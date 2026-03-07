use anyhow::{Context, Result, anyhow, ensure};
use burn::tensor::activation::silu;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dispatch::{Dispatch, DispatchDevice};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::config::ModelConfig;
use crate::layers::attention::Attention;
use crate::layers::rotary_embedding::RotaryEmbedding;
use crate::utils::context::AttentionContext;

struct LayerWeights {
    qkv_proj_w: Tensor<Dispatch, 2>, // [q+kv+kv, hidden]
    o_proj_w: Tensor<Dispatch, 2>,   // [hidden, q]
    q_norm_w: Option<Tensor<Dispatch, 1>>,
    k_norm_w: Option<Tensor<Dispatch, 1>>,
    gate_up_w: Tensor<Dispatch, 2>, // [2*inter, hidden]
    down_proj_w: Tensor<Dispatch, 2>, // [hidden, inter]
    input_ln_w: Tensor<Dispatch, 1>,
    post_ln_w: Tensor<Dispatch, 1>,
    attn: Attention,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

pub struct Qwen3ForCausalLM {
    embed_tokens: Tensor<Dispatch, 2>,
    layers: Vec<LayerWeights>,
    norm_w: Tensor<Dispatch, 1>,
    lm_head_weight: Tensor<Dispatch, 2>,
    rms_norm_eps: f64,
}

impl Qwen3ForCausalLM {
    pub fn new(
        config: &ModelConfig,
        model_path: &std::path::Path,
        device: &DispatchDevice,
    ) -> Result<Self> {
        let repo = SafetensorRepo::from_model_dir(model_path)?;

        let embed_tokens = repo.tensor2("model.embed_tokens.weight", device)?;
        let norm_w = repo.tensor1("model.norm.weight", device)?;

        let head_dim = config.head_dim();
        let q_size = config.num_attention_heads * head_dim;
        let kv_size = config.num_key_value_heads * head_dim;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let base = format!("model.layers.{i}");

            let q_w = repo.tensor2(&format!("{base}.self_attn.q_proj.weight"), device)?;
            let k_w = repo.tensor2(&format!("{base}.self_attn.k_proj.weight"), device)?;
            let v_w = repo.tensor2(&format!("{base}.self_attn.v_proj.weight"), device)?;
            let q_data = q_w.to_data().to_vec::<f32>()?;
            let k_data = k_w.to_data().to_vec::<f32>()?;
            let v_data = v_w.to_data().to_vec::<f32>()?;
            let mut qkv = Vec::with_capacity(q_data.len() + k_data.len() + v_data.len());
            qkv.extend_from_slice(&q_data);
            qkv.extend_from_slice(&k_data);
            qkv.extend_from_slice(&v_data);
            let qkv_proj_w = Tensor::<Dispatch, 2>::from_data(
                TensorData::new(qkv, [q_size + kv_size + kv_size, config.hidden_size]),
                device,
            );

            let gate_w = repo.tensor2(&format!("{base}.mlp.gate_proj.weight"), device)?;
            let up_w = repo.tensor2(&format!("{base}.mlp.up_proj.weight"), device)?;
            let gate_data = gate_w.to_data().to_vec::<f32>()?;
            let up_data = up_w.to_data().to_vec::<f32>()?;
            let mut gate_up = Vec::with_capacity(gate_data.len() + up_data.len());
            gate_up.extend_from_slice(&gate_data);
            gate_up.extend_from_slice(&up_data);
            let gate_up_w = Tensor::<Dispatch, 2>::from_data(
                TensorData::new(gate_up, [config.intermediate_size * 2, config.hidden_size]),
                device,
            );

            let q_norm_w = if !config.attention_bias {
                Some(repo.tensor1(&format!("{base}.self_attn.q_norm.weight"), device)?)
            } else {
                None
            };
            let k_norm_w = if !config.attention_bias {
                Some(repo.tensor1(&format!("{base}.self_attn.k_norm.weight"), device)?)
            } else {
                None
            };

            layers.push(LayerWeights {
                qkv_proj_w,
                o_proj_w: repo.tensor2(&format!("{base}.self_attn.o_proj.weight"), device)?,
                q_norm_w,
                k_norm_w,
                gate_up_w,
                down_proj_w: repo.tensor2(&format!("{base}.mlp.down_proj.weight"), device)?,
                input_ln_w: repo.tensor1(&format!("{base}.input_layernorm.weight"), device)?,
                post_ln_w: repo.tensor1(&format!("{base}.post_attention_layernorm.weight"), device)?,
                attn: Attention::new(config.num_attention_heads, config.num_key_value_heads, head_dim),
                rotary: RotaryEmbedding::new(
                    head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    device,
                )?,
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim,
                q_size,
                kv_size,
            });
        }

        let lm_head_weight = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            repo.tensor2("lm_head.weight", device)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm_w,
            lm_head_weight,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<Dispatch, 1, Int>,
        positions: &Tensor<Dispatch, 1, Int>,
        k_caches: &mut [Tensor<Dispatch, 4>],
        v_caches: &mut [Tensor<Dispatch, 4>],
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 2>> {
        ensure!(k_caches.len() == self.layers.len(), "k cache layer mismatch");
        ensure!(v_caches.len() == self.layers.len(), "v cache layer mismatch");

        let mut hidden_states = self.embed_tokens.clone().select(0, input_ids.clone());
        let mut residual: Option<Tensor<Dispatch, 2>> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let (normed, new_residual) = if let Some(res) = &residual {
                let x = hidden_states.clone() + res.clone();
                (rms_norm_2d(&x, &layer.input_ln_w, self.rms_norm_eps)?, x)
            } else {
                (
                    rms_norm_2d(&hidden_states, &layer.input_ln_w, self.rms_norm_eps)?,
                    hidden_states.clone(),
                )
            };

            let qkv = linear_2d(&normed, &layer.qkv_proj_w);
            let (mut q, mut k, v) = split_qkv(
                &qkv,
                layer.q_size,
                layer.kv_size,
                layer.num_heads,
                layer.num_kv_heads,
                layer.head_dim,
            )?;

            if let (Some(qn), Some(kn)) = (&layer.q_norm_w, &layer.k_norm_w) {
                q = rms_norm_3d(&q, qn, self.rms_norm_eps)?;
                k = rms_norm_3d(&k, kn, self.rms_norm_eps)?;
            }

            let (q, k) = layer.rotary.forward(positions, &q, &k)?;
            let o = layer
                .attn
                .forward(&q, &k, &v, &mut k_caches[i], &mut v_caches[i], ctx)?;
            let n = o.shape().as_slice()[0];
            let o = o.reshape([n, layer.num_heads * layer.head_dim]);
            let attn_out = linear_2d(&o, &layer.o_proj_w);

            let x = attn_out + new_residual.clone();
            let normed = rms_norm_2d(&x, &layer.post_ln_w, self.rms_norm_eps)?;
            let gate_up = linear_2d(&normed, &layer.gate_up_w);
            let mlp_act = silu_and_mul_2d(&gate_up)?;
            let mlp_out = linear_2d(&mlp_act, &layer.down_proj_w);

            hidden_states = mlp_out;
            residual = Some(x);
        }

        let hidden_states = if let Some(res) = residual {
            let x = hidden_states + res;
            rms_norm_2d(&x, &self.norm_w, self.rms_norm_eps)?
        } else {
            rms_norm_2d(&hidden_states, &self.norm_w, self.rms_norm_eps)?
        };

        Ok(hidden_states)
    }

    pub fn compute_logits(
        &self,
        hidden_states: &Tensor<Dispatch, 2>,
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 2>> {
        let hidden = if ctx.is_prefill {
            let cu: Vec<i32> = ctx
                .cu_seqlens_q
                .to_data()
                .to_vec::<i32>()
                .context("failed to read cu_seqlens_q")?;
            let batch = cu.len().saturating_sub(1);
            let mut indices = Vec::with_capacity(batch);
            for i in 0..batch {
                indices.push(cu[i + 1] - 1);
            }
            let idx = Tensor::<Dispatch, 1, Int>::from_data(
                TensorData::new(indices, [batch]),
                &hidden_states.device(),
            );
            hidden_states.clone().select(0, idx)
        } else {
            hidden_states.clone()
        };

        Ok(hidden.matmul(self.lm_head_weight.clone().transpose()))
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

fn linear_2d(x: &Tensor<Dispatch, 2>, w_out_in: &Tensor<Dispatch, 2>) -> Tensor<Dispatch, 2> {
    x.clone().matmul(w_out_in.clone().transpose())
}

fn rms_norm_2d(x: &Tensor<Dispatch, 2>, weight: &Tensor<Dispatch, 1>, eps: f64) -> Result<Tensor<Dispatch, 2>> {
    let shape = x.shape();
    let dims = shape.as_slice();
    ensure!(dims.len() == 2, "rms_norm_2d expects rank-2");
    let h = dims[1];
    ensure!(weight.shape().as_slice() == [h], "rms_norm_2d weight mismatch");
    let inv = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(1)
        .add_scalar(eps as f32)
        .sqrt()
        .recip();
    let w = weight.clone().unsqueeze_dim::<2>(0);
    Ok(x.clone() * inv * w)
}

fn rms_norm_3d(x: &Tensor<Dispatch, 3>, weight: &Tensor<Dispatch, 1>, eps: f64) -> Result<Tensor<Dispatch, 3>> {
    let dims = x.shape().as_slice().to_vec();
    ensure!(dims.len() == 3, "rms_norm_3d expects rank-3");
    let d = dims[2];
    ensure!(weight.shape().as_slice() == [d], "rms_norm_3d weight mismatch");
    let inv = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps as f32)
        .sqrt()
        .recip();
    let w = weight
        .clone()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0);
    Ok(x.clone() * inv * w)
}

fn silu_and_mul_2d(x: &Tensor<Dispatch, 2>) -> Result<Tensor<Dispatch, 2>> {
    let dims = x.shape().as_slice().to_vec();
    ensure!(dims.len() == 2, "silu_and_mul expects rank-2");
    let n = dims[0];
    let d2 = dims[1];
    ensure!(d2 % 2 == 0, "silu_and_mul last dim must be even");
    let d = d2 / 2;
    let gate = x.clone().slice([0..n, 0..d]);
    let up = x.clone().slice([0..n, d..d2]);
    Ok(silu(gate) * up)
}

fn split_qkv(
    qkv: &Tensor<Dispatch, 2>,
    q_size: usize,
    kv_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor<Dispatch, 3>, Tensor<Dispatch, 3>, Tensor<Dispatch, 3>)> {
    let dims = qkv.shape().as_slice().to_vec();
    ensure!(dims.len() == 2, "qkv must be rank-2");
    let n = dims[0];
    ensure!(dims[1] == q_size + kv_size + kv_size, "qkv width mismatch");

    Ok((
        qkv.clone()
            .slice([0..n, 0..q_size])
            .reshape([n, num_heads, head_dim]),
        qkv.clone()
            .slice([0..n, q_size..q_size + kv_size])
            .reshape([n, num_kv_heads, head_dim]),
        qkv.clone()
            .slice([0..n, q_size + kv_size..q_size + kv_size + kv_size])
            .reshape([n, num_kv_heads, head_dim]),
    ))
}

struct SafetensorRepo {
    files: Vec<SafeTensors<'static>>,
}

impl SafetensorRepo {
    fn from_model_dir(model_path: &std::path::Path) -> Result<Self> {
        let mut safetensors_files: Vec<_> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        safetensors_files.sort();
        ensure!(
            !safetensors_files.is_empty(),
            "no .safetensors file found under {}",
            model_path.display()
        );

        let mut files = Vec::with_capacity(safetensors_files.len());
        for file in safetensors_files {
            let bytes = std::fs::read(&file)
                .with_context(|| format!("failed to read safetensors file {}", file.display()))?;
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            files.push(SafeTensors::deserialize(leaked).context("invalid safetensors")?);
        }
        Ok(Self { files })
    }

    fn tensor_view(&self, key: &str) -> Result<TensorView<'_>> {
        for file in &self.files {
            if let Ok(v) = file.tensor(key) {
                return Ok(v);
            }
        }
        Err(anyhow!("missing tensor key: {key}"))
    }

    fn tensor1(&self, key: &str, device: &DispatchDevice) -> Result<Tensor<Dispatch, 1>> {
        let view = self.tensor_view(key)?;
        tensor_from_view_1d(view, device)
    }

    fn tensor2(&self, key: &str, device: &DispatchDevice) -> Result<Tensor<Dispatch, 2>> {
        let view = self.tensor_view(key)?;
        tensor_from_view_2d(view, device)
    }
}

fn tensor_from_view_1d(view: TensorView<'_>, device: &DispatchDevice) -> Result<Tensor<Dispatch, 1>> {
    let shape = view.shape().to_vec();
    ensure!(shape.len() == 1, "expected rank-1 tensor, got shape {shape:?}");
    let data = view_to_f32_vec(view)?;
    Ok(Tensor::<Dispatch, 1>::from_data(
        TensorData::new(data, [shape[0]]),
        device,
    ))
}

fn tensor_from_view_2d(view: TensorView<'_>, device: &DispatchDevice) -> Result<Tensor<Dispatch, 2>> {
    let shape = view.shape().to_vec();
    ensure!(shape.len() == 2, "expected rank-2 tensor, got shape {shape:?}");
    let data = view_to_f32_vec(view)?;
    Ok(Tensor::<Dispatch, 2>::from_data(
        TensorData::new(data, [shape[0], shape[1]]),
        device,
    ))
}

fn view_to_f32_vec(view: TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(view.data().len() / 4);
            for chunk in view.data().chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(view.data().len() / 2);
            for chunk in view.data().chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(view.data().len() / 2);
            for chunk in view.data().chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported dtype in safetensors: {other:?}")),
    }
}

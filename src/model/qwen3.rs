use anyhow::{Context, Result, anyhow};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dispatch::{Dispatch, DispatchDevice};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use crate::config::ModelConfig;
use crate::utils::context::AttentionContext;

pub struct Qwen3ForCausalLM {
    embed_tokens: Tensor<Dispatch, 2>,
    lm_head_weight: Tensor<Dispatch, 2>,
}

impl Qwen3ForCausalLM {
    pub fn new(
        config: &ModelConfig,
        model_path: &std::path::Path,
        device: &DispatchDevice,
    ) -> Result<Self> {
        let tensors = load_all_safetensors(model_path)?;

        let embed_view = tensors
            .tensor("model.embed_tokens.weight")
            .context("missing tensor: model.embed_tokens.weight")?;
        let embed = tensor_from_view_2d(embed_view, device)?;

        let lm_head_weight = if config.tie_word_embeddings {
            embed.clone()
        } else {
            let lm_view = tensors
                .tensor("lm_head.weight")
                .context("missing tensor: lm_head.weight")?;
            tensor_from_view_2d(lm_view, device)?
        };

        Ok(Self {
            embed_tokens: embed,
            lm_head_weight,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor<Dispatch, 1, Int>,
        _positions: &Tensor<Dispatch, 1, Int>,
        _k_caches: &mut [Tensor<Dispatch, 4>],
        _v_caches: &mut [Tensor<Dispatch, 4>],
        _ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 2>> {
        Ok(self.embed_tokens.clone().select(0, input_ids.clone()))
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

        let logits = hidden.matmul(self.lm_head_weight.clone().transpose());
        Ok(logits)
    }

    pub fn num_layers(&self) -> usize {
        0
    }
}

fn load_all_safetensors(model_path: &std::path::Path) -> Result<SafeTensors<'static>> {
    let mut safetensors_files: Vec<_> = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    safetensors_files.sort();

    let first = safetensors_files
        .first()
        .ok_or_else(|| anyhow!("no .safetensors file found under {}", model_path.display()))?;

    let bytes = std::fs::read(first)
        .with_context(|| format!("failed to read safetensors file {}", first.display()))?;
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let tensors = SafeTensors::deserialize(leaked).context("invalid safetensors")?;
    Ok(tensors)
}

fn tensor_from_view_2d(
    view: TensorView<'_>,
    device: &DispatchDevice,
) -> Result<Tensor<Dispatch, 2>> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(anyhow!("expected rank-2 tensor, got shape {shape:?}"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let data = view_to_f32_vec(view)?;
    Ok(Tensor::<Dispatch, 2>::from_data(
        TensorData::new(data, [rows, cols]),
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

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::layers::attention::Attention;
use crate::layers::rotary_embedding::RotaryEmbedding;
use crate::utils::context::AttentionContext;

/// SiLU(gate) * up
fn silu_and_mul(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(candle_core::D::Minus1)? / 2;
    let gate = x.narrow(candle_core::D::Minus1, 0, half)?;
    let up = x.narrow(candle_core::D::Minus1, half, half)?;
    Ok((candle_nn::ops::silu(&gate)? * up)?)
}

/// RMSNorm with optional fused residual add.
struct FusedRmsNorm {
    norm: RmsNorm,
}

impl FusedRmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: candle_nn::rms_norm(size, eps, vb)?,
        })
    }

    /// Apply RMSNorm without residual.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.norm.forward(x)?)
    }

    /// Fused: x = x + residual, then RMSNorm(x). Returns (normed, new_residual).
    fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = (x + residual)?;
        let normed = self.norm.forward(&x)?;
        Ok((normed, x))
    }
}

struct Qwen3Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<FusedRmsNorm>,
    k_norm: Option<FusedRmsNorm>,
    rotary_emb: RotaryEmbedding,
    attn: Attention,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl Qwen3Attention {
    fn new(config: &ModelConfig, _layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Load separate q/k/v weights and concatenate into qkv_proj
        let q_w = vb.get((q_size, config.hidden_size), "q_proj.weight")?;
        let k_w = vb.get((kv_size, config.hidden_size), "k_proj.weight")?;
        let v_w = vb.get((kv_size, config.hidden_size), "v_proj.weight")?;
        let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let qkv_proj = Linear::new(qkv_w, None);

        let o_proj = linear_no_bias(q_size, config.hidden_size, vb.pp("o_proj"))?;

        let (q_norm, k_norm) = if !config.attention_bias {
            let q_norm = FusedRmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
            let k_norm = FusedRmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;

        let attn = Attention::new(num_heads, num_kv_heads, head_dim);

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            attn,
            num_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
        })
    }

    fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        let qkv = self.qkv_proj.forward(hidden_states)?;

        let mut q = qkv.narrow(candle_core::D::Minus1, 0, self.q_size)?;
        let mut k = qkv.narrow(candle_core::D::Minus1, self.q_size, self.kv_size)?;
        let v = qkv.narrow(candle_core::D::Minus1, self.q_size + self.kv_size, self.kv_size)?;

        // Reshape to (N, num_heads, head_dim)
        let n = q.dim(0)?;
        q = q.reshape((n, self.num_heads, self.head_dim))?;
        k = k.reshape((n, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((n, self.num_kv_heads, self.head_dim))?;

        // Apply q/k norms if present (Qwen3 without attention_bias)
        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(&q)?;
            k = k_norm.forward(&k)?;
        }

        // RoPE
        let (q, k) = self.rotary_emb.forward(positions, &q, &k)?;

        // Attention with KV cache
        let o = self.attn.forward(&q, &k, &v, k_cache, v_cache, ctx)?;

        // (N, num_heads, head_dim) -> (N, hidden_size)
        let o = o.reshape((n, self.num_heads * self.head_dim))?;
        let output = self.o_proj.forward(&o)?;
        Ok(output)
    }
}

struct Qwen3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3MLP {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        // Load gate and up separately, concatenate
        let gate_w = vb.get((inter, hidden), "gate_proj.weight")?;
        let up_w = vb.get((inter, hidden), "up_proj.weight")?;
        let gate_up_w = Tensor::cat(&[&gate_w, &up_w], 0)?;
        let gate_up_proj = Linear::new(gate_up_w, None);

        let down_proj = linear_no_bias(inter, hidden, vb.pp("down_proj"))?;

        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let x = silu_and_mul(&gate_up)?;
        let x = self.down_proj.forward(&x)?;
        Ok(x)
    }
}

struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: FusedRmsNorm,
    post_attention_layernorm: FusedRmsNorm,
}

impl Qwen3DecoderLayer {
    fn new(config: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Qwen3Attention::new(config, layer_idx, vb.pp("self_attn"))?,
            mlp: Qwen3MLP::new(config, vb.pp("mlp"))?,
            input_layernorm: FusedRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: FusedRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        positions: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        ctx: &AttentionContext,
    ) -> Result<(Tensor, Tensor)> {
        let (normed, residual) = if let Some(res) = residual {
            self.input_layernorm.forward_with_residual(hidden_states, res)?
        } else {
            let normed = self.input_layernorm.forward(hidden_states)?;
            (normed, hidden_states.clone())
        };

        let hidden_states = self.self_attn.forward(positions, &normed, k_cache, v_cache, ctx)?;
        let (normed, residual) = self
            .post_attention_layernorm
            .forward_with_residual(&hidden_states, &residual)?;
        let hidden_states = self.mlp.forward(&normed)?;

        Ok((hidden_states, residual))
    }
}

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: FusedRmsNorm,
}

impl Qwen3Model {
    fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(
                config,
                i,
                vb.pp(format!("layers.{i}")),
            )?);
        }

        let norm = FusedRmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        k_caches: &mut [Tensor],
        v_caches: &mut [Tensor],
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward(
                positions,
                &hidden_states,
                residual.as_ref(),
                &mut k_caches[i],
                &mut v_caches[i],
                ctx,
            )?;
            hidden_states = h;
            residual = Some(r);
        }

        let (hidden_states, _) = self
            .norm
            .forward_with_residual(&hidden_states, residual.as_ref().unwrap())?;
        Ok(hidden_states)
    }
}

pub struct Qwen3ForCausalLM {
    model: Qwen3Model,
    lm_head_weight: Tensor,
}

impl Qwen3ForCausalLM {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let model = Qwen3Model::new(config, vb.pp("model"))?;

        let lm_head_weight = if config.tie_word_embeddings {
            model.embed_tokens.embeddings().clone()
        } else {
            vb.get((config.vocab_size, config.hidden_size), "lm_head.weight")?
        };

        Ok(Self {
            model,
            lm_head_weight,
        })
    }

    /// Run model forward pass, returns hidden states (N, hidden_size).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        k_caches: &mut [Tensor],
        v_caches: &mut [Tensor],
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        self.model
            .forward(input_ids, positions, k_caches, v_caches, ctx)
    }

    /// Compute logits from hidden states.
    /// During prefill, extracts only the last token per sequence.
    pub fn compute_logits(
        &self,
        hidden_states: &Tensor,
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        let hidden = if ctx.is_prefill {
            // Extract last token per sequence using cu_seqlens_q
            let cu_seqlens: Vec<u32> = ctx.cu_seqlens_q.to_vec1()?;
            let batch = cu_seqlens.len() - 1;
            let indices: Vec<u32> = (0..batch).map(|i| cu_seqlens[i + 1] - 1).collect();
            let indices = Tensor::from_vec(indices, batch, hidden_states.device())?;
            hidden_states.index_select(&indices, 0)?
        } else {
            hidden_states.clone()
        };

        // logits = hidden @ lm_head_weight^T
        let logits = hidden.matmul(&self.lm_head_weight.t()?)?;
        Ok(logits)
    }

    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }
}

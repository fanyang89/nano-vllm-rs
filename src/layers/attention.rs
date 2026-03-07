use anyhow::Result;
use burn::tensor::Tensor;
use burn_dispatch::Dispatch;

use crate::utils::context::AttentionContext;

/// Burn migration placeholder for paged attention.
pub struct Attention {
    #[allow(dead_code)]
    num_heads: usize,
    #[allow(dead_code)]
    num_kv_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
}

impl Attention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &Tensor<Dispatch, 3>,
        _k: &Tensor<Dispatch, 3>,
        _v: &Tensor<Dispatch, 3>,
        _k_cache: &mut Tensor<Dispatch, 4>,
        _v_cache: &mut Tensor<Dispatch, 4>,
        _ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 3>> {
        // TODO: replace with Burn-native paged attention (prefill + decode).
        Ok(q.clone())
    }
}

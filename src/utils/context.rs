use burn::tensor::{Int, Tensor};
use burn_dispatch::Dispatch;

/// Context passed through the model forward pass to attention layers.
pub struct AttentionContext {
    pub is_prefill: bool,
    /// Cumulative sequence lengths for queries, shape [batch+1], i32.
    pub cu_seqlens_q: Tensor<Dispatch, 1, Int>,
    /// Cumulative sequence lengths for keys, shape [batch+1], i32.
    pub cu_seqlens_k: Tensor<Dispatch, 1, Int>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    /// Maps each token position to a flat slot in the KV cache, shape [num_tokens], i32.
    pub slot_mapping: Tensor<Dispatch, 1, Int>,
    /// Per-sequence context length for decode, shape [batch], i32.
    pub context_lens: Option<Tensor<Dispatch, 1, Int>>,
    /// Block table mapping sequences to KV cache blocks, shape [batch, max_blocks], i32.
    pub block_tables: Option<Tensor<Dispatch, 2, Int>>,
}

use burn::tensor::{backend::Backend, Int, Tensor};

/// Context passed through the model forward pass to attention layers.
pub struct AttentionContext<B: Backend> {
    pub is_prefill: bool,
    pub seq_ids: Vec<usize>,
    /// Cumulative sequence lengths for queries, shape [batch+1], i32.
    pub cu_seqlens_q: Tensor<B, 1, Int>,
    /// Host-side mirror of `cu_seqlens_q` to avoid repeated device synchronization.
    pub cu_seqlens_q_host: Vec<i32>,
    /// Cumulative sequence lengths for keys, shape [batch+1], i32.
    pub cu_seqlens_k: Tensor<B, 1, Int>,
    /// Host-side mirror of `cu_seqlens_k` to avoid repeated device synchronization.
    pub cu_seqlens_k_host: Vec<i32>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    /// Maps each token position to a flat slot in the KV cache, shape [num_tokens], i32.
    pub slot_mapping: Tensor<B, 1, Int>,
    /// Host-side mirror of `slot_mapping` for fast scalar/range decisions.
    pub slot_mapping_host: Vec<i32>,
    /// Per-sequence context length for decode, shape [batch], i32.
    pub context_lens: Option<Tensor<B, 1, Int>>,
    /// Host-side mirror of `context_lens` to avoid repeated device synchronization.
    pub context_lens_host: Option<Vec<i32>>,
    /// Current block id for decode token append.
    pub last_block_ids: Option<Vec<usize>>,
    /// Number of valid tokens in the current last block after append.
    pub last_block_lens: Option<Vec<usize>>,
    /// Per-sequence flat slot indices into the KV cache.
    pub kv_slot_indices: Option<Vec<Tensor<B, 1, Int>>>,
}

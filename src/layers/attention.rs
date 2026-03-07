use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};

use crate::utils::context::AttentionContext;

/// Paged attention layer with KV cache support.
///
/// Handles both prefill (variable-length sequences) and decode (single token per sequence)
/// using block-based KV cache storage.
pub struct Attention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// Forward pass for attention with paged KV cache.
    ///
    /// - q: (N, num_heads, head_dim)
    /// - k: (N, num_kv_heads, head_dim)  [N=total_tokens for prefill, N=batch for decode]
    /// - v: (N, num_kv_heads, head_dim)
    /// - k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    /// - v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
    /// - ctx: attention context with slot_mapping, block_tables, etc.
    ///
    /// Returns: (N, num_heads, head_dim) or (batch, num_heads, head_dim)
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        // Store K/V into paged cache
        store_kvcache(k, v, k_cache, v_cache, &ctx.slot_mapping)?;

        if ctx.is_prefill {
            self.prefill_attention(q, k, v, k_cache, v_cache, ctx)
        } else {
            self.decode_attention(q, k_cache, v_cache, ctx)
        }
    }

    /// Prefill: variable-length multi-token attention with causal mask.
    fn prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        let cu_seqlens_q: Vec<u32> = ctx.cu_seqlens_q.to_vec1()?;
        let cu_seqlens_k: Vec<u32> = ctx.cu_seqlens_k.to_vec1()?;
        let batch_size = cu_seqlens_q.len() - 1;
        let has_prefix_cache = ctx.block_tables.is_some();

        let mut outputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let q_start = cu_seqlens_q[i] as usize;
            let q_end = cu_seqlens_q[i + 1] as usize;
            let q_len = q_end - q_start;

            let k_start = cu_seqlens_k[i] as usize;
            let k_end = cu_seqlens_k[i + 1] as usize;
            let k_len = k_end - k_start;

            // q_i: (q_len, num_heads, head_dim)
            let q_i = q.narrow(0, q_start, q_len)?;

            let (k_i, v_i) = if has_prefix_cache {
                // With prefix cache: read full K/V from paged cache
                self.gather_kv_from_cache(k_cache, v_cache, ctx, i, k_len)?
            } else {
                // Without prefix cache: use computed K/V directly
                let k_i = k.narrow(0, k_start, k_len)?;
                let v_i = v.narrow(0, k_start, k_len)?;
                (k_i, v_i)
            };

            let out = self.scaled_dot_product_attention(&q_i, &k_i, &v_i, true, q_len, k_len)?;
            outputs.push(out);
        }

        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Decode: single query token per sequence, attend over cached K/V.
    fn decode_attention(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        ctx: &AttentionContext,
    ) -> Result<Tensor> {
        let context_lens: Vec<u32> = ctx
            .context_lens
            .as_ref()
            .expect("decode requires context_lens")
            .to_vec1()?;
        let batch_size = q.dim(0)?;

        let mut outputs = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let ctx_len = context_lens[i] as usize;

            // q_i: (1, num_heads, head_dim)
            let q_i = q.narrow(0, i, 1)?;

            // Gather K/V from paged cache for this sequence
            let (k_i, v_i) = self.gather_kv_from_cache(k_cache, v_cache, ctx, i, ctx_len)?;

            let out = self.scaled_dot_product_attention(&q_i, &k_i, &v_i, false, 1, ctx_len)?;
            outputs.push(out);
        }

        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Gather K/V blocks from paged cache for sequence i.
    /// Returns (k, v) each of shape (seq_len, num_kv_heads, head_dim).
    fn gather_kv_from_cache(
        &self,
        k_cache: &Tensor,
        v_cache: &Tensor,
        ctx: &AttentionContext,
        seq_idx: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let block_tables = ctx
            .block_tables
            .as_ref()
            .expect("gather_kv requires block_tables");

        // block_table_i: (max_blocks,)
        let block_table_i: Vec<u32> = block_tables.i(seq_idx)?.to_vec1()?;

        let block_size = k_cache.dim(1)?;
        let blocks_needed = seq_len.div_ceil(block_size);
        let block_ids = Tensor::from_vec(
            block_table_i[..blocks_needed].to_vec(),
            blocks_needed,
            k_cache.device(),
        )?;

        // Gather all blocks in one shot: (B, block, kv_heads, head_dim)
        let k_blocks = k_cache.index_select(&block_ids, 0)?;
        let v_blocks = v_cache.index_select(&block_ids, 0)?;

        // Flatten blocks then trim to exact sequence length.
        let k = k_blocks
            .reshape((blocks_needed * block_size, self.num_kv_heads, self.head_dim))?
            .narrow(0, 0, seq_len)?;
        let v = v_blocks
            .reshape((blocks_needed * block_size, self.num_kv_heads, self.head_dim))?
            .narrow(0, 0, seq_len)?;
        Ok((k, v))
    }

    /// Standard scaled dot-product attention.
    ///
    /// - q: (q_len, num_heads, head_dim)
    /// - k: (kv_len, num_kv_heads, head_dim)
    /// - v: (kv_len, num_kv_heads, head_dim)
    ///
    /// Returns: (q_len, num_heads, head_dim)
    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
        q_len: usize,
        kv_len: usize,
    ) -> Result<Tensor> {
        let num_groups = self.num_heads / self.num_kv_heads;

        // Expand KV heads to match Q heads via GQA repeat
        // k: (kv_len, num_kv_heads, head_dim) -> (kv_len, num_heads, head_dim)
        let k = if num_groups > 1 {
            k.unsqueeze(2)?
                .expand((kv_len, self.num_kv_heads, num_groups, self.head_dim))?
                .reshape((kv_len, self.num_heads, self.head_dim))?
        } else {
            k.clone()
        };
        let v = if num_groups > 1 {
            v.unsqueeze(2)?
                .expand((kv_len, self.num_kv_heads, num_groups, self.head_dim))?
                .reshape((kv_len, self.num_heads, self.head_dim))?
        } else {
            v.clone()
        };

        // Transpose to (num_heads, seq_len, head_dim) for batched matmul
        let q = q.transpose(0, 1)?.to_dtype(DType::F32)?; // (num_heads, q_len, head_dim)
        let k = k.transpose(0, 1)?.to_dtype(DType::F32)?; // (num_heads, kv_len, head_dim)
        let v = v.transpose(0, 1)?.to_dtype(DType::F32)?; // (num_heads, kv_len, head_dim)

        // attn_weights = q @ k^T * scale
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?.affine(self.scale, 0.0)?;

        // Apply causal mask if needed
        let attn_weights = if causal && q_len > 1 {
            let device = attn_weights.device();
            let mask = create_causal_mask(q_len, kv_len, device)?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        // softmax + matmul with V
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let output = attn_weights.matmul(&v)?; // (num_heads, q_len, head_dim)

        // Transpose back to (q_len, num_heads, head_dim)
        let output = output.transpose(0, 1)?.to_dtype(q.dtype())?;
        Ok(output)
    }
}

/// Store K/V into paged cache using slot_mapping.
///
/// - key: (N, num_kv_heads, head_dim)
/// - value: (N, num_kv_heads, head_dim)
/// - k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
/// - v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
/// - slot_mapping: (N,) i32, maps each token to flat cache slot
fn store_kvcache(
    key: &Tensor,
    value: &Tensor,
    k_cache: &mut Tensor,
    v_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let slots: Vec<u32> = slot_mapping
        .to_vec1::<i32>()?
        .into_iter()
        .map(|s| if s < 0 { u32::MAX } else { s as u32 })
        .collect();
    let d = k_cache.dim(2)? * k_cache.dim(3)?; // num_kv_heads * head_dim

    // Flatten cache to (num_blocks * block_size, D) for easy scatter
    let cache_shape = k_cache.dims().to_vec();
    let total_slots = cache_shape[0] * cache_shape[1];
    let k_flat = k_cache.reshape((total_slots, d))?;
    let v_flat = v_cache.reshape((total_slots, d))?;

    let key_flat = key.reshape((key.dim(0)?, d))?;
    let val_flat = value.reshape((value.dim(0)?, d))?;

    let n = slots.len();
    let slot_ids = Tensor::from_vec(slots, n, key.device())?;
    let indexes = slot_ids.unsqueeze(1)?.expand((n, d))?.contiguous()?;
    k_flat.scatter_set(&indexes, &key_flat, 0)?;
    v_flat.scatter_set(&indexes, &val_flat, 0)?;

    // Reshape back
    *k_cache = k_flat.reshape(cache_shape.as_slice())?;
    *v_cache = v_flat.reshape(cache_shape.as_slice())?;

    Ok(())
}

/// Create a causal mask: 0 for allowed positions, -inf for masked.
fn create_causal_mask(q_len: usize, kv_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    // For each query position i (0..q_len), it can attend to key positions
    // 0..(kv_len - q_len + i + 1), i.e., offset = kv_len - q_len
    let offset = kv_len as i64 - q_len as i64;
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for qi in 0..q_len {
        let max_kv = (offset + qi as i64 + 1) as usize;
        for kj in max_kv..kv_len {
            mask_data[qi * kv_len + kj] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_data, (1, q_len, kv_len), device)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3, 3, &Device::Cpu).unwrap();
        let data: Vec<Vec<Vec<f32>>> = mask.to_vec3().unwrap();
        // Row 0: can attend to position 0 only
        assert!(data[0][0][0] == 0.0);
        assert!(data[0][0][1] == f32::NEG_INFINITY);
        // Row 2: can attend to all 3
        assert!(data[0][2][0] == 0.0);
        assert!(data[0][2][2] == 0.0);
    }

    #[test]
    fn test_store_kvcache() {
        let device = Device::Cpu;
        let num_blocks = 4;
        let block_size = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        let mut k_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let mut v_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();

        // Store 2 tokens: slot 0 (block0, offset0) and slot 3 (block1, offset1)
        let key = Tensor::ones((2, num_kv_heads, head_dim), DType::F32, &device).unwrap();
        let value = Tensor::ones((2, num_kv_heads, head_dim), DType::F32, &device).unwrap();
        let slot_mapping = Tensor::from_vec(vec![0i32, 3], 2, &device).unwrap();

        store_kvcache(&key, &value, &mut k_cache, &mut v_cache, &slot_mapping).unwrap();

        // Check slot 0 (block 0, offset 0) is filled
        let val: f32 = k_cache.i((0, 0, 0, 0)).unwrap().to_scalar().unwrap();
        assert_eq!(val, 1.0);

        // Check slot 3 (block 1, offset 1) is filled
        let val: f32 = k_cache.i((1, 1, 0, 0)).unwrap().to_scalar().unwrap();
        assert_eq!(val, 1.0);

        // Check slot 1 (block 0, offset 1) is still zero
        let val: f32 = k_cache.i((0, 1, 0, 0)).unwrap().to_scalar().unwrap();
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_attention_prefill() {
        let device = Device::Cpu;
        let attn = Attention::new(4, 2, 8);

        let num_blocks = 4;
        let block_size = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let mut k_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let mut v_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();

        // 2 sequences: seq0 has 3 tokens, seq1 has 2 tokens
        let total_tokens = 5;
        let q = Tensor::randn(0.0f32, 1.0, (total_tokens, 4, head_dim), &device).unwrap();
        let k =
            Tensor::randn(0.0f32, 1.0, (total_tokens, num_kv_heads, head_dim), &device).unwrap();
        let v =
            Tensor::randn(0.0f32, 1.0, (total_tokens, num_kv_heads, head_dim), &device).unwrap();

        let ctx = AttentionContext {
            is_prefill: true,
            cu_seqlens_q: Tensor::from_vec(vec![0u32, 3, 5], 3, &device).unwrap(),
            cu_seqlens_k: Tensor::from_vec(vec![0u32, 3, 5], 3, &device).unwrap(),
            max_seqlen_q: 3,
            max_seqlen_k: 3,
            slot_mapping: Tensor::from_vec(vec![0i32, 1, 2, 4, 5], total_tokens, &device).unwrap(),
            context_lens: None,
            block_tables: None,
        };

        let output = attn
            .forward(&q, &k, &v, &mut k_cache, &mut v_cache, &ctx)
            .unwrap();
        assert_eq!(output.dims(), &[total_tokens, 4, head_dim]);
    }
}

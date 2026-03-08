use anyhow::{Result, ensure};
use burn::tensor::activation::softmax;
use burn::tensor::{IndexingUpdateOp, Int, Tensor};
use burn_dispatch::Dispatch;

use crate::utils::context::AttentionContext;

/// Paged attention with KV cache support.
pub struct Attention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor<Dispatch, 3>,
        k: &Tensor<Dispatch, 3>,
        v: &Tensor<Dispatch, 3>,
        k_cache: &mut Tensor<Dispatch, 4>,
        v_cache: &mut Tensor<Dispatch, 4>,
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 3>> {
        store_kvcache(k, v, k_cache, v_cache, &ctx.slot_mapping)?;

        if ctx.is_prefill {
            self.prefill_attention(q, k, v, k_cache, v_cache, ctx)
        } else {
            self.decode_attention(q, k_cache, v_cache, ctx)
        }
    }

    fn prefill_attention(
        &self,
        q: &Tensor<Dispatch, 3>,
        k: &Tensor<Dispatch, 3>,
        v: &Tensor<Dispatch, 3>,
        k_cache: &Tensor<Dispatch, 4>,
        v_cache: &Tensor<Dispatch, 4>,
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 3>> {
        let cu_seqlens_q: Vec<i32> = ctx.cu_seqlens_q.to_data().to_vec::<i32>()?;
        let cu_seqlens_k: Vec<i32> = ctx.cu_seqlens_k.to_data().to_vec::<i32>()?;
        let batch_size = cu_seqlens_q.len().saturating_sub(1);
        let has_prefix_cache = ctx.block_tables.is_some();

        let mut outputs = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let q_start = cu_seqlens_q[i] as usize;
            let q_end = cu_seqlens_q[i + 1] as usize;
            let q_len = q_end.saturating_sub(q_start);
            let k_start = cu_seqlens_k[i] as usize;
            let k_end = cu_seqlens_k[i + 1] as usize;
            let k_len = k_end.saturating_sub(k_start);

            let q_i = q
                .clone()
                .slice([q_start..q_end, 0..self.num_heads, 0..self.head_dim]);

            let (k_i, v_i) = if has_prefix_cache {
                self.gather_kv_from_cache(
                    k_cache,
                    v_cache,
                    ctx.block_tables.as_ref().expect("checked"),
                    i,
                    k_len,
                )?
            } else {
                (
                    k.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                    v.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                )
            };

            outputs.push(self.scaled_dot_product_attention(&q_i, &k_i, &v_i, true)?);
            debug_assert_eq!(q_len, outputs.last().unwrap().shape().as_slice()[0]);
        }

        Ok(Tensor::<Dispatch, 3>::cat(outputs, 0))
    }

    fn decode_attention(
        &self,
        q: &Tensor<Dispatch, 3>,
        k_cache: &Tensor<Dispatch, 4>,
        v_cache: &Tensor<Dispatch, 4>,
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 3>> {
        let context_lens: Vec<i32> = ctx
            .context_lens
            .as_ref()
            .expect("decode requires context_lens")
            .to_data()
            .to_vec::<i32>()?;

        let batch_size = q.shape().as_slice()[0];
        let mut outputs = Vec::with_capacity(batch_size);
        for (i, &ctx_len) in context_lens.iter().enumerate() {
            let q_i = q
                .clone()
                .slice([i..i + 1, 0..self.num_heads, 0..self.head_dim]);
            let (k_i, v_i) = self.gather_kv_from_cache(
                k_cache,
                v_cache,
                ctx.block_tables.as_ref().expect("decode block_tables"),
                i,
                ctx_len as usize,
            )?;
            outputs.push(self.scaled_dot_product_attention(&q_i, &k_i, &v_i, false)?);
        }
        Ok(Tensor::<Dispatch, 3>::cat(outputs, 0))
    }

    fn gather_kv_from_cache(
        &self,
        k_cache: &Tensor<Dispatch, 4>,
        v_cache: &Tensor<Dispatch, 4>,
        block_tables: &Tensor<Dispatch, 2, Int>,
        seq_idx: usize,
        seq_len: usize,
    ) -> Result<(Tensor<Dispatch, 3>, Tensor<Dispatch, 3>)> {
        let cache_dims = k_cache.shape().as_slice().to_vec();
        let block_size = cache_dims[1];
        let blocks_needed = seq_len.div_ceil(block_size);
        let block_ids = block_tables
            .clone()
            .slice([seq_idx..seq_idx + 1, 0..blocks_needed])
            .reshape([blocks_needed]);

        let k_blocks = k_cache.clone().select(0, block_ids.clone());
        let v_blocks = v_cache.clone().select(0, block_ids);

        let k = k_blocks
            .reshape([blocks_needed * block_size, self.num_kv_heads, self.head_dim])
            .slice([0..seq_len, 0..self.num_kv_heads, 0..self.head_dim]);
        let v = v_blocks
            .reshape([blocks_needed * block_size, self.num_kv_heads, self.head_dim])
            .slice([0..seq_len, 0..self.num_kv_heads, 0..self.head_dim]);
        Ok((k, v))
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Tensor<Dispatch, 3>,
        k: &Tensor<Dispatch, 3>,
        v: &Tensor<Dispatch, 3>,
        causal: bool,
    ) -> Result<Tensor<Dispatch, 3>> {
        let q_len = q.shape().as_slice()[0];
        let kv_len = k.shape().as_slice()[0];
        let groups = self.num_heads / self.num_kv_heads;

        let k = if groups > 1 {
            k.clone()
                .unsqueeze_dim::<4>(2)
                .repeat(&[1, 1, groups, 1])
                .reshape([kv_len, self.num_heads, self.head_dim])
        } else {
            k.clone()
        };
        let v = if groups > 1 {
            v.clone()
                .unsqueeze_dim::<4>(2)
                .repeat(&[1, 1, groups, 1])
                .reshape([kv_len, self.num_heads, self.head_dim])
        } else {
            v.clone()
        };

        // [q, h, d] -> [h, q, d], [kv, h, d] -> [h, kv, d]
        let qh = q.clone().swap_dims(0, 1);
        let kh = k.clone().swap_dims(0, 1);
        let vh = v.clone().swap_dims(0, 1);
        let mut attn = qh.matmul(kh.clone().swap_dims(1, 2)).mul_scalar(self.scale);

        if causal && q_len > 1 {
            let mask = create_causal_mask(q_len, kv_len, &attn.device());
            attn = attn + mask;
        }

        let attn = softmax(attn, 2);
        let out = attn.matmul(vh); // [h, q, d]
        Ok(out.swap_dims(0, 1)) // [q, h, d]
    }
}

/// Store K/V into paged cache using tensor scatter ops.
fn store_kvcache(
    key: &Tensor<Dispatch, 3>,
    value: &Tensor<Dispatch, 3>,
    k_cache: &mut Tensor<Dispatch, 4>,
    v_cache: &mut Tensor<Dispatch, 4>,
    slot_mapping: &Tensor<Dispatch, 1, Int>,
) -> Result<()> {
    let dims = k_cache.shape().as_slice().to_vec();
    ensure!(dims.len() == 4, "k_cache must be rank-4");
    let total_slots = dims[0] * dims[1];
    let row_width = dims[2] * dims[3];

    let device = key.device();
    let n = key.shape().as_slice()[0];

    let mut slots = slot_mapping.clone();
    let valid_mask = slots.clone().greater_equal_elem(0);
    slots = slots.clamp(0, (total_slots.saturating_sub(1)) as i32);

    let indices = slots.unsqueeze_dim::<2>(1).repeat(&[1, row_width]); // [n, row_width]
    let valid = valid_mask
        .unsqueeze_dim::<2>(1)
        .repeat(&[1, row_width])
        .float();

    let key_flat = key.clone().reshape([n, row_width]).mul(valid.clone());
    let val_flat = value.clone().reshape([n, row_width]).mul(valid.clone());

    let k_flat = k_cache.clone().reshape([total_slots, row_width]);
    let v_flat = v_cache.clone().reshape([total_slots, row_width]);

    // Emulate assign semantics via Add scatter:
    // new = old - old_selected + selected_values
    let old_k = k_flat.clone().gather(0, indices.clone()).mul(valid.clone());
    let old_v = v_flat.clone().gather(0, indices.clone()).mul(valid);
    let k_flat = k_flat.scatter(
        0,
        indices.clone(),
        old_k.mul_scalar(-1.0),
        IndexingUpdateOp::Add,
    );
    let k_flat = k_flat.scatter(0, indices.clone(), key_flat, IndexingUpdateOp::Add);
    let v_flat = v_flat.scatter(
        0,
        indices.clone(),
        old_v.mul_scalar(-1.0),
        IndexingUpdateOp::Add,
    );
    let v_flat = v_flat.scatter(0, indices, val_flat, IndexingUpdateOp::Add);

    *k_cache = k_flat.reshape([dims[0], dims[1], dims[2], dims[3]]);
    *v_cache = v_flat.reshape([dims[0], dims[1], dims[2], dims[3]]);
    let _ = device;
    Ok(())
}

fn create_causal_mask(
    q_len: usize,
    kv_len: usize,
    device: &burn_dispatch::DispatchDevice,
) -> Tensor<Dispatch, 3> {
    let offset = (kv_len as i64 - q_len as i64) as i32;
    let q_idx = Tensor::<Dispatch, 1, Int>::arange(0..q_len as i64, device)
        .reshape([q_len, 1])
        .repeat(&[1, kv_len]);
    let k_idx = Tensor::<Dispatch, 1, Int>::arange(0..kv_len as i64, device)
        .reshape([1, kv_len])
        .repeat(&[q_len, 1]);

    let allowed = k_idx.lower_equal(q_idx.add_scalar(offset));
    let mask = Tensor::<Dispatch, 2>::zeros([q_len, kv_len], device)
        .mask_fill(allowed.bool_not(), f32::NEG_INFINITY);
    mask.unsqueeze_dim::<3>(0)
}

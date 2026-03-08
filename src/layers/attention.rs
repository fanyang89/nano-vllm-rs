use anyhow::{ensure, Result};
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor};

use crate::utils::context::AttentionContext;
use crate::utils::profiler::Scope;

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
}

impl Attention {
    pub fn forward<B: Backend>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        k_cache: &mut Tensor<B, 4>,
        v_cache: &mut Tensor<B, 4>,
        ctx: &AttentionContext<B>,
    ) -> Result<Tensor<B, 3>> {
        let _scope = Scope::new("store_kvcache");
        store_kvcache(k, v, k_cache, v_cache, &ctx.slot_mapping)?;
        drop(_scope);

        if ctx.is_prefill {
            self.prefill_attention(q, k, v, k_cache, v_cache, ctx)
        } else {
            self.decode_attention(q, k_cache, v_cache, ctx)
        }
    }

    fn prefill_attention<B: Backend>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        ctx: &AttentionContext<B>,
    ) -> Result<Tensor<B, 3>> {
        let cu_seqlens_q = &ctx.cu_seqlens_q_host;
        let cu_seqlens_k = &ctx.cu_seqlens_k_host;
        let batch_size = cu_seqlens_q.len().saturating_sub(1);
        let kv_slot_indices = ctx.kv_slot_indices.as_ref();
        let has_prefix_cache = kv_slot_indices.is_some();

        let mut outputs = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let q_start = cu_seqlens_q[i] as usize;
            let q_end = cu_seqlens_q[i + 1] as usize;
            let q_len = q_end.saturating_sub(q_start);
            let k_start = cu_seqlens_k[i] as usize;
            let k_end = cu_seqlens_k[i + 1] as usize;

            let q_i = q
                .clone()
                .slice([q_start..q_end, 0..self.num_heads, 0..self.head_dim]);

            let (k_i, v_i) = if has_prefix_cache {
                let _scope = Scope::new("prefill_gather_kv");
                self.gather_kv_from_cache(k_cache, v_cache, &kv_slot_indices.expect("checked")[i])?
            } else {
                (
                    k.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                    v.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                )
            };

            let _scope = Scope::new("prefill_sdpa");
            outputs.push(self.scaled_dot_product_attention(&q_i, &k_i, &v_i, true)?);
            drop(_scope);
            debug_assert_eq!(q_len, outputs.last().unwrap().shape().as_slice()[0]);
        }

        Ok(Tensor::<B, 3>::cat(outputs, 0))
    }

    fn decode_attention<B: Backend>(
        &self,
        q: &Tensor<B, 3>,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        ctx: &AttentionContext<B>,
    ) -> Result<Tensor<B, 3>> {
        let context_lens = ctx
            .context_lens_host
            .as_ref()
            .expect("decode requires context_lens");
        let kv_slot_indices = ctx
            .kv_slot_indices
            .as_ref()
            .expect("decode requires kv_slot_indices");

        let batch_size = q.shape().as_slice()[0];
        let max_ctx_len = context_lens.iter().copied().max().unwrap_or(0) as usize;
        let device = q.device();
        let mut k_batch = Vec::with_capacity(batch_size);
        let mut v_batch = Vec::with_capacity(batch_size);

        for (i, &ctx_len) in context_lens.iter().enumerate() {
            debug_assert_eq!(kv_slot_indices[i].shape().as_slice()[0], ctx_len as usize);
            let _scope = Scope::new("decode_gather_kv");
            let (k_i, v_i) = self.gather_kv_from_cache(k_cache, v_cache, &kv_slot_indices[i])?;
            drop(_scope);

            let pad_len = max_ctx_len.saturating_sub(ctx_len as usize);
            let k_i = if pad_len > 0 {
                Tensor::<B, 3>::cat(
                    vec![
                        k_i,
                        Tensor::<B, 3>::zeros([pad_len, self.num_kv_heads, self.head_dim], &device),
                    ],
                    0,
                )
            } else {
                k_i
            };
            let v_i = if pad_len > 0 {
                Tensor::<B, 3>::cat(
                    vec![
                        v_i,
                        Tensor::<B, 3>::zeros([pad_len, self.num_kv_heads, self.head_dim], &device),
                    ],
                    0,
                )
            } else {
                v_i
            };

            k_batch.push(k_i.unsqueeze_dim::<4>(0));
            v_batch.push(v_i.unsqueeze_dim::<4>(0));
        }

        let k = Tensor::<B, 4>::cat(k_batch, 0);
        let v = Tensor::<B, 4>::cat(v_batch, 0);
        self.batched_decode_attention(q, &k, &v, context_lens)
    }

    fn batched_decode_attention<B: Backend>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 4>,
        v: &Tensor<B, 4>,
        context_lens: &[i32],
    ) -> Result<Tensor<B, 3>> {
        let _scope = Scope::new("decode_sdpa");
        let batch_size = q.shape().as_slice()[0];
        let kv_len = k.shape().as_slice()[1];
        let groups = self.num_heads / self.num_kv_heads;
        let qg = q
            .clone()
            .reshape([batch_size, self.num_kv_heads, groups, self.head_dim])
            .unsqueeze_dim::<5>(3);
        let kg = k.clone().swap_dims(1, 2).unsqueeze_dim::<5>(2);
        let vg = v.clone().swap_dims(1, 2).unsqueeze_dim::<5>(2);
        let mut attn = (qg * kg)
            .sum_dim(4)
            .squeeze_dim::<4>(4)
            .mul_scalar(self.scale);

        let mut mask = Vec::with_capacity(batch_size * kv_len);
        for &len in context_lens {
            let len = len as usize;
            mask.extend((0..kv_len).map(|idx| if idx < len { 0.0 } else { f32::NEG_INFINITY }));
        }
        let mask = Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(mask, [batch_size, 1, 1, kv_len]),
            &attn.device(),
        );
        attn = attn + mask;

        let attn = softmax(attn, 3).unsqueeze_dim::<5>(4);
        let out = (attn * vg).sum_dim(3).squeeze_dim::<4>(3);
        Ok(out.reshape([batch_size, self.num_heads, self.head_dim]))
    }

    fn gather_kv_from_cache<B: Backend>(
        &self,
        k_cache: &Tensor<B, 4>,
        v_cache: &Tensor<B, 4>,
        slot_indices: &Tensor<B, 1, Int>,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let cache_dims = k_cache.shape().as_slice().to_vec();
        let total_slots = cache_dims[0] * cache_dims[1];
        let seq_len = slot_indices.shape().as_slice()[0];
        let k_flat = k_cache
            .clone()
            .reshape([total_slots, self.num_kv_heads, self.head_dim]);
        let v_flat = v_cache
            .clone()
            .reshape([total_slots, self.num_kv_heads, self.head_dim]);

        let k = k_flat.select(0, slot_indices.clone()).reshape([
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]);
        let v = v_flat.select(0, slot_indices.clone()).reshape([
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]);
        Ok((k, v))
    }

    fn scaled_dot_product_attention<B: Backend>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
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
fn store_kvcache<B: Backend>(
    key: &Tensor<B, 3>,
    value: &Tensor<B, 3>,
    k_cache: &mut Tensor<B, 4>,
    v_cache: &mut Tensor<B, 4>,
    slot_mapping: &Tensor<B, 1, Int>,
) -> Result<()> {
    let dims = k_cache.shape().as_slice().to_vec();
    ensure!(dims.len() == 4, "k_cache must be rank-4");
    let total_slots = dims[0] * dims[1];
    let row_width = dims[2] * dims[3];

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

    let k_flat = k_flat.scatter(0, indices.clone(), key_flat, IndexingUpdateOp::Add);
    let v_flat = v_flat.scatter(0, indices, val_flat, IndexingUpdateOp::Add);

    *k_cache = k_flat.reshape([dims[0], dims[1], dims[2], dims[3]]);
    *v_cache = v_flat.reshape([dims[0], dims[1], dims[2], dims[3]]);
    Ok(())
}

fn create_causal_mask<B: Backend>(q_len: usize, kv_len: usize, device: &B::Device) -> Tensor<B, 3> {
    let offset = (kv_len as i64 - q_len as i64) as i32;
    let q_idx = Tensor::<B, 1, Int>::arange(0..q_len as i64, device)
        .reshape([q_len, 1])
        .repeat(&[1, kv_len]);
    let k_idx = Tensor::<B, 1, Int>::arange(0..kv_len as i64, device)
        .reshape([1, kv_len])
        .repeat(&[q_len, 1]);

    let allowed = k_idx.lower_equal(q_idx.add_scalar(offset));
    let mask = Tensor::<B, 2>::zeros([q_len, kv_len], device)
        .mask_fill(allowed.bool_not(), f32::NEG_INFINITY);
    mask.unsqueeze_dim::<3>(0)
}

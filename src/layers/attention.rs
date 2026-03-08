use anyhow::Result;
use burn::tensor::activation::softmax;
use burn::tensor::{backend::Backend, DType, Element, Int, Tensor};

use crate::engine::kv_cache::KvCache;
use crate::utils::context::AttentionContext;
use crate::utils::profiler::{self, Scope};

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
    pub fn forward<B: Backend<IntElem = i32>>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        kv_cache: &mut KvCache<B>,
        ctx: &AttentionContext<B>,
    ) -> Result<Tensor<B, 3>> {
        if ctx.is_prefill {
            let _scope = Scope::new("store_kvcache");
            kv_cache.store_prefill(k, v, &ctx.slot_mapping, &ctx.slot_mapping_host)?;
            profiler::sync_backend::<B>(kv_cache.device())?;
            drop(_scope);
            self.prefill_attention(q, k, v, kv_cache, ctx)
        } else {
            self.decode_attention(q, k, v, kv_cache, ctx)
        }
    }

    fn prefill_attention<B: Backend<IntElem = i32>>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        kv_cache: &KvCache<B>,
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
                let out = kv_cache.gather(
                    ctx.seq_ids[i],
                    &kv_slot_indices.expect("checked")[i],
                    k_end,
                )?;
                profiler::sync_backend::<B>(kv_cache.device())?;
                out
            } else {
                (
                    k.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                    v.clone()
                        .slice([k_start..k_end, 0..self.num_kv_heads, 0..self.head_dim]),
                )
            };

            let _scope = Scope::new("prefill_sdpa");
            let out = self.scaled_dot_product_attention(&q_i, &k_i, &v_i, true)?;
            profiler::sync_backend::<B>(&out.device())?;
            outputs.push(out);
            drop(_scope);
            debug_assert_eq!(q_len, outputs.last().unwrap().shape().as_slice()[0]);
        }

        Ok(Tensor::<B, 3>::cat(outputs, 0))
    }

    fn decode_attention<B: Backend<IntElem = i32>>(
        &self,
        q: &Tensor<B, 3>,
        k: &Tensor<B, 3>,
        v: &Tensor<B, 3>,
        kv_cache: &mut KvCache<B>,
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
        let last_block_ids = ctx
            .last_block_ids
            .as_ref()
            .expect("decode requires last_block_ids");
        let last_block_lens = ctx
            .last_block_lens
            .as_ref()
            .expect("decode requires last_block_lens");

        let batch_size = q.shape().as_slice()[0];
        if batch_size == 1 {
            let ctx_len = context_lens[0] as usize;
            debug_assert_eq!(kv_slot_indices[0].shape().as_slice()[0], ctx_len);
            let _scope = Scope::new("store_kvcache");
            let k_i = k
                .clone()
                .slice([0..1, 0..self.num_kv_heads, 0..self.head_dim])
                .reshape([self.num_kv_heads, self.head_dim]);
            let v_i = v
                .clone()
                .slice([0..1, 0..self.num_kv_heads, 0..self.head_dim])
                .reshape([self.num_kv_heads, self.head_dim]);
            kv_cache.stage_decode_token(
                ctx.seq_ids[0],
                last_block_ids[0],
                last_block_lens[0],
                k_i,
                v_i,
            )?;
            profiler::sync_backend::<B>(kv_cache.device())?;
            drop(_scope);
            let _scope = Scope::new("decode_gather_kv");
            let (k_i, v_i) = kv_cache.gather(ctx.seq_ids[0], &kv_slot_indices[0], ctx_len)?;
            profiler::sync_backend::<B>(kv_cache.device())?;
            drop(_scope);
            return self.scaled_dot_product_attention(q, &k_i, &v_i, false);
        }

        let max_ctx_len = context_lens.iter().copied().max().unwrap_or(0) as usize;
        let device = q.device();
        let mut k_batch = Vec::with_capacity(batch_size);
        let mut v_batch = Vec::with_capacity(batch_size);

        for (i, &ctx_len) in context_lens.iter().enumerate() {
            let _scope = Scope::new("store_kvcache");
            let k_i = k
                .clone()
                .slice([i..i + 1, 0..self.num_kv_heads, 0..self.head_dim])
                .reshape([self.num_kv_heads, self.head_dim]);
            let v_i = v
                .clone()
                .slice([i..i + 1, 0..self.num_kv_heads, 0..self.head_dim])
                .reshape([self.num_kv_heads, self.head_dim]);
            kv_cache.stage_decode_token(
                ctx.seq_ids[i],
                last_block_ids[i],
                last_block_lens[i],
                k_i,
                v_i,
            )?;
            profiler::sync_backend::<B>(kv_cache.device())?;
            drop(_scope);
            debug_assert_eq!(kv_slot_indices[i].shape().as_slice()[0], ctx_len as usize);
            let _scope = Scope::new("decode_gather_kv");
            let (k_i, v_i) =
                kv_cache.gather(ctx.seq_ids[i], &kv_slot_indices[i], ctx_len as usize)?;
            profiler::sync_backend::<B>(kv_cache.device())?;
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

    fn batched_decode_attention<B: Backend<IntElem = i32>>(
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
        )
        .cast(DType::F32);
        let native_dtype = <B::FloatElem as Element>::dtype();
        attn = attn.cast(DType::F32) + mask;

        let attn = softmax(attn, 3).cast(native_dtype).unsqueeze_dim::<5>(4);
        let out = (attn * vg).sum_dim(3).squeeze_dim::<4>(3);
        profiler::sync_backend::<B>(&out.device())?;
        Ok(out.reshape([batch_size, self.num_heads, self.head_dim]))
    }

    fn scaled_dot_product_attention<B: Backend<IntElem = i32>>(
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
        let native_dtype = <B::FloatElem as Element>::dtype();
        attn = attn.cast(DType::F32);

        if causal && q_len > 1 {
            let mask = create_causal_mask::<B>(q_len, kv_len, &attn.device()).cast(DType::F32);
            attn = attn + mask;
        }

        let attn = softmax(attn, 2).cast(native_dtype);
        let out = attn.matmul(vh); // [h, q, d]
        profiler::sync_backend::<B>(&out.device())?;
        Ok(out.swap_dims(0, 1)) // [q, h, d]
    }
}

fn create_causal_mask<B: Backend<IntElem = i32>>(
    q_len: usize,
    kv_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
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

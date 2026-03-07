use anyhow::{Context, Result, ensure};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dispatch::Dispatch;

use crate::utils::context::AttentionContext;

/// Paged attention with KV cache (reference implementation, correctness-first).
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

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: &Tensor<Dispatch, 3>,
        k: &Tensor<Dispatch, 3>,
        v: &Tensor<Dispatch, 3>,
        k_cache: &mut Tensor<Dispatch, 4>,
        v_cache: &mut Tensor<Dispatch, 4>,
        ctx: &AttentionContext,
    ) -> Result<Tensor<Dispatch, 3>> {
        let device = q.device();
        let q_dims = q.shape().as_slice().to_vec();
        ensure!(q_dims.len() == 3, "q must be rank-3");
        let n = q_dims[0];
        ensure!(q_dims[1] == self.num_heads, "q num_heads mismatch");
        ensure!(q_dims[2] == self.head_dim, "q head_dim mismatch");

        let k_dims = k.shape().as_slice().to_vec();
        let v_dims = v.shape().as_slice().to_vec();
        ensure!(k_dims == v_dims, "k/v shape mismatch");
        ensure!(k_dims.len() == 3, "k must be rank-3");

        let mut k_cache_data = k_cache
            .to_data()
            .to_vec::<f32>()
            .context("failed to read k_cache")?;
        let mut v_cache_data = v_cache
            .to_data()
            .to_vec::<f32>()
            .context("failed to read v_cache")?;

        let cache_dims = k_cache.shape().as_slice().to_vec();
        ensure!(cache_dims.len() == 4, "k_cache must be rank-4");
        let num_blocks = cache_dims[0];
        let block_size = cache_dims[1];
        let kv_heads = cache_dims[2];
        let hd = cache_dims[3];
        ensure!(
            kv_heads == self.num_kv_heads && hd == self.head_dim,
            "cache dims mismatch"
        );

        let key_data = k.to_data().to_vec::<f32>().context("failed to read k")?;
        let val_data = v.to_data().to_vec::<f32>().context("failed to read v")?;
        let slot_mapping = to_i32_vec_1d(&ctx.slot_mapping)?;

        // Store new kv into paged cache.
        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            if slot >= num_blocks * block_size {
                continue;
            }
            let block = slot / block_size;
            let offset = slot % block_size;
            for kvh in 0..self.num_kv_heads {
                for d in 0..self.head_dim {
                    let src = ((token_idx * self.num_kv_heads + kvh) * self.head_dim) + d;
                    let dst =
                        (((block * block_size + offset) * self.num_kv_heads + kvh) * self.head_dim)
                            + d;
                    k_cache_data[dst] = key_data[src];
                    v_cache_data[dst] = val_data[src];
                }
            }
        }

        *k_cache = Tensor::<Dispatch, 4>::from_data(
            TensorData::new(k_cache_data.clone(), [num_blocks, block_size, kv_heads, hd]),
            &device,
        );
        *v_cache = Tensor::<Dispatch, 4>::from_data(
            TensorData::new(v_cache_data.clone(), [num_blocks, block_size, kv_heads, hd]),
            &device,
        );

        let q_data = q.to_data().to_vec::<f32>().context("failed to read q")?;
        let mut out = Vec::<f32>::new();

        if ctx.is_prefill {
            let cu_q = to_i32_vec_1d(&ctx.cu_seqlens_q)?;
            let cu_k = to_i32_vec_1d(&ctx.cu_seqlens_k)?;
            let has_prefix_cache = ctx.block_tables.is_some();
            let batch = cu_q.len().saturating_sub(1);
            for i in 0..batch {
                let q_start = cu_q[i] as usize;
                let q_end = cu_q[i + 1] as usize;
                let q_len = q_end.saturating_sub(q_start);
                let k_start = cu_k[i] as usize;
                let k_end = cu_k[i + 1] as usize;
                let k_len = k_end.saturating_sub(k_start);

                let q_i = &q_data[q_start * self.num_heads * self.head_dim
                    ..q_end * self.num_heads * self.head_dim];
                let (k_i, v_i) = if has_prefix_cache {
                    gather_kv_from_cache(
                        &k_cache_data,
                        &v_cache_data,
                        cache_dims.as_slice(),
                        self.num_kv_heads,
                        self.head_dim,
                        ctx.block_tables.as_ref().expect("checked"),
                        i,
                        k_len,
                    )?
                } else {
                    let from = k_start * self.num_kv_heads * self.head_dim;
                    let to = k_end * self.num_kv_heads * self.head_dim;
                    (key_data[from..to].to_vec(), val_data[from..to].to_vec())
                };
                let out_i =
                    sdpa(q_i, &k_i, &v_i, q_len, k_len, self.num_heads, self.num_kv_heads, self.head_dim, self.scale, true);
                out.extend_from_slice(&out_i);
            }
        } else {
            let context_lens = to_i32_vec_1d(ctx.context_lens.as_ref().expect("decode context_lens"))?;
            for (i, &ctx_len) in context_lens.iter().enumerate() {
                let q_from = i * self.num_heads * self.head_dim;
                let q_to = q_from + self.num_heads * self.head_dim;
                let q_i = &q_data[q_from..q_to];
                let (k_i, v_i) = gather_kv_from_cache(
                    &k_cache_data,
                    &v_cache_data,
                    cache_dims.as_slice(),
                    self.num_kv_heads,
                    self.head_dim,
                    ctx.block_tables.as_ref().expect("decode block_tables"),
                    i,
                    ctx_len as usize,
                )?;
                let out_i =
                    sdpa(q_i, &k_i, &v_i, 1, ctx_len as usize, self.num_heads, self.num_kv_heads, self.head_dim, self.scale, false);
                out.extend_from_slice(&out_i);
            }
        }

        Ok(Tensor::<Dispatch, 3>::from_data(
            TensorData::new(out, [n, self.num_heads, self.head_dim]),
            &device,
        ))
    }
}

fn to_i32_vec_1d(t: &Tensor<Dispatch, 1, Int>) -> Result<Vec<i32>> {
    Ok(t.to_data().to_vec::<i32>()?)
}

fn to_i32_vec_row(t: &Tensor<Dispatch, 2, Int>, row: usize) -> Result<Vec<i32>> {
    let shape = t.shape();
    let dims = shape.as_slice();
    ensure!(dims.len() == 2, "block_tables must be rank-2");
    let rows = dims[0];
    let cols = dims[1];
    ensure!(row < rows, "row out of range");
    let data = t.to_data().to_vec::<i32>()?;
    let from = row * cols;
    let to = from + cols;
    Ok(data[from..to].to_vec())
}

fn gather_kv_from_cache(
    k_cache: &[f32],
    v_cache: &[f32],
    cache_dims: &[usize],
    num_kv_heads: usize,
    head_dim: usize,
    block_tables: &Tensor<Dispatch, 2, Int>,
    seq_idx: usize,
    seq_len: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let block_size = cache_dims[1];
    let blocks_needed = seq_len.div_ceil(block_size);
    let row = to_i32_vec_row(block_tables, seq_idx)?;
    let mut k_out = vec![0f32; seq_len * num_kv_heads * head_dim];
    let mut v_out = vec![0f32; seq_len * num_kv_heads * head_dim];
    for t in 0..seq_len {
        let block_idx = t / block_size;
        let offset = t % block_size;
        let block_id = row
            .get(block_idx)
            .copied()
            .unwrap_or_default()
            .max(0) as usize;
        if block_idx >= blocks_needed {
            continue;
        }
        for kvh in 0..num_kv_heads {
            for d in 0..head_dim {
                let src = (((block_id * block_size + offset) * num_kv_heads + kvh) * head_dim) + d;
                let dst = ((t * num_kv_heads + kvh) * head_dim) + d;
                k_out[dst] = k_cache[src];
                v_out[dst] = v_cache[src];
            }
        }
    }
    Ok((k_out, v_out))
}

#[allow(clippy::too_many_arguments)]
fn sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; q_len * num_heads * head_dim];
    let groups = num_heads / num_kv_heads.max(1);
    let offset = kv_len as isize - q_len as isize;

    for qi in 0..q_len {
        for h in 0..num_heads {
            let kvh = h / groups.max(1);

            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for kj in 0..kv_len {
                if causal {
                    let max_k = offset + qi as isize;
                    if kj as isize > max_k {
                        continue;
                    }
                }
                let mut dot = 0f32;
                for d in 0..head_dim {
                    let qv = q[(qi * num_heads + h) * head_dim + d];
                    let kv = k[(kj * num_kv_heads + kvh) * head_dim + d];
                    dot += qv * kv;
                }
                scores[kj] = dot * scale;
            }

            let max_score = scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
            }

            for d in 0..head_dim {
                let mut acc = 0f32;
                for kj in 0..kv_len {
                    let vv = v[(kj * num_kv_heads + kvh) * head_dim + d];
                    acc += scores[kj] * vv;
                }
                out[(qi * num_heads + h) * head_dim + d] = acc;
            }
        }
    }

    out
}

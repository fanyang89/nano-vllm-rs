use std::collections::HashMap;

use anyhow::{ensure, Result};
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor};

#[derive(Debug)]
struct TailState<B: Backend> {
    block_id: usize,
    len: usize,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
}

pub struct KvCache<B: Backend> {
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: B::Device,
    tails: HashMap<usize, TailState<B>>,
}

impl<B: Backend<IntElem = i32>> KvCache<B> {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let total_slots = num_blocks * block_size;
        Self {
            k: Tensor::<B, 3>::zeros([total_slots, num_kv_heads, head_dim], device),
            v: Tensor::<B, 3>::zeros([total_slots, num_kv_heads, head_dim], device),
            block_size,
            num_kv_heads,
            head_dim,
            device: device.clone(),
            tails: HashMap::new(),
        }
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    pub fn store_prefill(
        &mut self,
        key: &Tensor<B, 3>,
        value: &Tensor<B, 3>,
        slot_mapping: &Tensor<B, 1, Int>,
        slot_mapping_host: &[i32],
    ) -> Result<()> {
        let total_slots = self.k.shape().as_slice()[0];
        let n = key.shape().as_slice()[0];

        if n == 0 {
            return Ok(());
        }

        if n == 1 && slot_mapping_host.len() == 1 {
            let slot = slot_mapping_host[0];
            ensure!(slot >= 0, "decode slot must be non-negative");
            let slot = slot as usize;
            ensure!(slot < total_slots, "decode slot out of bounds");

            self.k = self.k.clone().slice_assign(
                [slot..slot + 1, 0..self.num_kv_heads, 0..self.head_dim],
                key.clone(),
            );
            self.v = self.v.clone().slice_assign(
                [slot..slot + 1, 0..self.num_kv_heads, 0..self.head_dim],
                value.clone(),
            );
            return Ok(());
        }

        let mut slots = slot_mapping.clone();
        let valid_mask = slots.clone().greater_equal_elem(0);
        slots = slots.clamp(0, (total_slots.saturating_sub(1)) as i32);

        let indices = slots;
        let valid = valid_mask
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .repeat(&[1, self.num_kv_heads, self.head_dim])
            .float();

        let key = key.clone().mul(valid.clone());
        let value = value.clone().mul(valid);

        self.k = self
            .k
            .clone()
            .select_assign(0, indices.clone(), key, IndexingUpdateOp::Add);
        self.v = self
            .v
            .clone()
            .select_assign(0, indices, value, IndexingUpdateOp::Add);
        Ok(())
    }

    pub fn stage_decode_token(
        &mut self,
        seq_id: usize,
        block_id: usize,
        block_len: usize,
        key: Tensor<B, 2>,
        value: Tensor<B, 2>,
    ) -> Result<()> {
        ensure!(
            block_len >= 1 && block_len <= self.block_size,
            "invalid block len"
        );

        let key = key.unsqueeze_dim::<3>(0);
        let value = value.unsqueeze_dim::<3>(0);

        let maybe_prev = self.tails.remove(&seq_id);
        let next = if let Some(prev) = maybe_prev {
            if prev.block_id == block_id && prev.len + 1 == block_len {
                let pos = prev.len;
                TailState {
                    block_id,
                    len: block_len,
                    k: prev
                        .k
                        .slice_assign([pos..pos + 1, 0..self.num_kv_heads, 0..self.head_dim], key),
                    v: prev.v.slice_assign(
                        [pos..pos + 1, 0..self.num_kv_heads, 0..self.head_dim],
                        value,
                    ),
                }
            } else {
                self.init_tail_from_frozen(block_id, block_len, key, value)?
            }
        } else {
            self.init_tail_from_frozen(block_id, block_len, key, value)?
        };

        if next.len == self.block_size {
            let start = block_id * self.block_size;
            self.k = self.k.clone().slice_assign(
                [
                    start..start + self.block_size,
                    0..self.num_kv_heads,
                    0..self.head_dim,
                ],
                next.k,
            );
            self.v = self.v.clone().slice_assign(
                [
                    start..start + self.block_size,
                    0..self.num_kv_heads,
                    0..self.head_dim,
                ],
                next.v,
            );
        } else {
            self.tails.insert(seq_id, next);
        }

        Ok(())
    }

    fn init_tail_from_frozen(
        &self,
        block_id: usize,
        block_len: usize,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
    ) -> Result<TailState<B>> {
        let existing = block_len - 1;
        let base = block_id * self.block_size;
        let mut k = Tensor::<B, 3>::zeros(
            [self.block_size, self.num_kv_heads, self.head_dim],
            &self.device,
        );
        let mut v = Tensor::<B, 3>::zeros(
            [self.block_size, self.num_kv_heads, self.head_dim],
            &self.device,
        );

        if existing > 0 {
            let prefix =
                Tensor::<B, 1, Int>::arange(base as i64..(base + existing) as i64, &self.device);
            let k_prefix = self.k.clone().select(0, prefix.clone());
            let v_prefix = self.v.clone().select(0, prefix);
            k = k.slice_assign(
                [0..existing, 0..self.num_kv_heads, 0..self.head_dim],
                k_prefix,
            );
            v = v.slice_assign(
                [0..existing, 0..self.num_kv_heads, 0..self.head_dim],
                v_prefix,
            );
        }
        k = k.slice_assign(
            [
                existing..existing + 1,
                0..self.num_kv_heads,
                0..self.head_dim,
            ],
            key,
        );
        v = v.slice_assign(
            [
                existing..existing + 1,
                0..self.num_kv_heads,
                0..self.head_dim,
            ],
            value,
        );

        Ok(TailState {
            block_id,
            len: block_len,
            k,
            v,
        })
    }

    pub fn gather(
        &self,
        seq_id: usize,
        slot_indices: &Tensor<B, 1, Int>,
        seq_len: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        if let Some(tail) = self.tails.get(&seq_id) {
            let prefix_len = seq_len.saturating_sub(tail.len);
            if prefix_len == 0 {
                return Ok((
                    tail.k
                        .clone()
                        .slice([0..tail.len, 0..self.num_kv_heads, 0..self.head_dim]),
                    tail.v
                        .clone()
                        .slice([0..tail.len, 0..self.num_kv_heads, 0..self.head_dim]),
                ));
            }

            let prefix_slots = slot_indices.clone().slice([0..prefix_len]);
            let k = Tensor::<B, 3>::cat(
                vec![
                    self.k.clone().select(0, prefix_slots.clone()),
                    tail.k
                        .clone()
                        .slice([0..tail.len, 0..self.num_kv_heads, 0..self.head_dim]),
                ],
                0,
            );
            let v = Tensor::<B, 3>::cat(
                vec![
                    self.v.clone().select(0, prefix_slots),
                    tail.v
                        .clone()
                        .slice([0..tail.len, 0..self.num_kv_heads, 0..self.head_dim]),
                ],
                0,
            );
            return Ok((k, v));
        }

        Ok((
            self.k.clone().select(0, slot_indices.clone()),
            self.v.clone().select(0, slot_indices.clone()),
        ))
    }

    pub fn gather_padded_batch(
        &self,
        seq_ids: &[usize],
        slot_indices: &[Tensor<B, 1, Int>],
        context_lens: &[i32],
        max_ctx_len: usize,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        if context_lens.iter().all(|&len| len as usize == max_ctx_len) {
            let mut k_batch = Vec::with_capacity(seq_ids.len());
            let mut v_batch = Vec::with_capacity(seq_ids.len());
            let mut all_tail_only = true;

            for (&seq_id, &ctx_len_i32) in seq_ids.iter().zip(context_lens.iter()) {
                let ctx_len = ctx_len_i32 as usize;
                let Some(tail) = self.tails.get(&seq_id) else {
                    all_tail_only = false;
                    break;
                };
                if tail.len != ctx_len {
                    all_tail_only = false;
                    break;
                }

                k_batch.push(
                    tail.k
                        .clone()
                        .slice([0..ctx_len, 0..self.num_kv_heads, 0..self.head_dim])
                        .unsqueeze_dim::<4>(0),
                );
                v_batch.push(
                    tail.v
                        .clone()
                        .slice([0..ctx_len, 0..self.num_kv_heads, 0..self.head_dim])
                        .unsqueeze_dim::<4>(0),
                );
            }

            if all_tail_only {
                return Ok((
                    Tensor::<B, 4>::cat(k_batch, 0),
                    Tensor::<B, 4>::cat(v_batch, 0),
                ));
            }
        }

        let batch = seq_ids.len();
        let mut k_batch = Tensor::<B, 4>::zeros(
            [batch, max_ctx_len, self.num_kv_heads, self.head_dim],
            &self.device,
        );
        let mut v_batch = Tensor::<B, 4>::zeros(
            [batch, max_ctx_len, self.num_kv_heads, self.head_dim],
            &self.device,
        );

        for (i, (&seq_id, &ctx_len_i32)) in seq_ids.iter().zip(context_lens.iter()).enumerate() {
            let ctx_len = ctx_len_i32 as usize;
            if ctx_len == 0 {
                continue;
            }

            let (k_i, v_i) = self.gather(seq_id, &slot_indices[i], ctx_len)?;
            k_batch = k_batch.slice_assign(
                [i..i + 1, 0..ctx_len, 0..self.num_kv_heads, 0..self.head_dim],
                k_i.unsqueeze_dim::<4>(0),
            );
            v_batch = v_batch.slice_assign(
                [i..i + 1, 0..ctx_len, 0..self.num_kv_heads, 0..self.head_dim],
                v_i.unsqueeze_dim::<4>(0),
            );
        }

        Ok((k_batch, v_batch))
    }

    pub fn clear_sequence(&mut self, seq_id: usize) {
        self.tails.remove(&seq_id);
    }
}

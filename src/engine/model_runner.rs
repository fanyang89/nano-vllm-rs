use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dispatch::{Dispatch, DispatchDevice};

use crate::config::{EngineConfig, ModelConfig};
use crate::engine::sequence::Sequence;
use crate::layers::sampler;
use crate::model::qwen3::Qwen3ForCausalLM;
use crate::utils::context::AttentionContext;

pub struct ModelRunner {
    model: Qwen3ForCausalLM,
    k_caches: Vec<Tensor<Dispatch, 4>>,
    v_caches: Vec<Tensor<Dispatch, 4>>,
    block_size: usize,
    device: DispatchDevice,
}

impl ModelRunner {
    pub fn new(
        engine_config: &mut EngineConfig,
        model_config: &ModelConfig,
        device: DispatchDevice,
    ) -> Result<Self> {
        let model_path = &engine_config.model_path;
        let model = Qwen3ForCausalLM::new(model_config, model_path, &device)?;

        let (k_caches, v_caches, num_blocks) =
            Self::allocate_kv_cache(engine_config, model_config, &model, &device)?;
        engine_config.num_kvcache_blocks = num_blocks;

        Ok(Self {
            model,
            k_caches,
            v_caches,
            block_size: engine_config.kvcache_block_size,
            device,
        })
    }

    fn allocate_kv_cache(
        engine_config: &EngineConfig,
        model_config: &ModelConfig,
        model: &Qwen3ForCausalLM,
        device: &DispatchDevice,
    ) -> Result<(Vec<Tensor<Dispatch, 4>>, Vec<Tensor<Dispatch, 4>>, usize)> {
        let num_layers = model.num_layers();
        let num_kv_heads = model_config.num_key_value_heads;
        let head_dim = model_config.head_dim();
        let block_size = engine_config.kvcache_block_size;

        let max_blocks_per_seq = engine_config.max_model_len.div_ceil(block_size);
        let num_blocks = (max_blocks_per_seq * engine_config.max_num_seqs).min(1024);

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_caches.push(Tensor::<Dispatch, 4>::zeros(
                [num_blocks, block_size, num_kv_heads, head_dim],
                device,
            ));
            v_caches.push(Tensor::<Dispatch, 4>::zeros(
                [num_blocks, block_size, num_kv_heads, head_dim],
                device,
            ));
        }

        Ok((k_caches, v_caches, num_blocks))
    }

    fn int_tensor_1d(&self, data: Vec<i32>) -> Tensor<Dispatch, 1, Int> {
        let n = data.len();
        Tensor::<Dispatch, 1, Int>::from_data(TensorData::new(data, [n]), &self.device)
    }

    /// Prepare inputs for prefill phase.
    fn prepare_prefill(
        &self,
        seqs: &[&Sequence],
    ) -> Result<(
        Tensor<Dispatch, 1, Int>,
        Tensor<Dispatch, 1, Int>,
        AttentionContext,
    )> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0i32];
        let mut cu_seqlens_k = vec![0i32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping = Vec::new();
        let mut need_block_tables = false;

        for seq in seqs {
            let seqlen = seq.len();
            let cached = seq.num_cached_tokens;

            for &t in seq.uncached_token_ids() {
                input_ids.push(t as i32);
            }
            for pos in cached..seqlen {
                positions.push(pos as i32);
            }

            let seqlen_q = seqlen - cached;
            let seqlen_k = seqlen;
            cu_seqlens_q.push(cu_seqlens_q.last().copied().unwrap_or(0) + seqlen_q as i32);
            cu_seqlens_k.push(cu_seqlens_k.last().copied().unwrap_or(0) + seqlen_k as i32);
            max_seqlen_q = max_seqlen_q.max(seqlen_q);
            max_seqlen_k = max_seqlen_k.max(seqlen_k);

            if cached > 0 {
                need_block_tables = true;
            }

            if seq.block_table.is_empty() {
                continue;
            }
            for i in seq.num_cached_blocks()..seq.num_blocks() {
                let start = seq.block_table[i] * self.block_size;
                let end = if i != seq.num_blocks() - 1 {
                    start + self.block_size
                } else {
                    start + seq.last_block_num_tokens()
                };
                for slot in start..end {
                    slot_mapping.push(slot as i32);
                }
            }
        }

        let block_tables = if need_block_tables {
            Some(self.prepare_block_tables(seqs)?)
        } else {
            None
        };

        let input_ids = self.int_tensor_1d(input_ids);
        let positions = self.int_tensor_1d(positions);
        let slot_mapping = self.int_tensor_1d(slot_mapping);
        let cu_seqlens_q = self.int_tensor_1d(cu_seqlens_q);
        let cu_seqlens_k = self.int_tensor_1d(cu_seqlens_k);

        let ctx = AttentionContext {
            is_prefill: true,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            context_lens: None,
            block_tables,
        };

        Ok((input_ids, positions, ctx))
    }

    /// Prepare inputs for decode phase.
    fn prepare_decode(
        &self,
        seqs: &[&Sequence],
    ) -> Result<(
        Tensor<Dispatch, 1, Int>,
        Tensor<Dispatch, 1, Int>,
        AttentionContext,
    )> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();

        for seq in seqs {
            input_ids.push(seq.last_token() as i32);
            positions.push((seq.len() - 1) as i32);
            context_lens.push(seq.len() as i32);
            let last_slot =
                seq.block_table.last().unwrap() * self.block_size + seq.last_block_num_tokens() - 1;
            slot_mapping.push(last_slot as i32);
        }

        let n = input_ids.len();
        let block_tables = self.prepare_block_tables(seqs)?;

        let input_ids = self.int_tensor_1d(input_ids);
        let positions = self.int_tensor_1d(positions);
        let slot_mapping = self.int_tensor_1d(slot_mapping);
        let context_lens_t = self.int_tensor_1d(context_lens);

        let cu_seqlens_q: Vec<i32> = (0..=n as i32).collect();
        let cu_seqlens_k = cu_seqlens_q.clone();
        let cu_seqlens_q = self.int_tensor_1d(cu_seqlens_q);
        let cu_seqlens_k = self.int_tensor_1d(cu_seqlens_k);

        let ctx = AttentionContext {
            is_prefill: false,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q: 1,
            max_seqlen_k: 1,
            slot_mapping,
            context_lens: Some(context_lens_t),
            block_tables: Some(block_tables),
        };

        Ok((input_ids, positions, ctx))
    }

    fn prepare_block_tables(&self, seqs: &[&Sequence]) -> Result<Tensor<Dispatch, 2, Int>> {
        let max_len = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(0);
        let mut data = Vec::new();
        for seq in seqs {
            for &bid in &seq.block_table {
                data.push(bid as i32);
            }
            for _ in seq.block_table.len()..max_len {
                data.push(0);
            }
        }
        let n = seqs.len();
        Ok(Tensor::<Dispatch, 2, Int>::from_data(
            TensorData::new(data, [n, max_len]),
            &self.device,
        ))
    }

    fn prepare_temperatures(&self, seqs: &[&Sequence]) -> Tensor<Dispatch, 1> {
        let temps: Vec<f32> = seqs.iter().map(|s| s.temperature).collect();
        let n = temps.len();
        Tensor::<Dispatch, 1>::from_data(TensorData::new(temps, [n]), &self.device)
    }

    pub fn run(&mut self, seqs: &[&Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, ctx) = if is_prefill {
            self.prepare_prefill(seqs)?
        } else {
            self.prepare_decode(seqs)?
        };

        let temperatures = self.prepare_temperatures(seqs);
        let hidden_states = self.model.forward(
            &input_ids,
            &positions,
            &mut self.k_caches,
            &mut self.v_caches,
            &ctx,
        )?;

        let logits = self.model.compute_logits(&hidden_states, &ctx)?;
        let do_sample = seqs.first().is_none_or(|s| s.do_sample);
        debug_assert!(seqs.iter().all(|s| s.do_sample == do_sample));
        sampler::sample(&logits, &temperatures, do_sample)
    }
}

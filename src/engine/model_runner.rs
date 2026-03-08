use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::config::{EngineConfig, ModelConfig};
use crate::engine::kv_cache::KvCache;
use crate::engine::sequence::Sequence;
use crate::layers::sampler;
use crate::model::qwen3::Qwen3ForCausalLM;
use crate::utils::context::AttentionContext;
use crate::utils::profiler::{self, Scope};

pub struct ModelRunner<B: Backend<IntElem = i32>> {
    model: Qwen3ForCausalLM<B>,
    kv_caches: Vec<KvCache<B>>,
    block_size: usize,
    device: B::Device,
}

impl<B: Backend<IntElem = i32>> ModelRunner<B> {
    pub fn new(
        engine_config: &mut EngineConfig,
        model_config: &ModelConfig,
        device: B::Device,
    ) -> Result<Self> {
        let model_path = &engine_config.model_path;
        let model = Qwen3ForCausalLM::<B>::new(model_config, model_path, &device)?;

        let (kv_caches, num_blocks) =
            Self::allocate_kv_cache(engine_config, model_config, &model, &device)?;
        engine_config.num_kvcache_blocks = num_blocks;

        Ok(Self {
            model,
            kv_caches,
            block_size: engine_config.kvcache_block_size,
            device,
        })
    }

    fn allocate_kv_cache(
        engine_config: &EngineConfig,
        model_config: &ModelConfig,
        model: &Qwen3ForCausalLM<B>,
        device: &B::Device,
    ) -> Result<(Vec<KvCache<B>>, usize)> {
        let num_layers = model.num_layers();
        let num_kv_heads = model_config.num_key_value_heads;
        let head_dim = model_config.head_dim();
        let block_size = engine_config.kvcache_block_size;

        let max_blocks_per_seq = engine_config.max_model_len.div_ceil(block_size);
        let num_blocks = (max_blocks_per_seq * engine_config.max_num_seqs).min(1024);

        let mut kv_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            kv_caches.push(KvCache::new(
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                device,
            ));
        }

        Ok((kv_caches, num_blocks))
    }

    fn int_tensor_1d(&self, data: Vec<i32>) -> Tensor<B, 1, Int> {
        let n = data.len();
        Tensor::<B, 1, Int>::from_data(TensorData::new(data, [n]), &self.device)
    }

    fn seq_kv_slots(&self, seq: &Sequence, len: usize) -> Vec<i32> {
        let mut slots = Vec::with_capacity(len);
        let full_blocks = len / self.block_size;
        let tail = len % self.block_size;

        for &block_id in seq.block_table.iter().take(full_blocks) {
            let base = block_id * self.block_size;
            slots.extend((base..base + self.block_size).map(|slot| slot as i32));
        }

        if tail > 0 {
            let block_id = seq.block_table[full_blocks];
            let base = block_id * self.block_size;
            slots.extend((base..base + tail).map(|slot| slot as i32));
        }

        slots
    }

    fn prepare_prefill(
        &self,
        seqs: &[&Sequence],
    ) -> Result<(Tensor<B, 1, Int>, Tensor<B, 1, Int>, AttentionContext<B>)> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0i32];
        let mut cu_seqlens_k = vec![0i32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping = Vec::new();
        let mut need_block_tables = false;
        let mut kv_slot_indices = Vec::new();

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

            kv_slot_indices.push(self.int_tensor_1d(self.seq_kv_slots(seq, seqlen)));

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

        let cu_seqlens_q_host = cu_seqlens_q.clone();
        let cu_seqlens_k_host = cu_seqlens_k.clone();
        let input_ids = self.int_tensor_1d(input_ids);
        let positions = self.int_tensor_1d(positions);
        let slot_mapping_host = slot_mapping.clone();
        let slot_mapping = self.int_tensor_1d(slot_mapping);
        let cu_seqlens_q = self.int_tensor_1d(cu_seqlens_q);
        let cu_seqlens_k = self.int_tensor_1d(cu_seqlens_k);

        let ctx = AttentionContext {
            is_prefill: true,
            seq_ids: seqs.iter().map(|seq| seq.seq_id).collect(),
            cu_seqlens_q,
            cu_seqlens_q_host,
            cu_seqlens_k,
            cu_seqlens_k_host,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            slot_mapping_host,
            context_lens: None,
            context_lens_host: None,
            last_block_ids: None,
            last_block_lens: None,
            kv_slot_indices: need_block_tables.then_some(kv_slot_indices),
        };

        Ok((input_ids, positions, ctx))
    }

    fn prepare_decode(
        &self,
        seqs: &[&Sequence],
    ) -> Result<(Tensor<B, 1, Int>, Tensor<B, 1, Int>, AttentionContext<B>)> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();
        let mut kv_slot_indices = Vec::with_capacity(seqs.len());
        let mut last_block_ids = Vec::with_capacity(seqs.len());
        let mut last_block_lens = Vec::with_capacity(seqs.len());

        for seq in seqs {
            input_ids.push(seq.last_token() as i32);
            positions.push((seq.len() - 1) as i32);
            context_lens.push(seq.len() as i32);
            kv_slot_indices.push(self.int_tensor_1d(self.seq_kv_slots(seq, seq.len())));
            let last_block_id = *seq.block_table.last().unwrap();
            let last_block_len = seq.last_block_num_tokens();
            last_block_ids.push(last_block_id);
            last_block_lens.push(last_block_len);
            let last_slot = last_block_id * self.block_size + last_block_len - 1;
            slot_mapping.push(last_slot as i32);
        }

        let n = input_ids.len();
        let input_ids = self.int_tensor_1d(input_ids);
        let positions = self.int_tensor_1d(positions);
        let slot_mapping_host = slot_mapping.clone();
        let slot_mapping = self.int_tensor_1d(slot_mapping);
        let context_lens_t = self.int_tensor_1d(context_lens.clone());

        let cu_seqlens_q: Vec<i32> = (0..=n as i32).collect();
        let cu_seqlens_k = cu_seqlens_q.clone();
        let cu_seqlens_q_host = cu_seqlens_q.clone();
        let cu_seqlens_k_host = cu_seqlens_k.clone();
        let cu_seqlens_q = self.int_tensor_1d(cu_seqlens_q);
        let cu_seqlens_k = self.int_tensor_1d(cu_seqlens_k);

        let ctx = AttentionContext {
            is_prefill: false,
            seq_ids: seqs.iter().map(|seq| seq.seq_id).collect(),
            cu_seqlens_q,
            cu_seqlens_q_host,
            cu_seqlens_k,
            cu_seqlens_k_host,
            max_seqlen_q: 1,
            max_seqlen_k: 1,
            slot_mapping,
            slot_mapping_host,
            context_lens: Some(context_lens_t),
            context_lens_host: Some(context_lens),
            last_block_ids: Some(last_block_ids),
            last_block_lens: Some(last_block_lens),
            kv_slot_indices: Some(kv_slot_indices),
        };

        Ok((input_ids, positions, ctx))
    }

    fn prepare_temperatures(&self, seqs: &[&Sequence]) -> Tensor<B, 1> {
        let temps: Vec<f32> = seqs.iter().map(|s| s.temperature).collect();
        let n = temps.len();
        Tensor::<B, 1>::from_data(TensorData::new(temps, [n]), &self.device)
    }

    pub fn run(&mut self, seqs: &[&Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, ctx) = if is_prefill {
            let _scope = Scope::new("prepare_prefill");
            self.prepare_prefill(seqs)?
        } else {
            let _scope = Scope::new("prepare_decode");
            self.prepare_decode(seqs)?
        };

        let _scope = Scope::new("model_forward");
        let hidden_states =
            self.model
                .forward(&input_ids, &positions, &mut self.kv_caches, &ctx)?;
        profiler::sync_backend::<B>(&self.device)?;

        drop(_scope);
        let _scope = Scope::new("compute_logits");
        let logits = self.model.compute_logits(&hidden_states, &ctx)?;
        profiler::sync_backend::<B>(&self.device)?;
        let do_sample = seqs.first().is_none_or(|s| s.do_sample);
        debug_assert!(seqs.iter().all(|s| s.do_sample == do_sample));
        drop(_scope);
        let _scope = Scope::new("sample");
        let temperatures = do_sample.then(|| self.prepare_temperatures(seqs));
        let tokens = sampler::sample(&logits, temperatures.as_ref(), do_sample)?;
        profiler::sync_backend::<B>(&self.device)?;
        Ok(tokens)
    }

    pub fn clear_sequences(&mut self, seq_ids: &[usize]) {
        for &seq_id in seq_ids {
            for cache in &mut self.kv_caches {
                cache.clear_sequence(seq_id);
            }
        }
    }
}

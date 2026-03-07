use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::{EngineConfig, ModelConfig};
use crate::engine::sequence::Sequence;
use crate::layers::sampler;
use crate::model::qwen3::Qwen3ForCausalLM;
use crate::utils::context::AttentionContext;

pub struct ModelRunner {
    model: Qwen3ForCausalLM,
    k_caches: Vec<Tensor>,
    v_caches: Vec<Tensor>,
    block_size: usize,
    device: Device,
}

impl ModelRunner {
    fn select_dtype(device: &Device) -> DType {
        match device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        }
    }

    pub fn new(
        engine_config: &mut EngineConfig,
        model_config: &ModelConfig,
        device: Device,
    ) -> Result<Self> {
        let dtype = Self::select_dtype(&device);
        let model_path = &engine_config.model_path;

        // Load model weights from safetensors
        let safetensors_files: Vec<_> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)?
        };

        let model = Qwen3ForCausalLM::new(model_config, vb)?;

        // Allocate KV cache
        let (k_caches, v_caches, num_blocks) =
            Self::allocate_kv_cache(engine_config, model_config, &model, &device, dtype)?;
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
        device: &Device,
        dtype: DType,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, usize)> {
        let num_layers = model.num_layers();
        let num_kv_heads = model_config.num_key_value_heads;
        let head_dim = model_config.head_dim();
        let block_size = engine_config.kvcache_block_size;

        // For CPU, use a fixed number of blocks based on max_model_len
        let max_blocks_per_seq =
            (engine_config.max_model_len + block_size - 1) / block_size;
        let num_blocks = max_blocks_per_seq * engine_config.max_num_seqs;
        // Cap at a reasonable size for CPU
        let num_blocks = num_blocks.min(1024);

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            k_caches.push(Tensor::zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype,
                device,
            )?);
            v_caches.push(Tensor::zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype,
                device,
            )?);
        }

        Ok((k_caches, v_caches, num_blocks))
    }

    /// Prepare inputs for prefill phase.
    fn prepare_prefill(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, AttentionContext)> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping = Vec::new();
        let mut need_block_tables = false;

        for seq in seqs {
            let seqlen = seq.len();
            let cached = seq.num_cached_tokens;

            // Input tokens: skip cached portion
            for &t in seq.uncached_token_ids() {
                input_ids.push(t);
            }
            for pos in cached..seqlen {
                positions.push(pos as u32);
            }

            let seqlen_q = seqlen - cached;
            let seqlen_k = seqlen;
            cu_seqlens_q.push(cu_seqlens_q.last().unwrap() + seqlen_q as u32);
            cu_seqlens_k.push(cu_seqlens_k.last().unwrap() + seqlen_k as u32);
            max_seqlen_q = max_seqlen_q.max(seqlen_q);
            max_seqlen_k = max_seqlen_k.max(seqlen_k);

            if cached > 0 {
                need_block_tables = true;
            }

            // Slot mapping for uncached blocks
            if seq.block_table.is_empty() {
                continue; // warmup
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

        let n = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, n, &self.device)?;
        let positions = Tensor::from_vec(positions, n, &self.device)?;
        let sm_len = slot_mapping.len();
        let slot_mapping = Tensor::from_vec(slot_mapping, sm_len, &self.device)?;
        let cu_seqlens_q =
            Tensor::from_vec(cu_seqlens_q.clone(), cu_seqlens_q.len(), &self.device)?;
        let cu_seqlens_k =
            Tensor::from_vec(cu_seqlens_k.clone(), cu_seqlens_k.len(), &self.device)?;

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
    fn prepare_decode(&self, seqs: &[&Sequence]) -> Result<(Tensor, Tensor, AttentionContext)> {
        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut context_lens = Vec::new();

        for seq in seqs {
            input_ids.push(seq.last_token());
            positions.push((seq.len() - 1) as u32);
            context_lens.push(seq.len() as u32);
            let last_slot =
                seq.block_table.last().unwrap() * self.block_size + seq.last_block_num_tokens() - 1;
            slot_mapping.push(last_slot as i32);
        }

        let n = input_ids.len();
        let block_tables = self.prepare_block_tables(seqs)?;

        let input_ids = Tensor::from_vec(input_ids, n, &self.device)?;
        let positions = Tensor::from_vec(positions, n, &self.device)?;
        let slot_mapping = Tensor::from_vec(slot_mapping, n, &self.device)?;
        let context_lens_t = Tensor::from_vec(context_lens, n, &self.device)?;

        // For decode, cu_seqlens are simple: each sequence has 1 query token
        let cu_seqlens_q: Vec<u32> = (0..=n as u32).collect();
        let cu_seqlens_k = cu_seqlens_q.clone(); // not used in decode but needed for struct
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, n + 1, &self.device)?;
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, n + 1, &self.device)?;

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

    fn prepare_block_tables(&self, seqs: &[&Sequence]) -> Result<Tensor> {
        let max_len = seqs.iter().map(|s| s.block_table.len()).max().unwrap_or(0);
        let mut data = Vec::new();
        for seq in seqs {
            for &bid in &seq.block_table {
                data.push(bid as u32);
            }
            // Pad with 0
            for _ in seq.block_table.len()..max_len {
                data.push(0);
            }
        }
        let n = seqs.len();
        Ok(Tensor::from_vec(data, (n, max_len), &self.device)?)
    }

    fn prepare_temperatures(&self, seqs: &[&Sequence]) -> Result<Tensor> {
        let temps: Vec<f32> = seqs.iter().map(|s| s.temperature).collect();
        let n = temps.len();
        Ok(Tensor::from_vec(temps, n, &self.device)?)
    }

    /// Run a single inference step: prepare inputs, forward, sample.
    pub fn run(&mut self, seqs: &[&Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, ctx) = if is_prefill {
            self.prepare_prefill(seqs)?
        } else {
            self.prepare_decode(seqs)?
        };

        let temperatures = self.prepare_temperatures(seqs)?;

        let hidden_states = self.model.forward(
            &input_ids,
            &positions,
            &mut self.k_caches,
            &mut self.v_caches,
            &ctx,
        )?;

        let logits = self.model.compute_logits(&hidden_states, &ctx)?;
        let token_ids = sampler::sample(&logits, &temperatures)?;

        Ok(token_ids)
    }
}

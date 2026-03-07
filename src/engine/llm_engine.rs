use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use candle_core::Device;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::{EngineConfig, ModelConfig};
use crate::engine::model_runner::ModelRunner;
use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::sampling_params::SamplingParams;

pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
}

pub struct LLMEngine {
    tokenizer: tokenizers::Tokenizer,
    scheduler: Scheduler,
    model_runner: ModelRunner,
    block_size: usize,
}

impl LLMEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let model_path = std::path::PathBuf::from(model_path);
        let model_config = ModelConfig::from_dir(&model_path)?;
        let mut engine_config = EngineConfig::new(model_path.clone(), &model_config)?;

        let device = Device::Cpu;
        let model_runner = ModelRunner::new(&mut engine_config, &model_config, device)?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Set EOS token
        let vocab = tokenizer.get_vocab(true);
        if let Some(&eos_id) = vocab
            .get("<|endoftext|>")
            .or_else(|| vocab.get("</s>"))
        {
            engine_config.eos_token_id = eos_id;
        }

        let block_size = engine_config.kvcache_block_size;
        let scheduler = Scheduler::new(&engine_config);

        Ok(Self {
            tokenizer,
            scheduler,
            model_runner,
            block_size,
        })
    }

    pub fn add_request(&mut self, prompt: &str, sampling_params: &SamplingParams) -> Result<()> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq = Sequence::new(token_ids, sampling_params, self.block_size);
        self.scheduler.add(seq);
        Ok(())
    }

    fn step(&mut self) -> Result<(Vec<(usize, Vec<u32>)>, i64)> {
        let (seq_ids, is_prefill) = self.scheduler.schedule();

        // Collect sequence references for model_runner
        let seq_refs: Vec<&Sequence> = seq_ids
            .iter()
            .map(|id| self.scheduler.sequences.get(id).unwrap())
            .collect();

        let num_tokens = if is_prefill {
            seq_refs.iter().map(|s| s.len() as i64).sum()
        } else {
            -(seq_refs.len() as i64)
        };

        let token_ids = self.model_runner.run(&seq_refs, is_prefill)?;
        let finished = self.scheduler.postprocess(&seq_ids, &token_ids);

        Ok((finished, num_tokens))
    }

    pub fn generate(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        use_tqdm: bool,
    ) -> Result<Vec<GenerationOutput>> {
        for prompt in prompts {
            self.add_request(prompt, sampling_params)?;
        }

        let pb = if use_tqdm {
            let pb = ProgressBar::new(prompts.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40}] {pos}/{len}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            Some(pb)
        } else {
            None
        };

        let mut outputs: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut prefill_tps = 0.0f64;
        let mut decode_tps = 0.0f64;

        while !self.scheduler.is_finished() {
            let t = Instant::now();
            let (finished, num_tokens) = self.step()?;
            let elapsed = t.elapsed().as_secs_f64();

            if num_tokens > 0 {
                prefill_tps = num_tokens as f64 / elapsed;
            } else {
                decode_tps = (-num_tokens) as f64 / elapsed;
            }

            for (seq_id, token_ids) in finished {
                outputs.insert(seq_id, token_ids);
                if let Some(pb) = &pb {
                    pb.inc(1);
                }
            }

            if let Some(pb) = &pb {
                pb.set_message(format!(
                    "Prefill: {:.0} tok/s  Decode: {:.0} tok/s",
                    prefill_tps, decode_tps
                ));
            }
        }

        if let Some(pb) = &pb {
            pb.finish();
        }

        // Sort by seq_id and decode
        let mut sorted: Vec<_> = outputs.into_iter().collect();
        sorted.sort_by_key(|(id, _)| *id);

        let results: Vec<GenerationOutput> = sorted
            .into_iter()
            .map(|(_, token_ids)| {
                let text = self
                    .tokenizer
                    .decode(&token_ids, true)
                    .unwrap_or_default();
                GenerationOutput { text, token_ids }
            })
            .collect();

        Ok(results)
    }
}

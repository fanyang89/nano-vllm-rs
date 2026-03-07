use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use candle_core::Device;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

use crate::config::{EngineConfig, ModelConfig};
use crate::engine::model_runner::ModelRunner;
use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::sampling_params::SamplingParams;

pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Cuda,
}

pub struct LLMEngine {
    tokenizer: tokenizers::Tokenizer,
    scheduler: Scheduler,
    model_runner: ModelRunner,
    block_size: usize,
    use_qwen3_chat: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<EosTokenId>,
}

impl LLMEngine {
    pub fn new(model_path: &str, runtime_device: RuntimeDevice) -> Result<Self> {
        let model_path = std::path::PathBuf::from(model_path);
        let model_config = ModelConfig::from_dir(&model_path)?;
        let mut engine_config = EngineConfig::new(model_path.clone(), &model_config)?;

        let device = create_device(runtime_device)?;
        let model_runner = ModelRunner::new(&mut engine_config, &model_config, device)?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        // Set EOS token, preferring generation_config.json when available.
        let generation_config_path = model_path.join("generation_config.json");
        if let Ok(content) = std::fs::read_to_string(&generation_config_path) {
            if let Ok(gen_cfg) = serde_json::from_str::<GenerationConfig>(&content) {
                match gen_cfg.eos_token_id {
                    Some(EosTokenId::Single(id)) => engine_config.eos_token_id = id,
                    Some(EosTokenId::Multiple(ids)) => {
                        if let Some(id) = ids.first().copied() {
                            engine_config.eos_token_id = id;
                        }
                    }
                    None => {}
                }
            }
        }

        // Fallback EOS selection from model config / tokenizer vocab.
        if engine_config.eos_token_id == 0 {
            if let Some(id) = model_config.eos_token_id {
                engine_config.eos_token_id = id;
            } else {
                let vocab = tokenizer.get_vocab(true);
                if let Some(&eos_id) = vocab
                    .get("<|im_end|>")
                    .or_else(|| vocab.get("<|endoftext|>"))
                    .or_else(|| vocab.get("</s>"))
                {
                    engine_config.eos_token_id = eos_id;
                }
            }
        }

        let use_qwen3_chat = model_config.model_type.as_deref() == Some("qwen3");

        let block_size = engine_config.kvcache_block_size;
        let scheduler = Scheduler::new(&engine_config);

        Ok(Self {
            tokenizer,
            scheduler,
            model_runner,
            block_size,
            use_qwen3_chat,
        })
    }

    pub fn add_request(&mut self, prompt: &str, sampling_params: &SamplingParams) -> Result<()> {
        let prompt = if self.use_qwen3_chat {
            format!(
                "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        } else {
            prompt.to_string()
        };
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
        let mut last_pb_update = Instant::now();

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
                if last_pb_update.elapsed().as_millis() >= 100 {
                    pb.set_message(format!(
                        "Prefill: {:.0} tok/s  Decode: {:.0} tok/s",
                        prefill_tps, decode_tps
                    ));
                    last_pb_update = Instant::now();
                }
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
                let text = self.tokenizer.decode(&token_ids, true).unwrap_or_default();
                GenerationOutput { text, token_ids }
            })
            .collect();

        Ok(results)
    }
}

fn create_device(runtime_device: RuntimeDevice) -> Result<Device> {
    match runtime_device {
        RuntimeDevice::Cpu => Ok(Device::Cpu),
        RuntimeDevice::Cuda => create_cuda_device(),
    }
}

#[cfg(feature = "cuda")]
fn create_cuda_device() -> Result<Device> {
    Device::new_cuda(0).map_err(|e| anyhow::anyhow!("failed to initialize CUDA device 0: {e}"))
}

#[cfg(not(feature = "cuda"))]
fn create_cuda_device() -> Result<Device> {
    anyhow::bail!(
        "CUDA requested but this binary was built without CUDA support. Rebuild with `--features cuda`."
    )
}

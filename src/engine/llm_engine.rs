use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

#[cfg(feature = "cpu")]
use crate::backend::CpuBackend;
#[cfg(feature = "rocm")]
use crate::backend::RocmBackend;
use crate::chat::{ChatFormat, format_single_prompt};
use crate::config::{EngineConfig, ModelConfig};
use crate::engine::model_runner::ModelRunner;
use crate::engine::scheduler::Scheduler;
use crate::engine::sequence::Sequence;
use crate::sampling_params::SamplingParams;
use crate::utils::profiler;

#[derive(Debug, Clone)]
pub struct ProgressConfig {
    pub label: String,
    pub refresh_interval: std::time::Duration,
}

impl ProgressConfig {
    pub fn new(label: impl Into<String>, refresh_interval: std::time::Duration) -> Self {
        Self {
            label: label.into(),
            refresh_interval,
        }
    }
}

pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub total_time_s: f64,
    pub prefill_tokens: u64,
    pub prefill_time_s: f64,
    pub decode_tokens: u64,
    pub decode_time_s: f64,
    pub decode_steady_tokens: u64,
    pub decode_steady_time_s: f64,
    pub first_decode_latency_s: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDevice {
    Cpu,
    Rocm,
}

pub enum LLMEngine {
    #[cfg(feature = "cpu")]
    Cpu(LLMEngineImpl<CpuBackend>),
    #[cfg(feature = "rocm")]
    Rocm(LLMEngineImpl<RocmBackend>),
}

pub struct LLMEngineImpl<B: Backend<IntElem = i32>> {
    tokenizer: tokenizers::Tokenizer,
    scheduler: Scheduler,
    model_runner: ModelRunner<B>,
    block_size: usize,
    chat_format: ChatFormat,
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
        match runtime_device {
            #[cfg(feature = "cpu")]
            RuntimeDevice::Cpu => Ok(Self::Cpu(LLMEngineImpl::<CpuBackend>::new(
                model_path,
                burn_ndarray::NdArrayDevice::Cpu,
            )?)),
            #[cfg(not(feature = "cpu"))]
            RuntimeDevice::Cpu => {
                anyhow::bail!(
                    "CPU requested but this binary was built without `cpu` feature support"
                )
            }

            #[cfg(feature = "rocm")]
            RuntimeDevice::Rocm => Ok(Self::Rocm(LLMEngineImpl::<RocmBackend>::new(
                model_path,
                burn_rocm::RocmDevice::default(),
            )?)),
            #[cfg(not(feature = "rocm"))]
            RuntimeDevice::Rocm => {
                anyhow::bail!(
                    "ROCm requested but this binary was built without `rocm` feature support"
                )
            }
        }
    }

    pub fn add_request(&mut self, prompt: &str, sampling_params: &SamplingParams) -> Result<()> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.add_request(prompt, sampling_params),
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => engine.add_request(prompt, sampling_params),
        }
    }

    pub fn add_formatted_request(
        &mut self,
        prompt: &str,
        sampling_params: &SamplingParams,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.add_formatted_request(prompt, sampling_params),
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => engine.add_formatted_request(prompt, sampling_params),
        }
    }

    pub fn add_request_token_ids(
        &mut self,
        prompt_token_ids: Vec<u32>,
        sampling_params: &SamplingParams,
    ) -> Result<()> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.add_request_token_ids(prompt_token_ids, sampling_params),
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => engine.add_request_token_ids(prompt_token_ids, sampling_params),
        }
    }

    pub fn generate(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        use_tqdm: bool,
    ) -> Result<Vec<GenerationOutput>> {
        let progress = use_tqdm
            .then(|| ProgressConfig::new("Generating", std::time::Duration::from_millis(100)));
        let (outputs, _stats) =
            self.generate_with_stats_and_progress(prompts, sampling_params, progress.as_ref())?;
        Ok(outputs)
    }

    pub fn generate_formatted(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        use_tqdm: bool,
    ) -> Result<Vec<GenerationOutput>> {
        let progress = use_tqdm
            .then(|| ProgressConfig::new("Generating", std::time::Duration::from_millis(100)));
        let (outputs, _stats) = self.generate_formatted_with_stats_and_progress(
            prompts,
            sampling_params,
            progress.as_ref(),
        )?;
        Ok(outputs)
    }

    pub fn generate_token_ids_batch(
        &mut self,
        prompt_token_ids: &[Vec<u32>],
        sampling_params: &[SamplingParams],
        use_tqdm: bool,
    ) -> Result<Vec<GenerationOutput>> {
        let progress = use_tqdm
            .then(|| ProgressConfig::new("Generating", std::time::Duration::from_millis(100)));
        let (outputs, _stats) = self.generate_token_ids_batch_with_stats_and_progress(
            prompt_token_ids,
            sampling_params,
            progress.as_ref(),
        )?;
        Ok(outputs)
    }

    pub fn generate_token_ids_batch_with_stats(
        &mut self,
        prompt_token_ids: &[Vec<u32>],
        sampling_params: &[SamplingParams],
        use_tqdm: bool,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        let progress = use_tqdm
            .then(|| ProgressConfig::new("Generating", std::time::Duration::from_millis(100)));
        self.generate_token_ids_batch_with_stats_and_progress(
            prompt_token_ids,
            sampling_params,
            progress.as_ref(),
        )
    }

    pub fn generate_token_ids_batch_with_stats_and_progress(
        &mut self,
        prompt_token_ids: &[Vec<u32>],
        sampling_params: &[SamplingParams],
        progress: Option<&ProgressConfig>,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.generate_token_ids_batch_with_stats(
                prompt_token_ids,
                sampling_params,
                progress,
            ),
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => engine.generate_token_ids_batch_with_stats(
                prompt_token_ids,
                sampling_params,
                progress,
            ),
        }
    }

    pub fn generate_with_stats(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        use_tqdm: bool,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        let progress = use_tqdm
            .then(|| ProgressConfig::new("Generating", std::time::Duration::from_millis(100)));
        self.generate_with_stats_and_progress(prompts, sampling_params, progress.as_ref())
    }

    pub fn generate_formatted_with_stats_and_progress(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        progress: Option<&ProgressConfig>,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => {
                engine.generate_with_stats_impl(prompts, sampling_params, progress, true)
            }
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => {
                engine.generate_with_stats_impl(prompts, sampling_params, progress, true)
            }
        }
    }

    pub fn generate_with_stats_and_progress(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        progress: Option<&ProgressConfig>,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(engine) => engine.generate_with_stats(prompts, sampling_params, progress),
            #[cfg(feature = "rocm")]
            Self::Rocm(engine) => engine.generate_with_stats(prompts, sampling_params, progress),
        }
    }
}

fn create_progress_bar(
    total_tokens: u64,
    progress: Option<&ProgressConfig>,
) -> Option<ProgressBar> {
    let progress = progress?;
    let pb = ProgressBar::new(total_tokens);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(progress.label.clone());
    Some(pb)
}

fn prefill_tps(stats: &GenerationStats) -> f64 {
    if stats.prefill_time_s > 0.0 {
        stats.prefill_tokens as f64 / stats.prefill_time_s
    } else {
        0.0
    }
}

fn decode_tps(stats: &GenerationStats) -> f64 {
    if stats.decode_time_s > 0.0 {
        stats.decode_tokens as f64 / stats.decode_time_s
    } else {
        0.0
    }
}

fn update_progress_message(pb: &ProgressBar, progress: &ProgressConfig, stats: &GenerationStats) {
    pb.set_message(format!(
        "{} | Prefill: {:.0} tok/s | Decode: {:.0} tok/s",
        progress.label,
        prefill_tps(stats),
        decode_tps(stats)
    ));
}

impl<B: Backend<IntElem = i32>> LLMEngineImpl<B> {
    fn new(model_path: &str, device: B::Device) -> Result<Self> {
        let model_path = std::path::PathBuf::from(model_path);
        let model_config = ModelConfig::from_dir(&model_path)?;
        let mut engine_config = EngineConfig::new(model_path.clone(), &model_config)?;
        let model_runner = ModelRunner::<B>::new(&mut engine_config, &model_config, device)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

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

        let chat_format = ChatFormat::from_model_config(&model_config);
        let block_size = engine_config.kvcache_block_size;
        let scheduler = Scheduler::new(&engine_config);

        Ok(Self {
            tokenizer,
            scheduler,
            model_runner,
            block_size,
            chat_format,
        })
    }

    fn add_request(&mut self, prompt: &str, sampling_params: &SamplingParams) -> Result<()> {
        let prompt = format_single_prompt(prompt, self.chat_format);
        self.add_formatted_request(&prompt, sampling_params)
    }

    fn add_formatted_request(
        &mut self,
        prompt: &str,
        sampling_params: &SamplingParams,
    ) -> Result<()> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        self.add_request_token_ids(token_ids, sampling_params)
    }

    fn add_request_token_ids(
        &mut self,
        token_ids: Vec<u32>,
        sampling_params: &SamplingParams,
    ) -> Result<()> {
        let seq = Sequence::new(token_ids, sampling_params, self.block_size);
        self.scheduler.add(seq);
        Ok(())
    }

    fn step(&mut self) -> Result<(Vec<(usize, Vec<u32>)>, i64, usize)> {
        let (seq_ids, is_prefill) = self.scheduler.schedule();
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
        let generated_tokens = token_ids.len();
        let finished = self.scheduler.postprocess(&seq_ids, &token_ids);
        let finished_ids: Vec<usize> = finished.iter().map(|(seq_id, _)| *seq_id).collect();
        self.model_runner.clear_sequences(&finished_ids);

        Ok((finished, num_tokens, generated_tokens))
    }

    fn generate_with_stats(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        progress: Option<&ProgressConfig>,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        self.generate_with_stats_impl(prompts, sampling_params, progress, false)
    }

    fn generate_with_stats_impl(
        &mut self,
        prompts: &[&str],
        sampling_params: &SamplingParams,
        progress: Option<&ProgressConfig>,
        prompts_are_formatted: bool,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        profiler::reset();

        for prompt in prompts {
            if prompts_are_formatted {
                self.add_formatted_request(prompt, sampling_params)?;
            } else {
                self.add_request(prompt, sampling_params)?;
            }
        }

        let total_start = Instant::now();
        let pb = create_progress_bar(
            prompts.len().saturating_mul(sampling_params.max_tokens) as u64,
            progress,
        );

        let mut outputs: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut stats = GenerationStats::default();
        let mut seen_decode_step = false;
        let mut last_pb_update = Instant::now();

        while !self.scheduler.is_finished() {
            let t = Instant::now();
            let (finished, num_tokens, generated_tokens) = self.step()?;
            let elapsed = t.elapsed().as_secs_f64();

            if num_tokens > 0 {
                stats.prefill_tokens += num_tokens as u64;
                stats.prefill_time_s += elapsed;
            } else {
                let decode_tokens = (-num_tokens) as u64;
                stats.decode_tokens += decode_tokens;
                stats.decode_time_s += elapsed;
                if seen_decode_step {
                    stats.decode_steady_tokens += decode_tokens;
                    stats.decode_steady_time_s += elapsed;
                } else {
                    seen_decode_step = true;
                    stats.first_decode_latency_s = Some(total_start.elapsed().as_secs_f64());
                }
            }

            for (seq_id, token_ids) in finished {
                outputs.insert(seq_id, token_ids);
            }

            if let Some(pb) = &pb {
                let remaining = pb.length().unwrap_or(0).saturating_sub(pb.position());
                pb.inc((generated_tokens as u64).min(remaining));
            }

            if let Some(pb) = &pb {
                if progress.is_some_and(|cfg| last_pb_update.elapsed() >= cfg.refresh_interval) {
                    update_progress_message(pb, progress.expect("checked is_some"), &stats);
                    last_pb_update = Instant::now();
                }
            }
        }

        if let Some(pb) = &pb {
            if let Some(progress) = progress {
                update_progress_message(pb, progress, &stats);
            }
            if let Some(len) = pb.length() {
                let pos = pb.position();
                if pos < len {
                    pb.set_length(pos);
                }
            }
            pb.finish();
        }
        stats.total_time_s = total_start.elapsed().as_secs_f64();

        let mut sorted: Vec<_> = outputs.into_iter().collect();
        sorted.sort_by_key(|(id, _)| *id);

        let results: Vec<GenerationOutput> = sorted
            .into_iter()
            .map(|(_, token_ids)| {
                let text = self.tokenizer.decode(&token_ids, true).unwrap_or_default();
                GenerationOutput { text, token_ids }
            })
            .collect();

        if let Some(report) = profiler::report() {
            println!("{report}");
        }

        Ok((results, stats))
    }

    fn generate_token_ids_batch_with_stats(
        &mut self,
        prompt_token_ids: &[Vec<u32>],
        sampling_params: &[SamplingParams],
        progress: Option<&ProgressConfig>,
    ) -> Result<(Vec<GenerationOutput>, GenerationStats)> {
        anyhow::ensure!(
            prompt_token_ids.len() == sampling_params.len(),
            "prompt_token_ids and sampling_params length mismatch"
        );

        profiler::reset();
        for (prompt, params) in prompt_token_ids.iter().zip(sampling_params.iter()) {
            self.add_request_token_ids(prompt.clone(), params)?;
        }

        let total_start = Instant::now();
        let pb = create_progress_bar(
            sampling_params
                .iter()
                .map(|sp| sp.max_tokens)
                .sum::<usize>() as u64,
            progress,
        );

        let mut outputs: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut stats = GenerationStats::default();
        let mut seen_decode_step = false;
        let mut last_pb_update = Instant::now();

        while !self.scheduler.is_finished() {
            let t = Instant::now();
            let (finished, num_tokens, generated_tokens) = self.step()?;
            let elapsed = t.elapsed().as_secs_f64();

            if num_tokens > 0 {
                stats.prefill_tokens += num_tokens as u64;
                stats.prefill_time_s += elapsed;
            } else {
                let decode_tokens = (-num_tokens) as u64;
                stats.decode_tokens += decode_tokens;
                stats.decode_time_s += elapsed;
                if seen_decode_step {
                    stats.decode_steady_tokens += decode_tokens;
                    stats.decode_steady_time_s += elapsed;
                } else {
                    seen_decode_step = true;
                    stats.first_decode_latency_s = Some(total_start.elapsed().as_secs_f64());
                }
            }

            for (seq_id, token_ids) in finished {
                outputs.insert(seq_id, token_ids);
            }

            if let Some(pb) = &pb {
                let remaining = pb.length().unwrap_or(0).saturating_sub(pb.position());
                pb.inc((generated_tokens as u64).min(remaining));
            }

            if let Some(pb) = &pb {
                if progress.is_some_and(|cfg| last_pb_update.elapsed() >= cfg.refresh_interval) {
                    update_progress_message(pb, progress.expect("checked is_some"), &stats);
                    last_pb_update = Instant::now();
                }
            }
        }

        if let Some(pb) = &pb {
            if let Some(progress) = progress {
                update_progress_message(pb, progress, &stats);
            }
            if let Some(len) = pb.length() {
                let pos = pb.position();
                if pos < len {
                    pb.set_length(pos);
                }
            }
            pb.finish();
        }
        stats.total_time_s = total_start.elapsed().as_secs_f64();

        let mut sorted: Vec<_> = outputs.into_iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        let results: Vec<GenerationOutput> = sorted
            .into_iter()
            .map(|(_, token_ids)| {
                let text = self.tokenizer.decode(&token_ids, true).unwrap_or_default();
                GenerationOutput { text, token_ids }
            })
            .collect();

        if let Some(report) = profiler::report() {
            println!("{report}");
        }

        Ok((results, stats))
    }
}

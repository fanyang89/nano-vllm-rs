use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use nano_vllm_rs::chat::{
    ChatFormat, ChatMessage, ChatRole, render_chat_prompt, strip_think_blocks,
};
use nano_vllm_rs::config::ModelConfig;
use nano_vllm_rs::{GenerationStats, LLMEngine, ProgressConfig, RuntimeDevice, SamplingParams};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CliDevice {
    Cpu,
    Cuda,
    Rocm,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run text generation inference
    Run(RunArgs),
    /// Start an interactive chat REPL
    Repl(ReplArgs),
    /// Run benchmark compatible with nano-vllm/bench.py
    Bench(BenchArgs),
    /// Build a deterministic Qwen3 weight remap plan from safetensors
    ConvertModel(ConvertModelArgs),
}

#[derive(Parser, Debug)]
#[command(name = "nano-vllm-rs", about = "Minimal vLLM inference engine in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Path to model directory (containing safetensors + config.json + tokenizer.json)
    #[arg(long)]
    model: String,

    /// Runtime device to use
    #[arg(long, value_enum)]
    device: CliDevice,

    /// Prompts to generate from (can specify multiple)
    #[arg(long, num_args = 1..)]
    prompt: Vec<String>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Repeat each prompt N times to increase decode batch concurrency
    #[arg(long, default_value_t = 1)]
    repeat_prompt: usize,

    /// Disable random sampling and use greedy decoding (argmax)
    #[arg(long, default_value_t = false)]
    greedy: bool,
}

#[derive(Parser, Debug)]
struct ReplArgs {
    /// Path to model directory (containing safetensors + config.json + tokenizer.json)
    #[arg(long)]
    model: String,

    /// Runtime device to use
    #[arg(long, value_enum)]
    device: CliDevice,

    /// System prompt to keep for the whole session
    #[arg(long)]
    system: Option<String>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Disable random sampling and use greedy decoding (argmax)
    #[arg(long, default_value_t = false)]
    greedy: bool,
}

#[derive(Parser, Debug)]
struct ConvertModelArgs {
    /// Source model directory containing *.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// Output JSON file containing remap metadata and concat groups.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Parser, Debug)]
struct BenchArgs {
    /// Path to model directory (containing safetensors + config.json + tokenizer.json)
    #[arg(long)]
    model: String,

    /// Runtime device to use
    #[arg(long, value_enum)]
    device: CliDevice,

    /// Number of requests to issue in one batch
    #[arg(long, default_value_t = 256)]
    num_seqs: usize,

    /// Maximum random prompt token length
    #[arg(long, default_value_t = 1024)]
    max_input_len: usize,

    /// Maximum random output token length
    #[arg(long, default_value_t = 1024)]
    max_output_len: usize,

    /// Warmup iterations before timed run
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Timed iterations
    #[arg(long, default_value_t = 1)]
    iters: usize,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Disable live progress updates during timed iterations
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

#[derive(Debug, Serialize)]
struct TensorEntry {
    source_key: String,
    source_file: String,
    dtype: String,
    shape: Vec<usize>,
    target_key: String,
}

#[derive(Debug, Serialize)]
struct ConcatGroup {
    target_key: String,
    dim: usize,
    sources: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ConversionPlan {
    model_type: String,
    files_scanned: usize,
    tensor_count: usize,
    tensors: Vec<TensorEntry>,
    concat_groups: Vec<ConcatGroup>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run(args) => run_inference(args),
        Commands::Repl(args) => run_repl(args),
        Commands::Bench(args) => run_benchmark(args),
        Commands::ConvertModel(args) => run_convert_model(args),
    }
}

#[derive(Debug, Clone, Copy)]
struct IterMetrics {
    prefill_tps: f64,
    decode_tps: f64,
}

#[derive(Debug, Clone)]
struct ReplSession {
    chat_format: ChatFormat,
    system_prompt: Option<String>,
    history: Vec<ChatMessage>,
}

impl ReplSession {
    fn new(chat_format: ChatFormat, system_prompt: Option<String>) -> Self {
        Self {
            chat_format,
            system_prompt,
            history: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.history.clear();
    }

    fn add_user_message(&mut self, content: impl Into<String>) {
        self.history.push(ChatMessage::new(ChatRole::User, content));
    }

    fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.history
            .push(ChatMessage::new(ChatRole::Assistant, content));
    }

    fn messages(&self) -> Vec<ChatMessage> {
        let mut messages =
            Vec::with_capacity(self.history.len() + usize::from(self.system_prompt.is_some()));
        if let Some(system_prompt) = &self.system_prompt {
            messages.push(ChatMessage::new(ChatRole::System, system_prompt.clone()));
        }
        messages.extend(self.history.iter().cloned());
        messages
    }

    fn render_prompt(&self) -> String {
        let messages = self.messages();
        render_chat_prompt(&messages, self.chat_format, true)
    }

    fn print_history(&self) {
        let messages = self.messages();
        if messages.is_empty() {
            println!("(history is empty)");
            return;
        }

        for message in messages {
            println!("\n{}:\n{}", message.role.label(), message.content);
        }
        println!();
    }
}

fn run_inference(args: RunArgs) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- run ...` for real performance."
    );

    let base_prompts: Vec<String> = default_prompts(args.prompt);
    let repeat = args.repeat_prompt.max(1);
    let expanded_prompts: Vec<String> = expand_prompts(&base_prompts, repeat);
    let prompts: Vec<&str> = expanded_prompts.iter().map(|s| s.as_str()).collect();

    let sampling_params =
        SamplingParams::new(args.temperature, args.max_tokens, false, !args.greedy);

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let runtime_device = match args.device {
        CliDevice::Cpu => RuntimeDevice::Cpu,
        CliDevice::Cuda => RuntimeDevice::Cuda,
        CliDevice::Rocm => RuntimeDevice::Rocm,
    };
    let mut engine = LLMEngine::new(&args.model, runtime_device)?;

    println!(
        "Generating {} prompt(s)... (base={}, repeat={})",
        prompts.len(),
        base_prompts.len(),
        repeat
    );
    let outputs = engine.generate(&prompts, &sampling_params, true)?;

    for (i, output) in outputs.iter().enumerate() {
        println!("\n--- Output {} ---", i + 1);
        println!("{}", output.text);
    }

    Ok(())
}

fn run_repl(args: ReplArgs) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- repl ...` for real performance."
    );

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let runtime_device = match args.device {
        CliDevice::Cpu => RuntimeDevice::Cpu,
        CliDevice::Cuda => RuntimeDevice::Cuda,
        CliDevice::Rocm => RuntimeDevice::Rocm,
    };
    let model_config = ModelConfig::from_dir(Path::new(&args.model))?;
    let chat_format = ChatFormat::from_model_config(&model_config);
    let mut session = ReplSession::new(chat_format, args.system.clone());
    let sampling_params =
        SamplingParams::new(args.temperature, args.max_tokens, false, !args.greedy);
    let mut engine = LLMEngine::new(&args.model, runtime_device)?;

    println!("Chat REPL ready.");
    println!("Commands: /history, /clear, /exit");
    if let Some(system_prompt) = &args.system {
        println!("System: {}", system_prompt);
    }

    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        line.clear();
        let read = stdin.read_line(&mut line)?;
        if read == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/clear" => {
                session.clear();
                println!("Cleared conversation history. System prompt preserved.");
                continue;
            }
            "/history" => {
                session.print_history();
                continue;
            }
            _ => {}
        }

        session.add_user_message(input);
        let prompt = session.render_prompt();
        let outputs = engine.generate_formatted(&[prompt.as_str()], &sampling_params, true)?;
        let reply = outputs
            .into_iter()
            .next()
            .map(|output| output.text)
            .unwrap_or_default();
        let visible_reply = strip_think_blocks(&reply);

        println!("\nassistant:\n{}\n", visible_reply);
        session.add_assistant_message(visible_reply);
    }

    Ok(())
}

fn run_benchmark(args: BenchArgs) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- bench ...` for real performance."
    );

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let runtime_device = match args.device {
        CliDevice::Cpu => RuntimeDevice::Cpu,
        CliDevice::Cuda => RuntimeDevice::Cuda,
        CliDevice::Rocm => RuntimeDevice::Rocm,
    };
    let mut engine = LLMEngine::new(&args.model, runtime_device)?;

    let mut rng = StdRng::seed_from_u64(args.seed);
    let prompt_token_ids: Vec<Vec<u32>> = (0..args.num_seqs)
        .map(|_| {
            let len = rng.random_range(100..=args.max_input_len.max(100));
            (0..len).map(|_| rng.random_range(0..=10_000u32)).collect()
        })
        .collect();
    let sampling_params: Vec<SamplingParams> = (0..args.num_seqs)
        .map(|_| {
            SamplingParams::new(
                0.6,
                rng.random_range(100..=args.max_output_len.max(100)),
                true,
                true,
            )
        })
        .collect();

    println!(
        "Python-compatible benchmark config: num_seqs={}, max_input_len={}, max_output_len={}, warmup={}, iters={}, seed={}, quiet={}",
        args.num_seqs,
        args.max_input_len,
        args.max_output_len,
        args.warmup,
        args.iters,
        args.seed,
        args.quiet
    );

    let warmup_prompt = SamplingParams::default();
    for i in 0..args.warmup {
        let _ = engine.generate_with_stats(&["Benchmark: "], &warmup_prompt, false)?;
        println!("Warmup {}/{} done", i + 1, args.warmup);
    }

    let mut throughputs = Vec::with_capacity(args.iters);
    let mut prefill_tps_values = Vec::with_capacity(args.iters);
    let mut decode_tps_values = Vec::with_capacity(args.iters);
    let total_tokens: usize = sampling_params.iter().map(|sp| sp.max_tokens).sum();
    for iter in 0..args.iters {
        let progress_label = format!(
            "Bench {}/{} | num_seqs={} | in<={} | out<={}",
            iter + 1,
            args.iters,
            args.num_seqs,
            args.max_input_len,
            args.max_output_len
        );
        let progress =
            (!args.quiet).then(|| ProgressConfig::new(progress_label, Duration::from_secs(1)));
        let started = Instant::now();
        let (_, stats) = engine.generate_token_ids_batch_with_stats_and_progress(
            &prompt_token_ids,
            &sampling_params,
            progress.as_ref(),
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        let throughput = total_tokens as f64 / elapsed;
        let metrics = to_metrics(&stats, total_tokens, elapsed);
        throughputs.push(throughput);
        prefill_tps_values.push(metrics.prefill_tps);
        decode_tps_values.push(metrics.decode_tps);
        println!(
            "Iter {}/{}: Total: {}tok, Time: {:.2}s, Throughput: {:.2}tok/s, Prefill: {:.2}tok/s, Decode: {:.2}tok/s",
            iter + 1,
            args.iters,
            total_tokens,
            elapsed,
            throughput,
            metrics.prefill_tps,
            metrics.decode_tps,
        );
    }

    let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let mean_prefill_tps = prefill_tps_values.iter().sum::<f64>() / prefill_tps_values.len() as f64;
    let mean_decode_tps = decode_tps_values.iter().sum::<f64>() / decode_tps_values.len() as f64;
    println!(
        "Mean throughput over {} iter(s): {:.2}tok/s | Prefill: {:.2}tok/s | Decode: {:.2}tok/s",
        throughputs.len(),
        mean,
        mean_prefill_tps,
        mean_decode_tps,
    );
    Ok(())
}

fn default_prompts(prompt: Vec<String>) -> Vec<String> {
    if prompt.is_empty() {
        vec!["Hello, world!".to_string()]
    } else {
        prompt
    }
}

fn expand_prompts(base_prompts: &[String], repeat: usize) -> Vec<String> {
    base_prompts
        .iter()
        .flat_map(|p| std::iter::repeat_n(p.clone(), repeat))
        .collect()
}

fn to_metrics(stats: &GenerationStats, _output_tokens: usize, _wall_time_s: f64) -> IterMetrics {
    let prefill_tps = if stats.prefill_time_s > 0.0 {
        stats.prefill_tokens as f64 / stats.prefill_time_s
    } else {
        0.0
    };
    let decode_tps = if stats.decode_time_s > 0.0 {
        stats.decode_tokens as f64 / stats.decode_time_s
    } else {
        0.0
    };
    IterMetrics {
        prefill_tps,
        decode_tps,
    }
}

fn run_convert_model(args: ConvertModelArgs) -> Result<()> {
    let plan = build_plan(&args.model)?;
    let json = serde_json::to_string_pretty(&plan)?;
    fs::write(&args.out, json)
        .with_context(|| format!("failed to write plan file: {}", args.out.display()))?;
    println!(
        "Wrote conversion plan with {} tensors to {}",
        plan.tensor_count,
        args.out.display()
    );
    Ok(())
}

fn build_plan(model_dir: &Path) -> Result<ConversionPlan> {
    if !model_dir.is_dir() {
        bail!("model path is not a directory: {}", model_dir.display());
    }

    let mut files: Vec<PathBuf> = fs::read_dir(model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no .safetensors files found in {}", model_dir.display());
    }

    let mut tensors = Vec::new();
    let mut seen = BTreeSet::new();

    for file in &files {
        let bytes = fs::read(file).with_context(|| format!("failed to read {}", file.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("invalid safetensors: {}", file.display()))?;
        for key in st.names() {
            if !seen.insert(key.to_string()) {
                continue;
            }
            let t = st
                .tensor(key)
                .with_context(|| format!("failed to load tensor key {key}"))?;
            tensors.push(TensorEntry {
                source_key: key.to_string(),
                source_file: file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| file.display().to_string()),
                dtype: format!("{:?}", t.dtype()),
                shape: t.shape().to_vec(),
                target_key: remap_key(key),
            });
        }
    }

    tensors.sort_by(|a, b| a.source_key.cmp(&b.source_key));

    let mut by_layer_qkv: BTreeMap<String, (String, String, String)> = BTreeMap::new();
    let mut by_layer_mlp: BTreeMap<String, (String, String)> = BTreeMap::new();

    for t in &tensors {
        if let Some(layer) = t.source_key.strip_suffix(".self_attn.q_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.0 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".self_attn.k_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.1 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".self_attn.v_proj.weight") {
            let e = by_layer_qkv.entry(layer.to_string()).or_default();
            e.2 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".mlp.gate_proj.weight") {
            let e = by_layer_mlp.entry(layer.to_string()).or_default();
            e.0 = t.source_key.clone();
        } else if let Some(layer) = t.source_key.strip_suffix(".mlp.up_proj.weight") {
            let e = by_layer_mlp.entry(layer.to_string()).or_default();
            e.1 = t.source_key.clone();
        }
    }

    let mut concat_groups = Vec::new();
    for (layer, (q, k, v)) in by_layer_qkv {
        if !(q.is_empty() || k.is_empty() || v.is_empty()) {
            concat_groups.push(ConcatGroup {
                target_key: format!("{layer}.self_attn.qkv_proj.weight"),
                dim: 0,
                sources: vec![q, k, v],
            });
        }
    }
    for (layer, (gate, up)) in by_layer_mlp {
        if !(gate.is_empty() || up.is_empty()) {
            concat_groups.push(ConcatGroup {
                target_key: format!("{layer}.mlp.gate_up_proj.weight"),
                dim: 0,
                sources: vec![gate, up],
            });
        }
    }
    concat_groups.sort_by(|a, b| a.target_key.cmp(&b.target_key));

    Ok(ConversionPlan {
        model_type: "qwen3".to_string(),
        files_scanned: files.len(),
        tensor_count: tensors.len(),
        tensors,
        concat_groups,
    })
}

fn remap_key(source: &str) -> String {
    source.to_string()
}

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use tablestream::{Column, Stream};

use nano_vllm_rs::{GenerationStats, LLMEngine, RuntimeDevice, SamplingParams};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CliDevice {
    Cpu,
    Rocm,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run text generation inference
    Run(RunArgs),
    /// Run performance benchmarks
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

    /// Prompts to benchmark (can specify multiple)
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

    /// Sweep repeat counts to measure throughput scaling, e.g. 1,2,4,8,16
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    throughput_repeats: Vec<usize>,

    /// Disable random sampling and use greedy decoding (argmax)
    #[arg(long, default_value_t = false)]
    greedy: bool,

    /// Ignore EOS and always run to max_tokens for throughput measurements
    #[arg(long, default_value_t = false)]
    ignore_eos: bool,

    /// Warmup iterations before timed runs
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Timed benchmark iterations
    #[arg(long, default_value_t = 5)]
    iters: usize,
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
        Commands::Bench(args) => run_benchmark(args),
        Commands::ConvertModel(args) => run_convert_model(args),
    }
}

#[derive(Debug, Clone, Copy)]
struct IterMetrics {
    total_time_s: f64,
    output_tokens: usize,
    prefill_tps: f64,
    decode_tps: f64,
    decode_steady_tps: f64,
    ttft_ms: f64,
}

#[derive(Debug, Clone)]
struct BenchScenario {
    repeat: usize,
    prompt_count: usize,
    metrics: Vec<IterMetrics>,
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

fn run_benchmark(args: BenchArgs) -> Result<()> {
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- bench ...` for real performance."
    );

    let base_prompts: Vec<String> = default_prompts(args.prompt);
    let mut repeats = if args.throughput_repeats.is_empty() {
        vec![args.repeat_prompt.max(1)]
    } else {
        let mut repeats: Vec<usize> = args
            .throughput_repeats
            .iter()
            .copied()
            .map(|v| v.max(1))
            .collect();
        repeats.sort_unstable();
        repeats.dedup();
        repeats
    };
    if repeats.is_empty() {
        repeats.push(1);
    }
    let is_sweep = repeats.len() > 1;
    let sampling_params = SamplingParams::new(
        args.temperature,
        args.max_tokens,
        args.ignore_eos,
        !args.greedy,
    );

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let runtime_device = match args.device {
        CliDevice::Cpu => RuntimeDevice::Cpu,
        CliDevice::Rocm => RuntimeDevice::Rocm,
    };
    let mut engine = LLMEngine::new(&args.model, runtime_device)?;

    println!(
        "Benchmark config: base_prompts={}, repeats={:?}, warmup={}, iters={}, max_tokens={}, greedy={}, ignore_eos={}",
        base_prompts.len(),
        repeats,
        args.warmup,
        args.iters,
        args.max_tokens,
        args.greedy,
        args.ignore_eos
    );

    let mut scenarios = Vec::with_capacity(repeats.len());
    for &repeat in &repeats {
        let expanded_prompts: Vec<String> = expand_prompts(&base_prompts, repeat);
        let prompts: Vec<&str> = expanded_prompts.iter().map(|s| s.as_str()).collect();

        println!(
            "\nScenario: prompts={} (base={}, repeat={})",
            prompts.len(),
            base_prompts.len(),
            repeat
        );

        for i in 0..args.warmup {
            let _ = engine.generate_with_stats(&prompts, &sampling_params, false)?;
            println!("Warmup {}/{} done", i + 1, args.warmup);
        }

        let mut all = Vec::with_capacity(args.iters);
        for _ in 0..args.iters {
            let started = Instant::now();
            let (outputs, stats) = engine.generate_with_stats(&prompts, &sampling_params, false)?;
            let wall_time_s = started.elapsed().as_secs_f64();
            let output_tokens: usize = outputs.iter().map(|o| o.token_ids.len()).sum();
            let metrics = to_metrics(&stats, output_tokens, wall_time_s);
            all.push(metrics);
        }

        print_benchmark_tables(&all);
        scenarios.push(BenchScenario {
            repeat,
            prompt_count: prompts.len(),
            metrics: all,
        });
    }

    if is_sweep {
        print_throughput_sweep_table(&scenarios);
    }
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

fn to_metrics(stats: &GenerationStats, output_tokens: usize, wall_time_s: f64) -> IterMetrics {
    let total_time_s = if stats.total_time_s > 0.0 {
        stats.total_time_s
    } else {
        wall_time_s
    };
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
    let decode_steady_tps = if stats.decode_steady_time_s > 0.0 {
        stats.decode_steady_tokens as f64 / stats.decode_steady_time_s
    } else {
        decode_tps
    };
    let ttft_ms = stats.first_decode_latency_s.unwrap_or(0.0) * 1000.0;

    IterMetrics {
        total_time_s,
        output_tokens,
        prefill_tps,
        decode_tps,
        decode_steady_tps,
        ttft_ms,
    }
}

#[derive(Clone)]
struct BenchRow {
    iter: usize,
    total_s: String,
    output_tokens: usize,
    e2e_tps: String,
    prefill_tps: String,
    decode_tps: String,
    decode_steady_tps: String,
    ttft_ms: String,
}

#[derive(Clone)]
struct SummaryRow {
    metric: &'static str,
    value: String,
}

#[derive(Clone)]
struct ThroughputSweepRow {
    scenario: String,
    metric: &'static str,
    value: String,
}

fn print_benchmark_tables(all: &[IterMetrics]) {
    if all.is_empty() {
        println!("No benchmark iterations were run.");
        return;
    }

    let rows: Vec<BenchRow> = all
        .iter()
        .enumerate()
        .map(|(i, m)| BenchRow {
            iter: i + 1,
            total_s: format!("{:.3}", m.total_time_s),
            output_tokens: m.output_tokens,
            e2e_tps: format!(
                "{:.2}",
                if m.total_time_s > 0.0 {
                    m.output_tokens as f64 / m.total_time_s
                } else {
                    0.0
                }
            ),
            prefill_tps: format!("{:.2}", m.prefill_tps),
            decode_tps: format!("{:.2}", m.decode_tps),
            decode_steady_tps: format!("{:.2}", m.decode_steady_tps),
            ttft_ms: format!("{:.1}", m.ttft_ms),
        })
        .collect();

    let mut table = Stream::new(
        std::io::stdout(),
        vec![
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.iter)).header("iter"),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.total_s))
                .header("total_s")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.output_tokens))
                .header("output_tokens")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.e2e_tps))
                .header("e2e_tps")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.prefill_tps))
                .header("prefill_tps")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.decode_tps))
                .header("decode_tps")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.decode_steady_tps))
                .header("decode_steady_tps")
                .right(),
            Column::new(|f, r: &BenchRow| write!(f, "{}", r.ttft_ms))
                .header("ttft_ms")
                .right(),
        ],
    )
    .borders(true)
    .title("Benchmark Iterations");
    for row in rows {
        let _ = table.row(row);
    }
    let _ = table.finish();

    let n = all.len() as f64;
    let mean_total_s = all.iter().map(|m| m.total_time_s).sum::<f64>() / n;
    let mean_output_tokens = all.iter().map(|m| m.output_tokens as f64).sum::<f64>() / n;
    let mean_e2e_tps = all
        .iter()
        .map(|m| {
            if m.total_time_s > 0.0 {
                m.output_tokens as f64 / m.total_time_s
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n;
    let mean_prefill_tps = all.iter().map(|m| m.prefill_tps).sum::<f64>() / n;
    let mean_decode_tps = all.iter().map(|m| m.decode_tps).sum::<f64>() / n;
    let mean_decode_steady_tps = all.iter().map(|m| m.decode_steady_tps).sum::<f64>() / n;
    let mean_ttft_ms = all.iter().map(|m| m.ttft_ms).sum::<f64>() / n;

    let min_decode_steady_tps = all
        .iter()
        .map(|m| m.decode_steady_tps)
        .fold(f64::INFINITY, f64::min);
    let max_decode_steady_tps = all
        .iter()
        .map(|m| m.decode_steady_tps)
        .fold(0.0_f64, f64::max);

    let rows = vec![
        SummaryRow {
            metric: "mean_total_s",
            value: format!("{:.3}", mean_total_s),
        },
        SummaryRow {
            metric: "mean_output_tokens",
            value: format!("{:.1}", mean_output_tokens),
        },
        SummaryRow {
            metric: "mean_e2e_tps",
            value: format!("{:.2}", mean_e2e_tps),
        },
        SummaryRow {
            metric: "mean_prefill_tps",
            value: format!("{:.2}", mean_prefill_tps),
        },
        SummaryRow {
            metric: "mean_decode_tps",
            value: format!("{:.2}", mean_decode_tps),
        },
        SummaryRow {
            metric: "mean_decode_steady_tps",
            value: format!("{:.2}", mean_decode_steady_tps),
        },
        SummaryRow {
            metric: "mean_ttft_ms",
            value: format!("{:.1}", mean_ttft_ms),
        },
        SummaryRow {
            metric: "min_decode_steady_tps",
            value: format!("{:.2}", min_decode_steady_tps),
        },
        SummaryRow {
            metric: "max_decode_steady_tps",
            value: format!("{:.2}", max_decode_steady_tps),
        },
    ];

    let mut summary = Stream::new(
        std::io::stdout(),
        vec![
            Column::new(|f, r: &SummaryRow| write!(f, "{}", r.metric)).header("metric"),
            Column::new(|f, r: &SummaryRow| write!(f, "{}", r.value))
                .header("value")
                .right(),
        ],
    )
    .borders(true)
    .title("Benchmark Summary");
    for row in rows {
        let _ = summary.row(row);
    }
    let _ = summary.finish();
}

fn print_throughput_sweep_table(scenarios: &[BenchScenario]) {
    if scenarios.is_empty() {
        return;
    }

    let rows: Vec<ThroughputSweepRow> = scenarios
        .iter()
        .flat_map(|scenario| {
            let n = scenario.metrics.len() as f64;
            let mean_output_tokens = scenario
                .metrics
                .iter()
                .map(|m| m.output_tokens as f64)
                .sum::<f64>()
                / n;
            let mean_e2e_tps_total = scenario
                .metrics
                .iter()
                .map(|m| {
                    if m.total_time_s > 0.0 {
                        m.output_tokens as f64 / m.total_time_s
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / n;
            let mean_decode_steady_tps_total = scenario
                .metrics
                .iter()
                .map(|m| m.decode_steady_tps)
                .sum::<f64>()
                / n;
            let mean_ttft_ms = scenario.metrics.iter().map(|m| m.ttft_ms).sum::<f64>() / n;
            let prompt_count = scenario.prompt_count.max(1);
            let scenario_label = format!("repeat={} prompts={}", scenario.repeat, prompt_count);

            vec![
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "mean_output_tokens",
                    value: format!("{:.1}", mean_output_tokens),
                },
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "tokens_per_req",
                    value: format!("{:.1}", mean_output_tokens / prompt_count as f64),
                },
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "e2e_tps_total",
                    value: format!("{:.2}", mean_e2e_tps_total),
                },
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "e2e_tps_per_req",
                    value: format!("{:.2}", mean_e2e_tps_total / prompt_count as f64),
                },
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "decode_steady_tps_total",
                    value: format!("{:.2}", mean_decode_steady_tps_total),
                },
                ThroughputSweepRow {
                    scenario: scenario_label.clone(),
                    metric: "decode_steady_tps_per_req",
                    value: format!("{:.2}", mean_decode_steady_tps_total / prompt_count as f64),
                },
                ThroughputSweepRow {
                    scenario: scenario_label,
                    metric: "mean_ttft_ms",
                    value: format!("{:.1}", mean_ttft_ms),
                },
            ]
        })
        .collect();

    let mut table = Stream::new(
        std::io::stdout(),
        vec![
            Column::new(|f, r: &ThroughputSweepRow| write!(f, "{}", r.scenario)).header("scenario"),
            Column::new(|f, r: &ThroughputSweepRow| write!(f, "{}", r.metric)).header("metric"),
            Column::new(|f, r: &ThroughputSweepRow| write!(f, "{}", r.value))
                .header("value")
                .right(),
        ],
    )
    .borders(true)
    .title("Throughput Sweep");
    for row in rows {
        let _ = table.row(row);
    }
    let _ = table.finish();
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

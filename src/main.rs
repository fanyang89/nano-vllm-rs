use anyhow::Result;
use clap::{Parser, ValueEnum};

use nano_vllm_rs::{LLMEngine, RuntimeDevice, SamplingParams};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum CliDevice {
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(name = "nano-vllm-rs", about = "Minimal vLLM inference engine in Rust")]
struct Args {
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

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- ...` for real performance."
    );

    let base_prompts: Vec<String> = if args.prompt.is_empty() {
        vec!["Hello, world!".to_string()]
    } else {
        args.prompt
    };
    let repeat = args.repeat_prompt.max(1);
    let expanded_prompts: Vec<String> = base_prompts
        .iter()
        .flat_map(|p| std::iter::repeat_n(p.clone(), repeat))
        .collect();
    let prompts: Vec<&str> = expanded_prompts.iter().map(|s| s.as_str()).collect();

    let sampling_params =
        SamplingParams::new(args.temperature, args.max_tokens, false, !args.greedy);

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let runtime_device = match args.device {
        CliDevice::Cpu => RuntimeDevice::Cpu,
        CliDevice::Cuda => RuntimeDevice::Cuda,
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

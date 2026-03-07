use anyhow::Result;
use clap::Parser;

use nano_vllm_rs::{LLMEngine, SamplingParams};

#[derive(Parser)]
#[command(name = "nano-vllm-rs", about = "Minimal vLLM inference engine in Rust")]
struct Args {
    /// Path to model directory (containing safetensors + config.json + tokenizer.json)
    #[arg(long)]
    model: String,

    /// Prompts to generate from (can specify multiple)
    #[arg(long, num_args = 1..)]
    prompt: Vec<String>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.6)]
    temperature: f32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(debug_assertions)]
    eprintln!(
        "Warning: debug build is slow for inference. Use `cargo run --release -- ...` for real performance."
    );

    let prompts: Vec<&str> = if args.prompt.is_empty() {
        vec!["Hello, world!"]
    } else {
        args.prompt.iter().map(|s| s.as_str()).collect()
    };

    let sampling_params = SamplingParams::new(args.temperature, args.max_tokens, false);

    tracing_subscriber::fmt::init();

    println!("Loading model from: {}", args.model);
    let mut engine = LLMEngine::new(&args.model)?;

    println!("Generating {} prompt(s)...", prompts.len());
    let outputs = engine.generate(&prompts, &sampling_params, true)?;

    for (i, output) in outputs.iter().enumerate() {
        println!("\n--- Output {} ---", i + 1);
        println!("{}", output.text);
    }

    Ok(())
}

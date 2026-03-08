# nano-vllm-rs

## Overview

`nano-vllm-rs` is a lightweight vLLM-style inference engine written in Rust with [Burn](https://burn.dev),
inspired by [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

## Run Inference

Run on CPU:

```bash
cargo run --release -- run --device cpu --model ./models/Qwen3-0.6B --prompt "Hello"
```

Run on ROCm (build with the `rocm` feature):

```bash
cargo run --release --features rocm -- run --device rocm --model ./models/Qwen3-0.6B --prompt "Hello"
```

Notes:
- The tensor and model runner path now uses Burn with the `Dispatch` backend on CPU and ROCm devices.
- Full paged attention and the rotary path are still being refined in later optimization stages.

## Chat REPL

Start an interactive multi-turn chat session with an optional system prompt:

```bash
cargo run --release -- repl --device cpu --model ./models/Qwen3-0.6B --system "You are a concise assistant."
```

Built-in REPL commands:
- `/history`: print the current conversation history
- `/clear`: clear the conversation history and keep the `--system` prompt
- `/exit`: leave the REPL

## Chat REPL

Start an interactive multi-turn chat session with optional system prompt:

```bash
cargo run --release -- repl --device cpu --model ./models/Qwen3-0.6B --system "You are a concise assistant."
```

Built-in REPL commands:
- `/history`: print the current conversation history
- `/clear`: clear the conversation history and keep the `--system` prompt
- `/exit`: leave the REPL

The current REPL keeps context by replaying the full transcript each turn. It does not yet persist KV cache across turns.

## Benchmark

Use the built-in `bench` subcommand to measure both latency and throughput with structured CSV-like output:

```bash
cargo run --release -- bench --model ./models/Qwen3-0.6B --device cpu --prompt "Hello" \
--greedy --warmup 1 --iters 5 --max-tokens 128
```

Output columns:
- `total_s`: end-to-end request time
- `output_tokens`: generated completion tokens
- `e2e_tps`: `output_tokens / total_s`
- `prefill_tps`: prefill phase throughput
- `decode_tps`: decode throughput including first decode step
- `decode_steady_tps`: decode throughput excluding first decode step (more stable)
- `ttft_ms`: time to first generated token

Task shortcuts:

```bash
task bench-cpu
task bench-rocm
```

Recommended benchmark protocol:
1. Run at least one warmup iteration (`--warmup 1` or more).
2. Compare `decode_steady_tps` across versions for steady-state decode performance.
3. Sweep `--repeat-prompt` (e.g. `1, 2, 4, 8`) to evaluate batching/concurrency scaling.
4. Keep prompt and `--max-tokens` fixed when comparing CPU/ROCm or different commits.

## Weight Conversion

Generate a deterministic Qwen3 remap/concatenation plan from HuggingFace `safetensors` weights:

```bash
cargo run --release -- convert-model --model ./models/Qwen3-0.6B --out ./models/Qwen3-0.6B/weights-map.json
```

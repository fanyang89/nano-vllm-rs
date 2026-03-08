# nano-vllm-rs

## Run

CPU:

```bash
cargo run --release -- run --device cpu --model ./models/Qwen3-0.6B --prompt "Hello"
```

ROCm migration target (build with feature):

```bash
cargo run --release --features rocm -- run --device rocm --model ./models/Qwen3-0.6B --prompt "Hello"
```

Note: the tensor/model runner path now uses Burn (`Dispatch` backend with CPU/ROCm devices).
The full paged-attention + rotary path is still being refined in follow-up optimization phases.

## Benchmark

Use the built-in `bench` subcommand to measure both latency and throughput with structured CSV-like output:

```bash
cargo run --release -- bench --model ./models/Qwen3-0.6B --device cpu --prompt "Hello" --greedy --warmup 1 --iters 5 --max-tokens 128
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

## Weight Conversion Planning

Generate a deterministic Qwen3 remap/concat plan from HuggingFace safetensors:

```bash
cargo run --release -- convert-model --model ./models/Qwen3-0.6B --out ./models/Qwen3-0.6B/weights-map.json
```

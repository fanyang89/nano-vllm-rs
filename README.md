# nano-vllm-rs

## Run

CPU:

```bash
cargo run --release -- run --device cpu --model ./models/qwen3-0.6b --prompt "Hello"
```

ROCm migration target (build with feature):

```bash
cargo run --release --features rocm -- run --device rocm --model ./models/qwen3-0.6b --prompt "Hello"
```

Note: the tensor/model runner path now uses Burn (`Dispatch` backend with CPU/ROCm devices).
The full paged-attention + rotary path is still being refined in follow-up optimization phases.

## Weight Conversion Planning

Generate a deterministic Qwen3 remap/concat plan from HuggingFace safetensors:

```bash
cargo run --release -- convert-model --model ./models/qwen3-0.6b --out ./models/qwen3-0.6b/conversion-plan.json
```

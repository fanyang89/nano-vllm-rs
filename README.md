# nano-vllm-rs

## Run

CPU:

```bash
cargo run --release -- --device cpu --model ./models/qwen3-0.6b --prompt "Hello"
```

CUDA (requires NVIDIA CUDA runtime + build with feature):

```bash
cargo run --release --features cuda -- --device cuda --model ./models/qwen3-0.6b --prompt "Hello"
```

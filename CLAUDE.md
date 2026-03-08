# CLAUDE.md — nano-vllm-rs

## 项目简介

将 Python 项目 [nano-vllm](/home/fanmi/workspace/nano-vllm) 移植为 Rust + burn 实现的轻量级 vLLM 推理引擎。

## 参考代码

- Python 原版: `/home/fanmi/workspace/nano-vllm/nanovllm/`
- 路线图: `ROADMAP.md`

## 架构概览

```
LLMEngine.generate()
  → Tokenize → Scheduler.schedule() → ModelRunner.run() → Sampler → Scheduler.postprocess()
```

核心组件:
- **Scheduler**: 两阶段调度 (prefill/decode) + 抢占机制
- **BlockManager**: 基于块的 KV Cache 管理 + xxh64 前缀缓存
- **ModelRunner**: 输入准备、模型前向、采样
- **Qwen3 Model**: 解码器 Transformer (GQA, RoPE, SiLU MLP)

## 技术栈

- **ML 框架**: burn 0.21.0-pre.2 (burn-core, burn-dispatch)
- **后端**: CPU (burn-ndarray, 默认) / ROCm (burn-rocm, feature flag)
- **精度**: BF16
- **Tokenizer**: tokenizers crate (HuggingFace)
- **权重格式**: safetensors (本地目录加载)
- **哈希**: xxhash-rust (前缀缓存)
- **CLI**: clap (derive) 子命令模式
- **基准测试输出**: tablestream

## 开发约定

- 初始版本: 单设备、无 CUDA Graphs、Naive Attention
- AttentionContext 通过参数显式传递（非全局变量）
- 序列存储在 HashMap<seq_id, Sequence>，队列仅持有 id
- 权重加载时手动合并 QKV 和 gate_up

## 构建与测试

```bash
cargo build                          # 编译 (默认 CPU/ndarray 后端)
cargo build --features rocm          # 编译 ROCm 后端
cargo test                           # 运行测试

# 推理
cargo run --release -- run --model /path/to/qwen3 --device cpu --prompt "Hello" --max-tokens 50
cargo run --release -- run --model /path/to/qwen3 --device rocm --prompt "Hello" --greedy

# 基准测试
cargo run --release -- bench --model /path/to/qwen3 --device cpu --warmup 1 --iters 5

# 权重转换计划
cargo run -- convert-model --model /path/to/qwen3 --out plan.json
```

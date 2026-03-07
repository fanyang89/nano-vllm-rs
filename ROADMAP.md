# nano-vllm-rs 开发路线图

基于 [nano-vllm](https://github.com/user/nano-vllm) (Python/PyTorch) 移植到 Rust + [candle](https://github.com/huggingface/candle) 框架。

## 项目目标

将 nano-vllm (~1,200 行 Python) 移植为等价的 Rust 实现，保留核心 vLLM 架构：
- 基于块的 KV Cache 管理与前缀缓存 (Prefix Caching)
- 两阶段调度器 (Prefill / Decode) 与抢占机制
- Qwen3 模型支持
- 高效推理引擎

### 初始版本简化

| 特性 | Python 版 | Rust 初始版 |
|------|-----------|-------------|
| 张量并行 (TP) | 多 GPU (NCCL) | 单 GPU |
| CUDA Graphs | 支持 | 不支持 |
| Flash Attention | prefill + decode | Naive Attention |
| 计算精度 | 自动 (BF16) | BF16 |
| 模型加载 | 本地目录 | 本地目录 |

---

## 模块结构

```
src/
├── lib.rs                     # 公共 API: LLM, SamplingParams
├── main.rs                    # CLI 入口
├── config.rs                  # ModelConfig (config.json) + EngineConfig
├── sampling_params.rs         # 采样参数
├── engine/
│   ├── mod.rs
│   ├── llm_engine.rs          # 推理引擎主循环
│   ├── scheduler.rs           # 两阶段调度器
│   ├── sequence.rs            # 序列状态跟踪
│   ├── block_manager.rs       # KV Cache 块管理 + 前缀缓存
│   └── model_runner.rs        # 模型执行: 输入准备、推理、采样
├── model/
│   ├── mod.rs
│   └── qwen3.rs               # Qwen3 模型 (适配分页注意力)
├── layers/
│   ├── mod.rs
│   ├── attention.rs           # 分页注意力: KV 存储 + prefill/decode
│   ├── rotary_embedding.rs    # RoPE 旋转位置编码
│   └── sampler.rs             # 温度采样 (Gumbel-max)
└── utils/
    ├── mod.rs
    └── context.rs             # AttentionContext 注意力上下文
```

---

## 实现阶段

### 阶段 1: 基础类型（无张量运算）

**Step 1 — 项目依赖 `Cargo.toml`**
- `candle-core` (cuda feature)、`candle-nn`
- `tokenizers`、`serde`/`serde_json`、`anyhow`
- `xxhash-rust` (xxh64)、`indicatif`、`rand`

**Step 2 — 配置 `config.rs`**
- `ModelConfig`: 从 `config.json` 反序列化 (vocab_size, hidden_size, num_layers, heads, head_dim, intermediate_size, rms_norm_eps, rope_theta 等)
- `EngineConfig`: 用户参数 (model_path, max_num_batched_tokens=16384, max_num_seqs=512, max_model_len=4096, gpu_memory_utilization=0.9, kvcache_block_size=256)

**Step 3 — 采样参数 `sampling_params.rs`**
- `SamplingParams { temperature, max_tokens, ignore_eos }`

**Step 4 — 序列 `engine/sequence.rs`**
- `SequenceStatus` 枚举 (Waiting, Running, Finished)
- `Sequence` 结构体: token_ids, block_table, 各种状态计数
- 方法: len(), num_blocks(), block(i), append_token() 等

**Step 5 — 块管理器 `engine/block_manager.rs`**
- `Block` 结构体 (block_id, ref_count, hash, token_ids)
- `BlockManager`: 分配/释放块，前缀缓存哈希
- 使用 xxh64 哈希链实现前缀匹配

**Step 6 — 调度器 `engine/scheduler.rs`**
- 两阶段调度: prefill (分配新请求) + decode (继续运行请求)
- 抢占机制: KV Cache 满时驱逐低优先级序列
- 所有序列存储在 `HashMap<usize, Sequence>`，队列仅持有 seq_id

### 阶段 2: 模型与层

**Step 7 — 注意力上下文 `utils/context.rs`**
- `AttentionContext` 结构体: is_prefill, cu_seqlens_q/k, slot_mapping, block_tables 等
- 作为参数显式传递（替代 Python 的线程局部变量）

**Step 8 — RoPE `layers/rotary_embedding.rs`**
- 预计算 cos/sin 缓存
- 按位置索引并应用旋转: `y = cat(x1*cos - x2*sin, x2*cos + x1*sin)`

**Step 9 — 分页注意力 `layers/attention.rs`**（核心复杂层）
- KV Cache 存储: 用 slot_mapping 将 K/V 散射到分页缓存（初始用 CPU 循环 + slice_set）
- Prefill 注意力: 逐序列标准注意力 + 因果掩码
- Decode 注意力: 逐序列从分页缓存中收集 K/V，单 query 注意力

**Step 10 — 采样器 `layers/sampler.rs`**
- Gumbel-max 采样: logits/temp → softmax → 除以指数分布采样 → argmax

**Step 11 — Qwen3 模型 `model/qwen3.rs`**
- 使用 `candle_nn` 的 Linear, Embedding, RmsNorm
- 合并 QKV 权重加载 (q_proj + k_proj + v_proj → qkv_proj)
- 合并 gate_up 权重 (gate_proj + up_proj → gate_up_proj)
- RMSNorm with residual: fused add + normalize
- LM Head: prefill 时提取每序列最后 token 的 logits

### 阶段 3: 引擎集成

**Step 12 — 模型运行器 `engine/model_runner.rs`**
- 加载模型权重 (VarBuilder + safetensors mmap)
- 分配 KV Cache (查询 CUDA 可用内存，计算块数)
- prepare_prefill / prepare_decode: 构建输入张量和 AttentionContext
- run(): 前向传播 + 采样

**Step 13 — 推理引擎 `engine/llm_engine.rs`**
- 加载 tokenizer (tokenizer.json)
- add_request → tokenize → Sequence → scheduler
- step() → schedule → run → postprocess
- generate() → 循环 step 直到全部完成，解码输出

**Step 14 — 公共 API 与 CLI**
- `src/lib.rs`: 导出 LLM, SamplingParams, EngineConfig
- `src/main.rs`: CLI 参数 (--model, --prompt, --temperature, --max-tokens)

---

## 关键设计决策

| 决策 | 方案 |
|------|------|
| 序列所有权 | HashMap<seq_id, Sequence> + 队列持有 id，避免借用冲突 |
| KV Cache 写入 | CPU 循环 + slice_set（初始版），后续可加自定义 CUDA kernel |
| Decode 注意力 | 逐序列循环收集 K/V 块 + 标准 SDPA，后续可接入 flash-attn |
| 权重合并 | 加载时手动 cat q/k/v → qkv，gate/up → gate_up |
| 上下文传递 | &AttentionContext 显式参数（非全局变量） |
| 精度 | BF16 |
| 模型来源 | 仅本地目录 |

---

## 验证方案

1. **编译**: `cargo build` 无错误
2. **单元测试**: block_manager (分配/释放/前缀缓存)、scheduler (调度/抢占)、sequence (属性计算)
3. **集成测试**: 加载 Qwen3-0.6B，单条 prompt 完整跑通 prefill + decode
4. **对比测试**: 相同 prompt + temperature=0 (贪婪)，对比 Python 和 Rust 版本输出 token 是否一致
5. **CLI 测试**: `cargo run -- --model /path/to/qwen3 --prompt "Hello" --max-tokens 50`

---

## 后续优化方向（不在初始版本范围内）

- [ ] Flash Attention (candle-flash-attn) 用于 prefill 加速
- [ ] 自定义 CUDA kernel 替代 CPU 循环的 KV Cache 写入
- [ ] 张量并行 (多 GPU)
- [ ] CUDA Graphs 等价优化
- [ ] 更多模型支持 (Llama, Mistral 等)
- [ ] HuggingFace Hub 自动下载
- [ ] Streaming 输出

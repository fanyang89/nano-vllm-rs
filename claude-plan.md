Plan: Port nano-vllm to Rust with Candle

Context

The user has a Python project nano-vllm (~1,200 lines) implementing a minimal vLLM
inference engine with block-based KV cache, prefix caching, scheduling, and Qwen3
model support. The goal is to create an equivalent Rust implementation using the
candle ML framework at /home/fanmi/workspace/nano-vllm-rs.

Simplifications for initial Rust version:
- No tensor parallelism (single GPU only) — removes all dist.*, shared memory IPC,
multi-process logic
- No CUDA graphs — candle doesn't support this
- Naive attention instead of flash attention for decode (candle lacks
flash_attn_with_kvcache)
- Optional flash attention for prefill via candle-flash-attn feature flag
- Keep block-based KV cache, prefix caching, and scheduler logic intact

---
Module Structure

nano-vllm-rs/
├── Cargo.toml
└── src/
    ├── lib.rs                     # Public API: LLM, SamplingParams re-exports
    ├── main.rs                    # CLI entry point
    ├── config.rs                  # ModelConfig (from config.json) + EngineConfig
    ├── sampling_params.rs         # SamplingParams struct
    ├── engine/
    │   ├── mod.rs
    │   ├── llm_engine.rs          # LLMEngine: tokenize, generate loop
    │   ├── scheduler.rs           # Two-phase scheduler with preemption
    │   ├── sequence.rs            # Sequence, SequenceStatus
    │   ├── block_manager.rs       # Block, BlockManager with prefix caching
    │   └── model_runner.rs        # ModelRunner: prepare inputs, run model, sample
    ├── model/
    │   ├── mod.rs
    │   └── qwen3.rs               # Qwen3 model (custom, paged-attention-aware)
    ├── layers/
    │   ├── mod.rs
    │   ├── attention.rs           # Paged attention: store KV + prefill/decode attn
    │   ├── rotary_embedding.rs    # RoPE
    │   └── sampler.rs             # Temperature sampling (Gumbel-max)
    └── utils/
        ├── mod.rs
        └── context.rs             # AttentionContext struct

---
Implementation Steps

Step 1: Cargo.toml & Project Skeleton

Update Cargo.toml with all dependencies:
- candle-core (with cuda feature)
- candle-nn (VarBuilder, Linear, RmsNorm, Embedding)
- candle-transformers (reference only, we write our own model)
- tokenizers (HuggingFace tokenizers)
- hf-hub (model downloading)
- serde, serde_json (config parsing)
- anyhow (error handling)
- xxhash-rust with xxh64 feature (prefix caching)
- indicatif (progress bar)
- rand (sampling)

Optional feature: candle-flash-attn for prefill acceleration.

Create all module files with mod declarations.

File: Cargo.toml

Step 2: Config & SamplingParams

src/config.rs — Two structs:
- ModelConfig: Deserialize from config.json in model dir. Fields: vocab_size,
hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim,
intermediate_size, max_position_embeddings, rms_norm_eps, rope_theta,
tie_word_embeddings, torch_dtype.
- EngineConfig: User-provided. Fields: model_path, max_num_batched_tokens (16384),
max_num_seqs (512), max_model_len (4096), gpu_memory_utilization (0.9),
kvcache_block_size (256), dtype (BF16/F16).

src/sampling_params.rs — SamplingParams { temperature: f32, max_tokens: usize,
ignore_eos: bool }

Source ref: nanovllm/config.py, nanovllm/sampling_params.py

Step 3: Sequence & SequenceStatus

src/engine/sequence.rs

Port Sequence class. Key fields:
- seq_id: usize (atomic counter)
- status: SequenceStatus (enum: Waiting, Running, Finished)
- token_ids: Vec<u32>
- num_tokens, num_prompt_tokens, num_cached_tokens: usize
- block_table: Vec<usize>
- temperature: f32, max_tokens: usize, ignore_eos: bool

Methods: len(), num_blocks(), num_cached_blocks(), last_block_num_tokens(), block(i)
-> &[u32], append_token(), num_completion_tokens(), is_finished(), indexing into
token_ids.

Source ref: nanovllm/engine/sequence.py (83 lines)

Step 4: BlockManager

src/engine/block_manager.rs

Port Block and BlockManager. Use:
- VecDeque<usize> for free_block_ids
- HashSet<usize> for used_block_ids
- HashMap<u64, usize> for hash_to_block_id
- xxhash_rust::xxh64::xxh64 for prefix hash computation

Key methods: can_allocate(), allocate(), deallocate(), can_append(), may_append(),
compute_hash().

Hash computation: hash the token IDs bytes with optional prefix hash chaining
(matching the Python xxhash usage).

Source ref: nanovllm/engine/block_manager.py (112 lines)

Step 5: Scheduler

src/engine/scheduler.rs

Port two-phase scheduling. Use VecDeque<usize> (seq_ids) for waiting and running
queues, with sequences stored in a HashMap<usize, Sequence> owned by the scheduler
(avoids Rust borrow issues).

Key methods:
- add(seq) — push to waiting
- schedule() -> (Vec<usize>, bool) — returns scheduled seq_ids and is_prefill flag
- preempt(seq_id) — deallocate blocks, move to waiting front
- postprocess(seq_ids, token_ids) — append tokens, check EOS/max_tokens, deallocate
finished
- is_finished() -> bool
- get_sequences(&self, ids: &[usize]) -> Vec<&Sequence> — borrow sequences for
model_runner

Source ref: nanovllm/engine/scheduler.py (71 lines)

Step 6: AttentionContext

src/utils/context.rs

Simple struct (replaces Python's thread-local global):

pub struct AttentionContext {
    pub is_prefill: bool,
    pub cu_seqlens_q: Tensor,    // [batch+1], i32
    pub cu_seqlens_k: Tensor,    // [batch+1], i32
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Tensor,    // [num_tokens], i32
    pub context_lens: Option<Tensor>,   // [batch], i32 (decode only)
    pub block_tables: Option<Tensor>,   // [batch, max_blocks], i32
}

Passed as &AttentionContext through model forward methods.

Source ref: nanovllm/utils/context.py (27 lines)

Step 7: RoPE

src/layers/rotary_embedding.rs

Precompute cos_sin_cache: Tensor of shape (max_position, head_dim) at init. In
forward:
1. Index into cache with positions: cos_sin = cos_sin_cache.index_select(&positions,
0)?
2. Split into cos/sin halves
3. Apply rotation: y = cat(x1*cos - x2*sin, x2*cos + x1*sin, dim=-1)

Input q/k shape: (N, num_heads, head_dim), positions shape: (N,).

Source ref: nanovllm/layers/rotary_embedding.py (61 lines)

Step 8: Attention Layer (Paged KV Cache)

src/layers/attention.rs

The most complex layer. Three operations:

a) KV cache store (replaces Triton kernel):
- KV cache shape: (num_blocks, block_size, num_kv_heads, head_dim) per layer, per
k/v
- slot_mapping maps each token to a flat slot index
- Flatten cache to (num_blocks * block_size, num_kv_heads * head_dim), use index_put
or scatter with slot_mapping
- Implementation: build index tensor from slot_mapping, use Tensor::index_add or a
CPU loop with slice_set

b) Prefill attention:
- Without flash-attn: Implement per-sequence attention with causal mask
- For each sequence (using cu_seqlens to split), compute Q @ K^T, apply causal
mask, softmax, @ V
- Handle prefix cache: when block_tables is Some, read K/V from the paged cache
using block_tables
- With flash-attn feature: use candle_flash_attn::flash_attn_varlen(q, k, v,
cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal)

c) Decode attention:
- For each sequence: gather K/V blocks from paged cache using block_table, compute
single-query attention
- Loop over batch: for seq_i, extract block_table[i], gather context_lens[i] tokens
of K/V from cache, compute q @ K^T * scale -> softmax -> @ V

Source ref: nanovllm/layers/attention.py (75 lines)

Step 9: Sampler

src/layers/sampler.rs

Gumbel-max trick:
1. logits = logits.to_f32() / temperatures (broadcast)
2. probs = softmax(logits, dim=-1)
3. Generate uniform random u ~ (0,1), compute -log(u) for exponential samples
4. sample = probs / exponential_samples
5. token_ids = argmax(sample, dim=-1)

Use rand crate to generate random tensor, or generate on CPU and transfer.

Source ref: nanovllm/layers/sampler.py (15 lines)

Step 10: Qwen3 Model

src/model/qwen3.rs

Port the Qwen3 architecture using candle_nn:

Qwen3ForCausalLM
├── Qwen3Model
│   ├── Embedding (vocab_size, hidden_size) — candle_nn::Embedding
│   ├── Vec<Qwen3DecoderLayer>
│   │   ├── Qwen3Attention
│   │   │   ├── Linear (qkv_proj: hidden -> q_size + 2*kv_size)
│   │   │   ├── RMSNorm (q_norm, k_norm) — when no qkv_bias
│   │   │   ├── RotaryEmbedding
│   │   │   ├── Attention (paged)
│   │   │   └── Linear (o_proj: num_heads*head_dim -> hidden)
│   │   └── Qwen3MLP
│   │       ├── Linear (gate_up_proj: hidden -> 2*intermediate)
│   │       ├── SiLU + Mul
│   │       └── Linear (down_proj: intermediate -> hidden)
│   │   ├── RMSNorm (input_layernorm)
│   │   └── RMSNorm (post_attention_layernorm)
│   └── RMSNorm (final norm)
└── Linear (lm_head: hidden -> vocab) — with optional weight tying

Forward signature: fn forward(&self, input_ids: &Tensor, positions: &Tensor, ctx:
&AttentionContext) -> Result<Tensor>

Weight loading: Use candle_nn::VarBuilder::from_mmaped_safetensors. For merged
QKV/gate_up weights:
- Load q_proj, k_proj, v_proj separately then Tensor::cat them into qkv_proj
- Load gate_proj, up_proj separately then Tensor::cat into gate_up_proj
- Build a custom VarBuilder or do manual weight assembly in the model constructor

RMSNorm with residual: Implement fused add_rms_norm(x, residual) that returns
(normed, new_residual) where new_residual = x + residual.

LM Head: In prefill, extract last token per sequence using cu_seqlens_q[1:] - 1 as
indices. Compute hidden @ weight^T for logits.

Source ref: nanovllm/models/qwen3.py (215 lines), nanovllm/layers/layernorm.py,
nanovllm/layers/activation.py, nanovllm/layers/linear.py,
nanovllm/layers/embed_head.py

Step 11: ModelRunner

src/engine/model_runner.rs

Key responsibilities:

new(config, model_config):
1. Load model via VarBuilder from safetensors files in model_path
2. Allocate KV cache: query CUDA free memory, compute num_blocks = available_bytes /
block_bytes
3. Create KV cache tensors: k_caches: Vec<Tensor> and v_caches: Vec<Tensor>, one per
layer, shape (num_blocks, block_size, num_kv_heads, head_dim)
4. Pass cache references to attention layers

prepare_prefill(seqs) -> (input_ids, positions, AttentionContext):
- Port the Python logic: build flattened input_ids (skip cached tokens), positions,
cu_seqlens_q/k, slot_mapping
- All tensors created on CPU then moved to GPU

prepare_decode(seqs) -> (input_ids, positions, AttentionContext):
- One token per sequence: last_token, position = len-1
- Build slot_mapping, context_lens, block_tables

run(seqs, is_prefill) -> Vec<u32>:
- Prepare inputs
- Forward through model
- Sample tokens

Source ref: nanovllm/engine/model_runner.py (251 lines) — skip tensor parallelism,
CUDA graphs, shared memory IPC

Step 12: LLMEngine

src/engine/llm_engine.rs

Top-level orchestrator:
- Load tokenizer via tokenizers::Tokenizer::from_file(model_path/tokenizer.json)
- add_request(prompt, sampling_params): tokenize, create Sequence, add to scheduler
- step(): schedule -> model_runner.run -> postprocess
- generate(prompts, sampling_params) -> Vec<Output>: loop step() until finished,
collect outputs, decode tokens

Source ref: nanovllm/engine/llm_engine.py (93 lines)

Step 13: Public API & CLI

src/lib.rs: Re-export LLM (alias for LLMEngine), SamplingParams, EngineConfig

src/main.rs: CLI with args: --model, --prompt (repeatable), --temperature,
--max-tokens. Instantiate LLM, generate, print results.

---
Key Design Decisions

Sequence Ownership (avoiding Rust borrow issues)

Store all sequences in HashMap<usize, Sequence> inside the Scheduler. Queues
(waiting, running) hold only seq_id: usize. When model_runner needs sequence data,
scheduler provides immutable snapshots or temporary borrows via accessor methods.

KV Cache Store Without Triton

For initial correctness, iterate on CPU:
- Copy slot_mapping to CPU
- For each slot, use tensor slicing to write K/V into the cache
- Later optimization: build a custom CUDA kernel or use batched scatter

Decode Attention Without flash_attn_with_kvcache

Per-sequence loop:
1. Read block_table for the sequence
2. Gather K/V blocks from paged cache into a contiguous tensor
3. Standard scaled dot-product attention with the single query token
4. This is correct but slower than a fused kernel — acceptable for initial version

Weight Loading Strategy

Use candle_nn::VarBuilder with manual QKV/gate_up merging:
1. Memory-map all safetensors files
2. For each layer, load q_proj.weight, k_proj.weight, v_proj.weight separately
3. Tensor::cat(&[q, k, v], 0) to create merged qkv_proj.weight
4. Same for gate_proj + up_proj → gate_up_proj

---
Verification Plan

1. Build: cargo build compiles without errors
2. Unit tests: Test block_manager (allocate/deallocate/prefix cache), scheduler
(prefill/decode/preemption), sequence (properties)
3. Integration test: Load a small Qwen3 model (e.g., Qwen3-0.6B), run a single
prompt through the full pipeline, verify valid token output
4. Comparison test: Run same prompt with same temperature=0 (greedy) in both Python
and Rust versions, compare output tokens
5. CLI test: cargo run -- --model /path/to/qwen3 --prompt "Hello, world!"
--max-tokens 50

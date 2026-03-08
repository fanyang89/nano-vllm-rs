use anyhow::{Result, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Model architecture config deserialized from config.json in the model directory.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_false")]
    pub attention_bias: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_true() -> bool {
    true
}
fn default_false() -> bool {
    false
}

impl ModelConfig {
    pub fn from_dir(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let content = std::fs::read_to_string(&config_path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Engine-level configuration provided by the user.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub max_model_len: usize,
    pub gpu_memory_utilization: f64,
    pub kvcache_block_size: usize,
    pub num_kvcache_blocks: usize,
    pub eos_token_id: u32,
}

impl EngineConfig {
    pub fn new(model_path: PathBuf, model_config: &ModelConfig) -> Result<Self> {
        ensure!(model_path.is_dir(), "model path must be a directory");

        let max_model_len = 4096.min(model_config.max_position_embeddings);

        let config = Self {
            model_path,
            max_num_batched_tokens: 16384,
            max_num_seqs: 512,
            max_model_len,
            gpu_memory_utilization: 0.9,
            kvcache_block_size: 256,
            num_kvcache_blocks: 0, // computed later by model_runner
            eos_token_id: 0,       // set after tokenizer load
        };

        ensure!(
            config.max_num_batched_tokens >= config.max_model_len,
            "max_num_batched_tokens must >= max_model_len"
        );
        ensure!(
            config.kvcache_block_size % 256 == 0,
            "kvcache_block_size must be multiple of 256"
        );

        Ok(config)
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sampling_params::SamplingParams;

static SEQ_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id: usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub block_table: Vec<usize>,
    pub temperature: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub do_sample: bool,
    block_size: usize,
}

impl Sequence {
    pub fn new(token_ids: Vec<u32>, sampling_params: &SamplingParams, block_size: usize) -> Self {
        let num_tokens = token_ids.len();
        Self {
            seq_id: SEQ_COUNTER.fetch_add(1, Ordering::Relaxed),
            status: SequenceStatus::Waiting,
            token_ids,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            temperature: sampling_params.temperature,
            max_tokens: sampling_params.max_tokens,
            ignore_eos: sampling_params.ignore_eos,
            do_sample: sampling_params.do_sample,
            block_size,
        }
    }

    pub fn len(&self) -> usize {
        self.num_tokens
    }

    pub fn last_token(&self) -> u32 {
        self.token_ids[self.num_tokens - 1]
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    pub fn num_blocks(&self) -> usize {
        (self.num_tokens + self.block_size - 1) / self.block_size
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }

    pub fn last_block_num_tokens(&self) -> usize {
        self.num_tokens - (self.num_blocks() - 1) * self.block_size
    }

    /// Get token_ids for the i-th block.
    pub fn block(&self, i: usize) -> &[u32] {
        assert!(i < self.num_blocks());
        let start = i * self.block_size;
        let end = ((i + 1) * self.block_size).min(self.num_tokens);
        &self.token_ids[start..end]
    }

    /// Get token_ids from num_cached_tokens onward (uncached portion).
    pub fn uncached_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_cached_tokens..]
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.num_tokens += 1;
    }
}

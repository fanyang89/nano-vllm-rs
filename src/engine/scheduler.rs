use std::collections::{HashMap, VecDeque};

use crate::config::EngineConfig;
use crate::engine::block_manager::BlockManager;
use crate::engine::sequence::{Sequence, SequenceStatus};

pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    eos_token_id: u32,
    block_manager: BlockManager,
    waiting: VecDeque<usize>,
    running: VecDeque<usize>,
    pub sequences: HashMap<usize, Sequence>,
}

impl Scheduler {
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos_token_id: config.eos_token_id,
            block_manager: BlockManager::new(config.num_kvcache_blocks, config.kvcache_block_size),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            sequences: HashMap::new(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn add(&mut self, seq: Sequence) {
        let seq_id = seq.seq_id;
        self.sequences.insert(seq_id, seq);
        self.waiting.push_back(seq_id);
    }

    /// Schedule sequences for the next step.
    /// Returns (scheduled_seq_ids, is_prefill).
    pub fn schedule(&mut self) -> (Vec<usize>, bool) {
        // Phase 1: Prefill — try to schedule waiting sequences
        let mut scheduled = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        while let Some(&seq_id) = self.waiting.front() {
            if num_seqs >= self.max_num_seqs {
                break;
            }
            let seq = &self.sequences[&seq_id];
            let seq_len = seq.len();

            if num_batched_tokens + seq_len > self.max_num_batched_tokens
                || !self.block_manager.can_allocate(seq)
            {
                break;
            }

            num_seqs += 1;
            self.waiting.pop_front();

            let seq = self.sequences.get_mut(&seq_id).unwrap();
            self.block_manager.allocate(seq);
            let cached = seq.num_cached_tokens;
            num_batched_tokens += seq_len - cached;
            seq.status = SequenceStatus::Running;

            self.running.push_back(seq_id);
            scheduled.push(seq_id);
        }

        if !scheduled.is_empty() {
            return (scheduled, true);
        }

        // Phase 2: Decode — continue running sequences
        let mut old_running: VecDeque<usize> = std::mem::take(&mut self.running);

        while let Some(seq_id) = old_running.pop_front() {
            if num_seqs >= self.max_num_seqs {
                // Put remaining back
                for id in old_running {
                    self.running.push_back(id);
                }
                break;
            }

            let seq = &self.sequences[&seq_id];
            let mut can_append = self.block_manager.can_append(seq);

            while !can_append {
                // Preempt from the back of old_running or self
                if let Some(victim_id) = old_running.pop_back() {
                    self.preempt(victim_id);
                    let seq = &self.sequences[&seq_id];
                    can_append = self.block_manager.can_append(seq);
                } else {
                    // Must preempt ourselves
                    self.preempt(seq_id);
                    can_append = false;
                    break;
                }
            }

            if can_append {
                num_seqs += 1;
                let seq = self.sequences.get_mut(&seq_id).unwrap();
                self.block_manager.may_append(seq);
                scheduled.push(seq_id);
            }
        }

        assert!(!scheduled.is_empty(), "scheduler: no sequences scheduled");

        // Put scheduled sequences back into running (at front)
        let mut new_running = VecDeque::from(scheduled.clone());
        new_running.append(&mut self.running);
        self.running = new_running;

        (scheduled, false)
    }

    fn preempt(&mut self, seq_id: usize) {
        let seq = self.sequences.get_mut(&seq_id).unwrap();
        seq.status = SequenceStatus::Waiting;
        self.block_manager.deallocate(seq);
        self.waiting.push_front(seq_id);
    }

    /// Process generated tokens: append to sequences, mark finished ones.
    /// Returns list of (seq_id, completion_token_ids) for finished sequences.
    pub fn postprocess(&mut self, seq_ids: &[usize], token_ids: &[u32]) -> Vec<(usize, Vec<u32>)> {
        let mut finished = Vec::new();

        for (&seq_id, &token_id) in seq_ids.iter().zip(token_ids.iter()) {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.append_token(token_id);

            let should_finish = (!seq.ignore_eos && token_id == self.eos_token_id)
                || seq.num_completion_tokens() == seq.max_tokens;

            if should_finish {
                seq.status = SequenceStatus::Finished;
                self.block_manager.deallocate(seq);
                self.running.retain(|&id| id != seq_id);
                finished.push((seq_id, seq.completion_token_ids().to_vec()));
            }
        }

        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_params::SamplingParams;

    fn test_config(num_blocks: usize, block_size: usize) -> EngineConfig {
        EngineConfig {
            model_path: std::path::PathBuf::from("/tmp"),
            max_num_batched_tokens: 1024,
            max_num_seqs: 4,
            max_model_len: 256,
            gpu_memory_utilization: 0.9,
            kvcache_block_size: block_size,
            num_kvcache_blocks: num_blocks,
            eos_token_id: 0,
        }
    }

    #[test]
    fn test_prefill_scheduling() {
        let config = test_config(16, 4);
        let mut sched = Scheduler::new(&config);
        let sp = SamplingParams::default();

        sched.add(Sequence::new(vec![1, 2, 3], &sp, 4));
        sched.add(Sequence::new(vec![4, 5], &sp, 4));

        let (ids, is_prefill) = sched.schedule();
        assert!(is_prefill);
        assert_eq!(ids.len(), 2);
        assert!(sched.waiting.is_empty());
        assert_eq!(sched.running.len(), 2);
    }

    #[test]
    fn test_decode_scheduling() {
        let config = test_config(16, 4);
        let mut sched = Scheduler::new(&config);
        let sp = SamplingParams::default();

        sched.add(Sequence::new(vec![1, 2], &sp, 4));

        // Prefill first
        let (ids, is_prefill) = sched.schedule();
        assert!(is_prefill);

        // Simulate a generated token
        sched.postprocess(&ids, &[10]);

        // Now decode
        let (ids, is_prefill) = sched.schedule();
        assert!(!is_prefill);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_finish_on_eos() {
        let config = test_config(16, 4);
        let mut sched = Scheduler::new(&config);
        let sp = SamplingParams::default();

        sched.add(Sequence::new(vec![1, 2], &sp, 4));

        let (ids, _) = sched.schedule();
        let finished = sched.postprocess(&ids, &[0]); // eos_token_id = 0
        assert_eq!(finished.len(), 1);
        assert!(sched.is_finished());
    }
}

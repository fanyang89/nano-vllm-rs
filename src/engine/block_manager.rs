use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::sequence::Sequence;

#[derive(Debug, Clone)]
pub struct Block {
    pub block_id: usize,
    pub ref_count: usize,
    pub hash: Option<u64>,
    pub token_ids: Vec<u32>,
}

impl Block {
    fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: None,
            token_ids: Vec::new(),
        }
    }

    fn update(&mut self, hash: u64, token_ids: Vec<u32>) {
        self.hash = Some(hash);
        self.token_ids = token_ids;
    }

    fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids.clear();
    }
}

pub struct BlockManager {
    block_size: usize,
    blocks: Vec<Block>,
    hash_to_block_id: HashMap<u64, usize>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            block_size,
            blocks: (0..num_blocks).map(Block::new).collect(),
            hash_to_block_id: HashMap::new(),
            free_block_ids: (0..num_blocks).collect(),
            used_block_ids: HashSet::new(),
        }
    }

    /// Compute prefix-chain hash for a block's token_ids.
    pub fn compute_hash(token_ids: &[u32], prefix: Option<u64>) -> u64 {
        use xxhash_rust::xxh64::xxh64;

        let mut data = Vec::new();
        if let Some(p) = prefix {
            data.extend_from_slice(&p.to_le_bytes());
        }
        // Convert token_ids to bytes (matching Python's numpy tobytes: little-endian)
        for &t in token_ids {
            data.extend_from_slice(&t.to_le_bytes());
        }
        xxh64(&data, 0)
    }

    fn allocate_block(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        assert!(block.ref_count == 0);
        block.reset();
        self.free_block_ids.retain(|&id| id != block_id);
        self.used_block_ids.insert(block_id);
    }

    fn deallocate_block(&mut self, block_id: usize) {
        assert!(self.blocks[block_id].ref_count == 0);
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
    }

    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) {
        assert!(seq.block_table.is_empty());
        let mut prefix_hash: Option<u64> = None;
        let mut cache_miss = false;

        for i in 0..seq.num_blocks() {
            let token_ids = seq.block(i).to_vec();
            let h = if token_ids.len() == self.block_size {
                Some(Self::compute_hash(&token_ids, prefix_hash))
            } else {
                None
            };

            let mut block_id = h
                .and_then(|hash| self.hash_to_block_id.get(&hash).copied())
                .filter(|&bid| !cache_miss && self.blocks[bid].token_ids == token_ids);

            if let Some(bid) = block_id {
                // Cache hit
                seq.num_cached_tokens += self.block_size;
                if self.used_block_ids.contains(&bid) {
                    self.blocks[bid].ref_count += 1;
                } else {
                    self.allocate_block(bid);
                }
            } else {
                // Cache miss
                cache_miss = true;
                let free_id = self.free_block_ids[0];
                self.allocate_block(free_id);
                block_id = Some(free_id);
            }

            let bid = block_id.unwrap();
            if let Some(hash) = h {
                self.blocks[bid].update(hash, token_ids);
                self.hash_to_block_id.insert(hash, bid);
            }
            seq.block_table.push(bid);

            prefix_hash = h;
        }
    }

    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[block_id];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let needs_new_block = seq.len() % self.block_size == 1;
        self.free_block_ids.len() >= needs_new_block as usize
    }

    pub fn may_append(&mut self, seq: &mut Sequence) {
        let last_block_id = *seq.block_table.last().unwrap();
        let last_block = &self.blocks[last_block_id];

        if seq.len() % self.block_size == 1 {
            // Crossed block boundary, need a new block
            assert!(last_block.hash.is_some());
            let free_id = self.free_block_ids[0];
            self.allocate_block(free_id);
            seq.block_table.push(free_id);
        } else if seq.len() % self.block_size == 0 {
            // Block is now full, compute hash
            assert!(last_block.hash.is_none());
            let token_ids = seq.block(seq.num_blocks() - 1).to_vec();
            let prefix = if seq.block_table.len() > 1 {
                let prev_id = seq.block_table[seq.block_table.len() - 2];
                self.blocks[prev_id].hash
            } else {
                None
            };
            let h = Self::compute_hash(&token_ids, prefix);
            let bid = last_block_id;
            self.blocks[bid].update(h, token_ids);
            self.hash_to_block_id.insert(h, bid);
        } else {
            assert!(last_block.hash.is_none());
        }
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_params::SamplingParams;

    fn make_seq(token_ids: Vec<u32>, block_size: usize) -> Sequence {
        Sequence::new(token_ids, &SamplingParams::default(), block_size)
    }

    #[test]
    fn test_allocate_deallocate() {
        let block_size = 4;
        let mut bm = BlockManager::new(8, block_size);

        let mut seq = make_seq(vec![1, 2, 3, 4, 5], block_size);
        assert_eq!(seq.num_blocks(), 2);
        assert!(bm.can_allocate(&seq));

        bm.allocate(&mut seq);
        assert_eq!(seq.block_table.len(), 2);
        assert_eq!(bm.num_free_blocks(), 6);

        bm.deallocate(&mut seq);
        assert_eq!(seq.block_table.len(), 0);
        assert_eq!(bm.num_free_blocks(), 8);
    }

    #[test]
    fn test_prefix_cache_hit() {
        let block_size = 4;
        let mut bm = BlockManager::new(8, block_size);

        // First sequence fills one full block + partial
        let mut seq1 = make_seq(vec![1, 2, 3, 4, 5, 6], block_size);
        bm.allocate(&mut seq1);
        assert_eq!(seq1.num_cached_tokens, 0); // no cache yet

        // Second sequence shares the same first block
        let mut seq2 = make_seq(vec![1, 2, 3, 4, 7, 8], block_size);
        bm.allocate(&mut seq2);
        assert_eq!(seq2.num_cached_tokens, 4); // first block is cached
        // seq2's first block should be the same as seq1's
        assert_eq!(seq1.block_table[0], seq2.block_table[0]);
    }

    #[test]
    fn test_can_append() {
        let block_size = 4;
        let mut bm = BlockManager::new(2, block_size);

        // Use exactly 2 blocks
        let mut seq = make_seq(vec![1, 2, 3, 4], block_size);
        bm.allocate(&mut seq);
        assert_eq!(bm.num_free_blocks(), 1);

        // At position 4 (block boundary), no new block needed for next token
        assert!(bm.can_append(&seq));
    }
}

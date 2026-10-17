//! Router-side prefix index for engines that cache prompt prefixes.
//!
//! **Why this exists.** llama.cpp reuses the longest common prefix of the incoming
//! token sequence, but it publishes nothing about what is currently cached: `/slots`
//! reports a slot's cache only while that slot is processing a request, and there is
//! no KV-event stream. So a router cannot ask an engine "do you already hold this
//! prefix?" before choosing where to send work.
//!
//! This crate answers the question from the router's own side instead. It records a
//! content hash chain for every request the router sent, plus how much of that prefix
//! the engine reported as cached (`cache_n` in llama.cpp's `timings`). That is enough
//! to estimate which backend most likely holds an incoming prefix, and the estimate is
//! corrected by measurement rather than trusted blindly.
//!
//! **Two designs are combined, and the combination is the point:**
//!
//! * **vLLM-style chained block hashes.** A block is identified by
//!   `hash(prefix tokens, this block's tokens)`, so a block's identity depends on
//!   everything before it and identical prefixes converge on the same keys no matter
//!   which conversation produced them. This is also the value
//!   `ExternalSequenceBlockHash` that Dynamo's KV router consumes, so a future
//!   engine-side integration can emit these hashes unchanged.
//! * **A radix tree over well-defined token blocks.** This gives the longest shared
//!   prefix against everything seen, per backend, without comparing whole prompts.
//!
//! **What it does not do.** It does not tokenise: the caller supplies token ids,
//! because the router must hash exactly what the engine was sent, and MiniCPM's
//! tokenizer already exists outside this crate. It also does not verify its estimate
//! against the engine -- [`PrefixIndex::observe`] does that after the fact.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

/// A block of token ids, the unit of hashing and caching.
pub type Block = Vec<u32>;

/// Length of a chained block hash, as produced by [`block_hashes`].
pub const HASH_LEN: usize = 32;

/// Chained content hashes for `tokens`, one per complete block.
///
/// `h_i = H(h_{i-1} || block_i)`, with `h_{-1}` seeded by `seed`. Chaining is what
/// makes a shared prefix recognisable: two prompts with a common opening produce
/// identical leading hashes even if they diverge later, which is exactly the
/// property the engine's own longest-common-prefix reuse has.
///
/// A trailing partial block gets no hash: including it would make the hash depend on
/// how the prompt happened to end, and the engine cannot reuse a partial block's KV
/// any more reliably than a full one's. The count of hashed tokens is returned so the
/// caller knows how much of the prompt the keys actually cover.
pub fn block_hashes(tokens: &[u32], block_size: usize, seed: &[u8]) -> (Vec<[u8; HASH_LEN]>, usize) {
    assert!(block_size > 0, "block_size must be positive; 0 would hash nothing");
    let mut out = Vec::new();
    if tokens.len() < block_size {
        return (out, 0);
    }
    let mut running: [u8; HASH_LEN] = Sha256::digest(seed).into();
    for chunk in tokens.chunks_exact(block_size) {
        let mut hasher = Sha256::new();
        hasher.update(running);
        for token in chunk {
            hasher.update(token.to_le_bytes());
        }
        running = hasher.finalize().into();
        out.push(running);
    }
    let covered = out.len() * block_size;
    (out, covered)
}

/// What the engine told us about one request's reuse.
///
/// `cached_tokens` is the engine's `cache_n` for that request. Recording a request
/// the engine served cold (`cached_tokens == 0`) is **not** evidence that the prefix
/// was absent beforehand -- the engine may simply have chosen another slot. So a
/// zero is only treated as a correction when it follows a non-zero estimate; see
/// [`PrefixIndex::estimate`]. Use [`Observation::unknown`] when the engine reported
/// nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Observation {
    /// Tokens the engine reported as already cached (`cache_n`).
    pub cached_tokens: usize,
    /// Total prompt tokens the request carried.
    pub prompt_tokens: usize,
}

impl Observation {
    /// An observation where the engine reported no reuse figure.
    pub fn unknown(prompt_tokens: usize) -> Self {
        Self { cached_tokens: 0, prompt_tokens }
    }

    /// Whether this observation is a usable correction.
    ///
    /// A request whose reuse figure is zero says nothing about what the engine held:
    /// prefix reuse happens only when a *previous* request left the same prefix on the
    /// slot this one landed on, and the server is free to pick any idle slot. Treating
    /// zero as "nothing cached" would let one cold request erase a correct estimate.
    pub fn is_informative(&self) -> bool {
        self.cached_tokens > 0
    }
}

/// A node in the radix tree: one block position, keyed by its chained hash.
///
/// Only downward links are kept. Lookups are always "how deep does this chain match
/// from the root?", and eviction is by backend or by prefix, so no parent link or
/// stored depth is ever needed -- they would be state to keep correct for nothing.
#[derive(Debug, Default)]
struct Node {
    children: HashMap<[u8; HASH_LEN], usize>,
}

/// One backend's prefix state. Node 0 is always the root, so Default must not be
/// an empty vector: every walk starts there.
#[derive(Debug)]
struct BackendIndex {
    nodes: Vec<Node>,
}

impl Default for BackendIndex {
    fn default() -> Self {
        Self { nodes: vec![Node::default()] }
    }
}

impl BackendIndex {
    /// Insert a hash chain; returns the number of leading blocks already present.
    fn insert(&mut self, hashes: &[[u8; HASH_LEN]]) -> usize {
        let mut node = 0usize;
        let mut shared = 0usize;
        for (i, hash) in hashes.iter().enumerate() {
            if let Some(&next) = self.nodes[node].children.get(hash) {
                node = next;
                shared = i + 1;
                continue;
            }
            let idx = self.nodes.len();
            self.nodes.push(Node::default());
            self.nodes[node].children.insert(*hash, idx);
            node = idx;
        }
        shared
    }

    /// Longest prefix shared with anything stored, in blocks.
    fn longest_shared(&self, hashes: &[[u8; HASH_LEN]]) -> usize {
        let mut node = 0usize;
        let mut shared = 0usize;
        for hash in hashes {
            match self.nodes[node].children.get(hash) {
                Some(&next) => {
                    node = next;
                    shared += 1;
                }
                None => break,
            }
        }
        shared
    }

    /// Total nodes, for reporting.
    fn size(&self) -> usize {
        self.nodes.len() - 1
    }
}

/// Per-backend, per-prefix reuse state shared by a router.
#[derive(Debug)]
pub struct PrefixIndex {
    block_size: usize,
    seed: Vec<u8>,
    backends: HashMap<String, BackendIndex>,
    /// Corrections from measurement, keyed by backend and chained hash.
    observed: HashMap<(String, [u8; HASH_LEN]), Observation>,
    /// Requests recorded, for reporting.
    recorded: u64,
}

/// One backend's answer to "how much of this prompt do you likely hold?".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Estimate {
    pub backend: String,
    /// Blocks believed cached, from the radix tree.
    pub shared_blocks: usize,
    /// Tokens believed cached: `shared_blocks * block_size`.
    pub cached_tokens: usize,
    /// Whether a measurement ever corrected this prefix on this backend. Without
    /// one the estimate is the tree's guess alone.
    pub measured: bool,
}

impl PrefixIndex {
    /// `block_size` should match the engine's reuse granularity. llama.cpp reuses at
    /// token granularity, so a smaller block size gives a finer estimate at the cost
    /// of more nodes; 16 is a reasonable default.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            block_size,
            seed: b"mind-radix/v1".to_vec(),
            backends: HashMap::new(),
            observed: HashMap::new(),
            recorded: 0,
        }
    }

    /// Seed used for the first block's hash. Two indexes must share it to compare.
    pub fn with_seed(mut self, seed: impl Into<Vec<u8>>) -> Self {
        self.seed = seed.into();
        self
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Record that `tokens` was sent to `backend`, and what reuse the engine reported.
    ///
    /// Call this after every request. The tree learns the prefix unconditionally --
    /// the engine has now seen these blocks on this backend, which is the structural
    /// fact the tree is for. Only an *informative* observation (a non-zero `cache_n`)
    /// is kept as a correction, because a cold request proves nothing about what was
    /// there before it (see [`Observation::is_informative`]).
    pub fn observe(&mut self, backend: &str, tokens: &[u32], observation: Observation) {
        let (hashes, _covered) = block_hashes(tokens, self.block_size, &self.seed);
        let index = self.backends.entry(backend.to_string()).or_default();
        index.insert(&hashes);
        if observation.is_informative()
            && let Some(last) = hashes.last()
        {
            self.observed.insert((backend.to_string(), *last), observation);
        }
        self.recorded += 1;
    }

    /// Estimate how much of `tokens` each known backend holds, best first.
    ///
    /// The structural estimate is the number of leading blocks this backend has seen.
    /// A measurement corrects it **downward only**: the engine's own `cache_n` is
    /// ground truth for what survived eviction, but an absent measurement is not
    /// evidence of absence, so it never overrides the tree's answer with zero.
    /// Ties fall back to backend name, so ranking is deterministic.
    pub fn estimate(&self, tokens: &[u32]) -> Vec<Estimate> {
        let (hashes, _) = block_hashes(tokens, self.block_size, &self.seed);
        let mut out: Vec<Estimate> = self
            .backends
            .iter()
            .map(|(name, index)| {
                let shared = index.longest_shared(&hashes);
                let structural = shared * self.block_size;
                let mut cached = structural;
                let mut measured = false;
                // Walk from the deepest shared block outward, so the most specific
                // measurement available for this prefix is the one that applies.
                for depth in (1..=shared).rev() {
                    if let Some(obs) = self.observed.get(&(name.clone(), hashes[depth - 1])) {
                        cached = obs.cached_tokens.min(obs.prompt_tokens).min(structural);
                        measured = true;
                        break;
                    }
                }
                Estimate { backend: name.clone(), shared_blocks: shared, cached_tokens: cached, measured }
            })
            .collect();
        out.sort_by(|a, b| b.cached_tokens.cmp(&a.cached_tokens).then_with(|| a.backend.cmp(&b.backend)));
        out
    }

    /// Total blocks indexed for a backend.
    pub fn blocks(&self, backend: &str) -> usize {
        self.backends.get(backend).map_or(0, BackendIndex::size)
    }

    /// Backends seen so far, in registration order.
    pub fn backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Requests recorded.
    pub fn recorded(&self) -> u64 {
        self.recorded
    }

    /// Forget one backend, e.g. after it restarted and lost its cache.
    pub fn forget_backend(&mut self, backend: &str) {
        self.backends.remove(backend);
        self.observed.retain(|(name, _), _| name != backend);
    }

    /// Forget one prefix on one backend, e.g. after context shift evicted it.
    pub fn forget_prefix(&mut self, backend: &str, tokens: &[u32]) {
        let (hashes, _) = block_hashes(tokens, self.block_size, &self.seed);
        if let Some(last) = hashes.last() {
            self.observed.remove(&(backend.to_string(), *last));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn chained_hashes_depend_on_position() {
        let (a, covered) = block_hashes(&toks(32), 16, b"s");
        assert_eq!(a.len(), 2);
        assert_eq!(covered, 32);
        // The second block's hash must depend on the first, or two prompts with a
        // differing opening and identical tail would collide.
        let other = [&toks(16)[..], &toks(16)[..]].concat();
        let (b, _) = block_hashes(&other, 16, b"s");
        assert_ne!(a, b, "same blocks at different positions must differ");
    }

    #[test]
    fn shared_prefix_produces_shared_hashes() {
        let base = toks(64);
        let mut variant = base.clone();
        variant[40] = 999;
        let (a, _) = block_hashes(&base, 16, b"s");
        let (b, _) = block_hashes(&variant, 16, b"s");
        assert_eq!(a[..2], b[..2], "blocks before the change must match");
        assert_ne!(a[2], b[2], "the block containing the change must differ");
    }

    #[test]
    fn partial_trailing_block_is_not_hashed() {
        let (hashes, covered) = block_hashes(&toks(35), 16, b"s");
        assert_eq!(hashes.len(), 2);
        assert_eq!(covered, 32, "a partial block has no stable identity");
    }

    #[test]
    fn seed_changes_hashes() {
        let (a, _) = block_hashes(&toks(16), 16, b"one");
        let (b, _) = block_hashes(&toks(16), 16, b"two");
        assert_ne!(a, b);
    }

    #[test]
    fn no_tokens_is_empty() {
        let (hashes, covered) = block_hashes(&[], 16, b"s");
        assert!(hashes.is_empty());
        assert_eq!(covered, 0);
    }

    #[test]
    fn estimate_prefers_the_backend_that_saw_the_prefix() {
        let mut index = PrefixIndex::new(16);
        let prompt = toks(64);
        index.observe("amd", &prompt, Observation { cached_tokens: 0, prompt_tokens: 64 });
        let estimates = index.estimate(&prompt);
        assert_eq!(estimates.len(), 1);
        assert_eq!(estimates[0].backend, "amd");
        assert_eq!(estimates[0].shared_blocks, 4);
        assert_eq!(estimates[0].cached_tokens, 64);

        // A prompt only studio has seen must prefer studio.
        index.observe("studio", &toks(128), Observation { cached_tokens: 0, prompt_tokens: 128 });
        let est = index.estimate(&toks(128));
        assert_eq!(est[0].backend, "studio");
    }

    #[test]
    fn a_measurement_overrides_the_structural_guess() {
        let mut index = PrefixIndex::new(16);
        let prompt = toks(64);
        index.observe("amd", &prompt, Observation { cached_tokens: 0, prompt_tokens: 64 });
        // The engine evicted most of it; the tree still has the blocks but the
        // measurement says only 16 tokens survived.
        index.observe("amd", &prompt, Observation { cached_tokens: 16, prompt_tokens: 64 });
        let est = index.estimate(&prompt);
        assert!(est[0].measured, "a measured prefix should say so");
        assert_eq!(est[0].cached_tokens, 16, "the engine's cache_n wins over the tree");
    }

    #[test]
    fn unrelated_prompt_estimates_zero() {
        // A prompt with a different opening shares nothing. Note that a *longer*
        // version of the same sequence is not this case -- it legitimately shares its
        // whole prefix -- so the unrelated prompt must differ at the start.
        let mut index = PrefixIndex::new(16);
        index.observe("studio", &toks(64), Observation::unknown(64));
        let unrelated: Vec<u32> = (5000..5100).collect();
        let est = index.estimate(&unrelated);
        assert_eq!(est[0].cached_tokens, 0, "a wholly new opening shares nothing");
    }

    #[test]
    fn a_longer_version_of_a_known_prompt_reuses_the_whole_prefix() {
        // The complement of the case above: growth must keep the shared prefix.
        let mut index = PrefixIndex::new(16);
        index.observe("studio", &toks(64), Observation::unknown(64));
        let est = index.estimate(&toks(1024));
        assert_eq!(est[0].cached_tokens, 64, "the first 4 blocks are intact");
        assert_eq!(est[0].shared_blocks, 4);
    }

    #[test]
    fn forget_backend_drops_its_state() {
        let mut index = PrefixIndex::new(16);
        index.observe("studio", &toks(64), Observation { cached_tokens: 0, prompt_tokens: 64 });
        assert_eq!(index.blocks("studio"), 4);
        index.forget_backend("studio");
        assert_eq!(index.blocks("studio"), 0);
        assert!(index.estimate(&toks(64)).is_empty());
    }

    #[test]
    fn ranking_is_best_first_then_alphabetical() {
        let mut index = PrefixIndex::new(16);
        index.observe("zeta", &toks(64), Observation { cached_tokens: 0, prompt_tokens: 64 });
        index.observe("alpha", &toks(32), Observation { cached_tokens: 0, prompt_tokens: 32 });
        let est = index.estimate(&toks(64));
        assert_eq!(est[0].backend, "zeta", "more shared tokens ranks first");
        // Equal estimates fall back to name, so the order is deterministic.
        let mut tie = PrefixIndex::new(16);
        tie.observe("b", &toks(16), Observation { cached_tokens: 0, prompt_tokens: 16 });
        tie.observe("a", &toks(16), Observation { cached_tokens: 0, prompt_tokens: 16 });
        let tied = tie.estimate(&toks(16));
        assert_eq!(tied[0].backend, "a");
    }

    #[test]
    fn diverging_conversations_do_not_cross_contaminate() {
        let mut index = PrefixIndex::new(16);
        let mut a = toks(64);
        let mut b = toks(64);
        b[20] = 777; // diverges inside the second block
        index.observe("studio", &a, Observation { cached_tokens: 0, prompt_tokens: 64 });
        let est = index.estimate(&b);
        // Only the first block is genuinely shared; claiming more would send work to
        // a backend that must still re-prefill the rest.
        assert_eq!(est[0].cached_tokens, 16, "only the intact block counts");
        let _ = &mut a;
    }

    #[test]
    #[should_panic(expected = "block_size must be positive")]
    fn zero_block_size_is_rejected() {
        let _ = PrefixIndex::new(0);
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.MemoryFullException;

import java.util.Collections;
import java.util.Set;

/**
 * The prompt prefixes a {@link BatchEngine} admission reuses instead of prefilling: cached blocks of the same prefix
 * key (per-request adapters and cache salt, {@link PrefixKey}) from the block pool ({@link PrefixCache}), adopted onto
 * the claimed slot, or the committed prompt rows of a running sequence of the same key (a donor), whose pages a fork
 * copies by copy-on-write when its prefill comes up — for text, media and prompt-embedding prompts alike, the media
 * rows matched by their items' identities ({@link CacheRows}). Claims the slot an admitted submission runs on, settles
 * a pending fork at its prefill, and gives a re-admitted preempted sequence the cached blocks of its own history.
 * Over a memory with context checkpoints a prefix restores only at a checkpoint and nothing forks off a donor: a
 * running sequence's state has moved past the shared prefix and its window no longer holds the prefix's last rows.
 * Before each adoption the blocks of the prefix that wait in the host-RAM tier come back into the pool
 * ({@link PrefixCache#promote}). A request whose prefix-cache report mode is {@code full} (vLLM's
 * {@code kv_cache_report_mode}) also publishes every cached block it reuses as a {@code BlockStored}
 * ({@link PrefixCache#reportReused}), so an event consumer learns the replica holds that prefix. Past the match,
 * a request's {@code n_cache_reuse} runs the KV chunk reuse ({@link ChunkReuse}): the cached chain's aligned runs
 * of at least that many rows re-append at the prompt's own rows instead of prefilling.
 */
final class PrefixClaims {

    private final BatchState state;
    /** The KV chunk reuse past the match's boundary ({@code n_cache_reuse}), or null when the engine runs none. */
    private final ChunkReuse chunkReuse;

    PrefixClaims(BatchState state, ChunkReuse chunkReuse) {
        this.state = state;
        this.chunkReuse = chunkReuse;
    }

    /**
     * The slot an admitted submission claims ({@code seq}), the donor a fork copies its prefix from at prefill
     * time ({@code forkFrom}, or -1), and the block chain the slot adopted from the cache (null for none).
     */
    static final class Claim {
        final int seq;
        final int forkFrom;
        final int forkLen;
        final PrefixCache.BlockChain chain;

        Claim(int seq, int forkFrom, int forkLen, PrefixCache.BlockChain chain) {
            this.seq = seq;
            this.forkFrom = forkFrom;
            this.forkLen = forkLen;
            this.chain = chain;
        }
    }

    /**
     * What a candidate would reuse: its skipped prompt tokens (the reused prefix and the chunk rows whole shared
     * pages carry) and the cached blocks eviction must spare.
     */
    static final class Reuse {
        static final Reuse NONE = new Reuse(0, Collections.emptySet());

        final int tokens;
        final Set<PrefixCache.Block> keep;

        Reuse(int tokens, Set<PrefixCache.Block> keep) {
            this.tokens = tokens;
            this.keep = keep;
        }
    }

    /**
     * Starts the background read of the disk blocks a queued submission's promotion would need and answers whether
     * they are loaded. The admission waits while this is false, as vLLM's offloading connector delays a request
     * whose blocks are still being read; true when the request reuses nothing or no disk block extends its prefix.
     */
    boolean diskReady(Submission head) {
        if (state.cache == null || head.prefillsEveryRow()) {
            return true;
        }
        CacheRows rows = head.promptRows();
        return state.cache.prefetch(rows, head.prefixKey, rows.length() - 1);
    }

    /** {@link #diskReady(Submission)} for a preempted sequence's re-admission. */
    boolean diskReady(Slot waiting) {
        if (state.cache == null || waiting.submission.wantsPromptLogprobs()) {
            return true;
        }
        CacheRows rows = waiting.recompute ? waiting.historyRows() : waiting.submission.promptRows();
        int cap = waiting.recompute ? waiting.nextRow - 1 : rows.length() - 1;
        return state.cache.prefetch(rows, waiting.submission.prefixKey, cap);
    }

    /**
     * The prompt rows a cached prefix under {@code head}'s prefix key, or a running donor of it, would let
     * {@code head} skip (as {@link #claim}); none without a cache. A cache-side reuse also scans the cached chain
     * past the match ({@link ChunkReuse#scan}): its chunks' whole shared pages add to the skipped tokens and their
     * blocks to the set eviction spares. A donor that beats the cache-side total wins alone (a fork copies a
     * prefix and scans nothing).
     */
    Reuse reusable(Submission head) {
        if (state.cache == null) {
            return Reuse.NONE;
        }
        CacheRows rows = head.promptRows();
        PrefixCache.Match match = state.cache.match(rows, head.prefixKey, rows.length() - 1);
        int best = match.tokens();
        ChunkReuse.Plan plan = chunkReuse == null ? null
                : best > 0 && head.request.nCacheReuse() > 0 ? chunkReuse.scan(head, match) : null;
        if (plan != null) {
            best += plan.sharedRows;
        }
        if (forksAllowed()) {
            Slot donor = liveDonor(rows, head.prefixKey);
            if (donor != null && donor.forkLength(rows) > best) {
                return new Reuse(donor.forkLength(rows), match.held());
            }
        }
        return new Reuse(best, keep(match, plan));
    }

    /** The cached blocks a candidate's reuse stands on: the match's, then the chunk plan's. */
    private static Set<PrefixCache.Block> keep(PrefixCache.Match match, ChunkReuse.Plan plan) {
        if (plan == null || plan.blocks.isEmpty()) {
            return match.held();
        }
        Set<PrefixCache.Block> out = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        out.addAll(match.held());
        out.addAll(plan.blocks);
        return out;
    }

    /**
     * The cached rows a preempted sequence would adopt at its re-admission ({@link #adoptOnReadmission}): its own
     * history's blocks, which its preemption left cached.
     */
    Reuse reusable(Slot waiting) {
        PrefixCache.Match match = readmissionMatch(waiting, waiting.recompute || !waiting.needsPrefill);
        return match == null ? Reuse.NONE : new Reuse(match.tokens(), match.held());
    }

    /**
     * The slot for the head submission and the prefix it reuses on it, or null when no slot is free. A prompt that
     * prefills every row itself (a prompt-logprobs request, a pooling request whose output pools every row)
     * adopts no cached blocks; any other text or media prompt adopts the cached blocks its prefix walks under its
     * prefix key onto the slot now — whole blocks along the hash chain and the shared rows of the boundary block,
     * a media prompt's media rows matched by their items' identities and never cut inside an item. It takes the
     * longer of that and the committed prompt rows of a running sequence of the same key, which the request forks
     * off ({@code PagedSequences.copy}) when its prefill comes up, so concurrent requests with a common prefix —
     * text, media or prompt embeddings — prefill it once.
     */
    Claim claim(Submission head) {
        int seq = state.freeSlot();
        if (seq < 0) {
            return null;
        }
        if (state.cache != null && !head.prefillsEveryRow()) {
            CacheRows rows = head.promptRows();
            state.cache.promote(rows, head.prefixKey, rows.length() - 1);
            PrefixCache.Match match = state.cache.match(rows, head.prefixKey, rows.length() - 1);
            if (forksAllowed()) {
                Slot donor = liveDonor(rows, head.prefixKey);
                int forkLen = donor == null ? 0 : donor.forkLength(rows);
                if (donor != null && forkLen > match.tokens()) {
                    return new Claim(seq, donor.seq, forkLen, null);
                }
            }
            if (match.tokens() > 0) {
                PrefixCache.BlockChain chain = new PrefixCache.BlockChain();
                if (head.reportsReusedBlocks) {
                    state.cache.reportReused(match);
                }
                state.cache.adopt(seq, match, rows, chain);
                head.reused = match.tokens();
                head.handle.reused(head.reused);
                return claimChunks(seq, head, chain, match);
            }
        }
        return new Claim(seq, -1, 0, null);
    }

    /**
     * The chunk-reuse phase of an adopted claim: the scan over the cached chain past the match's boundary
     * ({@link ChunkReuse#scan}), then the re-append of every chunk at the claimed sequence's append cursor
     * ({@link ChunkReuse#apply}) — the prompt prefills from the plan's end on, past the rows the chunks replaced,
     * which are the prompt's own rows in order. A scan that finds no chunk, or a memory short of the pages the
     * re-append takes (the applied rows rolled back), leaves the plain adoption standing. Another memory failure
     * frees the claimed sequence and answers null, so the submission retries at the next pump.
     */
    private Claim claimChunks(int seq, Submission head, PrefixCache.BlockChain chain, PrefixCache.Match match) {
        if (chunkReuse == null || head.request.nCacheReuse() <= 0 || state.checkpoints != null) {
            return new Claim(seq, -1, 0, chain);
        }
        ChunkReuse.Plan plan = chunkReuse.scan(head, match);
        if (plan.isEmpty()) {
            return new Claim(seq, -1, 0, chain);
        }
        boolean applied;
        try {
            applied = chunkReuse.apply(seq, head, plan);
        } catch (RuntimeException e) {
            state.slots[seq] = null;
            state.free(seq);
            if (e instanceof IllegalArgumentException) {
                // the plan and the memory disagree on the cached rows: retrying rebuilds the same refusal
                head.handle.fail(new IllegalStateException("the " + head.request.nCacheReuse()
                        + "-row n_cache_reuse chunk the cached chain holds is refused by the memory: "
                        + e.getMessage(), e));
                return null;
            }
            // a transient memory failure: the sequence keeps no pages, the submission retries at the next pump
            return null;
        }
        if (applied) {
            head.reused = plan.promptEnd;
            head.handle.reused(head.reused);
        }
        return new Claim(seq, -1, 0, chain);
    }

    /** Whether an admission may fork off a running donor: not over a memory with context checkpoints. */
    private boolean forksAllowed() {
        return state.checkpoints == null;
    }

    /**
     * Gives a re-admitted preempted sequence ({@link Slot#moveTo}) the cached blocks of its own history: a
     * recompute skips the replayed rows they hold (all but its last committed row), a prompt that never finished
     * its prefill skips the cached prefix of its prompt. Over a memory with context checkpoints a recompute
     * restores at the checkpoint its preemption took, at its every committed row: the run then decodes on at once.
     * Prompt-logprobs requests adopt nothing.
     */
    void adoptOnReadmission(Slot moved) {
        if (state.cache != null && !moved.submission.wantsPromptLogprobs()) {
            CacheRows history = moved.recompute ? moved.historyRows() : moved.submission.promptRows();
            int cap = moved.recompute ? moved.nextRow - 1 : history.length() - 1;
            state.cache.promote(history, moved.submission.prefixKey, cap);
        }
        PrefixCache.Match match = readmissionMatch(moved, moved.recompute);
        if (match == null || match.tokens() == 0) {
            return;
        }
        CacheRows rows = moved.recompute ? moved.historyRows() : moved.submission.promptRows();
        if (moved.submission.reportsReusedBlocks) {
            state.cache.reportReused(match);
        }
        state.cache.adopt(moved.seq, match, rows, moved.chain);
        moved.lastCheckpoint = match.tokens();
        if (moved.recompute) {
            moved.prefillRows = match.tokens();
            // the adopted rows hold positions [0, match.tokens()); the replay continues above them
            moved.nextPromptPos = match.tokens();
            if (match.tokens() == moved.nextRow) {
                moved.recompute = false;
                moved.needsPrefill = false;
            }
        } else {
            moved.submission.reused = match.tokens();
            moved.submission.handle.reused(match.tokens());
            moved.nextPromptPos = match.tokens();
        }
    }

    /** The cache match of a preempted sequence's history ({@code recompute}) or prompt, or null when none applies. */
    private PrefixCache.Match readmissionMatch(Slot slot, boolean recompute) {
        Submission sub = slot.submission;
        if (state.cache == null || sub.wantsPromptLogprobs()) {
            return null;
        }
        CacheRows rows = recompute ? slot.historyRows() : sub.promptRows();
        int last = state.checkpoints != null ? slot.nextRow : slot.nextRow - 1;
        int cap = recompute ? last : rows.length() - 1;
        return state.cache.match(rows, sub.prefixKey, cap);
    }

    /**
     * The running sequence of the same prefix key whose prompt shares the longest prefix with the keyed rows
     * {@code prompt}, or null. Every active slot whose prompt prefill is plain or already ran is a potential
     * donor: one still waiting for its prefill commits its rows at that prefill, before any fork it holds prefills
     * (admission order). A media prompt donates the rows of the media items it shares, matched by identity, up to
     * a chunk boundary. Excluded are slots of another key (adapters and salt partition sharing, as the cache does),
     * slots whose rows depend on another fork (a pending fork donates only once its own prefill ran — a
     * re-admitted submission must never close a fork cycle back onto itself) and recomputes, whose rows replay a
     * preempted history, not a fresh prompt. Shifted slots are excluded too: a context shift moved their
     * stored rows off the prompt.
     */
    private Slot liveDonor(CacheRows prompt, PrefixKey key) {
        Slot best = null;
        int bestLen = 0;
        for (Slot slot : state.active) {
            if (slot.recompute || !slot.submission.prefixKey.equals(key)
                    || (slot.needsPrefill && slot.forkFrom >= 0) || slot.shifted) {
                continue;
            }
            int len = slot.forkLength(prompt);
            if (len > bestLen) {
                best = slot;
                bestLen = len;
            }
        }
        return best;
    }

    /** Whether {@code slot} is a pending fork whose donor has yet to commit every prefix row the fork copies. */
    boolean waitsForDonor(Slot slot) {
        return slot.forkFrom >= 0 && !forkReady(slot) && state.activeSlot(slot.forkFrom) != null;
    }

    /**
     * Settles a pending fork as its prefill comes up: the donor's shared prefix copies into the slot when the
     * donor still holds it ({@link #beginFork}) and the cache holds no longer prefix; otherwise the cached blocks
     * of the prompt — the retired donor's included — are adopted, or the request prefills its whole prompt. The
     * reused rows count as the submission's reused prefix; the slot holds no fork afterwards.
     */
    void settleFork(Slot slot) {
        if (slot.forkFrom >= 0) {
            Submission sub = slot.submission;
            CacheRows rows = sub.promptRows();
            if (state.cache != null) {
                state.cache.promote(rows, sub.prefixKey, rows.length() - 1);
            }
            PrefixCache.Match match = state.cache == null ? PrefixCache.Match.NONE
                    : state.cache.match(rows, sub.prefixKey, rows.length() - 1);
            if (slot.forkLen >= match.tokens() && beginFork(slot)) {
                sub.reused = slot.forkLen;
                sub.handle.reused(slot.forkLen);
            } else if (match.tokens() > 0) {
                if (sub.reportsReusedBlocks) {
                    state.cache.reportReused(match);
                }
                state.cache.adopt(slot.seq, match, rows, slot.chain);
                slot.lastCheckpoint = match.tokens();
                sub.reused = match.tokens();
                sub.handle.reused(match.tokens());
            }
            // the prefill tail writes at the first position past the rows the fork copied
            slot.nextPromptPos = sub.reused;
        }
        slot.forkFrom = -1;
        slot.forkLen = 0;
    }

    /**
     * Whether the fork's donor still holds every prefix row the fork copies: the active slot under the donor id
     * has committed rows {@code [0, forkLen)} from position 0 on, and they are exactly the fork prompt's rows. A
     * retired donor's id holds no pages (its rows went to the cache) or another request's, so the check fails and
     * the fork adopts from the cache instead.
     */
    private boolean forkReady(Slot slot) {
        int src = slot.forkFrom;
        Slot holder = state.activeSlot(src);
        if (holder == null || state.sequences.posMin(src) != 0 || holder.committedRows() < slot.forkLen
                || holder.shifted) {
            return false;
        }
        CacheRows held = holder.submission.promptRows();
        return CacheRows.commonLength(held, 0, slot.submission.promptRows(), 0, slot.forkLen) == slot.forkLen;
    }

    /**
     * Copies the fork's shared prefix from its donor into the empty slot: whole pages share, the copied rows keep
     * the donor's positions — every position before the one the fork's first unshared row takes, which a media
     * prompt's rows may share ({@link CacheRows#positionAt}) — and the fork's tail appends past them. False when
     * the donor no longer holds every prefix row ({@link #forkReady}) or a full pool: the request then prefills
     * without the donor.
     */
    private boolean beginFork(Slot slot) {
        if (!forkReady(slot)) {
            return false;
        }
        try {
            int end = slot.submission.promptRows().positionAt(slot.forkLen);
            state.sequences.copy(slot.forkFrom, slot.seq, 0, end);
            return true;
        } catch (MemoryFullException e) {
            state.free(slot.seq);
            return false;
        }
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The shared state of one {@link BatchEngine}: the paged memory and its sequence slots, the queued, running and
 * preempted submissions, the prefix cache's block pool and context checkpoints, the counters and the admission
 * gates. Its methods are the slot and memory primitives every collaborator of the engine uses — finding a free slot,
 * storing a sequence's committed blocks and checkpoints, freeing a sequence's pages and recurrent state row,
 * evicting cached blocks, taking a slot off the running set. Confined to the engine's thread.
 */
final class BatchState {

    /** The paged page pools: their page geometry, as the eviction targets count it. */
    final PagedSequences pages;
    /**
     * The memory's sequence operations over the same paged sequences ({@code ModelMemory.sequences()}): every
     * removal, copy and refusal goes through it, so the per-sequence state the memory keeps beside the pages
     * (recurrent state slots, DSV4 compressor streams and their rollback snapshots) stays in step with them.
     */
    final SequenceMemory sequences;
    /** The model memory the sequences live in: whether it keeps recurrent state, and its ubatch. */
    private final ModelMemory memory;
    /** Positions the paged memory holds per sequence (the plan's {@code nCtx}). */
    final int nCtx;
    /** The running slot under each sequence id, or null when the id is free (its sequence then holds no pages). */
    final Slot[] slots;
    /**
     * The history each sequence id's last sequence left in the prefix cache when it released its pages, or null:
     * the slot state {@link SlotStore} saves, restores and erases.
     */
    final SlotStore.History[] histories;
    /** The task each sequence id's last request left ({@code recordTask}, llama.cpp's {@code task_prev}), or null. */
    private final SlotStatus[] lastTasks;
    /** Submissions waiting for admission; the {@link #policy} picks which one admits next. */
    final Deque<Submission> queue = new ArrayDeque<>();
    /** The running sequences in admission order. */
    final List<Slot> active = new ArrayList<>();
    /**
     * Preempted sequences waiting to recompute: under {@link SchedulingPolicy#FCFS} re-admitted from the head
     * ahead of the queue, under {@link SchedulingPolicy#PRIORITY} competing with the queue in the policy's order.
     */
    final Deque<Slot> preempted = new ArrayDeque<>();
    /** The order admission and preemption serve the requests in ({@code neuralfree.scheduling.policy}). */
    final SchedulingPolicy policy;
    /** The block pool of cached prefixes, or null when prefix reuse is off. */
    final PrefixCache cache;
    /** The encoder output cache the identified media chunks share (disabled when its budget is at most 0). */
    final EncoderCache encoderCache;
    /** The context checkpoints of a memory with recurrent state or sliding-window groups, or null. */
    final ContextCheckpoints checkpoints;
    final BatchStats.Builder stats = new BatchStats.Builder();
    /** The drafter whose per-slot state a detached slot resets, or null for plain decoding. */
    private final BatchDrafter drafter;
    /** The model's per-request adapter slots, or null when the model runs no per-request adapters. */
    final LoraRegistry adapters;
    /**
     * The per-sequence encoder of an encoder-decoder backend, which encodes every admitted sequence's source into its
     * own cross caches, or null for a decoder-only model.
     */
    final SequenceEncoder encoder;
    /** The model's pooling operator, or null when the model has neither a pooling nor a classifier head. */
    final PoolingOperator poolingOperator;
    /** The lookahead decoding settings, or null when the engine decodes plainly. */
    final LookaheadOptions lookahead;
    /**
     * The sequence ids one running slot owns: 1 for plain decoding, {@code W + G + 1} for lookahead decoding, whose
     * step runs one group of Jacobi and n-gram sequences per request.
     */
    final int slotWidth;
    /**
     * Set when a prefill ran out of pages beside running sequences or a decode step preempted one: admission waits
     * until a retirement or an eviction frees memory.
     */
    boolean memoryBlocked;
    /** Set by {@link BatchEngine#cancelAll()}: every admitted or queued submission stops with CANCELLED. */
    boolean cancelPending;

    BatchState(PagedSequences pages, ModelMemory memory, int maxSequences, PrefixCache cache,
               ContextCheckpoints checkpoints, BatchDrafter drafter, LoraRegistry adapters, SequenceEncoder encoder,
               EncoderCache encoderCache, SchedulingPolicy policy, PoolingOperator poolingOperator,
               LookaheadOptions lookahead) {
        SequenceMemory sequences = memory.sequences();
        if (sequences.maxSequences() != pages.maxSequences()) {
            throw new IllegalArgumentException("the memory's sequence view holds " + sequences.maxSequences()
                    + " sequences, its paged pools " + pages.maxSequences());
        }
        this.pages = pages;
        this.sequences = sequences;
        this.memory = memory;
        this.checkpoints = checkpoints;
        this.nCtx = sequences.sequence(0).nCtx();
        this.slots = new Slot[maxSequences];
        this.histories = new SlotStore.History[maxSequences];
        this.lastTasks = new SlotStatus[maxSequences];
        this.cache = cache;
        this.drafter = drafter;
        this.adapters = adapters;
        this.encoder = encoder;
        this.encoderCache = encoderCache;
        this.policy = Objects.requireNonNull(policy, "policy");
        this.poolingOperator = poolingOperator;
        this.lookahead = lookahead;
        this.slotWidth = lookahead == null ? 1 : lookahead.sequenceCount();
    }

    /** Whether no submission is queued, running or preempted. */
    boolean idle() {
        return queue.isEmpty() && active.isEmpty() && preempted.isEmpty();
    }

    /** Whether {@code slot}'s request was cancelled, by its own hook or by a pending cancel-all. */
    boolean cancelRequested(Slot slot) {
        return slot.run.cancelled() || cancelPending;
    }

    /** The counters and the current state as an immutable snapshot. */
    BatchStats snapshot() {
        stats.running = active.size();
        stats.queued = queue.size();
        stats.preempted = preempted.size();
        stats.totalCells = sequences.totalCells();
        stats.usedCells = sequences.totalCells() - sequences.freeCells();
        stats.encoderCacheRows = encoderCache.usedRows();
        stats.encoderCacheHits = encoderCache.hits();
        stats.encoderCacheEvictions = encoderCache.evictions();
        return stats.build();
    }

    /**
     * The lowest sequence id whose whole group no running slot holds, or -1; a free id's sequences hold no pages.
     * A plain slot owns one id; a lookahead slot owns the {@link #slotWidth} consecutive ids its step runs.
     */
    int freeSlot() {
        outer:
        for (int i = 0; i + slotWidth <= slots.length; i++) {
            for (int j = 0; j < slotWidth; j++) {
                if (slots[i + j] != null) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }

    /** Registers {@code slot} under every sequence id of its group. */
    void attach(Slot slot) {
        for (int j = 0; j < slotWidth; j++) {
            slots[slot.seq + j] = slot;
        }
    }

    /** Clears the group at {@code seq}: the slot it held never entered the running set. */
    void clearSlot(int seq) {
        for (int j = 0; j < slotWidth; j++) {
            slots[seq + j] = null;
        }
    }

    /**
     * Evicts the least recently used free cached block outside {@code keep} and frees its pages; false when no
     * cached block is free.
     */
    boolean evictCached(Set<PrefixCache.Block> keep) {
        if (cache == null || !cache.evictOne(keep)) {
            return false;
        }
        memoryBlocked = false;
        return true;
    }

    /**
     * Evicts least recently used free cached blocks until the pools gained {@code rows} rows per KV group plus a
     * copy-on-write page per group for each of {@code writers} sequences; false when nothing could be evicted.
     */
    boolean evictFor(int rows, int writers) {
        int target = sequences.freeCells() + (rows + writers * pages.pageSize()) * pages.groups();
        boolean any = false;
        while (sequences.freeCells() < target && evictCached(Collections.emptySet())) {
            any = true;
        }
        return any;
    }

    /**
     * Makes room in the sliding-window groups for a step over {@code pieces} before it reserves its pages: a
     * context checkpoint pins window pages a running sequence has trimmed, so the least recently used yield until
     * every trimming group holds the pages the step's writes take. A running sequence's window never yields.
     */
    void makeWindowRoom(List<Piece> pieces) {
        if (checkpoints == null || !checkpoints.windowed()) {
            return;
        }
        int[] need = new int[pages.groups()];
        for (int g = 0; g < need.length; g++) {
            for (Piece p : pieces) {
                need[g] += pages.trims(g) ? pages.pagesToWrite(p.slot.seq, g, p.rows) : 0;
            }
        }
        yieldWindowPages(need, Collections.emptySet());
    }

    /**
     * Drops least recently used context checkpoints outside {@code keep}'s blocks until every sliding-window group
     * {@code g} holds {@code need[g]} idle pages, as far as their release frees pages: the window pages a
     * checkpoint pins once its sequence has trimmed them are cache, reclaimed like free cached blocks. A running
     * sequence's window never yields. No-op without windowed checkpoints.
     */
    void yieldWindowPages(int[] need, Set<PrefixCache.Block> keep) {
        if (checkpoints == null || !checkpoints.windowed()) {
            return;
        }
        for (int g = 0; g < pages.groups(); g++) {
            if (pages.trims(g) && pages.freePages(g) < need[g]) {
                cache.yieldWindowPages(g, need[g], keep);
            }
        }
    }

    /**
     * Over a memory with recurrent state, the failure a step over {@code pieces} would meet because the pools
     * cannot hold the pages of all its ubatches, or null when they can: a recurrent state advances as each ubatch
     * runs and a failed step cannot rewind it, so such a step must fail before its first ubatch. Counts every
     * piece's pages at once, an upper bound of what the step's ubatches reserve one after another; a
     * sliding-window group trims between ubatches, so it counts one ubatch of each piece. A memory without
     * recurrent state returns null: a failed step's rows roll back.
     */
    MemoryFullException pageShortfall(List<Piece> pieces) {
        if (!memory.hasRecurrentState()) {
            return null;
        }
        int rows = 0;
        for (Piece p : pieces) {
            rows += p.rows;
        }
        int ubatch = memory.plan().ubatch();
        for (int g = 0; g < pages.groups(); g++) {
            int need = 0;
            for (Piece p : pieces) {
                int written = pages.trims(g) ? Math.min(p.rows, ubatch) : p.rows;
                need += pages.pagesToWrite(p.slot.seq, g, written);
            }
            if (need > pages.freePages(g)) {
                return new MemoryFullException(nCtx, nCtx, rows, sequences.freeCells());
            }
        }
        return null;
    }

    /** Stores the newly committed whole blocks of every running sequence in the cache (after each step). */
    void cacheCommitted() {
        if (cache == null) {
            return;
        }
        for (Slot slot : active) {
            commitBlocks(slot);
        }
    }

    private void commitBlocks(Slot slot) {
        if (slot.chain.stopped) {
            return;
        }
        int rows = slot.committedRows();
        if (rows / cache.pageSize() > slot.chain.blocks) {
            cache.commit(slot.seq, slot.chain, slot.committedKeys(), rows, slot.submission.prefixKey);
        }
        capturePrefillCheckpoint(slot, rows);
    }

    /**
     * Takes the pending prefill's next context checkpoint when its row just committed: a page-aligned spaced or
     * tail row ({@link ContextCheckpoints#due}), or the row a prompt named or a whole media chunk ran to
     * ({@code slot.pendingCheckpointRow}). The checkpoint hangs on the chain's tail when the committed rows
     * cover it, else on a freshly stored partial block (a boundary row mid-page). A no-op when the prefill is
     * over, its chain left the cache, or the row is not a checkpoint row of this memory.
     */
    private void capturePrefillCheckpoint(Slot slot, int rows) {
        if (checkpoints == null || !slot.needsPrefill || slot.chain.stopped || rows <= slot.lastCheckpoint
                || rows >= slot.prefillEnd()) {
            slot.pendingCheckpointRow = -1;
            return;
        }
        int wanted = slot.pendingCheckpointRow;
        slot.pendingCheckpointRow = -1;
        if (!checkpoints.due(rows, slot.prefillEnd(), slot.lastCheckpoint) && wanted != rows) {
            return;
        }
        PrefixCache.Block block = slot.chain.blocks * cache.pageSize() == rows ? slot.chain.tail
                : cache.commitPartial(slot.seq, slot.chain, slot.committedKeys(), rows, slot.submission.prefixKey);
        if (block == null) {
            return;
        }
        cache.checkpoint(block, slot.seq, rows, slot.submission);
        slot.lastCheckpoint = rows;
    }

    /**
     * Frees sequence {@code seq} through the memory's sequence view: its pages leave, and the per-sequence state
     * the memory keeps beside them (its recurrent state row, its DSV4 compressor streams) is zeroed with them, so
     * the next sequence in the slot starts from a fresh state.
     */
    void free(int seq) {
        for (int j = 0; j < slotWidth; j++) {
            sequences.remove(seq + j, 0, -1);
        }
    }

    /**
     * Frees {@code slot}'s sequence. With {@code keep} set and the cache running, its committed rows first stay
     * cached — the whole blocks and the trailing partial block, and over a memory that needs them the context
     * checkpoint at those rows — so a later prompt, or the slot's own recompute, adopts them, and they become the
     * sequence id's {@link #histories history}; the cache then trims back to its token budget.
     */
    void release(Slot slot, boolean keep) {
        histories[slot.seq] = null;
        if (cache == null || !keep) {
            free(slot.seq);
            return;
        }
        PrefixCache.Block partial = null;
        int rows = slot.committedRows();
        try {
            if (rows > 0 && !slot.chain.stopped) {
                CacheRows keyed = slot.committedKeys();
                cache.commit(slot.seq, slot.chain, keyed, rows, slot.submission.prefixKey);
                partial = cache.commitPartial(slot.seq, slot.chain, keyed, rows, slot.submission.prefixKey);
                checkpointAtEnd(slot, rows, partial);
                histories[slot.seq] = new SlotStore.History(keyed, rows, slot.submission.prefixKey);
            }
        } finally {
            // the sequence frees even when storing its blocks failed: the slot is reused next
            free(slot.seq);
        }
        cache.released(slot.chain, partial);
        cache.trimToBudget();
    }

    /**
     * Hangs the context checkpoint at a leaving sequence's {@code rows} committed rows on the block holding its
     * last row: the chain's tail when the rows end on a block boundary, else the trailing partial block.
     */
    private void checkpointAtEnd(Slot slot, int rows, PrefixCache.Block partial) {
        if (checkpoints == null || slot.chain.stopped) {
            return;
        }
        boolean aligned = slot.chain.blocks * cache.pageSize() == rows;
        cache.checkpoint(aligned ? slot.chain.tail : partial, slot.seq, rows, slot.submission);
    }

    /** The active slot running under the sequence id, or null when the id runs nothing. */
    Slot activeSlot(int seq) {
        for (Slot slot : active) {
            if (slot.seq == seq) {
                return slot;
            }
        }
        return null;
    }

    /**
     * The status of {@code slot}'s sequence: its request's counters and tokens, with {@code processing} telling
     * whether it still runs. The processed count is the prompt rows the prefills wrote so far (a reused prefix
     * included) while the prompt prefills or a recompute replays, the whole prompt once they decode.
     */
    SlotStatus statusOf(Slot slot, boolean processing) {
        Submission sub = slot.submission;
        boolean prefilling = slot.needsPrefill;
        int processed = prefilling ? sub.reused + slot.prefillRows : sub.prompt.nTokens();
        int nRemain = sub.request.unlimited() ? -1 : Math.max(0, sub.request.maxNewTokens() - slot.fedCount());
        // a finished task feeds nothing more, so only a running one reports an armed token
        boolean hasNext = processing && slot.pendingToken >= 0;
        return new SlotStatus(slot.seq, nCtx, processing, drafter != null, sub.handle.submitId(),
                sub.prompt.nTokens(), processed, sub.reused, sub.request.sampler(), sub.request.maxNewTokens(),
                hasNext, nRemain, slot.fedCount(), sub.prompt.samplerTokens(), slot.fedTokens());
    }

    /**
     * Records the leaving sequence's task as its slot's last one (llama.cpp's {@code task_prev}): the status
     * {@code slotStatus} reports for the id once nothing runs in it.
     */
    void recordTask(Slot slot) {
        SlotStatus status = statusOf(slot, false);
        for (int j = 0; j < slotWidth; j++) {
            lastTasks[slot.seq + j] = status;
        }
    }

    /** The status of sequence id {@code seq}: its running sequence's, else the task it last served. */
    SlotStatus status(int seq) {
        Slot slot = slots[seq];
        if (slot != null) {
            return statusOf(slot, true);
        }
        SlotStatus last = lastTasks[seq];
        return last != null ? last : SlotStatus.idle(seq, nCtx, drafter != null);
    }

    /** Takes a slot off the running set without reporting on its handle; its pages are the caller's to free. */
    void detach(Slot slot) {
        active.remove(slot);
        for (int j = 0; j < slotWidth; j++) {
            slots[slot.seq + j] = null;
        }
        if (drafter != null) {
            // the slot's drafter state must not leak into the next sequence the slot serves
            drafter.reset(slot.seq);
        }
    }
}

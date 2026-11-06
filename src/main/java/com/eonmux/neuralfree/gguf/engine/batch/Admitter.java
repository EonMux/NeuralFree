package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.memory.paged.PageDemand;

/**
 * The admission side of a {@link BatchEngine} step, in the order of the engine's {@link SchedulingPolicy}: under
 * {@code fcfs} it re-admits preempted sequences ahead of the queue, then queued submissions by arrival; under
 * {@code priority} preempted sequences and queued submissions compete in one order — the lowest {@code priority}
 * first, among equals the earliest arrival — as vLLM's waiting queue holds both. It admits while a sequence slot
 * is free and every KV group's pool holds the pages the candidate's prompt needs beside the admitted prompts still
 * waiting for prefill ({@link PageDemand}, vLLM's {@code allocate_slots} count), evicting least recently used free
 * cached blocks to make room. A re-admitted preempted sequence adopts the cached blocks of its own history first.
 * A submission with per-request adapters is a candidate only while its adapter slot can be acquired (vLLM's
 * scheduler skips a request whose LoRA would exceed {@code max_loras}); admission acquires the slot.
 */
final class Admitter {

    /** {@link #acquireAdapters}: every adapter slot serves running requests of other keys. */
    private static final int WAITS = -1;
    /** {@link #acquireAdapters}: the submission failed and was reported. */
    private static final int FAILED = -2;

    private final BatchState state;
    private final PrefixClaims claims;
    private final SlotOpener opener;
    private final SlotLifecycle lifecycle;
    private final PageDemand demand;

    Admitter(BatchState state, PrefixClaims claims, SlotOpener opener, SlotLifecycle lifecycle) {
        this.state = state;
        this.claims = claims;
        this.opener = opener;
        this.lifecycle = lifecycle;
        this.demand = new PageDemand(state.pages);
    }

    /**
     * Admits preempted sequences and queued submissions in the policy's order while slots and memory allow. A
     * blocked memory gate holds admission while sequences run.
     */
    void admit() {
        if (state.memoryBlocked) {
            if (!state.active.isEmpty()) {
                return;
            }
            // nothing runs that could free memory: the gate would never open
            state.memoryBlocked = false;
        }
        lifecycle.dropCancelledPreempted();
        if (state.policy == SchedulingPolicy.PRIORITY) {
            admitInPriorityOrder();
        } else if (readmitPreempted()) {
            admitQueued();
        }
    }

    /**
     * Moves preempted sequences, from the head of their queue, onto free slots while their recompute fits; false
     * when one still waits (the queue then waits behind it).
     */
    private boolean readmitPreempted() {
        while (!state.preempted.isEmpty()) {
            if (!readmit(state.preempted.peek())) {
                return false;
            }
        }
        return true;
    }

    /** Admits queued submissions in arrival order while each fits (strict: the head waits until it fits). */
    private void admitQueued() {
        while (!state.queue.isEmpty()) {
            Submission head = bestQueued();
            if (head == null || !admitQueued(head)) {
                return;
            }
        }
    }

    /**
     * Admits preempted sequences and queued submissions as one order — the lowest {@code priority} first, among
     * equals the earliest arrival — while each fits; strict priority: the candidate served first waits, and every
     * one behind it, until it fits (vLLM's priority waiting queue, which holds preempted requests too).
     */
    private void admitInPriorityOrder() {
        while (true) {
            Slot preempted = bestPreempted();
            Submission queued = bestQueued();
            if (preempted == null && queued == null) {
                return;
            }
            boolean admitted = preempted != null
                    && (queued == null || state.policy.servesBefore(preempted.submission, queued))
                    ? readmit(preempted) : admitQueued(queued);
            if (!admitted) {
                return;
            }
        }
    }

    /**
     * Moves the preempted sequence {@code waiting} onto a free slot when its recompute fits; false when it still
     * waits. An encoder-decoder sequence encodes its source again into its new sequence id's cross caches before
     * its recompute; a failing encoding retires that sequence alone.
     */
    private boolean readmit(Slot waiting) {
        if (!claims.diskReady(waiting)) {
            return false;
        }
        int rows = waiting.recompute || !waiting.needsPrefill ? waiting.nextRow
                : waiting.submission.prompt.nTokens();
        if (!makeRoom(rows, claims.reusable(waiting))) {
            return false;
        }
        int seq = state.freeSlot();
        if (seq < 0) {
            return false;
        }
        state.preempted.remove(waiting);
        Slot moved = waiting.moveTo(seq);
        claims.adoptOnReadmission(moved);
        state.attach(moved);
        state.active.add(moved);
        moved.submission.handle.admit(seq);
        try {
            opener.encode(seq, moved.submission);
        } catch (RuntimeException e) {
            lifecycle.retire(moved, null, false, e);
        }
        return true;
    }

    /**
     * Admits the queued submission {@code head} when its prompt fits; false when it waits, left in the queue. A
     * submission whose adapter left the registry, or whose pipeline refuses to build, fails alone (true: the
     * candidates behind it go on).
     */
    private boolean admitQueued(Submission head) {
        // a prompt whose disk blocks are still being read waits, as vLLM's offloading connector delays its request
        if (!head.prefillsEveryRow() && !claims.diskReady(head)) {
            return false;
        }
        // a submission whose output reads every prompt row back (prompt logprobs, a pooling output that pools
        // every row) prefills every row itself and reuses no cached prefix
        PrefixClaims.Reuse reuse = head.prefillsEveryRow() ? PrefixClaims.Reuse.NONE : claims.reusable(head);
        if (!makeRoom(head.prompt.nTokens(), reuse)) {
            return false;
        }
        int adapterSlot = acquireAdapters(head);
        if (adapterSlot == FAILED) {
            state.queue.remove(head);
            return true;
        }
        if (adapterSlot == WAITS) {
            return false;
        }
        PrefixClaims.Claim claim = claims.claim(head);
        if (claim == null) {
            lifecycle.releaseAdapters(head);
            return false;
        }
        state.queue.remove(head);
        try {
            state.active.add(opener.open(claim, head));
        } catch (RuntimeException e) {
            // the request's own pipeline refused to build: free what the claim adopted and report it
            state.clearSlot(claim.seq);
            state.free(claim.seq);
            lifecycle.failed(head, e);
        }
        return true;
    }

    /**
     * Gives a submission being admitted its adapter slot ({@code sub.adapterSlot}) and returns it: 0 for one
     * without per-request adapters, {@link #WAITS} when every slot serves running requests of other keys,
     * {@link #FAILED} when the submission failed (its adapter was unloaded after it was submitted, or the slot
     * could not be opened).
     */
    private int acquireAdapters(Submission sub) {
        if (sub.adapters.isDefault()) {
            return 0;
        }
        int slot;
        try {
            slot = sub.adapters.key() == null ? 0 : state.adapters.acquire(sub.adapters.key());
            if (slot < 0) {
                return WAITS;
            }
            sub.adapterSlot = slot;
            LoraSlotKey pre = sub.adapters.preKey();
            sub.preAdapterSlot = pre == null ? 0 : pre.equals(sub.adapters.key()) ? slot : state.adapters.acquire(pre);
        } catch (RuntimeException e) {
            lifecycle.failed(sub, e);
            return FAILED;
        }
        return slot;
    }

    /**
     * The queued submission the engine admits next, left in the queue: the one the policy serves first (the
     * earliest arrival, under {@code priority} the lowest {@code priority} value first). A submission whose
     * adapter slot cannot be acquired now is passed over; null when every queued one is.
     */
    private Submission bestQueued() {
        Submission best = null;
        for (Submission candidate : state.queue) {
            if (adaptersAvailable(candidate)
                    && (best == null || state.policy.servesBefore(candidate, best))) {
                best = candidate;
            }
        }
        return best;
    }

    /** The preempted sequence the policy serves first, or null when none waits. */
    private Slot bestPreempted() {
        Slot best = null;
        for (Slot candidate : state.preempted) {
            if (best == null || state.policy.servesBefore(candidate.submission, best.submission)) {
                best = candidate;
            }
        }
        return best;
    }

    /** Whether {@code sub} runs no per-request adapters or its adapter slot can be acquired now. */
    private boolean adaptersAvailable(Submission sub) {
        return sub.adapters.key() == null || state.adapters.canAcquire(sub.adapters.key());
    }

    /**
     * The admission hint: evicts least recently used free cached blocks until every KV group's idle pages cover the
     * pages a candidate needs to hold its {@code rows} prompt rows, the prefix it would reuse ({@code reuse}: from
     * the cache or a running donor; a prompt-logprobs request reads back every prompt row and reuses nothing) shared
     * whole pages and a shared partial last page its copy-on-write copy, plus the pages the admitted prompts still
     * waiting for prefill take. A sliding-window group short of pages first takes the window pages context
     * checkpoints pin ({@link BatchState#yieldWindowPages}). The blocks the candidate would adopt, and their
     * checkpoints, are spared. With nothing left to evict it admits only onto an engine with no running sequence
     * (the prefill then fits or fails by name); otherwise the candidate waits for the running sequences.
     */
    private boolean makeRoom(int rows, PrefixClaims.Reuse reuse) {
        int[] need = pendingPrefillPages();
        for (int g = 0; g < need.length; g++) {
            need[g] += demand.toHold(g, rows, reuse.tokens);
        }
        state.yieldWindowPages(need, reuse.keep);
        while (!fits(need)) {
            if (!state.evictCached(reuse.keep)) {
                return state.active.isEmpty();
            }
        }
        return true;
    }

    /** Whether every KV group's pool holds {@code need[g]} idle pages. */
    private boolean fits(int[] need) {
        for (int g = 0; g < need.length; g++) {
            if (demand.freePages(g) < need[g]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Pages per KV group the admitted sequences that still wait for their prefill take to write their unwritten
     * rows: a pending fork, which holds nothing until it copies its donor's prefix, the pages past the whole ones
     * it will share (its copy-on-write page of a shared partial last page included); any other sequence the pages
     * its remaining rows reach beyond those it holds, privatizing the shared ones.
     */
    private int[] pendingPrefillPages() {
        int[] pages = new int[demand.groups()];
        for (Slot slot : state.active) {
            if (!slot.needsPrefill) {
                continue;
            }
            for (int g = 0; g < pages.length; g++) {
                pages[g] += slot.forkFrom >= 0
                        ? demand.toHold(g, slot.submission.prompt.nTokens(), slot.forkLen)
                        : demand.toAppend(slot.seq, g, slot.pendingTotal() - slot.prefillRows);
            }
        }
        return pages;
    }
}

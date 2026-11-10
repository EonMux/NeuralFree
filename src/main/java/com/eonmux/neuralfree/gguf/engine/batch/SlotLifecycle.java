package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * How a {@link BatchEngine} sequence leaves the running set: retirement (finished, cancelled or failed, reported
 * on the handle, its pages cached or freed), recompute preemption (pages freed, run kept, re-admitted as the
 * scheduling policy orders it), and the recovery of a step that ran out of pages — a prefill requeues at the head
 * of the queue, a recompute preempts again, a decode step evicts cached blocks or preempts the running sequence the
 * scheduling policy serves last ({@link #victim}). A sequence that leaves without error keeps its committed rows in
 * the prefix cache's block pool.
 */
final class SlotLifecycle {

    private final BatchState state;

    SlotLifecycle(BatchState state) {
        this.state = state;
    }

    /** Stops a sequence with an external reason: its result lands on the handle and its slot frees at once. */
    void finish(Slot slot, StopReason reason) {
        slot.run.stop(reason);
        retire(slot, slot.run.result(), reason == StopReason.CANCELLED, null);
    }

    /**
     * Frees the paged sequence and the slot, and reports the outcome on the handle. A sequence that ended without
     * error leaves its committed rows cached ({@link BatchState#release}) when the cache runs and its prompt is
     * text-only.
     */
    void retire(Slot slot, GenerationResult result, boolean cancelled, Throwable error) {
        state.detach(slot);
        state.recordTask(slot);
        state.memoryBlocked = false;
        try {
            state.release(slot, error == null);
        } catch (RuntimeException e) {
            if (error != null) {
                e.addSuppressed(error);
            }
            failed(slot.submission, e);
            return;
        }
        if (error != null) {
            failed(slot.submission, error);
        } else {
            completed(slot.submission, result, cancelled);
        }
    }

    /**
     * Reports a finished submission on its handle and counts it under its stop reason and its length; its adapter
     * slot and its encoder cache references are released.
     */
    void completed(Submission sub, GenerationResult result, boolean cancelled) {
        releaseAdapters(sub);
        releaseMedia(sub);
        StopReason reason = cancelled || result == null ? StopReason.CANCELLED : result.stopReason();
        state.stats.finished[reason.ordinal()]++;
        int length = sub.prompt.nTokens() + (result == null ? 0 : result.tokens().length);
        state.stats.maxSequenceLength = Math.max(state.stats.maxSequenceLength, length);
        if (sub.promptLogprobs != null && result != null) {
            // only a fully prefilled prompt reports the capture: a stopped or cancelled run drops its partial
            result = sub.promptLogprobs.attach(result);
        }
        sub.handle.complete(result, cancelled);
    }

    /**
     * Reports a failed submission on its handle and counts it; its adapter slot and its encoder cache references are
     * released.
     */
    void failed(Submission sub, Throwable error) {
        releaseAdapters(sub);
        releaseMedia(sub);
        state.stats.failed++;
        sub.handle.fail(error);
    }

    /**
     * Retires a pooling request at its prompt's completion: its rows stay cached like a finished generation's, its
     * adapter slot and encoder cache references release, and its pooling run applies the operator over the
     * accumulated rows and completes the handle ({@code DONE}). A run the operator refuses (or a failing release)
     * fails the request instead.
     */
    void poolingRetire(Slot slot) {
        state.detach(slot);
        state.recordTask(slot);
        state.memoryBlocked = false;
        Submission sub = slot.submission;
        Throwable failure = null;
        try {
            state.release(slot, true);
        } catch (RuntimeException e) {
            failure = e;
        }
        releaseAdapters(sub);
        releaseMedia(sub);
        if (failure == null) {
            try {
                sub.pooling.complete(state.poolingOperator, sub.prompt.samplerTokens());
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        if (failure != null) {
            state.stats.failed++;
            sub.handle.fail(failure);
        }
    }

    /** Drops {@code sub}'s encoder cache references, once: a later outcome report of the same submission keeps them. */
    void releaseMedia(Submission sub) {
        if (!sub.mediaReleased) {
            sub.mediaReleased = true;
            state.encoderCache.release(sub.mediaIds);
        }
    }

    /**
     * Releases the adapter slot {@code sub} holds, once: a submission holds its slot from admission through
     * preemption until it retires or requeues.
     */
    void releaseAdapters(Submission sub) {
        int pre = sub.preAdapterSlot;
        int slot = sub.adapterSlot;
        sub.preAdapterSlot = 0;
        sub.adapterSlot = 0;
        if (slot > 0) {
            state.adapters.release(slot);
        }
        if (pre > 0 && pre != slot) {
            state.adapters.release(pre);
        }
    }

    /** Preempts every running sequence (see {@link BatchEngine#preemptAll()}) and reopens admission. */
    void preemptAll() {
        for (Slot slot : new ArrayList<>(state.active)) {
            state.detach(slot);
            state.release(slot, true);
            state.preempted.addLast(slot);
            countPreemption(slot);
        }
        state.memoryBlocked = false;
    }

    /** Finishes the preempted sequences whose request was cancelled while they waited (they hold no pages). */
    void dropCancelledPreempted() {
        for (Iterator<Slot> it = state.preempted.iterator(); it.hasNext(); ) {
            Slot slot = it.next();
            if (state.cancelRequested(slot)) {
                it.remove();
                slot.run.stop(StopReason.CANCELLED);
                completed(slot.submission, slot.run.result(), true);
            }
        }
    }

    /** Fails every running, preempted and queued submission with {@code reason} and empties the engine. */
    void failAll(String reason) {
        for (Slot slot : new ArrayList<>(state.active)) {
            retire(slot, null, false, new IllegalStateException(reason));
        }
        state.active.clear();
        for (Slot slot : state.preempted) {
            failed(slot.submission, new IllegalStateException(reason));
        }
        state.preempted.clear();
        for (Submission sub : state.queue) {
            failed(sub, new IllegalStateException(reason));
        }
        state.queue.clear();
    }

    /**
     * Frees memory after a step ran out of pages: a single-piece step recovers along its own path (a decode
     * through {@link #relieve}, a recompute preempts again, a prompt chunk requeues), a step over several pieces
     * cannot attribute the failing ubatch, so memory is freed by {@link #relieve} or every piece's sequence
     * fails with the exception.
     */
    void recoverMemory(List<Piece> pieces, MemoryFullException e) {
        if (pieces.size() == 1) {
            Piece p = pieces.get(0);
            if (p.decode) {
                if (!relieve(pieces)) {
                    retire(p.slot, null, false, e);
                }
            } else if (p.slot.recompute) {
                preemptAgain(p.slot, e);
            } else {
                requeue(p.slot, e);
            }
            return;
        }
        if (!relieve(pieces)) {
            for (Piece p : pieces) {
                retire(p.slot, null, false, e);
            }
        }
    }

    /**
     * Puts a submission whose prefill ran out of pages back at the head of the queue: its sequence frees, the
     * rows it committed staying cached (the retry adopts them), then cached blocks are evicted for the rows the
     * prefill still lacks, or admission waits for the running sequences. With neither, nothing else holds memory
     * and the request fails with the exception.
     */
    private void requeue(Slot slot, MemoryFullException e) {
        Submission sub = slot.submission;
        state.detach(slot);
        int lacking = slot.pendingTotal() - slot.prefillRows;
        state.release(slot, true);
        sub.reused = 0;
        sub.handle.reused(0);
        // the prefill starts over: a prompt-logprobs capture and a pooling run restart with it
        sub.promptLogprobs = null;
        if (sub.pooling != null) {
            sub.pooling.reset();
        }
        // a queued submission holds no adapter slot: admission acquires it again
        releaseAdapters(sub);
        if (!blockAdmission(lacking)) {
            failed(sub, e);
            return;
        }
        sub.handle.requeued();
        state.queue.addFirst(sub);
    }

    /**
     * Re-preempts a sequence whose recompute ran out of pages: its sequence frees, the replayed rows staying
     * cached (the re-admission adopts them), cached blocks are evicted for the rows the replay still lacks, or
     * admission then waits for the running sequences. With neither, nothing else holds memory and the sequence
     * fails with the exception.
     */
    private void preemptAgain(Slot slot, MemoryFullException e) {
        state.detach(slot);
        int lacking = slot.pendingTotal() - slot.prefillRows;
        state.release(slot, true);
        slot.prefillRows = 0;
        if (!blockAdmission(lacking)) {
            failed(slot.submission, e);
            return;
        }
        state.preempted.addFirst(slot);
    }

    /**
     * Makes room for a prefill that ran out of pages to retry: evicts cached blocks for its {@code rows}, and
     * blocks admission until a retirement when nothing could be evicted. False when nothing else holds memory (no
     * sequence runs), so the retry could never fit.
     */
    private boolean blockAdmission(int rows) {
        if (!state.evictFor(rows, 1)) {
            if (state.active.isEmpty()) {
                return false;
            }
            state.memoryBlocked = true;
        }
        return true;
    }

    /**
     * Frees memory after a step over {@code pieces} ran out of pages: free cached blocks go first, else the
     * running sequence the policy serves last is preempted ({@link #victim}), so the one it serves first always
     * progresses. The preempted sequence's committed rows stay cached for its recompute, and as the least recent
     * free blocks give way to the retry; admission waits for a retirement, so the survivors
     * decode before the preempted sequence recomputes (an immediate re-admission would refill the pool and
     * re-preempt it). A fork that only just copied its prefix frees no page of its own, but its leaving un-shares
     * the pages its donor writes next; its rows are its donor's or the cache's, so they are not cached again (a
     * cached copy would keep the shared page pinned). False when nothing yields memory.
     */
    private boolean relieve(List<Piece> pieces) {
        int rows = 0;
        for (Piece p : pieces) {
            rows += p.rows;
        }
        if (state.evictFor(rows, pieces.size())) {
            return true;
        }
        if (state.active.size() < 2) {
            return false;
        }
        Slot victim = victim();
        boolean keep = !freshFork(victim);
        state.detach(victim);
        state.release(victim, keep);
        state.preempted.addFirst(victim);
        countPreemption(victim);
        state.evictFor(rows, pieces.size());
        state.memoryBlocked = true;
        return true;
    }

    /**
     * The running sequence a memory shortage preempts, as vLLM's scheduler picks it: under
     * {@link SchedulingPolicy#FCFS} the youngest (the back of the running set, its admission order), under
     * {@link SchedulingPolicy#PRIORITY} the one the policy serves last — the highest {@code priority} value, among
     * equals the latest arrival ({@code max(running, key=(priority, arrival_time))}).
     */
    private Slot victim() {
        Slot victim = state.active.get(state.active.size() - 1);
        if (state.policy == SchedulingPolicy.FCFS) {
            return victim;
        }
        for (Slot slot : state.active) {
            if (state.policy.servesBefore(victim.submission, slot.submission)) {
                victim = slot;
            }
        }
        return victim;
    }

    /** Counts a preemption of {@code slot} on the engine's counters and on its request's handle. */
    private void countPreemption(Slot slot) {
        state.stats.preemptions++;
        slot.submission.handle.preempted();
    }

    /**
     * Whether {@code slot} holds only a just-copied or adopted prefix, its pages shared with another holder: it
     * committed no row of its own yet.
     */
    private boolean freshFork(Slot slot) {
        return slot.needsPrefill && !slot.recompute && slot.prefillRows == 0 && slot.submission.reused > 0
                && state.sequences.sharesPages(slot.seq);
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.Arrays;

/**
 * A snapshot of a {@link BatchEngine}'s work since it opened (the counters) and of its state at the snapshot (the
 * gauges): the figures behind llama.cpp's {@code /metrics} ({@code server_task_result_metrics}) and vLLM's scheduler
 * stats. Immutable; {@link BatchEngine#stats()} builds one on the engine's thread, which may hand it to any other.
 */
public final class BatchStats {

    /** A snapshot of an engine that has done nothing. */
    public static final BatchStats EMPTY = new Builder().build();

    private final long promptTokens;
    private final long promptTokensCached;
    private final long promptNanos;
    private final long generatedTokens;
    private final long decodeNanos;
    private final long decodeSteps;
    private final long decodeRows;
    private final long recomputedTokens;
    private final long preemptions;
    private final long[] finished;
    private final long failed;
    private final int maxSequenceLength;
    private final int running;
    private final int queued;
    private final int preempted;
    private final int usedCells;
    private final int totalCells;

    private BatchStats(Builder b) {
        this.promptTokens = b.promptTokens;
        this.promptTokensCached = b.promptTokensCached;
        this.promptNanos = b.promptNanos;
        this.generatedTokens = b.generatedTokens;
        this.decodeNanos = b.decodeNanos;
        this.decodeSteps = b.decodeSteps;
        this.decodeRows = b.decodeRows;
        this.recomputedTokens = b.recomputedTokens;
        this.preemptions = b.preemptions;
        this.finished = b.finished.clone();
        this.failed = b.failed;
        this.maxSequenceLength = b.maxSequenceLength;
        this.running = b.running;
        this.queued = b.queued;
        this.preempted = b.preempted;
        this.usedCells = b.usedCells;
        this.totalCells = b.totalCells;
    }

    /** Prompt tokens prefilled, excluding those reused from the prefix cache. */
    public long promptTokens() { return promptTokens; }

    /** Prompt tokens reused from the prefix cache instead of prefilled. */
    public long promptTokensCached() { return promptTokensCached; }

    /** Time spent prefilling prompts. */
    public double promptSeconds() { return promptNanos / 1e9; }

    /** Tokens sampled. */
    public long generatedTokens() { return generatedTokens; }

    /** Time spent in decode steps. */
    public double decodeSeconds() { return decodeNanos / 1e9; }

    /** Batched decode forwards. */
    public long decodeSteps() { return decodeSteps; }

    /** Sequences decoded per batched decode forward, on average (llama.cpp's {@code n_busy_slots_per_decode}). */
    public double rowsPerDecode() { return decodeRows / (double) Math.max(decodeSteps, 1L); }

    /** Tokens prefilled again after a preemption. */
    public long recomputedTokens() { return recomputedTokens; }

    /** Running sequences preempted to free memory. */
    public long preemptions() { return preemptions; }

    /** Sequences that finished for {@code reason} (cancellations under {@link StopReason#CANCELLED}). */
    public long finished(StopReason reason) { return finished[reason.ordinal()]; }

    /** Sequences that failed. */
    public long failed() { return failed; }

    /** The longest finished request: its prompt and generated tokens (llama.cpp's {@code n_tokens_max}). */
    public int maxSequenceLength() { return maxSequenceLength; }

    /** Sequences prefilling or decoding at the snapshot. */
    public int running() { return running; }

    /** Submissions waiting for a slot, and preempted sequences waiting to recompute, at the snapshot. */
    public int waiting() { return queued + preempted; }

    /** Preempted sequences waiting to recompute at the snapshot. */
    public int preempted() { return preempted; }

    /**
     * The share of the KV memory's rows in use at the snapshot, in [0, 1]; rows the prefix cache holds count as
     * used until evicted.
     */
    public double kvUsage() { return totalCells <= 0 ? 0.0 : usedCells / (double) totalCells; }

    @Override
    public String toString() {
        return "BatchStats[prompt=" + promptTokens + ", cached=" + promptTokensCached + ", generated="
                + generatedTokens + ", steps=" + decodeSteps + ", preemptions=" + preemptions + ", finished="
                + Arrays.toString(finished) + ", failed=" + failed + ", running=" + running + ", waiting="
                + waiting() + "]";
    }

    /** The engine's accumulator: mutable, touched by the engine's thread only. */
    static final class Builder {
        long promptTokens;
        long promptTokensCached;
        long promptNanos;
        long generatedTokens;
        long decodeNanos;
        long decodeSteps;
        long decodeRows;
        long recomputedTokens;
        long preemptions;
        final long[] finished = new long[StopReason.values().length];
        long failed;
        int maxSequenceLength;
        int running;
        int queued;
        int preempted;
        int usedCells;
        int totalCells;

        BatchStats build() {
            return new BatchStats(this);
        }
    }
}

package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.util.Arrays;
import java.util.Objects;

/**
 * A snapshot of a {@link BatchEngine}'s work since it opened (the counters) and of its state at the snapshot (the
 * gauges): the figures behind llama.cpp's {@code /metrics} ({@code server_task_result_metrics}) and vLLM's scheduler
 * stats, with vLLM's speculative-decoding series ({@code spec_decode_num_drafted_tokens} and friends) when the
 * engine speculates. Immutable; {@link BatchEngine#stats()} builds one on the engine's thread, which may hand it
 * to any other.
 */
public final class BatchStats {

    /** A snapshot of an engine that has done nothing. */
    public static final BatchStats EMPTY = new Builder().build();

    private final long promptTokens;
    private final long promptTokensCached;
    private final long promptNanos;
    private final long generatedTokens;
    private final long deviceSampledRows;
    private final long decodeNanos;
    private final long decodeSteps;
    private final long decodeRows;
    private final long recomputedTokens;
    private final long preemptions;
    private final long contextShifts;
    private final long grpAttnEdits;
    private final long chunkRowsReused;
    private final long specRounds;
    private final long specDrafted;
    private final long specAccepted;
    private final long lookaheadRounds;
    private final long lookaheadEmitted;
    private final long lookaheadAccepted;
    private final long[] finished;
    private final long failed;
    private final int maxSequenceLength;
    private final int running;
    private final int queued;
    private final int preempted;
    private final int usedCells;
    private final int totalCells;
    private final int encoderCacheRows;
    private final long encoderCacheHits;
    private final long encoderCacheEvictions;

    private BatchStats(Builder b) {
        this.promptTokens = b.promptTokens;
        this.promptTokensCached = b.promptTokensCached;
        this.promptNanos = b.promptNanos;
        this.generatedTokens = b.generatedTokens;
        this.deviceSampledRows = b.deviceSampledRows;
        this.decodeNanos = b.decodeNanos;
        this.decodeSteps = b.decodeSteps;
        this.decodeRows = b.decodeRows;
        this.recomputedTokens = b.recomputedTokens;
        this.preemptions = b.preemptions;
        this.contextShifts = b.contextShifts;
        this.grpAttnEdits = b.grpAttnEdits;
        this.chunkRowsReused = b.chunkRowsReused;
        this.specRounds = b.specRounds;
        this.specDrafted = b.specDrafted;
        this.specAccepted = b.specAccepted;
        this.lookaheadRounds = b.lookaheadRounds;
        this.lookaheadEmitted = b.lookaheadEmitted;
        this.lookaheadAccepted = b.lookaheadAccepted;
        this.finished = b.finished.clone();
        this.failed = b.failed;
        this.maxSequenceLength = b.maxSequenceLength;
        this.running = b.running;
        this.queued = b.queued;
        this.preempted = b.preempted;
        this.usedCells = b.usedCells;
        this.totalCells = b.totalCells;
        this.encoderCacheRows = b.encoderCacheRows;
        this.encoderCacheHits = b.encoderCacheHits;
        this.encoderCacheEvictions = b.encoderCacheEvictions;
    }

    /** Prompt tokens prefilled, excluding those reused from the prefix cache. */
    public long promptTokens() { return promptTokens; }

    /** Prompt tokens reused from the prefix cache instead of prefilled. */
    public long promptTokensCached() { return promptTokensCached; }

    /** Time spent in forward passes that carried prompt rows (a mixed step counts in this and {@link #decodeSeconds()}). */
    public double promptSeconds() { return promptNanos / 1e9; }

    /** Tokens sampled. */
    public long generatedTokens() { return generatedTokens; }

    /**
     * Tokens sampled over a device top-k readback ({@code LLM_Row_TopK}, llama.cpp backend sampling {@code -bs}):
     * the batched steps whose chain kept its candidates inside the cone, so only the candidates were read back.
     */
    public long deviceSampledRows() { return deviceSampledRows; }

    /** Time spent in forward passes that carried decode rows (a mixed step counts in this and {@link #promptSeconds()}). */
    public double decodeSeconds() { return decodeNanos / 1e9; }

    /** Batched forwards that carried decode rows (a chunked prefill's step beside decodes counts here too). */
    public long decodeSteps() { return decodeSteps; }

    /** Sequences decoded per batched decode forward, on average (llama.cpp's {@code n_busy_slots_per_decode}). */
    public double rowsPerDecode() { return decodeRows / (double) Math.max(decodeSteps, 1L); }

    /** Tokens prefilled again after a preemption. */
    public long recomputedTokens() { return recomputedTokens; }

    /** Running sequences preempted to free memory. */
    public long preemptions() { return preemptions; }

    /** Context shifts that moved a decoding sequence's stored positions (llama.cpp's shifted slots). */
    public long contextShifts() { return contextShifts; }

    /** Position compressions a self-extend (group attention) ran: one per group window (llama.cpp's {@code ga_n}). */
    public long grpAttnEdits() { return grpAttnEdits; }

    /** Prompt rows a KV chunk reuse re-appended from cached pages, past the rows the prefix cache already shared. */
    public long chunkRowsReused() { return chunkRowsReused; }

    /** Speculative verify rounds run ({@code vllm:spec_decode_num_steps}). */
    public long specRounds() { return specRounds; }

    /** Proposed tokens the target verified ({@code vllm:spec_decode_num_drafted_tokens}). */
    public long specDrafted() { return specDrafted; }

    /** Proposed tokens the target confirmed ({@code vllm:spec_decode_num_accepted_tokens}). */
    public long specAccepted() { return specAccepted; }

    /** {@code specAccepted / specDrafted}, 0 when nothing was drafted. */
    public double specAcceptanceRate() {
        return specDrafted == 0 ? 0.0 : (double) specAccepted / specDrafted;
    }

    /** Lookahead (Jacobi) steps run (llama.cpp's lookahead example {@code n_drafted}). */
    public long lookaheadRounds() { return lookaheadRounds; }

    /** Tokens lookahead steps emitted, the target's own token of each step included. */
    public long lookaheadEmitted() { return lookaheadEmitted; }

    /** Tokens lookahead steps accepted from an observed n-gram beyond each step's own token. */
    public long lookaheadAccepted() { return lookaheadAccepted; }

    /** {@code lookaheadAccepted / lookaheadRounds}, the mean accepted tokens per step beyond its own token. */
    public double lookaheadAcceptanceRate() {
        return lookaheadRounds == 0 ? 0.0 : (double) lookaheadAccepted / lookaheadRounds;
    }

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

    /** Rows of shared encoder output the engine's encoder cache holds at the snapshot (vLLM's encoder cache size in use). */
    public int encoderCacheRows() { return encoderCacheRows; }

    /** Identified media chunks whose encoder output the encoder cache already held, so their rows were shared. */
    public long encoderCacheHits() { return encoderCacheHits; }

    /** Encoder cache entries evicted since the engine opened (vLLM's freed {@code mm_hash} count). */
    public long encoderCacheEvictions() { return encoderCacheEvictions; }

    /** The share of prompt tokens served from the prefix cache so far, in [0, 1]; 0 before any prompt work. */
    public double prefixCacheHitRate() {
        long total = promptTokens + promptTokensCached;
        return total == 0 ? 0.0 : promptTokensCached / (double) total;
    }

    /**
     * The combined snapshot of several engines (the data-parallel aggregate): the counters add, the running and
     * waiting counts and the KV cells add, and the longest finished sequence keeps the larger of the two.
     */
    public BatchStats plus(BatchStats other) {
        Objects.requireNonNull(other, "other");
        Builder b = new Builder();
        b.promptTokens = promptTokens + other.promptTokens;
        b.promptTokensCached = promptTokensCached + other.promptTokensCached;
        b.promptNanos = promptNanos + other.promptNanos;
        b.generatedTokens = generatedTokens + other.generatedTokens;
        b.deviceSampledRows = deviceSampledRows + other.deviceSampledRows;
        b.decodeNanos = decodeNanos + other.decodeNanos;
        b.decodeSteps = decodeSteps + other.decodeSteps;
        b.decodeRows = decodeRows + other.decodeRows;
        b.recomputedTokens = recomputedTokens + other.recomputedTokens;
        b.preemptions = preemptions + other.preemptions;
        b.contextShifts = contextShifts + other.contextShifts;
        b.grpAttnEdits = grpAttnEdits + other.grpAttnEdits;
        b.chunkRowsReused = chunkRowsReused + other.chunkRowsReused;
        b.specRounds = specRounds + other.specRounds;
        b.specDrafted = specDrafted + other.specDrafted;
        b.specAccepted = specAccepted + other.specAccepted;
        b.lookaheadRounds = lookaheadRounds + other.lookaheadRounds;
        b.lookaheadEmitted = lookaheadEmitted + other.lookaheadEmitted;
        b.lookaheadAccepted = lookaheadAccepted + other.lookaheadAccepted;
        for (int i = 0; i < finished.length; i++) {
            b.finished[i] = finished[i] + other.finished[i];
        }
        b.failed = failed + other.failed;
        b.maxSequenceLength = Math.max(maxSequenceLength, other.maxSequenceLength);
        b.running = running + other.running;
        b.queued = queued + other.queued;
        b.preempted = preempted + other.preempted;
        b.usedCells = usedCells + other.usedCells;
        b.totalCells = totalCells + other.totalCells;
        b.encoderCacheRows = encoderCacheRows + other.encoderCacheRows;
        b.encoderCacheHits = encoderCacheHits + other.encoderCacheHits;
        b.encoderCacheEvictions = encoderCacheEvictions + other.encoderCacheEvictions;
        return b.build();
    }

    @Override
    public String toString() {
        return "BatchStats[prompt=" + promptTokens + ", cached=" + promptTokensCached + ", generated="
                + generatedTokens + ", device=" + deviceSampledRows + ", steps=" + decodeSteps
                + ", preemptions=" + preemptions + ", spec="
                + specAccepted + "/" + specDrafted + " over " + specRounds + " rounds, lookahead="
                + lookaheadAccepted + "/" + lookaheadEmitted + " over " + lookaheadRounds + " rounds, finished="
                + Arrays.toString(finished) + ", failed=" + failed + ", running=" + running + ", waiting="
                + waiting() + "]";
    }

    /** The engine's accumulator: mutable, touched by the engine's thread only. */
    static final class Builder {
        long promptTokens;
        long promptTokensCached;
        long promptNanos;
        long generatedTokens;
        long deviceSampledRows;
        long decodeNanos;
        long decodeSteps;
        long decodeRows;
        long recomputedTokens;
        long preemptions;
        long contextShifts;
        long grpAttnEdits;
        long chunkRowsReused;
        long specRounds;
        long specDrafted;
        long specAccepted;
        long lookaheadRounds;
        long lookaheadEmitted;
        long lookaheadAccepted;
        final long[] finished = new long[StopReason.values().length];
        long failed;
        int maxSequenceLength;
        int running;
        int queued;
        int preempted;
        int usedCells;
        int totalCells;
        int encoderCacheRows;
        long encoderCacheHits;
        long encoderCacheEvictions;

        BatchStats build() {
            return new BatchStats(this);
        }
    }
}

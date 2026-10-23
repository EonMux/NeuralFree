package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * A target model with one bound drafter and the {@link SpeculativeDecoder} over both, built by
 * {@link LLMModelFactory#openSpeculative} or {@link LLMModelFactory#speculate}.
 *
 * <p>While the session is open the drafter observes every forward call of the target (its hidden-state tap, held
 * through {@link HiddenTapSlot}), so the target's forward passes must run through this session: other uses of the
 * target (plain generation, embeddings) would feed the drafter rows it never consumes. {@link #close()} detaches the
 * drafter, releases the drafter's resources and the tap slot, and closes the target when the session opened it.
 * Not thread-safe.
 */
public final class SpeculativeSession implements AutoCloseable {

    private final EngineModel target;
    private final boolean ownsTarget;
    private final DraftModel drafter;
    private final String drafterKind;
    private final List<AutoCloseable> resources;
    private final HiddenTapSlot.Lease lease;
    private final SpeculativeDecoder decoder;
    private SpeculativeDecoder.Stats lastStats;
    private boolean closed;

    /**
     * @param resources closed after the drafter, last first (draft weights, mapped files)
     * @param decoder   built over the target and the drafter; its stats must reach {@link #record}
     */
    SpeculativeSession(EngineModel target, boolean ownsTarget, DraftModel drafter, String drafterKind,
                       List<AutoCloseable> resources, HiddenTapSlot.Lease lease, SpeculativeDecoderFactory decoder) {
        this.target = Objects.requireNonNull(target, "target");
        this.ownsTarget = ownsTarget;
        this.drafter = Objects.requireNonNull(drafter, "drafter");
        this.drafterKind = Objects.requireNonNull(drafterKind, "drafterKind");
        this.resources = Collections.unmodifiableList(new ArrayList<>(resources));
        this.lease = Objects.requireNonNull(lease, "lease");
        this.decoder = decoder.build(this::record);
    }

    /** Builds the session's decoder once the stats sink exists. */
    interface SpeculativeDecoderFactory {
        SpeculativeDecoder build(Consumer<SpeculativeDecoder.Stats> statsSink);
    }

    private void record(SpeculativeDecoder.Stats stats) {
        this.lastStats = stats;
    }

    /** The target model. Its forward passes belong to this session while it is open. */
    public EngineModel target() {
        return target;
    }

    /** The bound drafter. */
    public DraftModel drafter() {
        return drafter;
    }

    /**
     * What drafts: {@code mtp}, {@code eagle3}, {@code gemma4-assistant}, {@code dflash} or
     * {@code draft model <architecture>}.
     */
    public String drafterKind() {
        return drafterKind;
    }

    /** Most tokens drafted per round, after the drafter's own limit. */
    public int draftMax() {
        return decoder.draftMax();
    }

    /** Shortest draft that is verified. */
    public int draftMin() {
        return decoder.draftMin();
    }

    /** The reasoning parser of the target's chat template. */
    public ReasoningFilter reasoning() {
        return decoder.reasoning();
    }

    /**
     * Speculative generation after {@code prompt}, streaming content to {@code callback}.
     *
     * @see SpeculativeDecoder#generate(int[], GenerationRequest, TokenCallback)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        ensureOpen();
        return decoder.generate(prompt, request, callback);
    }

    /**
     * Speculative generation after {@code prompt}, routing every channel to {@code sink}.
     *
     * @see SpeculativeDecoder#generate(int[], GenerationRequest, ReasoningFilter.Sink, String)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        ensureOpen();
        return decoder.generate(prompt, request, sink, renderedPrompt);
    }

    /** Draft counts of the last generation, or null before the first. */
    public SpeculativeDecoder.Stats lastStats() {
        return lastStats;
    }

    /** Empties the target memory and the drafter's state. */
    public void reset() {
        ensureOpen();
        target.resetKVCache();
        drafter.reset();
    }

    /**
     * Detaches and closes the drafter, closes its resources, frees the tap slot and closes the target when the
     * session opened it. Every step runs; the first failure is thrown with the others suppressed. Idempotent.
     */
    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        Throwable first = LLMModelFactory.closeStep(drafter, null);
        for (int i = resources.size() - 1; i >= 0; i--) {
            first = LLMModelFactory.closeStep(resources.get(i), first);
        }
        first = LLMModelFactory.closeStep(lease, first);
        if (ownsTarget) {
            first = LLMModelFactory.closeStep(target, first);
        }
        LLMModelFactory.rethrow(first, "closing the speculative session of " + target.name() + " failed");
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the speculative session is closed");
        }
    }

    @Override
    public String toString() {
        return "SpeculativeSession{" + target.name() + " + " + drafterKind + ", draftMax=" + decoder.draftMax()
                + ", draftMin=" + decoder.draftMin() + (closed ? ", closed" : "") + '}';
    }
}

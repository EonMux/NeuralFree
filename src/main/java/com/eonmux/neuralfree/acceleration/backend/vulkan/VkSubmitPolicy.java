package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;

import java.util.function.BooleanSupplier;

/**
 * When a {@link VkComputeQueue} submits the command batch it is recording, so the device starts
 * on recorded work while the host records the rest, without paying one submission per dispatch.
 *
 * <ul>
 *   <li><b>Dispatch threshold ramp.</b> A batch is submitted once it holds {@link #threshold()}
 *       dispatches. After the queue drains (every submitted batch has completed, so the device is
 *       idle) the threshold restarts at {@value #FIRST_SUBMIT_DISPATCHES}, and it doubles with each
 *       submission up to the configured cap ({@code neuralfree.vulkan.dispatchesPerSubmit}). The
 *       first work of a step therefore reaches the idle device after a few dispatches, and later
 *       batches grow while the device has a backlog. llama.cpp ramps its submissions the same way:
 *       a small first budget that doubles over the first submits, plus a node cap per submit
 *       (llama.cpp ggml/src/ggml-vulkan/ggml-vulkan.cpp:4272, 14360-14407, 14692-14695).</li>
 *   <li><b>Flush.</b> {@code ComputeQueue.flush()} at a forward-step boundary submits the batch
 *       unless doing so would only add a submission: it is deferred when the batch holds fewer than
 *       {@value #MIN_FLUSH_DISPATCHES} dispatches and nothing else (no copies, fills, readbacks or
 *       cross-queue waits) while the device is still executing earlier batches. A deferred batch is
 *       submitted by a later flush (at the latest the first one after the device has finished its
 *       earlier batches), by the dispatch threshold, by a wait on one of its events, or by
 *       {@code finish()}.</li>
 * </ul>
 */
@NotThreadSafe("Used under the owning queue's monitor.")
final class VkSubmitPolicy {

    /** Dispatch threshold of the first submission after the queue drains. */
    static final int FIRST_SUBMIT_DISPATCHES = 16;
    /** Dispatches a flushed batch needs to be submitted while the device is busy. */
    static final int MIN_FLUSH_DISPATCHES = 32;

    private final int cap;
    private int threshold;

    /**
     * @param cap largest dispatch threshold ({@code neuralfree.vulkan.dispatchesPerSubmit})
     * @throws IllegalArgumentException when the cap is not positive
     */
    VkSubmitPolicy(int cap) {
        if (cap <= 0) {
            throw new IllegalArgumentException("dispatches per submit must be positive, got " + cap);
        }
        this.cap = cap;
        this.threshold = Math.min(FIRST_SUBMIT_DISPATCHES, cap);
    }

    /** Largest dispatch threshold. */
    int cap() {
        return cap;
    }

    /** Dispatches after which the batch being recorded is submitted. */
    int threshold() {
        return threshold;
    }

    /** True when a batch holding {@code dispatches} dispatches is submitted after recording its latest one. */
    boolean submitAfterDispatch(int dispatches) {
        return dispatches >= threshold;
    }

    /**
     * True when a flush submits a non-empty batch.
     *
     * @param dispatches dispatches in the batch
     * @param otherWork  true when the batch also holds copies, fills, readbacks or cross-queue waits
     * @param deviceIdle whether every batch submitted earlier has completed; asked only when the
     *                   batch alone does not decide
     */
    boolean submitOnFlush(int dispatches, boolean otherWork, BooleanSupplier deviceIdle) {
        return otherWork || dispatches >= Math.min(MIN_FLUSH_DISPATCHES, cap) || deviceIdle.getAsBoolean();
    }

    /** Records a submission: the threshold doubles up to the cap. */
    void onSubmit() {
        threshold = (int) Math.min(cap, 2L * threshold);
    }

    /** Records that the queue drained: the next submission comes after {@value #FIRST_SUBMIT_DISPATCHES} dispatches again. */
    void onDrained() {
        threshold = Math.min(FIRST_SUBMIT_DISPATCHES, cap);
    }

    @Override
    public String toString() {
        return "VkSubmitPolicy{threshold=" + threshold + ", cap=" + cap + '}';
    }
}

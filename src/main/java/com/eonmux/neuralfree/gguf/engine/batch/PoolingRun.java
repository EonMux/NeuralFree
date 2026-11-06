package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The pooled rows of one pooling submission while its prompt prefills: every chunk's hidden rows append here in
 * prompt order, the completion applies the model's {@link PoolingOperator} over them and publishes the output on
 * the {@link PoolingSequence}. A prefill that restarts (a preempted sequence's recompute, a requeue) resets the
 * rows, so the completion pools exactly the rows the finished run computed.
 */
final class PoolingRun {

    private final PoolingOutput output;
    private final PoolingSequence handle;
    /** Whether the output reads every prompt row's hidden state (the submission then reuses no cached prefix). */
    private final boolean readsAllRows;
    /** Whether the completion applies the classifier's final probability activation (vLLM's {@code use_activation}). */
    private final boolean activation;
    private final List<float[]> rows = new ArrayList<>();

    PoolingRun(PoolingOutput output, PoolingSequence handle, boolean readsAllRows, boolean activation) {
        this.output = Objects.requireNonNull(output, "output");
        this.handle = Objects.requireNonNull(handle, "handle");
        this.readsAllRows = readsAllRows;
        this.activation = activation;
    }

    PoolingSequence handle() {
        return handle;
    }

    boolean readsAllRows() {
        return readsAllRows;
    }

    /** Appends one chunk's hidden rows (in prompt order). */
    void append(float[][] hiddenRows) {
        for (float[] row : hiddenRows) {
            rows.add(row);
        }
    }

    /** Drops the rows of a prefill that starts over. */
    void reset() {
        rows.clear();
    }

    /**
     * Pools the accumulated rows and completes the request ({@code DONE}), the classifier's final activation
     * applied or skipped as the request asked.
     *
     * @throws RuntimeException when the pooling operator refuses the output or fails: the caller retires the
     *                          sequence with the failure
     */
    void complete(PoolingOperator operator, int[] tokens) {
        handle.complete(operator.apply(output, rows.toArray(new float[0][]), tokens, activation));
    }
}

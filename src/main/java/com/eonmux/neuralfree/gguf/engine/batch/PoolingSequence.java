package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.PoolingOutput;

import java.util.Objects;

/**
 * One pooling request's handle in a {@link BatchEngine}: its {@link BatchSequence} lifecycle and the pooled output
 * published when the request reaches {@code DONE} — the vector, per-token rows, scores or classifier output the
 * pooling operator computed over the prompt's hidden rows (vLLM's {@code pooler_output}). A pooling handle carries
 * no {@link BatchSequence#result()}; read its output here.
 */
public final class PoolingSequence {

    private final BatchSequence handle;
    private final PoolingOutput output;
    private volatile Object pooled;
    private volatile boolean done;

    PoolingSequence(BatchSequence handle, PoolingOutput output) {
        this.handle = Objects.requireNonNull(handle, "handle");
        this.output = Objects.requireNonNull(output, "output");
    }

    /** The lifecycle handle: admission, preemption and the terminal state report there. */
    public BatchSequence handle() {
        return handle;
    }

    /** The pooling task the request runs. */
    public PoolingOutput output() {
        return output;
    }

    /**
     * The pooled output once the request reached {@code DONE} ({@link #done()}), null before that or after a
     * failure or cancellation: {@code float[]} for {@code POOLED}, {@code SCORES} and {@code RERANK_SCORE},
     * {@code float[][]} for {@code TOKEN_EMBEDDINGS} and {@code TOKEN_SCORES}, {@code ClassifierOutput} for
     * {@code CLASSIFIER_OUTPUT}.
     */
    public Object pooled() {
        return pooled;
    }

    /** Whether the pooled output is ready. */
    public boolean done() {
        return done;
    }

    /** Publishes the pooled output and completes the handle ({@code DONE}, the engine thread). */
    void complete(Object pooled) {
        this.pooled = Objects.requireNonNull(pooled, "pooled");
        this.done = true;
        handle.pooledDone();
    }
}

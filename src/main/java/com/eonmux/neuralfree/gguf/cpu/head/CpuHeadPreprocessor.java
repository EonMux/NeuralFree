package com.eonmux.neuralfree.gguf.cpu.head;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

/**
 * A model-level transform of the final hidden rows that runs after the last layer and before the output norm (for
 * example collapsing hyper-connection streams or AltUp predictions back to {@code n_embd}); the twin of
 * {@code GpuHeadPreprocessor}.
 *
 * <p>The forward backend binds every preprocessor whose {@link #appliesTo} accepts the model, in registration order,
 * and calls {@link #apply} once per step on the output rows. Per-step rules as the layer blocks: no per-token
 * allocation.</p>
 */
public interface CpuHeadPreprocessor extends AutoCloseable {

    /** True when this preprocessor must run for the model. */
    boolean appliesTo(ModelSpec spec);

    /** Binds to the model context. Called once, before the first step. */
    void bind(CpuForwardContext ctx);

    /**
     * Rewrites {@code nRows} hidden rows in place. On entry each row has the residual stream width; on return each
     * row has the width the output norm expects ({@code n_embd}), rows packed from index 0.
     */
    void apply(int nRows, float[] hidden);

    /** Releases what {@link #bind} created. */
    @Override
    default void close() {
    }
}

package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The feed-forward block of one layer on the GPU (dense FFN or mixture of experts), enqueued on the context queue.
 *
 * <p>Lifecycle and per-step rules as {@link GpuMixer}. {@code in} holds the normalized FFN input ({@code nTokens}
 * rows of the stream width); the block writes its output, before the residual, to {@code out} in the same layout.
 * {@code in} and {@code out} are distinct.</p>
 */
public interface GpuFfnBlock extends AutoCloseable {

    /** Binds this block to a model context and its layer. Called once, before the first step. */
    void bind(GpuForwardContext ctx, LayerSpec layer);

    /** Runs the block over {@code nTokens} rows. */
    void forward(int nTokens, ScratchSlot in, ScratchSlot out);

    /**
     * Whether the block's decode step enqueues only kernel dispatches, so the decode-step replay can record it.
     * A block that reads step results back to the host (a hosted MoE expert pass reading the routed ids and the
     * expert input rows) returns false; its steps run directly.
     */
    default boolean capturableDecode() {
        return true;
    }

    /**
     * Whether every output row depends on the same input row alone and a step leaves no state behind (no token
     * shift, no carry between layers or steps), so the executor may run the block over a subset of a step's rows:
     * the last layer of a step whose stack output is read at a few rows runs its FFN over those rows only
     * ({@code GpuLayerExecutor#prunesOutputRows}). A block opts in; the default keeps every row.
     */
    default boolean rowLocal() {
        return false;
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

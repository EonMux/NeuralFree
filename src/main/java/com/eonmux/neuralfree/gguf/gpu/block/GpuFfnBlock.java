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

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

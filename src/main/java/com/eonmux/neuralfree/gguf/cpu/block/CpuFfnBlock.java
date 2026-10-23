package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The feed-forward block of one layer on the CPU (dense FFN or mixture of experts); the twin of {@code GpuFfnBlock}.
 *
 * <p>Lifecycle and per-step rules as {@link CpuMixer}. {@code in} holds the normalized FFN input ({@code n} rows of
 * the stream width, row-major from index 0); the block writes its output, before the residual, to {@code out} in the
 * same layout. {@code in} and {@code out} are distinct arrays.</p>
 */
public interface CpuFfnBlock extends AutoCloseable {

    /** Binds this block to a model context and its layer. Called once, before the first step. */
    void bind(CpuForwardContext ctx, LayerSpec layer);

    /** Runs the block over {@code n} rows. */
    void forward(int n, float[] in, float[] out);

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

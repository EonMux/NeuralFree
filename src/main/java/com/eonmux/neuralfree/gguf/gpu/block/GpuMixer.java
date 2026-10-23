package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The token mixer of one layer on the GPU (attention, recurrent or convolutional), enqueued on the context queue.
 *
 * <p>Lifecycle: {@link #bind} once with the layer's context (look up weights, reserve scratch and memory views,
 * prepare dispatches), then {@link #decode} or {@link #prefill} per step, then {@link #close()}. Per step a mixer
 * allocates nothing and never waits on the host: every buffer comes from {@code GpuScratch}, {@code GpuMemory} or
 * {@code GpuWeights}.</p>
 *
 * <p>{@code in} holds the normalized layer input ({@code nTokens} rows of the stream width); the mixer writes its
 * branch output, before the residual, to {@code out} in the same layout. {@code in} and {@code out} are distinct.</p>
 */
public interface GpuMixer extends AutoCloseable {

    /** Binds this mixer to a model context and its layer. Called once, before the first step. */
    void bind(GpuForwardContext ctx, LayerSpec layer);

    /** One token at position {@code pos}. */
    void decode(int pos, ScratchSlot in, ScratchSlot out);

    /** {@code nTokens} consecutive tokens at positions {@code pos0 ..}. */
    void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out);

    /**
     * Drops cached positions {@code [nKeep, nKeep + nDiscard)} and shifts the later ones down (context shift).
     *
     * @throws UnsupportedOperationException when this mixer cannot shift its memory
     */
    default void shiftContext(int nKeep, int nDiscard) {
        throw new UnsupportedOperationException(shiftRefusal());
    }

    /**
     * Why {@link #shiftContext} would refuse on this bound mixer, or null when it can shift. The forward backend asks
     * every cache-owning mixer before the memory moves any row, so a refused shift leaves the cache intact.
     */
    default String shiftRefusal() {
        return "context shift unsupported for " + getClass().getSimpleName();
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

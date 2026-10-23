package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * A per-layer feature around the mixer and FFN (hyper-connections, AltUp, Laurel, per-layer embeddings, residual
 * scales, ...), identified by a {@code LayerSpec.Extras} id; the twin of {@code GpuLayerExtra}.
 *
 * <p>The layer executor calls the extras of a layer in {@code LayerSpec.extras()} order at each hook:
 * {@link #beforeBlock} before the attention norm; {@link #afterMixer} with the mixer branch, before the residual add;
 * {@link #afterFfn} with the FFN branch, before the residual add; {@link #afterBlock} at the end of the layer.
 * {@code x} is the residual stream and {@code branch} the sub-block output, each {@code n} rows of the stream width
 * in row-major {@code [t][width]} layout from index 0. A hook that returns true has folded the branch into {@code x}
 * itself, and the executor skips its own residual add (later extras' residual hooks of that step see the updated
 * {@code x}). Lifecycle and per-step rules as {@link CpuMixer}.</p>
 */
public interface CpuLayerExtra extends AutoCloseable {

    /** Binds this extra to a model context and its layer. Called once, before the first step. */
    void bind(CpuForwardContext ctx, LayerSpec layer);

    /** Before the layer: may rewrite the residual stream {@code x} ({@code n} rows). */
    default void beforeBlock(int pos0, int n, float[] x) {
    }

    /**
     * After the mixer, before its residual add.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
        return false;
    }

    /**
     * After the FFN, before its residual add.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterFfn(int pos0, int n, float[] x, float[] branch) {
        return false;
    }

    /** After the layer: may rewrite the residual stream {@code x}. */
    default void afterBlock(int pos0, int n, float[] x) {
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

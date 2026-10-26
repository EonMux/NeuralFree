package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * A per-layer feature around the mixer and FFN (hyper-connections, AltUp, Laurel, per-layer embeddings, residual
 * scales, ...), identified by a {@code LayerSpec.Extras} id.
 *
 * <p>The layer executor calls the extras of a layer in {@code LayerSpec.extras()} order at each hook:
 * {@link #beforeBlock} before the attention norm; {@link #afterMixer} with the mixer branch, before the residual add;
 * {@link #afterFfn} with the FFN branch, before the residual add; {@link #afterBlock} at the end of the layer. A hook
 * that returns true has folded the branch into the residual stream {@code x} itself, and the executor skips its own
 * residual add (and the later extras' residual hooks of that step see the updated {@code x}). Lifecycle and per-step
 * rules as {@link GpuMixer}.</p>
 */
public interface GpuLayerExtra extends AutoCloseable {

    /** Binds this extra to a model context and its layer. Called once, before the first step. */
    void bind(GpuForwardContext ctx, LayerSpec layer);

    /** Before the layer: may rewrite the residual stream {@code x} ({@code nTokens} rows). */
    default void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
    }

    /**
     * After the mixer, before its residual add.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        return false;
    }

    /**
     * After the FFN, before its residual add.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterFfn(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
        return false;
    }

    /** After the layer: may rewrite the residual stream {@code x}. */
    default void afterBlock(int pos0, int nTokens, ScratchSlot x) {
    }

    /**
     * Why this extra has no batched hooks over {@link StepLayout#contiguousSingle()} == false, or null when it
     * supports them. Default: refused, today's extras run the contiguous step only.
     */
    default String batchRefusal() {
        return getClass().getSimpleName() + " has no multi-sequence step";
    }

    /** Batched form of {@link #beforeBlock}: the contiguous step delegates with {@code pos0 = s.pos0(0)}. */
    default void beforeBlock(StepLayout s, ScratchSlot x) {
        if (s.contiguousSingle()) {
            beforeBlock(s.pos0(0), s.nTokens(), x);
            return;
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /**
     * Batched form of {@link #afterMixer}: the contiguous step delegates with {@code pos0 = s.pos0(0)}.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterMixer(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        if (s.contiguousSingle()) {
            return afterMixer(s.pos0(0), s.nTokens(), x, branch);
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /**
     * Batched form of {@link #afterFfn}: the contiguous step delegates with {@code pos0 = s.pos0(0)}.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     */
    default boolean afterFfn(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        if (s.contiguousSingle()) {
            return afterFfn(s.pos0(0), s.nTokens(), x, branch);
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /** Batched form of {@link #afterBlock}: the contiguous step delegates with {@code pos0 = s.pos0(0)}. */
    default void afterBlock(StepLayout s, ScratchSlot x) {
        if (s.contiguousSingle()) {
            afterBlock(s.pos0(0), s.nTokens(), x);
            return;
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

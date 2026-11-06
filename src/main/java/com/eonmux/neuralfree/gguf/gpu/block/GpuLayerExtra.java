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
     * supports them. Default: refused, the extra runs the contiguous step only.
     */
    default String batchRefusal() {
        return getClass().getSimpleName() + " has no multi-sequence step";
    }

    /**
     * Batched form of {@link #beforeBlock}: the contiguous step delegates with {@code pos0 = s.pos0(0)}; a
     * batch-capable extra ({@link #batchRefusal()} null) runs the same hook over the whole step's rows, which is
     * exact for the row-wise hooks, over paged and multi-segment layouts alike.
     *
     * @throws UnsupportedOperationException when the step is not contiguous-single and {@link #batchRefusal()}
     *                                       answers a reason
     */
    default void beforeBlock(StepLayout s, ScratchSlot x) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        beforeBlock(s.pos0(0), s.nTokens(), x);
    }

    /**
     * Batched form of {@link #afterMixer}, delegating as {@link #beforeBlock(StepLayout, ScratchSlot)} does.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, ScratchSlot)}
     */
    default boolean afterMixer(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        return afterMixer(s.pos0(0), s.nTokens(), x, branch);
    }

    /**
     * Batched form of {@link #afterFfn}, delegating as {@link #beforeBlock(StepLayout, ScratchSlot)} does.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, ScratchSlot)}
     */
    default boolean afterFfn(StepLayout s, ScratchSlot x, ScratchSlot branch) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        return afterFfn(s.pos0(0), s.nTokens(), x, branch);
    }

    /**
     * Batched form of {@link #afterBlock}, delegating as {@link #beforeBlock(StepLayout, ScratchSlot)} does.
     *
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, ScratchSlot)}
     */
    default void afterBlock(StepLayout s, ScratchSlot x) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        afterBlock(s.pos0(0), s.nTokens(), x);
    }

    /** The refusal {@link #batchRefusal()} answers, thrown for a step that is not contiguous-single. */
    default void refuseBatch() {
        String why = batchRefusal();
        if (why != null) {
            throw new UnsupportedOperationException(why);
        }
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
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

    /**
     * Why this extra has no batched hooks over a step that is not {@link StepLayout#contiguousSingle()}, or null
     * when its hooks are exact over any layout (a row-wise extra's hooks are: every row transforms on its own, so
     * the segments' sequences and positions cannot change the result). Default: refused; an extra whose hooks are
     * row-wise overrides this to null, an extra with layout-dependent math overrides the batched hooks themselves.
     */
    default String batchRefusal() {
        return getClass().getSimpleName() + " has no multi-sequence step";
    }

    /**
     * Batched form of {@link #beforeBlock}: the contiguous step delegates with {@code pos0 = s.pos0(0)}; a
     * batch-capable extra ({@link #batchRefusal()} null) runs the same hook over the whole step's rows, which is
     * exact for the row-wise hooks, over paged and multi-segment layouts alike.
     *
     * @throws UnsupportedOperationException when the step is not contiguous-un-paged and {@link #batchRefusal()}
     *                                       answers a reason
     */
    default void beforeBlock(StepLayout s, float[] x) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        beforeBlock(s.pos0(0), s.nTokens(), x);
    }

    /**
     * Batched form of {@link #afterMixer}: the contiguous step delegates with {@code pos0 = s.pos0(0)}; a
     * batch-capable extra runs the same hook over the whole step's rows.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, float[])}
     */
    default boolean afterMixer(StepLayout s, float[] x, float[] branch) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        return afterMixer(s.pos0(0), s.nTokens(), x, branch);
    }

    /**
     * Batched form of {@link #afterFfn}: the contiguous step delegates with {@code pos0 = s.pos0(0)}; a
     * batch-capable extra runs the same hook over the whole step's rows.
     *
     * @return true when this extra has added (or mixed) {@code branch} into {@code x} itself
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, float[])}
     */
    default boolean afterFfn(StepLayout s, float[] x, float[] branch) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        return afterFfn(s.pos0(0), s.nTokens(), x, branch);
    }

    /**
     * Batched form of {@link #afterBlock}: the contiguous step delegates with {@code pos0 = s.pos0(0)}; a
     * batch-capable extra runs the same hook over the whole step's rows.
     *
     * @throws UnsupportedOperationException as {@link #beforeBlock(StepLayout, float[])}
     */
    default void afterBlock(StepLayout s, float[] x) {
        if (!s.contiguousSingle()) {
            refuseBatch();
        }
        afterBlock(s.pos0(0), s.nTokens(), x);
    }

    /** The refusal {@link #batchRefusal()} answers, thrown for a non-contiguous step. */
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

package com.eonmux.neuralfree.gguf.cpu.block;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The token mixer of one layer on the CPU (attention, recurrent or convolutional); the twin of {@code GpuMixer}.
 *
 * <p>Lifecycle: {@link #bind} once with the layer's context (look up weights, reserve scratch and memory views),
 * then {@link #decode} or {@link #prefill} per step, then {@link #close()}. Per step a mixer allocates no per-token
 * arrays: every working array comes from {@code CpuScratch}, {@code CpuMemory} or {@code CpuWeights}.</p>
 *
 * <p>{@code in} holds the normalized layer input, {@code n} rows of the stream width in row-major {@code [t][width]}
 * layout from index 0; the mixer writes its branch output, before the residual, to {@code out} in the same layout.
 * {@code in} and {@code out} are distinct arrays.</p>
 */
public interface CpuMixer extends AutoCloseable {

    /** Binds this mixer to a model context and its layer. Called once, before the first step. */
    void bind(CpuForwardContext ctx, LayerSpec layer);

    /** One token at position {@code pos}. */
    void decode(int pos, float[] in, float[] out);

    /**
     * {@code n} consecutive tokens at positions {@code pos0 ..}; also the no-memory (encoder) path when the layer
     * uses no memory.
     */
    void prefill(int pos0, int n, float[] in, float[] out);

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

    /**
     * Why this mixer has no batched step over {@link StepLayout#contiguousSingle()} == false, or null when it can
     * run one. Default: refused, today's mixers run the contiguous step only.
     */
    default String batchRefusal() {
        return getClass().getSimpleName() + " has no multi-sequence step";
    }

    /**
     * One step over the rows of {@code s}: today's contiguous step ({@link #decode} / {@link #prefill} at
     * {@code pos0(0)}) when the step is contiguous and the memory un-paged, the backend's batched step otherwise.
     *
     * @throws UnsupportedOperationException when the step is not contiguous-un-paged and this mixer has no
     *                                       batched step ({@link #batchRefusal()})
     */
    default void batch(StepLayout s, float[] in, float[] out) {
        if (s.contiguousSingle()) {
            if (s.nTokens() == 1) {
                decode(s.pos0(0), in, out);
            } else {
                prefill(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        throw new UnsupportedOperationException(batchRefusal());
    }

    /** Releases what {@link #bind} created; the context's shared resources stay. */
    @Override
    default void close() {
    }
}

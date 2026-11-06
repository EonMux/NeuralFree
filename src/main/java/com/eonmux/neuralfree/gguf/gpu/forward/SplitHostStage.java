package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.memory.ModelMemory;

/**
 * The host part of a layer split whose first trunk layers stay on the host (llama.cpp {@code -ngl} below the layer
 * count over {@code --split-mode layer}): the embedding and layers {@code [0, layers())} run on the CPU over a heap
 * memory of their own, and the residual stream they leave is handed to the first device. Implemented by the hybrid
 * CPU+GPU machinery ({@code com.eonmux.neuralfree.gguf.hybrid.HostLayerStage}). Not thread-safe.
 */
public interface SplitHostStage extends AutoCloseable {

    /** The hidden-state captures of the host layers, in model layer indices. */
    interface Taps {
        /** True when the tap wants the stream entering layer {@code layer}. */
        boolean wants(int layer);

        /** The {@code n} rows of {@code width} floats of the stream entering {@code layer}. */
        void capture(int layer, float[] x, int n, int width);
    }

    /** The trunk layers that run on the host: {@code [0, layers())}. */
    int layers();

    /** Floats per token of the residual stream the host hands to the first device. */
    int outputWidth();

    /** The most tokens one step takes. */
    int ubatch();

    /** The heap memory of the host layers. */
    ModelMemory memory();

    /** Marks the rows of a step of {@code n} tokens at position {@code p} in the host memory. */
    void beginStep(int p, int n);

    /**
     * Embeds tokens {@code [off, off + n)} at position {@code p} and runs the host layers over them.
     *
     * @param types the token types of the whole call, or null
     * @return the residual stream of the step: {@code n} rows of {@link #outputWidth()} floats from offset 0
     */
    float[] forward(int[] tokens, int[] types, int off, int n, int p, Taps taps);

    /** Checks the inputs of a call before any step runs. */
    void checkInputs(int[] tokens, int[] types, int pos0);

    /** Why a context shift cannot run on the host layers, or null when it can. */
    String shiftRefusal();

    /** Moves the stored rows of the host memory and re-rotates the keys of the host layers. */
    void shiftContext(int nKeep, int nDiscard);

    /** Closes the host blocks; the memory and the context belong to whoever built them. */
    @Override
    void close();
}

package com.eonmux.neuralfree.gguf.engine;

/**
 * Observer of the residual stream during a forward pass (eagle3 target features, diagnostics).
 *
 * <p>Tap point {@code layer} is the hidden state entering trunk layer {@code layer}: tap 0 is the embedding-stage
 * output and tap {@code nLayer} is the output of the last trunk layer before the final output norm. This matches
 * llama.cpp's layer-input extraction ({@code t_layer_inp[il]}, llama.cpp src/llama-graph.cpp set_outputs) and its use
 * of {@code il == n_layer} for the pre-norm final hidden state (llama.cpp common/speculative.cpp eagle3 setup).
 *
 * <p>Forward backends ask {@link #wants(int)} before copying anything, so an unused tap costs nothing. The rows
 * passed to {@link #onHidden} cover every token of the current forward call in order, row-major
 * {@code [nRows][width]}, where {@code width} is the residual stream width at that point (n_embd, or
 * {@code hc * n_embd} for hyper-connection models). The array may be reused after the call returns, so an
 * implementation that keeps the data must copy it.
 */
public interface HiddenTap {

    /** A tap that observes nothing. */
    HiddenTap NONE = new HiddenTap() {
        @Override
        public void onHidden(int layer, float[] rows, int nRows, int width) {
        }

        @Override
        public boolean wants(int layer) {
            return false;
        }

        @Override
        public String toString() {
            return "HiddenTap.NONE";
        }
    };

    /**
     * Receives the hidden rows at tap point {@code layer}; called only when {@link #wants(int)} is true for it.
     *
     * @param layer tap point in {@code [0, nLayer]}
     * @param rows  row-major {@code [nRows][width]} hidden states (at least {@code nRows * width} floats)
     * @param nRows tokens in the current forward call
     * @param width floats per row
     */
    void onHidden(int layer, float[] rows, int nRows, int width);

    /** Whether this tap wants the rows at tap point {@code layer}. */
    boolean wants(int layer);
}

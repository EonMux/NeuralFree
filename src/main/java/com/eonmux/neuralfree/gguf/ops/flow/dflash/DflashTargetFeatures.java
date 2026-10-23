package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.engine.HiddenTap;

import java.util.Arrays;
import java.util.Objects;

/**
 * Hidden tap on the target that gathers the drafter's input features: the inputs of the {@code target_layers},
 * concatenated per token as {@code [layer_0 | layer_1 | ...]} of {@code n_embd} values each (common/speculative.cpp
 * process(): {@code llama_get_embeddings_layer_inp} per extract layer). A hyper-connection target delivers
 * {@code hc * n_embd} values per row; they are collapsed to the stream mean, the layer input llama.cpp extracts for
 * those models ({@code build_hc_mean}, deepseek4.cpp:269-277). Once every target layer of one forward call has
 * arrived, the rows go to the {@link Sink}. Not thread-safe; the forward thread drives it.
 */
final class DflashTargetFeatures implements HiddenTap {

    /** Receives the features of one forward call. */
    interface Sink {
        /**
         * @param features row-major {@code [nRows][nTargets * nEmbd]}; reused after the call returns
         */
        void onFeatures(float[] features, int nRows);
    }

    private final int[] targetLayers;
    private final int[] distinct;
    private final int nEmbd;
    private final int rowWidth;
    private final Sink sink;
    private final boolean[] received;
    private float[] features = new float[0];
    private int pendingRows = -1;
    private int nReceived;

    /**
     * @param targetLayers tap points in feature order
     * @param nEmbd        target hidden size
     * @throws IllegalArgumentException for an empty layer list, a negative layer or a non-positive width
     */
    DflashTargetFeatures(int[] targetLayers, int nEmbd, Sink sink) {
        if (targetLayers.length == 0 || nEmbd <= 0) {
            throw new IllegalArgumentException("dflash target features need layers and a positive width");
        }
        this.targetLayers = targetLayers.clone();
        this.distinct = Arrays.stream(targetLayers).distinct().sorted().toArray();
        if (distinct[0] < 0) {
            throw new IllegalArgumentException("dflash target layer " + distinct[0] + " is negative");
        }
        this.nEmbd = nEmbd;
        this.rowWidth = targetLayers.length * nEmbd;
        this.sink = Objects.requireNonNull(sink, "sink");
        this.received = new boolean[distinct.length];
    }

    /** Values per feature row: {@code nTargets * nEmbd}. */
    int rowWidth() {
        return rowWidth;
    }

    @Override
    public boolean wants(int layer) {
        return Arrays.binarySearch(distinct, layer) >= 0;
    }

    @Override
    public void onHidden(int layer, float[] rows, int nRows, int width) {
        final int slot = Arrays.binarySearch(distinct, layer);
        if (slot < 0 || nRows <= 0) {
            return;
        }
        if (width <= 0 || width % nEmbd != 0) {
            throw new IllegalStateException("dflash drafter: target layer " + layer + " delivered rows of " + width
                    + " values, not a multiple of the target width " + nEmbd);
        }
        if (nRows != pendingRows || received[slot]) {
            discard();
            pendingRows = nRows;
            if ((long) nRows * rowWidth > Integer.MAX_VALUE - 8) {
                throw new IllegalStateException("dflash drafter: " + nRows + " target rows exceed one feature buffer");
            }
            if (features.length < nRows * rowWidth) {
                features = new float[nRows * rowWidth];
            }
        }
        final int streams = width / nEmbd;
        for (int k = 0; k < targetLayers.length; k++) {
            if (targetLayers[k] == layer) {
                copy(rows, nRows, streams, k * nEmbd);
            }
        }
        received[slot] = true;
        if (++nReceived == distinct.length) {
            final int n = pendingRows;
            discard();
            sink.onFeatures(features, n);
        }
    }

    /** Drops a partially gathered call. */
    void discard() {
        Arrays.fill(received, false);
        nReceived = 0;
        pendingRows = -1;
    }

    private void copy(float[] rows, int nRows, int streams, int dstCol) {
        final int width = streams * nEmbd;
        final float inv = 1.0f / streams;
        for (int t = 0; t < nRows; t++) {
            final int src = t * width;
            final int dst = t * rowWidth + dstCol;
            if (streams == 1) {
                System.arraycopy(rows, src, features, dst, nEmbd);
                continue;
            }
            for (int i = 0; i < nEmbd; i++) {
                float acc = rows[src + i];
                for (int s = 1; s < streams; s++) {
                    acc += rows[src + s * nEmbd + i];
                }
                features[dst + i] = acc * inv;
            }
        }
    }

    @Override
    public String toString() {
        return "DflashTargetFeatures{layers=" + Arrays.toString(targetLayers) + ", nEmbd=" + nEmbd + '}';
    }
}

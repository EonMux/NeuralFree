package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;

import java.util.Locale;

/**
 * How the vision-tower rows of one image become one embedding vector in a CLIP dual-encoder pooling model: vLLM's
 * vision feature select strategy (vllm/model_executor/models/clip.py {@code _POOLING_TYPE_TO_STRATEGY}), picked
 * by the model's sequence pooling — CLS and LAST keep the class-token row ({@code "class"}), MEAN averages every
 * row ({@code "full"} followed by the mean pooler). Stateless; every call allocates its result.
 */
public enum ImageFeatureSelect {

    /** The vector is the class-token row, the first row of the tower (vLLM's {@code "class"}). */
    CLASS,
    /** The vector is the mean over every row of the tower (vLLM's {@code "full"} under MEAN pooling). */
    FULL;

    /**
     * The select of a model's pooling kind: CLS and LAST keep the class row, MEAN averages, and the kinds without
     * one pooled image vector refuse by name — NONE pools per token, RANK is a classifier.
     *
     * @throws IllegalArgumentException for a null kind and the NONE and RANK pooling kinds
     */
    public static ImageFeatureSelect of(PoolingKind pooling) {
        if (pooling == null) {
            throw new IllegalArgumentException("the pooling kind is null");
        }
        switch (pooling) {
            case CLS:
            case LAST:
                return CLASS;
            case MEAN:
                return FULL;
            default:
                throw new IllegalArgumentException("no image feature select is defined for pooling type "
                        + pooling.name().toLowerCase(Locale.ROOT) + "; image embeddings pool by cls, last or mean");
        }
    }

    /**
     * One vector over {@code nTokens} token-major rows of {@code width}: the first row for CLASS, the mean of the
     * rows for FULL.
     *
     * @return a fresh array of {@code width} values
     * @throws IllegalArgumentException when there are no rows or the buffer holds fewer than {@code nTokens}
     */
    public float[] select(float[] rows, int nTokens, int width) {
        if (nTokens <= 0 || width <= 0 || rows.length < (long) nTokens * width) {
            throw new IllegalArgumentException("the select needs " + nTokens + " rows of " + width + ", the buffer"
                    + " holds " + rows.length + " floats");
        }
        float[] out = new float[width];
        if (this == CLASS) {
            System.arraycopy(rows, 0, out, 0, width);
            return out;
        }
        double[] sum = new double[width];
        for (int t = 0; t < nTokens; t++) {
            int base = t * width;
            for (int i = 0; i < width; i++) {
                sum[i] += rows[base + i];
            }
        }
        for (int i = 0; i < width; i++) {
            out[i] = (float) (sum[i] / nTokens);
        }
        return out;
    }

    @Override
    public String toString() {
        return this == CLASS ? "class" : "full";
    }
}

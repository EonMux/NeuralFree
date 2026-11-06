package com.eonmux.neuralfree.gguf.engine;

/**
 * llama.cpp's {@code common_embd_normalize}: the normalization of a pooled embedding by mode — -1 none, 0 max
 * absolute scaled to the int16 range (32760), 1 taxicab, 2 euclidean (the servers' default), and any larger
 * {@code p} the p-norm. A zero vector stays zero. Stateless; inputs are never modified.
 */
public final class EmbeddingNorm {

    /** No normalization. */
    public static final int NONE = -1;

    /** Max absolute value scaled to 32760, llama.cpp's int16 range. */
    public static final int MAX_ABS_INT16 = 0;

    /** The euclidean (L2) norm, llama.cpp's and vLLM's default. */
    public static final int EUCLIDEAN = 2;

    private static final double INT16_RANGE = 32760.0;

    private EmbeddingNorm() {
    }

    /**
     * {@code v} normalized by {@code mode}.
     *
     * @throws IllegalArgumentException when {@code mode} is below -1
     */
    public static float[] normalize(float[] v, int mode) {
        if (mode < NONE) {
            throw new IllegalArgumentException("embedding normalization must be -1 (none), 0 (max absolute),"
                    + " 1 (taxicab), 2 (euclidean) or a larger p-norm, got " + mode);
        }
        double sum = divisor(v, mode);
        float scale = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * scale;
        }
        return out;
    }

    private static double divisor(float[] v, int mode) {
        double sum = 0.0;
        switch (mode) {
            case NONE:
                return 1.0;
            case MAX_ABS_INT16:
                for (float x : v) {
                    sum = Math.max(sum, Math.abs(x));
                }
                return sum / INT16_RANGE;
            case EUCLIDEAN:
                for (float x : v) {
                    sum += (double) x * x;
                }
                return Math.sqrt(sum);
            default:
                for (float x : v) {
                    sum += Math.pow(Math.abs(x), mode);
                }
                return Math.pow(sum, 1.0 / mode);
        }
    }
}

package com.eonmux.neuralfree.gguf.engine;

/**
 * llama.cpp's {@code common_embd_similarity_cos} (common/common.cpp:1983): the cosine similarity of two embeddings,
 * with float products accumulated in double as the C function does. Two zero vectors are similar (1), one zero
 * vector is not (0). Stateless; inputs are never modified.
 */
public final class EmbeddingSimilarity {

    private EmbeddingSimilarity() {
    }

    /**
     * The cosine similarity of two equal-length embeddings: the dot product over both Euclidean norms.
     *
     * @throws IllegalArgumentException for embeddings of different lengths or no values
     */
    public static float cosine(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("cosine of " + a.length + " against " + b.length
                    + " values needs equal lengths");
        }
        if (a.length == 0) {
            throw new IllegalArgumentException("cosine of empty embeddings");
        }
        double sum = 0.0;
        double sumA = 0.0;
        double sumB = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
            sumA += a[i] * a[i];
            sumB += b[i] * b[i];
        }
        if (sumA == 0.0 || sumB == 0.0) {
            return sumA == 0.0 && sumB == 0.0 ? 1.0f : 0.0f;
        }
        return (float) (sum / (Math.sqrt(sumA) * Math.sqrt(sumB)));
    }
}

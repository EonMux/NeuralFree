package com.eonmux.neuralfree.gguf.speech;

import java.util.Random;

/**
 * The MaskGIT sampling of llama.cpp's lfm3a audio helper (tools/mtmd/mtmd-helper-gen.cpp:1528-1626): a fresh
 * codebook order per frame, the repetition penalty over the frame history and the temperature draw over one
 * codebook's logits.
 *
 * <p>The draws come from a {@link java.util.Random} where the reference uses {@code std::mt19937}, so the sample
 * sequence of a given seed differs from llama.cpp's while staying deterministic within NeuralFree; every rule
 * around the draws (the shuffle's Fisher-Yates shape, the penalty formula and window, the k-th-best top-k cut with
 * ties, the cumulative walk) follows the reference exactly.
 */
final class Lfm3aMaskgitSampler {

    private Lfm3aMaskgitSampler() {
    }

    /**
     * A fresh random order of the codebooks: the reference shuffles {@code iota(n)} with {@code std::shuffle}
     * once per frame; this is the same Fisher-Yates walk over the caller's {@code rng}.
     */
    static int[] shuffledOrder(int nCodebooks, Random rng) {
        if (nCodebooks <= 0) {
            throw new IllegalArgumentException("the codebook order needs at least one codebook, got " + nCodebooks);
        }
        int[] order = new int[nCodebooks];
        for (int i = 0; i < nCodebooks; i++) {
            order[i] = i;
        }
        for (int i = nCodebooks - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return order;
    }

    /**
     * The CTRL repetition penalty over one codebook's logits, in place: every code the codebook took in the last
     * {@code repWindow} frames of the frame-major {@code history} divides a positive logit by {@code repPenalty}
     * and multiplies a non-positive one, first occurrence only.
     *
     * <p>A penalty of at most 1 or a window below 1 leaves the logits untouched, as in the reference.
     */
    static void applyRepPenalty(float[] logits, int codebook, int[] history, int nCodebooks, int repWindow,
                                float repPenalty) {
        if (repPenalty <= 1.0f || repWindow <= 0) {
            return;
        }
        int nHist = history.length / nCodebooks;
        int nWin = Math.min(nHist, repWindow);
        boolean[] seen = new boolean[logits.length];
        for (int f = nHist - nWin; f < nHist; f++) {
            int c = history[f * nCodebooks + codebook];
            if (c >= 0 && c < logits.length && !seen[c]) {
                seen[c] = true;
                logits[c] = logits[c] > 0.0f ? logits[c] / repPenalty : logits[c] * repPenalty;
            }
        }
    }

    /**
     * One codebook's draw: temperature at most 0 takes the first maximum; otherwise the codes below the k-th best
     * score (ties kept, top-k at most 0 or at least the vocabulary disabled) get probability 0 and the rest
     * {@code exp((l - max) / temp)}, and one uniform of {@code rng} walks the cumulative distribution, falling
     * back to the last code when the sum rounds away (mtmd-helper-gen.cpp:1600-1626).
     */
    static int sample(float[] logits, float temp, int topK, Random rng) {
        int n = logits.length;
        if (temp <= 0.0f) {
            int best = 0;
            for (int v = 1; v < n; v++) {
                if (logits[v] > logits[best]) {
                    best = v;
                }
            }
            return best;
        }
        float lmin = Float.NEGATIVE_INFINITY;
        if (topK > 0 && topK < n) {
            float[] sorted = logits.clone();
            descendingSort(sorted);
            lmin = sorted[topK - 1];
        }
        float lmax = logits[0];
        for (int v = 1; v < n; v++) {
            if (logits[v] > lmax) {
                lmax = logits[v];
            }
        }
        double[] prob = new double[n];
        double sum = 0.0;
        for (int v = 0; v < n; v++) {
            prob[v] = logits[v] < lmin ? 0.0 : Math.exp((double) (logits[v] - lmax) / temp);
            sum += prob[v];
        }
        double r = rng.nextDouble() * sum;
        for (int v = 0; v < n; v++) {
            r -= prob[v];
            if (r <= 0.0) {
                return v;
            }
        }
        return n - 1;
    }

    /** In-place descending sort; the k-th best is all {@code nth_element} provides. */
    private static void descendingSort(float[] v) {
        java.util.Arrays.sort(v);
        for (int i = 0, j = v.length - 1; i < j; i++, j--) {
            float tmp = v[i];
            v[i] = v[j];
            v[j] = tmp;
        }
    }
}

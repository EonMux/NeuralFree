package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@code LLM_TopK_Rows}: per-row top-k column selection of DSA indexer scores.
 *
 * <p>Row {@code r} of {@code out} ({@code k} slots) receives the columns of the {@code kk = min(k, nValid)} largest
 * eligible scores, where a score is eligible when it is greater than {@code -INF} (NaN never is), then {@code -1} in
 * the remaining slots. Ties at the k-th value go to the lowest columns. Output order is the kernel's: the columns
 * above the k-th value in increasing column order, then the selected columns equal to it in increasing column order.
 * Equal floats are equal keys, so {@code -0} ties with {@code +0}. The selected set is the one {@code ggml_top_k}
 * returns (llama.cpp src/models/deepseek32.cpp:314-315) once masked cells are excluded.
 *
 * <p>The k-th value is found by the kernel's radix select: three digit passes of 11, 11 and 10 bits over the
 * order-preserving unsigned key of each score.
 */
public final class CpuTopK {

    private static final int BINS = 2048;
    private static final int[] SHIFTS = {21, 10, 0};
    private static final int[] DIGIT_MASKS = {0x7FF, 0x7FF, 0x3FF};

    private CpuTopK() {
    }

    /**
     * Selects the top {@code k} columns of every row of {@code scores[scoresOff + r*nCols ..]} into
     * {@code out[outOff + r*k ..]}.
     *
     * @throws IllegalArgumentException for negative sizes or {@code k > TOPK_MAX}
     */
    public static void topKRows(float[] scores, int scoresOff, int nRows, int nCols, int k, int[] out, int outOff) {
        check(scores, nRows, nCols, k, out);
        CpuParallel.parallelFor(nRows, 1, (from, to) -> {
            int[] hist = new int[BINS];
            for (int r = from; r < to; r++) {
                topKRow(scores, scoresOff + r * nCols, nCols, k, out, outOff + r * k, hist);
            }
        });
    }

    /**
     * Selects the top {@code k} columns of one row.
     *
     * @return the number of selected columns {@code kk}; slots {@code [kk, k)} hold -1
     * @throws IllegalArgumentException for negative sizes or {@code k > TOPK_MAX}
     */
    public static int topKRow(float[] scores, int off, int nCols, int k, int[] out, int outOff) {
        check(scores, 1, nCols, k, out);
        return topKRow(scores, off, nCols, k, out, outOff, new int[BINS]);
    }

    private static void check(float[] scores, int nRows, int nCols, int k, int[] out) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(out, "out");
        if (nRows < 0 || nCols < 0 || k < 0 || k > SparseKernels.TOPK_MAX) {
            throw new IllegalArgumentException("top-k needs nRows >= 0, nCols >= 0 and 0 <= k <= "
                    + SparseKernels.TOPK_MAX + ", got " + nRows + "/" + nCols + "/" + k);
        }
    }

    /** Order-preserving unsigned key (returned in an int): {@code a > b} iff {@code key(a) >u key(b)}. */
    static int key(float x) {
        int u = x == 0.0f ? 0 : Float.floatToRawIntBits(x);
        return u < 0 ? ~u : (u | 0x80000000);
    }

    private static boolean eligible(float x) {
        return x > Float.NEGATIVE_INFINITY;
    }

    private static int topKRow(float[] scores, int off, int nCols, int k, int[] out, int outOff, int[] hist) {
        int prefix = 0;
        int pmask = 0;
        int kk = 0;
        int remaining = 0;
        for (int pass = 0; pass < SHIFTS.length; pass++) {
            int shift = SHIFTS[pass];
            int dmask = DIGIT_MASKS[pass];
            Arrays.fill(hist, 0);
            int candidates = 0;
            for (int c = 0; c < nCols; c++) {
                float x = scores[off + c];
                if (eligible(x)) {
                    int key = key(x);
                    if ((key & pmask) == prefix) {
                        hist[(key >>> shift) & dmask]++;
                        candidates++;
                    }
                }
            }
            if (pass == 0) {
                kk = Math.min(k, candidates);
                remaining = kk;
            }
            if (kk == 0) {
                break;
            }
            int above = 0;
            int digit = BINS - 1;
            while (above + hist[digit] < remaining) {
                above += hist[digit];
                digit--;
            }
            prefix |= digit << shift;
            pmask |= dmask << shift;
            remaining -= above;
        }
        int nAbove = kk - remaining;
        int aboveWritten = 0;
        int equalTaken = 0;
        if (kk > 0) {
            for (int c = 0; c < nCols; c++) {
                float x = scores[off + c];
                if (!eligible(x)) {
                    continue;
                }
                int key = key(x);
                if (Integer.compareUnsigned(key, prefix) > 0) {
                    out[outOff + aboveWritten++] = c;
                } else if (key == prefix && equalTaken < remaining) {
                    out[outOff + nAbove + equalTaken++] = c;
                }
            }
        }
        for (int i = kk; i < k; i++) {
            out[outOff + i] = -1;
        }
        return kk;
    }
}

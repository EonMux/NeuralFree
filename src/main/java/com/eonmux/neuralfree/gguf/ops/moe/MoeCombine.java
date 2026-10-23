package com.eonmux.neuralfree.gguf.ops.moe;

import java.util.Objects;

/**
 * CPU twins of the MoE helper kernels in {@code kernels/llm/moe.cl}, with the same index formulas and
 * accumulation order (pair index {@code p = t * k + j}):
 * {@code LLM_MoE_HashIds} ({@link #hashIds}), {@code LLM_MoE_WeightedSum} ({@link #weightedSum}),
 * {@code LLM_AddId} ({@link #addId}), {@code LLM_MoE_Bucketize} ({@link #bucketize}) and
 * {@code LLM_MoE_ScaleInput} ({@link #scaleInput}).
 */
public final class MoeCombine {

    private MoeCombine() {
    }

    /**
     * deepseek4 hash layers (models/deepseek4.cpp:1297-1299): {@code ids[t * k + j] = tid2eid[tokens[t] * k + j]},
     * {@code tid2eid} being the {@code [k, n_vocab]} I32 tensor {@code ffn_gate_tid2eid}.
     *
     * @throws IllegalArgumentException when a token is outside the table
     */
    public static int[] hashIds(int[] tokens, int nTokens, int k, int[] tid2eid) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(tid2eid, "tid2eid");
        checkCount("tokens", tokens.length, nTokens);
        positive("k", k);
        int nVocab = tid2eid.length / k;
        int[] ids = new int[nTokens * k];
        for (int t = 0; t < nTokens; t++) {
            int tok = tokens[t];
            if (tok < 0 || tok >= nVocab) {
                throw new IllegalArgumentException("token " + tok + " at " + t + " is outside the hash table of "
                        + nVocab + " tokens");
            }
            System.arraycopy(tid2eid, tok * k, ids, t * k, k);
        }
        return ids;
    }

    /**
     * Validates a hash table once at load: {@code [k, nVocab]} entries, each in {@code [0, nExpert)}, so the
     * kernels never index outside the expert tensors.
     *
     * @throws IllegalArgumentException naming the first bad entry
     */
    public static void validateHashTable(int[] tid2eid, int k, int nVocab, int nExpert) {
        Objects.requireNonNull(tid2eid, "tid2eid");
        positive("k", k);
        positive("n_vocab", nVocab);
        positive("n_expert", nExpert);
        if (tid2eid.length != (long) k * nVocab) {
            throw new IllegalArgumentException("ffn_gate_tid2eid must hold " + k + " x " + nVocab + " ids, got "
                    + tid2eid.length);
        }
        for (int i = 0; i < tid2eid.length; i++) {
            if (tid2eid[i] < 0 || tid2eid[i] >= nExpert) {
                throw new IllegalArgumentException("ffn_gate_tid2eid[" + (i / k) + "][" + (i % k) + "] = " + tid2eid[i]
                        + " is not an expert id in [0, " + nExpert + ")");
            }
        }
    }

    /**
     * {@code dst[t][c] = sum_j w[t][j] * expertOut[t * k + j][c]}, j ascending (llama-graph.cpp:2541-2567);
     * {@link MoeKernels#WSUM_UNWEIGHTED} takes {@code w = 1}, {@link MoeKernels#WSUM_ACCUMULATE} adds the sum to
     * {@code dst}. Writes {@code dst[0 .. nTokens * nEmbd)} in place.
     */
    public static void weightedSum(int nTokens, int nEmbd, int k, int flags, float[] expertOut, float[] weights,
                                   float[] dst) {
        Objects.requireNonNull(expertOut, "expertOut");
        Objects.requireNonNull(dst, "dst");
        positive("n_embd", nEmbd);
        positive("k", k);
        boolean unweighted = (flags & MoeKernels.WSUM_UNWEIGHTED) != 0;
        boolean accumulate = (flags & MoeKernels.WSUM_ACCUMULATE) != 0;
        checkCount("expert_out", expertOut.length, (long) nTokens * k * nEmbd);
        checkCount("dst", dst.length, (long) nTokens * nEmbd);
        if (!unweighted) {
            checkCount("weights", Objects.requireNonNull(weights, "weights").length, (long) nTokens * k);
        }
        for (int t = 0; t < nTokens; t++) {
            for (int c = 0; c < nEmbd; c++) {
                float acc = 0.0f;
                for (int j = 0; j < k; j++) {
                    int p = t * k + j;
                    float v = expertOut[p * nEmbd + c];
                    acc += unweighted ? v : v * weights[p];
                }
                int o = t * nEmbd + c;
                dst[o] = accumulate ? acc + dst[o] : acc;
            }
        }
    }

    /** {@code data[s][c] += bias[slots[s]][c]} (ggml_add_id), in place. */
    public static void addId(int nSel, int dim, int[] slots, float[] bias, float[] data) {
        Objects.requireNonNull(slots, "slots");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(data, "data");
        positive("dim", dim);
        checkCount("slots", slots.length, nSel);
        checkCount("data", data.length, (long) nSel * dim);
        for (int s = 0; s < nSel; s++) {
            int e = slots[s];
            if (e < 0 || (long) (e + 1) * dim > bias.length) {
                throw new IllegalArgumentException("slot " + s + " selects expert " + e + " outside the bias tensor");
            }
            for (int c = 0; c < dim; c++) {
                data[s * dim + c] += bias[e * dim + c];
            }
        }
    }

    /** llama4 weight-before-FFN (llama-graph.cpp:2360-2365): {@code out[t * k + j][c] = x[t][c] * w[t][j]}. */
    public static float[] scaleInput(int nTokens, int nEmbd, int k, float[] x, float[] weights) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(weights, "weights");
        positive("n_embd", nEmbd);
        positive("k", k);
        checkCount("x", x.length, (long) nTokens * nEmbd);
        checkCount("weights", weights.length, (long) nTokens * k);
        float[] out = new float[nTokens * k * nEmbd];
        for (int p = 0; p < nTokens * k; p++) {
            int t = p / k;
            for (int c = 0; c < nEmbd; c++) {
                out[p * nEmbd + c] = x[t * nEmbd + c] * weights[p];
            }
        }
        return out;
    }

    /** Result of {@link #bucketize}; immutable. */
    public static final class Buckets {
        private final int[] counts;
        private final int[] offsets;
        private final int[] rowToken;
        private final int[] rowSlot;

        private Buckets(int[] counts, int[] offsets, int[] rowToken, int[] rowSlot) {
            this.counts = counts;
            this.offsets = offsets;
            this.rowToken = rowToken;
            this.rowSlot = rowSlot;
        }

        /** Pairs routed to each expert. */
        public int[] counts() { return counts.clone(); }
        /** First sorted row of each expert (exclusive prefix sum of {@link #counts()}). */
        public int[] offsets() { return offsets.clone(); }
        /** Pair index of each sorted row. */
        public int[] rowToken() { return rowToken.clone(); }
        /** Expert of each sorted row; -1 for a pair whose id is outside {@code [0, nExpert)}. */
        public int[] rowSlot() { return rowSlot.clone(); }
        public int rows() { return rowToken.length; }
    }

    /**
     * Stable counting sort of the {@code nTokens * k} pairs by expert ({@code LLM_MoE_Bucketize}): rows of expert e
     * start at {@code offsets[e]} and list its pairs in ascending pair order; pairs with an id outside
     * {@code [0, nExpert)} follow all valid rows with slot -1.
     */
    public static Buckets bucketize(int nTokens, int k, int nExpert, int[] ids) {
        Objects.requireNonNull(ids, "ids");
        positive("k", k);
        positive("n_expert", nExpert);
        int nRows = nTokens * k;
        checkCount("ids", ids.length, nRows);
        int[] cur = new int[nExpert + 1];
        for (int p = 0; p < nRows; p++) {
            cur[bucket(ids[p], nExpert)]++;
        }
        int[] counts = new int[nExpert];
        int[] offsets = new int[nExpert];
        System.arraycopy(cur, 0, counts, 0, nExpert);
        int run = 0;
        for (int b = 0; b <= nExpert; b++) {
            int c = cur[b];
            if (b < nExpert) {
                offsets[b] = run;
            }
            cur[b] = run;
            run += c;
        }
        int[] rowToken = new int[nRows];
        int[] rowSlot = new int[nRows];
        for (int p = 0; p < nRows; p++) {
            int b = bucket(ids[p], nExpert);
            int pos = cur[b]++;
            rowToken[pos] = p;
            rowSlot[pos] = b < nExpert ? b : -1;
        }
        return new Buckets(counts, offsets, rowToken, rowSlot);
    }

    private static int bucket(int id, int nExpert) {
        return id >= 0 && id < nExpert ? id : nExpert;
    }

    private static void positive(String name, int v) {
        if (v <= 0) {
            throw new IllegalArgumentException(name + " must be positive, got " + v);
        }
    }

    private static void checkCount(String name, long have, long need) {
        if (need < 0 || need > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(name + ": element count " + need + " is out of range");
        }
        if (have < need) {
            throw new IllegalArgumentException(name + " must hold " + need + " values, got " + have);
        }
    }
}

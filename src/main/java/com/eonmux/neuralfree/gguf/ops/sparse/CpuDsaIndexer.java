package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;

import java.util.Objects;

/**
 * CPU twin of {@code LLM_DSA_IndexerScore_<KV>}: DeepSeek sparse-attention lightning-indexer scores
 * (llama.cpp ggml/src/ggml-cpu/ops.cpp:12531-12614; graph llama.cpp src/models/deepseek32.cpp:220-316).
 *
 * <p>For query token {@code t} and cache row {@code s}:
 * {@code scores[t*nKv + s] = sum_h max(iq[t][h] . k[s], 0) * w[t][h]} when {@link #visible} holds, else
 * {@code -INF}. {@code iq} is {@code [nTokens][nIdxHeads][idxDim]}, {@code w} is {@code [nTokens][nIdxHeads]}
 * already prescaled by {@code DsaSpec.weightScale()}, keys are the indexer cache rows (F32, F16 or Q8_0) read as F32,
 * {@code qPos[t]} is the token position and {@code kvPos[s]} the position held by row {@code s} (-1 when empty).
 */
public final class CpuDsaIndexer {

    /** Reads one cache row as F32; implementations must allow concurrent reads of distinct or equal rows. */
    @FunctionalInterface
    public interface RowReader {

        /** Writes row {@code row} into {@code dst[off, off + width)}. */
        void read(int row, float[] dst, int off);

        /** Rows {@code [base + r*width, base + (r+1)*width)} of a dense F32 array. */
        static RowReader ofArray(float[] rows, int base, int width) {
            Objects.requireNonNull(rows, "rows");
            if (base < 0 || width < 1) {
                throw new IllegalArgumentException("row array needs base >= 0 and width >= 1, got " + base + "/" + width);
            }
            return (row, dst, off) -> System.arraycopy(rows, base + row * width, dst, off, width);
        }

        /**
         * Full K rows of a single-head cache (a DSA indexer cache or an MLA latent cache), {@code headDimK} values each.
         *
         * @throws IllegalArgumentException when the cache holds more than one KV head per row
         */
        static RowReader ofKeys(KvAccessor cache) {
            Objects.requireNonNull(cache, "cache");
            if (cache.nHeadKv() != 1) {
                throw new IllegalArgumentException("DSA row reader needs a single-head cache, layer " + cache.layer()
                        + " has " + cache.nHeadKv() + " KV heads per row");
            }
            return (row, dst, off) -> cache.readK(row, 0, dst, off);
        }
    }

    /** Rows scored per parallel task. */
    private static final int ROW_GRAIN = 64;

    private CpuDsaIndexer() {
    }

    /**
     * llama.cpp KV-mask visibility of a row holding position {@code pKv} for a query at {@code pQ}
     * (llama.cpp src/llama-kv-cache.cpp:1664-1710 and {@code llama_hparams::is_masked_swa}, llama.cpp
     * src/llama-hparams.h:506-540).
     */
    public static boolean visible(boolean causal, SwaType swaType, int nSwa, int pKv, int pQ) {
        if (pKv < 0) {
            return false;
        }
        if (causal && pKv > pQ) {
            return false;
        }
        switch (swaType) {
            case NONE:
                return true;
            case STANDARD:
                return pQ - pKv < nSwa;
            case CHUNKED:
                return pKv >= (pQ / nSwa) * nSwa;
            case SYMMETRIC: {
                int halfNSwa = nSwa / 2;
                int diff = pQ - pKv;
                return diff >= -halfNSwa && diff <= halfNSwa;
            }
            default:
                throw new IllegalArgumentException("unknown SWA type " + swaType);
        }
    }

    /**
     * Scores every (token, row) pair, writing {@code scores[scoresOff + t*nKv + s]}.
     *
     * @param keys   indexer cache rows of {@code idxDim} values, rows {@code [0, nKv)}
     * @throws IllegalArgumentException for invalid shapes or a SWA type that needs a window without one
     */
    public static void scores(int nTokens, int nKv, int nIdxHeads, int idxDim, boolean causal, SwaType swaType, int nSwa,
                              float[] iq, int iqOff, float[] w, int wOff, RowReader keys,
                              int[] qPos, int qPosOff, int[] kvPos, int kvPosOff, float[] scores, int scoresOff) {
        checkShape(nTokens, nKv, nIdxHeads, idxDim, swaType, nSwa);
        Objects.requireNonNull(iq, "iq");
        Objects.requireNonNull(w, "w");
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(qPos, "qPos");
        Objects.requireNonNull(kvPos, "kvPos");
        Objects.requireNonNull(scores, "scores");
        CpuParallel.parallelFor(nKv, ROW_GRAIN, (from, to) -> {
            float[] k = new float[idxDim];
            for (int s = from; s < to; s++) {
                int pKv = kvPos[kvPosOff + s];
                boolean read = false;
                for (int t = 0; t < nTokens; t++) {
                    int dst = scoresOff + t * nKv + s;
                    if (!visible(causal, swaType, nSwa, pKv, qPos[qPosOff + t])) {
                        scores[dst] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    if (!read) {
                        keys.read(s, k, 0);
                        read = true;
                    }
                    scores[dst] = score(iq, iqOff + t * nIdxHeads * idxDim, w, wOff + t * nIdxHeads, k, nIdxHeads, idxDim);
                }
            }
        });
    }

    /**
     * Scores with keys taken from a dense F32 array of rows {@code keys[keysOff + s*idxDim ..]}.
     *
     * @see #scores(int, int, int, int, boolean, SwaType, int, float[], int, float[], int, RowReader, int[], int, int[],
     * int, float[], int)
     */
    public static void scores(int nTokens, int nKv, int nIdxHeads, int idxDim, boolean causal, SwaType swaType, int nSwa,
                              float[] iq, int iqOff, float[] w, int wOff, float[] keys, int keysOff,
                              int[] qPos, int qPosOff, int[] kvPos, int kvPosOff, float[] scores, int scoresOff) {
        scores(nTokens, nKv, nIdxHeads, idxDim, causal, swaType, nSwa, iq, iqOff, w, wOff,
                RowReader.ofArray(keys, keysOff, idxDim), qPos, qPosOff, kvPos, kvPosOff, scores, scoresOff);
    }

    /** {@code sum_h max(q_h . k, 0) * w_h} over the heads of one token. */
    static float score(float[] iq, int qOff, float[] w, int wOff, float[] k, int nIdxHeads, int idxDim) {
        float score = 0.0f;
        for (int h = 0; h < nIdxHeads; h++) {
            int q = qOff + h * idxDim;
            float qk = 0.0f;
            for (int d = 0; d < idxDim; d++) {
                qk += iq[q + d] * k[d];
            }
            score += Math.max(qk, 0.0f) * w[wOff + h];
        }
        return score;
    }

    private static void checkShape(int nTokens, int nKv, int nIdxHeads, int idxDim, SwaType swaType, int nSwa) {
        if (nTokens < 1 || nKv < 0 || nIdxHeads < 1 || idxDim < 1) {
            throw new IllegalArgumentException("DSA indexer needs nTokens >= 1, nKv >= 0, heads >= 1 and dim >= 1, got "
                    + nTokens + "/" + nKv + "/" + nIdxHeads + "/" + idxDim);
        }
        SparseKernels.checkIndexerMask(swaType, nSwa);
    }
}

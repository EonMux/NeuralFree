package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;

import java.util.Objects;

/**
 * CPU twin of {@code LLM_QSA_KpoolUpdate_<KV>} and {@code LLM_QSA_KpoolScore_<KV>} ({@code kernels/llm/sparse/qsa.cl}),
 * with the same index formulas; see {@link MsaKernels} for the step sequence and {@link CpuMsa#gatherAttn} for the
 * attention over the selected pools.
 *
 * <p>Pool {@code i} is positions {@code [i*kpool, (i+1)*kpool)} (one sequence from position 0, llama.cpp
 * src/llama-memory-hybrid-idx.cpp:389-474); position {@code p} lives in indexer row {@code p % kvRows} and the pooled
 * key of a pool is stored in its last member's row (:652, :852-854). The indexer cache is a one-head K-only accessor
 * whose rows of {@code rowWidth = idxDim * kpoolRow} values hold {@code [key | gate | pooled]} (gated) or
 * {@code [raw key | pooled]} (mean), in the cache's element type (any KV type; block types with whole-block slots,
 * {@link MsaKernels#checkKpoolCache}).
 */
public final class CpuQsa {

    private CpuQsa() {
    }

    /** First pool completed by a step at positions {@code [pos0, pos0 + nTokens)}. */
    public static int firstNewPool(int pos0, int kpool) {
        return MsaKernels.firstNewPool(pos0, kpool);
    }

    /** Pools whose last member the step at positions {@code [pos0, pos0 + nTokens)} writes. */
    public static int newPoolCount(int pos0, int nTokens, int kpool) {
        return MsaKernels.newPoolCount(pos0, nTokens, kpool);
    }

    /** Complete pools among positions {@code [0, nPos)}. */
    public static int poolCount(int nPos, int kpool) {
        return MsaKernels.completePools(nPos, kpool);
    }

    /**
     * {@code LLM_QSA_KpoolUpdate} stage POOL: for pool {@code i = firstPool + j}, member rows
     * {@code r_k = (i*kpool + k) % kvRows} and value {@code d < idxDim}:
     * <pre>
     *   GATED: x_k = gate_k[d] + ape[k*idxDim + d];  pooled[j*idxDim + d] = sum_k (e^(x_k - max x) * (1/sum e^(x - max x))) * key_k[d]
     *          (llama.cpp src/models/glm5-next.cpp:926-941)
     *   MEAN:  pooled[j*idxDim + d] = (key_0[d] + ... + key_{kpool-1}[d]) * (1/kpool)
     *          (llama.cpp src/models/qwen4exp.cpp:770-780)
     * </pre>
     * key, gate and pooled are the slots at offsets 0, {@code gateOff} and {@code pooledOff} of a cached row.
     *
     * @param mode {@link MsaKernels#KPOOL_MODE_GATED} or {@link MsaKernels#KPOOL_MODE_MEAN}
     * @param ape {@code kpool * idxDim} learned pool position offsets ({@code indexer_kpool_ape}); unused (may be
     *            null) in mode MEAN
     * @param pooled output of {@code nNew * idxDim} values
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void kpoolPool(int nNew, int firstPool, int mode, int kpool, int idxDim, int gateOff, int kvRows,
                                 float[] ape, KvAccessor idx, float[] pooled) {
        int rowWidth = checkCache(idx);
        MsaKernels.checkKpool(kpool, idxDim, rowWidth, gateOff, rowWidth - idxDim, kvRows, mode);
        MsaKernels.checkKpoolCache(idx.layout().dtypeK(), idxDim, rowWidth, gateOff, rowWidth - idxDim, mode);
        checkNew(nNew, firstPool);
        Objects.requireNonNull(pooled, "pooled");
        if (mode == MsaKernels.KPOOL_MODE_GATED) {
            Objects.requireNonNull(ape, "ape");
        }
        CpuParallel.parallelFor(nNew, 1, (from, to) -> {
            float[] rows = new float[kpool * rowWidth];
            float[] x = new float[kpool];
            for (int j = from; j < to; j++) {
                long pos0 = ((long) firstPool + j) * kpool;
                for (int k = 0; k < kpool; k++) {
                    idx.readK((int) ((pos0 + k) % kvRows), 0, rows, k * rowWidth);
                }
                int o = j * idxDim;
                for (int d = 0; d < idxDim; d++) {
                    if (mode == MsaKernels.KPOOL_MODE_GATED) {
                        float mx = Float.NEGATIVE_INFINITY;
                        for (int k = 0; k < kpool; k++) {
                            x[k] = rows[k * rowWidth + gateOff + d] + ape[k * idxDim + d];
                            mx = Math.max(mx, x[k]);
                        }
                        float sum = 0f;
                        for (int k = 0; k < kpool; k++) {
                            sum += (float) Math.exp(x[k] - mx);
                        }
                        float inv = 1f / sum;
                        float acc = 0f;
                        for (int k = 0; k < kpool; k++) {
                            acc += ((float) Math.exp(x[k] - mx) * inv) * rows[k * rowWidth + d];
                        }
                        pooled[o + d] = acc;
                    } else {
                        float acc = 0f;
                        for (int k = 0; k < kpool; k++) {
                            acc += rows[k * rowWidth + d];
                        }
                        pooled[o + d] = acc * (1f / kpool);
                    }
                }
            }
        });
    }

    /**
     * {@code LLM_QSA_KpoolUpdate} stage STORE: writes {@code pooled[j*idxDim ..]} into the pooled slot
     * ({@code pooledOff}) of the last member row {@code ((firstPool + j + 1)*kpool - 1) % kvRows}; the cache rounds to
     * its element type (F16 / BF16 to nearest even, a block type per 32-value block by its reference quantizer). Only
     * the pooled slot is written ({@link KvAccessor#writeKSlice}, the ggml_set_rows into the pooled view); the key and
     * gate slots keep their stored bits. Runs on the calling thread.
     *
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void kpoolStore(int nNew, int firstPool, int kpool, int idxDim, int pooledOff, int kvRows,
                                  float[] pooled, KvAccessor idx) {
        int rowWidth = checkCache(idx);
        MsaKernels.checkKpool(kpool, idxDim, rowWidth, 0, pooledOff, kvRows, MsaKernels.KPOOL_MODE_MEAN);
        MsaKernels.checkKpoolCache(idx.layout().dtypeK(), idxDim, rowWidth, 0, pooledOff, MsaKernels.KPOOL_MODE_MEAN);
        checkNew(nNew, firstPool);
        Objects.requireNonNull(pooled, "pooled");
        for (int j = 0; j < nNew; j++) {
            int rep = (int) ((((long) firstPool + j + 1) * kpool - 1) % kvRows);
            idx.writeKSlice(rep, pooledOff, idxDim, pooled, j * idxDim);
        }
    }

    /**
     * {@code LLM_QSA_KpoolScore}: for token {@code t} and pool {@code i < nPool} with last member {@code e = (i+1)*kpool - 1}
     * <pre>
     *   scores[t*nPool + i] = sum_h max(iq[iqOff + (t*nIdxHeads + h)*idxDim ..] . pooled_i, 0) * w_h   when e &lt;= qPos[t]
     *                         and kvPos[e % kvRows] == e, -INF otherwise
     *   w_h = (w == null ? 1 : w[wOff + t*nIdxHeads + h]) * wScale
     * </pre>
     * (llama.cpp src/models/glm5-next.cpp:951-973, src/models/qwen4exp.cpp:797-810; pool mask of
     * src/llama-memory-hybrid-idx.cpp:921-943).
     *
     * @param w per-token head weights, or null for constant weights (qwen4exp)
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void kpoolScore(int nTokens, int nPool, int nIdxHeads, int idxDim, int kpool, int pooledOff, int kvRows,
                                  float wScale, float[] iq, int iqOff, float[] w, int wOff, KvAccessor idx,
                                  int[] qPos, int[] kvPos, float[] scores) {
        int rowWidth = checkCache(idx);
        if (nTokens < 1 || nPool < 1) {
            throw new IllegalArgumentException("QSA pool score needs nTokens and nPool >= 1, got " + nTokens + "/" + nPool);
        }
        MsaKernels.checkKpoolScore(nIdxHeads, idxDim, kpool, rowWidth, pooledOff, kvRows, wScale, 1);
        MsaKernels.checkKpoolCache(idx.layout().dtypeK(), idxDim, rowWidth, 0, pooledOff, MsaKernels.KPOOL_MODE_MEAN);
        Objects.requireNonNull(iq, "iq");
        Objects.requireNonNull(qPos, "qPos");
        Objects.requireNonNull(kvPos, "kvPos");
        Objects.requireNonNull(scores, "scores");
        float[] keys = new float[nPool * idxDim];
        boolean[] cached = new boolean[nPool];
        int[] last = new int[nPool];
        CpuParallel.parallelFor(nPool, 16, (from, to) -> {
            float[] row = new float[rowWidth];
            for (int i = from; i < to; i++) {
                int e = (int) (((long) i + 1) * kpool - 1);
                last[i] = e;
                cached[i] = kvPos[e % kvRows] == e;
                if (cached[i]) {
                    idx.readK(e % kvRows, 0, row, 0);
                    System.arraycopy(row, pooledOff, keys, i * idxDim, idxDim);
                }
            }
        });
        CpuParallel.parallelFor(nTokens, 1, (from, to) -> {
            for (int t = from; t < to; t++) {
                for (int i = 0; i < nPool; i++) {
                    if (!cached[i] || last[i] > qPos[t]) {
                        scores[t * nPool + i] = Float.NEGATIVE_INFINITY;
                        continue;
                    }
                    float score = 0f;
                    for (int h = 0; h < nIdxHeads; h++) {
                        float a = CpuMsa.dot(iq, iqOff + (t * nIdxHeads + h) * idxDim, keys, i * idxDim, idxDim);
                        float wh = (w == null ? 1f : w[wOff + t * nIdxHeads + h]) * wScale;
                        score += Math.max(a, 0f) * wh;
                    }
                    scores[t * nPool + i] = score;
                }
            }
        });
    }

    private static int checkCache(KvAccessor idx) {
        if (Objects.requireNonNull(idx, "idx").nHeadKv() != 1) {
            throw new IllegalArgumentException("QSA indexer cache must be a one-head K-only cache, has " + idx.nHeadKv()
                    + " heads");
        }
        return idx.headDimK();
    }

    private static void checkNew(int nNew, int firstPool) {
        if (nNew < 1 || firstPool < 0) {
            throw new IllegalArgumentException("QSA k-pool update needs nNew >= 1 and firstPool >= 0, got " + nNew + "/"
                    + firstPool);
        }
    }
}

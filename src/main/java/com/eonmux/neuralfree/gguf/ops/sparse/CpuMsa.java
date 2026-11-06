package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@code LLM_MSA_BlockScore} and {@code LLM_MSA_GatherAttn} ({@code kernels/llm/sparse/msa.cl}), with the
 * same index formulas; see {@link MsaKernels} for the step sequence.
 *
 * <p>The kernels address one sequence's cached positions through a {@link PositionRows} map: position {@code p} lives
 * in row {@code rows.row(p)} and is present when that row is not -1 (llama.cpp {@code set_input_pos_slot},
 * src/llama-kv-cache-msa.cpp:303-345). The forms taking {@code (kvRows, kvPos)} address a contiguous cache:
 * {@code p} lives in KV row {@code p % kvRows} and is present when {@code kvPos[p % kvRows] == p}
 * ({@link PositionRows#ring}). A present position is visible to a token at position {@code qPos} when also
 * {@code p <= qPos} (llama.cpp src/llama-kv-cache-msa.cpp:348-395), or always in a non-causal step (the rows of a
 * bidirectional media chunk).
 */
public final class CpuMsa {

    private CpuMsa() {
    }

    /** Position blocks covering positions {@code [0, nPos)}. */
    public static int blockCount(int nPos, int blk) {
        return MsaKernels.blockCount(nPos, blk);
    }

    /** Whether position {@code p} is cached and causally visible to a token at {@code pQ}. */
    public static boolean visible(int p, int pQ, int kvRows, int[] kvPos) {
        return p >= 0 && p <= pQ && kvPos[p % kvRows] == p;
    }

    /** Whether position {@code p} is cached and visible to a token at {@code pQ} (any cached one when non-causal). */
    static boolean visible(int p, int pQ, PositionRows rows, boolean causal) {
        return p >= 0 && (!causal || p <= pQ) && rows.row(p) >= 0;
    }

    /**
     * {@code LLM_MSA_BlockScore}: for token {@code t}, index head {@code h} and block {@code b}
     * <pre>
     *   bs = max over visible p in [b*blk, min((b+1)*blk, nPos)) of iq[iqOff + (t*nKvHeads + h)*idxDim ..] . ik[p % kvRows]
     *        (-INF without a visible position; ggml_pool_2d MAX of llama.cpp src/models/minimax-m3.cpp:393, :475)
     *   scores[(t*nKvHeads + h)*nblk + b] = bs + 1e30 for L - localBlocks &lt; b &lt;= L (L = qPos[t]/blk), else bs
     *        (local-force bias, llama.cpp src/models/minimax-m3.cpp:107-121)
     * </pre>
     *
     * @param ik one-head index-key cache whose rows hold {@code idxDim} values (for example {@code CpuMemory.msa(il)})
     * @param scores output of {@code nTokens * nKvHeads * blockCount(nPos, blk)} values
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void blockScore(int nTokens, int nPos, int nKvHeads, int idxDim, int blk, int localBlocks, int kvRows,
                                  float[] iq, int iqOff, KvAccessor ik, int[] qPos, int[] kvPos, float[] scores) {
        if (kvRows < 1) {
            throw new IllegalArgumentException("MSA block score needs kvRows >= 1, got " + kvRows);
        }
        blockScore(nTokens, nPos, nKvHeads, idxDim, blk, localBlocks, iq, iqOff, ik, qPos,
                PositionRows.ring(kvRows, Objects.requireNonNull(kvPos, "kvPos")), true, scores);
    }

    /**
     * {@code LLM_MSA_BlockScore} over one sequence's positions {@code [0, nPos)} as {@code rows} addresses them; the
     * formula of {@link #blockScore(int, int, int, int, int, int, int, float[], int, KvAccessor, int[], int[],
     * float[])} with {@code ik[rows.row(p)]}, a non-causal step seeing every cached position.
     *
     * @param causal whether a position after the token's is masked
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void blockScore(int nTokens, int nPos, int nKvHeads, int idxDim, int blk, int localBlocks,
                                  float[] iq, int iqOff, KvAccessor ik, int[] qPos, PositionRows rows, boolean causal,
                                  float[] scores) {
        if (nTokens < 1 || nPos < 1 || nKvHeads < 1 || idxDim < 1 || blk < 1 || localBlocks < 0) {
            throw new IllegalArgumentException("MSA block score needs positive nTokens/nPos/nKvHeads/idxDim/blk and"
                    + " localBlocks >= 0, got " + nTokens + "/" + nPos + "/" + nKvHeads + "/" + idxDim + "/" + blk + "/"
                    + localBlocks);
        }
        Objects.requireNonNull(iq, "iq");
        Objects.requireNonNull(qPos, "qPos");
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(scores, "scores");
        if (Objects.requireNonNull(ik, "ik").headDimK() != idxDim || ik.nHeadKv() != 1) {
            throw new IllegalArgumentException("MSA index-key cache must hold one head of " + idxDim + " values, has "
                    + ik.nHeadKv() + " of " + ik.headDimK());
        }
        int nblk = blockCount(nPos, blk);
        CpuParallel.parallelFor(nTokens * nblk, 1, (from, to) -> {
            float[] key = new float[idxDim];
            float[] best = new float[nKvHeads];
            for (int g = from; g < to; g++) {
                int t = g / nblk;
                int b = g - t * nblk;
                int pQ = qPos[t];
                Arrays.fill(best, Float.NEGATIVE_INFINITY);
                int pEnd = Math.min(b * blk + blk, nPos);
                for (int p = b * blk; p < pEnd; p++) {
                    if (!visible(p, pQ, rows, causal)) {
                        continue;
                    }
                    ik.readK(rows.row(p), 0, key, 0);
                    for (int h = 0; h < nKvHeads; h++) {
                        best[h] = Math.max(best[h], dot(iq, iqOff + (t * nKvHeads + h) * idxDim, key, 0, idxDim));
                    }
                }
                int lBlk = pQ / blk;
                float bias = (b <= lBlk && b > lBlk - localBlocks) ? MsaKernels.LOCAL_BIAS : 0f;
                for (int h = 0; h < nKvHeads; h++) {
                    scores[(t * nKvHeads + h) * nblk + b] = best[h] + bias;
                }
            }
        });
    }

    /**
     * {@code LLM_MSA_GatherAttn}: head {@code h} of token {@code t} attends, with an exact softmax, over the visible
     * positions of its selected blocks
     * <pre>
     *   kvh = h / (nHead / nKvHeads), sh = selHeads == 1 ? 0 : kvh
     *   slots: blocks[(t*selHeads + sh)*k + s] for s &lt; k (-1 unused), plus with tail the block qPos[t]/blk when
     *          (qPos[t] + 1) % blk != 0; cells p = block*blk + o, o &lt; blk, kept when visible
     *   s_p = (q[qOff + t*qStride + h*hd ..] . K_kvh[p % kvRows]) * kqScale
     *   out[outOff + t*outStride + h*hd + d] = sum_p softmax(s)_p * V_kvh[p % kvRows][d]   (0 without a cell)
     * </pre>
     * MSA: llama.cpp src/models/minimax-m3.cpp:410-447 and :490-522; QSA k-pool attention: llama.cpp
     * src/models/qwen4exp.cpp:812-880, src/models/glm5-next.cpp:975-1042 with the tail cells of
     * src/llama-memory-hybrid-idx.cpp:946-966. The tail block is never among the selected blocks (an incomplete pool
     * scores -INF), so no cell is counted twice.
     *
     * @param kCache cache whose K heads are {@code hd} values (an MLA latent cache passes itself as {@code vCache})
     * @param vCache cache whose V heads are {@code hd} values
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void gatherAttn(int nTokens, int nHead, int nKvHeads, int hd, int blk, int k, float kqScale,
                                  int selHeads, boolean tail, int kvRows, float[] q, int qOff, int qStride,
                                  KvAccessor kCache, KvAccessor vCache, int[] qPos, int[] kvPos, int[] blocks,
                                  float[] out, int outOff, int outStride) {
        if (kvRows < 1) {
            throw new IllegalArgumentException("MSA gather needs kvRows >= 1, got " + kvRows);
        }
        gatherAttn(nTokens, nHead, nKvHeads, hd, blk, k, kqScale, selHeads, tail, 0, q, qOff, qStride, kCache, vCache,
                qPos, PositionRows.ring(kvRows, Objects.requireNonNull(kvPos, "kvPos")), true, blocks, 0, out, outOff,
                outStride);
    }

    /**
     * {@code LLM_MSA_GatherAttn} over one sequence's positions as {@code rows} addresses them: the formula of
     * {@link #gatherAttn(int, int, int, int, int, int, float, int, boolean, int, float[], int, int, KvAccessor,
     * KvAccessor, int[], int[], int[], float[], int, int)} with {@code K, V[rows.row(p)]} and token {@code t}'s
     * selection at {@code blocks[blocksOff + (t*selHeads + sh)*k ..]}. A non-causal step sees every cached position;
     * its tail block is the one the sequence's last position {@code nPos - 1} falls in (when incomplete), the block
     * a causal token's own position falls in otherwise, so a selected (complete) block is never counted twice.
     *
     * @param nPos   positions the sequence holds (its highest position + 1); read by a non-causal tail only
     * @param causal whether a position after the token's is masked
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void gatherAttn(int nTokens, int nHead, int nKvHeads, int hd, int blk, int k, float kqScale,
                                  int selHeads, boolean tail, int nPos, float[] q, int qOff, int qStride,
                                  KvAccessor kCache, KvAccessor vCache, int[] qPos, PositionRows rows, boolean causal,
                                  int[] blocks, int blocksOff, float[] out, int outOff, int outStride) {
        if (nTokens < 1 || nHead < 1 || nKvHeads < 1 || nHead % nKvHeads != 0 || hd < 1 || blk < 1 || k < 0
                || (selHeads != 1 && selHeads != nKvHeads) || blocksOff < 0 || (!causal && tail && nPos < 1)) {
            throw new IllegalArgumentException("MSA gather needs positive nTokens/nHead/hd/blk, nHead divisible by"
                    + " nKvHeads, k >= 0, selHeads in {1, nKvHeads}, blocksOff >= 0 and a non-causal tail over"
                    + " nPos >= 1, got " + nTokens + "/" + nHead + "/" + nKvHeads + "/" + hd + "/" + blk + "/" + k + "/"
                    + selHeads + "/" + blocksOff + "/" + nPos);
        }
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(qPos, "qPos");
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(blocks, "blocks");
        Objects.requireNonNull(out, "out");
        checkHeads("K", Objects.requireNonNull(kCache, "kCache").nHeadKv(), kCache.headDimK(), nKvHeads, hd);
        checkHeads("V", Objects.requireNonNull(vCache, "vCache").nHeadKv(), vCache.headDimV(), nKvHeads, hd);
        int nCells = (k + (tail ? 1 : 0)) * blk;
        CpuParallel.parallelFor(nTokens * nHead, 1, (from, to) -> {
            float[] qh = new float[hd];
            float[] row = new float[hd];
            float[] s = new float[nCells];
            int[] cells = new int[nCells];
            float[] acc = new float[hd];
            for (int g = from; g < to; g++) {
                int t = g / nHead;
                int h = g - t * nHead;
                int kvh = h / (nHead / nKvHeads);
                int sel = blocksOff + (t * selHeads + (selHeads == 1 ? 0 : kvh)) * k;
                int pQ = qPos[t];
                int tailEnd = causal ? pQ : nPos - 1;
                int tailBlk = tail && (tailEnd + 1) % blk != 0 ? tailEnd / blk : -1;
                System.arraycopy(q, qOff + t * qStride + h * hd, qh, 0, hd);
                float m = Float.NEGATIVE_INFINITY;
                for (int c = 0; c < nCells; c++) {
                    int slot = c / blk;
                    int bk = slot < k ? blocks[sel + slot] : tailBlk;
                    int p = bk * blk + (c - slot * blk);
                    cells[c] = -1;
                    s[c] = Float.NEGATIVE_INFINITY;
                    if (bk >= 0 && visible(p, pQ, rows, causal)) {
                        cells[c] = rows.row(p);
                        kCache.readK(cells[c], kvh, row, 0);
                        s[c] = dot(qh, 0, row, 0, hd) * kqScale;
                        m = Math.max(m, s[c]);
                    }
                }
                Arrays.fill(acc, 0f);
                float l = 0f;
                for (int c = 0; c < nCells; c++) {
                    if (cells[c] < 0) {
                        continue;
                    }
                    float w = (float) Math.exp(s[c] - m);
                    l += w;
                    vCache.readV(cells[c], kvh, row, 0);
                    for (int d = 0; d < hd; d++) {
                        acc[d] += w * row[d];
                    }
                }
                int o = outOff + t * outStride + h * hd;
                for (int d = 0; d < hd; d++) {
                    out[o + d] = l > 0f ? acc[d] / l : 0f;
                }
            }
        });
    }

    static float dot(float[] a, int aOff, float[] b, int bOff, int n) {
        float acc = 0f;
        for (int i = 0; i < n; i++) {
            acc += a[aOff + i] * b[bOff + i];
        }
        return acc;
    }

    private static void checkHeads(String which, int heads, int dim, int nKvHeads, int hd) {
        if (heads != nKvHeads || dim != hd) {
            throw new IllegalArgumentException("MSA gather " + which + " cache must hold " + nKvHeads + " heads of " + hd
                    + " values, has " + heads + " of " + dim);
        }
    }
}

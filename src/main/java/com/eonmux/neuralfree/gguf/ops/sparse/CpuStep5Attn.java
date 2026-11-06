package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU attention of the Step5 GQA layers, reading the cached keys and values through a {@link KvAccessor} and applying
 * the per-head sigmoid gate (vLLM {@code common/ops/sparse_decode_attention.py} and the FlashAttention window path of
 * the sliding blocks).
 *
 * <p>The sparse form attends the tokens of each token's selected regions up to its own position; the windowed form
 * attends the {@code window} positions ending at its own (the whole causal history when {@code window <= 0}). Both
 * reproduce the vLLM BF16 arithmetic: the softmax weights are rounded to BF16 before the value accumulation, the
 * normalized output is rounded to BF16, multiplied by the BF16-rounded sigmoid and rounded once more. A row missing
 * from the position map (an empty cache row) is skipped.
 *
 * <p>Work is sharded over (token, head); each worker allocates its own score, key, value and accumulator buffers.
 */
public final class CpuStep5Attn {

    /** {@code log2(e)}, the base of the vLLM exponentials. */
    private static final float LOG2E = 1.4426950408889634f;

    private CpuStep5Attn() {
    }

    /**
     * Sparse attention over selected regions.
     *
     * @param q        queries, {@code [token][head][headDim]} from {@code qOff}
     * @param scale    {@code 1/sqrt(headDim)}
     * @param kv       K/V cache holding {@code nKv} heads of {@code headDim} values
     * @param rowAt    cache row of a position, or -1 when it is not held
     * @param qPos     positions of the query rows
     * @param sel      per-token selections, {@code slotCount} entries per KV group, {@code -1} unused
     * @param gate     per-head gate logits ({@code [token][head]}) or null when the layer has no gate
     * @param out      output, {@code [token][head][headDim]} from {@code outOff}
     * @throws IllegalArgumentException for invalid shapes
     */
    public static void sparse(int nTokens, int nHead, int nKv, int headDim, int regionBlock, int slotCount,
                              float[] q, int qOff, int qStride, float scale, KvAccessor kv, IntUnaryOperator rowAt,
                              int[] qPos, int[] sel, int selOff, int selRow, int selGroup,
                              float[] gate, int gateOff, int gateStride, float[] out, int outOff, int outStride) {
        checkCommon(nTokens, nHead, nKv, headDim, q, qOff, qStride, kv, qPos, out, outOff, outStride);
        if (regionBlock <= 0 || slotCount < 0) {
            throw new IllegalArgumentException("sparse attention needs a positive region block and slot count, got "
                    + regionBlock + "/" + slotCount);
        }
        int headsPerGroup = nHead / nKv;
        CpuParallel.parallelFor(nTokens * nHead, 1, (from, to) -> {
            Work w = new Work(headDim, slotCount * regionBlock);
            for (int g = from; g < to; g++) {
                int t = g / nHead;
                int h = g - t * nHead;
                int group = h / headsPerGroup;
                int pos = qPos[t];
                int selBase = selOff + t * selRow + group * selGroup;
                w.begin(q, qOff + t * qStride + h * headDim);
                for (int s = 0; s < slotCount; s++) {
                    int region = sel[selBase + s];
                    if (region < 0) {
                        continue;
                    }
                    int first = region * regionBlock;
                    int last = Math.min(first + regionBlock - 1, pos);
                    for (int p = first; p <= last; p++) {
                        int row = rowAt.applyAsInt(p);
                        if (row >= 0) {
                            w.add(kv, row, group, scale);
                        }
                    }
                }
                w.finish(kv, group, gate == null ? Float.NaN : gate[gateOff + t * gateStride + h],
                        out, outOff + t * outStride + h * headDim);
            }
        });
    }

    /**
     * Dense causal attention over the last {@code window} positions of every query token (the whole causal history
     * when {@code window <= 0}), with the same arithmetic as {@link #sparse}.
     *
     * @throws IllegalArgumentException for invalid shapes or a negative window
     */
    public static void windowed(int nTokens, int nHead, int nKv, int headDim, int window, float[] q, int qOff,
                                int qStride, float scale, KvAccessor kv, IntUnaryOperator rowAt, int[] qPos,
                                float[] gate, int gateOff, int gateStride, float[] out, int outOff, int outStride) {
        checkCommon(nTokens, nHead, nKv, headDim, q, qOff, qStride, kv, qPos, out, outOff, outStride);
        if (window < 0) {
            throw new IllegalArgumentException("sliding window " + window + " is negative");
        }
        int headsPerGroup = nHead / nKv;
        int slots = Math.max(1, window);
        CpuParallel.parallelFor(nTokens * nHead, 1, (from, to) -> {
            Work w = new Work(headDim, slots);
            for (int g = from; g < to; g++) {
                int t = g / nHead;
                int h = g - t * nHead;
                int group = h / headsPerGroup;
                int pos = qPos[t];
                int first = window > 0 ? Math.max(0, pos - window + 1) : 0;
                w.begin(q, qOff + t * qStride + h * headDim);
                for (int p = first; p <= pos; p++) {
                    int row = rowAt.applyAsInt(p);
                    if (row >= 0) {
                        w.add(kv, row, group, scale);
                    }
                }
                w.finish(kv, group, gate == null ? Float.NaN : gate[gateOff + t * gateStride + h],
                        out, outOff + t * outStride + h * headDim);
            }
        });
    }

    private static void checkCommon(int nTokens, int nHead, int nKv, int headDim, float[] q, int qOff, int qStride,
                                    KvAccessor kv, int[] qPos, float[] out, int outOff, int outStride) {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(qPos, "qPos");
        Objects.requireNonNull(out, "out");
        if (nTokens < 1 || nHead < 1 || nKv < 1 || headDim < 1 || nHead % nKv != 0) {
            throw new IllegalArgumentException("Step5 attention needs positive tokens/heads/head dim and query heads"
                    + " divisible into KV heads, got " + nTokens + "/" + nHead + "/" + nKv + "/" + headDim);
        }
        if (kv.nHeadKv() != nKv || kv.headDimK() != headDim || kv.headDimV() != headDim) {
            throw new IllegalArgumentException("Step5 attention cache holds " + kv.nHeadKv() + " heads of "
                    + kv.headDimK() + "/" + kv.headDimV() + ", the mixer needs " + nKv + " of " + headDim);
        }
        if (qOff < 0 || outOff < 0 || qStride < 0 || outStride < 0
                || qOff + (long) (nTokens - 1) * qStride + nHead * headDim > q.length
                || outOff + (long) (nTokens - 1) * outStride + nHead * headDim > out.length) {
            throw new IllegalArgumentException("Step5 attention query or output view is outside its array");
        }
    }

    /** Per-worker softmax state of one (token, head): the vLLM two-pass form with BF16 probabilities. */
    private static final class Work {
        private final float[] qh;
        private final float[] row;
        private float[] scores;
        private int[] cells;
        private int count;

        Work(int headDim, int maxCells) {
            this.qh = new float[headDim];
            this.row = new float[headDim];
            this.scores = new float[Math.max(1, maxCells)];
            this.cells = new int[Math.max(1, maxCells)];
        }

        void begin(float[] q, int qBase) {
            System.arraycopy(q, qBase, qh, 0, qh.length);
            count = 0;
        }

        void add(KvAccessor kv, int rowIndex, int group, float scale) {
            if (count >= scores.length) {
                int grown = scores.length * 2;
                scores = Arrays.copyOf(scores, grown);
                cells = Arrays.copyOf(cells, grown);
            }
            kv.readK(rowIndex, group, row, 0);
            float dot = 0f;
            for (int d = 0; d < qh.length; d++) {
                dot += qh[d] * row[d];
            }
            cells[count] = rowIndex;
            scores[count++] = dot * scale * LOG2E;
        }

        void finish(KvAccessor kv, int group, float gateLogit, float[] out, int outBase) {
            float[] acc = new float[qh.length];
            if (count == 0) {
                Arrays.fill(out, outBase, outBase + qh.length, 0f);
                return;
            }
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
                max = Math.max(max, scores[i]);
            }
            double total = 0.0;
            for (int i = 0; i < count; i++) {
                float e = Step5Dtypes.exp2(scores[i] - max);
                scores[i] = e;
                total += e;
            }
            for (int i = 0; i < count; i++) {
                float weight = Step5Dtypes.bf16(scores[i]);
                kv.readV(cells[i], group, row, 0);
                for (int d = 0; d < qh.length; d++) {
                    acc[d] += weight * row[d];
                }
            }
            float inverse = total > 0.0 ? (float) (1.0 / Math.max(total, 1e-30)) : 0f;
            float sigmoid = Float.isNaN(gateLogit) ? 1f : Step5Dtypes.gateSigmoid(gateLogit);
            for (int d = 0; d < qh.length; d++) {
                out[outBase + d] = Step5Dtypes.bf16(Step5Dtypes.bf16(acc[d] * inverse) * sigmoid);
            }
        }
    }
}

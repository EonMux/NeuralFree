package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;

import java.util.Objects;

/**
 * CPU dual-chunk attention (Qwen2.5-1M, vLLM's dual-chunk flash-attention backend): the query's chunk ({@code intra},
 * causal), the preceding chunk ({@code succ}, non-causal) and every earlier chunk ({@code inter}, non-causal) are
 * scored separately and merged by one softmax over the concatenated logits.
 *
 * <p>Each stage reads a different query rotation plane and a different key range of the same sequence. Prefill walks
 * the sequence chunk by chunk ({@code DualChunkFlashAttentionImpl._dual_chunk_flash_attn_prefill_func}); a one-token
 * step takes the three stages of its current chunk directly ({@code _dual_chunk_flash_attn_decoding}). The stages are
 * merged with the log-sum-exp of every row, so the result is a single softmax over all visible keys.
 *
 * <p>Key ranges are logical slots of a {@link KeySpan}: contiguous cache rows, or page-table slots of a paged
 * sequence. The caller rotates the query planes and scales the logits before the call. Immutable.
 */
public final class CpuDualChunkAttention {

    /** The key rows a stage reads: a contiguous cache view or a paged sequence's slot range. */
    public interface KeySpan {
        /** Logical slots of the sequence (its cache length). */
        int keys();

        /**
         * Attention of {@code n} query rows of {@code q} (row {@code t} at {@code qOff + t * qRowStride}, position
         * {@code pos[posOff + t]}) against stage slots {@code [keyStart, keyEnd)}, writing normalized rows to
         * {@code out} and their log-sum-exp to {@code lse}.
         */
        void attend(CpuAttentionCore.Params p, int n, float[] q, int qOff, int qRowStride, int[] pos, int posOff,
                    int keyStart, int keyEnd, float[] out, int outOff, float[] lse, int lseOff);
    }

    private CpuDualChunkAttention() {
    }

    /**
     * The attention of {@code nTokens} query rows whose sequence occupies {@code keys.keys()} slots.
     *
     * @param q            the five query planes; plane {@code s} starts at {@code s * planeStride}
     * @param qTokenBase   row of the call's first token within a query plane and within {@code out}
     * @param pos          positions of the call's tokens
     * @param out          output rows, row {@code t} at {@code outOff + t * outRowStride}
     * @param scratchOut   at least {@code chunkLength * outDim} floats
     * @param scratchLse   at least {@code chunkLength * nHead} floats
     * @param mergeLse     at least {@code chunkLength * nHead} floats; per-row log-sum-exp accumulator
     * @throws IllegalArgumentException when the token count, plane stride or scratch sizes are inconsistent
     */
    public static void attend(DualChunkSpec dc, AttentionSpec spec, CpuAttentionCore.Params causal,
                              CpuAttentionCore.Params nonCausal, KeySpan keys, int nTokens, float[] q,
                              int planeStride, int qRowStride, int qTokenBase, int[] pos, int posOff, float[] out,
                              int outOff, int outRowStride, float[] scratchOut, float[] scratchLse,
                              float[] mergeLse) {
        Objects.requireNonNull(dc, "dc");
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(causal, "causal");
        Objects.requireNonNull(nonCausal, "nonCausal");
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(pos, "pos");
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(scratchOut, "scratchOut");
        Objects.requireNonNull(scratchLse, "scratchLse");
        Objects.requireNonNull(mergeLse, "mergeLse");
        if (nTokens < 0 || qRowStride <= 0 || outRowStride <= 0 || qTokenBase < 0) {
            throw new IllegalArgumentException("dual-chunk attention needs nTokens >= 0 and positive strides, got "
                    + nTokens + " / " + qRowStride + " / " + outRowStride + " / " + qTokenBase);
        }
        final int nHead = spec.nHead();
        final int outDim = spec.outDim();
        final int kCount = keys.keys();
        final long neededOut = (long) Math.min(nTokens, dc.chunkLength()) * outDim;
        final long neededLse = (long) Math.min(nTokens, dc.chunkLength()) * nHead;
        if (scratchOut.length < neededOut || scratchLse.length < neededLse || mergeLse.length < neededLse) {
            throw new IllegalArgumentException("dual-chunk scratch holds " + scratchOut.length + " / "
                    + scratchLse.length + " / " + mergeLse.length + " values, need " + neededOut + " / " + neededLse);
        }
        if (nTokens == 0) {
            return;
        }
        if (planeStride < (long) nHead * spec.headDimK()) {
            throw new IllegalArgumentException("dual-chunk query plane stride " + planeStride + " is below the query"
                    + " width " + (long) nHead * spec.headDimK());
        }
        if (kCount < nTokens) {
            throw new IllegalArgumentException("dual-chunk attention over " + kCount + " keys cannot serve " + nTokens
                    + " query rows");
        }
        if (nTokens == 1) {
            decode(dc, spec, causal, nonCausal, keys, q, planeStride, qRowStride, qTokenBase, pos, posOff, out, outOff,
                    outRowStride, scratchOut, scratchLse, mergeLse);
        } else {
            prefill(dc, spec, causal, nonCausal, keys, nTokens, q, planeStride, qRowStride, qTokenBase, pos, posOff,
                    out, outOff, outRowStride, scratchOut, scratchLse, mergeLse);
        }
    }

    /** vLLM's prefill grouping: one chunk of query rows at a time, its own/succ/inter key ranges. */
    private static void prefill(DualChunkSpec dc, AttentionSpec spec, CpuAttentionCore.Params causal,
                                CpuAttentionCore.Params nonCausal, KeySpan keys, int nTokens, float[] q,
                                int planeStride, int qRowStride, int qTokenBase, int[] pos, int posOff, float[] out,
                                int outOff, int outRowStride, float[] scratchOut, float[] scratchLse,
                                float[] mergeLse) {
        final int nHead = spec.nHead();
        final int outDim = spec.outDim();
        final int chunkLen = dc.chunkLength();
        final int kCount = keys.keys();
        final int firstToken = kCount - nTokens;
        int begin = firstToken;
        while (begin < kCount) {
            final int prev = begin / chunkLen * chunkLen;
            final int end = Math.min(prev + chunkLen, kCount);
            final int qb = begin - firstToken;
            final int qe = end - firstToken;
            final int len = qe - qb;
            final int row = qTokenBase + qb;
            stage(keys, causal, 0, planeStride, q, qRowStride, row, pos, posOff + qb, len, prev, end,
                    outDim, nHead, scratchOut, scratchLse);
            merge(len, nHead, outDim, scratchOut, scratchLse, out, outOff + qb * outRowStride, outRowStride,
                    mergeLse, true);
            if (prev - chunkLen >= 0) {
                stage(keys, nonCausal, 1, planeStride, q, qRowStride, row, pos, posOff + qb, len, prev - chunkLen,
                        prev, outDim, nHead, scratchOut, scratchLse);
                merge(len, nHead, outDim, scratchOut, scratchLse, out, outOff + qb * outRowStride, outRowStride,
                        mergeLse, false);
                if (prev - 2 * chunkLen >= 0) {
                    stage(keys, nonCausal, 2, planeStride, q, qRowStride, row, pos, posOff + qb, len, 0,
                            prev - chunkLen, outDim, nHead, scratchOut, scratchLse);
                    merge(len, nHead, outDim, scratchOut, scratchLse, out, outOff + qb * outRowStride, outRowStride,
                            mergeLse, false);
                }
            }
            begin = end;
        }
    }

    /** vLLM's decode grouping: the one row's intra, succ and inter key ranges. */
    private static void decode(DualChunkSpec dc, AttentionSpec spec, CpuAttentionCore.Params causal,
                               CpuAttentionCore.Params nonCausal, KeySpan keys, float[] q, int planeStride,
                               int qRowStride, int qTokenBase, int[] pos, int posOff, float[] out, int outOff,
                               int outRowStride, float[] scratchOut, float[] scratchLse, float[] mergeLse) {
        final int nHead = spec.nHead();
        final int outDim = spec.outDim();
        final int chunkLen = dc.chunkLength();
        final int kCount = keys.keys();
        final int chunk = (kCount - 1) / chunkLen;
        final int intraStart = chunk * chunkLen;
        stage(keys, nonCausal, 0, planeStride, q, qRowStride, qTokenBase, pos, posOff, 1, intraStart, kCount,
                outDim, nHead, scratchOut, scratchLse);
        merge(1, nHead, outDim, scratchOut, scratchLse, out, outOff, outRowStride, mergeLse, true);
        if (chunk >= 1) {
            stage(keys, nonCausal, 1, planeStride, q, qRowStride, qTokenBase, pos, posOff, 1,
                    (chunk - 1) * chunkLen, chunk * chunkLen, outDim, nHead, scratchOut, scratchLse);
            merge(1, nHead, outDim, scratchOut, scratchLse, out, outOff, outRowStride, mergeLse, false);
            if (chunk >= 2) {
                stage(keys, nonCausal, 2, planeStride, q, qRowStride, qTokenBase, pos, posOff, 1, 0,
                        (chunk - 1) * chunkLen, outDim, nHead, scratchOut, scratchLse);
                merge(1, nHead, outDim, scratchOut, scratchLse, out, outOff, outRowStride, mergeLse, false);
            }
        }
    }

    private static void stage(KeySpan keys, CpuAttentionCore.Params p, int plane, int planeStride, float[] q,
                              int qRowStride, int row, int[] pos, int posOff, int n, int keyStart, int keyEnd,
                              int outDim, int nHead, float[] scratchOut, float[] scratchLse) {
        if (keyEnd <= keyStart || n <= 0) {
            java.util.Arrays.fill(scratchLse, 0, n * nHead, Float.NEGATIVE_INFINITY);
            java.util.Arrays.fill(scratchOut, 0, n * outDim, 0.0f);
            return;
        }
        keys.attend(p, n, q, plane * planeStride + row * qRowStride, qRowStride, pos, posOff, keyStart, keyEnd,
                scratchOut, 0, scratchLse, 0);
    }

    /**
     * Merges a stage's rows into the output: the first stage of a chunk copies its rows, later stages combine them
     * with the running log-sum-exp,
     * {@code out = (out e^(lse0 - m) + stage e^(lse - m)) / (e^(lse0 - m) + e^(lse - m))}.
     */
    private static void merge(int n, int nHead, int outDim, float[] stageOut, float[] stageLse, float[] out,
                              int outOff, int outRowStride, float[] mergeLse, boolean first) {
        for (int t = 0; t < n; t++) {
            final int sb = t * outDim;
            final int ob = outOff + t * outRowStride;
            final int lb = t * nHead;
            if (first) {
                System.arraycopy(stageOut, sb, out, ob, outDim);
                System.arraycopy(stageLse, lb, mergeLse, lb, nHead);
                continue;
            }
            for (int h = 0; h < nHead; h++) {
                final float a = mergeLse[lb + h];
                final float b = stageLse[lb + h];
                if (b == Float.NEGATIVE_INFINITY) {
                    continue;
                }
                if (a == Float.NEGATIVE_INFINITY) {
                    System.arraycopy(stageOut, sb + h * (outDim / nHead), out, ob + h * (outDim / nHead),
                            outDim / nHead);
                    mergeLse[lb + h] = b;
                    continue;
                }
                final float m = Math.max(a, b);
                final float wa = (float) Math.exp(a - m);
                final float wb = (float) Math.exp(b - m);
                final float denom = wa + wb;
                final int hb = h * (outDim / nHead);
                final int w = outDim / nHead;
                for (int d = 0; d < w; d++) {
                    out[ob + hb + d] = (out[ob + hb + d] * wa + stageOut[sb + hb + d] * wb) / denom;
                }
                mergeLse[lb + h] = m + (float) Math.log(denom);
            }
        }
    }
}

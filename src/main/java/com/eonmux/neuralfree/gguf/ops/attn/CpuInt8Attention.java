package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Fusion;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Kernels;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuInt8Quantizer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.KvInt8Rows;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU attention over an int8 KV cache, the fused twin of {@link CpuAttentionCore}: the same scores, softmax,
 * windows and position biases, but a key row is read in its stored format and scored with the exact int8 dot
 * product of {@link CpuInt8Kernels}, and a value row is dequantized inside the accumulation instead of into a float
 * buffer. The generic core dequantizes every K and V row into float before it scores, which this kernel fuses.
 *
 * <p>Every query head is quantized once per step to the symmetric int8 format of {@link CpuInt8Quantizer}
 * ({@code max(absmax / 127, 1e-6)} per (token, head), round half to even), so the score of a key row is
 * {@code int_dot(query codes, key codes) * queryScale * keyScale} scaled by {@link CpuAttentionCore.Params#scale()}
 * and capped by the softcap like the generic core. The value accumulation adds
 * {@code p * (code * valueScale)} per value, the same per-element product the generic core computes from the
 * dequantized row, so a call differs from {@link CpuAttentionCore} only by the query quantization: the relative
 * score error is bounded by the int8 code spacing of the query head, and the outputs agree to the tolerance the
 * fused-path tests state. A query head whose values are exact multiples of its scale is scored exactly.</p>
 *
 * <p>Layouts, the online softmax, the sink logits, key splits, GQA and paged rows follow {@link CpuAttentionCore}
 * exactly; {@code rowOf} maps a logical key index (a cache row, or a slot of the paged slot space) to the physical
 * cache row, so the contiguous call passes the identity. Learned static sink <em>tokens</em>
 * ({@link CpuAttentionCore.Params#sinkTokens()}) are refused by name: their rows are parameters, not int8 cache
 * rows.</p>
 */
public final class CpuInt8Attention {

    /** Keys scored per block; bounds the score scratch to {@code HEAD_CHUNK * KEY_BLOCK} floats per thread. */
    private static final int KEY_BLOCK = CpuAttentionCore.KEY_BLOCK;
    /** Query heads of one KV head processed together; {@code NF_FD_MAX_G}. */
    private static final int HEAD_CHUNK = LlmKernelConstants.FD_MAX_G;

    private static final CpuInt8Kernels KERNELS = CpuInt8Fusion.kernels();

    private static final IntUnaryOperator IDENTITY = row -> row;

    private CpuInt8Attention() {
    }

    /**
     * Attention of {@code nTokens} query rows against the cache rows {@code [rowStart, rowEnd)} of {@code kv}: row
     * {@code t} at {@code qOff + t * qRowStride}, position {@code qPos[qPosOff + t]}, output at
     * {@code outOff + t * outRowStride}. Parallel over (token, KV head, head chunk) units and, when those are fewer
     * than the worker threads, over key splits (see {@link CpuAttentionCore}).
     *
     * @param rowPos position stored in a physical cache row, -1 for an empty row
     * @throws IllegalArgumentException      when a shape or range is inconsistent
     * @throws UnsupportedOperationException when the parameters carry learned static sink tokens
     */
    public static void attendRows(CpuAttentionCore.Params p, int nTokens, float[] q, int qOff, int qRowStride,
                                  int[] qPos, int qPosOff, KvInt8Rows kv, IntUnaryOperator rowPos, int rowStart,
                                  int rowEnd, float[] out, int outOff, int outRowStride) {
        attendRows(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, kv, rowPos, IDENTITY, rowStart, rowEnd, out,
                outOff, outRowStride);
    }

    /**
     * {@link #attendRows(CpuAttentionCore.Params, int, float[], int, int, int[], int, KvInt8Rows, IntUnaryOperator,
     * int, int, float[], int, int)} over the rows {@code rowOf} maps each logical key index to.
     *
     * @param rowOf the physical cache row of a logical key index; the contiguous call passes the identity, a paged
     *              call maps slots through its page table
     */
    public static void attendRows(CpuAttentionCore.Params p, int nTokens, float[] q, int qOff, int qRowStride,
                                  int[] qPos, int qPosOff, KvInt8Rows kv, IntUnaryOperator rowPos,
                                  IntUnaryOperator rowOf, int rowStart, int rowEnd, float[] out, int outOff,
                                  int outRowStride) {
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, kv, rowPos, rowOf, rowStart, rowEnd, out, outOff,
                outRowStride, 0);
    }

    /**
     * {@link #attendRows(CpuAttentionCore.Params, int, float[], int, int, int[], int, KvInt8Rows, IntUnaryOperator,
     * int, int, float[], int, int)} with an explicit key-split count instead of the one derived from the worker
     * pool; {@code splits} is capped at one split per key block.
     *
     * @throws IllegalArgumentException when {@code splits < 1}
     */
    static void attendRowsSplit(CpuAttentionCore.Params p, int nTokens, float[] q, int qOff, int qRowStride,
                                int[] qPos, int qPosOff, KvInt8Rows kv, IntUnaryOperator rowPos, int rowStart,
                                int rowEnd, float[] out, int outOff, int outRowStride, int splits) {
        if (splits < 1) {
            throw new IllegalArgumentException("key splits must be positive, got " + splits);
        }
        attendMany(p, nTokens, q, qOff, qRowStride, qPos, qPosOff, kv, rowPos, IDENTITY, rowStart, rowEnd, out,
                outOff, outRowStride, splits);
    }

    // ------------------------------------------------------------------ validation

    private static void checkShape(CpuAttentionCore.Params p, KvInt8Rows kv, int rowStart, int rowEnd) {
        Objects.requireNonNull(p, "params");
        Objects.requireNonNull(kv, "kv");
        if (kv.nHeadKv() <= 0 || kv.headDimK() <= 0 || kv.headDimV() <= 0) {
            throw new IllegalArgumentException("int8 KV view needs positive heads and head dims, got "
                    + kv.nHeadKv() + " x (" + kv.headDimK() + ", " + kv.headDimV() + ")");
        }
        if (p.nHead() % kv.nHeadKv() != 0) {
            throw new IllegalArgumentException("n_head " + p.nHead() + " is not a multiple of n_head_kv "
                    + kv.nHeadKv());
        }
        if (rowStart < 0 || rowEnd < rowStart) {
            throw new IllegalArgumentException("invalid key row range [" + rowStart + ", " + rowEnd + ")");
        }
        if (p.sinkTokens() > 0) {
            throw new UnsupportedOperationException("the fused int8 attention reads stored int8 rows; the "
                    + p.sinkTokens() + " learned static sink tokens are not cache rows");
        }
        if (p.kScale() != null || p.vScale() != null) {
            throw new UnsupportedOperationException("the fused int8 attention scales every key and value row by its"
                    + " own int8 scale; the per-KV-head checkpoint scales of an FP8 cache do not apply");
        }
    }

    /** Checks that the query rows {@code qFirst, qLast} and output rows {@code outFirst, outLast} fit the arrays. */
    private static void checkRows(CpuAttentionCore.Params p, KvInt8Rows kv, float[] q, long qFirst, long qLast,
                                  float[] out, long outFirst, long outLast) {
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(out, "out");
        long qLen = (long) p.nHead() * kv.headDimK();
        long outLen = (long) p.nHead() * kv.headDimV();
        if (qFirst < 0 || qLast + qLen > q.length || outFirst < 0 || outLast + outLen > out.length) {
            throw new IllegalArgumentException("query or output rows out of bounds: q[" + qFirst + ".." + qLast
                    + " + " + qLen + "] of " + q.length + ", out[" + outFirst + ".." + outLast + " + " + outLen
                    + "] of " + out.length);
        }
    }

    // ------------------------------------------------------------------ implementation

    /** Reusable work arrays of one thread, sized for one key block of one head chunk; grows on demand. Not thread-safe. */
    private static final class Scratch {
        private byte[] qCodes = new byte[0];
        private float[] qScale = new float[0];
        private byte[] kCodes = new byte[0];
        private byte[] vCodes = new byte[0];
        private float[] scores = new float[0];
        private float[] acc = new float[0];
        private float[] max = new float[0];
        private float[] sum = new float[0];

        void ensure(int dK, int dV) {
            if (scores.length < HEAD_CHUNK * KEY_BLOCK) {
                scores = new float[HEAD_CHUNK * KEY_BLOCK];
                max = new float[HEAD_CHUNK];
                sum = new float[HEAD_CHUNK];
            }
            if (qCodes.length < HEAD_CHUNK * dK) {
                qCodes = new byte[HEAD_CHUNK * dK];
            }
            if (qScale.length < HEAD_CHUNK) {
                qScale = new float[HEAD_CHUNK];
            }
            if (kCodes.length < dK) {
                kCodes = new byte[dK];
            }
            if (vCodes.length < dV) {
                vCodes = new byte[dV];
            }
            if (acc.length < HEAD_CHUNK * dV) {
                acc = new float[HEAD_CHUNK * dV];
            }
        }
    }

    /** {@code forcedSplits} 0 derives the key splits from the worker pool. */
    private static void attendMany(CpuAttentionCore.Params p, int nTokens, float[] q, int qOff, int qRowStride,
                                   int[] qPos, int qPosOff, KvInt8Rows kv, IntUnaryOperator rowPos,
                                   IntUnaryOperator rowOf, int rowStart, int rowEnd, float[] out, int outOff,
                                   int outRowStride, int forcedSplits) {
        Objects.requireNonNull(rowPos, "rowPos");
        Objects.requireNonNull(rowOf, "rowOf");
        checkShape(p, kv, rowStart, rowEnd);
        if (nTokens < 0 || qRowStride < 0 || outRowStride < 0 || qPosOff < 0
                || qPosOff + (long) nTokens > qPos.length) {
            throw new IllegalArgumentException("invalid query rows: n=" + nTokens + ", strides " + qRowStride + "/"
                    + outRowStride + ", positions " + qPosOff + ".." + (qPosOff + (long) nTokens) + " of "
                    + qPos.length);
        }
        if (nTokens == 0) {
            return;
        }
        long last = nTokens - 1L;
        checkRows(p, kv, q, qOff, qOff + last * qRowStride, out, outOff, outOff + last * outRowStride);
        final int nHeadKv = kv.nHeadKv();
        final int g0 = p.nHead() / nHeadKv;
        final int chunks = (g0 + HEAD_CHUNK - 1) / HEAD_CHUNK;
        final int unitsPerToken = nHeadKv * chunks;
        long units = (long) nTokens * unitsPerToken;
        if (units > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(nTokens + " query rows x " + unitsPerToken + " head chunks exceed "
                    + Integer.MAX_VALUE + " work units");
        }
        int keys = rowEnd - rowStart;
        int splits = forcedSplits > 0 ? Math.min(forcedSplits, Math.max(1, (keys + KEY_BLOCK - 1) / KEY_BLOCK))
                : CpuAttentionCore.splitCount(units, keys, CpuParallel.parallelism());
        if (splits > 1) {
            attendSplit(p, (int) units, unitsPerToken, chunks, g0, splits, q, qOff, qRowStride, qPos, qPosOff, kv,
                    rowPos, rowOf, rowStart, rowEnd, out, outOff, outRowStride);
            return;
        }
        CpuParallel.parallelFor((int) units, 1, (from, to) -> {
            Scratch scratch = new Scratch();
            for (int u = from; u < to; u++) {
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                attendChunk(p, q, qOff + t * qRowStride, qPos[qPosOff + t], qPosOff + t, kv, rowPos, rowOf, rowStart,
                        rowEnd, out, outOff + t * outRowStride, scratch, kvh * g0 + g, Math.min(HEAD_CHUNK, g0 - g),
                        kvh);
            }
        });
    }

    /** Query heads {@code [h0, h0 + gc)} (all reading KV head {@code kvh}) of one query row. */
    private static void attendChunk(CpuAttentionCore.Params p, float[] q, int qOff, int qPos, int qRow,
                                    KvInt8Rows kv, IntUnaryOperator rowPos, IntUnaryOperator rowOf, int rowStart,
                                    int rowEnd, float[] out, int outOff, Scratch s, int h0, int gc, int kvh) {
        accumulate(p, q, qOff, qPos, qRow, kv, rowPos, rowOf, rowStart, rowEnd, s, h0, gc, kvh);
        finish(p, s.max, 0, s.sum, 0, s.acc, 0, kv.headDimV(), h0, gc, out, outOff);
    }

    /**
     * The unnormalized online-softmax state {@code (m, l, o)} of heads {@code [h0, h0 + gc)} over keys
     * {@code [rowStart, rowEnd)}, in {@code s.max}, {@code s.sum} and {@code s.acc}.
     */
    private static void accumulate(CpuAttentionCore.Params p, float[] q, int qOff, int qPos, int qRow,
                                   KvInt8Rows kv, IntUnaryOperator rowPos, IntUnaryOperator rowOf, int rowStart,
                                   int rowEnd, Scratch s, int h0, int gc, int kvh) {
        final int dK = kv.headDimK();
        final int dV = kv.headDimV();
        s.ensure(dK, dV);
        quantizeQuery(q, qOff, dK, s, h0, gc);
        Arrays.fill(s.max, 0, gc, Float.NEGATIVE_INFINITY);
        Arrays.fill(s.sum, 0, gc, 0.0f);
        Arrays.fill(s.acc, 0, gc * dV, 0.0f);
        for (int b = rowStart; b < rowEnd; b += KEY_BLOCK) {
            final int nb = Math.min(KEY_BLOCK, rowEnd - b);
            scoreBlock(p, qPos, qRow, kv, rowPos, rowOf, b, nb, s, h0, gc, kvh);
            rescaleBlock(s, nb, gc, dV);
            addValues(kv, rowOf, b, nb, s, gc, kvh, dV);
        }
    }

    /** Quantizes query heads {@code [h0, h0 + gc)} into {@code s.qCodes} with one scale per head. */
    private static void quantizeQuery(float[] q, int qOff, int dK, Scratch s, int h0, int gc) {
        for (int g = 0; g < gc; g++) {
            final int qb = qOff + (h0 + g) * dK;
            final float scale = CpuInt8Quantizer.scale(q, qb, dK);
            s.qScale[g] = scale;
            CpuInt8Quantizer.quantize(q, qb, dK, s.qCodes, g * dK, scale);
        }
    }

    /**
     * Scores {@code s.scores[g * KEY_BLOCK + r]} of heads {@code [h0, h0 + gc)} against keys {@code [b, b + nb)};
     * keys outside the query's window score -INF.
     */
    private static void scoreBlock(CpuAttentionCore.Params p, int qPos, int qRow, KvInt8Rows kv,
                                   IntUnaryOperator rowPos, IntUnaryOperator rowOf, int b, int nb, Scratch s, int h0,
                                   int gc, int kvh) {
        final int dK = kv.headDimK();
        final float[] sc = s.scores;
        final KvWindow window = p.window();
        for (int r = 0; r < nb; r++) {
            final int phys = rowOf.applyAsInt(b + r);
            final int pk = rowPos.applyAsInt(phys);
            if (!window.visible(qPos, pk)) {
                for (int g = 0; g < gc; g++) {
                    sc[g * KEY_BLOCK + r] = Float.NEGATIVE_INFINITY;
                }
                continue;
            }
            kv.readKCodes(phys, kvh, s.kCodes, 0);
            final float keyScale = kv.kScale(phys, kvh);
            for (int g = 0; g < gc; g++) {
                final int h = h0 + g;
                final int dot = KERNELS.dot(s.qCodes, g * dK, s.kCodes, 0, dK);
                sc[g * KEY_BLOCK + r] = CpuAttentionCore.score(dot * (s.qScale[g] * keyScale), p.scale(),
                        p.softcap()) + p.bias(h, qPos, pk, qRow);
            }
        }
    }

    /**
     * Online-softmax step of one key block per head: {@code m' = max(m, block max)}, scores become
     * {@code p = e^(s - m')}, {@code l = l * e^(m - m') + sum p} and the accumulator is scaled by {@code e^(m - m')}.
     */
    private static void rescaleBlock(Scratch s, int nb, int gc, int dV) {
        final float[] sc = s.scores;
        final float[] acc = s.acc;
        final float[] mx = s.max;
        final float[] sum = s.sum;
        for (int g = 0; g < gc; g++) {
            final int sb = g * KEY_BLOCK;
            float m = mx[g];
            for (int r = 0; r < nb; r++) {
                m = Math.max(m, sc[sb + r]);
            }
            float l = 0.0f;
            for (int r = 0; r < nb; r++) {
                final float e = CpuAttentionCore.rescale(sc[sb + r], m);
                sc[sb + r] = e;
                l += e;
            }
            final float alpha = CpuAttentionCore.rescale(mx[g], m);
            if (alpha != 1.0f) {
                final int ab = g * dV;
                for (int d = 0; d < dV; d++) {
                    acc[ab + d] *= alpha;
                }
            }
            sum[g] = sum[g] * alpha + l;
            mx[g] = m;
        }
    }

    /** {@code o += p V} over keys {@code [b, b + nb)}; a value row no head weights is not read. */
    private static void addValues(KvInt8Rows kv, IntUnaryOperator rowOf, int b, int nb, Scratch s, int gc, int kvh,
                                  int dV) {
        final float[] sc = s.scores;
        final float[] acc = s.acc;
        for (int r = 0; r < nb; r++) {
            boolean used = false;
            for (int g = 0; g < gc && !used; g++) {
                used = sc[g * KEY_BLOCK + r] != 0.0f;
            }
            if (!used) {
                continue;
            }
            final int phys = rowOf.applyAsInt(b + r);
            kv.readVCodes(phys, kvh, s.vCodes, 0);
            final float valueScale = kv.vScale(phys, kvh);
            for (int g = 0; g < gc; g++) {
                final float pr = sc[g * KEY_BLOCK + r];
                if (pr != 0.0f) {
                    KERNELS.accumulateScaled(s.vCodes, 0, valueScale, pr, acc, g * dV, dV);
                }
            }
        }
    }

    /**
     * Sink and normalization of heads {@code [h0, h0 + gc)} from their state {@code m = mx[mOff + g]},
     * {@code l = sum[mOff + g]}, {@code o = acc[accOff + g * dV + d]}: {@code m' = max(m, sink)},
     * {@code out = o e^(m - m') / (l e^(m - m') + e^(sink - m'))} with a sink logit, else {@code out = o / l}; zeros
     * when the denominator is 0. The same arithmetic as {@link CpuAttentionCore}.
     */
    private static void finish(CpuAttentionCore.Params p, float[] mx, int mOff, float[] sum, int lOff, float[] acc,
                               int accOff, int dV, int h0, int gc, float[] out, int outOff) {
        for (int g = 0; g < gc; g++) {
            final int h = h0 + g;
            final float m = mx[mOff + g];
            final float l = sum[lOff + g];
            float factor = 1.0f;
            float denom = l;
            if (p.hasSinks()) {
                final float sink = p.sink(h);
                final float m2 = Math.max(m, sink);
                factor = CpuAttentionCore.rescale(m, m2);
                denom = l * factor + (float) Math.exp(sink - m2);
            }
            final int ob = outOff + h * dV;
            final int ab = accOff + g * dV;
            for (int d = 0; d < dV; d++) {
                out[ob + d] = denom > 0.0f ? acc[ab + d] * factor / denom : 0.0f;
            }
        }
    }

    /**
     * Split-K form of the multi-row call: task {@code (unit, split)} accumulates the unit's heads over the split's
     * keys (whole {@link #KEY_BLOCK}s), then each unit merges its splits in split order and finishes.
     */
    private static void attendSplit(CpuAttentionCore.Params p, int units, int unitsPerToken, int chunks, int g0,
                                    int splits, float[] q, int qOff, int qRowStride, int[] qPos, int qPosOff,
                                    KvInt8Rows kv, IntUnaryOperator rowPos, IntUnaryOperator rowOf, int rowStart,
                                    int rowEnd, float[] out, int outOff, int outRowStride) {
        final int dV = kv.headDimV();
        final int keys = rowEnd - rowStart;
        final int blocks = (keys + KEY_BLOCK - 1) / KEY_BLOCK;
        final int splitLen = ((blocks + splits - 1) / splits) * KEY_BLOCK;
        final int nSplit = (keys + splitLen - 1) / splitLen;
        long states = (long) units * nSplit * HEAD_CHUNK;
        if (states * dV > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(units + " units x " + nSplit + " key splits of " + HEAD_CHUNK + " x "
                    + dV + " partial values exceed one array");
        }
        final float[] pm = new float[(int) states];
        final float[] pl = new float[(int) states];
        final float[] po = new float[(int) (states * dV)];
        CpuParallel.parallelFor(units * nSplit, 1, (from, to) -> {
            Scratch scratch = new Scratch();
            for (int i = from; i < to; i++) {
                final int u = i / nSplit;
                final int sp = i - u * nSplit;
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                final int gc = Math.min(HEAD_CHUNK, g0 - g);
                final int k0 = rowStart + sp * splitLen;
                accumulate(p, q, qOff + t * qRowStride, qPos[qPosOff + t], qPosOff + t, kv, rowPos, rowOf, k0,
                        Math.min(rowEnd, k0 + splitLen), scratch, kvh * g0 + g, gc, kvh);
                final int base = i * HEAD_CHUNK;
                System.arraycopy(scratch.max, 0, pm, base, gc);
                System.arraycopy(scratch.sum, 0, pl, base, gc);
                System.arraycopy(scratch.acc, 0, po, base * dV, gc * dV);
            }
        });
        CpuParallel.parallelFor(units, 1, (from, to) -> {
            final float[] m = new float[HEAD_CHUNK];
            final float[] l = new float[HEAD_CHUNK];
            final float[] o = new float[HEAD_CHUNK * dV];
            for (int u = from; u < to; u++) {
                final int t = u / unitsPerToken;
                final int rest = u - t * unitsPerToken;
                final int kvh = rest / chunks;
                final int g = (rest - kvh * chunks) * HEAD_CHUNK;
                final int gc = Math.min(HEAD_CHUNK, g0 - g);
                combine(pm, pl, po, u * nSplit, nSplit, gc, dV, m, l, o);
                finish(p, m, 0, l, 0, o, 0, dV, kvh * g0 + g, gc, out, outOff + t * outRowStride);
            }
        });
    }

    /** Merges the states of splits {@code [first, first + nSplit)} into {@code m}, {@code l}, {@code o}. */
    private static void combine(float[] pm, float[] pl, float[] po, int first, int nSplit, int gc, int dV,
                                float[] m, float[] l, float[] o) {
        for (int g = 0; g < gc; g++) {
            float mm = Float.NEGATIVE_INFINITY;
            for (int sp = 0; sp < nSplit; sp++) {
                mm = Math.max(mm, pm[(first + sp) * HEAD_CHUNK + g]);
            }
            float ll = 0.0f;
            final int ob = g * dV;
            Arrays.fill(o, ob, ob + dV, 0.0f);
            for (int sp = 0; sp < nSplit; sp++) {
                final int st = (first + sp) * HEAD_CHUNK + g;
                final float w = CpuAttentionCore.rescale(pm[st], mm);
                if (w == 0.0f) {
                    continue;
                }
                ll += pl[st] * w;
                final int pb = st * dV;
                for (int d = 0; d < dV; d++) {
                    o[ob + d] += po[pb + d] * w;
                }
            }
            m[g] = mm;
            l[g] = ll;
        }
    }
}

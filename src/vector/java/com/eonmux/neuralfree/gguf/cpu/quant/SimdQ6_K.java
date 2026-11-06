package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QL;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q6_K weights with Q8_K activations, {@link KQuantVecDot#Q6_K} with the unpack in byte
 * lanes and the scaled 16-element sub-block dots in int16 lanes: each 16-byte slice of {@code ql} loads with
 * its 16 {@code qh} bytes, quarter {@code q} of slice {@code t} in half {@code c} decoding to sub-block
 * {@code 8c + 2q + t} ({@code (ql quarter | qh pair << 4) - 32}). A sub-block's decodes widen to short lanes,
 * multiply the pre-widened activation lanes in {@code vpmullw}, fold pairwise into int lanes and are scaled by
 * their signed 8-bit scale there, so one int accumulator carries the 16 sub-blocks of a super-block and the
 * horizontal sum runs once per token instead of once per sub-block. The float combination
 * {@code sumf += (d_x * d_y) * isum} follows {@code ggml_vec_dot_q6_K_q8_K_generic}, so the kernel returns what
 * {@code KQuantVecDot.Q6_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) serves eight tokens from one unpack of the super-block.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ6_K implements CpuVecDot {

    /** The Q6_K vec-dot. */
    public static final SimdQ6_K INSTANCE = new SimdQ6_K();

    /** Tokens one pass of the batched kernel decodes a weight super-block for. */
    private static final int TOKENS = 8;

    /** The two low bits of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 32} in every byte, for {@link Q8VecDot#subtractBias}. */
    private static final long BIAS_32 = 0x6060606060606060L;

    private SimdQ6_K() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x, SimdI8.activationsShort(x));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x, SimdI8.activationsShort(x));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q6_K);
        x.requireCompatible(w, GGMLType.Q8_K);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                        float[] y, int yOff, int yStride) {
        dotTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
    }

    @Override
    public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                         float[] y, int yOff, int yStride) {
        if (nTokens == 0) {
            return;
        }
        checkTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
        final short[][] ys = SimdI8.activationsAllShort(x, nTokens);
        final byte[] row = CpuVecDotRows.buffer(w);
        final float[] acc = SimdI8.tileAccumulators(TOKENS);
        final int nb = x[0].n / QK_K;
        for (int r = 0; r < rowCount; r++) {
            CpuVecDotRows.copyRow(w, rowStart + r, row);
            dotRowTokens(row, x, nTokens, ys, nb, y, yOff + r, yStride, acc);
        }
    }

    /**
     * One weight row against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time: the super-block
     * unpacks once per tile and every token of the tile dots its pre-widened lanes against it.
     */
    private static void dotRowTokens(byte[] b, QuantizedActivations[] x, int nTokens, short[][] ys, int nb,
                                     float[] y, int y0, int yStride, float[] sum) {
        final byte[] q = Q8VecDot.scratch().quants;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = 0;
            for (int i = 0; i < nb; i++, o += Q6_K_BYTES) {
                unpack(b, o, q);
                final int sc = o + Q6_K_SCALES;
                final int xo = i * QK_K;
                final float d = fp16At(b, o + Q6_K_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    IntVector acc = SimdI8.zero(SimdI8.I8);
                    for (int g = 0; g < QK_K / 16; g++) {
                        final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                        final int yo = xo + 16 * g;
                        acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, a, yo)),
                                SimdI8.zero(SimdI8.I8)).mul(b[sc + g]));
                    }
                    sum[ti] += d * x[t0 + ti].d[i] * SimdI8.sum(acc);
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        final byte[] q = Q8VecDot.scratch().quants;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q6_K_BYTES) {
            unpack(b, off, q);
            final int sc = off + Q6_K_SCALES;
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 16 * g;
                acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, ys, yo)),
                        SimdI8.zero(SimdI8.I8)).mul(b[sc + g]));
            }
            sumf += fp16At(b, off + Q6_K_D) * yd[i] * SimdI8.sum(acc);
        }
        return sumf;
    }

    /**
     * Unpacks the 256 signed 6-bit quants of one super-block into {@code q}: quarter {@code q} of slice
     * {@code t} in half {@code c} is sub-block {@code 8c + 2q + t}, its value {@code (ql nibble | qh pair << 4)
     * - 32}. Eight quants at a time with the 64-bit SWAR of {@link Q8VecDot#subtractBias}: the byte-lane form
     * of the same unpack measured slower and allocated, so the SWAR form stays.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int ql = off + Q6_K_QL;
        final int qh = off + Q6_K_QH;
        for (int c = 0; c < QK_K / 128; c++) {
            final int a = 128 * c;
            for (int t = 0; t < 4; t++) {
                final long lo = longAt(b, ql + 64 * c + 8 * t);
                final long hi = longAt(b, ql + 64 * c + 32 + 8 * t);
                final long h = longAt(b, qh + 32 * c + 8 * t);
                Q8VecDot.putLong(q, a + 8 * t,
                        Q8VecDot.subtractBias((lo & Q8VecDot.LOW_NIBBLES) | ((h & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 32 + 8 * t,
                        Q8VecDot.subtractBias((hi & Q8VecDot.LOW_NIBBLES) | (((h >>> 2) & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 64 + 8 * t,
                        Q8VecDot.subtractBias(((lo >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 4) & LOW_PAIRS) << 4),
                                BIAS_32));
                Q8VecDot.putLong(q, a + 96 + 8 * t,
                        Q8VecDot.subtractBias(((hi >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 6) & LOW_PAIRS) << 4),
                                BIAS_32));
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ6_K{Q6_K x Q8_K}";
    }
}

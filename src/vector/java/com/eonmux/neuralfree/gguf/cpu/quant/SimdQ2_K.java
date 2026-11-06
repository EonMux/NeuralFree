package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q2_K weights with Q8_K activations, {@link KQuantVecDot#Q2_K} with the two-bit codes
 * unpacked into the thread's scratch row and the 16-element group dots in int16 lanes: each group's quants
 * widen to short lanes, multiply the pre-widened activation lanes in {@code vpmullw}, fold pairwise into int
 * lanes and are scaled by the group's 4-bit scale there, so one int accumulator carries the sixteen groups of a
 * super-block. The mins sum over the 4-bit group mins runs in the same short lanes, one pass over the sixteen
 * {@code bsums} of the super-block. The float combination {@code dall * isum - dmin * summs} follows
 * {@code ggml_vec_dot_q2_K_q8_K_generic}, so the kernel returns what {@link KQuantVecDot#Q2_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) serves eight tokens from one unpack of the super-block.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ2_K implements CpuVecDot {

    /** The Q2_K vec-dot. */
    public static final SimdQ2_K INSTANCE = new SimdQ2_K();

    /** Tokens one pass of the batched kernel decodes a weight super-block for. */
    private static final int TOKENS = 8;

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    /** Bits 0-1 of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;

    private SimdQ2_K() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q2_K);
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

    /** One weight row against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time. */
    private static void dotRowTokens(byte[] b, QuantizedActivations[] x, int nTokens, short[][] ys, int nb,
                                     float[] y, int y0, int yStride, float[] sum) {
        final byte[] q = Q8VecDot.scratch().quants;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = 0;
            for (int i = 0; i < nb; i++, o += Q2_K_BYTES) {
                unpack(b, o, q);
                final int xo = i * QK_K;
                final float d = fp16At(b, o + Q2_K_D);
                final float dminBlock = fp16At(b, o + Q2_K_DMIN);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    IntVector acc = SimdI8.zero(SimdI8.I8);
                    for (int g = 0; g < QK_K / 16; g++) {
                        final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                        final int yo = xo + 16 * g;
                        final int sc = b[o + Q2_K_SCALES + g] & 0xFF;
                        acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, a, yo)),
                                SimdI8.zero(SimdI8.I8)).mul(sc & 0x0F));
                    }
                    final QuantizedActivations tok = x[t0 + ti];
                    final float dall = tok.d[i] * d;
                    final float dmin = tok.d[i] * dminBlock;
                    sum[ti] += dall * SimdI8.sum(acc) - dmin * minsSum(tok.bsums, i, b, o);
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final byte[] q = Q8VecDot.scratch().quants;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q2_K_BYTES) {
            unpack(b, off, q);
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 16 * g;
                final int sc = b[off + Q2_K_SCALES + g] & 0xFF;
                acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, ys, yo)),
                        SimdI8.zero(SimdI8.I8)).mul(sc & 0x0F));
            }
            final int summs = minsSum(bsums, i, b, off);
            final float dall = yd[i] * fp16At(b, off + Q2_K_D);
            final float dmin = yd[i] * fp16At(b, off + Q2_K_DMIN);
            sumf += dall * SimdI8.sum(acc) - dmin * summs;
        }
        return sumf;
    }

    /**
     * Unpacks the 256 two-bit codes of one super-block into {@code q} in element order: element
     * {@code e = 128k + 32s + 8t + l} is {@code (qs[32k + 8t + l] >> 2s) & 3}.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        for (int k = 0; k < QK_K / 128; k++) {
            final int q2 = off + Q2_K_QS + 32 * k;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q2 + 8 * t);
                for (int s = 0; s < 4; s++) {
                    Q8VecDot.putLong(q, 128 * k + 32 * s + 8 * t, (v >>> (2 * s)) & LOW_PAIRS);
                }
            }
        }
    }

    /**
     * {@code sum_g bsums[g] * (scales[g] >>> 4)} over the 16 groups of super-block {@code i}, in one pass: the
     * sixteen group sums widen to short lanes, the sixteen 4-bit mins are the high nibbles of the scale bytes,
     * and every product fits a short (a group sum of sixteen int8 quants is at most {@code 16 * 127} and a min
     * at most 15), so the paired fold into int lanes is exact.
     */
    private static int minsSum(short[] bsums, int i, byte[] b, int off) {
        final ShortVector sums = ShortVector.fromArray(SimdI8.S16, bsums, i * BSUMS_PER_BLOCK);
        final ShortVector mins = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q2_K_SCALES)
                .lanewise(VectorOperators.LSHR, 4)
                .and((byte) 0x0F)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        return SimdI8.sum(SimdI8.pairSum(sums.mul(mins), SimdI8.zero(SimdI8.I8)));
    }

    @Override
    public String toString() {
        return "SimdQ2_K{Q2_K x Q8_K}";
    }
}

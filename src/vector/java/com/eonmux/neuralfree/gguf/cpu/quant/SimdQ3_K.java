package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.KQuantBlockDecoders;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_HMASK;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q3_K weights with Q8_K activations, {@link KQuantVecDot#Q3_K} with the three-bit codes
 * unpacked into the thread's scratch row and the 16-element group dots in int16 lanes: the low two bits come
 * from the two-bit slices of each 8-byte {@code qs} group and the third bit from the matching {@code hmask}
 * byte, giving {@code (low | bit << 2) - 4}; each group's quants widen to short lanes, multiply the pre-widened
 * activation lanes, fold pairwise into int lanes and are scaled by the group's signed 6-bit scale there, so one
 * int accumulator carries the sixteen groups of a super-block. The scale unpack and the float combination
 * follow {@code ggml_vec_dot_q3_K_q8_K_generic}, so the kernel returns what {@link KQuantVecDot#Q3_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) serves eight tokens from one unpack of the super-block.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ3_K implements CpuVecDot {

    /** The Q3_K vec-dot. */
    public static final SimdQ3_K INSTANCE = new SimdQ3_K();

    /** Tokens one pass of the batched kernel decodes a weight super-block for. */
    private static final int TOKENS = 8;

    /** {@code 0x01} in every byte. */
    private static final long BYTE_ONES = 0x0101010101010101L;
    /** Bits 0-1 of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 4} in every byte: maps 3-bit values {@code 0..7} to {@code -4..3}. */
    private static final long BIAS_4 = 0x7C7C7C7C7C7C7C7CL;

    private SimdQ3_K() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q3_K);
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
        final byte[] scales = Q8VecDot.scratch().scales;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = 0;
            for (int i = 0; i < nb; i++, o += Q3_K_BYTES) {
                unpack(b, o, q);
                KQuantBlockDecoders.q3kScales(b, o + Q3_K_SCALES, scales);
                final int xo = i * QK_K;
                final float d = fp16At(b, o + Q3_K_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    IntVector acc = SimdI8.zero(SimdI8.I8);
                    for (int g = 0; g < QK_K / 16; g++) {
                        final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                        final int yo = xo + 16 * g;
                        acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, a, yo)),
                                SimdI8.zero(SimdI8.I8)).mul(scales[g] - 32));
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
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] scales = Q8VecDot.scratch().scales;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q3_K_BYTES) {
            unpack(b, off, q);
            KQuantBlockDecoders.q3kScales(b, off + Q3_K_SCALES, scales);
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 16 * g;
                acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, ys, yo)),
                        SimdI8.zero(SimdI8.I8)).mul(scales[g] - 32));
            }
            final float d = fp16At(b, off + Q3_K_D) * yd[i];
            sumf += d * SimdI8.sum(acc);
        }
        return sumf;
    }

    /**
     * Unpacks the 256 signed 3-bit quants of one super-block into {@code q}: element {@code e = 128c + 32s +
     * 8t + l} is {@code ((qs[32c + 8t + l] >> 2s) & 3) | ((hmask[8t + l] >> (4c + s)) & 1) << 2}, minus 4.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int hm = off + Q3_K_HMASK;
        for (int c = 0; c < QK_K / 128; c++) {
            final int q3 = off + Q3_K_QS + 32 * c;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q3 + 8 * t);
                final long h = longAt(b, hm + 8 * t);
                for (int s = 0; s < 4; s++) {
                    final long low = (v >>> (2 * s)) & LOW_PAIRS;
                    final long high = ((h >>> (4 * c + s)) & BYTE_ONES) << 2;
                    Q8VecDot.putLong(q, 128 * c + 32 * s + 8 * t, Q8VecDot.subtractBias(low | high, BIAS_4));
                }
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ3_K{Q3_K x Q8_K}";
    }
}

package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q4_0x8;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q4_0 weights in the interleaved {@link Q4_0x8} layout with Q8_0 activations,
 * {@link Q4_0x8VecDot} with the nibble decode in byte lanes: each 8-byte slice of a block loads as one byte
 * vector, its low nibbles decode with {@code SHL 4, ASHR 4} and its high nibbles with {@code AND 0xF0,
 * ASHR 4} (both recover the signed quant {@code nibble - 8} of the XOR-0x88 storage), and the four decoded
 * 8-element halves widen to int lanes against the activation quants at the element offsets the layout pairs.
 * The float combination {@code sumi * d_x * d_y} per block follows {@code LegacyVecDot.Q4_0}, so the kernel
 * returns exactly what the plain kernels return. Immutable, stateless and thread-safe.
 */
public final class SimdQ4_0x8 implements CpuVecDot {

    /** The Q4_0x8 vec-dot. */
    public static final SimdQ4_0x8 INSTANCE = new SimdQ4_0x8();

    /** Tokens one pass of the batched row loop decodes a weight block for. */
    private static final int TOKENS = 4;

    private SimdQ4_0x8() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x, SimdI8.activations(x));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x, SimdI8.activations(x));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                        float[] y, int yOff, int yStride) {
        checkCompatible(w, x[0]);
        final int[][] yw = SimdI8.activationsAll(x, nTokens);
        final int cols = x[0].n;
        final float[] sum = new float[TOKENS];
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, (row, off, r) ->
                dotTokens(row, off, x, nTokens, yw, cols, y, yOff + r, yStride, sum));
    }

    /**
     * One weight row dotted with the {@code nTokens} activation rows, {@link #TOKENS} tokens at a time: the
     * nibbles of a block decode once per tile and every token of the tile multiplies its pre-widened activation
     * lanes by them.
     */
    private static void dotTokens(byte[] b, int off, QuantizedActivations[] x, int nTokens, int[][] yw, int cols,
                                  float[] y, int y0, int yStride, float[] sum) {
        final int groups = cols / Q4_0x8.ELEMENTS;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = off;
            int ib = 0;
            for (int g = 0; g < groups; g++, o += Q4_0x8.BYTES) {
                final int qs = o + Q4_0x8.QS;
                for (int j = 0; j < Q4_0x8.BLOCKS; j++, ib++) {
                    final IntVector loSlice = SimdI8.widen(
                            ByteVector.fromArray(SimdI8.B8, b, qs + 8 * j), 0, SimdI8.I8);
                    final IntVector hiSlice = SimdI8.widen(
                            ByteVector.fromArray(SimdI8.B8, b, qs + 64 + 8 * j), 0, SimdI8.I8);
                    final IntVector lo0 = loSlice.lanewise(VectorOperators.LSHL, 28)
                            .lanewise(VectorOperators.ASHR, 28);
                    final IntVector lo1 = loSlice.lanewise(VectorOperators.LSHL, 24)
                            .lanewise(VectorOperators.ASHR, 28);
                    final IntVector hi0 = hiSlice.lanewise(VectorOperators.LSHL, 28)
                            .lanewise(VectorOperators.ASHR, 28);
                    final IntVector hi1 = hiSlice.lanewise(VectorOperators.LSHL, 24)
                            .lanewise(VectorOperators.ASHR, 28);
                    final float d = fp16At(b, o + Q4_0x8.D + 2 * j);
                    final int yb = ib * 32;
                    for (int ti = 0; ti < n; ti++) {
                        final int[] a = yw[t0 + ti];
                        final IntVector acc = lo0.mul(IntVector.fromArray(SimdI8.I8, a, yb))
                                .add(lo1.mul(IntVector.fromArray(SimdI8.I8, a, yb + 16)))
                                .add(hi0.mul(IntVector.fromArray(SimdI8.I8, a, yb + 8)))
                                .add(hi1.mul(IntVector.fromArray(SimdI8.I8, a, yb + 24)));
                        sum[ti] += SimdI8.sum(acc) * d * x[t0 + ti].d[ib];
                    }
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q4_0);
        x.requireCompatible(w, GGMLType.Q8_0);
        if (w.cols() % Q4_0x8.ELEMENTS != 0) {
            throw new IllegalArgumentException("the Q4_0x8 kernel reads rows of whole 256-element groups; "
                    + w + " has " + w.cols() + " columns");
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final int groups = x.n / Q4_0x8.ELEMENTS;
        float sumf = 0;
        int ib = 0;
        for (int g = 0; g < groups; g++, off += Q4_0x8.BYTES) {
            final int qs = off + Q4_0x8.QS;
            for (int j = 0; j < Q4_0x8.BLOCKS; j++, ib++) {
                // each 8-byte slice widens to 8 int lanes; the signed nibble decode of the XOR-0x88 storage
                // is (w << 28) >> 28 for the low nibble and (w << 24) >> 28 for the high one; the first 64
                // quant bytes hold elements 0..7 and 16..23 of each block, the second 64 hold 8..15 and
                // 24..31
                final IntVector loSlice = SimdI8.widen(
                        ByteVector.fromArray(SimdI8.B8, b, qs + 8 * j), 0, SimdI8.I8);
                final IntVector hiSlice = SimdI8.widen(
                        ByteVector.fromArray(SimdI8.B8, b, qs + 64 + 8 * j), 0, SimdI8.I8);
                final int y = ib * 32;
                final IntVector acc = loSlice.lanewise(VectorOperators.LSHL, 28)
                        .lanewise(VectorOperators.ASHR, 28)
                        .mul(IntVector.fromArray(SimdI8.I8, yw, y))
                        .add(loSlice.lanewise(VectorOperators.LSHL, 24).lanewise(VectorOperators.ASHR, 28)
                                .mul(IntVector.fromArray(SimdI8.I8, yw, y + 16)))
                        .add(hiSlice.lanewise(VectorOperators.LSHL, 28).lanewise(VectorOperators.ASHR, 28)
                                .mul(IntVector.fromArray(SimdI8.I8, yw, y + 8)))
                        .add(hiSlice.lanewise(VectorOperators.LSHL, 24).lanewise(VectorOperators.ASHR, 28)
                                .mul(IntVector.fromArray(SimdI8.I8, yw, y + 24)));
                final int sumi = SimdI8.sum(acc);
                sumf += sumi * fp16At(b, off + Q4_0x8.D + 2 * j) * yd[ib];
            }
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ4_0x8{Q4_0x8 x Q8_0}";
    }
}

package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of F16 (IEEE half) weights with F32 activations, {@link Fp16VecDot#F16} with the widening
 * as bit arithmetic instead of the 65536-entry lookup: the eight little-endian halves of a group widen to int
 * lanes, and a normal half's float bits are {@code (h & 0x7FFF) << 13} plus the exponent rebias
 * {@code 0x38000000} with the sign bit set back on, which is exact for every half from {@code 2^-14} to
 * {@code 65504}. A group holding a subnormal, zero, infinite or NaN lane (exponent field 0 or 31) takes the
 * exceptional path, which builds the subnormal value as {@code mantissa * 2^-24} (times {@code +-1}) and the
 * infinite or NaN bits as {@code sign | 0x7F800000 | mantissa << 13} — the values
 * {@code ggml_compute_fp16_to_fp32} produces — so the kernel returns exactly what {@link Fp16VecDot#F16}
 * returns. Both paths then multiply the eight float lanes and add them to the eight scalar partial sums of
 * {@code ggml_vec_dot_f16}, which reduce pairwise in float and complete in double as that function does.
 * Immutable, stateless and thread-safe.
 */
public final class SimdF16 implements CpuVecDot {

    /** The F16 vec-dot. */
    public static final SimdF16 INSTANCE = new SimdF16();

    private SimdF16() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x.f32, x.n);
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x.f32, x.n);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.F16);
        x.requireCompatible(w, GGMLType.F32);
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
        final byte[] row = CpuVecDotRows.buffer(w);
        final int n = x[0].n;
        for (int r = 0; r < rowCount; r++) {
            CpuVecDotRows.copyRow(w, rowStart + r, row);
            int t0 = 0;
            for (; t0 + SimdFp.TOKENS <= nTokens; t0 += SimdFp.TOKENS) {
                dotTile8(row, 0, x, t0, n, y, yOff + r, yStride);
            }
            for (; t0 < nTokens; t0++) {
                y[yOff + r + t0 * yStride] = dot(row, 0, x[t0].f32, n);
            }
        }
    }

    /**
     * Eight tokens of one weight row: eight lane accumulators, one per token, so the row's halves widen once
     * for the whole tile.
     */
    private static void dotTile8(byte[] b, int off, QuantizedActivations[] x, int t0, int n, float[] y,
                                 int y0, int yStride) {
        FloatVector a0 = FloatVector.zero(SimdFp.F8);
        FloatVector a1 = FloatVector.zero(SimdFp.F8);
        FloatVector a2 = FloatVector.zero(SimdFp.F8);
        FloatVector a3 = FloatVector.zero(SimdFp.F8);
        FloatVector a4 = FloatVector.zero(SimdFp.F8);
        FloatVector a5 = FloatVector.zero(SimdFp.F8);
        FloatVector a6 = FloatVector.zero(SimdFp.F8);
        FloatVector a7 = FloatVector.zero(SimdFp.F8);
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final ShortVector h = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0);
            final IntVector bits = (IntVector) h.convertShape(VectorOperators.S2I, SimdI8.I8, 0);
            final IntVector sign = bits.and(0x8000).lanewise(VectorOperators.LSHL, 16);
            final IntVector exponent = bits.and(0x7C00);
            FloatVector w = (FloatVector) bits.and(0x7FFF).lanewise(VectorOperators.LSHL, 13)
                    .add(0x38000000).or(sign).reinterpretShape(SimdFp.F8, 0);
            final VectorMask<Integer> subnormal = exponent.eq(0);
            final VectorMask<Integer> infinite = exponent.eq(0x7C00);
            if (subnormal.or(infinite).anyTrue()) {
                w = exceptional(bits, sign, w, subnormal.cast(SimdFp.F8), infinite.cast(SimdFp.F8));
            }
            a0 = a0.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0].f32, i)));
            a1 = a1.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 1].f32, i)));
            a2 = a2.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 2].f32, i)));
            a3 = a3.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 3].f32, i)));
            a4 = a4.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 4].f32, i)));
            a5 = a5.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 5].f32, i)));
            a6 = a6.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 6].f32, i)));
            a7 = a7.add(w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 7].f32, i)));
        }
        y[y0 + t0 * yStride] = tail(SimdFp.pairwise(a0), b, off, x[t0].f32, i, n);
        y[y0 + (t0 + 1) * yStride] = tail(SimdFp.pairwise(a1), b, off, x[t0 + 1].f32, i, n);
        y[y0 + (t0 + 2) * yStride] = tail(SimdFp.pairwise(a2), b, off, x[t0 + 2].f32, i, n);
        y[y0 + (t0 + 3) * yStride] = tail(SimdFp.pairwise(a3), b, off, x[t0 + 3].f32, i, n);
        y[y0 + (t0 + 4) * yStride] = tail(SimdFp.pairwise(a4), b, off, x[t0 + 4].f32, i, n);
        y[y0 + (t0 + 5) * yStride] = tail(SimdFp.pairwise(a5), b, off, x[t0 + 5].f32, i, n);
        y[y0 + (t0 + 6) * yStride] = tail(SimdFp.pairwise(a6), b, off, x[t0 + 6].f32, i, n);
        y[y0 + (t0 + 7) * yStride] = tail(SimdFp.pairwise(a7), b, off, x[t0 + 7].f32, i, n);
    }

    /** The elements past the last full group of eight, added one product at a time in double. */
    private static float tail(float sum, byte[] b, int off, float[] x, int i, int n) {
        double s = sum;
        for (; i < n; i++) {
            s += fp16At(b, off + 2 * i) * x[i];
        }
        return (float) s;
    }

    private static float dot(byte[] b, int off, float[] x, int n) {
        FloatVector acc = FloatVector.zero(SimdFp.F8);
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            // the sixteen bytes of the group are eight little-endian halves: widen them to int lanes and move
            // the sign, exponent and mantissa into the float fields
            final ShortVector h = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0);
            final IntVector bits = (IntVector) h.convertShape(VectorOperators.S2I, SimdI8.I8, 0);
            final IntVector sign = bits.and(0x8000).lanewise(VectorOperators.LSHL, 16);
            final IntVector exponent = bits.and(0x7C00);
            FloatVector w = (FloatVector) bits.and(0x7FFF).lanewise(VectorOperators.LSHL, 13)
                    .add(0x38000000).or(sign).reinterpretShape(SimdFp.F8, 0);
            final VectorMask<Integer> subnormal = exponent.eq(0);
            final VectorMask<Integer> infinite = exponent.eq(0x7C00);
            if (subnormal.or(infinite).anyTrue()) {
                w = exceptional(bits, sign, w, subnormal.cast(SimdFp.F8), infinite.cast(SimdFp.F8));
            }
            acc = acc.add(w.mul(FloatVector.fromArray(SimdFp.F8, x, i)));
        }
        double sum = SimdFp.pairwise(acc);
        for (; i < n; i++) {
            sum += fp16At(b, off + 2 * i) * x[i];
        }
        return (float) sum;
    }

    /**
     * The lanes of a group whose half is subnormal or zero, or infinite or NaN: their exact values replace the
     * normal-path ones, leaving every other lane as it was.
     */
    private static FloatVector exceptional(IntVector bits, IntVector sign, FloatVector normal,
                                           VectorMask<Float> subnormal, VectorMask<Float> infinite) {
        final IntVector mant = bits.and(0x03FF);
        final FloatVector tiny = ((FloatVector) mant.convertShape(VectorOperators.I2F, SimdFp.F8, 0))
                .mul(0x1.0p-24f)
                .mul((FloatVector) sign.or(0x3F800000).reinterpretShape(SimdFp.F8, 0));
        final FloatVector special = (FloatVector) sign.or(0x7F800000).or(mant.lanewise(VectorOperators.LSHL, 13))
                .reinterpretShape(SimdFp.F8, 0);
        return normal.blend(tiny, subnormal).blend(special, infinite);
    }

    @Override
    public String toString() {
        return "SimdF16{F16 x F32}";
    }
}

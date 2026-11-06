package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.bf16ToFp32;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.shortAt;

/**
 * Vector API vec-dot of BF16 weights with F32 activations, {@link Fp16VecDot#BF16} with the bfloat16 widen as a
 * shift instead of a lookup: the eight little-endian halves of a group bitcast to a short vector, widen to int
 * lanes, shift their sixteen bits into the float exponent field and reinterpret as floats, which is exactly
 * {@code ggml_compute_bf16_to_fp32} without its masking. Every partial sum is the scalar kernel's eight-lane
 * one, reduced pairwise in float and completed in double, so the kernel returns what {@link Fp16VecDot#BF16}
 * returns. Immutable, stateless and thread-safe.
 */
public final class SimdBf16 implements CpuVecDot {

    /** The BF16 vec-dot. */
    public static final SimdBf16 INSTANCE = new SimdBf16();

    private SimdBf16() {
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
        Q8VecDot.requireWeightType(w, GGMLType.BF16);
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
     * Eight tokens of one weight row: eight lane accumulators, one per token, so the row's bfloat16 values
     * widen once for the whole tile.
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
            final IntVector bits = (IntVector) ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0)).convertShape(VectorOperators.S2I, SimdI8.I8, 0)
                    .lanewise(VectorOperators.LSHL, 16);
            final FloatVector w = (FloatVector) bits.reinterpretShape(SimdFp.F8, 0);
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

    /** The elements past the last full group of eight, added one product at a time. */
    private static float tail(float sum, byte[] b, int off, float[] x, int i, int n) {
        double s = sum;
        for (; i < n; i++) {
            s += bf16ToFp32(shortAt(b, off + 2 * i)) * x[i];
        }
        return (float) s;
    }

    private static float dot(byte[] b, int off, float[] x, int n) {
        FloatVector acc = FloatVector.zero(SimdFp.F8);
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            // the sixteen bytes of the group are eight little-endian bf16 values: bitcast to halves, widen to
            // int lanes, move the sixteen bits into the exponent field, reinterpret as floats
            final ShortVector halves = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0);
            final IntVector bits = (IntVector) halves.convertShape(VectorOperators.S2I, SimdI8.I8, 0)
                    .lanewise(VectorOperators.LSHL, 16);
            acc = acc.add(((FloatVector) bits.reinterpretShape(SimdFp.F8, 0))
                    .mul(FloatVector.fromArray(SimdFp.F8, x, i)));
        }
        double sum = SimdFp.pairwise(acc);
        for (; i < n; i++) {
            sum += bf16ToFp32(shortAt(b, off + 2 * i)) * x[i];
        }
        return (float) sum;
    }

    @Override
    public String toString() {
        return "SimdBf16{BF16 x F32}";
    }
}

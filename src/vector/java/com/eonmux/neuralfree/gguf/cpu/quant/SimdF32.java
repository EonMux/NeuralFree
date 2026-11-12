package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;

import java.nio.ByteBuffer;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.floatAt;

/**
 * Vector API vec-dot of F32 weights with F32 activations, {@link Fp16VecDot#F32} with the eight float partial
 * sums of {@code ggml_vec_dot_f32} as the eight lanes of one float vector. The lane-wise products and sums are
 * the scalar ones, and the lanes reduce pairwise in the same order, so the kernel returns exactly what
 * {@link Fp16VecDot#F32} returns. Immutable, stateless and thread-safe.
 */
public final class SimdF32 implements CpuVecDot {

    /** The F32 vec-dot. */
    public static final SimdF32 INSTANCE = new SimdF32();

    private SimdF32() {
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
        Q8VecDot.requireWeightType(w, GGMLType.F32);
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
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        final float[] lanes = new float[SimdFp.TOKENS * SimdFp.F8.length()];
        final int n = x[0].n;
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, row);
            int t0 = 0;
            for (; t0 + SimdFp.TOKENS <= nTokens; t0 += SimdFp.TOKENS) {
                dotTile8(row, 0, x, t0, n, lanes, y, yOff + r, yStride);
            }
            for (; t0 < nTokens; t0++) {
                y[yOff + r + t0 * yStride] = dot(row, 0, x[t0].f32, n);
            }
        }
    }

    /**
     * Eight tokens of one weight row: eight lane accumulators, one per token, so the row's float lanes load
     * once for the whole tile.
     * The accumulators reduce from the caller's scratch {@code lanes} ({@link SimdFp#pairwise(float[], int)}).
     */
    private static void dotTile8(byte[] b, int off, QuantizedActivations[] x, int t0, int n, float[] lanes,
                                 float[] y, int y0, int yStride) {
        FloatVector a0 = FloatVector.zero(SimdFp.F8);
        FloatVector a1 = FloatVector.zero(SimdFp.F8);
        FloatVector a2 = FloatVector.zero(SimdFp.F8);
        FloatVector a3 = FloatVector.zero(SimdFp.F8);
        FloatVector a4 = FloatVector.zero(SimdFp.F8);
        FloatVector a5 = FloatVector.zero(SimdFp.F8);
        FloatVector a6 = FloatVector.zero(SimdFp.F8);
        FloatVector a7 = FloatVector.zero(SimdFp.F8);
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 32) {
            final FloatVector w = (FloatVector) ByteVector.fromArray(SimdFp.B32, b, p)
                    .reinterpretShape(SimdFp.F8, 0);
            a0 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0].f32, i)).add(a0);
            a1 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 1].f32, i)).add(a1);
            a2 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 2].f32, i)).add(a2);
            a3 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 3].f32, i)).add(a3);
            a4 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 4].f32, i)).add(a4);
            a5 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 5].f32, i)).add(a5);
            a6 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 6].f32, i)).add(a6);
            a7 = w.mul(FloatVector.fromArray(SimdFp.F8, x[t0 + 7].f32, i)).add(a7);
        }
        final int l = SimdFp.F8.length();
        ((FloatVector) a0.check(SimdFp.F8)).intoArray(lanes, 0);
        ((FloatVector) a1.check(SimdFp.F8)).intoArray(lanes, l);
        ((FloatVector) a2.check(SimdFp.F8)).intoArray(lanes, 2 * l);
        ((FloatVector) a3.check(SimdFp.F8)).intoArray(lanes, 3 * l);
        ((FloatVector) a4.check(SimdFp.F8)).intoArray(lanes, 4 * l);
        ((FloatVector) a5.check(SimdFp.F8)).intoArray(lanes, 5 * l);
        ((FloatVector) a6.check(SimdFp.F8)).intoArray(lanes, 6 * l);
        ((FloatVector) a7.check(SimdFp.F8)).intoArray(lanes, 7 * l);
        for (int t = 0; t < SimdFp.TOKENS; t++) {
            y[y0 + (t0 + t) * yStride] = tail(SimdFp.pairwise(lanes, t * l), b, off, x[t0 + t].f32, i, n);
        }
    }

    /** The elements past the last full group of eight, added one product at a time. */
    private static float tail(float sum, byte[] b, int off, float[] x, int i, int n) {
        float s = sum;
        for (; i < n; i++) {
            s += floatAt(b, off + 4 * i) * x[i];
        }
        return s;
    }

    private static float dot(byte[] b, int off, float[] x, int n) {
        FloatVector acc = FloatVector.zero(SimdFp.F8);
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 32) {
            // the 32 bytes of the group are the eight little-endian floats: bitcast them into the lanes
            // instead of decoding element by element into a buffer the vector then reloads
            final FloatVector w = (FloatVector) ByteVector.fromArray(SimdFp.B32, b, p)
                    .reinterpretShape(SimdFp.F8, 0);
            acc = w.mul(FloatVector.fromArray(SimdFp.F8, x, i)).add(acc);
        }
        float sum = SimdFp.pairwise((FloatVector) acc.check(SimdFp.F8));
        for (; i < n; i++) {
            sum += floatAt(b, off + 4 * i) * x[i];
        }
        return sum;
    }

    @Override
    public String toString() {
        return "SimdF32{F32 x F32}";
    }
}

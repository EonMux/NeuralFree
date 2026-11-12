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

    /** Per-thread float row of the token tiles ({@link #floatRow}). */
    private static final ThreadLocal<float[]> ROW = new ThreadLocal<>();

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
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        final float[] lanes = new float[SimdFp.TOKENS * SimdFp.F8.length()];
        final int n = x[0].n;
        // two tiles or more read the row's values from one float copy; a single tile widens them in registers,
        // where the copy's store and reload would cost more than the widening it saves
        final float[] wf = nTokens >= 2 * SimdFp.TOKENS ? floatRow(n) : null;
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, row);
            int t0 = 0;
            if (wf != null) {
                widen(row, 0, n, wf);
                for (; t0 + SimdFp.TOKENS <= nTokens; t0 += SimdFp.TOKENS) {
                    dotTile8(wf, x, t0, n, lanes, y, yOff + r, yStride);
                }
            } else if (nTokens >= SimdFp.TOKENS) {
                dotTile8(row, 0, x, t0, n, lanes, y, yOff + r, yStride);
                t0 += SimdFp.TOKENS;
            }
            for (; t0 < nTokens; t0++) {
                y[yOff + r + t0 * yStride] = dot(row, 0, x[t0].f32, n);
            }
        }
    }

    /**
     * The calling thread's float row of at least {@code n} lanes, which holds a weight row's widened values for all
     * of its token tiles.
     */
    private static float[] floatRow(int n) {
        float[] wf = ROW.get();
        if (wf == null || wf.length < n) {
            wf = new float[n];
            ROW.set(wf);
        }
        return wf;
    }

    /**
     * Widens the {@code n} bfloat16 values at {@code b[off ..]} into {@code wf}, the groups of eight as {@link #dot}
     * widens them and the elements past the last group with {@code bf16ToFp32}, so every value is the one {@code dot}
     * uses.
     */
    private static void widen(byte[] b, int off, int n, float[] wf) {
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final IntVector bits = (IntVector) ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0)).convertShape(VectorOperators.S2I, SimdI8.I8, 0)
                    .lanewise(VectorOperators.LSHL, 16);
            ((FloatVector) bits.reinterpretShape(SimdFp.F8, 0)).intoArray(wf, i);
        }
        for (; i < n; i++) {
            wf[i] = bf16ToFp32(shortAt(b, off + 2 * i));
        }
    }

    /**
     * Eight tokens of one weight row, widened once per row into {@code wf} for all of its tiles: eight lane
     * accumulators, one per token. The accumulators reduce from the caller's scratch {@code lanes}
     * ({@link SimdFp#pairwise(float[], int)}), since 64 lane extractions after the loop would take the method past
     * C2's inlining node cutoff.
     */
    private static void dotTile8(float[] wf, QuantizedActivations[] x, int t0, int n, float[] lanes,
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
        for (; i + 8 <= n; i += 8) {
            final FloatVector w = FloatVector.fromArray(SimdFp.F8, wf, i);
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
            y[y0 + (t0 + t) * yStride] = tail(SimdFp.pairwise(lanes, t * l), wf, x[t0 + t].f32, i, n);
        }
    }

    /** The elements past the last full group of eight, added one product at a time in double. */
    private static float tail(float sum, float[] wf, float[] x, int i, int n) {
        double s = sum;
        for (; i < n; i++) {
            s += wf[i] * x[i];
        }
        return (float) s;
    }

    /**
     * Eight tokens of one weight row widened in registers, for a product of a single tile: eight lane accumulators,
     * one per token, so the row's bfloat16 values widen once for the tile. The accumulators reduce from the caller's
     * scratch {@code lanes} ({@link SimdFp#pairwise(float[], int)}).
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
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final IntVector bits = (IntVector) ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, p)
                    .reinterpretShape(SimdFp.S8, 0)).convertShape(VectorOperators.S2I, SimdI8.I8, 0)
                    .lanewise(VectorOperators.LSHL, 16);
            final FloatVector w = (FloatVector) bits.reinterpretShape(SimdFp.F8, 0);
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

    /** The elements past the last full group of eight of a bfloat16 row, added one product at a time in double. */
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
            acc = (((FloatVector) bits.reinterpretShape(SimdFp.F8, 0))
                    .mul(FloatVector.fromArray(SimdFp.F8, x, i))).add(acc);
        }
        double sum = SimdFp.pairwise((FloatVector) acc.check(SimdFp.F8));
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

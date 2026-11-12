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

    /** Per-thread float row of the token tiles ({@link #floatRow}). */
    private static final ThreadLocal<float[]> ROW = new ThreadLocal<>();

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
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        final float[] lanes = new float[SimdFp.TOKENS * SimdFp.F8.length()];
        final int n = x[0].n;
        final float[] wf = floatRow(n);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, row);
            int t0 = 0;
            if (nTokens >= SimdFp.TOKENS) {
                widen(row, 0, n, wf);
            }
            for (; t0 + SimdFp.TOKENS <= nTokens; t0 += SimdFp.TOKENS) {
                dotTile8(wf, x, t0, n, lanes, y, yOff + r, yStride);
            }
            for (; t0 < nTokens; t0++) {
                y[yOff + r + t0 * yStride] = dot(row, 0, x[t0].f32, n);
            }
        }
    }

    /**
     * The calling thread's float row of at least {@code n} lanes, which holds a weight row's widened halves for
     * all of its token tiles.
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
     * Widens the {@code n} halves at {@code b[off ..]} into {@code wf}, the groups of eight as {@link #dot} widens
     * them and the elements past the last group with {@code fp16At}, so every value is the one {@code dot} uses.
     */
    private static void widen(byte[] b, int off, int n, float[] wf) {
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
            w.intoArray(wf, i);
        }
        for (; i < n; i++) {
            wf[i] = fp16At(b, off + 2 * i);
        }
    }

    /**
     * Eight tokens of one weight row, widened once per row into {@code wf} for all of its tiles: eight lane
     * accumulators, one per token. The accumulators reduce from the caller's scratch {@code lanes}
     * ({@link SimdFp#pairwise(float[], int)}). Widening inside the tile loop would put the half conversion and its
     * exceptional path beside the eight accumulators in one compilation, past C2's inlining node cutoff.
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
            acc = w.mul(FloatVector.fromArray(SimdFp.F8, x, i)).add(acc);
        }
        double sum = SimdFp.pairwise((FloatVector) acc.check(SimdFp.F8));
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

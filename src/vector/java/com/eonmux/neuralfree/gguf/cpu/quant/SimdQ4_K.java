package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q4_K weights with Q8_K activations, {@link KQuantVecDot#Q4_K} with the nibble unpack in
 * byte lanes and the sub-block dots in int16 lanes: the 256 quants of a super-block unpack into the thread's
 * scratch row, each 32-element sub-block dots the matching activation sub-block (two 16-lane products added in
 * short lanes, folded pairwise into int lanes) and its sum joins one int accumulator scaled by the 6-bit
 * sub-block scale, so the horizontal sum runs once per token instead of once per sub-block. The scale unpack,
 * the mins sum over {@code bsums} and the float combination ({@code sums += d * isum}, {@code sumf -= dmin *
 * sumi}, result {@code sumf + sums}) follow {@code ggml_vec_dot_q4_K_q8_K_generic} exactly, so the kernel
 * returns what {@code KQuantVecDot.Q4_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) serves eight tokens from one unpack of the super-block.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_K implements CpuVecDot {

    /** The Q4_K vec-dot. */
    public static final SimdQ4_K INSTANCE = new SimdQ4_K();

    /** Tokens one pass of the batched kernel decodes a weight super-block for. */
    private static final int TOKENS = 8;

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    private SimdQ4_K() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q4_K);
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
        final short[][] pairs = SimdI8.bsumPairs(x, nTokens);
        final byte[] row = CpuVecDotRows.buffer(w);
        final float[] acc = SimdI8.tileAccumulators(2 * TOKENS);
        final int nb = x[0].n / QK_K;
        for (int r = 0; r < rowCount; r++) {
            CpuVecDotRows.copyRow(w, rowStart + r, row);
            dotRowTokens(row, x, nTokens, ys, pairs, nb, y, yOff + r, yStride, acc);
        }
    }

    /**
     * One weight row against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time: the super-block
     * unpacks once per tile and every token of the tile dots its pre-widened lanes against it.
     */
    private static void dotRowTokens(byte[] b, QuantizedActivations[] x, int nTokens, short[][] ys,
                                     short[][] pairs, int nb, float[] y, int y0, int yStride, float[] acc) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(acc, 0, 2 * TOKENS, 0.0f);
            int o = 0;
            for (int i = 0; i < nb; i++, o += Q4_K_BYTES) {
                unpack(b, o, q);
                KQuantVecDot.unpackScalesMins(b, o + Q4_K_SCALES, sm);
                final int xo = i * QK_K;
                final float d = fp16At(b, o + Q4_K_D);
                final float dmin = fp16At(b, o + Q4_K_DMIN);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    IntVector sum = SimdI8.zero(SimdI8.I8);
                    for (int g = 0; g < QK_K / 32; g++) {
                        final ShortVector lo = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g)
                                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                        final ShortVector hi = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g + 16)
                                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                        final int yo = xo + 32 * g;
                        sum = sum.add(SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, a, yo))
                                .add(hi.mul(ShortVector.fromArray(SimdI8.S16, a, yo + 16))),
                                SimdI8.zero(SimdI8.I8)).mul(sm[g]));
                    }
                    final float yd = x[t0 + ti].d[i];
                    acc[ti] += (d * yd) * SimdI8.sum(sum);
                    acc[TOKENS + ti] -= (dmin * yd) * SimdI8.minsDot(pairs[t0 + ti], i * (QK_K / 32), sm);
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = acc[TOKENS + ti] + acc[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        final float[] yd = x.d;
        final short[] pairs = SimdI8.bsumPairs(x);
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q4_K_BYTES) {
            unpack(b, off, q);
            KQuantVecDot.unpackScalesMins(b, off + Q4_K_SCALES, sm);
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 32; g++) {
                final ShortVector lo = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 32 * g;
                acc = acc.add(SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, yo))
                        .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 16))),
                        SimdI8.zero(SimdI8.I8)).mul(sm[g]));
            }
            final float d = fp16At(b, off + Q4_K_D) * yd[i];
            sums += d * SimdI8.sum(acc);
            final float dmin = fp16At(b, off + Q4_K_DMIN) * yd[i];
            sumf -= dmin * SimdI8.minsDot(pairs, i * (QK_K / 32), sm);
        }
        return sumf + sums;
    }

    /**
     * Unpacks the 256 four-bit quants of one super-block into {@code q}: group {@code j} of 64 elements takes the
     * low nibbles of its 32 quant bytes as elements {@code 64j .. 64j + 31} and their high nibbles as
     * {@code 64j + 32 .. 64j + 63}, which is the order {@code dequantize_row_q4_K} lays them out in.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        for (int j = 0; j < QK_K / 64; j++) {
            final int q4 = off + Q4_K_QS + 32 * j;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q4 + 8 * t);
                Q8VecDot.putLong(q, 64 * j + 8 * t, v & Q8VecDot.LOW_NIBBLES);
                Q8VecDot.putLong(q, 64 * j + 32 + 8 * t, (v >>> 4) & Q8VecDot.LOW_NIBBLES);
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ4_K{Q4_K x Q8_K}";
    }
}

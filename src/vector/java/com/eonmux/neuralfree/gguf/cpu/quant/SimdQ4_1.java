package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_1;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q4_1 weights with Q8_1 activations, {@link LegacyVecDot#Q4_1} with the unsigned
 * nibbles in int16 lanes: the 16 quant bytes widen to 16 short lanes, the low nibbles dot elements
 * {@code 0..15} and the shifted high nibbles elements {@code 16..31}, the two product vectors add in short
 * lanes and fold pairwise into int lanes ({@link SimdI8#pairSum}). A nibble times an int8 activation fits a
 * short, and so does the sum of two of them. The float combination
 * {@code (d_x * d_y) * sumi + m_x * s_y} per block follows {@code ggml_vec_dot_q4_1_q8_1_generic}, so the
 * kernel returns exactly what {@link LegacyVecDot#Q4_1} returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of two weight rows by eight tokens.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_1 implements CpuVecDot {

    /** The Q4_1 vec-dot. */
    public static final SimdQ4_1 INSTANCE = new SimdQ4_1();

    /** Tokens one pass of the batched kernel decodes a weight block for. */
    private static final int TOKENS = 8;

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;

    private SimdQ4_1() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q4_1);
        x.requireCompatible(w, GGMLType.Q8_1);
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
        final byte[][] rows = CpuVecDotRows.buffers(w, TILE_ROWS);
        final float[] acc = SimdI8.tileAccumulators(TILE_ROWS * TOKENS);
        final int nb = x[0].n / QK4_1;
        int r = 0;
        for (; r + TILE_ROWS <= rowCount; r += TILE_ROWS) {
            CpuVecDotRows.copyRow(w, rowStart + r, rows[0]);
            CpuVecDotRows.copyRow(w, rowStart + r + 1, rows[1]);
            dotTile2(rows[0], rows[1], x, nTokens, ys, nb, y, yOff + r, yStride, acc);
        }
        for (; r < rowCount; r++) {
            CpuVecDotRows.copyRow(w, rowStart + r, rows[0]);
            dotRowTokens(rows[0], x, nTokens, ys, nb, y, yOff + r, yStride, acc);
        }
    }

    /** Two weight rows against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time. */
    private static void dotTile2(byte[] b0, byte[] b1, QuantizedActivations[] x, int nTokens, short[][] ys,
                                 int nb, float[] y, int y0, int yStride, float[] sum) {
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, TILE_ROWS * TOKENS, 0.0f);
            int o = 0;
            for (int ib = 0; ib < nb; ib++, o += Q4_1_BYTES) {
                final ByteVector q0 = ByteVector.fromArray(SimdI8.B16, b0, o + Q4_1_QS);
                final ShortVector w0 = (ShortVector) q0.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector lo0 = w0.and((short) 0x0F);
                final ShortVector hi0 = w0.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
                final ByteVector q1 = ByteVector.fromArray(SimdI8.B16, b1, o + Q4_1_QS);
                final ShortVector w1 = (ShortVector) q1.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector lo1 = w1.and((short) 0x0F);
                final ShortVector hi1 = w1.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
                final int yb = ib * QK4_1;
                final float d0 = fp16At(b0, o + Q4_1_D);
                final float m0 = fp16At(b0, o + Q4_1_M);
                final float d1 = fp16At(b1, o + Q4_1_D);
                final float m1 = fp16At(b1, o + Q4_1_M);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    final ShortVector av = ShortVector.fromArray(SimdI8.S16, a, yb);
                    final ShortVector avh = ShortVector.fromArray(SimdI8.S16, a, yb + 16);
                    final QuantizedActivations tok = x[t0 + ti];
                    final int base = ti * TILE_ROWS;
                    sum[base] += (d0 * tok.d[ib]) * SimdI8.sum(SimdI8.pairSum(lo0.mul(av).add(hi0.mul(avh)),
                            SimdI8.zero(SimdI8.I8))) + m0 * tok.s[ib];
                    sum[base + 1] += (d1 * tok.d[ib]) * SimdI8.sum(SimdI8.pairSum(lo1.mul(av).add(hi1.mul(avh)),
                            SimdI8.zero(SimdI8.I8))) + m1 * tok.s[ib];
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (ti + t0) * yStride] = sum[ti * TILE_ROWS];
                y[y0 + 1 + (ti + t0) * yStride] = sum[ti * TILE_ROWS + 1];
            }
        }
    }

    /** One weight row against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time. */
    private static void dotRowTokens(byte[] b, QuantizedActivations[] x, int nTokens, short[][] ys, int nb,
                                     float[] y, int y0, int yStride, float[] sum) {
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = 0;
            for (int ib = 0; ib < nb; ib++, o += Q4_1_BYTES) {
                final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, o + Q4_1_QS);
                final ShortVector w = (ShortVector) q.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector lo = w.and((short) 0x0F);
                final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
                final int yb = ib * QK4_1;
                final float d = fp16At(b, o + Q4_1_D);
                final float m = fp16At(b, o + Q4_1_M);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    final QuantizedActivations tok = x[t0 + ti];
                    sum[ti] += (d * tok.d[ib]) * SimdI8.sum(SimdI8.pairSum(
                            lo.mul(ShortVector.fromArray(SimdI8.S16, a, yb))
                                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, a, yb + 16))),
                            SimdI8.zero(SimdI8.I8))) + m * tok.s[ib];
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final float[] ya = x.s;
        final int nb = x.n / QK4_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q4_1_BYTES) {
            final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, off + Q4_1_QS);
            final ShortVector w = (ShortVector) q.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final ShortVector lo = w.and((short) 0x0F);
            final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
            final int y = ib * QK4_1;
            final int sumi = SimdI8.sum(SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))), SimdI8.zero(SimdI8.I8)));
            sumf += (fp16At(b, off + Q4_1_D) * yd[ib]) * sumi + fp16At(b, off + Q4_1_M) * ya[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ4_1{Q4_1 x Q8_1}";
    }
}

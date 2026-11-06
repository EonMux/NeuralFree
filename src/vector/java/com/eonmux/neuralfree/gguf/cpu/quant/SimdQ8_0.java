package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q8_0 weights with Q8_0 activations, {@link Q8VecDot} with the 32-element int8 block
 * dots in int16 lanes: the two 16-byte halves of a block widen to short lanes and multiply the pre-widened
 * activation halves, and the two product vectors add in short lanes before one pairwise fold into int lanes
 * ({@link SimdI8#pairSum}) — every int8 product and every sum of two of them fits a short. The float
 * combination {@code sumi * (d_x * d_y)} per block follows {@code ggml_vec_dot_q8_0_q8_0_generic}, so the
 * kernel returns exactly what {@code Q8VecDot} returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of two weight rows by eight tokens,
 * so a decoded weight block and a widened activation block are each reused across the tile. Immutable,
 * stateless and thread-safe.</p>
 */
public final class SimdQ8_0 implements CpuVecDot {

    /** The Q8_0 vec-dot. */
    public static final SimdQ8_0 INSTANCE = new SimdQ8_0();

    /** Tokens one pass of the batched kernel decodes a weight block for. */
    private static final int TOKENS = 8;

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;

    private SimdQ8_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        x.requireCompatible(w, GGMLType.Q8_0);
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
        final int nb = x[0].n / QK8_0;
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
            for (int ib = 0; ib < nb; ib++, o += Q8_0_BYTES) {
                final ShortVector a0 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b0, o + Q8_0_QS)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector a1 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b0, o + Q8_0_QS + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector c0 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b1, o + Q8_0_QS)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector c1 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b1, o + Q8_0_QS + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yb = ib * QK8_0;
                final float d0 = fp16At(b0, o + Q8_0_D);
                final float d1 = fp16At(b1, o + Q8_0_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    final ShortVector v0 = ShortVector.fromArray(SimdI8.S16, a, yb);
                    final ShortVector v1 = ShortVector.fromArray(SimdI8.S16, a, yb + 16);
                    final int base = ti * TILE_ROWS;
                    sum[base] += SimdI8.sum(SimdI8.pairSum(a0.mul(v0).add(a1.mul(v1)),
                            SimdI8.zero(SimdI8.I8))) * (d0 * x[t0 + ti].d[ib]);
                    sum[base + 1] += SimdI8.sum(SimdI8.pairSum(c0.mul(v0).add(c1.mul(v1)),
                            SimdI8.zero(SimdI8.I8))) * (d1 * x[t0 + ti].d[ib]);
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
            for (int ib = 0; ib < nb; ib++, o += Q8_0_BYTES) {
                final ShortVector a0 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, o + Q8_0_QS)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector a1 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, o + Q8_0_QS + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yb = ib * QK8_0;
                final float d = fp16At(b, o + Q8_0_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    sum[ti] += SimdI8.sum(SimdI8.pairSum(a0.mul(ShortVector.fromArray(SimdI8.S16, a, yb))
                            .add(a1.mul(ShortVector.fromArray(SimdI8.S16, a, yb + 16))),
                            SimdI8.zero(SimdI8.I8))) * (d * x[t0 + ti].d[ib]);
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK8_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q8_0_BYTES) {
            // the two 16-byte halves of the block widen to 16 short lanes each and multiply the matching
            // pre-widened activation halves in vpmullw; every int8 product fits a short and so does the sum of
            // two of them, so the block dot is the int-lane one
            final ShortVector q0 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final ShortVector q1 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS + 16)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final int y = ib * QK8_0;
            final int sumi = SimdI8.sum(SimdI8.pairSum(q0.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(q1.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))), SimdI8.zero(SimdI8.I8)));
            sumf += sumi * (fp16At(b, off + Q8_0_D) * yd[ib]);
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ8_0{Q8_0 x Q8_0}";
    }
}

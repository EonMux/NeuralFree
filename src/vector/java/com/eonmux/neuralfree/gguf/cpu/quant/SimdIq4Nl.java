package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_NL;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of IQ4_NL weights with Q8_0 activations, {@link IQuantVecDot#IQ4_NL} with the codebook as a
 * vector lookup: the 16 quant bytes give sixteen low nibbles (elements {@code 0..15}) and sixteen shifted high
 * nibbles (elements {@code 16..31}), each selects its lane out of the {@code kvalues_iq4nl} table with
 * {@code selectFrom}, and the selected int8 values dot the pre-widened short activation lanes in
 * {@code vpmullw}, the two product vectors adding in short lanes before one pairwise fold into int lanes
 * ({@link SimdI8#pairSum}). Two int8 codebook values sum without wrapping a short, and so do their products
 * with an int8 activation. The float combination {@code d_x * d_y * sumi} per block follows
 * {@code ggml_vec_dot_iq4_nl_q8_0_generic}, so the kernel returns exactly what {@link IQuantVecDot#IQ4_NL}
 * returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of two weight rows by eight tokens, so
 * the codebook lookup and the activation lanes of a block are each reused across the tile. Immutable,
 * stateless and thread-safe.</p>
 */
public final class SimdIq4Nl implements CpuVecDot {

    /** The IQ4_NL vec-dot. */
    public static final SimdIq4Nl INSTANCE = new SimdIq4Nl();

    /** Tokens one pass of the batched kernel decodes a weight block for. */
    private static final int TOKENS = 8;

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;

    /** {@code kvalues_iq4nl} as sixteen byte lanes: the source of {@code selectFrom}. */
    private static final ByteVector TABLE = table();

    private SimdIq4Nl() {
    }

    /** The codebook as a 16-lane byte vector. */
    private static ByteVector table() {
        final byte[] t = new byte[QK4_NL / 2];
        for (int i = 0; i < t.length; i++) {
            t[i] = (byte) QuantTables.kvalueIq4nl(i);
        }
        return ByteVector.fromArray(SimdI8.B16, t, 0);
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
        Q8VecDot.requireWeightType(w, GGMLType.IQ4_NL);
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
        final int nb = x[0].n / QK4_NL;
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
            for (int ib = 0; ib < nb; ib++, o += IQ4_NL_BYTES) {
                final ByteVector q0 = ByteVector.fromArray(SimdI8.B16, b0, o + IQ4_NL_QS);
                final ShortVector lo0 = (ShortVector) q0.and((byte) 0x0F).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi0 = (ShortVector) q0.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ByteVector q1 = ByteVector.fromArray(SimdI8.B16, b1, o + IQ4_NL_QS);
                final ShortVector lo1 = (ShortVector) q1.and((byte) 0x0F).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi1 = (ShortVector) q1.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yb = ib * QK4_NL;
                final float d0 = fp16At(b0, o + IQ4_NL_D);
                final float d1 = fp16At(b1, o + IQ4_NL_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    final ShortVector av = ShortVector.fromArray(SimdI8.S16, a, yb);
                    final ShortVector avh = ShortVector.fromArray(SimdI8.S16, a, yb + 16);
                    final int base = ti * TILE_ROWS;
                    sum[base] += (d0 * x[t0 + ti].d[ib]) * SimdI8.sum(SimdI8.pairSum(lo0.mul(av)
                            .add(hi0.mul(avh)), SimdI8.zero(SimdI8.I8)));
                    sum[base + 1] += (d1 * x[t0 + ti].d[ib]) * SimdI8.sum(SimdI8.pairSum(lo1.mul(av)
                            .add(hi1.mul(avh)), SimdI8.zero(SimdI8.I8)));
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
            for (int ib = 0; ib < nb; ib++, o += IQ4_NL_BYTES) {
                final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, o + IQ4_NL_QS);
                final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yb = ib * QK4_NL;
                final float d = fp16At(b, o + IQ4_NL_D);
                for (int ti = 0; ti < n; ti++) {
                    final short[] a = ys[t0 + ti];
                    sum[ti] += (d * x[t0 + ti].d[ib]) * SimdI8.sum(SimdI8.pairSum(
                            lo.mul(ShortVector.fromArray(SimdI8.S16, a, yb))
                                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, a, yb + 16))),
                            SimdI8.zero(SimdI8.I8)));
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK4_NL;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += IQ4_NL_BYTES) {
            final int y = ib * QK4_NL;
            final int sumi = SimdI8.sum(subDot(b, off + IQ4_NL_QS, ys, y));
            sumf += (fp16At(b, off + IQ4_NL_D) * yd[ib]) * sumi;
        }
        return sumf;
    }

    /** The 32 codebook values of the sixteen quant bytes dot the thirty-two activation lanes at {@code y}. */
    private static IntVector subDot(byte[] b, int qs, short[] ys, int y) {
        final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, qs);
        final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        return SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                        .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + QK4_NL / 2))),
                SimdI8.zero(SimdI8.I8));
    }

    @Override
    public String toString() {
        return "SimdIq4Nl{IQ4_NL x Q8_0}";
    }
}

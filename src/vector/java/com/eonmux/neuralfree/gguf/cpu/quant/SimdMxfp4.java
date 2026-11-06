package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_E;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_MXFP4;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.e8m0ToFp32Half;

/**
 * Vector API vec-dot of MXFP4 weights with Q8_0 activations, {@link Fp4VecDot#MXFP4} with the E2M1 codebook as a
 * vector lookup: the 16 quant bytes give sixteen low nibbles (elements {@code 0..15}) and sixteen shifted high
 * nibbles (elements {@code 16..31}), each selects its lane out of the {@code kvalues_mxfp4} table with
 * {@code selectFrom}, and the selected int8 values dot the pre-widened short activation lanes in
 * {@code vpmullw}, folding pairwise into int lanes ({@link SimdI8#pairSum}). The float combination
 * {@code e8m0 * d_y * sumi} per block follows {@code ggml_vec_dot_mxfp4_q8_0_generic}, so the kernel returns
 * exactly what {@link Fp4VecDot#MXFP4} returns. Immutable, stateless and thread-safe.
 */
public final class SimdMxfp4 implements CpuVecDot {

    /** The MXFP4 vec-dot. */
    public static final SimdMxfp4 INSTANCE = new SimdMxfp4();

    /** Tokens one pass of the batched row loop decodes a weight block for. */
    private static final int TOKENS = 8;

    /** {@code kvalues_mxfp4}, the doubled E2M1 values, as sixteen byte lanes: the source of {@code selectFrom}. */
    private static final ByteVector TABLE = table();

    private SimdMxfp4() {
    }

    /** The codebook as a 16-lane byte vector. */
    private static ByteVector table() {
        final byte[] t = new byte[QK_MXFP4 / 2];
        for (int i = 0; i < t.length; i++) {
            t[i] = (byte) QuantTables.kvalueFp4(i);
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
        Q8VecDot.requireWeightType(w, GGMLType.MXFP4);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_MXFP4;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += MXFP4_BYTES) {
            final int y = ib * QK_MXFP4;
            final IntVector acc = subDot(b, off + MXFP4_QS, ys, y);
            sumf += (e8m0ToFp32Half(b[off + MXFP4_E]) * yd[ib]) * SimdI8.sum(acc);
        }
        return sumf;
    }

    /** The 32 codebook values of the sixteen quant bytes at {@code qs} dot the activations at {@code y}. */
    private static IntVector subDot(byte[] b, int qs, short[] ys, int y) {
        final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, qs);
        final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        IntVector acc = SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y)),
                SimdI8.zero(SimdI8.I8));
        return SimdI8.pairSum(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + QK_MXFP4 / 2)), acc);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                        float[] y, int yOff, int yStride) {
        checkCompatible(w, x[0]);
        final short[][] ys = SimdI8.activationsAllShort(x, nTokens);
        final int cols = x[0].n;
        final float[] sum = new float[TOKENS];
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, (row, off, r) ->
                dotTokens(row, off, x, nTokens, ys, cols, y, yOff + r, yStride, sum));
    }

    /** One weight row dotted with the {@code nTokens} activation rows, {@link #TOKENS} tokens at a time. */
    private static void dotTokens(byte[] b, int off, QuantizedActivations[] x, int nTokens, short[][] ys, int cols,
                                  float[] y, int y0, int yStride, float[] sum) {
        final int nb = cols / QK_MXFP4;
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = off;
            for (int ib = 0; ib < nb; ib++, o += MXFP4_BYTES) {
                final int yb = ib * QK_MXFP4;
                final float d = e8m0ToFp32Half(b[o + MXFP4_E]);
                for (int ti = 0; ti < n; ti++) {
                    sum[ti] += (d * x[t0 + ti].d[ib]) * SimdI8.sum(subDot(b, o + MXFP4_QS, ys[t0 + ti], yb));
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    @Override
    public String toString() {
        return "SimdMxfp4{MXFP4 x Q8_0}";
    }
}

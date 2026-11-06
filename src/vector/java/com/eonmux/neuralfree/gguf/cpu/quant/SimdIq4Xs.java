package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_H;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_L;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.shortAt;

/**
 * Vector API vec-dot of IQ4_XS weights with Q8_K activations, {@link IQuantVecDot#IQ4_XS} with the
 * {@code kvalues_iq4nl} codebook as a vector lookup, the same one that serves {@link SimdIq4Nl}: the sixteen
 * quant bytes of a 32-element sub-block give sixteen low nibbles (elements {@code 0..15}) and sixteen shifted
 * high nibbles (elements {@code 16..31}), each selects its lane out of the table with {@code selectFrom}, and
 * the selected int8 values dot the pre-widened short activation lanes, the two products adding in short lanes
 * before one pairwise fold into int lanes. The sub-block's signed 6-bit scale (four low bits from
 * {@code scales_l}, two more from {@code scales_h}) scales the block sum in float exactly as
 * {@code ggml_vec_dot_iq4_xs_q8_K_generic} does, so the kernel returns what {@link IQuantVecDot#IQ4_XS} returns.
 *
 * <p>Keeping the codebook lookup and the activation lanes of a sub-block in a method of their own is what makes
 * the lookup intrinsic: written inside the super-block loop, the same source fell back to the allocating
 * template. The batched product ({@link #dotTiles}) serves eight tokens from one lookup of each sub-block.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdIq4Xs implements CpuVecDot {

    /** The IQ4_XS vec-dot. */
    public static final SimdIq4Xs INSTANCE = new SimdIq4Xs();

    /** Tokens one pass of the batched kernel decodes a weight super-block for. */
    private static final int TOKENS = 8;

    /** Elements of one sub-block, i.e. of the activation run one lookup dots. */
    private static final int SUB = 32;

    /** {@code kvalues_iq4nl} as sixteen byte lanes: the source of {@code selectFrom}. */
    private static final ByteVector TABLE = table();

    private SimdIq4Xs() {
    }

    /** The codebook as a 16-lane byte vector. */
    private static ByteVector table() {
        final byte[] t = new byte[16];
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
        Q8VecDot.requireWeightType(w, GGMLType.IQ4_XS);
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
        final byte[] row = CpuVecDotRows.buffer(w);
        final float[] acc = SimdI8.tileAccumulators(TOKENS);
        final int nb = x[0].n / QK_K;
        for (int r = 0; r < rowCount; r++) {
            CpuVecDotRows.copyRow(w, rowStart + r, row);
            dotRowTokens(row, x, nTokens, ys, nb, y, yOff + r, yStride, acc);
        }
    }

    /**
     * One weight row against {@code nTokens} activation rows, {@link #TOKENS} tokens at a time: each 32-element
     * sub-block is looked up once and dotted with every token of the tile.
     */
    private static void dotRowTokens(byte[] b, QuantizedActivations[] x, int nTokens, short[][] ys, int nb,
                                     float[] y, int y0, int yStride, float[] sum) {
        for (int t0 = 0; t0 < nTokens; t0 += TOKENS) {
            final int n = Math.min(TOKENS, nTokens - t0);
            java.util.Arrays.fill(sum, 0, n, 0.0f);
            int o = 0;
            for (int ibl = 0; ibl < nb; ibl++, o += IQ4_XS_BYTES) {
                final float d = fp16At(b, o + IQ4_XS_D);
                int h = shortAt(b, o + IQ4_XS_SCALES_H) & 0xFFFF;
                int qs = o + IQ4_XS_QS;
                final int xo = ibl * QK_K;
                for (int sub = 0; sub < QK_K / SUB; sub += 2) {
                    final int sl = b[o + IQ4_XS_SCALES_L + sub / 2] & 0xFF;
                    final int ls1 = (sl & 0x0F) | ((h << 4) & 0x30);
                    final int ls2 = (sl >>> 4) | ((h << 2) & 0x30);
                    h >>>= 4;
                    for (int ti = 0; ti < n; ti++) {
                        final short[] a = ys[t0 + ti];
                        final float d4d8 = d * x[t0 + ti].d[ibl];
                        sum[ti] += (d4d8 * (ls1 - 32)) * subDot(b, qs, a, xo + sub * SUB);
                        sum[ti] += (d4d8 * (ls2 - 32)) * subDot(b, qs + 16, a, xo + (sub + 1) * SUB);
                    }
                    qs += 32;
                }
            }
            for (int ti = 0; ti < n; ti++) {
                y[y0 + (t0 + ti) * yStride] = sum[ti];
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int ibl = 0; ibl < nb; ibl++, off += IQ4_XS_BYTES) {
            final float d4d8 = fp16At(b, off + IQ4_XS_D) * yd[ibl];
            int h = shortAt(b, off + IQ4_XS_SCALES_H) & 0xFFFF;
            int qs = off + IQ4_XS_QS;
            final int o = ibl * QK_K;
            for (int sub = 0; sub < QK_K / SUB; sub += 2) {
                final int sl = b[off + IQ4_XS_SCALES_L + sub / 2] & 0xFF;
                final int ls1 = (sl & 0x0F) | ((h << 4) & 0x30);
                final int ls2 = (sl >>> 4) | ((h << 2) & 0x30);
                h >>>= 4;
                sumf += (d4d8 * (ls1 - 32)) * subDot(b, qs, ys, o + sub * SUB);
                sumf += (d4d8 * (ls2 - 32)) * subDot(b, qs + 16, ys, o + (sub + 1) * SUB);
                qs += 32;
            }
        }
        return sumf;
    }

    /**
     * The 32 codebook values of one sub-block's sixteen quant bytes, dotted with the activation lanes at
     * {@code y}: the lookup and the two 16-lane products of {@link SimdIq4Nl} in a method small enough for the
     * JIT to keep the {@code selectFrom} intrinsic.
     */
    private static int subDot(byte[] b, int qs, short[] ys, int y) {
        final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, qs);
        final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        return SimdI8.sum(SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                        .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))),
                SimdI8.zero(SimdI8.I8)));
    }

    @Override
    public String toString() {
        return "SimdIq4Xs{IQ4_XS x Q8_K}";
    }
}

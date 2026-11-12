package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QL;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q6_K weights with Q8_K activations, {@link KQuantVecDot#Q6_K} with the unpack and the scaled
 * 16-element sub-block dots in int16 lanes: per 128-element half of a super-block, each 16-byte slice {@code t} of
 * the low and high 32 {@code ql} bytes widens to short lanes with its 16 {@code qh} bytes, and quarter {@code q} of
 * the slice decodes to sub-block {@code 8c + 2q + t} ({@code (ql nibble | qh pair << 4) - 32}). A sub-block's
 * decodes multiply the pre-widened activation lanes in {@code vpmullw}, fold pairwise into int lanes and are scaled
 * by their signed 8-bit scale there, so one int accumulator carries the 16 sub-blocks of a super-block and the
 * horizontal sum runs once per super-block. The float combination {@code sumf += (d_x * d_y) * isum} follows
 * {@code ggml_vec_dot_q6_K_q8_K_generic}, so the kernel returns what {@code KQuantVecDot.Q6_K} returns.
 *
 * <p>The batched product ({@link SimdKTiles}, form {@link SimdKTiles.Form#SIGNED_16}) unpacks each weight row once
 * and multiplies it with interleaved tiles of eight tokens, one int lane per token. The vector code keeps the rules
 * of {@link SimdI8}.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ6_K implements CpuVecDot {

    /** The Q6_K vec-dot. */
    public static final SimdQ6_K INSTANCE = new SimdQ6_K();

    /** The sub-block structure of the token-tile product. */
    private static final SimdKTiles.Form FORM = SimdKTiles.Form.SIGNED_16;

    /** The two low bits of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 32} in every byte, for {@link Q8VecDot#subtractBias}. */
    private static final long BIAS_32 = 0x6060606060606060L;

    /** The low-nibble mask of the single-token dot's short lanes, a constant so no call broadcasts it. */
    private static final ShortVector LOW4 = ShortVector.broadcast(SimdI8.S16, (short) 0x0F);

    /** The mask of the two high bits once shifted to bits 4-5, a constant so no call broadcasts it. */
    private static final ShortVector HIGH2 = ShortVector.broadcast(SimdI8.S16, (short) 0x30);

    private SimdQ6_K() {
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
        CpuVecDotRows.check(w, rowStart, rowCount, y, yOff);
        final short[] ys = SimdI8.activationsShort(x);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, buf);
            y[yOff + r] = dot(buf, 0, x, ys);
        }
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q6_K);
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
        SimdKTiles.product(w, rowStart, rowCount, SimdI8.tokenTiles(x, nTokens, SimdKTiles.groupSpan(FORM)), 0,
                nTokens, y, yOff, yStride, FORM, SimdQ6_K::unpackRow);
    }

    @Override
    public TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
        SimdI8.checkTileTokens(x, nTokens, GGMLType.Q8_K);
        return nTokens == 0 ? null : SimdI8.tokenTiles(x, nTokens, SimdKTiles.groupSpan(FORM));
    }

    /**
     * Token panels of whole pairs of 8-token tiles: a panel of a wide row (9728 columns) then holds 16 tokens
     * instead of a single tile, so each weight row a task decodes serves twice the tokens.
     */
    @Override
    public int tilePanelTokens() {
        return 2 * SimdI8.TILE_TOKENS;
    }

    @Override
    public int tileTokens() {
        return SimdI8.TILE_TOKENS;
    }

    @Override
    public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                         QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (!(tiles instanceof SimdI8.TokenTiles)) {
            CpuVecDot.super.dotTiles(w, rowStart, rowCount, tiles, x, tokenStart, nTokens, y, yOff, yStride);
            return;
        }
        SimdKTiles.checkedProduct(w, GGMLType.Q6_K, rowStart, rowCount, (SimdI8.TokenTiles) tiles, tokenStart,
                nTokens, y, yOff, yStride, FORM, SimdQ6_K::unpackRow);
    }

    /**
     * One row's signed quants as int16 element pairs, its sixteen signed sub-block scales per super-block and its
     * {@code d} ({@link SimdKTiles.Rows}).
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] q = Q8VecDot.scratch().quants;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q6_K_BYTES) {
            unpack(b, o, q);
            SimdKTiles.packPairs(q, pairs, i * (QK_K / 2));
            for (int g = 0; g < QK_K / 16; g++) {
                scales[(QK_K / 16) * i + g] = b[o + Q6_K_SCALES + g];
            }
            d[2 * i] = fp16At(b, o + Q6_K_D);
            d[2 * i + 1] = 0;
        }
    }

    /**
     * One row against {@code x} with its short lanes {@code ys}. The quants unpack in registers: per 32-element
     * slice of a half super-block, its 32 {@code ql} bytes and 16 {@code qh} bytes widen to short lanes once, and
     * each 16-element sub-block takes its nibble and its two high bits from them by shifts, so no quant passes
     * through memory between the unpack and the product. The products take the unsigned 6-bit values, and each
     * super-block's integer sum then drops {@code 32 * sum_j scale_j * bsum_j} over its sixteen 16-element sums
     * {@code bsums}, the same integer as the sum of the {@code (q - 32) * y} products. One loop body serves the
     * low-nibble pair of sub-blocks and the high-nibble pair (shifts by 0 or 4): C2 parses each Vector API call into
     * a few hundred nodes before it intrinsifies it, and a method whose calls approach its inlining node cutoff
     * loses the intrinsics past the cutoff, allocating a vector per operation; two product paths per loop body
     * keep the method inside it.
     */
    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q6_K_BYTES) {
            final int xo = i * QK_K;
            IntVector acc = IntVector.zero(SimdI8.I8);
            for (int c = 0; c < QK_K / 128; c++) {
                final int ql = off + Q6_K_QL + 64 * c;
                final int qh = off + Q6_K_QH + 32 * c;
                final int sc = off + Q6_K_SCALES + 8 * c;
                final int yo = xo + 128 * c;
                for (int t = 0; t < 2; t++) {
                    final ShortVector a = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, ql + 16 * t)
                            .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                    final ShortVector e = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, ql + 32 + 16 * t)
                            .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                    final ShortVector h4 = ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, qh + 16 * t)
                            .convertShape(VectorOperators.B2S, SimdI8.S16, 0)).lanewise(VectorOperators.LSHL, 4);
                    for (int k = 0; k < 2; k++) {
                        final IntVector pa = (IntVector) a.lanewise(VectorOperators.LSHR, 4 * k).and(LOW4)
                                .or(h4.lanewise(VectorOperators.LSHR, 4 * k).and(HIGH2))
                                .mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 64 * k + 16 * t))
                                .reinterpretShape(SimdI8.I8, 0);
                        final IntVector pe = (IntVector) e.lanewise(VectorOperators.LSHR, 4 * k).and(LOW4)
                                .or(h4.lanewise(VectorOperators.LSHR, 4 * k + 2).and(HIGH2))
                                .mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 32 + 64 * k + 16 * t))
                                .reinterpretShape(SimdI8.I8, 0);
                        acc = pa.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                                .add(pa.lanewise(VectorOperators.ASHR, 16)).mul(b[sc + 4 * k + t])
                                .add(pe.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                                        .add(pe.lanewise(VectorOperators.ASHR, 16)).mul(b[sc + 4 * k + 2 + t]))
                                .add(acc);
                    }
                }
            }
            int bias = 0;
            for (int j = 0; j < QK_K / 16; j++) {
                bias += b[off + Q6_K_SCALES + j] * x.bsums[i * (QK_K / 16) + j];
            }
            sumf += fp16At(b, off + Q6_K_D) * yd[i]
                    * (((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD) - 32 * bias);
        }
        return sumf;
    }

    /**
     * Unpacks the 256 signed 6-bit quants of one super-block into {@code q}: quarter {@code q} of slice
     * {@code t} in half {@code c} is sub-block {@code 8c + 2q + t}, its value {@code (ql nibble | qh pair << 4)
     * - 32}. Eight quants at a time with the 64-bit SWAR of {@link Q8VecDot#subtractBias}: the byte-lane form
     * of the same unpack measured slower and allocated, so the SWAR form stays.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int ql = off + Q6_K_QL;
        final int qh = off + Q6_K_QH;
        for (int c = 0; c < QK_K / 128; c++) {
            final int a = 128 * c;
            for (int t = 0; t < 4; t++) {
                final long lo = longAt(b, ql + 64 * c + 8 * t);
                final long hi = longAt(b, ql + 64 * c + 32 + 8 * t);
                final long h = longAt(b, qh + 32 * c + 8 * t);
                Q8VecDot.putLong(q, a + 8 * t,
                        Q8VecDot.subtractBias((lo & Q8VecDot.LOW_NIBBLES) | ((h & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 32 + 8 * t,
                        Q8VecDot.subtractBias((hi & Q8VecDot.LOW_NIBBLES) | (((h >>> 2) & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 64 + 8 * t,
                        Q8VecDot.subtractBias(((lo >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 4) & LOW_PAIRS) << 4),
                                BIAS_32));
                Q8VecDot.putLong(q, a + 96 + 8 * t,
                        Q8VecDot.subtractBias(((hi >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 6) & LOW_PAIRS) << 4),
                                BIAS_32));
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ6_K{Q6_K x Q8_K}";
    }
}

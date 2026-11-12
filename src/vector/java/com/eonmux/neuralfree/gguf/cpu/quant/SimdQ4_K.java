package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
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
 * short lanes and the sub-block dots in int16 lanes: the 32 quant bytes of each 64-element group widen to short
 * lanes once and give their low nibbles to the group's first sub-block and their high nibbles to its second, each
 * 32-element sub-block dots the matching activation sub-block (two 16-lane products added in short lanes, folded
 * pairwise into int lanes) and its sum joins one int accumulator scaled by the 6-bit
 * sub-block scale, so the horizontal sum runs once per token instead of once per sub-block. The scale unpack,
 * the mins sum over {@code bsums} and the float combination ({@code sums += d * isum}, {@code sumf -= dmin *
 * sumi}, result {@code sumf + sums}) follow {@code ggml_vec_dot_q4_K_q8_K_generic} exactly, so the kernel
 * returns what {@code KQuantVecDot.Q4_K} returns.
 *
 * <p>The batched product ({@link SimdKTiles}, form {@link SimdKTiles.Form#MINS_32}) unpacks each weight row once
 * and multiplies it with interleaved tiles of eight tokens, one int lane per token. The vector code keeps the rules
 * of {@link SimdI8}.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_K implements CpuVecDot {

    /** The Q4_K vec-dot. */
    public static final SimdQ4_K INSTANCE = new SimdQ4_K();

    /** The sub-block structure of the token-tile product. */
    private static final SimdKTiles.Form FORM = SimdKTiles.Form.MINS_32;

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
        CpuVecDotRows.check(w, rowStart, rowCount, y, yOff);
        final short[] ys = SimdI8.activationsShort(x);
        final short[] pairs = SimdI8.bsumPairs(x);
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, buf);
            y[yOff + r] = dotRow(buf, 0, x, ys, pairs, sm);
        }
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
        SimdKTiles.product(w, rowStart, rowCount, SimdI8.tokenTiles(x, nTokens, SimdKTiles.groupSpan(FORM)), 0,
                nTokens, y, yOff, yStride, FORM, SimdQ4_K::unpackRow);
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
        SimdKTiles.checkedProduct(w, GGMLType.Q4_K, rowStart, rowCount, (SimdI8.TokenTiles) tiles, tokenStart,
                nTokens, y, yOff, yStride, FORM, SimdQ4_K::unpackRow);
    }

    /**
     * One row's quants as int16 element pairs, its 6-bit sub-block scales and mins, and {@code d} / {@code dmin}
     * ({@link SimdKTiles.Rows}).
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q4_K_BYTES) {
            unpack(b, o, q);
            KQuantVecDot.unpackScalesMins(b, o + Q4_K_SCALES, sm);
            SimdKTiles.packPairs(q, pairs, i * (QK_K / 2));
            for (int g = 0; g < 8; g++) {
                scales[8 * i + g] = sm[g];
                mins[8 * i + g] = sm[8 + g];
            }
            d[2 * i] = fp16At(b, o + Q4_K_D);
            d[2 * i + 1] = fp16At(b, o + Q4_K_DMIN);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        return dotRow(b, off, x, ys, SimdI8.bsumPairs(x), Q8VecDot.scratch().scales);
    }

    /**
     * One row against {@code x} with its short lanes {@code ys} and group-sum pairs {@code pairs}; {@code sm} is the
     * caller's scale scratch. The nibbles unpack in registers: the 32 quant bytes of a 64-element group widen to
     * two 16-lane short vectors once, whose low nibbles are the group's first sub-block and whose high nibbles its
     * second, so no quant passes through memory between the unpack and the product. One loop body serves both
     * nibbles (a shift by 0 or 4): C2 parses each Vector API call into a few hundred nodes before it intrinsifies
     * it, and a method whose calls approach its inlining node cutoff loses the intrinsics past the cutoff, allocating
     * a vector per operation; one product path per sub-block keeps the method well inside the cutoff.
     */
    private static float dotRow(byte[] b, int off, QuantizedActivations x, short[] ys, short[] pairs, byte[] sm) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q4_K_BYTES) {
            KQuantVecDot.unpackScalesMins(b, off + Q4_K_SCALES, sm);
            final int xo = i * QK_K;
            IntVector acc = IntVector.zero(SimdI8.I8);
            for (int j = 0; j < QK_K / 64; j++) {
                final int qs = off + Q4_K_QS + 32 * j;
                final int yo = xo + 64 * j;
                final ShortVector w0 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, qs)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector w1 = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, qs + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                for (int h = 0; h < 2; h++) {
                    final IntVector p = (IntVector) w0.lanewise(VectorOperators.LSHR, 4 * h).and((short) 0x0F)
                            .mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 32 * h))
                            .add(w1.lanewise(VectorOperators.LSHR, 4 * h).and((short) 0x0F)
                                    .mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 32 * h + 16)))
                            .reinterpretShape(SimdI8.I8, 0);
                    acc = p.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(p.lanewise(VectorOperators.ASHR, 16)).mul(sm[2 * j + h]).add(acc);
                }
            }
            final float d = fp16At(b, off + Q4_K_D) * yd[i];
            sums += d * ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
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

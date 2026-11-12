package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q5_K weights with Q8_K activations, {@link KQuantVecDot#Q5_K} with the nibble unpack in
 * byte lanes and the sub-block dots in int16 lanes: the 256 five-bit quants of a super-block unpack into the
 * thread's scratch row (each element's fifth bit comes from the matching {@code qh} byte), each 32-element
 * sub-block dots the matching activation sub-block (two 16-lane products added in short lanes, folded pairwise
 * into int lanes) and its sum joins one int accumulator scaled by the 6-bit sub-block scale. The scale unpack,
 * the mins sum over {@code bsums} and the float combination follow {@code ggml_vec_dot_q5_K_q8_K_generic}, so
 * the kernel returns what {@link KQuantVecDot#Q5_K} returns.
 *
 * <p>The batched product ({@link SimdKTiles}, form {@link SimdKTiles.Form#MINS_32_FOLD_8}) unpacks each weight row
 * once and multiplies it with interleaved tiles of eight tokens, one int lane per token. The vector code keeps the
 * rules of {@link SimdI8}.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ5_K implements CpuVecDot {

    /** The Q5_K vec-dot. */
    public static final SimdQ5_K INSTANCE = new SimdQ5_K();

    /** The sub-block structure of the token-tile product. */
    private static final SimdKTiles.Form FORM = SimdKTiles.Form.MINS_32_FOLD_8;

    /** {@code 0x01} in every byte. */
    private static final long BYTE_ONES = 0x0101010101010101L;

    private SimdQ5_K() {
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
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, buf);
            y[yOff + r] = dot(buf, 0, x, ys, pairs);
        }
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q5_K);
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
                nTokens, y, yOff, yStride, FORM, SimdQ5_K::unpackRow);
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
        SimdKTiles.checkedProduct(w, GGMLType.Q5_K, rowStart, rowCount, (SimdI8.TokenTiles) tiles, tokenStart,
                nTokens, y, yOff, yStride, FORM, SimdQ5_K::unpackRow);
    }

    /**
     * One row's quants as int16 element pairs, its 6-bit sub-block scales and mins, and {@code d} / {@code dmin}
     * ({@link SimdKTiles.Rows}).
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q5_K_BYTES) {
            unpack(b, o, q);
            KQuantVecDot.unpackScalesMins(b, o + Q5_K_SCALES, sm);
            SimdKTiles.packPairs(q, pairs, i * (QK_K / 2));
            for (int g = 0; g < 8; g++) {
                scales[8 * i + g] = sm[g];
                mins[8 * i + g] = sm[8 + g];
            }
            d[2 * i] = fp16At(b, o + Q5_K_D);
            d[2 * i + 1] = fp16At(b, o + Q5_K_DMIN);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        return dot(b, off, x, ys, SimdI8.bsumPairs(x));
    }

    /** One row against {@code x} with its short lanes {@code ys} and group-sum pairs {@code pairs}. */
    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys, short[] pairs) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q5_K_BYTES) {
            unpack(b, off, q);
            KQuantVecDot.unpackScalesMins(b, off + Q5_K_SCALES, sm);
            final int xo = i * QK_K;
            IntVector acc = IntVector.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 32; g++) {
                final int yo = xo + 32 * g;
                final ShortVector lo = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final IntVector lanes = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, ys, yo))
                        .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 16)))
                        .reinterpretShape(SimdI8.I8, 0);
                acc = lanes.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                        .add(lanes.lanewise(VectorOperators.ASHR, 16)).mul(sm[g]).add(acc);
            }
            final float d = fp16At(b, off + Q5_K_D) * yd[i];
            sums += d * ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
            final float dmin = fp16At(b, off + Q5_K_DMIN) * yd[i];
            sumf -= dmin * SimdI8.minsDot(pairs, i * (QK_K / 32), sm);
        }
        return sumf + sums;
    }

    /**
     * Unpacks the 256 five-bit quants of one super-block into {@code q}: group {@code j} of 64 elements takes
     * the low nibbles of its 32 quant bytes as elements {@code 64j .. 64j + 31} and their high nibbles as
     * {@code 64j + 32 .. 64j + 63}, and the same bytes of {@code qh} supply bit {@code 2j} (low half) and
     * {@code 2j + 1} (high half) as each element's fifth bit.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int qh = off + Q5_K_QH;
        for (int j = 0; j < QK_K / 64; j++) {
            final int q4 = off + Q5_K_QS + 32 * j;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q4 + 8 * t);
                final long h = longAt(b, qh + 8 * t);
                final long hLow = ((h >>> (2 * j)) & BYTE_ONES) << 4;
                final long hHigh = ((h >>> (2 * j + 1)) & BYTE_ONES) << 4;
                Q8VecDot.putLong(q, 64 * j + 8 * t, (v & Q8VecDot.LOW_NIBBLES) | hLow);
                Q8VecDot.putLong(q, 64 * j + 32 + 8 * t, ((v >>> 4) & Q8VecDot.LOW_NIBBLES) | hHigh);
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ5_K{Q5_K x Q8_K}";
    }
}

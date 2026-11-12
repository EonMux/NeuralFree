package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.KQuantBlockDecoders;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_HMASK;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q3_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q3_K weights with Q8_K activations, {@link KQuantVecDot#Q3_K} with the three-bit codes
 * unpacked into the thread's scratch row and the 16-element group dots in int16 lanes: the low two bits come
 * from the two-bit slices of each 8-byte {@code qs} group and the third bit from the matching {@code hmask}
 * byte, giving {@code (low | bit << 2) - 4}; each group's quants widen to short lanes, multiply the pre-widened
 * activation lanes, fold pairwise into int lanes and are scaled by the group's signed 6-bit scale there, so one
 * int accumulator carries the sixteen groups of a super-block. The scale unpack and the float combination
 * follow {@code ggml_vec_dot_q3_K_q8_K_generic}, so the kernel returns what {@link KQuantVecDot#Q3_K} returns.
 *
 * <p>The batched product ({@link SimdKTiles}, form {@link SimdKTiles.Form#SIGNED_16}) unpacks each weight row once
 * and multiplies it with interleaved tiles of eight tokens, one int lane per token. The vector code keeps the rules
 * of {@link SimdI8}.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ3_K implements CpuVecDot {

    /** The Q3_K vec-dot. */
    public static final SimdQ3_K INSTANCE = new SimdQ3_K();

    /** The sub-block structure of the token-tile product. */
    private static final SimdKTiles.Form FORM = SimdKTiles.Form.SIGNED_16;

    /** {@code 0x01} in every byte. */
    private static final long BYTE_ONES = 0x0101010101010101L;
    /** Bits 0-1 of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 4} in every byte: maps 3-bit values {@code 0..7} to {@code -4..3}. */
    private static final long BIAS_4 = 0x7C7C7C7C7C7C7C7CL;

    private SimdQ3_K() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q3_K);
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
                nTokens, y, yOff, yStride, FORM, SimdQ3_K::unpackRow);
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
        SimdKTiles.checkedProduct(w, GGMLType.Q3_K, rowStart, rowCount, (SimdI8.TokenTiles) tiles, tokenStart,
                nTokens, y, yOff, yStride, FORM, SimdQ3_K::unpackRow);
    }

    /**
     * One row's signed quants as int16 element pairs, its sixteen signed sub-block scales ({@code scale - 32}) per
     * super-block and its {@code d} ({@link SimdKTiles.Rows}).
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] sc = Q8VecDot.scratch().scales;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q3_K_BYTES) {
            unpack(b, o, q);
            KQuantBlockDecoders.q3kScales(b, o + Q3_K_SCALES, sc);
            SimdKTiles.packPairs(q, pairs, i * (QK_K / 2));
            for (int g = 0; g < QK_K / 16; g++) {
                scales[(QK_K / 16) * i + g] = sc[g] - 32;
            }
            d[2 * i] = fp16At(b, o + Q3_K_D);
            d[2 * i + 1] = 0;
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] scales = Q8VecDot.scratch().scales;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q3_K_BYTES) {
            unpack(b, off, q);
            KQuantBlockDecoders.q3kScales(b, off + Q3_K_SCALES, scales);
            final int xo = i * QK_K;
            IntVector acc = IntVector.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final IntVector lanes = (IntVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0)
                        .mul(ShortVector.fromArray(SimdI8.S16, ys, xo + 16 * g))
                        .reinterpretShape(SimdI8.I8, 0);
                acc = lanes.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                        .add(lanes.lanewise(VectorOperators.ASHR, 16)).mul(scales[g] - 32).add(acc);
            }
            final float d = fp16At(b, off + Q3_K_D) * yd[i];
            sumf += d * ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
        }
        return sumf;
    }

    /**
     * Unpacks the 256 signed 3-bit quants of one super-block into {@code q}: element {@code e = 128c + 32s +
     * 8t + l} is {@code ((qs[32c + 8t + l] >> 2s) & 3) | ((hmask[8t + l] >> (4c + s)) & 1) << 2}, minus 4.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int hm = off + Q3_K_HMASK;
        for (int c = 0; c < QK_K / 128; c++) {
            final int q3 = off + Q3_K_QS + 32 * c;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q3 + 8 * t);
                final long h = longAt(b, hm + 8 * t);
                for (int s = 0; s < 4; s++) {
                    final long low = (v >>> (2 * s)) & LOW_PAIRS;
                    final long high = ((h >>> (4 * c + s)) & BYTE_ONES) << 2;
                    Q8VecDot.putLong(q, 128 * c + 32 * s + 8 * t, Q8VecDot.subtractBias(low | high, BIAS_4));
                }
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ3_K{Q3_K x Q8_K}";
    }
}

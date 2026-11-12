package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_S_SIGNS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XS_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ2_XXS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_SCALES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_S_SIGNS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ3_XXS_SAS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.intAt;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of the grid-coded i-quants ({@code IQ2_XXS}, {@code IQ2_XS}, {@code IQ2_S}, {@code IQ3_XXS},
 * {@code IQ3_S}) with Q8_K activations, {@link IQuantVecDot} with the grid decode done once per super-block and
 * the products in int16 lanes. Each 8-element grid entry (two 4-element entries for the 3-bit kinds) is a packed
 * row of unsigned magnitudes, all of them in {@code [1, 62]}; its sign bits select a byte mask from
 * {@link #SIGN_MASKS}, and {@code (grid ^ mask) + (mask & 0x01..01)} negates the selected bytes in one 64-bit step
 * (no byte is zero, so no lane carries into the next). The 256 signed quants of a super-block land in the thread's
 * scratch row with one scale per 16 elements ({@code 2 * ls + 1}; a kind whose scale covers 32 elements repeats it),
 * and the 16-element groups then dot the pre-widened activation lanes exactly as {@link SimdQ3_K} does: every
 * product fits a short, the pairwise fold and the scale multiply run in int lanes, and one int accumulator carries
 * the super-block. The integer sum is the scalar kernel's {@code bsum}, and the float combination
 * ({@code sumf += (d_x * d_y) * bsum}, then {@code 0.125f * sumf}, {@code 0.25f * sumf} or {@code sumf}) follows
 * {@code ggml_vec_dot_iq*_q8_K_generic}, so every kind returns what its scalar kernel returns.
 *
 * <p>The batched product ({@link SimdKTiles}, form {@link SimdKTiles.Form#SIGNED_16}) unpacks each weight row once
 * into int16 element pairs with the same scales and multiplies it with interleaved tiles of eight tokens: four
 * products of a magnitude up to 62 and an int8 activation fit a short. The vector code keeps the rules of
 * {@link SimdI8}. Immutable, stateless and thread-safe.</p>
 */
public final class SimdIqGrid implements CpuVecDot {

    /** The IQ2_XXS vec-dot. */
    public static final SimdIqGrid IQ2_XXS = new SimdIqGrid(Kind.IQ2_XXS);
    /** The IQ2_XS vec-dot. */
    public static final SimdIqGrid IQ2_XS = new SimdIqGrid(Kind.IQ2_XS);
    /** The IQ2_S vec-dot. */
    public static final SimdIqGrid IQ2_S = new SimdIqGrid(Kind.IQ2_S);
    /** The IQ3_XXS vec-dot. */
    public static final SimdIqGrid IQ3_XXS = new SimdIqGrid(Kind.IQ3_XXS);
    /** The IQ3_S vec-dot. */
    public static final SimdIqGrid IQ3_S = new SimdIqGrid(Kind.IQ3_S);

    /** The sub-block structure of the token-tile product. */
    private static final SimdKTiles.Form FORM = SimdKTiles.Form.SIGNED_16;

    /** {@code 0x01} in every byte. */
    private static final long BYTE_ONES = 0x0101010101010101L;

    /** For each sign byte {@code s}, {@code 0xFF} in byte {@code j} where bit {@code j} of {@code s} is set. */
    static final long[] SIGN_MASKS = signMasks();

    /** The grid kinds, their weight types, block sizes and the constant their scalar kernel scales the sum by. */
    enum Kind {
        IQ2_XXS(GGMLType.IQ2_XXS, IQ2_XXS_BYTES, IQ2_XXS_D, 0.125f),
        IQ2_XS(GGMLType.IQ2_XS, IQ2_XS_BYTES, IQ2_XS_D, 0.125f),
        IQ2_S(GGMLType.IQ2_S, IQ2_S_BYTES, IQ2_S_D, 0.125f),
        IQ3_XXS(GGMLType.IQ3_XXS, IQ3_XXS_BYTES, IQ3_XXS_D, 0.25f),
        IQ3_S(GGMLType.IQ3_S, IQ3_S_BYTES, IQ3_S_D, 1f);

        final GGMLType type;
        final int blockBytes;
        final int dField;
        final float post;

        Kind(GGMLType type, int blockBytes, int dField, float post) {
            this.type = type;
            this.blockBytes = blockBytes;
            this.dField = dField;
            this.post = post;
        }
    }

    private final Kind kind;

    private SimdIqGrid(Kind kind) {
        this.kind = kind;
    }

    /** The vec-dot of {@code type}, or null when this class has none. */
    static SimdIqGrid forType(GGMLType type) {
        for (SimdIqGrid k : new SimdIqGrid[]{IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S}) {
            if (k.kind.type == type) {
                return k;
            }
        }
        return null;
    }

    private static long[] signMasks() {
        final long[] m = new long[256];
        for (int s = 0; s < 256; s++) {
            long v = 0;
            for (int j = 0; j < 8; j++) {
                if (((s >>> j) & 1) != 0) {
                    v |= 0xFFL << (8 * j);
                }
            }
            m[s] = v;
        }
        return m;
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

    private void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, kind.type);
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
                nTokens, y, yOff, yStride, FORM, this::unpackRow, kind.post);
    }

    @Override
    public TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
        SimdI8.checkTileTokens(x, nTokens, GGMLType.Q8_K);
        return nTokens == 0 ? null : SimdI8.tokenTiles(x, nTokens, SimdKTiles.groupSpan(FORM));
    }

    /** Token panels of whole pairs of 8-token tiles, as the K-quants. */
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
        SimdKTiles.checkedProduct(w, kind.type, rowStart, rowCount, (SimdI8.TokenTiles) tiles, tokenStart, nTokens,
                y, yOff, yStride, FORM, this::unpackRow, kind.post);
    }

    /**
     * One row's signed quants as int16 element pairs, its sixteen scales per super-block and its {@code d}
     * ({@link SimdKTiles.Rows}).
     */
    private void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] sc = Q8VecDot.scratch().scales;
        int o = 0;
        for (int i = 0; i < nb; i++, o += kind.blockBytes) {
            unpack(kind, b, o, q, sc);
            SimdKTiles.packPairs(q, pairs, i * (QK_K / 2));
            for (int g = 0; g < QK_K / 16; g++) {
                scales[(QK_K / 16) * i + g] = sc[g];
            }
            d[2 * i] = fp16At(b, o + kind.dField);
            d[2 * i + 1] = 0;
        }
    }

    /** One row against {@code x} with its short lanes {@code ys}. */
    private float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final Kind k = kind;
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] scales = Q8VecDot.scratch().scales;
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += k.blockBytes) {
            unpack(k, b, off, q, scales);
            final int xo = i * QK_K;
            IntVector acc = IntVector.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final IntVector lanes = (IntVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0)
                        .mul(ShortVector.fromArray(SimdI8.S16, ys, xo + 16 * g))
                        .reinterpretShape(SimdI8.I8, 0);
                acc = lanes.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                        .add(lanes.lanewise(VectorOperators.ASHR, 16)).mul(scales[g]).add(acc);
            }
            final float d = fp16At(b, off + k.dField) * yd[i];
            sumf += d * ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
        }
        return k.post == 1f ? sumf : k.post * sumf;
    }

    /**
     * Decodes the super-block of {@code kind} at {@code b[off..)}: its 256 signed quants into {@code q} in element
     * order and the scale of each 16-element group ({@code 2 * ls + 1}) into {@code scales[0..16)}.
     */
    static void unpack(Kind kind, byte[] b, int off, byte[] q, byte[] scales) {
        switch (kind) {
            case IQ2_XXS:
                unpackIq2Xxs(b, off, q, scales);
                return;
            case IQ2_XS:
                unpackIq2Xs(b, off, q, scales);
                return;
            case IQ2_S:
                unpackIq2S(b, off, q, scales);
                return;
            case IQ3_XXS:
                unpackIq3Xxs(b, off, q, scales);
                return;
            case IQ3_S:
                unpackIq3S(b, off, q, scales);
                return;
            default:
                throw new IllegalStateException(kind.name());
        }
    }

    /** The eight magnitudes of {@code grid} with the bytes that {@code signs} selects negated. */
    private static long signed(long grid, int signs) {
        final long m = SIGN_MASKS[signs];
        return (grid ^ m) + (m & BYTE_ONES);
    }

    /** {@code dotIq2Xxs}: per 32 elements four 8-entry grid rows, seven sign bits each, one scale. */
    private static void unpackIq2Xxs(byte[] b, int off, byte[] q, byte[] scales) {
        int q2 = off + IQ2_XXS_QS;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q2 += 8) {
            final long aux = longAt(b, q2);
            final int aux0 = (int) aux;
            final int aux1 = (int) (aux >>> 32);
            final byte ls = (byte) (2 * (aux1 >>> 28) + 1);
            scales[2 * ib32] = ls;
            scales[2 * ib32 + 1] = ls;
            for (int l = 0; l < 4; ++l) {
                Q8VecDot.putLong(q, 32 * ib32 + 8 * l, signed(QuantTables.iq2xxsGrid((aux0 >>> (8 * l)) & 0xFF),
                        QuantTables.ksignsIq2xs((aux1 >>> (7 * l)) & 127)));
            }
        }
    }

    /** {@code dotIq2Xs}: per 32 elements four 9-bit grid indices with seven sign bits, a scale per 16. */
    private static void unpackIq2Xs(byte[] b, int off, byte[] q, byte[] scales) {
        int q2 = off + IQ2_XS_QS;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q2 += 8) {
            final int sc = b[off + IQ2_XS_SCALES + ib32] & 0xFF;
            scales[2 * ib32] = (byte) (2 * (sc & 0xf) + 1);
            scales[2 * ib32 + 1] = (byte) (2 * (sc >>> 4) + 1);
            final long q4 = longAt(b, q2);
            for (int l = 0; l < 4; ++l) {
                final int v = (int) (q4 >>> (16 * l)) & 0xFFFF;
                Q8VecDot.putLong(q, 32 * ib32 + 8 * l,
                        signed(QuantTables.iq2xsGrid(v & 511), QuantTables.ksignsIq2xs(v >>> 9)));
            }
        }
    }

    /** {@code dotIq2S}: per 32 elements four 10-bit grid indices (two bits from {@code qh}), explicit sign bytes. */
    private static void unpackIq2S(byte[] b, int off, byte[] q, byte[] scales) {
        int qs = off + IQ2_S_QS;
        int signs = off + IQ2_S_SIGNS;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, qs += 4, signs += 4) {
            final int sc = b[off + IQ2_S_SCALES + ib32] & 0xFF;
            scales[2 * ib32] = (byte) (1 + 2 * (sc & 0xf));
            scales[2 * ib32 + 1] = (byte) (1 + 2 * (sc >>> 4));
            final int qh = b[off + IQ2_S_QH + ib32] & 0xFF;
            final int q4 = intAt(b, qs);
            final int s4 = intAt(b, signs);
            for (int l = 0; l < 4; ++l) {
                final int idx = ((q4 >>> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
                Q8VecDot.putLong(q, 32 * ib32 + 8 * l, signed(QuantTables.iq2sGrid(idx), (s4 >>> (8 * l)) & 0xFF));
            }
        }
    }

    /** {@code dotIq3Xxs}: per 32 elements eight 4-entry grid rows, seven sign bits per pair of rows, one scale. */
    private static void unpackIq3Xxs(byte[] b, int off, byte[] q, byte[] scales) {
        int q3 = off + IQ3_XXS_QS;
        int gas = off + IQ3_XXS_SAS;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, q3 += 8, gas += 4) {
            final int aux32 = intAt(b, gas);
            final byte ls = (byte) (2 * (aux32 >>> 28) + 1);
            scales[2 * ib32] = ls;
            scales[2 * ib32 + 1] = ls;
            final long idx8 = longAt(b, q3);
            for (int l = 0; l < 4; ++l) {
                final long grid1 = QuantTables.iq3xxsGrid((int) (idx8 >>> (16 * l)) & 0xFF) & 0xFFFFFFFFL;
                final long grid2 = QuantTables.iq3xxsGrid((int) (idx8 >>> (16 * l + 8)) & 0xFF) & 0xFFFFFFFFL;
                Q8VecDot.putLong(q, 32 * ib32 + 8 * l,
                        signed(grid1 | grid2 << 32, QuantTables.ksignsIq2xs((aux32 >>> (7 * l)) & 127)));
            }
        }
    }

    /**
     * {@code dotIq3S}: per 32 elements eight 9-bit grid indices (the ninth bit from {@code qh}), explicit sign
     * bytes, a scale per 32 from the nibbles of {@code scales}.
     */
    private static void unpackIq3S(byte[] b, int off, byte[] q, byte[] scales) {
        int qs = off + IQ3_S_QS;
        int signs = off + IQ3_S_SIGNS;
        final int qh = off + IQ3_S_QH;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32, qs += 8, signs += 4) {
            final int sc = b[off + IQ3_S_SCALES + ib32 / 2] & 0xFF;
            final byte ls = (byte) (2 * ((ib32 & 1) == 0 ? sc & 0xf : sc >>> 4) + 1);
            scales[2 * ib32] = ls;
            scales[2 * ib32 + 1] = ls;
            final long idx8 = longAt(b, qs);
            final int h = b[qh + ib32] & 0xFF;
            final int s4 = intAt(b, signs);
            for (int l = 0; l < 4; ++l) {
                final int i1 = ((int) (idx8 >>> (16 * l)) & 0xFF) | ((h << (8 - 2 * l)) & 256);
                final int i2 = ((int) (idx8 >>> (16 * l + 8)) & 0xFF) | ((h << (7 - 2 * l)) & 256);
                final long grid = (QuantTables.iq3sGrid(i1) & 0xFFFFFFFFL)
                        | (QuantTables.iq3sGrid(i2) & 0xFFFFFFFFL) << 32;
                Q8VecDot.putLong(q, 32 * ib32 + 8 * l, signed(grid, (s4 >>> (8 * l)) & 0xFF));
            }
        }
    }

    @Override
    public String toString() {
        return "SimdIqGrid{" + kind.type + " x Q8_K}";
    }
}

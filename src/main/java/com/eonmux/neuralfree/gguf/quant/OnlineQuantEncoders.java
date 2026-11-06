package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.F8B32_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.F8B128_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_E;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP8_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP8_E;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.MXFP8_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.NVFP4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.NVFP4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_F8B128;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_F8B32;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_I8B128;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_MXFP4;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_MXFP8;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_NVFP4;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_NVFP4_SUB;

/**
 * The online weight quantizers of the vLLM load-time quantization methods: fp8 per tensor, per 128x128 block
 * and per output channel into F8_E4M3_B128 (F8_E4M3_B32 for rows that are no whole number of 128-value blocks,
 * each 32-value run carrying the scale of its 128-wide scale block), int8 per output channel ({@code int8_per_channel_weight_only} and
 * {@code experts_int8}) into I8_B128, mxfp8 into MXFP8, mxfp4 into MXFP4 and nvfp4_per_token into NVFP4. The
 * scale rules are the references' (vLLM model_executor/layers/quantization/online/):
 * <ul>
 *   <li>fp8_per_tensor: {@code scale = amax / 448} per expert; codes {@code e4m3(x * (1/scale))} (the static
 *       kernel multiplies by the reciprocal, fp8.py:84-86). A zero scale keeps every code 0.</li>
 *   <li>fp8_per_block: one scale per 128x128 tile per expert, {@code scale = clamp(amax, 1e-4) * (1/448)}, the
 *       float reciprocal the compiled {@code amax / 448} multiplies by; codes {@code e4m3(x * (1/scale))}
 *       (per_block_cast_to_fp8, deep_gemm.py:1060-1079); rows and columns past a whole tile are part of the last
 *       tile's amax, where vLLM pads with zeros.</li>
 *   <li>fp8_per_channel: {@code scale = max(amax / 448, 1/(448*512))} per row; codes {@code e4m3(x / scale)} by
 *       true division (fp8.py:89-108).</li>
 *   <li>int8 per channel: {@code scale = amax / 127} per row; codes {@code round(x / scale)} to nearest even,
 *       clamped to [-127, 127] (int8.py:83-89); a zero row keeps every code 0.</li>
 *   <li>mxfp8: per 32-element run, the exponent {@code clamp(ceil(log2(max(amax, FLT_MIN)/448)) + 127, 0, 254)}
 *       as the E8M0 byte and codes {@code e4m3(x / 2^e)} ({@link ActFakeQuant#mxfp8Exponent},
 *       mxfp8_utils.py:55-70).</li>
 *   <li>mxfp4: per 32-element run, the OCP MX scale of vLLM's Triton quantizer (mxfp4_utils.py
 *       {@code _compute_quant_and_scale}): {@code amax} rounded to a power of two at three quarters of a binade,
 *       the unbiased exponent {@code clamp(floor(log2(r)) - 2, -127, 127)}, and E2M1 codes of {@code x * 2^-e}
 *       by the kernel's bit arithmetic ({@link #mxfp4Code}).</li>
 *   <li>nvfp4_per_token: per expert the global scale {@code gs = (1 / clamp(amax, 1e-8)) * (6 * 448)} and
 *       {@code weight_scale_2 = 1 / gs}, carried as the output multiplier; per 16-element run the scale byte is
 *       {@code e4m3(min(gs * (amax * (1/6)), 448))} and the codes {@code e2m1(x * (1 / (scale * weight_scale_2)))}
 *       (nvfp4.py:60-79, ref_nvfp4_quant), so a stored element is the dequantized value without the global scale
 *       and the output multiplier restores it.</li>
 * </ul>
 * Quantizing multiplies or divides as the reference kernel does, so a midpoint rounds the same way. Stateless
 * prototypes; a scanner is single-threaded until {@link ScanningQuantizer#freeze}.
 */
public final class OnlineQuantEncoders {

    /** Largest E4M3FN magnitude, the FP8 scale divisor. */
    private static final float FP8_MAX = 448.0f;
    /** Largest E2M1 magnitude, the NVFP4 block-scale divisor. */
    private static final float FP4_MAX = 6.0f;
    /** Largest int8 level, the int8 scale divisor. */
    private static final int INT8_MAX = 127;

    /**
     * {@code 1 / (448 * 512)} rounded to float: the per-channel FP8 scale floor (fp8.py:89-91), the constant
     * {@link ActFakeQuant#FP8_MIN_SCALE} shares.
     */
    private static final float FP8_MIN_SCALE = ActFakeQuant.FP8_MIN_SCALE;

    /** The amax floor of the block FP8 scales (per_block_cast_to_fp8's {@code clamp(1e-4)}). */
    private static final float AMAX_FLOOR = 1e-4f;
    /** {@code 1/448} in float, the factor the compiled block-scale division multiplies by. */
    private static final float INV_FP8_MAX = 1.0f / FP8_MAX;
    /** The amax floor of the NVFP4 global scales (nvfp4.py:66). */
    private static final float NVFP4_AMAX_FLOOR = 1e-8f;

    private OnlineQuantEncoders() {
    }

    /** The fp8_per_tensor prototype: one scale per expert, computed from the whole expert's rows. */
    public static ScanningQuantizer fp8PerTensor() {
        return new Fp8PerTensor();
    }

    /** The fp8_per_block prototype: one scale per 128x128 tile per expert. */
    public static ScanningQuantizer fp8PerBlock() {
        return new Fp8PerBlock();
    }

    /** The fp8_per_channel quantizer: one scale per row, computed row-locally. */
    public static OnlineQuantizer fp8PerChannel() {
        return Fp8PerChannel.INSTANCE;
    }

    /** The int8 per-channel quantizer of {@code int8_per_channel_weight_only} and {@code experts_int8}. */
    public static OnlineQuantizer int8PerChannel() {
        return Int8PerChannel.INSTANCE;
    }

    /** The mxfp8 quantizer: one power-of-two scale per 32 elements, computed run-locally. */
    public static OnlineQuantizer mxfp8() {
        return Mxfp8.INSTANCE;
    }

    /** The mxfp4 quantizer: one power-of-two scale per 32 elements, computed run-locally. */
    public static OnlineQuantizer mxfp4() {
        return Mxfp4.INSTANCE;
    }

    /** The nvfp4_per_token prototype: a per-expert global scale and per-16-run scales. */
    public static ScanningQuantizer nvfp4PerToken() {
        return new Nvfp4PerToken();
    }

    // ------------------------------------------------------------------ shared writers

    /** {@code max|x|} over {@code n} values at {@code x[off]}, NaN ignored as the kernels' fmax reduction does. */
    private static float amax(float[] x, int off, int n) {
        float m = 0.0f;
        for (int i = off; i < off + n; i++) {
            float a = Math.abs(x[i]);
            if (a > m) {
                m = a;
            }
        }
        return m;
    }

    /** One I8_B128 block: {@code d} as little-endian f32, then the clamped rounded codes of {@code x / d}. */
    private static void putI8B128(byte[] dst, int dstOff, float d, float[] x, int xOff) {
        putF32(dst, dstOff, d);
        for (int j = 0; j < QK_I8B128; j++) {
            int q = (int) Math.max(-INT8_MAX, Math.min(INT8_MAX, Math.rint(x[xOff + j] / d)));
            dst[dstOff + I8B128_QS + j] = (byte) q;
        }
    }

    /** {@code v} as little-endian f32 at {@code dst[dstOff]}. */
    private static void putF32(byte[] dst, int dstOff, float v) {
        int bits = Float.floatToRawIntBits(v);
        dst[dstOff] = (byte) bits;
        dst[dstOff + 1] = (byte) (bits >>> 8);
        dst[dstOff + 2] = (byte) (bits >>> 16);
        dst[dstOff + 3] = (byte) (bits >>> 24);
    }

    /** The expert of flat row {@code row}, and the rows of one expert. */
    private static int expertOf(long row, long perExpert, long experts) {
        return (int) Math.min(experts - 1, row / perExpert);
    }

    /**
     * The row check of every quantizer: {@code n} is a positive whole number of {@code block}-element blocks.
     *
     * @throws IllegalArgumentException naming {@code type} otherwise
     */
    private static void checkRow(int n, int block, GGMLType type) {
        if (n <= 0 || n % block != 0) {
            throw new IllegalArgumentException("a row of " + n + " values is not a whole number of " + type
                    + " blocks of " + block);
        }
    }

    /** The common shape check of {@link ScanningQuantizer#open}. */
    private static void checkShape(long rows, int inner, long experts) {
        if (rows < 0 || inner <= 0 || experts <= 0 || rows % experts != 0) {
            throw new IllegalArgumentException(rows + " rows of " + inner + " over " + experts + " experts is"
                    + " not a whole number of rows per expert");
        }
    }

    // ------------------------------------------------------------------ fp8 storage

    /**
     * The FP8 storage type of a row of {@code n} values: F8_E4M3_B128 for whole 128-value blocks, else
     * F8_E4M3_B32, whose 32-value runs each carry the scale of the 128-wide scale block they lie in.
     */
    static GGMLType fp8Type(int n) {
        return n % QK_F8B128 == 0 ? GGMLType.F8_E4M3_B128 : GGMLType.F8_E4M3_B32;
    }

    /** The check of a transposed member's FP8 type: {@code type} is an FP8 block type {@code n} fills whole. */
    private static void checkColumnType(int n, GGMLType type) {
        if (type != GGMLType.F8_E4M3_B128 && type != GGMLType.F8_E4M3_B32) {
            throw new IllegalArgumentException(type + " is not an FP8 block type of the online quantizers");
        }
        checkRow(n, type.getBlockSize(), type);
    }

    /** The row check of the FP8 quantizers: {@code n} is a positive whole number of the type's blocks. */
    private static GGMLType checkFp8Row(int n) {
        GGMLType t = fp8Type(n);
        checkRow(n, t.getBlockSize(), t);
        return t;
    }

    /**
     * One FP8 block of {@code type} (F8_E4M3_B128 or F8_E4M3_B32): {@code d} as little-endian f32, then the
     * E4M3FN codes of {@code x / f} when {@code div} (per channel, the true division the dynamic kernel does) or
     * {@code x * f} otherwise (the reciprocal form the static and block kernels multiply by).
     */
    private static void putF8(GGMLType type, byte[] dst, int dstOff, float d, float[] x, int xOff, float f,
                              boolean div) {
        putF32(dst, dstOff, d);
        int qs = type == GGMLType.F8_E4M3_B128 ? F8B128_QS : F8B32_QS;
        for (int j = 0; j < type.getBlockSize(); j++) {
            float v = x[xOff + j];
            dst[dstOff + qs + j] = (byte) FpBlockDecoders.fp32ToE4m3(div ? v / f : v * f);
        }
    }

    // ------------------------------------------------------------------ fp8 per tensor

    /**
     * One E4M3 scale per expert, {@code amax / 448} over the expert's flattened weight (fp8.py:242-257 for a
     * linear, :643-655 per expert of a MoE); a zero scale keeps every code 0. Any run of the weight carries the
     * one scale, so a transposed member stores as its rows' type.
     */
    private static final class Fp8PerTensor implements ScanningQuantizer {

        private long perExpert;
        private long experts;
        private float[] amax;
        private float[] scales;

        @Override
        public GGMLType type() {
            return GGMLType.F8_E4M3_B128;
        }

        @Override
        public GGMLType typeFor(int n) {
            return fp8Type(n);
        }

        @Override
        public GGMLType columnTypeFor(int n, long startAlign) {
            return n > 0 && n % QK_F8B32 == 0 ? fp8Type(n) : null;
        }

        @Override
        public ScanningQuantizer open(long rows, int inner, long experts) {
            checkShape(rows, inner, experts);
            checkFp8Row(inner);
            Fp8PerTensor s = new Fp8PerTensor();
            s.perExpert = rows / experts;
            s.experts = experts;
            s.amax = new float[Math.toIntExact(experts)];
            return s;
        }

        @Override
        public void scanRow(long row, float[] x, int off, int n) {
            int e = expertOf(row, perExpert, experts);
            amax[e] = Math.max(amax[e], amax(x, off, n));
        }

        @Override
        public void scanColumn(long row, int col, float[] x, int off, int n) {
            int e = columnExpert(row, n);
            amax[e] = Math.max(amax[e], amax(x, off, n));
        }

        private int columnExpert(long row, int n) {
            int e = expertOf(row, perExpert, experts);
            if (n <= 0 || expertOf(row + n - 1, perExpert, experts) != e) {
                throw new IllegalArgumentException("a column run of rows [" + row + ", " + (row + n)
                        + ") crosses an expert of " + perExpert + " rows");
            }
            return e;
        }

        @Override
        public void freeze() {
            float[] sc = new float[amax.length];
            for (int e = 0; e < sc.length; e++) {
                sc[e] = amax[e] / FP8_MAX;
            }
            scales = sc;
        }

        @Override
        public void quantizeRow(long row, float[] x, int off, int n, byte[] dst, int dstOff) {
            write(checkFp8Row(n), scales[expertOf(row, perExpert, experts)], x, off, n, dst, dstOff);
        }

        @Override
        public void quantizeColumn(long row, int col, float[] x, int off, int n, GGMLType type, byte[] dst,
                                   int dstOff) {
            checkColumnType(n, type);
            write(type, scales[columnExpert(row, n)], x, off, n, dst, dstOff);
        }

        private static void write(GGMLType t, float s, float[] x, int off, int n, byte[] dst, int dstOff) {
            float f = s == 0.0f ? 0.0f : 1.0f / s;
            int bs = t.getBlockSize();
            for (int b = 0; b < n; b += bs) {
                putF8(t, dst, dstOff + b / bs * t.getTypeSize(), s, x, off + b, f, false);
            }
        }
    }

    // ------------------------------------------------------------------ fp8 per block

    /**
     * One E4M3 scale per 128x128 tile per expert (fp8.py:344-357 over each expert's 2-D weight, padded with zeros
     * to whole tiles by per_block_cast_to_fp8): tile {@code (rt, ct)} covers rows {@code [rt * 128, rt * 128 + 128)}
     * of the expert and columns {@code [ct * 128, ct * 128 + 128)}. Rows of a partial last tile, and the columns of
     * a partial last column tile, join the tile's amax, where vLLM's zero padding adds nothing. A stored row that is
     * no whole number of 128-value blocks is F8_E4M3_B32, each 32-value run carrying its tile's scale; a transposed
     * member's runs must each lie in one row tile.
     */
    private static final class Fp8PerBlock implements ScanningQuantizer {

        private long perExpert;
        private long experts;
        private int inner;
        private int rowTiles;
        private int colTiles;
        private float[] tileAmax;
        private float[] scales;

        @Override
        public GGMLType type() {
            return GGMLType.F8_E4M3_B128;
        }

        @Override
        public GGMLType typeFor(int n) {
            return fp8Type(n);
        }

        @Override
        public GGMLType columnTypeFor(int n, long startAlign) {
            if (n > 0 && n % QK_F8B128 == 0 && startAlign % QK_F8B128 == 0) {
                return GGMLType.F8_E4M3_B128;
            }
            if (n > 0 && n % QK_F8B32 == 0 && startAlign % QK_F8B32 == 0) {
                return GGMLType.F8_E4M3_B32;
            }
            return null;
        }

        @Override
        public ScanningQuantizer open(long rows, int inner, long experts) {
            checkShape(rows, inner, experts);
            checkFp8Row(inner);
            Fp8PerBlock s = new Fp8PerBlock();
            s.perExpert = rows / experts;
            s.experts = experts;
            s.inner = inner;
            s.rowTiles = (int) ((s.perExpert + QK_F8B128 - 1) / QK_F8B128);
            s.colTiles = (inner + QK_F8B128 - 1) / QK_F8B128;
            long tiles = Math.multiplyExact(Math.multiplyExact(experts, s.rowTiles), s.colTiles);
            s.tileAmax = new float[Math.toIntExact(tiles)];
            return s;
        }

        /** The index of the tile row {@code row} of the scanned tensor and column tile {@code ct} lie in. */
        private int tile(long row, int ct) {
            int e = expertOf(row, perExpert, experts);
            int rt = (int) ((row - e * perExpert) / QK_F8B128);
            return (e * rowTiles + rt) * colTiles + ct;
        }

        @Override
        public void scanRow(long row, float[] x, int off, int n) {
            checkWidth(n);
            for (int ct = 0; ct < colTiles; ct++) {
                int at = tile(row, ct);
                int from = ct * QK_F8B128;
                tileAmax[at] = Math.max(tileAmax[at], amax(x, off + from, Math.min(QK_F8B128, n - from)));
            }
        }

        @Override
        public void scanColumn(long row, int col, float[] x, int off, int n) {
            checkColumn(col);
            int ct = col / QK_F8B128;
            for (int i = 0; i < n; i++) {
                int at = tile(row + i, ct);
                tileAmax[at] = Math.max(tileAmax[at], Math.abs(x[off + i]));
            }
        }

        @Override
        public void freeze() {
            scales = new float[tileAmax.length];
            for (int i = 0; i < tileAmax.length; i++) {
                scales[i] = Math.max(tileAmax[i], AMAX_FLOOR) * INV_FP8_MAX;
            }
        }

        @Override
        public void quantizeRow(long row, float[] x, int off, int n, byte[] dst, int dstOff) {
            GGMLType t = checkFp8Row(n);
            checkWidth(n);
            int bs = t.getBlockSize();
            for (int b = 0; b < n; b += bs) {
                float s = scales[tile(row, b / QK_F8B128)];
                putF8(t, dst, dstOff + b / bs * t.getTypeSize(), s, x, off + b, 1.0f / s, false);
            }
        }

        @Override
        public void quantizeColumn(long row, int col, float[] x, int off, int n, GGMLType type, byte[] dst,
                                   int dstOff) {
            checkColumn(col);
            checkColumnType(n, type);
            int bs = type.getBlockSize();
            int ct = col / QK_F8B128;
            for (int b = 0; b < n; b += bs) {
                int at = tile(row + b, ct);
                if (tile(row + b + bs - 1, ct) != at) {
                    throw new IllegalArgumentException("rows [" + (row + b) + ", " + (row + b + bs)
                            + ") of a column run cross a 128-row tile");
                }
                float s = scales[at];
                putF8(type, dst, dstOff + b / bs * type.getTypeSize(), s, x, off + b, 1.0f / s, false);
            }
        }

        private void checkWidth(int n) {
            if (n != inner) {
                throw new IllegalArgumentException("a row of " + n + " values is not the scanned row of " + inner);
            }
        }

        private void checkColumn(int col) {
            if (col < 0 || col >= inner) {
                throw new IllegalArgumentException("column " + col + " lies outside the scanned rows of " + inner);
            }
        }
    }

    // ------------------------------------------------------------------ fp8 per channel

    /** One E4M3 scale per row, clamped to at least {@code 1/(448*512)} and applied by true division. */
    private static final class Fp8PerChannel implements OnlineQuantizer {

        private static final OnlineQuantizer INSTANCE = new Fp8PerChannel();

        @Override
        public GGMLType type() {
            return GGMLType.F8_E4M3_B128;
        }

        @Override
        public GGMLType typeFor(int n) {
            return fp8Type(n);
        }

        @Override
        public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
            GGMLType t = checkFp8Row(n);
            float s = Math.max(amax(x, off, n) / FP8_MAX, FP8_MIN_SCALE);
            int bs = t.getBlockSize();
            for (int b = 0; b < n; b += bs) {
                putF8(t, dst, dstOff + b / bs * t.getTypeSize(), s, x, off + b, s, true);
            }
        }
    }

    // ------------------------------------------------------------------ int8 per channel

    /** One int8 scale per row ({@code amax / 127}); codes round to nearest even and clamp to [-127, 127]. */
    private static final class Int8PerChannel implements OnlineQuantizer {

        private static final OnlineQuantizer INSTANCE = new Int8PerChannel();

        @Override
        public GGMLType type() {
            return GGMLType.I8_B128;
        }

        @Override
        public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
            checkRow(n, QK_I8B128, type());
            float amax = amax(x, off, n);
            if (amax == 0.0f) {
                Arrays.fill(dst, dstOff, dstOff + n / QK_I8B128 * I8B128_BYTES, (byte) 0);
                return;
            }
            float d = amax / INT8_MAX;
            for (int b = 0; b < n; b += QK_I8B128) {
                putI8B128(dst, dstOff + b / QK_I8B128 * I8B128_BYTES, d, x, off + b);
            }
        }
    }

    // ------------------------------------------------------------------ mxfp8

    /** One power-of-two E8M0 scale per 32 elements and E4M3 codes under it. */
    private static final class Mxfp8 implements OnlineQuantizer {

        private static final OnlineQuantizer INSTANCE = new Mxfp8();

        @Override
        public GGMLType type() {
            return GGMLType.MXFP8;
        }

        @Override
        public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
            checkRow(n, QK_MXFP8, type());
            for (int b = 0; b < n; b += QK_MXFP8) {
                int e = ActFakeQuant.mxfp8Exponent(amax(x, off + b, QK_MXFP8));
                dst[dstOff + b / QK_MXFP8 * MXFP8_BYTES + MXFP8_E] = (byte) (e + 127);
                int qs = dstOff + b / QK_MXFP8 * MXFP8_BYTES + MXFP8_QS;
                for (int j = 0; j < QK_MXFP8; j++) {
                    dst[qs + j] = (byte) FpBlockDecoders.fp32ToE4m3(Math.scalb(x[off + b + j], -e));
                }
            }
        }
    }

    // ------------------------------------------------------------------ mxfp4

    /**
     * One E8M0 scale per 32 elements and E2M1 codes under it, as vLLM's Triton quantizer {@code downcast_to_mxfp}
     * writes them (mxfp4_utils.py {@code _compute_quant_and_scale}).
     */
    private static final class Mxfp4 implements OnlineQuantizer {

        private static final OnlineQuantizer INSTANCE = new Mxfp4();

        @Override
        public GGMLType type() {
            return GGMLType.MXFP4;
        }

        @Override
        public void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff) {
            checkRow(n, QK_MXFP4, type());
            final int half = QK_MXFP4 / 2;
            for (int b = 0; b < n; b += QK_MXFP4) {
                int e = mxfp4Exponent(amax(x, off + b, QK_MXFP4));
                int block = dstOff + b / QK_MXFP4 * MXFP4_BYTES;
                dst[block + MXFP4_E] = (byte) (e + 127);
                float q = Math.scalb(1.0f, -e);
                for (int j = 0; j < half; j++) {
                    int lo = mxfp4Code(x[off + b + j] * q);
                    int hi = mxfp4Code(x[off + b + j + half] * q);
                    dst[block + MXFP4_QS + j] = (byte) (lo | hi << 4);
                }
            }
        }
    }

    /**
     * The unbiased E8M0 exponent of a 32-element run of absolute maximum {@code amax}: {@code amax} rounded up to
     * the next power of two from {@code 1.75 * 2^k} on ({@code (bits + 0x200000) & 0x7F800000}), then
     * {@code floor(log2(r)) - 2} clamped to {@code [-127, 127]}, a zero {@code r} taking {@code log2(2^-126)}.
     * The E8M0 byte is the exponent plus 127, 0 for {@code -127}.
     */
    static int mxfp4Exponent(float amax) {
        int bits = (Float.floatToRawIntBits(amax) + 0x200000) & 0x7F800000;
        int unbiased;
        if (bits == 0) {
            unbiased = -126 - 2;
        } else if (bits == 0x7F800000) {
            unbiased = 127;
        } else {
            unbiased = (bits >>> 23) - 127 - 2;
        }
        return Math.max(-127, Math.min(127, unbiased));
    }

    /**
     * The E2M1 code of {@code v} by the bit arithmetic of vLLM's Triton quantizer: below 1 the mantissa (with its
     * implicit bit, its lowest bit dropped) is shifted into the subnormal range with the shifted-out bits kept as a
     * sticky bit, then rounded to nearest with ties to even and saturated at 6; the sign bit is kept. A shift of 32
     * or more yields zero, as the GPU's clamped shift does.
     */
    static int mxfp4Code(float v) {
        int bits = Float.floatToRawIntBits(v);
        int sign = (bits >>> 28) & 0x8;
        int exponent = (bits >>> 23) & 0xFF;
        int mantissa = bits & 0x7FFFFF;
        if (exponent < 127) {
            int shift = 127 - (exponent + 1);
            int folded = 0x400000 | (mantissa >>> 1);
            int denormal = shift >= 32 ? 0 : folded >>> shift;
            int back = shift >= 32 ? 0 : denormal << shift;
            mantissa = denormal | (back != folded ? 1 : 0);
        }
        int biased = Math.max(exponent, 126) - 126;
        int rounded = (mantissa + ((mantissa >>> 22) & 1) + 0x1FFFFF) >>> 22;
        return sign | Math.min((biased << 1) + rounded, 0x7);
    }

    // ------------------------------------------------------------------ nvfp4 per token

    /**
     * NVFP4 experts (nvfp4.py:40-79 with the reference quantizer ref_nvfp4_quant, nvfp4_emulation_utils.py:754-768):
     * per expert the global scale {@code gs = (1 / clamp(amax, 1e-8)) * (6 * 448)} and {@code weight_scale_2 = 1 / gs},
     * carried as the output multiplier; per 16-element run the E4M3 scale {@code e4m3(min(gs * (amax * (1/6)),
     * 448))} and the E2M1 codes of {@code x * (1 / (scale * weight_scale_2))} (0 for a zero scale). A stored
     * element is the dequantized value without the global scale; the multiplier restores it.
     */
    private static final class Nvfp4PerToken implements ScanningQuantizer {

        private long perExpert;
        private long experts;
        private float[] amax;
        private float[] weightScale;

        @Override
        public GGMLType type() {
            return GGMLType.NVFP4;
        }

        @Override
        public ScanningQuantizer open(long rows, int inner, long experts) {
            checkShape(rows, inner, experts);
            checkRow(inner, QK_NVFP4, type());
            Nvfp4PerToken s = new Nvfp4PerToken();
            s.perExpert = rows / experts;
            s.experts = experts;
            s.amax = new float[Math.toIntExact(experts)];
            return s;
        }

        @Override
        public void scanRow(long row, float[] x, int off, int n) {
            int e = expertOf(row, perExpert, experts);
            amax[e] = Math.max(amax[e], amax(x, off, n));
        }

        /** NVFP4's 16-value scales run along the rows; a transposed member has none. */
        @Override
        public GGMLType columnTypeFor(int n, long startAlign) {
            return null;
        }

        @Override
        public void scanColumn(long row, int col, float[] x, int off, int n) {
            throw new IllegalArgumentException("nvfp4_per_token quantizes 16-value runs along rows; a transposed"
                    + " member has no NVFP4 layout");
        }

        @Override
        public void quantizeColumn(long row, int col, float[] x, int off, int n, GGMLType type, byte[] dst,
                                   int dstOff) {
            throw new IllegalArgumentException("nvfp4_per_token quantizes 16-value runs along rows; a transposed"
                    + " member has no NVFP4 layout");
        }

        @Override
        public void freeze() {
            float[] ws2 = new float[amax.length];
            for (int e = 0; e < ws2.length; e++) {
                ws2[e] = 1.0f / globalScale(e);
            }
            weightScale = ws2;
        }

        /**
         * The global scale {@code gs} of expert {@code e}: {@code (6 * 448) / amax} as PyTorch evaluates a scalar
         * over a tensor, the rounded reciprocal of {@code amax} times {@code 2688}.
         */
        private float globalScale(int e) {
            return (1.0f / Math.max(amax[e], NVFP4_AMAX_FLOOR)) * (FP4_MAX * FP8_MAX);
        }

        @Override
        public float[] outputScale() {
            return weightScale.clone();
        }

        @Override
        public void quantizeRow(long row, float[] x, int off, int n, byte[] dst, int dstOff) {
            checkRow(n, QK_NVFP4, type());
            int e = expertOf(row, perExpert, experts);
            float gs = globalScale(e);
            float ws2 = weightScale[e];
            for (int b = 0; b < n; b += QK_NVFP4) {
                int dstBlock = dstOff + b / QK_NVFP4 * NVFP4_BYTES;
                for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; s++) {
                    int run = off + b + s * QK_NVFP4_SUB;
                    float raw = Math.min(gs * (amax(x, run, QK_NVFP4_SUB) * (1.0f / FP4_MAX)), FP8_MAX);
                    int code = FpBlockDecoders.fp32ToE4m3(raw);
                    float scale = FpBlockDecoders.e4m3ToFp32(code);
                    dst[dstBlock + s] = (byte) code;
                    float d = scale * ws2;
                    float inv = d == 0.0f ? 0.0f : 1.0f / d;
                    int qs = dstBlock + NVFP4_QS + s * (QK_NVFP4_SUB / 2);
                    for (int j = 0; j < QK_NVFP4_SUB / 2; j++) {
                        int lo = FpBlockDecoders.fp32ToE2m1(x[run + j] * inv);
                        int hi = FpBlockDecoders.fp32ToE2m1(x[run + j + QK_NVFP4_SUB / 2] * inv);
                        dst[qs + j] = (byte) (lo | hi << 4);
                    }
                }
            }
        }
    }
}

package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.nio.ByteBuffer;

/**
 * Element conversions of the KV cache types: the BF16 store and the 32-value blocks of Q8_0, Q4_0, Q4_1, IQ4_NL,
 * Q5_0 and Q5_1 (CPU twin of kernels/llm/common/kv_types.h). Block layouts (little-endian, ggml-common.h):
 * <pre>
 *   Q8_0    34 B  d(f16) @0, qs[32] int8 @2                       value j = qs[j] * d
 *   Q4_0    18 B  d @0, qs[16] @2                                 value j = (nib_j - 8) * d
 *   Q4_1    20 B  d @0, m @2, qs[16] @4                           value j = nib_j * d + m
 *   IQ4_NL  18 B  d @0, qs[16] @2                                 value j = kvalues_iq4nl[nib_j] * d
 *   Q5_0    22 B  d @0, qh(u32) @2, qs[16] @6                     value j = ((nib_j | hbit_j &lt;&lt; 4) - 16) * d
 *   Q5_1    24 B  d @0, m @2, qh(u32) @4, qs[16] @8               value j = (nib_j | hbit_j &lt;&lt; 4) * d + m
 * </pre>
 * where nib_j is the low nibble of qs[j] for j &lt; 16 and the high nibble of qs[j - 16] otherwise, and hbit_j is bit
 * j of qh (dequantize_row_*, llama.cpp ggml-quants.c:328-548, 2726-2743). The quantizers are the ggml reference
 * quantizers ggml_set_rows stores with (from_float of llama.cpp ggml-cpu/ggml-cpu.c type traits): quantize_row_q8_0_ref
 * (llama.cpp ggml-quants.c:276-299), _q4_0_ref (:113-147), _q4_1_ref (:149-182), _q5_0_ref (:184-222), _q5_1_ref
 * (:224-260) and quantize_row_iq4_nl_ref (:5072-5085 with quantize_row_iq4_nl_impl :4938-5047, ntry = -1).
 */
public final class KvBlockCodec {

    /** Values per block of every block type. */
    public static final int BLOCK = 32;

    /** Largest block, in bytes (Q8_0). */
    public static final int MAX_BLOCK_BYTES = 34;

    /**
     * Smallest max |x| of an IQ4_NL block quantized with a non-zero scale (GROUP_MAX_EPS, llama.cpp ggml-quants.c:20).
     */
    private static final float IQ4_NL_EPS = 1e-15f;

    private KvBlockCodec() {
    }

    // ------------------------------------------------------------------ scalars

    /**
     * f32 to bfloat16 bits, round to nearest even, NaN kept quiet ({@code ggml_compute_fp32_to_bf16}, llama.cpp
     * ggml-impl.h:623-636).
     */
    public static int fp32ToBf16(float f) {
        int u = Float.floatToRawIntBits(f);
        if ((u & 0x7FFFFFFF) > 0x7F800000) {
            return ((u >>> 16) | 64) & 0xFFFF;
        }
        return ((u + (0x7FFF + ((u >>> 16) & 1))) >>> 16) & 0xFFFF;
    }

    /** C {@code roundf}: round half away from zero. */
    static int roundf(float v) {
        double a = Math.floor(Math.abs((double) v) + 0.5);
        return (int) (v < 0 ? -a : a);
    }

    /** Index of the IQ4_NL value nearest {@code x} ({@code best_index_int8}, llama.cpp ggml-quants.c:28-37). */
    static int iq4nlIndex(float x) {
        if (x <= QuantTables.kvalueIq4nl(0)) {
            return 0;
        }
        if (x >= QuantTables.kvalueIq4nl(15)) {
            return 15;
        }
        int ml = 0;
        int mu = 15;
        while (mu - ml > 1) {
            int mav = (ml + mu) / 2;
            if (x < QuantTables.kvalueIq4nl(mav)) {
                mu = mav;
            } else {
                ml = mav;
            }
        }
        return x - QuantTables.kvalueIq4nl(mu - 1) < QuantTables.kvalueIq4nl(mu) - x ? mu - 1 : mu;
    }

    // ------------------------------------------------------------------ blocks

    /**
     * Quantizes {@code x[xOff, xOff + 32)} into the block of type {@code t} at {@code y[yOff, yOff + t.blockBytes())}.
     *
     * @throws IllegalArgumentException when {@code t} is not a block type
     */
    public static void quantize(KvDType t, float[] x, int xOff, byte[] y, int yOff) {
        switch (t) {
            case Q8_0:
                quantizeQ8_0(x, xOff, y, yOff);
                break;
            case Q4_0:
                quantizeSym(x, xOff, y, yOff, false);
                break;
            case Q5_0:
                quantizeSym(x, xOff, y, yOff, true);
                break;
            case Q4_1:
                quantizeAffine(x, xOff, y, yOff, false);
                break;
            case Q5_1:
                quantizeAffine(x, xOff, y, yOff, true);
                break;
            case IQ4_NL:
                quantizeIq4Nl(x, xOff, y, yOff);
                break;
            default:
                throw new IllegalArgumentException(t + " is not a KV block type");
        }
    }

    /**
     * {@link #quantize(KvDType, float[], int, byte[], int)} into the block at absolute byte {@code yOff} of
     * {@code y}.
     */
    public static void quantize(KvDType t, float[] x, int xOff, ByteBuffer y, int yOff) {
        byte[] b = new byte[t.blockBytes()];
        quantize(t, x, xOff, b, 0);
        for (int i = 0; i < b.length; i++) {
            y.put(yOff + i, b[i]);
        }
    }

    /**
     * Dequantizes values {@code [j, j + n)} of the block of type {@code t} at {@code y[yOff..]} into
     * {@code dst[off, off + n)}.
     *
     * @throws IllegalArgumentException when {@code t} is not a block type or the range leaves the block
     */
    public static void dequantize(KvDType t, byte[] y, int yOff, int j, int n, float[] dst, int off) {
        if (!t.quantized()) {
            throw new IllegalArgumentException(t + " is not a KV block type");
        }
        if (j < 0 || n < 0 || j + n > BLOCK) {
            throw new IllegalArgumentException("values [" + j + ", " + (j + n) + ") outside a block of " + BLOCK);
        }
        float d = half(y, yOff);
        if (t == KvDType.Q8_0) {
            for (int i = 0; i < n; i++) {
                dst[off + i] = y[yOff + 2 + j + i] * d;
            }
            return;
        }
        int qs = qsOffset(t);
        float m = t == KvDType.Q4_1 || t == KvDType.Q5_1 ? half(y, yOff + 2) : 0f;
        int qh = t == KvDType.Q5_0 ? u32(y, yOff + 2) : t == KvDType.Q5_1 ? u32(y, yOff + 4) : 0;
        for (int i = 0; i < n; i++) {
            int e = j + i;
            int b = y[yOff + qs + (e & 15)] & 0xFF;
            int q = (e < 16 ? b & 0xF : b >>> 4) | (((qh >>> e) & 1) << 4);
            dst[off + i] = codeValue(t, q, d, m);
        }
    }

    /**
     * {@link #dequantize(KvDType, byte[], int, int, int, float[], int)} of a whole block at absolute byte
     * {@code yOff}.
     */
    public static void dequantize(KvDType t, ByteBuffer y, int yOff, float[] dst, int off) {
        byte[] b = new byte[t.blockBytes()];
        for (int i = 0; i < b.length; i++) {
            b[i] = y.get(yOff + i);
        }
        dequantize(t, b, 0, 0, BLOCK, dst, off);
    }

    private static float codeValue(KvDType t, int q, float d, float m) {
        switch (t) {
            case Q4_0:
                return (q - 8) * d;
            case Q5_0:
                return (q - 16) * d;
            case IQ4_NL:
                return d * QuantTables.kvalueIq4nl(q);
            default:
                return q * d + m;
        }
    }

    private static int qsOffset(KvDType t) {
        switch (t) {
            case Q4_1:
                return 4;
            case Q5_0:
                return 6;
            case Q5_1:
                return 8;
            default:
                return 2;
        }
    }

    private static void quantizeQ8_0(float[] x, int xOff, byte[] y, int yOff) {
        float amax = 0.0f;
        for (int j = 0; j < BLOCK; j++) {
            float a = Math.abs(x[xOff + j]);
            amax = amax > a ? amax : a;
        }
        final float d = amax / ((1 << 7) - 1);
        final float id = d != 0.0f ? 1.0f / d : 0.0f;
        putHalf(y, yOff, d);
        for (int j = 0; j < BLOCK; j++) {
            y[yOff + 2 + j] = (byte) roundf(x[xOff + j] * id);
        }
    }

    /** Q4_0 / Q5_0: {@code d = max / -2^(bits-1)} of the value with the largest |x|. */
    private static void quantizeSym(float[] x, int xOff, byte[] y, int yOff, boolean five) {
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < BLOCK; j++) {
            float v = x[xOff + j];
            if (amax < Math.abs(v)) {
                amax = Math.abs(v);
                max = v;
            }
        }
        final float d = max / (five ? -16 : -8);
        final float id = d != 0.0f ? 1.0f / d : 0.0f;
        final float bias = five ? 16.5f : 8.5f;
        final int top = five ? 31 : 15;
        final int qs = five ? 6 : 2;
        putHalf(y, yOff, d);
        int qh = 0;
        for (int j = 0; j < BLOCK / 2; j++) {
            int xi0 = Math.min(top, (byte) (int) (x[xOff + j] * id + bias));
            int xi1 = Math.min(top, (byte) (int) (x[xOff + j + BLOCK / 2] * id + bias));
            y[yOff + qs + j] = (byte) ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= ((xi0 & 0x10) >>> 4) << j;
            qh |= ((xi1 & 0x10) >>> 4) << (j + BLOCK / 2);
        }
        if (five) {
            putU32(y, yOff + 2, qh);
        }
    }

    /** Q4_1 / Q5_1: {@code d = (max - min) / (2^bits - 1)}, {@code m = min}. */
    private static void quantizeAffine(float[] x, int xOff, byte[] y, int yOff, boolean five) {
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (int j = 0; j < BLOCK; j++) {
            float v = x[xOff + j];
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
        }
        final float d = (max - min) / (five ? 31 : 15);
        final float id = d != 0.0f ? 1.0f / d : 0.0f;
        final int qs = five ? 8 : 4;
        putHalf(y, yOff, d);
        putHalf(y, yOff + 2, min);
        int qh = 0;
        for (int j = 0; j < BLOCK / 2; j++) {
            float x0 = (x[xOff + j] - min) * id;
            float x1 = (x[xOff + j + BLOCK / 2] - min) * id;
            int xi0 = five ? ((int) (x0 + 0.5f)) & 0xFF : Math.min(15, (byte) (int) (x0 + 0.5f));
            int xi1 = five ? ((int) (x1 + 0.5f)) & 0xFF : Math.min(15, (byte) (int) (x1 + 0.5f));
            y[yOff + qs + j] = (byte) ((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= ((xi0 & 0x10) >>> 4) << j;
            qh |= ((xi1 & 0x10) >>> 4) << (j + BLOCK / 2);
        }
        if (five) {
            putU32(y, yOff + 4, qh);
        }
    }

    /**
     * IQ4_NL: indices of {@code x / (max / kvalues[0])}, then the weighted least-squares scale
     * {@code d = sum(w q x) / sum(w q^2)}, {@code w = x^2}, accumulated in value order. A block with
     * {@code max |x| < 1e-15} stores {@code d = 0} and zero indices (llama.cpp leaves its indices unset; every index
     * decodes to 0 under {@code d = 0}).
     */
    private static void quantizeIq4Nl(float[] x, int xOff, byte[] y, int yOff) {
        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < BLOCK; j++) {
            float ax = Math.abs(x[xOff + j]);
            if (ax > amax) {
                amax = ax;
                max = x[xOff + j];
            }
        }
        for (int j = 0; j < BLOCK / 2; j++) {
            y[yOff + 2 + j] = 0;
        }
        float d = 0.0f;
        if (amax >= IQ4_NL_EPS) {
            final float d0 = max / QuantTables.kvalueIq4nl(0);
            final float id = 1 / d0;
            float sumqx = 0;
            float sumq2 = 0;
            for (int j = 0; j < BLOCK; j++) {
                float xj = x[xOff + j];
                int l = iq4nlIndex(id * xj);
                y[yOff + 2 + (j & 15)] |= (byte) (j < BLOCK / 2 ? l : l << 4);
                float q = QuantTables.kvalueIq4nl(l);
                float w = xj * xj;
                sumqx += w * q * xj;
                sumq2 += w * q * q;
            }
            d = sumq2 > 0 ? sumqx / sumq2 : 0.0f;
        }
        putHalf(y, yOff, d);
    }

    private static float half(byte[] y, int off) {
        return FpBlockDecoders.fp16ToFp32((y[off] & 0xFF) | (y[off + 1] & 0xFF) << 8);
    }

    private static void putHalf(byte[] y, int off, float v) {
        int h = FpBlockDecoders.fp32ToFp16(v);
        y[off] = (byte) h;
        y[off + 1] = (byte) (h >>> 8);
    }

    private static int u32(byte[] y, int off) {
        return (y[off] & 0xFF) | (y[off + 1] & 0xFF) << 8 | (y[off + 2] & 0xFF) << 16 | (y[off + 3] & 0xFF) << 24;
    }

    private static void putU32(byte[] y, int off, int v) {
        y[off] = (byte) v;
        y[off + 1] = (byte) (v >>> 8);
        y[off + 2] = (byte) (v >>> 16);
        y[off + 3] = (byte) (v >>> 24);
    }
}

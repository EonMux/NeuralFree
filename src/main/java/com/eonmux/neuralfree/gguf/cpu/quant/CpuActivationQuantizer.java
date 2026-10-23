package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_1;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;

/**
 * Quantizes F32 activations into the block format each weight type's CPU vec-dot consumes, bit-identical
 * to ggml's reference quantizers: {@code quantize_row_q8_0_ref}, {@code quantize_row_q8_1_ref} and
 * {@code quantize_row_q8_K_ref} (llama.cpp ggml/src/ggml-quants.c). The pairing of weight type and activation
 * type follows {@code type_traits_cpu[].vec_dot_type} (llama.cpp ggml/src/ggml-cpu/ggml-cpu.c:215-416), except
 * that F16 and BF16 weights take F32 activations (direct float dot products).
 * <p>
 * Stateless and thread-safe. The input array is only read.
 */
public final class CpuActivationQuantizer {

    private CpuActivationQuantizer() {
    }

    /**
     * The activation type the vec-dot of {@code weightType} consumes: Q8_0, Q8_1, Q8_K, or F32.
     *
     * @throws UnsupportedOperationException for types ggml-cpu has no vec-dot for (Q8_1, Q8_K, the integer
     *                                       types, F64, removed and unknown types)
     */
    public static GGMLType activationType(GGMLType weightType) {
        if (weightType == null) {
            throw new IllegalArgumentException("weight type is null");
        }
        switch (weightType) {
            case F32:
            case F16:
            case BF16:
                return GGMLType.F32;
            case Q4_0:
            case Q5_0:
            case Q8_0:
            case MXFP4:
            case NVFP4:
            case IQ4_NL:
            case Q1_0:
            case Q2_0:
                return GGMLType.Q8_0;
            case Q4_1:
            case Q5_1:
                return GGMLType.Q8_1;
            case Q2_K:
            case Q3_K:
            case Q4_K:
            case Q5_K:
            case Q6_K:
            case IQ2_XXS:
            case IQ2_XS:
            case IQ2_S:
            case IQ3_XXS:
            case IQ3_S:
            case IQ1_S:
            case IQ1_M:
            case IQ4_XS:
            case TQ1_0:
            case TQ2_0:
                return GGMLType.Q8_K;
            default:
                throw new UnsupportedOperationException("ggml type " + weightType
                        + " has no CPU vec-dot; it cannot be used as a matmul weight");
        }
    }

    /** Prepares {@code x[off .. off + n)} for a vec-dot against {@code weightType} weights. */
    public static QuantizedActivations quantizeFor(GGMLType weightType, float[] x, int off, int n) {
        GGMLType t = activationType(weightType);
        switch (t) {
            case Q8_0:
                return quantizeQ8_0(x, off, n);
            case Q8_1:
                return quantizeQ8_1(x, off, n);
            case Q8_K:
                return quantizeQ8_K(x, off, n);
            default:
                return f32(x, off, n);
        }
    }

    /** Copies {@code x[off .. off + n)} as F32 activations. */
    public static QuantizedActivations f32(float[] x, int off, int n) {
        checkRange(x, off, n, 1, GGMLType.F32);
        float[] v = new float[n];
        System.arraycopy(x, off, v, 0, n);
        return new QuantizedActivations(GGMLType.F32, n, null, null, null, null, v);
    }

    /**
     * {@code quantize_row_q8_0_ref} (llama.cpp ggml-quants.c:276-299): per 32 elements {@code d = amax / 127},
     * {@code q = roundf(x / d)}; {@code d} is stored as a half and read back by the vec-dot.
     */
    public static QuantizedActivations quantizeQ8_0(float[] x, int off, int n) {
        checkRange(x, off, n, QK8_0, GGMLType.Q8_0);
        final int nb = n / QK8_0;
        byte[] qs = new byte[n];
        float[] ds = new float[nb];
        for (int i = 0; i < nb; i++) {
            final int base = off + i * QK8_0;
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                amax = cMax(amax, Math.abs(x[base + j]));
            }
            final float d = amax / ((1 << 7) - 1);
            final float id = d != 0.0f ? 1.0f / d : 0.0f;
            ds[i] = FpBlockDecoders.roundToFp16(d);
            for (int j = 0; j < QK8_0; ++j) {
                qs[i * QK8_0 + j] = (byte) roundf(x[base + j] * id);
            }
        }
        return new QuantizedActivations(GGMLType.Q8_0, n, qs, ds, null, null, null);
    }

    /**
     * {@code quantize_row_q8_1_ref} (llama.cpp ggml-quants.c:302-335): Q8_0 quantization plus
     * {@code s = half(d * sum(q))} per block, where {@code d} is the unrounded float scale.
     */
    public static QuantizedActivations quantizeQ8_1(float[] x, int off, int n) {
        checkRange(x, off, n, QK8_1, GGMLType.Q8_1);
        final int nb = n / QK8_1;
        final int half = QK8_1 / 2;
        byte[] qs = new byte[n];
        float[] ds = new float[nb];
        float[] ss = new float[nb];
        for (int i = 0; i < nb; i++) {
            final int base = off + i * QK8_1;
            float amax = 0.0f;
            for (int j = 0; j < QK8_1; j++) {
                amax = cMax(amax, Math.abs(x[base + j]));
            }
            final float d = amax / ((1 << 7) - 1);
            final float id = d != 0.0f ? 1.0f / d : 0.0f;
            ds[i] = FpBlockDecoders.roundToFp16(d);
            int sum = 0;
            for (int j = 0; j < half; ++j) {
                final byte q0 = (byte) roundf(x[base + j] * id);
                final byte q1 = (byte) roundf(x[base + half + j] * id);
                qs[i * QK8_1 + j] = q0;
                qs[i * QK8_1 + half + j] = q1;
                sum += q0;
                sum += q1;
            }
            ss[i] = FpBlockDecoders.roundToFp16(sum * d);
        }
        return new QuantizedActivations(GGMLType.Q8_1, n, qs, ds, ss, null, null);
    }

    /**
     * {@code quantize_row_q8_K_ref} (llama.cpp ggml-quants.c:2769-2806): per 256 elements
     * {@code iscale = -127 / max} with {@code max} the first element of largest magnitude,
     * {@code q = min(127, nearest_int(iscale * x))}, {@code d = 1 / iscale} kept as a float, and
     * {@code bsums} over each 16 quants. An all-zero block gets {@code d = 0} and zero quants and sums.
     */
    public static QuantizedActivations quantizeQ8_K(float[] x, int off, int n) {
        checkRange(x, off, n, QK_K, GGMLType.Q8_K);
        final int nb = n / QK_K;
        byte[] qs = new byte[n];
        float[] ds = new float[nb];
        short[] bsums = new short[nb * (QK_K / 16)];
        for (int i = 0; i < nb; i++) {
            final int base = off + i * QK_K;
            float max = 0;
            float amax = 0;
            for (int j = 0; j < QK_K; ++j) {
                final float ax = Math.abs(x[base + j]);
                if (ax > amax) {
                    amax = ax;
                    max = x[base + j];
                }
            }
            if (amax == 0.0f) {
                ds[i] = 0.0f;
                continue;
            }
            final float iscale = -127.f / max;
            for (int j = 0; j < QK_K; ++j) {
                final int v = nearestInt(iscale * x[base + j]);
                qs[i * QK_K + j] = (byte) Math.min(127, v);
            }
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += qs[i * QK_K + j * 16 + ii];
                }
                bsums[i * (QK_K / 16) + j] = (short) sum;
            }
            ds[i] = 1 / iscale;
        }
        return new QuantizedActivations(GGMLType.Q8_K, n, qs, ds, null, bsums, null);
    }

    /**
     * ggml's {@code MAX(a, b)} macro ({@code (a) > (b) ? (a) : (b)}, llama.cpp ggml-impl.h:40). Unlike
     * {@link Math#max(float, float)} it does not propagate a NaN held in {@code a}, so a NaN input affects
     * the block scale exactly as it does in ggml.
     */
    private static float cMax(float a, float b) {
        return a > b ? a : b;
    }

    /** C {@code roundf}: round half away from zero. Exact for every float in the quantizer's range. */
    static int roundf(float v) {
        final double a = Math.floor(Math.abs((double) v) + 0.5);
        return (int) (v < 0 ? -a : a);
    }

    /**
     * {@code nearest_int} (llama.cpp ggml-quants.c:621-626): round half to even through the 1.5 * 2^23 magic
     * constant; valid for {@code |fval| <= 4194303}.
     */
    static int nearestInt(float fval) {
        final float val = fval + 12582912.f;
        final int i = Float.floatToRawIntBits(val);
        return (i & 0x007fffff) - 0x00400000;
    }

    private static void checkRange(float[] x, int off, int n, int block, GGMLType type) {
        if (x == null) {
            throw new IllegalArgumentException("activation array is null");
        }
        if (n < 0 || off < 0 || (long) off + n > x.length) {
            throw new IndexOutOfBoundsException("activation range [" + off + ", " + ((long) off + n)
                    + ") exceeds array length " + x.length);
        }
        if (n % block != 0) {
            throw new IllegalArgumentException(n + " activations are not a multiple of the " + type
                    + " block size " + block);
        }
    }
}

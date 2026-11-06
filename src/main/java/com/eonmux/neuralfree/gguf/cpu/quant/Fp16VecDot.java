package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.shortAt;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.floatAt;

/**
 * CPU dot product of plain float weight rows (F32, F16, BF16) and of the per-element FP8 code type
 * F8_E4M3FN with F32 activations. The float elements are widened exactly ({@code GGML_CPU_FP16_TO_FP32},
 * {@code GGML_BF16_TO_FP32}) and the E4M3FN codes decoded through {@link FpBlockDecoders#e4m3ToFp32(int)};
 * all are multiplied with the F32 activations, which {@link CpuActivationQuantizer} keeps in F32 for these
 * types, so the activation side carries no extra rounding. F8_E4M3FN serves any row length (one byte per
 * element), as {@code ggml_vec_dot_f8_e4m3_f32} does.
 * <p>
 * Accumulation follows the SIMD paths of {@code ggml_vec_dot_f32}, {@code ggml_vec_dot_f16},
 * {@code ggml_vec_dot_bf16} (llama.cpp ggml/src/ggml-cpu/vec.cpp:11, 139, 264) and
 * {@code ggml_vec_dot_f8_e4m3_f32} (llama.cpp ggml-cpu/quants.c:370-411): float partial sums over 8 lanes
 * reduced pairwise, then the elements past the last full group of 8 are added one product at a time, in
 * float for F32 and F8_E4M3FN and in double ({@code ggml_float}) for F16 and BF16, as those functions do.
 * Weights are fetched 8 bytes at a time to keep buffer reads off the critical path.
 * <p>
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class Fp16VecDot implements CpuVecDot {

    public static final Fp16VecDot F32 = new Fp16VecDot(GGMLType.F32, Fp16VecDot::dotF32);
    public static final Fp16VecDot F16 = new Fp16VecDot(GGMLType.F16, Fp16VecDot::dotF16);
    public static final Fp16VecDot BF16 = new Fp16VecDot(GGMLType.BF16, Fp16VecDot::dotBf16);
    public static final Fp16VecDot F8_E4M3FN = new Fp16VecDot(GGMLType.F8_E4M3FN, Fp16VecDot::dotF8E4m3fn);

    private static final Map<GGMLType, Fp16VecDot> BY_TYPE = index(F32, F16, BF16, F8_E4M3FN);

    /** Value of every E4M3FN code, from {@link FpBlockDecoders#e4m3ToFp32(int)} so it matches the decoder. */
    private static final float[] E4M3 = buildE4m3();

    private static float[] buildE4m3() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e4m3ToFp32(i);
        }
        return t;
    }

    /** One row kernel: {@code n} elements starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(byte[] b, int off, float[] x, int n);
    }

    private final GGMLType type;
    private final RowKernel kernel;

    private Fp16VecDot(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.kernel = kernel;
    }

    private static Map<GGMLType, Fp16VecDot> index(Fp16VecDot... dots) {
        Map<GGMLType, Fp16VecDot> m = new EnumMap<>(GGMLType.class);
        for (Fp16VecDot d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
    }

    /** True when this class has a vec-dot for {@code type}. */
    public static boolean supports(GGMLType type) {
        return type != null && BY_TYPE.containsKey(type);
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return BY_TYPE.keySet();
    }

    /**
     * Returns the vec-dot of {@code type}.
     *
     * @throws IllegalArgumentException when {@code type} is not F32, F16 or BF16
     */
    public static Fp16VecDot forType(GGMLType type) {
        Fp16VecDot d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("Fp16VecDot has no kernel for ggml type " + type
                    + "; it handles " + BY_TYPE.keySet());
        }
        return d;
    }

    /** The weight type of this vec-dot. */
    public GGMLType weightType() {
        return type;
    }

    /** The activation type this vec-dot consumes (F32). */
    public GGMLType activationType() {
        return GGMLType.F32;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return kernel.dot(CpuVecDotRows.copyRow(w, row), 0, x.f32, x.n);
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return kernel.dot(row, off, x.f32, x.n);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private void check(CpuTensorRef w, QuantizedActivations x) {
        if (w == null || x == null) {
            throw new IllegalArgumentException("weight tensor and activations must not be null");
        }
        if (w.type() != type) {
            throw new IllegalArgumentException("the " + type + " vec-dot cannot read a " + w.type() + " tensor");
        }
        x.requireCompatible(w, GGMLType.F32);
    }

    /** Low float of a little-endian 8-byte word. */
    private static float lo(long v) {
        return Float.intBitsToFloat((int) v);
    }

    /** High float of a little-endian 8-byte word. */
    private static float hi(long v) {
        return Float.intBitsToFloat((int) (v >>> 32));
    }

    /** Half {@code k} (0-3) of a little-endian 8-byte word, widened. */
    private static float half(long v, int k) {
        return FpBlockDecoders.fp16ToFp32((int) (v >>> (k << 4)));
    }

    /** bfloat16 {@code k} (0-3) of a little-endian 8-byte word, widened. */
    private static float bf16(long v, int k) {
        return Float.intBitsToFloat(((int) (v >>> (k << 4)) & 0xFFFF) << 16);
    }

    /** F32 weights: {@code ggml_vec_dot_f32} (llama.cpp ggml-cpu/vec.cpp:11-137). */
    private static float dotF32(byte[] b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 32) {
            final long v0 = longAt(b, p);
            final long v1 = longAt(b, p + 8);
            final long v2 = longAt(b, p + 16);
            final long v3 = longAt(b, p + 24);
            s0 += lo(v0) * x[i];
            s1 += hi(v0) * x[i + 1];
            s2 += lo(v1) * x[i + 2];
            s3 += hi(v1) * x[i + 3];
            s4 += lo(v2) * x[i + 4];
            s5 += hi(v2) * x[i + 5];
            s6 += lo(v3) * x[i + 6];
            s7 += hi(v3) * x[i + 7];
        }
        float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for (; i < n; i++) {
            sum += floatAt(b, off + 4 * i) * x[i];
        }
        return sum;
    }

    /** F16 weights widened per element: {@code ggml_vec_dot_f16} (llama.cpp ggml-cpu/vec.cpp:264-378). */
    private static float dotF16(byte[] b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final long v0 = longAt(b, p);
            final long v1 = longAt(b, p + 8);
            s0 += half(v0, 0) * x[i];
            s1 += half(v0, 1) * x[i + 1];
            s2 += half(v0, 2) * x[i + 2];
            s3 += half(v0, 3) * x[i + 3];
            s4 += half(v1, 0) * x[i + 4];
            s5 += half(v1, 1) * x[i + 5];
            s6 += half(v1, 2) * x[i + 6];
            s7 += half(v1, 3) * x[i + 7];
        }
        double sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for (; i < n; i++) {
            sum += FpBlockDecoders.fp16At(b, off + 2 * i) * x[i];
        }
        return (float) sum;
    }

    /** BF16 weights widened per element: {@code ggml_vec_dot_bf16} (llama.cpp ggml-cpu/vec.cpp:139-262). */
    private static float dotBf16(byte[] b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final long v0 = longAt(b, p);
            final long v1 = longAt(b, p + 8);
            s0 += bf16(v0, 0) * x[i];
            s1 += bf16(v0, 1) * x[i + 1];
            s2 += bf16(v0, 2) * x[i + 2];
            s3 += bf16(v0, 3) * x[i + 3];
            s4 += bf16(v1, 0) * x[i + 4];
            s5 += bf16(v1, 1) * x[i + 5];
            s6 += bf16(v1, 2) * x[i + 6];
            s7 += bf16(v1, 3) * x[i + 7];
        }
        double sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for (; i < n; i++) {
            sum += FpBlockDecoders.bf16ToFp32(shortAt(b, off + 2 * i)) * x[i];
        }
        return (float) sum;
    }

    /** E4M3FN code {@code k} (0-7) of a little-endian 8-byte word, decoded. */
    private static float e4m3(long v, int k, float[] lut) {
        return lut[(int) (v >>> (k << 3)) & 0xFF];
    }

    /** F8_E4M3FN weights: {@code ggml_vec_dot_f8_e4m3_f32} (llama.cpp ggml-cpu/quants.c:370-411). */
    private static float dotF8E4m3fn(byte[] b, int off, float[] x, int n) {
        final float[] lut = E4M3;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 8) {
            final long v = longAt(b, p);
            s0 += e4m3(v, 0, lut) * x[i];
            s1 += e4m3(v, 1, lut) * x[i + 1];
            s2 += e4m3(v, 2, lut) * x[i + 2];
            s3 += e4m3(v, 3, lut) * x[i + 3];
            s4 += e4m3(v, 4, lut) * x[i + 4];
            s5 += e4m3(v, 5, lut) * x[i + 5];
            s6 += e4m3(v, 6, lut) * x[i + 6];
            s7 += e4m3(v, 7, lut) * x[i + 7];
        }
        float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        for (; i < n; i++) {
            sum += lut[b[off + i] & 0xFF] * x[i];
        }
        return sum;
    }

    @Override
    public String toString() {
        return "Fp16VecDot{" + type + " x F32}";
    }
}

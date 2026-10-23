package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

/**
 * CPU dot product of plain float weight rows (F32, F16, BF16) with F32 activations. The weight elements
 * are widened exactly ({@code GGML_CPU_FP16_TO_FP32}, {@code GGML_BF16_TO_FP32}) and multiplied with the
 * F32 activations; {@link CpuActivationQuantizer} keeps activations in F32 for these types, so the
 * activation side carries no F16/BF16 rounding.
 * <p>
 * Accumulation follows the SIMD paths of {@code ggml_vec_dot_f32}, {@code ggml_vec_dot_f16} and
 * {@code ggml_vec_dot_bf16} (llama.cpp ggml/src/ggml-cpu/vec.cpp:11, 139, 264): float partial sums over 8 lanes
 * reduced pairwise, then the elements past the last full group of 8 are added one product at a time, in
 * float for F32 and in double ({@code ggml_float}) for F16 and BF16, as those functions do. Weights are
 * fetched 8 bytes at a time to keep buffer reads off the critical path.
 * <p>
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class Fp16VecDot implements CpuVecDot {

    public static final Fp16VecDot F32 = new Fp16VecDot(GGMLType.F32, Fp16VecDot::dotF32);
    public static final Fp16VecDot F16 = new Fp16VecDot(GGMLType.F16, Fp16VecDot::dotF16);
    public static final Fp16VecDot BF16 = new Fp16VecDot(GGMLType.BF16, Fp16VecDot::dotBf16);

    private static final Map<GGMLType, Fp16VecDot> BY_TYPE = index(F32, F16, BF16);

    /** One row kernel: {@code n} elements starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(ByteBuffer b, int off, float[] x, int n);
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
        return kernel.dot(w.rowBuffer(row), w.rowOffset(row), x.f32, x.n);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final float[] xf = x.f32;
        final int n = x.n;
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> kernel.dot(b, off, xf, n));
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
    private static float dotF32(ByteBuffer b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 32) {
            final long v0 = b.getLong(p);
            final long v1 = b.getLong(p + 8);
            final long v2 = b.getLong(p + 16);
            final long v3 = b.getLong(p + 24);
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
            sum += b.getFloat(off + 4 * i) * x[i];
        }
        return sum;
    }

    /** F16 weights widened per element: {@code ggml_vec_dot_f16} (llama.cpp ggml-cpu/vec.cpp:264-378). */
    private static float dotF16(ByteBuffer b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final long v0 = b.getLong(p);
            final long v1 = b.getLong(p + 8);
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
    private static float dotBf16(ByteBuffer b, int off, float[] x, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (int p = off; i + 8 <= n; i += 8, p += 16) {
            final long v0 = b.getLong(p);
            final long v1 = b.getLong(p + 8);
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
            sum += FpBlockDecoders.bf16ToFp32(b.getShort(off + 2 * i)) * x[i];
        }
        return (float) sum;
    }

    @Override
    public String toString() {
        return "Fp16VecDot{" + type + " x F32}";
    }
}

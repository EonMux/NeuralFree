package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.utils.numeric.Fp8Codec;

import java.util.Objects;

/**
 * Activation fake-quantization on the CPU: {@code x' = dequant(quant(x))} per {@link ActQuantScheme}, in place, with
 * the formulas and the operation order the scheme's javadoc gives and the {@code LLM_ActFakeQuant} kernels
 * ({@code kernels/llm/act_quant.cl}, {@code shaders/llm/act_quant.comp}) implement, so the host and the device give
 * bit-identical results for normal operands. Every quotient and reciprocal is the correctly rounded float operation,
 * every product a single float rounding, and the scaling by {@code 2^e} of the MX schemes is exact.
 *
 * <p>Rules shared with the kernels:
 * <ul>
 *   <li>A zero dequant multiplier ({@code s}, or {@code s16*g} for NVFP4) gives {@code +0}; int8 levels are integers,
 *       so a zero level dequantizes to {@code +0} too.</li>
 *   <li>The int8 round-to-nearest-even reads a NaN quotient as level 0 (vLLM's {@code cvt.rni.sat}); with finite
 *       inputs that happens only for a zero input times the infinite reciprocal of a vanishing scale.</li>
 *   <li>{@code e4m3(v)} saturates the infinities to {@code +-448} (the clamp before the cast), NaN stays NaN;
 *       {@code e2m1(v)} saturates at {@code +-6} and maps negative zero to {@code +0}.</li>
 *   <li>NVFP4 dequantizes {@code (e2m1 * s16) * g}, left to right.</li>
 *   <li>MXFP4 takes {@code e = 127} when the rounded {@code amax} is infinite (vLLM mxfp4_utils.py:270-280).</li>
 * </ul>
 * Results are specified for finite inputs. Stateless and thread-safe.
 */
public final class ActFakeQuant {

    /** Largest int8 level, the divisor of the int8 scales. */
    private static final float INT8_MAX = 127.0f;
    /** Largest E4M3FN magnitude, the divisor of the FP8 scales. */
    private static final float FP8_MAX = 448.0f;
    /** Largest E2M1 magnitude, the NVFP4 block-scale divisor. */
    private static final float FP4_MAX = 6.0f;

    /** {@code 1e-10f} ({@code 0x2EDBE6FF}): the {@code amax} floor of the per-group schemes (vLLM's {@code eps}). */
    public static final float GROUP_EPS = 1e-10f;

    /**
     * {@code 1 / (448 * 512)} rounded to float ({@code 0x36924925}): the scale floor of
     * {@link ActQuantScheme#FP8_TOKEN} (vLLM {@code min_scaling_factor}, csrc/quantization/utils.cuh:50-54).
     */
    public static final float FP8_MIN_SCALE = Float.intBitsToFloat(0x36924925);

    private ActFakeQuant() {
    }

    /**
     * Fake-quantizes {@code rows} contiguous rows of {@code cols} elements at {@code x[off ..]} in place.
     *
     * @param s           the scheme
     * @param scale       the static {@code .input_scale} of {@link ActQuantScheme#isStatic() static} schemes; for
     *                    {@link ActQuantScheme#FP8_TOKEN} the upper bound {@code ub} on {@code amax}
     *                    ({@link Float#POSITIVE_INFINITY} for none); ignored by the other schemes
     * @param zeroPoint   the {@code .input_zero_point} of {@link ActQuantScheme#INT8_STATIC_ASYM}; ignored otherwise
     * @param globalScale the NVFP4 global scale {@code g} ({@code .input_scale}); ignored otherwise
     * @throws IllegalArgumentException when the range lies outside {@code x}, {@code cols} is not a whole number of
     *                                  the scheme's runs, or a parameter the scheme reads is out of range
     */
    public static void apply(ActQuantScheme s, float scale, int zeroPoint, float globalScale, float[] x, int off,
                             int rows, int cols) {
        Objects.requireNonNull(s, "scheme");
        Objects.requireNonNull(x, "x");
        if (rows < 0 || off < 0 || (long) off + (long) rows * cols > x.length) {
            throw new IllegalArgumentException("activation range of " + rows + " rows of " + cols + " at " + off
                    + " lies outside an array of " + x.length);
        }
        checkParameters(s, scale, zeroPoint, globalScale);
        int run = s.runLength(cols);
        int end = off + rows * cols;
        for (int b = off; b < end; b += run) {
            applyRun(s, scale, zeroPoint, globalScale, x, b, run);
        }
    }

    /**
     * Checks the parameters {@code s} reads: a finite positive {@code scale} for the static schemes, a positive
     * {@code scale} (infinity included) for {@link ActQuantScheme#FP8_TOKEN}, a finite positive
     * {@code globalScale} for NVFP4, and a zero point within {@code +-2^24} (exact as a float, and {@code q - azp} cannot
     * overflow).
     *
     * @throws IllegalArgumentException naming the scheme and the parameter
     */
    public static void checkParameters(ActQuantScheme s, float scale, int zeroPoint, float globalScale) {
        if (s.isStatic() && !(scale > 0.0f && scale < Float.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(s + " needs a positive finite input scale, got " + scale);
        }
        if (s == ActQuantScheme.FP8_TOKEN && !(scale > 0.0f)) {
            throw new IllegalArgumentException(s + " needs a positive amax upper bound (infinity for none), got "
                    + scale);
        }
        if (s.needsZeroPoint() && (zeroPoint < -(1 << 24) || zeroPoint > 1 << 24)) {
            throw new IllegalArgumentException(s + " zero point " + zeroPoint + " lies outside [-2^24, 2^24]");
        }
        if (s.needsGlobalScale() && !(globalScale > 0.0f && globalScale < Float.POSITIVE_INFINITY)) {
            throw new IllegalArgumentException(s + " needs a positive finite global scale, got " + globalScale);
        }
    }

    private static void applyRun(ActQuantScheme s, float scale, int azp, float g, float[] x, int b, int n) {
        float amax = s.isStatic() ? 0.0f : amax(x, b, n);
        switch (s) {
            case INT8_TOKEN: {
                float d = amax / INT8_MAX;
                float inv = amax == 0.0f ? 0.0f : INT8_MAX / amax;
                for (int i = b; i < b + n; i++) {
                    x[i] = d == 0.0f ? 0.0f : int8(rne(x[i] * inv), 0, d);
                }
                return;
            }
            case INT8_STATIC:
                for (int i = b; i < b + n; i++) {
                    x[i] = int8(rne(x[i] / scale), 0, scale);
                }
                return;
            case INT8_STATIC_ASYM: {
                float inv = 1.0f / scale;
                for (int i = b; i < b + n; i++) {
                    x[i] = int8(rne(x[i] * inv), azp, scale);
                }
                return;
            }
            case INT8_GROUP128: {
                float d = Math.max(amax, GROUP_EPS) / INT8_MAX;
                for (int i = b; i < b + n; i++) {
                    x[i] = (int) clamp(x[i] / d, -128.0f, INT8_MAX) * d;
                }
                return;
            }
            case FP8_TOKEN: {
                float d = Math.max(Math.min(amax, scale) / FP8_MAX, FP8_MIN_SCALE);
                for (int i = b; i < b + n; i++) {
                    x[i] = e4m3(x[i] / d) * d;
                }
                return;
            }
            case FP8_STATIC:
                fp8Reciprocal(x, b, n, scale);
                return;
            case FP8_GROUP128:
                fp8Reciprocal(x, b, n, Math.max(amax, GROUP_EPS) / FP8_MAX);
                return;
            case NVFP4:
                nvfp4(x, b, n, amax, g);
                return;
            case MXFP4: {
                int e = mxfp4Exponent(amax);
                for (int i = b; i < b + n; i++) {
                    x[i] = Math.scalb(e2m1(Math.scalb(x[i], -e)), e);
                }
                return;
            }
            case MXFP8: {
                int e = mxfp8Exponent(amax);
                for (int i = b; i < b + n; i++) {
                    x[i] = Math.scalb(e4m3(Math.scalb(x[i], -e)), e);
                }
                return;
            }
            default:
                throw new IllegalStateException("no fake-quantization for " + s);
        }
    }

    /** {@code e4m3(x * (1/d)) * d} over the run: FP8_STATIC and FP8_GROUP128. */
    private static void fp8Reciprocal(float[] x, int b, int n, float d) {
        float inv = 1.0f / d;
        for (int i = b; i < b + n; i++) {
            x[i] = e4m3(x[i] * inv) * d;
        }
    }

    /** NVFP4 over a 16-element run: {@code s16 = e4m3(amax/6/g)}, {@code x' = (e2m1(x/(s16*g)) * s16) * g}. */
    private static void nvfp4(float[] x, int b, int n, float amax, float g) {
        float s16 = e4m3(amax / FP4_MAX / g);
        float d = s16 * g;
        for (int i = b; i < b + n; i++) {
            x[i] = d == 0.0f ? 0.0f : e2m1(x[i] / d) * s16 * g;
        }
    }

    /** {@code max|x|} over the run, ignoring NaN as the kernels' {@code fmax} reduction does. */
    private static float amax(float[] x, int b, int n) {
        float m = 0.0f;
        for (int i = b; i < b + n; i++) {
            float a = Math.abs(x[i]);
            if (a > m) {
                m = a;
            }
        }
        return m;
    }

    /** {@code v} rounded to nearest even, NaN read as level 0. */
    private static float rne(float v) {
        return Float.isNaN(v) ? 0.0f : (float) Math.rint(v);
    }

    /** {@code (q - azp) * s} for the integer level {@code q = sat8(level + azp)}. */
    private static float int8(float level, int azp, float s) {
        int q = (int) clamp(level + azp, -128.0f, INT8_MAX);
        return (q - azp) * s;
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /** The value of the E4M3FN round-to-nearest-even code of {@code clamp(v, -448, 448)}; NaN stays NaN. */
    static float e4m3(float v) {
        return Fp8Codec.E4M3FN.decode(Fp8Codec.E4M3FN.encode(clamp(v, -FP8_MAX, FP8_MAX)));
    }

    /** The E2M1 round-to-nearest-even value of {@code v}, saturating at {@code +-6}; negative zero gives {@code +0}. */
    static float e2m1(float v) {
        return FpBlockDecoders.e2m1ToFp32(FpBlockDecoders.fp32ToE2m1(v));
    }

    /**
     * The MXFP4 scale exponent of a run with maximum magnitude {@code amax}: {@code amax} rounded to a 3-bit mantissa
     * with ties to even, {@code e = clamp(floor(log2(r)) - 2, -127, 127)}, {@code r = 0} read as {@code 2^-126} and
     * an infinite {@code r} giving 127 (vLLM mxfp4_utils.py:255-280).
     */
    static int mxfp4Exponent(float amax) {
        int r = (Float.floatToRawIntBits(amax) + 0x200000) & 0x7F800000;
        if (r == 0x7F800000) {
            return 127;
        }
        int floorLog2 = r == 0 ? -126 : (r >>> 23) - 127;
        return Math.max(-127, Math.min(127, floorLog2 - 2));
    }

    /**
     * The MXFP8 scale exponent of a run with maximum magnitude {@code amax}:
     * {@code clamp(ceil(log2(max(amax, FLT_MIN) / 448)) + 127, 0, 254) - 127}, with {@code ceil(log2)} taken exactly
     * from the bits of the correctly rounded quotient (vLLM mxfp8_utils.py:55-70).
     */
    static int mxfp8Exponent(float amax) {
        int d = Float.floatToRawIntBits(Math.max(amax, Float.MIN_NORMAL) / FP8_MAX);
        int ex = d >>> 23;
        int m = d & 0x7FFFFF;
        int ceilLog2;
        if (ex == 0xFF) {
            ceilLog2 = 128;
        } else if (ex == 0) {
            ceilLog2 = (31 - Integer.numberOfLeadingZeros(m)) - 149 + ((m & (m - 1)) != 0 ? 1 : 0);
        } else {
            ceilLog2 = ex - 127 + (m != 0 ? 1 : 0);
        }
        return Math.max(0, Math.min(254, ceilLog2 + 127)) - 127;
    }
}

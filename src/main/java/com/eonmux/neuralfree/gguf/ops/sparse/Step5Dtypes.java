package com.eonmux.neuralfree.gguf.ops.sparse;

/**
 * Element roundings of the Step5 attention path, mirroring the vLLM casts:
 * <ul>
 *   <li>indexer keys, the open-region ring and the attention gate are {@code bfloat16}
 *       ({@code indexer_norm_rope.py}: {@code x.to(K.dtype.element_ty)}, the ring stores, and
 *       {@code attention_gate} rounding the sigmoid to the gate dtype);</li>
 *   <li>indexer queries and closed-region summaries are {@code float8_e4m3fn}
 *       ({@code indexer_norm_rope.py} stores the pre-rope BF16 value into the FP8 query tensor, and
 *       {@code summary_update.py} rounds the summary to BF16 and then to E4M3).</li>
 * </ul>
 * Round-to-nearest-even, with E4M3 saturating at {@value #E4M3_MAX} as PyTorch's conversion does. NaNs and infinities
 * pass through the BF16 conversion unchanged.
 */
public final class Step5Dtypes {

    /** Largest finite E4M3 value. */
    public static final float E4M3_MAX = 448f;
    private static final float E4M3_STEP = 0.001953125f;   // 2^-9, the E4M3 subnormal step
    private static final double LN2 = 0.6931471805599453;

    private Step5Dtypes() {
    }

    /** {@code x} rounded to bfloat16 and widened back to F32. */
    public static float bf16(float x) {
        int bits = Float.floatToRawIntBits(x);
        if ((bits & 0x7F800000) == 0x7F800000) {
            return x;
        }
        int lsb = (bits >>> 16) & 1;
        bits += 0x7FFF + lsb;
        return Float.intBitsToFloat(bits & 0xFFFF0000);
    }

    /** {@code x} rounded to float8_e4m3fn and widened back to F32. */
    public static float e4m3(float x) {
        if (Float.isNaN(x)) {
            return Float.NaN;
        }
        float sign = x < 0f || (x == 0f && 1f / x < 0f) ? -1f : 1f;
        double a = Math.abs((double) x);
        if (a >= E4M3_MAX) {
            return sign * E4M3_MAX;
        }
        double step;
        if (a == 0.0) {
            return sign * 0f;
        }
        int exponent = (int) Math.floor(Math.log(a) / LN2);
        if (exponent >= -6) {
            step = Math.pow(2.0, exponent - 3);
        } else {
            step = E4M3_STEP;
        }
        double scaled = Math.rint(a / step);
        double value = scaled * step;
        if (value > E4M3_MAX) {
            value = E4M3_MAX;
        }
        return (float) (sign * value);
    }

    /** {@code sigmoid(x)} in F32, rounded to bfloat16 and widened back (the head gate's dtype). */
    public static float gateSigmoid(float x) {
        return bf16((float) (1.0 / (1.0 + Math.exp(-(double) x))));
    }

    /** {@code 2^x} in F32, the attention and summary exponentials of the vLLM Triton kernels. */
    public static float exp2(float x) {
        return (float) Math.exp((double) x * LN2);
    }
}

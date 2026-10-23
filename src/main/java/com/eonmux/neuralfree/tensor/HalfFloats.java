package com.eonmux.neuralfree.tensor;

/**
 * FLOAT16 (IEEE binary16) and BFLOAT16 conversions on bit patterns, the CPU reference of the train library's cast
 * kernels ({@code tr_f32_to_f16_bits} and friends in {@code kernels/train/common/train.h}), which implement the same
 * integer algorithm so device and host casts are bit-identical.
 *
 * <ul>
 *   <li>float to FLOAT16/BFLOAT16: round to nearest, ties to even; overflow to signed infinity; subnormal results
 *       preserved; NaN becomes a quiet NaN keeping the sign ({@code 0x7E00|s} for FLOAT16, {@code 0x7FC0|s} for
 *       BFLOAT16).</li>
 *   <li>FLOAT16/BFLOAT16 to float: exact, sign and NaN payload kept.</li>
 * </ul>
 *
 * <p>16-bit values are passed and returned as {@code int}s holding the bits in the low 16 bits. No floating-point
 * arithmetic is involved, so the results do not depend on denormal handling.</p>
 */
public final class HalfFloats {

    private HalfFloats() {
    }

    /** FLOAT16 bits of a float, rounded to nearest even. */
    public static int floatToHalf(float value) {
        return floatBitsToHalf(Float.floatToRawIntBits(value));
    }

    /** FLOAT16 bits of the float with raw bits {@code x}, rounded to nearest even. */
    public static int floatBitsToHalf(int x) {
        int sign = (x >>> 16) & 0x8000;
        int ax = x & 0x7fffffff;
        if (ax > 0x7f800000) {
            return sign | 0x7e00;
        }
        if (ax >= 0x477ff000) {
            return sign | 0x7c00;
        }
        if (ax >= 0x38800000) {
            int m = ax - 0x38000000;
            return sign | ((m + 0x0fff + ((m >>> 13) & 1)) >>> 13);
        }
        if (ax <= 0x33000000) {
            return sign;
        }
        int e = ax >>> 23;
        int mant = (ax & 0x7fffff) | 0x800000;
        int shift = 126 - e;
        int h = mant >>> shift;
        int rem = mant & ((1 << shift) - 1);
        int halfway = 1 << (shift - 1);
        if (rem > halfway || (rem == halfway && (h & 1) != 0)) {
            h += 1;
        }
        return sign | h;
    }

    /** The float a FLOAT16 bit pattern (low 16 bits of {@code h}) denotes, exactly. */
    public static float halfToFloat(int h) {
        return Float.intBitsToFloat(halfToFloatBits(h));
    }

    /** Raw float bits of a FLOAT16 bit pattern (low 16 bits of {@code h}), exactly. */
    public static int halfToFloatBits(int h) {
        int sign = (h & 0x8000) << 16;
        int e = (h >>> 10) & 0x1f;
        int m = h & 0x3ff;
        if (e == 0x1f) {
            return sign | 0x7f800000 | (m << 13);
        }
        if (e != 0) {
            return sign | ((e + 112) << 23) | (m << 13);
        }
        if (m == 0) {
            return sign;
        }
        int fe = 113;
        while ((m & 0x400) == 0) {
            m <<= 1;
            fe -= 1;
        }
        return sign | (fe << 23) | ((m & 0x3ff) << 13);
    }

    /** BFLOAT16 bits of a float, rounded to nearest even. */
    public static int floatToBFloat16(float value) {
        return floatBitsToBFloat16(Float.floatToRawIntBits(value));
    }

    /** BFLOAT16 bits of the float with raw bits {@code x}, rounded to nearest even. */
    public static int floatBitsToBFloat16(int x) {
        if ((x & 0x7fffffff) > 0x7f800000) {
            return ((x >>> 16) & 0x8000) | 0x7fc0;
        }
        return ((x + 0x7fff + ((x >>> 16) & 1)) >>> 16) & 0xffff;
    }

    /** The float a BFLOAT16 bit pattern (low 16 bits of {@code b}) denotes, exactly. */
    public static float bfloat16ToFloat(int b) {
        return Float.intBitsToFloat(bfloat16ToFloatBits(b));
    }

    /** Raw float bits of a BFLOAT16 bit pattern (low 16 bits of {@code b}), exactly. */
    public static int bfloat16ToFloatBits(int b) {
        return (b & 0xffff) << 16;
    }
}

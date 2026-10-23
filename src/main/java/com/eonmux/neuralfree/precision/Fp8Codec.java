package com.eonmux.neuralfree.precision;

/**
 * The OCP 8-bit floating-point formats, with the semantics of {@code torch.float8_e4m3fn} and
 * {@code torch.float8_e5m2}: a sign bit, an exponent field and a mantissa field, subnormal values when the exponent
 * field is 0, and these special encodings:
 * <ul>
 *   <li>{@link #E4M3FN}: bias 7, largest finite value 448, no infinities; {@code S.1111.111} is NaN.</li>
 *   <li>{@link #E5M2}: bias 15, largest finite value 57344; an all-ones exponent field holds the IEEE infinities
 *       (mantissa 0) and NaNs (mantissa not 0).</li>
 * </ul>
 *
 * <p>Codes are the bit patterns as {@code int}s in {@code [0, 255]}. {@link #encode} rounds to nearest, ties to
 * even, and saturates every finite value whose magnitude reaches the largest finite value to that value with its
 * sign, as the scaled FP8 recipes do before casting. NaN encodes to the canonical NaN with the input's sign bit; an
 * infinity encodes to the infinity of the same sign in E5M2 and to NaN with its sign in E4M3FN, which has no
 * infinities. {@link #decode} is exact. Instances are immutable and thread-safe.</p>
 */
final class Fp8Codec {
    /** E4M3 without infinities ({@code float8_e4m3fn}): bias 7, largest finite 448, smallest subnormal 2^-9. */
    static final Fp8Codec E4M3FN = new Fp8Codec("E4M3FN", 4, 3, false);

    /**
     * E5M2 with IEEE infinities and NaNs ({@code float8_e5m2}): bias 15, largest finite 57344, smallest subnormal
     * 2^-16.
     */
    static final Fp8Codec E5M2 = new Fp8Codec("E5M2", 5, 2, true);

    private static final int SIGN = 0x80;
    private static final int MAGNITUDE = 0x7F;

    private final String name;
    private final int mantissaBits;
    private final int bias;
    private final int maxExponentField;
    private final boolean ieeeSpecials;
    private final int maxFiniteCode;
    private final int nanCode;
    private final int infinityCode;
    private final float maxFinite;
    private final float[] values;

    private Fp8Codec(String name, int exponentBits, int mantissaBits, boolean ieeeSpecials) {
        this.name = name;
        this.mantissaBits = mantissaBits;
        this.bias = (1 << (exponentBits - 1)) - 1;
        this.maxExponentField = (1 << exponentBits) - 1;
        this.ieeeSpecials = ieeeSpecials;
        int mantissaMask = (1 << mantissaBits) - 1;
        if (ieeeSpecials) {
            this.maxFiniteCode = ((maxExponentField - 1) << mantissaBits) | mantissaMask;
            this.infinityCode = maxExponentField << mantissaBits;
            this.nanCode = (maxExponentField << mantissaBits) | (1 << (mantissaBits - 1));
        } else {
            this.maxFiniteCode = (maxExponentField << mantissaBits) | (mantissaMask - 1);
            this.infinityCode = -1;
            this.nanCode = (maxExponentField << mantissaBits) | mantissaMask;
        }
        this.values = new float[256];
        for (int code = 0; code < 256; code++) {
            values[code] = computeValue(code);
        }
        this.maxFinite = values[maxFiniteCode];
    }

    /** The largest finite value: 448 for E4M3FN, 57344 for E5M2. */
    float maxFinite() {
        return maxFinite;
    }

    /** The smallest normal value {@code 2^(1 - bias)}: {@code 2^-6} for E4M3FN, {@code 2^-14} for E5M2. */
    float minNormal() {
        return Math.scalb(1.0f, 1 - bias);
    }

    /** The width of the mantissa field: 3 for E4M3FN, 2 for E5M2. */
    int mantissaBits() {
        return mantissaBits;
    }

    /** True when the format encodes the infinities (E5M2); E4M3FN encodes an infinity as NaN. */
    boolean hasInfinities() {
        return ieeeSpecials;
    }

    /** The positive canonical NaN code: {@code 0x7F} for E4M3FN, {@code 0x7E} for E5M2. */
    int nanCode() {
        return nanCode;
    }

    /** True when {@code code} encodes NaN. */
    boolean isNaN(int code) {
        int magnitude = checkCode(code) & MAGNITUDE;
        if (ieeeSpecials) {
            return (magnitude >>> mantissaBits) == maxExponentField && (magnitude & ((1 << mantissaBits) - 1)) != 0;
        }
        return magnitude == nanCode;
    }

    /**
     * The value of {@code code}, exactly; a NaN code gives a quiet float NaN carrying the code's sign bit.
     *
     * @throws IllegalArgumentException when {@code code} is outside {@code [0, 255]}
     */
    float decode(int code) {
        return values[checkCode(code)];
    }

    /** The code of {@code value}, rounded to nearest with ties to even and saturated as the class describes. */
    int encode(float value) {
        int sign = (Float.floatToRawIntBits(value) >>> 24) & SIGN;
        if (Float.isNaN(value)) {
            return sign | nanCode;
        }
        float magnitude = Math.abs(value);
        if (Float.isInfinite(magnitude)) {
            return sign | (infinityCode >= 0 ? infinityCode : nanCode);
        }
        if (magnitude >= maxFinite) {
            return sign | maxFiniteCode;
        }
        if (magnitude == 0.0f) {
            return sign;
        }
        int minNormalExponent = 1 - bias;
        int exponent = Math.max(Math.getExponent(magnitude), minNormalExponent);
        long units = (long) Math.rint(Math.scalb((double) magnitude, mantissaBits - exponent));
        int implicitOne = 1 << mantissaBits;
        if (units < implicitOne) {
            return sign | (int) units;
        }
        if (units == (long) implicitOne << 1) {
            units = implicitOne;
            exponent++;
        }
        int bits = ((exponent + bias) << mantissaBits) | ((int) units - implicitOne);
        return sign | Math.min(bits, maxFiniteCode);
    }

    private float computeValue(int code) {
        boolean negative = (code & SIGN) != 0;
        int magnitude = code & MAGNITUDE;
        int exponentField = magnitude >>> mantissaBits;
        int mantissa = magnitude & ((1 << mantissaBits) - 1);
        if (ieeeSpecials && exponentField == maxExponentField) {
            if (mantissa == 0) {
                return negative ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            }
            return signedNaN(negative);
        }
        if (!ieeeSpecials && magnitude == nanCode) {
            return signedNaN(negative);
        }
        float value = exponentField == 0
            ? Math.scalb((float) mantissa, 1 - bias - mantissaBits)
            : Math.scalb((float) ((1 << mantissaBits) | mantissa), exponentField - bias - mantissaBits);
        return negative ? -value : value;
    }

    private static float signedNaN(boolean negative) {
        return Float.intBitsToFloat(negative ? 0xFFC00000 : 0x7FC00000);
    }

    private static int checkCode(int code) {
        if (code < 0 || code > 0xFF) {
            throw new IllegalArgumentException("FP8 code must be in [0, 255], got: " + code);
        }
        return code;
    }

    @Override
    public String toString() {
        return name;
    }
}

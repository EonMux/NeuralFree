package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

/**
 * FP8 fake quantization as a {@link TensorOps} composition on the tensor's backend: per element the value
 * {@code decode(encode(x / scale)) * scale} of {@link Quantizer#fp8Code} and {@link Fp8Codec#decode}, the value
 * post-training quantization of an FP8 type reconstructs.
 *
 * <p><b>Rounding without bit access.</b> With {@code y = x / scale}, the FP8 grid spacing at {@code y} is
 * {@code step = ufp(max(|y|, minNormal)) * 2^-m} ({@code m} mantissa bits; the subnormal spacing below the smallest
 * normal value), where {@code ufp(a) = 2^floor(log2 a)} is computed exactly with Rump's unit-in-the-first-place
 * algorithm {@code ufp(a) = |phi * a - (1 - 2^-24) * (phi * a)|}, {@code phi = 2^23 + 1}. Adding and subtracting
 * {@code 1.5 * 2^23 * step} to {@code clamp(y, -fmax, fmax)} rounds it to a multiple of {@code step} with ties to even,
 * since the sum lies in the binade whose spacing is {@code step} and {@code 1.5 * 2^23} is even; the clamp gives the
 * saturation to {@code +-fmax}. A result of 0 takes the sign of {@code y} ({@code y * 0}). Every step is an exact
 * IEEE operation, so the composition equals the codec for every finite {@code y} (checked over all 2^32 FLOAT32 bit
 * patterns for both formats). An infinite {@code x} gives the codec's encoding of an infinity: itself in E5M2 (or
 * NaN when {@code scale} is infinite), NaN in E4M3FN; a finite {@code x} whose quotient overflows saturates, as
 * {@link Quantizer#fp8Code} does; NaN stays NaN.</p>
 *
 * <p><b>Tolerance.</b> Only the division leaves the bit-identical classes of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}: on a device it is in the 4-ulp class, so a
 * quotient within 4 ulp of a rounding boundary may take the neighbouring FP8 value; the scaling, rounding and final
 * product are single-rounding operations, and a quotient below {@code FLT_MIN} may flush to a zero of either sign.</p>
 */
final class Fp8FakeQuantize {
    /** {@code phi = 2^23 + 1} of the unit-in-the-first-place algorithm for FLOAT32. */
    private static final float UFP_SPREAD = 8388609.0f;
    /** {@code 1 - 2^-24}, the largest FLOAT32 below 1. */
    private static final float UFP_SHRINK = Math.nextDown(1.0f);

    private Fp8FakeQuantize() {
    }

    /**
     * The FP8 fake-quantized values of {@code x} (FLOAT32, of {@code x}'s shape) with {@code scale} (a FLOAT32
     * tensor broadcast against {@code x}: rank 0, or {@code [C, 1, ..., 1]} per channel) and, with {@code withMask},
     * the BOOL in-range mask {@code |x / scale| <= fmax} (false for NaN and infinite quotients), the clipped
     * straight-through estimator's mask. New tensors; the intermediates are closed. A non-FLOAT32 {@code x} is
     * converted first.
     */
    static Tensor[] quantize(Tensor x, Tensor scale, Fp8Codec codec, boolean withMask) {
        Tensor input = x.dtype() == DType.FLOAT32 ? x : x.to(DType.FLOAT32);
        try (Tensor y = TensorOps.div(input, scale);
             Tensor magnitude = TensorOps.abs(y)) {
            Tensor values = values(input, y, magnitude, scale, codec);
            if (!withMask) {
                return new Tensor[]{values};
            }
            try (Tensor inRange = TensorOps.clamp(magnitude, 0.0f, codec.maxFinite())) {
                return new Tensor[]{values, TensorOps.eq(magnitude, inRange)};
            } catch (RuntimeException e) {
                closeAfterFailure(values, e);
                throw e;
            }
        } finally {
            if (input != x) {
                input.close();
            }
        }
    }

    /** {@code decode(encode(y)) * scale} per element, with the infinite inputs of the class description. */
    private static Tensor values(Tensor x, Tensor y, Tensor magnitude, Tensor scale, Fp8Codec codec) {
        try (Tensor rounded = rounded(y, magnitude, codec);
             Tensor scaled = TensorOps.mul(rounded, scale);
             Tensor absolute = TensorOps.abs(x);
             Tensor infinity = Quantizer.scalarOn(x.backend(), Float.POSITIVE_INFINITY);
             Tensor infinite = TensorOps.eq(absolute, infinity)) {
            if (codec.hasInfinities()) {
                return TensorOps.where(infinite, y, scaled);
            }
            try (Tensor zeros = Tensor.zeros(x.backend(), x.shape(), DType.FLOAT32);
                 Tensor nan = TensorOps.add(zeros, Float.NaN)) {
                return TensorOps.where(infinite, nan, scaled);
            }
        }
    }

    /** {@code y} rounded to the FP8 grid with ties to even and saturated to {@code +-fmax}; NaN stays NaN. */
    private static Tensor rounded(Tensor y, Tensor magnitude, Fp8Codec codec) {
        float fmax = codec.maxFinite();
        float shifterPerBinade = 1.5f * Math.scalb(1.0f, 23 - codec.mantissaBits());
        try (Tensor bounded = TensorOps.clamp(magnitude, codec.minNormal(), fmax);
             Tensor spread = TensorOps.mul(bounded, UFP_SPREAD);
             Tensor shrunk = TensorOps.mul(spread, UFP_SHRINK);
             Tensor difference = TensorOps.sub(spread, shrunk);
             Tensor binade = TensorOps.abs(difference);
             Tensor shifter = TensorOps.mul(binade, shifterPerBinade);
             Tensor saturated = TensorOps.clamp(y, -fmax, fmax);
             Tensor shifted = TensorOps.add(saturated, shifter);
             Tensor gridded = TensorOps.sub(shifted, shifter);
             Tensor zero = Quantizer.scalarOn(y.backend(), 0.0f);
             Tensor isZero = TensorOps.eq(gridded, zero);
             Tensor signedZero = TensorOps.mul(y, 0.0f)) {
            return TensorOps.where(isZero, signedZero, gridded);
        }
    }

    /** Closes {@code t}, adding a close failure to {@code failure} as suppressed. */
    private static void closeAfterFailure(Tensor t, RuntimeException failure) {
        try {
            t.close();
        } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
        }
    }
}

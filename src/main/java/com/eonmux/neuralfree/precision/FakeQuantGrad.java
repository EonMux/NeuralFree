package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Fake quantization as one recorded operation, with the clipped straight-through estimator as its gradient.
 *
 * <p><b>Forward</b> ({@link #apply}). Per element, on the tensor's backend, the host formula
 * {@code q = clamp(Math.round(x / scale) + zeroPoint, qMin, qMax); y = (q - zeroPoint) * scale}, as the
 * {@link TensorOps} composition: divide by a device scalar, {@code round} (Java {@code Math.round} kept as a float),
 * NaN to the code 0 (as {@code Math.round(NaN)}), add the zero point, {@code clamp}, subtract the zero point, multiply
 * by the scale. Every step is exact on integer values except the division and the final product, which round once as
 * the host formula does (on a device the division is in the 4-ulp class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}, so a quotient within 4 ulp of a half-way point may
 * take the neighbouring code); a quotient beyond the int range saturates to {@code qMin}/{@code qMax}.</p>
 *
 * <p><b>Per channel</b> ({@link #applyPerChannel}). The same formula with channel {@code c} of dimension 0 using its
 * own {@code scale[c]} and {@code zeroPoint[c]} (PyTorch's {@code fake_quantize_per_channel_affine} with axis 0): the
 * scales and zero points are uploaded as {@code [C, 1, ..., 1]} constants and broadcast, so every step rounds as the
 * per-tensor composition does.</p>
 *
 * <p><b>FP8</b> ({@link #applyFp8}, {@link #applyFp8PerChannel}). The FP8 types fake-quantize to the value their
 * post-training codes reconstruct, {@code y = decode(encode(x / scale)) * scale} with the OCP FP8 encoding of
 * {@link Quantizer} (round to nearest even, saturation to {@code +-fmax}), computed on the tensor's backend by an exact
 * composition ({@code Fp8FakeQuantize}) whose only inexact step on a device is the division; the code range of the
 * estimator below is {@code |x / scale| <= fmax}.</p>
 *
 * <p><b>Backward</b> (clipped straight-through estimator, as PyTorch's {@code fake_quantize_per_tensor_affine}):
 * {@code gx = where(qMin <= round(x / scale) + zeroPoint <= qMax, gy, 0)}. Elements whose code is clamped receive no
 * gradient, and so do NaN and infinite inputs (their code is not in range). Saved: the in-range mask, a BOOL tensor of
 * the input's shape computed from the input without recording (owned by this function and closed by
 * {@link #clearSavedTensors()}). Tolerance: bit-identical (selection). Create-graph is supported: the backward is one
 * recorded {@code where} over {@code gy} with a constant mask, so the estimator's second derivative is 0.</p>
 */
public final class FakeQuantGrad implements GradientFunction {
    private Tensor inRange;
    private List<Tensor> saved;

    /** Takes over {@code inRange}, the BOOL in-range mask of the recorded input. */
    private FakeQuantGrad(Tensor inRange) {
        this.inRange = inRange;
        this.saved = Collections.singletonList(inRange);
    }

    /**
     * {@code x} fake-quantized with {@code scale}, {@code zeroPoint} and the code range {@code [qMin, qMax]}, as a new
     * tensor the caller owns. Recorded with a {@link FakeQuantGrad} over {@code x} when {@code x} requires grad and
     * grad mode is on ({@link Autograd#needsGrad}); otherwise the result does not require grad.
     *
     * @throws IllegalArgumentException when {@code scale} is not positive (or NaN) or {@code qMin > qMax}
     */
    public static Tensor apply(Tensor x, float scale, int zeroPoint, int qMin, int qMax) {
        Objects.requireNonNull(x, "Tensor cannot be null");
        if (!(scale > 0.0f)) {
            throw new IllegalArgumentException("Fake quantization scale must be positive, got: " + scale);
        }
        if (qMin > qMax) {
            throw new IllegalArgumentException("Code range [" + qMin + ", " + qMax + "] is empty");
        }
        if (!Autograd.needsGrad(x)) {
            return quantize(x, scale, zeroPoint, qMin, qMax, false)[0];
        }
        Tensor[] raw = Autograd.untracked(() -> quantize(x, scale, zeroPoint, qMin, qMax, true));
        return record(x, raw[0], raw[1]);
    }

    /**
     * {@code x} fake-quantized per channel along dimension 0, channel {@code c} with {@code scales[c]},
     * {@code zeroPoints[c]} and the code range {@code [qMin, qMax]}, as a new tensor the caller owns; recorded as
     * {@link #apply} is.
     *
     * @throws IllegalArgumentException when {@code x} has rank 0, the arrays do not have one entry per channel, a scale
     *                                  is not positive (or NaN), or {@code qMin > qMax}
     */
    public static Tensor applyPerChannel(Tensor x, float[] scales, int[] zeroPoints, int qMin, int qMax) {
        Objects.requireNonNull(x, "Tensor cannot be null");
        Objects.requireNonNull(zeroPoints, "Zero points cannot be null");
        Shape broadcast = channelShape(x, scales);
        int channels = scales.length;
        if (zeroPoints.length != channels) {
            throw new IllegalArgumentException("Per-channel fake quantization of " + x.shape() + " needs " + channels
                + " zero points, got " + zeroPoints.length);
        }
        float[] zeros = new float[channels];
        for (int c = 0; c < channels; c++) {
            zeros[c] = zeroPoints[c];
        }
        if (qMin > qMax) {
            throw new IllegalArgumentException("Code range [" + qMin + ", " + qMax + "] is empty");
        }
        boolean recorded = Autograd.needsGrad(x);
        Tensor[] raw = Autograd.untracked(() -> {
            try (Tensor scaleRow = Tensor.fromArray(x.backend(), scales);
                 Tensor scaleColumn = scaleRow.reshape(broadcast);
                 Tensor zeroRow = Tensor.fromArray(x.backend(), zeros);
                 Tensor zeroColumn = zeroRow.reshape(broadcast)) {
                return quantize(x, scaleColumn, zeroColumn, qMin, qMax, recorded);
            }
        });
        return recorded ? record(x, raw[0], raw[1]) : raw[0];
    }

    /**
     * {@code x} fake-quantized to the FP8 type {@code type} with {@code scale} (class description), as a new FLOAT32
     * tensor the caller owns; recorded as {@link #apply} is, with the in-range mask {@code |x / scale| <= fmax}.
     *
     * @throws IllegalArgumentException when {@code type} is not an FP8 type or {@code scale} is not positive (or NaN)
     */
    public static Tensor applyFp8(Tensor x, float scale, Quantizer.QuantizationType type) {
        Objects.requireNonNull(x, "Tensor cannot be null");
        Fp8Codec codec = Quantizer.codec(Objects.requireNonNull(type, "Quantization type cannot be null"));
        if (!(scale > 0.0f)) {
            throw new IllegalArgumentException("Fake quantization scale must be positive, got: " + scale);
        }
        boolean recorded = Autograd.needsGrad(x);
        Tensor[] raw = Autograd.untracked(() -> {
            try (Tensor scaleTensor = Quantizer.scalarOn(x.backend(), scale)) {
                return Fp8FakeQuantize.quantize(x, scaleTensor, codec, recorded);
            }
        });
        return recorded ? record(x, raw[0], raw[1]) : raw[0];
    }

    /**
     * {@code x} fake-quantized to the FP8 type {@code type} per channel along dimension 0, channel {@code c} with
     * {@code scales[c]}, as a new FLOAT32 tensor the caller owns; recorded as {@link #applyFp8} is.
     *
     * @throws IllegalArgumentException when {@code type} is not an FP8 type, {@code x} has rank 0, {@code scales} does
     *                                  not have one entry per channel, or a scale is not positive (or NaN)
     */
    public static Tensor applyFp8PerChannel(Tensor x, float[] scales, Quantizer.QuantizationType type) {
        Objects.requireNonNull(x, "Tensor cannot be null");
        Fp8Codec codec = Quantizer.codec(Objects.requireNonNull(type, "Quantization type cannot be null"));
        Shape broadcast = channelShape(x, scales);
        boolean recorded = Autograd.needsGrad(x);
        Tensor[] raw = Autograd.untracked(() -> {
            try (Tensor scaleRow = Tensor.fromArray(x.backend(), scales);
                 Tensor scaleColumn = scaleRow.reshape(broadcast)) {
                return Fp8FakeQuantize.quantize(x, scaleColumn, codec, recorded);
            }
        });
        return recorded ? record(x, raw[0], raw[1]) : raw[0];
    }

    /**
     * The broadcast shape {@code [C, 1, ..., 1]} of per-channel constants for {@code x}, after checking that
     * {@code x} has rank 1 or more and {@code scales} holds one positive scale per channel of dimension 0.
     */
    private static Shape channelShape(Tensor x, float[] scales) {
        Objects.requireNonNull(scales, "Scales cannot be null");
        if (x.shape().rank() == 0) {
            throw new IllegalArgumentException("Per-channel fake quantization needs a tensor of rank >= 1");
        }
        int channels = x.shape().dim(0);
        if (scales.length != channels) {
            throw new IllegalArgumentException("Per-channel fake quantization of " + x.shape() + " needs " + channels
                + " scales, got " + scales.length);
        }
        for (int c = 0; c < channels; c++) {
            if (!(scales[c] > 0.0f)) {
                throw new IllegalArgumentException("Fake quantization scale of channel " + c
                    + " must be positive, got: " + scales[c]);
            }
        }
        int[] dims = new int[x.shape().rank()];
        Arrays.fill(dims, 1);
        dims[0] = channels;
        return Shape.of(dims);
    }

    /** Records {@code values} as the fake quantization of {@code x} with the in-range {@code mask} (taken over). */
    private static Tensor record(Tensor x, Tensor values, Tensor mask) {
        boolean[] handedOver = {false};
        try {
            return Autograd.record(values, Collections.singletonList(x), out -> {
                handedOver[0] = true;
                return new FakeQuantGrad(mask);
            });
        } catch (RuntimeException e) {
            if (!handedOver[0]) {
                closeAfterFailure(mask, e);
            }
            throw e;
        }
    }

    /**
     * {@link #quantize(Tensor, float, int, int, int, boolean)} with broadcast scale and zero-point tensors
     * ({@code [C, 1, ..., 1]}, FLOAT32): the per-channel formula, every step rounding as the per-tensor one does.
     */
    private static Tensor[] quantize(Tensor x, Tensor scale, Tensor zeroPoint, int qMin, int qMax, boolean withMask) {
        try (Tensor scaled = TensorOps.div(x, scale);
             Tensor rounded = TensorOps.round(scaled);
             Tensor notNaN = TensorOps.eq(rounded, rounded);
             Tensor zeros = Tensor.zeros(rounded.backend(), rounded.shape());
             Tensor codes = TensorOps.where(notNaN, rounded, zeros);
             Tensor shifted = TensorOps.add(codes, zeroPoint);
             Tensor clamped = TensorOps.clamp(shifted, qMin, qMax);
             Tensor centered = TensorOps.sub(clamped, zeroPoint)) {
            Tensor values = TensorOps.mul(centered, scale);
            if (!withMask) {
                return new Tensor[]{values};
            }
            try (Tensor unclamped = TensorOps.add(rounded, zeroPoint);
                 Tensor inRangeCodes = TensorOps.clamp(unclamped, qMin, qMax)) {
                return new Tensor[]{values, TensorOps.eq(unclamped, inRangeCodes)};
            } catch (RuntimeException e) {
                closeAfterFailure(values, e);
                throw e;
            }
        }
    }

    /**
     * The fake-quantized values and, with {@code withMask}, the BOOL in-range mask
     * {@code round(x / scale) + zeroPoint == clamp(round(x / scale) + zeroPoint, qMin, qMax)} (false for NaN), both
     * new tensors; the intermediates are closed.
     */
    private static Tensor[] quantize(Tensor x, float scale, int zeroPoint, int qMin, int qMax, boolean withMask) {
        try (Tensor divisor = Quantizer.scalarOn(x.backend(), scale);
             Tensor scaled = TensorOps.div(x, divisor);
             Tensor rounded = TensorOps.round(scaled);
             Tensor notNaN = TensorOps.eq(rounded, rounded);
             Tensor zeros = Tensor.zeros(rounded.backend(), rounded.shape());
             Tensor codes = TensorOps.where(notNaN, rounded, zeros);
             Tensor shifted = TensorOps.add(codes, (float) zeroPoint);
             Tensor clamped = TensorOps.clamp(shifted, qMin, qMax);
             Tensor centered = TensorOps.add(clamped, (float) -zeroPoint)) {
            Tensor values = TensorOps.mul(centered, scale);
            if (!withMask) {
                return new Tensor[]{values};
            }
            try (Tensor unclamped = TensorOps.add(rounded, (float) zeroPoint);
                 Tensor inRangeCodes = TensorOps.clamp(unclamped, qMin, qMax)) {
                return new Tensor[]{values, TensorOps.eq(unclamped, inRangeCodes)};
            } catch (RuntimeException e) {
                closeAfterFailure(values, e);
                throw e;
            }
        }
    }

    /**
     * {@code where(inRange, gradOutput, 0)}, a new tensor of the input's shape.
     *
     * @throws IllegalStateException    when the node was released
     * @throws IllegalArgumentException when {@code gradOutput} does not have the input's shape
     */
    @Override
    public List<Tensor> backward(Tensor gradOutput) {
        Tensor mask = inRange;
        if (mask == null) {
            throw new IllegalStateException("The fake quantization node was released; its mask is no longer available");
        }
        if (!mask.shape().equals(gradOutput.shape())) {
            throw new IllegalArgumentException("Fake quantization gradient " + gradOutput.shape()
                + " does not match the input shape " + mask.shape());
        }
        try (Tensor zeros = Tensor.zeros(gradOutput.backend(), gradOutput.shape(), gradOutput.dtype())) {
            return Collections.singletonList(TensorOps.where(mask, gradOutput, zeros));
        }
    }

    @Override
    public List<Tensor> savedTensors() {
        return saved;
    }

    @Override
    public void clearSavedTensors() {
        Tensor mask = inRange;
        if (mask != null) {
            inRange = null;
            saved = Collections.emptyList();
            mask.close();
        }
    }

    @Override
    public boolean supportsCreateGraph() {
        return true;
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

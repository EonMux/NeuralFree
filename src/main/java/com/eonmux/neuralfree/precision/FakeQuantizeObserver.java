package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Objects;

/**
 * The observer state of one fake-quantized site of a {@link QATModule}, as PyTorch's {@code FakeQuantize} with a
 * {@code MinMaxObserver} holds it: the running range {@code min_val} and {@code max_val} and the derived {@code scale}
 * and {@code zero_point}, each a persistent FLOAT32 buffer of one element per channel (one for a per-tensor site), so
 * {@link Module#stateDict()}, the model serializer and checkpoints save and restore the calibration. The buffers keep
 * FLOAT32 through {@link Module#to(com.eonmux.neuralfree.tensor.DType)}; the zero point holds integer values.
 *
 * <p>A new observer has the empty range ({@code min_val = Float.MAX_VALUE}, {@code max_val = -Float.MAX_VALUE}),
 * scale 1 and zero point 0. The buffers are the only copy of the state: {@link #observe} reads the running range
 * back in one transfer and writes all four buffers, and {@link #quantParams()} reads the scale and zero point back in
 * one transfer, so a state loaded into the buffers is what the next forward pass uses.</p>
 */
final class FakeQuantizeObserver extends Module {
    static final String MIN_VAL = "min_val";
    static final String MAX_VAL = "max_val";
    static final String SCALE = "scale";
    static final String ZERO_POINT = "zero_point";

    private final int channels;
    private final Quantizer.QuantizationType quantType;
    private final float eps;

    /**
     * An observer of {@code channels} channels (1 for a per-tensor site) whose buffers live on {@code backend}.
     *
     * @throws IllegalArgumentException when {@code channels < 1} or {@code eps} is not positive
     */
    FakeQuantizeObserver(ComputeBackend backend, int channels, Quantizer.QuantizationType quantType, float eps) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        this.quantType = Objects.requireNonNull(quantType, "Quantization type cannot be null");
        if (channels < 1) {
            throw new IllegalArgumentException("An observer needs at least one channel, got " + channels);
        }
        if (!(eps > 0.0f)) {
            throw new IllegalArgumentException("Observer eps must be positive, got: " + eps);
        }
        this.channels = channels;
        this.eps = eps;
        Quantizer.QuantizationParams empty = Quantizer.emptyRangeParams(quantType);
        register(backend, MIN_VAL, Float.MAX_VALUE);
        register(backend, MAX_VAL, -Float.MAX_VALUE);
        register(backend, SCALE, empty.scale());
        register(backend, ZERO_POINT, empty.zeroPoint());
    }

    private void register(ComputeBackend backend, String name, float value) {
        float[] values = new float[channels];
        Arrays.fill(values, value);
        registerBuffer(name, Tensor.fromArray(backend, values), true, true);
    }

    int channels() {
        return channels;
    }

    /**
     * Folds the observed per-channel range {@code {mins, maxs}} into the running range and stores it with the scale
     * and zero point derived from it, as post-training quantization derives them from a MIN_MAX range: for the
     * integer types {@link Quantizer#integerParams}; for the FP8 types zero point 0 and
     * {@link Quantizer#fp8Scale} of {@code max(|min|, |max|)}. A channel whose running range is still empty
     * ({@code min > max}, as after observing only NaN values) keeps PyTorch's empty-range parameters, scale 1 and zero
     * point 0 ({@link Quantizer#emptyRangeParams}), the parameters of a new observer. Returns those parameters.
     *
     * @throws IllegalArgumentException when the arrays do not have one entry per channel
     */
    QuantParams observe(float[] mins, float[] maxs) {
        if (mins.length != channels || maxs.length != channels) {
            throw new IllegalArgumentException("Observed " + mins.length + " minima and " + maxs.length
                + " maxima for an observer of " + channels + " channels");
        }
        float[] running = readPair(MIN_VAL, MAX_VAL);
        float[] newMins = new float[channels];
        float[] newMaxs = new float[channels];
        float[] scales = new float[channels];
        int[] zeroPoints = new int[channels];
        float[] zeroPointValues = new float[channels];
        for (int c = 0; c < channels; c++) {
            newMins[c] = Math.min(running[c], mins[c]);
            newMaxs[c] = Math.max(running[channels + c], maxs[c]);
            if (newMins[c] > newMaxs[c]) {
                Quantizer.QuantizationParams empty = Quantizer.emptyRangeParams(quantType);
                scales[c] = empty.scale();
                zeroPoints[c] = empty.zeroPoint();
                zeroPointValues[c] = empty.zeroPoint();
            } else if (Quantizer.isFp8(quantType)) {
                scales[c] = Quantizer.fp8Scale(quantType, Math.max(Math.abs(newMins[c]), Math.abs(newMaxs[c])));
            } else {
                Quantizer.QuantizationParams params = Quantizer.integerParams(quantType, newMins[c], newMaxs[c], eps);
                scales[c] = params.scale();
                zeroPoints[c] = params.zeroPoint();
                zeroPointValues[c] = params.zeroPoint();
            }
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            GpuOptimizerOps.storeFloats(buffer(MIN_VAL), newMins);
            GpuOptimizerOps.storeFloats(buffer(MAX_VAL), newMaxs);
            GpuOptimizerOps.storeFloats(buffer(SCALE), scales);
            GpuOptimizerOps.storeFloats(buffer(ZERO_POINT), zeroPointValues);
        }
        return new QuantParams(scales, zeroPoints);
    }

    /** The stored scale and zero point of every channel. */
    QuantParams quantParams() {
        float[] stored = readPair(SCALE, ZERO_POINT);
        float[] scales = Arrays.copyOf(stored, channels);
        int[] zeroPoints = new int[channels];
        for (int c = 0; c < channels; c++) {
            zeroPoints[c] = Math.round(stored[channels + c]);
        }
        return new QuantParams(scales, zeroPoints);
    }

    /** The values of buffers {@code first} and {@code second}, concatenated, read back in one transfer. */
    private float[] readPair(String first, String second) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor both = TensorOps.cat(Arrays.asList(buffer(first), buffer(second)), 0)) {
            return both.toFloatArray();
        }
    }

    /** An observer holds state only; the fake quantization itself is {@link QATModule}'s. */
    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("FakeQuantizeObserver has no forward pass; QATModule applies it");
    }

    @Override
    public String moduleName() {
        return "FakeQuantizeObserver";
    }

    /** Per-channel scales and zero points, one entry per channel. */
    static final class QuantParams {
        final float[] scales;
        final int[] zeroPoints;

        QuantParams(float[] scales, int[] zeroPoints) {
            this.scales = scales;
            this.zeroPoints = zeroPoints;
        }
    }
}

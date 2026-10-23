package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.precision.Quantizer.CalibrationMethod;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationParams;
import com.eonmux.neuralfree.precision.Quantizer.QuantizationType;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The post-training calibration state of a {@link Quantizer}, keyed by parameter path, and the quantization
 * parameters derived from it: the running range of every parameter for MIN_MAX, its {@link CalibrationHistogram} for
 * the histogram methods. Observations merge, so the derived parameters depend only on the values observed.
 *
 * <p>With {@code perChannel}, a parameter of rank at least 1 whose dimension 0 is not empty is calibrated per channel
 * along dimension 0: every channel keeps its own range or histogram and derives its own scale and zero point exactly as
 * a parameter holding only that channel's values would, and the parameters are
 * {@link QuantizationParams#perChannel}. A parameter keeps the layout of its first observation.</p>
 */
final class PtqCalibrator {
    /** The channel count recorded for a parameter calibrated per tensor. */
    private static final int PER_TENSOR = 0;
    /** The channel count reported for a parameter without calibration state. */
    private static final int UNOBSERVED = -1;

    private final QuantizationType quantType;
    private final CalibrationMethod method;
    private final float percentile;
    private final int maxSamples;
    private final boolean perChannel;
    private final Map<String, float[]> observedRanges = new HashMap<>();
    private final Map<String, float[][]> observedChannelRanges = new HashMap<>();
    private final Map<String, CalibrationHistogram> histograms = new HashMap<>();
    private final Map<String, CalibrationHistogram[]> channelHistograms = new HashMap<>();

    PtqCalibrator(QuantizationType quantType, CalibrationMethod method, float percentile, int maxSamples,
                  boolean perChannel) {
        this.quantType = Objects.requireNonNull(quantType, "Quantization type cannot be null");
        this.method = Objects.requireNonNull(method, "Calibration method cannot be null");
        this.percentile = percentile;
        this.maxSamples = maxSamples;
        this.perChannel = perChannel;
    }

    /** True when the parameter at {@code path} has been observed. */
    boolean isObserved(String path) {
        return observedChannels(path) != UNOBSERVED;
    }

    /**
     * The channel count of the calibration state at {@code path}: {@link #PER_TENSOR} for a per-tensor parameter,
     * {@link #UNOBSERVED} without state.
     */
    private int observedChannels(String path) {
        if (observedRanges.containsKey(path) || histograms.containsKey(path)) {
            return PER_TENSOR;
        }
        float[][] ranges = observedChannelRanges.get(path);
        if (ranges != null) {
            return ranges[0].length;
        }
        CalibrationHistogram[] rows = channelHistograms.get(path);
        return rows == null ? UNOBSERVED : rows.length;
    }

    /** The channel count {@code tensor} is calibrated with: its dimension 0 per channel, else {@link #PER_TENSOR}. */
    private int channelsOf(Tensor tensor) {
        Shape shape = tensor.shape();
        return perChannel && shape.rank() >= 1 && shape.dim(0) >= 1 ? shape.dim(0) : PER_TENSOR;
    }

    /**
     * Observes every entry of {@code parameters} and then merges the observations into the state: the observed range
     * ({@link Quantizer#observedRange}, per channel {@link Quantizer#observedChannelRanges}) for MIN_MAX, the
     * histogram of the calibration values otherwise. Nothing changes when an observation throws.
     *
     * @throws IllegalArgumentException when a histogram method observes an infinite value, or a parameter calibrated
     *                                  per channel is observed with another channel count or per tensor (or the
     *                                  reverse)
     */
    void observe(Map<String, Parameter> parameters) {
        if (method == CalibrationMethod.MIN_MAX) {
            observeRanges(parameters);
        } else {
            observeHistograms(parameters);
        }
    }

    private void observeRanges(Map<String, Parameter> parameters) {
        Map<String, float[]> merged = new LinkedHashMap<>();
        Map<String, float[][]> mergedChannels = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String path = entry.getKey();
            Tensor tensor = entry.getValue().tensor();
            int channels = requireLayout(path, channelsOf(tensor));
            if (channels == PER_TENSOR) {
                float[] range = Quantizer.observedRange(tensor);
                float[] previous = observedRanges.get(path);
                merged.put(path, previous == null ? range
                    : new float[]{Math.min(previous[0], range[0]), Math.max(previous[1], range[1])});
                continue;
            }
            float[][] ranges = Quantizer.observedChannelRanges(tensor);
            float[][] previous = observedChannelRanges.get(path);
            if (previous != null) {
                float[] mins = new float[channels];
                float[] maxs = new float[channels];
                for (int c = 0; c < channels; c++) {
                    mins[c] = Math.min(previous[0][c], ranges[0][c]);
                    maxs[c] = Math.max(previous[1][c], ranges[1][c]);
                }
                ranges = new float[][]{mins, maxs};
            }
            mergedChannels.put(path, ranges);
        }
        observedRanges.putAll(merged);
        observedChannelRanges.putAll(mergedChannels);
    }

    private void observeHistograms(Map<String, Parameter> parameters) {
        Map<String, CalibrationHistogram> merged = new LinkedHashMap<>();
        Map<String, CalibrationHistogram[]> mergedChannels = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String path = entry.getKey();
            Tensor tensor = entry.getValue().tensor();
            int channels = requireLayout(path, channelsOf(tensor));
            if (channels == PER_TENSOR) {
                CalibrationHistogram histogram = histogramOf(site(path), calibrationValues(tensor, 1));
                CalibrationHistogram previous = histograms.get(path);
                merged.put(path, previous == null ? histogram : previous.merge(histogram));
                continue;
            }
            float[] values = calibrationValues(tensor, channels);
            int rowSamples = values.length / channels;
            CalibrationHistogram[] previous = channelHistograms.get(path);
            CalibrationHistogram[] rows = new CalibrationHistogram[channels];
            for (int c = 0; c < channels; c++) {
                float[] row = Arrays.copyOfRange(values, c * rowSamples, (c + 1) * rowSamples);
                CalibrationHistogram histogram = histogramOf(channelSite(path, c), row);
                rows[c] = previous == null ? histogram : previous[c].merge(histogram);
            }
            mergedChannels.put(path, rows);
        }
        histograms.putAll(merged);
        channelHistograms.putAll(mergedChannels);
    }

    /**
     * {@code channels}, the layout of a new observation of {@code path}, after checking that it matches the layout of
     * the state already held for it.
     *
     * @throws IllegalArgumentException when the parameter was calibrated with another layout
     */
    private int requireLayout(String path, int channels) {
        int previous = observedChannels(path);
        if (previous != UNOBSERVED && previous != channels) {
            throw new IllegalArgumentException("Parameter '" + path + "' was calibrated " + layout(previous)
                + " and is now observed " + layout(channels));
        }
        return channels;
    }

    private static String layout(int channels) {
        return channels == PER_TENSOR ? "per tensor" : "per channel with " + channels + " channels";
    }

    /**
     * The quantization parameters of the observed parameter at {@code path}, per channel when it was calibrated per
     * channel. A parameter or channel without non-NaN values (an empty observed range, {@code min > max}, or an empty
     * histogram) gets {@link Quantizer#emptyRangeParams} with every method.
     *
     * @throws IllegalStateException    when the parameter has not been observed
     * @throws IllegalArgumentException when an FP8 clip threshold is infinite
     */
    QuantizationParams params(String path) {
        int channels = observedChannels(path);
        if (channels == UNOBSERVED) {
            throw new IllegalStateException("Parameter '" + path + "' has not been calibrated");
        }
        if (channels == PER_TENSOR) {
            return method == CalibrationMethod.MIN_MAX
                ? rangeParams(site(path), observedRanges.get(path))
                : histogramParams(site(path), histograms.get(path));
        }
        float[] scales = new float[channels];
        int[] zeroPoints = new int[channels];
        for (int c = 0; c < channels; c++) {
            QuantizationParams channel = method == CalibrationMethod.MIN_MAX
                ? rangeParams(channelSite(path, c), new float[]{
                    observedChannelRanges.get(path)[0][c], observedChannelRanges.get(path)[1][c]})
                : histogramParams(channelSite(path, c), channelHistograms.get(path)[c]);
            scales[c] = channel.scale();
            zeroPoints[c] = channel.zeroPoint();
        }
        return Quantizer.observedParams(quantType, scales, zeroPoints, true);
    }

    /** The MIN_MAX parameters of the observed range {@code {min, max}} of {@code site}. */
    private QuantizationParams rangeParams(String site, float[] observed) {
        float min = observed[0];
        float max = observed[1];
        if (min > max) {
            return Quantizer.emptyRangeParams(quantType);
        }
        if (Quantizer.isFp8(quantType)) {
            return fp8Params(site, Math.max(Math.abs(min), Math.abs(max)));
        }
        return Quantizer.integerParams(quantType, min, max, Quantizer.SCALE_EPS);
    }

    /** The parameters a histogram method derives from the histogram of {@code site}. */
    private QuantizationParams histogramParams(String site, CalibrationHistogram histogram) {
        if (histogram.total() == 0) {
            return Quantizer.emptyRangeParams(quantType);
        }
        if (Quantizer.isAsymmetric(quantType)) {
            float[] range = asymmetricClipRange(histogram);
            return Quantizer.asymmetricParams(range[0], range[1], Quantizer.codeMin(quantType),
                Quantizer.codeMax(quantType), Quantizer.SCALE_EPS, quantType);
        }
        float threshold = symmetricThreshold(histogram);
        if (Quantizer.isFp8(quantType)) {
            return fp8Params(site, threshold);
        }
        return symmetricParams(threshold);
    }

    /** The name of the parameter at {@code path} in messages. */
    private static String site(String path) {
        return "'" + path + "'";
    }

    /** The name of channel {@code channel} of the parameter at {@code path} in messages. */
    private static String channelSite(String path, int channel) {
        return "'" + path + "' (channel " + channel + ")";
    }

    private QuantizationParams fp8Params(String site, float amax) {
        try {
            return Quantizer.fp8Params(quantType, amax);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Cannot quantize parameter " + site + " to " + quantType + ": "
                + e.getMessage(), e);
        }
    }

    private QuantizationParams symmetricParams(float threshold) {
        return Quantizer.symmetricParams(-threshold, threshold, Quantizer.codeMin(quantType),
            Quantizer.codeMax(quantType), Quantizer.SCALE_EPS, quantType);
    }

    /** The clip threshold of a symmetric type from its magnitude histogram. */
    private float symmetricThreshold(CalibrationHistogram histogram) {
        switch (method) {
            case PERCENTILE:
                return HistogramCalibration.percentile(histogram, percentile / 100.0);
            case MSE:
                return HistogramCalibration.symmetricMse(histogram, symmetricQuantizer());
            case ENTROPY:
                return HistogramCalibration.symmetricEntropy(histogram, 1 << (Quantizer.codeBits(quantType) - 1));
            default:
                throw new IllegalStateException("No histogram calibration for " + method);
        }
    }

    /** The clip range {@code {lo, hi}} of an asymmetric type from its signed histogram. */
    private float[] asymmetricClipRange(CalibrationHistogram histogram) {
        switch (method) {
            case PERCENTILE:
                return new float[]{
                    HistogramCalibration.percentile(histogram, 1.0 - percentile / 100.0),
                    HistogramCalibration.percentile(histogram, percentile / 100.0)};
            case MSE:
                return HistogramCalibration.asymmetricMse(histogram, (lo, hi) -> {
                    QuantizationParams p = Quantizer.asymmetricParams(lo, hi, Quantizer.codeMin(quantType),
                        Quantizer.codeMax(quantType), Quantizer.SCALE_EPS, quantType);
                    return x -> Quantizer.integerValue(Quantizer.integerCode(x, p), p);
                });
            case ENTROPY:
                return HistogramCalibration.asymmetricEntropy(histogram, 1 << Quantizer.codeBits(quantType));
            default:
                throw new IllegalStateException("No histogram calibration for " + method);
        }
    }

    /** The quantize-dequantize map of a symmetric threshold, coding exactly as {@link Quantizer#quantizeTensor}. */
    private HistogramCalibration.SymmetricQuantizer symmetricQuantizer() {
        if (Quantizer.isFp8(quantType)) {
            Fp8Codec codec = Quantizer.codec(quantType);
            return threshold -> {
                float scale = Quantizer.fp8Params(quantType, threshold).scale();
                return x -> codec.decode(Quantizer.fp8Code(codec, x, scale)) * scale;
            };
        }
        return threshold -> {
            QuantizationParams p = symmetricParams(threshold);
            return x -> Quantizer.integerValue(Quantizer.integerCode(x, p), p);
        };
    }

    /** The histogram of the calibration values of {@code site}. */
    private CalibrationHistogram histogramOf(String site, float[] values) {
        try {
            return CalibrationHistogram.of(values, !Quantizer.isAsymmetric(quantType));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Cannot calibrate parameter " + site + " with " + method + ": "
                + e.getMessage(), e);
        }
    }

    /**
     * The calibration values of {@code tensor} viewed as {@code rows} rows of {@code rowLen = n / rows} row-major
     * elements, row after row with the same count per row, read back once without recording. When the tensor has
     * more than {@code maxSamples} elements, each row contributes only its elements at {@code floor(i * rowLen / k)}
     * for {@code i < k = max(1, maxSamples / rows)}, gathered on the tensor's backend before the readback.
     */
    private float[] calibrationValues(Tensor tensor, int rows) {
        long n = tensor.numElements();
        if (n == 0) {
            return new float[0];
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor values = tensor.dtype() == DType.FLOAT32 ? tensor : tensor.to(DType.FLOAT32);
            try {
                if (n <= maxSamples) {
                    return values.toFloatArray();
                }
                try (Tensor flat = values.reshape(Shape.of(Math.toIntExact(n)));
                     Tensor index = Tensor.fromArray(flat.backend(), rowSampleIndices(n, rows, maxSamples));
                     Tensor sampled = TensorOps.gather(flat, 0, index)) {
                    return sampled.toFloatArray();
                }
            } finally {
                if (values != tensor) {
                    values.close();
                }
            }
        }
    }

    /**
     * The flat indices of a uniform-stride sample of {@code k = max(1, maxSamples / rows)} elements from each of the
     * {@code rows} rows of {@code n / rows} elements of an {@code n}-element tensor with {@code n > maxSamples}: row
     * {@code c} contributes {@code c * rowLen + floor(i * rowLen / k)} for {@code i < k}.
     */
    static int[] rowSampleIndices(long n, int rows, int maxSamples) {
        long rowLen = n / rows;
        int perRow = Math.max(1, maxSamples / rows);
        int[] rowIndices = sampleIndices(rowLen, perRow);
        int[] indices = new int[Math.multiplyExact(rows, perRow)];
        for (int c = 0; c < rows; c++) {
            int offset = Math.toIntExact(c * rowLen);
            for (int i = 0; i < perRow; i++) {
                indices[c * perRow + i] = offset + rowIndices[i];
            }
        }
        return indices;
    }

    /**
     * The {@code count} indices {@code floor(i * n / count)}: a deterministic uniform-stride sample of
     * {@code n >= count} elements.
     */
    static int[] sampleIndices(long n, int count) {
        int[] indices = new int[count];
        for (int i = 0; i < count; i++) {
            indices[i] = Math.toIntExact(i * n / count);
        }
        return indices;
    }
}

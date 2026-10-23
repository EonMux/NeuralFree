package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.util.*;

/**
 * Post-training quantization of a model's parameters to integer or FP8 codes.
 *
 * <p>Parameters are identified by their dotted path in the model ({@link Module#namedParameters()}, such as
 * {@code "0.weight"}): calibration state, {@link #getQuantParams} and the {@link #quantizeModel} result are keyed by
 * it, so every parameter of a multi-layer model has its own clip range and codes.</p>
 *
 * <p><b>Codes.</b> The integer types use signed linear codes ({@code [-128, 127]} for INT8, {@code [-8, 7]} for INT4)
 * {@code clamp(Math.round(x / scale) + zeroPoint, quantMin, quantMax)}, a quotient beyond the int range saturating
 * to {@code quantMin} or {@code quantMax}, reconstructing {@code (code - zeroPoint) * scale}: symmetric types have
 * zero point 0 and
 * {@code scale = t / codeMax} for the clip threshold {@code t}; asymmetric types use PyTorch's affine parameters of the
 * clip range ({@link #asymmetricParams}). The FP8 types store the OCP 8-bit float bit pattern ({@code 0..255}) of
 * {@code x / scale} with {@code scale = amax / fmax} (448 for E4M3, 57344 for E5M2; scale 1 when {@code amax} is 0),
 * rounded to nearest-even and saturated to {@code +-fmax} ({@link Fp8Codec}); they reconstruct
 * {@code decode(code) * scale}, and their parameters report zero point 0 and the code range {@code [0, 255]}.</p>
 *
 * <p><b>Calibration.</b> {@link #calibrate} accumulates, per parameter, what the {@link CalibrationMethod} needs, and
 * repeated calls merge it, so the result depends only on the values observed, not on the call order:</p>
 * <ul>
 *   <li>{@link CalibrationMethod#MIN_MAX} keeps the running range {@code [min, max]} of the non-NaN values, observed on
 *       the parameter's backend without recording autograd operations: two device reductions whose two-float result
 *       is read back ({@link #observedRange}).</li>
 *   <li>{@link CalibrationMethod#PERCENTILE}, {@link CalibrationMethod#MSE} and {@link CalibrationMethod#ENTROPY}
 *       keep a {@link CalibrationHistogram} of up to 2048 bins (of {@code |x|} for the symmetric types, of {@code x}
 *       for the asymmetric ones). Post-training calibration is an offline host step: each call reads the parameter's
 *       values back once (or a deterministic uniform sample of at most {@code numCalibrationSamples} of them, gathered
 *       on the device) and bins the non-NaN ones on the host; an infinite value is rejected. PERCENTILE clips the
 *       symmetric types at the given percentile {@code p} of {@code |x|} and the asymmetric types at the
 *       {@code (100 - p, p)} percentiles of {@code x}; MSE minimizes the expected squared quantization error;
 *       ENTROPY is TensorRT's KL-divergence calibration ({@link HistogramCalibration}). The FP8 types support
 *       MIN_MAX, PERCENTILE and MSE (with the FP8 codec in the error); ENTROPY is defined over the uniform levels of
 *       integer codes and is rejected for them by {@link Builder#build()}.</li>
 * </ul>
 * <p>With every method, a parameter (or channel) without non-NaN values gets PyTorch's empty-range parameters, scale 1
 * and zero point 0, as a QAT observer that never saw a value exports them ({@link #emptyRangeParams}). With
 * {@link Builder#perChannel} every slice of dimension 0 of a parameter of rank at least 1 is calibrated on its own, as
 * a parameter holding only that slice's values would be, into {@link QuantizationParams#perChannel} parameters.</p>
 *
 * <p><b>Per-channel parameters.</b> {@link QuantizationParams#perChannel} holds one scale and zero point per slice of
 * dimension 0, as PyTorch's {@code per_channel_affine} and {@code per_channel_symmetric} schemes with axis 0: element
 * {@code i} of a row-major tensor of {@code n} elements and {@code dim(0) = C} channels uses channel
 * {@code i / (n / C)}, for integer and FP8 codes alike.</p>
 *
 * <p><b>Observed parameters.</b> The quantizer {@link QATModule#toQuantizer()} exports carries the scales and zero
 * points the QAT observers hold, as PyTorch's {@code convert} takes each {@code FakeQuantize}'s parameters: its
 * {@link #quantizeModel} codes the fake-quantized weights with exactly those parameters (per channel where they were
 * observed per channel), leaves every other parameter (biases, normalization parameters) in full precision and
 * calibrates nothing; {@link #getActivationQuantParams} gives the activation parameters.</p>
 *
 * <p>{@link #quantizeTensor} returns the codes as a Java array, so it reads the values to the host and records the
 * source backend in the {@link QuantizedTensor}; {@link #dequantizeTensor} builds its result on that backend.</p>
 */
public final class Quantizer {
    public enum QuantizationType {
        INT8_SYMMETRIC,
        INT8_ASYMMETRIC,
        INT4_SYMMETRIC,
        INT4_ASYMMETRIC,
        FP8_E4M3,
        FP8_E5M2
    }

    public enum CalibrationMethod {
        MIN_MAX,
        PERCENTILE,
        ENTROPY,
        MSE
    }

    /** The smallest integer scale, so an all-zero range still divides. */
    static final float SCALE_EPS = 1e-10f;
    /** The code range FP8 parameters report: the bit patterns. */
    private static final int FP8_CODE_MIN = 0;
    private static final int FP8_CODE_MAX = 0xFF;

    private final QuantizationType quantType;
    private final CalibrationMethod calibrationMethod;
    private final float percentile;
    private final int numCalibrationSamples;
    private final boolean perChannel;
    private final Map<String, QuantizationParams> paramQuantParams;
    /** The calibration state; null for a quantizer of observed parameters. */
    private final PtqCalibrator calibrator;
    /** The observed weight parameters by parameter path; null for a calibrating quantizer. */
    private final Map<String, QuantizationParams> observedWeightParams;
    /** The observed activation parameters by site name; empty for a calibrating quantizer. */
    private final Map<String, QuantizationParams> activationParams;
    private boolean calibrated;

    private Quantizer(Builder builder) {
        this(builder, null, Collections.emptyMap());
    }

    private Quantizer(Builder builder, Map<String, QuantizationParams> observedWeightParams,
                      Map<String, QuantizationParams> activationParams) {
        this.quantType = builder.quantType;
        this.calibrationMethod = builder.calibrationMethod;
        this.percentile = builder.percentile;
        this.numCalibrationSamples = builder.numCalibrationSamples;
        this.perChannel = builder.perChannel;
        this.paramQuantParams = new HashMap<>();
        this.observedWeightParams = observedWeightParams;
        this.activationParams = activationParams;
        if (observedWeightParams == null) {
            this.calibrator = new PtqCalibrator(quantType, calibrationMethod, percentile, numCalibrationSamples,
                perChannel);
            this.calibrated = false;
        } else {
            this.calibrator = null;
            this.paramQuantParams.putAll(observedWeightParams);
            this.calibrated = true;
        }
    }

    /**
     * A quantizer of {@code type} that codes with observed parameters instead of calibrating (class description):
     * {@code weightParams} by parameter path, {@code activationParams} by activation site name. Its calibration
     * method is MIN_MAX, the method of the observers.
     *
     * @throws IllegalArgumentException when a parameter set is of another type
     */
    static Quantizer fromObserved(QuantizationType type, Map<String, QuantizationParams> weightParams,
                                  Map<String, QuantizationParams> activationParams) {
        Builder builder = builder().quantType(type).calibrationMethod(CalibrationMethod.MIN_MAX);
        return new Quantizer(builder, observedCopy(type, weightParams, "Weight"),
            observedCopy(type, activationParams, "Activation"));
    }

    /** An unmodifiable copy of {@code params} in its iteration order, every entry checked to be of {@code type}. */
    private static Map<String, QuantizationParams> observedCopy(QuantizationType type,
                                                                Map<String, QuantizationParams> params, String kind) {
        Objects.requireNonNull(params, kind + " parameters cannot be null");
        Map<String, QuantizationParams> copy = new LinkedHashMap<>();
        for (Map.Entry<String, QuantizationParams> entry : params.entrySet()) {
            String site = Objects.requireNonNull(entry.getKey(), kind + " site name cannot be null");
            QuantizationParams p = Objects.requireNonNull(entry.getValue(),
                kind + " site '" + site + "' has no parameters");
            if (p.quantType() != type) {
                throw new IllegalArgumentException(kind + " site '" + site + "' has " + p.quantType()
                    + " parameters, not " + type);
            }
            copy.put(site, p);
        }
        return Collections.unmodifiableMap(copy);
    }

    /**
     * The parameters of observed scales and zero points, one entry per channel: per channel along dimension 0 when
     * {@code perChannel}, else per tensor from the single entry; the code range is the integer type's, or the FP8
     * bit patterns {@code [0, 255]}.
     *
     * @throws IllegalArgumentException when a per-tensor site has more than one entry, or as the
     *                                  {@link QuantizationParams} factories
     */
    static QuantizationParams observedParams(QuantizationType type, float[] scales, int[] zeroPoints,
                                             boolean perChannel) {
        Objects.requireNonNull(type, "Quantization type cannot be null");
        int qMin = isFp8(type) ? FP8_CODE_MIN : codeMin(type);
        int qMax = isFp8(type) ? FP8_CODE_MAX : codeMax(type);
        if (perChannel) {
            return QuantizationParams.perChannel(scales, zeroPoints, qMin, qMax, type);
        }
        if (scales.length != 1 || zeroPoints.length != 1) {
            throw new IllegalArgumentException("Per-tensor parameters need one scale and one zero point, got "
                + scales.length + " and " + zeroPoints.length);
        }
        return new QuantizationParams(scales[0], zeroPoints[0], qMin, qMax, type);
    }

    /**
     * The per-tensor parameters of {@code type} for a range without values: scale 1 and zero point 0, the result of
     * PyTorch's {@code MinMaxObserver} for an empty range, which post-training calibration and the QAT observers
     * share.
     */
    static QuantizationParams emptyRangeParams(QuantizationType type) {
        return observedParams(type, new float[]{1.0f}, new int[]{0}, false);
    }

    /**
     * Merges an observation of every parameter of {@code model}, keyed by its dotted path, into the calibration
     * state (see the class description). Every parameter is observed before any state changes, so a call that throws
     * leaves the state as it was.
     *
     * @throws IllegalArgumentException when a histogram method observes an infinite value, or a parameter calibrated
     *                                  per channel is observed with another dimension 0 or per tensor (or the
     *                                  reverse)
     * @throws IllegalStateException    for a quantizer of observed parameters, which does not calibrate
     */
    public void calibrate(Module model) {
        Objects.requireNonNull(model, "Model cannot be null");
        if (calibrator == null) {
            throw new IllegalStateException("This quantizer codes with the parameters its QAT observers hold "
                + "and does not calibrate");
        }
        calibrator.observe(model.namedParameters());
        calibrated = true;
    }

    /**
     * The codes of every parameter of {@code model} by dotted path ({@link Module#namedParameters()}, in its order),
     * each with the parameters derived from its own calibration state (per channel with {@link Builder#perChannel}).
     * A parameter without calibration state (every parameter, when this quantizer has not been calibrated) is
     * calibrated from its current values first.
     *
     * <p>A quantizer of observed parameters codes only the parameters of {@code model} at the observed weight paths,
     * with the observed parameters, and leaves the others out of the result (they stay in full precision); pass the
     * QAT module's base module, whose parameter paths the weight sites are keyed by.</p>
     *
     * @throws IllegalArgumentException when an FP8 clip range is infinite, a parameter's per-channel parameters do
     *                                  not have one channel per entry of its dimension 0, or as {@link #calibrate};
     *                                  for a quantizer of observed parameters, when none of its weight sites is a
     *                                  parameter of {@code model}
     */
    public Map<String, QuantizedTensor> quantizeModel(Module model) {
        Objects.requireNonNull(model, "Model cannot be null");
        if (observedWeightParams != null) {
            return quantizeObservedWeights(model);
        }
        Map<String, Parameter> parameters = model.namedParameters();
        Map<String, Parameter> uncalibrated = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            if (!calibrator.isObserved(entry.getKey())) {
                uncalibrated.put(entry.getKey(), entry.getValue());
            }
        }
        calibrator.observe(uncalibrated);
        calibrated = true;

        Map<String, QuantizedTensor> quantizedParams = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            String name = entry.getKey();
            QuantizationParams qParams = calibrator.params(name);
            quantizedParams.put(name, quantizeParameter(name, entry.getValue(), qParams));
            paramQuantParams.put(name, qParams);
        }
        return quantizedParams;
    }

    /** {@link #quantizeModel} of a quantizer of observed parameters. */
    private Map<String, QuantizedTensor> quantizeObservedWeights(Module model) {
        Map<String, QuantizedTensor> codes = new LinkedHashMap<>();
        for (Map.Entry<String, Parameter> entry : model.namedParameters().entrySet()) {
            String path = entry.getKey();
            QuantizationParams params = observedWeightParams.get(path);
            if (params == null) {
                continue;
            }
            codes.put(path, quantizeParameter(path, entry.getValue(), params));
        }
        if (codes.isEmpty() && !observedWeightParams.isEmpty()) {
            throw new IllegalArgumentException("None of the observed weight sites " + observedWeightParams.keySet()
                + " is a parameter of the model; quantize the QAT module's base module");
        }
        return codes;
    }

    /** {@link #quantizeTensor} of the parameter at {@code path}, whose failure names the parameter. */
    private QuantizedTensor quantizeParameter(String path, Parameter parameter, QuantizationParams params) {
        try {
            return quantizeTensor(parameter.tensor(), params);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Cannot quantize parameter '" + path + "': " + e.getMessage(), e);
        }
    }

    /**
     * The codes of a tensor in row-major order, computed on the host because the result is a Java array; the result
     * remembers {@code tensor}'s backend. Integer types: {@code clamp(Math.round(x / scale) + zeroPoint, quantMin,
     * quantMax)} ({@link #integerCode}, saturating). FP8 types: the bit pattern of {@code x / scale} (a finite
     * {@code x} whose quotient overflows saturates like any out-of-range finite value). Per-channel parameters code
     * element {@code i} with channel {@code i / (n / dim(0))}.
     *
     * @throws IllegalArgumentException when per-channel parameters do not have one channel per entry of
     *                                  {@code tensor}'s dimension 0
     */
    public QuantizedTensor quantizeTensor(Tensor tensor, QuantizationParams params) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        Objects.requireNonNull(params, "QuantizationParams cannot be null");
        int channelSize = channelSize(tensor.shape(), params);

        float[] data = tensor.toFloatArray();
        int[] quantizedData = new int[data.length];
        if (isFp8(params.quantType)) {
            Fp8Codec codec = codec(params.quantType);
            for (int i = 0; i < data.length; i++) {
                quantizedData[i] = fp8Code(codec, data[i], params.scales[i / channelSize]);
            }
        } else {
            for (int i = 0; i < data.length; i++) {
                int c = i / channelSize;
                quantizedData[i] = integerCode(data[i], params.scales[c], params.zeroPoints[c], params.quantMin,
                    params.quantMax);
            }
        }
        return new QuantizedTensor(quantizedData, tensor.shape(), params, tensor.backend());
    }

    /**
     * The number of consecutive row-major elements of a tensor of {@code shape} that share one channel of
     * {@code params}: every element ({@code Integer.MAX_VALUE}) for per-tensor parameters, {@code n / dim(0)} (at
     * least 1) for per-channel ones.
     *
     * @throws IllegalArgumentException when per-channel parameters do not have one channel per entry of dimension 0
     */
    private static int channelSize(Shape shape, QuantizationParams params) {
        if (!params.perChannel) {
            return Integer.MAX_VALUE;
        }
        if (shape.rank() == 0 || shape.dim(0) != params.scales.length) {
            throw new IllegalArgumentException("Per-channel parameters of " + params.scales.length
                + " channels need a tensor whose dimension 0 has that size, got shape " + shape);
        }
        return Math.max(1, Math.toIntExact(shape.numElements() / shape.dim(0)));
    }

    /**
     * The values the codes reconstruct, as a new FLOAT32 tensor of the quantized shape on the backend the codes were
     * quantized from: {@code (code - zeroPoint) * scale} for the integer types, {@code decode(code) * scale} for FP8.
     * The codes are host data, so the values are computed on the host and uploaded once.
     *
     * @throws IllegalStateException when {@code quantized} records no backend; use
     *                               {@link #dequantizeTensor(QuantizedTensor, ComputeBackend)}
     */
    public Tensor dequantizeTensor(QuantizedTensor quantized) {
        Objects.requireNonNull(quantized, "QuantizedTensor cannot be null");
        if (quantized.backend() == null) {
            throw new IllegalStateException("QuantizedTensor records no backend; pass the target backend explicitly");
        }
        return dequantizeTensor(quantized, quantized.backend());
    }

    /**
     * {@link #dequantizeTensor(QuantizedTensor)} with the result on {@code backend}. Per-channel codes reconstruct
     * element {@code i} with channel {@code i / (n / dim(0))}.
     */
    public Tensor dequantizeTensor(QuantizedTensor quantized, ComputeBackend backend) {
        Objects.requireNonNull(quantized, "QuantizedTensor cannot be null");
        Objects.requireNonNull(backend, "Backend cannot be null");

        int[] quantizedData = quantized.data;
        QuantizationParams params = quantized.params;
        int channelSize = channelSize(quantized.shape, params);
        float[] dequantizedData = new float[quantizedData.length];
        if (isFp8(params.quantType)) {
            Fp8Codec codec = codec(params.quantType);
            for (int i = 0; i < quantizedData.length; i++) {
                dequantizedData[i] = codec.decode(quantizedData[i]) * params.scales[i / channelSize];
            }
        } else {
            for (int i = 0; i < quantizedData.length; i++) {
                int c = i / channelSize;
                dequantizedData[i] = integerValue(quantizedData[i], params.scales[c], params.zeroPoints[c]);
            }
        }
        TensorStorage storage = TensorStorage.fromHostFloat(backend, dequantizedData);
        try {
            return Tensor.fromStorage(backend, quantized.shape, DType.FLOAT32, storage);
        } catch (RuntimeException e) {
            try {
                storage.close();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /**
     * {@code {min, max}} of the non-NaN values of {@code tensor}, with the bounds of the observer loop
     * {@code min = Float.MAX_VALUE; max = -Float.MAX_VALUE; if (v < min) min = v; if (v > max) max = v;}: NaN values
     * are skipped, the minimum is at most {@code Float.MAX_VALUE} and the maximum at least {@code -Float.MAX_VALUE}
     * (also for an empty tensor). Computed with device reductions over NaN-replaced copies; only the two floats are
     * read back. Where the loop keeps the first of {@code -0.0f} and {@code 0.0f}, the reductions
     * give {@code -0.0f} as the minimum and {@code 0.0f} as the maximum, which derives the same scale and zero point.
     * A non-FLOAT32 tensor is converted first ({@link Tensor#to(DType)}). An observation, not part of any loss: it
     * runs under {@link ComputationGraph#noGrad()} and records nothing, also inside a training forward pass.
     */
    static float[] observedRange(Tensor tensor) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        long n = tensor.numElements();
        if (n == 0) {
            return new float[]{Float.MAX_VALUE, -Float.MAX_VALUE};
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return observedRange(tensor, n);
        }
    }

    /** {@link #observedRange(Tensor)} of a non-empty tensor of {@code n} elements, without recording. */
    private static float[] observedRange(Tensor tensor, long n) {
        Tensor values = tensor.dtype() == DType.FLOAT32 ? tensor : tensor.to(DType.FLOAT32);
        try (Tensor flat = values.reshape(Shape.of(Math.toIntExact(n)));
             Tensor notNaN = TensorOps.eq(flat, flat);
             Tensor zeros = Tensor.zeros(flat.backend(), flat.shape());
             Tensor highFill = TensorOps.add(zeros, Float.MAX_VALUE);
             Tensor lowFill = TensorOps.add(zeros, -Float.MAX_VALUE);
             Tensor forMin = TensorOps.where(notNaN, flat, highFill);
             Tensor forMax = TensorOps.where(notNaN, flat, lowFill);
             Tensor min = TensorOps.min(forMin, 0);
             Tensor max = TensorOps.max(forMax, 0);
             Tensor both = TensorOps.stack(Arrays.asList(min, max), 0)) {
            float[] range = both.toFloatArray();
            return new float[]{Math.min(Float.MAX_VALUE, range[0]), Math.max(-Float.MAX_VALUE, range[1])};
        } finally {
            if (values != tensor) {
                values.close();
            }
        }
    }

    /**
     * The per-channel {@link #observedRange} of {@code tensor} along dimension 0: {@code {mins, maxs}}, each of
     * length {@code tensor.shape().dim(0)}, with the same NaN rule and bounds per channel (a channel without elements
     * gives {@code Float.MAX_VALUE} and {@code -Float.MAX_VALUE}). Computed with two row reductions over NaN-replaced
     * copies; the {@code 2 * channels} floats are read back at once. Runs under
     * {@link ComputationGraph#noGrad()} and records nothing.
     *
     * @throws IllegalArgumentException when {@code tensor} has rank 0
     */
    static float[][] observedChannelRanges(Tensor tensor) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        if (tensor.shape().rank() == 0) {
            throw new IllegalArgumentException("Per-channel observation needs a tensor of rank >= 1");
        }
        int channels = tensor.shape().dim(0);
        long n = tensor.numElements();
        float[] mins = new float[channels];
        float[] maxs = new float[channels];
        if (n == 0) {
            Arrays.fill(mins, Float.MAX_VALUE);
            Arrays.fill(maxs, -Float.MAX_VALUE);
            return new float[][]{mins, maxs};
        }
        int perChannel = Math.toIntExact(n / channels);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor values = tensor.dtype() == DType.FLOAT32 ? tensor : tensor.to(DType.FLOAT32);
            try (Tensor rows = values.reshape(Shape.of(channels, perChannel));
                 Tensor notNaN = TensorOps.eq(rows, rows);
                 Tensor zeros = Tensor.zeros(rows.backend(), rows.shape());
                 Tensor highFill = TensorOps.add(zeros, Float.MAX_VALUE);
                 Tensor lowFill = TensorOps.add(zeros, -Float.MAX_VALUE);
                 Tensor forMin = TensorOps.where(notNaN, rows, highFill);
                 Tensor forMax = TensorOps.where(notNaN, rows, lowFill);
                 Tensor min = TensorOps.min(forMin, 1);
                 Tensor max = TensorOps.max(forMax, 1);
                 Tensor both = TensorOps.stack(Arrays.asList(min, max), 0)) {
                float[] read = both.toFloatArray();
                for (int c = 0; c < channels; c++) {
                    mins[c] = Math.min(Float.MAX_VALUE, read[c]);
                    maxs[c] = Math.max(-Float.MAX_VALUE, read[channels + c]);
                }
                return new float[][]{mins, maxs};
            } finally {
                if (values != tensor) {
                    values.close();
                }
            }
        }
    }

    /**
     * A rank-0 FLOAT32 tensor holding {@code value} (not {@code -0.0f}) on {@code backend}: {@code 0 + value} over a
     * zero-filled scalar ({@link Tensor#zeros}, filled on the device when the train kernels are available, so no host
     * transfer is made). Never requires grad.
     */
    static Tensor scalarOn(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar(), DType.FLOAT32)) {
            return TensorOps.add(zero, value);
        }
    }

    /** {@link #integerCode(float, float, int, int, int)} with per-tensor {@code params}. */
    static int integerCode(float x, QuantizationParams params) {
        return integerCode(x, params.scale(), params.zeroPoint(), params.quantMin(), params.quantMax());
    }

    /**
     * The integer code of {@code x}: {@code clamp(Math.round(x / scale) + zeroPoint, quantMin, quantMax)}, the sum
     * taken without overflow, so a quotient beyond the int range (where {@code Math.round} saturates) clamps to
     * {@code quantMin} or {@code quantMax} by its sign, as the float composition of {@link FakeQuantGrad} does. NaN
     * gives the zero point.
     */
    static int integerCode(float x, float scale, int zeroPoint, int quantMin, int quantMax) {
        long code = (long) Math.round(x / scale) + zeroPoint;
        return (int) Math.max(quantMin, Math.min(quantMax, code));
    }

    /** {@link #integerValue(int, float, int)} with per-tensor {@code params}. */
    static float integerValue(int code, QuantizationParams params) {
        return integerValue(code, params.scale(), params.zeroPoint());
    }

    /** The value an integer code reconstructs: {@code (code - zeroPoint) * scale}. */
    static float integerValue(int code, float scale, int zeroPoint) {
        return (code - zeroPoint) * scale;
    }

    /**
     * The FP8 code of {@code x / scale}; a finite {@code x} whose quotient overflows to an infinity saturates to the
     * largest finite code of its sign.
     */
    static int fp8Code(Fp8Codec codec, float x, float scale) {
        float scaled = x / scale;
        if (Float.isInfinite(scaled) && !Float.isInfinite(x)) {
            scaled = Math.copySign(Float.MAX_VALUE, scaled);
        }
        return codec.encode(scaled);
    }

    /**
     * The FP8 parameters of the clip threshold {@code amax}: {@code scale = amax / fmax} (at least
     * {@code Float.MIN_VALUE}), or 1 when {@code amax} is 0; zero point 0 and the code range {@code [0, 255]}.
     *
     * @throws IllegalArgumentException when {@code type} is not an FP8 type or {@code amax} is negative, NaN or
     *                                  infinite
     */
    static QuantizationParams fp8Params(QuantizationType type, float amax) {
        codec(type);
        if (!(amax >= 0.0f) || Float.isInfinite(amax)) {
            throw new IllegalArgumentException("An FP8 clip threshold must be finite and non-negative, got " + amax);
        }
        return new QuantizationParams(fp8Scale(type, amax), 0, FP8_CODE_MIN, FP8_CODE_MAX, type);
    }

    /**
     * The FP8 scale of the threshold {@code amax >= 0}: {@code max(amax / fmax, Float.MIN_VALUE)}, or 1 when
     * {@code amax} is 0. An infinite {@code amax} gives an infinite scale, whose fake quantization is NaN, as an
     * infinite integer scale's is; {@link #fp8Params} rejects it for post-training codes.
     *
     * @throws IllegalArgumentException when {@code type} is not an FP8 type
     */
    static float fp8Scale(QuantizationType type, float amax) {
        Fp8Codec codec = codec(type);
        return amax == 0.0f ? 1.0f : Math.max(amax / codec.maxFinite(), Float.MIN_VALUE);
    }

    /**
     * The codec of an FP8 type: {@link Fp8Codec#E4M3FN} for {@code FP8_E4M3}, {@link Fp8Codec#E5M2} for
     * {@code FP8_E5M2}.
     *
     * @throws IllegalArgumentException for the integer types
     */
    static Fp8Codec codec(QuantizationType type) {
        if (type == QuantizationType.FP8_E4M3) {
            return Fp8Codec.E4M3FN;
        }
        if (type == QuantizationType.FP8_E5M2) {
            return Fp8Codec.E5M2;
        }
        throw new IllegalArgumentException(type + " is not an FP8 type");
    }

    /** The smallest integer code of {@code type}: -8 for the INT4 types, -128 otherwise. */
    static int codeMin(QuantizationType type) {
        return isInt4(type) ? -8 : -128;
    }

    /** The largest integer code of {@code type}: 7 for the INT4 types, 127 otherwise. */
    static int codeMax(QuantizationType type) {
        return isInt4(type) ? 7 : 127;
    }

    /** The code width of an integer type: 4 for the INT4 types, 8 otherwise. */
    static int codeBits(QuantizationType type) {
        return isInt4(type) ? 4 : 8;
    }

    private static boolean isInt4(QuantizationType type) {
        return type == QuantizationType.INT4_SYMMETRIC || type == QuantizationType.INT4_ASYMMETRIC;
    }

    /** True for the types with affine (zero-point) codes. */
    static boolean isAsymmetric(QuantizationType type) {
        return type == QuantizationType.INT8_ASYMMETRIC || type == QuantizationType.INT4_ASYMMETRIC;
    }

    /** True for the 8-bit floating-point types. */
    static boolean isFp8(QuantizationType type) {
        return type == QuantizationType.FP8_E4M3 || type == QuantizationType.FP8_E5M2;
    }

    /**
     * The scale and zero point of the observed range {@code [min, max]} for the integer codes of {@code type}
     * ({@link #codeMin}, {@link #codeMax}): {@link #asymmetricParams} for the asymmetric types,
     * {@link #symmetricParams} otherwise.
     *
     * @throws IllegalArgumentException for the FP8 types, whose parameters are {@link #fp8Params}
     */
    static QuantizationParams integerParams(QuantizationType type, float min, float max, float eps) {
        if (isFp8(type)) {
            throw new IllegalArgumentException(type + " has FP8 codes, not integer codes");
        }
        int qMin = codeMin(type);
        int qMax = codeMax(type);
        if (isAsymmetric(type)) {
            return asymmetricParams(min, max, qMin, qMax, eps, type);
        }
        return symmetricParams(min, max, qMin, qMax, eps, type);
    }

    /** Zero point 0 and {@code scale = max(max(|min|, |max|) / qMax, eps)}. */
    static QuantizationParams symmetricParams(float min, float max, int qMin, int qMax, float eps,
                                              QuantizationType type) {
        float absMax = Math.max(Math.abs(min), Math.abs(max));
        float scale = Math.max(absMax / Math.max(1, qMax), eps);
        return new QuantizationParams(scale, 0, qMin, qMax, type);
    }

    /**
     * PyTorch's affine observer parameters ({@code MinMaxObserver._calculate_qparams}): the range is first widened to
     * contain 0 ({@code minNeg = min(min, 0)}, {@code maxPos = max(max, 0)}), so 0 is exactly representable and a
     * range of one sign keeps its full extent; then {@code scale = max((maxPos - minNeg) / (qMax - qMin), eps)} and
     * {@code zeroPoint = clamp(qMin - round(minNeg / scale), qMin, qMax)}.
     */
    static QuantizationParams asymmetricParams(float min, float max, int qMin, int qMax, float eps,
                                               QuantizationType type) {
        float minNeg = Math.min(min, 0.0f);
        float maxPos = Math.max(max, 0.0f);
        float scale = Math.max((maxPos - minNeg) / (qMax - qMin), eps);
        int zeroPoint = qMin - Math.round(minNeg / scale);
        zeroPoint = Math.max(qMin, Math.min(qMax, zeroPoint));
        return new QuantizationParams(scale, zeroPoint, qMin, qMax, type);
    }

    /**
     * The parameters {@link #quantizeModel} used for the parameter at dotted path {@code paramName} (such as
     * {@code "0.weight"}), or null when no parameter of that path has been quantized. A quantizer of observed
     * parameters returns the observed parameters of every weight site from the start.
     */
    public QuantizationParams getQuantParams(String paramName) {
        return paramQuantParams.get(paramName);
    }

    /**
     * The observed per-tensor parameters of the activation site {@code name} of a quantizer exported by
     * {@link QATModule#toQuantizer()} ({@code "input"} and {@code "output"} for the base module's input and output),
     * or null when there is no such site (always for a calibrating quantizer).
     */
    public QuantizationParams getActivationQuantParams(String name) {
        return activationParams.get(name);
    }

    /** The observed activation parameters by site name, in site creation order; empty for a calibrating quantizer. */
    public Map<String, QuantizationParams> activationQuantParams() {
        return activationParams;
    }

    public QuantizationType quantType() {
        return quantType;
    }

    public CalibrationMethod calibrationMethod() {
        return calibrationMethod;
    }

    /** The percentile PERCENTILE calibration clips at, in {@code (0, 100]}. */
    public float percentile() {
        return percentile;
    }

    /** The most values of one parameter a histogram calibration reads back per {@link #calibrate} call. */
    public int numCalibrationSamples() {
        return numCalibrationSamples;
    }

    /**
     * True when calibration derives per-channel parameters along dimension 0 ({@link Builder#perChannel}). A
     * quantizer of observed parameters does not calibrate and reports false; its parameters are per channel where the
     * QAT observers were.
     */
    public boolean perChannel() {
        return perChannel;
    }

    public boolean isCalibrated() {
        return calibrated;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Settings of a {@link Quantizer}: INT8_SYMMETRIC with MIN_MAX calibration, percentile 99.9, every value of a
     * parameter histogrammed ({@code numCalibrationSamples = Integer.MAX_VALUE}) and per-tensor parameters unless set.
     */
    public static final class Builder {
        private QuantizationType quantType = QuantizationType.INT8_SYMMETRIC;
        private CalibrationMethod calibrationMethod = CalibrationMethod.MIN_MAX;
        private float percentile = 99.9f;
        private int numCalibrationSamples = Integer.MAX_VALUE;
        private boolean perChannel = false;

        public Builder quantType(QuantizationType quantType) {
            this.quantType = Objects.requireNonNull(quantType, "Quantization type cannot be null");
            return this;
        }

        public Builder calibrationMethod(CalibrationMethod calibrationMethod) {
            this.calibrationMethod = Objects.requireNonNull(calibrationMethod, "Calibration method cannot be null");
            return this;
        }

        public Builder percentile(float percentile) {
            if (!(percentile > 0 && percentile <= 100)) {
                throw new IllegalArgumentException("Percentile must be in (0, 100], got: " + percentile);
            }
            this.percentile = percentile;
            return this;
        }

        /**
         * The most values of one parameter that a histogram calibration (PERCENTILE, MSE, ENTROPY) reads back per
         * {@link Quantizer#calibrate} call: a larger parameter contributes a deterministic uniform-stride sample of
         * this many values, which bounds the readback and host work for huge tensors. A parameter calibrated per
         * channel ({@link #perChannel}) contributes such a sample of {@code max(1, numCalibrationSamples / C)} values
         * from each of its {@code C} channels. MIN_MAX reduces every value on the device and is not affected.
         */
        public Builder numCalibrationSamples(int numCalibrationSamples) {
            if (numCalibrationSamples < 1) {
                throw new IllegalArgumentException("Num calibration samples must be at least 1, got: "
                    + numCalibrationSamples);
            }
            this.numCalibrationSamples = numCalibrationSamples;
            return this;
        }

        /**
         * Whether calibration derives per-channel parameters along dimension 0, as PyTorch's
         * {@code per_channel_affine} and {@code per_channel_symmetric} schemes with axis 0: each slice of dimension 0
         * of a parameter of rank at least 1 is calibrated with the chosen method on its own values only, and the
         * parameter gets {@link QuantizationParams#perChannel} parameters. Rank-0 parameters and parameters whose
         * dimension 0 is empty stay per tensor.
         */
        public Builder perChannel(boolean perChannel) {
            this.perChannel = perChannel;
            return this;
        }

        /**
         * The quantizer of these settings.
         *
         * @throws IllegalArgumentException for ENTROPY with an FP8 type, or PERCENTILE below 50 with an asymmetric
         *                                  type (whose lower clip point would lie above the upper one)
         */
        public Quantizer build() {
            if (isFp8(quantType) && calibrationMethod == CalibrationMethod.ENTROPY) {
                throw new IllegalArgumentException("ENTROPY calibration is defined over the uniform levels of the "
                    + "integer types; calibrate " + quantType + " with MIN_MAX, PERCENTILE or MSE");
            }
            if (calibrationMethod == CalibrationMethod.PERCENTILE && isAsymmetric(quantType) && percentile < 50) {
                throw new IllegalArgumentException("PERCENTILE calibration of " + quantType
                    + " clips at the (100 - p, p) percentiles and needs p >= 50, got: " + percentile);
            }
            return new Quantizer(this);
        }
    }

    /**
     * A scale and zero point per tensor, or one per channel of dimension 0 ({@link #perChannel}), with the code range
     * and type of the codes they produce.
     *
     * <p>Valid parameters only: every scale is positive and not NaN (an infinite scale, the parameters of an infinite
     * observed range, reconstructs NaN, as its fake quantization does); the integer types have
     * {@code codeMin <= quantMin <= quantMax <= codeMax} of their signed codes ({@code [-128, 127]} for INT8,
     * {@code [-8, 7]} for INT4) and every zero point in {@code [quantMin, quantMax]}; the FP8 types have the bit
     * pattern range {@code [0, 255]} and zero point 0.</p>
     */
    public static final class QuantizationParams {
        private final float[] scales;
        private final int[] zeroPoints;
        private final boolean perChannel;
        private final int quantMin;
        private final int quantMax;
        private final QuantizationType quantType;

        /**
         * Per-tensor parameters.
         *
         * @throws IllegalArgumentException when the parameters are not valid (class description)
         */
        public QuantizationParams(float scale, int zeroPoint, int quantMin, int quantMax, QuantizationType quantType) {
            this(new float[]{scale}, new int[]{zeroPoint}, false, quantMin, quantMax, quantType);
        }

        private QuantizationParams(float[] scales, int[] zeroPoints, boolean perChannel, int quantMin, int quantMax,
                                   QuantizationType quantType) {
            this.quantType = Objects.requireNonNull(quantType, "Quantization type cannot be null");
            if (scales.length == 0 || scales.length != zeroPoints.length) {
                throw new IllegalArgumentException("Quantization parameters need one scale and one zero point per "
                    + "channel, got " + scales.length + " scales and " + zeroPoints.length + " zero points");
            }
            checkCodeRange(quantType, quantMin, quantMax);
            for (int c = 0; c < scales.length; c++) {
                checkChannel(quantType, scales[c], zeroPoints[c], quantMin, quantMax, c);
            }
            this.scales = scales;
            this.zeroPoints = zeroPoints;
            this.perChannel = perChannel;
            this.quantMin = quantMin;
            this.quantMax = quantMax;
        }

        /**
         * Per-channel parameters along dimension 0: channel {@code c} has {@code scales[c]} and
         * {@code zeroPoints[c]} (the arrays are copied).
         *
         * @throws IllegalArgumentException when the arrays are empty or of different lengths, or the parameters are
         *                                  not valid (class description)
         */
        public static QuantizationParams perChannel(float[] scales, int[] zeroPoints, int quantMin, int quantMax,
                                                    QuantizationType quantType) {
            Objects.requireNonNull(scales, "Scales cannot be null");
            Objects.requireNonNull(zeroPoints, "Zero points cannot be null");
            return new QuantizationParams(scales.clone(), zeroPoints.clone(), true, quantMin, quantMax, quantType);
        }

        private static void checkCodeRange(QuantizationType type, int quantMin, int quantMax) {
            if (isFp8(type)) {
                if (quantMin != FP8_CODE_MIN || quantMax != FP8_CODE_MAX) {
                    throw new IllegalArgumentException(type + " codes are the bit patterns [" + FP8_CODE_MIN + ", "
                        + FP8_CODE_MAX + "], got [" + quantMin + ", " + quantMax + "]");
                }
            } else if (quantMin > quantMax || quantMin < codeMin(type) || quantMax > codeMax(type)) {
                throw new IllegalArgumentException("The code range of " + type + " must lie within [" + codeMin(type)
                    + ", " + codeMax(type) + "] with quantMin <= quantMax, got [" + quantMin + ", " + quantMax + "]");
            }
        }

        private static void checkChannel(QuantizationType type, float scale, int zeroPoint, int quantMin,
                                         int quantMax, int channel) {
            if (!(scale > 0.0f)) {
                throw new IllegalArgumentException("Scale must be positive, got " + scale + " (channel " + channel
                    + ")");
            }
            boolean outOfRange = isFp8(type) ? zeroPoint != 0 : (zeroPoint < quantMin || zeroPoint > quantMax);
            if (outOfRange) {
                throw new IllegalArgumentException("Zero point " + zeroPoint + " (channel " + channel + ") of "
                    + type + (isFp8(type) ? " must be 0" : " must lie in [" + quantMin + ", " + quantMax + "]"));
            }
        }

        /** True for per-channel parameters along dimension 0. */
        public boolean isPerChannel() {
            return perChannel;
        }

        /** The number of channels: 1 for per-tensor parameters. */
        public int channels() {
            return scales.length;
        }

        /**
         * The per-tensor scale.
         *
         * @throws IllegalStateException for per-channel parameters; use {@link #scale(int)}
         */
        public float scale() {
            requirePerTensor();
            return scales[0];
        }

        /**
         * The per-tensor zero point.
         *
         * @throws IllegalStateException for per-channel parameters; use {@link #zeroPoint(int)}
         */
        public int zeroPoint() {
            requirePerTensor();
            return zeroPoints[0];
        }

        /** The scale of {@code channel} (channel 0 of per-tensor parameters). */
        public float scale(int channel) {
            return scales[channel];
        }

        /** The zero point of {@code channel} (channel 0 of per-tensor parameters). */
        public int zeroPoint(int channel) {
            return zeroPoints[channel];
        }

        /** A copy of the scales, one per channel. */
        public float[] scales() {
            return scales.clone();
        }

        /** A copy of the zero points, one per channel. */
        public int[] zeroPoints() {
            return zeroPoints.clone();
        }

        private void requirePerTensor() {
            if (perChannel) {
                throw new IllegalStateException("Per-channel parameters of " + scales.length
                    + " channels have no single scale and zero point");
            }
        }

        public int quantMin() {
            return quantMin;
        }

        public int quantMax() {
            return quantMax;
        }

        public QuantizationType quantType() {
            return quantType;
        }

        @Override
        public String toString() {
            String values = perChannel
                ? "scales=" + Arrays.toString(scales) + ", zeroPoints=" + Arrays.toString(zeroPoints)
                : "scale=" + scales[0] + ", zeroPoint=" + zeroPoints[0];
            return "QuantizationParams{" + values +
                ", quantMin=" + quantMin +
                ", quantMax=" + quantMax +
                ", quantType=" + quantType +
                '}';
        }
    }

    /**
     * Codes in row-major order with their shape, parameters and, when known, the source backend: integer codes, or
     * FP8 bit patterns in {@code [0, 255]} for the FP8 types ({@link #quantType()}).
     */
    public static final class QuantizedTensor {
        private final int[] data;
        private final Shape shape;
        private final QuantizationParams params;
        private final ComputeBackend backend;

        /** Codes with no recorded backend. */
        public QuantizedTensor(int[] data, Shape shape, QuantizationParams params) {
            this(data, shape, params, null);
        }

        /**
         * Codes quantized from a tensor on {@code backend} (null when unknown); {@code data} is copied.
         *
         * @throws IllegalArgumentException when {@code data} does not have one code per element of {@code shape}, a
         *                                  code lies outside {@code [quantMin, quantMax]}, or per-channel
         *                                  {@code params} do not have one channel per entry of dimension 0
         */
        public QuantizedTensor(int[] data, Shape shape, QuantizationParams params, ComputeBackend backend) {
            Objects.requireNonNull(data, "Codes cannot be null");
            this.shape = Objects.requireNonNull(shape, "Shape cannot be null");
            this.params = Objects.requireNonNull(params, "QuantizationParams cannot be null");
            this.data = data.clone();
            if (this.data.length != shape.numElements()) {
                throw new IllegalArgumentException(this.data.length + " codes for shape " + shape + " of "
                    + shape.numElements() + " elements");
            }
            channelSize(shape, params);
            for (int i = 0; i < this.data.length; i++) {
                int code = this.data[i];
                if (code < params.quantMin || code > params.quantMax) {
                    throw new IllegalArgumentException("Code " + code + " at index " + i + " lies outside ["
                        + params.quantMin + ", " + params.quantMax + "]");
                }
            }
            this.backend = backend;
        }

        /** The backend the codes were quantized from, or null when unknown. */
        public ComputeBackend backend() {
            return backend;
        }

        public int[] data() {
            return data.clone();
        }

        public Shape shape() {
            return shape;
        }

        public QuantizationParams params() {
            return params;
        }

        /** The type of the codes, as their parameters record it. */
        public QuantizationType quantType() {
            return params.quantType();
        }

        public int numElements() {
            return data.length;
        }

        @Override
        public String toString() {
            return "QuantizedTensor{" +
                "shape=" + shape +
                ", params=" + params +
                ", numElements=" + numElements() +
                '}';
        }
    }
}

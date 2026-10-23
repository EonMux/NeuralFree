package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.ModuleDict;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Quantization-aware training wrapper: fake-quantizes the weights, and the input and output activations, of a base
 * module with scales derived from running min/max observers, as PyTorch's QAT modules do.
 *
 * <p><b>Weights</b> ({@code quantizeWeights}). Each forward pass fake-quantizes every floating-point parameter of rank
 * 2 or more of the base module (linear, convolution and embedding weights; biases and normalization parameters stay
 * in full precision, as in PyTorch) and runs the base forward with them substituted ({@link Module#substitute}, which
 * nests inside an outer substitution such as MAML's), so the straight-through gradient reaches the weights. Their
 * observers are keyed by the parameter's path in the base module ({@code weight}, {@code 0.weight}, ...), the key
 * {@link #fakeQuantizeWeight} takes. A parameter reachable under several paths (a module registered twice, tied
 * weights) is one weight site, under its first path ({@link Module#namedParameters()} order): every one of its paths
 * names that site's observer. With {@code perChannelQuantization} each output channel (dimension 0) has its
 * own running range, scale and zero point ({@link FakeQuantGrad#applyPerChannel}); activations are always per
 * tensor.</p>
 *
 * <p>Fake quantization is {@link FakeQuantGrad#apply}: a {@link TensorOps} composition on the tensor's backend that
 * reproduces, per element, the host formula {@code q = clamp(Math.round(x / scale) + zeroPoint, qMin, qMax);
 * y = (q - zeroPoint) * scale}, recorded as one autograd node whose gradient is the clipped straight-through estimator
 * {@code gx = gy} where the code is in {@code [qMin, qMax]} and 0 where it is clamped (or NaN). Gradients therefore
 * reach the base module's parameters, and the input when it requires grad. The scale and zero point come from
 * {@link Quantizer#integerParams}: symmetric codes around 0, or PyTorch's affine parameters over the observed range
 * widened to contain 0. The observers read back the observed range without recording
 * ({@link Quantizer#observedRange}); they update whenever they are enabled ({@link #enableObservers},
 * {@link #freezeObservers}), in training and evaluation mode alike and whether or not fake quantization is enabled
 * ({@link #enableFakeQuant}), as PyTorch's {@code FakeQuantize} observes independently of {@code fake_quant_enabled}:
 * with observers enabled and fake quantization disabled a forward pass calibrates every quantized site and returns
 * the base module's unquantized output.</p>
 *
 * <p><b>FP8.</b> The types {@code FP8_E4M3} and {@code FP8_E5M2} fake-quantize to the values their post-training
 * codes reconstruct ({@link FakeQuantGrad#applyFp8}, {@link FakeQuantGrad#applyFp8PerChannel}):
 * {@code decode(encode(x / scale)) * scale} with the OCP FP8 encoding {@link Quantizer} stores, scale
 * {@code amax / fmax} of the observed range's largest magnitude ({@code fmax} 448 or 57344; scale 1 when it is 0) and
 * zero point 0. The straight-through gradient is 0 where {@code |x / scale| > fmax} (saturated). The observer
 * {@code eps} applies to the integer types only.</p>
 *
 * <p><b>State.</b> Each site's running range, scale and zero point are the persistent buffers of a
 * {@link FakeQuantizeObserver} (PyTorch's {@code FakeQuantize}), registered in the children {@code weight_fake_quant}
 * (under the weight's path with {@code '.'} replaced by {@code '_'}) and {@code activation_fake_quant} (under the
 * activation's name), so {@link #stateDict()}, the model serializer and checkpoints save and restore the calibration
 * (for example {@code weight_fake_quant.weight.scale}, {@code activation_fake_quant.input.min_val}). The observers of
 * the base module's weights, and of the {@code input} and {@code output} activations when the base module has a
 * parameter to place them by, exist from construction; an observer for any other site, or for activations of a base
 * module without parameters, is created by the first observation or fake quantization of that site and is part of the
 * state from then on.</p>
 *
 * <p>The base module is registered as the child {@code base}, so {@link #parameters()}, {@link #train()},
 * {@link #eval()}, {@code to(...)}, {@link #zeroGrad()} and {@link #close()} reach it.</p>
 */
public final class QATModule extends Module {
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";

    private final Module baseModule;
    private final Quantizer.QuantizationType quantType;
    private final boolean quantizeWeights;
    private final boolean quantizeActivations;
    private final boolean perChannelQuantization;
    private final float observerEps;

    private final ModuleDict weightObservers;
    private final ModuleDict activationObservers;
    /** The site name each observer key was created for, per container, to refuse two names with one key. */
    private final Map<String, String> weightSiteNames = new LinkedHashMap<>();
    private final Map<String, String> activationSiteNames = new LinkedHashMap<>();
    /** The weight sites observed per output channel. */
    private final Set<String> perChannelSites = new HashSet<>();

    private boolean observersEnabled = true;
    private boolean fakeQuantEnabled = true;

    private QATModule(Builder builder) {
        this.baseModule = registerModule("base",
            Objects.requireNonNull(builder.baseModule, "Base module cannot be null"));
        this.quantType = builder.quantType != null ? builder.quantType : Quantizer.QuantizationType.INT8_SYMMETRIC;
        this.quantizeWeights = builder.quantizeWeights;
        this.quantizeActivations = builder.quantizeActivations;
        this.perChannelQuantization = builder.perChannelQuantization;
        this.observerEps = builder.observerEps;
        this.weightObservers = registerModule("weight_fake_quant", new ModuleDict());
        this.activationObservers = registerModule("activation_fake_quant", new ModuleDict());
        initializeObservers();
    }

    /**
     * The observers that can be placed at construction: one per weight of the base module (by its first path) when
     * weights are fake-quantized, and the {@code input} and {@code output} observers on the base module's backend when
     * activations are.
     */
    private void initializeObservers() {
        if (quantizeWeights) {
            for (Map.Entry<String, Parameter> site : baseWeightSites().entrySet()) {
                weightObserver(site.getKey(), site.getValue().tensor());
            }
        }
        ComputeBackend backend = baseModule.backend();
        if (quantizeActivations && backend != null) {
            activationObserver(INPUT, backend);
            activationObserver(OUTPUT, backend);
        }
    }

    /**
     * The base module's output with its weights and its input and output activations fake-quantized as enabled
     * (class description), recorded through {@link FakeQuantGrad} when grad mode is on and a fake-quantized tensor
     * requires grad. The fake-quantized weights are substituted only for the base forward pass and closed after it;
     * intermediate results this module creates are released ({@link #forwardStep}). With fake quantization disabled
     * the enabled observers still fold in the weights and the input and output activations, and the base module's
     * output is returned as it computes it.
     */
    @Override
    public Tensor forward(Tensor input) {
        if ((!quantizeWeights && !quantizeActivations) || !sitesActive()) {
            return baseModule.forward(input);
        }
        Map<Parameter, Tensor> weights = quantizeWeights ? fakeQuantizedWeights() : Collections.emptyMap();
        Tensor output = null;
        try {
            try (Parameter.Substitution scope = weights.isEmpty() ? null : baseModule.substitute(weights)) {
                output = quantizeActivations ? activationQuantizedForward(input) : baseModule.forward(input);
            }
            return output;
        } finally {
            for (Tensor weight : weights.values()) {
                if (weight != output) {
                    weight.close();
                }
            }
        }
    }

    /** The base forward with the input and output activations fake-quantized. */
    private Tensor activationQuantizedForward(Tensor input) {
        Tensor quantizedInput = fakeQuantizeActivation(input, INPUT);
        Tensor output = forwardStep(quantizedInput, input, baseModule::forward);
        return forwardStep(output, input, out -> fakeQuantizeActivation(out, OUTPUT));
    }

    /**
     * The fake-quantized value of every weight site of the base module ({@link #baseWeightSites}), new tensors the
     * caller closes. Each site's enabled observer folds its weight in first; with fake quantization disabled that is
     * all that happens and the map is empty.
     */
    private Map<Parameter, Tensor> fakeQuantizedWeights() {
        Map<Parameter, Tensor> quantized = new IdentityHashMap<>();
        try {
            for (Map.Entry<String, Parameter> site : baseWeightSites().entrySet()) {
                Tensor weight = site.getValue().tensor();
                Tensor value = quantizedWeight(weight, site.getKey());
                if (value != weight) {
                    quantized.put(site.getValue(), value);
                }
            }
            return quantized;
        } catch (RuntimeException e) {
            for (Tensor weight : quantized.values()) {
                try {
                    weight.close();
                } catch (RuntimeException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        }
    }

    /**
     * The weight sites of the base module in {@link Module#namedParameters()} order: every weight ({@link #isWeight})
     * once, keyed by its first path.
     */
    private Map<String, Parameter> baseWeightSites() {
        Map<String, Parameter> sites = new LinkedHashMap<>();
        Set<Parameter> placed = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Map.Entry<String, Parameter> entry : baseModule.namedParameters().entrySet()) {
            Parameter param = entry.getValue();
            if (isWeight(param) && placed.add(param)) {
                sites.put(entry.getKey(), param);
            }
        }
        return sites;
    }

    /** A floating-point parameter of rank 2 or more: the parameters PyTorch's QAT modules fake-quantize. */
    private static boolean isWeight(Parameter param) {
        return param.shape().rank() >= 2 && param.dtype().isFloatingPoint();
    }

    /**
     * The first path of every parameter of {@code named} (a base module's {@link Module#namedParameters()}), by
     * identity: the weight site of a parameter reachable under several paths.
     */
    private static Map<Parameter, String> firstPaths(Map<String, Parameter> named) {
        Map<Parameter, String> first = new IdentityHashMap<>();
        for (Map.Entry<String, Parameter> entry : named.entrySet()) {
            first.putIfAbsent(entry.getValue(), entry.getKey());
        }
        return first;
    }

    /**
     * The weight site of {@code path}: the first path of the base module's parameter at {@code path}, or {@code path}
     * itself when it is not a parameter path of the base module.
     */
    private String weightSiteOf(String path) {
        Map<String, Parameter> named = baseModule.namedParameters();
        Parameter param = named.get(path);
        return param == null ? path : firstPaths(named).get(param);
    }

    /** Whether {@code weight} is observed per output channel: per-channel quantization and a rank of at least 1. */
    private boolean perChannel(Tensor weight) {
        return perChannelQuantization && weight.shape().rank() >= 1;
    }

    /** Whether a site does anything: its observer folds values in, or fake quantization is applied, or both. */
    private boolean sitesActive() {
        return observersEnabled || fakeQuantEnabled;
    }

    /**
     * {@code weight} fake-quantized with the observer of the weight site {@code site}, per channel or per tensor, in
     * the weight's dtype (a new tensor the caller owns), or {@code weight} itself when fake quantization is disabled.
     * An enabled observer first folds the weight's range in, whether or not fake quantization is enabled.
     */
    private Tensor quantizedWeight(Tensor weight, String site) {
        if (!sitesActive()) {
            return weight;
        }
        FakeQuantizeObserver observer = weightObserver(site, weight);
        boolean perChannel = perChannelSites.contains(site);
        if (observersEnabled) {
            FakeQuantizeObserver.QuantParams observed = foldWeight(observer, weight, perChannel);
            return fakeQuantEnabled ? fakeQuantizeWeightWith(weight, observed, perChannel) : weight;
        }
        return fakeQuantizeWeightWith(weight, observer.quantParams(), perChannel);
    }

    /** {@code weight} fake-quantized with {@code params}, per channel or per tensor, in the weight's dtype. */
    private Tensor fakeQuantizeWeightWith(Tensor weight, FakeQuantizeObserver.QuantParams params, boolean perChannel) {
        Tensor quantized = perChannel ? fakeQuantizePerChannel(weight, params) : fakeQuantize(weight, params);
        if (quantized.dtype() == weight.dtype()) {
            return quantized;
        }
        try (Tensor full = quantized) {
            return full.to(weight.dtype());
        }
    }

    /**
     * {@code weight} fake-quantized with the observed scale and zero point of {@code paramName} (per output channel
     * with {@code perChannelQuantization}), a new tensor the caller owns, recorded through {@link FakeQuantGrad} when
     * {@code weight} requires grad and grad mode is on; or {@code weight} itself when weights are not quantized or
     * fake quantization is disabled. An enabled observer folds the weight's range in either way, so with fake
     * quantization disabled this calibrates the site. {@link #forward} applies it to every weight of the base module
     * under the weight's path. Every path of a parameter the base module reaches under several paths names the
     * observer of its first path.
     *
     * @throws IllegalArgumentException when {@code paramName} is empty, maps to the observer key of another name, or
     *                                  names a per-channel observer whose channel count differs from {@code weight}'s
     */
    public Tensor fakeQuantizeWeight(Tensor weight, String paramName) {
        if (!quantizeWeights) {
            return weight;
        }
        Objects.requireNonNull(paramName, "Site name cannot be null");
        return quantizedWeight(weight, weightSiteOf(paramName));
    }

    /**
     * {@code activation} fake-quantized with the observed scale and zero point of {@code name} (a new tensor the
     * caller owns, recorded through {@link FakeQuantGrad} when {@code activation} requires grad and grad mode is on),
     * or {@code activation} itself when activations are not quantized or fake quantization is disabled. An enabled
     * observer folds the activation's range in either way, so with fake quantization disabled this calibrates the
     * site.
     *
     * @throws IllegalArgumentException when {@code name} is empty or contains {@code '.'}
     */
    public Tensor fakeQuantizeActivation(Tensor activation, String name) {
        if (!quantizeActivations || !sitesActive()) {
            return activation;
        }
        FakeQuantizeObserver observer = activationObserver(name, activation.backend());
        if (observersEnabled) {
            FakeQuantizeObserver.QuantParams observed = observeRange(observer, activation);
            return fakeQuantEnabled ? fakeQuantize(activation, observed) : activation;
        }
        return fakeQuantize(activation, observer.quantParams());
    }

    /** {@code tensor} fake-quantized per tensor with the first channel of {@code params}, in this module's type. */
    private Tensor fakeQuantize(Tensor tensor, FakeQuantizeObserver.QuantParams params) {
        if (Quantizer.isFp8(quantType)) {
            return FakeQuantGrad.applyFp8(tensor, params.scales[0], quantType);
        }
        return FakeQuantGrad.apply(tensor, params.scales[0], params.zeroPoints[0], codeMin(), codeMax());
    }

    /** {@code tensor} fake-quantized per channel of dimension 0 with {@code params}, in this module's type. */
    private Tensor fakeQuantizePerChannel(Tensor tensor, FakeQuantizeObserver.QuantParams params) {
        if (Quantizer.isFp8(quantType)) {
            return FakeQuantGrad.applyFp8PerChannel(tensor, params.scales, quantType);
        }
        return FakeQuantGrad.applyPerChannel(tensor, params.scales, params.zeroPoints, codeMin(), codeMax());
    }

    /**
     * Folds the range of {@code weight} into {@code observer}, per output channel when {@code perChannel}, and returns
     * the parameters derived from the updated running range.
     */
    private static FakeQuantizeObserver.QuantParams foldWeight(FakeQuantizeObserver observer, Tensor weight,
                                                               boolean perChannel) {
        if (!perChannel) {
            return observeRange(observer, weight);
        }
        float[][] range = Quantizer.observedChannelRanges(weight);
        return observer.observe(range[0], range[1]);
    }

    /** Folds the per-tensor range of {@code tensor} into {@code observer}. */
    private static FakeQuantizeObserver.QuantParams observeRange(FakeQuantizeObserver observer, Tensor tensor) {
        float[] range = Quantizer.observedRange(tensor);
        return observer.observe(new float[]{range[0]}, new float[]{range[1]});
    }

    /**
     * The observer of the weight site {@code paramName}, created on {@code weight}'s backend (one channel per row of
     * dimension 0 with per-channel quantization) when the site has none yet.
     */
    private FakeQuantizeObserver weightObserver(String paramName, Tensor weight) {
        boolean perChannel = perChannel(weight);
        int channels = perChannel ? weight.shape().dim(0) : 1;
        FakeQuantizeObserver observer = observer(weightObservers, weightSiteNames, paramName.replace('.', '_'),
            paramName, channels, weight.backend());
        if (perChannel) {
            perChannelSites.add(paramName);
        }
        return observer;
    }

    /** The observer of the activation site {@code name}, created on {@code backend} when the site has none yet. */
    private FakeQuantizeObserver activationObserver(String name, ComputeBackend backend) {
        Objects.requireNonNull(name, "Activation name cannot be null");
        if (name.indexOf('.') >= 0) {
            throw new IllegalArgumentException("Activation name cannot contain '.': " + name);
        }
        return observer(activationObservers, activationSiteNames, name, name, 1, backend);
    }

    /**
     * The observer registered under {@code key} in {@code container}, or a new one of {@code channels} channels on
     * {@code backend} registered there for the site {@code site}.
     */
    private FakeQuantizeObserver observer(ModuleDict container, Map<String, String> siteNames, String key,
                                          String site, int channels, ComputeBackend backend) {
        Objects.requireNonNull(site, "Site name cannot be null");
        if (key.isEmpty()) {
            throw new IllegalArgumentException("Fake quantization site name cannot be empty");
        }
        FakeQuantizeObserver existing = (FakeQuantizeObserver) container.get(key);
        if (existing != null) {
            String owner = siteNames.get(key);
            if (!site.equals(owner)) {
                throw new IllegalArgumentException("Sites '" + owner + "' and '" + site
                    + "' would share the observer key '" + key + "'");
            }
            if (existing.channels() != channels) {
                throw new IllegalArgumentException("Site '" + site + "' has " + channels
                    + " channels, its observer " + existing.channels());
            }
            return existing;
        }
        FakeQuantizeObserver created = new FakeQuantizeObserver(backend, channels, quantType, observerEps);
        container.put(key, created);
        siteNames.put(key, site);
        return created;
    }

    private int codeMin() {
        return Quantizer.codeMin(quantType);
    }

    private int codeMax() {
        return Quantizer.codeMax(quantType);
    }

    public void enableObservers(boolean enabled) {
        this.observersEnabled = enabled;
    }

    public void enableFakeQuant(boolean enabled) {
        this.fakeQuantEnabled = enabled;
    }

    public void freezeObservers() {
        this.observersEnabled = false;
    }

    /**
     * The {@link Quantizer} of this module's observed parameters, as PyTorch's {@code convert} takes each
     * {@code FakeQuantize}'s scale and zero point: every weight site carries the scale and zero point of its observer
     * (per channel along dimension 0 for a per-channel site), keyed by the weight's path in the base module, and every
     * activation site its stored per-tensor parameters ({@link Quantizer#getActivationQuantParams}). A parameter the
     * base module reaches under several paths carries the parameters of its one site under every path.
     *
     * <p>While the observers are enabled, the observer of every weight of the base module first folds in the weight's
     * current range, as PyTorch's {@code from_float} passes the final weight through the weight's {@code FakeQuantize}
     * and as the next forward pass would; this updates the observers. Frozen observers keep their stored buffers. The
     * activation sites export the parameters their forward passes stored, without observing. Either way
     * {@code toQuantizer().quantizeModel(baseModule())} codes every weight exactly as a forward pass after the export
     * fake-quantizes it: it codes the fake-quantized weights only, leaves biases and normalization parameters in full
     * precision, and does not recalibrate. A site that has never been observed has scale 1 and zero point 0,
     * PyTorch's parameters of an empty range.</p>
     *
     * @throws IllegalArgumentException when a site's stored state (for example one loaded from a checkpoint) is not
     *                                  a valid scale and zero point of this module's type
     */
    public Quantizer toQuantizer() {
        Map<String, Parameter> named = baseModule.namedParameters();
        Map<String, Quantizer.QuantizationParams> sites = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : weightSiteNames.entrySet()) {
            String site = entry.getValue();
            FakeQuantizeObserver observer = (FakeQuantizeObserver) weightObservers.get(entry.getKey());
            boolean perChannel = perChannelSites.contains(site);
            Parameter param = named.get(site);
            if (observersEnabled && param != null && isWeight(param)) {
                foldWeight(observer, param.tensor(), perChannel);
            }
            sites.put(site, observedParams(site, observer, perChannel));
        }
        Map<String, Quantizer.QuantizationParams> weights = new LinkedHashMap<>();
        Map<Parameter, String> first = firstPaths(named);
        for (Map.Entry<String, Parameter> entry : named.entrySet()) {
            Quantizer.QuantizationParams params = sites.get(first.get(entry.getValue()));
            if (params != null) {
                weights.put(entry.getKey(), params);
            }
        }
        for (Map.Entry<String, Quantizer.QuantizationParams> entry : sites.entrySet()) {
            weights.putIfAbsent(entry.getKey(), entry.getValue());
        }
        Map<String, Quantizer.QuantizationParams> activations = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : activationSiteNames.entrySet()) {
            String site = entry.getValue();
            FakeQuantizeObserver observer = (FakeQuantizeObserver) activationObservers.get(entry.getKey());
            activations.put(site, observedParams(site, observer, false));
        }
        return Quantizer.fromObserved(quantType, weights, activations);
    }

    /** The stored scale and zero point of the observer of {@code site} as quantization parameters of this type. */
    private Quantizer.QuantizationParams observedParams(String site, FakeQuantizeObserver observer,
                                                        boolean perChannel) {
        FakeQuantizeObserver.QuantParams stored = observer.quantParams();
        try {
            return Quantizer.observedParams(quantType, stored.scales, stored.zeroPoints, perChannel);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Fake quantization site '" + site + "' holds invalid parameters: "
                + e.getMessage(), e);
        }
    }

    public Module baseModule() {
        return baseModule;
    }

    public Quantizer.QuantizationType quantType() {
        return quantType;
    }

    public boolean isQuantizeWeights() {
        return quantizeWeights;
    }

    public boolean isQuantizeActivations() {
        return quantizeActivations;
    }

    /** The stored scale of every per-tensor weight site, by weight path (read back from the observers). */
    public Map<String, Float> weightScaleObservers() {
        Map<String, Float> scales = new LinkedHashMap<>();
        for (Map.Entry<String, FakeQuantizeObserver> entry : weightSites(false).entrySet()) {
            scales.put(entry.getKey(), entry.getValue().quantParams().scales[0]);
        }
        return Collections.unmodifiableMap(scales);
    }

    /** The stored zero point of every per-tensor weight site, by weight path. */
    public Map<String, Integer> weightZeroPointObservers() {
        Map<String, Integer> zeroPoints = new LinkedHashMap<>();
        for (Map.Entry<String, FakeQuantizeObserver> entry : weightSites(false).entrySet()) {
            zeroPoints.put(entry.getKey(), entry.getValue().quantParams().zeroPoints[0]);
        }
        return Collections.unmodifiableMap(zeroPoints);
    }

    /** The stored per-channel scales of every per-channel weight site, by weight path. */
    public Map<String, float[]> weightChannelScales() {
        Map<String, float[]> scales = new LinkedHashMap<>();
        for (Map.Entry<String, FakeQuantizeObserver> entry : weightSites(true).entrySet()) {
            scales.put(entry.getKey(), entry.getValue().quantParams().scales);
        }
        return Collections.unmodifiableMap(scales);
    }

    /** The stored per-channel zero points of every per-channel weight site, by weight path. */
    public Map<String, int[]> weightChannelZeroPoints() {
        Map<String, int[]> zeroPoints = new LinkedHashMap<>();
        for (Map.Entry<String, FakeQuantizeObserver> entry : weightSites(true).entrySet()) {
            zeroPoints.put(entry.getKey(), entry.getValue().quantParams().zeroPoints);
        }
        return Collections.unmodifiableMap(zeroPoints);
    }

    /** The weight sites observed per channel ({@code perChannel}) or per tensor, by weight path. */
    private Map<String, FakeQuantizeObserver> weightSites(boolean perChannel) {
        Map<String, FakeQuantizeObserver> sites = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : weightSiteNames.entrySet()) {
            String site = entry.getValue();
            if (perChannelSites.contains(site) == perChannel) {
                sites.put(site, (FakeQuantizeObserver) weightObservers.get(entry.getKey()));
            }
        }
        return sites;
    }

    public boolean isPerChannelQuantization() {
        return perChannelQuantization;
    }

    @Override
    public String moduleName() {
        return "QATModule[" + baseModule.moduleName() + "]";
    }

    @Override
    public String toString() {
        return "QATModule{" +
            "baseModule=" + baseModule.moduleName() +
            ", quantType=" + quantType +
            ", quantizeWeights=" + quantizeWeights +
            ", quantizeActivations=" + quantizeActivations +
            ", observersEnabled=" + observersEnabled +
            ", fakeQuantEnabled=" + fakeQuantEnabled +
            '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Module baseModule;
        private Quantizer.QuantizationType quantType = Quantizer.QuantizationType.INT8_SYMMETRIC;
        private boolean quantizeWeights = true;
        private boolean quantizeActivations = true;
        private boolean perChannelQuantization = false;
        private float observerEps = 1e-5f;

        public Builder baseModule(Module baseModule) {
            this.baseModule = baseModule;
            return this;
        }

        public Builder quantType(Quantizer.QuantizationType quantType) {
            this.quantType = quantType;
            return this;
        }

        public Builder quantizeWeights(boolean quantizeWeights) {
            this.quantizeWeights = quantizeWeights;
            return this;
        }

        public Builder quantizeActivations(boolean quantizeActivations) {
            this.quantizeActivations = quantizeActivations;
            return this;
        }

        public Builder perChannelQuantization(boolean perChannelQuantization) {
            this.perChannelQuantization = perChannelQuantization;
            return this;
        }

        public Builder observerEps(float observerEps) {
            if (observerEps <= 0) {
                throw new IllegalArgumentException("Observer eps must be positive, got: " + observerEps);
            }
            this.observerEps = observerEps;
            return this;
        }

        public QATModule build() {
            return new QATModule(this);
        }
    }
}

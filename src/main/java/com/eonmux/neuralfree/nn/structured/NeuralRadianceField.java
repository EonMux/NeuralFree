package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.nn.activations.Sigmoid;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Neural radiance field: an MLP from a 3D position (with sinusoidal encoding) to density and, together with the
 * encoded view direction, to colour, plus volume rendering of rays through it. Every step, the positional encodings,
 * skip concatenations, heads and compositing included, is a {@link TensorOps} composition on the input's backend, so
 * the field runs on the device when the train kernels are available there; intermediates are released before
 * returning. Every operation records, the encodings' {@link TensorOps#sin} and {@link TensorOps#cos} included, so a
 * loss on {@link #forward} or {@link #renderRays} reaches every layer, and the input when it requires grad.
 */
public final class NeuralRadianceField extends Module {
    private final int posEncodingDim;
    private final int dirEncodingDim;
    private final int hiddenDim;
    private final int numLayers;
    private final int skips[];
    private final float near;
    private final float far;
    private final int numSamples;

    private final List<Linear> layers;
    private Linear densityLayer;
    private Linear rgbLayer1;
    private Linear rgbLayer2;
    private ReLU relu;
    private Sigmoid sigmoid;

    private boolean initialized = false;

    public NeuralRadianceField(int hiddenDim, int numLayers) {
        this(10, 4, hiddenDim, numLayers, new int[]{4}, 0.0f, 1.0f, 64);
    }

    public NeuralRadianceField(int posEncodingDim, int dirEncodingDim, int hiddenDim,
                               int numLayers, int[] skips, float near, float far, int numSamples) {
        if (hiddenDim <= 0) {
            throw new IllegalArgumentException("hiddenDim must be positive: " + hiddenDim);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (posEncodingDim <= 0) {
            throw new IllegalArgumentException("posEncodingDim must be positive: " + posEncodingDim);
        }
        if (dirEncodingDim <= 0) {
            throw new IllegalArgumentException("dirEncodingDim must be positive: " + dirEncodingDim);
        }
        if (numSamples <= 0) {
            throw new IllegalArgumentException("numSamples must be positive: " + numSamples);
        }
        if (far <= near) {
            throw new IllegalArgumentException("far must be greater than near: far=" + far + ", near=" + near);
        }

        this.posEncodingDim = posEncodingDim;
        this.dirEncodingDim = dirEncodingDim;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.skips = skips != null ? skips.clone() : new int[0];
        this.near = near;
        this.far = far;
        this.numSamples = numSamples;
        this.layers = new ArrayList<>();
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        int inputDim = networkInputDim();

        for (int i = 0; i < numLayers; i++) {
            int inFeatures = i == 0 ? inputDim : hiddenDim + (isSkipLayer(i) ? inputDim : 0);
            Linear layer = new Linear(inFeatures, hiddenDim);
            layer.initializeXavier(backend, random);
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }

        densityLayer = new Linear(hiddenDim, 1);
        densityLayer.initializeXavier(backend, random);
        registerModule("density_layer", densityLayer);

        int dirInputDim = hiddenDim + 3 + 2 * 3 * dirEncodingDim;
        rgbLayer1 = new Linear(dirInputDim, hiddenDim / 2);
        rgbLayer1.initializeXavier(backend, random);
        registerModule("rgb_layer_1", rgbLayer1);

        rgbLayer2 = new Linear(hiddenDim / 2, 3);
        rgbLayer2.initializeXavier(backend, random);
        registerModule("rgb_layer_2", rgbLayer2);

        relu = new ReLU();
        registerModule("relu", relu);

        sigmoid = new Sigmoid();
        registerModule("sigmoid", sigmoid);

        initialized = true;
    }

    /**
     * Maps {@code [batch, 6]} rows (position, then view direction) to new {@code [batch, 4]} rows
     * {@code [r, g, b, density]}. The network input is {@code [p, enc(p)]}; layer {@code i > 0} listed in the skips
     * reads {@code [h, [p, enc(p)]]}; density is {@code relu(densityLayer(h))} and colour is
     * {@code sigmoid(rgb2(relu(rgb1([h, d, enc(d)]))))}, every layer followed by ReLU.
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        if (inputShape.rank() != 2 || inputShape.dim(1) != 6) {
            throw new IllegalArgumentException("Input must be [batch, 6] (3D position + 3D direction), got: " + inputShape);
        }

        try (Tensor positions = input.slice(1, 0, 3);
             Tensor directions = input.slice(1, 3, 6);
             Tensor posEncoded = positionalEncoding(positions, posEncodingDim);
             Tensor networkInput = TensorOps.cat(Arrays.asList(positions, posEncoded), 1);
             Tensor features = trunk(networkInput);
             Tensor densityLogits = densityLayer.forward(features);
             Tensor density = relu.forward(densityLogits);
             Tensor rgb = colorHead(features, directions)) {
            return TensorOps.cat(Arrays.asList(rgb, density), 1);
        }
    }

    private Tensor trunk(Tensor networkInput) {
        Tensor x = networkInput;
        try {
            for (int i = 0; i < numLayers; i++) {
                if (i > 0 && isSkipLayer(i)) {
                    x = replace(x, TensorOps.cat(Arrays.asList(x, networkInput), 1), networkInput);
                }
                x = replace(x, layers.get(i).forward(x), networkInput);
                x = replace(x, relu.forward(x), networkInput);
            }
            return x;
        } catch (RuntimeException e) {
            if (x != networkInput) {
                x.close();
            }
            throw e;
        }
    }

    private Tensor colorHead(Tensor features, Tensor directions) {
        try (Tensor dirEncoded = positionalEncoding(directions, dirEncodingDim);
             Tensor rgbInput = TensorOps.cat(Arrays.asList(features, directions, dirEncoded), 1);
             Tensor hidden = rgbLayer1.forward(rgbInput);
             Tensor activated = relu.forward(hidden);
             Tensor logits = rgbLayer2.forward(activated)) {
            return sigmoid.forward(logits);
        }
    }

    private static Tensor replace(Tensor previous, Tensor next, Tensor input) {
        if (previous != input && previous != next) {
            previous.close();
        }
        return next;
    }

    private boolean isSkipLayer(int layer) {
        for (int skip : skips) {
            if (layer == skip) {
                return true;
            }
        }
        return false;
    }

    private int networkInputDim() {
        return 3 + 2 * 3 * posEncodingDim;
    }

    /**
     * {@code [n, 3]} coordinates to {@code [n, 6K]} features: for coordinate {@code j} and frequency {@code k},
     * columns {@code 2(jK + k)} and {@code 2(jK + k) + 1} hold {@code sin(a)} and {@code cos(a)} of
     * {@code a = (2^k * c) * (float) PI}, the angle formed in float arithmetic and its sine and cosine by
     * {@link TensorOps#sin} and {@link TensorOps#cos}.
     */
    private static Tensor positionalEncoding(Tensor coords, int numFrequencies) {
        ComputeBackend backend = coords.backend();
        int n = coords.shape().dim(0);
        float[] frequencies = new float[numFrequencies];
        for (int k = 0; k < numFrequencies; k++) {
            frequencies[k] = (float) Math.pow(2.0, k);
        }
        try (Tensor column = coords.reshape(Shape.of(n, 3, 1));
             Tensor freqs = upload(backend, Shape.of(1, 1, numFrequencies), frequencies);
             Tensor scaled = TensorOps.mul(column, freqs);
             Tensor angles = TensorOps.mul(scaled, (float) Math.PI);
             Tensor sin = TensorOps.sin(angles);
             Tensor cos = TensorOps.cos(angles);
             Tensor sinLast = sin.reshape(Shape.of(n, 3, numFrequencies, 1));
             Tensor cosLast = cos.reshape(Shape.of(n, 3, numFrequencies, 1));
             Tensor pairs = TensorOps.cat(Arrays.asList(sinLast, cosLast), 3)) {
            return pairs.reshape(Shape.matrix(n, 3 * 2 * numFrequencies));
        }
    }

    /**
     * Volume-renders rays given by {@code [numRays, 3]} origins and directions over {@code numSamples} equally
     * spaced depths in {@code [near, far]} and returns new {@code [numRays, 4]} rows {@code [r, g, b, opacity]}.
     * Sample {@code s} has {@code alpha = 1 - exp(-sigma * delta_s)}, weight {@code alpha * T} and transmittance
     * update {@code T *= 1 - alpha}, sample by sample in depth order; colour and opacity are the weight-sums. Every
     * step is a tensor op on the rays' backend (the weight sums in the reduction class of the
     * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}).
     */
    public Tensor renderRays(Tensor rayOrigins, Tensor rayDirections) {
        ensureInitialized();
        Objects.requireNonNull(rayOrigins, "rayOrigins cannot be null");
        Objects.requireNonNull(rayDirections, "rayDirections cannot be null");

        Shape originsShape = rayOrigins.shape();
        if (originsShape.rank() != 2 || originsShape.dim(1) != 3 || !originsShape.equals(rayDirections.shape())) {
            throw new IllegalArgumentException("rayOrigins and rayDirections must both be [numRays, 3], got "
                + originsShape + " and " + rayDirections.shape());
        }
        int numRays = originsShape.dim(0);
        ComputeBackend backend = rayOrigins.backend();
        float[] tVals = linspace(near, far, numSamples);

        Tensor raw;
        try (Tensor samples = samplePoints(rayOrigins, rayDirections, tVals);
             Tensor flatRaw = forward(samples)) {
            raw = flatRaw.reshape(Shape.of(numRays, numSamples, 4));
        }
        try (Tensor r = raw;
             Tensor rgb = r.slice(2, 0, 3);
             Tensor sigma = r.slice(2, 3, 4);
             Tensor sigmaRows = sigma.reshape(Shape.matrix(numRays, numSamples));
             Tensor deltas = upload(backend, Shape.vector(numSamples), sampleDeltas(tVals));
             Tensor negSigma = TensorOps.neg(sigmaRows);
             Tensor exponent = TensorOps.mul(negSigma, deltas);
             Tensor decay = TensorOps.exp(exponent);
             Tensor negDecay = TensorOps.neg(decay);
             Tensor alpha = TensorOps.add(negDecay, 1.0f);
             Tensor weights = compositingWeights(alpha, numRays);
             Tensor weights3 = weights.reshape(Shape.of(numRays, numSamples, 1));
             Tensor weightedRgb = TensorOps.mul(weights3, rgb);
             Tensor color = TensorOps.sum(weightedRgb, 1);
             Tensor opacity = TensorOps.sum(weights, 1);
             Tensor opacityColumn = opacity.reshape(Shape.matrix(numRays, 1))) {
            return TensorOps.cat(Arrays.asList(color, opacityColumn), 1);
        }
    }

    /** {@code [numRays * numSamples, 6]} rows {@code [o + t * d, d]}, ray-major. */
    private Tensor samplePoints(Tensor origins, Tensor directions, float[] tVals) {
        int numRays = origins.shape().dim(0);
        Shape perSample = Shape.of(numRays, numSamples, 3);
        try (Tensor o = origins.reshape(Shape.of(numRays, 1, 3));
             Tensor d = directions.reshape(Shape.of(numRays, 1, 3));
             Tensor t = upload(origins.backend(), Shape.of(1, numSamples, 1), tVals);
             Tensor offsets = TensorOps.mul(t, d);
             Tensor points = TensorOps.add(o, offsets);
             Tensor dirs = d.expand(perSample);
             Tensor rows = TensorOps.cat(Arrays.asList(points, dirs), 2)) {
            return rows.reshape(Shape.matrix(numRays * numSamples, 6));
        }
    }

    /**
     * {@code [numRays, numSamples]} weights {@code alpha_s * T_s}, with {@code T_0 = 1} and
     * {@code T_{s+1} = T_s * (1 - alpha_s)}.
     */
    private Tensor compositingWeights(Tensor alpha, int numRays) {
        List<Tensor> columns = new ArrayList<>(numSamples);
        Tensor transmittance = Tensor.ones(alpha.backend(), Shape.matrix(numRays, 1));
        try {
            for (int s = 0; s < numSamples; s++) {
                try (Tensor a = alpha.slice(1, s, s + 1)) {
                    columns.add(TensorOps.mul(a, transmittance));
                    if (s < numSamples - 1) {
                        try (Tensor negA = TensorOps.neg(a);
                             Tensor survive = TensorOps.add(negA, 1.0f);
                             Tensor previous = transmittance) {
                            transmittance = TensorOps.mul(previous, survive);
                        }
                    }
                }
            }
            return TensorOps.cat(columns, 1);
        } finally {
            transmittance.close();
            for (Tensor column : columns) {
                column.close();
            }
        }
    }

    /** Depth step of each sample: to the next sample, and for the last one the step from the previous sample. */
    private float[] sampleDeltas(float[] tVals) {
        float[] deltas = new float[numSamples];
        for (int s = 0; s < numSamples; s++) {
            deltas[s] = (s < numSamples - 1) ? tVals[s + 1] - tVals[s] : tVals[s] - (s > 0 ? tVals[s - 1] : near);
        }
        return deltas;
    }

    private static Tensor upload(ComputeBackend backend, Shape shape, float[] data) {
        try (Tensor flat = Tensor.fromArray(backend, data)) {
            return flat.reshape(shape);
        }
    }

    /** {@code num} equally spaced values from {@code start} to {@code end}; {@code {start}} when {@code num} is 1. */
    private static float[] linspace(float start, float end, int num) {
        float[] result = new float[num];
        if (num == 1) {
            result[0] = start;
            return result;
        }
        float step = (end - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            result[i] = start + i * step;
        }
        return result;
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("NeuralRadianceField not initialized. Call initialize(backend) first.");
        }
    }

    public int posEncodingDim() { return posEncodingDim; }
    public int dirEncodingDim() { return dirEncodingDim; }
    public int hiddenDim() { return hiddenDim; }
    public int numLayers() { return numLayers; }
    public float near() { return near; }
    public float far() { return far; }
    public int numSamples() { return numSamples; }
    public List<Linear> layers() { return new ArrayList<>(layers); }
    public Linear densityLayer() { ensureInitialized(); return densityLayer; }
    public Linear rgbLayer1() { ensureInitialized(); return rgbLayer1; }
    public Linear rgbLayer2() { ensureInitialized(); return rgbLayer2; }

    @Override
    public String toString() {
        return "NeuralRadianceField{" +
            "posEncodingDim=" + posEncodingDim +
            ", dirEncodingDim=" + dirEncodingDim +
            ", hiddenDim=" + hiddenDim +
            ", numLayers=" + numLayers +
            ", numSamples=" + numSamples +
            ", initialized=" + initialized +
            '}';
    }
}

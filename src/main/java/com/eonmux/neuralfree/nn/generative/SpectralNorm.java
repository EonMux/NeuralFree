package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Spectral normalization of a wrapped module's 2-D weight {@code W [out, in]}. Each forward in training mode runs
 * {@code numPowerIterations} power-iteration steps {@code v = normalize(W^T u)}, {@code u = normalize(W v)} with
 * {@code normalize(x) = x / sqrt(sum(x^2) + 1e-8)} under {@link ComputationGraph#noGrad()} and stores the results
 * in the buffers; in evaluation mode it uses the stored {@code u} and {@code v} as they are and updates nothing
 * (PyTorch's spectral norm). It then estimates {@code sigma = |u . (W v)|} and runs the wrapped forward with the
 * weight substituted ({@link Parameter#substitute}) by {@code W_sn = W * normCoeff / (sigma + 1e-8)}. The parameter's
 * own values are never written, so optimizers keep updating the unnormalized weight. Every step is a
 * {@link TensorOps} composition on the weight's backend (device when the train kernels are available, CPU reference
 * otherwise) with no readback.
 *
 * <p>Gradients: {@code sigma} and {@code W_sn} are recorded from the weight and the power-iteration results
 * {@code u}, {@code v} of this forward, which are constants of the graph (as in PyTorch's spectral norm), so the
 * wrapped module's gradient with respect to {@code W_sn} reaches {@code W} both directly and through {@code sigma}
 * ({@code ∂sigma/∂W = sign(u . (W v)) u vᵀ}). Each forward records its own {@code sigma}, so several
 * forwards in one loss (such as a discriminator's real and fake passes) differentiate through their own
 * normalization. The vectors the graph saves are fresh tensors; the buffers are updated by a copy under
 * {@code noGrad} and never alias them.</p>
 *
 * <p>The power-iteration vectors are the module buffers {@code u [out]} and {@code v [in]}, persisted across
 * forwards and moved by {@link Module#to(ComputeBackend)}. The weight is the wrapped parameter registered as
 * {@code weight}, else the one whose {@link Parameter#name()} is {@code weight}, else the first parameter. Optimizer
 * steps and {@link Module#to} happen between forwards, outside the substitution.</p>
 */
public final class SpectralNorm extends Module {

    private static final String U = "u";
    private static final String V = "v";
    private static final float NORM_EPS = 1e-8f;

    private final Module wrappedModule;
    private final float normCoeff;
    private final int numPowerIterations;

    private String weightKey;
    private boolean initialized;

    private volatile boolean closed = false;

    public SpectralNorm(Module wrappedModule) {
        this(wrappedModule, 1.0f, 1);
    }

    public SpectralNorm(Module wrappedModule, float normCoeff, int numPowerIterations) {
        Objects.requireNonNull(wrappedModule, "Wrapped module cannot be null");

        if (normCoeff <= 0) {
            throw new IllegalArgumentException("Norm coefficient must be positive, got: " + normCoeff);
        }
        if (numPowerIterations < 1) {
            throw new IllegalArgumentException("Number of power iterations must be at least 1");
        }

        this.wrappedModule = wrappedModule;
        this.normCoeff = normCoeff;
        this.numPowerIterations = numPowerIterations;

        registerModule("wrapped", wrappedModule);
    }

    /** {@link #initialize(ComputeBackend, Random)} with an unseeded {@link Random}. */
    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Selects the weight and creates the buffers {@code u} and {@code v} as normalized standard-normal vectors drawn
     * from {@code random} on the host ({@code u} first) and uploaded to {@code backend}.
     *
     * @throws IllegalStateException    when the wrapped module has no parameters or this module is already
     *                                  initialized
     * @throws IllegalArgumentException when the selected weight is not 2-D
     */
    public void initialize(ComputeBackend backend, Random random) {
        ensureOpen();
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (initialized) {
            throw new IllegalStateException("SpectralNorm is already initialized");
        }

        String key = findWeightKey();
        int[] shape = wrappedModule.getParameter(key).shape().dims();
        if (shape.length != 2) {
            throw new IllegalArgumentException("SpectralNorm only supports 2D weight matrices, got shape rank "
                + shape.length);
        }

        Tensor u;
        Tensor v;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            u = randomUnitVector(backend, shape[0], random);
            try {
                v = randomUnitVector(backend, shape[1], random);
            } catch (RuntimeException e) {
                u.close();
                throw e;
            }
        }
        registerBuffer(U, u);
        registerBuffer(V, v);
        weightKey = key;
        initialized = true;
    }

    /**
     * The wrapped forward under the spectrally normalized weight (class description). The substitution ends even when
     * the wrapped forward throws.
     *
     * @throws IllegalStateException when this module is not initialized, the weight parameter is gone, or the weight
     *                               no longer matches the power-iteration vectors
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        ensureInitialized();

        Parameter weightParam = wrappedModule.getParameter(weightKey);
        if (weightParam == null) {
            throw new IllegalStateException("Wrapped module no longer has the weight parameter '" + weightKey + "'");
        }
        Tensor weight = weightParam.tensor();
        requireMatchingVectors(weight);
        boolean training = isTraining();
        Tensor[] vectors = training ? powerIteration(weight) : storedVectors();
        Tensor u = vectors[0];
        Tensor v = vectors[1];
        Tensor normalized = null;
        Tensor output = null;
        try {
            try (Tensor scale = scale(weight, u, v)) {
                normalized = TensorOps.mul(weight, scale);
            }
            if (training) {
                storeVectors(u, v);
            }
            try (Parameter.Substitution substitution = weightParam.substitute(normalized)) {
                output = wrappedModule.forward(input);
            }
            return output;
        } finally {
            if (normalized != null && normalized != output) {
                normalized.close();
            }
            u.close();
            v.close();
        }
    }

    /** Throws {@link IllegalStateException} unless the weight is {@code [len(u), len(v)]}. */
    private void requireMatchingVectors(Tensor weight) {
        Shape expected = Shape.of(buffer(U).shape().dim(0), buffer(V).shape().dim(0));
        if (!weight.shape().equals(expected)) {
            throw new IllegalStateException("Weight of shape " + weight.shape() + " does not match the power-iteration"
                + " vectors of " + expected);
        }
    }

    /**
     * Copies of the stored {@code u} and {@code v} under {@link ComputationGraph#noGrad()}, owned by the caller: the
     * graph saves them, and a later training forward writes the buffers in place.
     */
    private Tensor[] storedVectors() {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor u = TensorOps.cat(Collections.singletonList(buffer(U)), 0);
            try {
                return new Tensor[]{u, TensorOps.cat(Collections.singletonList(buffer(V)), 0)};
            } catch (RuntimeException e) {
                u.close();
                throw e;
            }
        }
    }

    /**
     * {@code numPowerIterations} power-iteration steps from the stored {@code u} under
     * {@link ComputationGraph#noGrad()}: {@code {u, v}} as new tensors owned by the caller (never the buffers).
     */
    private Tensor[] powerIteration(Tensor weight) {
        Tensor storedU = buffer(U);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor currentU = null;
            Tensor currentV = null;
            try {
                for (int i = 0; i < numPowerIterations; i++) {
                    Tensor source = currentU != null ? currentU : storedU;
                    Tensor nextV;
                    try (Tensor weightT = weight.transpose();
                         Tensor rawV = TensorOps.matmul(weightT, source)) {
                        nextV = normalize(rawV);
                    }
                    release(currentV);
                    currentV = nextV;
                    Tensor nextU;
                    try (Tensor rawU = TensorOps.matmul(weight, currentV)) {
                        nextU = normalize(rawU);
                    }
                    release(currentU);
                    currentU = nextU;
                }
                return new Tensor[]{currentU, currentV};
            } catch (RuntimeException e) {
                release(currentU);
                release(currentV);
                throw e;
            }
        }
    }

    /**
     * The rank-0 scale {@code normCoeff / (|u . (W v)| + 1e-8)}, recorded with respect to {@code W} when grad mode is
     * on and the weight requires grad ({@code u} and {@code v} do not require grad).
     */
    private Tensor scale(Tensor weight, Tensor u, Tensor v) {
        try (Tensor weightV = TensorOps.matmul(weight, v);
             Tensor products = TensorOps.mul(weightV, u);
             Tensor dot = TensorOps.sum(products);
             Tensor sigma = TensorOps.abs(dot);
             Tensor shifted = TensorOps.add(sigma, NORM_EPS);
             Tensor coeff = constant(weight.backend(), normCoeff)) {
            return TensorOps.div(coeff, shifted);
        }
    }

    /** Copies the new vectors into the buffers {@code u} and {@code v} under {@link ComputationGraph#noGrad()}. */
    private void storeVectors(Tensor u, Tensor v) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            assign(buffer(U), u);
            assign(buffer(V), v);
        }
    }

    /** {@code x / sqrt(sum(x^2) + 1e-8)}. */
    private static Tensor normalize(Tensor x) {
        try (Tensor squares = TensorOps.mul(x, x);
             Tensor sumSquares = TensorOps.sum(squares);
             Tensor shifted = TensorOps.add(sumSquares, NORM_EPS);
             Tensor norm = TensorOps.sqrt(shifted)) {
            return TensorOps.div(x, norm);
        }
    }

    private static Tensor randomUnitVector(ComputeBackend backend, int length, Random random) {
        try (Tensor raw = Tensor.randn(backend, Shape.vector(length), DType.FLOAT32, random)) {
            return normalize(raw);
        }
    }

    /**
     * Writes {@code src}'s values into the buffer {@code dst} in place: on the device when the train kernels are
     * available, else through the host. Callers run it under {@link ComputationGraph#noGrad()}.
     */
    private static void assign(Tensor dst, Tensor src) {
        if (TrainGpu.forBackend(dst.backend()) != null) {
            GpuCoreOps.copyInto(src, dst);
        } else {
            GpuOptimizerOps.storeFloats(dst, src.toFloatArray());
        }
    }

    /** A rank-0 tensor holding {@code value}, filled on the device when the train kernels are available. */
    private static Tensor constant(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar())) {
            return TensorOps.add(zero, value);
        }
    }

    private static void release(Tensor t) {
        if (t != null) {
            t.close();
        }
    }

    /** The registration key of the weight (class description). */
    private String findWeightKey() {
        Map<String, Parameter> params = wrappedModule.namedParameters();
        if (params.isEmpty()) {
            throw new IllegalStateException("Wrapped module has no parameters");
        }
        if (params.containsKey("weight")) {
            return "weight";
        }
        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            if ("weight".equals(entry.getValue().name())) {
                return entry.getKey();
            }
        }
        return params.keySet().iterator().next();
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("SpectralNorm not initialized. Call initialize(backend) first.");
        }
    }

    public Module wrappedModule() {
        return wrappedModule;
    }

    public float normCoeff() {
        return normCoeff;
    }

    public int numPowerIterations() {
        return numPowerIterations;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("SpectralNorm has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            super.close();
        }
    }

    @Override
    public String toString() {
        return "SpectralNorm{" +
            "wrappedModule=" + wrappedModule.moduleName() +
            ", normCoeff=" + normCoeff +
            ", numPowerIterations=" + numPowerIterations +
            ", initialized=" + initialized +
            '}';
    }
}

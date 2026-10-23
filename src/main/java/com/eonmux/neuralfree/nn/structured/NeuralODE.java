package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.activations.Tanh;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Neural ordinary differential equation {@code dz/dt = f(z)}, with {@code f} a stack of linear layers joined by tanh,
 * integrated by a fixed-step solver (explicit Euler, classical RK4, or the fifth-order Dormand-Prince stages). Every
 * solver stage is a {@link TensorOps} composition over the state, so the integration runs on the device when the train
 * kernels are available on the state's backend; stage temporaries are released as soon as they are consumed and the
 * caller's tensors are never closed.
 *
 * <p>The state is {@code [rows, inputDim]} (or any shape whose last dimension is {@code inputDim} and whose leading
 * dimension is the row count); {@code f} acts on each row independently. The time argument of {@code f} is unused.
 * Stage scalings are {@link TensorOps#mul(Tensor, float)} in the single-rounding class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 *
 * <p><b>Training.</b> The forward integration records every solver stage, so a loss on its result back-propagates
 * through the discretized solver to the layers and the initial state (discretize-then-optimize); this is the training
 * path. {@link #adjoint} is the continuous adjoint method, a separate utility that integrates the adjoint backward
 * with exact vector-Jacobian products of {@code f} and returns the gradients without writing any gradient slot.</p>
 */
public final class NeuralODE extends Module {
    private static final int DEFAULT_TIME_POINTS = 50;
    /** Dormand-Prince 5(4) nodes {@code c_i} of the six stages a fifth-order step evaluates. */
    private static final double[] DOPRI_C = {0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0};
    /** Dormand-Prince stage rows {@code a_ij}: stage {@code i} reads {@code state + dt * sum_j a_ij * k_j}. */
    private static final double[][] DOPRI_A = {
        {},
        {1.0 / 5},
        {3.0 / 40, 9.0 / 40},
        {44.0 / 45, -56.0 / 15, 32.0 / 9},
        {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
        {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656}};
    /** Dormand-Prince fifth-order weights {@code b_j}, summing to 1. */
    private static final double[] DOPRI_B = {35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84};

    public enum Solver {
        EULER,
        RK4,
        DOPRI5
    }

    private final int inputDim;
    private final int hiddenDim;
    private final int numLayers;
    private final Solver solver;
    private final float rtol;
    private final float atol;
    private final List<Linear> layers;
    private final List<Tanh> activations;

    private boolean initialized = false;

    public NeuralODE(int inputDim, int hiddenDim) {
        this(inputDim, hiddenDim, 1, Solver.RK4, 1e-5f, 1e-8f);
    }

    public NeuralODE(int inputDim, int hiddenDim, int numLayers, Solver solver, float rtol, float atol) {
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive: " + inputDim);
        }
        if (hiddenDim <= 0) {
            throw new IllegalArgumentException("hiddenDim must be positive: " + hiddenDim);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (rtol <= 0) {
            throw new IllegalArgumentException("rtol must be positive: " + rtol);
        }
        if (atol <= 0) {
            throw new IllegalArgumentException("atol must be positive: " + atol);
        }

        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.numLayers = numLayers;
        this.solver = solver;
        this.rtol = rtol;
        this.atol = atol;
        this.layers = new ArrayList<>();
        this.activations = new ArrayList<>();
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        for (int i = 0; i < numLayers; i++) {
            int inFeatures = (i == 0) ? inputDim : hiddenDim;
            int outFeatures = (i == numLayers - 1) ? inputDim : hiddenDim;

            Linear layer = new Linear(inFeatures, outFeatures);
            layer.initializeXavier(backend, random);
            layers.add(layer);
            registerModule("layer_" + i, layer);

            Tanh activation = new Tanh();
            activations.add(activation);
            registerModule("activation_" + i, activation);
        }

        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, 0.0f, 1.0f);
    }

    /** Integrates {@code input} from {@code t0} to {@code t1} in 49 equal steps; returns a new tensor. */
    public Tensor forward(Tensor input, float t0, float t1) {
        return integrate(input, t0, t1, DEFAULT_TIME_POINTS);
    }

    /**
     * Integrates {@code input} from {@code t0} to {@code t1} over {@code numSteps} equally spaced time points and
     * returns the final state as a new tensor.
     *
     * @throws IllegalArgumentException when {@code numSteps < 2}
     */
    public Tensor forwardWithTrajectory(Tensor input, float t0, float t1, int numSteps) {
        return integrate(input, t0, t1, numSteps);
    }

    private Tensor integrate(Tensor input, float t0, float t1, int numSteps) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        float[] timeSpan = linspace(t0, t1, requireTimePoints(numSteps));

        Tensor state = input;
        try {
            for (int i = 0; i < timeSpan.length - 1; i++) {
                float t = timeSpan[i];
                float dt = timeSpan[i + 1] - t;
                Tensor next = step(t, state, dt);
                release(state, input);
                state = next;
            }
            return state;
        } catch (RuntimeException e) {
            release(state, input);
            throw e;
        }
    }

    private Tensor step(float t, Tensor state, float dt) {
        switch (solver) {
            case EULER:
                return eulerStep(t, state, dt);
            case RK4:
                return rk4Step(t, state, dt);
            case DOPRI5:
                return dopri5Step(t, state, dt);
            default:
                throw new IllegalStateException("Unknown solver: " + solver);
        }
    }

    /** {@code state + f(t, state) * dt}. */
    private Tensor eulerStep(float t, Tensor state, float dt) {
        try (Tensor k1 = dynamics(t, state)) {
            return addScaled(state, k1, dt);
        }
    }

    /** Classical RK4: {@code state + ((k1*dt/6 + k2*dt/3) + (k3*dt/3 + k4*dt/6))}. */
    private Tensor rk4Step(float t, Tensor state, float dt) {
        float dtHalf = dt / 2.0f;

        try (Tensor k1 = dynamics(t, state);
             Tensor state2 = addScaled(state, k1, dtHalf);
             Tensor k2 = dynamics(t + dtHalf, state2);
             Tensor state3 = addScaled(state, k2, dtHalf);
             Tensor k3 = dynamics(t + dtHalf, state3);
             Tensor state4 = addScaled(state, k3, dt);
             Tensor k4 = dynamics(t + dt, state4);
             Tensor t1 = TensorOps.mul(k1, dt / 6.0f);
             Tensor t2 = TensorOps.mul(k2, dt / 3.0f);
             Tensor t3 = TensorOps.mul(k3, dt / 3.0f);
             Tensor t4 = TensorOps.mul(k4, dt / 6.0f);
             Tensor sum1 = TensorOps.add(t1, t2);
             Tensor sum2 = TensorOps.add(t3, t4);
             Tensor sum = TensorOps.add(sum1, sum2)) {
            return TensorOps.add(state, sum);
        }
    }

    /**
     * One fixed step of the Dormand-Prince 5(4) method (torchdiffeq's {@code dopri5} tableau, fifth-order solution):
     * six stages {@code k_i = f(t + c_i * dt, state + dt * sum_j a_ij * k_j)} and
     * {@code state + dt * sum_j b_j * k_j}, each sum accumulated left to right over its nonzero coefficients.
     */
    private Tensor dopri5Step(float t, Tensor state, float dt) {
        Tensor[] k = new Tensor[DOPRI_C.length];
        try {
            k[0] = dynamics(t, state);
            for (int i = 1; i < k.length; i++) {
                try (Tensor stage = combine(state, k, DOPRI_A[i], dt)) {
                    k[i] = dynamics(t + (float) (DOPRI_C[i] * dt), stage);
                }
            }
            return combine(state, k, DOPRI_B, dt);
        } finally {
            for (Tensor stage : k) {
                if (stage != null) {
                    stage.close();
                }
            }
        }
    }

    /**
     * {@code base + dt * sum_j weights[j] * k[j]} as a new tensor, skipping zero weights; {@code weights[0]} is
     * nonzero in every row of the tableau.
     */
    private static Tensor combine(Tensor base, Tensor[] k, double[] weights, float dt) {
        Tensor acc = addScaled(base, k[0], (float) (weights[0] * dt));
        try {
            for (int j = 1; j < weights.length; j++) {
                if (weights[j] != 0.0) {
                    acc = replace(acc, addScaled(acc, k[j], (float) (weights[j] * dt)), null);
                }
            }
            return acc;
        } catch (RuntimeException e) {
            acc.close();
            throw e;
        }
    }

    /** {@code base + k * scale} as a new tensor. */
    private static Tensor addScaled(Tensor base, Tensor k, float scale) {
        try (Tensor scaled = TensorOps.mul(k, scale)) {
            return TensorOps.add(base, scaled);
        }
    }

    /** The dynamics network {@code f(t, state)}: the layers with tanh between them (none after the last). */
    private Tensor dynamics(float t, Tensor state) {
        Tensor x = state;
        try {
            for (int i = 0; i < numLayers; i++) {
                x = replace(x, layers.get(i).forward(x), state);
                if (i < numLayers - 1) {
                    x = replace(x, activations.get(i).forward(x), state);
                }
            }
            return x;
        } catch (RuntimeException e) {
            release(x, state);
            throw e;
        }
    }

    private static Tensor replace(Tensor previous, Tensor next, Tensor input) {
        if (previous != next) {
            release(previous, input);
        }
        return next;
    }

    private static void release(Tensor t, Tensor input) {
        if (t != input) {
            t.close();
        }
    }

    private static int requireTimePoints(int numSteps) {
        if (numSteps < 2) {
            throw new IllegalArgumentException("numSteps must be at least 2 time points: " + numSteps);
        }
        return numSteps;
    }

    private float[] linspace(float start, float end, int num) {
        float[] result = new float[num];
        float step = (end - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            result[i] = start + i * step;
        }
        return result;
    }

    /**
     * Integrates the adjoint {@code a} of the loss with respect to the state backward from {@code t1} to {@code t0}
     * over {@code numSteps} equally spaced time points with explicit Euler steps, alongside the state itself, and
     * returns the adjoint at {@code t0} as a new tensor shaped like {@code outputGrad}, owned by the caller. Each step
     * is {@link #adjoint}'s state update; the parameter gradients are not formed.
     *
     * @throws IllegalArgumentException when the shapes of {@code outputGrad} and {@code finalState} differ, they are
     *                                  on different backends, or {@code numSteps < 2}
     */
    public Tensor backwardAdjoint(Tensor outputGrad, Tensor finalState, float t0, float t1, int numSteps) {
        return integrateAdjoint(outputGrad, finalState, t0, t1, numSteps, false).stateAdjoint();
    }

    /**
     * The adjoint method: integrates the state adjoint {@code a} (seeded with {@code outputGrad}, the loss gradient at
     * {@code t1}) backward from {@code t1} to {@code t0} over {@code numSteps} equally spaced time points with explicit
     * Euler steps of {@code dt = t_{i+1} - t_i} (negative for {@code t0 < t1}), together with the state (seeded with
     * {@code finalState}) and the parameter gradients. At each point {@code t} with state {@code z}, the
     * vector-Jacobian products {@code [vz, v_theta] = a^T [df/dz, df/dtheta]} at {@code f(t, z)} are computed exactly
     * by {@link Autograd#grad} on a fresh leaf {@code z}, then {@code a <- a - dt * vz},
     * {@code g_theta <- g_theta - dt * v_theta} and {@code z <- z + dt * f(t, z)}. The result approximates
     * {@code dL/dz(t0)} and {@code dL/dtheta = integral over [t0, t1] of a^T df/dtheta dt}.
     *
     * <p>Each step runs in grad mode ({@link ComputationGraph#enableGrad()}, so it also works when called under
     * {@code noGrad}) inside a {@link ComputationGraph.StepScope}: the nodes it records are released, the graph's size
     * returns to its value before the call, and no gradient slot is written. The updates run under {@code noGrad}, so
     * every returned tensor is a new tensor that does not require grad; the caller owns the result and closes it.
     * Parameters whose tensor does not require grad get no gradient.</p>
     *
     * @throws IllegalArgumentException when the shapes of {@code outputGrad} and {@code finalState} differ, they are
     *                                  on different backends, or {@code numSteps < 2}
     */
    public AdjointResult adjoint(Tensor outputGrad, Tensor finalState, float t0, float t1, int numSteps) {
        return integrateAdjoint(outputGrad, finalState, t0, t1, numSteps, true);
    }

    private AdjointResult integrateAdjoint(Tensor outputGrad, Tensor finalState, float t0, float t1, int numSteps,
                                           boolean withParameters) {
        ensureInitialized();
        Objects.requireNonNull(outputGrad, "outputGrad cannot be null");
        Objects.requireNonNull(finalState, "finalState cannot be null");
        if (!outputGrad.shape().equals(finalState.shape())) {
            throw new IllegalArgumentException("outputGrad " + outputGrad.shape() + " and finalState "
                + finalState.shape() + " must have the same shape");
        }
        if (outputGrad.backend() != finalState.backend()) {
            throw new IllegalArgumentException("outputGrad and finalState must be on the same backend");
        }
        float[] timeSpan = linspace(t1, t0, requireTimePoints(numSteps));
        List<Parameter> params = Collections.unmodifiableList(parameters());
        Tensor[] sums = new Tensor[params.size()];

        Tensor adjoint = outputGrad;
        Tensor state = finalState;
        try {
            for (int i = 0; i < timeSpan.length - 1; i++) {
                float t = timeSpan[i];
                float dt = timeSpan[i + 1] - t;
                try (VectorJacobian vjp = vectorJacobian(t, state, adjoint, params, withParameters);
                     ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    adjoint = replace(adjoint, addScaled(adjoint, vjp.state, -dt), outputGrad);
                    for (int k = 0; k < sums.length; k++) {
                        Tensor v = vjp.parameters[k];
                        if (v != null) {
                            sums[k] = sums[k] == null ? TensorOps.mul(v, -dt)
                                : replace(sums[k], addScaled(sums[k], v, -dt), null);
                        }
                    }
                    state = replace(state, addScaled(state, vjp.dynamics, dt), finalState);
                }
            }
            for (int k = 0; k < sums.length; k++) {
                Parameter p = params.get(k);
                if (sums[k] == null && withParameters && p.tensor().requiresGrad()) {
                    sums[k] = Tensor.zeros(p.backend(), p.shape());
                }
            }
            List<Tensor> gradients = new ArrayList<>(params.size());
            Collections.addAll(gradients, sums);
            return new AdjointResult(adjoint, params, gradients);
        } catch (RuntimeException e) {
            release(adjoint, outputGrad);
            for (Tensor sum : sums) {
                if (sum != null) {
                    sum.close();
                }
            }
            throw e;
        } finally {
            release(state, finalState);
        }
    }

    /**
     * {@code a^T df/dz} and, when {@code withParameters}, {@code a^T df/dtheta} for every parameter whose tensor
     * requires grad, at {@code f(t, z)} with {@code z} a fresh leaf over {@code state}'s values, together with
     * {@code f(t, z)} itself; computed in grad mode inside a step scope that leaves no node behind.
     */
    private VectorJacobian vectorJacobian(float t, Tensor state, Tensor adjoint, List<Parameter> params,
                                         boolean withParameters) {
        ComputationGraph graph = ComputationGraph.current();
        try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad();
             ComputationGraph.StepScope step = graph.step();
             Tensor z = freshLeaf(state);
             Tensor f = dynamics(t, z)) {
            List<Tensor> targets = new ArrayList<>(params.size() + 1);
            int[] positions = new int[params.size()];
            targets.add(z);
            for (int k = 0; k < params.size(); k++) {
                Tensor p = params.get(k).tensor();
                positions[k] = withParameters && p.requiresGrad() ? targets.size() : -1;
                if (positions[k] >= 0) {
                    targets.add(p);
                }
            }
            List<Tensor> grads = Autograd.grad(f, targets, adjoint, false, false);
            VectorJacobian result = new VectorJacobian(params.size());
            try {
                result.dynamics = f.detach();
                Tensor vz = grads.get(0);
                result.state = vz != null ? vz : Tensor.zeros(state.backend(), state.shape());
                for (int k = 0; k < positions.length; k++) {
                    result.parameters[k] = positions[k] >= 0 ? grads.get(positions[k]) : null;
                }
                return result;
            } catch (RuntimeException e) {
                result.close();
                for (Tensor g : grads) {
                    if (g != null) {
                        g.close();
                    }
                }
                throw e;
            }
        }
    }

    /** A new leaf that requires grad over {@code state}'s storage, with no path to {@code state}. */
    private static Tensor freshLeaf(Tensor state) {
        Tensor detached = state.detach();
        Tensor leaf = detached.withRequiresGrad(true);
        if (leaf != detached) {
            detached.close();
        }
        return leaf;
    }

    /** One adjoint step's vector-Jacobian products and dynamics, each owned by this object. */
    private static final class VectorJacobian implements AutoCloseable {
        Tensor dynamics;
        Tensor state;
        final Tensor[] parameters;

        VectorJacobian(int numParameters) {
            this.parameters = new Tensor[numParameters];
        }

        @Override
        public void close() {
            if (dynamics != null) {
                dynamics.close();
            }
            if (state != null) {
                state.close();
            }
            for (Tensor p : parameters) {
                if (p != null) {
                    p.close();
                }
            }
        }
    }

    /**
     * The result of {@link #adjoint}: the state adjoint at {@code t0} and the parameter gradients, in the order of
     * {@link #parameters()}, owned by the holder; {@link #close()} closes them all.
     */
    public static final class AdjointResult implements AutoCloseable {
        private final Tensor stateAdjoint;
        private final List<Parameter> parameters;
        private final List<Tensor> parameterGradients;

        private AdjointResult(Tensor stateAdjoint, List<Parameter> parameters, List<Tensor> parameterGradients) {
            this.stateAdjoint = stateAdjoint;
            this.parameters = parameters;
            this.parameterGradients = Collections.unmodifiableList(parameterGradients);
        }

        /** {@code dL/dz} at {@code t0}, shaped like the output gradient. */
        public Tensor stateAdjoint() {
            return stateAdjoint;
        }

        /** The module's parameters, in the order of {@link #parameterGradients()}. */
        public List<Parameter> parameters() {
            return parameters;
        }

        /** {@code dL/dtheta} per parameter, null for a parameter whose tensor does not require grad. */
        public List<Tensor> parameterGradients() {
            return parameterGradients;
        }

        /**
         * The gradient of {@code parameter} (by identity), null when its tensor does not require grad.
         *
         * @throws IllegalArgumentException when {@code parameter} is not one of the module's parameters
         */
        public Tensor gradientOf(Parameter parameter) {
            for (int k = 0; k < parameters.size(); k++) {
                if (parameters.get(k) == parameter) {
                    return parameterGradients.get(k);
                }
            }
            throw new IllegalArgumentException("not a parameter of this NeuralODE: " + parameter);
        }

        @Override
        public void close() {
            stateAdjoint.close();
            for (Tensor g : parameterGradients) {
                if (g != null) {
                    g.close();
                }
            }
        }
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("NeuralODE not initialized. Call initialize(backend) first.");
        }
    }

    public int inputDim() {
        return inputDim;
    }

    public int hiddenDim() {
        return hiddenDim;
    }

    public int numLayers() {
        return numLayers;
    }

    public Solver solver() {
        return solver;
    }

    /**
     * The configured relative tolerance. Every solver takes the fixed steps of the time grid, so it does not control
     * the step size.
     */
    public float rtol() {
        return rtol;
    }

    /**
     * The configured absolute tolerance. Every solver takes the fixed steps of the time grid, so it does not control
     * the step size.
     */
    public float atol() {
        return atol;
    }

    public List<Linear> layers() {
        return new ArrayList<>(layers);
    }

    @Override
    public String toString() {
        return "NeuralODE{" +
            "inputDim=" + inputDim +
            ", hiddenDim=" + hiddenDim +
            ", numLayers=" + numLayers +
            ", solver=" + solver +
            ", initialized=" + initialized +
            '}';
    }
}

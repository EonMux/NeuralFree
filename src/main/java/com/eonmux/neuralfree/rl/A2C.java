package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Advantage actor-critic over n-step rollouts.
 *
 * <p><b>Networks.</b> The actor maps a state to {@code actionSize} logits and the critic to one value. Each is the
 * perceptron {@code Linear(stateSize, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)} with
 * two hidden layers of width {@code hidden = hiddenSize} (the layout of Stable-Baselines3's default {@code net_arch},
 * with ReLU activations); the networks are initialized from the seed, the actor first (He-normal hidden layers, a
 * Glorot-normal output layer, zero biases).</p>
 *
 * <p>{@link #trainStep()} evaluates the whole rollout as one batch on the device: the actor and the
 * critic run once over the {@code [n, stateSize]} states, and the policy term {@code -log(max(p(a), 1e-10)) * adv},
 * the value term {@code (R - V(s))^2} with the discounted return {@code R} and the entropy {@code -sum(p log p)} over
 * {@code p > 0} combine into the per-step loss {@code policy + valueCoeff * value - entropyCoeff * entropy}; its mean
 * over the rollout is the loss of one optimizer step. The discounted returns are a sequential scan over the rewards the
 * Java environment supplied, bootstrapped from the critic, so they stay on the host; the advantages
 * {@code R - V_stored(s)} are normalized to zero mean and unit deviation on the device. Action sampling, the stored
 * value and the bootstrap value of the last next state are Java decisions and read back the {@code |A|}
 * probabilities or a scalar.</p>
 *
 * <p><b>Training.</b> The actor and the critic are trained together by one optimizer, built by the factory given to
 * the constructor over both networks' parameters. Each {@link #trainStep()} records the rollout loss, back-propagates
 * it and steps the optimizer once, inside a {@link ComputationGraph.StepScope}, so the graph is back
 * at its size before the step afterwards. The returns, the stored values and the normalized advantages are host data,
 * so the gradient reaches the actor only through {@code log p(a)} and the entropy, and the critic only through
 * {@code V(s)}. Without an optimizer the step evaluates the loss under {@link ComputationGraph#noGrad()} and updates
 * nothing. Acting, {@link #storeTransition}, {@link #getValue} and {@link #getActionProbs} run under
 * {@link ComputationGraph#noGrad()} and record nothing.</p>
 *
 * <p>Tolerances against the CPU reference ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): softmax,
 * log and the entropy terms are in the transcendental class, the advantage normalization and the loss mean in the
 * reduction class.</p>
 */
public final class A2C implements Policy<float[], Integer>, AutoCloseable {
    private static final float MIN_PROB = 1e-10f;
    private static final float NORMALIZE_EPS = 1e-8f;

    private final ComputeBackend backend;
    private final int stateSize;
    private final int actionSize;
    private final int hiddenSize;
    
    private final Sequential actor;
    private final Sequential critic;
    private final Optimizer optimizer;
    
    private final float gamma;
    private final float entropyCoeff;
    private final float valueCoeff;
    private final int nSteps;
    
    private final List<RolloutStep> rollout;
    private final Random random;
    private volatile boolean closed;
    
    /**
     * An agent whose actor and critic (see the class description) are initialized from {@code seed};
     * {@code optimizerFactory} builds the optimizer over the actor's and the critic's parameters (null for an agent
     * that evaluates without updating).
     *
     * @throws IllegalArgumentException when a size is not positive, or the factory returns null or an optimizer that
     *                                  does not cover the parameters it is given
     */
    public A2C(ComputeBackend backend, int stateSize, int actionSize, int hiddenSize,
               Function<List<Parameter>, ? extends Optimizer> optimizerFactory, float gamma, float entropyCoeff,
               float valueCoeff, int nSteps, long seed) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (stateSize <= 0) throw new IllegalArgumentException("State size must be positive");
        if (actionSize <= 0) throw new IllegalArgumentException("Action size must be positive");
        RlNetworks.requirePositiveHiddenSize(hiddenSize);
        
        this.backend = backend;
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.hiddenSize = hiddenSize;
        this.gamma = gamma;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.nSteps = nSteps;
        this.random = new Random(seed);
        this.rollout = new ArrayList<>();
        
        Sequential actorNetwork = null;
        Sequential criticNetwork = null;
        Optimizer built;
        try {
            actorNetwork = RlNetworks.mlp(backend, stateSize, hiddenSize, actionSize, random);
            criticNetwork = RlNetworks.mlp(backend, stateSize, hiddenSize, 1, random);
            built = RlTensorMath.buildOptimizer(optimizerFactory,
                    RlTensorMath.parametersOf(actorNetwork, criticNetwork), "A2C");
        } catch (RuntimeException e) {
            try {
                RlTensorMath.closeAll(actorNetwork, criticNetwork);
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        this.actor = actorNetwork;
        this.critic = criticNetwork;
        this.optimizer = built;
    }
    
    public static A2CBuilder builder() {
        return new A2CBuilder();
    }
    
    /**
     * Samples an action from the policy when {@code explore} (the first action whose running probability reaches a
     * uniform draw, the last action when rounding leaves none), otherwise the first most probable action. Runs under
     * {@link ComputationGraph#noGrad()}.
     */
    @Override
    public Integer selectAction(float[] state, boolean explore) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            float[] probs = actionProbabilities(state);
            if (explore) {
                return RlTensorMath.sampleIndex(probs, random.nextFloat(), actionSize - 1);
            }
            return RlTensorMath.firstMaxIndex(probs);
        }
    }
    
    /**
     * Records one step with the critic's value of {@code state}, computed under {@link ComputationGraph#noGrad()} and
     * read back as one scalar, since the rollout keeps it as a Java value. The states are copied; the next state
     * is kept only for a step that does not end its episode, as the bootstrap of a rollout that ends there.
     *
     * @throws IndexOutOfBoundsException when {@code action} is outside {@code [0, actionSize)}
     * @throws NullPointerException      when {@code state} is null, or {@code nextState} is null and {@code done} is
     *                                   false
     * @throws IllegalArgumentException  when such a state does not hold {@code stateSize} values
     */
    public void storeTransition(float[] state, int action, float reward, float[] nextState, boolean done) {
        checkNotClosed();
        float[] stateCopy = RlInputs.copyOfState(state, stateSize, "State");
        float[] nextStateCopy = done ? null : RlInputs.copyOfState(nextState, stateSize, "Next state");
        if (action < 0 || action >= actionSize) {
            throw new IndexOutOfBoundsException("Action " + action + " out of range [0, " + actionSize + ")");
        }
        
        float value;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            value = value(stateCopy);
        }
        
        rollout.add(new RolloutStep(stateCopy, action, reward, nextStateCopy, done, value));
    }
    
    /**
     * One update over the stored rollout, once it holds at least {@code nSteps} steps: the mean per-step loss is
     * back-propagated and the optimizer steps once (see the class description). The rollout is then cleared.
     *
     * @return the mean per-step loss {@code policy + valueCoeff * value - entropyCoeff * entropy} before the update,
     *         or 0 before the rollout is full
     * @throws IllegalStateException when the loss does not require grad (grad mode is disabled)
     */
    public float trainStep() {
        checkNotClosed();
        
        if (rollout.isEmpty() || rollout.size() < nSteps) {
            return 0.0f;
        }
        
        float[] returns = discountedReturns();
        float meanLoss;
        if (optimizer == null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor loss = rolloutLoss(returns)) {
                meanLoss = loss.toFloatArray()[0];
            }
        } else {
            try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                 Tensor loss = rolloutLoss(returns)) {
                meanLoss = RlTensorMath.optimizerStep(optimizer, loss, "A2C");
            }
        }
        
        rollout.clear();
        
        return meanLoss;
    }
    
    /** The mean per-step loss over the rollout, as a recorded scalar (see the class description). */
    private Tensor rolloutLoss(float[] returns) {
        int n = rollout.size();
        float[][] states = new float[n][];
        int[] actions = new int[n];
        float[] rawAdvantages = new float[n];
        for (int i = 0; i < n; i++) {
            RolloutStep step = rollout.get(i);
            states[i] = step.state;
            actions[i] = step.action;
            rawAdvantages[i] = returns[i] - step.value;
        }
        try (Tensor stateBatch = RlTensorMath.matrix(backend, states);
             Tensor valueTarget = Tensor.fromArray(backend, returns);
             Tensor actionIndex = RlTensorMath.columnIndex(backend, actions, actionSize);
             Tensor raw = Tensor.fromArray(backend, rawAdvantages);
             Tensor advantages = normalized(raw);
             Tensor logits = actor.forward(stateBatch);
             Tensor probs = TensorOps.softmax(logits, 1);
             Tensor chosen = RlTensorMath.pickColumns(probs, actionIndex);
             Tensor logProb = RlTensorMath.clampedLog(chosen, MIN_PROB);
             Tensor weighted = TensorOps.mul(logProb, advantages);
             Tensor policyLoss = TensorOps.neg(weighted);
             Tensor criticOut = critic.forward(stateBatch);
             Tensor newValue = criticOut.reshape(Shape.vector(n));
             Tensor valueError = TensorOps.sub(valueTarget, newValue);
             Tensor valueLoss = TensorOps.mul(valueError, valueError);
             Tensor entropy = RlTensorMath.rowEntropy(probs, 0.0f);
             Tensor scaledValueLoss = TensorOps.mul(valueLoss, valueCoeff);
             Tensor partial = TensorOps.add(policyLoss, scaledValueLoss);
             Tensor scaledEntropy = TensorOps.mul(entropy, entropyCoeff);
             Tensor loss = TensorOps.sub(partial, scaledEntropy)) {
            return TensorOps.mean(loss);
        }
    }
    
    /**
     * Discounted returns {@code R_i = r_i + gamma * R_{i+1}} (restarting at episode ends), by a backward scan over the
     * rollout (host data: environment rewards). A rollout that does not end its episode bootstraps from the critic's
     * value of the last next state, read back as one scalar.
     */
    private float[] discountedReturns() {
        int size = rollout.size();
        float[] returnsOut = new float[size];
        
        float lastValue = 0.0f;
        RolloutStep lastStep = rollout.get(size - 1);
        if (!lastStep.done) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                lastValue = value(lastStep.nextState);
            }
        }
        
        float returns = lastValue;
        for (int i = size - 1; i >= 0; i--) {
            RolloutStep step = rollout.get(i);
            
            if (step.done) {
                returns = step.reward;
            } else {
                returns = step.reward + gamma * returns;
            }
            
            returnsOut[i] = returns;
        }
        return returnsOut;
    }
    
    /** {@code (a - mean(a)) / sqrt(var(a) + 1e-8)} on the device, with the population variance (host data, no grad). */
    private static Tensor normalized(Tensor raw) {
        try (Tensor mean = TensorOps.mean(raw);
             Tensor centered = TensorOps.sub(raw, mean);
             Tensor squared = TensorOps.mul(centered, centered);
             Tensor variance = TensorOps.mean(squared);
             Tensor shifted = TensorOps.add(variance, NORMALIZE_EPS);
             Tensor std = TensorOps.sqrt(shifted)) {
            return TensorOps.div(centered, std);
        }
    }
    
    /**
     * The critic's value of {@code state}, read back as one scalar; runs under {@link ComputationGraph#noGrad()}.
     */
    public float getValue(float[] state) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return value(state);
        }
    }
    
    private float value(float[] state) {
        try (Tensor stateTensor = RlTensorMath.row(backend, state);
             Tensor valueTensor = critic.forward(stateTensor)) {
            return valueTensor.toFloatArray()[0];
        }
    }
    
    /**
     * The policy's action probabilities in {@code state} (the {@code |A|} values are read back); runs under
     * {@link ComputationGraph#noGrad()}.
     */
    public float[] getActionProbs(float[] state) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return actionProbabilities(state);
        }
    }
    
    private float[] actionProbabilities(float[] state) {
        try (Tensor stateTensor = RlTensorMath.row(backend, state);
             Tensor logits = actor.forward(stateTensor);
             Tensor probs = TensorOps.softmax(logits, 1)) {
            return probs.toFloatArray();
        }
    }
    
    public int rolloutSize() {
        return rollout.size();
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    @Override
    public void reset() {
        rollout.clear();
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("A2C has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RlTensorMath.closeAll(RlTensorMath.closer(optimizer), actor, critic);
        }
    }
    
    @Override
    public String toString() {
        return "A2C{stateSize=" + stateSize + ", actionSize=" + actionSize + 
               ", hiddenSize=" + hiddenSize + ", rolloutSize=" + rollout.size() + "}";
    }
    
    /** One stored step; {@code nextState} is null for a step that ends its episode. */
    private static final class RolloutStep {
        final float[] state;
        final int action;
        final float reward;
        final float[] nextState;
        final boolean done;
        final float value;
        
        RolloutStep(float[] state, int action, float reward, float[] nextState, boolean done, float value) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
            this.value = value;
        }
    }
    
    public static final class A2CBuilder {
        private ComputeBackend backend;
        private int stateSize;
        private int actionSize;
        private int hiddenSize = 64;
        private Function<List<Parameter>, ? extends Optimizer> optimizerFactory;
        private float gamma = 0.99f;
        private float entropyCoeff = 0.01f;
        private float valueCoeff = 0.5f;
        private int nSteps = 5;
        private long seed = System.currentTimeMillis();
        
        public A2CBuilder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public A2CBuilder stateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }
        
        public A2CBuilder actionSize(int actionSize) {
            this.actionSize = actionSize;
            return this;
        }
        
        /** The width of the two hidden layers of the actor and of the critic (default 64); must be positive. */
        public A2CBuilder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }
        
        /**
         * The factory of the optimizer that trains the actor and the critic: it receives their parameters (the
         * networks exist only inside the agent) and returns an optimizer over all of them, such as
         * {@code params -> new Adam(params, 7e-4f)}. Without one the agent evaluates without updating.
         */
        public A2CBuilder optimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.optimizerFactory = optimizerFactory;
            return this;
        }
        
        public A2CBuilder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }
        
        public A2CBuilder entropyCoeff(float entropyCoeff) {
            this.entropyCoeff = entropyCoeff;
            return this;
        }
        
        public A2CBuilder valueCoeff(float valueCoeff) {
            this.valueCoeff = valueCoeff;
            return this;
        }
        
        public A2CBuilder nSteps(int nSteps) {
            this.nSteps = nSteps;
            return this;
        }
        
        public A2CBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }
        
        public A2C build() {
            return new A2C(backend, stateSize, actionSize, hiddenSize, optimizerFactory,
                          gamma, entropyCoeff, valueCoeff, nSteps, seed);
        }
    }
}

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
 * Proximal policy optimization with generalized advantage estimation and a clipped surrogate objective.
 *
 * <p><b>Networks.</b> The actor maps a state to {@code actionSize} logits and the critic to one value. Each is the
 * perceptron {@code Linear(stateSize, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)} with
 * two hidden layers of width {@code hidden = hiddenSize} (the layout of Stable-Baselines3's default {@code net_arch},
 * with ReLU activations); the networks are initialized from the seed, the actor first (He-normal hidden layers, a
 * Glorot-normal output layer, zero biases).</p>
 *
 * <p>{@link #trainStep()} evaluates the whole trajectory as one batch on the device: the actor and
 * the critic run once over the {@code [n, stateSize]} states, and the ratio {@code exp(log(max(p(a), 1e-10)) -
 * oldLogProb)}, the clipped surrogate {@code -min(ratio * adv, clamp(ratio, 1 - eps, 1 + eps) * adv)}, the value
 * term {@code (return - V(s))^2} and the entropy {@code -sum(p log p)} over {@code p > 0} combine into one per-step
 * loss. The advantages are a backward scan of generalized advantage estimation over the rewards and values the caller
 * stored, all Java values, so they stay on the host and are uploaded once per update. A trajectory whose last step
 * does not end its episode is bootstrapped from the critic's value of that step's next state, as Stable-Baselines3
 * and CleanRL do. Action sampling, probabilities and values for the caller, and that bootstrap value, are Java
 * decisions and read back the {@code |A|} probabilities or a scalar.</p>
 *
 * <p><b>Training.</b> One optimizer, built by the factory given to the constructor over the actor's and the critic's
 * parameters, trains both networks. An update runs {@code epochs} passes over the trajectory; each pass records the
 * mean per-step loss with the current parameters, back-propagates it and steps the optimizer once, inside its own
 * {@link ComputationGraph.StepScope}, so later passes see the updated policy and the ratio moves away
 * from 1 as PPO intends. The minimum of the surrogate pair is a recorded {@code min}, which routes the gradient to the
 * first smaller term; the clamp passes the gradient only inside {@code [1 - eps, 1 + eps]}. The advantages, returns and
 * old log-probabilities are host data, so the gradient reaches the actor through the ratio and the entropy and the
 * critic through {@code V(s)}. Without an optimizer the update evaluates the loss once under
 * {@link ComputationGraph#noGrad()} and changes nothing. Acting and the getters run under {@code noGrad} and record
 * nothing.</p>
 *
 * <p>Tolerances against the CPU reference ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): softmax,
 * log, exp and the entropy terms are in the transcendental class, the clipping and the minimum are exact on their
 * inputs, and the loss mean is in the reduction class.</p>
 */
public final class PPO implements Policy<float[], Integer>, AutoCloseable {
    private static final float MIN_PROB = 1e-10f;

    private final ComputeBackend backend;
    private final int stateSize;
    private final int actionSize;
    private final int hiddenSize;
    
    private final Sequential actor;
    private final Sequential critic;
    private final Optimizer optimizer;
    
    private final float gamma;
    private final float lambda;
    private final float clipEpsilon;
    private final float entropyCoeff;
    private final float valueCoeff;
    private final int epochs;
    private final int batchSize;
    
    private final List<TrajectoryStep> trajectory;
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
    public PPO(ComputeBackend backend, int stateSize, int actionSize, int hiddenSize,
               Function<List<Parameter>, ? extends Optimizer> optimizerFactory, float gamma, float lambda,
               float clipEpsilon, float entropyCoeff, float valueCoeff, int epochs, int batchSize, long seed) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (stateSize <= 0) throw new IllegalArgumentException("State size must be positive");
        if (actionSize <= 0) throw new IllegalArgumentException("Action size must be positive");
        RlNetworks.requirePositiveHiddenSize(hiddenSize);
        
        this.backend = backend;
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.hiddenSize = hiddenSize;
        this.gamma = gamma;
        this.lambda = lambda;
        this.clipEpsilon = clipEpsilon;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.random = new Random(seed);
        this.trajectory = new ArrayList<>();
        
        Sequential actorNetwork = null;
        Sequential criticNetwork = null;
        Optimizer built;
        try {
            actorNetwork = RlNetworks.mlp(backend, stateSize, hiddenSize, actionSize, random);
            criticNetwork = RlNetworks.mlp(backend, stateSize, hiddenSize, 1, random);
            built = RlTensorMath.buildOptimizer(optimizerFactory,
                    RlTensorMath.parametersOf(actorNetwork, criticNetwork), "PPO");
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
    
    public static PPOBuilder builder() {
        return new PPOBuilder();
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
     * Records one step with the log-probability and value the caller observed when acting. The states are copied; the
     * next state is kept only for a step that does not end its episode, as the bootstrap of a trajectory that ends
     * there.
     *
     * @throws IndexOutOfBoundsException when {@code action} is outside {@code [0, actionSize)}
     * @throws NullPointerException      when {@code state} is null, or {@code nextState} is null and {@code done} is
     *                                   false
     * @throws IllegalArgumentException  when such a state does not hold {@code stateSize} values
     */
    public void storeTransition(float[] state, int action, float reward, float[] nextState, boolean done,
                                float logProb, float value) {
        checkNotClosed();
        float[] stateCopy = RlInputs.copyOfState(state, stateSize, "State");
        float[] nextStateCopy = done ? null : RlInputs.copyOfState(nextState, stateSize, "Next state");
        if (action < 0 || action >= actionSize) {
            throw new IndexOutOfBoundsException("Action " + action + " out of range [0, " + actionSize + ")");
        }
        trajectory.add(new TrajectoryStep(stateCopy, action, reward, nextStateCopy, done, logProb, value));
    }
    
    /**
     * One update over the stored trajectory, once it holds at least {@code batchSize} steps: {@code epochs} passes,
     * each one optimizer step on that pass's mean per-step loss (see the class description). The trajectory is then
     * cleared.
     *
     * @return the mean per-step loss averaged over the passes, each pass's value taken before its step; 0 before the
     *         trajectory is full or when {@code epochs} is not positive
     * @throws IllegalStateException when the loss does not require grad (grad mode is disabled)
     */
    public float trainStep() {
        checkNotClosed();
        
        if (trajectory.isEmpty() || trajectory.size() < batchSize) {
            return 0.0f;
        }
        
        float meanLoss = epochs > 0 ? update() : 0.0f;
        
        trajectory.clear();
        
        return meanLoss;
    }
    
    /**
     * The {@code epochs} passes of {@link #trainStep()}: the host data is uploaded once, then each pass records its
     * loss, reads it back (one scalar) and steps; returns the mean of the pass values.
     */
    private float update() {
        int n = trajectory.size();
        float[] advantages = new float[n];
        float[] returns = new float[n];
        computeReturnsAndAdvantages(advantages, returns);
        float[][] states = new float[n][];
        int[] actions = new int[n];
        float[] oldLogProbs = new float[n];
        for (int i = 0; i < n; i++) {
            TrajectoryStep step = trajectory.get(i);
            states[i] = step.state;
            actions[i] = step.action;
            oldLogProbs[i] = step.logProb;
        }
        try (Tensor stateBatch = RlTensorMath.matrix(backend, states);
             Tensor advantage = Tensor.fromArray(backend, advantages);
             Tensor target = Tensor.fromArray(backend, returns);
             Tensor oldLogProb = Tensor.fromArray(backend, oldLogProbs);
             Tensor actionIndex = RlTensorMath.columnIndex(backend, actions, actionSize)) {
            Batch batch = new Batch(stateBatch, advantage, target, oldLogProb, actionIndex);
            if (optimizer == null) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                     Tensor loss = trajectoryLoss(batch)) {
                    return loss.toFloatArray()[0];
                }
            }
            float sum = 0.0f;
            for (int epoch = 0; epoch < epochs; epoch++) {
                try (ComputationGraph.StepScope step = ComputationGraph.current().step();
                     Tensor loss = trajectoryLoss(batch)) {
                    sum += RlTensorMath.optimizerStep(optimizer, loss, "PPO");
                }
            }
            return sum / epochs;
        }
    }
    
    /** The mean per-step loss over the trajectory with the current parameters, as a recorded scalar. */
    private Tensor trajectoryLoss(Batch batch) {
        int n = trajectory.size();
        try (Tensor logits = actor.forward(batch.states);
             Tensor probs = TensorOps.softmax(logits, 1);
             Tensor chosen = RlTensorMath.pickColumns(probs, batch.actionIndex);
             Tensor newLogProb = RlTensorMath.clampedLog(chosen, MIN_PROB);
             Tensor logRatio = TensorOps.sub(newLogProb, batch.oldLogProb);
             Tensor ratio = TensorOps.exp(logRatio);
             Tensor clipped = TensorOps.clamp(ratio, 1.0f - clipEpsilon, 1.0f + clipEpsilon);
             Tensor surrogate = TensorOps.mul(ratio, batch.advantage);
             Tensor clippedSurrogate = TensorOps.mul(clipped, batch.advantage);
             Tensor pessimistic = RlTensorMath.pairMin(surrogate, clippedSurrogate);
             Tensor policyLoss = TensorOps.neg(pessimistic);
             Tensor criticOut = critic.forward(batch.states);
             Tensor newValue = criticOut.reshape(Shape.vector(n));
             Tensor valueError = TensorOps.sub(batch.target, newValue);
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
     * Generalized advantage estimation by a backward scan over the stored steps (host data); {@code returns[i] =
     * advantages[i] + value[i]}. When the last step does not end its episode, the scan starts from the critic's value
     * of its next state, evaluated under {@link ComputationGraph#noGrad()} and read back as one scalar.
     */
    private void computeReturnsAndAdvantages(float[] advantages, float[] returns) {
        TrajectoryStep lastStep = trajectory.get(trajectory.size() - 1);
        float lastValue = 0.0f;
        if (!lastStep.done) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                lastValue = value(lastStep.nextState);
            }
        }
        float lastAdvantage = 0.0f;
        
        for (int i = trajectory.size() - 1; i >= 0; i--) {
            TrajectoryStep step = trajectory.get(i);
            
            float delta = step.reward + (step.done ? 0.0f : gamma * lastValue) - step.value;
            float advantage = delta + gamma * lambda * (step.done ? 0.0f : lastAdvantage);
            
            advantages[i] = advantage;
            returns[i] = advantage + step.value;
            
            lastValue = step.value;
            lastAdvantage = advantage;
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
    
    public int trajectorySize() {
        return trajectory.size();
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    @Override
    public void reset() {
        trajectory.clear();
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("PPO has been closed");
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
        return "PPO{stateSize=" + stateSize + ", actionSize=" + actionSize + 
               ", hiddenSize=" + hiddenSize + ", trajectorySize=" + trajectory.size() + "}";
    }
    
    /** The host data of one update, uploaded once and shared by its passes (none of it requires grad). */
    private static final class Batch {
        final Tensor states;
        final Tensor advantage;
        final Tensor target;
        final Tensor oldLogProb;
        final Tensor actionIndex;
        
        Batch(Tensor states, Tensor advantage, Tensor target, Tensor oldLogProb, Tensor actionIndex) {
            this.states = states;
            this.advantage = advantage;
            this.target = target;
            this.oldLogProb = oldLogProb;
            this.actionIndex = actionIndex;
        }
    }
    
    /** One stored step; {@code nextState} is null for a step that ends its episode. */
    private static final class TrajectoryStep {
        final float[] state;
        final int action;
        final float reward;
        final float[] nextState;
        final boolean done;
        final float logProb;
        final float value;
        
        TrajectoryStep(float[] state, int action, float reward, float[] nextState, 
                       boolean done, float logProb, float value) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
            this.logProb = logProb;
            this.value = value;
        }
    }
    
    public static final class PPOBuilder {
        private ComputeBackend backend;
        private int stateSize;
        private int actionSize;
        private int hiddenSize = 64;
        private Function<List<Parameter>, ? extends Optimizer> optimizerFactory;
        private float gamma = 0.99f;
        private float lambda = 0.95f;
        private float clipEpsilon = 0.2f;
        private float entropyCoeff = 0.01f;
        private float valueCoeff = 0.5f;
        private int epochs = 4;
        private int batchSize = 64;
        private long seed = System.currentTimeMillis();
        
        public PPOBuilder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public PPOBuilder stateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }
        
        public PPOBuilder actionSize(int actionSize) {
            this.actionSize = actionSize;
            return this;
        }
        
        /** The width of the two hidden layers of the actor and of the critic (default 64); must be positive. */
        public PPOBuilder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }
        
        /**
         * The factory of the optimizer that trains the actor and the critic: it receives their parameters (the
         * networks exist only inside the agent) and returns an optimizer over all of them, such as
         * {@code params -> new Adam(params, 3e-4f)}. Without one the agent evaluates without updating.
         */
        public PPOBuilder optimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.optimizerFactory = optimizerFactory;
            return this;
        }
        
        public PPOBuilder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }
        
        public PPOBuilder lambda(float lambda) {
            this.lambda = lambda;
            return this;
        }
        
        public PPOBuilder clipEpsilon(float clipEpsilon) {
            this.clipEpsilon = clipEpsilon;
            return this;
        }
        
        public PPOBuilder entropyCoeff(float entropyCoeff) {
            this.entropyCoeff = entropyCoeff;
            return this;
        }
        
        public PPOBuilder valueCoeff(float valueCoeff) {
            this.valueCoeff = valueCoeff;
            return this;
        }
        
        public PPOBuilder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }
        
        public PPOBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public PPOBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }
        
        public PPO build() {
            return new PPO(backend, stateSize, actionSize, hiddenSize, optimizerFactory,
                          gamma, lambda, clipEpsilon, entropyCoeff, valueCoeff, epochs, batchSize, seed);
        }
    }
}

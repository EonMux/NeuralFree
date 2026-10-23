package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Deep Q-learning with a target network and epsilon-greedy exploration.
 *
 * <p><b>Networks.</b> The Q-network maps a state to the {@code actionSize} action values through the perceptron
 * {@code Linear(stateSize, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, actionSize)} with two
 * hidden layers of width {@code hidden = hiddenSize}, as Stable-Baselines3 builds it, initialized from the seed
 * (He-normal hidden layers, a Glorot-normal output layer, zero biases). The target network has the same
 * architecture.</p>
 *
 * <p>{@link #trainStep()} evaluates a sampled replay batch as one batch on the device: the Q-network
 * and the target network each run once over the {@code [batch, stateSize]} states, the taken actions' values are
 * gathered, the bootstrap is the target network's row maximum (NaN entries skipped, as a scan from {@code -inf}
 * that keeps strictly greater values does) and the loss is the mean of the importance-weighted squared TD errors.
 * Only that mean is read back, as the logged loss, plus the batch's TD errors when the buffer uses prioritized replay
 * (they become the sampled transitions' new priorities, and the buffer's {@code beta} is annealed once per update).
 * Action selection reads back the {@code |A|} Q-values of one state and picks the first strict maximum on the host;
 * exploration draws come from the seeded {@link Random}.</p>
 *
 * <p><b>Training.</b> The optimizer, built by the factory given to the constructor over the Q-network's parameters,
 * steps once per {@link #trainStep()} after the loss is back-propagated, inside a
 * {@link ComputationGraph.StepScope}. The target network starts as a copy of the Q-network, is frozen
 * ({@link com.eonmux.neuralfree.nn.Module#setRequiresGrad(boolean)}) and is evaluated under
 * {@link ComputationGraph#noGrad()}, so the TD target is a constant and the gradient reaches the Q-network only
 * through {@code Q(s, a)}; the target synchronization runs under {@code noGrad} too. Without an optimizer the
 * step evaluates the loss under {@code noGrad} and updates nothing. Acting and the Q-value getters run under
 * {@code noGrad} and record nothing.</p>
 *
 * <p>Tolerances against the CPU reference ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): the
 * Q-values are GEMM results (reduction class); the TD targets and errors add single roundings to them; the loss is a
 * reduction over the batch.</p>
 */
public final class DQN implements Policy<float[], Integer>, AutoCloseable {
    private final ComputeBackend backend;
    private final int stateSize;
    private final int actionSize;
    private final int hiddenSize;
    
    private final Sequential qNetwork;
    private final Sequential targetNetwork;
    private final Optimizer optimizer;
    private final ReplayBuffer replayBuffer;
    
    private final float gamma;
    private final float epsilonStart;
    private final float epsilonEnd;
    private final float epsilonDecay;
    private final int targetUpdateFrequency;
    private final int batchSize;
    
    private float epsilon;
    private int stepCount;
    private int updateCount;
    private final Random random;
    private volatile boolean closed;
    
    /**
     * An agent with a Q-network (see the class description) initialized from {@code seed} and a frozen target network
     * that starts as a copy of it; {@code optimizerFactory} builds the optimizer over the Q-network's parameters (null
     * for an agent that evaluates without updating).
     *
     * @throws IllegalArgumentException when a size or frequency is not positive, or the factory returns null or an
     *                                  optimizer that does not cover the parameters it is given
     */
    public DQN(ComputeBackend backend, int stateSize, int actionSize, int hiddenSize,
               Function<List<Parameter>, ? extends Optimizer> optimizerFactory, ReplayBuffer replayBuffer,
               float gamma, float epsilonStart, float epsilonEnd, float epsilonDecay,
               int targetUpdateFrequency, int batchSize, long seed) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (stateSize <= 0) throw new IllegalArgumentException("State size must be positive");
        if (actionSize <= 0) throw new IllegalArgumentException("Action size must be positive");
        RlNetworks.requirePositiveHiddenSize(hiddenSize);
        if (batchSize <= 0) throw new IllegalArgumentException("Batch size must be positive: " + batchSize);
        if (targetUpdateFrequency <= 0) {
            throw new IllegalArgumentException("Target update frequency must be positive: " + targetUpdateFrequency);
        }
        
        this.backend = backend;
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.hiddenSize = hiddenSize;
        this.replayBuffer = replayBuffer;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonDecay = epsilonDecay;
        this.targetUpdateFrequency = targetUpdateFrequency;
        this.batchSize = batchSize;
        this.random = new Random(seed);
        this.epsilon = epsilonStart;
        this.stepCount = 0;
        this.updateCount = 0;
        
        Sequential online = null;
        Sequential target = null;
        Optimizer built;
        try {
            online = RlNetworks.mlp(backend, stateSize, hiddenSize, actionSize, random);
            target = RlNetworks.mlp(backend, stateSize, hiddenSize, actionSize, random);
            target.setRequiresGrad(false);
            RlTensorMath.copyParameters(online, target);
            built = RlTensorMath.buildOptimizer(optimizerFactory, online.parameters(), "DQN");
        } catch (RuntimeException e) {
            try {
                RlTensorMath.closeAll(online, target);
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        this.qNetwork = online;
        this.targetNetwork = target;
        this.optimizer = built;
    }
    
    public static DQNBuilder builder() {
        return new DQNBuilder();
    }
    
    /** Epsilon-greedy over the Q-network's values; runs under {@link ComputationGraph#noGrad()}. */
    @Override
    public Integer selectAction(float[] state, boolean explore) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        
        if (explore && random.nextFloat() < epsilon) {
            return random.nextInt(actionSize);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return RlTensorMath.firstMaxIndex(qValues(state));
        }
    }
    
    /**
     * One update over a replay batch: {@code target = reward + (done ? 0 : gamma * max_a Q_target(s', a))} and the
     * loss {@code mean(weight * (target - Q(s, action))^2)}, back-propagated into the Q-network before one optimizer
     * step (see the class description). Every {@code targetUpdateFrequency} updates the target network takes the
     * Q-network's parameters.
     *
     * @return the mean weighted squared TD error before the update, or 0 while the buffer holds fewer than
     *         {@code batchSize} transitions
     * @throws IllegalStateException when the loss does not require grad (grad mode is disabled)
     */
    public float trainStep() {
        checkNotClosed();
        Objects.requireNonNull(replayBuffer, "DQN was built without a replay buffer");
        
        if (replayBuffer.size() < batchSize) {
            return 0.0f;
        }
        
        ReplayBuffer.SampleBatch batch = replayBuffer.sample(batchSize, random);
        float meanLoss;
        if (optimizer == null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                meanLoss = evaluate(batch, null);
            }
        } else {
            try (ComputationGraph.StepScope step = ComputationGraph.current().step()) {
                meanLoss = evaluate(batch, optimizer);
            }
        }
        
        updateCount++;
        if (updateCount % targetUpdateFrequency == 0) {
            RlTensorMath.copyParameters(qNetwork, targetNetwork);
        }
        
        decayEpsilon();
        stepCount++;
        
        return meanLoss;
    }
    
    /**
     * The batch loss of {@link #trainStep()}, read back with the TD errors under prioritized replay, which then
     * become the sampled transitions' priorities; with an {@code optimizer} the loss is then back-propagated and the
     * optimizer steps. A loss that cannot be back-propagated is refused before the buffer's priorities and
     * {@code beta} change.
     */
    private float evaluate(ReplayBuffer.SampleBatch batch, Optimizer optimizer) {
        int n = batch.batchSize();
        try (Tensor states = RlTensorMath.matrix(backend, batch.states());
             Tensor nextStates = RlTensorMath.matrix(backend, batch.nextStates());
             Tensor rewards = Tensor.fromArray(backend, batch.rewards());
             Tensor weights = Tensor.fromArray(backend, batch.importanceWeights());
             Tensor live = RlTensorMath.liveMask(backend, batch.dones());
             Tensor actions = RlTensorMath.columnIndex(backend, batch.actions(), actionSize);
             Tensor target = tdTarget(nextStates, rewards, live);
             Tensor q = qNetwork.forward(states);
             Tensor currentQ = RlTensorMath.pickColumns(q, actions);
             Tensor tdError = TensorOps.sub(target, currentQ);
             Tensor squared = TensorOps.mul(tdError, tdError);
             Tensor weighted = TensorOps.mul(squared, weights);
             Tensor loss = TensorOps.mean(weighted)) {
            if (optimizer != null) {
                RlTensorMath.requireTrainable(loss, "DQN");
            }
            float meanLoss;
            if (replayBuffer.usesPrioritizedReplay()) {
                float[] errorsAndLoss = errorsAndLoss(tdError, loss);
                meanLoss = errorsAndLoss[n];
                replayBuffer.updatePriorities(batch.indices(), Arrays.copyOf(errorsAndLoss, n));
                replayBuffer.incrementBeta();
            } else {
                meanLoss = loss.toFloatArray()[0];
            }
            if (optimizer != null) {
                RlTensorMath.backwardAndStep(optimizer, loss, "DQN");
            }
            return meanLoss;
        }
    }
    
    /** {@code reward + (live ? gamma * rowMax(Q_target(s')) : 0)}, computed under {@link ComputationGraph#noGrad()}. */
    private Tensor tdTarget(Tensor nextStates, Tensor rewards, Tensor live) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor nextQ = targetNetwork.forward(nextStates);
             Tensor maxNextQ = RlTensorMath.rowMaxSkippingNaN(nextQ);
             Tensor discounted = TensorOps.mul(maxNextQ, gamma);
             Tensor bootstrap = RlTensorMath.whereLive(live, discounted)) {
            return TensorOps.add(rewards, bootstrap);
        }
    }
    
    /** The {@code n} TD errors followed by the loss, read back together under {@link ComputationGraph#noGrad()}. */
    private static float[] errorsAndLoss(Tensor tdError, Tensor loss) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor lossRow = loss.reshape(Shape.vector(1));
             Tensor both = TensorOps.cat(Arrays.asList(tdError, lossRow), 0)) {
            return both.toFloatArray();
        }
    }
    
    private void decayEpsilon() {
        epsilon = Math.max(epsilonEnd, epsilon * epsilonDecay);
    }
    
    /**
     * The Q-network's value of {@code action} in {@code state}; reads back that one value and runs under
     * {@link ComputationGraph#noGrad()}.
     *
     * @throws IndexOutOfBoundsException when {@code action} is outside {@code [0, actionSpaceSize())}
     */
    public float getQValue(float[] state, int action) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        if (action < 0 || action >= actionSize) {
            throw new IndexOutOfBoundsException("Action " + action + " out of range [0, " + actionSize + ")");
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor stateTensor = RlTensorMath.row(backend, state);
             Tensor q = qNetwork.forward(stateTensor);
             Tensor chosen = q.slice(1, action, action + 1L)) {
            return chosen.toFloatArray()[0];
        }
    }
    
    /**
     * The Q-network's values of every action in {@code state} (the {@code |A|} values are read back); runs under
     * {@link ComputationGraph#noGrad()}.
     */
    public float[] getAllQValues(float[] state) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return qValues(state);
        }
    }
    
    private float[] qValues(float[] state) {
        try (Tensor stateTensor = RlTensorMath.row(backend, state);
             Tensor q = qNetwork.forward(stateTensor)) {
            return q.toFloatArray();
        }
    }
    
    public float epsilon() {
        return epsilon;
    }
    
    public int stepCount() {
        return stepCount;
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    public int actionSpaceSize() {
        return actionSize;
    }
    
    @Override
    public void reset() {
        epsilon = epsilonStart;
        stepCount = 0;
        updateCount = 0;
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("DQN has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RlTensorMath.closeAll(RlTensorMath.closer(optimizer), qNetwork, targetNetwork);
        }
    }
    
    @Override
    public String toString() {
        return "DQN{stateSize=" + stateSize + ", actionSize=" + actionSize + 
               ", hiddenSize=" + hiddenSize + ", epsilon=" + epsilon + "}";
    }
    
    public static final class DQNBuilder {
        private ComputeBackend backend;
        private int stateSize;
        private int actionSize;
        private int hiddenSize = 128;
        private Function<List<Parameter>, ? extends Optimizer> optimizerFactory;
        private ReplayBuffer replayBuffer;
        private float gamma = 0.99f;
        private float epsilonStart = 1.0f;
        private float epsilonEnd = 0.01f;
        private float epsilonDecay = 0.995f;
        private int targetUpdateFrequency = 100;
        private int batchSize = 32;
        private long seed = System.currentTimeMillis();
        
        public DQNBuilder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public DQNBuilder stateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }
        
        public DQNBuilder actionSize(int actionSize) {
            this.actionSize = actionSize;
            return this;
        }
        
        /** The width of the two hidden layers of the Q-network and its target (default 128); must be positive. */
        public DQNBuilder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }
        
        /**
         * The factory of the optimizer that trains the Q-network: it receives the Q-network's parameters (the network
         * exists only inside the agent) and returns an optimizer over all of them, such as
         * {@code params -> new Adam(params, 1e-4f)}. Without one the agent evaluates without updating.
         */
        public DQNBuilder optimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.optimizerFactory = optimizerFactory;
            return this;
        }
        
        public DQNBuilder replayBuffer(ReplayBuffer replayBuffer) {
            this.replayBuffer = replayBuffer;
            return this;
        }
        
        public DQNBuilder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }
        
        public DQNBuilder epsilonStart(float epsilonStart) {
            this.epsilonStart = epsilonStart;
            return this;
        }
        
        public DQNBuilder epsilonEnd(float epsilonEnd) {
            this.epsilonEnd = epsilonEnd;
            return this;
        }
        
        public DQNBuilder epsilonDecay(float epsilonDecay) {
            this.epsilonDecay = epsilonDecay;
            return this;
        }
        
        public DQNBuilder targetUpdateFrequency(int targetUpdateFrequency) {
            this.targetUpdateFrequency = targetUpdateFrequency;
            return this;
        }
        
        public DQNBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public DQNBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }
        
        public DQN build() {
            return new DQN(backend, stateSize, actionSize, hiddenSize, optimizerFactory, replayBuffer,
                          gamma, epsilonStart, epsilonEnd, epsilonDecay, targetUpdateFrequency, batchSize, seed);
        }
    }
}

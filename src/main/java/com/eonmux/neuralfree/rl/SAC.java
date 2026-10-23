package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Soft actor-critic (Haarnoja et al. 2018, with the automatic entropy tuning of the follow-up paper) with twin
 * critics, Polyak-averaged target critics and a tanh-squashed Gaussian policy.
 *
 * <p><b>Networks.</b> Every network is a perceptron with two hidden layers of width {@code hidden = hiddenSize}, as
 * Stable-Baselines3 builds them: {@code Linear(in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden,
 * out)}, with He-normal hidden layers, a Glorot-normal output layer and zero biases. The actor ({@code in = stateSize})
 * ends in one output layer of {@code 2 actionSize} units on its hidden trunk: the mean head is its first
 * {@code actionSize} outputs and the log-standard-deviation head the rest. Each critic takes the concatenation
 * {@code (state, action)} ({@code in = stateSize + actionSize}) to one value; each target critic has its critic's
 * architecture and starts as a copy of it. The networks are initialized from the seed in the order actor, first
 * critic, second critic, first target, second target.</p>
 *
 * <p>The actor maps a state to {@code actionSize} means {@code mu} and {@code actionSize} log standard deviations
 * (clamped to {@code [-20, 2]}, {@code sigma = exp(logStd)}). A policy sample is the reparameterized
 * {@code a = tanh(u)}, {@code u = mu + sigma * eps}, with standard-normal {@code eps} drawn from the seeded
 * {@link Random} on the host and uploaded; a greedy action is {@code tanh(mu)}. Its log-density
 * is the Gaussian density of {@code u} with the tanh change of variables,
 * {@code log pi(a|s) = sum_j [-eps_j^2 / 2 - log(2 pi) / 2 - logStd_j - log(1 - tanh(u_j)^2)]}, where
 * {@code log(1 - tanh(u)^2) = 2 (log 2 - |u| - softplus(-2 |u|))} is evaluated in that overflow-free form.</p>
 *
 * <p>{@link #trainStep()} samples one replay batch and runs, in order, each inside its own
 * {@link ComputationGraph.StepScope}:</p>
 * <ol>
 *   <li>the critic update: the TD target {@code y = r + gamma (1 - d) (min(Q1', Q2')(s', a') - alpha log pi(a'|s'))}
 *   with {@code a'} sampled from the current policy is computed under {@link ComputationGraph#noGrad()} (a constant),
 *   and the loss {@code 0.5 mean((Q1(s, a) - y)^2) + 0.5 mean((Q2(s, a) - y)^2)} trains both critics;</li>
 *   <li>the actor update: with the critics frozen ({@link com.eonmux.neuralfree.nn.Module#setRequiresGrad(boolean)},
 *   restored in a {@code finally}), the loss {@code mean(alpha log pi(a~|s) - min(Q1, Q2)(s, a~))} with the
 *   reparameterized {@code a~} trains the actor through both the log-density and the critics' inputs, and deposits
 *   nothing in the critics;</li>
 *   <li>with {@code autoEntropy}, the temperature update: {@code log alpha} is a parameter trained on
 *   {@code -mean(log alpha (log pi(a~|s) + targetEntropy))} with the actor step's {@code log pi} detached, so this loss
 *   never trains the actor; {@code targetEntropy = -actionSize}. Without an actor optimizer the same sample's
 *   {@code log pi} is computed under {@code noGrad} for it.</li>
 * </ol>
 * <p>All three use the temperature {@code alpha} of the start of the step ({@code exp(log alpha)} under
 * {@code noGrad}, or the fixed {@code entropyAlpha} without {@code autoEntropy}, where {@code entropyAlpha} is also
 * the initial temperature with it). Both target critics are frozen copies of the critics that follow them by
 * {@code target = tau * critic + (1 - tau) * target} under {@code noGrad} after every step. The objective is the
 * standard SAC one, with the policy's own log-density as the entropy term.</p>
 *
 * <p>The replay buffer stores the continuous actions as given ({@link ReplayBuffer#add(float[], float[], float,
 * float[], boolean)}), and the critics are trained on exactly those actions. Each update has its own optimizer, built
 * by its factory over its parameters (the critic factory over both critics, the actor factory over the actor, the
 * temperature factory, or the actor factory when none is given, over {@code log alpha}); an update without an
 * optimizer is skipped, and without a critic optimizer the critic loss is only evaluated, under {@code noGrad}.
 * Acting and the Q-value getters run under {@code noGrad} and record nothing. The returned loss and the actions
 * handed to the caller are read back.</p>
 *
 * <p>Tolerances against the CPU reference ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}): the
 * network outputs are GEMM results (reduction class), the squashing, exponentials and log-densities are in the
 * transcendental class, the minima are exact on their inputs, and the losses are reductions over the batch.</p>
 */
public final class SAC implements Policy<float[], float[]>, AutoCloseable {
    private static final float LOG_STD_MIN = -20.0f;
    private static final float LOG_STD_MAX = 2.0f;
    private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2.0 * Math.PI));
    private static final float LOG_TWO = (float) Math.log(2.0);

    private final ComputeBackend backend;
    private final int stateSize;
    private final int actionSize;
    private final int hiddenSize;
    
    private final Sequential actor;
    private final Sequential q1;
    private final Sequential q2;
    private final Sequential q1Target;
    private final Sequential q2Target;
    private final Parameter logAlpha;
    private final Optimizer actorOptimizer;
    private final Optimizer criticOptimizer;
    private final Optimizer alphaOptimizer;
    
    private final float gamma;
    private final float tau;
    private final float entropyAlpha;
    private final boolean autoEntropy;
    private final float targetEntropy;
    private final int batchSize;
    
    private final ReplayBuffer replayBuffer;
    private final Random random;
    private volatile boolean closed;
    
    /**
     * An agent whose networks are initialized from {@code seed}, with the temperature fixed at {@code entropyAlpha}, or
     * starting there and tuned when {@code autoEntropy}; the temperature optimizer is the actor factory's.
     *
     * @see #SAC(ComputeBackend, int, int, int, Function, Function, Function, ReplayBuffer, float, float, float,
     *      boolean, int, long)
     */
    public SAC(ComputeBackend backend, int stateSize, int actionSize, int hiddenSize,
               Function<List<Parameter>, ? extends Optimizer> actorOptimizerFactory,
               Function<List<Parameter>, ? extends Optimizer> criticOptimizerFactory, ReplayBuffer replayBuffer,
               float gamma, float tau, float entropyAlpha, boolean autoEntropy,
               int batchSize, long seed) {
        this(backend, stateSize, actionSize, hiddenSize, actorOptimizerFactory, criticOptimizerFactory, null,
                replayBuffer, gamma, tau, entropyAlpha, autoEntropy, batchSize, seed);
    }
    
    /**
     * An agent whose networks are initialized from {@code seed}. Each factory builds the optimizer of one update over
     * the parameters it is given (see the class description); {@code alphaOptimizerFactory} is used only with
     * {@code autoEntropy}, and the actor factory takes its place when it is null.
     *
     * @throws IllegalArgumentException when a size is not positive, {@code tau} is outside {@code [0, 1]},
     *                                  {@code entropyAlpha} is negative (not positive with {@code autoEntropy}) or not
     *                                  finite, or a factory returns null or an optimizer that does not cover the
     *                                  parameters it is given
     */
    public SAC(ComputeBackend backend, int stateSize, int actionSize, int hiddenSize,
               Function<List<Parameter>, ? extends Optimizer> actorOptimizerFactory,
               Function<List<Parameter>, ? extends Optimizer> criticOptimizerFactory,
               Function<List<Parameter>, ? extends Optimizer> alphaOptimizerFactory, ReplayBuffer replayBuffer,
               float gamma, float tau, float entropyAlpha, boolean autoEntropy,
               int batchSize, long seed) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        if (stateSize <= 0) throw new IllegalArgumentException("State size must be positive");
        if (actionSize <= 0) throw new IllegalArgumentException("Action size must be positive");
        RlNetworks.requirePositiveHiddenSize(hiddenSize);
        if (!(tau >= 0.0f && tau <= 1.0f)) throw new IllegalArgumentException("Tau must be in [0, 1]: " + tau);
        if (batchSize <= 0) throw new IllegalArgumentException("Batch size must be positive: " + batchSize);
        if (!Float.isFinite(entropyAlpha) || entropyAlpha < 0.0f || (autoEntropy && entropyAlpha == 0.0f)) {
            throw new IllegalArgumentException("Entropy alpha must be finite and " + (autoEntropy ? "positive" 
                    : "non-negative") + " (it is the initial temperature with autoEntropy): " + entropyAlpha);
        }
        
        this.backend = backend;
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.hiddenSize = hiddenSize;
        this.replayBuffer = replayBuffer;
        this.gamma = gamma;
        this.tau = tau;
        this.entropyAlpha = entropyAlpha;
        this.autoEntropy = autoEntropy;
        this.targetEntropy = -actionSize;
        this.batchSize = batchSize;
        this.random = new Random(seed);
        
        int criticInputs = stateSize + actionSize;
        Sequential actorNetwork = null;
        Sequential critic1 = null;
        Sequential critic2 = null;
        Sequential target1 = null;
        Sequential target2 = null;
        Parameter alphaParam = null;
        Optimizer actorOpt = null;
        Optimizer criticOpt = null;
        Optimizer alphaOpt = null;
        try {
            actorNetwork = RlNetworks.mlp(backend, stateSize, hiddenSize, 2 * actionSize, random);
            critic1 = RlNetworks.mlp(backend, criticInputs, hiddenSize, 1, random);
            critic2 = RlNetworks.mlp(backend, criticInputs, hiddenSize, 1, random);
            target1 = RlNetworks.mlp(backend, criticInputs, hiddenSize, 1, random);
            target2 = RlNetworks.mlp(backend, criticInputs, hiddenSize, 1, random);
            target1.setRequiresGrad(false);
            target2.setRequiresGrad(false);
            RlTensorMath.copyParameters(critic1, target1);
            RlTensorMath.copyParameters(critic2, target2);
            if (autoEntropy) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    alphaParam = new Parameter(Tensor.fromArray(backend,
                            new float[]{(float) Math.log(entropyAlpha)}), "log_alpha");
                }
            }
            actorOpt = RlTensorMath.buildOptimizer(actorOptimizerFactory, actorNetwork.parameters(), "SAC actor");
            criticOpt = RlTensorMath.buildOptimizer(criticOptimizerFactory,
                    RlTensorMath.parametersOf(critic1, critic2), "SAC critic");
            if (alphaParam != null) {
                alphaOpt = RlTensorMath.buildOptimizer(
                        alphaOptimizerFactory != null ? alphaOptimizerFactory : actorOptimizerFactory,
                        Collections.singletonList(alphaParam), "SAC temperature");
            }
        } catch (RuntimeException e) {
            try {
                RlTensorMath.closeAll(RlTensorMath.closer(actorOpt), RlTensorMath.closer(criticOpt),
                        RlTensorMath.closer(alphaOpt), alphaParam, actorNetwork, critic1, critic2, target1, target2);
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        this.actor = actorNetwork;
        this.q1 = critic1;
        this.q2 = critic2;
        this.q1Target = target1;
        this.q2Target = target2;
        this.logAlpha = alphaParam;
        this.actorOptimizer = actorOpt;
        this.criticOptimizer = criticOpt;
        this.alphaOptimizer = alphaOpt;
    }
    
    public static SACBuilder builder() {
        return new SACBuilder();
    }
    
    /**
     * A squashed policy action for {@code state}: a sample when {@code explore}, else {@code tanh(mu)}. Runs under
     * {@link ComputationGraph#noGrad()}; the action is read back.
     */
    @Override
    public float[] selectAction(float[] state, boolean explore) {
        checkNotClosed();
        Objects.requireNonNull(state, "State cannot be null");
        
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor stateTensor = RlTensorMath.row(backend, state);
             Noise noise = explore ? drawNoise(1) : null) {
            if (noise == null) {
                try (Tensor params = actor.forward(stateTensor);
                     Tensor mean = params.slice(1, 0, actionSize)) {
                    return toArray(TensorOps.tanh(mean));
                }
            }
            try (PolicySample sample = samplePolicy(stateTensor, noise)) {
                return sample.action.toFloatArray();
            }
        }
    }
    
    private static float[] toArray(Tensor t) {
        try (Tensor owned = t) {
            return owned.toFloatArray();
        }
    }
    
    /**
     * {@code rows} standard-normal noise rows drawn from the seeded {@link Random} and their Gaussian log-density
     * terms {@code -eps^2 / 2 - log(2 pi) / 2}, computed on the host from the same draws; both uploaded.
     */
    private Noise drawNoise(int rows) {
        float[][] eps = new float[rows][actionSize];
        float[][] logDensity = new float[rows][actionSize];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < actionSize; j++) {
                float e = (float) random.nextGaussian();
                eps[i][j] = e;
                logDensity[i][j] = -0.5f * e * e - HALF_LOG_TWO_PI;
            }
        }
        Tensor epsTensor = RlTensorMath.matrix(backend, eps);
        try {
            return new Noise(epsTensor, RlTensorMath.matrix(backend, logDensity));
        } catch (RuntimeException e) {
            epsTensor.close();
            throw e;
        }
    }
    
    /**
     * The reparameterized sample {@code a = tanh(mu + sigma * eps)} for a {@code [n, stateSize]} state batch and its
     * {@code [n]} log-density (see the class description), both recorded when the actor's parameters require grad.
     */
    private PolicySample samplePolicy(Tensor states, Noise noise) {
        Tensor action = null;
        try (Tensor params = actor.forward(states);
             Tensor mean = params.slice(1, 0, actionSize);
             Tensor rawLogStd = params.slice(1, actionSize, 2L * actionSize);
             Tensor logStd = TensorOps.clamp(rawLogStd, LOG_STD_MIN, LOG_STD_MAX);
             Tensor std = TensorOps.exp(logStd);
             Tensor scaled = TensorOps.mul(noise.eps, std);
             Tensor preSquash = TensorOps.add(mean, scaled);
             Tensor magnitude = TensorOps.abs(preSquash);
             Tensor doubled = TensorOps.mul(magnitude, -2.0f);
             Tensor tail = TensorOps.softplus(doubled);
             Tensor halfLogCosh = TensorOps.add(magnitude, tail);
             Tensor shifted = TensorOps.add(halfLogCosh, -LOG_TWO);
             Tensor squashCorrection = TensorOps.mul(shifted, 2.0f);
             Tensor gaussian = TensorOps.sub(noise.logDensity, logStd);
             Tensor perElement = TensorOps.add(gaussian, squashCorrection)) {
            action = TensorOps.tanh(preSquash);
            return new PolicySample(action, TensorOps.sum(perElement, 1));
        } catch (RuntimeException e) {
            if (action != null) {
                action.close();
            }
            throw e;
        }
    }
    
    /**
     * Stores one transition with its continuous action (copied) in the replay buffer.
     *
     * @throws IllegalArgumentException when {@code action} does not have {@code actionSize} values
     */
    public void storeTransition(float[] state, float[] action, float reward, float[] nextState, boolean done) {
        checkNotClosed();
        Objects.requireNonNull(replayBuffer, "SAC was built without a replay buffer");
        Objects.requireNonNull(action, "Action cannot be null");
        if (action.length != actionSize) {
            throw new IllegalArgumentException("Action has " + action.length + " values, expected " + actionSize);
        }
        replayBuffer.add(state, action, reward, nextState, done);
    }
    
    /**
     * One step over a replay batch: the critic, actor and temperature updates of the class description, then the soft
     * update of both target critics.
     *
     * @return the critic loss {@code 0.5 mean((Q1 - y)^2) + 0.5 mean((Q2 - y)^2)} before the update, or 0 while the
     *         buffer holds fewer than {@code batchSize} transitions
     * @throws IllegalStateException when a loss does not require grad (grad mode is disabled)
     */
    public float trainStep() {
        checkNotClosed();
        Objects.requireNonNull(replayBuffer, "SAC was built without a replay buffer");
        
        if (replayBuffer.size() < batchSize) {
            return 0.0f;
        }
        
        ReplayBuffer.SampleBatch batch = replayBuffer.sample(batchSize, random);
        float[][] states = batch.states();
        float[][] actions = batch.continuousActions();
        float[][] stateActions = new float[states.length][];
        for (int i = 0; i < states.length; i++) {
            stateActions[i] = concat(states[i], replayedAction(actions[i]));
        }
        int n = states.length;
        float criticLoss;
        Tensor detachedLogProb = null;
        try (Tensor pairs = RlTensorMath.matrix(backend, stateActions);
             Tensor stateBatch = RlTensorMath.matrix(backend, states);
             Tensor nextStates = RlTensorMath.matrix(backend, batch.nextStates());
             Tensor rewards = Tensor.fromArray(backend, batch.rewards());
             Tensor live = RlTensorMath.liveMask(backend, batch.dones());
             Noise nextNoise = drawNoise(n);
             Noise noise = drawNoise(n);
             Tensor alpha = temperature()) {
            criticLoss = criticUpdate(pairs, nextStates, rewards, live, nextNoise, alpha);
            if (actorOptimizer != null) {
                detachedLogProb = actorUpdate(stateBatch, noise, alpha);
            } else if (alphaOptimizer != null) {
                detachedLogProb = sampleLogProb(stateBatch, noise);
            }
            if (alphaOptimizer != null) {
                temperatureUpdate(detachedLogProb);
            }
        } finally {
            if (detachedLogProb != null) {
                detachedLogProb.close();
            }
        }
        
        RlTensorMath.softUpdate(q1, q1Target, tau);
        RlTensorMath.softUpdate(q2, q2Target, tau);
        
        return criticLoss;
    }
    
    /** The temperature of this step as a {@code [1]} tensor that does not require grad. */
    private Tensor temperature() {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (logAlpha != null) {
                return TensorOps.exp(logAlpha.tensor());
            }
            return RlTensorMath.constant(backend, Shape.vector(1), entropyAlpha);
        }
    }
    
    /** The critic update (step 1 of the class description); returns the critic loss before the update. */
    private float criticUpdate(Tensor pairs, Tensor nextStates, Tensor rewards, Tensor live, Noise nextNoise,
                               Tensor alpha) {
        if (criticOptimizer == null) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                 Tensor target = softTarget(nextStates, rewards, live, nextNoise, alpha);
                 Tensor loss = criticLoss(pairs, target)) {
                return loss.toFloatArray()[0];
            }
        }
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             Tensor target = softTarget(nextStates, rewards, live, nextNoise, alpha);
             Tensor loss = criticLoss(pairs, target)) {
            return RlTensorMath.optimizerStep(criticOptimizer, loss, "SAC critic");
        }
    }
    
    /**
     * {@code r + gamma (1 - d) (min(Q1', Q2')(s', a') - alpha log pi(a'|s'))} with {@code a'} sampled from the current
     * policy, computed under {@link ComputationGraph#noGrad()}.
     */
    private Tensor softTarget(Tensor nextStates, Tensor rewards, Tensor live, Noise nextNoise, Tensor alpha) {
        int n = nextStates.shape().dim(0);
        Shape column = Shape.vector(n);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             PolicySample next = samplePolicy(nextStates, nextNoise);
             Tensor nextPairs = TensorOps.cat(Arrays.asList(nextStates, next.action), 1);
             Tensor t1Out = q1Target.forward(nextPairs);
             Tensor t2Out = q2Target.forward(nextPairs);
             Tensor t1Value = t1Out.reshape(column);
             Tensor t2Value = t2Out.reshape(column);
             Tensor nextQ = RlTensorMath.pairMin(t1Value, t2Value);
             Tensor entropyTerm = TensorOps.mul(next.logProb, alpha);
             Tensor softValue = TensorOps.sub(nextQ, entropyTerm);
             Tensor discounted = TensorOps.mul(softValue, gamma);
             Tensor bootstrap = RlTensorMath.whereLive(live, discounted)) {
            return TensorOps.add(rewards, bootstrap);
        }
    }
    
    /** {@code 0.5 mean((Q1(s, a) - y)^2) + 0.5 mean((Q2(s, a) - y)^2)} for the replayed pairs. */
    private Tensor criticLoss(Tensor pairs, Tensor target) {
        try (Tensor l1 = criticError(q1, pairs, target);
             Tensor l2 = criticError(q2, pairs, target);
             Tensor both = TensorOps.add(l1, l2)) {
            return TensorOps.mul(both, 0.5f);
        }
    }
    
    /** {@code mean((critic(pairs) - target)^2)}. */
    private static Tensor criticError(Module critic, Tensor pairs, Tensor target) {
        try (Tensor out = critic.forward(pairs);
             Tensor value = out.reshape(target.shape());
             Tensor error = TensorOps.sub(value, target);
             Tensor squared = TensorOps.mul(error, error)) {
            return TensorOps.mean(squared);
        }
    }
    
    /**
     * The actor update (step 2 of the class description), with both critics frozen for its duration; returns a
     * detached handle of the sample's {@code [n]} log-density for the temperature update, owned by the caller.
     */
    private Tensor actorUpdate(Tensor states, Noise noise, Tensor alpha) {
        q1.setRequiresGrad(false);
        q2.setRequiresGrad(false);
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             PolicySample sample = samplePolicy(states, noise)) {
            Shape column = Shape.vector(states.shape().dim(0));
            Tensor detached = sample.logProb.detach();
            try (Tensor pairs = TensorOps.cat(Arrays.asList(states, sample.action), 1);
                 Tensor out1 = q1.forward(pairs);
                 Tensor out2 = q2.forward(pairs);
                 Tensor value1 = out1.reshape(column);
                 Tensor value2 = out2.reshape(column);
                 Tensor minQ = RlTensorMath.pairMin(value1, value2);
                 Tensor weighted = TensorOps.mul(sample.logProb, alpha);
                 Tensor objective = TensorOps.sub(weighted, minQ);
                 Tensor loss = TensorOps.mean(objective)) {
                RlTensorMath.optimizerStep(actorOptimizer, loss, "SAC actor");
                return detached;
            } catch (RuntimeException e) {
                detached.close();
                throw e;
            }
        } finally {
            q1.setRequiresGrad(true);
            q2.setRequiresGrad(true);
        }
    }
    
    /**
     * The {@code [n]} log-density of the policy sample for {@code noise}, computed under
     * {@link ComputationGraph#noGrad()} for a temperature update without an actor update; owned by the caller.
     */
    private Tensor sampleLogProb(Tensor states, Noise noise) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             PolicySample sample = samplePolicy(states, noise)) {
            return sample.logProb.detach();
        }
    }

    /** The temperature update (step 3 of the class description) on the actor step's detached log-density. */
    private void temperatureUpdate(Tensor logProb) {
        try (ComputationGraph.StepScope step = ComputationGraph.current().step();
             Tensor gap = TensorOps.add(logProb, targetEntropy);
             Tensor weighted = TensorOps.mul(logAlpha.tensor(), gap);
             Tensor mean = TensorOps.mean(weighted);
             Tensor loss = TensorOps.neg(mean)) {
            RlTensorMath.optimizerStep(alphaOptimizer, loss, "SAC temperature");
        }
    }
    
    /**
     * A replayed continuous action, checked against {@code actionSize}.
     *
     * @throws IllegalStateException when the buffer holds a discrete transition or an action of another size
     */
    private float[] replayedAction(float[] action) {
        if (action == null) {
            throw new IllegalStateException("SAC replays continuous actions; the replay buffer holds a transition with"
                + " a discrete action (store transitions through storeTransition or ReplayBuffer.add(float[], float[],"
                + " float, float[], boolean))");
        }
        if (action.length != actionSize) {
            throw new IllegalStateException("Replayed action has " + action.length + " values, expected "
                + actionSize);
        }
        return action;
    }
    
    private float[] concat(float[] state, float[] action) {
        float[] combined = new float[stateSize + actionSize];
        System.arraycopy(state, 0, combined, 0, stateSize);
        System.arraycopy(action, 0, combined, stateSize, actionSize);
        return combined;
    }
    
    /** {@code critic(state, action)}, read back as one scalar, under {@link ComputationGraph#noGrad()}. */
    private float criticValue(Module critic, float[] state, float[] action) {
        try (Tensor input = RlTensorMath.row(backend, concat(state, action));
             Tensor output = critic.forward(input)) {
            return output.toFloatArray()[0];
        }
    }
    
    /** The first critic's value of {@code (state, action)}; runs under {@link ComputationGraph#noGrad()}. */
    public float getQ1Value(float[] state, float[] action) {
        checkNotClosed();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return criticValue(q1, state, action);
        }
    }
    
    /** The second critic's value of {@code (state, action)}; runs under {@link ComputationGraph#noGrad()}. */
    public float getQ2Value(float[] state, float[] action) {
        checkNotClosed();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            return criticValue(q2, state, action);
        }
    }
    
    /**
     * The current temperature {@code alpha}: {@code exp(log alpha)} read back under
     * {@link ComputationGraph#noGrad()} with {@code autoEntropy}, else the fixed {@code entropyAlpha}.
     */
    public float entropyCoefficient() {
        checkNotClosed();
        if (logAlpha == null) {
            return entropyAlpha;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor alpha = TensorOps.exp(logAlpha.tensor())) {
            return alpha.toFloatArray()[0];
        }
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    @Override
    public void reset() {
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("SAC has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            RlTensorMath.closeAll(RlTensorMath.closer(actorOptimizer), RlTensorMath.closer(criticOptimizer),
                    RlTensorMath.closer(alphaOptimizer), logAlpha, actor, q1, q2, q1Target, q2Target);
        }
    }
    
    @Override
    public String toString() {
        return "SAC{stateSize=" + stateSize + ", actionSize=" + actionSize + 
               ", hiddenSize=" + hiddenSize + "}";
    }
    
    /** Uploaded policy noise and its Gaussian log-density terms (see {@link #drawNoise}). */
    private static final class Noise implements AutoCloseable {
        final Tensor eps;
        final Tensor logDensity;
        
        Noise(Tensor eps, Tensor logDensity) {
            this.eps = eps;
            this.logDensity = logDensity;
        }
        
        @Override
        public void close() {
            RlTensorMath.closeAll(eps, logDensity);
        }
    }
    
    /** A policy sample: the {@code [n, actionSize]} squashed action and its {@code [n]} log-density. */
    private static final class PolicySample implements AutoCloseable {
        final Tensor action;
        final Tensor logProb;
        
        PolicySample(Tensor action, Tensor logProb) {
            this.action = action;
            this.logProb = logProb;
        }
        
        @Override
        public void close() {
            RlTensorMath.closeAll(action, logProb);
        }
    }
    
    public static final class SACBuilder {
        private ComputeBackend backend;
        private int stateSize;
        private int actionSize;
        private int hiddenSize = 256;
        private Function<List<Parameter>, ? extends Optimizer> actorOptimizerFactory;
        private Function<List<Parameter>, ? extends Optimizer> criticOptimizerFactory;
        private Function<List<Parameter>, ? extends Optimizer> alphaOptimizerFactory;
        private ReplayBuffer replayBuffer;
        private float gamma = 0.99f;
        private float tau = 0.005f;
        private float entropyAlpha = 0.2f;
        private boolean autoEntropy = true;
        private int batchSize = 256;
        private long seed = System.currentTimeMillis();
        
        public SACBuilder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public SACBuilder stateSize(int stateSize) {
            this.stateSize = stateSize;
            return this;
        }
        
        public SACBuilder actionSize(int actionSize) {
            this.actionSize = actionSize;
            return this;
        }
        
        /** The width of the two hidden layers of the actor and of every critic (default 256); must be positive. */
        public SACBuilder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }
        
        /**
         * The factory of the actor's optimizer: it receives the actor's parameters (the networks exist only inside the
         * agent) and returns an optimizer over all of them, such as {@code params -> new Adam(params, 3e-4f)}. It
         * also builds the temperature optimizer when {@link #alphaOptimizer} is not set. Without it the actor is not
         * trained, and neither is the temperature unless {@link #alphaOptimizer} is set.
         */
        public SACBuilder actorOptimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.actorOptimizerFactory = optimizerFactory;
            return this;
        }
        
        /** The factory of the critics' optimizer, over both critics' parameters; without it they are not trained. */
        public SACBuilder criticOptimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.criticOptimizerFactory = optimizerFactory;
            return this;
        }
        
        /** The factory of the temperature optimizer (over {@code log alpha}), used with {@code autoEntropy}. */
        public SACBuilder alphaOptimizer(Function<List<Parameter>, ? extends Optimizer> optimizerFactory) {
            this.alphaOptimizerFactory = optimizerFactory;
            return this;
        }
        
        public SACBuilder replayBuffer(ReplayBuffer replayBuffer) {
            this.replayBuffer = replayBuffer;
            return this;
        }
        
        public SACBuilder gamma(float gamma) {
            this.gamma = gamma;
            return this;
        }
        
        public SACBuilder tau(float tau) {
            this.tau = tau;
            return this;
        }
        
        public SACBuilder entropyAlpha(float entropyAlpha) {
            this.entropyAlpha = entropyAlpha;
            return this;
        }
        
        public SACBuilder autoEntropy(boolean autoEntropy) {
            this.autoEntropy = autoEntropy;
            return this;
        }
        
        public SACBuilder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public SACBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }
        
        public SAC build() {
            return new SAC(backend, stateSize, actionSize, hiddenSize, actorOptimizerFactory, criticOptimizerFactory,
                          alphaOptimizerFactory, replayBuffer, gamma, tau, entropyAlpha, autoEntropy, batchSize, seed);
        }
    }
}

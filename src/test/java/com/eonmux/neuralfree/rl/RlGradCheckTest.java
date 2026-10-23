package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.autograd.GradientCheckUtil;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.Sequential;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Finite-difference gradient checks of the reinforcement-learning objectives, on the CPU path and on each GPU API. The
 * rl package has no module class of its own: A2C, PPO, DQN and SAC train the two-hidden-layer perceptrons of
 * {@link RlNetworks} on recorded losses built from {@link RlTensorMath}, and RLHF trains the models its caller supplies
 * ({@link Linear} layers here). Each check forms one trainer's loss from the same helpers and operation order as the
 * trainer's loss method (A2C's rollout loss, PPO's clipped surrogate, DQN's weighted TD error, SAC's actor, twin critic
 * and temperature losses, RLHF's preference loss and clipped PPO step), over networks of the trainer's
 * architecture with designed weights, and compares its recorded gradient in the networks' parameters with central
 * differences ({@link GradientCheckUtil}: step {@value #EPSILON}, tolerance {@value #TOLERANCE}).
 *
 * <p>How the kinks stay away from every evaluation point:</p>
 * <ul>
 *   <li><b>ReLU.</b> Every perceptron input is in {@code [-1, 1]} (states, squashed actions and one-hot actions) and
 *   the perceptrons have {@value #HIDDEN} hidden units per layer with weights in {@code [-0.1, 0.1]} and biases
 *   alternating {@code +1} (live units) and {@code -1} (dead units). A first-layer pre-activation is therefore at
 *   least {@code 1 - 5 * 0.1 = 0.5} from 0 and a live unit's output at most 1.5; a second-layer pre-activation sums
 *   {@value #HIDDEN} / 2 live inputs, so it is at least {@code 1 - 2 * 0.1 * 1.5 = 0.7} from 0. A step of
 *   {@value #EPSILON} in any parameter moves them by far less, so both sides of every ReLU are exercised without
 *   crossing a kink.</li>
 *   <li>The probability floors ({@code 1e-10}) of {@code clampedLog} and the entropy thresholds are far below the
 *   softmax probabilities of these small logits.</li>
 *   <li><b>Clipped surrogates</b> (PPO, RLHF). The old log-probabilities are the current ones minus fixed offsets
 *   {@code delta}, so each ratio is {@code exp(delta)}: inside the clip range ({@code |delta| <= 0.1}, at least 0.12
 *   from {@code log(1 +- 0.2)}), where the surrogate and its clipped form are the same function, or outside it
 *   ({@code |delta| >= 0.3}, at least 0.11 away), where the two differ by at least {@code 0.07}; rows of both signs of
 *   the advantage select each side of the minimum.</li>
 *   <li><b>SAC.</b> The pre-squash actions {@code u = mu + sigma eps} are placed at fixed values with
 *   {@code |u| >= 0.3} by choosing {@code eps} from the actor's current outputs, so {@code |u|} is smooth; the
 *   log-standard deviations stay within {@code +-1.2}, inside the clamp {@code [-20, 2]}; the second critic is the
 *   first plus {@code 2 s_0} with {@code |s_0| >= 0.55}, so their minimum is at least 1.1 from a tie.</li>
 *   <li><b>RLHF preferences.</b> The reward margins {@code d} are {@code 0.6}, {@code -0.4} and {@code 0.475} by
 *   construction, away from the kinks of {@code relu(-d)} and {@code |d|} at 0.</li>
 *   <li>DQN's target and SAC's critic target are computed under {@code noGrad}, as the trainers compute them.</li>
 * </ul>
 */
@RunWith(Enclosed.class)
public class RlGradCheckTest {

    static final float EPSILON = 1e-3f;
    static final float TOLERANCE = 1e-2f;
    static final int ELEMENTS = 6;

    static final float MIN_PROB = 1e-10f;
    static final float CLIP_EPSILON = 0.2f;
    static final float VALUE_COEFF = 0.5f;
    static final float ENTROPY_COEFF = 0.01f;
    static final float GAMMA = 0.9f;
    static final int ACTIONS = 3;
    static final int SAC_ACTIONS = 2;
    static final float SAC_ALPHA = 0.2f;
    static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2.0 * Math.PI));
    static final float LOG_TWO = (float) Math.log(2.0);
    /** The hidden width of the checked perceptrons. */
    static final int HIDDEN = 4;
    /** The bound of the hidden layers' weights (see the class description). */
    static final float HIDDEN_WEIGHT = 0.1f;

    /** Four states whose first component is at least 0.55 in magnitude. */
    static final float[][] STATES = {
        {0.8f, -0.3f, 0.2f}, {-0.6f, 0.5f, -0.1f}, {0.55f, 0.9f, -0.7f}, {-0.9f, -0.4f, 0.6f}};
    static final float[][] NEXT_STATES = {
        {0.1f, 0.7f, -0.5f}, {0.4f, -0.8f, 0.3f}, {-0.2f, 0.1f, 0.9f}, {0.6f, 0.2f, -0.4f}};

    /** The checks, on the backend of the subclass. */
    public abstract static class Checks {

        protected abstract ComputeBackend backend();

        /** A2C: {@code mean(-log pi(a) A + c_v (R - V)^2 - c_e H)} over a rollout of four steps. */
        @Test
        public void advantageActorCritic() {
            ComputeBackend b = backend();
            Sequential actor = mlp(b, 3, ACTIONS, 501, 0.5f);
            Sequential critic = mlp(b, 3, 1, 502, 0.5f);
            try (Tensor states = RlTensorMath.matrix(b, STATES);
                 Tensor actions = RlTensorMath.columnIndex(b, new int[]{0, 2, 1, 2}, ACTIONS);
                 Tensor returns = Tensor.fromArray(b, new float[]{0.7f, -0.2f, 1.1f, 0.3f});
                 Tensor advantages = Tensor.fromArray(b, new float[]{0.9f, -1.2f, 0.4f, -0.1f})) {
                assertParameters("A2C rollout loss",
                    () -> actorCriticLoss(actor, critic, states, actions, returns, advantages, null),
                    parametersOf(actor, critic));
            } finally {
                actor.close();
                critic.close();
            }
        }

        /** PPO: the clipped surrogate with value and entropy terms, both sides of the clip and of the minimum. */
        @Test
        public void proximalPolicyOptimization() {
            ComputeBackend b = backend();
            Sequential actor = mlp(b, 3, ACTIONS, 511, 0.5f);
            Sequential critic = mlp(b, 3, 1, 512, 0.5f);
            float[] delta = {0.05f, 0.4f, -0.4f, 0.4f};
            try (Tensor states = RlTensorMath.matrix(b, STATES);
                 Tensor actions = RlTensorMath.columnIndex(b, new int[]{1, 0, 2, 1}, ACTIONS);
                 Tensor returns = Tensor.fromArray(b, new float[]{0.3f, 0.8f, -0.4f, 0.1f});
                 Tensor advantages = Tensor.fromArray(b, new float[]{0.9f, 0.8f, 0.6f, -0.5f});
                 Tensor oldLogProb = Tensor.fromArray(b, shiftedLogProbs(actor, states, actions, delta))) {
                assertParameters("PPO trajectory loss",
                    () -> actorCriticLoss(actor, critic, states, actions, returns, advantages, oldLogProb),
                    parametersOf(actor, critic));
            } finally {
                actor.close();
                critic.close();
            }
        }

        /** DQN: {@code mean(w (y - Q(s, a))^2)} with the target {@code y} of a frozen target network. */
        @Test
        public void deepQNetwork() {
            ComputeBackend b = backend();
            Sequential q = mlp(b, 3, 2, 521, 0.5f);
            Sequential target = mlp(b, 3, 2, 522, 0.5f);
            target.setRequiresGrad(false);
            try (Tensor states = RlTensorMath.matrix(b, STATES);
                 Tensor actions = RlTensorMath.columnIndex(b, new int[]{1, 0, 1, 0}, 2);
                 Tensor weights = Tensor.fromArray(b, new float[]{1.0f, 0.5f, 0.8f, 1.2f});
                 Tensor y = tdTarget(target, RlTensorMath.matrix(b, NEXT_STATES),
                     new float[]{1.0f, 0.0f, -0.5f, 0.2f}, new boolean[]{false, true, false, false})) {
                assertParameters("DQN weighted TD loss", () -> {
                    try (Tensor out = q.forward(states);
                         Tensor current = RlTensorMath.pickColumns(out, actions);
                         Tensor error = TensorOps.sub(y, current);
                         Tensor squared = TensorOps.mul(error, error);
                         Tensor weighted = TensorOps.mul(squared, weights)) {
                        return TensorOps.mean(weighted);
                    }
                }, q.parameters());
            } finally {
                q.close();
                target.close();
            }
        }

        /** SAC's actor loss through the squashed Gaussian and the frozen twin critics. */
        @Test
        public void softActorCriticActor() {
            ComputeBackend b = backend();
            Sequential actor = mlp(b, 3, 2 * SAC_ACTIONS, 531, 0.3f);
            Sequential q1 = mlp(b, 3 + SAC_ACTIONS, 1, 532, 0.5f);
            Module q2 = new ShiftedCritic(q1);
            q1.setRequiresGrad(false);
            q2.setRequiresGrad(false);
            float[] preSquash = {0.4f, -0.6f, 0.7f, -0.3f, -0.5f, 0.35f, 0.6f, -0.45f};
            try (Tensor states = RlTensorMath.matrix(b, STATES);
                 Tensor alpha = RlTensorMath.constant(b, Shape.vector(1), SAC_ALPHA)) {
                float[][][] noise = noiseFor(actor, states, preSquash);
                try (Tensor eps = RlTensorMath.matrix(b, noise[0]);
                     Tensor logDensity = RlTensorMath.matrix(b, noise[1])) {
                    assertParameters("SAC actor loss", () -> {
                        try (PolicySample sample = samplePolicy(actor, states, eps, logDensity);
                             Tensor pairs = TensorOps.cat(Arrays.asList(states, sample.action), 1);
                             Tensor out1 = q1.forward(pairs);
                             Tensor out2 = q2.forward(pairs);
                             Tensor value1 = out1.reshape(Shape.vector(STATES.length));
                             Tensor value2 = out2.reshape(Shape.vector(STATES.length));
                             Tensor minQ = RlTensorMath.pairMin(value1, value2);
                             Tensor weighted = TensorOps.mul(sample.logProb, alpha);
                             Tensor objective = TensorOps.sub(weighted, minQ)) {
                            return TensorOps.mean(objective);
                        }
                    }, actor.parameters());
                }
            } finally {
                actor.close();
                q1.close();
                q2.close();
            }
        }

        /** SAC's twin-critic loss on a constant target, and the temperature loss on a detached log-density. */
        @Test
        public void softActorCriticCriticsAndTemperature() {
            ComputeBackend b = backend();
            Sequential q1 = mlp(b, 3 + SAC_ACTIONS, 1, 541, 0.5f);
            Sequential q2 = mlp(b, 3 + SAC_ACTIONS, 1, 542, 0.5f);
            Parameter logAlpha = new Parameter(Tensor.fromArray(b, new float[]{(float) Math.log(SAC_ALPHA)}),
                "log_alpha");
            float[][] pairRows = new float[STATES.length][];
            for (int i = 0; i < STATES.length; i++) {
                pairRows[i] = new float[]{STATES[i][0], STATES[i][1], STATES[i][2], i % 2, 1 - i % 2};
            }
            try (Tensor pairs = RlTensorMath.matrix(b, pairRows);
                 Tensor y = Tensor.fromArray(b, new float[]{0.6f, -0.4f, 1.2f, 0.1f});
                 Tensor logProb = Tensor.fromArray(b, new float[]{-1.3f, -0.4f, -2.2f, 0.5f})) {
                assertParameters("SAC critic loss", () -> {
                    try (Tensor l1 = criticError(q1, pairs, y);
                         Tensor l2 = criticError(q2, pairs, y);
                         Tensor both = TensorOps.add(l1, l2)) {
                        return TensorOps.mul(both, 0.5f);
                    }
                }, parametersOf(q1, q2));
                assertParameters("SAC temperature loss", () -> {
                    try (Tensor gap = TensorOps.add(logProb, -(float) SAC_ACTIONS);
                         Tensor weighted = TensorOps.mul(logAlpha.tensor(), gap);
                         Tensor mean = TensorOps.mean(weighted)) {
                        return TensorOps.neg(mean);
                    }
                }, Collections.singletonList(logAlpha));
            } finally {
                q1.close();
                q2.close();
                logAlpha.close();
            }
        }

        /** RLHF's preference loss {@code mean(relu(-d) + softplus(-|d|))} over three pairs. */
        @Test
        public void rlhfPreferenceLoss() {
            ComputeBackend b = backend();
            Linear reward = new Linear(4, 2);
            float[] base = {0.25f, 0.5f, 0.75f, 1.0f};
            float[][] differences = {{1.0f, 0.0f, 0.5f, -0.25f}, {-0.5f, 0.5f, 0.0f, 0.25f}, {0.2f, -0.5f, 0.25f, 0.5f}};
            try {
                reward.initialize(b, new Random(551));
                Map<String, Object> state = new LinkedHashMap<>();
                state.put("weight", new float[]{0.5f, -0.4f, 0.3f, 0.2f, -0.3f, 0.6f, 0.1f, -0.2f});
                state.put("bias", new float[]{0.1f, -0.2f});
                reward.loadStateDict(state);
                assertParameters("RLHF preference loss", () -> {
                    List<Tensor> losses = new ArrayList<>();
                    try {
                        for (float[] difference : differences) {
                            float[] preferred = base.clone();
                            for (int j = 0; j < preferred.length; j++) {
                                preferred[j] += difference[j];
                            }
                            losses.add(preferenceLoss(reward, b, preferred, base));
                        }
                        try (Tensor all = TensorOps.cat(losses, 0)) {
                            return TensorOps.mean(all);
                        }
                    } finally {
                        for (Tensor t : losses) {
                            t.close();
                        }
                    }
                }, reward.parameters());
            } finally {
                reward.close();
            }
        }

        /**
         * RLHF's PPO step loss with the entropy bonus; the KL penalty is part of the advantages, a constant here, as
         * in the trainer.
         */
        @Test
        public void rlhfClippedPolicyLoss() {
            ComputeBackend b = backend();
            Linear policy = linear(b, 3, 6, 561, 0.5f);
            float[][] states = {STATES[0], STATES[2]};
            int[][] positions = {{2, 3}, {1, 4}};
            float[] delta = {0.3f, -0.1f};
            float[] advantage = {-0.5f, 0.7f};
            try {
                float[] old = new float[states.length];
                for (int i = 0; i < states.length; i++) {
                    try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
                         Tensor logProb = sequenceLogProb(policy, b, states[i], positions[i])) {
                        old[i] = logProb.toFloatArray()[0] - delta[i];
                    }
                }
                assertParameters("RLHF clipped PPO loss", () -> {
                    List<Tensor> losses = new ArrayList<>();
                    try {
                        for (int i = 0; i < states.length; i++) {
                            losses.add(rlhfPpoStepLoss(policy, b, states[i], positions[i], old[i], advantage[i]));
                        }
                        try (Tensor all = TensorOps.cat(losses, 0)) {
                            return TensorOps.mean(all);
                        }
                    } finally {
                        for (Tensor t : losses) {
                            t.close();
                        }
                    }
                }, policy.parameters());
            } finally {
                policy.close();
            }
        }
    }

    /** The CPU path. */
    public static class HostPath extends Checks {
        private ComputeBackend host;

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            TrainGpu.release(host);
        }

        @Override
        protected ComputeBackend backend() {
            return host;
        }
    }

    /** The device path on each GPU API. */
    @RunWith(Parameterized.class)
    public static class Device extends Checks {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            ComputationGraph.setGradEnabled(true);
            session = TrainParity.open(api);
        }

        @After
        public void tearDown() {
            ComputationGraph.current().clear();
            if (session != null) {
                session.close();
            }
        }

        @Override
        protected ComputeBackend backend() {
            return session.backend();
        }
    }

    /**
     * The per-step mean of {@code policy + c_v (R - V)^2 - c_e H}: with {@code oldLogProb} null A2C's policy term
     * {@code -log pi(a) A}, otherwise PPO's {@code -min(rA, clip(r, 1 - eps, 1 + eps) A)} with
     * {@code r = exp(log pi(a) - oldLogProb)}.
     */
    static Tensor actorCriticLoss(Module actor, Module critic, Tensor states, Tensor actions, Tensor returns,
                                  Tensor advantages, Tensor oldLogProb) {
        int n = states.shape().dim(0);
        try (Tensor logits = actor.forward(states);
             Tensor probs = TensorOps.softmax(logits, 1);
             Tensor chosen = RlTensorMath.pickColumns(probs, actions);
             Tensor logProb = RlTensorMath.clampedLog(chosen, MIN_PROB);
             Tensor policyLoss = oldLogProb == null ? policyGradientTerm(logProb, advantages)
                 : clippedSurrogateTerm(logProb, oldLogProb, advantages);
             Tensor criticOut = critic.forward(states);
             Tensor value = criticOut.reshape(Shape.vector(n));
             Tensor valueError = TensorOps.sub(returns, value);
             Tensor valueLoss = TensorOps.mul(valueError, valueError);
             Tensor entropy = RlTensorMath.rowEntropy(probs, 0.0f);
             Tensor scaledValueLoss = TensorOps.mul(valueLoss, VALUE_COEFF);
             Tensor partial = TensorOps.add(policyLoss, scaledValueLoss);
             Tensor scaledEntropy = TensorOps.mul(entropy, ENTROPY_COEFF);
             Tensor loss = TensorOps.sub(partial, scaledEntropy)) {
            return TensorOps.mean(loss);
        }
    }

    private static Tensor policyGradientTerm(Tensor logProb, Tensor advantages) {
        try (Tensor weighted = TensorOps.mul(logProb, advantages)) {
            return TensorOps.neg(weighted);
        }
    }

    private static Tensor clippedSurrogateTerm(Tensor logProb, Tensor oldLogProb, Tensor advantages) {
        try (Tensor logRatio = TensorOps.sub(logProb, oldLogProb);
             Tensor ratio = TensorOps.exp(logRatio);
             Tensor clipped = TensorOps.clamp(ratio, 1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
             Tensor surrogate = TensorOps.mul(ratio, advantages);
             Tensor clippedSurrogate = TensorOps.mul(clipped, advantages);
             Tensor pessimistic = RlTensorMath.pairMin(surrogate, clippedSurrogate)) {
            return TensorOps.neg(pessimistic);
        }
    }

    /** The current {@code log pi(a|s)} per row minus {@code delta}, read back under {@code noGrad}. */
    static float[] shiftedLogProbs(Module actor, Tensor states, Tensor actions, float[] delta) {
        float[] values;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor logits = actor.forward(states);
             Tensor probs = TensorOps.softmax(logits, 1);
             Tensor chosen = RlTensorMath.pickColumns(probs, actions);
             Tensor logProb = RlTensorMath.clampedLog(chosen, MIN_PROB)) {
            values = logProb.toFloatArray();
        }
        for (int i = 0; i < values.length; i++) {
            values[i] -= delta[i];
        }
        return values;
    }

    /**
     * DQN's target {@code r + gamma * live * max_a' Q_target(s', a')} under {@code noGrad}; {@code nextStates} is
     * closed.
     */
    static Tensor tdTarget(Module target, Tensor nextStates, float[] rewards, boolean[] dones) {
        ComputeBackend b = nextStates.backend();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor next = nextStates;
             Tensor rewardTensor = Tensor.fromArray(b, rewards);
             Tensor live = RlTensorMath.liveMask(b, dones);
             Tensor nextQ = target.forward(next);
             Tensor best = RlTensorMath.rowMaxSkippingNaN(nextQ);
             Tensor discounted = TensorOps.mul(best, GAMMA);
             Tensor bootstrap = RlTensorMath.whereLive(live, discounted)) {
            return TensorOps.add(rewardTensor, bootstrap);
        }
    }

    /** {@code mean((critic(pairs) - target)^2)}. */
    static Tensor criticError(Module critic, Tensor pairs, Tensor target) {
        try (Tensor out = critic.forward(pairs);
             Tensor value = out.reshape(target.shape());
             Tensor error = TensorOps.sub(value, target);
             Tensor squared = TensorOps.mul(error, error)) {
            return TensorOps.mean(squared);
        }
    }

    /** A squashed Gaussian sample and its log-density, both recorded. */
    static final class PolicySample implements AutoCloseable {
        final Tensor action;
        final Tensor logProb;

        PolicySample(Tensor action, Tensor logProb) {
            this.action = action;
            this.logProb = logProb;
        }

        @Override
        public void close() {
            action.close();
            logProb.close();
        }
    }

    /**
     * SAC's reparameterized sample {@code a = tanh(mu + sigma eps)} and
     * {@code log pi = sum_j [logDensity_j - logStd_j + 2 (|u_j| + softplus(-2 |u_j|) - log 2)]}, as SAC forms them.
     */
    static PolicySample samplePolicy(Module actor, Tensor states, Tensor eps, Tensor logDensity) {
        Tensor action = null;
        try (Tensor params = actor.forward(states);
             Tensor mean = params.slice(1, 0, SAC_ACTIONS);
             Tensor rawLogStd = params.slice(1, SAC_ACTIONS, 2L * SAC_ACTIONS);
             Tensor logStd = TensorOps.clamp(rawLogStd, -20.0f, 2.0f);
             Tensor std = TensorOps.exp(logStd);
             Tensor scaled = TensorOps.mul(eps, std);
             Tensor preSquash = TensorOps.add(mean, scaled);
             Tensor magnitude = TensorOps.abs(preSquash);
             Tensor doubled = TensorOps.mul(magnitude, -2.0f);
             Tensor tail = TensorOps.softplus(doubled);
             Tensor halfLogCosh = TensorOps.add(magnitude, tail);
             Tensor shifted = TensorOps.add(halfLogCosh, -LOG_TWO);
             Tensor squashCorrection = TensorOps.mul(shifted, 2.0f);
             Tensor gaussian = TensorOps.sub(logDensity, logStd);
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
     * The noise rows {@code eps = (u - mu) / sigma} that put the actor's pre-squash actions at {@code preSquash}
     * (row-major), and their Gaussian log-density terms {@code -eps^2 / 2 - log(2 pi) / 2}, from the actor's current
     * outputs read back under {@code noGrad}.
     */
    static float[][][] noiseFor(Module actor, Tensor states, float[] preSquash) {
        float[] out;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor params = actor.forward(states)) {
            out = params.toFloatArray();
        }
        int rows = states.shape().dim(0);
        float[][] eps = new float[rows][SAC_ACTIONS];
        float[][] logDensity = new float[rows][SAC_ACTIONS];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < SAC_ACTIONS; j++) {
                float mean = out[i * 2 * SAC_ACTIONS + j];
                float logStd = out[i * 2 * SAC_ACTIONS + SAC_ACTIONS + j];
                assertTrue("log-standard deviation " + logStd + " inside the clamp", Math.abs(logStd) <= 1.2f);
                float e = (preSquash[i * SAC_ACTIONS + j] - mean) / (float) Math.exp(logStd);
                eps[i][j] = e;
                logDensity[i][j] = -0.5f * e * e - HALF_LOG_TWO_PI;
            }
        }
        return new float[][][]{eps, logDensity};
    }

    /**
     * The second critic {@code Q2(x) = Q1(x) + 2 x_0} over {@code Q1}, read through (not owned): on the
     * {@code (state, action)} pairs {@code x_0} is the first state component, so the two critics differ by
     * {@code 2 s_0} and their minimum is the first critic where {@code s_0 > 0} and the second where {@code s_0 < 0}.
     */
    static final class ShiftedCritic extends Module {
        private final Module base;

        ShiftedCritic(Module base) {
            this.base = base;
        }

        @Override
        public Tensor forward(Tensor input) {
            try (Tensor value = base.forward(input);
                 Tensor first = input.slice(1, 0, 1);
                 Tensor shift = TensorOps.mul(first, 2.0f)) {
                return TensorOps.add(value, shift);
            }
        }
    }

    /**
     * RLHF's {@code relu(-d) + softplus(-|d|)} with {@code d = r(preferred) - r(rejected)}, as a {@code [1]} tensor.
     */
    static Tensor preferenceLoss(Linear reward, ComputeBackend b, float[] preferred, float[] rejected) {
        try (Tensor preferredTensor = RlTensorMath.row(b, preferred);
             Tensor rejectedTensor = RlTensorMath.row(b, rejected);
             Tensor preferredOut = reward.forward(preferredTensor);
             Tensor rejectedOut = reward.forward(rejectedTensor);
             Tensor preferredReward = RlTensorMath.firstElement(preferredOut, "reward model");
             Tensor rejectedReward = RlTensorMath.firstElement(rejectedOut, "reward model");
             Tensor margin = TensorOps.sub(preferredReward, rejectedReward);
             Tensor negMargin = TensorOps.neg(margin);
             Tensor linearPart = TensorOps.relu(negMargin);
             Tensor absMargin = TensorOps.abs(margin);
             Tensor negAbs = TensorOps.neg(absMargin);
             Tensor smoothPart = TensorOps.softplus(negAbs)) {
            return TensorOps.add(linearPart, smoothPart);
        }
    }

    /**
     * {@code sum_t log(max(p_t, 1e-10))} over the row-major {@code positions} of the {@code [2, 3]} token
     * probabilities of {@code model(state)}, as a {@code [1]} tensor.
     */
    static Tensor sequenceLogProb(Linear model, ComputeBackend b, float[] state, int[] positions) {
        try (Tensor stateRow = RlTensorMath.row(b, state);
             Tensor logits = model.forward(stateRow);
             Tensor probs = tokenProbabilities(logits)) {
            return pickedLogProb(probs, b, positions);
        }
    }

    private static Tensor tokenProbabilities(Tensor logits) {
        try (Tensor rows = logits.reshape(Shape.matrix(2, 3))) {
            return TensorOps.softmax(rows, 1);
        }
    }

    private static Tensor pickedLogProb(Tensor probs, ComputeBackend b, int[] positions) {
        try (Tensor flatIndex = Tensor.fromArray(b, positions);
             Tensor index = flatIndex.reshape(Shape.matrix(1, positions.length));
             Tensor flat = probs.reshape(Shape.matrix(1, (int) probs.numElements()));
             Tensor picked = TensorOps.gather(flat, 1, index);
             Tensor logs = RlTensorMath.clampedLog(picked, MIN_PROB)) {
            return TensorOps.sum(logs, 1);
        }
    }

    /**
     * RLHF's step loss {@code -min(rA, clip(r) A) - c_e mean_t H_t} with {@code r = exp(log pi - oldLogProb)}.
     */
    static Tensor rlhfPpoStepLoss(Linear policy, ComputeBackend b, float[] state, int[] positions, float oldLogProb,
                                  float advantage) {
        try (Tensor stateRow = RlTensorMath.row(b, state);
             Tensor logits = policy.forward(stateRow);
             Tensor probs = tokenProbabilities(logits);
             Tensor newLogProb = pickedLogProb(probs, b, positions);
             Tensor logRatio = TensorOps.add(newLogProb, -oldLogProb);
             Tensor ratio = TensorOps.exp(logRatio);
             Tensor clipped = TensorOps.clamp(ratio, 1.0f - CLIP_EPSILON, 1.0f + CLIP_EPSILON);
             Tensor surrogate = TensorOps.mul(ratio, advantage);
             Tensor clippedSurrogate = TensorOps.mul(clipped, advantage);
             Tensor pessimistic = RlTensorMath.pairMin(surrogate, clippedSurrogate);
             Tensor policyTerm = TensorOps.neg(pessimistic);
             Tensor rowEntropies = RlTensorMath.rowEntropy(probs, MIN_PROB);
             Tensor meanEntropy = TensorOps.mean(rowEntropies);
             Tensor entropy = meanEntropy.reshape(Shape.vector(1));
             Tensor entropyBonus = TensorOps.mul(entropy, ENTROPY_COEFF)) {
            return TensorOps.sub(policyTerm, entropyBonus);
        }
    }

    /**
     * A perceptron of the trainers' architecture ({@link RlNetworks#mlp}, {@value #HIDDEN} hidden units) on {@code b}
     * with the designed weights of the class description: hidden weights seeded in {@code [-0.1, 0.1]}, hidden biases
     * alternating {@code +1} and {@code -1}, output weights and biases seeded in {@code [-scale, scale]}.
     */
    static Sequential mlp(ComputeBackend b, int in, int out, long seed, float scale) {
        Sequential network = RlNetworks.mlp(b, in, HIDDEN, out, new Random(seed));
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("0.weight", TrainParity.uniform(seed * 31 + 1, HIDDEN * in, -HIDDEN_WEIGHT, HIDDEN_WEIGHT));
        state.put("0.bias", liveAndDeadBiases());
        state.put("2.weight", TrainParity.uniform(seed * 31 + 2, HIDDEN * HIDDEN, -HIDDEN_WEIGHT, HIDDEN_WEIGHT));
        state.put("2.bias", liveAndDeadBiases());
        state.put("4.weight", TrainParity.uniform(seed * 31 + 3, out * HIDDEN, -scale, scale));
        state.put("4.bias", TrainParity.uniform(seed * 31 + 4, out, -scale, scale));
        try {
            assertEquals("the designed weights name every parameter of the perceptron",
                network.namedParameters().keySet(), state.keySet());
            network.loadStateDict(state);
        } catch (RuntimeException | AssertionError e) {
            network.close();
            throw e;
        }
        return network;
    }

    /** {@code +1} for the even hidden units and {@code -1} for the odd ones. */
    private static float[] liveAndDeadBiases() {
        float[] bias = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) {
            bias[i] = i % 2 == 0 ? 1.0f : -1.0f;
        }
        return bias;
    }

    /** A {@link Linear} on {@code b} with seeded weights and biases in {@code [-scale, scale]}. */
    static Linear linear(ComputeBackend b, int in, int out, long seed, float scale) {
        Linear layer = new Linear(in, out);
        layer.initialize(b, new Random(seed));
        Map<String, Object> state = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : layer.stateDict().entrySet()) {
            int n = ((float[]) e.getValue()).length;
            state.put(e.getKey(), TrainParity.uniform(seed * 31 + e.getKey().hashCode(), n, -scale, scale));
        }
        layer.loadStateDict(state);
        return layer;
    }

    static List<Parameter> parametersOf(Module first, Module second) {
        List<Parameter> params = new ArrayList<>(first.parameters());
        params.addAll(second.parameters());
        return params;
    }

    /**
     * Asserts that the recorded gradients of {@code loss} in {@code params} match central differences on up to
     * {@value #ELEMENTS} elements per parameter, and that the check leaves the graph unchanged.
     */
    static void assertParameters(String what, Supplier<Tensor> loss, List<Parameter> params) {
        assertFalse(what + " has no parameters", params.isEmpty());
        int before = ComputationGraph.current().size();
        GradientCheckUtil.GradientCheckResult result =
            GradientCheckUtil.gradCheckParameters(loss, null, params, EPSILON, TOLERANCE, ELEMENTS);
        assertTrue(what + " parameter gradients: " + result, result.passed());
        assertEquals(what + ": the check leaves the graph as it found it", before, ComputationGraph.current().size());
    }
}

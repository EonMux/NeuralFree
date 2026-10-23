package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.inference.SpeculativeDecoder;
import com.eonmux.neuralfree.inference.SpeculativeDecoderConfig;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
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
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The reinforcement-learning algorithms and speculative decoding on the train dispatchers.
 * {@link CpuPath} checks each batched device composition on the host reference backend against an independent
 * per-sample definition, evaluated in double from the networks' own outputs. {@link GpuParity} runs the same
 * scenarios on every available GPU API against the host backend. The losses compose GEMMs, softmax, logarithms and
 * short sums, so they are compared in the reduction class with a relative bound of {@code 1e-4} ({@linkplain
 * com.eonmux.neuralfree.tensor.gpu numerical contract}); greedy decoding over single-product logits is bit-identical,
 * so its tokens must match exactly.
 *
 * <p>The trained scenarios give each algorithm an SGD optimizer through its factory and check one update end to end.
 * For A2C, PPO and DQN the analytic gradient of the documented loss at the networks' outputs is
 * back-propagated through an independent double-precision copy of their two-hidden-layer perceptrons
 * ({@link HostMlp}, read from the parameters the factory receives), the SGD update is applied to that copy, and its
 * outputs at a probe state predict the agent's outputs after the step; for the RLHF reward model, a {@link Linear}
 * the test supplies, the prediction is {@code delta r(probe) = -lr * sum_i g_i (s_i . probe + 1)}. A gradient that
 * misses a term, a parameter, a layer or a sign fails. SAC checks that every actor parameter and {@code log alpha}
 * receive a gradient while the frozen critics keep none from the actor update. Every training step must leave the
 * autograd graph at its size before the step.</p>
 */
@RunWith(Enclosed.class)
public class RlParityTest {

    private static final long SEED = 42L;
    private static final int STATE = 4;
    private static final int ACTIONS = 3;
    private static final int VOCAB = 6;
    private static final double REL = 1e-4;

    private static final float[] S = {0.3f, -0.7f, 0.5f, 0.1f};
    private static final float[] S_NEXT = {-0.2f, 0.4f, 0.9f, -0.6f};
    private static final float[][] ROLLOUT_STATES = {
        {0.2f, 0.1f, -0.3f, 0.8f}, {-0.5f, 0.6f, 0.2f, -0.1f}, {0.9f, -0.4f, 0.3f, 0.5f}};
    private static final int[] ROLLOUT_ACTIONS = {0, 2, 1};
    private static final float[] ROLLOUT_REWARDS = {1.0f, -0.5f, 0.25f};

    // ------------------------------------------------------------------ CPU path against the definitions

    public static class CpuPath {
        private final ComputeBackend host = TrainParity.hostBackend();

        @Test
        public void dqnLossIsTheSquaredTemporalDifferenceError() {
            DqnRun live = runDqn(host, false);
            double target = 0.5 + 0.9 * max(live.qNext);
            double td = target - live.qState[1];
            assertClose("DQN loss after the target sync", td * td, live.secondLoss);
            assertClose("DQN first loss (the target network starts as a copy)", td * td, live.firstLoss);

            DqnRun ended = runDqn(host, true);
            double endTd = 0.5 - ended.qState[1];
            assertClose("DQN loss at episode end", endTd * endTd, ended.secondLoss);
            assertClose("DQN first loss at episode end", endTd * endTd, ended.firstLoss);
        }

        @Test
        public void a2cLossMatchesTheRolloutDefinition() {
            A2cRun run = runA2c(host);
            int n = ROLLOUT_STATES.length;
            double[] adv = new double[n];
            double returns = run.lastValue;
            for (int i = n - 1; i >= 0; i--) {
                returns = ROLLOUT_REWARDS[i] + 0.9 * returns;
                adv[i] = returns - run.values[i];
            }
            double mean = Arrays.stream(adv).sum() / n;
            double var = 0.0;
            for (double a : adv) {
                var += (a - mean) * (a - mean);
            }
            double std = Math.sqrt(var / n + 1e-8);
            double total = 0.0;
            for (int i = 0; i < n; i++) {
                double a = (adv[i] - mean) / std;
                double logP = Math.log(Math.max(run.probs[i][ROLLOUT_ACTIONS[i]], 1e-10));
                total += -logP * a + 0.5 * adv[i] * adv[i] - 0.01 * entropy(run.probs[i], 0.0);
            }
            assertClose("A2C loss (the critic regresses on the discounted return)", total / n, run.loss);
        }

        @Test
        public void ppoLossMatchesTheClippedSurrogate() {
            PpoRun run = runPpo(host, true);
            assertClose("PPO loss (mean per-step loss over the epochs)", expectedPpoLoss(run), run.loss);
        }

        @Test
        public void ppoBootstrapsATrajectoryCutInsideAnEpisodeFromTheCriticsNextValue() {
            PpoRun run = runPpo(host, false);
            assertTrue("the bootstrap value is not trivially zero", Math.abs(run.nextValue) > 1e-3f);
            assertClose("PPO loss of a trajectory that ends inside an episode", expectedPpoLoss(run), run.loss);
        }

        @Test
        public void sacActionsAreSquashedAndTheSeededRunIsReproducible() {
            SacRun first = runSac(host);
            SacRun second = runSac(host);
            for (float a : first.explore) {
                assertTrue("exploring action in [-1, 1]: " + a, a >= -1.0f && a <= 1.0f);
            }
            for (float a : first.greedy) {
                assertTrue("greedy action in [-1, 1]: " + a, a >= -1.0f && a <= 1.0f);
            }
            assertArrayEquals(first.explore, second.explore, 0.0f);
            assertEquals(first.firstLoss, second.firstLoss, 0.0f);
            assertEquals(first.secondLoss, second.secondLoss, 0.0f);
            assertTrue("SAC loss is finite and non-negative", Float.isFinite(first.firstLoss) && first.firstLoss >= 0);
        }

        @Test
        public void rlhfLossesMatchTheirDefinitions() {
            RlhfRun run = runRlhf(host);
            double preference = 0.0;
            for (int k = 0; k < 2; k++) {
                double d = run.preferredRewards[k] - run.rejectedRewards[k];
                preference += Math.max(-d, 0.0) + Math.log1p(Math.exp(-Math.abs(d)));
            }
            assertClose("reward-model loss", preference / 2, run.rewardLoss);

            assertClose("RLHF PPO loss (the KL penalty enters through the rewards)",
                    expectedRlhfPpoLoss(run.policyLogits, run.referenceLogits, true, 0.0), run.ppoLoss);
            double[] penalties = rlhfKlPenalties(run.referenceLogits);
            assertClose("RLHF mean KL penalty", (penalties[0] + penalties[1]) / 2, run.klPenalty);
        }

        @Test
        public void rlhfBootstrapsABufferCutInsideAnEpisodeFromTheGivenNextValue() {
            RlhfCutRun run = runRlhfCut(host);
            assertTrue("a buffer that ends inside an episode is refused without a next value", run.refusedWithout);
            assertEquals("the refused buffer is kept", 2, run.bufferAfterRefusal);
            assertClose("RLHF PPO loss of a buffer that ends inside an episode",
                    expectedRlhfPpoLoss(run.policyLogits, run.referenceLogits, false, RLHF_NEXT_VALUE), run.ppoLoss);
        }

        @Test
        public void a2cUpdateFollowsTheLossGradient() {
            TrainedA2c run = runA2cTrained(host);
            int n = ROLLOUT_STATES.length;
            assertEquals("graph size after the step", run.graphBefore, run.graphAfter);
            double[] raw = new double[n];
            double returns = run.lastValue;
            for (int i = n - 1; i >= 0; i--) {
                returns = ROLLOUT_REWARDS[i] + 0.9 * returns;
                raw[i] = returns - run.values[i];
            }
            double[] a = normalize(raw);
            double[][] dLogits = new double[n][ACTIONS];
            double[][] dValue = new double[n][1];
            for (int i = 0; i < n; i++) {
                double h = entropy(run.probs[i], 0.0);
                for (int k = 0; k < ACTIONS; k++) {
                    double p = run.probs[i][k];
                    double chosen = k == ROLLOUT_ACTIONS[i] ? 1.0 : 0.0;
                    dLogits[i][k] = (-a[i] * (chosen - p) + 0.01 * p * (Math.log(p) + h)) / n;
                }
                dValue[i][0] = -2.0 * 0.5 * raw[i] / n;
            }
            assertEquals("one parameter group", 1, run.groups);
            assertReferenceMatches("A2C", run.actor, run.critic, run.probeProbsBefore, run.probeValueBefore);
            HostMlp actor = run.actor.sgdStep(ROLLOUT_STATES, dLogits, LR);
            HostMlp critic = run.critic.sgdStep(ROLLOUT_STATES, dValue, LR);
            assertCloseAll("A2C probabilities at the probe after the step", softmax(actor.forward(S)),
                    run.probeProbsAfter);
            assertClose("A2C value at the probe after the step", critic.forward(S)[0], run.probeValueAfter);
        }

        @Test
        public void ppoUpdateFollowsTheClippedSurrogateGradient() {
            TrainedPpo run = runPpoTrained(host);
            int n = ROLLOUT_STATES.length;
            assertEquals("graph size after the step", run.graphBefore, run.graphAfter);
            double[] adv = new double[n];
            double[] ret = new double[n];
            gae(adv, ret, true, 0.0);
            double[][] dLogits = new double[n][ACTIONS];
            double[][] dValue = new double[n][1];
            for (int i = 0; i < n; i++) {
                double p0 = run.probs[i][ROLLOUT_ACTIONS[i]];
                double ratio = Math.exp(Math.log(Math.max(p0, 1e-10)) - run.oldLogProbs[i]);
                double clipped = Math.max(0.8, Math.min(1.2, ratio));
                double dLogP = ratio * adv[i] <= clipped * adv[i] ? -adv[i] * ratio : 0.0;
                double h = entropy(run.probs[i], 0.0);
                for (int k = 0; k < ACTIONS; k++) {
                    double p = run.probs[i][k];
                    double chosen = k == ROLLOUT_ACTIONS[i] ? 1.0 : 0.0;
                    dLogits[i][k] = (dLogP * (chosen - p) + 0.01 * p * (Math.log(p) + h)) / n;
                }
                dValue[i][0] = -2.0 * 0.5 * (ret[i] - run.values[i]) / n;
            }
            assertReferenceMatches("PPO", run.actor, run.critic, run.probeProbsBefore, run.probeValueBefore);
            HostMlp actor = run.actor.sgdStep(ROLLOUT_STATES, dLogits, LR);
            HostMlp critic = run.critic.sgdStep(ROLLOUT_STATES, dValue, LR);
            assertCloseAll("PPO probabilities at the probe after the step", softmax(actor.forward(S)),
                    run.probeProbsAfter);
            assertClose("PPO value at the probe after the step", critic.forward(S)[0], run.probeValueAfter);
        }

        @Test
        public void dqnUpdateFollowsTheTakenActionsTemporalDifferenceGradient() {
            TrainedDqn run = runDqnTrained(host);
            assertEquals("graph size after the step", run.graphBefore, run.graphAfter);
            assertCloseAll("host reference of the Q-network", run.qNetwork.forward(S), run.qBefore);
            double q = run.qBefore[1];
            int batch = DQN_BATCH;
            float[][] states = new float[batch][];
            double[][] dq = new double[batch][ACTIONS];
            for (int i = 0; i < batch; i++) {
                states[i] = S;
                dq[i][1] = -2.0 * (0.5 - q) / batch;
            }
            HostMlp trained = run.qNetwork.sgdStep(states, dq, LR);
            assertCloseAll("DQN Q-values after the step (the hidden layers are shared by every action)",
                    trained.forward(S), run.qAfter);
            assertClose("DQN logged loss", (0.5 - q) * (0.5 - q), run.loss);
        }

        @Test
        public void sacUpdatesTrainTheActorAndTemperatureWithoutTouchingTheFrozenCritics() {
            TrainedSac run = runSacTrained(host);
            assertEquals("graph size after the step", run.graphBefore, run.graphAfter);
            assertTrue("critic loss is finite", Float.isFinite(run.criticLoss));
            for (int i = 0; i < run.criticGradNull.length; i++) {
                assertTrue("critic parameter " + i + " holds no gradient after the actor update",
                        run.criticGradNull[i]);
            }
            for (int i = 0; i < run.actorHasGrad.length; i++) {
                assertTrue("actor parameter " + i + " received a gradient", run.actorHasGrad[i]);
            }
            assertTrue("log alpha received a gradient", run.alphaHasGrad);
            assertFalse("actor changed", Arrays.deepEquals(run.actorBefore, run.actorAfter));
            assertFalse("critics changed", Arrays.deepEquals(run.criticBefore, run.criticAfter));
            assertFalse("log alpha changed", run.logAlphaBefore == run.logAlphaAfter);
            assertClose("temperature is exp(log alpha)", Math.exp(run.logAlphaAfter), run.alphaAfter);
            assertClose("initial temperature", 0.2, (float) Math.exp(run.logAlphaBefore));
        }

        @Test
        public void rlhfUpdatesTrainTheRewardModelAndPolicyAndLeaveTheReferenceFrozen() {
            TrainedRlhf run = runRlhfTrained(host);
            assertEquals("graph size after the reward step", run.graphBefore, run.graphAfterReward);
            assertEquals("graph size after the PPO step", run.graphBefore, run.graphAfterPpo);
            double expected = run.probeRewardBefore;
            for (int k = 0; k < 2; k++) {
                double d = run.preferredRewards[k] - run.rejectedRewards[k];
                double sigmoidNeg = 1.0 / (1.0 + Math.exp(d));
                expected += LR / 2 * sigmoidNeg
                        * (dot(RLHF_PREFERRED[k], S_NEXT) - dot(RLHF_REJECTED[k], S_NEXT));
            }
            assertClose("reward at the probe after the reward step", expected, run.probeRewardAfter);
            assertFalse("reference parameters are frozen", run.referenceRequiresGrad);
            assertTrue("reference parameters hold no gradient", run.referenceGradNull);
            assertArrayEquals("reference output unchanged", run.referenceBefore, run.referenceAfter, 0.0f);
            assertFalse("policy output changed", Arrays.equals(run.policyBefore, run.policyAfter));
        }

        @Test
        public void speculativeGreedyDecodingOfAnIdenticalDraftAcceptsEveryProposal() {
            try (LastTokenModel model = new LastTokenModel(VOCAB)) {
                model.initialize(host, SEED);
                SpeculativeDecoder decoder = SpeculativeDecoder.builder().draftModel(model).targetModel(model)
                        .config(new SpeculativeDecoderConfig(3, 1.0f, 0.0f)).seed(SEED).build();
                float[] prompt = {2.0f, 5.0f};
                float[] out = decoder.generate(prompt, 7);
                assertArrayEquals(greedyDecode(model, host, prompt, 7), out, 0.0f);
                assertEquals(1.0, decoder.acceptanceRate(), 0.0);
                assertEquals("rounds: 3 proposals plus a bonus token, then the last 3 proposals", 2,
                        decoder.totalDecodingSteps());
                assertEquals(6, decoder.totalAcceptedTokens());
            }
        }

        @Test
        public void speculativeDecodingIsTheSameForPerPositionAndLastRowTargets() {
            try (LastTokenModel draft = new LastTokenModel(VOCAB);
                 LastTokenModel lastRow = new LastTokenModel(VOCAB);
                 PerPositionModel perPosition = new PerPositionModel(VOCAB)) {
                draft.initialize(host, SEED + 1);
                lastRow.initialize(host, SEED);
                perPosition.initialize(host, SEED);
                SpeculativeDecoderConfig config = new SpeculativeDecoderConfig(4, 0.8f, 1.0f);
                float[] prompt = {1.0f, 3.0f, 4.0f};
                float[] viaLastRow = new SpeculativeDecoder(draft, lastRow, config, SEED).generate(prompt, 12);
                float[] viaPerPosition = new SpeculativeDecoder(draft, perPosition, config, SEED).generate(prompt, 12);
                assertEquals(prompt.length + 12, viaLastRow.length);
                assertArrayEquals(viaLastRow, viaPerPosition, 0.0f);
                for (int i = prompt.length; i < viaLastRow.length; i++) {
                    float t = viaLastRow[i];
                    assertTrue("token id " + t, t >= 0 && t < VOCAB && t == Math.rint(t));
                }
            }
        }
    }

    // ------------------------------------------------------------------ GPU against the host backend

    @RunWith(Parameterized.class)
    public static class GpuParity {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;
        private ComputeBackend host;

        public GpuParity(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
            host = TrainParity.hostBackend();
        }

        @After
        public void tearDown() {
            if (session != null) {
                session.close();
            }
        }

        @Test
        public void dqn() {
            for (boolean done : new boolean[]{false, true}) {
                DqnRun ref = runDqn(host, done);
                DqnRun dev = runDqn(gpu, done);
                assertCloseAll(api + " DQN Q(s)", ref.qState, dev.qState);
                assertCloseAll(api + " DQN Q(s')", ref.qNext, dev.qNext);
                assertClose(api + " DQN first loss", ref.firstLoss, dev.firstLoss);
                assertClose(api + " DQN second loss", ref.secondLoss, dev.secondLoss);
            }
        }

        @Test
        public void a2c() {
            A2cRun ref = runA2c(host);
            A2cRun dev = runA2c(gpu);
            for (int i = 0; i < ref.probs.length; i++) {
                assertCloseAll(api + " A2C probabilities " + i, ref.probs[i], dev.probs[i]);
            }
            assertClose(api + " A2C loss", ref.loss, dev.loss);
        }

        @Test
        public void ppo() {
            for (boolean lastDone : new boolean[]{true, false}) {
                PpoRun ref = runPpo(host, lastDone);
                PpoRun dev = runPpo(gpu, lastDone);
                assertClose(api + " PPO loss (last step done: " + lastDone + ")", ref.loss, dev.loss);
            }
        }

        @Test
        public void sac() {
            SacRun ref = runSac(host);
            SacRun dev = runSac(gpu);
            assertCloseAll(api + " SAC exploring action", ref.explore, dev.explore);
            assertCloseAll(api + " SAC greedy action", ref.greedy, dev.greedy);
            assertClose(api + " SAC first loss", ref.firstLoss, dev.firstLoss);
            assertClose(api + " SAC second loss", ref.secondLoss, dev.secondLoss);
            assertClose(api + " SAC Q1 after the soft update", ref.q1, dev.q1);
        }

        @Test
        public void rlhf() {
            RlhfRun ref = runRlhf(host);
            RlhfRun dev = runRlhf(gpu);
            assertCloseAll(api + " preferred rewards", ref.preferredRewards, dev.preferredRewards);
            assertClose(api + " reward-model loss", ref.rewardLoss, dev.rewardLoss);
            assertClose(api + " RLHF PPO loss", ref.ppoLoss, dev.ppoLoss);
            assertClose(api + " RLHF mean KL penalty", ref.klPenalty, dev.klPenalty);
            RlhfCutRun refCut = runRlhfCut(host);
            RlhfCutRun devCut = runRlhfCut(gpu);
            assertClose(api + " RLHF PPO loss of a bootstrapped buffer", refCut.ppoLoss, devCut.ppoLoss);
        }

        @Test
        public void trainedUpdates() {
            TrainedA2c a2cRef = runA2cTrained(host);
            TrainedA2c a2cDev = runA2cTrained(gpu);
            assertCloseAll(api + " A2C probabilities after the step", a2cRef.probeProbsAfter, a2cDev.probeProbsAfter);
            assertClose(api + " A2C value after the step", a2cRef.probeValueAfter, a2cDev.probeValueAfter);
            TrainedPpo ppoRef = runPpoTrained(host);
            TrainedPpo ppoDev = runPpoTrained(gpu);
            assertCloseAll(api + " PPO probabilities after the step", ppoRef.probeProbsAfter, ppoDev.probeProbsAfter);
            assertClose(api + " PPO value after the step", ppoRef.probeValueAfter, ppoDev.probeValueAfter);
            TrainedDqn dqnRef = runDqnTrained(host);
            TrainedDqn dqnDev = runDqnTrained(gpu);
            assertCloseAll(api + " DQN Q-values after the step", dqnRef.qAfter, dqnDev.qAfter);
            TrainedSac sacRef = runSacTrained(host);
            TrainedSac sacDev = runSacTrained(gpu);
            assertClose(api + " SAC critic loss", sacRef.criticLoss, sacDev.criticLoss);
            for (int i = 0; i < sacRef.actorAfter.length; i++) {
                assertCloseAll(api + " SAC actor parameter " + i, sacRef.actorAfter[i], sacDev.actorAfter[i]);
            }
            for (int i = 0; i < sacRef.criticAfter.length; i++) {
                assertCloseAll(api + " SAC critic parameter " + i, sacRef.criticAfter[i], sacDev.criticAfter[i]);
            }
            assertClose(api + " SAC log alpha", sacRef.logAlphaAfter, sacDev.logAlphaAfter);
            TrainedRlhf rlhfRef = runRlhfTrained(host);
            TrainedRlhf rlhfDev = runRlhfTrained(gpu);
            assertClose(api + " RLHF reward after the step", rlhfRef.probeRewardAfter, rlhfDev.probeRewardAfter);
            assertCloseAll(api + " RLHF policy output after the step", rlhfRef.policyAfter, rlhfDev.policyAfter);
        }

        @Test
        public void speculativeGreedyDecoding() {
            try (LastTokenModel onHost = new LastTokenModel(VOCAB);
                 PerPositionModel onGpu = new PerPositionModel(VOCAB);
                 LastTokenModel draftOnGpu = new LastTokenModel(VOCAB)) {
                onHost.initialize(host, SEED);
                onGpu.initialize(gpu, SEED);
                draftOnGpu.initialize(gpu, SEED + 1);
                SpeculativeDecoderConfig greedy = new SpeculativeDecoderConfig(3, 1.0f, 0.0f);
                float[] prompt = {2.0f, 5.0f};
                float[] expected = greedyDecode(onHost, host, prompt, 9);
                float[] actual = new SpeculativeDecoder(draftOnGpu, onGpu, greedy, SEED).generate(prompt, 9);
                assertArrayEquals(api + " greedy tokens", expected, actual, 0.0f);
            }
        }
    }

    // ------------------------------------------------------------------ scenarios (any backend)

    private static final class DqnRun {
        float firstLoss;
        float secondLoss;
        float[] qState;
        float[] qNext;
    }

    /** Four identical transitions, so every sampled batch is the same; the target syncs after every update. */
    private static DqnRun runDqn(ComputeBackend b, boolean done) {
        DqnRun run = new DqnRun();
        try (ReplayBuffer buffer = new ReplayBuffer(8, STATE, b);
             DQN dqn = DQN.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).hiddenSize(8)
                     .replayBuffer(buffer).gamma(0.9f).batchSize(4).targetUpdateFrequency(1).seed(SEED).build()) {
            for (int i = 0; i < 4; i++) {
                buffer.add(S, 1, 0.5f, S_NEXT, done);
            }
            run.firstLoss = dqn.trainStep();
            run.qState = dqn.getAllQValues(S);
            run.qNext = dqn.getAllQValues(S_NEXT);
            run.secondLoss = dqn.trainStep();
            assertEquals(run.qState[1], dqn.getQValue(S, 1), 0.0f);
        }
        return run;
    }

    private static final class A2cRun {
        float loss;
        float[][] probs;
        float[] values;
        float lastValue;
    }

    private static A2cRun runA2c(ComputeBackend b) {
        A2cRun run = new A2cRun();
        int n = ROLLOUT_STATES.length;
        try (A2C a2c = A2C.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).nSteps(n).gamma(0.9f)
                .valueCoeff(0.5f).entropyCoeff(0.01f).seed(SEED).build()) {
            run.probs = new float[n][];
            run.values = new float[n];
            for (int i = 0; i < n; i++) {
                run.probs[i] = a2c.getActionProbs(ROLLOUT_STATES[i]);
                run.values[i] = a2c.getValue(ROLLOUT_STATES[i]);
                a2c.storeTransition(ROLLOUT_STATES[i], ROLLOUT_ACTIONS[i], ROLLOUT_REWARDS[i], S_NEXT, false);
            }
            run.lastValue = a2c.getValue(S_NEXT);
            run.loss = a2c.trainStep();
            assertEquals(0, a2c.rolloutSize());
        }
        return run;
    }

    private static final float[] PPO_VALUES = {0.1f, -0.2f, 0.3f};

    private static final class PpoRun {
        float loss;
        float[][] probs;
        float[] values;
        float[] oldLogProbs;
        boolean lastDone;
        float nextValue;
    }

    /**
     * Stored log-probabilities 0.5 below the current ones, so every ratio is e^0.5 and clipping engages; the last step
     * ends its episode when {@code lastDone}, otherwise the trajectory is cut there and {@code nextValue} is the
     * critic's value of its next state.
     */
    private static PpoRun runPpo(ComputeBackend b, boolean lastDone) {
        PpoRun run = new PpoRun();
        run.lastDone = lastDone;
        int n = ROLLOUT_STATES.length;
        try (PPO ppo = PPO.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).batchSize(n).epochs(2)
                .gamma(0.9f).lambda(0.8f).clipEpsilon(0.2f).valueCoeff(0.5f).entropyCoeff(0.01f).seed(SEED)
                .build()) {
            run.probs = new float[n][];
            run.values = new float[n];
            run.oldLogProbs = new float[n];
            for (int i = 0; i < n; i++) {
                run.probs[i] = ppo.getActionProbs(ROLLOUT_STATES[i]);
                run.values[i] = ppo.getValue(ROLLOUT_STATES[i]);
                run.oldLogProbs[i] = (float) Math.log(run.probs[i][ROLLOUT_ACTIONS[i]]) - 0.5f;
                ppo.storeTransition(ROLLOUT_STATES[i], ROLLOUT_ACTIONS[i], ROLLOUT_REWARDS[i], S_NEXT,
                        lastDone && i == n - 1, run.oldLogProbs[i], PPO_VALUES[i]);
            }
            run.nextValue = ppo.getValue(S_NEXT);
            run.loss = ppo.trainStep();
        }
        return run;
    }

    private static final class SacRun {
        float[] explore;
        float[] greedy;
        float firstLoss;
        float secondLoss;
        float q1;
    }

    private static SacRun runSac(ComputeBackend b) {
        SacRun run = new SacRun();
        try (ReplayBuffer buffer = new ReplayBuffer(16, STATE, b);
             SAC sac = SAC.builder().backend(b).stateSize(STATE).actionSize(2).replayBuffer(buffer).batchSize(4)
                     .gamma(0.9f).tau(0.5f).entropyAlpha(0.2f).seed(SEED).build()) {
            for (int i = 0; i < 6; i++) {
                float[] action = {i % 2 == 0 ? 0.6f : -0.3f, i % 2 == 0 ? -0.2f : 0.7f};
                int row = i % ROLLOUT_STATES.length;
                sac.storeTransition(ROLLOUT_STATES[row], action, ROLLOUT_REWARDS[row], S_NEXT, i == 5);
            }
            run.explore = sac.selectAction(S, true);
            run.greedy = sac.selectAction(S, false);
            run.firstLoss = sac.trainStep();
            run.secondLoss = sac.trainStep();
            run.q1 = sac.getQ1Value(S, run.greedy);
        }
        return run;
    }

    private static final float[] RLHF_REWARDS = {0.7f, -0.4f};
    private static final float[] RLHF_VALUES = {0.2f, 0.5f};
    private static final float RLHF_OLD_LOG_PROB = -1.5f;
    /** The caller's value of the last next state of {@link #runRlhfCut}. */
    private static final float RLHF_NEXT_VALUE = 0.6f;

    private static final class RlhfRun {
        float rewardLoss;
        float ppoLoss;
        float[] preferredRewards = new float[2];
        float[] rejectedRewards = new float[2];
        float[][] policyLogits = new float[2][];
        float[][] referenceLogits = new float[2][];
        float klPenalty;
    }

    private static RlhfRun runRlhf(ComputeBackend b) {
        RlhfRun run = new RlhfRun();
        try (Linear reward = new Linear(STATE, 1);
             Linear policy = new Linear(STATE, VOCAB);
             Linear reference = new Linear(STATE, VOCAB)) {
            reward.initialize(b, new Random(SEED));
            policy.initialize(b, new Random(SEED + 1));
            reference.initialize(b, new Random(SEED + 2));
            try (RLHFTrainer trainer = RLHFTrainer.builder().backend(b).policyModel(policy).rewardModel(reward)
                    .referenceModel(reference).batchSize(2).ppoEpochs(1).klCoeff(0.1f).clipEpsilon(0.2f)
                    .entropyCoeff(0.01f).gamma(0.9f).lambda(0.8f).seed(SEED).build()) {
                float[][] preferred = {ROLLOUT_STATES[0], ROLLOUT_STATES[1]};
                float[][] rejected = {ROLLOUT_STATES[2], S};
                for (int k = 0; k < 2; k++) {
                    trainer.addPreferencePair(S_NEXT, preferred[k], rejected[k]);
                    run.preferredRewards[k] = trainer.getReward(preferred[k]);
                    run.rejectedRewards[k] = trainer.getReward(rejected[k]);
                }
                run.rewardLoss = trainer.trainRewardModel(1);

                float[][] states = {S, S_NEXT};
                for (int k = 0; k < 2; k++) {
                    run.policyLogits[k] = forward(policy, b, states[k]);
                    run.referenceLogits[k] = forward(reference, b, states[k]);
                    trainer.storePPOTransition(states[k], new float[]{k + 1}, RLHF_REWARDS[k], k == 1,
                            RLHF_OLD_LOG_PROB, RLHF_VALUES[k]);
                }
                run.ppoLoss = trainer.trainPPOStep();
                run.klPenalty = trainer.lastKLPenalty();
            }
        }
        return run;
    }

    private static final class RlhfCutRun {
        float ppoLoss;
        boolean refusedWithout;
        int bufferAfterRefusal;
        float[][] policyLogits = new float[2][];
        float[][] referenceLogits = new float[2][];
    }

    /**
     * {@link #runRlhf}'s PPO step with neither step ending its episode: {@code trainPPOStep()} must refuse the buffer
     * and keep it, and {@code trainPPOStep(RLHF_NEXT_VALUE)} bootstraps from the given value.
     */
    private static RlhfCutRun runRlhfCut(ComputeBackend b) {
        RlhfCutRun run = new RlhfCutRun();
        try (Linear reward = new Linear(STATE, 1);
             Linear policy = new Linear(STATE, VOCAB);
             Linear reference = new Linear(STATE, VOCAB)) {
            reward.initialize(b, new Random(SEED));
            policy.initialize(b, new Random(SEED + 1));
            reference.initialize(b, new Random(SEED + 2));
            try (RLHFTrainer trainer = RLHFTrainer.builder().backend(b).policyModel(policy).rewardModel(reward)
                    .referenceModel(reference).batchSize(2).ppoEpochs(1).klCoeff(0.1f).clipEpsilon(0.2f)
                    .entropyCoeff(0.01f).gamma(0.9f).lambda(0.8f).seed(SEED).build()) {
                float[][] states = {S, S_NEXT};
                for (int k = 0; k < 2; k++) {
                    run.policyLogits[k] = forward(policy, b, states[k]);
                    run.referenceLogits[k] = forward(reference, b, states[k]);
                    trainer.storePPOTransition(states[k], new float[]{k + 1}, RLHF_REWARDS[k], false,
                            RLHF_OLD_LOG_PROB, RLHF_VALUES[k]);
                }
                try {
                    trainer.trainPPOStep();
                } catch (IllegalStateException e) {
                    run.refusedWithout = e.getMessage().contains("inside an episode");
                }
                run.bufferAfterRefusal = trainer.ppoBufferSize();
                run.ppoLoss = trainer.trainPPOStep(RLHF_NEXT_VALUE);
            }
        }
        return run;
    }

    // ------------------------------------------------------------------ trained scenarios (SGD through the factories)

    private static final float LR = 0.5f;
    /** The parameter tensors of one perceptron: three layers' weight and bias. */
    private static final int MLP_PARAMETERS = 6;
    /** The trained DQN scenario's batch: that many identical terminal transitions. */
    private static final int DQN_BATCH = 4;
    private static final float SAC_LR = 0.05f;
    private static final float[][] RLHF_PREFERRED = {ROLLOUT_STATES[0], ROLLOUT_STATES[1]};
    private static final float[][] RLHF_REJECTED = {ROLLOUT_STATES[2], S};

    /** An SGD factory that records the parameter groups it is given, in call order. */
    private static final class SgdFactory implements Function<List<Parameter>, Optimizer> {
        private final float lr;
        private final List<List<Parameter>> groups = new ArrayList<>();

        SgdFactory(float lr) {
            this.lr = lr;
        }

        @Override
        public Optimizer apply(List<Parameter> params) {
            groups.add(params);
            return new SGD(params, lr);
        }
    }

    private static final class TrainedA2c {
        float[][] probs;
        float[] values;
        float lastValue;
        float[] probeProbsBefore;
        float probeValueBefore;
        float[] probeProbsAfter;
        float probeValueAfter;
        int graphBefore;
        int graphAfter;
        int groups;
        HostMlp actor;
        HostMlp critic;
    }

    /** {@link #runA2c} with an SGD step; the probe is {@code S}. */
    private static TrainedA2c runA2cTrained(ComputeBackend b) {
        TrainedA2c run = new TrainedA2c();
        int n = ROLLOUT_STATES.length;
        SgdFactory factory = new SgdFactory(LR);
        try (A2C a2c = A2C.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).nSteps(n).gamma(0.9f)
                .valueCoeff(0.5f).entropyCoeff(0.01f).seed(SEED).optimizer(factory).build()) {
            run.probs = new float[n][];
            run.values = new float[n];
            for (int i = 0; i < n; i++) {
                run.probs[i] = a2c.getActionProbs(ROLLOUT_STATES[i]);
                run.values[i] = a2c.getValue(ROLLOUT_STATES[i]);
                a2c.storeTransition(ROLLOUT_STATES[i], ROLLOUT_ACTIONS[i], ROLLOUT_REWARDS[i], S_NEXT, false);
            }
            run.lastValue = a2c.getValue(S_NEXT);
            run.probeProbsBefore = a2c.getActionProbs(S);
            run.probeValueBefore = a2c.getValue(S);
            run.actor = HostMlp.of(factory.groups.get(0), 0);
            run.critic = HostMlp.of(factory.groups.get(0), MLP_PARAMETERS);
            run.graphBefore = ComputationGraph.current().size();
            a2c.trainStep();
            run.graphAfter = ComputationGraph.current().size();
            run.probeProbsAfter = a2c.getActionProbs(S);
            run.probeValueAfter = a2c.getValue(S);
            run.groups = factory.groups.size();
        }
        return run;
    }

    private static final class TrainedPpo {
        float[][] probs;
        float[] values;
        float[] oldLogProbs;
        float[] probeProbsBefore;
        float probeValueBefore;
        float[] probeProbsAfter;
        float probeValueAfter;
        int graphBefore;
        int graphAfter;
        HostMlp actor;
        HostMlp critic;
    }

    /** {@link #runPpo} with one epoch and an SGD step; the probe is {@code S}. */
    private static TrainedPpo runPpoTrained(ComputeBackend b) {
        TrainedPpo run = new TrainedPpo();
        int n = ROLLOUT_STATES.length;
        SgdFactory factory = new SgdFactory(LR);
        try (PPO ppo = PPO.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).batchSize(n).epochs(1)
                .gamma(0.9f).lambda(0.8f).clipEpsilon(0.2f).valueCoeff(0.5f).entropyCoeff(0.01f).seed(SEED)
                .optimizer(factory).build()) {
            run.probs = new float[n][];
            run.values = new float[n];
            run.oldLogProbs = new float[n];
            for (int i = 0; i < n; i++) {
                run.probs[i] = ppo.getActionProbs(ROLLOUT_STATES[i]);
                run.values[i] = ppo.getValue(ROLLOUT_STATES[i]);
                run.oldLogProbs[i] = (float) Math.log(run.probs[i][ROLLOUT_ACTIONS[i]]) - 0.5f;
                ppo.storeTransition(ROLLOUT_STATES[i], ROLLOUT_ACTIONS[i], ROLLOUT_REWARDS[i], S_NEXT, i == n - 1,
                        run.oldLogProbs[i], PPO_VALUES[i]);
            }
            run.probeProbsBefore = ppo.getActionProbs(S);
            run.probeValueBefore = ppo.getValue(S);
            run.actor = HostMlp.of(factory.groups.get(0), 0);
            run.critic = HostMlp.of(factory.groups.get(0), MLP_PARAMETERS);
            run.graphBefore = ComputationGraph.current().size();
            ppo.trainStep();
            run.graphAfter = ComputationGraph.current().size();
            run.probeProbsAfter = ppo.getActionProbs(S);
            run.probeValueAfter = ppo.getValue(S);
        }
        return run;
    }

    private static final class TrainedDqn {
        float[] qBefore;
        float[] qAfter;
        float loss;
        int graphBefore;
        int graphAfter;
        HostMlp qNetwork;
    }

    /** Four identical terminal transitions (target = reward), one SGD step, no target sync. */
    private static TrainedDqn runDqnTrained(ComputeBackend b) {
        TrainedDqn run = new TrainedDqn();
        SgdFactory factory = new SgdFactory(LR);
        try (ReplayBuffer buffer = new ReplayBuffer(8, STATE, b);
             DQN dqn = DQN.builder().backend(b).stateSize(STATE).actionSize(ACTIONS).hiddenSize(8)
                     .replayBuffer(buffer).gamma(0.9f).batchSize(DQN_BATCH).targetUpdateFrequency(100).seed(SEED)
                     .optimizer(factory).build()) {
            for (int i = 0; i < DQN_BATCH; i++) {
                buffer.add(S, 1, 0.5f, S_NEXT, true);
            }
            run.qBefore = dqn.getAllQValues(S);
            run.qNetwork = HostMlp.of(factory.groups.get(0), 0);
            run.graphBefore = ComputationGraph.current().size();
            run.loss = dqn.trainStep();
            run.graphAfter = ComputationGraph.current().size();
            run.qAfter = dqn.getAllQValues(S);
        }
        return run;
    }

    private static final class TrainedSac {
        float criticLoss;
        float[][] actorBefore;
        float[][] actorAfter;
        float[][] criticBefore;
        float[][] criticAfter;
        float logAlphaBefore;
        float logAlphaAfter;
        float alphaAfter;
        boolean[] criticGradNull;
        boolean[] actorHasGrad;
        boolean alphaHasGrad;
        int graphBefore;
        int graphAfter;
    }

    /**
     * {@link #runSac}'s transitions, one step with SGD for the actor, critics and (through the actor factory) alpha.
     */
    private static TrainedSac runSacTrained(ComputeBackend b) {
        TrainedSac run = new TrainedSac();
        SgdFactory actorFactory = new SgdFactory(SAC_LR);
        SgdFactory criticFactory = new SgdFactory(SAC_LR);
        try (ReplayBuffer buffer = new ReplayBuffer(16, STATE, b);
             SAC sac = SAC.builder().backend(b).stateSize(STATE).actionSize(2).replayBuffer(buffer).batchSize(4)
                     .gamma(0.9f).tau(0.5f).entropyAlpha(0.2f).autoEntropy(true).seed(SEED)
                     .actorOptimizer(actorFactory).criticOptimizer(criticFactory).build()) {
            assertEquals("actor and temperature groups", 2, actorFactory.groups.size());
            assertEquals("critic group", 1, criticFactory.groups.size());
            List<Parameter> actor = actorFactory.groups.get(0);
            Parameter logAlpha = actorFactory.groups.get(1).get(0);
            List<Parameter> critics = criticFactory.groups.get(0);
            for (int i = 0; i < 6; i++) {
                float[] action = {i % 2 == 0 ? 0.6f : -0.3f, i % 2 == 0 ? -0.2f : 0.7f};
                int row = i % ROLLOUT_STATES.length;
                sac.storeTransition(ROLLOUT_STATES[row], action, ROLLOUT_REWARDS[row], S_NEXT, i == 5);
            }
            run.actorBefore = values(actor);
            run.criticBefore = values(critics);
            run.logAlphaBefore = logAlpha.tensor().toFloatArray()[0];
            run.graphBefore = ComputationGraph.current().size();
            run.criticLoss = sac.trainStep();
            run.graphAfter = ComputationGraph.current().size();
            run.criticGradNull = new boolean[critics.size()];
            for (int i = 0; i < critics.size(); i++) {
                run.criticGradNull[i] = critics.get(i).grad() == null;
            }
            run.actorHasGrad = new boolean[actor.size()];
            for (int i = 0; i < actor.size(); i++) {
                run.actorHasGrad[i] = actor.get(i).grad() != null;
            }
            run.alphaHasGrad = logAlpha.grad() != null;
            run.actorAfter = values(actor);
            run.criticAfter = values(critics);
            run.logAlphaAfter = logAlpha.tensor().toFloatArray()[0];
            run.alphaAfter = sac.entropyCoefficient();
        }
        return run;
    }

    private static final class TrainedRlhf {
        float[] preferredRewards = new float[2];
        float[] rejectedRewards = new float[2];
        float probeRewardBefore;
        float probeRewardAfter;
        float[] policyBefore;
        float[] policyAfter;
        float[] referenceBefore;
        float[] referenceAfter;
        boolean referenceRequiresGrad;
        boolean referenceGradNull;
        int graphBefore;
        int graphAfterReward;
        int graphAfterPpo;
    }

    /** One reward-model SGD step on two pairs (probe {@code S_NEXT}), then one PPO epoch with an SGD step. */
    private static TrainedRlhf runRlhfTrained(ComputeBackend b) {
        TrainedRlhf run = new TrainedRlhf();
        try (Linear reward = new Linear(STATE, 1);
             Linear policy = new Linear(STATE, VOCAB);
             Linear reference = new Linear(STATE, VOCAB)) {
            reward.initialize(b, new Random(SEED));
            policy.initialize(b, new Random(SEED + 1));
            reference.initialize(b, new Random(SEED + 2));
            SGD rewardOptimizer = new SGD(reward.parameters(), LR);
            SGD policyOptimizer = new SGD(policy.parameters(), LR);
            try (RLHFTrainer trainer = RLHFTrainer.builder().backend(b).policyModel(policy).rewardModel(reward)
                    .referenceModel(reference).policyOptimizer(policyOptimizer).rewardOptimizer(rewardOptimizer)
                    .batchSize(2).ppoEpochs(1).klCoeff(0.1f).clipEpsilon(0.2f).entropyCoeff(0.01f).gamma(0.9f)
                    .lambda(0.8f).seed(SEED).build()) {
                for (int k = 0; k < 2; k++) {
                    trainer.addPreferencePair(S_NEXT, RLHF_PREFERRED[k], RLHF_REJECTED[k]);
                    run.preferredRewards[k] = trainer.getReward(RLHF_PREFERRED[k]);
                    run.rejectedRewards[k] = trainer.getReward(RLHF_REJECTED[k]);
                }
                run.probeRewardBefore = trainer.getReward(S_NEXT);
                run.graphBefore = ComputationGraph.current().size();
                trainer.trainRewardModel(1);
                run.graphAfterReward = ComputationGraph.current().size();
                run.probeRewardAfter = trainer.getReward(S_NEXT);

                run.policyBefore = forward(policy, b, S);
                run.referenceBefore = forward(reference, b, S);
                float[][] states = {S, S_NEXT};
                for (int k = 0; k < 2; k++) {
                    trainer.storePPOTransition(states[k], new float[]{k + 1}, RLHF_REWARDS[k], k == 1,
                            RLHF_OLD_LOG_PROB, RLHF_VALUES[k]);
                }
                trainer.trainPPOStep();
                run.graphAfterPpo = ComputationGraph.current().size();
                run.policyAfter = forward(policy, b, S);
                run.referenceAfter = forward(reference, b, S);
                run.referenceRequiresGrad = false;
                run.referenceGradNull = true;
                for (Parameter p : reference.parameters()) {
                    run.referenceRequiresGrad |= p.requiresGrad();
                    run.referenceGradNull &= p.grad() == null;
                }
            } finally {
                rewardOptimizer.close();
                policyOptimizer.close();
            }
        }
        return run;
    }

    // ------------------------------------------------------------------ token models

    /** Logits {@code w * token} for the last token only: one row, whatever the sequence length. */
    static final class LastTokenModel extends Module {
        private final Linear head;

        LastTokenModel(int vocab) {
            this.head = registerModule("head", new Linear(1, vocab));
        }

        void initialize(ComputeBackend b, long seed) {
            head.initialize(b, new Random(seed));
        }

        @Override
        public Tensor forward(Tensor input) {
            int len = input.shape().dim(1);
            try (Tensor last = input.slice(1, len - 1, len)) {
                return head.forward(last);
            }
        }
    }

    /** Logits {@code w * token} for every position: one row per input token. */
    static final class PerPositionModel extends Module {
        private final Linear head;

        PerPositionModel(int vocab) {
            this.head = registerModule("head", new Linear(1, vocab));
        }

        void initialize(ComputeBackend b, long seed) {
            head.initialize(b, new Random(seed));
        }

        @Override
        public Tensor forward(Tensor input) {
            int len = input.shape().dim(1);
            try (Tensor column = input.reshape(Shape.matrix(len, 1))) {
                return head.forward(column);
            }
        }
    }

    /** Plain greedy decoding: each next token is the first maximal logit of the model's last row. */
    private static float[] greedyDecode(Module model, ComputeBackend b, float[] prompt, int steps) {
        float[] tokens = Arrays.copyOf(prompt, prompt.length + steps);
        for (int i = prompt.length; i < tokens.length; i++) {
            float[] logits = forward(model, b, Arrays.copyOf(tokens, i));
            float[] last = Arrays.copyOfRange(logits, logits.length - VOCAB, logits.length);
            int best = 0;
            float bestValue = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < last.length; v++) {
                if (last[v] > bestValue) {
                    bestValue = last[v];
                    best = v;
                }
            }
            tokens[i] = best;
        }
        return tokens;
    }

    // ------------------------------------------------------------------ helpers

    /** {@code model(row)} read back, under {@code noGrad} (an inspection that records nothing). */
    private static float[] forward(Module model, ComputeBackend b, float[] row) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = Tensor.fromArray(b, new float[][]{row});
             Tensor output = model.forward(input)) {
            return output.toFloatArray();
        }
    }

    private static float[][] values(List<Parameter> params) {
        float[][] out = new float[params.size()][];
        for (int i = 0; i < params.size(); i++) {
            out[i] = params.get(i).tensor().toFloatArray();
        }
        return out;
    }

    private static double dot(float[] a, float[] b) {
        double d = 0.0;
        for (int i = 0; i < a.length; i++) {
            d += (double) a[i] * b[i];
        }
        return d;
    }

    /** {@code (a - mean) / sqrt(var + 1e-8)} with the population variance. */
    private static double[] normalize(double[] raw) {
        double mean = Arrays.stream(raw).sum() / raw.length;
        double var = 0.0;
        for (double a : raw) {
            var += (a - mean) * (a - mean);
        }
        double std = Math.sqrt(var / raw.length + 1e-8);
        double[] out = new double[raw.length];
        for (int i = 0; i < raw.length; i++) {
            out[i] = (raw[i] - mean) / std;
        }
        return out;
    }

    /**
     * Asserts that the host copies of an agent's actor and critic reproduce its probabilities and value at the probe
     * before the step, so the predicted update starts from the agent's own networks.
     */
    private static void assertReferenceMatches(String what, HostMlp actor, HostMlp critic, float[] probeProbs,
                                               float probeValue) {
        assertCloseAll(what + " host reference of the actor", softmax(actor.forward(S)), probeProbs);
        assertClose(what + " host reference of the critic", critic.forward(S)[0], probeValue);
    }

    /**
     * PPO's generalized advantages and returns ({@code gamma = 0.9}, {@code lambda = 0.8}) for the rollout rewards and
     * {@link #PPO_VALUES}: the last step ends its episode when {@code lastDone}, otherwise the scan starts from
     * {@code nextValue}, the value of its next state.
     */
    private static void gae(double[] adv, double[] ret, boolean lastDone, double nextValue) {
        double[] rewards = new double[adv.length];
        double[] values = new double[adv.length];
        for (int i = 0; i < adv.length; i++) {
            rewards[i] = ROLLOUT_REWARDS[i];
            values[i] = PPO_VALUES[i];
        }
        double[] advantages = generalizedAdvantages(rewards, values, lastDone, nextValue);
        for (int i = 0; i < adv.length; i++) {
            adv[i] = advantages[i];
            ret[i] = advantages[i] + values[i];
        }
    }

    /**
     * Generalized advantages ({@code gamma = 0.9}, {@code lambda = 0.8}) of a trajectory in which only the last step
     * may end its episode ({@code lastDone}); otherwise the scan starts from {@code nextValue}.
     */
    private static double[] generalizedAdvantages(double[] rewards, double[] values, boolean lastDone,
                                                  double nextValue) {
        int n = rewards.length;
        double[] adv = new double[n];
        double lastValue = nextValue;
        double lastAdv = 0.0;
        for (int i = n - 1; i >= 0; i--) {
            boolean done = lastDone && i == n - 1;
            double delta = rewards[i] + (done ? 0.0 : 0.9 * lastValue) - values[i];
            adv[i] = delta + 0.9 * 0.8 * (done ? 0.0 : lastAdv);
            lastValue = values[i];
            lastAdv = adv[i];
        }
        return adv;
    }

    /** The mean per-step PPO loss of {@link #runPpo} from the definition, evaluated in double. */
    private static double expectedPpoLoss(PpoRun run) {
        int n = ROLLOUT_STATES.length;
        double[] adv = new double[n];
        double[] ret = new double[n];
        gae(adv, ret, run.lastDone, run.nextValue);
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            double logP = Math.log(Math.max(run.probs[i][ROLLOUT_ACTIONS[i]], 1e-10));
            double ratio = Math.exp(logP - run.oldLogProbs[i]);
            double clipped = Math.max(0.8, Math.min(1.2, ratio));
            double policy = -Math.min(ratio * adv[i], clipped * adv[i]);
            double valueError = ret[i] - run.values[i];
            total += policy + 0.5 * valueError * valueError - 0.01 * entropy(run.probs[i], 0.0);
        }
        return total / n;
    }

    /**
     * The RLHF scenarios' per-step KL penalties {@code 0.1 * (logp_old - log pi_ref(token))}, with step {@code k}'s
     * action the token {@code k + 1}.
     */
    private static double[] rlhfKlPenalties(float[][] referenceLogits) {
        double[] penalties = new double[referenceLogits.length];
        for (int k = 0; k < penalties.length; k++) {
            double refLogP = Math.log(Math.max(softmax(referenceLogits[k])[k + 1], 1e-10));
            penalties[k] = 0.1 * (RLHF_OLD_LOG_PROB - refLogP);
        }
        return penalties;
    }

    /**
     * The RLHF scenarios' mean PPO step loss {@code -min(rA, clip(r) A) - 0.01 H}, with the advantages estimated from
     * the rewards minus their KL penalties (bootstrapped from {@code nextValue} unless the last step is done).
     */
    private static double expectedRlhfPpoLoss(float[][] policyLogits, float[][] referenceLogits, boolean lastDone,
                                              double nextValue) {
        int n = policyLogits.length;
        double[] penalties = rlhfKlPenalties(referenceLogits);
        double[] rewards = new double[n];
        double[] values = new double[n];
        for (int k = 0; k < n; k++) {
            rewards[k] = RLHF_REWARDS[k] - penalties[k];
            values[k] = RLHF_VALUES[k];
        }
        double[] adv = generalizedAdvantages(rewards, values, lastDone, nextValue);
        double total = 0.0;
        for (int k = 0; k < n; k++) {
            double[] p = softmax(policyLogits[k]);
            double logP = Math.log(Math.max(p[k + 1], 1e-10));
            double ratio = Math.exp(logP - RLHF_OLD_LOG_PROB);
            double clipped = Math.max(0.8, Math.min(1.2, ratio));
            double policy = -Math.min(ratio * adv[k], clipped * adv[k]);
            total += policy - 0.01 * entropy(toFloats(p), 1e-10);
        }
        return total / n;
    }

    private static double max(float[] values) {
        double best = Double.NEGATIVE_INFINITY;
        for (float v : values) {
            best = Math.max(best, v);
        }
        return best;
    }

    private static double[] softmax(float[] logits) {
        double[] values = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            values[i] = logits[i];
        }
        return softmax(values);
    }

    private static double[] softmax(double[] logits) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double v : logits) {
            maxLogit = Math.max(maxLogit, v);
        }
        double sum = 0.0;
        double[] p = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            p[i] = Math.exp(logits[i] - maxLogit);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= sum;
        }
        return p;
    }

    private static float[] toFloats(double[] values) {
        float[] out = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = (float) values[i];
        }
        return out;
    }

    private static double entropy(float[] probs, double threshold) {
        double h = 0.0;
        for (float p : probs) {
            if (p > threshold) {
                h -= p * Math.log(p);
            }
        }
        return h;
    }

    private static void assertClose(String what, double ref, float actual) {
        double bound = REL * Math.max(1.0, Math.abs(ref));
        if (!(Math.abs(actual - ref) <= bound)) {
            fail(what + ": expected " + ref + ", got " + actual + " (bound " + bound + ")");
        }
    }

    private static void assertCloseAll(String what, double[] ref, float[] actual) {
        assertEquals(what + ": length", ref.length, actual.length);
        for (int i = 0; i < ref.length; i++) {
            assertClose(what + " [" + i + "]", ref[i], actual[i]);
        }
    }

    private static void assertCloseAll(String what, float[] ref, float[] actual) {
        assertEquals(what + ": length", ref.length, actual.length);
        for (int i = 0; i < ref.length; i++) {
            assertClose(what + " [" + i + "]", ref[i], actual[i]);
        }
    }
}

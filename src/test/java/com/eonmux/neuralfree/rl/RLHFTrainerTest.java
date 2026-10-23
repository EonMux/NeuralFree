package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

public class RLHFTrainerTest {
    
    private ComputeBackend backend;
    private MockModule policyModel;
    private MockModule rewardModel;

    /** The trainer runs on the CPU reference backend, whose buffers hold what is written to them. */
    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
        policyModel = new MockModule(64, 32);
        rewardModel = new MockModule(64, 1);
        
        policyModel.initialize(backend, new Random(42));
        rewardModel.initialize(backend, new Random(42));
    }

    @After
    public void tearDown() {
        try {
            policyModel.close();
            rewardModel.close();
        } finally {
            backend.close();
        }
    }
    
    @Test
    public void testRLHFTrainerCreation() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        assertNotNull(trainer);
        assertEquals(backend, trainer.backend());
        assertEquals(policyModel, trainer.policyModel());
        assertEquals(rewardModel, trainer.rewardModel());
        assertNotNull(trainer.referenceModel());
        assertEquals(0, trainer.preferenceDataSize());
        assertEquals(0, trainer.ppoBufferSize());
    }
    
    @Test
    public void testRLHFTrainerBuilderWithCustomParams() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .klCoeff(0.05f)
            .clipEpsilon(0.3f)
            .entropyCoeff(0.02f)
            .gamma(0.95f)
            .lambda(0.9f)
            .ppoEpochs(8)
            .batchSize(32)
            .seed(12345)
            .build();
        
        assertNotNull(trainer);
    }
    
    @Test
    public void testRLHFTrainerDefaultReferenceModel() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        assertNotNull(trainer);
        assertEquals(policyModel, trainer.referenceModel());
    }
    
    @Test(expected = NullPointerException.class)
    public void testRLHFTrainerNullBackendRejected() {
        RLHFTrainer.builder()
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
    }
    
    @Test(expected = NullPointerException.class)
    public void testRLHFTrainerNullPolicyModelRejected() {
        RLHFTrainer.builder()
            .backend(backend)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
    }
    
    @Test(expected = NullPointerException.class)
    public void testRLHFTrainerNullRewardModelRejected() {
        RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .referenceModel(policyModel)
            .build();
    }
    
    @Test
    public void testAddPreferencePair() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        float[] prompt = new float[]{1.0f, 2.0f, 3.0f};
        float[] preferred = new float[]{4.0f, 5.0f};
        float[] rejected = new float[]{6.0f, 7.0f};
        
        trainer.addPreferencePair(prompt, preferred, rejected);
        
        assertEquals(1, trainer.preferenceDataSize());
    }
    
    @Test
    public void testAddPreferencePairsList() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        List<RLHFTrainer.PreferencePair> pairs = new ArrayList<>();
        pairs.add(new RLHFTrainer.PreferencePair(
            new float[]{1.0f}, new float[]{2.0f}, new float[]{3.0f}));
        pairs.add(new RLHFTrainer.PreferencePair(
            new float[]{4.0f}, new float[]{5.0f}, new float[]{6.0f}));
        
        trainer.addPreferencePairs(pairs);
        
        assertEquals(2, trainer.preferenceDataSize());
    }
    
    @Test(expected = NullPointerException.class)
    public void testAddPreferencePairNullPromptRejected() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.addPreferencePair(null, new float[]{1.0f}, new float[]{2.0f});
    }
    
    @Test(expected = NullPointerException.class)
    public void testAddPreferencePairNullPreferredRejected() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.addPreferencePair(new float[]{1.0f}, null, new float[]{2.0f});
    }
    
    @Test(expected = NullPointerException.class)
    public void testAddPreferencePairNullRejectedRejected() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.addPreferencePair(new float[]{1.0f}, new float[]{2.0f}, null);
    }
    
    @Test
    public void testClearPreferenceData() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.addPreferencePair(new float[]{1.0f}, new float[]{2.0f}, new float[]{3.0f});
        assertEquals(1, trainer.preferenceDataSize());
        
        trainer.clearPreferenceData();
        assertEquals(0, trainer.preferenceDataSize());
    }
    
    @Test
    public void testTrainRewardModelEmptyData() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        float loss = trainer.trainRewardModel(1);
        assertEquals(0.0f, loss, 0.0f);
    }
    
    @Test
    public void testTrainRewardModelWithPreferenceData() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .batchSize(2)
            .build();
        
        for (int i = 0; i < 4; i++) {
            float[] prompt = new float[64];
            float[] preferred = new float[64];
            float[] rejected = new float[64];
            for (int j = 0; j < 64; j++) {
                prompt[j] = (float) (j + 1 + i * 64);
                preferred[j] = (float) (j + 1 + i * 64 + 100);
                rejected[j] = (float) (j + 1 + i * 64 + 200);
            }
            trainer.addPreferencePair(prompt, preferred, rejected);
        }
        
        float loss = trainer.trainRewardModel(1);
        assertTrue(loss >= 0.0f || Float.isNaN(loss));
    }
    
    @Test
    public void testStorePPOTransition() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        trainer.storePPOTransition(
            new float[]{1.0f, 2.0f},
            new float[]{3.0f, 4.0f},
            1.0f,
            false,
            0.0f,
            0.5f
        );
        
        assertEquals(1, trainer.ppoBufferSize());
    }
    
    @Test
    public void testTrainPPOStepEmptyBuffer() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        float loss = trainer.trainPPOStep();
        assertEquals(0.0f, loss, 0.0f);
    }
    
    @Test
    public void testTrainPPOStepWithBuffer() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .batchSize(2)
            .ppoEpochs(1)
            .build();
        
        for (int i = 0; i < 4; i++) {
            float[] state = new float[64];
            float[] action = new float[32];
            for (int j = 0; j < 64; j++) state[j] = (float) (j + 1 + i * 64);
            for (int j = 0; j < 32; j++) action[j] = (float) (j + 1 + i * 32);
            
            trainer.storePPOTransition(
                state,
                action,
                1.0f,
                false,
                -0.5f,
                0.5f
            );
        }
        
        float loss = trainer.trainPPOStep(0.5f);
        assertTrue(Float.isFinite(loss) || loss == 0.0f);
        assertEquals(0, trainer.ppoBufferSize());
    }
    
    @Test
    public void testClearPPOBuffer() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        trainer.storePPOTransition(
            new float[]{1.0f}, new float[]{2.0f}, 1.0f, false, 0.0f, 0.5f
        );
        
        assertEquals(1, trainer.ppoBufferSize());
        
        trainer.clearPPOBuffer();
        assertEquals(0, trainer.ppoBufferSize());
    }
    
    @Test
    public void testGetReward() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        float[] tokens = new float[64];
        for (int i = 0; i < 64; i++) {
            tokens[i] = (float) (i + 1);
        }
        float reward = trainer.getReward(tokens);
        
        assertTrue(Float.isFinite(reward));
    }
    
    @Test(expected = NullPointerException.class)
    public void testGetRewardNullTokensRejected() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(policyModel)
            .build();
        
        trainer.getReward(null);
    }
    
    @Test
    public void testGenerate() {
        MockAutoregressiveModel genModel = new MockAutoregressiveModel(32);
        genModel.initialize(backend, new Random(42));
        
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(genModel)
            .rewardModel(rewardModel)
            .referenceModel(genModel)
            .build();
        
        float[] prompt = new float[32];
        for (int i = 0; i < 32; i++) {
            prompt[i] = (float) (i + 1);
        }
        genModel.train();
        float[] generated = trainer.generate(genModel, prompt, 3, 1.0f);
        
        assertNotNull(generated);
        assertEquals(32 + 3, generated.length);
        for (int i = 32; i < generated.length; i++) {
            float t = generated[i];
            assertTrue("token id " + t + " is a vocabulary index of the last row", t >= 0 && t < 32 && t == Math.rint(t));
        }
        assertTrue("the model's training mode is restored", genModel.isTraining());
    }
    
    @Test
    public void testLastLossAccessors() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        assertEquals(0.0f, trainer.lastRewardLoss(), 0.0f);
        assertEquals(0.0f, trainer.lastPolicyLoss(), 0.0f);
        assertEquals(0.0f, trainer.lastKLPenalty(), 0.0f);
    }
    
    @Test
    public void testPreferencePairImmutability() {
        float[] prompt = new float[]{1.0f, 2.0f};
        float[] preferred = new float[]{3.0f, 4.0f};
        float[] rejected = new float[]{5.0f, 6.0f};
        
        RLHFTrainer.PreferencePair pair = new RLHFTrainer.PreferencePair(
            prompt, preferred, rejected);
        
        prompt[0] = 999.0f;
        preferred[0] = 888.0f;
        rejected[0] = 777.0f;
        
        assertNotEquals(999.0f, pair.promptTokens()[0], 0.0f);
        assertNotEquals(888.0f, pair.preferredTokens()[0], 0.0f);
        assertNotEquals(777.0f, pair.rejectedTokens()[0], 0.0f);
    }
    
    @Test
    public void testToString() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .klCoeff(0.02f)
            .build();
        
        String str = trainer.toString();
        assertNotNull(str);
        assertTrue(str.contains("RLHFTrainer"));
        assertTrue(str.contains("preferenceDataSize=0"));
        assertTrue(str.contains("klCoeff=0.02"));
    }
    
    @Test
    public void testClose() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.close();
        
        try {
            trainer.addPreferencePair(new float[]{1.0f}, new float[]{2.0f}, new float[]{3.0f});
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }
    
    @Test
    public void testOperationsAfterCloseThrow() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.close();
        
        try {
            trainer.getReward(new float[]{1.0f});
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
        
        try {
            trainer.trainRewardModel(1);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
        
        try {
            trainer.trainPPOStep();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }
    
    @Test
    public void testCloseIsIdempotent() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        
        trainer.close();
        trainer.close();
    }
    
    @Test
    public void testReferenceModelIsFrozen() {
        MockModule separateRefModel = new MockModule(64, 32);
        separateRefModel.initialize(backend, new Random(99));
        
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .referenceModel(separateRefModel)
            .build();
        
        Module refModel = trainer.referenceModel();
        assertNotNull(refModel);
        assertFalse(refModel.isTraining());
        for (Parameter p : refModel.parameters()) {
            assertFalse("a separate reference model is frozen", p.requiresGrad());
        }
        for (Parameter p : policyModel.parameters()) {
            assertTrue("the policy stays trainable", p.requiresGrad());
        }
        
        trainer.close();
        separateRefModel.close();
    }
    
    @Test
    public void testDefaultReferenceModelLeavesThePolicyTrainable() {
        policyModel.train();
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        for (Parameter p : policyModel.parameters()) {
            assertTrue("the policy is its own reference and stays trainable", p.requiresGrad());
        }
        assertTrue("the policy keeps its training mode", policyModel.isTraining());
        trainer.close();
    }
    
    @Test
    public void testGetRewardRestoresTheTrainingModeAndRecordsNothing() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .build();
        rewardModel.train();
        int before = ComputationGraph.current().size();
        float[] tokens = new float[64];
        Arrays.fill(tokens, 0.01f);
        assertTrue(Float.isFinite(trainer.getReward(tokens)));
        assertTrue(rewardModel.isTraining());
        assertEquals("getReward records no node", before, ComputationGraph.current().size());
        trainer.close();
    }
    
    @Test
    public void testTrainingStepsUpdateTheModelsAndLeaveNoNodes() {
        float[] probe = new float[64];
        for (int j = 0; j < 64; j++) {
            probe[j] = 0.01f * (j % 7) - 0.03f;
        }
        Adam policyOptimizer = new Adam(policyModel.parameters(), 1e-2f);
        Adam rewardOptimizer = new Adam(rewardModel.parameters(), 1e-2f);
        try (MockModule reference = new MockModule(64, 32)) {
            reference.initialize(backend, new Random(7));
            RLHFTrainer trainer = RLHFTrainer.builder()
                .backend(backend)
                .policyModel(policyModel)
                .rewardModel(rewardModel)
                .referenceModel(reference)
                .policyOptimizer(policyOptimizer)
                .rewardOptimizer(rewardOptimizer)
                .batchSize(2)
                .ppoEpochs(2)
                .seed(42)
                .build();
            for (int i = 0; i < 2; i++) {
                float[] preferred = new float[64];
                float[] rejected = new float[64];
                for (int j = 0; j < 64; j++) {
                    preferred[j] = 0.02f * ((i + j) % 5);
                    rejected[j] = -0.02f * ((i + 2 * j) % 3);
                }
                trainer.addPreferencePair(probe, preferred, rejected);
            }
            int before = ComputationGraph.current().size();
            float rewardBefore = trainer.getReward(probe);
            float rewardLoss = trainer.trainRewardModel(1);
            assertTrue(Float.isFinite(rewardLoss) && rewardLoss >= 0.0f);
            assertNotEquals("the reward model was updated", rewardBefore, trainer.getReward(probe), 0.0f);
            assertEquals("reward training leaves no node", before, ComputationGraph.current().size());
            
            float[] policyBefore = outputs(policyModel, probe);
            float[] referenceBefore = outputs(reference, probe);
            for (int i = 0; i < 2; i++) {
                float[] state = new float[64];
                Arrays.fill(state, 0.01f * (i + 1));
                trainer.storePPOTransition(state, new float[]{i + 3}, 1.0f - i, i == 1, -3.0f, 0.2f);
            }
            float ppoLoss = trainer.trainPPOStep();
            assertTrue(Float.isFinite(ppoLoss));
            assertTrue(Float.isFinite(trainer.lastKLPenalty()));
            assertEquals("PPO training leaves no node", before, ComputationGraph.current().size());
            assertFalse("the policy was updated", Arrays.equals(policyBefore, outputs(policyModel, probe)));
            assertArrayEquals("the frozen reference is unchanged", referenceBefore, outputs(reference, probe), 0.0f);
            for (Parameter p : reference.parameters()) {
                assertNull("the frozen reference holds no gradient", p.grad());
            }
            trainer.close();
        } finally {
            policyOptimizer.close();
            rewardOptimizer.close();
        }
    }
    
    @Test
    public void testPolicyAuxiliaryLossIsAddedAndTheReferencePassIsDropped() {
        try (AuxModule policy = new AuxModule(8, 6)) {
            policy.initialize(backend, new Random(5));
            RLHFTrainer trainer = RLHFTrainer.builder()
                .backend(backend)
                .policyModel(policy)
                .rewardModel(rewardModel)
                .batchSize(2)
                .ppoEpochs(1)
                .seed(42)
                .build();
            float[][] states = {{0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f, 0.2f, 0.4f},
                                {-0.3f, 0.1f, 0.0f, 0.2f, -0.4f, 0.3f, 0.1f, -0.2f}};
            
            policy.auxScale = 0.0f;
            storeSteps(trainer, states);
            float plain = trainer.trainPPOStep();
            
            policy.auxScale = 0.5f;
            storeSteps(trainer, states);
            float withAux = trainer.trainPPOStep();
            
            double auxMean = 0.0;
            for (float[] state : states) {
                double sum = 0.0;
                for (float v : outputs(policy, state)) {
                    sum += v;
                }
                auxMean += 0.5 * sum / states.length;
            }
            assertEquals("the policy's auxiliary terms join the loss; the reference pass's are dropped",
                    plain + auxMean, withAux, 1e-4 * Math.max(1.0, Math.abs(withAux)));
            assertNull("no auxiliary term is left pending", policy.takeAuxiliaryLoss());
            trainer.close();
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testPolicyOutputMustSplitIntoOneRowPerActionToken() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .batchSize(1)
            .ppoEpochs(1)
            .build();
        try {
            trainer.storePPOTransition(new float[64], new float[]{1, 2, 3, 4, 5}, 1.0f, true, -1.0f, 0.0f);
            trainer.trainPPOStep();
        } finally {
            trainer.close();
        }
    }
    
    @Test
    public void testABufferEndingInsideAnEpisodeNeedsTheNextValue() {
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .batchSize(2)
            .ppoEpochs(1)
            .build();
        float[] state = new float[64];
        Arrays.fill(state, 0.02f);
        trainer.storePPOTransition(state, new float[]{3}, 1.0f, false, -3.0f, 0.2f);
        trainer.storePPOTransition(state, new float[]{4}, 0.5f, false, -3.0f, 0.1f);
        try {
            trainer.trainPPOStep();
            fail("a buffer that ends inside an episode needs the value of its last next state");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("inside an episode"));
        }
        assertEquals("the refused buffer is kept", 2, trainer.ppoBufferSize());
        try {
            trainer.trainPPOStep(Float.NaN);
            fail("a non-finite next value is refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("finite"));
        }
        assertTrue(Float.isFinite(trainer.trainPPOStep(0.3f)));
        assertEquals(0, trainer.ppoBufferSize());
        trainer.close();
    }
    
    /**
     * With the stored log-probabilities equal to the policy's, every ratio is 1 inside the clip range, so the loss is
     * {@code -mean(A)} and raising the next value by 1 lowers it by {@code gamma (1 + gamma lambda) / 2} for two
     * steps; when the last step ends its episode the next value is unused.
     */
    @Test
    public void testTheNextValueBootstrapsTheAdvantages() {
        float[][] states = new float[2][64];
        Arrays.fill(states[0], 0.03f);
        Arrays.fill(states[1], -0.02f);
        float[] tokens = {5, 9};
        float[] logProbs = new float[2];
        for (int i = 0; i < 2; i++) {
            float[] logits = outputs(policyModel, states[i]);
            logProbs[i] = (float) Math.log(softmaxAt(logits, (int) tokens[i]));
        }
        RLHFTrainer trainer = RLHFTrainer.builder()
            .backend(backend)
            .policyModel(policyModel)
            .rewardModel(rewardModel)
            .batchSize(2)
            .ppoEpochs(1)
            .entropyCoeff(0.0f)
            .gamma(0.9f)
            .lambda(0.8f)
            .build();
        float[] losses = new float[4];
        float[] nextValues = {0.0f, 1.0f, 0.0f, 1.0f};
        for (int run = 0; run < 4; run++) {
            boolean lastDone = run >= 2;
            for (int i = 0; i < 2; i++) {
                trainer.storePPOTransition(states[i], new float[]{tokens[i]}, 0.5f, lastDone && i == 1, logProbs[i],
                        0.1f);
            }
            losses[run] = trainer.trainPPOStep(nextValues[run]);
        }
        assertEquals("the next value bootstraps a buffer cut inside an episode", -0.9 * (1 + 0.9 * 0.8) / 2,
                losses[1] - losses[0], 1e-4);
        assertEquals("the next value is unused when the last step ends its episode", losses[2], losses[3], 0.0f);
        trainer.close();
    }
    
    /**
     * The KL penalty is a constant subtracted from the rewards: with the policy as its own reference and the stored
     * log-probabilities {@code delta} above the policy's, each penalty is {@code klCoeff * delta}, the logged mean
     * penalty is that value, and the loss equals the loss without the penalty on rewards lowered by it.
     */
    @Test
    public void testTheKlPenaltyEntersThroughTheRewards() {
        float[][] states = new float[2][64];
        Arrays.fill(states[0], 0.01f);
        Arrays.fill(states[1], 0.04f);
        float[] tokens = {2, 7};
        float delta = 0.05f;
        float[] logProbs = new float[2];
        for (int i = 0; i < 2; i++) {
            logProbs[i] = (float) Math.log(softmaxAt(outputs(policyModel, states[i]), (int) tokens[i])) + delta;
        }
        float klCoeff = 0.5f;
        float[] rewards = {0.3f, -0.2f};
        RLHFTrainer penalized = RLHFTrainer.builder().backend(backend).policyModel(policyModel)
            .rewardModel(rewardModel).batchSize(2).ppoEpochs(1).klCoeff(klCoeff).build();
        RLHFTrainer plain = RLHFTrainer.builder().backend(backend).policyModel(policyModel)
            .rewardModel(rewardModel).batchSize(2).ppoEpochs(1).klCoeff(0.0f).build();
        for (int i = 0; i < 2; i++) {
            penalized.storePPOTransition(states[i], new float[]{tokens[i]}, rewards[i], i == 1, logProbs[i], 0.1f);
            plain.storePPOTransition(states[i], new float[]{tokens[i]}, rewards[i] - klCoeff * delta, i == 1,
                    logProbs[i], 0.1f);
        }
        float penalizedLoss = penalized.trainPPOStep();
        float plainLoss = plain.trainPPOStep();
        assertEquals("the logged penalty is klCoeff * (logp_old - logp_ref)", klCoeff * delta,
                penalized.lastKLPenalty(), 1e-5f);
        assertEquals("the penalty acts as a reward reduction", plainLoss, penalizedLoss, 1e-5f);
        assertEquals(0.0f, plain.lastKLPenalty(), 1e-6f);
        penalized.close();
        plain.close();
    }
    
    @Test
    public void testStoredStepsAreCopies() {
        float[] state = new float[64];
        Arrays.fill(state, 0.02f);
        float[] tokens = {6};
        float logProb = (float) Math.log(softmaxAt(outputs(policyModel, state), 6));
        RLHFTrainer reused = RLHFTrainer.builder().backend(backend).policyModel(policyModel)
            .rewardModel(rewardModel).batchSize(1).ppoEpochs(1).build();
        RLHFTrainer fresh = RLHFTrainer.builder().backend(backend).policyModel(policyModel)
            .rewardModel(rewardModel).batchSize(1).ppoEpochs(1).build();
        reused.storePPOTransition(state, tokens, 0.4f, true, logProb, 0.1f);
        fresh.storePPOTransition(state.clone(), tokens.clone(), 0.4f, true, logProb, 0.1f);
        Arrays.fill(state, -0.5f);
        tokens[0] = 1;
        assertEquals("changing the caller's arrays after storing changes nothing", fresh.trainPPOStep(),
                reused.trainPPOStep(), 0.0f);
        reused.close();
        fresh.close();
    }
    
    /** {@code softmax(logits)[index]} in double. */
    private static double softmaxAt(float[] logits, int index) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0.0;
        for (float v : logits) {
            sum += Math.exp(v - max);
        }
        return Math.exp(logits[index] - max) / sum;
    }
    
    private static void storeSteps(RLHFTrainer trainer, float[][] states) {
        for (int i = 0; i < states.length; i++) {
            trainer.storePPOTransition(states[i], new float[]{i + 1}, 0.5f - i, i == states.length - 1, -1.5f, 0.1f);
        }
    }
    
    /** {@code model(row)} read back under {@code noGrad}; any auxiliary term the pass leaves is dropped. */
    private float[] outputs(Module model, float[] row) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor input = Tensor.fromArray(backend, new float[][]{row});
             Tensor output = model.forward(input)) {
            try (Tensor aux = model.takeAuxiliaryLoss()) {
                return output.toFloatArray();
            }
        }
    }
    
    /** A linear model whose forward pass leaves {@code auxScale * sum(output)} as its auxiliary loss when non-zero. */
    private static final class AuxModule extends Module {
        private final Linear linear;
        float auxScale;
        
        AuxModule(int inputSize, int outputSize) {
            this.linear = registerModule("linear", new Linear(inputSize, outputSize));
        }
        
        void initialize(ComputeBackend backend, Random random) {
            linear.initialize(backend, random);
        }
        
        @Override
        public Tensor forward(Tensor input) {
            Tensor out = linear.forward(input);
            if (auxScale != 0.0f) {
                try (Tensor sum = TensorOps.sum(out)) {
                    setAuxiliaryLoss(TensorOps.mul(sum, auxScale));
                }
            }
            return out;
        }
    }
    
    private static class MockModule extends Module {
        private final Linear linear;
        private final int inputSize;
        private final int outputSize;
        
        MockModule(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.linear = new Linear(inputSize, outputSize);
            registerModule("linear", linear);
        }
        
        void initialize(ComputeBackend backend, Random random) {
            linear.initialize(backend, random);
        }
        
        @Override
        public Tensor forward(Tensor input) {
            return linear.forward(input);
        }
    }
    
    private static class MockAutoregressiveModel extends Module {
        private final int vocabSize;
        private ComputeBackend cachedBackend;
        private Random cachedRandom;
        
        MockAutoregressiveModel(int vocabSize) {
            this.vocabSize = vocabSize;
        }
        
        void initialize(ComputeBackend backend, Random random) {
            this.cachedBackend = backend;
            this.cachedRandom = random;
        }
        
        @Override
        public Tensor forward(Tensor input) {
            long seqLen = input.shape().dim(1);
            float[] logits = new float[(int) (seqLen * vocabSize)];
            for (int i = 0; i < logits.length; i++) {
                logits[i] = cachedRandom.nextFloat() * 2.0f - 1.0f;
            }
            try (Tensor flat = Tensor.fromArray(cachedBackend, logits)) {
                return flat.reshape(Shape.matrix((int) seqLen, vocabSize));
            }
        }
    }
}

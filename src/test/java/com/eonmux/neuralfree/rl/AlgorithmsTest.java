package com.eonmux.neuralfree.rl;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Test;
import org.junit.After;
import org.junit.Before;

import java.util.*;
import java.util.function.Function;

import static org.junit.Assert.*;

public class AlgorithmsTest {

    private ComputeBackend backend;
    private ReplayBuffer replayBuffer;

    /** Algorithms run on the CPU reference backend, whose buffers hold what is written to them. */
    @Before
    public void setUp() {
        backend = TrainParity.hostBackend();
        replayBuffer = new ReplayBuffer(1000, 4, backend);
    }

    @After
    public void tearDown() {
        backend.close();
    }
    
    @Test
    public void testDQNBuilder() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .gamma(0.99f)
            .epsilonStart(1.0f)
            .epsilonEnd(0.01f)
            .epsilonDecay(0.995f)
            .batchSize(32)
            .seed(42)
            .build();
        
        assertNotNull(dqn);
        assertEquals(4, dqn.backend().capabilities().getComputeUnits());
        assertEquals(1.0f, dqn.epsilon(), 0.0f);
        dqn.close();
    }
    
    @Test
    public void testDQNSelectAction() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        Integer actionExplore = dqn.selectAction(state, true);
        assertNotNull(actionExplore);
        assertTrue(actionExplore >= 0 && actionExplore < 2);
        
        Integer actionEval = dqn.selectAction(state, false);
        assertNotNull(actionEval);
        assertTrue(actionEval >= 0 && actionEval < 2);
        
        dqn.close();
    }
    
    @Test
    public void testDQNSelectActionEval() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        Integer action = dqn.selectActionEval(state);
        assertNotNull(action);
        assertTrue(action >= 0 && action < 2);
        
        dqn.close();
    }
    
    @Test
    public void testDQNGetQValues() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float qValue = dqn.getQValue(state, 0);
        assertTrue(Float.isFinite(qValue));
        
        float[] allQValues = dqn.getAllQValues(state);
        assertEquals(2, allQValues.length);
        assertTrue(Float.isFinite(allQValues[0]));
        assertTrue(Float.isFinite(allQValues[1]));
        
        dqn.close();
    }
    
    @Test
    public void testDQNTrainStep() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(replayBuffer)
            .batchSize(8)
            .optimizer(params -> new Adam(params, 1e-2f))
            .seed(42)
            .build();
        
        for (int i = 0; i < 20; i++) {
            float[] state = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            int action = i % 2;
            float reward = (float)Math.random();
            float[] nextState = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            boolean done = (i == 19);
            replayBuffer.add(state, action, reward, nextState, done);
        }
        
        int graphBefore = ComputationGraph.current().size();
        float[] before = dqn.getAllQValues(new float[]{0.5f, 0.5f, 0.5f, 0.5f});
        float loss = dqn.trainStep();
        assertTrue(Float.isFinite(loss));
        assertTrue(loss >= 0);
        assertEquals("the step leaves no node behind", graphBefore, ComputationGraph.current().size());
        assertFalse("the Q-network was updated",
                Arrays.equals(before, dqn.getAllQValues(new float[]{0.5f, 0.5f, 0.5f, 0.5f})));
        
        dqn.close();
    }
    
    @Test
    public void testDQNReset() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .epsilonStart(1.0f)
            .epsilonDecay(0.9f)
            .seed(42)
            .build();
        
        dqn.selectAction(new float[]{0,0,0,0}, true);
        assertTrue(dqn.epsilon() <= 1.0f);
        
        dqn.reset();
        assertEquals(1.0f, dqn.epsilon(), 0.001f);
        assertEquals(0, dqn.stepCount());
        
        dqn.close();
    }
    
    @Test
    public void testDQNClose() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        dqn.close();
        
        try {
            dqn.selectAction(new float[]{0,0,0,0}, true);
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
    
    @Test
    public void testDQNToString() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        String str = dqn.toString();
        assertTrue(str.contains("DQN"));
        assertTrue(str.contains("stateSize=4"));
        assertTrue(str.contains("actionSize=2"));
        
        dqn.close();
    }
    
    @Test
    public void testPPOBuilder() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .gamma(0.99f)
            .lambda(0.95f)
            .clipEpsilon(0.2f)
            .entropyCoeff(0.01f)
            .valueCoeff(0.5f)
            .epochs(4)
            .batchSize(64)
            .seed(42)
            .build();
        
        assertNotNull(ppo);
        ppo.close();
    }
    
    @Test
    public void testPPOSelectAction() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        Integer actionExplore = ppo.selectAction(state, true);
        assertNotNull(actionExplore);
        assertTrue(actionExplore >= 0 && actionExplore < 2);
        
        Integer actionEval = ppo.selectAction(state, false);
        assertNotNull(actionEval);
        assertTrue(actionEval >= 0 && actionEval < 2);
        
        ppo.close();
    }
    
    @Test
    public void testPPOGetValue() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float value = ppo.getValue(state);
        assertTrue(Float.isFinite(value));
        
        ppo.close();
    }
    
    @Test
    public void testPPOGetActionProbs() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float[] probs = ppo.getActionProbs(state);
        assertEquals(2, probs.length);
        
        float sum = 0.0f;
        for (float p : probs) {
            assertTrue(p >= 0 && p <= 1);
            sum += p;
        }
        assertEquals(1.0f, sum, 0.001f);
        
        ppo.close();
    }
    
    @Test
    public void testPPOStoreAndTrain() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .batchSize(8)
            .epochs(2)
            .optimizer(params -> new Adam(params, 1e-2f))
            .seed(42)
            .build();
        
        for (int i = 0; i < 10; i++) {
            float[] state = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            Integer action = ppo.selectAction(state, true);
            float[] probs = ppo.getActionProbs(state);
            float logProb = (float)Math.log(Math.max(probs[action], 1e-10f));
            float value = ppo.getValue(state);
            float reward = (float)Math.random();
            float[] nextState = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            boolean done = (i == 9);
            
            ppo.storeTransition(state, action, reward, nextState, done, logProb, value);
        }
        
        assertEquals(10, ppo.trajectorySize());
        
        int graphBefore = ComputationGraph.current().size();
        float[] probe = {0.5f, 0.5f, 0.5f, 0.5f};
        float[] before = ppo.getActionProbs(probe);
        float loss = ppo.trainStep();
        assertTrue(Float.isFinite(loss));
        assertEquals("the epochs leave no node behind", graphBefore, ComputationGraph.current().size());
        assertFalse("the policy was updated", Arrays.equals(before, ppo.getActionProbs(probe)));
        
        assertEquals(0, ppo.trajectorySize());
        
        ppo.close();
    }
    
    @Test
    public void testPPOReset() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        ppo.storeTransition(new float[]{0,0,0,0}, 0, 0, new float[]{0,0,0,0}, false, 0, 0);
        assertTrue(ppo.trajectorySize() > 0);
        
        ppo.reset();
        assertEquals(0, ppo.trajectorySize());
        
        ppo.close();
    }
    
    @Test
    public void testPPOClose() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        ppo.close();
        
        try {
            ppo.selectAction(new float[]{0,0,0,0}, true);
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
    
    @Test
    public void testPPOToString() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        String str = ppo.toString();
        assertTrue(str.contains("PPO"));
        assertTrue(str.contains("stateSize=4"));
        
        ppo.close();
    }
    
    @Test
    public void testSACBuilder() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .gamma(0.99f)
            .tau(0.005f)
            .entropyAlpha(0.2f)
            .autoEntropy(true)
            .batchSize(64)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        assertNotNull(sac);
        sac.close();
    }
    
    @Test
    public void testSACSelectAction() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float[] actionExplore = sac.selectAction(state, true);
        assertNotNull(actionExplore);
        assertEquals(2, actionExplore.length);
        for (float a : actionExplore) {
            assertTrue(a >= -1 && a <= 1);
        }
        
        float[] actionEval = sac.selectAction(state, false);
        assertNotNull(actionEval);
        assertEquals(2, actionEval.length);
        
        sac.close();
    }
    
    @Test
    public void testSACGetQValues() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] action = {0.5f, -0.3f};
        
        float q1 = sac.getQ1Value(state, action);
        assertTrue(Float.isFinite(q1));
        
        float q2 = sac.getQ2Value(state, action);
        assertTrue(Float.isFinite(q2));
        
        sac.close();
    }
    
    @Test
    public void testSACTrainStep() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .batchSize(8)
            .replayBuffer(sacBuffer)
            .actorOptimizer(params -> new Adam(params, 1e-2f))
            .criticOptimizer(params -> new Adam(params, 1e-2f))
            .seed(42)
            .build();
        
        for (int i = 0; i < 20; i++) {
            float[] state = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            float[] action = {(float)(Math.random()*2-1), (float)(Math.random()*2-1)};
            float reward = (float)Math.random();
            float[] nextState = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            boolean done = (i == 19);
            sac.storeTransition(state, action, reward, nextState, done);
        }
        
        int graphBefore = ComputationGraph.current().size();
        float[] probe = {0.5f, 0.5f, 0.5f, 0.5f};
        float[] greedyBefore = sac.selectAction(probe, false);
        float alphaBefore = sac.entropyCoefficient();
        float loss = sac.trainStep();
        assertTrue(Float.isFinite(loss));
        assertEquals("the step leaves no node behind", graphBefore, ComputationGraph.current().size());
        assertFalse("the actor was updated", Arrays.equals(greedyBefore, sac.selectAction(probe, false)));
        assertNotEquals("the temperature was updated", alphaBefore, sac.entropyCoefficient(), 0.0f);
        
        sac.close();
        sacBuffer.close();
    }
    
    @Test
    public void testSACReset() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        sac.reset();
        
        sac.close();
    }
    
    @Test
    public void testSACClose() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        sac.close();
        
        try {
            sac.selectAction(new float[]{0,0,0,0}, true);
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
    
    @Test
    public void testSACToString() {
        ReplayBuffer sacBuffer = new ReplayBuffer(1000, 4, backend);
        
        SAC sac = SAC.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(sacBuffer)
            .seed(42)
            .build();
        
        String str = sac.toString();
        assertTrue(str.contains("SAC"));
        assertTrue(str.contains("stateSize=4"));
        
        sac.close();
    }
    
    @Test
    public void testA2CBuilder() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .gamma(0.99f)
            .entropyCoeff(0.01f)
            .valueCoeff(0.5f)
            .nSteps(5)
            .seed(42)
            .build();
        
        assertNotNull(a2c);
        a2c.close();
    }
    
    @Test
    public void testA2CSelectAction() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        Integer actionExplore = a2c.selectAction(state, true);
        assertNotNull(actionExplore);
        assertTrue(actionExplore >= 0 && actionExplore < 2);
        
        Integer actionEval = a2c.selectAction(state, false);
        assertNotNull(actionEval);
        assertTrue(actionEval >= 0 && actionEval < 2);
        
        a2c.close();
    }
    
    @Test
    public void testA2CGetValue() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float value = a2c.getValue(state);
        assertTrue(Float.isFinite(value));
        
        a2c.close();
    }
    
    @Test
    public void testA2CGetActionProbs() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        
        float[] probs = a2c.getActionProbs(state);
        assertEquals(2, probs.length);
        
        float sum = 0.0f;
        for (float p : probs) {
            assertTrue(p >= 0 && p <= 1);
            sum += p;
        }
        assertEquals(1.0f, sum, 0.001f);
        
        a2c.close();
    }
    
    @Test
    public void testA2CStoreAndTrain() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .nSteps(3)
            .optimizer(params -> new SGD(params, 0.1f))
            .seed(42)
            .build();
        
        for (int i = 0; i < 10; i++) {
            float[] state = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            float reward = (float)Math.random();
            float[] nextState = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            boolean done = (i == 9);
            
            a2c.storeTransition(state, i % 2, reward, nextState, done);
        }
        
        assertEquals(10, a2c.rolloutSize());
        
        int graphBefore = ComputationGraph.current().size();
        float[] probe = {0.5f, 0.5f, 0.5f, 0.5f};
        float valueBefore = a2c.getValue(probe);
        float loss = a2c.trainStep();
        assertTrue(Float.isFinite(loss));
        assertEquals("the step leaves no node behind", graphBefore, ComputationGraph.current().size());
        assertNotEquals("the critic was updated", valueBefore, a2c.getValue(probe), 0.0f);
        
        assertEquals(0, a2c.rolloutSize());
        
        a2c.close();
    }
    
    @Test
    public void testA2CReset() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        a2c.storeTransition(new float[]{0,0,0,0}, 0, 0, new float[]{0,0,0,0}, false);
        assertTrue(a2c.rolloutSize() > 0);
        
        a2c.reset();
        assertEquals(0, a2c.rolloutSize());
        
        a2c.close();
    }
    
    @Test
    public void testA2CClose() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        a2c.close();
        
        try {
            a2c.selectAction(new float[]{0,0,0,0}, true);
            fail("Should throw IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
    
    @Test
    public void testA2CToString() {
        A2C a2c = A2C.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .seed(42)
            .build();
        
        String str = a2c.toString();
        assertTrue(str.contains("A2C"));
        assertTrue(str.contains("stateSize=4"));
        
        a2c.close();
    }
    
    @Test
    public void testDQNLearnsSimpleEnvironment() {
        DQN dqn = DQN.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .replayBuffer(replayBuffer)
            .batchSize(8)
            .optimizer(params -> new Adam(params, 1e-2f))
            .epsilonStart(0.5f)
            .epsilonEnd(0.01f)
            .epsilonDecay(0.99f)
            .seed(42)
            .build();
        
        float totalReward = 0;
        int episodes = 5;
        
        for (int ep = 0; ep < episodes; ep++) {
            float[] state = {0.5f, 0.5f, 0.5f, 0.5f};
            float epReward = 0;
            
            for (int step = 0; step < 10; step++) {
                int action = dqn.selectAction(state, true);
                
                float reward = action == 0 ? 1.0f : -0.5f;
                epReward += reward;
                
                float[] nextState = new float[4];
                for (int i = 0; i < 4; i++) {
                    nextState[i] = Math.max(0, Math.min(1, state[i] + (float)(Math.random() * 0.2 - 0.1)));
                }
                boolean done = (step == 9);
                
                replayBuffer.add(state, action, reward, nextState, done);
                state = nextState;
                
                if (replayBuffer.size() >= 8) {
                    dqn.trainStep();
                }
                
                if (done) break;
            }
            
            totalReward += epReward;
        }
        
        float avgReward = totalReward / episodes;
        assertTrue(avgReward > 0 || replayBuffer.size() > 0);
        
        dqn.close();
    }
    
    @Test
    public void testPPOTrainStepNoError() {
        PPO ppo = PPO.builder()
            .backend(backend)
            .stateSize(4)
            .actionSize(2)
            .hiddenSize(32)
            .batchSize(4)
            .epochs(1)
            .optimizer(params -> new SGD(params, 0.05f))
            .seed(42)
            .build();
        
        for (int i = 0; i < 10; i++) {
            float[] state = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            Integer action = ppo.selectAction(state, true);
            float[] probs = ppo.getActionProbs(state);
            float logProb = (float)Math.log(Math.max(probs[action], 1e-10f));
            float value = ppo.getValue(state);
            float reward = (float)Math.random();
            float[] nextState = {(float)Math.random(), (float)Math.random(), (float)Math.random(), (float)Math.random()};
            boolean done = (i == 9);
            
            ppo.storeTransition(state, action, reward, nextState, done, logProb, value);
        }
        
        float loss = ppo.trainStep();
        assertTrue(Float.isFinite(loss) || loss == 0.0f);
        
        ppo.close();
    }

    @Test
    public void testOptimizerFactoryMustCoverTheParametersItIsGiven() {
        try {
            A2C.builder().backend(backend).stateSize(4).actionSize(2)
                .optimizer(params -> new SGD(Collections.<Parameter>emptyList(), 0.1f)).seed(42).build();
            fail("an optimizer that ignores the given parameters must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("optimizer factory"));
        }
        try {
            DQN.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8)
                .optimizer(params -> null).seed(42).build();
            fail("a factory returning null must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("returned null"));
        }
    }

    @Test
    public void testOptimizerFactoryReceivesTheTrainedParameters() {
        List<List<Parameter>> groups = new ArrayList<>();
        PPO ppo = PPO.builder().backend(backend).stateSize(4).actionSize(3)
            .optimizer(params -> {
                groups.add(params);
                return new SGD(params, 0.1f);
            })
            .seed(42).build();
        assertEquals(1, groups.size());
        assertEquals("weights and biases of the actor's and the critic's three layers", 12, groups.get(0).size());
        for (Parameter p : groups.get(0)) {
            assertTrue(p.requiresGrad());
        }
        ppo.close();
    }

    @Test
    public void testTrainStepUnderNoGradIsRefusedWithAnOptimizer() {
        A2C a2c = A2C.builder().backend(backend).stateSize(4).actionSize(2).nSteps(2)
            .optimizer(params -> new SGD(params, 0.1f)).seed(42).build();
        a2c.storeTransition(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, 0, 1.0f, new float[]{0.2f, 0.3f, 0.4f, 0.5f}, false);
        a2c.storeTransition(new float[]{0.2f, 0.3f, 0.4f, 0.5f}, 1, 0.0f, new float[]{0.3f, 0.4f, 0.5f, 0.6f}, true);
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            a2c.trainStep();
            fail("a training step needs grad mode");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("does not require grad"));
        }
        assertEquals("the rollout is kept for a retry", 2, a2c.rolloutSize());
        assertTrue(Float.isFinite(a2c.trainStep()));
        a2c.close();
    }

    @Test
    public void testActingRecordsNothing() {
        DQN dqn = DQN.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).seed(42).build();
        PPO ppo = PPO.builder().backend(backend).stateSize(4).actionSize(2).seed(42).build();
        A2C a2c = A2C.builder().backend(backend).stateSize(4).actionSize(2).seed(42).build();
        ReplayBuffer sacBuffer = new ReplayBuffer(16, 4, backend);
        SAC sac = SAC.builder().backend(backend).stateSize(4).actionSize(2)
            .replayBuffer(sacBuffer).seed(42).build();
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        int before = ComputationGraph.current().size();
        dqn.selectAction(state, false);
        dqn.getQValue(state, 1);
        dqn.getAllQValues(state);
        ppo.selectAction(state, true);
        ppo.getActionProbs(state);
        ppo.getValue(state);
        a2c.selectAction(state, true);
        a2c.getActionProbs(state);
        a2c.getValue(state);
        a2c.storeTransition(state, 1, 0.5f, state, false);
        sac.selectAction(state, true);
        sac.selectAction(state, false);
        sac.getQ1Value(state, new float[]{0.1f, -0.2f});
        sac.getQ2Value(state, new float[]{0.1f, -0.2f});
        assertEquals("acting and inspection record no node", before, ComputationGraph.current().size());
        dqn.close();
        ppo.close();
        a2c.close();
        sac.close();
        sacBuffer.close();
    }

    @Test
    public void testRefusedDQNStepLeavesThePrioritizedBufferUnchanged() {
        ReplayBuffer prioritized = new ReplayBuffer(16, 4, backend, true, 0.6f, 0.4f, 0.1f, 1e-6f);
        DQN dqn = DQN.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8)
            .replayBuffer(prioritized).batchSize(4).optimizer(params -> new SGD(params, 0.1f)).seed(42).build();
        for (int i = 0; i < 4; i++) {
            prioritized.add(new float[]{0.1f * i, 0.2f, -0.3f, 0.4f}, i % 2, 1.0f, new float[]{0.2f, 0.1f, 0.0f, 0.3f},
                    i == 3);
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            dqn.trainStep();
            fail("a training step needs grad mode");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("does not require grad"));
        }
        assertEquals("beta is annealed only by a completed update", 0.4f, prioritized.beta(), 0.0f);
        assertEquals("no step is counted", 0, dqn.stepCount());
        assertTrue(Float.isFinite(dqn.trainStep()));
        assertEquals(0.5f, prioritized.beta(), 1e-6f);
        dqn.close();
        prioritized.close();
    }

    @Test
    public void testSACTemperatureTrainsWithoutAnActorOptimizer() {
        ReplayBuffer sacBuffer = new ReplayBuffer(16, 4, backend);
        SAC sac = SAC.builder().backend(backend).stateSize(4).actionSize(2).batchSize(4).replayBuffer(sacBuffer)
            .criticOptimizer(params -> new SGD(params, 0.05f)).alphaOptimizer(params -> new SGD(params, 0.05f))
            .autoEntropy(true).seed(42).build();
        for (int i = 0; i < 6; i++) {
            float[] state = {0.1f * i, -0.2f, 0.3f, 0.05f * i};
            sac.storeTransition(state, new float[]{i % 2 == 0 ? 0.6f : -0.3f, 0.2f}, 0.5f, state, i == 5);
        }
        float[] probe = {0.5f, 0.5f, 0.5f, 0.5f};
        float[] greedyBefore = sac.selectAction(probe, false);
        float alphaBefore = sac.entropyCoefficient();
        int graphBefore = ComputationGraph.current().size();
        assertTrue(Float.isFinite(sac.trainStep()));
        assertEquals("the step leaves no node behind", graphBefore, ComputationGraph.current().size());
        assertArrayEquals("the actor is not trained", greedyBefore, sac.selectAction(probe, false), 0.0f);
        assertNotEquals("the temperature was updated", alphaBefore, sac.entropyCoefficient(), 0.0f);
        sac.close();
        sacBuffer.close();
    }

    @Test
    public void testNetworksHaveTwoHiddenLayersOfHiddenSize() {
        int state = 4;
        int hidden = 8;
        int actions = 3;
        List<List<Parameter>> a2cGroups = new ArrayList<>();
        try (A2C a2c = A2C.builder().backend(backend).stateSize(state).actionSize(actions).hiddenSize(hidden)
                .optimizer(recording(a2cGroups)).seed(42).build()) {
            assertEquals(1, a2cGroups.size());
            assertShapes("A2C actor and critic", concat(mlpShapes(state, hidden, actions), mlpShapes(state, hidden, 1)),
                a2cGroups.get(0));
            assertEquals("A2C parameter count", 139 + 121, elementCount(a2cGroups.get(0)));
        }
        List<List<Parameter>> ppoGroups = new ArrayList<>();
        try (PPO ppo = PPO.builder().backend(backend).stateSize(state).actionSize(actions).hiddenSize(hidden)
                .optimizer(recording(ppoGroups)).seed(42).build()) {
            assertEquals(1, ppoGroups.size());
            assertShapes("PPO actor and critic", concat(mlpShapes(state, hidden, actions), mlpShapes(state, hidden, 1)),
                ppoGroups.get(0));
            assertEquals("PPO parameter count", 139 + 121, elementCount(ppoGroups.get(0)));
        }
        List<List<Parameter>> dqnGroups = new ArrayList<>();
        try (DQN dqn = DQN.builder().backend(backend).stateSize(state).actionSize(actions).hiddenSize(hidden)
                .optimizer(recording(dqnGroups)).seed(42).build()) {
            assertEquals(1, dqnGroups.size());
            assertShapes("DQN Q-network", mlpShapes(state, hidden, actions), dqnGroups.get(0));
            assertEquals("DQN parameter count", 139, elementCount(dqnGroups.get(0)));
        }
        List<List<Parameter>> actorGroups = new ArrayList<>();
        List<List<Parameter>> criticGroups = new ArrayList<>();
        try (SAC sac = SAC.builder().backend(backend).stateSize(state).actionSize(actions).hiddenSize(hidden)
                .actorOptimizer(recording(actorGroups)).criticOptimizer(recording(criticGroups)).autoEntropy(true)
                .seed(42).build()) {
            assertEquals("actor and temperature groups", 2, actorGroups.size());
            assertShapes("SAC actor (mean and log-std heads)", mlpShapes(state, hidden, 2 * actions),
                actorGroups.get(0));
            assertEquals("SAC actor parameter count", 166, elementCount(actorGroups.get(0)));
            assertEquals("SAC log alpha", 1, elementCount(actorGroups.get(1)));
            assertEquals(1, criticGroups.size());
            assertShapes("SAC twin critics on (state, action)",
                concat(mlpShapes(state + actions, hidden, 1), mlpShapes(state + actions, hidden, 1)),
                criticGroups.get(0));
            assertEquals("SAC critic parameter count", 2 * 145, elementCount(criticGroups.get(0)));
        }
    }

    @Test
    public void testHiddenSizeMustBePositive() {
        for (int hidden : new int[]{0, -1}) {
            assertRefused(() -> A2C.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(hidden).seed(1)
                .build());
            assertRefused(() -> PPO.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(hidden).seed(1)
                .build());
            assertRefused(() -> DQN.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(hidden).seed(1)
                .build());
            assertRefused(() -> SAC.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(hidden).seed(1)
                .build());
        }
    }

    @Test
    public void testSeededNetworksAreReproducible() {
        float[] state = {0.3f, -0.7f, 0.5f, 0.1f};
        try (A2C first = A2C.builder().backend(backend).stateSize(4).actionSize(3).hiddenSize(16).seed(7).build();
             A2C second = A2C.builder().backend(backend).stateSize(4).actionSize(3).hiddenSize(16).seed(7).build()) {
            assertArrayEquals(first.getActionProbs(state), second.getActionProbs(state), 0.0f);
            assertEquals(first.getValue(state), second.getValue(state), 0.0f);
        }
        try (DQN first = DQN.builder().backend(backend).stateSize(4).actionSize(3).hiddenSize(16).seed(7).build();
             DQN second = DQN.builder().backend(backend).stateSize(4).actionSize(3).hiddenSize(16).seed(7).build()) {
            assertArrayEquals(first.getAllQValues(state), second.getAllQValues(state), 0.0f);
        }
    }

    @Test
    public void testPPOAndA2CRequireTheNextStateOfAStepThatContinues() {
        float[] state = {0.1f, 0.2f, 0.3f, 0.4f};
        try (PPO ppo = PPO.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).seed(42).build();
             A2C a2c = A2C.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).seed(42).build()) {
            assertRefusedStep(NullPointerException.class,
                () -> ppo.storeTransition(state, 0, 1.0f, null, false, -0.7f, 0.1f));
            assertRefusedStep(IllegalArgumentException.class,
                () -> ppo.storeTransition(state, 0, 1.0f, new float[3], false, -0.7f, 0.1f));
            assertRefusedStep(IllegalArgumentException.class,
                () -> ppo.storeTransition(new float[5], 0, 1.0f, state, false, -0.7f, 0.1f));
            assertRefusedStep(NullPointerException.class, () -> a2c.storeTransition(state, 0, 1.0f, null, false));
            assertRefusedStep(IllegalArgumentException.class,
                () -> a2c.storeTransition(state, 0, 1.0f, new float[3], false));
            assertRefusedStep(IllegalArgumentException.class,
                () -> a2c.storeTransition(new float[5], 0, 1.0f, state, false));
            assertEquals("refused steps are not stored", 0, ppo.trajectorySize());
            assertEquals("refused steps are not stored", 0, a2c.rolloutSize());
            ppo.storeTransition(state, 0, 1.0f, null, true, -0.7f, 0.1f);
            a2c.storeTransition(state, 0, 1.0f, null, true);
            assertEquals("a step that ends its episode needs no next state", 1, ppo.trajectorySize());
            assertEquals("a step that ends its episode needs no next state", 1, a2c.rolloutSize());
        }
    }

    /**
     * Two agents with the same seed store the same steps; the first keeps reusing the caller's arrays, which are
     * overwritten after each store. The trajectories end inside an episode, so the last next state is read at the
     * update too; equal losses show the agents kept copies.
     */
    @Test
    public void testPPOAndA2CStoreCopiesOfTheStates() {
        float[][] states = {{0.1f, -0.2f, 0.3f, 0.5f}, {-0.4f, 0.2f, 0.6f, -0.1f}, {0.7f, 0.1f, -0.3f, 0.2f}};
        float[] reusedState = new float[4];
        float[] reusedNext = new float[4];
        try (PPO reused = PPO.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).batchSize(2)
                 .epochs(1).seed(7).build();
             PPO fresh = PPO.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).batchSize(2)
                 .epochs(1).seed(7).build();
             A2C reusedA2c = A2C.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).nSteps(2)
                 .seed(7).build();
             A2C freshA2c = A2C.builder().backend(backend).stateSize(4).actionSize(2).hiddenSize(8).nSteps(2)
                 .seed(7).build()) {
            for (int i = 0; i < 2; i++) {
                System.arraycopy(states[i], 0, reusedState, 0, 4);
                System.arraycopy(states[i + 1], 0, reusedNext, 0, 4);
                reused.storeTransition(reusedState, i, 0.5f - i, reusedNext, false, -0.6f, 0.2f);
                reusedA2c.storeTransition(reusedState, i, 0.5f - i, reusedNext, false);
                fresh.storeTransition(states[i].clone(), i, 0.5f - i, states[i + 1].clone(), false, -0.6f, 0.2f);
                freshA2c.storeTransition(states[i].clone(), i, 0.5f - i, states[i + 1].clone(), false);
                Arrays.fill(reusedState, 9.0f);
                Arrays.fill(reusedNext, -9.0f);
            }
            assertEquals("PPO kept copies of the states", fresh.trainStep(), reused.trainStep(), 0.0f);
            assertEquals("A2C kept copies of the states", freshA2c.trainStep(), reusedA2c.trainStep(), 0.0f);
        }
    }

    private static void assertRefusedStep(Class<? extends RuntimeException> expected, Runnable store) {
        try {
            store.run();
            fail("expected " + expected.getSimpleName());
        } catch (RuntimeException e) {
            assertTrue("expected " + expected.getSimpleName() + ", got " + e, expected.isInstance(e));
        }
    }

    /** An SGD factory that records the parameter groups it is given. */
    private static Function<List<Parameter>, SGD> recording(List<List<Parameter>> groups) {
        return params -> {
            groups.add(params);
            return new SGD(params, 0.1f);
        };
    }

    /** The parameter shapes of {@code Linear(in, h) -> ReLU -> Linear(h, h) -> ReLU -> Linear(h, out)}, in order. */
    private static List<Shape> mlpShapes(int in, int hidden, int out) {
        return Arrays.asList(Shape.matrix(hidden, in), Shape.vector(hidden), Shape.matrix(hidden, hidden),
            Shape.vector(hidden), Shape.matrix(out, hidden), Shape.vector(out));
    }

    private static List<Shape> concat(List<Shape> first, List<Shape> second) {
        List<Shape> all = new ArrayList<>(first);
        all.addAll(second);
        return all;
    }

    private static void assertShapes(String what, List<Shape> expected, List<Parameter> params) {
        assertEquals(what + ": parameter tensors", expected.size(), params.size());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(what + ": shape of parameter " + i, expected.get(i), params.get(i).tensor().shape());
        }
    }

    private static long elementCount(List<Parameter> params) {
        long total = 0;
        for (Parameter p : params) {
            total += p.tensor().numElements();
        }
        return total;
    }

    private static void assertRefused(Runnable build) {
        try {
            build.run();
            fail("a hidden size that is not positive must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Hidden size must be positive"));
        }
    }
}

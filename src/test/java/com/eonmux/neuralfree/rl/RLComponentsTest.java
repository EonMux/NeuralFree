package com.eonmux.neuralfree.rl;

import org.junit.Test;
import org.junit.Before;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.*;

public class RLComponentsTest {

    private static class MockEnvironment implements Environment<float[], Integer> {
        private float[] currentState;
        private int stepCount;
        private final int maxSteps;

        MockEnvironment(int maxSteps) {
            this.maxSteps = maxSteps;
            reset();
        }

        @Override
        public float[] reset() {
            currentState = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
            stepCount = 0;
            return currentState.clone();
        }

        @Override
        public StepResult<float[]> step(Integer action) {
            stepCount++;
            float reward = action == 0 ? 1.0f : -1.0f;
            boolean done = stepCount >= maxSteps;
            
            currentState[action % 4] += 1.0f;
            
            Map<String, Object> info = new HashMap<>();
            info.put("step", stepCount);
            
            return new StepResult<>(currentState.clone(), reward, done, info);
        }

        @Override
        public int actionSpaceSize() {
            return 4;
        }

        @Override
        public java.util.List<Integer> observationShape() {
            return java.util.List.of(4);
        }
    }

    private static class MockPolicy implements Policy<float[], Integer> {
        private final Random random = new Random(42);
        private boolean lastExplore;

        @Override
        public Integer selectAction(float[] state, boolean explore) {
            lastExplore = explore;
            if (explore) {
                return random.nextInt(4);
            }
            return 0;
        }

        @Override
        public void reset() {
            random.setSeed(42);
        }

        public boolean lastExplore() {
            return lastExplore;
        }
    }

    @Test
    public void testStepResultBasic() {
        float[] state = {1.0f, 2.0f};
        StepResult<float[]> result = new StepResult<>(state, 5.0f, false);

        assertArrayEquals(state, result.state(), 0.0f);
        assertEquals(5.0f, result.reward(), 0.0f);
        assertFalse(result.done());
        assertTrue(result.info().isEmpty());
    }

    @Test
    public void testStepResultWithInfo() {
        float[] state = {1.0f, 2.0f};
        Map<String, Object> info = new HashMap<>();
        info.put("score", 100);
        info.put("level", 5);

        StepResult<float[]> result = new StepResult<>(state, 3.0f, true, info);

        assertEquals(100, result.info("score"));
        assertEquals(5, result.info("level"));
        assertTrue(result.hasInfo("score"));
        assertFalse(result.hasInfo("unknown"));
        assertEquals("default", result.info("unknown", "default"));
    }

    @Test
    public void testStepResultDone() {
        StepResult<float[]> result = StepResult.of(new float[]{0}, 1.0f, true);
        assertTrue(result.done());
    }

    @Test
    public void testStepResultOfFactory() {
        Map<String, Object> info = new HashMap<>();
        info.put("key", "value");

        StepResult<float[]> result = StepResult.of(new float[]{1}, 2.0f, false, info);

        assertEquals(2.0f, result.reward(), 0.0f);
        assertEquals("value", result.info("key"));
    }

    @Test
    public void testEnvironmentReset() {
        MockEnvironment env = new MockEnvironment(10);
        
        float[] state = env.reset();
        
        assertNotNull(state);
        assertEquals(4, state.length);
        assertArrayEquals(new float[]{0.0f, 0.0f, 0.0f, 0.0f}, state, 0.0f);
    }

    @Test
    public void testEnvironmentStep() {
        MockEnvironment env = new MockEnvironment(10);
        env.reset();

        StepResult<float[]> result = env.step(0);

        assertNotNull(result.state());
        assertEquals(1.0f, result.reward(), 0.0f);
        assertFalse(result.done());
        assertEquals(1, result.info("step"));
    }

    @Test
    public void testEnvironmentDone() {
        MockEnvironment env = new MockEnvironment(3);
        env.reset();

        env.step(0);
        env.step(0);
        StepResult<float[]> result = env.step(0);

        assertTrue(result.done());
    }

    @Test
    public void testEnvironmentActionSpace() {
        MockEnvironment env = new MockEnvironment(10);
        assertEquals(4, env.actionSpaceSize());
    }

    @Test
    public void testEnvironmentObservationShape() {
        MockEnvironment env = new MockEnvironment(10);
        assertEquals(1, env.observationShape().size());
        assertEquals(Integer.valueOf(4), env.observationShape().get(0));
    }

    @Test
    public void testPolicySelectAction() {
        MockPolicy policy = new MockPolicy();
        float[] state = {1.0f, 2.0f, 3.0f, 4.0f};

        Integer action = policy.selectAction(state, true);
        
        assertNotNull(action);
        assertTrue(action >= 0 && action < 4);
        assertTrue(policy.lastExplore());
    }

    @Test
    public void testPolicySelectActionEval() {
        MockPolicy policy = new MockPolicy();
        float[] state = {1.0f, 2.0f, 3.0f, 4.0f};

        Integer action = policy.selectActionEval(state);
        
        assertNotNull(action);
        assertEquals(Integer.valueOf(0), action);
        assertFalse(policy.lastExplore());
    }

    @Test
    public void testPolicyReset() {
        MockPolicy policy = new MockPolicy();
        
        Integer first = policy.selectAction(new float[]{0}, true);
        policy.reset();
        Integer second = policy.selectAction(new float[]{0}, true);
        
        assertEquals(first, second);
    }

    @Test
    public void testTransitionBasic() {
        float[] state = {1.0f, 2.0f};
        float[] nextState = {3.0f, 4.0f};

        Transition t = new Transition(state, 1, 5.0f, nextState, false);

        assertArrayEquals(state, t.state(), 0.0f);
        assertEquals(1, t.action());
        assertEquals(5.0f, t.reward(), 0.0f);
        assertArrayEquals(nextState, t.nextState(), 0.0f);
        assertFalse(t.done());
        assertEquals(1.0f, t.priority(), 0.0f);
    }

    @Test
    public void testTransitionWithPriorityConstructor() {
        Transition t = new Transition(new float[]{0}, 0, 1.0f, new float[]{0}, false, 2.5f);
        assertEquals(2.5f, t.priority(), 0.0f);
    }

    @Test
    public void testTransitionWithPriorityMethod() {
        Transition t = new Transition(new float[]{0}, 0, 1.0f, new float[]{0}, false);
        Transition t2 = t.withPriority(5.0f);

        assertEquals(1.0f, t.priority(), 0.0f);
        assertEquals(5.0f, t2.priority(), 0.0f);
    }

    @Test
    public void testTransitionStateSize() {
        Transition t = new Transition(new float[]{1, 2, 3, 4, 5}, 0, 0, new float[5], false);
        assertEquals(5, t.stateSize());
    }

    @Test
    public void testTransitionRefMethods() {
        float[] state = {1.0f};
        float[] nextState = {2.0f};
        Transition t = new Transition(state, 0, 0, nextState, false);

        assertNotSame(state, t.stateRef());
        assertNotSame(nextState, t.nextStateRef());
        assertArrayEquals(state, t.stateRef(), 0.0f);
        assertArrayEquals(nextState, t.nextStateRef(), 0.0f);

        state[0] = 99.0f;
        nextState[0] = 98.0f;
        assertArrayEquals(new float[]{1.0f}, t.state(), 0.0f);
        assertArrayEquals(new float[]{2.0f}, t.nextState(), 0.0f);
    }

    @Test
    public void testContinuousTransitionCopiesItsArrays() {
        float[] state = {1.0f, 2.0f};
        float[] action = {0.5f, -0.5f};
        float[] nextState = {3.0f, 4.0f};
        Transition t = new Transition(state, action, 1.0f, nextState, false, 1.0f);

        state[0] = 99.0f;
        action[0] = 99.0f;
        nextState[0] = 99.0f;
        assertArrayEquals(new float[]{1.0f, 2.0f}, t.state(), 0.0f);
        assertArrayEquals(new float[]{0.5f, -0.5f}, t.continuousAction(), 0.0f);
        assertArrayEquals(new float[]{3.0f, 4.0f}, t.nextState(), 0.0f);
        assertEquals(Transition.NO_DISCRETE_ACTION, t.action());
    }

    @Test
    public void testTransitionNullStatesStayNull() {
        Transition t = new Transition(null, 0, 0, null, true);
        assertNull(t.state());
        assertNull(t.nextState());
        assertEquals(0, t.stateSize());
    }

    @Test
    public void testOwnedStatesFactoryKeepsStatesAndCopiesAction() {
        float[] state = {1.0f};
        float[] nextState = {2.0f};
        Transition discrete = Transition.ofOwnedStates(state, 3, 1.0f, nextState, false, 2.0f);
        assertSame(state, discrete.stateRef());
        assertSame(nextState, discrete.nextStateRef());
        assertNull(discrete.continuousActionRef());
        assertEquals(3, discrete.action());
        assertEquals(2.0f, discrete.priority(), 0.0f);

        float[] action = {0.25f};
        Transition continuous = Transition.ofOwnedStates(state, action, 1.0f, nextState, false, 1.0f);
        assertSame(state, continuous.stateRef());
        assertNotSame(action, continuous.continuousActionRef());
        assertArrayEquals(action, continuous.continuousActionRef(), 0.0f);
        assertEquals(Transition.NO_DISCRETE_ACTION, continuous.action());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOwnedStatesFactoryRejectsEmptyAction() {
        Transition.ofOwnedStates(new float[]{1.0f}, new float[0], 1.0f, new float[]{2.0f}, false, 1.0f);
    }

    @Test
    public void testWithPriorityKeepsValues() {
        Transition t = new Transition(new float[]{1.0f}, 2, 3.0f, new float[]{4.0f}, true);
        Transition t2 = t.withPriority(7.0f);
        assertArrayEquals(t.state(), t2.state(), 0.0f);
        assertArrayEquals(t.nextState(), t2.nextState(), 0.0f);
        assertEquals(2, t2.action());
        assertEquals(3.0f, t2.reward(), 0.0f);
        assertTrue(t2.done());
    }

    @Test
    public void testTransitionEqualsAndHashCode() {
        Transition t1 = new Transition(new float[]{1, 2}, 0, 1.0f, new float[]{3, 4}, true);
        Transition t2 = new Transition(new float[]{1, 2}, 0, 1.0f, new float[]{3, 4}, true);
        Transition t3 = new Transition(new float[]{1, 2}, 1, 1.0f, new float[]{3, 4}, true);

        assertEquals(t1, t2);
        assertNotEquals(t1, t3);
        assertEquals(t1.hashCode(), t2.hashCode());
    }

    @Test
    public void testTransitionToString() {
        Transition t = new Transition(new float[]{1}, 2, 3.0f, new float[]{4}, true);
        String str = t.toString();
        
        assertTrue(str.contains("action=2"));
        assertTrue(str.contains("reward=3.0"));
        assertTrue(str.contains("done=true"));
    }
}

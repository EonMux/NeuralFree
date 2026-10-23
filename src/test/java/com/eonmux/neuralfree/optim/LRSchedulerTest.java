package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.nn.Parameter;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class LRSchedulerTest {
    
    private Optimizer createMockOptimizer(float lr) {
        List<Parameter> params = new ArrayList<>();
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        
        return new Optimizer(params, defaults) {
            @Override
            public void step() {
                incrementStep();
            }
        };
    }

    @Test
    public void testLRSchedulerBaseClass() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        LRScheduler scheduler = new StepLR(optimizer, 10, 0.5f);
        
        assertEquals(0, scheduler.stepCount());
        assertEquals(0, scheduler.epochCount());
        assertEquals(0.1f, scheduler.initialLr(), 1e-6f);
        assertNotNull(scheduler.optimizer());
    }

    @Test
    public void testStepLRBasic() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR scheduler = new StepLR(optimizer, 2, 0.5f);
        
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
        
        scheduler.stepEpoch();
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
        
        scheduler.stepEpoch();
        assertEquals(0.05f, scheduler.getLR(), 1e-6f);
        
        scheduler.stepEpoch();
        assertEquals(0.05f, scheduler.getLR(), 1e-6f);
        
        scheduler.stepEpoch();
        assertEquals(0.025f, scheduler.getLR(), 1e-6f);
    }

    @Test
    public void testStepLRStepMethod() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR scheduler = new StepLR(optimizer, 2, 0.5f);
        
        scheduler.step();
        assertEquals(0.1f, optimizer.getLearningRate(), 1e-6f);
        
        scheduler.step();
        scheduler.stepEpoch();
        scheduler.stepEpoch();
        scheduler.step();
        assertEquals(0.05f, optimizer.getLearningRate(), 1e-6f);
    }

    @Test
    public void testStepLRInvalidStepSize() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        try {
            new StepLR(optimizer, 0, 0.5f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        try {
            new StepLR(optimizer, -1, 0.5f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }

    @Test
    public void testStepLRInvalidGamma() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        try {
            new StepLR(optimizer, 10, 0.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        try {
            new StepLR(optimizer, 10, 1.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }

    @Test
    public void testCosineAnnealingLRBasic() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, 4, 0.0f);
        
        assertEquals(0.1f, scheduler.getLR(), 0.01f);
        
        scheduler.stepEpoch();
        float lr1 = scheduler.getLR();
        assertTrue("LR should decrease during cosine annealing", lr1 < 0.1f && lr1 > 0.0f);
        
        scheduler.stepEpoch();
        scheduler.stepEpoch();
        float lr3 = scheduler.getLR();
        assertTrue("LR should continue decreasing", lr3 < lr1);
        
        scheduler.stepEpoch();
        float lr4 = scheduler.getLR();
        assertEquals("LR should reach minimum at T_max", 0.0f, lr4, 0.001f);
    }

    @Test
    public void testCosineAnnealingLRWarmRestart() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, 8, 0.0f, 2);
        
        assertEquals(8, scheduler.tMax());
        assertEquals(2, scheduler.numCycles());
        
        for (int i = 0; i < 2; i++) {
            scheduler.stepEpoch();
        }
        float midLr = scheduler.getLR();
        assertTrue("Mid-cycle LR should be lower than initial", midLr < 0.1f);
        
        for (int i = 0; i < 2; i++) {
            scheduler.stepEpoch();
        }
        float endLr = scheduler.getLR();
        assertTrue("End of first cycle should restart from initial", endLr >= 0.0f);
    }

    @Test
    public void testCosineAnnealingLRInvalidTMax() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        try {
            new CosineAnnealingLR(optimizer, 0, 0.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }

    @Test
    public void testWarmupLRBasic() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        WarmupLR scheduler = new WarmupLR(optimizer, 4);
        
        assertEquals(0.025f, scheduler.getLR(), 1e-6f);
        assertTrue(scheduler.inWarmup());
        
        scheduler.step();
        assertEquals(0.05f, scheduler.getLR(), 1e-6f);
        
        scheduler.step();
        scheduler.step();
        scheduler.step();
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
        assertFalse(scheduler.inWarmup());
    }

    @Test
    public void testWarmupLRWithBaseScheduler() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR baseScheduler = new StepLR(optimizer, 2, 0.5f);
        WarmupLR scheduler = new WarmupLR(optimizer, 2, baseScheduler);
        
        assertEquals("Initial LR should be warmup factor * base", 0.05f, scheduler.getLR(), 1e-6f);
        
        scheduler.step();
        assertEquals("After step 1, warmup factor = 1.0", 0.1f, scheduler.getLR(), 1e-6f);
        
        scheduler.step();
        assertTrue("After warmup should use base scheduler", scheduler.getLR() <= 0.1f);
        
        scheduler.stepEpoch();
        scheduler.stepEpoch();
        scheduler.step();
        assertTrue("After base scheduler decay, LR should be lower", scheduler.getLR() < 0.1f);
    }

    @Test
    public void testWarmupLRProgress() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        WarmupLR scheduler = new WarmupLR(optimizer, 4);
        
        assertEquals(0.0f, scheduler.warmupProgress(), 0.01f);
        
        scheduler.step();
        assertEquals(0.25f, scheduler.warmupProgress(), 0.01f);
        
        scheduler.step();
        scheduler.step();
        scheduler.step();
        assertEquals(1.0f, scheduler.warmupProgress(), 0.01f);
    }

    @Test
    public void testOneCycleLRBasic() {
        Optimizer optimizer = createMockOptimizer(0.004f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "cos");
        
        assertEquals(0.1f, scheduler.maxLr(), 1e-6f);
        assertEquals(10, scheduler.totalSteps());
        assertEquals(0.3f, scheduler.pctStart(), 1e-6f);
        
        assertTrue(scheduler.getLR() > 0);
        assertTrue(scheduler.getLR() < 0.1f);
    }

    @Test
    public void testOneCycleLRProgress() {
        Optimizer optimizer = createMockOptimizer(0.004f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10);
        
        float prevLr = scheduler.getLR();
        for (int i = 0; i < 10; i++) {
            scheduler.step();
            float currentLr = scheduler.getLR();
            assertTrue(currentLr > 0);
            prevLr = currentLr;
        }
        
        assertTrue(scheduler.isFinished());
    }

    @Test
    public void testOneCycleLRLinearStrategy() {
        Optimizer optimizer = createMockOptimizer(0.004f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "linear");
        
        assertEquals("linear", scheduler.annealStrategy());
        
        for (int i = 0; i < 10; i++) {
            scheduler.step();
            float lr = scheduler.getLR();
            assertTrue(lr > 0);
        }
    }

    @Test
    public void testOneCycleLRInvalidParams() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        
        try {
            new OneCycleLR(optimizer, 0.0f, 10);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new OneCycleLR(optimizer, 0.1f, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new OneCycleLR(optimizer, 0.1f, 10, 0.0f, 1e4f, 0.3f, "cos");
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.0f, "cos");
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 1.0f, "cos");
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }

    @Test
    public void testReduceOnPlateauLRMinMode() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 2);
        
        assertEquals("min", scheduler.mode());
        assertEquals(0.5f, scheduler.factor(), 1e-6f);
        assertEquals(2, scheduler.patience());
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
        
        scheduler.step(1.0f);
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
        
        scheduler.step(1.0f);
        scheduler.step(1.0f);
        scheduler.step(1.0f);
        assertTrue("LR should reduce after patience exceeded", scheduler.getLR() < 0.1f);
    }

    @Test
    public void testReduceOnPlateauLRMaxMode() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "max", 0.5f, 2);
        
        assertEquals("max", scheduler.mode());
        
        scheduler.step(1.0f);
        scheduler.step(0.9f);
        scheduler.step(0.9f);
        scheduler.step(0.9f);
        assertTrue("LR should reduce after patience exceeded in max mode", scheduler.getLR() < 0.1f);
    }

    @Test
    public void testReduceOnPlateauLRCooldown() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 1, 1e-4f, "rel", 2, 0.0f);
        
        assertEquals(2, scheduler.cooldown());
        
        scheduler.step(1.0f);
        scheduler.step(1.0f);
        scheduler.step(1.0f);
        assertTrue("LR should reduce", scheduler.getLR() < 0.1f);
        assertTrue("Should be in cooldown", scheduler.inCooldown());
    }

    @Test
    public void testReduceOnPlateauLRMinLr() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.1f, 1, 1e-4f, "rel", 0, 0.05f);
        
        assertEquals(0.05f, scheduler.minLr(), 1e-6f);
        
        for (int i = 0; i < 20; i++) {
            scheduler.step(1.0f);
        }
        
        assertTrue("LR should not go below min_lr", scheduler.getLR() >= 0.05f);
    }

    @Test
    public void testReduceOnPlateauLRInvalidParams() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        
        try {
            new ReduceOnPlateauLR(optimizer, "invalid", 0.5f, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new ReduceOnPlateauLR(optimizer, "min", 0.0f, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new ReduceOnPlateauLR(optimizer, "min", 1.0f, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        
        try {
            new ReduceOnPlateauLR(optimizer, "min", 0.5f, -1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }

    @Test
    public void testSchedulerReset() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR scheduler = new StepLR(optimizer, 2, 0.5f);
        
        scheduler.stepEpoch();
        scheduler.stepEpoch();
        scheduler.step();
        assertTrue(scheduler.getLR() < 0.1f);
        
        scheduler.reset();
        assertEquals(0, scheduler.stepCount());
        assertEquals(0, scheduler.epochCount());
        assertEquals(0.1f, scheduler.getLR(), 1e-6f);
    }

    @Test
    public void testSchedulerClose() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        LRScheduler scheduler = new StepLR(optimizer, 2, 0.5f);
        
        scheduler.close();
        
        try {
            scheduler.step();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
        }
    }

    @Test
    public void testNullOptimizerRejected() {
        try {
            new StepLR(null, 10, 0.5f);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
        }
        
        try {
            new CosineAnnealingLR(null, 10);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
        }
        
        try {
            new WarmupLR(null, 10);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
        }
        
        try {
            new OneCycleLR(null, 0.1f, 10);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
        }
        
        try {
            new ReduceOnPlateauLR(null, "min");
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
        }
    }

    @Test
    public void testStepLRAccessors() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR scheduler = new StepLR(optimizer, 10, 0.5f);
        
        assertEquals(10, scheduler.stepSize());
        assertEquals(0.5f, scheduler.gamma(), 1e-6f);
    }

    @Test
    public void testCosineAnnealingLRAccessors() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, 100, 0.001f, 3);
        
        assertEquals(100, scheduler.tMax());
        assertEquals(0.001f, scheduler.etaMin(), 1e-6f);
        assertEquals(3, scheduler.numCycles());
    }

    @Test
    public void testWarmupLRAccessors() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        WarmupLR scheduler = new WarmupLR(optimizer, 100);
        
        assertEquals(100, scheduler.warmupSteps());
        assertNull(scheduler.baseScheduler());
    }

    @Test
    public void testReduceOnPlateauLRAccessors() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 10, 1e-4f, "rel", 5, 0.001f);
        
        assertEquals(0.5f, scheduler.factor(), 1e-6f);
        assertEquals(10, scheduler.patience());
        assertEquals(1e-4f, scheduler.threshold(), 1e-8f);
        assertEquals("rel", scheduler.thresholdMode());
        assertEquals(5, scheduler.cooldown());
        assertEquals(0.001f, scheduler.minLr(), 1e-6f);
    }

    @Test
    public void testCosineAnnealingLRProducesCorrectCurve() {
        Optimizer optimizer = createMockOptimizer(1.0f);
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, 4, 0.0f);
        
        float[] actualLrs = new float[5];
        
        for (int i = 0; i < 4; i++) {
            actualLrs[i] = scheduler.getLR();
            scheduler.stepEpoch();
        }
        actualLrs[4] = scheduler.getLR();
        
        assertEquals("At epoch 0, LR should be initial", 1.0f, actualLrs[0], 0.01f);
        assertTrue("LR should decrease over time", actualLrs[3] < actualLrs[0]);
    }

    @Test
    public void testStepEpochAppliesRateBeforeNextEpochsFirstBatch() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR scheduler = new StepLR(optimizer, 1, 0.5f);
        float[][] inEffect = new float[3][4];

        for (int epoch = 0; epoch < 3; epoch++) {
            for (int batch = 0; batch < 4; batch++) {
                inEffect[epoch][batch] = optimizer.getLearningRate();
                scheduler.step();
            }
            scheduler.stepEpoch();
            assertEquals("rate recorded after epoch " + epoch + " is the next epoch's",
                    0.1f * (float) Math.pow(0.5, epoch + 1), optimizer.getLearningRate(), 1e-7f);
        }

        for (int epoch = 0; epoch < 3; epoch++) {
            float expected = 0.1f * (float) Math.pow(0.5, epoch);
            for (int batch = 0; batch < 4; batch++) {
                assertEquals("epoch " + epoch + " batch " + batch, expected, inEffect[epoch][batch], 1e-7f);
            }
        }
    }

    @Test
    public void testStepAppliesRateForAdvancedCount() {
        Optimizer optimizer = createMockOptimizer(0.004f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "linear");

        assertEquals(0.004f, optimizer.getLearningRate(), 1e-7f);
        scheduler.step();
        assertEquals(1, scheduler.stepCount());
        assertEquals("after one step the rate is the schedule's value at step 1",
                0.052f, optimizer.getLearningRate(), 1e-6f);
        assertEquals(scheduler.getLR(), optimizer.getLearningRate(), 0.0f);
    }

    @Test
    public void testCosineAnnealingMayReachZero() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        CosineAnnealingLR scheduler = new CosineAnnealingLR(optimizer, 2, 0.0f);

        scheduler.stepEpoch();
        assertEquals(0.05f, optimizer.getLearningRate(), 1e-6f);
        scheduler.stepEpoch();
        assertEquals(0.0f, optimizer.getLearningRate(), 1e-7f);
    }

    @Test
    public void testWarmupLRAppliesFirstRateAtConstruction() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        new WarmupLR(optimizer, 4);

        assertEquals("the first batch runs at the first warmup rate", 0.025f, optimizer.getLearningRate(), 1e-7f);
    }

    @Test
    public void testWarmupLREpochBoundaryInsideWarmupKeepsWarmupRate() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR base = new StepLR(optimizer, 1, 0.5f);
        WarmupLR scheduler = new WarmupLR(optimizer, 5, base);

        scheduler.step();
        scheduler.stepEpoch();
        assertEquals("an epoch boundary inside the warmup keeps the warmup rate",
                0.04f, optimizer.getLearningRate(), 1e-7f);

        scheduler.step();
        scheduler.step();
        scheduler.step();
        assertEquals(0.1f, optimizer.getLearningRate(), 1e-7f);
        scheduler.step();
        assertEquals("after the warmup the base rate for one elapsed epoch applies",
                0.05f, optimizer.getLearningRate(), 1e-7f);
        assertEquals(base.getLR(), scheduler.getLR(), 0.0f);
    }

    @Test
    public void testWarmupLRResetRestoresFirstWarmupRate() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        StepLR base = new StepLR(optimizer, 1, 0.5f);
        WarmupLR scheduler = new WarmupLR(optimizer, 2, base);

        for (int i = 0; i < 5; i++) {
            scheduler.step();
            scheduler.stepEpoch();
        }
        scheduler.reset();

        assertEquals(0, base.epochCount());
        assertEquals(0.05f, optimizer.getLearningRate(), 1e-7f);
    }

    @Test
    public void testOneCycleLRTwoPhaseMatchesPyTorch() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "linear");
        float minLr = 0.004f / 1e4f;
        float[] expected = new float[10];
        expected[0] = 0.004f;
        expected[1] = 0.052f;
        expected[2] = 0.1f;
        for (int step = 3; step < 10; step++) {
            expected[step] = 0.1f + (minLr - 0.1f) * (step - 2) / 7.0f;
        }

        assertFalse(scheduler.threePhase());
        for (int step = 0; step < 10; step++) {
            assertEquals("step " + step, expected[step], optimizer.getLearningRate(), 1e-6f);
            scheduler.step();
        }
        assertEquals("stepping to total_steps holds the final rate", minLr, optimizer.getLearningRate(), 1e-9f);
    }

    @Test
    public void testOneCycleLRCosineMidpoints() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 1.0f, 22, 10.0f, 100.0f, 0.5f, "cos");
        float minLr = 0.1f / 100.0f;

        for (int step = 0; step < 22; step++) {
            float lr = scheduler.getLR();
            if (step == 0) {
                assertEquals(0.1f, lr, 1e-6f);
            }
            if (step == 5) {
                assertEquals("cosine midpoint of the rise", 0.55f, lr, 1e-6f);
            }
            if (step == 10) {
                assertEquals(1.0f, lr, 1e-6f);
            }
            if (step == 21) {
                assertEquals(minLr, lr, 1e-7f);
            }
            scheduler.step();
        }
    }

    @Test
    public void testOneCycleLRThreePhaseMatchesPyTorch() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "linear", true);
        float minLr = 0.004f / 1e4f;

        assertTrue(scheduler.threePhase());
        float[] lrs = new float[11];
        for (int step = 0; step <= 10; step++) {
            lrs[step] = optimizer.getLearningRate();
            if (step < 10) {
                scheduler.step();
            }
        }
        assertEquals(0.004f, lrs[0], 1e-6f);
        assertEquals(0.1f, lrs[2], 1e-6f);
        assertEquals("halfway back to the starting rate", 0.052f, lrs[3], 1e-6f);
        assertEquals(0.004f, lrs[4], 1e-6f);
        assertEquals(0.004f + (minLr - 0.004f) * 0.4f, lrs[6], 1e-7f);
        assertEquals(minLr, lrs[9], 1e-9f);
        assertEquals(minLr, lrs[10], 1e-9f);
    }

    @Test
    public void testOneCycleLRRejectsSteppingPastTotalSteps() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        OneCycleLR scheduler = new OneCycleLR(optimizer, 0.1f, 3);
        scheduler.step();
        scheduler.step();
        scheduler.step();

        try {
            scheduler.step();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertEquals(3, scheduler.stepCount());
        }
    }

    @Test
    public void testOneCycleLRRejectsUnknownAnnealStrategy() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        try {
            new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "exp");
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
        assertEquals("cos", new OneCycleLR(optimizer, 0.1f, 10, 25.0f, 1e4f, 0.3f, "COS").annealStrategy());
    }

    @Test
    public void testReduceOnPlateauLRMaxModeWithNonPositiveMetrics() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "max", 0.5f, 1);

        assertEquals(Float.NEGATIVE_INFINITY, scheduler.bestMetric(), 0.0f);
        scheduler.step(-0.5f);
        scheduler.step(-0.4f);
        scheduler.step(0.0f);
        scheduler.step(0.1f);

        assertEquals("an improving metric never reduces the rate", 0.1f, optimizer.getLearningRate(), 0.0f);
        assertEquals(0.1f, scheduler.bestMetric(), 0.0f);
        assertEquals(0, scheduler.numBadEpochs());

        scheduler.resetMetrics();
        assertEquals(Float.NEGATIVE_INFINITY, scheduler.bestMetric(), 0.0f);
    }

    @Test
    public void testReduceOnPlateauLRMinModeInitialBest() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 0, 1e-4f, "abs", 0, 0.0f);

        assertEquals(Float.POSITIVE_INFINITY, scheduler.bestMetric(), 0.0f);
        scheduler.step(Float.MAX_VALUE);
        assertEquals("any finite first metric is the best", Float.MAX_VALUE, scheduler.bestMetric(), 0.0f);
        assertEquals(0.1f, optimizer.getLearningRate(), 0.0f);
    }

    @Test
    public void testReduceOnPlateauLRTracksBestDuringCooldown() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 0, 1e-4f, "rel", 2, 0.0f);

        scheduler.step(1.0f);
        scheduler.step(1.0f);
        assertEquals(0.05f, optimizer.getLearningRate(), 1e-7f);
        assertTrue(scheduler.inCooldown());

        scheduler.step(0.5f);
        assertEquals("a better metric during cooldown becomes the best", 0.5f, scheduler.bestMetric(), 0.0f);
        scheduler.step(0.6f);
        assertFalse(scheduler.inCooldown());
        assertEquals("bad calls during cooldown are not counted", 0.05f, optimizer.getLearningRate(), 1e-7f);

        scheduler.step(0.6f);
        assertEquals(0.025f, optimizer.getLearningRate(), 1e-7f);
    }

    @Test
    public void testReduceOnPlateauLRResetRestoresInitialRate() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        ReduceOnPlateauLR scheduler = new ReduceOnPlateauLR(optimizer, "min", 0.5f, 0);

        scheduler.step(1.0f);
        scheduler.step(1.0f);
        assertEquals(0.05f, optimizer.getLearningRate(), 1e-7f);

        scheduler.reset();
        assertEquals(0.1f, scheduler.getLR(), 0.0f);
        assertEquals(0.1f, optimizer.getLearningRate(), 0.0f);
    }

    @Test
    public void testReduceOnPlateauLRRejectsUnknownThresholdMode() {
        Optimizer optimizer = createMockOptimizer(0.1f);
        try {
            new ReduceOnPlateauLR(optimizer, "min", 0.5f, 2, 1e-4f, "relative", 0, 0.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
        }
    }
}

package com.eonmux.neuralfree.data;

import org.junit.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.*;

public class WeightedRandomSamplerTest {

    @Test
    public void testWeightedSampling() {
        double[] weights = {0.1, 0.9};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 1000, true, 42);
        
        List<Integer> indices = sampler.sample(2);
        assertEquals(1000, indices.size());
        
        int count0 = 0, count1 = 0;
        for (int idx : indices) {
            if (idx == 0) count0++;
            else if (idx == 1) count1++;
        }
        
        assertTrue("Index 1 should be sampled more frequently than index 0", count1 > count0);
        assertTrue("Index 0 should still be sampled sometimes", count0 > 0);
    }

    @Test
    public void testReproducible() {
        double[] weights = {0.3, 0.5, 0.2};
        
        WeightedRandomSampler sampler1 = new WeightedRandomSampler(weights, 10, true, 123);
        WeightedRandomSampler sampler2 = new WeightedRandomSampler(weights, 10, true, 123);
        
        List<Integer> indices1 = sampler1.sample(3);
        List<Integer> indices2 = sampler2.sample(3);
        
        assertEquals(indices1, indices2);
    }

    @Test
    public void testNumSamples() {
        double[] weights = {0.5, 0.5};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 50, true);
        
        assertEquals(50, sampler.numSamples());
    }

    @Test
    public void testReplacement() {
        double[] weights = {0.5, 0.5};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 10, true);
        
        assertTrue(sampler.replacement());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEmptyWeights() {
        new WeightedRandomSampler(new double[0], 10, true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeWeights() {
        new WeightedRandomSampler(new double[]{0.5, -0.1}, 10, true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testZeroWeights() {
        new WeightedRandomSampler(new double[]{0.0, 0.0}, 10, true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testZeroNumSamples() {
        new WeightedRandomSampler(new double[]{0.5, 0.5}, 0, true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMismatchedDatasetSize() {
        double[] weights = {0.5, 0.5};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 10, true);
        sampler.sample(3);
    }

    @Test
    public void testWithReplacementNeverDrawsZeroWeightIndices() {
        double[] weights = {0.0, 1.0, 0.0, 2.0, 0.0};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 2000, true, 5L);
        for (int idx : sampler.sample(weights.length)) {
            assertTrue("zero-weight index drawn: " + idx, weights[idx] > 0);
        }
    }

    @Test
    public void testWithoutReplacementDrawsDistinctPositiveWeightIndices() {
        double[] weights = {0.0, 0.2, 0.0, 0.5, 0.1, 0.9, 0.0, 0.3};
        for (long seed = 0; seed < 50; seed++) {
            WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 5, false, seed);
            List<Integer> indices = sampler.sample(weights.length);
            assertEquals(5, indices.size());
            Set<Integer> distinct = new HashSet<>(indices);
            assertEquals("indices repeat: " + indices, 5, distinct.size());
            for (int idx : indices) {
                assertTrue("zero-weight index drawn: " + idx, weights[idx] > 0);
            }
        }
    }

    @Test
    public void testWithoutReplacementFavoursHeavierWeightsFirst() {
        double[] weights = {1.0, 9.0};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 1, false, 11L);
        int firstHeavy = 0;
        for (int trial = 0; trial < 2000; trial++) {
            if (sampler.sample(2).get(0) == 1) {
                firstHeavy++;
            }
        }
        double fraction = firstHeavy / 2000.0;
        assertEquals("P(first draw is index 1) should be 0.9", 0.9, fraction, 0.03);
    }

    @Test
    public void testWithoutReplacementReproducibleAfterReset() {
        double[] weights = {0.3, 0.5, 0.2, 0.7};
        WeightedRandomSampler sampler = new WeightedRandomSampler(weights, 3, false, 77L);
        List<Integer> first = sampler.sample(4);
        sampler.reset();
        assertEquals(first, sampler.sample(4));
        assertFalse(sampler.replacement());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWithoutReplacementRejectsMoreSamplesThanPositiveWeights() {
        new WeightedRandomSampler(new double[]{0.5, 0.0, 0.5}, 3, false);
    }

    @Test
    public void testWithReplacementAllowsMoreSamplesThanPositiveWeights() {
        WeightedRandomSampler sampler = new WeightedRandomSampler(new double[]{0.5, 0.0, 0.5}, 3, true, 1L);
        assertEquals(3, sampler.sample(3).size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNonFiniteWeightRejected() {
        new WeightedRandomSampler(new double[]{0.5, Double.NaN}, 1, true);
    }
}

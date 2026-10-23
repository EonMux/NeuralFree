package com.eonmux.neuralfree.data;

import org.junit.Test;

import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

public class RandomSamplerTest {

    @Test
    public void testRandomSamplingSize() {
        RandomSampler sampler = new RandomSampler(42);
        List<Integer> indices = sampler.sample(100);
        
        assertEquals(100, indices.size());
    }

    @Test
    public void testReproducible() {
        RandomSampler sampler1 = new RandomSampler(12345);
        RandomSampler sampler2 = new RandomSampler(12345);
        
        List<Integer> indices1 = sampler1.sample(50);
        List<Integer> indices2 = sampler2.sample(50);
        
        assertEquals(indices1, indices2);
    }

    @Test
    public void testReset() {
        RandomSampler sampler = new RandomSampler(999);
        
        List<Integer> indices1 = sampler.sample(10);
        sampler.reset();
        List<Integer> indices2 = sampler.sample(10);
        
        assertEquals(indices1, indices2);
    }

    @Test
    public void testAllIndicesPresent() {
        RandomSampler sampler = new RandomSampler(42);
        List<Integer> indices = sampler.sample(20);
        
        boolean[] seen = new boolean[20];
        for (int idx : indices) {
            assertFalse("Duplicate index: " + idx, seen[idx]);
            seen[idx] = true;
        }
        
        for (int i = 0; i < 20; i++) {
            assertTrue("Missing index: " + i, seen[i]);
        }
    }

    @Test
    public void testCustomRandom() {
        Random random = new Random(111);
        RandomSampler sampler = new RandomSampler(random);
        
        List<Integer> indices = sampler.sample(5);
        assertEquals(5, indices.size());
    }

    @Test(expected = NullPointerException.class)
    public void testNullRandom() {
        new RandomSampler(null);
    }
}

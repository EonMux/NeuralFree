package com.eonmux.neuralfree.data;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

public class SequentialSamplerTest {

    @Test
    public void testSequentialSampling() {
        SequentialSampler sampler = new SequentialSampler();
        List<Integer> indices = sampler.sample(5);
        
        assertEquals(5, indices.size());
        assertEquals(Integer.valueOf(0), indices.get(0));
        assertEquals(Integer.valueOf(1), indices.get(1));
        assertEquals(Integer.valueOf(2), indices.get(2));
        assertEquals(Integer.valueOf(3), indices.get(3));
        assertEquals(Integer.valueOf(4), indices.get(4));
    }

    @Test
    public void testZeroSize() {
        SequentialSampler sampler = new SequentialSampler();
        List<Integer> indices = sampler.sample(0);
        
        assertEquals(0, indices.size());
    }

    @Test
    public void testReset() {
        SequentialSampler sampler = new SequentialSampler();
        sampler.reset();
        List<Integer> indices = sampler.sample(3);
        assertEquals(3, indices.size());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeSize() {
        SequentialSampler sampler = new SequentialSampler();
        sampler.sample(-1);
    }
}

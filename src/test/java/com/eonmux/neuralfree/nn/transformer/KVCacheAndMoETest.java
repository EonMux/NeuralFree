package com.eonmux.neuralfree.nn.transformer;

import org.junit.Test;

import static org.junit.Assert.*;

public class KVCacheAndMoETest {

    @Test
    public void testKVCacheConstruction() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        
        assertEquals(6, cache.numLayers());
        assertEquals(8, cache.numHeads());
        assertEquals(64, cache.headDim());
        assertEquals(512, cache.maxSeqLen());
        assertEquals(2, cache.batchSize());
        
        cache.close();
    }
    
    @Test
    public void testKVCacheInvalidConstruction() {
        try {
            new KVCache(0, 8, 64, 512, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numLayers"));
        }
        
        try {
            new KVCache(6, 0, 64, 512, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numHeads"));
        }
        
        try {
            new KVCache(6, 8, 0, 512, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("headDim"));
        }
    }
    
    @Test
    public void testKVCacheGetSeqLengthBeforeInit() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        assertEquals(0, cache.getSeqLength(0));
        cache.close();
    }
    
    @Test
    public void testKVCacheClearBeforeInit() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        cache.clear();
        cache.close();
    }
    
    @Test
    public void testKVCacheGetKeysBeforeInit() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        try {
            cache.getKeys(0);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("initialized"));
        }
        cache.close();
    }
    
    @Test
    public void testKVCacheClosedState() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        cache.close();
        
        try {
            cache.getSeqLength(0);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("closed"));
        }
    }
    
    @Test
    public void testKVCacheToString() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        String str = cache.toString();
        
        assertTrue(str.contains("numLayers=6"));
        assertTrue(str.contains("numHeads=8"));
        assertTrue(str.contains("headDim=64"));
        assertTrue(str.contains("maxSeqLen=512"));
        assertTrue(str.contains("batchSize=2"));
        
        cache.close();
        
        str = cache.toString();
        assertTrue(str.contains("closed=true"));
    }
    
    @Test
    public void testMixtureOfExpertsConstruction() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2);
        
        assertEquals(256, moe.dModel());
        assertEquals(1024, moe.dFF());
        assertEquals(8, moe.numExperts());
        assertEquals(2, moe.topK());
        
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsInvalidConstruction() {
        try {
            new MixtureOfExperts(0, 1024, 8, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dModel"));
        }
        
        try {
            new MixtureOfExperts(256, 0, 8, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dFF"));
        }
        
        try {
            new MixtureOfExperts(256, 1024, 0, 2);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numExperts"));
        }
    }
    
    @Test
    public void testMixtureOfExpertsNotInitialized() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2);
        
        try {
            moe.router();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("initialized"));
        }
        
        try {
            moe.experts();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("initialized"));
        }
        
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsToString() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2);
        String str = moe.toString();
        
        assertTrue(str.contains("dModel=256"));
        assertTrue(str.contains("dFF=1024"));
        assertTrue(str.contains("numExperts=8"));
        assertTrue(str.contains("topK=2"));
        assertTrue(str.contains("initialized=false"));
        
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsLoadBalanceLossDefault() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2);
        assertEquals(0.0f, moe.getLoadBalanceLoss(), 1e-6f);
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsWithNoise() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2, 0.1f, true);
        
        assertEquals(0.1f, moe.noiseStd(), 1e-6f);
        assertTrue(moe.isLoadBalancing());
        
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsWithoutLoadBalancing() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 2, 0.0f, false);
        
        assertFalse(moe.isLoadBalancing());
        
        moe.close();
    }
    
    @Test
    public void testKVCacheLayerIndexBounds() {
        KVCache cache = new KVCache(6, 8, 64, 512, 2);
        
        try {
            cache.getSeqLength(-1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("out of bounds"));
        }
        
        try {
            cache.getSeqLength(6);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("out of bounds"));
        }
        
        cache.close();
    }
    
    @Test
    public void testMixtureOfExpertsTopKGreaterThanExperts() {
        try {
            new MixtureOfExperts(256, 1024, 4, 8);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("topK"));
        }
    }
    
    @Test
    public void testMixtureOfExpertsNegativeNoise() {
        try {
            new MixtureOfExperts(256, 1024, 8, 2, -0.1f, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("noiseStd"));
        }
    }
    
    @Test
    public void testMixtureOfExpertsTopKOne() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 1);
        assertEquals(1, moe.topK());
        moe.close();
    }
    
    @Test
    public void testMixtureOfExpertsTopKAllExperts() {
        MixtureOfExperts moe = new MixtureOfExperts(256, 1024, 8, 8);
        assertEquals(8, moe.topK());
        moe.close();
    }
}

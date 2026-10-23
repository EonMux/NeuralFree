package com.eonmux.neuralfree.nn.transformer;

import org.junit.Test;

import static org.junit.Assert.*;

public class MultiHeadAttentionTest {
    
    @Test
    public void testConstruction() {
        MultiHeadAttention mha = new MultiHeadAttention(512, 8);
        assertEquals(512, mha.dModel());
        assertEquals(8, mha.numHeads());
        assertEquals(64, mha.headDim());
        assertEquals(0.0f, mha.dropoutRate(), 1e-6f);
        assertFalse(mha.isCausal());
    }
    
    @Test
    public void testConstructionWithDropoutAndCausal() {
        MultiHeadAttention mha = new MultiHeadAttention(256, 4, 0.1f, true);
        assertEquals(256, mha.dModel());
        assertEquals(4, mha.numHeads());
        assertEquals(64, mha.headDim());
        assertEquals(0.1f, mha.dropoutRate(), 1e-6f);
        assertTrue(mha.isCausal());
    }
    
    @Test
    public void testInvalidConstructionDModelZero() {
        try {
            new MultiHeadAttention(0, 8);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dModel"));
        }
    }
    
    @Test
    public void testInvalidConstructionNumHeadsZero() {
        try {
            new MultiHeadAttention(512, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numHeads"));
        }
    }
    
    @Test
    public void testInvalidConstructionNotDivisible() {
        try {
            new MultiHeadAttention(100, 8);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible"));
        }
    }
    
    @Test
    public void testInvalidConstructionDropoutTooHigh() {
        try {
            new MultiHeadAttention(512, 8, 1.0f, false);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropoutRate"));
        }
    }
    
    @Test
    public void testInvalidConstructionDropoutNegative() {
        try {
            new MultiHeadAttention(512, 8, -0.1f, false);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropoutRate"));
        }
    }
    
    @Test
    public void testForwardNotInitializedThrows() {
        MultiHeadAttention mha = new MultiHeadAttention(64, 4);
        try {
            mha.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testAccessorsBeforeInit() {
        MultiHeadAttention mha = new MultiHeadAttention(64, 4);
        assertEquals(64, mha.dModel());
        assertEquals(4, mha.numHeads());
        assertEquals(16, mha.headDim());
        assertEquals(0.0f, mha.dropoutRate(), 1e-6f);
        assertFalse(mha.isCausal());
    }
    
    @Test
    public void testAccessorsBeforeInitCausal() {
        MultiHeadAttention mha = new MultiHeadAttention(512, 8, 0.1f, true);
        assertTrue(mha.isCausal());
        assertEquals(0.1f, mha.dropoutRate(), 1e-6f);
    }
    
    @Test
    public void testTrainEvalMode() {
        MultiHeadAttention mha = new MultiHeadAttention(64, 4, 0.5f, false);
        assertTrue(mha.isTraining());
        
        mha.eval();
        assertFalse(mha.isTraining());
        
        mha.train();
        assertTrue(mha.isTraining());
    }
    
    @Test
    public void testToString() {
        MultiHeadAttention mha = new MultiHeadAttention(512, 8, 0.1f, true);
        String str = mha.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("numHeads=8"));
        assertTrue(str.contains("causal=true"));
        assertTrue(str.contains("headDim=64"));
    }
    
    @Test
    public void testHeadDimComputation() {
        assertEquals(64, new MultiHeadAttention(512, 8).headDim());
        assertEquals(128, new MultiHeadAttention(512, 4).headDim());
        assertEquals(32, new MultiHeadAttention(256, 8).headDim());
        assertEquals(16, new MultiHeadAttention(128, 8).headDim());
    }
    
    @Test
    public void testFlashAttentionThreshold() {
        assertTrue(FlashAttentionOp.shouldUseFlashAttention(256));
        assertTrue(FlashAttentionOp.shouldUseFlashAttention(512));
        assertTrue(FlashAttentionOp.shouldUseFlashAttention(1024));
        assertFalse(FlashAttentionOp.shouldUseFlashAttention(128));
        assertFalse(FlashAttentionOp.shouldUseFlashAttention(64));
        assertEquals(256, FlashAttentionOp.getThreshold());
    }
    
    @Test
    public void testCloseWithoutInit() {
        MultiHeadAttention mha = new MultiHeadAttention(64, 4);
        mha.close();
    }
    
    @Test
    public void testModuleName() {
        MultiHeadAttention mha = new MultiHeadAttention(64, 4);
        assertEquals("MultiHeadAttention", mha.moduleName());
    }
}

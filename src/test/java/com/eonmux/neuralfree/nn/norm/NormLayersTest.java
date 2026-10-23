package com.eonmux.neuralfree.nn.norm;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.*;

public class NormLayersTest {

    @Test
    public void testBatchNorm1dCreation() {
        BatchNorm1d bn = new BatchNorm1d(64);
        assertNotNull(bn);
        assertEquals(64, bn.numFeatures());
        assertEquals(1e-5f, bn.epsilon(), 1e-10f);
        assertEquals(0.1f, bn.momentum(), 1e-10f);
        assertTrue(bn.hasAffine());
    }

    @Test
    public void testBatchNorm1dCreationWithParams() {
        BatchNorm1d bn = new BatchNorm1d(32, 1e-6f, 0.2f, false);
        assertEquals(32, bn.numFeatures());
        assertEquals(1e-6f, bn.epsilon(), 1e-10f);
        assertEquals(0.2f, bn.momentum(), 1e-10f);
        assertFalse(bn.hasAffine());
    }

    @Test
    public void testBatchNorm1dInvalidNumFeatures() {
        try {
            new BatchNorm1d(0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numFeatures"));
        }
    }

    @Test
    public void testBatchNorm1dInvalidEpsilon() {
        try {
            new BatchNorm1d(10, 0f, 0.1f, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("epsilon"));
        }
    }

    @Test
    public void testBatchNorm1dInvalidMomentum() {
        try {
            new BatchNorm1d(10, 1e-5f, 1.5f, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("momentum"));
        }
    }

    @Test
    public void testBatchNorm1dNotInitialized() {
        BatchNorm1d bn = new BatchNorm1d(10);
        try {
            bn.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testBatchNorm1dNoAffine() {
        BatchNorm1d bn = new BatchNorm1d(10, 1e-5f, 0.1f, false);
        try {
            bn.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no affine"));
        }
    }

    @Test
    public void testBatchNorm1dToString() {
        BatchNorm1d bn = new BatchNorm1d(64);
        String str = bn.toString();
        assertTrue(str.contains("BatchNorm1d"));
        assertTrue(str.contains("numFeatures=64"));
    }

    @Test
    public void testBatchNorm2dCreation() {
        BatchNorm2d bn = new BatchNorm2d(64);
        assertNotNull(bn);
        assertEquals(64, bn.numFeatures());
        assertEquals(1e-5f, bn.epsilon(), 1e-10f);
        assertEquals(0.1f, bn.momentum(), 1e-10f);
        assertTrue(bn.hasAffine());
    }

    @Test
    public void testBatchNorm2dCreationWithParams() {
        BatchNorm2d bn = new BatchNorm2d(32, 1e-6f, 0.2f, false);
        assertEquals(32, bn.numFeatures());
        assertFalse(bn.hasAffine());
    }

    @Test
    public void testBatchNorm2dInvalidNumFeatures() {
        try {
            new BatchNorm2d(0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numFeatures"));
        }
    }

    @Test
    public void testBatchNorm2dToString() {
        BatchNorm2d bn = new BatchNorm2d(64);
        String str = bn.toString();
        assertTrue(str.contains("BatchNorm2d"));
        assertTrue(str.contains("numFeatures=64"));
    }

    @Test
    public void testLayerNormCreation() {
        LayerNorm ln = new LayerNorm(64);
        assertNotNull(ln);
        assertArrayEquals(new int[]{64}, ln.normalizedShape());
        assertEquals(1e-5f, ln.epsilon(), 1e-10f);
        assertTrue(ln.hasElementwiseAffine());
    }

    @Test
    public void testLayerNormCreationWithArray() {
        LayerNorm ln = new LayerNorm(new int[]{64, 32});
        assertArrayEquals(new int[]{64, 32}, ln.normalizedShape());
    }

    @Test
    public void testLayerNormCreationWithParams() {
        LayerNorm ln = new LayerNorm(64, 1e-6f, false);
        assertEquals(1e-6f, ln.epsilon(), 1e-10f);
        assertFalse(ln.hasElementwiseAffine());
    }

    @Test
    public void testLayerNormInvalidShape() {
        try {
            new LayerNorm(new int[]{});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("empty"));
        }
    }

    @Test
    public void testLayerNormInvalidDim() {
        try {
            new LayerNorm(new int[]{64, 0, 32});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("positive"));
        }
    }

    @Test
    public void testLayerNormNoAffine() {
        LayerNorm ln = new LayerNorm(10, 1e-5f, false);
        try {
            ln.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no affine"));
        }
    }

    @Test
    public void testLayerNormToString() {
        LayerNorm ln = new LayerNorm(64);
        String str = ln.toString();
        assertTrue(str.contains("LayerNorm"));
        assertTrue(str.contains("normalizedShape"));
    }

    @Test
    public void testGroupNormCreation() {
        GroupNorm gn = new GroupNorm(4, 16);
        assertNotNull(gn);
        assertEquals(4, gn.numGroups());
        assertEquals(16, gn.numChannels());
        assertEquals(1e-5f, gn.epsilon(), 1e-10f);
        assertTrue(gn.hasAffine());
    }

    @Test
    public void testGroupNormCreationWithParams() {
        GroupNorm gn = new GroupNorm(8, 32, 1e-6f, false);
        assertEquals(8, gn.numGroups());
        assertEquals(32, gn.numChannels());
        assertEquals(1e-6f, gn.epsilon(), 1e-10f);
        assertFalse(gn.hasAffine());
    }

    @Test
    public void testGroupNormInvalidNumGroups() {
        try {
            new GroupNorm(0, 16);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numGroups"));
        }
    }

    @Test
    public void testGroupNormInvalidNumChannels() {
        try {
            new GroupNorm(4, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numChannels"));
        }
    }

    @Test
    public void testGroupNormNotDivisible() {
        try {
            new GroupNorm(3, 16);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible"));
        }
    }

    @Test
    public void testGroupNormNoAffine() {
        GroupNorm gn = new GroupNorm(4, 16, 1e-5f, false);
        try {
            gn.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no affine"));
        }
    }

    @Test
    public void testGroupNormToString() {
        GroupNorm gn = new GroupNorm(4, 16);
        String str = gn.toString();
        assertTrue(str.contains("GroupNorm"));
        assertTrue(str.contains("numGroups=4"));
        assertTrue(str.contains("numChannels=16"));
    }

    @Test
    public void testRMSNormCreation() {
        RMSNorm rn = new RMSNorm(64);
        assertNotNull(rn);
        assertEquals(64, rn.normalizedShape());
        assertEquals(1e-8f, rn.epsilon(), 1e-10f);
        assertTrue(rn.hasWeight());
    }

    @Test
    public void testRMSNormCreationWithParams() {
        RMSNorm rn = new RMSNorm(32, 1e-6f, false);
        assertEquals(32, rn.normalizedShape());
        assertEquals(1e-6f, rn.epsilon(), 1e-10f);
        assertFalse(rn.hasWeight());
    }

    @Test
    public void testRMSNormInvalidShape() {
        try {
            new RMSNorm(0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("normalizedShape"));
        }
    }

    @Test
    public void testRMSNormInvalidEpsilon() {
        try {
            new RMSNorm(64, 0f, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("epsilon"));
        }
    }

    @Test
    public void testRMSNormNoWeight() {
        RMSNorm rn = new RMSNorm(64, 1e-8f, false);
        try {
            rn.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no weight"));
        }
    }

    @Test
    public void testRMSNormToString() {
        RMSNorm rn = new RMSNorm(64);
        String str = rn.toString();
        assertTrue(str.contains("RMSNorm"));
        assertTrue(str.contains("normalizedShape=64"));
    }

    @Test
    public void testBatchNorm1dTrainVsEvalMode() {
        BatchNorm1d bn = new BatchNorm1d(4);
        assertTrue(bn.isTraining());
        bn.eval();
        assertFalse(bn.isTraining());
        bn.train();
        assertTrue(bn.isTraining());
    }

    @Test
    public void testBatchNorm2dTrainVsEvalMode() {
        BatchNorm2d bn = new BatchNorm2d(4);
        assertTrue(bn.isTraining());
        bn.eval();
        assertFalse(bn.isTraining());
    }

    @Test
    public void testLayerNormTrainVsEvalMode() {
        LayerNorm ln = new LayerNorm(64);
        assertTrue(ln.isTraining());
        ln.eval();
        assertFalse(ln.isTraining());
    }

    @Test
    public void testGroupNormTrainVsEvalMode() {
        GroupNorm gn = new GroupNorm(4, 16);
        assertTrue(gn.isTraining());
        gn.eval();
        assertFalse(gn.isTraining());
    }

    @Test
    public void testRMSNormTrainVsEvalMode() {
        RMSNorm rn = new RMSNorm(64);
        assertTrue(rn.isTraining());
        rn.eval();
        assertFalse(rn.isTraining());
    }

    @Test
    public void testBatchNorm1dParameters() {
        BatchNorm1d bn = new BatchNorm1d(10);
        bn.toString();
        assertEquals(0, bn.parameters().size());
    }

    @Test
    public void testLayerNormParameters() {
        LayerNorm ln = new LayerNorm(10);
        assertEquals(0, ln.parameters().size());
    }

    @Test
    public void testGroupNormParameters() {
        GroupNorm gn = new GroupNorm(4, 16);
        assertEquals(0, gn.parameters().size());
    }

    @Test
    public void testRMSNormParameters() {
        RMSNorm rn = new RMSNorm(10);
        assertEquals(0, rn.parameters().size());
    }

    @Test
    public void testBatchNorm1dNotInitializedForward() {
        BatchNorm1d bn = new BatchNorm1d(10);
        try {
            bn.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testBatchNorm2dNotInitializedForward() {
        BatchNorm2d bn = new BatchNorm2d(10);
        try {
            bn.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testLayerNormNotInitializedForward() {
        LayerNorm ln = new LayerNorm(10);
        try {
            ln.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testGroupNormNotInitializedForward() {
        GroupNorm gn = new GroupNorm(4, 16);
        try {
            gn.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testRMSNormNotInitializedForward() {
        RMSNorm rn = new RMSNorm(10);
        try {
            rn.forward(null);
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
}

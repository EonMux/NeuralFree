package com.eonmux.neuralfree.nn.layers;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import org.junit.Test;

import static org.junit.Assert.*;

public class LayersTest {

    @Test
    public void testLinearCreation() {
        Linear linear = new Linear(10, 5);
        assertNotNull(linear);
        assertEquals(10, linear.inFeatures());
        assertEquals(5, linear.outFeatures());
        assertTrue(linear.hasBias());
    }

    @Test
    public void testLinearCreationNoBias() {
        Linear linear = new Linear(10, 5, false);
        assertNotNull(linear);
        assertEquals(10, linear.inFeatures());
        assertEquals(5, linear.outFeatures());
        assertFalse(linear.hasBias());
    }

    @Test
    public void testLinearInvalidInFeatures() {
        try {
            new Linear(0, 5);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inFeatures"));
        }
    }

    @Test
    public void testLinearInvalidOutFeatures() {
        try {
            new Linear(10, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outFeatures"));
        }
    }

    @Test
    public void testLinearNotInitialized() {
        Linear linear = new Linear(10, 5);
        try {
            linear.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testLinearBiasNotAvailable() {
        Linear linear = new Linear(10, 5, false);
        try {
            linear.bias();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("no bias"));
        }
    }

    @Test
    public void testLinearToString() {
        Linear linear = new Linear(10, 5);
        String str = linear.toString();
        assertTrue(str.contains("Linear"));
        assertTrue(str.contains("inFeatures=10"));
        assertTrue(str.contains("outFeatures=5"));
    }

    @Test
    public void testConv2dCreation() {
        Conv2d conv = new Conv2d(3, 16, new int[]{3, 3});
        assertNotNull(conv);
        assertEquals(3, conv.inChannels());
        assertEquals(16, conv.outChannels());
        assertArrayEquals(new int[]{3, 3}, conv.kernelSize());
        assertArrayEquals(new int[]{1, 1}, conv.stride());
        assertArrayEquals(new int[]{0, 0}, conv.padding());
        assertTrue(conv.hasBias());
    }

    @Test
    public void testConv2dWithStridePadding() {
        Conv2d conv = new Conv2d(3, 16, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 2}, conv.stride());
        assertArrayEquals(new int[]{1, 1}, conv.padding());
    }

    @Test
    public void testConv2dWithAllParams() {
        Conv2d conv = new Conv2d(4, 8, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1}, 
                                  new int[]{1, 1}, 2, true);
        assertEquals(2, conv.groups());
        assertArrayEquals(new int[]{1, 1}, conv.dilation());
    }

    @Test
    public void testConv2dInvalidInChannels() {
        try {
            new Conv2d(0, 16, new int[]{3, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testConv2dInvalidOutChannels() {
        try {
            new Conv2d(3, 0, new int[]{3, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outChannels"));
        }
    }

    @Test
    public void testConv2dInvalidKernelSize() {
        try {
            new Conv2d(3, 16, new int[]{0, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("kernelSize"));
        }
    }

    @Test
    public void testConv2dGroupsNotDivisible() {
        try {
            new Conv2d(3, 16, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0},
                       new int[]{1, 1}, 2, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible by groups"));
        }
    }

    @Test
    public void testConv2dToString() {
        Conv2d conv = new Conv2d(3, 16, new int[]{3, 3});
        String str = conv.toString();
        assertTrue(str.contains("Conv2d"));
        assertTrue(str.contains("inChannels=3"));
        assertTrue(str.contains("outChannels=16"));
    }

    @Test
    public void testConv1dCreation() {
        Conv1d conv = new Conv1d(3, 16, 3);
        assertNotNull(conv);
        assertEquals(3, conv.inChannels());
        assertEquals(16, conv.outChannels());
        assertEquals(3, conv.kernelSize());
        assertEquals(1, conv.stride());
        assertEquals(0, conv.padding());
        assertTrue(conv.hasBias());
    }

    @Test
    public void testConv1dWithStridePadding() {
        Conv1d conv = new Conv1d(3, 16, 3, 2, 1);
        assertEquals(2, conv.stride());
        assertEquals(1, conv.padding());
    }

    @Test
    public void testConv1dWithAllParams() {
        Conv1d conv = new Conv1d(4, 8, 3, 2, 1, 1, 2, false);
        assertEquals(2, conv.groups());
        assertEquals(1, conv.dilation());
        assertFalse(conv.hasBias());
    }

    @Test
    public void testConv1dInvalidInChannels() {
        try {
            new Conv1d(0, 16, 3);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testConv1dInvalidKernelSize() {
        try {
            new Conv1d(3, 16, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("kernelSize"));
        }
    }

    @Test
    public void testConv1dGroupsNotDivisible() {
        try {
            new Conv1d(3, 16, 3, 1, 0, 1, 2, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible by groups"));
        }
    }

    @Test
    public void testConv1dToString() {
        Conv1d conv = new Conv1d(3, 16, 3);
        String str = conv.toString();
        assertTrue(str.contains("Conv1d"));
        assertTrue(str.contains("inChannels=3"));
        assertTrue(str.contains("outChannels=16"));
    }

    @Test
    public void testConvTranspose2dCreation() {
        ConvTranspose2d conv = new ConvTranspose2d(16, 3, new int[]{3, 3});
        assertNotNull(conv);
        assertEquals(16, conv.inChannels());
        assertEquals(3, conv.outChannels());
        assertArrayEquals(new int[]{3, 3}, conv.kernelSize());
        assertArrayEquals(new int[]{1, 1}, conv.stride());
        assertArrayEquals(new int[]{0, 0}, conv.padding());
        assertTrue(conv.hasBias());
    }

    @Test
    public void testConvTranspose2dWithStridePadding() {
        ConvTranspose2d conv = new ConvTranspose2d(16, 3, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 2}, conv.stride());
        assertArrayEquals(new int[]{1, 1}, conv.padding());
    }

    @Test
    public void testConvTranspose2dWithAllParams() {
        ConvTranspose2d conv = new ConvTranspose2d(8, 4, new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1},
                                                    new int[]{0, 0}, new int[]{1, 1}, 2, true);
        assertEquals(2, conv.groups());
        assertArrayEquals(new int[]{0, 0}, conv.outputPadding());
        assertArrayEquals(new int[]{1, 1}, conv.dilation());
    }

    @Test
    public void testConvTranspose2dInvalidInChannels() {
        try {
            new ConvTranspose2d(0, 3, new int[]{3, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testConvTranspose2dInvalidKernelSize() {
        try {
            new ConvTranspose2d(16, 3, new int[]{0, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("kernelSize"));
        }
    }

    @Test
    public void testConvTranspose2dGroupsNotDivisible() {
        try {
            new ConvTranspose2d(3, 16, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0},
                                new int[]{0, 0}, new int[]{1, 1}, 2, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("divisible by groups"));
        }
    }

    /** PyTorch's rule: each output padding must be smaller than its stride or its dilation. */
    @Test
    public void testConvTranspose2dOutputPaddingMustBeSmallerThanStrideOrDilation() {
        assertArrayEquals(new int[]{1, 1}, new ConvTranspose2d(4, 4, new int[]{3, 3}, new int[]{2, 2},
                new int[]{0, 0}, new int[]{1, 1}, new int[]{1, 1}, 1, true).outputPadding());
        assertArrayEquals(new int[]{1, 0}, new ConvTranspose2d(4, 4, new int[]{3, 3}, new int[]{1, 1},
                new int[]{0, 0}, new int[]{1, 0}, new int[]{2, 1}, 1, true).outputPadding());
        try {
            new ConvTranspose2d(4, 4, new int[]{3, 3}, new int[]{2, 2}, new int[]{0, 0},
                                new int[]{0, 2}, new int[]{1, 2}, 1, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("output padding must be smaller than either stride or dilation"));
        }
        try {
            ConvOps.requireTransposeOutputPadding(new int[]{1, 0}, new int[]{1, 1}, new int[]{1, 1});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("output padding must be smaller than either stride or dilation"));
        }
    }

    @Test
    public void testConvTranspose2dToString() {
        ConvTranspose2d conv = new ConvTranspose2d(16, 3, new int[]{3, 3});
        String str = conv.toString();
        assertTrue(str.contains("ConvTranspose2d"));
        assertTrue(str.contains("inChannels=16"));
        assertTrue(str.contains("outChannels=3"));
    }

    @Test
    public void testLinearForwardShapeValidation() {
        Linear linear = new Linear(10, 5);
        try {
            linear.weight();
            fail("Expected IllegalStateException for uninitialized layer");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testConv2dForwardShapeValidation() {
        Conv2d conv = new Conv2d(3, 16, new int[]{3, 3});
        try {
            conv.weight();
            fail("Expected IllegalStateException for uninitialized layer");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testConv1dForwardShapeValidation() {
        Conv1d conv = new Conv1d(3, 16, 3);
        try {
            conv.weight();
            fail("Expected IllegalStateException for uninitialized layer");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testConvTranspose2dForwardShapeValidation() {
        ConvTranspose2d conv = new ConvTranspose2d(16, 3, new int[]{3, 3});
        try {
            conv.weight();
            fail("Expected IllegalStateException for uninitialized layer");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testMaxPool2dCreation() {
        MaxPool2d pool = new MaxPool2d(new int[]{3, 3});
        assertNotNull(pool);
        assertArrayEquals(new int[]{3, 3}, pool.kernelSize());
        assertArrayEquals(new int[]{3, 3}, pool.stride());
        assertArrayEquals(new int[]{0, 0}, pool.padding());
    }

    @Test
    public void testMaxPool2dWithStridePadding() {
        MaxPool2d pool = new MaxPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 2}, pool.stride());
        assertArrayEquals(new int[]{1, 1}, pool.padding());
    }

    @Test
    public void testMaxPool2dInvalidKernelSize() {
        try {
            new MaxPool2d(new int[]{0, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("kernelSize"));
        }
    }

    @Test
    public void testMaxPool2dInvalidStride() {
        try {
            new MaxPool2d(new int[]{3, 3}, new int[]{0, 2}, new int[]{0, 0});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("stride"));
        }
    }

    @Test
    public void testMaxPool2dOutputShape() {
        MaxPool2d pool = new MaxPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0});
        Shape inputShape = Shape.of(1, 3, 8, 8);
        Shape outputShape = pool.computeOutputShape(inputShape);
        assertEquals(4, outputShape.dim(2));
        assertEquals(4, outputShape.dim(3));
    }

    @Test
    public void testMaxPool2dToString() {
        MaxPool2d pool = new MaxPool2d(new int[]{3, 3});
        String str = pool.toString();
        assertTrue(str.contains("MaxPool2d"));
        assertTrue(str.contains("kernelSize"));
    }

    @Test
    public void testAvgPool2dCreation() {
        AvgPool2d pool = new AvgPool2d(new int[]{3, 3});
        assertNotNull(pool);
        assertArrayEquals(new int[]{3, 3}, pool.kernelSize());
        assertArrayEquals(new int[]{3, 3}, pool.stride());
        assertArrayEquals(new int[]{0, 0}, pool.padding());
        assertTrue(pool.isCountIncludePad());
    }

    @Test
    public void testAvgPool2dWithStridePadding() {
        AvgPool2d pool = new AvgPool2d(new int[]{3, 3}, new int[]{2, 2}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 2}, pool.stride());
        assertArrayEquals(new int[]{1, 1}, pool.padding());
    }

    @Test
    public void testAvgPool2dInvalidKernelSize() {
        try {
            new AvgPool2d(new int[]{0, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("kernelSize"));
        }
    }

    @Test
    public void testAvgPool2dOutputShape() {
        AvgPool2d pool = new AvgPool2d(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0});
        Shape inputShape = Shape.of(1, 3, 8, 8);
        Shape outputShape = pool.computeOutputShape(inputShape);
        assertEquals(4, outputShape.dim(2));
        assertEquals(4, outputShape.dim(3));
    }

    @Test
    public void testAvgPool2dToString() {
        AvgPool2d pool = new AvgPool2d(new int[]{3, 3});
        String str = pool.toString();
        assertTrue(str.contains("AvgPool2d"));
        assertTrue(str.contains("kernelSize"));
    }

    @Test
    public void testAdaptiveAvgPool2dCreation() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(7, 7);
        assertNotNull(pool);
        assertEquals(7, pool.outputH());
        assertEquals(7, pool.outputW());
    }

    @Test
    public void testAdaptiveAvgPool2dSingleSize() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(7);
        assertEquals(7, pool.outputH());
        assertEquals(7, pool.outputW());
    }

    @Test
    public void testAdaptiveAvgPool2dInvalidOutputSize() {
        try {
            new AdaptiveAvgPool2d(0, 7);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outputH"));
        }
    }

    @Test
    public void testAdaptiveAvgPool2dOutputShape() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(7, 7);
        Shape inputShape = Shape.of(2, 3, 32, 64);
        Shape outputShape = pool.computeOutputShape(inputShape);
        assertEquals(2, outputShape.dim(0));
        assertEquals(3, outputShape.dim(1));
        assertEquals(7, outputShape.dim(2));
        assertEquals(7, outputShape.dim(3));
    }

    @Test
    public void testAdaptiveAvgPool2dToString() {
        AdaptiveAvgPool2d pool = new AdaptiveAvgPool2d(7);
        String str = pool.toString();
        assertTrue(str.contains("AdaptiveAvgPool2d"));
        assertTrue(str.contains("outputSize"));
    }

    @Test
    public void testDropoutCreation() {
        Dropout dropout = new Dropout(0.5f);
        assertNotNull(dropout);
        assertEquals(0.5f, dropout.dropRate(), 0.001f);
        assertEquals(0.5f, dropout.keepProb(), 0.001f);
    }

    @Test
    public void testDropoutDefaultRate() {
        Dropout dropout = new Dropout();
        assertEquals(0.5f, dropout.dropRate(), 0.001f);
    }

    @Test
    public void testDropoutInvalidRate() {
        try {
            new Dropout(1.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropRate"));
        }
    }

    @Test
    public void testDropoutNegativeRate() {
        try {
            new Dropout(-0.1f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropRate"));
        }
    }

    @Test
    public void testDropoutTrainingMode() {
        Dropout dropout = new Dropout(0.5f);
        assertTrue(dropout.isTraining());
    }

    @Test
    public void testDropoutEvalMode() {
        Dropout dropout = new Dropout(0.5f);
        dropout.eval();
        assertFalse(dropout.isTraining());
    }

    @Test
    public void testDropoutToString() {
        Dropout dropout = new Dropout(0.3f);
        String str = dropout.toString();
        assertTrue(str.contains("Dropout"));
        assertTrue(str.contains("dropRate=0.3"));
    }

    @Test
    public void testEmbeddingCreation() {
        Embedding embedding = new Embedding(1000, 128);
        assertNotNull(embedding);
        assertEquals(1000, embedding.numEmbeddings());
        assertEquals(128, embedding.embeddingDim());
        assertEquals(-1, embedding.paddingIdx());
    }

    @Test
    public void testEmbeddingWithPaddingIdx() {
        Embedding embedding = new Embedding(1000, 128, 0);
        assertEquals(0, embedding.paddingIdx());
    }

    @Test
    public void testEmbeddingInvalidNumEmbeddings() {
        try {
            new Embedding(0, 128);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numEmbeddings"));
        }
    }

    @Test
    public void testEmbeddingInvalidEmbeddingDim() {
        try {
            new Embedding(1000, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("embeddingDim"));
        }
    }

    @Test
    public void testEmbeddingInvalidPaddingIdx() {
        try {
            new Embedding(1000, 128, 1000);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("paddingIdx"));
        }
    }

    @Test
    public void testEmbeddingNotInitialized() {
        Embedding embedding = new Embedding(1000, 128);
        try {
            embedding.weight();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }

    @Test
    public void testEmbeddingOutputShape() {
        Embedding embedding = new Embedding(1000, 128);
        Shape inputShape = Shape.of(2, 10);
        Shape outputShape = embedding.computeOutputShape(inputShape);
        assertEquals(2, outputShape.dim(0));
        assertEquals(10, outputShape.dim(1));
        assertEquals(128, outputShape.dim(2));
    }

    @Test
    public void testEmbeddingToString() {
        Embedding embedding = new Embedding(1000, 128);
        String str = embedding.toString();
        assertTrue(str.contains("Embedding"));
        assertTrue(str.contains("numEmbeddings=1000"));
        assertTrue(str.contains("embeddingDim=128"));
    }
}

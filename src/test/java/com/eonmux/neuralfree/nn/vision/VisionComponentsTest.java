package com.eonmux.neuralfree.nn.vision;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Conv2d;
import com.eonmux.neuralfree.nn.norm.BatchNorm2d;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.*;

public class VisionComponentsTest {

    @Test
    public void testDepthwiseSeparableConvCreation() {
        DepthwiseSeparableConv conv = new DepthwiseSeparableConv(32, 64, new int[]{3, 3});
        assertNotNull(conv);
        assertEquals(32, conv.inChannels());
        assertEquals(64, conv.outChannels());
        assertArrayEquals(new int[]{3, 3}, conv.kernelSize());
        assertTrue(conv.hasBatchNorm());
    }

    @Test
    public void testDepthwiseSeparableConvWithStridePadding() {
        DepthwiseSeparableConv conv = new DepthwiseSeparableConv(32, 64, new int[]{3, 3}, 
                                                                   new int[]{2, 2}, new int[]{1, 1});
        assertArrayEquals(new int[]{2, 2}, conv.stride());
        assertArrayEquals(new int[]{1, 1}, conv.padding());
    }

    @Test
    public void testDepthwiseSeparableConvInvalidInChannels() {
        try {
            new DepthwiseSeparableConv(0, 64, new int[]{3, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testDepthwiseSeparableConvInvalidOutChannels() {
        try {
            new DepthwiseSeparableConv(32, 0, new int[]{3, 3});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outChannels"));
        }
    }

    @Test
    public void testDepthwiseSeparableConvToString() {
        DepthwiseSeparableConv conv = new DepthwiseSeparableConv(32, 64, new int[]{3, 3});
        String str = conv.toString();
        assertTrue(str.contains("DepthwiseSeparableConv"));
        assertTrue(str.contains("inChannels=32"));
        assertTrue(str.contains("outChannels=64"));
    }

    @Test
    public void testResidualBlockCreation() {
        ResidualBlock block = new ResidualBlock(64, 64);
        assertNotNull(block);
        assertEquals(64, block.inChannels());
        assertEquals(64, block.outChannels());
        assertArrayEquals(new int[]{1, 1}, block.stride());
        assertFalse(block.hasProjection());
    }

    @Test
    public void testResidualBlockWithStride() {
        ResidualBlock block = new ResidualBlock(64, 128, new int[]{2, 2});
        assertEquals(128, block.outChannels());
        assertArrayEquals(new int[]{2, 2}, block.stride());
        assertTrue(block.hasProjection());
    }

    @Test
    public void testResidualBlockWithProjection() {
        ResidualBlock block = new ResidualBlock(64, 128);
        assertTrue(block.hasProjection());
        assertNotNull(block.projection());
    }

    @Test
    public void testResidualBlockNoProjection() {
        ResidualBlock block = new ResidualBlock(64, 64);
        assertFalse(block.hasProjection());
        assertNull(block.projection());
    }

    @Test
    public void testResidualBlockInvalidInChannels() {
        try {
            new ResidualBlock(0, 64);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testResidualBlockInvalidStride() {
        try {
            new ResidualBlock(64, 128, new int[]{0, 2});
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("stride"));
        }
    }

    @Test
    public void testResidualBlockToString() {
        ResidualBlock block = new ResidualBlock(64, 128, new int[]{2, 2});
        String str = block.toString();
        assertTrue(str.contains("ResidualBlock"));
        assertTrue(str.contains("inChannels=64"));
        assertTrue(str.contains("outChannels=128"));
        assertTrue(str.contains("useProjection=true"));
    }

    @Test
    public void testBottleneckBlockCreation() {
        BottleneckBlock block = new BottleneckBlock(64, 64, 256);
        assertNotNull(block);
        assertEquals(64, block.inChannels());
        assertEquals(64, block.midChannels());
        assertEquals(256, block.outChannels());
        assertFalse(block.hasSe());
    }

    @Test
    public void testBottleneckBlockWithStride() {
        BottleneckBlock block = new BottleneckBlock(64, 64, 256, new int[]{2, 2});
        assertArrayEquals(new int[]{2, 2}, block.stride());
        assertTrue(block.hasProjection());
    }

    @Test
    public void testBottleneckBlockWithSe() {
        BottleneckBlock block = new BottleneckBlock(64, 64, 256, new int[]{1, 1}, true, 16);
        assertTrue(block.hasSe());
        assertEquals(16, block.seReduction());
    }

    @Test
    public void testBottleneckBlockInvalidMidChannels() {
        try {
            new BottleneckBlock(64, 0, 256);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("midChannels"));
        }
    }

    @Test
    public void testBottleneckBlockInvalidSeReduction() {
        try {
            new BottleneckBlock(64, 64, 256, new int[]{1, 1}, true, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("seReduction"));
        }
    }

    @Test
    public void testBottleneckBlockToString() {
        BottleneckBlock block = new BottleneckBlock(64, 64, 256, new int[]{1, 1}, true, 16);
        String str = block.toString();
        assertTrue(str.contains("BottleneckBlock"));
        assertTrue(str.contains("midChannels=64"));
        assertTrue(str.contains("outChannels=256"));
        assertTrue(str.contains("useSe=true"));
    }

    @Test
    public void testResNet18Creation() {
        ResNet resnet = ResNet.resnet18(1000);
        assertNotNull(resnet);
        assertEquals(1000, resnet.numClasses());
        assertFalse(resnet.usesBottleneck());
        assertArrayEquals(ResNet.RESNET18_DEPTHS, resnet.layerDepths());
    }

    @Test
    public void testResNet50Creation() {
        ResNet resnet = ResNet.resnet50(1000);
        assertNotNull(resnet);
        assertTrue(resnet.usesBottleneck());
        assertArrayEquals(ResNet.RESNET50_DEPTHS, resnet.layerDepths());
    }

    @Test
    public void testResNet101Creation() {
        ResNet resnet = ResNet.resnet101(1000);
        assertNotNull(resnet);
        assertTrue(resnet.usesBottleneck());
        assertArrayEquals(ResNet.RESNET101_DEPTHS, resnet.layerDepths());
    }

    @Test
    public void testResNet152Creation() {
        ResNet resnet = ResNet.resnet152(1000);
        assertNotNull(resnet);
        assertTrue(resnet.usesBottleneck());
        assertArrayEquals(ResNet.RESNET152_DEPTHS, resnet.layerDepths());
    }

    @Test
    public void testResNet34Creation() {
        ResNet resnet = ResNet.resnet34(1000);
        assertNotNull(resnet);
        assertFalse(resnet.usesBottleneck());
        assertArrayEquals(ResNet.RESNET34_DEPTHS, resnet.layerDepths());
    }

    @Test
    public void testResNetOutputChannels() {
        ResNet resnet18 = ResNet.resnet18(1000);
        assertEquals(512, resnet18.outputChannels());
        
        ResNet resnet50 = ResNet.resnet50(1000);
        assertEquals(2048, resnet50.outputChannels());
    }

    @Test
    public void testResNetFeatureExtractor() {
        ResNet resnet = ResNet.resnet18(0);
        assertNotNull(resnet);
        assertEquals(0, resnet.numClasses());
        assertNull(resnet.fc());
    }

    @Test
    public void testResNetAccessors() {
        ResNet resnet = ResNet.resnet18(1000);
        assertNotNull(resnet.conv1());
        assertNotNull(resnet.bn1());
        assertNotNull(resnet.maxpool());
        assertNotNull(resnet.avgpool());
        assertNotNull(resnet.fc());
    }

    @Test
    public void testResNetToString() {
        ResNet resnet18 = ResNet.resnet18(1000);
        String str = resnet18.toString();
        assertTrue(str.contains("ResNet-18"));
        assertTrue(str.contains("numClasses=1000"));
        
        ResNet resnet50 = ResNet.resnet50(1000);
        str = resnet50.toString();
        assertTrue(str.contains("ResNet-50"));
    }

    @Test
    public void testUNetCreation() {
        UNet unet = new UNet(3, 1);
        assertNotNull(unet);
        assertEquals(3, unet.inChannels());
        assertEquals(1, unet.outChannels());
        assertTrue(unet.usesBatchNorm());
        assertEquals(5, unet.depth());
    }

    @Test
    public void testUNetCustomChannels() {
        int[] channels = {32, 64, 128, 256};
        UNet unet = new UNet(3, 1, channels, true);
        assertArrayEquals(channels, unet.channelDepths());
        assertEquals(4, unet.depth());
    }

    @Test
    public void testUNetWithoutBatchNorm() {
        UNet unet = new UNet(3, 1, new int[]{64, 128, 256}, false);
        assertFalse(unet.usesBatchNorm());
    }

    @Test
    public void testUNetInvalidInChannels() {
        try {
            new UNet(0, 1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("inChannels"));
        }
    }

    @Test
    public void testUNetInvalidOutChannels() {
        try {
            new UNet(3, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("outChannels"));
        }
    }

    @Test
    public void testUNetInvalidChannelDepths() {
        try {
            new UNet(3, 1, new int[]{64}, true);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("channelDepths"));
        }
    }

    @Test
    public void testUNetAccessors() {
        UNet unet = new UNet(3, 1);
        assertNotNull(unet.encoderBlocks());
        assertNotNull(unet.decoderBlocks());
        assertNotNull(unet.finalConv());
    }

    @Test
    public void testUNetToString() {
        UNet unet = new UNet(3, 1);
        String str = unet.toString();
        assertTrue(str.contains("UNet"));
        assertTrue(str.contains("inChannels=3"));
        assertTrue(str.contains("outChannels=1"));
        assertTrue(str.contains("depth=5"));
    }

    @Test
    public void testResNet18ForwardOutputShape() {
        ResNet resnet = ResNet.resnet18(10);
        
        try {
            resnet.initialize(null);
            fail("Expected NullPointerException for null backend");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Backend"));
        }
    }

    @Test
    public void testResidualBlockNotInitialized() {
        ResidualBlock block = new ResidualBlock(64, 64);
        try {
            block.forward(null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testBottleneckBlockNotInitialized() {
        BottleneckBlock block = new BottleneckBlock(64, 64, 256);
        try {
            block.forward(null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testDepthwiseSeparableConvNotInitialized() {
        DepthwiseSeparableConv conv = new DepthwiseSeparableConv(32, 64, new int[]{3, 3});
        try {
            conv.forward(null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testUNetNotInitialized() {
        UNet unet = new UNet(3, 1);
        try {
            unet.forward(null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testResNetNotInitialized() {
        ResNet resnet = ResNet.resnet18(10);
        try {
            resnet.forward(null);
            fail("Expected NullPointerException or IllegalStateException");
        } catch (NullPointerException | IllegalStateException e) {
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    public void testResidualBlockInputValidation() {
        ResidualBlock block = new ResidualBlock(64, 64);
        try {
            block.initialize(null);
            fail("Expected NullPointerException for null backend");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Backend"));
        }
    }

    @Test
    public void testResNetInputValidation() {
        ResNet resnet = ResNet.resnet18(10);
        try {
            resnet.initialize(null);
            fail("Expected NullPointerException for null backend");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Backend"));
        }
    }

    @Test
    public void testUNetInputValidation() {
        UNet unet = new UNet(3, 1);
        try {
            unet.initialize(null);
            fail("Expected NullPointerException for null backend");
        } catch (NullPointerException e) {
            assertTrue(e.getMessage().contains("Backend"));
        }
    }
}

package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import static org.junit.Assert.*;

public class TransformerComponentsTest {
    
    @Test
    public void testSinusoidalPEProperties() {
        SinusoidalPE pe = new SinusoidalPE(512, 1024, 0.1f);
        
        assertEquals(512, pe.dModel());
        assertEquals(1024, pe.maxSequenceLength());
        assertEquals(0.1f, pe.dropoutRate(), 1e-6f);
        assertEquals("SinusoidalPE", pe.name());
        
        pe.close();
    }
    
    @Test
    public void testRotaryPEProperties() {
        RotaryPE pe = new RotaryPE(512, 1024, 256, 10000.0f);
        
        assertEquals(512, pe.dModel());
        assertEquals(1024, pe.maxSequenceLength());
        assertEquals(256, pe.rotaryDim());
        assertEquals(10000.0f, pe.base(), 1e-6f);
        assertEquals("RotaryPE", pe.name());
        
        pe.close();
    }
    
    @Test
    public void testRotaryPERequiresEvenRotaryDim() {
        try {
            new RotaryPE(512, 1024, 255, 10000.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("even"));
        }
    }
    
    @Test
    public void testLearnedPEProperties() {
        LearnedPE pe = new LearnedPE(512, 1024);
        
        assertEquals(512, pe.dModel());
        assertEquals(1024, pe.maxSequenceLength());
        assertEquals("LearnedPE", pe.name());
    }
    
    @Test
    public void testALiBiPEProperties() {
        ALiBiPE pe = new ALiBiPE(512, 1024, 8);
        
        assertEquals(512, pe.dModel());
        assertEquals(1024, pe.maxSequenceLength());
        assertEquals(8, pe.numHeads());
        assertEquals("ALiBiPE", pe.name());
        
        pe.close();
    }
    
    @Test
    public void testALiBiPESlopes() {
        ALiBiPE pe = new ALiBiPE(512, 1024, 8);
        float[] slopes = pe.getSlopes();
        
        assertEquals(8, slopes.length);
        for (int h = 0; h < slopes.length; h++) {
            assertEquals("slope of head " + h, (float) Math.pow(2.0, -(h + 1)), slopes[h], 0.0f);
            assertEquals(slopes[h], pe.getSlope(h), 0.0f);
        }
        
        pe.close();
    }
    
    @Test
    public void testALiBiPESlopesForNonPowerOfTwoHeads() {
        ALiBiPE pe = new ALiBiPE(768, 64, 12);
        float[] slopes = pe.getSlopes();
        
        assertEquals(12, slopes.length);
        for (int h = 0; h < 8; h++) {
            assertEquals("slope of head " + h, (float) Math.pow(2.0, -(h + 1)), slopes[h], 0.0f);
        }
        for (int h = 8; h < 12; h++) {
            double expected = Math.pow(2.0, -4.0 * (2 * (h - 8) + 1) / 8);
            assertEquals("slope of head " + h, expected, slopes[h], 1e-6);
        }
        assertEquals(Math.pow(2.0, -0.5), slopes[8], 1e-6);
        assertEquals(Math.pow(2.0, -3.5), slopes[11], 1e-6);
        
        pe.close();
    }
    
    @Test
    public void testTransformerEncoderLayerConfig() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(
            512, 8, 2048, 0.1f, true, 
            TransformerEncoderLayer.NormType.LAYER_NORM, 
            TransformerEncoderLayer.FFNType.GEGLU, 
            false
        );
        
        assertEquals(512, layer.dModel());
        assertEquals(8, layer.numHeads());
        assertEquals(2048, layer.dFF());
        assertEquals(0.1f, layer.dropoutRate(), 1e-6f);
        assertTrue(layer.isPreNorm());
        assertEquals(TransformerEncoderLayer.NormType.LAYER_NORM, layer.normType());
        assertEquals(TransformerEncoderLayer.FFNType.GEGLU, layer.ffnType());
        assertFalse(layer.isCausal());
    }
    
    @Test
    public void testTransformerEncoderLayerGLURequiresEvenDFF() {
        try {
            new TransformerEncoderLayer(
                512, 8, 2047, 0.1f, true,
                TransformerEncoderLayer.NormType.LAYER_NORM,
                TransformerEncoderLayer.FFNType.GLU,
                false
            );
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("even"));
        }
    }
    
    @Test
    public void testTransformerDecoderLayerConfig() {
        TransformerDecoderLayer layer = new TransformerDecoderLayer(
            512, 8, 2048, 0.1f, true,
            TransformerDecoderLayer.NormType.RMS_NORM,
            TransformerDecoderLayer.FFNType.MLP
        );
        
        assertEquals(512, layer.dModel());
        assertEquals(8, layer.numHeads());
        assertEquals(2048, layer.dFF());
        assertEquals(0.1f, layer.dropoutRate(), 1e-6f);
        assertTrue(layer.isPreNorm());
        assertEquals(TransformerDecoderLayer.NormType.RMS_NORM, layer.normType());
        assertEquals(TransformerDecoderLayer.FFNType.MLP, layer.ffnType());
    }
    
    @Test
    public void testTransformerEncoderConfig() {
        TransformerEncoder encoder = new TransformerEncoder(
            6, 512, 8, 2048, 0.1f, true,
            TransformerEncoderLayer.NormType.LAYER_NORM,
            TransformerEncoderLayer.FFNType.MLP,
            false
        );
        
        assertEquals(6, encoder.numLayers());
        assertEquals(512, encoder.dModel());
        assertEquals(8, encoder.numHeads());
        assertEquals(2048, encoder.dFF());
        assertEquals(0.1f, encoder.dropoutRate(), 1e-6f);
        assertTrue(encoder.isPreNorm());
        assertEquals(TransformerEncoderLayer.NormType.LAYER_NORM, encoder.normType());
        assertEquals(TransformerEncoderLayer.FFNType.MLP, encoder.ffnType());
        assertFalse(encoder.isCausal());
    }
    
    @Test
    public void testTransformerDecoderConfig() {
        TransformerDecoder decoder = new TransformerDecoder(
            6, 512, 8, 2048, 0.1f, true,
            TransformerDecoderLayer.NormType.LAYER_NORM,
            TransformerDecoderLayer.FFNType.MLP
        );
        
        assertEquals(6, decoder.numLayers());
        assertEquals(512, decoder.dModel());
        assertEquals(8, decoder.numHeads());
        assertEquals(2048, decoder.dFF());
        assertEquals(0.1f, decoder.dropoutRate(), 1e-6f);
        assertTrue(decoder.isPreNorm());
        assertEquals(TransformerDecoderLayer.NormType.LAYER_NORM, decoder.normType());
        assertEquals(TransformerDecoderLayer.FFNType.MLP, decoder.ffnType());
    }
    
    @Test
    public void testTransformerEncoderLayerNotInitializedThrows() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(512, 8);
        
        try {
            layer.selfAttention();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testTransformerDecoderLayerNotInitializedThrows() {
        TransformerDecoderLayer layer = new TransformerDecoderLayer(512, 8);
        
        try {
            layer.selfAttention();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testTransformerEncoderNotInitializedThrows() {
        TransformerEncoder encoder = new TransformerEncoder(6, 512, 8);
        
        try {
            encoder.layers();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testTransformerDecoderNotInitializedThrows() {
        TransformerDecoder decoder = new TransformerDecoder(6, 512, 8);
        
        try {
            decoder.layers();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("not initialized"));
        }
    }
    
    @Test
    public void testPositionalEncodingInterfaceDefaultMethod() {
        PositionalEncoding pe = new PositionalEncoding() {
            @Override
            public Tensor apply(Tensor input, int offset) {
                return input;
            }
            
            @Override
            public int maxSequenceLength() {
                return 1024;
            }
            
            @Override
            public int dModel() {
                return 512;
            }
            
            @Override
            public String name() {
                return "TestPE";
            }
        };
        
        assertEquals(1024, pe.maxSequenceLength());
        assertEquals(512, pe.dModel());
        assertEquals("TestPE", pe.name());
    }
    
    @Test
    public void testInvalidDModelThrows() {
        try {
            new SinusoidalPE(0, 1024);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dModel"));
        }
        
        try {
            new SinusoidalPE(-1, 1024);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dModel"));
        }
    }
    
    @Test
    public void testInvalidMaxSeqLenThrows() {
        try {
            new SinusoidalPE(512, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("maxSeqLen"));
        }
        
        try {
            new SinusoidalPE(512, -1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("maxSeqLen"));
        }
    }
    
    @Test
    public void testInvalidDropoutRateThrows() {
        try {
            new SinusoidalPE(512, 1024, -0.1f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropoutRate"));
        }
        
        try {
            new SinusoidalPE(512, 1024, 1.0f);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropoutRate"));
        }
    }
    
    @Test
    public void testInvalidNumHeadsThrows() {
        try {
            new TransformerEncoderLayer(512, 0);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numHeads"));
        }
        
        try {
            new TransformerEncoderLayer(512, -1);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numHeads"));
        }
    }
    
    @Test
    public void testInvalidNumLayersThrows() {
        try {
            new TransformerEncoder(0, 512, 8);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numLayers"));
        }
        
        try {
            new TransformerDecoder(-1, 512, 8);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("numLayers"));
        }
    }
    
    @Test
    public void testSinusoidalPETtoString() {
        SinusoidalPE pe = new SinusoidalPE(512, 1024, 0.1f);
        String str = pe.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("maxSeqLen=1024"));
        assertTrue(str.contains("SinusoidalPE"));
    }
    
    @Test
    public void testRotaryPEToString() {
        RotaryPE pe = new RotaryPE(512, 1024, 256, 10000.0f);
        String str = pe.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("rotaryDim=256"));
        assertTrue(str.contains("RotaryPE"));
    }
    
    @Test
    public void testLearnedPEToString() {
        LearnedPE pe = new LearnedPE(512, 1024);
        String str = pe.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("LearnedPE"));
    }
    
    @Test
    public void testALiBiPEToString() {
        ALiBiPE pe = new ALiBiPE(512, 1024, 8);
        String str = pe.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("numHeads=8"));
        assertTrue(str.contains("ALiBiPE"));
    }
    
    @Test
    public void testTransformerEncoderLayerToString() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(512, 8, 2048, 0.1f);
        String str = layer.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("numHeads=8"));
        assertTrue(str.contains("TransformerEncoderLayer"));
    }
    
    @Test
    public void testTransformerDecoderLayerToString() {
        TransformerDecoderLayer layer = new TransformerDecoderLayer(512, 8, 2048, 0.1f);
        String str = layer.toString();
        
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("numHeads=8"));
        assertTrue(str.contains("TransformerDecoderLayer"));
    }
    
    @Test
    public void testTransformerEncoderToString() {
        TransformerEncoder encoder = new TransformerEncoder(6, 512, 8);
        String str = encoder.toString();
        
        assertTrue(str.contains("numLayers=6"));
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("TransformerEncoder"));
    }
    
    @Test
    public void testTransformerDecoderToString() {
        TransformerDecoder decoder = new TransformerDecoder(6, 512, 8);
        String str = decoder.toString();
        
        assertTrue(str.contains("numLayers=6"));
        assertTrue(str.contains("dModel=512"));
        assertTrue(str.contains("TransformerDecoder"));
    }
    
    @Test
    public void testCloseWithoutInit() {
        SinusoidalPE sinPE = new SinusoidalPE(512, 1024);
        sinPE.close();
        
        RotaryPE rotPE = new RotaryPE(512, 1024);
        rotPE.close();
        
        ALiBiPE alibiPE = new ALiBiPE(512, 1024, 8);
        alibiPE.close();
    }
    
    @Test
    public void testEncoderDefaultConstructor() {
        TransformerEncoder encoder = new TransformerEncoder(6, 512, 8);
        
        assertEquals(6, encoder.numLayers());
        assertEquals(512, encoder.dModel());
        assertEquals(8, encoder.numHeads());
        assertEquals(512 * 4, encoder.dFF());
        assertEquals(0.1f, encoder.dropoutRate(), 1e-6f);
    }
    
    @Test
    public void testDecoderDefaultConstructor() {
        TransformerDecoder decoder = new TransformerDecoder(6, 512, 8);
        
        assertEquals(6, decoder.numLayers());
        assertEquals(512, decoder.dModel());
        assertEquals(8, decoder.numHeads());
        assertEquals(512 * 4, decoder.dFF());
        assertEquals(0.1f, decoder.dropoutRate(), 1e-6f);
    }
    
    @Test
    public void testEncoderLayerDefaultConstructor() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(512, 8);
        
        assertEquals(512, layer.dModel());
        assertEquals(8, layer.numHeads());
        assertEquals(512 * 4, layer.dFF());
        assertEquals(0.1f, layer.dropoutRate(), 1e-6f);
        assertTrue(layer.isPreNorm());
        assertFalse(layer.isCausal());
    }
    
    @Test
    public void testDecoderLayerDefaultConstructor() {
        TransformerDecoderLayer layer = new TransformerDecoderLayer(512, 8);
        
        assertEquals(512, layer.dModel());
        assertEquals(8, layer.numHeads());
        assertEquals(512 * 4, layer.dFF());
        assertEquals(0.1f, layer.dropoutRate(), 1e-6f);
        assertTrue(layer.isPreNorm());
    }
}

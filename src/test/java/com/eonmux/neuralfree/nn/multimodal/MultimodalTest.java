package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class MultimodalTest {

    @Test
    public void testVisionEncoderConstruction() {
        VisionEncoder encoder = new VisionEncoder(224, 16, 3, 256, 4, 8);
        
        assertEquals(224, encoder.imageSize());
        assertEquals(16, encoder.patchSize());
        assertEquals(3, encoder.inChannels());
        assertEquals(256, encoder.hiddenDim());
        assertEquals(4, encoder.numLayers());
        assertEquals(8, encoder.numHeads());
        assertTrue(encoder.useCLS());
        assertEquals(0.1f, encoder.dropout(), 0.001f);
        assertEquals("transformer", encoder.backboneType());
        assertFalse(encoder.isInitialized());
    }

    @Test
    public void testVisionEncoderInvalidConstruction() {
        try { new VisionEncoder(0, 16, 3, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("imageSize")); }
        try { new VisionEncoder(224, 0, 3, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("patchSize")); }
        try { new VisionEncoder(224, 16, 0, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("inChannels")); }
        try { new VisionEncoder(224, 16, 3, 0, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("hiddenDim")); }
        try { new VisionEncoder(223, 16, 3, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("divisible")); }
        try { new VisionEncoder(224, 16, 3, 256, 4, 8, "transformer", true, -0.1f); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("dropout")); }
    }

    @Test
    public void testVisionEncoderToString() {
        VisionEncoder encoder = new VisionEncoder(224, 16, 3, 256, 4, 8);
        String str = encoder.toString();
        assertTrue(str.contains("VisionEncoder"));
        assertTrue(str.contains("imageSize=224"));
        assertTrue(str.contains("patchSize=16"));
    }

    @Test
    public void testTextEncoderConstruction() {
        TextEncoder encoder = new TextEncoder(30000, 512, 6, 8, 512);
        
        assertEquals(30000, encoder.vocabSize());
        assertEquals(512, encoder.hiddenDim());
        assertEquals(6, encoder.numLayers());
        assertEquals(8, encoder.numHeads());
        assertEquals(512, encoder.maxSeqLen());
        assertEquals(0.1f, encoder.dropout(), 0.001f);
        assertEquals("transformer", encoder.backboneType());
        assertFalse(encoder.isInitialized());
    }

    @Test
    public void testTextEncoderInvalidConstruction() {
        try { new TextEncoder(0, 512, 6, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("vocabSize")); }
        try { new TextEncoder(30000, 0, 6, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("hiddenDim")); }
        try { new TextEncoder(30000, 512, -1, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numLayers")); }
        try { new TextEncoder(30000, 512, 6, 0, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numHeads")); }
        try { new TextEncoder(30000, 512, 6, 8, 0); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("maxSeqLen")); }
    }

    @Test
    public void testTextEncoderToString() {
        TextEncoder encoder = new TextEncoder(30000, 512, 6, 8, 512);
        String str = encoder.toString();
        assertTrue(str.contains("TextEncoder"));
        assertTrue(str.contains("vocabSize=30000"));
        assertTrue(str.contains("hiddenDim=512"));
    }

    @Test
    public void testCrossAttentionFusionConstruction() {
        CrossAttentionFusion fusion = new CrossAttentionFusion(512, 512, 512, 512, 8);
        
        assertEquals(512, fusion.queryDim());
        assertEquals(512, fusion.keyDim());
        assertEquals(512, fusion.valueDim());
        assertEquals(512, fusion.hiddenDim());
        assertEquals(8, fusion.numHeads());
        assertEquals(0.1f, fusion.dropout(), 0.001f);
        assertTrue(fusion.gated());
        assertFalse(fusion.isInitialized());
    }

    @Test
    public void testCrossAttentionFusionInvalidConstruction() {
        try { new CrossAttentionFusion(0, 512, 512, 512, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("queryDim")); }
        try { new CrossAttentionFusion(512, 0, 512, 512, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("keyDim")); }
        try { new CrossAttentionFusion(512, 512, 0, 512, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("valueDim")); }
        try { new CrossAttentionFusion(512, 512, 512, 0, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("hiddenDim")); }
        try { new CrossAttentionFusion(512, 512, 512, 512, 0); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numHeads")); }
    }

    @Test
    public void testCrossAttentionFusionToString() {
        CrossAttentionFusion fusion = new CrossAttentionFusion(512, 512, 512, 512, 8);
        String str = fusion.toString();
        assertTrue(str.contains("CrossAttentionFusion"));
        assertTrue(str.contains("queryDim=512"));
        assertTrue(str.contains("numHeads=8"));
    }

    @Test
    public void testVisionLanguageModelConstruction() {
        VisionLanguageModel vlm = new VisionLanguageModel(30000, 256, 512, 512, 4, 8, 512);
        
        assertEquals(30000, vlm.vocabSize());
        assertEquals(256, vlm.imageHiddenDim());
        assertEquals(512, vlm.textHiddenDim());
        assertEquals(512, vlm.fusionHiddenDim());
        assertEquals(4, vlm.numFusionLayers());
        assertEquals(8, vlm.numHeads());
        assertEquals(512, vlm.maxSeqLen());
        assertFalse(vlm.isInitialized());
    }

    @Test
    public void testVisionLanguageModelInvalidConstruction() {
        try { new VisionLanguageModel(0, 256, 512, 512, 4, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("vocabSize")); }
        try { new VisionLanguageModel(30000, 0, 512, 512, 4, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("imageHiddenDim")); }
        try { new VisionLanguageModel(30000, 256, 0, 512, 4, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("textHiddenDim")); }
        try { new VisionLanguageModel(30000, 256, 512, 0, 4, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("fusionHiddenDim")); }
        try { new VisionLanguageModel(30000, 256, 512, 512, -1, 8, 512); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numFusionLayers")); }
    }

    @Test
    public void testVisionLanguageModelToString() {
        VisionLanguageModel vlm = new VisionLanguageModel(30000, 256, 512, 512, 4, 8, 512);
        String str = vlm.toString();
        assertTrue(str.contains("VisionLanguageModel"));
        assertTrue(str.contains("vocabSize=30000"));
    }

    @Test
    public void testAudioEncoderConstruction() {
        AudioEncoder encoder = new AudioEncoder(16000, 80, 256, 4, 8);
        
        assertEquals(16000, encoder.sampleRate());
        assertEquals(80, encoder.nMels());
        assertEquals(400, encoder.nFft());
        assertEquals(160, encoder.hopLength());
        assertEquals(256, encoder.hiddenDim());
        assertEquals(4, encoder.numLayers());
        assertEquals(8, encoder.numHeads());
        assertEquals("transformer", encoder.encoderType());
        assertEquals(0.1f, encoder.dropout(), 0.001f);
        assertFalse(encoder.isInitialized());
    }

    @Test
    public void testAudioEncoderFullConstruction() {
        AudioEncoder encoder = new AudioEncoder(16000, 80, 512, 256, 256, 4, 8, "cnn", 0.2f);
        
        assertEquals(16000, encoder.sampleRate());
        assertEquals(80, encoder.nMels());
        assertEquals(512, encoder.nFft());
        assertEquals(256, encoder.hopLength());
        assertEquals(256, encoder.hiddenDim());
        assertEquals(4, encoder.numLayers());
        assertEquals(8, encoder.numHeads());
        assertEquals("cnn", encoder.encoderType());
        assertEquals(0.2f, encoder.dropout(), 0.001f);
    }

    @Test
    public void testAudioEncoderInvalidConstruction() {
        try { new AudioEncoder(0, 80, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("sampleRate")); }
        try { new AudioEncoder(16000, 0, 256, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("nMels")); }
        try { new AudioEncoder(16000, 80, 0, 4, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("hiddenDim")); }
        try { new AudioEncoder(16000, 80, 256, -1, 8); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numLayers")); }
        try { new AudioEncoder(16000, 80, 256, 4, 0); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numHeads")); }
    }

    @Test
    public void testAudioEncoderToString() {
        AudioEncoder encoder = new AudioEncoder(16000, 80, 256, 4, 8);
        String str = encoder.toString();
        assertTrue(str.contains("AudioEncoder"));
        assertTrue(str.contains("sampleRate=16000"));
        assertTrue(str.contains("nMels=80"));
    }

    @Test
    public void testModalityRouterConstruction() {
        ModalityRouter router = new ModalityRouter(512, 3);
        
        assertEquals(512, router.hiddenDim());
        assertEquals(3, router.numModalities());
        assertEquals("softmax", router.routingStrategy());
        assertEquals(1.0f, router.routingTemperature(), 0.001f);
        assertFalse(router.useTopK());
        assertEquals(1, router.topK());
        assertFalse(router.isInitialized());
    }

    @Test
    public void testModalityRouterCustomConstruction() {
        ModalityRouter router = new ModalityRouter(512, 3, "sigmoid", 0.5f, true, 2);
        
        assertEquals(512, router.hiddenDim());
        assertEquals(3, router.numModalities());
        assertEquals("sigmoid", router.routingStrategy());
        assertEquals(0.5f, router.routingTemperature(), 0.001f);
        assertTrue(router.useTopK());
        assertEquals(2, router.topK());
    }

    @Test
    public void testModalityRouterInvalidConstruction() {
        try { new ModalityRouter(0, 3); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("hiddenDim")); }
        try { new ModalityRouter(512, 0); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("numModalities")); }
        try { new ModalityRouter(512, 3, "softmax", 0, false, 1); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("routingTemperature")); }
        try { new ModalityRouter(512, 3, "softmax", 1.0f, true, 0); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("topK")); }
        try { new ModalityRouter(512, 3, "softmax", 1.0f, true, 4); fail("Expected IllegalArgumentException"); } catch (IllegalArgumentException e) { assertTrue(e.getMessage().contains("topK")); }
    }

    @Test
    public void testModalityRouterConstants() {
        assertEquals("vision", ModalityRouter.MODALITY_VISION);
        assertEquals("text", ModalityRouter.MODALITY_TEXT);
        assertEquals("audio", ModalityRouter.MODALITY_AUDIO);
    }

    @Test
    public void testModalityRouterToString() {
        ModalityRouter router = new ModalityRouter(512, 3);
        String str = router.toString();
        assertTrue(str.contains("ModalityRouter"));
        assertTrue(str.contains("hiddenDim=512"));
        assertTrue(str.contains("numModalities=3"));
    }

    @Test
    public void testVisionEncoderNotInitializedThrows() {
        VisionEncoder encoder = new VisionEncoder(224, 16, 3, 256, 4, 8);
        try { encoder.encode(null); fail("Expected IllegalStateException"); } catch (IllegalStateException e) { assertTrue(e.getMessage().contains("initialized")); }
    }

    @Test
    public void testTextEncoderNotInitializedThrows() {
        TextEncoder encoder = new TextEncoder(30000, 512, 6, 8, 512);
        try { encoder.encode(null); fail("Expected IllegalStateException"); } catch (IllegalStateException e) { assertTrue(e.getMessage().contains("initialized")); }
    }

    @Test
    public void testCrossAttentionFusionNotInitializedThrows() {
        CrossAttentionFusion fusion = new CrossAttentionFusion(512, 512, 512, 512, 8);
        try { fusion.forward(null, null, null); fail("Expected IllegalStateException"); } catch (IllegalStateException e) { assertTrue(e.getMessage().contains("initialized")); }
    }

    @Test
    public void testModalityRouterRoutingResult() {
        float[] weights = {0.7f, 0.2f, 0.1f};
        int[] selected = {0};
        List<String> names = Arrays.asList("vision", "text", "audio");
        
        ModalityRouter.RoutingResult result = new ModalityRouter.RoutingResult(weights, selected, names);
        
        assertArrayEquals(weights, result.weights(), 0.001f);
        assertArrayEquals(selected, result.selectedModalities());
        assertEquals(3, result.modalityNames().size());
        assertEquals(0.7f, result.getWeight(0, 0), 0.001f);
        assertEquals("vision", result.getSelectedModalityName(0));
        assertEquals(0, result.getSelectedModalityIndex(0));
    }

    @Test
    public void testVisionEncoderNoUngatedConstruction() {
        VisionEncoder encoder = new VisionEncoder(224, 16, 3, 256, 4, 8, "transformer", false, 0.0f);
        assertFalse(encoder.useCLS());
        assertEquals(0.0f, encoder.dropout(), 0.001f);
    }

    @Test
    public void testCrossAttentionFusionNoGate() {
        CrossAttentionFusion fusion = new CrossAttentionFusion(512, 512, 512, 512, 8, 0.0f, false);
        assertFalse(fusion.gated());
    }
}

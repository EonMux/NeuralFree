package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The clip.* metadata parsing of ClipHparams against the defaults and refusals of clip.cpp load_hparams. */
public class ClipHparamsTest {

    private static Map<String, Object> visionMeta() {
        Map<String, Object> m = baseMeta("mlp");
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_mean", new float[]{0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[]{0.268f, 0.261f, 0.275f});
        m.put("clip.use_gelu", true);
        return m;
    }

    private static Map<String, Object> baseMeta(String projector) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_vision_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.vision.embedding_length", 1024);
        m.put("clip.vision.feed_forward_length", 4096);
        m.put("clip.vision.block_count", 24);
        m.put("clip.vision.projection_dim", 768);
        m.put("clip.vision.attention.head_count", 16);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-5f);
        m.put("clip.vision.image_mean", new float[]{0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[]{0.268f, 0.261f, 0.275f});
        return m;
    }

    private static ClipHparams parse(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.VISION);
    }

    @Test
    public void visionKeysParseWithTheReferenceDefaults() {
        ClipHparams hp = parse(visionMeta());
        assertEquals(ProjectorType.MLP, hp.projectorType());
        assertEquals(1024, hp.nEmbd());
        assertEquals(4096, hp.nFf());
        assertEquals(24, hp.nLayer());
        assertEquals(768, hp.projectionDim());
        assertEquals(16, hp.nHead());
        assertEquals(16, hp.nHeadKv());
        assertEquals(64, hp.dHead());
        assertEquals(1e-5f, hp.eps(), 0f);
        assertEquals(ClipFfnOp.GELU, hp.ffnOp());
        assertEquals(ClipHparams.PatchMerge.FLAT, hp.patchMerge());
        assertEquals(336, hp.imageSize());
        assertEquals(14, hp.patchSize());
        assertEquals(1, hp.nMerge());
        assertEquals(0, hp.nWaPattern());
        assertEquals(0, hp.featureLayers().length);
        assertArrayEquals(new float[]{0.481f, 0.457f, 0.408f}, hp.imageMean(), 0f);
    }

    @Test
    public void qwenDefaultsAndKeysResolveAsClipCpp() {
        Map<String, Object> m = baseMeta("qwen2.5o");
        m.put("clip.vision.projector_type", "qwen2.5vl_merger");
        m.put("clip.vision.image_size", 560);
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.n_wa_pattern", 2);
        m.put("clip.vision.window_size", 112);
        m.put("clip.vision.attention.head_count_kv", 2);
        m.put("clip.vision.feature_layer", new int[]{-2, -7});
        m.put("clip.use_silu", true);
        ClipHparams hp = parse(m);
        assertEquals(ProjectorType.QWEN25VL, hp.projectorType());
        assertEquals(64, hp.dHead());
        assertEquals(2, hp.nMerge());
        assertEquals(2, hp.nWaPattern());
        assertEquals(112, hp.attnWindowSize());
        assertArrayEquals(new int[]{-2, -7}, hp.featureLayers());
        assertEquals(ClipFfnOp.SILU, hp.ffnOp());
    }

    @Test
    public void qwenMergeDefaultsToTwoWhenTheKeyIsAbsent() {
        Map<String, Object> m = baseMeta("qwen2vl_merger");
        m.put("clip.vision.image_size", 28);
        m.put("clip.vision.patch_size", 14);
        assertEquals(2, parse(m).nMerge());
        assertEquals(1, parse(visionMeta()).nMerge());
    }

    @Test
    public void missingKeysAndBadValuesAreNamed() {
        Map<String, Object> m = visionMeta();
        m.remove("clip.vision.feed_forward_length");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> parse(m));
        assertTrue(e.getMessage().contains("clip.vision.feed_forward_length"));

        Map<String, Object> noProjector = visionMeta();
        noProjector.remove("clip.projector_type");
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(noProjector)).getMessage()
                .contains("clip.projector_type"));

        Map<String, Object> unknown = visionMeta();
        unknown.put("clip.projector_type", "warp_drive");
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(unknown)).getMessage()
                .contains("warp_drive"));

        Map<String, Object> badHeads = visionMeta();
        badHeads.put("clip.vision.attention.head_count_kv", 5);
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(badHeads)).getMessage()
                .contains("head_count_kv"));

        Map<String, Object> badGrid = visionMeta();
        badGrid.put("clip.vision.image_size", 340);
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(badGrid)).getMessage()
                .contains("image_size"));

        Map<String, Object> badFeature = visionMeta();
        badFeature.put("clip.vision.feature_layer", new int[]{30});
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(badFeature)).getMessage()
                .contains("feature_layer"));

        Map<String, Object> noColors = visionMeta();
        noColors.remove("clip.vision.image_mean");
        assertTrue(assertThrows(GgufFormatException.class, () -> parse(noColors)).getMessage()
                .contains("image_mean"));
    }

    private static Map<String, Object> audioMeta(String projector) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_audio_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.audio.num_mel_bins", 128);
        m.put("clip.audio.embedding_length", 1280);
        m.put("clip.audio.feed_forward_length", 5120);
        m.put("clip.audio.block_count", 32);
        m.put("clip.audio.projection_dim", 4096);
        m.put("clip.audio.attention.head_count", 20);
        m.put("clip.audio.attention.layer_norm_epsilon", 1e-5f);
        return m;
    }

    private static ClipHparams parseAudio(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.AUDIO);
    }

    @Test
    public void audioModalityUsesItsOwnPrefixAndFlag() {
        Map<String, Object> m = audioMeta("ultravox");
        m.put("clip.audio.projector.stack_factor", 8);
        ClipHparams hp = parseAudio(m);
        assertEquals(ProjectorModality.AUDIO, hp.modality());
        assertEquals(0, hp.imageSize());
        assertEquals(1, hp.patchSize());
        assertEquals(1280, hp.nEmbd());
        assertEquals(64, hp.dHead());
        assertEquals(ClipFfnOp.GELU_ERF, hp.ffnOp());
        ClipAudioHparams audio = hp.audio();
        assertEquals(128, audio.nMelBins());
        assertEquals(8, audio.stackFactor());
        assertEquals(30, audio.chunkSeconds());
        assertEquals(16000, audio.sampleRate());
        assertEquals(400, audio.nFft());
        assertEquals(400, audio.windowLen());
        assertEquals(160, audio.hopLen());
        assertTrue(audio.hasMelFrontend());
        assertThrows(IllegalArgumentException.class,
                () -> ClipHparams.from(GgufKv.ofMap(m), ProjectorModality.VISION));
        assertThrows(IllegalStateException.class, () -> parse(visionMeta()).audio());
    }

    @Test
    public void audioFramingFollowsTheProjectorTable() {
        ClipAudioHparams qwen3a = parseAudio(audioMeta("qwen3a")).audio();
        assertEquals(30, qwen3a.chunkSeconds());
        assertEquals(160, qwen3a.hopLen());
        assertEquals(0, qwen3a.stackFactor());

        Map<String, Object> qwen2a = audioMeta("qwen2a");
        qwen2a.put("clip.use_silu", true);
        // clip.cpp forces gelu_erf on every whisper-family tower whatever the ffn flags say
        assertEquals(ClipFfnOp.GELU_ERF, parseAudio(qwen2a).ffnOp());

        ClipAudioHparams lfm2a = parseAudio(audioMeta("lfm2a")).audio();
        assertEquals(1, lfm2a.chunkSeconds());
        assertEquals(512, lfm2a.nFft());
        assertEquals(400, lfm2a.windowLen());

        Map<String, Object> gemma = audioMeta("gemma4a");
        assertEquals(1e-6f, parseAudio(gemma).eps(), 0f);
        assertEquals(320, parseAudio(gemma).audio().windowLen());
    }

    @Test
    public void missingAudioKeysAreNamed() {
        Map<String, Object> noMel = audioMeta("qwen2a");
        noMel.remove("clip.audio.num_mel_bins");
        assertTrue(assertThrows(GgufFormatException.class, () -> parseAudio(noMel)).getMessage()
                .contains("clip.audio.num_mel_bins"));

        for (String stacked : new String[]{"ultravox", "voxtral", "meralion", "glma"}) {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> parseAudio(audioMeta(stacked)));
            assertTrue(stacked + ": " + e.getMessage(),
                    e.getMessage().contains("clip.audio.projector.stack_factor"));
        }

        Map<String, Object> zeroStack = audioMeta("qwen2a");
        zeroStack.put("clip.audio.projector.stack_factor", 0);
        assertTrue(assertThrows(GgufFormatException.class, () -> parseAudio(zeroStack)).getMessage()
                .contains("stack_factor"));

        assertTrue(assertThrows(GgufFormatException.class, () -> parseAudio(audioMeta("granite_speech")))
                .getMessage().contains("clip.audio.chunk_size"));
    }
}

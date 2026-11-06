package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioFixture.HIDDEN;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.N_MEL;
import static com.eonmux.neuralfree.gguf.mm.AudioFixture.TEXT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The audio geometry resolved from the clip hparams and tensor table, against the clip.cpp loader's checks. */
public class ClipAudioModelTest {

    private static ClipAudioModel model(MmprojFixture f, String projector, int stack) {
        return ClipAudioModel.from(AudioFixture.hparams(AudioFixture.whisperMeta(projector, stack)), f.catalog());
    }

    private static void reshape(MmprojFixture f, String name, long... ne) {
        f.remove(name);
        f.weight(name, ne);
    }

    @Test
    public void whisperGeometryResolvesEveryProjector() {
        String[] projectors = {"ultravox", "qwen2a", "voxtral", "musicflamingo", "meralion", "glma"};
        int[] stacks = {3, 0, 3, 0, 3, 2};
        for (int i = 0; i < projectors.length; i++) {
            ClipAudioModel m = model(AudioFixture.whisper(projectors[i], stacks[i], 1), projectors[i], stacks[i]);
            assertTrue(projectors[i], m.isWhisper());
            assertEquals(ProjectorEntry.ENCODER_WHISPER, m.encoderGraph());
            assertEquals(TEXT, m.outputWidth());
            assertEquals(AudioFixture.LAYERS, m.layers().size());
            assertEquals(Math.max(1, stacks[i]), m.stackFactor());
            assertEquals(AudioFixture.MAX_POS, m.maxPositions());
            assertNull(m.convChannels());
        }
    }

    @Test
    public void tokenCountsFollowTheGraphs() {
        // 3000 frames -> 1500 positions after the stride-2 conv
        assertEquals(750, model(AudioFixture.whisper("qwen2a", 0, 1), "qwen2a", 0).nTokens(3000));
        assertEquals(500, model(AudioFixture.whisper("ultravox", 3, 1), "ultravox", 3).nTokens(3000));
        assertEquals(250, model(AudioFixture.whisper("voxtral", 3, 1), "voxtral", 3).nTokens(3000));
        assertEquals(752, model(AudioFixture.whisper("glma", 2, 1), "glma", 2).nTokens(3000));
        ClipAudioModel qwen3a = ClipAudioModel.from(AudioFixture.hparams(AudioFixture.qwen3aMeta()),
                AudioFixture.qwen3a(1).catalog());
        assertFalse(qwen3a.isWhisper());
        assertEquals(30 * ClipAudioModel.QWEN3A_TOKENS_PER_CHUNK, qwen3a.nTokens(3000));
        assertEquals(3, qwen3a.convChannels().length);
        assertThrows(IllegalArgumentException.class, () -> qwen3a.nTokens(0));
    }

    @Test
    public void melFrontendCarriesTheProjectorFraming() {
        ClipAudioModel m = model(AudioFixture.whisper("qwen2a", 0, 1), "qwen2a", 0);
        assertEquals(MelFrontend.Flavor.WHISPER, m.melFlavor());
        MelFrontend.Config c = m.melConfig();
        assertEquals(N_MEL, c.nMelBins());
        assertEquals(16000, c.sampleRate());
        assertEquals(400, c.nFft());
        assertEquals(400, c.windowLen());
        assertEquals(160, c.hopLen());
        ClipAudioModel qwen3a = ClipAudioModel.from(AudioFixture.hparams(AudioFixture.qwen3aMeta()),
                AudioFixture.qwen3a(1).catalog());
        assertEquals(MelFrontend.Flavor.QWEN3A, qwen3a.melFlavor());
        assertEquals(AudioFixture.Q3_MEL, qwen3a.melConfig().nMelBins());
    }

    @Test
    public void missingProjectorTensorIsRefusedByName() {
        MmprojFixture f = AudioFixture.whisper("ultravox", 3, 1);
        f.remove("mm.a.norm_mid.weight");
        MissingTensorException e = assertThrows(MissingTensorException.class, () -> model(f, "ultravox", 3));
        assertTrue(e.getMessage(), e.getMessage().contains("mm.a.norm_mid.weight"));
    }

    @Test
    public void whisperKeyBiasIsRefusedByName() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 1);
        f.weight("a.blk.1.attn_k.bias", HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> model(f, "qwen2a", 0));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.1.attn_k.bias"));
    }

    @Test
    public void flatConvBiasIsRefused() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 1);
        reshape(f, "a.conv1d.1.bias", HIDDEN);
        assertTrue(assertThrows(IllegalArgumentException.class, () -> model(f, "qwen2a", 0)).getMessage()
                .contains("a.conv1d.1.bias"));
    }

    @Test
    public void projectorWidthOtherThanTheProjectionDimIsRefused() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 1);
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen2a", 0);
        meta.put("clip.audio.projection_dim", TEXT + 1);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioFixture.hparams(meta), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("mm.a.fc"));
    }

    @Test
    public void stackedWidthMustMatchTheStackFactor() {
        MmprojFixture f = AudioFixture.whisper("ultravox", 3, 1);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> model(f, "ultravox", 2));
        assertTrue(e.getMessage(), e.getMessage().contains("mm.a.norm_pre.weight"));
    }

    @Test
    public void missingStackFactorIsRefusedByKeyName() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> AudioFixture.hparams(AudioFixture.whisperMeta("ultravox", 0)));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.audio.projector.stack_factor"));
    }

    @Test
    public void visionProjectorDeclaredForTheAudioEncoderIsRefused() {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 1);
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen2a", 0);
        meta.put("clip.projector_type", "mlp");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioFixture.hparams(meta), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("VISION"));
    }

    @Test
    public void visionHparamsAreRefused() {
        ClipHparams vision = ClipHparams.from(GgufKv.ofMap(visionMeta()), ProjectorModality.VISION);
        assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(vision, AudioFixture.whisper("qwen2a", 0, 1).catalog()));
    }

    private static Map<String, Object> visionMeta() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_vision_encoder", true);
        m.put("clip.projector_type", "mlp");
        m.put("clip.vision.image_size", 8);
        m.put("clip.vision.patch_size", 4);
        m.put("clip.vision.embedding_length", 8);
        m.put("clip.vision.feed_forward_length", 12);
        m.put("clip.vision.block_count", 2);
        m.put("clip.vision.projection_dim", 6);
        m.put("clip.vision.attention.head_count", 2);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        m.put("clip.vision.image_mean", new float[]{0.5f, 0.5f, 0.5f});
        m.put("clip.vision.image_std", new float[]{0.5f, 0.5f, 0.5f});
        return m;
    }
}

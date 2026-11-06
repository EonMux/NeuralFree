package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Opening the qwen3tts generator file and the geometry the two graphs report. */
public class Qwen3TtsClipTest {

    @Test
    public void opensThePredictorAndDecoderGeometry() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(7L);
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(f.write().toString())) {
            Qwen3TtsCodePredictor p = clip.predictor();
            assertEquals(Qwen3TtsClipFixture.PROJ, p.nEmbd());
            assertEquals(Qwen3TtsClipFixture.TALKER, p.talkerWidth());
            assertEquals(Qwen3TtsClipFixture.PRED_LAYERS, p.nLayer());
            assertEquals(Qwen3TtsClipFixture.PAGES, p.nCodebooks());
            assertEquals(Qwen3TtsClipFixture.VOCAB, p.vocab());
            assertEquals(Qwen3TtsClipFixture.VOCAB0, p.vocab0());
            assertEquals(Qwen3TtsClipFixture.PAGES + 1, p.codesPerFrame());

            Qwen3TtsCode2Wav c = clip.code2Wav();
            assertEquals(Qwen3TtsClipFixture.WINDOW, c.window());
            assertEquals(Qwen3TtsClipFixture.HOP, c.hop());
            assertEquals(24000, c.sampleRate());
            assertEquals(Qwen3TtsClipFixture.PAGES, c.codebooks());
            assertEquals(Qwen3TtsClipFixture.PAGES + 1, c.codesPerFrame());
            assertEquals(Qwen3TtsClipFixture.TFM_LAYERS, c.nLayer());
            assertEquals(Arrays.asList("tfm_pos", "tfm_k_0", "tfm_v_0", "pre_conv", "up0_dwconv",
                    "dac_entry", "dac0_tail", "dac0_res0", "dac_post_conv"), c.stateSlots());
            assertNotNull(clip.path());
            assertTrue(clip.toString().contains("Qwen3TtsClip"));
        }
    }

    @Test
    public void refusesAnotherArchitecture() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(1L);
        String path = f.writer().architecture("llama").writeTempFile().toString();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Qwen3TtsClip.open(path));
        assertTrue(e.getMessage().contains("general.architecture"));
    }

    @Test
    public void refusesAnotherProjectorType() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(1L);
        String path = f.writer().str("clip.gen.audio.projector_type", "qwen3tts_spkenc").writeTempFile().toString();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Qwen3TtsClip.open(path));
        assertTrue(e.getMessage().contains("qwen3tts_gen"));
    }

    @Test
    public void refusesAMissingCodePredictorTensor() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(1L);
        String path = f.writer().removeTensor("a.gen.code.head.weight").writeTempFile().toString();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Qwen3TtsClip.open(path));
        assertTrue(e.getMessage().contains("a.gen.code.head.weight"));
    }

    @Test
    public void refusesAMissingDecoderTensor() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(1L);
        String path = f.writer().removeTensor("a.gen.wav.dac.blk.0.conv.weight").writeTempFile().toString();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Qwen3TtsClip.open(path));
        assertTrue(e.getMessage().contains("a.gen.wav.dac.blk.0.conv.weight"));
    }

    @Test
    public void refusesCodebookCountsThatDisagree() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(1L);
        // the code predictor head advertises three pages while its embedding table and the decoder hold two
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = f.writer();
        w.removeTensor("a.gen.code.head.weight");
        float[] values = new float[Qwen3TtsClipFixture.PROJ * Qwen3TtsClipFixture.VOCAB * 3];
        w.tensorF32("a.gen.code.head.weight", new long[]{Qwen3TtsClipFixture.PROJ, Qwen3TtsClipFixture.VOCAB, 3},
                values);
        String path = w.writeTempFile().toString();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> Qwen3TtsClip.open(path));
        assertTrue(e.getMessage().contains("codebook"));
    }
}

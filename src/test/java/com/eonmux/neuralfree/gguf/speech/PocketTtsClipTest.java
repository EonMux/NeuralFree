package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link PocketTtsClip}: the opened generator file and the refusals of another family or a broken tensor table. */
public class PocketTtsClipTest {

    @Test
    public void opensTheFixtureAndBindsItsGraphsAndPack() throws IOException {
        try (PocketTtsClip clip = PocketTtsClip.open(new PocketTtsFixtures(1L, "english").write().toString())) {
            assertNotNull(clip.flow());
            assertNotNull(clip.decoder());
            assertNotNull(clip.path());
            assertEquals("english", clip.pack().variant());
            assertEquals(PocketTtsFixtures.LATENT, clip.flow().latent());
            assertEquals(clip.flow().latent(), clip.decoder().latent());
        }
    }

    @Test
    public void refusesAnotherArchitecture() throws IOException {
        GgufTestWriter writer = new PocketTtsFixtures(2L).writer();
        writer.architecture("qwen3tts");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> PocketTtsClip.open(writer.writeTempFile().toString()));
        assertTrue(e.getMessage().contains("qwen3tts"));
    }

    @Test
    public void refusesAnotherGeneratorFamily() throws IOException {
        GgufTestWriter writer = new PocketTtsFixtures(3L).writer()
                .str("clip.gen.audio.projector_type", "qwen3tts_gen");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> PocketTtsClip.open(writer.writeTempFile().toString()));
        assertTrue(e.getMessage().contains("qwen3tts_gen"));
    }

    @Test
    public void refusesAFlowNetworkWhoseLatentsTheDecoderCannotTake() throws IOException {
        GgufTestWriter writer = new PocketTtsFixtures(4L).writer();
        writer.removeTensor("a.gen.wav.quant_out.weight");
        assertThrows(RuntimeException.class, () -> PocketTtsClip.open(writer.writeTempFile().toString()));
    }

    @Test
    public void closeIsIdempotent() throws IOException {
        PocketTtsClip clip = PocketTtsClip.open(new PocketTtsFixtures(5L).write().toString());
        clip.close();
        clip.close();
    }
}

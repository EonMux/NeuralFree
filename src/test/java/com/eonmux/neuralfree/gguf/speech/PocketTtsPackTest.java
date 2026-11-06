package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** {@link PocketTtsPack}: the reference's per-variant temperature, tail length, padding and language. */
public class PocketTtsPackTest {

    @Test
    public void knownVariantsCarryTheirReferenceSettings() {
        PocketTtsPack english = PocketTtsPack.forVariant("english");
        assertEquals(0.3f, english.temperature(), 0f);
        assertEquals(3, english.framesAfterEos(50));
        // a zero frames_after_eos falls back to the prompt-length guess: 5 up to six tokens, 3 above
        assertEquals(5, english.framesAfterEos(5));
        assertFalse(english.padShortText());
        assertEquals("english", english.language());

        PocketTtsPack january = PocketTtsPack.forVariant("english_2026-01");
        assertEquals(0.7f, january.temperature(), 0f);
        assertTrue(january.padShortText());

        PocketTtsPack april = PocketTtsPack.forVariant("english_2026-04");
        assertEquals(0.3f, april.temperature(), 0f);
        assertFalse(april.padShortText());

        PocketTtsPack french = PocketTtsPack.forVariant("french_24l");
        assertEquals(0.7f, french.temperature(), 0f);
        assertEquals(8, french.framesAfterEos(1));
        assertEquals(8, french.framesAfterEos(100));
        assertEquals("french", french.language());
    }

    @Test
    public void unknownVariantsUseTheModelDefaults() {
        PocketTtsPack pack = PocketTtsPack.forVariant("something-new");
        assertEquals(PocketTtsFlow.DEFAULT_TEMPERATURE, pack.temperature(), 0f);
        assertEquals(5, pack.framesAfterEos(4));
        assertEquals(3, pack.framesAfterEos(7));
        assertFalse(pack.padShortText());
        assertEquals("english", pack.language());
        assertEquals("something-new", pack.variant());
    }

    @Test
    public void blankAndMissingVariantsAreDefaults() {
        assertEquals(PocketTtsFlow.DEFAULT_TEMPERATURE, PocketTtsPack.forVariant(null).temperature(), 0f);
        assertEquals("", PocketTtsPack.forVariant(null).variant());
        assertEquals(PocketTtsFlow.DEFAULT_TEMPERATURE, PocketTtsPack.forVariant("  ").temperature(), 0f);
    }

    @Test
    public void readsTheVariantMetadataOfAGeneratorFile() throws java.io.IOException {
        GgufTestWriter writer = new PocketTtsFixtures(1L, "french_24l").writer();
        try (com.eonmux.neuralfree.gguf.ModelSource source = com.eonmux.neuralfree.gguf.ModelSources.open(
                writer.writeTempFile().toString(),
                com.eonmux.neuralfree.gguf.ModelSourceOptions.fromSystemProperties())) {
            PocketTtsPack pack = PocketTtsPack.of(com.eonmux.neuralfree.gguf.hparams.GgufKv.of(source));
            assertEquals("french_24l", pack.variant());
            assertEquals("french", pack.language());
        }
    }
}

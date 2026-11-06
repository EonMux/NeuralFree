package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link SpeechOptions}: defaults, withers, ISO language resolution and the immutable value semantics. */
public class SpeechOptionsTest {

    @Test
    public void defaultsFollowTheReference() {
        SpeechOptions o = SpeechOptions.defaults();
        assertEquals("", o.language());
        assertEquals("", o.voice());
        assertEquals(0L, o.seed());
        assertEquals(50, o.topK());
        assertEquals(1.0f, o.topP(), 0f);
        assertEquals(512, o.maxFrames());
        assertEquals(0, o.chunkFrames());
        assertFalse(o.hasVoice());
    }

    @Test
    public void withersReturnCopiesAndValidate() {
        SpeechOptions a = SpeechOptions.defaults();
        SpeechOptions b = a.withLanguage("fr").withVoice("alice").withSeed(7).withTopK(3).withTopP(0.9f)
                .withMaxFrames(4).withChunkFrames(2);
        assertEquals("", a.language());
        assertEquals("fr", b.language());
        assertEquals("alice", b.voice());
        assertEquals(7L, b.seed());
        assertEquals(3, b.topK());
        assertEquals(0.9f, b.topP(), 0f);
        assertEquals(4, b.maxFrames());
        assertEquals(2, b.chunkFrames());
        assertTrue(b.hasVoice());
        assertThrows(IllegalArgumentException.class, () -> a.withTopK(-1));
        assertThrows(IllegalArgumentException.class, () -> a.withTopP(1.5f));
        assertThrows(IllegalArgumentException.class, () -> a.withTopP(Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> a.withMaxFrames(0));
        assertThrows(IllegalArgumentException.class, () -> a.withChunkFrames(-1));
    }

    @Test
    public void blankLanguageAndVoiceMeanTheReferenceDefault() {
        SpeechOptions o = SpeechOptions.defaults().withLanguage("  ").withVoice(null);
        assertEquals("", o.language());
        assertEquals("", o.voice());
        assertEquals("english", SpeechOptions.resolveLanguage(o.language()));
    }

    @Test
    public void resolveLanguageMapsIsoCodes() {
        assertEquals("english", SpeechOptions.resolveLanguage(null));
        assertEquals("english", SpeechOptions.resolveLanguage("en"));
        assertEquals("english", SpeechOptions.resolveLanguage("EN"));
        assertEquals("chinese", SpeechOptions.resolveLanguage("zh"));
        assertEquals("japanese", SpeechOptions.resolveLanguage("ja"));
        assertEquals("french", SpeechOptions.resolveLanguage("fr"));
        assertEquals("german", SpeechOptions.resolveLanguage("de"));
        assertEquals("korean", SpeechOptions.resolveLanguage("ko"));
        assertEquals("russian", SpeechOptions.resolveLanguage("ru"));
        assertEquals("spanish", SpeechOptions.resolveLanguage("es"));
        assertEquals("italian", SpeechOptions.resolveLanguage("it"));
        assertEquals("portuguese", SpeechOptions.resolveLanguage("pt"));
        assertEquals("chinese", SpeechOptions.resolveLanguage("chinese"));
    }

    @Test
    public void resolveLanguageRefusesFreeText() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeechOptions.resolveLanguage("not a language!"));
        assertTrue(e.getMessage().contains("language"));
    }

    @Test
    public void valueSemantics() {
        SpeechOptions a = SpeechOptions.of("en", "", 1L, 5, 0.5f);
        SpeechOptions b = SpeechOptions.of("en", "", 1L, 5, 0.5f);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, b.withTopK(6));
        assertNotEquals(a, b.withSeed(2L));
    }
}

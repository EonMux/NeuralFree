package com.eonmux.neuralfree.gguf.media;

import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaLimits}: vLLM's {@code --limit-mm-per-prompt} count and object forms, the 999 default of an unset
 * modality, the refusal message of a prompt over the limit, and the named refusals of malformed settings.
 */
public class MediaLimitsTest {

    @Test
    public void unsetModalitiesAllowVllmsDefault() {
        MediaLimits limits = MediaLimits.parse(null);
        for (Modality m : Modality.values()) {
            assertEquals(MediaLimits.DEFAULT_LIMIT, limits.limit(m));
        }
        assertEquals(MediaLimits.DEFAULT_LIMIT, MediaLimits.parse("  ").limit(Modality.IMAGE));
        assertEquals(MediaLimits.DEFAULT_LIMIT, MediaLimits.defaults().limit(Modality.AUDIO));
    }

    @Test
    public void countAndObjectFormsSetTheirModality() {
        MediaLimits limits = MediaLimits.parse("{\"image\": 2, \"audio\": {\"count\": 0}}");
        assertEquals(2, limits.limit(Modality.IMAGE));
        assertEquals(0, limits.limit(Modality.AUDIO));
        assertEquals(MediaLimits.DEFAULT_LIMIT, limits.limit(Modality.VIDEO));
        assertEquals(MediaLimits.DEFAULT_LIMIT, MediaLimits.parse("{\"video\": {}}").limit(Modality.VIDEO));
    }

    @Test
    public void aPromptOverTheLimitIsRefusedWithVllmsMessage() {
        MediaLimits limits = MediaLimits.parse("{\"image\": 1, \"audio\": 0}");
        limits.check(Modality.IMAGE, 1);
        limits.check(Modality.AUDIO, 0);
        MediaInputException e = assertThrows(MediaInputException.class, () -> limits.check(Modality.IMAGE, 2));
        assertEquals("At most 1 image(s) may be provided in one prompt.", e.getMessage());
        e = assertThrows(MediaInputException.class, () -> limits.check(Modality.AUDIO, 1));
        assertEquals("At most 0 audio(s) may be provided in one prompt.", e.getMessage());
    }

    @Test
    public void malformedSettingsAreRefusedByName() {
        assertRefused("[1]", "must be a JSON object");
        assertRefused("{", "not valid JSON");
        assertRefused("{\"vision\": 1}", "'vision'");
        assertRefused("{\"image\": -1}", "non-negative whole number");
        assertRefused("{\"image\": 1.5}", "non-negative whole number");
        assertRefused("{\"image\": \"2\"}", "non-negative whole number");
        assertRefused("{\"image\": {\"count\": 1, \"width\": 512}}", "option 'width'");
    }

    private static void assertRefused(String json, String fragment) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> MediaLimits.parse(json));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--limit-mm-per-prompt"));
        assertTrue(e.getMessage(), e.getMessage().contains(fragment));
    }
}

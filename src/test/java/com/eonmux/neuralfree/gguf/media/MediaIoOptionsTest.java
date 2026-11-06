package com.eonmux.neuralfree.gguf.media;

import com.eonmux.neuralfree.gguf.image.ImageLoadOptions;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's {@code --media-io-kwargs} as {@link MediaIoOptions}: the image loader's {@code image_mode} and
 * {@code rgba_background_color} parse in vLLM's JSON form and become hash factors of image keys, an empty option set
 * is no factor, and every option or modality the loaders here do not take is refused by name.
 */
public class MediaIoOptionsTest {

    private static String refusal(String json) {
        return assertThrows(IllegalArgumentException.class, () -> MediaIoOptions.parse(json)).getMessage();
    }

    @Test
    public void unsetOptionsAreVllmsDefaultsWithoutHashFactors() {
        for (String json : new String[] {null, "", "  ", "{}", "{\"image\": {}}", "{\"audio\": {}}"}) {
            MediaIoOptions io = MediaIoOptions.parse(json);
            assertEquals(json, ImageLoadOptions.defaults(), io.image());
            assertEquals(json, "", io.hashFactors(Modality.IMAGE));
            assertTrue(json, io.isDefault());
        }
    }

    @Test
    public void imageOptionsParseInVllmsFormAndHashTheirModalityOnly() {
        MediaIoOptions io = MediaIoOptions.parse(
                "{\"image\": {\"image_mode\": \"L\", \"rgba_background_color\": [0, 128, 255]}}");
        assertEquals(ImageLoadOptions.Mode.L, io.image().mode());
        assertEquals(0x0080FF, io.image().background());
        String factors = io.hashFactors(Modality.IMAGE);
        assertTrue(factors, factors.contains("image_mode") && factors.contains("[0,128,255]"));
        assertEquals("the audio items keep their keys", "", io.hashFactors(Modality.AUDIO));
        assertEquals("key order does not matter", factors, MediaIoOptions.parse(
                "{\"image\": {\"rgba_background_color\": [0, 128, 255], \"image_mode\": \"L\"}}")
                .hashFactors(Modality.IMAGE));
        assertNotEquals("an explicit default is still a factor, as in vLLM", "",
                MediaIoOptions.parse("{\"image\": {\"image_mode\": \"RGB\"}}").hashFactors(Modality.IMAGE));
    }

    @Test
    public void optionsTheLoadersDoNotTakeAreRefusedByName() {
        assertTrue(refusal("{\"text\": {}}").contains("'text'"));
        assertTrue(refusal("{\"audio\": {\"audio_backend\": \"pyav\"}}").contains("'audio_backend'"));
        assertTrue(refusal("{\"video\": {\"backend\": \"pyav\"}}").contains("'backend'"));
        assertTrue(refusal("{\"image\": {\"max_pixels\": 100}}").contains("'max_pixels'"));
        assertTrue(refusal("{\"image\": {\"image_mode\": \"RGBA\"}}").contains("\"RGBA\""));
        assertTrue(refusal("{\"image\": {\"image_mode\": null}}").contains("image_mode null"));
        assertTrue(refusal("{\"image\": {\"rgba_background_color\": [256, 0, 0]}}")
                .contains("rgba_background_color must be a list of 3 integers in the range [0, 255]"));
        assertTrue(refusal("{\"image\": {\"rgba_background_color\": [1, 2]}}").contains("rgba_background_color"));
        assertTrue(refusal("{\"image\": {\"rgba_background_color\": [1.5, 2, 3]}}").contains("rgba_background_color"));
        assertTrue(refusal("{\"image\": \"L\"}").contains("image must be a JSON object"));
        assertTrue(refusal("{\"video\": {\"num_frames\": 0}}").contains("num_frames"));
        assertTrue(refusal("{\"video\": {\"fps\": 0}}").contains("fps"));
        assertTrue(refusal("{\"video\": {\"max_duration\": -1}}").contains("max_duration"));
        assertTrue(refusal("[1]").startsWith(MediaIoOptions.FLAG + " must be a JSON object"));
        assertTrue(refusal("{nope").startsWith(MediaIoOptions.FLAG + " is not valid JSON"));
    }

    @Test
    public void videoOptionsParseInVllmsFormAndHashTheirModalityOnly() {
        MediaIoOptions io = MediaIoOptions.parse("{\"video\": {\"num_frames\": 40, \"fps\": 2}}");
        assertEquals(40, io.video().numFrames());
        assertEquals(2.0, io.video().fps(), 0);
        assertEquals(300, io.video().maxDurationSeconds());
        String factors = io.hashFactors(Modality.VIDEO);
        assertTrue(factors, factors.contains("num_frames") && factors.contains("fps=2"));
        assertEquals("the image items keep their keys", "", io.hashFactors(Modality.IMAGE));
        assertEquals("the audio items keep their keys", "", io.hashFactors(Modality.AUDIO));
        assertEquals("key order does not matter", factors, MediaIoOptions.parse(
                "{\"video\": {\"fps\": 2, \"num_frames\": 40}}").hashFactors(Modality.VIDEO));
        // an explicit frame cap becomes a factor, an empty section is none
        assertNotEquals("", MediaIoOptions.parse("{\"video\": {\"max_duration\": 60}}").hashFactors(Modality.VIDEO));
        assertNotEquals("an explicitly configured default still changes the key", "",
                MediaIoOptions.parse("{\"video\": {\"max_duration\": 300}}").hashFactors(Modality.VIDEO));
        assertTrue(MediaIoOptions.parse("{\"video\": {}}").isDefault());
        // unset keeps vLLM's -1 sentinel and the 300 s default
        MediaIoOptions.Video unset = MediaIoOptions.parse("{\"audio\": {}}").video();
        assertEquals(MediaIoOptions.Video.UNSET, unset.numFrames());
        assertEquals(MediaIoOptions.Video.UNSET, unset.fps(), 0);
        assertEquals(MediaIoOptions.Video.DEFAULT_MAX_DURATION_SECONDS, unset.maxDurationSeconds());
        assertEquals(60, MediaIoOptions.parse("{\"video\": {\"max_duration\": 60}}").video()
                .maxDurationSeconds());
    }
}

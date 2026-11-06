package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.video.VideoOptions;
import com.eonmux.neuralfree.gguf.video.VideoPruner;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The video input settings of the {@code neuralfree.server.video.*} system properties (llama.cpp's
 * {@code --video-ffmpeg-dir}, {@code --video-fps} and {@code --video-timestamp-interval}, vLLM's
 * {@code --video-pruning-rate} and {@code --video-pruning-method}): parsing, the defaults, and the refusals.
 */
class MediaOptionsVideoTest {

    private static final String[] PROPERTIES = {MediaOptions.VIDEO_FFMPEG_DIR_PROPERTY,
            MediaOptions.VIDEO_FPS_PROPERTY, MediaOptions.VIDEO_TIMESTAMP_PROPERTY,
            MediaOptions.VIDEO_PRUNING_RATE_PROPERTY, MediaOptions.VIDEO_PRUNING_METHOD_PROPERTY};

    private final Map<String, String> saved = new HashMap<>();

    @BeforeEach
    void save() {
        for (String property : PROPERTIES) {
            saved.put(property, System.getProperty(property));
            System.clearProperty(property);
        }
    }

    @AfterEach
    void restore() {
        saved.forEach((property, value) -> {
            if (value == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, value);
            }
        });
    }

    private static VideoOptions video() {
        return MediaOptions.fromSystemProperties().video();
    }

    @Test
    void theDefaultsFollowLlamaCppAndPruneNothing() {
        VideoOptions options = video();
        assertNull(options.ffmpegDir(), "the binaries come from the PATH");
        assertEquals(4.0, options.fpsTarget(), 0);
        assertEquals(5000, options.timestampIntervalMs());
        assertEquals(0, options.pruningRate(), 0);
        assertEquals(VideoPruner.Method.EVS, options.pruningMethod());
        assertFalse(options.prunes());
    }

    @Test
    void thePropertiesConfigureTheDecoder(@TempDir Path binDir) {
        System.setProperty(MediaOptions.VIDEO_FFMPEG_DIR_PROPERTY, " " + binDir + " ");
        System.setProperty(MediaOptions.VIDEO_FPS_PROPERTY, " 2.5 ");
        System.setProperty(MediaOptions.VIDEO_TIMESTAMP_PROPERTY, "1000");
        System.setProperty(MediaOptions.VIDEO_PRUNING_RATE_PROPERTY, "0.5");
        System.setProperty(MediaOptions.VIDEO_PRUNING_METHOD_PROPERTY, " VIDCOM2 ");
        VideoOptions options = video();
        assertEquals(binDir, options.ffmpegDir());
        assertEquals(2.5, options.fpsTarget(), 0);
        assertEquals(1000, options.timestampIntervalMs());
        assertEquals(0.5, options.pruningRate(), 0);
        assertEquals(VideoPruner.Method.VIDCOM2, options.pruningMethod());
        assertTrue(options.prunes());
    }

    @Test
    void malformedPropertiesRefuseByName() {
        System.setProperty(MediaOptions.VIDEO_FPS_PROPERTY, "fast");
        assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
        System.setProperty(MediaOptions.VIDEO_FPS_PROPERTY, "4");
        System.setProperty(MediaOptions.VIDEO_FPS_PROPERTY, "-1");
        assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
        System.setProperty(MediaOptions.VIDEO_FPS_PROPERTY, "4");
        System.setProperty(MediaOptions.VIDEO_TIMESTAMP_PROPERTY, "-1");
        assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
        System.setProperty(MediaOptions.VIDEO_TIMESTAMP_PROPERTY, "5000");
        System.setProperty(MediaOptions.VIDEO_PRUNING_RATE_PROPERTY, "1");
        assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
        System.setProperty(MediaOptions.VIDEO_PRUNING_RATE_PROPERTY, "0.25");
        System.setProperty(MediaOptions.VIDEO_PRUNING_METHOD_PROPERTY, "attention");
        assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
    }

    @Test
    void aMissingFfmpegDirectoryRefusesByName() {
        System.setProperty(MediaOptions.VIDEO_FFMPEG_DIR_PROPERTY, "/definitely/not/a/directory");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, MediaOptionsVideoTest::video);
        assertTrue(e.getMessage().contains("ffmpeg directory"), e.getMessage());
    }
}

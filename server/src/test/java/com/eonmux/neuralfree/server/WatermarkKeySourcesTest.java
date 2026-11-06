package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The watermark key sources the server settings read: the {@code NEURALFREE_WATERMARK_KEY} environment variable,
 * the file {@code NEURALFREE_WATERMARK_KEY_FILE} names, and vLLM's JSON {@code key}, with the refusal when an
 * environment source and the JSON key are both given.
 */
class WatermarkKeySourcesTest {

    @TempDir
    Path dir;

    @Test
    void theEnvironmentSuppliesTheKeyWithoutAConfigProperty() {
        withConfigProperty(null, () -> {
            ServerOptions options = ServerOptions.fromSystemProperties(
                    Map.of(WatermarkConfig.ENV_KEY, " 42 "));
            assertNotNull(options.watermark());
            assertEquals(42L, options.watermark().key());
            assertEquals(WatermarkAlgorithm.GUMBEL, options.watermark().algorithm());
        });
    }

    @Test
    void theEnvironmentKeyCarriesTheJsonSettings() {
        withConfigProperty("{\"algorithm\": \"dual_key_gumbel\", \"context_width\": 2}", () -> {
            ServerOptions options = ServerOptions.fromSystemProperties(
                    Map.of(WatermarkConfig.ENV_KEY, "42"));
            assertNotNull(options.watermark());
            assertEquals(42L, options.watermark().key());
            assertEquals(WatermarkAlgorithm.DUAL_KEY_GUMBEL, options.watermark().algorithm());
            assertEquals(2, options.watermark().contextWidth());
        });
    }

    @Test
    void theKeyComesFromTheFileTheEnvironmentNames() throws IOException {
        Path file = dir.resolve("watermark.key");
        Files.write(file, "18446744073709551615\n".getBytes(StandardCharsets.UTF_8));
        withConfigProperty(null, () -> {
            ServerOptions options = ServerOptions.fromSystemProperties(
                    Map.of(WatermarkConfig.ENV_KEY_FILE, file.toString()));
            assertNotNull(options.watermark());
            assertEquals(-1L, options.watermark().key());
        });
    }

    @Test
    void anEnvironmentKeyBesideTheJsonKeyIsRefusedByName() {
        withConfigProperty("{\"key\": 7}", () -> {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> ServerOptions.fromSystemProperties(Map.of(WatermarkConfig.ENV_KEY, "42")));
            assertTrue(e.getMessage().contains(WatermarkConfig.ENV_KEY), e.getMessage());
            assertTrue(e.getMessage().contains("key"), e.getMessage());
        });
    }

    @Test
    void theJsonKeyStillWorksWithoutAnEnvironmentSource() {
        withConfigProperty("{\"key\": 7, \"algorithm\": \"dual_key_gumbel\"}", () -> {
            ServerOptions options = ServerOptions.fromSystemProperties(Collections.emptyMap());
            assertNotNull(options.watermark());
            assertEquals(7L, options.watermark().key());
            assertEquals(WatermarkAlgorithm.DUAL_KEY_GUMBEL, options.watermark().algorithm());
        });
    }

    @Test
    void noSourceServesWithoutAWatermark() {
        withConfigProperty(null, () -> {
            assertNull(ServerOptions.fromSystemProperties(Collections.emptyMap()).watermark());
        });
    }

    /** Runs {@code body} with {@code neuralfree.watermark.config} set to {@code json} (null clears it). */
    private static void withConfigProperty(String json, Runnable body) {
        String saved = System.getProperty(WatermarkConfig.PROP_CONFIG);
        try {
            if (json == null) {
                System.clearProperty(WatermarkConfig.PROP_CONFIG);
            } else {
                System.setProperty(WatermarkConfig.PROP_CONFIG, json);
            }
            body.run();
        } finally {
            if (saved == null) {
                System.clearProperty(WatermarkConfig.PROP_CONFIG);
            } else {
                System.setProperty(WatermarkConfig.PROP_CONFIG, saved);
            }
        }
    }
}

package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.mm.EncoderCacheConfig;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConnector;
import com.eonmux.neuralfree.gguf.mm.MmEncoderTpMode;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The encoder scale-out settings of {@link MediaOptions#fromSystemProperties()}: the rank group and parallelism of
 * the vision encoder ({@code neuralfree.mmproj.tp.mode}, {@code neuralfree.mmproj.devices}) and the shared encoder
 * cache of the deployment ({@code neuralfree.mm.encoder.cache.dir} and {@code .role}) reach the media options.
 */
class MediaOptionsEncoderCacheTest {

    private static final String[] PROPERTIES = {MmprojOptions.PROP_TP_MODE, MmprojOptions.PROP_DEVICES,
            EncoderCacheConfig.PROP_DIR, EncoderCacheConfig.PROP_ROLE};

    @TempDir
    Path dir;

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

    @Test
    void noPropertiesKeepTheEncodingsLocalAndTheEncoderWhole() {
        MediaOptions options = MediaOptions.fromSystemProperties();
        assertNull(options.encoderCache());
        assertEquals(MmEncoderTpMode.WEIGHTS, options.encoder().tpMode());
        assertEquals(List.of(), options.encoder().devices());
    }

    @Test
    void theSharedEncoderCacheReachesTheMediaOptions() {
        System.setProperty(EncoderCacheConfig.PROP_DIR, dir.toString());
        System.setProperty(EncoderCacheConfig.PROP_ROLE, "ec_consumer");
        EncoderCacheConfig cache = MediaOptions.fromSystemProperties().encoderCache();
        assertEquals(dir, cache.directory());
        assertEquals(EncoderCacheConnector.Role.CONSUMER, cache.role());
    }

    @Test
    void theDataParallelRankGroupReachesTheEncoderOptions() {
        System.setProperty(MmprojOptions.PROP_TP_MODE, "data");
        System.setProperty(MmprojOptions.PROP_DEVICES, "0,1");
        MmprojOptions encoder = MediaOptions.fromSystemProperties().encoder();
        assertEquals(MmEncoderTpMode.DATA, encoder.tpMode());
        assertEquals(Arrays.asList("0", "1"), encoder.devices());
    }

    @Test
    void aRankGroupWithoutDataModeIsRefusedByPropertyName() {
        System.setProperty(MmprojOptions.PROP_DEVICES, "0,1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                MediaOptions::fromSystemProperties);
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICES), e.getMessage());
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_TP_MODE), e.getMessage());
    }
}

package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.Config;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The HIP mode plumbing: {@code neuralfree.backend=hip} parses through the engine config and the
 * acceleration mode.
 */
public class HipModeWiringTest {

    @Test
    public void backendModeParsesHip() {
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.HIP,
                com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.parse("hip"));
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.HIP,
                com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.parse(" HIP "));
    }

    @Test
    public void backendModeMapsToTheAccelerationMode() {
        assertEquals(Config.ACCELERATION_MODE.HIP,
                com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.HIP.toAccelerationMode());
    }

    @Test
    public void invalidBackendNamesTheValidModes() {
        try {
            com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.parse("rocm");
            fail("an unknown backend mode must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("hip"));
        }
    }

    @Test
    public void neuralFreeBackendParsingAcceptsHip() {
        assertEquals(Config.ACCELERATION_MODE.HIP, com.eonmux.neuralfree.NeuralFree.parseBackend("hip"));
    }
}

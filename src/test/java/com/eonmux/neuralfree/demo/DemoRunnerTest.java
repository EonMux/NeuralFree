package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.Config;
import org.junit.Test;

import java.util.Locale;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link DemoRunner}: the usage text names every API {@link com.eonmux.neuralfree.NeuralFree#parseBackend} accepts
 * ({@code auto}, the built-in APIs and the accelerator APIs), and {@code --backend} takes them, metal included.
 */
public class DemoRunnerTest {

    @Test
    public void theUsageNamesEveryAccelerationMode() {
        for (Config.ACCELERATION_MODE mode : Config.ACCELERATION_MODE.values()) {
            assertTrue("the usage names " + mode + ": " + DemoRunner.USAGE,
                    DemoRunner.USAGE.contains(mode.name().toLowerCase(Locale.ROOT)));
        }
    }

    @Test
    public void theBackendFlagTakesHipAndMetal() {
        assertEquals(Config.ACCELERATION_MODE.HIP, DemoRunner.parseBackend(new String[]{"--backend", "hip"}));
        assertEquals(Config.ACCELERATION_MODE.METAL, DemoRunner.parseBackend(new String[]{"--backend", "Metal"}));
    }
}

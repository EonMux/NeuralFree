package com.eonmux.neuralfree.server;

import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The sleep-mode switch (vLLM's {@code --enable-sleep-mode}): off by default, the property decides it, unusable
 * values refuse by name, the launch flag sets the property, and the off switch refuses the checkpoint window with
 * the unregistered route's 404 and the sleep operations with a 400 naming the flag.
 */
public class SleepModeSwitchTest {

    @Test
    void theSwitchIsOffByDefault() {
        assertFalse(SleepModeSwitch.from(Map.of()).enabled());
        assertFalse(SleepModeSwitch.from(Map.of(SleepModeSwitch.PROPERTY, "  ")).enabled());
    }

    @Test
    void thePropertyDecidesBothWays() {
        assertTrue(SleepModeSwitch.from(Map.of(SleepModeSwitch.PROPERTY, "true")).enabled());
        assertTrue(SleepModeSwitch.from(Map.of(SleepModeSwitch.PROPERTY, "TRUE")).enabled());
        assertFalse(SleepModeSwitch.from(Map.of(SleepModeSwitch.PROPERTY, "false")).enabled());
    }

    @Test
    void unusableValuesRefuseByName() {
        for (String value : new String[]{"yes", "1", "on"}) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> SleepModeSwitch.from(Map.of(SleepModeSwitch.PROPERTY, value)), value);
            assertTrue(refused.getMessage().startsWith(SleepModeSwitch.PROPERTY), refused.getMessage());
            assertTrue(refused.getMessage().contains(value), refused.getMessage());
        }
    }

    @Test
    void theLaunchFlagSetsTheProperty() {
        assertEquals("true", ServerLaunchArgs.parse("--enable-sleep-mode").properties()
                .get(SleepModeSwitch.PROPERTY));
    }

    @Test
    void theOffSwitchRefusesTheSleepOperationsAndTheCheckpointWindow() {
        WebApplicationException sleep = assertThrows(WebApplicationException.class,
                () -> SleepModeSwitch.OFF.requireSleep());
        assertEquals(400, sleep.getResponse().getStatus());
        String sleepMessage = errorOf(() -> SleepModeSwitch.OFF.requireSleep());
        assertTrue(sleepMessage.contains(SleepModeSwitch.FLAG), sleepMessage);
        assertTrue(sleepMessage.contains(SleepModeSwitch.PROPERTY), sleepMessage);

        WebApplicationException checkpoint = assertThrows(WebApplicationException.class,
                () -> SleepModeSwitch.OFF.requireCheckpoint());
        assertEquals(404, checkpoint.getResponse().getStatus());
        String checkpointMessage = errorOf(() -> SleepModeSwitch.OFF.requireCheckpoint());
        assertTrue(checkpointMessage.contains(SleepModeSwitch.FLAG), checkpointMessage);
        assertTrue(checkpointMessage.contains(SleepModeSwitch.PROPERTY), checkpointMessage);
        assertTrue(checkpointMessage.contains("checkpoint"), checkpointMessage);

        assertDoesNotThrow(() -> SleepModeSwitch.ON.requireSleep());
        assertDoesNotThrow(() -> SleepModeSwitch.ON.requireCheckpoint());
    }
}

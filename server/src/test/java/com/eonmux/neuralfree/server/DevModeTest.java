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
 * The development-endpoint switch (vLLM's {@code VLLM_SERVER_DEV_MODE}): off by default, the property decides when it
 * is set, vLLM's integer environment value otherwise, unusable values refuse by name, the launch flag sets the
 * property, and the off switch refuses a development route with the 404 an unregistered vLLM route answers.
 */
public class DevModeTest {

    @Test
    void theSwitchIsOffByDefault() {
        assertFalse(DevMode.from(Map.of(), Map.of()).enabled());
        assertFalse(DevMode.from(Map.of(DevMode.PROPERTY, "  "), Map.of()).enabled(),
                "a blank property defers to the environment, which is unset");
    }

    @Test
    void thePropertyDecidesBothWays() {
        assertTrue(DevMode.from(Map.of(DevMode.PROPERTY, "true"), Map.of()).enabled());
        assertTrue(DevMode.from(Map.of(DevMode.PROPERTY, "TRUE"), Map.of()).enabled());
        assertFalse(DevMode.from(Map.of(DevMode.PROPERTY, "false"), Map.of()).enabled());
        assertFalse(DevMode.from(Map.of(DevMode.PROPERTY, "false"), Map.of(DevMode.ENV, "1")).enabled(),
                "the property wins over vLLM's environment value");
    }

    @Test
    void theEnvironmentCarriesVllmsIntegerForm() {
        assertTrue(DevMode.from(Map.of(), Map.of(DevMode.ENV, "1")).enabled());
        assertTrue(DevMode.from(Map.of(), Map.of(DevMode.ENV, "2")).enabled(), "any non-zero value is on");
        assertFalse(DevMode.from(Map.of(), Map.of(DevMode.ENV, "0")).enabled());
        assertTrue(DevMode.from(Map.of(DevMode.PROPERTY, ""), Map.of(DevMode.ENV, "1")).enabled(),
                "a blank property defers to the environment");
    }

    @Test
    void unusableValuesRefuseByName() {
        IllegalArgumentException property = assertThrows(IllegalArgumentException.class,
                () -> DevMode.from(Map.of(DevMode.PROPERTY, "yes"), Map.of()));
        assertTrue(property.getMessage().startsWith(DevMode.PROPERTY), property.getMessage());
        for (String value : new String[]{"true", "1.5", ""}) {
            IllegalArgumentException env = assertThrows(IllegalArgumentException.class,
                    () -> DevMode.from(Map.of(), Map.of(DevMode.ENV, value)), value);
            assertTrue(env.getMessage().startsWith(DevMode.ENV), env.getMessage());
        }
    }

    @Test
    void theLaunchFlagSetsTheProperty() {
        assertEquals("true", ServerLaunchArgs.parse("--dev-mode").properties().get(DevMode.PROPERTY));
    }

    @Test
    void theOffSwitchRefusesWithTheUnregisteredRoutes404() {
        WebApplicationException refused = assertThrows(WebApplicationException.class, () -> DevMode.OFF.require());
        assertEquals(404, refused.getResponse().getStatus());
        String message = errorOf(() -> DevMode.OFF.require());
        assertTrue(message.contains(DevMode.FLAG), message);
        assertTrue(message.contains(DevMode.PROPERTY), message);
        assertTrue(message.contains(DevMode.ENV), message);
        assertDoesNotThrow(() -> DevMode.ON.require());
    }
}

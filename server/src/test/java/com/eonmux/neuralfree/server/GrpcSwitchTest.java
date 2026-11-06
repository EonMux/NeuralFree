package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The gRPC switch and the {@code --grpc} launch flag: off unless asked, and the flag sets its property. */
class GrpcSwitchTest {

    @Test
    void offWithoutTheProperty() {
        assertFalse(GrpcSwitch.from(new Properties()).enabled());
        assertFalse(GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, " ")).enabled());
    }

    @Test
    void trueAndFalseParseBothWays() {
        assertTrue(GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, "true")).enabled());
        assertTrue(GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, "TRUE")).enabled());
        assertFalse(GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, "false")).enabled());
        assertFalse(GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, "False")).enabled());
    }

    @Test
    void otherValuesAreRefusedByName() {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> GrpcSwitch.from(Map.of(GrpcSwitch.PROPERTY, "yes")));
        assertTrue(refused.getMessage().contains(GrpcSwitch.PROPERTY), refused.getMessage());
    }

    @Test
    void theFlagSetsTheProperty() {
        Map<String, String> properties = ServerLaunchArgs.parse("--grpc", "-m", "model.gguf").properties();
        assertEquals("true", properties.get(GrpcSwitch.PROPERTY));
    }

    @Test
    void theFlagStaysOutWithoutIt() {
        assertEquals(null, ServerLaunchArgs.parse("-m", "model.gguf").properties().get(GrpcSwitch.PROPERTY));
    }
}

package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The distributed-serving switches: vLLM's {@code --enable-elastic-ep} and {@code --enable-fault-tolerance} map to
 * their properties, both are off by default, and a value that is neither true nor false refuses by name.
 */
class ClusterSwitchesTest {

    @Test
    void theLaunchFlagsTurnTheRoutesOn() {
        Map<String, String> elastic = ServerLaunchArgs.parse("--enable-elastic-ep").properties();
        assertEquals("true", elastic.get(ClusterSwitches.ELASTIC_PROPERTY));
        Map<String, String> fault = ServerLaunchArgs.parse("--enable-fault-tolerance").properties();
        assertEquals("true", fault.get(ClusterSwitches.FAULT_PROPERTY));
        Map<String, String> both = ServerLaunchArgs.parse("--enable-elastic-ep", "--enable-fault-tolerance")
                .properties();
        assertEquals("true", both.get(ClusterSwitches.ELASTIC_PROPERTY));
        assertEquals("true", both.get(ClusterSwitches.FAULT_PROPERTY));
    }

    @Test
    void bothAreOffByDefault() {
        ClusterSwitches off = ClusterSwitches.of(Map.of());
        assertFalse(off.elasticEp());
        assertFalse(off.faultTolerance());
        ClusterSwitches on = ClusterSwitches.of(Map.of(ClusterSwitches.ELASTIC_PROPERTY, "true",
                ClusterSwitches.FAULT_PROPERTY, "false"));
        assertTrue(on.elasticEp());
        assertFalse(on.faultTolerance());
    }

    @Test
    void anUnusableValueRefusesByProperty() {
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClusterSwitches.of(Map.of(ClusterSwitches.ELASTIC_PROPERTY, "yes"))).getMessage()
                .contains(ClusterSwitches.ELASTIC_PROPERTY));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClusterSwitches.of(Map.of(ClusterSwitches.FAULT_PROPERTY, "1"))).getMessage()
                .contains(ClusterSwitches.FAULT_PROPERTY));
    }
}

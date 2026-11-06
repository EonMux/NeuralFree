package com.eonmux.neuralfree.acceleration.metal;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The Metal runtime on a host that cannot load the shim: availability, the reason users read,
 * and the refusal to enumerate. On a macOS host with a built shim the assertions are skipped.
 */
public class MetalRuntimeTest {

    @Test
    public void aHostWithoutTheShimReportsHowToBuildIt() {
        Assume.assumeFalse("the Metal shim is loaded", MetalRuntime.available());
        assertFalse(MetalNative.isAvailable());
        String reason = MetalRuntime.unavailableReason();
        assertTrue("the reason names the shim build: " + reason, reason.contains("make native/metal"));
        assertTrue("the reason names the property: " + reason, reason.contains(MetalNative.SHIM_PATH_PROPERTY));
    }

    @Test
    public void enumerationRefusesWithoutTheShim() {
        Assume.assumeFalse("the Metal shim is loaded", MetalRuntime.available());
        try {
            MetalRuntime.enumerate();
            fail("enumeration without the shim must fail");
        } catch (NoBackendAvailableException expected) {
            assertTrue(expected.getMessage().contains("Metal shim"));
        }
    }
}

package com.eonmux.neuralfree.acceleration.vulkan;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Argument checks and the no-device message of {@link VkDeviceSelection}, the one device-selection rule of
 * {@link VkDevice} and the Vulkan compute backend. No Vulkan instance is needed.
 */
public class VkDeviceSelectionTest {

    @Test
    public void adoptRejectsAMissingManagerBeforeTouchingIt() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VkDeviceSelection.adopt(null, null));
        assertTrue(e.getMessage(), e.getMessage().contains("instance manager"));
    }

    @Test
    public void anEmptyDeviceListIsReportedAsNoPhysicalDevice() {
        assertEquals("no Vulkan physical devices found",
                VkDeviceSelection.noEligibleDeviceMessage(Collections.<VkPhysicalDeviceInfo>emptyList()));
    }
}

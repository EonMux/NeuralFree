package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class VkLimitsTest {

    /** {@code VkDeviceSize} all-ones, as the NVIDIA driver reports {@code maxMemoryAllocationSize}. */
    private static final long UNSIGNED_ALL_ONES = -1L;

    @Test
    public void aReportedRequiredLimitIsKept() {
        assertEquals(4L << 30, VkLimits.required("maxMemoryAllocationSize", 4L << 30));
    }

    @Test
    public void theUnsignedAllOnesSentinelReadsAsUnbounded() {
        assertEquals(Long.MAX_VALUE, VkLimits.required("maxMemoryAllocationSize", UNSIGNED_ALL_ONES));
        assertEquals(Long.MAX_VALUE, VkLimits.optional(UNSIGNED_ALL_ONES));
    }

    @Test
    public void aRequiredLimitOfZeroIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VkLimits.required("maxStorageBufferRange", 0));
        assertTrue(e.getMessage(), e.getMessage().contains("maxStorageBufferRange"));
    }

    @Test
    public void anUnreportedOptionalLimitReadsAsUnbounded() {
        assertEquals(Long.MAX_VALUE, VkLimits.optional(0));
        assertEquals(1L << 40, VkLimits.optional(1L << 40));
    }

    @Test
    public void theUnboundedSentinelIsClampedByTheOtherLimits() {
        long storageRange = 0xFFFF_FFFFL;
        long cap = Math.min(VkLimits.required("maxMemoryAllocationSize", UNSIGNED_ALL_ONES),
                VkLimits.required("maxStorageBufferRange", storageRange));
        assertEquals(storageRange, Math.min(cap, VkLimits.optional(0)));
    }
}

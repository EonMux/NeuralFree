package com.eonmux.neuralfree.acceleration.backend.vulkan;

/**
 * Reads the byte-count limits a Vulkan device reports. Each one is an unsigned {@code VkDeviceSize}
 * (or a {@code uint32_t} widened unsigned) held in a signed Java {@code long}, so the unsigned all-ones
 * value a driver reports to say it sets no bound of its own (the NVIDIA driver does this for
 * {@code maxMemoryAllocationSize}) reads as a negative number. These methods read any negative value as
 * unbounded ({@link Long#MAX_VALUE}); the callers take the min() with the device's other limits, which
 * clamps that case.
 */
final class VkLimits {

    private VkLimits() {
    }

    /**
     * A limit the device must report.
     *
     * @return the limit, or {@link Long#MAX_VALUE} for the unbounded sentinel
     * @throws IllegalArgumentException when the device reports zero, which leaves nothing to allocate
     */
    static long required(String name, long value) {
        if (value == 0) {
            throw new IllegalArgumentException(name + " must be positive, got 0");
        }
        return value < 0 ? Long.MAX_VALUE : value;
    }

    /**
     * A limit a device may leave unreported: zero (the field absent, as {@code maxBufferSize} is before
     * Vulkan 1.3) and the unbounded sentinel both read as {@link Long#MAX_VALUE}.
     */
    static long optional(long value) {
        return value <= 0 ? Long.MAX_VALUE : value;
    }
}

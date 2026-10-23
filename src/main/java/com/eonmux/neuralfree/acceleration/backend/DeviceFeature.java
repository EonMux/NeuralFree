package com.eonmux.neuralfree.acceleration.backend;

/**
 * Optional device capabilities that kernels and the engine query through
 * {@link DeviceCapabilities#hasFeature(DeviceFeature)}.
 *
 * <p>A backend reports a feature only when the device supports it <em>and</em>
 * the backend has enabled it, so a reported feature is always usable.</p>
 */
public enum DeviceFeature {
    /** Subgroup (warp, wavefront) arithmetic reductions and broadcasts are usable in kernels. */
    SUBGROUP_ARITHMETIC,
    /** Kernels may read and write 8-bit values in storage buffers. */
    STORAGE_8BIT,
    /** Kernels may read and write 16-bit values in storage buffers. */
    STORAGE_16BIT,
    /** 8-bit integer arithmetic is available in kernels. */
    SHADER_INT8,
    /** 64-bit integer arithmetic is available in kernels. */
    SHADER_INT64,
    /** 16-bit float arithmetic is available in kernels. */
    SHADER_FLOAT16,
    /** Descriptors can be pushed directly into the command stream (Vulkan {@code VK_KHR_push_descriptor}). */
    PUSH_DESCRIPTORS,
    /** The driver reports a per-heap memory budget (Vulkan {@code VK_EXT_memory_budget}). */
    MEMORY_BUDGET,
    /** Timeline semaphores are available for queue synchronisation. */
    TIMELINE_SEMAPHORE,
    /** Dispatches can require full subgroups (Vulkan {@code computeFullSubgroups}). */
    FULL_SUBGROUPS,
    /** Packed integer dot-product instructions are available. */
    INTEGER_DOT_PRODUCT,
    /** Work spread across several compute queues runs concurrently and is worth the synchronisation cost. */
    CONCURRENT_QUEUES_PROFITABLE
}

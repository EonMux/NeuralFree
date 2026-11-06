package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

/**
 * A buffer a Metal kernel binds: device memory of a {@link MetalComputeBackend} or the shared
 * memory of a {@link MetalHostMappedBuffer}. Kernel argument binding and the queue copy operations
 * accept either.
 */
interface MetalBuffer extends GpuBuffer {

    /** The {@code MTLBuffer} handle, or 0 once the buffer is closed. */
    long nativeHandle();

    /** The backend the buffer belongs to. */
    MetalComputeBackend backend();

    /** True once the buffer is closed. */
    boolean isClosed();
}

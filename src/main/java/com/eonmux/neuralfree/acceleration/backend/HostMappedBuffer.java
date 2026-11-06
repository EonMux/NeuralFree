package com.eonmux.neuralfree.acceleration.backend;

import java.nio.ByteBuffer;

/**
 * Page-locked host memory a device reads directly, without an upload: CUDA mapped host memory
 * ({@code cuMemHostAlloc} with {@code CU_MEMHOSTALLOC_DEVICEMAP} and {@code cuMemHostGetDevicePointer})
 * or Vulkan host-visible memory. A kernel binds it like any {@link GpuBuffer}; the interconnect
 * carries every read.
 *
 * <p>The buffer is filled once through {@link #hostView(long, int)} at load, flushed for devices
 * whose host-visible memory is not coherent, and then read-only. Closing it returns the host memory
 * to the driver.</p>
 */
public interface HostMappedBuffer extends GpuBuffer {

    /**
     * A host view of {@code length} bytes starting at {@code offset} of the buffer, in little-endian
     * byte order. The view is a copy of the mapping's position and limit, so callers may seek it
     * freely.
     *
     * @throws IllegalArgumentException when the range is outside the buffer or longer than an
     *                                  {@code int} byte count
     * @throws IllegalStateException    when the buffer is not host-visible or is closed
     */
    ByteBuffer hostView(long offset, int length);

    /**
     * Makes host writes to the whole buffer visible to the device; a no-op when the memory is
     * coherent. Called once after the buffer is filled.
     */
    default void flush() {
    }
}

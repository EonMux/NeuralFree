package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A shared-storage Metal buffer exposed to the host as a direct byte buffer: a weight the kernel
 * reads in place, with no upload.
 *
 * <p>Shared storage is coherent on Apple GPUs and CPU-accessible on every Mac, so {@link #flush()}
 * is a no-op. The host view is a direct buffer over the framework's mapping; it stays valid until
 * the buffer is closed. The buffer is filled once, before the first kernel reads it, and read-only
 * afterwards.</p>
 */
public final class MetalHostMappedBuffer extends MetalGpuBuffer implements HostMappedBuffer {

    /**
     * @param owner           backend that allocated the memory
     * @param handle          the shared {@code MTLBuffer} handle
     * @param size            allocation size in bytes
     * @param allocationBytes aligned allocation size
     */
    public MetalHostMappedBuffer(MetalComputeBackend owner, long handle, long size, long allocationBytes) {
        super(owner, handle, size, allocationBytes);
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException when the range is outside the buffer or longer than an
     *                                  {@code int} byte count
     * @throws IllegalStateException    when the buffer is closed
     */
    @Override
    public ByteBuffer hostView(long offset, int length) {
        if (length < 0) {
            throw new IllegalArgumentException("host view length must not be negative, got " + length);
        }
        long size = sizeInBytes();
        if (offset < 0 || length > size - offset) {
            throw new IllegalArgumentException("host view [" + offset + ", " + (offset + length)
                    + ") is outside the buffer of " + size + " bytes");
        }
        if (length > Integer.MAX_VALUE || offset > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("host view offset/length exceeds an int byte count: offset="
                    + offset + ", length=" + length);
        }
        long handle = nativeHandle();
        ByteBuffer full = MetalNative.bufferHostView(handle, size);
        if (full == null) {
            throw new IllegalStateException("Metal buffer mapping is not host-accessible");
        }
        ByteBuffer view = full.duplicate();
        view.position((int) offset);
        view.limit((int) offset + length);
        return view.order(ByteOrder.LITTLE_ENDIAN);
    }
}

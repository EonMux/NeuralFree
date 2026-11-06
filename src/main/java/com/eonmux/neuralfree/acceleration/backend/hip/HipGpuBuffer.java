package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.atomic.AtomicLong;

/**
 * One device allocation of a {@link HipComputeBackend}.
 *
 * <p>{@link #sizeInBytes()} is the size the caller requested; the device block may be larger
 * (rounded up to the backend's allocation granularity, or a reused pooled block within
 * {@link HipBufferPool}'s reuse bounds) and is what the backend's allocation counter tracks.
 * {@link #close()} returns the block to the backend's {@link HipBufferPool} without waiting for
 * the device. Closing the backend frees the memory of every buffer still open. Ids are unique
 * for the process and never reused.</p>
 */
public final class HipGpuBuffer implements GpuBuffer {

    private static final Logger LOGGER = LoggingUtil.getLogger(HipGpuBuffer.class);
    private static final AtomicLong NEXT_ID = new AtomicLong(1);

    private final HipComputeBackend owner;
    private final long sizeBytes;
    private final long allocationBytes;
    private final long id;
    private volatile Long devicePtr;

    /**
     * @param owner           backend that allocated the memory
     * @param ptr             the device allocation
     * @param size            size the caller requested, in bytes
     * @param allocationBytes bytes actually allocated, at least {@code size}
     * @throws IllegalArgumentException when an argument is null, the size is not positive, or
     *                                  the allocation is smaller than the size
     */
    public HipGpuBuffer(HipComputeBackend owner, long ptr, long size, long allocationBytes) {
        if (owner == null) {
            throw new IllegalArgumentException("owning HIP backend must not be null");
        }
        if (ptr == 0) {
            throw new IllegalArgumentException("device pointer must not be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("buffer size must be positive, got " + size);
        }
        if (allocationBytes < size) {
            throw new IllegalArgumentException("allocation of " + allocationBytes + " bytes is smaller than the buffer size "
                    + size);
        }
        this.owner = owner;
        this.devicePtr = ptr;
        this.sizeBytes = size;
        this.allocationBytes = allocationBytes;
        this.id = NEXT_ID.getAndIncrement();
    }

    /** The size the caller requested, in bytes. */
    @Override
    public long sizeInBytes() {
        return sizeBytes;
    }

    /** Bytes of device memory backing this buffer: the block size, at least the requested size. */
    public long allocationBytes() {
        return allocationBytes;
    }

    /** Process-unique id of this allocation, or -1 once closed. */
    @Override
    public long bufferId() {
        return devicePtr == null ? -1 : id;
    }

    /** The backend that allocated this buffer. */
    public HipComputeBackend backend() {
        return owner;
    }

    /** The device pointer, or null once the buffer is closed. */
    public Long getDevicePointer() {
        return devicePtr;
    }

    /** True once {@link #close()} has run. */
    public boolean isClosed() {
        return devicePtr == null;
    }

    /**
     * Returns the device memory to the backend's buffer pool. Never waits for the device;
     * commands enqueued before the close may still use the memory. Idempotent. After the backend
     * has closed it only marks the buffer closed: the backend already freed the memory.
     *
     * @throws RuntimeException the translated driver failure of fencing the block, which was
     *                          then freed; the buffer is closed either way
     */
    @Override
    public void close() {
        Long ptr;
        synchronized (this) {
            ptr = devicePtr;
            if (ptr == null) {
                return;
            }
            devicePtr = null;
        }
        if (owner.isClosed()) {
            LOGGER.debug("HIP buffer {} ({} bytes) closed after its backend, which already freed its memory", id,
                    sizeBytes);
            return;
        }
        owner.releaseAllocation(ptr, allocationBytes);
    }

    @Override
    public String toString() {
        return "HipGpuBuffer{id=" + id + ", " + sizeBytes + " bytes" + (devicePtr == null ? ", closed" : "") + "}";
    }
}

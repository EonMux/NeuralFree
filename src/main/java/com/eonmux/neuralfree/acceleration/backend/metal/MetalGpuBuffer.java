package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.atomic.AtomicLong;

/**
 * One allocation of a {@link MetalComputeBackend}: shared-storage {@code MTLBuffer} memory the CPU
 * and the GPU both reach.
 *
 * <p>{@link #sizeInBytes()} is the size the caller requested; the allocation is rounded up to
 * {@link MetalComputeBackend#BUFFER_ALIGNMENT} and that is what the backend's allocation counter
 * tracks. {@link #close()} releases the framework object; a command that already references the
 * buffer holds its own retain until it completes, so closing never races with queued work. Closing
 * the backend releases every buffer still open. Ids are unique for the process and never reused.</p>
 */
public class MetalGpuBuffer implements MetalBuffer {

    private static final Logger LOGGER = LoggingUtil.getLogger(MetalGpuBuffer.class);
    private static final AtomicLong NEXT_ID = new AtomicLong(1);

    private final MetalComputeBackend owner;
    private final long sizeBytes;
    private final long allocationBytes;
    private final long id;
    private volatile Long handle;

    /**
     * @param owner           backend that allocated the memory
     * @param handle          the {@code MTLBuffer} handle
     * @param size            size the caller requested, in bytes
     * @param allocationBytes bytes actually allocated, at least {@code size}
     * @throws IllegalArgumentException when the owner is null, the size is not positive, or the
     *                                  allocation is smaller than the size
     */
    public MetalGpuBuffer(MetalComputeBackend owner, long handle, long size, long allocationBytes) {
        if (owner == null) {
            throw new IllegalArgumentException("owning Metal backend must not be null");
        }
        if (handle == 0) {
            throw new IllegalArgumentException("Metal buffer handle must not be null");
        }
        if (size <= 0) {
            throw new IllegalArgumentException("buffer size must be positive, got " + size);
        }
        if (allocationBytes < size) {
            throw new IllegalArgumentException("allocation of " + allocationBytes + " bytes is smaller than the"
                    + " buffer size " + size);
        }
        this.owner = owner;
        this.handle = handle;
        this.sizeBytes = size;
        this.allocationBytes = allocationBytes;
        this.id = NEXT_ID.getAndIncrement();
    }

    /** The size the caller requested, in bytes. */
    @Override
    public long sizeInBytes() {
        return sizeBytes;
    }

    /** Bytes the framework allocated: the requested size rounded up to the backend's alignment. */
    public long allocationBytes() {
        return allocationBytes;
    }

    /** Process-unique id of this allocation, or -1 once closed. */
    @Override
    public long bufferId() {
        return handle == null ? -1 : id;
    }

    @Override
    public MetalComputeBackend backend() {
        return owner;
    }

    @Override
    public long nativeHandle() {
        Long live = handle;
        if (live == null) {
            throw new IllegalStateException("Metal buffer " + id + " is closed");
        }
        return live;
    }

    @Override
    public boolean isClosed() {
        return handle == null;
    }

    /**
     * Releases the framework object. Idempotent. After the backend has closed it only marks the
     * buffer closed: the backend already released the memory.
     */
    @Override
    public void close() {
        Long live;
        synchronized (this) {
            live = handle;
            if (live == null) {
                return;
            }
            handle = null;
        }
        if (owner.isClosed()) {
            LOGGER.debug("Metal buffer {} ({} bytes) closed after its backend, which already released it", id,
                    sizeBytes);
            return;
        }
        owner.releaseAllocation(live, allocationBytes);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{id=" + id + ", " + sizeBytes + " bytes"
                + (handle == null ? ", closed" : "") + "}";
    }
}

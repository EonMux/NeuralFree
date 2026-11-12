package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import org.jocl.cl_mem;

import java.util.concurrent.atomic.AtomicLong;

import static org.jocl.CL.*;

public class OclGpuBuffer implements GpuBuffer {
    private cl_mem mem;
    private final long sizeBytes;
    private final AtomicLong allocatedBytesTracker;
    private final java.util.function.Consumer<OclGpuBuffer> onClosed;
    private boolean closed = false;

    public OclGpuBuffer(cl_mem mem, long sizeBytes, AtomicLong allocatedBytesTracker) {
        this(mem, sizeBytes, allocatedBytesTracker, null);
    }

    public OclGpuBuffer(cl_mem mem, long sizeBytes) {
        this(mem, sizeBytes, null, null);
    }

    /**
     * @param onClosed run once after the {@code cl_mem} is released, so the allocating backend can stop
     *                 tracking the buffer; may be null
     */
    OclGpuBuffer(cl_mem mem, long sizeBytes, AtomicLong allocatedBytesTracker,
                 java.util.function.Consumer<OclGpuBuffer> onClosed) {
        if (mem == null) {
            throw new IllegalArgumentException("cl_mem cannot be null");
        }
        if (sizeBytes <= 0) {
            throw new IllegalArgumentException("sizeBytes must be positive");
        }
        this.mem = mem;
        this.sizeBytes = sizeBytes;
        this.allocatedBytesTracker = allocatedBytesTracker;
        this.onClosed = onClosed;
    }

    @Override
    public long sizeInBytes() {
        return sizeBytes;
    }

    @Override
    public synchronized long bufferId() {
        if (closed || mem == null) {
            return -1;
        }
        return System.identityHashCode(mem);
    }

    /** Releases the {@code cl_mem}; idempotent and safe against a concurrent close by the allocating backend. */
    @Override
    public synchronized void close() {
        if (!closed) {
            clReleaseMemObject(mem);
            mem = null;
            closed = true;
            if (allocatedBytesTracker != null) {
                allocatedBytesTracker.addAndGet(-sizeBytes);
            }
            if (onClosed != null) {
                onClosed.accept(this);
            }
        }
    }

    public synchronized cl_mem getCLMem() {
        return mem;
    }
}

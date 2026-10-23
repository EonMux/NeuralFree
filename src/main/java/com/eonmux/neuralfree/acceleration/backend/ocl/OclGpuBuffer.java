package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import org.jocl.cl_mem;

import java.util.concurrent.atomic.AtomicLong;

import static org.jocl.CL.*;

public final class OclGpuBuffer implements GpuBuffer {
    private cl_mem mem;
    private final long sizeBytes;
    private final AtomicLong allocatedBytesTracker;
    private boolean closed = false;

    public OclGpuBuffer(cl_mem mem, long sizeBytes, AtomicLong allocatedBytesTracker) {
        if (mem == null) {
            throw new IllegalArgumentException("cl_mem cannot be null");
        }
        if (sizeBytes <= 0) {
            throw new IllegalArgumentException("sizeBytes must be positive");
        }
        this.mem = mem;
        this.sizeBytes = sizeBytes;
        this.allocatedBytesTracker = allocatedBytesTracker;
    }

    public OclGpuBuffer(cl_mem mem, long sizeBytes) {
        this(mem, sizeBytes, null);
    }

    @Override
    public long sizeInBytes() {
        return sizeBytes;
    }

    @Override
    public long bufferId() {
        if (closed || mem == null) {
            return -1;
        }
        return System.identityHashCode(mem);
    }

    @Override
    public void close() {
        if (!closed) {
            clReleaseMemObject(mem);
            mem = null;
            closed = true;
            if (allocatedBytesTracker != null) {
                allocatedBytesTracker.addAndGet(-sizeBytes);
            }
        }
    }

    public cl_mem getCLMem() {
        return mem;
    }
}

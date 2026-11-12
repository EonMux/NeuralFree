package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import org.jocl.CLException;
import org.jocl.cl_command_queue;
import org.jocl.cl_mem;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.jocl.CL.CL_MAP_READ;
import static org.jocl.CL.CL_MAP_WRITE;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.clEnqueueMapBuffer;
import static org.jocl.CL.clEnqueueUnmapMemObject;
import static org.jocl.CL.clFinish;

/**
 * An OpenCL buffer in host memory ({@code CL_MEM_ALLOC_HOST_PTR}) on a device that shares the host's memory
 * ({@code CL_DEVICE_HOST_UNIFIED_MEMORY}): kernels read it in place, and the host writes it through a mapping
 * without a staging copy. The first {@link #hostView} maps the whole buffer ({@code clEnqueueMapBuffer}, blocking);
 * {@link #flush()} unmaps it and waits for the unmap, which makes the host writes visible to the device and ends
 * every view handed out — OpenCL leaves a kernel's access to a mapped buffer undefined, so the buffer is filled
 * first and flushed before any kernel reads it. A later {@link #hostView} maps it again. The buffer counts against
 * no device allocation. Thread-safe.
 */
final class OclHostMappedBuffer extends OclGpuBuffer implements HostMappedBuffer {

    private final long sizeBytes;
    private final Supplier<cl_command_queue> mapQueue;
    private ByteBuffer mapping;

    /**
     * @param mapQueue the queue mapping and unmapping the buffer; null once that queue is released
     * @param onClosed run once after the {@code cl_mem} is released
     */
    OclHostMappedBuffer(cl_mem mem, long sizeBytes, Supplier<cl_command_queue> mapQueue,
                        Consumer<OclGpuBuffer> onClosed) {
        super(mem, sizeBytes, null, onClosed);
        this.sizeBytes = sizeBytes;
        this.mapQueue = Objects.requireNonNull(mapQueue, "mapQueue");
    }

    /**
     * {@inheritDoc}
     *
     * <p>The view stays valid until {@link #flush()} or {@link #close()}.</p>
     */
    @Override
    public synchronized ByteBuffer hostView(long offset, int length) {
        if (length < 0) {
            throw new IllegalArgumentException("host view length must not be negative, got " + length);
        }
        if (offset < 0 || length > sizeBytes - offset) {
            throw new IllegalArgumentException("host view [" + offset + ", " + (offset + length)
                    + ") is outside the buffer of " + sizeBytes + " bytes");
        }
        cl_mem mem = getCLMem();
        if (mem == null) {
            throw new IllegalStateException("host-mapped OpenCL buffer is closed");
        }
        if (mapping == null) {
            mapping = map(mem);
        }
        ByteBuffer view = mapping.duplicate();
        view.position((int) offset).limit((int) offset + length);
        return view.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Unmaps the buffer and waits for the unmap; a no-op when it is not mapped. */
    @Override
    public synchronized void flush() {
        if (mapping == null) {
            return;
        }
        cl_command_queue queue = queue();
        ByteBuffer mapped = mapping;
        mapping = null;
        OclErrors.check("clEnqueueUnmapMemObject", sizeBytes + " bytes",
                () -> clEnqueueUnmapMemObject(queue, getCLMem(), mapped, 0, null, null));
        OclErrors.check("clFinish", "unmap of " + sizeBytes + " bytes", () -> clFinish(queue));
    }

    /** Unmaps the buffer when its queue is still open, then releases it. Idempotent. */
    @Override
    public synchronized void close() {
        RuntimeException failure = null;
        if (mapping != null && getCLMem() != null && mapQueue.get() != null) {
            try {
                flush();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        mapping = null;
        super.close();
        if (failure != null) {
            throw failure;
        }
    }

    private ByteBuffer map(cl_mem mem) {
        if (sizeBytes > Integer.MAX_VALUE) {
            throw new IllegalStateException("host-mapped OpenCL buffer of " + sizeBytes
                    + " bytes exceeds one host mapping");
        }
        int[] error = new int[1];
        ByteBuffer mapped;
        try {
            mapped = clEnqueueMapBuffer(queue(), mem, true, CL_MAP_READ | CL_MAP_WRITE, 0, sizeBytes, 0, null, null,
                    error);
        } catch (CLException e) {
            throw OclErrors.translate(e.getStatus(), "clEnqueueMapBuffer", sizeBytes + " bytes", -1, e);
        }
        if (error[0] != CL_SUCCESS) {
            throw OclErrors.translate(error[0], "clEnqueueMapBuffer", sizeBytes + " bytes", -1, null);
        }
        if (mapped == null) {
            throw new IllegalStateException("clEnqueueMapBuffer (" + sizeBytes
                    + " bytes) reported success but returned no mapping");
        }
        return mapped.order(ByteOrder.LITTLE_ENDIAN);
    }

    private cl_command_queue queue() {
        cl_command_queue queue = mapQueue.get();
        if (queue == null) {
            throw new IllegalStateException("the OpenCL backend of this host-mapped buffer is closed");
        }
        return queue;
    }
}

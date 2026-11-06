package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.backend.HostMappedBuffer;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP;
import static jcuda.driver.JCudaDriver.CU_MEMHOSTALLOC_PORTABLE;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.driver.JCudaDriver.cuMemHostGetDevicePointer;

/**
 * One page-locked host allocation of a {@link CudaComputeBackend}, mapped into the device address
 * space ({@code cuMemHostAlloc} with {@code CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP},
 * then {@code cuMemHostGetDevicePointer}): a kernel binds the device pointer and reads host memory
 * over the interconnect, with no copy and no device allocation.
 *
 * <p>Writes go through {@link #hostView(long, int)} before the first kernel reads the buffer; the
 * memory is written once and read-only afterwards. Freeing is idempotent, and the backend frees any
 * buffer still open when it closes.</p>
 */
public final class CudaHostMappedBuffer implements CudaBuffer, HostMappedBuffer {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaHostMappedBuffer.class);
    private static final AtomicLong NEXT_ID = new AtomicLong(1);

    private final CudaComputeBackend owner;
    private final long sizeBytes;
    private final long id;
    private final Pointer hostPointer;
    private volatile CUdeviceptr devicePointer;

    /**
     * Allocates and maps {@code sizeInBytes} of page-locked host memory. The backend's context must
     * be current on the calling thread.
     *
     * @param owner       backend the buffer belongs to
     * @param sizeInBytes bytes to allocate; positive
     * @throws IllegalArgumentException when an argument is null or the size is not positive
     * @throws RuntimeException         the translated driver failure
     */
    public CudaHostMappedBuffer(CudaComputeBackend owner, long sizeInBytes) {
        if (owner == null) {
            throw new IllegalArgumentException("owning CUDA backend must not be null");
        }
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("host-mapped buffer size must be positive, got " + sizeInBytes);
        }
        this.owner = owner;
        this.sizeBytes = sizeInBytes;
        this.id = NEXT_ID.getAndIncrement();
        owner.context().makeCurrent();
        Pointer host = new Pointer();
        boolean allocated = false;
        try {
            CudaErrors.check("cuMemHostAlloc(" + sizeInBytes + " bytes)",
                    () -> cuMemHostAlloc(host, sizeInBytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
            allocated = true;
            if (host.getByteBuffer(0, Math.min(sizeInBytes, Integer.MAX_VALUE)) == null) {
                throw new IllegalStateException("JCuda returned page-locked memory without a ByteBuffer view");
            }
            CUdeviceptr device = new CUdeviceptr();
            CudaErrors.call(cuMemHostGetDevicePointer(device, host, 0), "cuMemHostGetDevicePointer");
            this.hostPointer = host;
            this.devicePointer = device;
        } catch (RuntimeException e) {
            if (allocated) {
                try {
                    CudaErrors.check("cuMemFreeHost", () -> cuMemFreeHost(host));
                } catch (RuntimeException freeFailure) {
                    e.addSuppressed(freeFailure);
                }
            }
            throw e;
        }
    }

    @Override
    public long sizeInBytes() {
        return sizeBytes;
    }

    /** Process-unique id of this allocation, also the buffer id kernel binding reports. */
    @Override
    public long bufferId() {
        return devicePointer == null ? -1 : id;
    }

    @Override
    public CudaComputeBackend backend() {
        return owner;
    }

    @Override
    public CUdeviceptr getCUDevicePtr() {
        return devicePointer;
    }

    @Override
    public boolean isClosed() {
        return devicePointer == null;
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalStateException when the buffer is closed
     */
    @Override
    public ByteBuffer hostView(long offset, int length) {
        if (length < 0) {
            throw new IllegalArgumentException("host view length must not be negative, got " + length);
        }
        if (offset < 0 || length > sizeBytes - offset) {
            throw new IllegalArgumentException("host view [" + offset + ", " + (offset + length)
                    + ") is outside the buffer of " + sizeBytes + " bytes");
        }
        if (devicePointer == null) {
            throw new IllegalStateException("host-mapped buffer " + id + " is closed");
        }
        ByteBuffer view = hostPointer.getByteBuffer(offset, length);
        return view.order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Frees the host memory ({@code cuMemFreeHost}). Idempotent; the caller ensures no device work
     * still reads the buffer. After the backend closed, it only marks the buffer closed: the backend
     * already freed the memory.
     */
    @Override
    public void close() {
        CUdeviceptr device;
        synchronized (this) {
            device = devicePointer;
            if (device == null) {
                return;
            }
            devicePointer = null;
        }
        if (owner.isClosed()) {
            LOGGER.debug("host-mapped CUDA buffer {} ({} bytes) closed after its backend, which already freed it",
                    id, sizeBytes);
            return;
        }
        owner.releaseHostAllocation(this);
    }

    /** Frees the host allocation; called by the backend and by {@link #close()}. */
    void free() {
        CUdeviceptr device;
        synchronized (this) {
            device = devicePointer;
            devicePointer = null;
        }
        if (device == null) {
            return;
        }
        owner.context().makeCurrent();
        CudaErrors.check("cuMemFreeHost", () -> cuMemFreeHost(hostPointer));
    }

    @Override
    public String toString() {
        return "CudaHostMappedBuffer{id=" + id + ", " + sizeBytes + " bytes"
                + (devicePointer == null ? ", closed" : ", mapped") + "}";
    }
}

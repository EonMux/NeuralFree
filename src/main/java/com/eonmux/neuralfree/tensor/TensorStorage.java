package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A byte range {@code [offsetBytes, offsetBytes + lengthBytes)} of a reference-counted device buffer of one backend.
 *
 * <p><b>Allocation.</b> Every buffer this class allocates is rounded up to a multiple of 4 bytes (at least 4) while
 * the storage length stays exact, so kernels that access sub-word dtypes as 32-bit words stay inside the buffer.
 * A storage of 0 elements owns a 4-byte buffer and has length 0.</p>
 *
 * <p><b>Transfers.</b> Every host transfer runs under the backend's {@link BackendResources#lock} and is complete
 * when the method returns: it is ordered after pending train work ({@link TrainGpu#beforeHostRead}, which also
 * surfaces deferred index errors), the queue is finished, and a write gives the compute queue a dependency on the
 * upload ({@link TrainGpu#afterHostWrite}). The {@code queue} parameters keep their meaning: the transfer is issued on
 * the queue passed. Byte order is {@link ByteOrder#nativeOrder()}. Returned events belong to the caller, are already
 * complete, and should be closed.</p>
 */
public final class TensorStorage implements AutoCloseable {
    private final RefCountedBuffer refCountedBuffer;
    private final ComputeBackend backend;
    private final long offsetBytes;
    private final long lengthBytes;
    private final DType dtype;
    private final boolean isPrimaryOwner;
    private volatile boolean closed = false;

    private TensorStorage(RefCountedBuffer refCountedBuffer, ComputeBackend backend, long offsetBytes, long lengthBytes,
                          DType dtype, boolean isPrimaryOwner) {
        this.refCountedBuffer = Objects.requireNonNull(refCountedBuffer, "RefCountedBuffer cannot be null");
        this.backend = Objects.requireNonNull(backend, "Backend cannot be null");
        this.dtype = Objects.requireNonNull(dtype, "DType cannot be null");
        this.isPrimaryOwner = isPrimaryOwner;

        if (offsetBytes < 0) {
            throw new IllegalArgumentException("Offset must be non-negative: " + offsetBytes);
        }
        if (lengthBytes < 0) {
            throw new IllegalArgumentException("Length must be non-negative: " + lengthBytes);
        }
        if (offsetBytes + lengthBytes > refCountedBuffer.gpuBuffer.sizeInBytes()) {
            throw new IllegalArgumentException("Offset (" + offsetBytes + ") + length (" + lengthBytes +
                ") exceeds buffer size (" + refCountedBuffer.gpuBuffer.sizeInBytes() + ")");
        }

        this.offsetBytes = offsetBytes;
        this.lengthBytes = lengthBytes;
    }

    /** Uninitialized storage of {@code numElements} elements; the buffer is rounded up to a multiple of 4 bytes. */
    public static TensorStorage allocate(ComputeBackend backend, long numElements, DType dtype) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");

        if (numElements < 0) {
            throw new IllegalArgumentException("Number of elements must be non-negative: " + numElements);
        }
        if (numElements > Long.MAX_VALUE / dtype.byteSize() - Integer.BYTES) {
            throw new IllegalArgumentException("Storage of " + numElements + " " + dtype + " elements is too large");
        }
        return allocateBytes(backend, numElements * dtype.byteSize(), dtype);
    }

    /** Storage holding bytes {@code [position, limit)} of {@code hostData}. */
    public static TensorStorage fromHost(ComputeBackend backend, ByteBuffer hostData, DType dtype) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(hostData, "Host data cannot be null");
        Objects.requireNonNull(dtype, "DType cannot be null");

        byte[] bytes = new byte[hostData.remaining()];
        hostData.duplicate().get(bytes);
        TensorStorage storage = allocateBytes(backend, bytes.length, dtype);
        try {
            closeEvent(storage.writeBytes(0, bytes, backend.transferQueue()));
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
        return storage;
    }

    /** FLOAT32 storage holding {@code hostData}. */
    public static TensorStorage fromHostFloat(ComputeBackend backend, float[] hostData) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(hostData, "Host data cannot be null");

        TensorStorage storage = allocateBytes(backend, (long) hostData.length * DType.FLOAT32.byteSize(), DType.FLOAT32);
        try {
            closeEvent(storage.writeFloats(0, hostData, backend.transferQueue()));
        } catch (RuntimeException e) {
            storage.close();
            throw e;
        }
        return storage;
    }

    /**
     * Reads {@code min(lengthBytes, dst.remaining())} bytes from the start of the storage into {@code dst}, advancing
     * its position; null when nothing is transferred.
     */
    public GpuEvent copyToHost(ByteBuffer dst, ComputeQueue queue) {
        ensureOpen();
        Objects.requireNonNull(dst, "Destination buffer cannot be null");
        Objects.requireNonNull(queue, "Queue cannot be null");

        int copyBytes = (int) Math.min(lengthBytes, dst.remaining());
        if (copyBytes == 0) {
            return null;
        }
        byte[] tempArray = new byte[copyBytes];
        GpuEvent event = read(queue, (q, waits) ->
                q.enqueueReadBuffer(refCountedBuffer.gpuBuffer, offsetBytes, copyBytes, tempArray, waits));
        dst.put(tempArray, 0, copyBytes);
        return event;
    }

    /** Every element of a FLOAT32-aligned storage. */
    public float[] copyToHostFloat(ComputeQueue queue) {
        ensureOpen();
        if (lengthBytes % DType.FLOAT32.byteSize() != 0) {
            throw new IllegalStateException("Storage size is not aligned to FLOAT32");
        }
        return copyToHostFloat(0, lengthBytes / DType.FLOAT32.byteSize(), queue);
    }

    /**
     * {@code count} 32-bit floats starting {@code elementOffset * 4} bytes past the storage start.
     *
     * @throws IllegalArgumentException when the range is outside the storage or above {@code Integer.MAX_VALUE}
     */
    public float[] copyToHostFloat(long elementOffset, long count, ComputeQueue queue) {
        ensureOpen();
        Objects.requireNonNull(queue, "Queue cannot be null");
        long byteOffset = checkRange(elementOffset, count, DType.FLOAT32.byteSize());
        float[] result = new float[(int) count];
        if (count > 0) {
            closeEvent(read(queue, (q, waits) -> q.enqueueReadBuffer(
                    refCountedBuffer.gpuBuffer, offsetBytes + byteOffset, count * Float.BYTES, result, waits)));
        }
        return result;
    }

    /**
     * {@code byteCount} bytes starting {@code byteOffset} bytes past the storage start.
     *
     * @throws IllegalArgumentException when the range is outside the storage or above {@code Integer.MAX_VALUE}
     */
    public byte[] copyToHostBytes(long byteOffset, long byteCount, ComputeQueue queue) {
        ensureOpen();
        Objects.requireNonNull(queue, "Queue cannot be null");
        checkRange(byteOffset, byteCount, 1);
        byte[] result = new byte[(int) byteCount];
        if (byteCount > 0) {
            closeEvent(read(queue, (q, waits) -> q.enqueueReadBuffer(
                    refCountedBuffer.gpuBuffer, offsetBytes + byteOffset, byteCount, result, waits)));
        }
        return result;
    }

    /** Writes {@code min(lengthBytes, src.remaining())} bytes of {@code src} to the storage start, advancing it. */
    public GpuEvent copyFromHost(ByteBuffer src, ComputeQueue queue) {
        ensureOpen();
        Objects.requireNonNull(src, "Source buffer cannot be null");
        Objects.requireNonNull(queue, "Queue cannot be null");

        int copyBytes = (int) Math.min(lengthBytes, src.remaining());
        byte[] copyArray = new byte[copyBytes];
        src.get(copyArray, 0, copyBytes);
        return writeBytes(0, copyArray, queue);
    }

    /** Writes {@code src} to the storage start. */
    public GpuEvent copyFromHostFloat(float[] src, ComputeQueue queue) {
        return copyFromHostFloat(src, 0, queue);
    }

    /**
     * Writes {@code src} starting {@code elementOffset * 4} bytes past the storage start.
     *
     * @throws IllegalArgumentException when the range is outside the storage
     */
    public GpuEvent copyFromHostFloat(float[] src, long elementOffset, ComputeQueue queue) {
        ensureOpen();
        Objects.requireNonNull(src, "Source array cannot be null");
        Objects.requireNonNull(queue, "Queue cannot be null");
        if (elementOffset < 0 || (elementOffset + src.length) * DType.FLOAT32.byteSize() > lengthBytes) {
            throw new IllegalArgumentException("Source array of " + src.length + " floats at element " + elementOffset
                + " exceeds storage capacity (" + lengthBytes + " bytes)");
        }
        return writeFloats(elementOffset * DType.FLOAT32.byteSize(), src, queue);
    }

    public TensorStorage slice(long offsetElements, long lengthElements) {
        ensureOpen();

        if (offsetElements < 0) {
            throw new IllegalArgumentException("Offset must be non-negative: " + offsetElements);
        }
        if (lengthElements < 0) {
            throw new IllegalArgumentException("Length must be non-negative: " + lengthElements);
        }

        long newOffsetBytes = this.offsetBytes + offsetElements * dtype.byteSize();
        long newLengthBytes = lengthElements * dtype.byteSize();

        if (newOffsetBytes + newLengthBytes > this.offsetBytes + this.lengthBytes) {
            throw new IllegalArgumentException("Slice bounds exceed storage: offset=" +
                offsetElements + ", length=" + lengthElements + ", available=" +
                (this.lengthBytes / dtype.byteSize()) + " elements");
        }

        refCountedBuffer.retain();
        return new TensorStorage(refCountedBuffer, backend, newOffsetBytes, newLengthBytes, dtype, false);
    }

    public TensorStorage share() {
        ensureOpen();
        refCountedBuffer.retain();
        return new TensorStorage(refCountedBuffer, backend, offsetBytes, lengthBytes, dtype, false);
    }

    public GpuBuffer gpuBuffer() {
        ensureOpen();
        return refCountedBuffer.gpuBuffer;
    }

    /** The backend that owns the buffer. */
    public ComputeBackend backend() {
        return backend;
    }

    public long offsetBytes() {
        return offsetBytes;
    }

    public long lengthBytes() {
        return lengthBytes;
    }

    public long numElements() {
        return lengthBytes / dtype.byteSize();
    }

    public DType dtype() {
        return dtype;
    }

    public boolean ownsBuffer() {
        return isPrimaryOwner;
    }

    public boolean isView() {
        return !isPrimaryOwner;
    }

    public int refCount() {
        return refCountedBuffer.refCount.get();
    }

    private static TensorStorage allocateBytes(ComputeBackend backend, long lengthBytes, DType dtype) {
        long bufferBytes = Math.max(Integer.BYTES, (lengthBytes + Integer.BYTES - 1) / Integer.BYTES * Integer.BYTES);
        GpuBuffer buffer = backend.allocateBuffer(bufferBytes);
        RefCountedBuffer refCountedBuffer = new RefCountedBuffer(buffer);
        try {
            return new TensorStorage(refCountedBuffer, backend, 0, lengthBytes, dtype, true);
        } catch (RuntimeException e) {
            buffer.close();
            throw e;
        }
    }

    /** Byte offset of an element range inside the storage, after checking it; {@code count} fits an array. */
    private long checkRange(long elementOffset, long count, int elementBytes) {
        if (elementOffset < 0 || count < 0 || count > Integer.MAX_VALUE
                || elementOffset > (lengthBytes / elementBytes) - count) {
            throw new IllegalArgumentException("Range of " + count + " elements at element " + elementOffset
                + " is outside the storage of " + (lengthBytes / elementBytes) + " elements");
        }
        return elementOffset * elementBytes;
    }

    private GpuEvent writeBytes(long byteOffset, byte[] data, ComputeQueue queue) {
        if (data.length == 0) {
            return null;
        }
        return write(queue, (q, waits) ->
                q.enqueueWriteBuffer(refCountedBuffer.gpuBuffer, offsetBytes + byteOffset, data.length, data, waits));
    }

    private GpuEvent writeFloats(long byteOffset, float[] data, ComputeQueue queue) {
        if (data.length == 0) {
            return null;
        }
        long bytes = (long) data.length * Float.BYTES;
        return write(queue, (q, waits) ->
                q.enqueueWriteBuffer(refCountedBuffer.gpuBuffer, offsetBytes + byteOffset, bytes, data, waits));
    }

    /** A read under the backend lock, ordered after pending train work, complete on return. */
    private GpuEvent read(ComputeQueue queue, Transfer transfer) {
        return transfer(queue, transfer, false);
    }

    /** A write under the backend lock, ordered after pending train work, complete and ordered before later launches. */
    private GpuEvent write(ComputeQueue queue, Transfer transfer) {
        return transfer(queue, transfer, true);
    }

    private GpuEvent transfer(ComputeQueue queue, Transfer transfer, boolean isWrite) {
        ensureOpen();
        ReentrantLock lock = BackendResources.lock(backend);
        lock.lock();
        List<GpuEvent> waits = new ArrayList<>(1);
        try {
            TrainGpu.beforeHostRead(backend, queue, waits);
            GpuEvent event = transfer.enqueue(queue, waits.toArray(new GpuEvent[0]));
            try {
                queue.finish();
                if (isWrite) {
                    TrainGpu.afterHostWrite(backend, event);
                }
            } catch (RuntimeException e) {
                closeEvent(event);
                throw e;
            }
            return event;
        } finally {
            for (GpuEvent w : waits) {
                w.close();
            }
            lock.unlock();
        }
    }

    private static void closeEvent(GpuEvent event) {
        if (event != null) {
            event.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("TensorStorage has been closed");
        }
    }

    /** Releases this storage's reference to the buffer once; later calls do nothing. */
    @Override
    public void close() {
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
        }
        refCountedBuffer.release();
    }

    @Override
    public String toString() {
        return "TensorStorage{" +
            "offsetBytes=" + offsetBytes +
            ", lengthBytes=" + lengthBytes +
            ", dtype=" + dtype +
            ", isPrimaryOwner=" + isPrimaryOwner +
            ", refCount=" + (refCountedBuffer != null ? refCountedBuffer.refCount.get() : 0) +
            ", closed=" + closed +
            '}';
    }

    /** One enqueued transfer: issues it on {@code queue} with {@code waits} and returns its event. */
    @FunctionalInterface
    private interface Transfer {
        GpuEvent enqueue(ComputeQueue queue, GpuEvent[] waits);
    }

    private static final class RefCountedBuffer {
        private final GpuBuffer gpuBuffer;
        private final AtomicInteger refCount;

        RefCountedBuffer(GpuBuffer gpuBuffer) {
            this.gpuBuffer = Objects.requireNonNull(gpuBuffer, "GpuBuffer cannot be null");
            this.refCount = new AtomicInteger(1);
        }

        void retain() {
            int oldCount = refCount.getAndIncrement();
            if (oldCount <= 0) {
                refCount.getAndDecrement();
                throw new IllegalStateException("Cannot retain buffer with refCount=" + oldCount + " (already released)");
            }
        }

        void release() {
            int newCount = refCount.decrementAndGet();
            if (newCount == 0) {
                gpuBuffer.close();
            } else if (newCount < 0) {
                throw new IllegalStateException("Buffer released more times than retained: refCount=" + newCount);
            }
        }
    }
}

package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemsetD32Async;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.driver.JCudaDriver.cuStreamWaitEvent;

/**
 * One CUDA stream of a {@link CudaComputeBackend}, implementing the {@link ComputeQueue}
 * contract.
 *
 * <ul>
 *   <li><b>Dispatch.</b> {@code global} counts work-items and {@code local} is the block;
 *       the grid is {@code ceil(global / local)} per dimension ({@link WorkSizes#groupCounts}),
 *       so kernels bounds-check their tail. A null {@code local} uses
 *       {@link WorkSizes#defaultLocal}, which divides {@code global[0]}. Blocks are checked
 *       against the function's thread limit and grids against the device limits.
 *       Dispatches without an event ({@code NoEvent}, {@code PreBound}) record nothing.</li>
 *   <li><b>Arguments</b> are sticky on the {@link CudaKernelHandle}; a null or empty
 *       {@code args} array launches the stored arguments.</li>
 *   <li><b>Writes</b> copy the host data into a page-locked upload ring before returning,
 *       then issue {@code cuMemcpyHtoDAsync} from it; transfers larger than a ring slice are
 *       split. The caller may reuse its array or buffer immediately.</li>
 *   <li><b>Reads</b> copy device data into a page-locked readback ring. The destination is
 *       filled, in issue order, by {@link #finish()} or by {@link GpuEvent#awaitCompletion()}
 *       on any event of this queue recorded after the read.</li>
 *   <li><b>Waits</b> on events of other queues of the same backend become
 *       {@code cuStreamWaitEvent}; waits on this queue's own events are implied by stream order.</li>
 *   <li><b>Ownership.</b> {@code close()} of a backend-owned queue only finishes it (and does
 *       nothing once the backend is closed); the backend releases it with {@link #closeInternal()}.
 *       A caller-owned queue is released by {@code close()}, or by the backend's close when it is
 *       still open then; closing it afterwards does nothing.</li>
 *   <li><b>Buffer fences.</b> The backend's buffer pool records a point on the stream of every
 *       open queue when a buffer is released ({@link #recordFence()}), from any thread.</li>
 * </ul>
 */
@NotThreadSafe("One thread enqueues at a time. Events of this queue may be awaited from other threads; "
        + "read completion and staging memory are guarded by an internal lock.")
public final class CudaComputeQueue implements ComputeQueue {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaComputeQueue.class);

    /** Largest block z dimension of every CUDA device. */
    private static final long MAX_BLOCK_DIM_Z = 64;

    private final CudaComputeBackend backend;
    private final CudaContext context;
    private final CudaEventPool events;
    private final CUstream stream;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final long[] maxGrid;
    private final Object transferLock = new Object();
    private final Object fenceLock = new Object();
    private final CudaPinnedRing upload;
    private final CudaPinnedRing readback;
    private final ArrayDeque<PendingRead> pendingReads = new ArrayDeque<>();
    private final AtomicLong sequence = new AtomicLong();
    private volatile boolean closed;
    /** Guarded by {@link #fenceLock}: the stream takes buffer fences until it has been drained on release. */
    private boolean fenceable = true;

    /**
     * Wraps {@code stream}, which this queue owns and destroys on release (also when this
     * constructor fails). The backend's context, event pool and capabilities must exist.
     *
     * @param backend        the owning backend
     * @param stream         a stream created in the backend's context
     * @param type           compute or transfer
     * @param ownedByBackend true for the backend's own queues, whose {@code close()} only finishes
     * @throws IllegalArgumentException when an argument is null, the backend is not ready, or
     *                                  {@code -Dneuralfree.gpu.stagingMb} is invalid
     */
    public CudaComputeQueue(CudaComputeBackend backend, CUstream stream, QueueType type, boolean ownedByBackend) {
        if (stream == null) {
            throw new IllegalArgumentException("CUstream cannot be null");
        }
        try {
            if (backend == null) {
                throw new IllegalArgumentException("CUDA backend cannot be null");
            }
            if (type == null) {
                throw new IllegalArgumentException("queueType cannot be null");
            }
            if (backend.context() == null || backend.eventPool() == null || backend.capabilities() == null) {
                throw new IllegalArgumentException("CUDA backend must create its context, event pool and "
                        + "capabilities before its queues");
            }
            this.backend = backend;
            this.context = backend.context();
            this.events = backend.eventPool();
            this.stream = stream;
            this.queueType = type;
            this.ownedByBackend = ownedByBackend;
            this.maxGrid = backend.capabilities().getMaxWorkGroupCount();
            String label = type.name().toLowerCase(Locale.ROOT);
            this.upload = new CudaPinnedRing(context, label + " upload",
                    CudaPinnedRing.configuredMaxBytes(CudaPinnedRing.DEFAULT_UPLOAD_MB));
            this.readback = new CudaPinnedRing(context, label + " readback",
                    CudaPinnedRing.configuredMaxBytes(CudaPinnedRing.DEFAULT_READBACK_MB));
        } catch (RuntimeException e) {
            try {
                CudaErrors.check("cuStreamDestroy", () -> cuStreamDestroy(stream));
            } catch (RuntimeException destroy) {
                e.addSuppressed(destroy);
            }
            throw e;
        }
    }

    @Override
    public QueueType getQueueType() {
        return queueType;
    }

    /** The underlying stream. */
    public CUstream getCUStream() {
        return stream;
    }

    /** True for the backend's compute and transfer queues. */
    public boolean isOwnedByBackend() {
        return ownedByBackend;
    }

    /** True once the queue has been released. */
    public boolean isClosed() {
        return closed;
    }

    // ------------------------------------------------------------------------------------------
    // Host transfers
    // ------------------------------------------------------------------------------------------

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       byte[] srcData, GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source array must not be null");
        }
        checkHostRange(sizeInBytes, srcData.length, "write", "source byte array");
        return write(dstBuffer, dstOffset, sizeInBytes, (staging, from, n) -> staging.put(srcData, (int) from, n),
                waitOnEvents);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       float[] srcData, GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source array must not be null");
        }
        requireWholeFloats(sizeInBytes, "write");
        checkHostRange(sizeInBytes, (long) srcData.length * Float.BYTES, "write", "source float array");
        return write(dstBuffer, dstOffset, sizeInBytes,
                (staging, from, n) -> staging.asFloatBuffer().put(srcData, (int) (from / Float.BYTES), n / Float.BYTES),
                waitOnEvents);
    }

    /** Writes bytes {@code [0, sizeInBytes)} of {@code srcData}, heap or direct, read by absolute index. */
    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                       ByteBuffer srcData, GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source ByteBuffer must not be null");
        }
        checkHostRange(sizeInBytes, srcData.capacity(), "write", "source ByteBuffer capacity");
        return write(dstBuffer, dstOffset, sizeInBytes, (staging, from, n) -> {
            ByteBuffer view = srcData.duplicate();
            view.clear();
            view.position((int) from);
            view.limit((int) from + n);
            staging.put(view);
        }, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      byte[] dstData, GpuEvent... waitOnEvents) {
        if (dstData == null) {
            throw new IllegalArgumentException("read destination array must not be null");
        }
        checkHostRange(sizeInBytes, dstData.length, "read", "destination byte array");
        return read(srcBuffer, srcOffset, sizeInBytes, (staging, to, n) -> staging.get(dstData, (int) to, n),
                waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                      float[] dstData, GpuEvent... waitOnEvents) {
        if (dstData == null) {
            throw new IllegalArgumentException("read destination array must not be null");
        }
        requireWholeFloats(sizeInBytes, "read");
        checkHostRange(sizeInBytes, (long) dstData.length * Float.BYTES, "read", "destination float array");
        return read(srcBuffer, srcOffset, sizeInBytes,
                (staging, to, n) -> staging.asFloatBuffer().get(dstData, (int) (to / Float.BYTES), n / Float.BYTES),
                waitOnEvents);
    }

    // ------------------------------------------------------------------------------------------
    // Device-side commands
    // ------------------------------------------------------------------------------------------

    /**
     * Copies {@code sizeInBytes} bytes on the device ({@code cuMemcpyDtoDAsync}). Overlapping
     * ranges of one buffer are rejected.
     */
    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                  long sizeInBytes, GpuEvent... waitOnEvents) {
        enter();
        CUdeviceptr from = devicePointer(src, "copy source");
        CUdeviceptr to = devicePointer(dst, "copy destination");
        checkDeviceRange(src, srcOffset, sizeInBytes, "copy source");
        checkDeviceRange(dst, dstOffset, sizeInBytes, "copy destination");
        if (src == dst && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        validateWaits(waitOnEvents);
        applyWaits(waitOnEvents);
        CudaErrors.check("cuMemcpyDtoDAsync", () -> cuMemcpyDtoDAsync(to.withByteOffset(dstOffset),
                from.withByteOffset(srcOffset), sizeInBytes, stream));
    }

    /** Fills the range with the 32-bit {@code pattern} on the device ({@code cuMemsetD32Async}). */
    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        enter();
        if (dst == null) {
            throw new IllegalArgumentException("fill destination must not be null");
        }
        if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                    + " must be non-negative multiples of 4");
        }
        if (dstOffset + sizeInBytes > dst.sizeInBytes()) {
            throw new IllegalArgumentException("fill range [" + dstOffset + ", " + (dstOffset + sizeInBytes)
                    + ") exceeds the buffer size of " + dst.sizeInBytes() + " bytes");
        }
        CUdeviceptr ptr = devicePointer(dst, "fill destination");
        validateWaits(waitOnEvents);
        if (sizeInBytes == 0) {
            return;
        }
        applyWaits(waitOnEvents);
        long words = sizeInBytes / 4;
        CudaErrors.check("cuMemsetD32Async", () -> cuMemsetD32Async(ptr.withByteOffset(dstOffset), pattern, words,
                stream));
    }

    /** An event that completes once everything enqueued so far, and the waits, has completed. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        enter();
        validateWaits(waitOnEvents);
        applyWaits(waitOnEvents);
        return record();
    }

    // ------------------------------------------------------------------------------------------
    // Kernels
    // ------------------------------------------------------------------------------------------

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                  Object[] args, GpuEvent... waitOnEvents) {
        return launch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, true);
    }

    @Override
    public void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                     Object[] args, GpuEvent... waitOnEvents) {
        launch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, false);
    }

    /** Stores one argument on the handle; the queue it is set through does not matter. */
    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        enter();
        CudaKernelHandle handle = ownHandle(kernel);
        if (handle.isClosed()) {
            throw new IllegalStateException("CUDA kernel " + handle.kernelName() + " was used after it was closed");
        }
        handle.args().set(argIndex, arg);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      GpuEvent... waitOnEvents) {
        launch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, false);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                   long[] localWorkSize, GpuEvent... waitOnEvents) {
        return launch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, true);
    }

    // ------------------------------------------------------------------------------------------
    // Synchronisation and lifecycle
    // ------------------------------------------------------------------------------------------

    /** Commands are submitted when enqueued, so there is nothing to flush. */
    @Override
    public void flush() {
        enter();
    }

    /** Waits for the stream to drain, then fills every pending read destination in issue order. */
    @Override
    public void finish() {
        enter();
        CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(stream));
        retireReadsUpTo(Long.MAX_VALUE);
    }

    /**
     * A backend-owned queue only finishes, and does nothing once its backend is closed; a
     * caller-owned queue is released ({@link #closeInternal()}).
     */
    @Override
    public void close() {
        if (ownedByBackend) {
            if (closed || backend.isClosed()) {
                return;
            }
            finish();
            return;
        }
        closeInternal();
    }

    /**
     * Releases the queue: drains the stream, completes pending reads, frees both staging rings
     * and destroys the stream. Idempotent. It needs only the context, so the backend calls it
     * while closing, after marking itself closed and before releasing its context; the backend
     * releases every queue it created that way, so a queue never outlives the context retain
     * its stream and staging memory are freed under.
     *
     * @throws IllegalStateException when the backend's context was released while this queue was
     *                               still open; the stream and staging memory can then no longer
     *                               be freed by this queue
     * @throws RuntimeException      the first translated driver failure, after every step was attempted
     */
    public void closeInternal() {
        synchronized (transferLock) {
            if (closed) {
                return;
            }
            closed = true;
        }
        if (context.isReleased()) {
            stopFences();
            throw new IllegalStateException("CUDA " + queueType + " queue was still open when its backend released "
                    + "the primary context; the backend must release every queue it created first");
        }
        RuntimeException failure = null;
        boolean drained = false;
        try {
            context.makeCurrent();
            CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(stream));
            drained = true;
        } catch (RuntimeException e) {
            failure = e;
        }
        stopFences();
        synchronized (transferLock) {
            try {
                if (drained) {
                    retireReadsLocked(Long.MAX_VALUE);
                } else {
                    dropPendingReads();
                }
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
            failure = closeRing(upload, failure);
            failure = closeRing(readback, failure);
        }
        try {
            CudaErrors.check("cuStreamDestroy", () -> cuStreamDestroy(stream));
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "CudaComputeQueue{" + queueType + (ownedByBackend ? ", backend-owned" : "") + (closed ? ", closed" : "")
                + "}";
    }

    /**
     * Records a point on this stream for the backend's buffer pool: the point is reached once
     * everything enqueued on the stream so far has finished. The event has no owning queue, so
     * it retires no reads. Safe from any thread. A queue being released keeps taking fences until
     * {@link #closeInternal()} has drained its stream, so work still running on it is covered.
     *
     * @return the event, or null once the released queue's stream has been drained
     * @throws RuntimeException the translated driver failure
     */
    CudaGpuEvent recordFence() {
        synchronized (fenceLock) {
            if (!fenceable) {
                return null;
            }
            return events.record(stream);
        }
    }

    /** Next sequence number of a point recorded on this stream; numbers grow in issue order. */
    long nextSequence() {
        return sequence.incrementAndGet();
    }

    /**
     * Fills the destinations of every read recorded at or before {@code upTo}. The caller has
     * already waited for a point of this stream at or after {@code upTo}.
     */
    void retireReadsUpTo(long upTo) {
        synchronized (transferLock) {
            retireReadsLocked(upTo);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------------------------------

    private CudaGpuEvent launch(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                GpuEvent[] waits, boolean withEvent) {
        enter();
        CudaKernelHandle handle = ownHandle(kernel);
        CUfunction function = handle.function();
        String name = handle.kernelName();
        long[] block = local == null ? WorkSizes.defaultLocal(global, handle.maxThreadsPerBlock()) : local;
        WorkSizes.validateLocal(block, handle.maxThreadsPerBlock(), name);
        if (block.length == 3 && block[2] > MAX_BLOCK_DIM_Z) {
            throw new IllegalArgumentException(name + ": block z dimension " + block[2] + " exceeds the CUDA limit of "
                    + MAX_BLOCK_DIM_Z);
        }
        long[] grid = WorkSizes.groupCounts(global, block);
        for (int d = 0; d < grid.length; d++) {
            if (grid[d] > maxGrid[d]) {
                throw new IllegalArgumentException(name + ": " + grid[d] + " blocks in dimension " + d
                        + " (global " + global[d] + ", local " + block[d] + ") exceed the device limit of "
                        + maxGrid[d]);
            }
        }
        validateWaits(waits);
        if (args != null && args.length > 0) {
            handle.args().setAll(args);
        }
        Pointer params = handle.args().kernelParams();
        applyWaits(waits);
        int gx = (int) grid[0];
        int gy = grid.length > 1 ? (int) grid[1] : 1;
        int gz = grid.length > 2 ? (int) grid[2] : 1;
        int bx = (int) block[0];
        int by = block.length > 1 ? (int) block[1] : 1;
        int bz = block.length > 2 ? (int) block[2] : 1;
        int result;
        try {
            result = cuLaunchKernel(function, gx, gy, gz, bx, by, bz, 0, stream, params, null);
        } catch (CudaException e) {
            throw CudaErrors.translate(e, "cuLaunchKernel(" + name + ")", -1);
        }
        if (result != CudaErrors.SUCCESS) {
            throw CudaErrors.translate(result, "cuLaunchKernel(" + name + ")", -1);
        }
        return withEvent ? record() : null;
    }

    private GpuEvent write(GpuBuffer dst, long dstOffset, long size, HostSource source, GpuEvent[] waits) {
        enter();
        CUdeviceptr device = devicePointer(dst, "write destination");
        checkDeviceRange(dst, dstOffset, size, "write destination");
        validateWaits(waits);
        synchronized (transferLock) {
            applyWaits(waits);
            CudaGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, upload.maxSliceBytes());
                    CudaPinnedRing.Slice slice = upload.acquire(n);
                    CUdeviceptr target = device.withByteOffset(dstOffset + done);
                    try {
                        source.copyInto(slice.byteBuffer(), done, n);
                        Pointer staged = slice.pointer();
                        CudaErrors.check("cuMemcpyHtoDAsync", () -> cuMemcpyHtoDAsync(target, staged, n, stream));
                    } catch (RuntimeException e) {
                        slice.release();
                        throw e;
                    }
                    CudaGpuEvent event = recordAfterCopy(slice);
                    slice.releaseAfter(event.share());
                    if (last != null) {
                        last.close();
                    }
                    last = event;
                    done += n;
                }
            } catch (RuntimeException e) {
                if (last != null) {
                    last.close();
                }
                throw e;
            }
            return last;
        }
    }

    private GpuEvent read(GpuBuffer src, long srcOffset, long size, HostSink sink, GpuEvent[] waits) {
        enter();
        CUdeviceptr device = devicePointer(src, "read source");
        checkDeviceRange(src, srcOffset, size, "read source");
        validateWaits(waits);
        synchronized (transferLock) {
            applyWaits(waits);
            CudaGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, readback.maxSliceBytes());
                    CudaPinnedRing.Slice slice = acquireReadSlice(n);
                    CUdeviceptr origin = device.withByteOffset(srcOffset + done);
                    try {
                        Pointer staged = slice.pointer();
                        CudaErrors.check("cuMemcpyDtoHAsync", () -> cuMemcpyDtoHAsync(staged, origin, n, stream));
                    } catch (RuntimeException e) {
                        slice.release();
                        throw e;
                    }
                    CudaGpuEvent event = recordAfterCopy(slice);
                    pendingReads.addLast(new PendingRead(event.share(), slice, sink, done, n));
                    if (last != null) {
                        last.close();
                    }
                    last = event;
                    done += n;
                }
            } catch (RuntimeException e) {
                if (last != null) {
                    last.close();
                }
                throw e;
            }
            return last;
        }
    }

    /** A readback slice; when the ring is full of unretired reads, waits for the oldest and retires it. */
    private CudaPinnedRing.Slice acquireReadSlice(int n) {
        CudaPinnedRing.Slice slice = readback.tryAcquire(n);
        while (slice == null) {
            PendingRead oldest = pendingReads.peekFirst();
            if (oldest == null) {
                throw new IllegalStateException("CUDA " + queueType + " readback ring is full without pending reads");
            }
            oldest.event.synchronize();
            retireReadsLocked(oldest.sequence);
            slice = readback.tryAcquire(n);
        }
        return slice;
    }

    /**
     * Records the event that follows a staged copy. When recording fails, the slice is
     * returned to its ring only after the stream drained, so in-flight memory is never reused.
     */
    private CudaGpuEvent recordAfterCopy(CudaPinnedRing.Slice slice) {
        try {
            return record();
        } catch (RuntimeException e) {
            try {
                CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(stream));
                slice.release();
            } catch (RuntimeException drain) {
                e.addSuppressed(drain);
            }
            throw e;
        }
    }

    private CudaGpuEvent record() {
        return events.record(stream, this);
    }

    private void retireReadsLocked(long upTo) {
        RuntimeException failure = null;
        while (!pendingReads.isEmpty() && pendingReads.peekFirst().sequence <= upTo) {
            try {
                pendingReads.pollFirst().complete();
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Discards reads whose copies did not provably finish; their destinations stay unwritten. */
    private void dropPendingReads() {
        if (!pendingReads.isEmpty()) {
            LOGGER.warn("CUDA {} queue closed with {} unfinished reads; their destinations were not written",
                    queueType, pendingReads.size());
        }
        while (!pendingReads.isEmpty()) {
            pendingReads.pollFirst().event.close();
        }
    }

    /**
     * Ends buffer fencing once the stream has been drained, before the stream is destroyed, and
     * drops this queue from the backend's fenced queues.
     */
    private void stopFences() {
        synchronized (fenceLock) {
            fenceable = false;
        }
        backend.onQueueReleased(this);
    }

    private static RuntimeException closeRing(CudaPinnedRing ring, RuntimeException failure) {
        try {
            ring.close();
            return failure;
        } catch (RuntimeException e) {
            return Closeables.chain(failure, e);
        }
    }

    /**
     * Checks every wait event before any command is issued.
     *
     * @throws IllegalArgumentException for an event that is null, of another backend, or closed
     */
    private void validateWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (int i = 0; i < waits.length; i++) {
            GpuEvent e = waits[i];
            if (!(e instanceof CudaGpuEvent)) {
                throw new IllegalArgumentException("wait event " + i + " is not a CUDA event of this backend: "
                        + (e == null ? "null" : e.getClass().getName()));
            }
            CudaGpuEvent cuda = (CudaGpuEvent) e;
            if (!cuda.isFrom(events)) {
                throw new IllegalArgumentException("wait event " + i + " belongs to another CUDA backend");
            }
            if (cuda.isClosed()) {
                throw new IllegalArgumentException("wait event " + i + " was already closed");
            }
        }
    }

    /** Makes the stream wait on events of other streams; this stream's own events are already ordered. */
    private void applyWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (GpuEvent e : waits) {
            CudaGpuEvent cuda = (CudaGpuEvent) e;
            if (cuda.owner() == this) {
                continue;
            }
            CUevent event = cuda.getCUEvent();
            CudaErrors.check("cuStreamWaitEvent", () -> cuStreamWaitEvent(stream, event, 0));
        }
    }

    /**
     * Checks this queue and its backend are open and makes the context current.
     *
     * @throws IllegalStateException when either is closed
     */
    private void enter() {
        if (closed) {
            throw new IllegalStateException("CUDA " + queueType + " queue already closed");
        }
        backend.enter();
    }

    private CudaKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof CudaKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a CudaKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        CudaKernelHandle handle = (CudaKernelHandle) kernel;
        if (handle.backend() != backend) {
            throw new IllegalArgumentException("CUDA kernel " + handle.kernelName() + " belongs to another backend");
        }
        return handle;
    }

    private CUdeviceptr devicePointer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof CudaGpuBuffer)) {
            throw new IllegalArgumentException(role + " must be a CUDA buffer, got "
                    + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        CudaGpuBuffer cuda = (CudaGpuBuffer) buffer;
        if (cuda.backend() != backend) {
            throw new IllegalArgumentException(role + " belongs to another CUDA backend");
        }
        CUdeviceptr ptr = cuda.getCUDevicePtr();
        if (ptr == null) {
            throw new IllegalArgumentException(role + " was already closed");
        }
        return ptr;
    }

    private static void checkDeviceRange(GpuBuffer buffer, long offset, long size, String role) {
        if (offset < 0) {
            throw new IllegalArgumentException(role + " offset must not be negative, got " + offset);
        }
        if (size <= 0) {
            throw new IllegalArgumentException(role + " size must be positive, got " + size);
        }
        if (offset > buffer.sizeInBytes() - size) {
            throw new IllegalArgumentException(role + " range [" + offset + ", " + (offset + size)
                    + ") exceeds the buffer size of " + buffer.sizeInBytes() + " bytes");
        }
    }

    private static void checkHostRange(long size, long available, String op, String what) {
        if (size > available) {
            throw new IllegalArgumentException(op + " of " + size + " bytes exceeds the " + what + " of "
                    + available + " bytes");
        }
    }

    private static void requireWholeFloats(long size, String op) {
        if (size % Float.BYTES != 0) {
            throw new IllegalArgumentException(op + " size " + size + " is not a whole number of floats");
        }
    }

    /** Copies {@code n} host bytes starting at byte {@code from} into a staging view. */
    private interface HostSource {
        void copyInto(ByteBuffer staging, long from, int n);
    }

    /** Copies {@code n} bytes of a staging view into the host destination at byte {@code to}. */
    private interface HostSink {
        void copyFrom(ByteBuffer staging, long to, int n);
    }

    /** A device-to-host copy whose destination is filled once its stream point is reached. */
    private static final class PendingRead {
        private final CudaGpuEvent event;
        private final long sequence;
        private final CudaPinnedRing.Slice slice;
        private final HostSink sink;
        private final long hostOffset;
        private final int size;

        PendingRead(CudaGpuEvent event, CudaPinnedRing.Slice slice, HostSink sink, long hostOffset, int size) {
            this.event = event;
            this.sequence = event.sequence();
            this.slice = slice;
            this.sink = sink;
            this.hostOffset = hostOffset;
            this.size = size;
        }

        /** Copies the staged bytes to the destination and hands back the slice and event. */
        void complete() {
            try {
                sink.copyFrom(slice.byteBuffer(), hostOffset, size);
            } finally {
                slice.release();
                event.close();
            }
        }
    }
}

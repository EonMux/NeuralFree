package com.eonmux.neuralfree.acceleration.backend.ocl;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.*;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.jocl.CLException;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;

/**
 * An in-order OpenCL command queue.
 *
 * <p><b>Transfers.</b> Host data goes through direct staging buffers of an {@link OclStagingPool}:
 * writes copy the caller's data at enqueue and the staging buffer stays in flight until
 * {@link #finish()}. Reads land in a staging buffer that the read's {@code CL_COMPLETE} callback
 * copies into the caller's array; a read that terminated abnormally copies nothing and is reported
 * by {@link #finish()} or by its event's {@code awaitCompletion()}. Host arrays must cover the
 * transfer size, and float transfers must be whole floats.</p>
 *
 * <p><b>Errors.</b> Every driver failure is translated by {@link OclErrors}: exhausted memory becomes
 * {@code GpuOutOfMemoryException}, every other status a {@code GpuBackendException} naming the call.</p>
 *
 * <p><b>Dispatch.</b> A full argument list must match the kernel's {@code CL_KERNEL_NUM_ARGS}, and a
 * work-group may not exceed the kernel's {@code CL_KERNEL_WORK_GROUP_SIZE}; the driver applies the
 * remaining rules (divisibility, device limits).</p>
 */
@NotThreadSafe("OpenCL command queues are not thread-safe for concurrent enqueue operations. "
        + "Caller must synchronize access when using from multiple threads. "
        + "Recommended: create separate queue instances per thread, or use external mutex.")
public final class OclComputeQueue implements ComputeQueue {
    private static final Logger LOGGER = LoggingUtil.getLogger(OclComputeQueue.class);

    private cl_command_queue queue;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final boolean profiled;
    private boolean closed = false;
    /** Reads enqueued since the last finish; their callbacks run on a driver thread. */
    private final List<OclReadCompletion> pendingReads = new ArrayList<>();
    private final OclStagingPool staging = new OclStagingPool();

    // Reusable single-element arrays for clSetKernelArg scalar values.
    // clSetKernelArg copies the value immediately, so these are safe to reuse
    // across sequential setKernelArgs calls within the same thread.
    private final int[]    scalarIntBuf    = new int[1];
    private final long[]   scalarLongBuf   = new long[1];
    private final float[]  scalarFloatBuf  = new float[1];
    private final double[] scalarDoubleBuf = new double[1];
    private static final cl_event[] EMPTY_CL_EVENTS = new cl_event[0];
    // Reusable single-element cl_event array for the common case (1 wait event).
    // Safe because enqueue calls are single-threaded per queue (OpenCL spec).
    private final cl_event[] singleClEventBuf = new cl_event[1];

    /** A caller-owned queue without profiling; {@link #close()} releases it. */
    public OclComputeQueue(cl_command_queue queue, QueueType queueType) {
        this(queue, queueType, false, false);
    }

    /**
     * @param queue          the OpenCL command queue; released by this object
     * @param queueType      compute or transfer
     * @param ownedByBackend true for the backend's own queues: {@link #close()} then only finishes
     *                       and the backend releases the queue through {@link #closeInternal()}
     * @param profiled       true when the queue was created with {@code CL_QUEUE_PROFILING_ENABLE}
     */
    public OclComputeQueue(cl_command_queue queue, QueueType queueType, boolean ownedByBackend, boolean profiled) {
        if (queue == null) {
            throw new IllegalArgumentException("cl_command_queue cannot be null");
        }
        if (queueType == null) {
            throw new IllegalArgumentException("queueType cannot be null");
        }
        this.queue = queue;
        this.queueType = queueType;
        this.ownedByBackend = ownedByBackend;
        this.profiled = profiled;
    }

    /** True for the queues returned by {@code computeQueue()} and {@code transferQueue()}. */
    public boolean isOwnedByBackend() {
        return ownedByBackend;
    }

    @Override
    public QueueType getQueueType() {
        return queueType;
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                        byte[] srcData, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (srcData == null) {
            throw new IllegalArgumentException("srcData cannot be null");
        }
        checkDeviceRange(dstBuffer, dstOffset, sizeInBytes, "write destination");
        checkHostRange(sizeInBytes, srcData.length, "write", "source byte array");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = staging.acquire((int) sizeInBytes);
        buf.put(srcData, 0, (int) sizeInBytes);
        buf.flip();
        return enqueueStagedWrite(mem, dstOffset, sizeInBytes, buf, waitList);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                        float[] srcData, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (srcData == null) {
            throw new IllegalArgumentException("srcData cannot be null");
        }
        checkDeviceRange(dstBuffer, dstOffset, sizeInBytes, "write destination");
        requireWholeFloats(sizeInBytes, "write");
        checkHostRange(sizeInBytes, (long) srcData.length * Float.BYTES, "write", "source float array");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = staging.acquire((int) sizeInBytes);
        buf.asFloatBuffer().put(srcData, 0, (int) (sizeInBytes / Float.BYTES));
        return enqueueStagedWrite(mem, dstOffset, sizeInBytes, buf, waitList);
    }

    /**
     * Writes bytes {@code [0, sizeInBytes)} of {@code srcData}, read by absolute index into a staging
     * copy, so the caller may reuse the buffer at once and its position and limit stay untouched.
     */
    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                        java.nio.ByteBuffer srcData, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (srcData == null) {
            throw new IllegalArgumentException("srcData cannot be null");
        }
        checkDeviceRange(dstBuffer, dstOffset, sizeInBytes, "write destination");
        checkHostRange(sizeInBytes, srcData.capacity(), "write", "source ByteBuffer capacity");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = staging.acquire((int) sizeInBytes);
        ByteBuffer view = srcData.duplicate();
        view.clear();
        view.limit((int) sizeInBytes);
        buf.put(view);
        buf.flip();
        return enqueueStagedWrite(mem, dstOffset, sizeInBytes, buf, waitList);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                       byte[] dstData, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (dstData == null) {
            throw new IllegalArgumentException("dstData cannot be null");
        }
        checkDeviceRange(srcBuffer, srcOffset, sizeInBytes, "read source");
        checkHostRange(sizeInBytes, dstData.length, "read", "destination byte array");
        cl_mem mem = extractClMem(srcBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = staging.acquire((int) sizeInBytes);
        int n = (int) sizeInBytes;
        return enqueueStagedRead(mem, srcOffset, sizeInBytes, buf, waitList, () -> {
            buf.rewind();
            buf.get(dstData, 0, n);
        });
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                       float[] dstData, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (dstData == null) {
            throw new IllegalArgumentException("dstData cannot be null");
        }
        checkDeviceRange(srcBuffer, srcOffset, sizeInBytes, "read source");
        requireWholeFloats(sizeInBytes, "read");
        checkHostRange(sizeInBytes, (long) dstData.length * Float.BYTES, "read", "destination float array");
        cl_mem mem = extractClMem(srcBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = staging.acquire((int) sizeInBytes);
        FloatBuffer floats = buf.asFloatBuffer();
        int n = (int) (sizeInBytes / Float.BYTES);
        return enqueueStagedRead(mem, srcOffset, sizeInBytes, buf, waitList, () -> {
            floats.rewind();
            floats.get(dstData, 0, n);
        });
    }

    @Override
    public void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      Object[] args, GpuEvent... waitOnEvents) {
        ensureOpen();
        OclKernelHandle handle = prepareDispatch(kernel, globalWorkSize, localWorkSize);
        setKernelArgs(handle, args);
        dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, null);
    }

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                   Object[] args, GpuEvent... waitOnEvents) {
        ensureOpen();
        OclKernelHandle handle = prepareDispatch(kernel, globalWorkSize, localWorkSize);
        setKernelArgs(handle, args);
        cl_event event = new cl_event();
        dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, event);
        return new OclGpuEvent(event, null, profiled);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize,
                                       long[] localWorkSize, GpuEvent... waitOnEvents) {
        ensureOpen();
        OclKernelHandle handle = prepareDispatch(kernel, globalWorkSize, localWorkSize);
        dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, null);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                    long[] localWorkSize, GpuEvent... waitOnEvents) {
        ensureOpen();
        OclKernelHandle handle = prepareDispatch(kernel, globalWorkSize, localWorkSize);
        cl_event event = new cl_event();
        dispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, event);
        return new OclGpuEvent(event, null, profiled);
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                   long sizeInBytes, GpuEvent... waitOnEvents) {
        ensureOpen();
        if (src == null || dst == null) {
            throw new IllegalArgumentException("src and dst buffers cannot be null");
        }
        checkDeviceRange(src, srcOffset, sizeInBytes, "copy source");
        checkDeviceRange(dst, dstOffset, sizeInBytes, "copy destination");
        if (src == dst && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        cl_mem srcMem = extractClMem(src);
        cl_mem dstMem = extractClMem(dst);
        cl_event[] waitList = extractWaitList(waitOnEvents);
        OclErrors.check("clEnqueueCopyBuffer", sizeInBytes + " bytes",
                () -> clEnqueueCopyBuffer(queue, srcMem, dstMem, srcOffset, dstOffset, sizeInBytes,
                        waitList.length, waitList.length == 0 ? null : waitList, null));
    }

    /**
     * Fills the range with the 32-bit {@code pattern}, stored little-endian, through
     * {@code clEnqueueFillBuffer} (OpenCL 1.2).
     */
    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        ensureOpen();
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
        cl_mem mem = extractClMem(dst);
        cl_event[] waitList = extractWaitList(waitOnEvents);
        if (sizeInBytes == 0) {
            return;
        }
        byte[] patternBytes = {
                (byte) pattern, (byte) (pattern >>> 8), (byte) (pattern >>> 16), (byte) (pattern >>> 24)
        };
        OclErrors.check("clEnqueueFillBuffer", sizeInBytes + " bytes",
                () -> clEnqueueFillBuffer(queue, mem, Pointer.to(patternBytes), patternBytes.length,
                        dstOffset, sizeInBytes, waitList.length, waitList.length == 0 ? null : waitList, null));
    }

    /**
     * Enqueues a marker through {@code clEnqueueMarkerWithWaitList}. The queue executes in
     * order, so the marker completes after every command enqueued before it and after the waits.
     */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        ensureOpen();
        cl_event[] waitList = extractWaitList(waitOnEvents);
        cl_event event = new cl_event();
        OclErrors.check("clEnqueueMarkerWithWaitList", null,
                () -> clEnqueueMarkerWithWaitList(queue, waitList.length, waitList.length == 0 ? null : waitList,
                        event));
        return new OclGpuEvent(event, null, profiled);
    }

    /**
     * Replaces one argument of the kernel.
     *
     * @throws IllegalArgumentException when the index is outside the kernel's arguments or the value
     *                                  is null or unsupported
     */
    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        ensureOpen();
        if (kernel == null) {
            throw new IllegalArgumentException("kernel cannot be null");
        }
        if (arg == null) {
            throw new IllegalArgumentException("arg " + argIndex + " of " + kernel.kernelName() + " is null");
        }
        OclKernelHandle handle = extractHandle(kernel);
        if (argIndex < 0 || (handle.numArgs() != OclKernelHandle.UNKNOWN && argIndex >= handle.numArgs())) {
            throw new IllegalArgumentException("arg index " + argIndex + " is outside the "
                    + (handle.numArgs() == OclKernelHandle.UNKNOWN ? "" : handle.numArgs() + " ")
                    + "arguments of OpenCL kernel " + handle.kernelName());
        }
        setSingleKernelArg(handle, argIndex, arg);
    }

    @Override
    public void flush() {
        ensureOpen();
        OclErrors.check("clFlush", null, () -> clFlush(queue));
    }

    /**
     * Waits for every enqueued command and every read callback, recycles the staging buffers, then
     * reports the first read that terminated abnormally (later ones are attached as suppressed).
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException     when the queue or a read failed
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when it failed for lack of memory
     */
    @Override
    public void finish() {
        ensureOpen();
        OclErrors.check("clFinish", null, () -> clFinish(queue));

        // clFinish completes the device work, but JOCL runs each read's CL_COMPLETE callback on a
        // driver thread that may not have run yet. Every callback reads its staging buffer, so all
        // of them must have run before the staging buffers are recycled.
        List<OclReadCompletion> reads;
        synchronized (pendingReads) {
            reads = new ArrayList<>(pendingReads);
            pendingReads.clear();
        }
        for (int i = 0; i < reads.size(); i++) {
            try {
                reads.get(i).await();
            } catch (InterruptedException e) {
                // Every read stays pending, so the next finish() still waits for the callbacks that
                // have not run and reports the failures of those that have.
                synchronized (pendingReads) {
                    pendingReads.addAll(0, reads);
                }
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for OpenCL read callbacks", e);
            }
        }
        staging.recycleInFlight();

        RuntimeException failure = null;
        for (OclReadCompletion read : reads) {
            try {
                read.throwIfFailed();
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * For a backend-owned queue this is {@link #finish()}; the queue stays usable and the
     * backend releases it on its own close. For a caller-owned queue it finishes and releases.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (ownedByBackend) {
            finish();
            return;
        }
        closeInternal();
    }

    /** Finishes and releases the queue; called by the owning backend. Idempotent. */
    void closeInternal() {
        if (!closed) {
            try {
                // clReleaseCommandQueue only decrements the reference count and does not wait for
                // pending commands, so the queue is finished first.
                finish();
            } finally {
                // The queue is released even when finish() reports a device error.
                try {
                    clReleaseCommandQueue(queue);
                } catch (CLException e) {
                    LOGGER.warn("clReleaseCommandQueue of the OpenCL {} queue failed: {}", queueType,
                            OclErrors.nameOf(e.getStatus()));
                }
                queue = null;
                closed = true;
                synchronized (pendingReads) {
                    pendingReads.clear();
                }
                staging.clear();
            }
        }
    }

    public cl_command_queue getCLQueue() {
        return queue;
    }

    /** Enqueues a non-blocking write from a staging buffer and keeps the buffer in flight. */
    private GpuEvent enqueueStagedWrite(cl_mem mem, long dstOffset, long sizeInBytes, ByteBuffer buf,
                                        cl_event[] waitList) {
        cl_event event = new cl_event();
        try {
            OclErrors.check("clEnqueueWriteBuffer", sizeInBytes + " bytes",
                    () -> clEnqueueWriteBuffer(queue, mem, CL_FALSE, dstOffset, sizeInBytes, Pointer.to(buf),
                            waitList.length, waitList.length == 0 ? null : waitList, event));
        } catch (RuntimeException e) {
            staging.release(buf);
            throw e;
        }
        staging.retain(buf);
        return new OclGpuEvent(event, null, profiled);
    }

    /**
     * Enqueues a non-blocking read into a staging buffer, keeps the buffer in flight and registers
     * the callback that copies it into the destination when the read completes successfully.
     */
    private GpuEvent enqueueStagedRead(cl_mem mem, long srcOffset, long sizeInBytes, ByteBuffer buf,
                                       cl_event[] waitList, Runnable copyToDestination) {
        cl_event event = new cl_event();
        try {
            OclErrors.check("clEnqueueReadBuffer", sizeInBytes + " bytes",
                    () -> clEnqueueReadBuffer(queue, mem, CL_FALSE, srcOffset, sizeInBytes, Pointer.to(buf),
                            waitList.length, waitList.length == 0 ? null : waitList, event));
        } catch (RuntimeException e) {
            staging.release(buf);
            throw e;
        }
        // The driver may write the staging buffer until the read completes; it is recycled by finish().
        staging.retain(buf);

        OclReadCompletion completion = new OclReadCompletion(sizeInBytes);
        try {
            OclErrors.check("clSetEventCallback", "read of " + sizeInBytes + " bytes",
                    () -> clSetEventCallback(event, CL_COMPLETE,
                            (evt, executionStatus, userData) -> completion.complete(executionStatus,
                                    copyToDestination), null));
        } catch (RuntimeException e) {
            releaseQuietly(event);
            throw e;
        }
        synchronized (pendingReads) {
            pendingReads.add(completion);
        }
        return new OclGpuEvent(event, completion, profiled);
    }

    /** Validates a dispatch and returns the open handle. */
    private static OclKernelHandle prepareDispatch(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize) {
        if (kernel == null) {
            throw new IllegalArgumentException("kernel cannot be null");
        }
        validateWorkSizes(kernel, globalWorkSize, localWorkSize);
        OclKernelHandle handle = extractHandle(kernel);
        handle.checkLocalSize(localWorkSize);
        return handle;
    }

    private void dispatch(OclKernelHandle handle, long[] globalWorkSize, long[] localWorkSize,
                          GpuEvent[] waitOnEvents, cl_event event) {
        cl_kernel clKernel = handle.getCLKernel();
        cl_event[] waitList = extractWaitList(waitOnEvents);
        OclErrors.check("clEnqueueNDRangeKernel", handle.kernelName(),
                () -> clEnqueueNDRangeKernel(queue, clKernel, globalWorkSize.length, null, globalWorkSize,
                        localWorkSize, waitList.length, waitList.length == 0 ? null : waitList, event));
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ComputeQueue already closed");
        }
    }

    private static void checkDeviceRange(GpuBuffer buffer, long offset, long size, String role) {
        if (buffer == null) {
            throw new IllegalArgumentException(role + " buffer cannot be null");
        }
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

    private static cl_mem extractClMem(GpuBuffer buffer) {
        if (!(buffer instanceof OclGpuBuffer)) {
            throw new IllegalArgumentException("buffer must be an OclGpuBuffer");
        }
        cl_mem mem = ((OclGpuBuffer) buffer).getCLMem();
        if (mem == null) {
            throw new IllegalStateException("OpenCL buffer already closed");
        }
        return mem;
    }

    private static OclKernelHandle extractHandle(KernelHandle kernel) {
        if (!(kernel instanceof OclKernelHandle)) {
            throw new IllegalArgumentException("kernel must be an OclKernelHandle");
        }
        OclKernelHandle handle = (OclKernelHandle) kernel;
        if (handle.getCLKernel() == null) {
            throw new IllegalStateException("OpenCL kernel " + kernel.kernelName() + " already closed");
        }
        return handle;
    }

    private cl_event[] extractWaitList(GpuEvent... waitOnEvents) {
        if (waitOnEvents == null || waitOnEvents.length == 0) {
            return EMPTY_CL_EVENTS;
        }
        // Fast path: single wait event (most common case in streamed matvec)
        if (waitOnEvents.length == 1) {
            singleClEventBuf[0] = waitEvent(waitOnEvents[0], 0);
            return singleClEventBuf;
        }
        cl_event[] waitList = new cl_event[waitOnEvents.length];
        for (int i = 0; i < waitOnEvents.length; i++) {
            waitList[i] = waitEvent(waitOnEvents[i], i);
        }
        return waitList;
    }

    private static cl_event waitEvent(GpuEvent e, int index) {
        if (!(e instanceof OclGpuEvent)) {
            throw new IllegalArgumentException("wait event " + index + " is not an OpenCL event of this backend: "
                    + (e == null ? "null" : e.getClass().getName()));
        }
        OclGpuEvent ocl = (OclGpuEvent) e;
        if (ocl.isClosed()) {
            throw new IllegalArgumentException("wait event " + index + " was already closed");
        }
        return ocl.getCLEvent();
    }

    private static void releaseQuietly(cl_event event) {
        try {
            clReleaseEvent(event);
        } catch (CLException e) {
            LOGGER.debug("clReleaseEvent failed: {}", OclErrors.nameOf(e.getStatus()));
        }
    }

    /**
     * Sets every argument when {@code args} is non-empty; null or empty keeps the stored arguments.
     *
     * @throws IllegalArgumentException when the count differs from the kernel's argument count, or an
     *                                  argument is null or unsupported
     */
    private void setKernelArgs(OclKernelHandle handle, Object[] args) {
        if (args == null || args.length == 0) {
            return;
        }
        handle.checkArgCount(args.length);
        for (int i = 0; i < args.length; i++) {
            if (args[i] == null) {
                throw new IllegalArgumentException("arg " + i + " of " + handle.kernelName() + " is null");
            }
            setSingleKernelArg(handle, i, args[i]);
        }
    }

    private void setSingleKernelArg(OclKernelHandle handle, int index, Object arg) {
        long size;
        Pointer value;
        if (arg instanceof GpuBuffer) {
            size = Sizeof.cl_mem;
            value = Pointer.to(extractClMem((GpuBuffer) arg));
        } else if (arg instanceof byte[]) {
            byte[] arr = (byte[]) arg;
            size = (long) arr.length * Sizeof.cl_char;
            value = Pointer.to(arr);
        } else if (arg instanceof short[]) {
            short[] arr = (short[]) arg;
            size = (long) arr.length * Sizeof.cl_short;
            value = Pointer.to(arr);
        } else if (arg instanceof int[]) {
            int[] arr = (int[]) arg;
            size = (long) arr.length * Sizeof.cl_int;
            value = Pointer.to(arr);
        } else if (arg instanceof long[]) {
            long[] arr = (long[]) arg;
            size = (long) arr.length * Sizeof.cl_long;
            value = Pointer.to(arr);
        } else if (arg instanceof float[]) {
            float[] arr = (float[]) arg;
            size = (long) arr.length * Sizeof.cl_float;
            value = Pointer.to(arr);
        } else if (arg instanceof double[]) {
            double[] arr = (double[]) arg;
            size = (long) arr.length * Sizeof.cl_double;
            value = Pointer.to(arr);
        } else if (arg instanceof Boolean) {
            scalarIntBuf[0] = ((Boolean) arg) ? 1 : 0;
            size = Sizeof.cl_int;
            value = Pointer.to(scalarIntBuf);
        } else if (arg instanceof Integer) {
            scalarIntBuf[0] = (Integer) arg;
            size = Sizeof.cl_int;
            value = Pointer.to(scalarIntBuf);
        } else if (arg instanceof Long) {
            scalarLongBuf[0] = (Long) arg;
            size = Sizeof.cl_long;
            value = Pointer.to(scalarLongBuf);
        } else if (arg instanceof Float) {
            scalarFloatBuf[0] = (Float) arg;
            size = Sizeof.cl_float;
            value = Pointer.to(scalarFloatBuf);
        } else if (arg instanceof Double) {
            scalarDoubleBuf[0] = (Double) arg;
            size = Sizeof.cl_double;
            value = Pointer.to(scalarDoubleBuf);
        } else {
            throw new IllegalArgumentException("Unsupported kernel argument type at index " + index + " of "
                    + handle.kernelName() + ": " + arg.getClass().getName());
        }

        int status;
        CLException thrown = null;
        try {
            status = clSetKernelArg(handle.getCLKernel(), index, size, value);
        } catch (CLException e) {
            status = e.getStatus();
            thrown = e;
        }
        if (status != CL_SUCCESS) {
            throw OclErrors.translate(status, "clSetKernelArg", handle.kernelName() + " arg " + index, -1, thrown);
        }
    }

    /**
     * Checks the dispatch shape before it reaches the driver: 1 to 3 non-negative global sizes and,
     * when given, a local size of the same rank with positive entries.
     */
    private static void validateWorkSizes(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize) {
        if (globalWorkSize == null || globalWorkSize.length < 1 || globalWorkSize.length > 3) {
            throw new IllegalArgumentException("globalWorkSize of " + kernel.kernelName()
                    + " must have 1 to 3 dimensions");
        }
        if (localWorkSize != null && localWorkSize.length != globalWorkSize.length) {
            throw new IllegalArgumentException("localWorkSize of " + kernel.kernelName() + " has "
                    + localWorkSize.length + " dimensions but globalWorkSize has " + globalWorkSize.length);
        }
        for (int d = 0; d < globalWorkSize.length; d++) {
            if (globalWorkSize[d] < 0 || (localWorkSize != null && localWorkSize[d] <= 0)) {
                throw new IllegalArgumentException("invalid work sizes of " + kernel.kernelName() + ": global["
                        + d + "]=" + globalWorkSize[d] + (localWorkSize != null ? ", local[" + d + "]=" + localWorkSize[d] : ""));
            }
        }
    }
}

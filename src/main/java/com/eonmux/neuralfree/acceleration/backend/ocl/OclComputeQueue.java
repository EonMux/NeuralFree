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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import static org.jocl.CL.*;

/**
 * An in-order OpenCL command queue.
 *
 * <p><b>Transfers.</b> Host data goes through direct staging buffers of an {@link OclStagingPool}:
 * writes copy the caller's data at enqueue and the staging buffer stays in flight until the queue
 * knows the write completed. Reads land in a staging buffer that the read's {@code CL_COMPLETE}
 * callback copies into the caller's array; a read that terminated abnormally copies nothing and is
 * reported by {@link #finish()} or by its event's {@code awaitCompletion()}. Every transfer first
 * retires the oldest reads whose callbacks ran successfully: the queue is in order, so every staging
 * buffer retained up to such a read is free again and every dispatch timed before it has its
 * timestamps, and a run of transfers that never calls {@link #finish()} (chained decode steps that
 * wait only on their reads' events) holds only the buffers of the reads still in flight.
 * {@link #finish()} recycles the rest. Writes are also bounded without reads: a write that would
 * take the staging of the writes in flight above {@link OclPendingWrites#MAX_BYTES} first waits
 * for the oldest of them (and the callbacks of the reads enqueued before it) and recycles their
 * buffers, so a run of uploads that never reads or finishes holds a bounded amount of direct
 * memory. Host arrays must cover the transfer size, and float transfers must be whole floats.</p>
 *
 * <p><b>Errors.</b> Every driver failure is translated by {@link OclErrors}: exhausted memory becomes
 * {@code GpuOutOfMemoryException}, every other status a {@code GpuBackendException} naming the call.</p>
 *
 * <p><b>Dispatch.</b> A full argument list must match the kernel's {@code CL_KERNEL_NUM_ARGS}, and a
 * work-group may not exceed the kernel's {@code CL_KERNEL_WORK_GROUP_SIZE}; the driver applies the
 * remaining rules (divisibility, device limits).</p>
 *
 * <p><b>Graph capture.</b> On a device that reports {@link DeviceFeature#COMMAND_BUFFER},
 * {@link #beginCapture()} records the queue's dispatches, copies and fills into a
 * {@code cl_command_buffer_khr} through {@link OclCommandBufferApi} instead of executing them, and
 * {@link #endCapture()} returns the {@link OclGraph}. The recorded commands are chained with sync
 * points in capture order. A replay verifies the step body against the capture and updates changed
 * scalar arguments in place when mutable dispatch is available; host transfers, reads, markers and
 * synchronizations are refused while recording.</p>
 */
@NotThreadSafe("OpenCL command queues are not thread-safe for concurrent enqueue operations. "
        + "Caller must synchronize access when using from multiple threads. "
        + "Recommended: create separate queue instances per thread, or use external mutex.")
public final class OclComputeQueue implements ComputeQueue {
    private static final Logger LOGGER = LoggingUtil.getLogger(OclComputeQueue.class);

    /** What the queue records instead of executing: nothing, a capture, or a replay's argument updates. */
    private enum Capture {
        NONE,
        CAPTURING,
        REPLAYING
    }

    private cl_command_queue queue;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final boolean profiled;
    /** Per-command device time; non-null only on a queue created with profiling enabled. */
    private final KernelTiming timing;
    private final ArrayDeque<PendingTiming> pendingTiming = new ArrayDeque<>();
    /** The backend that owns this queue, or null for a standalone queue; its events are tracked there. */
    private final OclComputeBackend owner;
    /** The device's {@code cl_khr_command_buffer} entry points, or null when capture is unsupported. */
    private final OclCommandBufferApi commandBuffers;
    private boolean closed = false;
    /**
     * Reads enqueued since the last finish and not retired yet ({@link #retireCompletedReads}),
     * oldest first; their callbacks run on a driver thread.
     */
    private final ArrayDeque<OclReadCompletion> pendingReads = new ArrayDeque<>();
    /** Dispatch timings tracked since the queue was made ({@link #trackTiming}). */
    private long timingsTracked;
    private final OclStagingPool staging = new OclStagingPool();
    /** Writes whose staging may still be in flight, kept under {@link OclPendingWrites#MAX_BYTES}. */
    private final OclPendingWrites pendingWrites = new OclPendingWrites();

    private Capture capture = Capture.NONE;
    private long captureBuffer;
    private int captureSyncPoint = OclCommandBufferApi.NO_SYNC_POINT;
    private final List<OclGraph.Node> captureNodes = new ArrayList<>();
    /** The kernel-dispatch position the next recorded kernel command takes, reset by {@code beginCapture}. */
    private int captureKernelOrdinal;
    private final List<OclGraph> graphs = new ArrayList<>();
    private OclGraph replayGraph;

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

    /** A caller-owned queue without profiling and without command buffers; {@link #close()} releases it. */
    public OclComputeQueue(cl_command_queue queue, QueueType queueType) {
        this(queue, queueType, false, false, null, null);
    }

    /**
     * @param queue          the OpenCL command queue; released by this object
     * @param queueType      compute or transfer
     * @param ownedByBackend true for the backend's own queues: {@link #close()} then only finishes
     *                       and the backend releases the queue through {@link #closeInternal()}
     * @param profiled       true when the queue was created with {@code CL_QUEUE_PROFILING_ENABLE}
     */
    public OclComputeQueue(cl_command_queue queue, QueueType queueType, boolean ownedByBackend, boolean profiled) {
        this(queue, queueType, ownedByBackend, profiled, null, null);
    }

    /**
     * @param queue          the OpenCL command queue; released by this object
     * @param queueType      compute or transfer
     * @param ownedByBackend true for the backend's own queues: {@link #close()} then only finishes
     *                       and the backend releases the queue through {@link #closeInternal()}
     * @param profiled       true when the queue was created with {@code CL_QUEUE_PROFILING_ENABLE}
     * @param owner          the backend that owns this queue, or null for a standalone queue; every
     *                       event this queue issues is counted open there until its holder closes it
     * @param commandBuffers the device's {@code cl_khr_command_buffer} entry points, or null when the
     *                       device does not support capture
     */
    OclComputeQueue(cl_command_queue queue, QueueType queueType, boolean ownedByBackend, boolean profiled,
                    OclComputeBackend owner, OclCommandBufferApi commandBuffers) {
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
        this.timing = profiled ? new KernelTiming() : null;
        this.owner = owner;
        this.commandBuffers = commandBuffers;
    }

    /** True for the queues returned by {@code computeQueue()} and {@code transferQueue()}. */
    public boolean isOwnedByBackend() {
        return ownedByBackend;
    }

    /** True once the queue has been released ({@link #closeInternal()}). */
    boolean isReleased() {
        return closed;
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
        uncapturable("a write");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = stageWrite(sizeInBytes);
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
        uncapturable("a write");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = stageWrite(sizeInBytes);
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
        uncapturable("a write");
        cl_mem mem = extractClMem(dstBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = stageWrite(sizeInBytes);
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
        uncapturable("a read");
        cl_mem mem = extractClMem(srcBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = stage(sizeInBytes);
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
        uncapturable("a read");
        cl_mem mem = extractClMem(srcBuffer);
        cl_event[] waitList = extractWaitList(waitOnEvents);

        ByteBuffer buf = stage(sizeInBytes);
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
        dispatch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, false);
    }

    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                   Object[] args, GpuEvent... waitOnEvents) {
        return dispatch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, true);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize,
                                       long[] localWorkSize, GpuEvent... waitOnEvents) {
        dispatch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, false);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                    long[] localWorkSize, GpuEvent... waitOnEvents) {
        return dispatch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, true);
    }

    /**
     * One kernel dispatch: launches it, records it into an open capture, or verifies it against the
     * capture of a replay.
     */
    private GpuEvent dispatch(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                              Object[] args, GpuEvent[] waitOnEvents, boolean withEvent) {
        ensureOpen();
        OclKernelHandle handle = prepareDispatch(kernel, globalWorkSize, localWorkSize);
        if (args != null && args.length > 0) {
            setKernelArgs(handle, args);
        }
        boolean waitsOrEvent = withEvent || (waitOnEvents != null && waitOnEvents.length > 0);
        if (capture == Capture.REPLAYING) {
            if (waitsOrEvent) {
                throw new GpuGraphMismatchException("command replay of " + replayGraph + ": "
                        + handle.kernelName() + (withEvent ? " returns an event" : " waits on events")
                        + ", which the captured command sequence does not contain");
            }
            replayGraph.dispatch(handle, globalWorkSize.clone(),
                    localWorkSize == null ? null : localWorkSize.clone());
            return null;
        }
        if (capture == Capture.CAPTURING) {
            if (waitsOrEvent) {
                extractWaitList(waitOnEvents);
                throw new GpuUncapturableCommandException("a " + handle.kernelName() + " dispatch that "
                        + (withEvent ? "returns an event" : "waits on events") + " cannot be captured: enqueue it"
                        + " before beginCapture, or return no event and wait on this queue's order");
            }
            recordKernel(handle, globalWorkSize, localWorkSize);
            return null;
        }
        if (withEvent) {
            cl_event event = new cl_event();
            enqueueClDispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, event);
            if (timing != null) {
                trackTiming(timingKey(handle.kernelName(), globalWorkSize, localWorkSize), event, false);
            }
            return new OclGpuEvent(event, null, profiled, owner);
        }
        if (timing == null) {
            enqueueClDispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, null);
            return null;
        }
        cl_event event = new cl_event();
        enqueueClDispatch(handle, globalWorkSize, localWorkSize, waitOnEvents, event);
        trackTiming(timingKey(handle.kernelName(), globalWorkSize, localWorkSize), event, true);
        return null;
    }

    /** The report key of a dispatch: its kernel name and grid so per-shape times stay separable. */
    private static String timingKey(String name, long[] globalWorkSize, long[] localWorkSize) {
        long local = localWorkSize == null || localWorkSize.length == 0 ? 0 : localWorkSize[0];
        return name + "#" + globalWorkSize[0] + "x" + local;
    }

    /**
     * Queues a dispatch's event for the timing report, which reads its profiling timestamps at the
     * next {@link #finish()}; the queue releases the event unless a returned {@link OclGpuEvent}
     * owns it.
     */
    private void trackTiming(String name, cl_event event, boolean releaseOnDrain) {
        if (timing == null || event == null) {
            return;
        }
        synchronized (pendingTiming) {
            pendingTiming.addLast(new PendingTiming(name, event, releaseOnDrain, timingsTracked++));
        }
    }

    /** Moves every event completed by the finished queue's work into the timing report. */
    private void drainTiming() {
        drainTiming(Long.MAX_VALUE);
    }

    /**
     * Moves the tracked events numbered below {@code mark} into the timing report; the caller knows
     * they completed (a finished queue, or a later read that completed on the in-order queue).
     */
    private void drainTiming(long mark) {
        if (timing == null) {
            return;
        }
        List<PendingTiming> batch = new ArrayList<>();
        synchronized (pendingTiming) {
            while (!pendingTiming.isEmpty() && pendingTiming.peekFirst().sequence() < mark) {
                batch.add(pendingTiming.pollFirst());
            }
        }
        if (batch.isEmpty()) {
            return;
        }
        long[] start = new long[1];
        long[] end = new long[1];
        for (PendingTiming pending : batch) {
            RuntimeException failure = null;
            try {
                OclErrors.check("clGetEventProfilingInfo", "CL_PROFILING_COMMAND_START",
                        () -> clGetEventProfilingInfo(pending.event(), CL_PROFILING_COMMAND_START,
                                Sizeof.cl_ulong, Pointer.to(start), null));
                OclErrors.check("clGetEventProfilingInfo", "CL_PROFILING_COMMAND_END",
                        () -> clGetEventProfilingInfo(pending.event(), CL_PROFILING_COMMAND_END,
                                Sizeof.cl_ulong, Pointer.to(end), null));
                timing.add(pending.name(), end[0] - start[0]);
            } catch (RuntimeException e) {
                failure = e;
            } finally {
                if (pending.releaseOnDrain()) {
                    releaseQuietly(pending.event());
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }

    /**
     * A completed-report entry: the command's name, its event, who releases the event and its number
     * among the queue's tracked timings.
     */
    private static final class PendingTiming {
        private final String name;
        private final cl_event event;
        private final boolean releaseOnDrain;
        private final long sequence;

        PendingTiming(String name, cl_event event, boolean releaseOnDrain, long sequence) {
            this.name = name;
            this.event = event;
            this.releaseOnDrain = releaseOnDrain;
            this.sequence = sequence;
        }

        long sequence() {
            return sequence;
        }

        String name() {
            return name;
        }

        cl_event event() {
            return event;
        }

        boolean releaseOnDrain() {
            return releaseOnDrain;
        }
    }

    /** Records one dispatch of a capture, taking the sync point the next command waits on. */
    private void recordKernel(OclKernelHandle handle, long[] globalWorkSize, long[] localWorkSize) {
        handle.args().requireComplete();
        OclCommandBufferApi.Recorded recorded = commandBuffers.commandNDRangeKernel(captureBuffer,
                queue.getNativePointer(), handle.getCLKernel().getNativePointer(), globalWorkSize.length,
                globalWorkSize, localWorkSize, captureSyncPoint);
        captureSyncPoint = recorded.syncPoint;
        captureNodes.add(OclGraph.Node.kernel(recorded.mutableCommand, captureKernelOrdinal++, handle,
                globalWorkSize.clone(), localWorkSize == null ? null : localWorkSize.clone(),
                OclGraph.snapshot(handle.args())));
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
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device copy");
            replayGraph.copy((OclGpuBuffer) src, srcOffset, (OclGpuBuffer) dst, dstOffset, sizeInBytes);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device copy");
            captureSyncPoint = commandBuffers.commandCopyBuffer(captureBuffer, queue.getNativePointer(),
                    srcMem.getNativePointer(), dstMem.getNativePointer(), srcOffset, dstOffset, sizeInBytes,
                    captureSyncPoint);
            captureNodes.add(OclGraph.Node.copy((OclGpuBuffer) src, srcOffset, (OclGpuBuffer) dst, dstOffset,
                    sizeInBytes));
            return;
        }
        cl_event[] waitList = extractWaitList(waitOnEvents);
        cl_event event = timing != null ? new cl_event() : null;
        OclErrors.check("clEnqueueCopyBuffer", sizeInBytes + " bytes",
                () -> clEnqueueCopyBuffer(queue, srcMem, dstMem, srcOffset, dstOffset, sizeInBytes,
                        waitList.length, waitList.length == 0 ? null : waitList, event));
        trackTiming("memcpy DtoD", event, true);
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
        if (sizeInBytes == 0) {
            return;
        }
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device fill");
            replayGraph.fill((OclGpuBuffer) dst, dstOffset, sizeInBytes, pattern);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device fill");
            captureSyncPoint = commandBuffers.commandFillBuffer(captureBuffer, queue.getNativePointer(),
                    mem.getNativePointer(), pattern, dstOffset, sizeInBytes, captureSyncPoint);
            captureNodes.add(OclGraph.Node.fill((OclGpuBuffer) dst, dstOffset, sizeInBytes, pattern));
            return;
        }
        cl_event[] waitList = extractWaitList(waitOnEvents);
        byte[] patternBytes = {
                (byte) pattern, (byte) (pattern >>> 8), (byte) (pattern >>> 16), (byte) (pattern >>> 24)
        };
        cl_event event = timing != null ? new cl_event() : null;
        OclErrors.check("clEnqueueFillBuffer", sizeInBytes + " bytes",
                () -> clEnqueueFillBuffer(queue, mem, Pointer.to(patternBytes), patternBytes.length,
                        dstOffset, sizeInBytes, waitList.length, waitList.length == 0 ? null : waitList, event));
        trackTiming("memset", event, true);
    }

    /**
     * Enqueues a marker through {@code clEnqueueMarkerWithWaitList}. The queue executes in
     * order, so the marker completes after every command enqueued before it and after the waits.
     */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        ensureOpen();
        uncapturable("a marker");
        cl_event[] waitList = extractWaitList(waitOnEvents);
        cl_event event = new cl_event();
        OclErrors.check("clEnqueueMarkerWithWaitList", null,
                () -> clEnqueueMarkerWithWaitList(queue, waitList.length, waitList.length == 0 ? null : waitList,
                        event));
        return new OclGpuEvent(event, null, profiled, owner);
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
        if (arg instanceof GpuBuffer) {
            extractClMem((GpuBuffer) arg);
        }
        handle.args().set(argIndex, arg);
        applyChangedArgs(handle);
    }

    // ------------------------------------------------------------------------------------------
    // Graph capture
    // ------------------------------------------------------------------------------------------

    @Override
    public boolean supportsGraphCapture() {
        return commandBuffers != null;
    }

    /** True when a replayed dispatch can replace a scalar argument in place (mutable dispatch). */
    boolean supportsMutableDispatch() {
        return commandBuffers != null && commandBuffers.mutableArguments();
    }

    @Override
    public boolean isCapturing() {
        return capture == Capture.CAPTURING;
    }

    /**
     * Starts recording this queue's commands into a {@code cl_command_buffer_khr}. Kernel
     * dispatches, device copies and fills recorded from here on execute nothing until the
     * resulting graph is launched; host transfers, reads, markers and synchronizations are
     * refused while recording.
     *
     * @throws UnsupportedOperationException when the device does not support command buffers
     * @throws IllegalStateException         when the queue is closed, already capturing or replaying
     * @throws RuntimeException              the translated driver failure
     */
    @Override
    public void beginCapture() {
        enter();
        requireCaptureSupport();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue is already " + (capture == Capture.CAPTURING ? "capturing" : "replaying")
                    + " commands");
        }
        captureBuffer = commandBuffers.createCommandBuffer(queue.getNativePointer(),
                commandBuffers.simultaneousUse(), commandBuffers.mutableArguments());
        captureSyncPoint = OclCommandBufferApi.NO_SYNC_POINT;
        captureNodes.clear();
        captureKernelOrdinal = 0;
        capture = Capture.CAPTURING;
    }

    /** Refuses a capture call, naming the missing extension, on a queue without command buffers. */
    private void requireCaptureSupport() {
        if (commandBuffers == null) {
            throw new UnsupportedOperationException("OpenCL "
                    + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue does not support graph capture: the device does not support "
                    + "cl_khr_command_buffer");
        }
    }

    /**
     * Finalizes the recorded command buffer and returns its graph. The commands do not run until
     * the graph is launched. An empty capture returns null after releasing the buffer.
     *
     * @throws IllegalStateException when the queue is closed or not capturing
     * @throws RuntimeException      the translated driver failure; a failed capture is discarded
     */
    @Override
    public GpuGraph endCapture() {
        enter();
        requireCaptureSupport();
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue: endCapture needs a capture, a replay of " + replayGraph + " is recording");
        }
        if (capture != Capture.CAPTURING) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue is not capturing a graph");
        }
        List<OclGraph.Node> nodes = new ArrayList<>(captureNodes);
        long commandBuffer = captureBuffer;
        stopRecording();
        if (nodes.isEmpty()) {
            commandBuffers.releaseCommandBuffer(commandBuffer);
            return null;
        }
        try {
            commandBuffers.finalizeCommandBuffer(commandBuffer);
        } catch (RuntimeException e) {
            try {
                commandBuffers.releaseCommandBuffer(commandBuffer);
            } catch (RuntimeException release) {
                e.addSuppressed(release);
            }
            throw e;
        }
        OclGraph graph = new OclGraph(this, commandBuffers, commandBuffer, commandBuffers.simultaneousUse(), nodes);
        graphs.add(graph);
        return graph;
    }

    /**
     * Ends an open capture and discards the recorded commands. Does nothing when the queue is not
     * capturing, so a failure handler can call it unconditionally.
     *
     * @throws IllegalStateException when a replay is recording
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public void abortCapture() {
        enter();
        requireCaptureSupport();
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue: abortCapture needs a capture, a replay of " + replayGraph + " is recording");
        }
        if (capture != Capture.CAPTURING) {
            return;
        }
        long commandBuffer = captureBuffer;
        stopRecording();
        commandBuffers.releaseCommandBuffer(commandBuffer);
    }

    /** Clears the recording state and returns the command buffer it held in {@link #captureBuffer}. */
    private void stopRecording() {
        capture = Capture.NONE;
        captureBuffer = 0;
        captureSyncPoint = OclCommandBufferApi.NO_SYNC_POINT;
        captureNodes.clear();
    }

    /**
     * Refuses a command the recorded sequence cannot hold: during a capture the command is refused
     * before the driver sees it, and during a replay it reports the mismatch the caller
     * re-captures from.
     */
    private void uncapturable(String op) {
        if (capture == Capture.NONE) {
            return;
        }
        if (capture == Capture.REPLAYING) {
            throw new GpuGraphMismatchException("command replay of " + replayGraph + " hit " + op
                    + ", which the captured commands do not contain");
        }
        throw new GpuUncapturableCommandException("a " + op + " cannot be captured: perform host transfers and"
                + " reads before beginCapture, and read results only after the graph has been launched");
    }

    /** A recorded copy or fill carries no events, so a wait cannot be part of the capture. */
    private void requireNoWaitsForCapture(GpuEvent[] waits, String what) {
        if (waits != null && waits.length > 0) {
            extractWaitList(waits);
            throw new GpuUncapturableCommandException("a " + what + " that waits on events cannot be captured:"
                    + " enqueue it"
                    + " before beginCapture, or wait on this queue's order");
        }
    }

    /** A replayed command must match the captured sequence, which carries no events. */
    private void requireNoWaitsForReplay(GpuEvent[] waits, String what) {
        if (waits != null && waits.length > 0) {
            throw new GpuGraphMismatchException("command replay of " + replayGraph + ": " + what
                    + " waits on events, which the captured command sequence does not contain");
        }
    }

    /**
     * A replay's step body: commands verify against the capture instead of executing.
     *
     * @throws IllegalStateException when the queue is closed, another recording is active or the
     *                               graph does not belong to this queue
     */
    void beginGraphReplay(OclGraph graph) {
        enter();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue is already " + (capture == Capture.CAPTURING ? "capturing" : "replaying")
                    + " commands");
        }
        if (graph.isClosed()) {
            throw new IllegalStateException(graph + " is closed");
        }
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
        replayGraph = graph;
        capture = Capture.REPLAYING;
    }

    /** The replay submission: mutable argument updates, then one enqueue of the command buffer. */
    void submitGraphReplay(OclGraph graph) {
        enter();
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue: " + graph + " is not replaying");
        }
        graph.applyUpdates();
        enqueueCommandBuffer(graph, EMPTY_CL_EVENTS);
    }

    /** Ends a replay's step body: a failed body leaves the device untouched. */
    void endGraphReplay(OclGraph graph, boolean submitted) {
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue: " + graph + " is not replaying");
        }
        replayGraph = null;
        capture = Capture.NONE;
        if (!submitted) {
            LOGGER.debug("Command replay of {} did not reach its submission; the caller retries", graph);
        }
    }

    /**
     * One launch of a captured graph on {@code target}, with its recorded commands. The command
     * buffer was created for this queue, so it runs only here.
     *
     * @throws IllegalArgumentException when {@code target} is another queue
     * @throws IllegalStateException    when the queue or graph is closed, or another recording is active
     */
    GpuEvent launchGraph(OclGraph graph, ComputeQueue target, GpuEvent[] waitOnEvents) {
        if (target != this) {
            throw new IllegalArgumentException("OpenCL graph " + graph + " runs only on the queue that captured it,"
                    + " not on " + (target == null ? "null" : target.toString()));
        }
        enter();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("OpenCL " + queueType.name().toLowerCase(java.util.Locale.ROOT)
                    + " queue is already " + (capture == Capture.CAPTURING ? "capturing" : "replaying")
                    + " commands");
        }
        if (graph.isClosed()) {
            throw new IllegalStateException(graph + " is closed");
        }
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
        cl_event[] waitList = extractWaitList(waitOnEvents);
        enqueueCommandBuffer(graph, waitList);
        // the command buffer carries no event of its own; a read or finish on this in-order queue
        // observes the submission
        return null;
    }

    /** Drains the queue so the command buffer is not pending when its mutable commands are updated. */
    void drainForGraphUpdate() {
        ensureOpen();
        OclErrors.checkWait("clFinish", null, () -> clFinish(queue));
    }

    /** Enqueues the graph's command buffer after {@code waitList} (which the caller validated). */
    private void enqueueCommandBuffer(OclGraph graph, cl_event[] waitList) {
        long[] waits = new long[waitList.length];
        for (int i = 0; i < waitList.length; i++) {
            waits[i] = waitList[i].getNativePointer();
        }
        if (!graph.simultaneousUse()) {
            // a command buffer without simultaneous use cannot be enqueued while it is pending
            OclErrors.checkWait("clFinish", null, () -> clFinish(queue));
        }
        commandBuffers.enqueueCommandBuffer(graph.commandBuffer(), queue.getNativePointer(), waits);
    }

    /** Drains the queue, then releases a graph's command buffer. */
    void releaseGraph(OclGraph graph) {
        enter();
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
        RuntimeException failure = null;
        try {
            OclErrors.checkWait("clFinish", null, () -> clFinish(queue));
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            graph.destroy();
        } catch (RuntimeException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        graphs.remove(graph);
        if (failure != null) {
            throw failure;
        }
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
        uncapturable("a queue synchronization");
        OclErrors.checkWait("clFinish", null, () -> clFinish(queue));
        drainTiming();

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
                    for (int r = reads.size() - 1; r >= 0; r--) {
                        pendingReads.addFirst(reads.get(r));
                    }
                }
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for OpenCL read callbacks", e);
            }
        }
        pendingWrites.releaseAll();
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
            discardRecording();
            try {
                // clReleaseCommandQueue only decrements the reference count and does not wait for
                // pending commands, so the queue is finished first.
                finish();
            } finally {
                releaseGraphs();
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
                pendingWrites.releaseAll();
                staging.clear();
            }
        }
    }

    /** Drops an open capture and replay state so a queue being closed can finish and release. */
    private void discardRecording() {
        long commandBuffer = captureBuffer;
        boolean hadCapture = capture == Capture.CAPTURING;
        capture = Capture.NONE;
        replayGraph = null;
        captureBuffer = 0;
        captureSyncPoint = OclCommandBufferApi.NO_SYNC_POINT;
        captureNodes.clear();
        if (hadCapture && commandBuffer != 0) {
            try {
                commandBuffers.releaseCommandBuffer(commandBuffer);
            } catch (RuntimeException e) {
                LOGGER.warn("Releasing the OpenCL {} queue's open command buffer failed: {}", queueType,
                        e.toString());
            }
        }
    }

    /** Releases every graph's command buffer; the queue has drained. */
    private void releaseGraphs() {
        for (OclGraph graph : new ArrayList<>(graphs)) {
            try {
                graph.destroy();
            } catch (RuntimeException e) {
                LOGGER.warn("Releasing an OpenCL command buffer of the {} queue failed: {}", queueType,
                        e.toString());
            }
        }
        graphs.clear();
    }

    public cl_command_queue getCLQueue() {
        return queue;
    }

    @Override
    public KernelTiming kernelTiming() {
        return timing;
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
        long ticket = staging.retain(buf);
        try {
            OclErrors.check("clRetainEvent", "write of " + sizeInBytes + " bytes", () -> clRetainEvent(event));
        } catch (RuntimeException e) {
            releaseQuietly(event);
            throw e;
        }
        pendingWrites.add(event, ticket, buf.capacity());
        return new OclGpuEvent(event, null, profiled, owner);
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
        // The driver may write the staging buffer until the read completes; it is recycled once this
        // read or a later one completed, or by finish().
        long ticket = staging.retain(buf);

        OclReadCompletion completion = new OclReadCompletion(sizeInBytes, ticket, timingsTracked());
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
        return new OclGpuEvent(event, completion, profiled, owner);
    }

    /**
     * A staging buffer of at least {@code sizeInBytes} for a transfer, after retiring the reads that
     * completed so their buffers can serve it.
     */
    private ByteBuffer stage(long sizeInBytes) {
        retireCompletedReads();
        return staging.acquire((int) sizeInBytes);
    }

    /**
     * A staging buffer for a write: as {@link #stage}, after waiting for the oldest writes in flight while this one
     * would take their staging above {@link OclPendingWrites#MAX_BYTES}, so their buffers can serve it.
     */
    private ByteBuffer stageWrite(long sizeInBytes) {
        retireCompletedReads();
        long capacity = OclStagingPool.capacityFor((int) sizeInBytes);
        while (pendingWrites.overBudgetWith(capacity)) {
            awaitOldestWrite();
        }
        return staging.acquire((int) sizeInBytes);
    }

    /**
     * Waits for the oldest write in flight and recycles the staging buffers retained up to it. The queue runs in
     * order, so every command enqueued before that write completed too; the reads among them have their callbacks
     * awaited first, because a callback still copies out of its staging buffer.
     *
     * @throws com.eonmux.neuralfree.acceleration.GpuBackendException when the write or a command before it failed
     */
    private void awaitOldestWrite() {
        OclPendingWrites.Write oldest = pendingWrites.pollOldest();
        cl_event[] waitList = {oldest.event};
        try {
            OclErrors.checkWait("clWaitForEvents", "write of " + oldest.stagingBytes + " staged bytes",
                    () -> clWaitForEvents(1, waitList));
        } finally {
            OclPendingWrites.release(oldest.event);
        }
        awaitReadCallbacksBefore(oldest.ticket);
        staging.recycleThrough(oldest.ticket);
        retireCompletedReads();
    }

    /** Waits until the callback of every pending read retained before {@code ticket} has run. */
    private void awaitReadCallbacksBefore(long ticket) {
        List<OclReadCompletion> reads;
        synchronized (pendingReads) {
            reads = new ArrayList<>(pendingReads);
        }
        for (OclReadCompletion read : reads) {
            if (read.stagingTicket() >= ticket) {
                return;
            }
            try {
                read.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for OpenCL read callbacks", e);
            }
        }
    }

    /**
     * Retires the oldest pending reads whose callbacks ran for a read that completed normally: the
     * queue runs in order, so every command enqueued up to such a read completed, its staging
     * buffers return to the pool and its dispatch timings move to the report. Stops at the first
     * read still in flight or one that failed, which {@link #finish()} or its event reports.
     */
    private void retireCompletedReads() {
        OclReadCompletion through = null;
        synchronized (pendingReads) {
            while (!pendingReads.isEmpty() && pendingReads.peekFirst().completedNormally()) {
                through = pendingReads.pollFirst();
            }
        }
        if (through != null) {
            staging.recycleThrough(through.stagingTicket());
            pendingWrites.releaseThrough(through.stagingTicket());
            drainTiming(through.timingMark());
        }
    }

    /** The dispatch timings tracked so far, the mark a read enqueued now records. */
    private long timingsTracked() {
        synchronized (pendingTiming) {
            return timingsTracked;
        }
    }

    /** Staging buffers the driver may still access, for diagnostics and tests. */
    public int stagingBuffersInFlight() {
        return staging.inFlightCount();
    }

    /** Staging bytes of the writes that may still be in flight, for diagnostics and tests. */
    public long writeStagingBytesInFlight() {
        return pendingWrites.bytes();
    }

    /** Reads not yet retired by a later transfer or a finish, for diagnostics and tests. */
    public int pendingReadCount() {
        synchronized (pendingReads) {
            return pendingReads.size();
        }
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

    private void enqueueClDispatch(OclKernelHandle handle, long[] globalWorkSize, long[] localWorkSize,
                                   GpuEvent[] waitOnEvents, cl_event event) {
        cl_kernel clKernel = handle.getCLKernel();
        cl_event[] waitList = extractWaitList(waitOnEvents);
        OclErrors.check("clEnqueueNDRangeKernel", handle.kernelName(),
                () -> clEnqueueNDRangeKernel(queue, clKernel, globalWorkSize.length, null, globalWorkSize,
                        localWorkSize, waitList.length, waitList.length == 0 ? null : waitList, event));
    }

    /** Checks the queue is open before a public entry point acts. */
    private void enter() {
        ensureOpen();
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
     * Sets every argument when {@code args} is non-empty; null or empty keeps the stored arguments. Only the
     * arguments whose values differ from the ones the kernel object holds reach {@code clSetKernelArg}
     * ({@link #applyChangedArgs}); a buffer argument is checked to be open either way.
     *
     * @throws IllegalArgumentException when the count differs from the kernel's argument count, or an
     *                                  argument is null or unsupported
     * @throws IllegalStateException    when a buffer argument is closed
     */
    private void setKernelArgs(OclKernelHandle handle, Object[] args) {
        if (args == null || args.length == 0) {
            return;
        }
        handle.checkArgCount(args.length);
        for (int i = 0; i < args.length; i++) {
            if (args[i] instanceof GpuBuffer) {
                extractClMem((GpuBuffer) args[i]);
            }
        }
        // The sticky store validates the whole list before any argument reaches the driver.
        handle.args().setAll(args);
        applyChangedArgs(handle);
    }

    /**
     * Passes every argument stored since the kernel object last received it to {@code clSetKernelArg}. The kernel
     * object keeps an argument until it is set again, so an argument whose value did not change is not sent; one
     * the driver refused stays pending and is sent again by the next call.
     */
    private void applyChangedArgs(OclKernelHandle handle) {
        StickyArgs args = handle.args();
        BitSet changed = args.dirty();
        for (int i = changed.nextSetBit(0); i >= 0; i = changed.nextSetBit(i + 1)) {
            setClKernelArg(handle, i, args.get(i));
            args.clearDirty(i);
        }
    }

    private void setClKernelArg(OclKernelHandle handle, int index, Object arg) {
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

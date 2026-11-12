package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuUncapturableCommandException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTaskBatch;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.backend.LaunchGeometry;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUgraph;
import jcuda.driver.CUgraphExec;
import jcuda.driver.CUgraphNode;
import jcuda.driver.CUDA_KERNEL_NODE_PARAMS;
import jcuda.driver.CUDA_MEMCPY3D;
import jcuda.driver.CUDA_MEMSET_NODE_PARAMS;
import jcuda.driver.CUstream;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static jcuda.driver.JCudaDriver.cuCtxEnablePeerAccess;
import static jcuda.driver.JCudaDriver.cuDeviceCanAccessPeer;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventQuery;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuGraphAddKernelNode;
import static jcuda.driver.JCudaDriver.cuGraphAddMemcpyNode;
import static jcuda.driver.JCudaDriver.cuGraphAddMemsetNode;
import static jcuda.driver.JCudaDriver.cuGraphCreate;
import static jcuda.driver.JCudaDriver.cuGraphDestroy;
import static jcuda.driver.JCudaDriver.cuGraphInstantiate;
import static jcuda.driver.JCudaDriver.cuGraphLaunch;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoDAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyPeerAsync;
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
 *   <li><b>Graph capture.</b> {@code beginCapture()} records the queue's kernel dispatches, device
 *       copies and fills into an explicit CUDA graph instead of executing them ({@code cuGraphAdd*}
 *       chained in stream order); {@code endCapture()} instantiates it into a replayable
 *       {@link CudaGraph}. The step that captured it submits it once with
 *       {@link GpuGraph#launch(ComputeQueue, GpuEvent...)}, and a replay runs the launching code again,
 *       updates the nodes' parameters and submits once with {@code cuGraphLaunch}. Host transfers,
 *       reads, markers, synchronizations and waits on other queues' events are rejected while
 *       capturing, and no buffer fence is taken on the capturing stream: it executes nothing until the
 *       graph is launched. The captured commands replay with the arguments and addresses they were
 *       captured with, so a caller keeps every value that changes between replays in device memory.</li>
 *   <li><b>Ownership.</b> {@code close()} of a backend-owned queue only finishes it (and does
 *       nothing once the backend is closed); the backend releases it with {@link #closeInternal()}.
 *       A caller-owned queue is released by {@code close()}, or by the backend's close when it is
 *       still open then; closing it afterwards does nothing.</li>
 *   <li><b>Buffer fences.</b> The backend's buffer pool records a point on the stream of every
 *       open queue when a buffer is released ({@link #recordFence()}), from any thread.</li>
 *   <li><b>Kernel batching.</b> {@code beginKernelBatch()} returns a {@link CudaKernelBatch}:
 *       independent small kernels recorded with {@code add} and submitted as one launch of a captured
 *       graph.</li>
 * </ul>
 */
@NotThreadSafe("One thread enqueues at a time. Events of this queue may be awaited from other threads; "
        + "read completion and staging memory are guarded by an internal lock.")
public final class CudaComputeQueue implements ComputeQueue {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaComputeQueue.class);

    /** Largest block z dimension of every CUDA device. */
    private static final long MAX_BLOCK_DIM_Z = 64;

    /** Pending timing pairs converted per dispatch, bounding the deque between finishes. */
    private static final int TIMING_DRAIN_BATCH = 64;

    private final CudaComputeBackend backend;
    private final CudaContext context;
    private final CudaEventPool events;
    private final CudaEventPool timingPool;
    private final KernelTiming timing;
    private final ArrayDeque<PendingTiming> pendingTiming = new ArrayDeque<>();
    private final CUstream stream;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final Object fenceLock = new Object();
    private final CudaHostTransfers transfers;
    private final AtomicLong sequence = new AtomicLong();
    private final List<CudaGraph> graphs = new ArrayList<>();
    private volatile boolean closed;
    /** Guarded by {@link #fenceLock}: the stream takes buffer fences until it has been drained on release. */
    private boolean fenceable = true;
    /** What the queue records instead of launching: nothing, a capture, or a replay's node updates. */
    private Capture capture = Capture.NONE;
    private CUgraph captureGraph;
    private CUgraphNode captureLastNode;
    private final List<CudaGraph.Node> captureNodes = new ArrayList<>();
    /** The kernel-dispatch position the next captured kernel node takes, reset by {@code beginCapture}. */
    private int captureKernelOrdinal;
    private CudaGraph replayGraph;

    /** The command recording state between {@code beginCapture} and the replay launch. */
    private enum Capture {
        /** Dispatches launch as they are enqueued. */
        NONE,
        /** Dispatches record into the capture graph without executing. */
        CAPTURING,
        /** Dispatches update the replay graph's node parameters without executing. */
        REPLAYING
    }

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
            CudaEventPool timed = backend.timingEvents();
            this.timingPool = timed;
            this.timing = timed != null ? new KernelTiming() : null;
            this.stream = stream;
            this.queueType = type;
            this.ownedByBackend = ownedByBackend;
            this.transfers = new CudaHostTransfers(this);
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

    /** The backend that owns this queue. */
    CudaComputeBackend backendOf() {
        return backend;
    }

    /** The CUDA context the queue's stream and staging rings live in. */
    CudaContext context() {
        return context;
    }

    /** Marks the queue closed; called by {@link CudaHostTransfers} under the transfer lock. */
    void markClosed() {
        closed = true;
    }

    /**
     * The wait events of a graph or batch launch on this queue, checked and typed.
     *
     * @throws IllegalArgumentException for an event that is null, of another backend, or closed
     */
    CudaGpuEvent[] validatedWaits(GpuEvent[] waits) {
        validateWaits(waits);
        if (waits == null || waits.length == 0) {
            return new CudaGpuEvent[0];
        }
        CudaGpuEvent[] typed = new CudaGpuEvent[waits.length];
        for (int i = 0; i < waits.length; i++) {
            typed[i] = (CudaGpuEvent) waits[i];
        }
        return typed;
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
        CudaBuffer from = ownCudaBuffer(src, "copy source");
        CudaBuffer to = ownCudaBuffer(dst, "copy destination");
        checkDeviceRange(src, srcOffset, sizeInBytes, "copy source");
        checkDeviceRange(dst, dstOffset, sizeInBytes, "copy destination");
        if (src == dst && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        validateWaits(waitOnEvents);
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device copy");
            replayGraph.copy(from, srcOffset, to, dstOffset, sizeInBytes);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device copy");
            CUDA_MEMCPY3D params = CudaGraph.memcpyParams(from, srcOffset, to, dstOffset, sizeInBytes);
            CUgraphNode node = addNode("cuGraphAddMemcpyNode",
                    (n, deps, depCount) -> cuGraphAddMemcpyNode(n, captureGraph, deps, depCount, params,
                            context.context()));
            captureNodes.add(CudaGraph.Node.copy(node, from, srcOffset, to, dstOffset, sizeInBytes));
            return;
        }
        applyWaits(waitOnEvents);
        CUevent timedStart = beginTiming("memcpy DtoD");
        try {
            CudaErrors.check("cuMemcpyDtoDAsync", () -> cuMemcpyDtoDAsync(to.getCUDevicePtr().withByteOffset(dstOffset),
                    from.getCUDevicePtr().withByteOffset(srcOffset), sizeInBytes, stream));
        } catch (RuntimeException e) {
            cancelTiming(timedStart);
            throw e;
        }
        endTiming(timedStart, "memcpy DtoD");
    }

    /**
     * Whether this device can read {@code peer}'s device memory directly: the same physical device always, another
     * device when the driver reports peer access.
     */
    @Override
    public boolean supportsPeerCopy(ComputeQueue peer) {
        if (!(peer instanceof CudaComputeQueue)) {
            return false;
        }
        CudaComputeQueue other = (CudaComputeQueue) peer;
        if (other == this || other.backend == backend) {
            return true;
        }
        try {
            int[] canAccess = {0};
            CudaErrors.check("cuDeviceCanAccessPeer",
                    () -> cuDeviceCanAccessPeer(canAccess, backend.getCUDevice().getDeviceId(),
                            other.backend.getCUDevice().getDeviceId()));
            return canAccess[0] != 0;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * Copies a buffer of this device into a buffer of another CUDA device: a device-to-device copy when both queues
     * share the device, else {@code cuMemcpyPeerAsync} on this queue after enabling peer access to the destination's
     * device. Running on this queue orders the copy between the work already enqueued here and later work, so a
     * source buffer that later steps overwrite is captured correctly.
     *
     * @throws UnsupportedOperationException under a graph capture (a cross-device copy is not recorded) or when the
     *                                       driver cannot copy between the two devices
     */
    @Override
    public void enqueuePeerCopy(ComputeQueue destination, GpuBuffer src, long srcOffset, GpuBuffer dst,
                                long dstOffset, long sizeInBytes, GpuEvent... waitOnEvents) {
        enter();
        if (!(destination instanceof CudaComputeQueue)) {
            throw new UnsupportedOperationException("peer copy needs a CUDA destination queue, got "
                    + (destination == null ? "null" : destination.getClass().getName()));
        }
        CudaComputeQueue to = (CudaComputeQueue) destination;
        CudaBuffer srcBuffer = foreignCudaBuffer(src, "peer copy source");
        CudaBuffer dstBuffer = foreignCudaBuffer(dst, "peer copy destination");
        checkDeviceRange(src, srcOffset, sizeInBytes, "peer copy source");
        checkDeviceRange(dst, dstOffset, sizeInBytes, "peer copy destination");
        validateWaits(waitOnEvents);
        if (capture == Capture.CAPTURING || capture == Capture.REPLAYING) {
            throw new UnsupportedOperationException("a cross-device peer copy is not recorded in a graph capture");
        }
        applyWaits(waitOnEvents);
        CUdeviceptr srcPtr = srcBuffer.getCUDevicePtr().withByteOffset(srcOffset);
        CUdeviceptr dstPtr = dstBuffer.getCUDevicePtr().withByteOffset(dstOffset);
        if (to.backend.getCUDevice().getDeviceId().equals(backend.getCUDevice().getDeviceId())) {
            CUevent timedStart = beginTiming("memcpy DtoD");
            try {
                CudaErrors.check("cuMemcpyDtoDAsync",
                        () -> cuMemcpyDtoDAsync(dstPtr, srcPtr, sizeInBytes, stream));
            } catch (RuntimeException e) {
                cancelTiming(timedStart);
                throw e;
            }
            endTiming(timedStart, "memcpy DtoD");
            return;
        }
        int enabled = cuCtxEnablePeerAccess(to.context.context(), 0);
        if (enabled != CudaErrors.SUCCESS
                && enabled != jcuda.driver.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
            CudaErrors.call(enabled, "cuCtxEnablePeerAccess");
        }
        CUevent timedStart = beginTiming("memcpy peer");
        try {
            CudaErrors.check("cuMemcpyPeerAsync", () -> cuMemcpyPeerAsync(dstPtr, to.context.context(), srcPtr,
                    context.context(), sizeInBytes, stream));
        } catch (RuntimeException e) {
            cancelTiming(timedStart);
            throw e;
        }
        endTiming(timedStart, "memcpy peer");
    }

    /** A CUDA buffer of any CUDA backend, checked for open state only. */
    private static CudaBuffer foreignCudaBuffer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof CudaBuffer)) {
            throw new IllegalArgumentException(role + " must be a CUDA buffer, got "
                    + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        CudaBuffer cuda = (CudaBuffer) buffer;
        if (cuda.getCUDevicePtr() == null) {
            throw new IllegalArgumentException(role + " was already closed");
        }
        return cuda;
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
        CudaBuffer target = ownCudaBuffer(dst, "fill destination");
        validateWaits(waitOnEvents);
        if (sizeInBytes == 0) {
            return;
        }
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device fill");
            replayGraph.fill(target, dstOffset, sizeInBytes, pattern);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device fill");
            CUDA_MEMSET_NODE_PARAMS params = CudaGraph.memsetParams(target, dstOffset, sizeInBytes, pattern);
            CUgraphNode node = addNode("cuGraphAddMemsetNode",
                    (n, deps, depCount) -> cuGraphAddMemsetNode(n, captureGraph, deps, depCount, params,
                            context.context()));
            captureNodes.add(CudaGraph.Node.fill(node, target, dstOffset, sizeInBytes, pattern));
            return;
        }
        applyWaits(waitOnEvents);
        long words = sizeInBytes / 4;
        CUevent timedStart = beginTiming("memset");
        try {
            CudaErrors.check("cuMemsetD32Async", () -> cuMemsetD32Async(
                    target.getCUDevicePtr().withByteOffset(dstOffset), pattern, words, stream));
        } catch (RuntimeException e) {
            cancelTiming(timedStart);
            throw e;
        }
        endTiming(timedStart, "memset");
    }

    /** An event that completes once everything enqueued so far, and the waits, has completed. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        enter();
        requireNotCapturing("a marker");
        uncapturable("a marker");
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
    // Graph capture and kernel batching
    // ------------------------------------------------------------------------------------------

    @Override
    public boolean supportsGraphCapture() {
        return true;
    }

    /**
     * Starts capturing this queue's commands into a CUDA graph. The stream is drained first, so the
     * capture covers only the commands enqueued from here on. While capturing, kernel dispatches,
     * device copies and fills are recorded and not executed; host transfers, reads, markers,
     * synchronizations and waits on events are rejected with {@link IllegalStateException}, because
     * none of them can be captured.
     *
     * @throws IllegalStateException when the queue is closed, already capturing, or replaying a graph
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public void beginCapture() {
        enter();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("CUDA " + queueType + " queue is already "
                    + (capture == Capture.CAPTURING ? "capturing" : "replaying") + " commands");
        }
        CUgraph graph = new CUgraph();
        CudaErrors.check("cuGraphCreate", () -> cuGraphCreate(graph, 0));
        captureGraph = graph;
        captureLastNode = null;
        captureNodes.clear();
        captureKernelOrdinal = 0;
        capture = Capture.CAPTURING;
    }

    /**
     * Ends the capture and instantiates the captured command sequence as a {@link CudaGraph}. The
     * commands do not run until the graph is launched. The queue stops capturing whether or not the
     * instantiation succeeds; an empty capture returns null.
     *
     * @throws IllegalStateException when the queue is closed or not capturing
     * @throws RuntimeException      the translated driver failure; a failed capture is discarded
     */
    @Override
    public GpuGraph endCapture() {
        enter();
        List<CudaGraph.Node> nodes;
        CUgraph graph;
        if (capture == Capture.NONE) {
            throw new IllegalStateException("CUDA " + queueType + " queue is not capturing a graph");
        }
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("CUDA " + queueType + " queue: endCapture needs a capture, a replay of "
                    + replayGraph + " is recording");
        }
        nodes = new ArrayList<>(captureNodes);
        graph = captureGraph;
        stopRecording();
        if (nodes.isEmpty()) {
            destroyGraph(graph);
            return null;
        }
        CUgraphExec exec = new CUgraphExec();
        CUgraphExec nextExec = new CUgraphExec();
        try {
            CudaErrors.check("cuGraphInstantiate", () -> cuGraphInstantiate(exec, graph, 0L));
            CudaErrors.check("cuGraphInstantiate", () -> cuGraphInstantiate(nextExec, graph, 0L));
        } catch (RuntimeException e) {
            destroyGraph(graph);
            throw e;
        }
        CudaGraph captured = new CudaGraph(this, graph, exec, nextExec, nodes);
        graphs.add(captured);
        return captured;
    }

    /**
     * Ends an open capture and discards the recorded commands, so a failed step leaves no device work
     * behind. Does nothing when the queue is not capturing, so a failure handler can call it
     * unconditionally.
     *
     * @throws IllegalStateException when a replay is recording
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public void abortCapture() {
        enter();
        if (capture == Capture.NONE) {
            return;
        }
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("CUDA " + queueType + " queue: abortCapture needs a capture, a replay of "
                    + replayGraph + " is recording");
        }
        CUgraph graph = captureGraph;
        stopRecording();
        destroyGraph(graph);
    }

    @Override
    public boolean isCapturing() {
        return capture == Capture.CAPTURING;
    }

    @Override
    public boolean supportsKernelBatching() {
        return true;
    }

    /** A new caller-owned {@link CudaKernelBatch} on this queue's backend. */
    @Override
    public KernelTaskBatch beginKernelBatch() {
        enter();
        return new CudaKernelBatch(backend);
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
        requireNotCapturing("a stream synchronization");
        uncapturable("a finish");
        CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(stream));
        drainTiming(Integer.MAX_VALUE);
        retireReadsUpTo(Long.MAX_VALUE);
    }

    @Override
    public KernelTiming kernelTiming() {
        return timing;
    }

    /** A timing queue times every graph launch as one command ({@link KernelTiming#graphKey(int)}). */
    @Override
    public boolean timesGraphLaunches() {
        return timing != null;
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
        if (!transfers.beginClose()) {
            return;
        }
        if (capture == Capture.CAPTURING) {
            discardCapture();
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
            drainTiming(Integer.MAX_VALUE);
            drained = true;
        } catch (RuntimeException e) {
            failure = e;
        }
        stopFences();
        failure = transfers.finishClose(drained, failure);
        try {
            destroyGraphs();
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
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
     * @return the event, or null when the stream is drained or capturing: a capturing stream
     *         executes nothing until its graph is launched, so there is no work in flight to fence
     * @throws RuntimeException the translated driver failure
     */
    CudaGpuEvent recordFence() {
        synchronized (fenceLock) {
            if (!fenceable || capture == Capture.CAPTURING) {
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
        transfers.retire(upTo);
    }

    // ------------------------------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------------------------------

    private CudaGpuEvent launch(KernelHandle kernel, long[] global, long[] local, Object[] args,
                                GpuEvent[] waits, boolean withEvent) {
        enter();
        CudaKernelHandle handle = ownHandle(kernel);
        String name = handle.kernelName();
        CUfunction function = handle.function();
        LaunchGeometry geometry = gridBlockOf(handle, global, local);
        long[] grid = geometry.grid();
        long[] block = geometry.block();
        validateWaits(waits);
        if (args != null && args.length > 0) {
            handle.args().setAll(args);
        }
        Pointer params = handle.args().kernelParams();
        boolean waitsOrEvent = withEvent || (waits != null && waits.length > 0);
        if (capture == Capture.REPLAYING) {
            if (waitsOrEvent) {
                throw new GpuGraphMismatchException("command replay of " + replayGraph + ": " + name
                        + (withEvent ? " returns an event" : " waits on another queue's events")
                        + ", which the captured command sequence does not contain");
            }
            replayGraph.dispatch(handle, global, grid, block, name);
            return null;
        }
        if (capture == Capture.CAPTURING) {
            if (waitsOrEvent) {
                requireOwnQueueWaits(waits);
                throw new GpuUncapturableCommandException("a " + name + " dispatch that "
                        + (withEvent ? "returns an event" : "waits on events") + " cannot be captured: enqueue it"
                        + " before beginCapture, or return no event and wait on this queue's order");
            }
            CUDA_KERNEL_NODE_PARAMS nodeParams = CudaGraph.nodeParams(function, grid, block,
                    handle.dynamicSharedMemoryBytes(), params);
            CUgraphNode node = addNode("cuGraphAddKernelNode(" + name + ")",
                    (n, deps, depCount) -> cuGraphAddKernelNode(n, captureGraph, deps, depCount, nodeParams));
            captureNodes.add(CudaGraph.Node.kernel(node, captureKernelOrdinal++, handle, global.clone(),
                    grid.clone(), block.clone(), handle.args().snapshotValues(),
                    handle.args().sticky().version()));
            return null;
        }
        applyWaits(waits);
        String timedKey = timing == null ? null : timingKey(name, grid, block);
        CUevent timedStart = beginTiming(timedKey);
        int result;
        try {
            result = cuLaunchKernel(handle.function(), (int) grid[0], (int) grid[1], (int) grid[2],
                    (int) block[0], (int) block[1], (int) block[2], handle.dynamicSharedMemoryBytes(), stream,
                    params, null);
        } catch (CudaException e) {
            cancelTiming(timedStart);
            throw CudaErrors.translate(e, "cuLaunchKernel(" + handle.kernelName() + ")", -1);
        }
        if (result != CudaErrors.SUCCESS) {
            cancelTiming(timedStart);
            throw CudaErrors.translate(result, "cuLaunchKernel(" + handle.kernelName() + ")", -1);
        }
        endTiming(timedStart, timedKey);
        return withEvent ? record() : null;
    }

    /** The report key of a dispatch: its kernel name and grid so per-shape times stay separable. */
    private static String timingKey(String name, long[] grid, long[] block) {
        if (grid.length == 1) {
            return name + "#" + grid[0] + "x" + block[0];
        }
        return name + "#" + grid[0] + "x" + grid[1] + "x" + grid[2] + "b" + block[0];
    }

    // ------------------------------------------------------------------------------------------
    // Command timing (only with -Dneuralfree.gpu.profile=true or step)
    // ------------------------------------------------------------------------------------------

    /**
     * Records the start event of a timed command; the command then runs and
     * {@link #endTiming} records its stop event. Null on a queue without timing.
     *
     * @throws RuntimeException the translated driver failure of the record
     */
    CUevent beginTiming(String name) {
        if (timing == null) {
            return null;
        }
        CUevent start = timingPool.obtain();
        try {
            CudaErrors.check("cuEventRecord", () -> cuEventRecord(start, stream));
        } catch (RuntimeException e) {
            timingPool.recycle(start);
            throw e;
        }
        return start;
    }

    /** Returns a recorded start event of a failed command to the pool without measuring it. */
    void cancelTiming(CUevent start) {
        if (start != null) {
            timingPool.recycle(start);
        }
    }

    /**
     * Records the stop event of a timed command and queues the pair for
     * {@link #drainTiming}: the elapsed device time lands in the timing report once the stop
     * event has completed.
     *
     * @throws RuntimeException the translated driver failure of the record
     */
    void endTiming(CUevent start, String name) {
        if (start == null) {
            return;
        }
        CUevent stop = timingPool.obtain();
        try {
            CudaErrors.check("cuEventRecord", () -> cuEventRecord(stop, stream));
        } catch (RuntimeException e) {
            timingPool.recycle(stop);
            timingPool.recycle(start);
            throw e;
        }
        synchronized (pendingTiming) {
            pendingTiming.addLast(new PendingTiming(name, start, stop));
        }
        drainTiming(TIMING_DRAIN_BATCH);
    }

    /**
     * Moves every completed pending pair at the front of the queue into the timing report, at
     * most {@code max} of them: a completed stop event implies its start completed. Called from
     * {@link #finish()}, where the drained stream completed every pair, and opportunistically
     * from {@link #endTiming}.
     */
    private void drainTiming(int max) {
        if (timing == null) {
            return;
        }
        float[] ms = new float[1];
        synchronized (pendingTiming) {
            while (!pendingTiming.isEmpty() && max > 0) {
                PendingTiming head = pendingTiming.peekFirst();
                try {
                    if (cuEventQuery(head.stop()) != CudaErrors.SUCCESS) {
                        return;
                    }
                } catch (CudaException e) {
                    if (!CudaErrors.isNotReady(e)) {
                        throw CudaErrors.translate(e, "cuEventQuery", -1);
                    }
                    return;
                }
                try {
                    cuEventElapsedTime(ms, head.start(), head.stop());
                } catch (CudaException e) {
                    if (!CudaErrors.isNotReady(e)) {
                        throw e;
                    }
                    return;
                }
                timing.add(head.name(), (long) (ms[0] * 1.0e6f));
                pendingTiming.pollFirst();
                timingPool.recycle(head.start());
                timingPool.recycle(head.stop());
                max--;
            }
        }
    }

    /** A recorded command: its report name and the start and stop events bracketing it. */
    private static final class PendingTiming {
        private final String name;
        private final CUevent start;
        private final CUevent stop;

        PendingTiming(String name, CUevent start, CUevent stop) {
            this.name = name;
            this.start = start;
            this.stop = stop;
        }

        String name() {
            return name;
        }

        CUevent start() {
            return start;
        }

        CUevent stop() {
            return stop;
        }
    }

    /**
     * Validates a dispatch of {@code handle} and converts it to the grid and block
     * {@code cuLaunchKernel} takes: a null local size becomes
     * {@link WorkSizes#defaultLocal(long[], long)}, the block is checked against the function's
     * thread limit and the device's block z limit, the grid against the device limits. The grid and the
     * block are padded to three dimensions with 1. A dispatch that repeats the sizes of the handle's last one
     * reuses its geometry, which was validated when it was derived.
     */
    static LaunchGeometry gridBlockOf(CudaKernelHandle handle, long[] global, long[] local) {
        LaunchGeometry last = handle.lastGeometry();
        if (last != null && last.matches(global, local)) {
            return last;
        }
        String name = handle.kernelName();
        long[] blockRaw = local == null ? WorkSizes.defaultLocal(global, handle.maxThreadsPerBlock()) : local;
        WorkSizes.validateLocal(blockRaw, handle.maxThreadsPerBlock(), name);
        long[] block = WorkSizes.pad3(blockRaw, 1);
        if (block[2] > MAX_BLOCK_DIM_Z) {
            throw new IllegalArgumentException(name + ": block z dimension " + block[2] + " exceeds the CUDA limit of "
                    + MAX_BLOCK_DIM_Z);
        }
        long[] grid = WorkSizes.pad3(WorkSizes.groupCounts(global, blockRaw), 1);
        long[] maxGrid = handle.backend().capabilities().getMaxWorkGroupCount();
        for (int d = 0; d < grid.length; d++) {
            if (grid[d] > maxGrid[d]) {
                throw new IllegalArgumentException(name + ": " + grid[d] + " blocks in dimension " + d
                        + " (global " + global[d] + ", local " + blockRaw[d] + ") exceed the device limit of "
                        + maxGrid[d]);
            }
        }
        LaunchGeometry geometry = new LaunchGeometry(global, local, grid, block);
        handle.lastGeometry(geometry);
        return geometry;
    }

    /** Rejects waits on events of other queues: a captured graph records only this stream's order. */
    private void requireOwnQueueWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (int i = 0; i < waits.length; i++) {
            CudaGpuEvent cuda = (CudaGpuEvent) waits[i];
            if (cuda.owner() != this) {
                throw new GpuUncapturableCommandException("wait event " + i
                        + " was recorded on another queue of the backend; "
                        + "a capture records no cross-queue waits, wait on it before beginCapture or launch the "
                        + "graph on this queue's stream order");
            }
        }
    }

    /** A captured copy or fill carries no events, so a wait cannot be part of the recorded sequence. */
    private void requireNoWaitsForCapture(GpuEvent[] waits, String what) {
        if (waits != null && waits.length > 0) {
            requireOwnQueueWaits(waits);
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
     * Rejects a command whose result the host observes while a capture is recording.
     *
     * @throws IllegalStateException when the queue is capturing a graph
     */
    void requireNotCapturing(String what) {
        if (capture == Capture.CAPTURING) {
            throw new GpuUncapturableCommandException("a " + what + " cannot be captured: perform host transfers"
                    + " and reads before beginCapture, and read results only after the graph has been launched");
        }
    }

    // ------------------------------------------------------------------------------------------
    private void stopRecording() {
        captureGraph = null;
        captureLastNode = null;
        captureNodes.clear();
        capture = Capture.NONE;
    }

    /**
     * Adds one node of a capture after the previous node, so the nodes run in capture order.
     * {@code adder} receives the node and its dependency chain.
     */
    private CUgraphNode addNode(String operation, NodeAdder adder) {
        CUgraphNode node = new CUgraphNode();
        CUgraphNode[] deps = captureLastNode != null ? new CUgraphNode[]{captureLastNode} : null;
        int depCount = deps != null ? 1 : 0;
        CudaErrors.check(operation, () -> adder.add(node, deps, depCount));
        captureLastNode = node;
        return node;
    }

    /** One {@code cuGraphAdd*} of a captured node. */
    @FunctionalInterface
    private interface NodeAdder {
        int add(CUgraphNode node, CUgraphNode[] deps, int depCount);
    }

    private void destroyGraph(CUgraph graph) {
        if (graph == null) {
            return;
        }
        CudaErrors.check("cuGraphDestroy", () -> cuGraphDestroy(graph));
    }

    /**
     * A replay's step body: launches update the executable graph's node parameters instead of executing.
     *
     * @throws IllegalStateException when the queue is closed, another recording is active or the graph
     *                               does not belong to this queue
     */
    void beginGraphReplay(CudaGraph graph) {
        enter();
        ensureLaunchable(graph);
        replayGraph = graph;
        capture = Capture.REPLAYING;
    }

    /** The replay launch: one {@code cuGraphLaunch} of the whole captured sequence on this queue. */
    void launchGraphOnOwner(CudaGraph graph) {
        enter();
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("CUDA " + queueType + " queue: " + graph + " is not replaying");
        }
        graph.launchOn(this);
    }

    /**
     * One launch of a captured graph on {@code target}, with the capture-time commands. The graph may
     * run on any open queue of its backend, because every queue of a CUDA context shares the device.
     *
     * @throws IllegalArgumentException when the target queue is not a CUDA queue of this backend
     * @throws IllegalStateException    when the graph, its capturing queue or the target queue is
     *                                  closed, or the target is recording
     */
    GpuEvent launchGraph(CudaGraph graph, ComputeQueue targetQueue, GpuEvent[] waitOnEvents) {
        CudaComputeQueue target = launchQueueOf(backend, targetQueue);
        if (graph.owner().backendOf() != backend) {
            throw new IllegalArgumentException(graph + " was captured on another backend");
        }
        if (graph.owner().isClosed()) {
            throw new IllegalStateException(graph + " was captured on a closed queue");
        }
        if (graph.isClosed()) {
            throw new IllegalStateException(graph + " is closed");
        }
        target.requireNotRecording("launch a graph");
        CudaGpuEvent[] waits = target.validatedWaits(waitOnEvents);
        target.applyWaits(waits);
        graph.launchOn(target);
        return backend.eventPool().record(target.getCUStream(), target);
    }

    /** True when {@code graph} belongs to this open queue and nothing else is recording. */
    private void ensureLaunchable(CudaGraph graph) {
        requireNotRecording("replay a graph");
        if (graph.isClosed()) {
            throw new IllegalStateException(graph + " is closed");
        }
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
    }

    /** Rejects a capture or replay on this queue while another recording is active. */
    private void requireNotRecording(String what) {
        if (capture != Capture.NONE) {
            throw new IllegalStateException("CUDA " + queueType + " queue is already "
                    + (capture == Capture.CAPTURING ? "capturing" : "replaying") + " commands; it cannot " + what);
        }
    }

    /** Ends a replay's step body: a failed body leaves the device untouched (no launch happened). */
    void endGraphReplay(CudaGraph graph, boolean launched) {
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("CUDA " + queueType + " queue: " + graph + " is not replaying");
        }
        replayGraph = null;
        capture = Capture.NONE;
        if (!launched) {
            LOGGER.debug("Command replay of {} did not reach its launch; the caller retries", graph);
        }
    }

    /**
     * One launch of an executable graph holding {@code dispatches} kernel dispatches on the queue's stream,
     * timed as one command under {@link KernelTiming#graphKey(int)} while profiling is on.
     */
    void graphLaunch(CUgraphExec exec, int dispatches) {
        String key = timing != null ? KernelTiming.graphKey(dispatches) : null;
        CUevent timedStart = beginTiming(key);
        try {
            CudaErrors.check("cuGraphLaunch", () -> cuGraphLaunch(exec, stream));
        } catch (RuntimeException e) {
            cancelTiming(timedStart);
            throw e;
        }
        endTiming(timedStart, key);
    }

    /** Drains the stream, then destroys the executable graph. The queue must be open. */
    void releaseGraph(CudaGraph graph) {
        enter();
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
        RuntimeException failure = null;
        try {
            CudaErrors.check("cuStreamSynchronize", () -> cuStreamSynchronize(stream));
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            graph.destroy();
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
        }
        graphs.remove(graph);
        if (failure != null) {
            throw failure;
        }
    }

    /** Destroys every open executable graph; called by {@link #closeInternal()} after the stream drained. */
    private void destroyGraphs() {
        RuntimeException failure = null;
        for (CudaGraph graph : new ArrayList<>(graphs)) {
            try {
                graph.destroy();
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        graphs.clear();
        if (captureGraph != null) {
            CUgraph graph = captureGraph;
            stopRecording();
            try {
                destroyGraph(graph);
            } catch (RuntimeException e) {
                failure = Closeables.chain(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private GpuEvent write(GpuBuffer dst, long dstOffset, long size, CudaHostTransfers.HostSource source,
                           GpuEvent[] waits) {
        return transfers.write(dst, dstOffset, size, source, waits);
    }

    private GpuEvent read(GpuBuffer src, long srcOffset, long size, CudaHostTransfers.HostSink sink,
                          GpuEvent[] waits) {
        return transfers.read(src, srcOffset, size, sink, waits);
    }

    /** Records an event at the stream's current point; the event retires this queue's pending reads. */
    CudaGpuEvent record() {
        return events.record(stream, this);
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

    /**
     * Refuses a command the captured sequence does not contain while a replay is recording. A caller's
     * command that reaches this method during a replay is not part of the capture, so the replay cannot
     * produce the same device work and the caller captures the step again.
     */
    void uncapturable(String op) {
        if (capture == Capture.NONE) {
            return;
        }
        if (capture == Capture.REPLAYING) {
            throw new GpuGraphMismatchException("command replay of " + replayGraph + " hit " + op
                    + ", which the captured command sequence does not contain");
        }
    }

    /**
     * Checks every wait event before any command is issued.
     *
     * @throws IllegalArgumentException for an event that is null, of another backend, or closed
     */
    void validateWaits(GpuEvent[] waits) {
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
    void applyWaits(GpuEvent[] waits) {
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
     * Ends an open capture so the stream can be synchronized and destroyed, discarding the
     * recorded commands. Called only by {@link #closeInternal()} after the queue was marked
     * closed; failures are logged, the release continues.
     */
    private void discardCapture() {
        CUgraph graph = captureGraph;
        stopRecording();
        try {
            destroyGraph(graph);
        } catch (RuntimeException e) {
            LOGGER.warn("CUDA {} queue closed while capturing; discarding the capture failed: {}", queueType,
                    e.getMessage());
        }
    }

    /**
     * The open queue of {@code backend} a graph or batch launch runs on.
     *
     * @throws IllegalArgumentException when the queue is not a CUDA queue of this backend
     * @throws IllegalStateException    when the queue or the backend is closed
     */
    static CudaComputeQueue launchQueueOf(CudaComputeBackend backend, ComputeQueue queue) {
        if (!(queue instanceof CudaComputeQueue)) {
            throw new IllegalArgumentException("queue must be a CudaComputeQueue, got "
                    + (queue == null ? "null" : queue.getClass().getName()));
        }
        CudaComputeQueue cuda = (CudaComputeQueue) queue;
        if (cuda.isClosed()) {
            throw new IllegalStateException("CUDA queue already closed");
        }
        if (cuda.backendOf() != backend) {
            throw new IllegalArgumentException("CUDA queue belongs to another backend");
        }
        backend.enter();
        return cuda;
    }

    /**
     * Checks this queue and its backend are open and makes the context current.
     *
     * @throws IllegalStateException when either is closed
     */
    void enter() {
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

    CUdeviceptr devicePointer(GpuBuffer buffer, String role) {
        return ownCudaBuffer(buffer, role).getCUDevicePtr();
    }

    /** The CUDA buffer behind {@code buffer}, checked for ownership and open state. */
    private CudaBuffer ownCudaBuffer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof CudaBuffer)) {
            throw new IllegalArgumentException(role + " must be a CUDA buffer, got "
                    + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        CudaBuffer cuda = (CudaBuffer) buffer;
        if (cuda.backend() != backend) {
            throw new IllegalArgumentException(role + " belongs to another CUDA backend");
        }
        if (cuda.getCUDevicePtr() == null) {
            throw new IllegalArgumentException(role + " was already closed");
        }
        return cuda;
    }

    static void checkDeviceRange(GpuBuffer buffer, long offset, long size, String role) {
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

}

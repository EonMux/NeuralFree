package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuUncapturableCommandException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.LaunchGeometry;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One HIP stream of a {@link HipComputeBackend}, implementing the {@link ComputeQueue}
 * contract.
 *
 * <ul>
 *   <li><b>Dispatch.</b> {@code global} counts work-items and {@code local} is the block;
 *       the grid is {@code ceil(global / local)} per dimension ({@link WorkSizes#groupCounts}),
 *       so kernels bounds-check their tail. A null {@code local} uses
 *       {@link WorkSizes#defaultLocal}, which divides {@code global[0]}. Blocks are checked
 *       against the function's work-group limit and grids against the device limits.
 *       Dispatches without an event ({@code NoEvent}, {@code PreBound}) record nothing.</li>
 *   <li><b>Arguments</b> are sticky on the {@link HipKernelHandle}; a null or empty
 *       {@code args} array launches the stored arguments.</li>
 *   <li><b>Writes</b> copy the host data into a page-locked upload ring before returning,
 *       then issue {@code hipMemcpyAsync} from it; transfers larger than a ring slice are
 *       split. The caller may reuse its array or buffer immediately.</li>
 *   <li><b>Reads</b> copy device data into a page-locked readback ring. The destination is
 *       filled, in issue order, by {@link #finish()} or by {@link GpuEvent#awaitCompletion()}
 *       on any event of this queue recorded after the read.</li>
 *   <li><b>Waits</b> on events of other queues of the same backend become
 *       {@code hipStreamWaitEvent}; waits on this queue's own events are implied by stream
 *       order.</li>
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
public final class HipComputeQueue implements ComputeQueue {

    private static final Logger LOGGER = LoggingUtil.getLogger(HipComputeQueue.class);

    private final HipComputeBackend backend;
    private final HipContext context;
    private final HipEventPool events;
    private final long stream;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final long[] maxGrid;
    private final Object transferLock = new Object();
    private final Object fenceLock = new Object();
    private final HipPinnedRing upload;
    private final HipPinnedRing readback;
    private final ArrayDeque<PendingRead> pendingReads = new ArrayDeque<>();
    private final AtomicLong sequence = new AtomicLong();
    private volatile boolean closed;
    /** Guarded by {@link #fenceLock}: the stream takes buffer fences until it has been drained on release. */
    private boolean fenceable = true;

    /** What the queue records instead of launching: nothing, a capture, or a replay's node updates. */
    private Capture capture = Capture.NONE;
    private final List<HipGraph.Node> captureNodes = new ArrayList<>();
    /** The kernel-dispatch position the next captured kernel node takes, reset by {@code beginCapture}. */
    private int captureKernelOrdinal;
    private final List<HipGraph> graphs = new ArrayList<>();
    private HipGraph replayGraph;

    /** The command recording state between {@code beginCapture} and the replay launch. */
    private enum Capture {
        /** Commands launch as they are enqueued. */
        NONE,
        /** Commands record into the stream's capture graph without executing. */
        CAPTURING,
        /** Dispatches update the replay graph's node parameters without executing. */
        REPLAYING
    }

    /**
     * Wraps {@code stream}, which this queue owns and destroys on release (also when this
     * constructor fails). The backend's context, event pool and capabilities must exist.
     *
     * @param backend        the owning backend
     * @param stream         a stream created for the backend's device
     * @param type           compute or transfer
     * @param ownedByBackend true for the backend's own queues, whose {@code close()} only finishes
     * @throws IllegalArgumentException when an argument is null, the backend is not ready, or
     *                                  {@code -Dneuralfree.gpu.stagingMb} is invalid
     */
    public HipComputeQueue(HipComputeBackend backend, long stream, QueueType type, boolean ownedByBackend) {
        if (stream == 0) {
            throw new IllegalArgumentException("HIP stream handle must not be null");
        }
        try {
            if (backend == null) {
                throw new IllegalArgumentException("HIP backend cannot be null");
            }
            if (type == null) {
                throw new IllegalArgumentException("queueType cannot be null");
            }
            if (backend.context() == null || backend.eventPool() == null || backend.capabilities() == null) {
                throw new IllegalArgumentException("HIP backend must create its context, event pool and "
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
            this.upload = new HipPinnedRing(context, backend.driver(), label + " upload",
                    HipPinnedRing.configuredMaxBytes(HipPinnedRing.DEFAULT_UPLOAD_MB));
            this.readback = new HipPinnedRing(context, backend.driver(), label + " readback",
                    HipPinnedRing.configuredMaxBytes(HipPinnedRing.DEFAULT_READBACK_MB));
        } catch (RuntimeException e) {
            try {
                backend.driver().streamDestroy(stream);
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
    public long getHipStream() {
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
     * Copies {@code sizeInBytes} bytes on the device ({@code hipMemcpyAsync} DtoD). Overlapping
     * ranges of one buffer are rejected.
     */
    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                  long sizeInBytes, GpuEvent... waitOnEvents) {
        enter();
        long from = devicePointer(src, "copy source");
        long to = devicePointer(dst, "copy destination");
        checkDeviceRange(src, srcOffset, sizeInBytes, "copy source");
        checkDeviceRange(dst, dstOffset, sizeInBytes, "copy destination");
        if (src == dst && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        validateWaits(waitOnEvents);
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device copy");
            replayGraph.copy((HipGpuBuffer) src, srcOffset, (HipGpuBuffer) dst, dstOffset, sizeInBytes);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device copy");
            backend.driver().memcpyDtoDAsync(to + dstOffset, from + srcOffset, sizeInBytes, stream);
            captureNodes.add(HipGraph.Node.copy((HipGpuBuffer) src, srcOffset, (HipGpuBuffer) dst, dstOffset,
                    sizeInBytes));
            return;
        }
        applyWaits(waitOnEvents);
        backend.driver().memcpyDtoDAsync(to + dstOffset, from + srcOffset, sizeInBytes, stream);
    }

    /** Fills the range with the 32-bit {@code pattern} on the device ({@code hipMemsetD32Async}). */
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
        long ptr = devicePointer(dst, "fill destination");
        validateWaits(waitOnEvents);
        if (sizeInBytes == 0) {
            return;
        }
        long words = sizeInBytes / 4;
        if (capture == Capture.REPLAYING) {
            requireNoWaitsForReplay(waitOnEvents, "a device fill");
            replayGraph.fill((HipGpuBuffer) dst, dstOffset, sizeInBytes, pattern);
            return;
        }
        if (capture == Capture.CAPTURING) {
            requireNoWaitsForCapture(waitOnEvents, "a device fill");
            backend.driver().memsetD32Async(ptr + dstOffset, pattern, words, stream);
            captureNodes.add(HipGraph.Node.fill((HipGpuBuffer) dst, dstOffset, sizeInBytes, pattern));
            return;
        }
        applyWaits(waitOnEvents);
        backend.driver().memsetD32Async(ptr + dstOffset, pattern, words, stream);
    }

    /** An event that completes once everything enqueued so far, and the waits, has completed. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        enter();
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
        HipKernelHandle handle = ownHandle(kernel);
        if (handle.isClosed()) {
            throw new IllegalStateException("HIP kernel " + handle.kernelName() + " was used after it was closed");
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
    // Graph capture
    // ------------------------------------------------------------------------------------------

    @Override
    public boolean supportsGraphCapture() {
        return true;
    }

    @Override
    public boolean isCapturing() {
        return capture == Capture.CAPTURING;
    }

    /**
     * Starts recording this queue's commands into the stream's capture graph. Kernel dispatches,
     * device copies and fills recorded from here on execute nothing until the graph is launched;
     * host transfers, reads, markers and synchronizations are refused while recording.
     *
     * @throws IllegalStateException when the queue is closed, already capturing or replaying
     * @throws RuntimeException      the translated driver failure
     */
    @Override
    public void beginCapture() {
        enter();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("HIP " + queueType + " queue is already "
                    + (capture == Capture.CAPTURING ? "capturing" : "replaying") + " commands");
        }
        backend.driver().streamBeginCapture(stream);
        captureNodes.clear();
        captureKernelOrdinal = 0;
        capture = Capture.CAPTURING;
    }

    /**
     * Ends the capture and instantiates the recorded commands as a {@link HipGraph}. The commands do
     * not run until the graph is launched; an empty capture returns null.
     *
     * @throws IllegalStateException when the queue is closed or not capturing
     * @throws RuntimeException      the translated driver failure; a failed capture is discarded
     */
    @Override
    public GpuGraph endCapture() {
        enter();
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("HIP " + queueType + " queue: endCapture needs a capture, a replay of "
                    + replayGraph + " is recording");
        }
        if (capture != Capture.CAPTURING) {
            throw new IllegalStateException("HIP " + queueType + " queue is not capturing a graph");
        }
        List<HipGraph.Node> nodes = new ArrayList<>(captureNodes);
        stopRecording();
        long graph = backend.driver().streamEndCapture(stream);
        if (nodes.isEmpty()) {
            backend.driver().graphDestroy(graph);
            return null;
        }
        long exec;
        try {
            exec = backend.driver().graphInstantiate(graph);
        } catch (RuntimeException e) {
            backend.driver().graphDestroy(graph);
            throw e;
        }
        try {
            assignNodes(graph, nodes);
        } catch (RuntimeException e) {
            backend.driver().graphExecDestroy(exec);
            backend.driver().graphDestroy(graph);
            throw e;
        }
        HipGraph captured = new HipGraph(this, backend.driver(), graph, exec, nodes);
        graphs.add(captured);
        return captured;
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
        if (capture == Capture.REPLAYING) {
            throw new IllegalStateException("HIP " + queueType + " queue: abortCapture needs a capture, a replay of "
                    + replayGraph + " is recording");
        }
        if (capture != Capture.CAPTURING) {
            return;
        }
        stopRecording();
        discardCaptureGraph(backend.driver().streamEndCapture(stream));
    }

    /** Reads the captured nodes back in dependency order and pairs them with the recorded commands. */
    private void assignNodes(long graph, List<HipGraph.Node> nodes) {
        long[] handles = backend.driver().graphNodesInOrder(graph);
        if (handles.length != nodes.size()) {
            throw new IllegalStateException("the captured HIP graph holds " + handles.length
                    + " nodes but " + nodes.size() + " commands were recorded");
        }
        for (int i = 0; i < handles.length; i++) {
            HipGraph.Node node = nodes.get(i);
            int type = backend.driver().graphNodeType(handles[i]);
            int expected = node.kind == HipGraph.Kind.KERNEL ? com.eonmux.neuralfree.acceleration.hip.HipNative.GRAPH_NODE_KERNEL
                    : node.kind == HipGraph.Kind.COPY ? com.eonmux.neuralfree.acceleration.hip.HipNative.GRAPH_NODE_MEMCPY
                    : com.eonmux.neuralfree.acceleration.hip.HipNative.GRAPH_NODE_MEMSET;
            if (type != expected) {
                throw new IllegalStateException("captured HIP node " + i + " is of type " + type
                        + " but the recorded command is a " + node.kind);
            }
            node.graphNode = handles[i];
        }
    }

    /** Clears the recording state so the next command executes again. */
    private void stopRecording() {
        capture = Capture.NONE;
        captureNodes.clear();
    }

    /** Ends a capture nobody will use and destroys its graph, logging a failure that has no caller. */
    private void discardCaptureGraph(long graph) {
        try {
            backend.driver().graphDestroy(graph);
        } catch (RuntimeException e) {
            LOGGER.warn("Destroying the HIP {} queue's discarded capture graph failed: {}", queueType, e.toString());
        }
    }

    /** Ends an open capture and destroys its graph; used by close. */
    private void discardRecording() {
        if (capture == Capture.CAPTURING) {
            stopRecording();
            try {
                discardCaptureGraph(backend.driver().streamEndCapture(stream));
            } catch (RuntimeException e) {
                LOGGER.warn("Ending the HIP {} queue's open capture failed: {}", queueType, e.toString());
            }
        } else {
            capture = Capture.NONE;
            replayGraph = null;
        }
    }

    /** Refuses a command the capture cannot hold, or reports it as a replay mismatch. */
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

    /** Records capture no cross-queue waits: the captured sequence is one stream's order. */
    private void requireOwnQueueWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        validateWaits(waits);
    }

    /** A recorded copy or fill carries no events, so a wait cannot be part of the capture. */
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

    /** A replay's step body: commands verify against the capture instead of executing. */
    void beginGraphReplay(HipGraph graph) {
        enter();
        if (capture != Capture.NONE) {
            throw new IllegalStateException("HIP " + queueType + " queue is already "
                    + (capture == Capture.CAPTURING ? "capturing" : "replaying") + " commands");
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

    /** The replay's submission: one graph launch on the queue's stream. */
    void launchGraphOnOwner(HipGraph graph) {
        enter();
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("HIP " + queueType + " queue: " + graph + " is not replaying");
        }
        graph.launchOn(this);
    }

    /** Ends a replay's step body: a failed body leaves the device untouched. */
    void endGraphReplay(HipGraph graph, boolean launched) {
        if (replayGraph != graph || capture != Capture.REPLAYING) {
            throw new IllegalStateException("HIP " + queueType + " queue: " + graph + " is not replaying");
        }
        replayGraph = null;
        capture = Capture.NONE;
        if (!launched) {
            LOGGER.debug("Command replay of {} did not reach its launch; the caller retries", graph);
        }
    }

    /**
     * One launch of a captured graph on {@code target}, with its capture-time commands. The graph
     * may run on any open queue of the backend, because every queue of a HIP device shares it.
     *
     * @throws IllegalArgumentException when the target is not a HIP queue of this backend
     * @throws IllegalStateException    when the graph or a queue is closed, or the target is recording
     */
    GpuEvent launchGraph(HipGraph graph, ComputeQueue targetQueue, GpuEvent[] waitOnEvents) {
        if (!(targetQueue instanceof HipComputeQueue)) {
            throw new IllegalArgumentException("HIP graph " + graph + " needs a HIP queue, not "
                    + (targetQueue == null ? "null" : targetQueue.toString()));
        }
        HipComputeQueue target = (HipComputeQueue) targetQueue;
        if (target.backend != backend) {
            throw new IllegalArgumentException(graph + " was captured on another HIP backend");
        }
        if (graph.owner().isClosed()) {
            throw new IllegalStateException(graph + " was captured on a closed queue");
        }
        if (graph.isClosed()) {
            throw new IllegalStateException(graph + " is closed");
        }
        target.enter();
        if (target.capture != Capture.NONE) {
            throw new IllegalStateException("HIP " + target.queueType + " queue is already "
                    + (target.capture == Capture.CAPTURING ? "capturing" : "replaying") + " commands");
        }
        target.validateWaits(waitOnEvents);
        target.applyWaits(waitOnEvents);
        graph.launchOn(target);
        return target.record();
    }

    /** Drains the stream, then destroys the executable graph. The queue must be open. */
    void releaseGraph(HipGraph graph) {
        enter();
        if (!graphs.contains(graph)) {
            throw new IllegalArgumentException(graph + " was not captured on this queue");
        }
        RuntimeException failure = null;
        try {
            backend.driver().streamSynchronize(stream);
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

    /** Destroys every graph the queue captured; called after its stream drained. */
    private void releaseGraphs() {
        for (HipGraph graph : new ArrayList<>(graphs)) {
            try {
                graph.destroy();
            } catch (RuntimeException e) {
                LOGGER.warn("Destroying an executable HIP graph of the {} queue failed: {}", queueType,
                        e.toString());
            }
        }
        graphs.clear();
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
        uncapturable("a stream synchronization");
        backend.driver().streamSynchronize(stream);
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
     * while closing, after marking itself closed and before releasing the device. The backend
     * releases every queue it created that way, so a queue never outlives the stream and staging
     * memory it frees.
     *
     * @throws RuntimeException the first translated driver failure, after every step was attempted
     */
    public void closeInternal() {
        synchronized (transferLock) {
            if (closed) {
                return;
            }
            closed = true;
        }
        RuntimeException failure = null;
        boolean drained = false;
        try {
            context.makeCurrent(backend.driver());
            discardRecording();
            backend.driver().streamSynchronize(stream);
            drained = true;
        } catch (RuntimeException e) {
            failure = e;
        }
        stopFences();
        releaseGraphs();
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
            backend.driver().streamDestroy(stream);
        } catch (RuntimeException e) {
            failure = Closeables.chain(failure, e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "HipComputeQueue{" + queueType + (ownedByBackend ? ", backend-owned" : "") + (closed ? ", closed" : "")
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
    HipGpuEvent recordFence() {
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
        synchronized (transferLock) {
            retireReadsLocked(upTo);
        }
    }

    // ------------------------------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------------------------------

    private GpuEvent launch(KernelHandle kernel, long[] global, long[] local, Object[] args,
                            GpuEvent[] waits, boolean withEvent) {
        enter();
        HipKernelHandle handle = ownHandle(kernel);
        long function = handle.function();
        String name = handle.kernelName();
        LaunchGeometry geometry = geometryOf(handle, global, local);
        long[] block = geometry.block();
        long[] paddedGrid = geometry.grid();
        validateWaits(waits);
        if (args != null && args.length > 0) {
            handle.args().setAll(args);
        }
        int gx = (int) paddedGrid[0];
        int gy = (int) paddedGrid[1];
        int gz = (int) paddedGrid[2];
        int bx = (int) block[0];
        int by = (int) block[1];
        int bz = (int) block[2];
        boolean waitsOrEvent = withEvent || (waits != null && waits.length > 0);
        if (capture == Capture.REPLAYING) {
            if (waitsOrEvent) {
                throw new GpuGraphMismatchException("command replay of " + replayGraph + ": " + name
                        + (withEvent ? " returns an event" : " waits on events")
                        + ", which the captured command sequence does not contain");
            }
            replayGraph.dispatch(handle, global, paddedGrid, block, name);
            return null;
        }
        if (capture == Capture.CAPTURING) {
            if (waitsOrEvent) {
                requireOwnQueueWaits(waits);
                throw new GpuUncapturableCommandException("a " + name + " dispatch that "
                        + (withEvent ? "returns an event" : "waits on events") + " cannot be captured: enqueue it"
                        + " before beginCapture, or return no event and wait on this queue's order");
            }
            Object[] params = handle.args().launchParams();
            backend.driver().launchKernel(function, gx, gy, gz, bx, by, bz, 0, stream, params);
            captureNodes.add(HipGraph.Node.kernel(handle, captureKernelOrdinal++, global.clone(),
                    paddedGrid.clone(), block.clone(), handle.args().snapshotValues(),
                    handle.args().sticky().version()));
            return null;
        }
        Object[] params = handle.args().launchParams();
        applyWaits(waits);
        backend.driver().launchKernel(function, gx, gy, gz, bx, by, bz, 0, stream, params);
        return withEvent ? record() : null;
    }

    /**
     * The grid and block of a dispatch of {@code handle}: a null local size becomes
     * {@link WorkSizes#defaultLocal(long[], long)}, the block is checked against the function's thread limit and the
     * grid against the device limits, both padded to three dimensions with 1. A dispatch that repeats the sizes of
     * the handle's last one reuses its geometry, which was validated when it was derived.
     */
    private LaunchGeometry geometryOf(HipKernelHandle handle, long[] global, long[] local) {
        LaunchGeometry last = handle.lastGeometry();
        if (last != null && last.matches(global, local)) {
            return last;
        }
        String name = handle.kernelName();
        long[] rawBlock = local == null ? WorkSizes.defaultLocal(global, handle.maxThreadsPerBlock()) : local;
        WorkSizes.validateLocal(rawBlock, handle.maxThreadsPerBlock(), name);
        long[] grid = WorkSizes.groupCounts(global, rawBlock);
        for (int d = 0; d < grid.length; d++) {
            if (grid[d] > maxGrid[d]) {
                throw new IllegalArgumentException(name + ": " + grid[d] + " blocks in dimension " + d
                        + " (global " + global[d] + ", local " + rawBlock[d] + ") exceed the device limit of "
                        + maxGrid[d]);
            }
        }
        LaunchGeometry geometry = new LaunchGeometry(global, local, WorkSizes.pad3(grid, 1),
                WorkSizes.pad3(rawBlock, 1));
        handle.lastGeometry(geometry);
        return geometry;
    }

    private GpuEvent write(GpuBuffer dst, long dstOffset, long size, HostSource source, GpuEvent[] waits) {
        enter();
        long device = devicePointer(dst, "write destination");
        checkDeviceRange(dst, dstOffset, size, "write destination");
        uncapturable("a write");
        validateWaits(waits);
        synchronized (transferLock) {
            applyWaits(waits);
            HipGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, upload.maxSliceBytes());
                    HipPinnedRing.Slice slice = upload.acquire(n);
                    long target = device + dstOffset + done;
                    try {
                        source.copyInto(slice.byteBuffer(), done, n);
                        backend.driver().memcpyHtoDAsync(target, slice.pointer(), n, stream);
                    } catch (RuntimeException e) {
                        slice.release();
                        throw e;
                    }
                    HipGpuEvent event = recordAfterCopy(slice);
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
        long device = devicePointer(src, "read source");
        checkDeviceRange(src, srcOffset, size, "read source");
        uncapturable("a read");
        validateWaits(waits);
        synchronized (transferLock) {
            applyWaits(waits);
            HipGpuEvent last = null;
            long done = 0;
            try {
                while (done < size) {
                    int n = (int) Math.min(size - done, readback.maxSliceBytes());
                    HipPinnedRing.Slice slice = acquireReadSlice(n);
                    long origin = device + srcOffset + done;
                    try {
                        backend.driver().memcpyDtoHAsync(slice.pointer(), origin, n, stream);
                    } catch (RuntimeException e) {
                        slice.release();
                        throw e;
                    }
                    HipGpuEvent event = recordAfterCopy(slice);
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
    private HipPinnedRing.Slice acquireReadSlice(int n) {
        HipPinnedRing.Slice slice = readback.tryAcquire(n);
        while (slice == null) {
            PendingRead oldest = pendingReads.peekFirst();
            if (oldest == null) {
                throw new IllegalStateException("HIP " + queueType + " readback ring is full without pending reads");
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
    private HipGpuEvent recordAfterCopy(HipPinnedRing.Slice slice) {
        try {
            return record();
        } catch (RuntimeException e) {
            try {
                backend.driver().streamSynchronize(stream);
                slice.release();
            } catch (RuntimeException drain) {
                e.addSuppressed(drain);
            }
            throw e;
        }
    }

    HipGpuEvent record() {
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
            LOGGER.warn("HIP {} queue closed with {} unfinished reads; their destinations were not written",
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

    private static RuntimeException closeRing(HipPinnedRing ring, RuntimeException failure) {
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
            if (!(e instanceof HipGpuEvent)) {
                throw new IllegalArgumentException("wait event " + i + " is not a HIP event of this backend: "
                        + (e == null ? "null" : e.getClass().getName()));
            }
            HipGpuEvent hip = (HipGpuEvent) e;
            if (!hip.isFrom(events)) {
                throw new IllegalArgumentException("wait event " + i + " belongs to another HIP backend");
            }
            if (hip.isClosed()) {
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
            HipGpuEvent hip = (HipGpuEvent) e;
            if (hip.owner() == this) {
                continue;
            }
            backend.driver().streamWaitEvent(stream, hip.getHipEvent());
        }
    }

    /**
     * Checks this queue and its backend are open and makes the device current.
     *
     * @throws IllegalStateException when either is closed
     */
    private void enter() {
        if (closed) {
            throw new IllegalStateException("HIP " + queueType + " queue already closed");
        }
        backend.enter();
    }

    private HipKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof HipKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a HipKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        HipKernelHandle handle = (HipKernelHandle) kernel;
        if (handle.backend() != backend) {
            throw new IllegalArgumentException("HIP kernel " + handle.kernelName() + " belongs to another backend");
        }
        return handle;
    }

    private long devicePointer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof HipGpuBuffer)) {
            throw new IllegalArgumentException(role + " must be a HIP buffer, got "
                    + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        HipGpuBuffer hip = (HipGpuBuffer) buffer;
        if (hip.backend() != backend) {
            throw new IllegalArgumentException(role + " belongs to another HIP backend");
        }
        Long ptr = hip.getDevicePointer();
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
        private final HipGpuEvent event;
        private final long sequence;
        private final HipPinnedRing.Slice slice;
        private final HostSink sink;
        private final long hostOffset;
        private final int size;

        PendingRead(HipGpuEvent event, HipPinnedRing.Slice slice, HostSink sink, long hostOffset, int size) {
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

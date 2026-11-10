package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.GpuGraph;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphMismatchException;
import com.eonmux.neuralfree.acceleration.backend.GpuUncapturableCommandException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkTimelineSemaphoreSubmitInfo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK10.vkDestroyCommandPool;
import static org.lwjgl.vulkan.VK10.vkDestroySemaphore;

/**
 * A Vulkan queue with the {@link ComputeQueue} contract: commands are recorded into batched
 * command buffers, barriers are recorded only on hazards, and completion is tracked on a timeline
 * semaphore.
 *
 * <p><b>Batches.</b> Commands go into the current {@link VkCommandBatch}, which is submitted by
 * {@link #finish()}, a wait on one of its events, when the batch has staged half a staging ring,
 * before a dispatch that has to create its pipeline first (so the device runs the recorded work
 * while the pipeline compiles), and as {@link VkSubmitPolicy} decides: once the batch holds the
 * ramping dispatch threshold (16 after the queue drains, doubling per submission up to
 * {@value #DEFAULT_DISPATCHES_PER_SUBMIT}, {@value #DISPATCHES_PER_SUBMIT_PROPERTY}), and on
 * {@link #flush()} unless the batch holds only a few dispatches while the device is still busy.
 * Batch {@code n} signals timeline value {@code n};
 * a command's event carries the value of its batch. Batches retire in order once their value is
 * reached: readbacks are copied into their destination arrays, staging space and descriptor frames
 * are reclaimed and the command buffer is recycled.</p>
 *
 * <p><b>Hazards.</b> Each batch starts with a global barrier. Within a batch a
 * {@link VkHazardTracker} records which buffer ranges were read and written since the last barrier;
 * a global barrier is recorded before a command only when it reads a range written since then, or
 * writes a range read or written since then (llama.cpp records the same barrier before every node,
 * llama.cpp ggml-vulkan.cpp {@code ggml_vk_sync_buffers}).</p>
 *
 * <p><b>Waits.</b> An event of this queue is already ordered. An event of another queue of the
 * same backend is submitted on its queue first, then added as a timeline-semaphore wait of this
 * queue's next submission; the batch recorded so far is submitted before the wait is added, so
 * earlier work is not delayed. Nothing blocks on the host. A wait that a closed producer already
 * satisfied is dropped at submission; closing a producer while another thread submits work waiting
 * on it is a caller error.</p>
 *
 * <p><b>Buffer lifetime.</b> {@link #lastUseValue()} is the value of the batch that received the latest
 * command; the backend's release gate compares it with {@link #completedValue()}, so a buffer closed
 * after a command was recorded is destroyed only once that command's batch has retired. Retiring
 * batches lets the allocator destroy such buffers.</p>
 *
 * <p><b>Transfers.</b> Writes copy the host data into an upload {@link VkStagingRing} before
 * returning and record a buffer copy; reads record a copy into a readback ring and copy into the
 * destination array when the batch retires, so destinations are valid after {@link #finish()} or
 * the read event's {@link GpuEvent#awaitCompletion()}. Transfers larger than half a ring are split.</p>
 *
 * <p><b>Dispatch.</b> {@code global} counts work-items. The local size is the caller's for
 * specialisable dimensions and the shader's for fixed ones ({@link VkKernelHandle#resolveLocalSize});
 * groups are {@code ceil(global / local)}. The group count of a one-dimensional dispatch above
 * {@code maxComputeWorkGroupCount[0]} is folded into {@code (max, ceil(groups / max))}; kernels recover
 * the linear group id ({@code nf_wg_id()}) and discard the overflow groups. Multi-dimensional
 * dispatches are never folded, and any count above the device limits is rejected. A queue on a
 * transfer-only family rejects dispatches.</p>
 *
 * <p><b>Graph capture.</b> {@code beginCapture()} records dispatches into a dedicated simultaneous-use
 * command buffer instead of the current batch; {@code endCapture()} returns them as a replayable
 * {@link VkGraph}. The step that captured them submits them once with
 * {@link GpuGraph#launch(ComputeQueue, GpuEvent...)}, and a replay verifies the step's dispatches against
 * the capture and resubmits the recorded buffer once, ordered on the queue's timeline. Host transfers,
 * reads, copies, fills, markers and event dispatches are refused while capturing with
 * {@link IllegalStateException}; during a replay they report a mismatch. A graph launches only on the
 * queue that captured it. A replay refuses a dispatch whose bound arguments or group counts changed since
 * the capture: the recorded push constants and group counts are fixed.</p>
 */
@NotThreadSafe("Commands are recorded by one thread at a time. Event completion, waits from other queues and "
        + "flushes are thread-safe: all queue state is guarded by the queue's monitor, and a queue never "
        + "takes another queue's monitor while holding its own.")
public final class VkComputeQueue implements ComputeQueue, VkStagingRing.Timeline {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkComputeQueue.class);

    /** Largest number of dispatches recorded into one command buffer before it is submitted. */
    public static final String DISPATCHES_PER_SUBMIT_PROPERTY = "neuralfree.vulkan.dispatchesPerSubmit";
    /** Default of {@value #DISPATCHES_PER_SUBMIT_PROPERTY}. */
    public static final int DEFAULT_DISPATCHES_PER_SUBMIT = 256;

    private final VkComputeBackend backend;
    private final VkDevice device;
    private final VkQueueContext context;
    private final QueueType queueType;
    private final boolean ownedByBackend;
    private final boolean dispatchCapable;
    private final VkMemoryAllocator allocator;
    private final VkDescriptorStrategy strategy;
    private final long[] maxGroups;
    private final VkSubmitPolicy submitPolicy;
    /** Dispatch timing under {@code -Dneuralfree.gpu.profile=true}, or null. */
    private final VkDispatchTimer timer;
    private final long commandPool;
    private final long timeline;
    private final VkHazardTracker tracker = new VkHazardTracker();
    private final ArrayDeque<VkCommandBatch> freeBatches = new ArrayDeque<>();
    private final ArrayDeque<VkCommandBatch> inFlight = new ArrayDeque<>();
    private final List<VkCommandBatch> allBatches = new ArrayList<>();

    private VkCommandBatch current;
    private VkStagingRing uploadRing;
    private VkStagingRing readbackRing;
    private long uploadBytesInBatch;
    private long readbackBytesInBatch;
    private volatile long submittedValue;
    private volatile long completedValue;
    private volatile long lastUseValue;
    private volatile boolean closed;
    /** True between {@code beginCapture} and its end: dispatches record into the captured batch. */
    private boolean capturing;
    private VkCommandBatch captureBatch;
    private final List<VkGraph.Dispatch> captureDispatches = new ArrayList<>();
    private final List<VkGraph> graphs = new ArrayList<>();
    /** The graph whose replay body is dispatching, or null; dispatches verify against its capture. */
    private VkGraph replayGraph;
    /** Timeline values of submitted replays, in order; retired like the batches of {@link #inFlight}. */
    private final ArrayDeque<Long> replaySignals = new ArrayDeque<>();
    /** Largest submitted replay value not known complete; later submissions wait on it. */
    private long pendingReplaySignal;

    /**
     * @param backend        the owning backend (device, allocator, descriptor strategy, limits)
     * @param context        the {@code VkQueue} this queue submits to
     * @param queueType      reported type; a TRANSFER queue on a transfer-only family rejects dispatches
     * @param ownedByBackend true for the backend's own queues, whose {@link #close()} only finishes
     * @throws GpuOutOfMemoryException when the driver runs out of memory creating the command pool or timeline
     *                                 semaphore
     * @throws GpuBackendException     when the command pool or timeline semaphore cannot be created otherwise
     */
    public VkComputeQueue(VkComputeBackend backend, VkQueueContext context, QueueType queueType,
                          boolean ownedByBackend) {
        if (backend == null || context == null || queueType == null) {
            throw new IllegalArgumentException("backend, queue context and queue type must not be null");
        }
        this.backend = backend;
        this.device = backend.device();
        this.context = context;
        this.queueType = queueType;
        this.ownedByBackend = ownedByBackend;
        this.dispatchCapable = backend.info().queueFamilies().get(context.family()).has(VK_QUEUE_COMPUTE_BIT);
        this.allocator = backend.allocator();
        this.strategy = backend.descriptorStrategy();
        this.maxGroups = backend.info().maxComputeWorkGroupCount();
        this.submitPolicy = new VkSubmitPolicy(backend.dispatchesPerSubmit());
        this.timer = dispatchCapable
                ? VkDispatchTimer.create(device, backend.info(), context.family(), submitPolicy.cap()) : null;
        this.commandPool = VkQueueResources.createCommandPool(device, context.family());
        long semaphore;
        try {
            semaphore = VkQueueResources.createTimeline(device);
        } catch (RuntimeException e) {
            vkDestroyCommandPool(device, commandPool, null);
            throw e;
        }
        this.timeline = semaphore;
    }

    /**
     * {@value #DISPATCHES_PER_SUBMIT_PROPERTY}, or {@value #DEFAULT_DISPATCHES_PER_SUBMIT} when unset.
     *
     * @throws IllegalArgumentException when the property is not a positive integer
     */
    public static int configuredDispatchesPerSubmit() {
        String value = System.getProperty(DISPATCHES_PER_SUBMIT_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            return DEFAULT_DISPATCHES_PER_SUBMIT;
        }
        int n;
        try {
            n = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("-D" + DISPATCHES_PER_SUBMIT_PROPERTY
                    + " must be a positive integer, got '" + value + "'", e);
        }
        if (n <= 0) {
            throw new IllegalArgumentException("-D" + DISPATCHES_PER_SUBMIT_PROPERTY
                    + " must be a positive integer, got '" + value + "'");
        }
        return n;
    }

    @Override
    public QueueType getQueueType() {
        return queueType;
    }

    /** The {@code VkQueue} this queue submits to. */
    public VkQueueContext context() {
        return context;
    }

    /** The timeline semaphore the batches signal. */
    public long timelineSemaphore() {
        return timeline;
    }

    /** Value of the last submitted batch. */
    public long submittedValue() {
        return submittedValue;
    }

    /** Value up to which batches have retired. */
    public long completedValue() {
        return completedValue;
    }

    /** Value of the batch that received the latest command; 0 before the first command. */
    public long lastUseValue() {
        return lastUseValue;
    }

    public VkComputeBackend backend() {
        return backend;
    }

    public boolean isClosed() {
        return closed;
    }

    // ------------------------------------------------------------------ transfers

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, byte[] srcData,
                                       GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source array must not be null");
        }
        VkQueueChecks.requireWithin(sizeInBytes, srcData.length, "write source array");
        return write(dstBuffer, dstOffset, sizeInBytes,
                (from, dst) -> dst.put(srcData, (int) from, dst.remaining()), waitOnEvents);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData,
                                       GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source array must not be null");
        }
        VkQueueChecks.requireFloatSized(sizeInBytes, srcData.length, "write source array");
        return write(dstBuffer, dstOffset, sizeInBytes,
                (from, dst) -> dst.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        .put(srcData, (int) (from / Float.BYTES), dst.remaining() / Float.BYTES),
                waitOnEvents);
    }

    /** Reads bytes {@code [0, sizeInBytes)} of {@code srcData} by absolute index; its position and limit are untouched. */
    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, ByteBuffer srcData,
                                       GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source ByteBuffer must not be null");
        }
        VkQueueChecks.requireWithin(sizeInBytes, srcData.capacity(), "write source ByteBuffer");
        return write(dstBuffer, dstOffset, sizeInBytes, (from, dst) -> {
            ByteBuffer view = srcData.duplicate();
            view.clear();
            view.position((int) from).limit((int) from + dst.remaining());
            dst.put(view);
        }, waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, byte[] dstData,
                                      GpuEvent... waitOnEvents) {
        if (dstData == null) {
            throw new IllegalArgumentException("read destination array must not be null");
        }
        VkQueueChecks.requireWithin(sizeInBytes, dstData.length, "read destination array");
        return read(srcBuffer, srcOffset, sizeInBytes,
                (slice, done) -> VkCommandBatch.PendingRead.ofBytes(slice, dstData, (int) done), waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData,
                                      GpuEvent... waitOnEvents) {
        if (dstData == null) {
            throw new IllegalArgumentException("read destination array must not be null");
        }
        VkQueueChecks.requireFloatSized(sizeInBytes, dstData.length, "read destination array");
        return read(srcBuffer, srcOffset, sizeInBytes,
                (slice, done) -> VkCommandBatch.PendingRead.ofFloats(slice, dstData, (int) (done / Float.BYTES)),
                waitOnEvents);
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long sizeInBytes,
                                  GpuEvent... waitOnEvents) {
        VkGpuBuffer from = VkQueueChecks.ownBuffer(src, "copy source", allocator);
        VkGpuBuffer to = VkQueueChecks.ownBuffer(dst, "copy destination", allocator);
        VkQueueChecks.requireRange(from, srcOffset, sizeInBytes, "copy source");
        VkQueueChecks.requireRange(to, dstOffset, sizeInBytes, "copy destination");
        if (from == to && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        VkEventWaits.validate(waitOnEvents, backend);
        VkEventWaits.flushProducers(waitOnEvents, this);
        synchronized (this) {
            enter();
            uncapturable("a device copy");
            applyWaits(waitOnEvents);
            if (sizeInBytes == 0) {
                return;
            }
            VkCommandBatch b = batch();
            transferHazard(b, from.trackingId(), srcOffset, sizeInBytes, to.trackingId(), dstOffset, sizeInBytes);
            b.copy(from.vkBuffer(), from.offset() + srcOffset, to.vkBuffer(), to.offset() + dstOffset, sizeInBytes);
        }
    }

    /** Records {@code vkCmdFillBuffer}; {@code pattern} is stored little-endian in every 4-byte word. */
    @Override
    public void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                  GpuEvent... waitOnEvents) {
        VkGpuBuffer to = VkQueueChecks.ownBuffer(dst, "fill destination", allocator);
        if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                    + " must be non-negative multiples of 4");
        }
        VkQueueChecks.requireRange(to, dstOffset, sizeInBytes, "fill destination");
        VkEventWaits.validate(waitOnEvents, backend);
        VkEventWaits.flushProducers(waitOnEvents, this);
        synchronized (this) {
            enter();
            if (sizeInBytes == 0) {
                return;
            }
            uncapturable("a fill");
            applyWaits(waitOnEvents);
            VkCommandBatch b = batch();
            transferHazard(b, 0, 0, 0, to.trackingId(), dstOffset, sizeInBytes);
            b.fill(to.vkBuffer(), to.offset() + dstOffset, sizeInBytes, pattern);
        }
    }

    /** An event that completes when every command enqueued so far on this queue, and the waits, complete. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        VkEventWaits.validate(waitOnEvents, backend);
        VkEventWaits.flushProducers(waitOnEvents, this);
        synchronized (this) {
            enter();
            uncapturable("a marker");
            applyWaits(waitOnEvents);
            return markerLocked();
        }
    }

    // ------------------------------------------------------------------ kernels

    /** Replaces the handle's arguments with {@code args} and dispatches; null or empty {@code args} launch the stored ones. */
    @Override
    public GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args,
                                  GpuEvent... waitOnEvents) {
        return launch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, true);
    }

    @Override
    public void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize, Object[] args,
                                     GpuEvent... waitOnEvents) {
        launch(kernel, globalWorkSize, localWorkSize, args, waitOnEvents, false);
    }

    @Override
    public void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        VkQueueChecks.ownHandle(kernel, strategy).argState().set(argIndex, arg);
    }

    @Override
    public void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                      GpuEvent... waitOnEvents) {
        launch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, false);
    }

    @Override
    public GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                                   GpuEvent... waitOnEvents) {
        return launch(kernel, globalWorkSize, localWorkSize, null, waitOnEvents, true);
    }

    // ------------------------------------------------------------------ command capture

    @Override
    public boolean supportsGraphCapture() {
        return dispatchCapable;
    }

    @Override
    public boolean isCapturing() {
        synchronized (this) {
            return capturing;
        }
    }

    @Override
    public void beginCapture() {
        synchronized (this) {
            enter();
            if (!dispatchCapable) {
                throw new UnsupportedOperationException("Vulkan " + queueType + " queue on transfer-only family "
                        + context.family() + " cannot capture dispatches");
            }
            if (capturing || replayGraph != null) {
                throw new IllegalStateException("Vulkan " + queueType + " queue is already "
                        + (capturing ? "capturing" : "replaying") + " dispatches");
            }
            submitLocked();
            capturing = true;
            captureBatch = null;
            captureDispatches.clear();
            lastUseValue = pendingValue();
        }
    }

    @Override
    public GpuGraph endCapture() {
        synchronized (this) {
            enter();
            if (replayGraph != null) {
                throw new IllegalStateException("Vulkan " + queueType + " queue: endCapture needs a capture,"
                        + " a replay of " + replayGraph + " is dispatching");
            }
            if (!capturing) {
                throw new IllegalStateException("Vulkan " + queueType + " queue is not capturing a graph");
            }
            List<VkGraph.Dispatch> dispatches = new ArrayList<>(captureDispatches);
            VkCommandBatch b = captureBatch;
            capturing = false;
            captureBatch = null;
            captureDispatches.clear();
            if (b == null || dispatches.isEmpty()) {
                disposeCaptureBatch(b);
                return null;
            }
            try {
                b.end();
            } catch (RuntimeException e) {
                disposeCaptureBatch(b);
                throw e;
            }
            VkGraph graph = new VkGraph(this, b, dispatches);
            graphs.add(graph);
            return graph;
        }
    }

    @Override
    public void abortCapture() {
        synchronized (this) {
            enter();
            if (replayGraph != null) {
                throw new IllegalStateException("Vulkan " + queueType + " queue: abortCapture needs a"
                        + " capture, a replay of " + replayGraph + " is dispatching");
            }
            if (!capturing) {
                return;
            }
            discardCaptureLocked();
        }
    }

    /** One dispatch of a capture: recorded into the captured command batch, executed by no replay. */
    private void recordCaptureLocked(VkKernelHandle handle, VkKernelHandle.Variant variant, long[] groups,
                                     Object[] args) {
        VkArgState state = handle.argState();
        if (args != null && args.length > 0) {
            state.setAll(args);
        }
        state.materialize();
        VkCommandBatch b = captureBatchLocked();
        if (tracker.needsBarrier(state.readIds(), state.writeIds())) {
            b.barrier();
            tracker.reset();
        }
        b.bindPipeline(variant.pipeline());
        b.bindDescriptors(variant.program(), state);
        variant.program().pushConstants(b.commandBuffer(), state);
        b.dispatch((int) groups[0], (int) groups[1], (int) groups[2]);
        tracker.record(state.readIds(), state.writeIds());
        captureDispatches.add(new VkGraph.Dispatch(handle, groups.clone(), VkGraph.copyArgs(state)));
    }

    /** One dispatch of a replay body: matched against the next captured dispatch, never executed. */
    private void verifyReplayLocked(VkKernelHandle handle, long[] groups, Object[] args) {
        VkArgState state = handle.argState();
        if (args != null && args.length > 0) {
            state.setAll(args);
        }
        replayGraph.dispatch(handle, groups, VkGraph.copyArgs(state));
    }

    /** The command batch the capture records into, begun on first use. */
    private VkCommandBatch captureBatchLocked() {
        lastUseValue = pendingValue();
        if (captureBatch == null) {
            captureBatch = newReplayableBatch();
            captureDispatches.clear();
        }
        return captureBatch;
    }

    private VkCommandBatch newReplayableBatch() {
        VkCommandBuffer cb = VkQueueResources.allocateCommandBuffer(device, commandPool);
        VkDescriptorStrategy.Frame frame;
        try {
            frame = strategy.newFrame();
        } catch (RuntimeException e) {
            VkQueueResources.freeCommandBuffer(device, commandPool, cb);
            throw e;
        }
        VkCommandBatch b = new VkCommandBatch(cb, frame, dispatchCapable, true);
        try {
            b.begin();
        } catch (RuntimeException e) {
            step(() -> strategy.destroy(frame), e);
            b.close();
            VkQueueResources.freeCommandBuffer(device, commandPool, cb);
            throw e;
        }
        return b;
    }

    /** Returns an unused capture batch's command buffer and frame to the queue. */
    private void disposeCaptureBatch(VkCommandBatch b) {
        if (b == null) {
            return;
        }
        if (b.isRecording()) {
            b.reset();
        }
        strategy.destroy(b.descriptorFrame());
        b.close();
        VkQueueResources.freeCommandBuffer(device, commandPool, b.commandBuffer());
    }

    /**
     * Ends an open capture, discarding the dispatches recorded so far: a failed step leaves no device
     * work behind and the queue returns to normal dispatching.
     */
    private void discardCaptureLocked() {
        capturing = false;
        VkCommandBatch b = captureBatch;
        captureBatch = null;
        captureDispatches.clear();
        tracker.reset();
        disposeCaptureBatch(b);
    }

    /**
     * Refuses a command the captured dispatch sequence cannot hold: a host transfer, a read, a device
     * copy or fill, a marker or an event while a capture is recording, and any command that is not
     * part of the capture during a replay.
     */
    private void uncapturable(String op) {
        if (!capturing && replayGraph == null) {
            return;
        }
        if (replayGraph != null) {
            throw new GpuGraphMismatchException("command replay of " + replayGraph + " hit " + op
                    + ", which the captured dispatches do not contain");
        }
        throw new GpuUncapturableCommandException("a " + op + " cannot be captured: perform host transfers and"
                + " reads before beginCapture, and read results only after the graph has been launched");
    }

    /**
     * A replay's step body: dispatches verify against the capture instead of recording.
     *
     * @throws IllegalStateException when the queue is closed, another recording is active or the graph
     *                               does not belong to this queue
     */
    void beginGraphReplay(VkGraph graph) {
        synchronized (this) {
            enter();
            if (capturing || replayGraph != null) {
                throw new IllegalStateException("Vulkan " + queueType + " queue is already "
                        + (capturing ? "capturing" : "replaying") + " dispatches");
            }
            if (graph.isClosed()) {
                throw new IllegalStateException(graph + " is closed");
            }
            if (!graphs.contains(graph)) {
                throw new IllegalArgumentException(graph + " was not captured on this queue");
            }
            replayGraph = graph;
        }
    }

    /** The replay submission: one resubmission of the captured command buffer, ordered on the timeline. */
    void submitGraphReplay(VkGraph graph) {
        synchronized (this) {
            enter();
            if (replayGraph != graph) {
                throw new IllegalStateException("Vulkan " + queueType + " queue: " + graph + " is not replaying");
            }
            submitCapturedBatchLocked(graph.batch());
        }
    }

    /**
     * One submission of the graph captured by the step itself, with its capture-time push constants:
     * the step that recorded the dispatches submitted through
     * {@link GpuGraph#launch(ComputeQueue, GpuEvent...)}. A Vulkan graph runs only on the queue that
     * captured it, because its command buffer belongs to that queue.
     *
     * @throws IllegalArgumentException when {@code target} is another queue
     * @throws IllegalStateException    when the queue or graph is closed, or another recording is active
     */
    GpuEvent launchCapture(VkGraph graph, ComputeQueue target, GpuEvent[] waitOnEvents) {
        if (target != this) {
            throw new IllegalArgumentException("Vulkan graph " + graph + " runs only on the queue that captured"
                    + " it, not on " + (target == null ? "null" : target.toString()));
        }
        VkEventWaits.validate(waitOnEvents, backend);
        VkEventWaits.flushProducers(waitOnEvents, this);
        synchronized (this) {
            enter();
            if (capturing || replayGraph != null) {
                throw new IllegalStateException("Vulkan " + queueType + " queue is already "
                        + (capturing ? "capturing" : "replaying") + " dispatches");
            }
            if (graph.isClosed()) {
                throw new IllegalStateException(graph + " is closed");
            }
            if (!graphs.contains(graph)) {
                throw new IllegalArgumentException(graph + " was not captured on this queue");
            }
            applyWaitsTo(graph.batch(), waitOnEvents);
            long signal = submitCapturedBatchLocked(graph.batch());
            return new VkGpuEvent(this, signal);
        }
    }

    /** Adds a timeline wait for each event of another queue to {@code batch}, as {@link #applyWaits} does. */
    private void applyWaitsTo(VkCommandBatch batch, GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (GpuEvent e : waits) {
            VkGpuEvent event = (VkGpuEvent) e;
            VkComputeQueue producer = event.owner();
            long value = event.value();
            if (producer == this || value <= producer.completedValue) {
                continue;
            }
            if (value > producer.submittedValue) {
                throw new IllegalStateException("wait event value " + value + " of " + producer
                        + " was never submitted" + (producer.closed ? " before the queue closed" : ""));
            }
            if (current != null && current.commandCount() > 0) {
                submitLocked();
            }
            batch.addWait(producer, value);
        }
    }

    /**
     * Submits a captured graph's command buffer after the queue's recorded batch, on the timeline; a
     * later batch waits on the replay signal, so the queue's order is kept.
     *
     * @return the timeline value the submission signals
     */
    private long submitCapturedBatchLocked(VkCommandBatch batch) {
        submitLocked();
        long signal = submittedValue + 1;
        submitBatchLocked(batch, signal, submittedValue);
        replaySignals.addLast(signal);
        pendingReplaySignal = signal;
        submittedValue = signal;
        lastUseValue = signal;
        submitPolicy.onSubmit();
        return signal;
    }

    /** Ends a replay's step body: a failed body leaves the device untouched (nothing was submitted). */
    void endGraphReplay(VkGraph graph, boolean submitted) {
        synchronized (this) {
            if (replayGraph != graph) {
                throw new IllegalStateException("Vulkan " + queueType + " queue: " + graph + " is not replaying");
            }
            replayGraph = null;
            if (!submitted) {
                LOGGER.debug("Command replay of {} did not reach its submission; the caller retries", graph);
            }
        }
    }

    /** Waits for the graph's submitted replays, then frees its command buffer and descriptor frame. */
    void releaseGraph(VkGraph graph) {
        long last = 0;
        synchronized (this) {
            enter();
            if (!graphs.contains(graph)) {
                throw new IllegalArgumentException(graph + " was not captured on this queue");
            }
            for (Long value : replaySignals) {
                last = Math.max(last, value);
            }
        }
        if (last > 0 && last > completedValue) {
            VkTimelineWait.wait(device, timeline, last);
        }
        synchronized (this) {
            retireUpTo(last);
            graphs.remove(graph);
            graph.destroy();
            strategy.destroy(graph.batch().descriptorFrame());
            graph.batch().close();
            VkQueueResources.freeCommandBuffer(device, commandPool, graph.batch().commandBuffer());
        }
    }

    /** Frees every open graph's command buffer and descriptor frame; called on queue close. */
    private void closeGraphsLocked() {
        RuntimeException failure = null;
        for (VkGraph graph : graphs) {
            failure = step(() -> strategy.destroy(graph.batch().descriptorFrame()), failure);
            failure = step(graph.batch()::close, failure);
            failure = step(() -> VkQueueResources.freeCommandBuffer(device, commandPool,
                    graph.batch().commandBuffer()), failure);
            graph.destroy();
        }
        graphs.clear();
        if (captureBatch != null) {
            VkCommandBatch b = captureBatch;
            captureBatch = null;
            capturing = false;
            failure = step(() -> strategy.destroy(b.descriptorFrame()), failure);
            failure = step(b::close, failure);
            failure = step(() -> VkQueueResources.freeCommandBuffer(device, commandPool, b.commandBuffer()), failure);
        }
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ submission and completion

    /**
     * Submits the batch being recorded at a step boundary, as {@link VkSubmitPolicy} decides: always
     * when it holds copies, fills, readbacks or cross-queue waits, when it holds at least
     * {@value VkSubmitPolicy#MIN_FLUSH_DISPATCHES} dispatches, or when the device has finished every
     * earlier batch; a smaller batch recorded while the device is busy keeps collecting dispatches.
     * Asking the device polls the timeline and retires the batches that completed. Never blocks on
     * the device.
     */
    @Override
    public void flush() {
        synchronized (this) {
            enter();
            VkCommandBatch b = current;
            if (b != null && !b.isEmpty()) {
                boolean otherWork = b.commandCount() > b.dispatchCount() || b.waitCount() > 0
                        || b.hasPendingReads();
                if (!submitPolicy.submitOnFlush(b.dispatchCount(), otherWork, this::deviceIdleLocked)) {
                    return;
                }
            }
            submitLocked();
        }
    }

    /** Submits, waits until every submitted batch completes, and retires them (readbacks are then valid). */
    @Override
    public void finish() {
        long target;
        synchronized (this) {
            enter();
            submitLocked();
            target = submittedValue;
        }
        if (target > completedValue) {
            VkTimelineWait.wait(device, timeline, target);
        }
        synchronized (this) {
            retireUpTo(target);
        }
    }

    /**
     * The dispatch timing of this queue ({@link VkDispatchTimer}), or null when
     * {@code -Dneuralfree.gpu.profile=true} is unset or the queue family writes no timestamps.
     */
    @Override
    public KernelTiming kernelTiming() {
        return timer != null ? timer.timing() : null;
    }

    /**
     * The backend's own queues only finish (nothing once the backend is closed); a caller-owned queue
     * is released ({@link #closeInternal()}).
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

    /** The value the batch being recorded will signal. */
    @Override
    public long pendingValue() {
        return submittedValue + 1;
    }

    /**
     * Submits the batch holding {@code value} if it is still being recorded, blocks until the timeline
     * reaches it and retires up to it. Used by the staging rings; not interruptible.
     */
    @Override
    public void waitForValue(long value) {
        if (prepareWait(value)) {
            VkTimelineWait.wait(device, timeline, value);
        }
        synchronized (this) {
            retireUpTo(Math.min(value, submittedValue));
        }
    }

    /**
     * {@link #waitForValue} that can be interrupted while blocked on the device.
     *
     * @throws InterruptedException  when the thread is interrupted while waiting
     * @throws IllegalStateException when the queue closed without reaching {@code value}
     */
    void awaitValue(long value) throws InterruptedException {
        if (prepareWait(value)) {
            VkTimelineWait.waitInterruptibly(device, timeline, value, this);
        }
        synchronized (this) {
            retireUpTo(Math.min(value, submittedValue));
        }
    }

    /**
     * Submits the batch holding {@code value} if needed and reports whether the timeline reached it,
     * retiring the completed batches.
     *
     * @throws IllegalStateException when the queue closed without reaching {@code value}
     */
    boolean pollValue(long value) {
        if (value <= completedValue) {
            return true;
        }
        synchronized (this) {
            if (value <= completedValue) {
                return true;
            }
            ensureSubmitted(value);
            long reached = VkTimelineWait.counter(device, timeline);
            if (reached > completedValue) {
                retireUpTo(Math.min(reached, submittedValue));
            }
            return reached >= value;
        }
    }

    /** Submits the batch being recorded when it signals {@code value}; called by events and other queues. */
    void flushIfPending(long value) {
        synchronized (this) {
            if (!closed && value > submittedValue) {
                submitLocked();
            }
        }
    }

    /**
     * Releases the queue: drains it, completes pending readbacks, and destroys its batches, command
     * pool, timeline semaphore and staging rings. Idempotent. Every step runs even when an earlier
     * one fails; the first failure is rethrown at the end.
     */
    public void closeInternal() {
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
        }
        RuntimeException failure = null;
        try {
            long target;
            synchronized (this) {
                submitLocked();
                target = submittedValue;
            }
            if (target > completedValue) {
                VkTimelineWait.wait(device, timeline, target);
            }
            synchronized (this) {
                retireUpTo(target);
            }
        } catch (RuntimeException e) {
            failure = e;
            LOGGER.warn("Vulkan queue {} did not drain before closing: {}", this, e.getMessage());
        }
        synchronized (this) {
            failure = step(this::closeGraphsLocked, failure);
            for (VkCommandBatch b : allBatches) {
                failure = step(() -> strategy.destroy(b.descriptorFrame()), failure);
                failure = step(b::close, failure);
            }
            allBatches.clear();
            freeBatches.clear();
            inFlight.clear();
            replaySignals.clear();
            pendingReplaySignal = 0;
            current = null;
            if (timer != null) {
                failure = step(timer::close, failure);
            }
            failure = step(() -> vkDestroyCommandPool(device, commandPool, null), failure);
            failure = step(() -> vkDestroySemaphore(device, timeline, null), failure);
            if (uploadRing != null) {
                failure = step(uploadRing::close, failure);
            }
            if (readbackRing != null) {
                failure = step(readbackRing::close, failure);
            }
        }
        backend.onQueueClosed(this);
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "VkComputeQueue{" + queueType + ", " + context + (ownedByBackend ? ", backend-owned" : "")
                + ", submitted=" + submittedValue + ", completed=" + completedValue + (closed ? ", closed" : "") + '}';
    }

    // ------------------------------------------------------------------ recording

    /**
     * Records one dispatch. The variant (and its pipeline) is selected before the queue monitor is
     * taken, so creating a pipeline never blocks other threads' event polls or cross-queue flushes;
     * when a pipeline has to be created, the batch recorded so far is submitted first so the device
     * runs it during the compile.
     */
    private VkGpuEvent launch(KernelHandle kernel, long[] global, long[] local, Object[] args, GpuEvent[] waits,
                              boolean wantEvent) {
        VkKernelHandle handle = VkQueueChecks.ownHandle(kernel, strategy);
        enter();
        if (!dispatchCapable) {
            throw new UnsupportedOperationException("Vulkan " + queueType + " queue on transfer-only family "
                    + context.family() + " cannot dispatch kernel " + handle.kernelName());
        }
        VkEventWaits.validate(waits, backend);
        int[] local3 = handle.resolveLocalSize(global, local);
        long[] groups = VkDispatchLimits.groupCounts(handle.kernelName(), global, local3, maxGroups);
        VkKernelHandle.Variant variant = handle.preparedVariant(local3);
        if (variant == null) {
            synchronized (this) {
                enter();
                if (current != null && current.commandCount() > 0) {
                    submitLocked();
                }
            }
            variant = handle.select(local3);
        }
        VkEventWaits.flushProducers(waits, this);
        synchronized (this) {
            enter();
            if (capturing) {
                if (wantEvent || (waits != null && waits.length > 0)) {
                    throw new GpuUncapturableCommandException("a " + handle.kernelName() + " dispatch that "
                            + (wantEvent ? "returns an event" : "waits on events") + " cannot be captured: enqueue"
                            + " it before beginCapture, or return no event and wait on this queue's order");
                }
                recordCaptureLocked(handle, variant, groups, args);
                return null;
            } else if (replayGraph != null) {
                if (wantEvent || (waits != null && waits.length > 0)) {
                    throw new GpuGraphMismatchException("command replay of " + replayGraph + ": "
                            + handle.kernelName() + (wantEvent ? " returns an event" : " waits on another queue's"
                            + " events") + ", which the captured dispatches do not contain");
                }
                verifyReplayLocked(handle, groups, args);
                return null;
            }
            applyWaits(waits);
            VkArgState state = handle.argState();
            if (args != null && args.length > 0) {
                state.setAll(args);
            }
            state.materialize();

            VkCommandBatch b = batch();
            boolean timed = timer != null && timer.open(b, timingKey(handle.kernelName(), global, local3));
            if (timed) {
                tracker.reset();
            } else if (tracker.needsBarrier(state.readIds(), state.writeIds())) {
                b.barrier();
                tracker.reset();
            }
            b.bindPipeline(variant.pipeline());
            b.bindDescriptors(variant.program(), state);
            variant.program().pushConstants(b.commandBuffer(), state);
            b.dispatch((int) groups[0], (int) groups[1], (int) groups[2]);
            if (timed) {
                timer.close(b);
            }
            tracker.record(state.readIds(), state.writeIds());
            long value = pendingValue();
            if (submitPolicy.submitAfterDispatch(b.dispatchCount())) {
                submitLocked();
            }
            return wantEvent ? new VkGpuEvent(this, value) : null;
        }
    }

    /** The timing report key of a dispatch: its kernel name and grid, the OpenCL queue's key. */
    private static String timingKey(String name, long[] global, int[] local3) {
        return name + "#" + global[0] + "x" + local3[0];
    }

    private GpuEvent write(GpuBuffer dst, long dstOffset, long size, HostSource source, GpuEvent[] waits) {
        VkGpuBuffer to = VkQueueChecks.ownBuffer(dst, "write destination", allocator);
        VkQueueChecks.requireRange(to, dstOffset, size, "write destination");
        VkEventWaits.validate(waits, backend);
        VkEventWaits.flushProducers(waits, this);
        synchronized (this) {
            enter();
            uncapturable("a host write");
            applyWaits(waits);
            if (size == 0) {
                return markerLocked();
            }
            VkStagingRing ring = uploadRing();
            long chunk = chunkBytes(ring);
            long value = 0;
            for (long done = 0; done < size; ) {
                int n = (int) Math.min(chunk, size - done);
                VkStagingRing.Slice slice = ring.acquire(n, this);
                ByteBuffer staged = slice.mapped().duplicate();
                staged.clear();
                source.copy(done, staged);
                VkCommandBatch b = batch();
                requireSliceValue(slice);
                long at = dstOffset + done;
                transferHazard(b, 0, 0, 0, to.trackingId(), at, n);
                b.copy(slice.vkBuffer(), slice.offset(), to.vkBuffer(), to.offset() + at, n);
                value = pendingValue();
                done += n;
                uploadBytesInBatch += n;
                if (uploadBytesInBatch * 2 >= ring.capacity()) {
                    submitLocked();
                }
            }
            return new VkGpuEvent(this, value);
        }
    }

    private GpuEvent read(GpuBuffer src, long srcOffset, long size, ReadTarget target, GpuEvent[] waits) {
        VkGpuBuffer from = VkQueueChecks.ownBuffer(src, "read source", allocator);
        VkQueueChecks.requireRange(from, srcOffset, size, "read source");
        VkEventWaits.validate(waits, backend);
        VkEventWaits.flushProducers(waits, this);
        synchronized (this) {
            enter();
            uncapturable("a host read");
            applyWaits(waits);
            if (size == 0) {
                return markerLocked();
            }
            VkStagingRing ring = readbackRing();
            long chunk = chunkBytes(ring);
            long value = 0;
            for (long done = 0; done < size; ) {
                int n = (int) Math.min(chunk, size - done);
                VkStagingRing.Slice slice = ring.acquire(n, this);
                VkCommandBatch.PendingRead pending = target.pending(slice, done);
                VkCommandBatch b = batch();
                requireSliceValue(slice);
                long at = srcOffset + done;
                transferHazard(b, from.trackingId(), at, n, 0, 0, 0);
                b.copy(from.vkBuffer(), from.offset() + at, slice.vkBuffer(), slice.offset(), n);
                b.addPendingRead(pending);
                value = pendingValue();
                done += n;
                readbackBytesInBatch += n;
                if (readbackBytesInBatch * 2 >= ring.capacity()) {
                    submitLocked();
                }
            }
            return new VkGpuEvent(this, value);
        }
    }

    /** Records a barrier before a transfer when its ranges conflict with pending accesses, then records them. */
    private void transferHazard(VkCommandBatch b, long readId, long readOffset, long readSize,
                                long writeId, long writeOffset, long writeSize) {
        if (tracker.needsBarrier(readId, readOffset, readSize, writeId, writeOffset, writeSize)) {
            b.barrier();
            tracker.reset();
        }
        tracker.record(readId, readOffset, readSize, writeId, writeOffset, writeSize);
    }

    /** The batch being recorded, begun on first use; completed batches are retired before a new one is created. */
    private VkCommandBatch batch() {
        lastUseValue = submittedValue + 1;
        if (current != null) {
            return current;
        }
        VkCommandBatch b = freeBatches.pollFirst();
        if (b == null && !inFlight.isEmpty()) {
            long reached = VkTimelineWait.counter(device, timeline);
            if (reached > completedValue) {
                retireUpTo(Math.min(reached, submittedValue));
            }
            b = freeBatches.pollFirst();
        }
        if (b == null) {
            b = newBatch();
        }
        b.begin();
        tracker.reset();
        uploadBytesInBatch = 0;
        readbackBytesInBatch = 0;
        current = b;
        return b;
    }

    private VkCommandBatch newBatch() {
        VkCommandBuffer cb = VkQueueResources.allocateCommandBuffer(device, commandPool);
        VkDescriptorStrategy.Frame frame;
        try {
            frame = strategy.newFrame();
        } catch (RuntimeException e) {
            VkQueueResources.freeCommandBuffer(device, commandPool, cb);
            throw e;
        }
        VkCommandBatch b = new VkCommandBatch(cb, frame, dispatchCapable);
        allBatches.add(b);
        return b;
    }

    /** An event for the work enqueued so far (the pending batch if it holds anything, else the last submitted one). */
    private VkGpuEvent markerLocked() {
        boolean pending = current != null && !current.isEmpty();
        return new VkGpuEvent(this, pending ? pendingValue() : submittedValue);
    }

    /** Submits the current batch with its waits, signalling the next timeline value; no-op when it is empty. */
    private void submitLocked() {
        VkCommandBatch b = current;
        if (b == null || b.isEmpty()) {
            lastUseValue = submittedValue;
            return;
        }
        current = null;
        long signal = submittedValue + 1;
        // the batch orders after a pending replay, which runs outside the queue's batch sequence
        long selfWait = pendingReplaySignal;
        try {
            b.end();
            submitBatchLocked(b, signal, selfWait);
        } catch (RuntimeException e) {
            recycleUnsubmitted(b, e);
            throw e;
        }
        b.setSignalValue(signal);
        inFlight.addLast(b);
        submittedValue = signal;
        submitPolicy.onSubmit();
    }

    /**
     * Submits an ended batch, signalling {@code signal}: the batch's recorded cross-queue waits, and
     * {@code selfWaitValue} on this queue's own timeline when it is still pending (0 for none), so the
     * batch executes after that timeline point's work.
     */
    private void submitBatchLocked(VkCommandBatch b, long signal, long selfWaitValue) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkComputeQueue[] producers = b.waitProducers();
            long[] values = b.waitValues();
            LongBuffer waitSemaphores = stack.mallocLong(producers.length + 1);
            LongBuffer waitValues = stack.mallocLong(producers.length + 1);
            int waitCount = 0;
            for (int i = 0; i < producers.length; i++) {
                if (values[i] <= producers[i].completedValue) {
                    continue;
                }
                if (producers[i].closed) {
                    throw new IllegalStateException("wait on timeline value " + values[i] + " of " + producers[i]
                            + ", which closed without reaching it");
                }
                waitSemaphores.put(waitCount, producers[i].timeline);
                waitValues.put(waitCount, values[i]);
                waitCount++;
            }
            if (selfWaitValue > completedValue) {
                waitSemaphores.put(waitCount, timeline);
                waitValues.put(waitCount, selfWaitValue);
                waitCount++;
            }
            IntBuffer waitStages = null;
            if (waitCount > 0) {
                waitSemaphores.limit(waitCount);
                waitValues.limit(waitCount);
                waitStages = stack.mallocInt(waitCount);
                for (int i = 0; i < waitCount; i++) {
                    waitStages.put(i, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                }
            } else {
                waitSemaphores = null;
                waitValues = null;
            }
            VkTimelineSemaphoreSubmitInfo timelineInfo = VkTimelineSemaphoreSubmitInfo.calloc(stack)
                    .sType$Default()
                    .waitSemaphoreValueCount(waitCount)
                    .pWaitSemaphoreValues(waitValues)
                    .pSignalSemaphoreValues(stack.longs(signal));
            VkSubmitInfo info = VkSubmitInfo.calloc(stack)
                    .sType$Default()
                    .pNext(timelineInfo.address())
                    .waitSemaphoreCount(waitCount)
                    .pWaitSemaphores(waitSemaphores)
                    .pWaitDstStageMask(waitStages)
                    .pCommandBuffers(stack.pointers(b.commandBuffer()))
                    .pSignalSemaphores(stack.longs(timeline));
            context.submit(info, VK_NULL_HANDLE);
        }
    }

    /** Returns a batch whose submission failed to the free list; its readbacks never complete. */
    private void recycleUnsubmitted(VkCommandBatch b, RuntimeException cause) {
        if (timer != null) {
            timer.discard(b);
        }
        try {
            strategy.reset(b.descriptorFrame());
            b.reset();
            freeBatches.addLast(b);
        } catch (RuntimeException e) {
            cause.addSuppressed(e);
            destroyFailedBatch(b, cause);
        }
    }

    /**
     * Destroys a batch that cannot return to the free list after {@code cause}: its timing queries, descriptor
     * frame and command buffer. The batch is not executing (it completed, or its submission failed); failures of
     * the release steps are added to {@code cause}.
     */
    private void destroyFailedBatch(VkCommandBatch b, RuntimeException cause) {
        allBatches.remove(b);
        if (timer != null) {
            step(() -> timer.release(b), cause);
        }
        step(() -> strategy.destroy(b.descriptorFrame()), cause);
        b.close();
        VkQueueResources.freeCommandBuffer(device, commandPool, b.commandBuffer());
    }

    /**
     * Retires every in-flight batch and replay submission whose value is at most {@code value} (which
     * the timeline has reached): copies out its readbacks, resets its descriptor frame and command
     * buffer, and reclaims the staging space of every submission up to {@code value}. A captured
     * command graph's batch stays recorded for its replays; only its submissions retire here.
     */
    private void retireUpTo(long value) {
        while (!inFlight.isEmpty() && inFlight.peekFirst().signalValue() <= value) {
            VkCommandBatch b = inFlight.pollFirst();
            try {
                for (VkCommandBatch.PendingRead read : b.pendingReads()) {
                    read.copyOut();
                }
                if (timer != null) {
                    timer.collect(b);
                }
                strategy.reset(b.descriptorFrame());
                b.reset();
                freeBatches.addLast(b);
            } catch (RuntimeException e) {
                destroyFailedBatch(b, e);
                throw e;
            }
        }
        while (!replaySignals.isEmpty() && replaySignals.peekFirst() <= value) {
            replaySignals.pollFirst();
        }
        // a later batch must wait on the last submitted replay still outstanding, not only on the earliest
        pendingReplaySignal = replaySignals.isEmpty() ? 0 : replaySignals.peekLast();
        if (value > completedValue) {
            completedValue = value;
        }
        if (completedValue >= submittedValue) {
            submitPolicy.onDrained();
        }
        if (uploadRing != null) {
            uploadRing.releaseUpTo(value);
        }
        if (readbackRing != null) {
            readbackRing.releaseUpTo(value);
        }
        allocator.reclaimDeferred();
    }

    // ------------------------------------------------------------------ waits

    /**
     * Adds a timeline wait for each event of another queue that has not completed; the batch recorded
     * so far is submitted first so its commands do not wait.
     */
    private void applyWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (GpuEvent e : waits) {
            VkGpuEvent event = (VkGpuEvent) e;
            VkComputeQueue producer = event.owner();
            long value = event.value();
            if (producer == this || value <= producer.completedValue) {
                continue;
            }
            if (value > producer.submittedValue) {
                throw new IllegalStateException("wait event value " + value + " of " + producer
                        + " was never submitted" + (producer.closed ? " before the queue closed" : ""));
            }
            if (current != null && current.commandCount() > 0) {
                submitLocked();
            }
            batch().addWait(producer, value);
        }
    }

    // ------------------------------------------------------------------ timeline

    /** Submits the batch holding {@code value} if needed; returns true when the caller must wait on the device. */
    private boolean prepareWait(long value) {
        if (value <= completedValue) {
            return false;
        }
        synchronized (this) {
            if (value <= completedValue) {
                return false;
            }
            ensureSubmitted(value);
            return true;
        }
    }

    private void ensureSubmitted(long value) {
        if (value > submittedValue) {
            if (!closed) {
                submitLocked();
            }
            if (value > submittedValue) {
                throw new IllegalStateException("timeline value " + value + " was never submitted on " + this);
            }
        }
    }

    /**
     * True when the device has finished every submitted batch; polls the timeline when needed and
     * retires the batches that completed. Caller holds the monitor.
     */
    private boolean deviceIdleLocked() {
        if (submittedValue <= completedValue) {
            return true;
        }
        long reached = VkTimelineWait.counter(device, timeline);
        if (reached > completedValue) {
            retireUpTo(Math.min(reached, submittedValue));
        }
        return reached >= submittedValue;
    }

    // ------------------------------------------------------------------ helpers

    private VkStagingRing uploadRing() {
        if (uploadRing == null) {
            uploadRing = new VkStagingRing(allocator, true, VkStagingRing.configuredCapacity(true));
        }
        return uploadRing;
    }

    private VkStagingRing readbackRing() {
        if (readbackRing == null) {
            readbackRing = new VkStagingRing(allocator, false, VkStagingRing.configuredCapacity(false));
        }
        return readbackRing;
    }

    /** Largest transfer chunk: half the ring, a multiple of the slice alignment. */
    private static long chunkBytes(VkStagingRing ring) {
        long half = ring.capacity() / 2;
        return Math.max(VkStagingRing.ALIGNMENT, half - half % VkStagingRing.ALIGNMENT);
    }

    /** A slice must belong to the batch being recorded, or the ring would reclaim it too early. */
    private void requireSliceValue(VkStagingRing.Slice slice) {
        if (slice.batchValue() != pendingValue()) {
            throw new IllegalStateException("staging slice tagged with timeline value " + slice.batchValue()
                    + " but the recording batch signals " + pendingValue());
        }
    }

    /**
     * Checks this queue and its backend are open.
     *
     * @throws IllegalStateException when either is closed
     */
    private void enter() {
        if (closed) {
            throw new IllegalStateException("Vulkan queue " + queueType + " is closed");
        }
        if (backend.isClosed()) {
            throw new IllegalStateException("VkComputeBackend already closed");
        }
    }

    private static RuntimeException step(Runnable action, RuntimeException failure) {
        try {
            action.run();
            return failure;
        } catch (RuntimeException e) {
            LOGGER.warn("Vulkan queue close step failed: {}", e.getMessage());
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }

    /** Copies host bytes {@code [from, from + dst.remaining())} into a staging view. */
    private interface HostSource {
        void copy(long from, ByteBuffer dst);
    }

    /** Creates the readback that copies a staged chunk into the destination at byte offset {@code done}. */
    private interface ReadTarget {
        VkCommandBatch.PendingRead pending(VkStagingRing.Slice slice, long done);
    }
}

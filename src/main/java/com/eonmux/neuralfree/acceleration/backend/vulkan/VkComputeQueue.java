package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreTypeCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreWaitInfo;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkTimelineSemaphoreSubmitInfo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.VK_TIMEOUT;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkCreateCommandPool;
import static org.lwjgl.vulkan.VK10.vkCreateSemaphore;
import static org.lwjgl.vulkan.VK10.vkDestroyCommandPool;
import static org.lwjgl.vulkan.VK10.vkDestroySemaphore;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK12.VK_SEMAPHORE_TYPE_TIMELINE;
import static org.lwjgl.vulkan.VK12.vkGetSemaphoreCounterValue;
import static org.lwjgl.vulkan.VK12.vkWaitSemaphores;

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

    private static final long INFINITE_TIMEOUT = -1L;
    private static final long INTERRUPT_POLL_NANOS = 50_000_000L;

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
        this.commandPool = createCommandPool(device, context.family());
        long semaphore;
        try {
            semaphore = createTimeline(device);
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
        requireWithin(sizeInBytes, srcData.length, "write source array");
        return write(dstBuffer, dstOffset, sizeInBytes,
                (from, dst) -> dst.put(srcData, (int) from, dst.remaining()), waitOnEvents);
    }

    @Override
    public GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes, float[] srcData,
                                       GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("write source array must not be null");
        }
        requireFloatSized(sizeInBytes, srcData.length, "write source array");
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
        requireWithin(sizeInBytes, srcData.capacity(), "write source ByteBuffer");
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
        requireWithin(sizeInBytes, dstData.length, "read destination array");
        return read(srcBuffer, srcOffset, sizeInBytes,
                (slice, done) -> VkCommandBatch.PendingRead.ofBytes(slice, dstData, (int) done), waitOnEvents);
    }

    @Override
    public GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes, float[] dstData,
                                      GpuEvent... waitOnEvents) {
        if (dstData == null) {
            throw new IllegalArgumentException("read destination array must not be null");
        }
        requireFloatSized(sizeInBytes, dstData.length, "read destination array");
        return read(srcBuffer, srcOffset, sizeInBytes,
                (slice, done) -> VkCommandBatch.PendingRead.ofFloats(slice, dstData, (int) (done / Float.BYTES)),
                waitOnEvents);
    }

    @Override
    public void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset, long sizeInBytes,
                                  GpuEvent... waitOnEvents) {
        VkGpuBuffer from = ownBuffer(src, "copy source");
        VkGpuBuffer to = ownBuffer(dst, "copy destination");
        requireRange(from, srcOffset, sizeInBytes, "copy source");
        requireRange(to, dstOffset, sizeInBytes, "copy destination");
        if (from == to && srcOffset < dstOffset + sizeInBytes && dstOffset < srcOffset + sizeInBytes) {
            throw new IllegalArgumentException("copy ranges [" + srcOffset + ", " + (srcOffset + sizeInBytes)
                    + ") and [" + dstOffset + ", " + (dstOffset + sizeInBytes) + ") of one buffer overlap");
        }
        validateWaits(waitOnEvents);
        flushProducers(waitOnEvents);
        synchronized (this) {
            enter();
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
        VkGpuBuffer to = ownBuffer(dst, "fill destination");
        if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                    + " must be non-negative multiples of 4");
        }
        requireRange(to, dstOffset, sizeInBytes, "fill destination");
        validateWaits(waitOnEvents);
        flushProducers(waitOnEvents);
        synchronized (this) {
            enter();
            applyWaits(waitOnEvents);
            if (sizeInBytes == 0) {
                return;
            }
            VkCommandBatch b = batch();
            transferHazard(b, 0, 0, 0, to.trackingId(), dstOffset, sizeInBytes);
            b.fill(to.vkBuffer(), to.offset() + dstOffset, sizeInBytes, pattern);
        }
    }

    /** An event that completes when every command enqueued so far on this queue, and the waits, complete. */
    @Override
    public GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        validateWaits(waitOnEvents);
        flushProducers(waitOnEvents);
        synchronized (this) {
            enter();
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
        ownHandle(kernel).argState().set(argIndex, arg);
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
            waitTimeline(target);
        }
        synchronized (this) {
            retireUpTo(target);
        }
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
            waitTimeline(value);
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
            waitTimelineInterruptibly(value);
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
            long reached = counterValue();
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
                waitTimeline(target);
            }
            synchronized (this) {
                retireUpTo(target);
            }
        } catch (RuntimeException e) {
            failure = e;
            LOGGER.warn("Vulkan queue {} did not drain before closing: {}", this, e.getMessage());
        }
        synchronized (this) {
            for (VkCommandBatch b : allBatches) {
                failure = step(() -> strategy.destroy(b.descriptorFrame()), failure);
                failure = step(b::close, failure);
            }
            allBatches.clear();
            freeBatches.clear();
            inFlight.clear();
            current = null;
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
        VkKernelHandle handle = ownHandle(kernel);
        enter();
        if (!dispatchCapable) {
            throw new UnsupportedOperationException("Vulkan " + queueType + " queue on transfer-only family "
                    + context.family() + " cannot dispatch kernel " + handle.kernelName());
        }
        validateWaits(waits);
        int[] local3 = handle.resolveLocalSize(global, local);
        long[] groups = groupCounts(handle.kernelName(), global, local3);
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
        flushProducers(waits);
        synchronized (this) {
            enter();
            applyWaits(waits);
            VkArgState state = handle.argState();
            if (args != null && args.length > 0) {
                state.setAll(args);
            }
            state.materialize();

            VkCommandBatch b = batch();
            if (tracker.needsBarrier(state.readIds(), state.writeIds())) {
                b.barrier();
                tracker.reset();
            }
            b.bindPipeline(variant.pipeline());
            b.bindDescriptors(variant.program(), state);
            variant.program().pushConstants(b.commandBuffer(), state);
            b.dispatch((int) groups[0], (int) groups[1], (int) groups[2]);
            tracker.record(state.readIds(), state.writeIds());
            long value = pendingValue();
            if (submitPolicy.submitAfterDispatch(b.dispatchCount())) {
                submitLocked();
            }
            return wantEvent ? new VkGpuEvent(this, value) : null;
        }
    }

    /**
     * Group counts {@code ceil(global / local3)} padded to three dimensions. The count of a
     * one-dimensional dispatch ({@code global.length == 1}) above {@code maxComputeWorkGroupCount[0]}
     * is folded into the y dimension; multi-dimensional kernels read {@code gl_WorkGroupID.y} as their
     * own axis, so they are never folded.
     *
     * @throws IllegalArgumentException when a count exceeds the device limits after folding
     */
    private long[] groupCounts(String kernel, long[] global, int[] local3) {
        long[] local = new long[global.length];
        for (int d = 0; d < global.length; d++) {
            local[d] = local3[d];
        }
        long[] groups = WorkSizes.pad3(WorkSizes.groupCounts(global, local), 1);
        if (groups[0] > maxGroups[0]) {
            if (global.length != 1) {
                throw new IllegalArgumentException(kernel + ": " + groups[0] + " work-groups in x exceed "
                        + "maxComputeWorkGroupCount[0] = " + maxGroups[0] + "; only one-dimensional dispatches are "
                        + "folded (global " + Arrays.toString(global) + ", groups " + Arrays.toString(groups) + ")");
            }
            long folded = (groups[0] - 1) / maxGroups[0] + 1;
            groups[0] = maxGroups[0];
            groups[1] = folded;
        }
        for (int d = 0; d < 3; d++) {
            if (groups[d] > maxGroups[d]) {
                throw new IllegalArgumentException(kernel + ": work-group counts " + Arrays.toString(groups)
                        + " exceed maxComputeWorkGroupCount " + Arrays.toString(maxGroups));
            }
        }
        return groups;
    }

    private GpuEvent write(GpuBuffer dst, long dstOffset, long size, HostSource source, GpuEvent[] waits) {
        VkGpuBuffer to = ownBuffer(dst, "write destination");
        requireRange(to, dstOffset, size, "write destination");
        validateWaits(waits);
        flushProducers(waits);
        synchronized (this) {
            enter();
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
        VkGpuBuffer from = ownBuffer(src, "read source");
        requireRange(from, srcOffset, size, "read source");
        validateWaits(waits);
        flushProducers(waits);
        synchronized (this) {
            enter();
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
            long reached = counterValue();
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
        VkCommandBuffer cb = allocateCommandBuffer();
        VkDescriptorStrategy.Frame frame;
        try {
            frame = strategy.newFrame();
        } catch (RuntimeException e) {
            freeCommandBuffer(cb);
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
        try {
            b.end();
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkComputeQueue[] producers = b.waitProducers();
                long[] values = b.waitValues();
                LongBuffer waitSemaphores = stack.mallocLong(Math.max(1, producers.length));
                LongBuffer waitValues = stack.mallocLong(Math.max(1, producers.length));
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
        } catch (RuntimeException e) {
            recycleUnsubmitted(b, e);
            throw e;
        }
        b.setSignalValue(signal);
        inFlight.addLast(b);
        submittedValue = signal;
        submitPolicy.onSubmit();
    }

    /** Returns a batch whose submission failed to the free list; its readbacks never complete. */
    private void recycleUnsubmitted(VkCommandBatch b, RuntimeException cause) {
        try {
            strategy.reset(b.descriptorFrame());
            b.reset();
            freeBatches.addLast(b);
        } catch (RuntimeException e) {
            cause.addSuppressed(e);
            allBatches.remove(b);
            step(() -> strategy.destroy(b.descriptorFrame()), null);
            b.close();
            freeCommandBuffer(b.commandBuffer());
        }
    }

    /**
     * Retires every in-flight batch whose value is at most {@code value} (which the timeline has
     * reached): copies out its readbacks, resets its descriptor frame and command buffer, and
     * reclaims the staging space of every batch up to {@code value}.
     */
    private void retireUpTo(long value) {
        while (!inFlight.isEmpty() && inFlight.peekFirst().signalValue() <= value) {
            VkCommandBatch b = inFlight.pollFirst();
            try {
                for (VkCommandBatch.PendingRead read : b.pendingReads()) {
                    read.copyOut();
                }
                strategy.reset(b.descriptorFrame());
                b.reset();
                freeBatches.addLast(b);
            } catch (RuntimeException e) {
                allBatches.remove(b);
                step(() -> strategy.destroy(b.descriptorFrame()), null);
                b.close();
                freeCommandBuffer(b.commandBuffer());
                throw e;
            }
        }
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
     * Checks every wait is an open event of this backend.
     *
     * @throws IllegalArgumentException for a null, foreign or closed event
     */
    private void validateWaits(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (int i = 0; i < waits.length; i++) {
            GpuEvent e = waits[i];
            if (!(e instanceof VkGpuEvent)) {
                throw new IllegalArgumentException("wait event " + i + " is not a Vulkan event of this backend: "
                        + (e == null ? "null" : e.getClass().getName()));
            }
            VkGpuEvent event = (VkGpuEvent) e;
            if (event.owner().backend() != backend) {
                throw new IllegalArgumentException("wait event " + i + " belongs to another Vulkan backend");
            }
            if (event.isClosed()) {
                throw new IllegalArgumentException("wait event " + i + " was already closed");
            }
        }
    }

    /** Submits the producer batches of other queues' events; runs without this queue's monitor. */
    private void flushProducers(GpuEvent[] waits) {
        if (waits == null) {
            return;
        }
        for (GpuEvent e : waits) {
            VkGpuEvent event = (VkGpuEvent) e;
            if (event.owner() != this) {
                event.owner().flushIfPending(event.value());
            }
        }
    }

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

    private void waitTimeline(long value) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreWaitInfo info = waitInfo(stack, value);
            VkErrors.check(vkWaitSemaphores(device, info, INFINITE_TIMEOUT), "vkWaitSemaphores");
        }
    }

    private void waitTimelineInterruptibly(long value) throws InterruptedException {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreWaitInfo info = waitInfo(stack, value);
            while (true) {
                int result = vkWaitSemaphores(device, info, INTERRUPT_POLL_NANOS);
                if (result == VK_SUCCESS) {
                    return;
                }
                if (result != VK_TIMEOUT) {
                    throw VkErrors.translate(result, "vkWaitSemaphores on " + this);
                }
                if (Thread.interrupted()) {
                    throw new InterruptedException("interrupted while waiting for Vulkan timeline value " + value);
                }
            }
        }
    }

    private VkSemaphoreWaitInfo waitInfo(MemoryStack stack, long value) {
        return VkSemaphoreWaitInfo.calloc(stack)
                .sType$Default()
                .semaphoreCount(1)
                .pSemaphores(stack.longs(timeline))
                .pValues(stack.longs(value));
    }

    /**
     * True when the device has finished every submitted batch; polls the timeline when needed and
     * retires the batches that completed. Caller holds the monitor.
     */
    private boolean deviceIdleLocked() {
        if (submittedValue <= completedValue) {
            return true;
        }
        long reached = counterValue();
        if (reached > completedValue) {
            retireUpTo(Math.min(reached, submittedValue));
        }
        return reached >= submittedValue;
    }

    private long counterValue() {
        long[] out = new long[1];
        VkErrors.check(vkGetSemaphoreCounterValue(device, timeline, out), "vkGetSemaphoreCounterValue");
        return out[0];
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

    private VkCommandBuffer allocateCommandBuffer() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandBufferAllocateInfo info = VkCommandBufferAllocateInfo.calloc(stack)
                    .sType$Default()
                    .commandPool(commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1);
            PointerBuffer handle = stack.mallocPointer(1);
            VkErrors.check(vkAllocateCommandBuffers(device, info, handle), "vkAllocateCommandBuffers");
            return new VkCommandBuffer(handle.get(0), device);
        }
    }

    private void freeCommandBuffer(VkCommandBuffer cb) {
        vkFreeCommandBuffers(device, commandPool, cb);
    }

    private VkGpuBuffer ownBuffer(GpuBuffer buffer, String role) {
        if (!(buffer instanceof VkGpuBuffer)) {
            throw new IllegalArgumentException(role + " is not a Vulkan buffer: "
                    + (buffer == null ? "null" : buffer.getClass().getName()));
        }
        VkGpuBuffer vk = (VkGpuBuffer) buffer;
        if (vk.owner() != allocator) {
            throw new IllegalArgumentException(role + " " + vk + " belongs to another Vulkan backend");
        }
        if (vk.isClosed()) {
            throw new IllegalArgumentException(role + " " + vk + " is closed");
        }
        return vk;
    }

    private VkKernelHandle ownHandle(KernelHandle kernel) {
        if (!(kernel instanceof VkKernelHandle)) {
            throw new IllegalArgumentException("kernel must be a VkKernelHandle, got "
                    + (kernel == null ? "null" : kernel.getClass().getName()));
        }
        VkKernelHandle handle = (VkKernelHandle) kernel;
        if (handle.program().descriptorStrategy() != strategy) {
            throw new IllegalArgumentException("Vulkan kernel " + handle.kernelName() + " belongs to another backend");
        }
        if (handle.isClosed()) {
            throw new IllegalStateException("Vulkan kernel " + handle.kernelName() + " was used after it was closed");
        }
        return handle;
    }

    private static void requireRange(VkGpuBuffer buffer, long offset, long size, String role) {
        if (offset < 0 || size < 0 || offset > buffer.sizeInBytes() - size) {
            throw new IllegalArgumentException(role + " range [" + offset + ", " + (offset + size)
                    + ") is outside the buffer of " + buffer.sizeInBytes() + " bytes");
        }
    }

    private static void requireWithin(long size, long available, String what) {
        if (size < 0 || size > available) {
            throw new IllegalArgumentException("transfer of " + size + " bytes does not fit the " + available
                    + "-byte " + what);
        }
    }

    private static void requireFloatSized(long size, int floats, String what) {
        if (size < 0 || (size & 3) != 0 || size > (long) floats * Float.BYTES) {
            throw new IllegalArgumentException("float transfer of " + size + " bytes must be a non-negative multiple"
                    + " of 4 within the " + floats + "-float " + what);
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

    private static long createCommandPool(VkDevice device, int family) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkCommandPoolCreateInfo info = VkCommandPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                    .queueFamilyIndex(family);
            LongBuffer pool = stack.mallocLong(1);
            VkErrors.check(vkCreateCommandPool(device, info, null, pool), "vkCreateCommandPool");
            return pool.get(0);
        }
    }

    private static long createTimeline(VkDevice device) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreTypeCreateInfo type = VkSemaphoreTypeCreateInfo.calloc(stack)
                    .sType$Default()
                    .semaphoreType(VK_SEMAPHORE_TYPE_TIMELINE)
                    .initialValue(0);
            VkSemaphoreCreateInfo info = VkSemaphoreCreateInfo.calloc(stack)
                    .sType$Default()
                    .pNext(type.address());
            LongBuffer semaphore = stack.mallocLong(1);
            VkErrors.check(vkCreateSemaphore(device, info, null, semaphore), "vkCreateSemaphore(timeline)");
            return semaphore.get(0);
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

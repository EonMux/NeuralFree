package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;

import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_RESULT_64_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_RESULT_WAIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_TYPE_TIMESTAMP;
import static org.lwjgl.vulkan.VK10.vkCmdResetQueryPool;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkCmdWriteTimestamp;
import static org.lwjgl.vulkan.VK10.vkCreateQueryPool;
import static org.lwjgl.vulkan.VK10.vkDestroyQueryPool;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkGetQueryPoolResults;

/**
 * Device-side command timing of one {@link VkComputeQueue} while {@code -Dneuralfree.gpu.profile} is {@code true}
 * or {@code step} ({@link KernelTiming.Mode}), the Vulkan counterpart of the OpenCL queue's profiling events.
 *
 * <p>Each batch owns a timestamp query pool of two queries per dispatch it can hold. In
 * {@link KernelTiming.Mode#DISPATCH} a timed dispatch is preceded by the batch's global barrier and a bottom-of-pipe
 * timestamp, and followed by another one, so the pair brackets exactly that dispatch's execution: the barrier
 * serialises it behind the earlier work (the profiled step loses the overlap of independent dispatches, as an
 * in-order OpenCL queue does) and both timestamps are written once all earlier commands have completed. When the
 * batch retires, the pairs are read, converted with the device's {@code timestampPeriod} and added to the
 * {@link KernelTiming} report under {@code name#global_x x local_x}, the OpenCL queue's key. A batch that holds more
 * dispatches than its pool times only the first ones.</p>
 *
 * <p>In {@link KernelTiming.Mode#STEP} no dispatch is bracketed: each recorded batch carries one pair, written after
 * the barrier that opens it and at its end, and reports under {@link KernelTiming#SUBMISSION_KEY}, so the run keeps
 * its overlap and the report holds the device time of what it submitted. A batch keeps its pool while the queue
 * reuses it; a batch the queue destroys after a failure releases it ({@link #release}), and closing the timer
 * destroys the rest.</p>
 *
 * <p>In both modes each submission of a captured command graph, whether its captured simultaneous-use command buffer
 * or a re-recorded ring slot, is bracketed by a {@link GraphStamp} of its own: two prerecorded command buffers that
 * reset a two-query pool and write the opening timestamp, and write the closing one, submitted before and after the
 * graph's buffers in the same submission. A stamp is reused only after the submission it bracketed retired, so
 * back-to-back submissions of one captured buffer each report their own time, and a retire reads only the stamps of
 * the submissions it covers, which have completed: the read never waits on the device. Each retired submission adds
 * the graph's time under {@link KernelTiming#graphKey(int)}.</p>
 */
@NotThreadSafe("Used under the owning queue's monitor.")
final class VkDispatchTimer implements AutoCloseable {

    private final VkDevice device;
    private final KernelTiming.Mode mode;
    private final double nanosPerTick;
    private final long tickMask;
    private final int capacity;
    private final KernelTiming timing = new KernelTiming();
    private final Map<VkCommandBatch, BatchQueries> batches = new IdentityHashMap<>();
    /** The queue's command pool, which the stamps' command buffers come from. */
    private final long commandPool;
    /** Stamps whose last submission retired, ready for the next graph submission. */
    private final ArrayDeque<GraphStamp> freeStamps = new ArrayDeque<>();
    /** Stamps of submitted graphs in submission order, read as their timeline values retire. */
    private final ArrayDeque<GraphStamp> pendingStamps = new ArrayDeque<>();
    /** Every stamp this timer created, destroyed on {@link #close()}. */
    private final List<GraphStamp> allStamps = new ArrayList<>();

    /**
     * The timestamp pair of one graph submission: a two-query pool, the command buffer that resets it and writes
     * the opening timestamp, and the one that writes the closing timestamp, recorded once and resubmitted.
     */
    static final class GraphStamp {
        final long pool;
        final VkCommandBuffer open;
        final VkCommandBuffer close;
        /** The dispatches of the graph the current submission runs. */
        int dispatches;
        /** The timeline value the current submission signals. */
        long signal;

        private GraphStamp(long pool, VkCommandBuffer open, VkCommandBuffer close) {
            this.pool = pool;
            this.open = open;
            this.close = close;
        }

        /** {@code buffers} between this stamp's opening and closing command buffers, for one submission. */
        VkCommandBuffer[] around(VkCommandBuffer[] buffers) {
            VkCommandBuffer[] out = new VkCommandBuffer[buffers.length + 2];
            out[0] = open;
            System.arraycopy(buffers, 0, out, 1, buffers.length);
            out[out.length - 1] = close;
            return out;
        }
    }

    /** The queries and dispatch keys of one batch's current recording. */
    private static final class BatchQueries {
        final long pool;
        final List<String> names = new ArrayList<>();

        BatchQueries(long pool) {
            this.pool = pool;
        }
    }

    private VkDispatchTimer(VkDevice device, KernelTiming.Mode mode, double nanosPerTick, int validBits,
                            int dispatches, long commandPool) {
        this.device = device;
        this.mode = mode;
        this.commandPool = commandPool;
        this.nanosPerTick = nanosPerTick;
        this.tickMask = validBits >= 64 ? -1L : (1L << validBits) - 1;
        this.capacity = 2 * dispatches;
    }

    /**
     * The timer of a queue on {@code family} in {@code mode}, or null when the mode is {@link KernelTiming.Mode#OFF}
     * or the family writes no timestamps.
     *
     * @param dispatches  the most dispatches one batch of the queue records ({@link VkSubmitPolicy#cap()})
     * @param commandPool the queue's command pool, which outlives the timer
     */
    static VkDispatchTimer create(VkDevice device, VkPhysicalDeviceInfo info, int family, int dispatches,
                                  KernelTiming.Mode mode, long commandPool) {
        if (mode == KernelTiming.Mode.OFF) {
            return null;
        }
        int validBits = info.queueFamilies().get(family).timestampValidBits();
        if (validBits <= 0 || info.timestampPeriod() <= 0.0f) {
            return null;
        }
        return new VkDispatchTimer(device, mode, info.timestampPeriod(), validBits, Math.max(1, dispatches),
                commandPool);
    }

    /** The accumulated report. */
    KernelTiming timing() {
        return timing;
    }

    /**
     * Records the barrier and opening timestamp of a dispatch of {@code key} into {@code batch}; returns false
     * when the batch's pool is full and the dispatch goes untimed.
     */
    boolean open(VkCommandBatch batch, String key) {
        if (mode != KernelTiming.Mode.DISPATCH) {
            return false;
        }
        BatchQueries q = queries(batch);
        int slot = 2 * q.names.size();
        if (slot + 2 > capacity) {
            return false;
        }
        if (q.names.isEmpty()) {
            vkCmdResetQueryPool(batch.commandBuffer(), q.pool, 0, capacity);
        }
        batch.barrier();
        vkCmdWriteTimestamp(batch.commandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.pool, slot);
        q.names.add(key);
        return true;
    }

    /** Records the closing timestamp of the dispatch {@link #open} admitted last. */
    void close(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        vkCmdWriteTimestamp(batch.commandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.pool,
                2 * q.names.size() - 1);
    }

    /**
     * Opens the span of a batch that just began recording, in {@link KernelTiming.Mode#STEP}: the pool is reset and
     * the opening timestamp written behind the barrier that begins the batch.
     */
    void beginBatch(VkCommandBatch batch) {
        if (mode == KernelTiming.Mode.STEP) {
            openSpan(batch, KernelTiming.SUBMISSION_KEY);
        }
    }

    /** Closes the span {@link #beginBatch} opened, before the batch ends; nothing when none is open. */
    void endBatch(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        if (mode == KernelTiming.Mode.STEP && q != null && q.names.size() == 1) {
            close(batch);
        }
    }

    /**
     * The stamp that brackets the next submission of a captured graph of {@code dispatches} dispatches: one whose
     * last submission retired, or a new one. The queue hands it back with {@link #graphSubmitted} once submitted, or
     * with {@link #graphUnsubmitted} when the submission failed.
     */
    GraphStamp graphStamp(int dispatches) {
        GraphStamp stamp = freeStamps.pollFirst();
        if (stamp == null) {
            stamp = newStamp();
        }
        stamp.dispatches = dispatches;
        return stamp;
    }

    /** Notes the submission {@code stamp} brackets, read once {@code signal} retires. */
    void graphSubmitted(GraphStamp stamp, long signal) {
        stamp.signal = signal;
        pendingStamps.addLast(stamp);
    }

    /** Returns the stamp of a submission that failed; nothing it brackets reached the device. */
    void graphUnsubmitted(GraphStamp stamp) {
        freeStamps.addLast(stamp);
    }

    /**
     * Adds the graph submissions whose timeline values the queue reached to the report and frees their stamps.
     * Only completed submissions are read, so this never waits on the device.
     */
    void graphsRetired(long value) {
        while (!pendingStamps.isEmpty() && pendingStamps.peekFirst().signal <= value) {
            GraphStamp done = pendingStamps.pollFirst();
            long[] ticks = new long[2];
            VkErrors.check(vkGetQueryPoolResults(device, done.pool, 0, 2, ticks, Long.BYTES,
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
            timing.add(KernelTiming.graphKey(done.dispatches), Math.round(((ticks[1] - ticks[0]) & tickMask)
                    * nanosPerTick));
            freeStamps.addLast(done);
        }
    }

    /** A new stamp: its pool and its two command buffers, recorded once. */
    private GraphStamp newStamp() {
        long pool = createPool(2);
        VkCommandBuffer open = null;
        VkCommandBuffer close = null;
        try {
            open = VkQueueResources.allocateCommandBuffer(device, commandPool);
            close = VkQueueResources.allocateCommandBuffer(device, commandPool);
            try (MemoryStack stack = MemoryStack.stackPush()) {
                VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack).sType$Default();
                VkErrors.check(vkBeginCommandBuffer(open, begin), "vkBeginCommandBuffer");
                vkCmdResetQueryPool(open, pool, 0, 2);
                vkCmdWriteTimestamp(open, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 0);
                VkErrors.check(vkEndCommandBuffer(open), "vkEndCommandBuffer");
                VkErrors.check(vkBeginCommandBuffer(close, begin), "vkBeginCommandBuffer");
                vkCmdWriteTimestamp(close, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
                VkErrors.check(vkEndCommandBuffer(close), "vkEndCommandBuffer");
            }
        } catch (RuntimeException e) {
            if (open != null) {
                VkQueueResources.freeCommandBuffer(device, commandPool, open);
            }
            if (close != null) {
                VkQueueResources.freeCommandBuffer(device, commandPool, close);
            }
            vkDestroyQueryPool(device, pool, null);
            throw e;
        }
        GraphStamp stamp = new GraphStamp(pool, open, close);
        allStamps.add(stamp);
        return stamp;
    }

    /** Adds the timed dispatches of a completed batch to the report and clears them for its next recording. */
    void collect(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        if (q == null || q.names.isEmpty()) {
            return;
        }
        read(q);
        q.names.clear();
    }

    /** Reads the pairs of {@code q}'s recorded names into the report. */
    private void read(BatchQueries q) {
        int n = q.names.size();
        long[] ticks = new long[2 * n];
        VkErrors.check(vkGetQueryPoolResults(device, q.pool, 0, 2 * n, ticks, Long.BYTES,
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
        for (int i = 0; i < n; i++) {
            long delta = (ticks[2 * i + 1] - ticks[2 * i]) & tickMask;
            timing.add(q.names.get(i), Math.round(delta * nanosPerTick));
        }
    }

    /** Resets {@code batch}'s pool and writes the opening timestamp of a span reported under {@code key}. */
    private void openSpan(VkCommandBatch batch, String key) {
        BatchQueries q = queries(batch);
        q.names.clear();
        vkCmdResetQueryPool(batch.commandBuffer(), q.pool, 0, capacity);
        vkCmdWriteTimestamp(batch.commandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.pool, 0);
        q.names.add(key);
    }

    /** The queries of {@code batch}, its pool created on first use. */
    private BatchQueries queries(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        if (q == null) {
            q = new BatchQueries(createPool(capacity));
            batches.put(batch, q);
        }
        return q;
    }

    /** Drops the timed dispatches of a batch whose submission failed; the batch keeps its pool for reuse. */
    void discard(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        if (q != null) {
            q.names.clear();
        }
    }

    /**
     * Destroys the query pool of a batch the queue destroys after a failure, with its timed dispatches; the batch
     * must not be executing (it completed, or its submission failed).
     */
    void release(VkCommandBatch batch) {
        BatchQueries q = batches.remove(batch);
        if (q != null) {
            vkDestroyQueryPool(device, q.pool, null);
        }
    }

    /**
     * Destroys every batch's query pool and every graph stamp; the queue has drained, so nothing they hold is
     * executing, and the command pool is still alive.
     */
    @Override
    public void close() {
        for (BatchQueries q : batches.values()) {
            vkDestroyQueryPool(device, q.pool, null);
        }
        batches.clear();
        for (GraphStamp stamp : allStamps) {
            VkQueueResources.freeCommandBuffer(device, commandPool, stamp.open);
            VkQueueResources.freeCommandBuffer(device, commandPool, stamp.close);
            vkDestroyQueryPool(device, stamp.pool, null);
        }
        allStamps.clear();
        freeStamps.clear();
        pendingStamps.clear();
    }

    private long createPool(int queries) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkQueryPoolCreateInfo info = VkQueryPoolCreateInfo.calloc(stack).sType$Default()
                    .queryType(VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(queries);
            LongBuffer handle = stack.mallocLong(1);
            VkErrors.check(vkCreateQueryPool(device, info, null, handle), "vkCreateQueryPool");
            return handle.get(0);
        }
    }
}

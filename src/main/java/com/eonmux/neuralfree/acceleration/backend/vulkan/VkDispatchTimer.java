package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkQueryPoolCreateInfo;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_RESULT_64_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_RESULT_WAIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUERY_TYPE_TIMESTAMP;
import static org.lwjgl.vulkan.VK10.vkCmdResetQueryPool;
import static org.lwjgl.vulkan.VK10.vkCmdWriteTimestamp;
import static org.lwjgl.vulkan.VK10.vkCreateQueryPool;
import static org.lwjgl.vulkan.VK10.vkDestroyQueryPool;
import static org.lwjgl.vulkan.VK10.vkGetQueryPoolResults;

/**
 * Device-side dispatch timing of one {@link VkComputeQueue} while {@code -Dneuralfree.gpu.profile=true} is set
 * ({@link KernelTiming#PROFILE_PROPERTY}), the Vulkan counterpart of the OpenCL queue's profiling events.
 *
 * <p>Each batch owns a timestamp query pool of two queries per dispatch it can hold. A timed dispatch is preceded
 * by the batch's global barrier and a bottom-of-pipe timestamp, and followed by another one, so the pair brackets
 * exactly that dispatch's execution: the barrier serialises it behind the earlier work (the profiled step loses the
 * overlap of independent dispatches, as an in-order OpenCL queue does) and both timestamps are written once all
 * earlier commands have completed. When the batch retires, the pairs are read, converted with the device's
 * {@code timestampPeriod} and added to the {@link KernelTiming} report under {@code name#global_x x local_x}, the
 * OpenCL queue's key. A batch that holds more dispatches than its pool times only the first ones; captured and
 * replayed command graphs are never timed. A batch keeps its pool while the queue reuses it; a batch the queue
 * destroys after a failure releases it ({@link #release}), and closing the timer destroys the rest.</p>
 */
@NotThreadSafe("Used under the owning queue's monitor.")
final class VkDispatchTimer implements AutoCloseable {

    private final VkDevice device;
    private final double nanosPerTick;
    private final long tickMask;
    private final int capacity;
    private final KernelTiming timing = new KernelTiming();
    private final Map<VkCommandBatch, BatchQueries> batches = new IdentityHashMap<>();

    /** The queries and dispatch keys of one batch's current recording. */
    private static final class BatchQueries {
        final long pool;
        final List<String> names = new ArrayList<>();

        BatchQueries(long pool) {
            this.pool = pool;
        }
    }

    private VkDispatchTimer(VkDevice device, double nanosPerTick, int validBits, int dispatches) {
        this.device = device;
        this.nanosPerTick = nanosPerTick;
        this.tickMask = validBits >= 64 ? -1L : (1L << validBits) - 1;
        this.capacity = 2 * dispatches;
    }

    /**
     * The timer of a queue on {@code family}, or null when profiling is off or the family writes no timestamps.
     *
     * @param dispatches the most dispatches one batch of the queue records ({@link VkSubmitPolicy#cap()})
     */
    static VkDispatchTimer create(VkDevice device, VkPhysicalDeviceInfo info, int family, int dispatches) {
        if (!Boolean.getBoolean(KernelTiming.PROFILE_PROPERTY)) {
            return null;
        }
        int validBits = info.queueFamilies().get(family).timestampValidBits();
        if (validBits <= 0 || info.timestampPeriod() <= 0.0f) {
            return null;
        }
        return new VkDispatchTimer(device, info.timestampPeriod(), validBits, Math.max(1, dispatches));
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
        BatchQueries q = batches.get(batch);
        if (q == null) {
            q = new BatchQueries(createPool());
            batches.put(batch, q);
        }
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

    /** Adds the timed dispatches of a completed batch to the report and clears them for its next recording. */
    void collect(VkCommandBatch batch) {
        BatchQueries q = batches.get(batch);
        if (q == null || q.names.isEmpty()) {
            return;
        }
        int n = q.names.size();
        long[] ticks = new long[2 * n];
        VkErrors.check(vkGetQueryPoolResults(device, q.pool, 0, 2 * n, ticks, Long.BYTES,
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
        for (int i = 0; i < n; i++) {
            long delta = (ticks[2 * i + 1] - ticks[2 * i]) & tickMask;
            timing.add(q.names.get(i), Math.round(delta * nanosPerTick));
        }
        q.names.clear();
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

    /** Destroys every batch's query pool. */
    @Override
    public void close() {
        for (BatchQueries q : batches.values()) {
            vkDestroyQueryPool(device, q.pool, null);
        }
        batches.clear();
    }

    private long createPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkQueryPoolCreateInfo info = VkQueryPoolCreateInfo.calloc(stack).sType$Default()
                    .queryType(VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(capacity);
            LongBuffer handle = stack.mallocLong(1);
            VkErrors.check(vkCreateQueryPool(device, info, null, handle), "vkCreateQueryPool");
            return handle.get(0);
        }
    }
}

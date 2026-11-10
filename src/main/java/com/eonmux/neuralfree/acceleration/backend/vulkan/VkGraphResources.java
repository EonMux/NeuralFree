package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;

import java.util.function.LongSupplier;

/**
 * The command buffers and descriptor frames of one queue's captured graphs: the capture's own simultaneous-use batch,
 * the ring and frame of a graph's {@link VkGraphRecording}, and their release when the graph or the queue closes.
 * Every batch comes from the queue's command pool and every frame from the backend's descriptor strategy; a timeline
 * wait blocks on the queue's semaphore.
 */
@NotThreadSafe("Used under the owning queue's lock.")
final class VkGraphResources implements VkGraphRecording.Resources {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkGraphResources.class);

    private final VkDevice device;
    private final long commandPool;
    private final VkDescriptorStrategy strategy;
    private final boolean computeStages;
    private final long timeline;
    private final LongSupplier completedValue;
    /** Timeline waits that found the value pending and blocked on the device; see {@link #blockedWaits()}. */
    private int blockedWaits;

    /**
     * @param computeStages  whether the queue's family runs compute (the batches' barrier stages)
     * @param completedValue the queue's completed timeline value
     */
    VkGraphResources(VkDevice device, long commandPool, VkDescriptorStrategy strategy, boolean computeStages,
                     long timeline, LongSupplier completedValue) {
        this.device = device;
        this.commandPool = commandPool;
        this.strategy = strategy;
        this.computeStages = computeStages;
        this.timeline = timeline;
        this.completedValue = completedValue;
    }

    @Override
    public VkCommandBatch newBatch(VkDescriptorStrategy.Frame frame, boolean simultaneousUse) {
        return new VkCommandBatch(VkQueueResources.allocateCommandBuffer(device, commandPool), frame,
                computeStages, simultaneousUse);
    }

    @Override
    public VkDescriptorStrategy.Frame newFrame() {
        return strategy.newFrame();
    }

    @Override
    public void awaitTimeline(long value) {
        if (value > completedValue.getAsLong()) {
            blockedWaits++;
            VkTimelineWait.wait(device, timeline, value);
        }
    }

    /**
     * The timeline waits of the recordings that blocked on the device. A recording runs under the queue's lock and
     * its graph waits for the ring slot it takes before taking the lock, so the count stays 0 unless that order
     * breaks.
     */
    int blockedWaits() {
        return blockedWaits;
    }

    @Override
    public void freeBatch(VkCommandBatch batch) {
        batch.close();
        VkQueueResources.freeCommandBuffer(device, commandPool, batch.commandBuffer());
    }

    @Override
    public void destroyFrame(VkDescriptorStrategy.Frame frame) {
        strategy.destroy(frame);
    }

    /** A begun simultaneous-use batch with its own frame, which a capture records into. */
    VkCommandBatch newCaptureBatch() {
        VkCommandBuffer cb = VkQueueResources.allocateCommandBuffer(device, commandPool);
        VkDescriptorStrategy.Frame frame;
        try {
            frame = strategy.newFrame();
        } catch (RuntimeException e) {
            VkQueueResources.freeCommandBuffer(device, commandPool, cb);
            throw e;
        }
        VkCommandBatch b = new VkCommandBatch(cb, frame, computeStages, true);
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

    /** Frees a capture batch and its frame; {@code b} is not executing. Nothing for null. */
    RuntimeException freeCaptureBatch(VkCommandBatch b, RuntimeException failure) {
        if (b == null) {
            return failure;
        }
        if (b.isRecording()) {
            failure = step(b::reset, failure);
        }
        failure = step(() -> strategy.destroy(b.descriptorFrame()), failure);
        failure = step(b::close, failure);
        return step(() -> VkQueueResources.freeCommandBuffer(device, commandPool, b.commandBuffer()), failure);
    }

    /**
     * Frees a graph's re-recorded form and captured batch and marks it destroyed; its submissions have completed.
     *
     * @return {@code failure}, or the first failure of the release steps with the later ones suppressed
     */
    RuntimeException freeGraph(VkGraph graph, RuntimeException failure) {
        if (graph.recording() != null) {
            failure = step(graph.recording()::close, failure);
        }
        failure = freeCaptureBatch(graph.batch(), failure);
        graph.destroy();
        return failure;
    }

    /** Runs one release step, keeping the first failure and suppressing the later ones. */
    static RuntimeException step(Runnable action, RuntimeException failure) {
        try {
            action.run();
            return failure;
        } catch (RuntimeException e) {
            LOGGER.warn("Vulkan queue release step failed: {}", e.getMessage());
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }
}

package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkSubmitInfo;
import org.lwjgl.vulkan.VkTimelineSemaphoreSubmitInfo;

import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

/**
 * One timeline-ordered submission of a {@link VkComputeQueue}: ended command buffers, executed in order once the
 * waits they need are reached, signalling the queue's next timeline value. A wait that has already completed is
 * left out, so a submission whose producers are done carries no semaphore waits.
 */
final class VkTimelineSubmit {

    private VkTimelineSubmit() {
    }

    /**
     * Submits {@code buffers} on {@code context}, signalling {@code signal} on {@code timeline}: after the cross-queue
     * waits recorded in {@code waits} (none when null) and after {@code selfWaitValue} on {@code timeline} while it
     * is above {@code completedValue} (0 for none), so the buffers execute after that timeline point's work.
     *
     * @throws IllegalStateException when a producer queue closed without reaching the value it is waited on for
     */
    static void submit(VkQueueContext context, long timeline, long completedValue, VkCommandBatch waits,
                       VkCommandBuffer[] buffers, long signal, long selfWaitValue) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkComputeQueue[] producers = waits != null ? waits.waitProducers() : new VkComputeQueue[0];
            long[] values = waits != null ? waits.waitValues() : new long[0];
            LongBuffer waitSemaphores = stack.mallocLong(producers.length + 1);
            LongBuffer waitValues = stack.mallocLong(producers.length + 1);
            int waitCount = 0;
            for (int i = 0; i < producers.length; i++) {
                if (values[i] <= producers[i].completedValue()) {
                    continue;
                }
                if (producers[i].isClosed()) {
                    throw new IllegalStateException("wait on timeline value " + values[i] + " of " + producers[i]
                            + ", which closed without reaching it");
                }
                waitSemaphores.put(waitCount, producers[i].timelineSemaphore());
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
                    .pCommandBuffers(stack.pointers(buffers))
                    .pSignalSemaphores(stack.longs(timeline));
            context.submit(info, VK_NULL_HANDLE);
        }
    }
}

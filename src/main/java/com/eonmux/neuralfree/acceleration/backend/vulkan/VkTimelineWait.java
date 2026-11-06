package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSemaphoreWaitInfo;

import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.VK_TIMEOUT;
import static org.lwjgl.vulkan.VK12.vkGetSemaphoreCounterValue;
import static org.lwjgl.vulkan.VK12.vkWaitSemaphores;

/**
 * Host-side waits on a queue's timeline semaphore: blocking until a value is reached, an interruptible
 * wait that polls the interrupt flag, and a poll of the value the device has reached.
 */
final class VkTimelineWait {

    private static final long INFINITE_TIMEOUT = -1L;
    private static final long INTERRUPT_POLL_NANOS = 50_000_000L;

    private VkTimelineWait() {
    }

    /** Blocks until {@code timeline} reaches {@code value}. */
    static void wait(VkDevice device, long timeline, long value) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreWaitInfo info = waitInfo(stack, timeline, value);
            VkErrors.check(vkWaitSemaphores(device, info, INFINITE_TIMEOUT), "vkWaitSemaphores");
        }
    }

    /**
     * Blocks until {@code timeline} reaches {@code value}, polling so the calling thread can be
     * interrupted; {@code owner} names the queue in failures.
     *
     * @throws InterruptedException when the thread is interrupted while waiting
     */
    static void waitInterruptibly(VkDevice device, long timeline, long value, Object owner)
            throws InterruptedException {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkSemaphoreWaitInfo info = waitInfo(stack, timeline, value);
            while (true) {
                int result = vkWaitSemaphores(device, info, INTERRUPT_POLL_NANOS);
                if (result == VK_SUCCESS) {
                    return;
                }
                if (result != VK_TIMEOUT) {
                    throw VkErrors.translate(result, "vkWaitSemaphores on " + owner);
                }
                if (Thread.interrupted()) {
                    throw new InterruptedException("interrupted while waiting for Vulkan timeline value " + value);
                }
            }
        }
    }

    /** The value {@code timeline} has reached. */
    static long counter(VkDevice device, long timeline) {
        long[] out = new long[1];
        VkErrors.check(vkGetSemaphoreCounterValue(device, timeline, out), "vkGetSemaphoreCounterValue");
        return out[0];
    }

    private static VkSemaphoreWaitInfo waitInfo(MemoryStack stack, long timeline, long value) {
        return VkSemaphoreWaitInfo.calloc(stack)
                .sType$Default()
                .semaphoreCount(1)
                .pSemaphores(stack.longs(timeline))
                .pValues(stack.longs(value));
    }
}

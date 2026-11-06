package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkSemaphoreCreateInfo;
import org.lwjgl.vulkan.VkSemaphoreTypeCreateInfo;

import java.nio.LongBuffer;

import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers;
import static org.lwjgl.vulkan.VK10.vkCreateCommandPool;
import static org.lwjgl.vulkan.VK10.vkCreateSemaphore;
import static org.lwjgl.vulkan.VK10.vkFreeCommandBuffers;
import static org.lwjgl.vulkan.VK12.VK_SEMAPHORE_TYPE_TIMELINE;

/**
 * Vulkan objects one compute queue records with: its command pool, its timeline semaphore and the
 * primary command buffers allocated from the pool.
 */
final class VkQueueResources {

    private VkQueueResources() {
    }

    /** Creates a command pool whose command buffers reset individually, on the queue family {@code family}. */
    static long createCommandPool(VkDevice device, int family) {
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

    /** Creates the timeline semaphore that orders the queue's submissions, starting at value 0. */
    static long createTimeline(VkDevice device) {
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

    /** Allocates one primary command buffer from {@code commandPool}. */
    static VkCommandBuffer allocateCommandBuffer(VkDevice device, long commandPool) {
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

    /** Returns a command buffer to {@code commandPool}. */
    static void freeCommandBuffer(VkDevice device, long commandPool, VkCommandBuffer cb) {
        vkFreeCommandBuffers(device, commandPool, cb);
    }
}

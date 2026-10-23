package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkSubmitInfo;

import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkQueueSubmit;
import static org.lwjgl.vulkan.VK10.vkQueueWaitIdle;

/**
 * One {@link VkQueue} with its family and index; submissions are serialised by a
 * lock shared by every compute queue that uses the same {@code VkQueue}.
 *
 * <p>Vulkan requires external synchronisation of a {@code VkQueue}: every
 * {@code VkQueueContext} built for the same queue must be given the same
 * {@code submitLock}. Immutable and thread-safe.</p>
 */
public final class VkQueueContext {

    private final VkQueue queue;
    private final int family;
    private final int index;
    private final Object submitLock;

    /**
     * @param queue      the device queue
     * @param family     its queue family index
     * @param index      its index within the family
     * @param submitLock lock shared by every context of this {@code VkQueue}
     * @throws IllegalArgumentException when an argument is null or an index is negative
     */
    public VkQueueContext(VkQueue queue, int family, int index, Object submitLock) {
        if (queue == null) {
            throw new IllegalArgumentException("queue must not be null");
        }
        if (submitLock == null) {
            throw new IllegalArgumentException("submit lock must not be null");
        }
        if (family < 0 || index < 0) {
            throw new IllegalArgumentException("queue family and index must not be negative: family=" + family
                    + ", index=" + index);
        }
        this.queue = queue;
        this.family = family;
        this.index = index;
        this.submitLock = submitLock;
    }

    public VkQueue queue() {
        return queue;
    }

    public int family() {
        return family;
    }

    public int index() {
        return index;
    }

    /**
     * Submits under the lock.
     *
     * @param fence {@code VkFence} to signal, or {@code VK_NULL_HANDLE}
     * @throws GpuOutOfMemoryException when {@code vkQueueSubmit} reports exhausted host or device memory
     * @throws GpuBackendException     when {@code vkQueueSubmit} fails otherwise
     */
    public void submit(VkSubmitInfo info, long fence) {
        if (info == null) {
            throw new IllegalArgumentException("submit info must not be null");
        }
        int result;
        synchronized (submitLock) {
            result = vkQueueSubmit(queue, info, fence);
        }
        check(result, "vkQueueSubmit");
    }

    /**
     * Submits several batches in one call under the lock.
     *
     * @param fence {@code VkFence} to signal, or {@code VK_NULL_HANDLE}
     * @throws GpuOutOfMemoryException when {@code vkQueueSubmit} reports exhausted host or device memory
     * @throws GpuBackendException     when {@code vkQueueSubmit} fails otherwise
     */
    public void submit(VkSubmitInfo.Buffer infos, long fence) {
        if (infos == null || infos.remaining() == 0) {
            throw new IllegalArgumentException("at least one submit info is required");
        }
        int result;
        synchronized (submitLock) {
            result = vkQueueSubmit(queue, infos, fence);
        }
        check(result, "vkQueueSubmit of " + infos.remaining() + " batches");
    }

    /**
     * {@code vkQueueWaitIdle} under the lock.
     *
     * @throws GpuOutOfMemoryException when the wait reports exhausted host or device memory
     * @throws GpuBackendException     when the wait fails otherwise (for example on device loss)
     */
    public void waitIdle() {
        int result;
        synchronized (submitLock) {
            result = vkQueueWaitIdle(queue);
        }
        check(result, "vkQueueWaitIdle");
    }

    private void check(int result, String what) {
        if (result != VK_SUCCESS) {
            throw VkErrors.translate(result, what + " on queue family " + family + " index " + index);
        }
    }

    @Override
    public String toString() {
        return "VkQueueContext{family=" + family + ", index=" + index + '}';
    }
}

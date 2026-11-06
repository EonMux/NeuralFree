package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;

/**
 * Validations a Vulkan queue runs on the buffers and kernel handles it is handed: a buffer must belong to
 * the queue's allocator, a kernel to its descriptor strategy, and a transfer range to its buffer.
 */
final class VkQueueChecks {

    private VkQueueChecks() {
    }

    static VkGpuBuffer ownBuffer(GpuBuffer buffer, String role, VkMemoryAllocator allocator) {
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

    static VkKernelHandle ownHandle(KernelHandle kernel, VkDescriptorStrategy strategy) {
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

    static void requireRange(VkGpuBuffer buffer, long offset, long size, String role) {
        if (offset < 0 || size < 0 || offset > buffer.sizeInBytes() - size) {
            throw new IllegalArgumentException(role + " range [" + offset + ", " + (offset + size)
                    + ") is outside the buffer of " + buffer.sizeInBytes() + " bytes");
        }
    }

    static void requireWithin(long size, long available, String what) {
        if (size < 0 || size > available) {
            throw new IllegalArgumentException("transfer of " + size + " bytes does not fit the " + available
                    + "-byte " + what);
        }
    }

    static void requireFloatSized(long size, int floats, String what) {
        if (size < 0 || (size & 3) != 0 || size > (long) floats * Float.BYTES) {
            throw new IllegalArgumentException("float transfer of " + size + " bytes must be a non-negative multiple"
                    + " of 4 within the " + floats + "-float " + what);
        }
    }
}

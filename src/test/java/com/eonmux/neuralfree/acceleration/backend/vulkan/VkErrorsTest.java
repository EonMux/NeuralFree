package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.lwjgl.vulkan.VK10.VK_ERROR_DEVICE_LOST;
import static org.lwjgl.vulkan.VK10.VK_ERROR_OUT_OF_DEVICE_MEMORY;
import static org.lwjgl.vulkan.VK10.VK_ERROR_OUT_OF_HOST_MEMORY;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.VK_TIMEOUT;
import static org.lwjgl.vulkan.VK11.VK_ERROR_OUT_OF_POOL_MEMORY;

/**
 * Result mapping of {@link VkErrors}: the two out-of-memory results become {@link GpuOutOfMemoryException}
 * whichever call reported them, every other failure a {@link GpuBackendException} carrying the code. The result
 * codes are compile-time constants, so no Vulkan loader is needed.
 */
public class VkErrorsTest {

    @Test
    public void onlyHostAndDeviceMemoryResultsAreOutOfMemory() {
        assertTrue(VkErrors.isOutOfMemory(VK_ERROR_OUT_OF_HOST_MEMORY));
        assertTrue(VkErrors.isOutOfMemory(VK_ERROR_OUT_OF_DEVICE_MEMORY));
        assertFalse(VkErrors.isOutOfMemory(VK_ERROR_OUT_OF_POOL_MEMORY));
        assertFalse(VkErrors.isOutOfMemory(VK_ERROR_DEVICE_LOST));
        assertFalse(VkErrors.isOutOfMemory(VK_SUCCESS));
    }

    @Test
    public void successPassesTheCheck() {
        VkErrors.check(VK_SUCCESS, "vkQueueSubmit");
        VkErrors.check(VK_SUCCESS, "vkAllocateMemory", 1024);
    }

    @Test
    public void outOfMemoryFromASubmitIsTypedAsOutOfMemory() {
        GpuOutOfMemoryException e = assertThrows(GpuOutOfMemoryException.class,
                () -> VkErrors.check(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkQueueSubmit on queue family 0 index 0"));
        assertTrue(e.getMessage(), e.getMessage().contains("vkQueueSubmit on queue family 0 index 0"));
        assertTrue(e.getMessage(), e.getMessage().contains("VK_ERROR_OUT_OF_DEVICE_MEMORY"));
    }

    @Test
    public void outOfMemoryFromCommandBufferCallsIsTypedAsOutOfMemory() {
        assertThrows(GpuOutOfMemoryException.class,
                () -> VkErrors.check(VK_ERROR_OUT_OF_HOST_MEMORY, "vkBeginCommandBuffer"));
        assertThrows(GpuOutOfMemoryException.class,
                () -> VkErrors.check(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkAllocateCommandBuffers"));
    }

    @Test
    public void allocationOutOfMemoryReportsTheRequestedBytes() {
        GpuOutOfMemoryException e = assertThrows(GpuOutOfMemoryException.class,
                () -> VkErrors.check(VK_ERROR_OUT_OF_DEVICE_MEMORY, "vkAllocateMemory", 4096));
        assertEquals(4096, e.getRequestedBytes());
        assertEquals(VkErrors.BACKEND, e.getBackendType());
    }

    @Test
    public void otherFailuresCarryTheirCode() {
        GpuBackendException e = assertThrows(GpuBackendException.class,
                () -> VkErrors.check(VK_ERROR_DEVICE_LOST, "vkWaitSemaphores"));
        assertEquals(VK_ERROR_DEVICE_LOST, e.getCode());
        assertEquals(VkErrors.BACKEND, e.getBackend());
        assertTrue(e.getMessage(), e.getMessage().contains("VK_ERROR_DEVICE_LOST"));
    }

    @Test
    public void translateReturnsTheExceptionForAnAcceptedNonSuccessCaller() {
        RuntimeException e = VkErrors.translate(VK_ERROR_OUT_OF_HOST_MEMORY, "vkAllocateDescriptorSets");
        assertTrue(e instanceof GpuOutOfMemoryException);
        assertTrue(VkErrors.translate(VK_TIMEOUT, "vkWaitSemaphores") instanceof GpuBackendException);
        assertThrows(IllegalArgumentException.class, () -> VkErrors.translate(VK_SUCCESS, "vkQueueSubmit"));
    }

    @Test
    public void unknownCodesAreNamedByValue() {
        assertEquals("VK_ERROR_OUT_OF_POOL_MEMORY", VkErrors.nameOf(VK_ERROR_OUT_OF_POOL_MEMORY));
        assertEquals("VkResult -123456", VkErrors.nameOf(-123456));
    }
}

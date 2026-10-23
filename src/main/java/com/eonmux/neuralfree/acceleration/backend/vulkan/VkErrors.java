package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;

import static org.lwjgl.vulkan.VK10.VK_ERROR_DEVICE_LOST;
import static org.lwjgl.vulkan.VK10.VK_ERROR_EXTENSION_NOT_PRESENT;
import static org.lwjgl.vulkan.VK10.VK_ERROR_FEATURE_NOT_PRESENT;
import static org.lwjgl.vulkan.VK10.VK_ERROR_FORMAT_NOT_SUPPORTED;
import static org.lwjgl.vulkan.VK10.VK_ERROR_FRAGMENTED_POOL;
import static org.lwjgl.vulkan.VK10.VK_ERROR_INCOMPATIBLE_DRIVER;
import static org.lwjgl.vulkan.VK10.VK_ERROR_INITIALIZATION_FAILED;
import static org.lwjgl.vulkan.VK10.VK_ERROR_LAYER_NOT_PRESENT;
import static org.lwjgl.vulkan.VK10.VK_ERROR_MEMORY_MAP_FAILED;
import static org.lwjgl.vulkan.VK10.VK_ERROR_OUT_OF_DEVICE_MEMORY;
import static org.lwjgl.vulkan.VK10.VK_ERROR_OUT_OF_HOST_MEMORY;
import static org.lwjgl.vulkan.VK10.VK_ERROR_TOO_MANY_OBJECTS;
import static org.lwjgl.vulkan.VK10.VK_EVENT_RESET;
import static org.lwjgl.vulkan.VK10.VK_EVENT_SET;
import static org.lwjgl.vulkan.VK10.VK_INCOMPLETE;
import static org.lwjgl.vulkan.VK10.VK_NOT_READY;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.VK_TIMEOUT;
import static org.lwjgl.vulkan.VK11.VK_ERROR_INVALID_EXTERNAL_HANDLE;
import static org.lwjgl.vulkan.VK11.VK_ERROR_OUT_OF_POOL_MEMORY;
import static org.lwjgl.vulkan.VK12.VK_ERROR_FRAGMENTATION;
import static org.lwjgl.vulkan.VK12.VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS;
import static org.lwjgl.vulkan.VK12.VK_ERROR_UNKNOWN;
import static org.lwjgl.vulkan.VK13.VK_PIPELINE_COMPILE_REQUIRED;

/**
 * Maps {@code VkResult} codes to the backend's exception types, the one rule every Vulkan call of this
 * backend is checked with: {@code VK_ERROR_OUT_OF_HOST_MEMORY} and {@code VK_ERROR_OUT_OF_DEVICE_MEMORY}
 * become {@link GpuOutOfMemoryException}, every other failure a {@link GpuBackendException} carrying the
 * code. Callers branch on the type (an out-of-memory failure is retried with a smaller footprint), so the
 * mapping is the same whichever call reported the code.
 */
final class VkErrors {

    /** Backend name carried by {@link GpuBackendException} and {@link GpuOutOfMemoryException}. */
    static final String BACKEND = "Vulkan";

    private VkErrors() {
    }

    /** True for the results that report exhausted host or device memory. */
    static boolean isOutOfMemory(int result) {
        return result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    /** The specification name of a result code, for messages; unknown codes are named by their value. */
    static String nameOf(int result) {
        switch (result) {
            case VK_SUCCESS: return "VK_SUCCESS";
            case VK_NOT_READY: return "VK_NOT_READY";
            case VK_TIMEOUT: return "VK_TIMEOUT";
            case VK_EVENT_SET: return "VK_EVENT_SET";
            case VK_EVENT_RESET: return "VK_EVENT_RESET";
            case VK_INCOMPLETE: return "VK_INCOMPLETE";
            case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
            case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
            case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
            case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
            case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
            case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
            case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
            case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
            case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
            case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
            default: return "VkResult " + result;
        }
    }

    /**
     * Checks the result of a call that succeeds only with {@code VK_SUCCESS}.
     *
     * @param what the call and what it acted on, for example {@code "vkQueueSubmit on queue family 0"}
     * @throws GpuOutOfMemoryException when the result reports exhausted memory
     * @throws GpuBackendException     for every other result but {@code VK_SUCCESS}
     */
    static void check(int result, String what) {
        if (result != VK_SUCCESS) {
            throw translate(result, what, -1);
        }
    }

    /**
     * Checks the result of a call that allocates {@code requestedBytes}, which an out-of-memory failure reports.
     *
     * @throws GpuOutOfMemoryException when the result reports exhausted memory
     * @throws GpuBackendException     for every other result but {@code VK_SUCCESS}
     */
    static void check(int result, String what, long requestedBytes) {
        if (result != VK_SUCCESS) {
            throw translate(result, what, requestedBytes);
        }
    }

    /**
     * The exception for a failed result, for callers that accept more than {@code VK_SUCCESS} (a timeout, a
     * partial read) and throw only for the rest.
     *
     * @throws IllegalArgumentException when {@code result} is {@code VK_SUCCESS}
     */
    static RuntimeException translate(int result, String what) {
        return translate(result, what, -1);
    }

    /**
     * The exception for a failed result.
     *
     * @param what           the call and what it acted on
     * @param requestedBytes bytes the call allocates, reported on out-of-memory; -1 when not an allocation
     * @throws IllegalArgumentException when {@code result} is {@code VK_SUCCESS}
     */
    static RuntimeException translate(int result, String what, long requestedBytes) {
        if (result == VK_SUCCESS) {
            throw new IllegalArgumentException("VK_SUCCESS is not a failure of " + what);
        }
        String message = what + " failed: " + nameOf(result);
        if (isOutOfMemory(result)) {
            return requestedBytes >= 0
                    ? new GpuOutOfMemoryException(message, requestedBytes, BACKEND)
                    : new GpuOutOfMemoryException(message + " (backend=" + BACKEND + ")");
        }
        return new GpuBackendException(BACKEND, result, message);
    }
}

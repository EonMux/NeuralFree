package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo.Feature;

import java.util.List;
import java.util.Locale;

import static org.lwjgl.vulkan.VK10.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK12.VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;

/**
 * Builds {@link DeviceCapabilities} for a Vulkan device from its property
 * snapshot and the enabled feature set (VRAM, allocation and binding limits,
 * subgroup sizes, work-group limits, push-constant size, class, identity, features).
 *
 * <ul>
 *   <li>{@code totalVram}: {@link VkPhysicalDeviceInfo#totalDeviceLocalBytes()}.</li>
 *   <li>{@code maxAllocSize}: {@code min(maxMemoryAllocationSize, maxStorageBufferRange,
 *       maxBufferSize when Vulkan 1.3, largest device-local heap)}, further capped by the
 *       allocator's limit when one is given, so a tensor larger than one binding can address
 *       is split by the caller.</li>
 *   <li>{@code computeUnits}: shader cores when the driver reports them, else 0; the clock is 0
 *       (Vulkan does not report it).</li>
 *   <li>{@code maxWorkGroupSize}: {@code min(maxComputeWorkGroupInvocations, maxComputeWorkGroupSize[0])}.</li>
 *   <li>{@code supportsAsyncTransfer}: a dedicated transfer family exists, or the compute family has
 *       two or more queues.</li>
 *   <li>{@code driverVersion}: decoded per vendor (NVIDIA 10.8.8.6 bits, Intel on Windows 18.14 bits,
 *       otherwise the Vulkan version encoding), followed by the driver's name and build string.</li>
 *   <li>{@code supportsSubgroups}: subgroup arithmetic in compute shaders.</li>
 *   <li>Shared memory and local memory: {@code maxComputeSharedMemorySize}.</li>
 * </ul>
 */
public final class VkCapabilitiesBuilder {

    private static final int VENDOR_NVIDIA = 0x10DE;

    private VkCapabilitiesBuilder() {
    }

    /** Capabilities with {@code maxAllocSize} from the device limits alone. */
    public static DeviceCapabilities build(VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled) {
        return build(info, enabled, Long.MAX_VALUE);
    }

    /**
     * Capabilities whose {@code maxAllocSize} is also capped by {@code allocatorMaxAllocSize}
     * (the backend passes {@code VkMemoryAllocator.maxAllocSize()}, which includes the heap budget).
     *
     * @throws IllegalArgumentException when an argument is null or {@code allocatorMaxAllocSize} is not positive
     */
    public static DeviceCapabilities build(VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled,
                                           long allocatorMaxAllocSize) {
        if (info == null || enabled == null) {
            throw new IllegalArgumentException("physical device info and enabled features must not be null");
        }
        if (allocatorMaxAllocSize <= 0) {
            throw new IllegalArgumentException("allocator max allocation size must be positive, got "
                    + allocatorMaxAllocSize);
        }
        long totalVram = info.totalDeviceLocalBytes();
        long maxAlloc = Math.min(deviceMaxAllocSize(info), allocatorMaxAllocSize);
        long maxWorkGroup = Math.min(info.maxComputeWorkGroupInvocations() & 0xFFFFFFFFL,
                info.maxComputeWorkGroupSize()[0]);
        long sharedBytes = info.maxComputeSharedMemorySize() & 0xFFFFFFFFL;

        return new DeviceCapabilities.Builder()
                .deviceName(info.name())
                .totalVramBytes(totalVram)
                .maxAllocSizeBytes(maxAlloc)
                .computeUnits(info.shaderCoreCount())
                .maxClockFrequencyMhz(0)
                .maxWorkGroupSize(maxWorkGroup)
                .supportsAsyncTransfer(asyncTransfer(info))
                .driverVersion(driverVersionString(info))
                .supportsFloat64(enabled.isEnabled(Feature.SHADER_FLOAT64))
                .supportsFloat16(enabled.isEnabled(Feature.SHADER_FLOAT16))
                .supportsSubgroups(enabled.deviceFeatures().contains(DeviceFeature.SUBGROUP_ARITHMETIC))
                .maxLocalMemoryBytes(sharedBytes)
                .deviceClass(info.deviceClass())
                .identity(info.identity())
                .subgroupSize(info.subgroupSize())
                .minSubgroupSize(info.minSubgroupSize())
                .maxSubgroupSize(info.maxSubgroupSize())
                .maxWorkGroupCount(info.maxComputeWorkGroupCount())
                .maxPushConstantBytes(info.maxPushConstantsSize())
                .maxSharedMemoryBytes(sharedBytes)
                .maxStorageBufferRange(info.maxStorageBufferRange())
                .computeCapability(0)
                .features(enabled.deviceFeatures())
                .build();
    }

    /**
     * {@code min(maxMemoryAllocationSize, maxStorageBufferRange, maxBufferSize when non-zero,
     * largest device-local heap)}; the device-local term is the largest heap on an integrated GPU.
     *
     * @throws IllegalArgumentException when a required limit is missing (zero)
     */
    public static long deviceMaxAllocSize(VkPhysicalDeviceInfo info) {
        long maxMemoryAllocation = info.maxMemoryAllocationSize();
        long maxRange = info.maxStorageBufferRange();
        if (maxMemoryAllocation <= 0 || maxRange <= 0) {
            throw new IllegalArgumentException("Vulkan device '" + info.name() + "' reports no allocation limits "
                    + "(maxMemoryAllocationSize=" + maxMemoryAllocation + ", maxStorageBufferRange=" + maxRange + ")");
        }
        long limit = Math.min(maxMemoryAllocation, maxRange);
        if (info.maxBufferSize() > 0) {
            limit = Math.min(limit, info.maxBufferSize());
        }
        long largestHeap = largestDeviceHeap(info);
        return largestHeap > 0 ? Math.min(limit, largestHeap) : limit;
    }

    private static long largestDeviceHeap(VkPhysicalDeviceInfo info) {
        long[] sizes = info.memoryHeapSizes();
        int[] flags = info.memoryHeapFlags();
        boolean integrated = info.deviceClass() == DeviceClass.INTEGRATED;
        long largest = 0;
        for (int i = 0; i < sizes.length; i++) {
            boolean deviceLocal = (flags[i] & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            if (integrated || deviceLocal) {
                largest = Math.max(largest, sizes[i]);
            }
        }
        return largest;
    }

    private static boolean asyncTransfer(VkPhysicalDeviceInfo info) {
        if (info.transferQueueFamily() >= 0) {
            return true;
        }
        int compute = info.computeQueueFamily();
        List<VkPhysicalDeviceInfo.QueueFamily> families = info.queueFamilies();
        return compute >= 0 && families.get(compute).queueCount() >= 2;
    }

    /** Human-readable driver version: the vendor-decoded number, then the driver name and build string. */
    static String driverVersionString(VkPhysicalDeviceInfo info) {
        int v = info.driverVersion();
        String number;
        if (info.vendorId() == VENDOR_NVIDIA) {
            number = String.format(Locale.ROOT, "%d.%d.%d.%d", (v >>> 22) & 0x3FF, (v >>> 14) & 0xFF,
                    (v >>> 6) & 0xFF, v & 0x3F);
        } else if (info.driverId() == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) {
            number = String.format(Locale.ROOT, "%d.%d", v >>> 14, v & 0x3FFF);
        } else {
            number = String.format(Locale.ROOT, "%d.%d.%d", v >>> 22, (v >>> 12) & 0x3FF, v & 0xFFF);
        }
        String detail = (info.driverName() + " " + info.driverInfo()).trim();
        return detail.isEmpty() ? number : number + " (" + detail + ")";
    }
}

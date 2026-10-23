package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.Struct;
import org.lwjgl.vulkan.VkBaseOutStructure;
import org.lwjgl.vulkan.VkExtensionProperties;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceDriverProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceIDProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceLimits;
import org.lwjgl.vulkan.VkPhysicalDeviceMaintenance3Properties;
import org.lwjgl.vulkan.VkPhysicalDeviceMaintenance4Properties;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDevicePCIBusInfoPropertiesEXT;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties2;
import org.lwjgl.vulkan.VkPhysicalDevicePushDescriptorProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderCoreProperties2AMD;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderIntegerDotProductFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceShaderSMBuiltinsPropertiesNV;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupSizeControlFeatures;
import org.lwjgl.vulkan.VkPhysicalDeviceSubgroupSizeControlProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan11Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan13Features;
import org.lwjgl.vulkan.VkQueueFamilyProperties;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.lwjgl.vulkan.AMDShaderCoreProperties2.VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTPciBusInfo.VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;
import static org.lwjgl.vulkan.EXTSubgroupSizeControl.VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRPushDescriptor.VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRShaderIntegerDotProduct.VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
import static org.lwjgl.vulkan.NVShaderSmBuiltins.VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME;
import static org.lwjgl.vulkan.VK10.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
import static org.lwjgl.vulkan.VK10.VK_PHYSICAL_DEVICE_TYPE_CPU;
import static org.lwjgl.vulkan.VK10.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
import static org.lwjgl.vulkan.VK10.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
import static org.lwjgl.vulkan.VK10.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_GRAPHICS_BIT;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_TRANSFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkEnumerateDeviceExtensionProperties;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceFeatures;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceMemoryProperties;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceProperties;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceQueueFamilyProperties;
import static org.lwjgl.vulkan.VK11.vkGetPhysicalDeviceFeatures2;
import static org.lwjgl.vulkan.VK11.vkGetPhysicalDeviceProperties2;
import static org.lwjgl.vulkan.VK12.VK_API_VERSION_1_2;
import static org.lwjgl.vulkan.VK13.VK_API_VERSION_1_3;

/**
 * Immutable snapshot of one Vulkan physical device: properties, limits, memory
 * heaps and types, queue families, extensions, subgroup properties and the
 * feature bits relevant to compute, plus its cross-API identity.
 *
 * <p>Devices at Vulkan 1.2 or newer are queried through the {@code *2} entry points:
 * properties with the ID, subgroup, maintenance-3, driver and (when available)
 * maintenance-4, subgroup-size-control, PCI-bus-info, push-descriptor, NV SM-builtins
 * and AMD shader-core-2 structs; features with the Vulkan 1.1/1.2 (and 1.3) feature
 * structs. Devices below 1.2 cannot run the compute backend; they carry only their
 * Vulkan 1.0 properties, so ranking can name them and exclude them.</p>
 *
 * <p>Unsigned 32-bit limits are widened: limits returned as {@code long} are exact,
 * limits returned as {@code int} are clamped to {@link Integer#MAX_VALUE}.</p>
 */
public final class VkPhysicalDeviceInfo {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkPhysicalDeviceInfo.class);

    /** Mask that removes the patch number from a packed Vulkan version. */
    private static final int VERSION_NO_PATCH_MASK = 0xFFFFF000;
    private static final int UUID_BYTES = 16;

    /** Device features the compute backend may enable, as reported by {@code vkGetPhysicalDeviceFeatures2}. */
    public enum Feature {
        SHADER_INT64,
        SHADER_INT16,
        SHADER_FLOAT64,
        STORAGE_BUFFER_16BIT_ACCESS,
        UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS,
        STORAGE_BUFFER_8BIT_ACCESS,
        UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS,
        SHADER_INT8,
        SHADER_FLOAT16,
        TIMELINE_SEMAPHORE,
        SUBGROUP_BROADCAST_DYNAMIC_ID,
        SUBGROUP_SIZE_CONTROL,
        COMPUTE_FULL_SUBGROUPS,
        MAINTENANCE4,
        SHADER_INTEGER_DOT_PRODUCT
    }

    /** Immutable description of one queue family. */
    public static final class QueueFamily {
        private final int index;
        private final int flags;
        private final int queueCount;
        private final int timestampValidBits;

        public QueueFamily(int index, int flags, int queueCount, int timestampValidBits) {
            this.index = index;
            this.flags = flags;
            this.queueCount = queueCount;
            this.timestampValidBits = timestampValidBits;
        }

        public int index() {
            return index;
        }

        /** {@code VkQueueFlags} bits. */
        public int flags() {
            return flags;
        }

        public int queueCount() {
            return queueCount;
        }

        public int timestampValidBits() {
            return timestampValidBits;
        }

        public boolean has(int queueFlagBit) {
            return (flags & queueFlagBit) != 0;
        }
    }

    private final VkPhysicalDevice handle;
    private final String name;
    private final int vendorId;
    private final int deviceId;
    private final int driverVersion;
    private final int apiVersion;
    private final int driverId;
    private final String driverName;
    private final String driverInfo;
    private final DeviceClass deviceClass;
    private final DeviceIdentity identity;
    private final byte[] pipelineCacheUuid;
    private final byte[] deviceUuid;
    private final Set<String> extensions;
    private final Set<Feature> features;
    private final List<QueueFamily> queueFamilies;
    private final int computeQueueFamily;
    private final int transferQueueFamily;
    private final long[] memoryHeapSizes;
    private final int[] memoryHeapFlags;
    private final int[] memoryTypePropertyFlags;
    private final int[] memoryTypeHeapIndices;
    private final long maxStorageBufferRange;
    private final int maxPushConstantsSize;
    private final long[] maxComputeWorkGroupCount;
    private final int maxComputeWorkGroupInvocations;
    private final long[] maxComputeWorkGroupSize;
    private final int maxComputeSharedMemorySize;
    private final int maxPerStageDescriptorStorageBuffers;
    private final long minStorageBufferOffsetAlignment;
    private final long nonCoherentAtomSize;
    private final long maxMemoryAllocationSize;
    private final int maxMemoryAllocationCount;
    private final long maxBufferSize;
    private final int subgroupSize;
    private final int subgroupSupportedStages;
    private final int subgroupSupportedOperations;
    private final int minSubgroupSize;
    private final int maxSubgroupSize;
    private final int maxPushDescriptors;
    private final int shaderCoreCount;

    private VkPhysicalDeviceInfo(Snapshot s) {
        this.handle = s.handle;
        this.name = s.name;
        this.vendorId = s.vendorId;
        this.deviceId = s.deviceId;
        this.driverVersion = s.driverVersion;
        this.apiVersion = s.apiVersion;
        this.driverId = s.driverId;
        this.driverName = s.driverName;
        this.driverInfo = s.driverInfo;
        this.deviceClass = s.deviceClass;
        this.identity = s.identity;
        this.pipelineCacheUuid = s.pipelineCacheUuid.clone();
        this.deviceUuid = s.deviceUuid == null ? null : s.deviceUuid.clone();
        this.extensions = Collections.unmodifiableSet(new LinkedHashSet<>(s.extensions));
        this.features = Collections.unmodifiableSet(EnumSet.copyOf(s.features));
        this.queueFamilies = Collections.unmodifiableList(new ArrayList<>(s.queueFamilies));
        this.computeQueueFamily = selectComputeFamily(queueFamilies);
        this.transferQueueFamily = selectTransferFamily(queueFamilies);
        this.memoryHeapSizes = s.memoryHeapSizes.clone();
        this.memoryHeapFlags = s.memoryHeapFlags.clone();
        this.memoryTypePropertyFlags = s.memoryTypePropertyFlags.clone();
        this.memoryTypeHeapIndices = s.memoryTypeHeapIndices.clone();
        this.maxStorageBufferRange = s.maxStorageBufferRange;
        this.maxPushConstantsSize = s.maxPushConstantsSize;
        this.maxComputeWorkGroupCount = s.maxComputeWorkGroupCount.clone();
        this.maxComputeWorkGroupInvocations = s.maxComputeWorkGroupInvocations;
        this.maxComputeWorkGroupSize = s.maxComputeWorkGroupSize.clone();
        this.maxComputeSharedMemorySize = s.maxComputeSharedMemorySize;
        this.maxPerStageDescriptorStorageBuffers = s.maxPerStageDescriptorStorageBuffers;
        this.minStorageBufferOffsetAlignment = s.minStorageBufferOffsetAlignment;
        this.nonCoherentAtomSize = s.nonCoherentAtomSize;
        this.maxMemoryAllocationSize = s.maxMemoryAllocationSize;
        this.maxMemoryAllocationCount = s.maxMemoryAllocationCount;
        this.maxBufferSize = s.maxBufferSize;
        this.subgroupSize = s.subgroupSize;
        this.subgroupSupportedStages = s.subgroupSupportedStages;
        this.subgroupSupportedOperations = s.subgroupSupportedOperations;
        this.minSubgroupSize = s.minSubgroupSize;
        this.maxSubgroupSize = s.maxSubgroupSize;
        this.maxPushDescriptors = s.maxPushDescriptors;
        this.shaderCoreCount = s.shaderCoreCount;
    }

    /**
     * Queries every property and feature of {@code device} through the {@code *2} entry points.
     *
     * @param instanceApiVersion API version the instance was created with; caps the device version
     * @throws IllegalArgumentException when {@code device} is null
     * @throws IllegalStateException    when enumerating the device extensions fails
     */
    public static VkPhysicalDeviceInfo query(VkPhysicalDevice device, int instanceApiVersion) {
        if (device == null) {
            throw new IllegalArgumentException("physical device must not be null");
        }
        Snapshot s = new Snapshot();
        s.handle = device;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(stack);
            vkGetPhysicalDeviceProperties(device, props);
            s.extensions = queryExtensions(device, stack);
            readCoreProperties(s, props, instanceApiVersion);
            readMemory(s, device, stack);
            readQueueFamilies(s, device, stack);
            if (Integer.compareUnsigned(s.apiVersion, VK_API_VERSION_1_2) >= 0) {
                readProperties2(s, device, stack);
                readFeatures2(s, device, stack);
            } else {
                readCoreFeatures(s, device, stack);
            }
        }
        int identityVendor = s.vendorId < 0 ? DeviceIdentity.UNKNOWN_VENDOR : s.vendorId;
        s.identity = new DeviceIdentity(s.name, identityVendor, s.pciBusId,
                s.deviceUuid == null ? null : DeviceIdentity.hexUuid(s.deviceUuid));
        return new VkPhysicalDeviceInfo(s);
    }

    /** Raw handle, kept for device creation. */
    public VkPhysicalDevice handle() {
        return handle;
    }

    public String name() {
        return name;
    }

    public int vendorId() {
        return vendorId;
    }

    public int deviceId() {
        return deviceId;
    }

    /** Raw {@code driverVersion}; vendor-specific encoding. */
    public int driverVersion() {
        return driverVersion;
    }

    /** Device API version, capped at the instance API version (patch number cleared). */
    public int apiVersion() {
        return apiVersion;
    }

    /** True when the device supports at least {@code version} (a packed {@code VK_API_VERSION_*}). */
    public boolean apiAtLeast(int version) {
        return Integer.compareUnsigned(apiVersion, version) >= 0;
    }

    /** {@code VkDriverId}, or 0 when the device is below Vulkan 1.2. */
    public int driverId() {
        return driverId;
    }

    /** Driver name from {@code VkPhysicalDeviceDriverProperties}; empty when unknown. */
    public String driverName() {
        return driverName;
    }

    /** Driver build information from {@code VkPhysicalDeviceDriverProperties}; empty when unknown. */
    public String driverInfo() {
        return driverInfo;
    }

    public DeviceClass deviceClass() {
        return deviceClass;
    }

    public DeviceIdentity identity() {
        return identity;
    }

    /** {@code pipelineCacheUUID} (copy, 16 bytes). */
    public byte[] pipelineCacheUuid() {
        return pipelineCacheUuid.clone();
    }

    /** {@code deviceUUID} (copy, 16 bytes), or null when the device is below Vulkan 1.2. */
    public byte[] deviceUuid() {
        return deviceUuid == null ? null : deviceUuid.clone();
    }

    /** Unmodifiable set of supported device extension names. */
    public Set<String> extensions() {
        return extensions;
    }

    public boolean hasExtension(String extensionName) {
        return extensions.contains(extensionName);
    }

    public boolean supports(Feature feature) {
        return features.contains(feature);
    }

    /** Unmodifiable list of queue families, indexed by family index. */
    public List<QueueFamily> queueFamilies() {
        return queueFamilies;
    }

    /**
     * The compute queue family: the first family with COMPUTE, replaced by the first
     * compute family without GRAPHICS that has at least as many queues; -1 when the
     * device has no compute queue.
     */
    public int computeQueueFamily() {
        return computeQueueFamily;
    }

    /** A dedicated transfer family (TRANSFER without COMPUTE or GRAPHICS), or -1 when there is none. */
    public int transferQueueFamily() {
        return transferQueueFamily;
    }

    /**
     * Device memory in bytes: the sum of the DEVICE_LOCAL heaps, or the largest heap on an
     * integrated GPU (whose DEVICE_LOCAL heaps alias system memory).
     */
    public long totalDeviceLocalBytes() {
        long sum = 0;
        long largest = 0;
        for (int i = 0; i < memoryHeapSizes.length; i++) {
            largest = Math.max(largest, memoryHeapSizes[i]);
            if ((memoryHeapFlags[i] & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                sum += memoryHeapSizes[i];
            }
        }
        return deviceClass == DeviceClass.INTEGRATED ? largest : sum;
    }

    /** Size of each memory heap in bytes (copy). */
    public long[] memoryHeapSizes() {
        return memoryHeapSizes.clone();
    }

    /** {@code VkMemoryHeapFlags} of each heap (copy). */
    public int[] memoryHeapFlags() {
        return memoryHeapFlags.clone();
    }

    /** {@code VkMemoryPropertyFlags} of each memory type (copy). */
    public int[] memoryTypePropertyFlags() {
        return memoryTypePropertyFlags.clone();
    }

    /** Heap index of each memory type (copy). */
    public int[] memoryTypeHeapIndices() {
        return memoryTypeHeapIndices.clone();
    }

    /** {@code maxStorageBufferRange}, widened unsigned. */
    public long maxStorageBufferRange() {
        return maxStorageBufferRange;
    }

    public int maxPushConstantsSize() {
        return maxPushConstantsSize;
    }

    /** {@code maxComputeWorkGroupCount} (copy, three entries, widened unsigned). */
    public long[] maxComputeWorkGroupCount() {
        return maxComputeWorkGroupCount.clone();
    }

    public int maxComputeWorkGroupInvocations() {
        return maxComputeWorkGroupInvocations;
    }

    /** {@code maxComputeWorkGroupSize} (copy, three entries, widened unsigned). */
    public long[] maxComputeWorkGroupSize() {
        return maxComputeWorkGroupSize.clone();
    }

    public int maxComputeSharedMemorySize() {
        return maxComputeSharedMemorySize;
    }

    public int maxPerStageDescriptorStorageBuffers() {
        return maxPerStageDescriptorStorageBuffers;
    }

    public long minStorageBufferOffsetAlignment() {
        return minStorageBufferOffsetAlignment;
    }

    public long nonCoherentAtomSize() {
        return nonCoherentAtomSize;
    }

    /** {@code maxMemoryAllocationSize} (Maintenance3), or 0 when the device is below Vulkan 1.2. */
    public long maxMemoryAllocationSize() {
        return maxMemoryAllocationSize;
    }

    public int maxMemoryAllocationCount() {
        return maxMemoryAllocationCount;
    }

    /** {@code maxBufferSize} (Maintenance4), or 0 when the device is below Vulkan 1.3. */
    public long maxBufferSize() {
        return maxBufferSize;
    }

    /** Default subgroup size; 0 when the device is below Vulkan 1.2. */
    public int subgroupSize() {
        return subgroupSize;
    }

    /** {@code VkShaderStageFlags} where subgroup operations are supported. */
    public int subgroupSupportedStages() {
        return subgroupSupportedStages;
    }

    /** {@code VkSubgroupFeatureFlags} of supported subgroup operations. */
    public int subgroupSupportedOperations() {
        return subgroupSupportedOperations;
    }

    /** Minimum subgroup size under subgroup size control, or the default size without it. */
    public int minSubgroupSize() {
        return minSubgroupSize;
    }

    /** Maximum subgroup size under subgroup size control, or the default size without it. */
    public int maxSubgroupSize() {
        return maxSubgroupSize;
    }

    /** {@code maxPushDescriptors}, or 0 when {@code VK_KHR_push_descriptor} is absent. */
    public int maxPushDescriptors() {
        return maxPushDescriptors;
    }

    /**
     * Shader cores (NVIDIA SMs from {@code VK_NV_shader_sm_builtins}, AMD active compute
     * units from {@code VK_AMD_shader_core_properties2}); 0 when the driver does not say.
     */
    public int shaderCoreCount() {
        return shaderCoreCount;
    }

    /** The API version as {@code major.minor}. */
    public String apiVersionString() {
        return (apiVersion >>> 22) + "." + ((apiVersion >>> 12) & 0x3FF);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "VkPhysicalDeviceInfo{name='%s', api=%s, class=%s, vendor=0x%04x, "
                        + "device=0x%04x, driver='%s %s', vram=%d MB, cores=%d, subgroup=%d}",
                name, apiVersionString(), deviceClass, vendorId, deviceId, driverName, driverInfo,
                totalDeviceLocalBytes() / (1024 * 1024), shaderCoreCount, subgroupSize);
    }

    // ---------------------------------------------------------------- queue families

    static int selectComputeFamily(List<QueueFamily> families) {
        QueueFamily first = null;
        for (QueueFamily f : families) {
            if (f.has(VK_QUEUE_COMPUTE_BIT) && f.queueCount() > 0) {
                first = f;
                break;
            }
        }
        if (first == null) {
            return -1;
        }
        for (QueueFamily f : families) {
            if (f.has(VK_QUEUE_COMPUTE_BIT) && !f.has(VK_QUEUE_GRAPHICS_BIT) && f.queueCount() >= first.queueCount()) {
                return f.index();
            }
        }
        return first.index();
    }

    static int selectTransferFamily(List<QueueFamily> families) {
        for (QueueFamily f : families) {
            if (f.has(VK_QUEUE_TRANSFER_BIT) && !f.has(VK_QUEUE_COMPUTE_BIT) && !f.has(VK_QUEUE_GRAPHICS_BIT)
                    && f.queueCount() > 0) {
                return f.index();
            }
        }
        return -1;
    }

    // ---------------------------------------------------------------- queries

    private static Set<String> queryExtensions(VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        int result = vkEnumerateDeviceExtensionProperties(device, (CharSequence) null, count, null);
        if (result != VK_SUCCESS) {
            throw new IllegalStateException("vkEnumerateDeviceExtensionProperties failed with " + result);
        }
        Set<String> names = new LinkedHashSet<>();
        int n = count.get(0);
        if (n == 0) {
            return names;
        }
        VkExtensionProperties.Buffer props = VkExtensionProperties.malloc(n);
        try {
            result = vkEnumerateDeviceExtensionProperties(device, (CharSequence) null, count, props);
            if (result != VK_SUCCESS) {
                throw new IllegalStateException("vkEnumerateDeviceExtensionProperties failed with " + result);
            }
            for (int i = 0; i < count.get(0); i++) {
                names.add(props.get(i).extensionNameString());
            }
        } finally {
            props.free();
        }
        return names;
    }

    private static void readCoreProperties(Snapshot s, VkPhysicalDeviceProperties props, int instanceApiVersion) {
        s.name = props.deviceNameString();
        s.vendorId = props.vendorID();
        s.deviceId = props.deviceID();
        s.driverVersion = props.driverVersion();
        int deviceApi = props.apiVersion() & VERSION_NO_PATCH_MASK;
        int instanceApi = instanceApiVersion & VERSION_NO_PATCH_MASK;
        s.apiVersion = Integer.compareUnsigned(deviceApi, instanceApi) <= 0 ? deviceApi : instanceApi;
        s.deviceClass = classOf(props.deviceType());
        s.pipelineCacheUuid = new byte[UUID_BYTES];
        for (int i = 0; i < UUID_BYTES; i++) {
            s.pipelineCacheUuid[i] = props.pipelineCacheUUID(i);
        }
        VkPhysicalDeviceLimits limits = props.limits();
        s.maxStorageBufferRange = u32(limits.maxStorageBufferRange());
        s.maxPushConstantsSize = clampU32(limits.maxPushConstantsSize());
        s.maxComputeWorkGroupInvocations = clampU32(limits.maxComputeWorkGroupInvocations());
        s.maxComputeSharedMemorySize = clampU32(limits.maxComputeSharedMemorySize());
        s.maxPerStageDescriptorStorageBuffers = clampU32(limits.maxPerStageDescriptorStorageBuffers());
        s.maxMemoryAllocationCount = clampU32(limits.maxMemoryAllocationCount());
        s.minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment();
        s.nonCoherentAtomSize = limits.nonCoherentAtomSize();
        for (int d = 0; d < 3; d++) {
            s.maxComputeWorkGroupCount[d] = u32(limits.maxComputeWorkGroupCount(d));
            s.maxComputeWorkGroupSize[d] = u32(limits.maxComputeWorkGroupSize(d));
        }
    }

    private static void readMemory(Snapshot s, VkPhysicalDevice device, MemoryStack stack) {
        VkPhysicalDeviceMemoryProperties mem = VkPhysicalDeviceMemoryProperties.calloc(stack);
        vkGetPhysicalDeviceMemoryProperties(device, mem);
        int heaps = mem.memoryHeapCount();
        s.memoryHeapSizes = new long[heaps];
        s.memoryHeapFlags = new int[heaps];
        for (int i = 0; i < heaps; i++) {
            s.memoryHeapSizes[i] = mem.memoryHeaps(i).size();
            s.memoryHeapFlags[i] = mem.memoryHeaps(i).flags();
        }
        int types = mem.memoryTypeCount();
        s.memoryTypePropertyFlags = new int[types];
        s.memoryTypeHeapIndices = new int[types];
        for (int i = 0; i < types; i++) {
            s.memoryTypePropertyFlags[i] = mem.memoryTypes(i).propertyFlags();
            s.memoryTypeHeapIndices[i] = mem.memoryTypes(i).heapIndex();
        }
    }

    private static void readQueueFamilies(Snapshot s, VkPhysicalDevice device, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, null);
        int n = count.get(0);
        s.queueFamilies = new ArrayList<>(n);
        if (n == 0) {
            return;
        }
        VkQueueFamilyProperties.Buffer props = VkQueueFamilyProperties.calloc(n, stack);
        vkGetPhysicalDeviceQueueFamilyProperties(device, count, props);
        for (int i = 0; i < count.get(0); i++) {
            VkQueueFamilyProperties p = props.get(i);
            s.queueFamilies.add(new QueueFamily(i, p.queueFlags(), p.queueCount(), p.timestampValidBits()));
        }
    }

    /** Properties chain: Properties2 → ID → Subgroup → Maintenance3 → Driver → optional structs in listed order. */
    private static void readProperties2(Snapshot s, VkPhysicalDevice device, MemoryStack stack) {
        boolean v13 = Integer.compareUnsigned(s.apiVersion, VK_API_VERSION_1_3) >= 0;
        boolean sizeControl = v13 || s.extensions.contains(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        boolean pciInfo = s.extensions.contains(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
        boolean pushDescriptor = s.extensions.contains(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        boolean smBuiltins = s.extensions.contains(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME);
        boolean amdCore2 = s.extensions.contains(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME);

        VkPhysicalDeviceProperties2 props2 = VkPhysicalDeviceProperties2.calloc(stack).sType$Default();
        VkPhysicalDeviceIDProperties id = VkPhysicalDeviceIDProperties.calloc(stack).sType$Default();
        VkPhysicalDeviceSubgroupProperties subgroup = VkPhysicalDeviceSubgroupProperties.calloc(stack).sType$Default();
        VkPhysicalDeviceMaintenance3Properties maint3 =
                VkPhysicalDeviceMaintenance3Properties.calloc(stack).sType$Default();
        VkPhysicalDeviceDriverProperties driver = VkPhysicalDeviceDriverProperties.calloc(stack).sType$Default();
        VkPhysicalDeviceMaintenance4Properties maint4 =
                VkPhysicalDeviceMaintenance4Properties.calloc(stack).sType$Default();
        VkPhysicalDeviceSubgroupSizeControlProperties sizeProps =
                VkPhysicalDeviceSubgroupSizeControlProperties.calloc(stack).sType$Default();
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = VkPhysicalDevicePCIBusInfoPropertiesEXT.calloc(stack).sType$Default();
        VkPhysicalDevicePushDescriptorProperties push =
                VkPhysicalDevicePushDescriptorProperties.calloc(stack).sType$Default();
        VkPhysicalDeviceShaderSMBuiltinsPropertiesNV sm =
                VkPhysicalDeviceShaderSMBuiltinsPropertiesNV.calloc(stack).sType$Default();
        VkPhysicalDeviceShaderCoreProperties2AMD amd =
                VkPhysicalDeviceShaderCoreProperties2AMD.calloc(stack).sType$Default();

        List<Struct<?>> chain = new ArrayList<>();
        chain.add(props2);
        chain.add(id);
        chain.add(subgroup);
        chain.add(maint3);
        chain.add(driver);
        if (v13) {
            chain.add(maint4);
        }
        if (sizeControl) {
            chain.add(sizeProps);
        }
        if (pciInfo) {
            chain.add(pci);
        }
        if (pushDescriptor) {
            chain.add(push);
        }
        if (smBuiltins) {
            chain.add(sm);
        }
        if (amdCore2) {
            chain.add(amd);
        }
        link(chain);
        vkGetPhysicalDeviceProperties2(device, props2);

        s.deviceUuid = new byte[UUID_BYTES];
        for (int i = 0; i < UUID_BYTES; i++) {
            s.deviceUuid[i] = id.deviceUUID(i);
        }
        s.subgroupSize = subgroup.subgroupSize();
        s.subgroupSupportedStages = subgroup.supportedStages();
        s.subgroupSupportedOperations = subgroup.supportedOperations();
        s.minSubgroupSize = sizeControl ? sizeProps.minSubgroupSize() : s.subgroupSize;
        s.maxSubgroupSize = sizeControl ? sizeProps.maxSubgroupSize() : s.subgroupSize;
        s.maxMemoryAllocationSize = maint3.maxMemoryAllocationSize();
        s.driverId = driver.driverID();
        s.driverName = driver.driverNameString().trim();
        s.driverInfo = driver.driverInfoString().trim();
        s.maxBufferSize = v13 ? maint4.maxBufferSize() : 0;
        s.maxPushDescriptors = pushDescriptor ? clampU32(push.maxPushDescriptors()) : 0;
        if (pciInfo) {
            s.pciBusId = pciBusId(s.name, pci);
        }
        if (smBuiltins) {
            s.shaderCoreCount = clampU32(sm.shaderSMCount());
        } else if (amdCore2) {
            s.shaderCoreCount = clampU32(amd.activeComputeUnitCount());
        }
    }

    /** Features chain: Features2 → Vulkan11 → Vulkan12 → Vulkan13 (1.3) or the 1.2 extension structs. */
    private static void readFeatures2(Snapshot s, VkPhysicalDevice device, MemoryStack stack) {
        boolean v13 = Integer.compareUnsigned(s.apiVersion, VK_API_VERSION_1_3) >= 0;
        boolean sizeControlExt = !v13 && s.extensions.contains(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        boolean dotProductExt = !v13 && s.extensions.contains(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);

        VkPhysicalDeviceFeatures2 features2 = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
        VkPhysicalDeviceVulkan11Features v11 = VkPhysicalDeviceVulkan11Features.calloc(stack).sType$Default();
        VkPhysicalDeviceVulkan12Features v12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default();
        VkPhysicalDeviceVulkan13Features v13f = VkPhysicalDeviceVulkan13Features.calloc(stack).sType$Default();
        VkPhysicalDeviceSubgroupSizeControlFeatures sizeControl =
                VkPhysicalDeviceSubgroupSizeControlFeatures.calloc(stack).sType$Default();
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot =
                VkPhysicalDeviceShaderIntegerDotProductFeatures.calloc(stack).sType$Default();

        List<Struct<?>> chain = new ArrayList<>();
        chain.add(features2);
        chain.add(v11);
        chain.add(v12);
        if (v13) {
            chain.add(v13f);
        }
        if (sizeControlExt) {
            chain.add(sizeControl);
        }
        if (dotProductExt) {
            chain.add(dot);
        }
        link(chain);
        vkGetPhysicalDeviceFeatures2(device, features2);

        readCoreFeatureBits(s, features2.features());
        put(s, Feature.STORAGE_BUFFER_16BIT_ACCESS, v11.storageBuffer16BitAccess());
        put(s, Feature.UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS, v11.uniformAndStorageBuffer16BitAccess());
        put(s, Feature.STORAGE_BUFFER_8BIT_ACCESS, v12.storageBuffer8BitAccess());
        put(s, Feature.UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS, v12.uniformAndStorageBuffer8BitAccess());
        put(s, Feature.SHADER_INT8, v12.shaderInt8());
        put(s, Feature.SHADER_FLOAT16, v12.shaderFloat16());
        put(s, Feature.TIMELINE_SEMAPHORE, v12.timelineSemaphore());
        put(s, Feature.SUBGROUP_BROADCAST_DYNAMIC_ID, v12.subgroupBroadcastDynamicId());
        if (v13) {
            put(s, Feature.SUBGROUP_SIZE_CONTROL, v13f.subgroupSizeControl());
            put(s, Feature.COMPUTE_FULL_SUBGROUPS, v13f.computeFullSubgroups());
            put(s, Feature.MAINTENANCE4, v13f.maintenance4());
            put(s, Feature.SHADER_INTEGER_DOT_PRODUCT, v13f.shaderIntegerDotProduct());
        }
        if (sizeControlExt) {
            put(s, Feature.SUBGROUP_SIZE_CONTROL, sizeControl.subgroupSizeControl());
            put(s, Feature.COMPUTE_FULL_SUBGROUPS, sizeControl.computeFullSubgroups());
        }
        if (dotProductExt) {
            put(s, Feature.SHADER_INTEGER_DOT_PRODUCT, dot.shaderIntegerDotProduct());
        }
    }

    private static void readCoreFeatures(Snapshot s, VkPhysicalDevice device, MemoryStack stack) {
        VkPhysicalDeviceFeatures core = VkPhysicalDeviceFeatures.calloc(stack);
        vkGetPhysicalDeviceFeatures(device, core);
        readCoreFeatureBits(s, core);
    }

    private static void readCoreFeatureBits(Snapshot s, VkPhysicalDeviceFeatures core) {
        put(s, Feature.SHADER_INT64, core.shaderInt64());
        put(s, Feature.SHADER_INT16, core.shaderInt16());
        put(s, Feature.SHADER_FLOAT64, core.shaderFloat64());
    }

    private static void put(Snapshot s, Feature feature, boolean supported) {
        if (supported) {
            s.features.add(feature);
        }
    }

    /** Links {@code structs} so each one's {@code pNext} points at the next, in list order. */
    private static void link(List<Struct<?>> structs) {
        for (int i = 0; i + 1 < structs.size(); i++) {
            MemoryUtil.memPutAddress(structs.get(i).address() + VkBaseOutStructure.PNEXT, structs.get(i + 1).address());
        }
        MemoryUtil.memPutAddress(structs.get(structs.size() - 1).address() + VkBaseOutStructure.PNEXT, MemoryUtil.NULL);
    }

    private static String pciBusId(String deviceName, VkPhysicalDevicePCIBusInfoPropertiesEXT pci) {
        try {
            return DeviceIdentity.formatPciBusId(pci.pciDomain(), pci.pciBus(), pci.pciDevice(), pci.pciFunction());
        } catch (IllegalArgumentException e) {
            LOGGER.debug("Vulkan device '{}' reports an unusable PCI address: {}", deviceName, e.getMessage());
            return null;
        }
    }

    private static DeviceClass classOf(int deviceType) {
        switch (deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return DeviceClass.DISCRETE;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return DeviceClass.INTEGRATED;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return DeviceClass.VIRTUAL;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return DeviceClass.CPU;
            default:
                return DeviceClass.UNKNOWN;
        }
    }

    private static long u32(int value) {
        return value & 0xFFFFFFFFL;
    }

    private static int clampU32(int value) {
        return value < 0 ? Integer.MAX_VALUE : value;
    }

    /** Mutable collection point for one query; never escapes {@link #query}. */
    private static final class Snapshot {
        VkPhysicalDevice handle;
        String name = "";
        int vendorId;
        int deviceId;
        int driverVersion;
        int apiVersion;
        int driverId;
        String driverName = "";
        String driverInfo = "";
        DeviceClass deviceClass = DeviceClass.UNKNOWN;
        DeviceIdentity identity;
        String pciBusId;
        byte[] pipelineCacheUuid = new byte[UUID_BYTES];
        byte[] deviceUuid;
        Set<String> extensions = new LinkedHashSet<>();
        final Set<Feature> features = EnumSet.noneOf(Feature.class);
        List<QueueFamily> queueFamilies = new ArrayList<>();
        long[] memoryHeapSizes = new long[0];
        int[] memoryHeapFlags = new int[0];
        int[] memoryTypePropertyFlags = new int[0];
        int[] memoryTypeHeapIndices = new int[0];
        long maxStorageBufferRange;
        int maxPushConstantsSize;
        final long[] maxComputeWorkGroupCount = new long[3];
        int maxComputeWorkGroupInvocations;
        final long[] maxComputeWorkGroupSize = new long[3];
        int maxComputeSharedMemorySize;
        int maxPerStageDescriptorStorageBuffers;
        long minStorageBufferOffsetAlignment;
        long nonCoherentAtomSize;
        long maxMemoryAllocationSize;
        int maxMemoryAllocationCount;
        long maxBufferSize;
        int subgroupSize;
        int subgroupSupportedStages;
        int subgroupSupportedOperations;
        int minSubgroupSize;
        int maxSubgroupSize;
        int maxPushDescriptors;
        int shaderCoreCount;
    }
}

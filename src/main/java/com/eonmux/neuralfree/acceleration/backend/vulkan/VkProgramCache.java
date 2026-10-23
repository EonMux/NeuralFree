package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkDevice;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;

/**
 * Loads each classpath SPIR-V module once into a {@link VkProgram}; programs are
 * destroyed by {@link #close()}.
 *
 * <p>Loading reflects the module and validates it against the device before any Vulkan object is
 * created: every capability that needs a device feature must be enabled, subgroup operations must
 * be supported in the compute stage, the bindings must fit {@code maxPerStageDescriptorStorageBuffers}
 * and the descriptor strategy, and the push-constant block must fit {@code maxPushConstantsSize}.
 * A failure throws {@link UnsupportedKernelException} naming the resource and the reason.</p>
 *
 * <p>The cache also carries the device facts kernel handles need to pick a subgroup variant:
 * {@code maxSubgroupSize} and whether {@code REQUIRE_FULL_SUBGROUPS} may be requested. Thread-safe;
 * lookups of loaded programs are lock-free.</p>
 */
public final class VkProgramCache implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkProgramCache.class);

    /** File-name suffix of a kernel's subgroup-reduction module ({@code <name>__sg.spv}). */
    public static final String SUBGROUP_VARIANT_SUFFIX = "__sg.spv";
    /** File-name suffix of a SPIR-V module. */
    public static final String SPIRV_SUFFIX = ".spv";

    /** SPIR-V capabilities that need an enabled device feature. */
    private static final int CAP_STORAGE_UNIFORM_16 = 4434;
    private static final int CAP_STORAGE_PUSH_CONSTANT_16 = 4435;
    private static final int CAP_UNIFORM_AND_STORAGE_8 = 4449;
    private static final int CAP_STORAGE_PUSH_CONSTANT_8 = 4450;
    private static final int CAP_GROUP_NON_UNIFORM_FIRST = 61;
    private static final int CAP_GROUP_NON_UNIFORM_LAST = 68;
    private static final int CAP_DOT_PRODUCT_INPUT_ALL = 6016;
    private static final int CAP_DOT_PRODUCT = 6019;
    private static final int CAP_VULKAN_MEMORY_MODEL = 5345;

    private final VkDevice device;
    private final VkPhysicalDeviceInfo info;
    private final VkDeviceFeatures.Enabled enabled;
    private final VkPipelineCache pipelineCache;
    private final VkDescriptorStrategy descriptorStrategy;
    private final ConcurrentHashMap<String, VkProgram> programs = new ConcurrentHashMap<>();
    private final ClassLoader classLoader;
    private volatile boolean closed;

    public VkProgramCache(VkDevice device, VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled,
                          VkPipelineCache pipelineCache, VkDescriptorStrategy descriptorStrategy) {
        if (device == null || info == null || enabled == null || pipelineCache == null || descriptorStrategy == null) {
            throw new IllegalArgumentException("device, info, enabled features, pipeline cache and descriptor"
                    + " strategy must not be null");
        }
        this.device = device;
        this.info = info;
        this.enabled = enabled;
        this.pipelineCache = pipelineCache;
        this.descriptorStrategy = descriptorStrategy;
        this.classLoader = VkProgramCache.class.getClassLoader();
    }

    /**
     * The program of a classpath {@code .spv} resource, loaded and reflected on first use.
     *
     * @throws UnsupportedKernelException when the resource is missing, malformed, or needs a
     *                                    capability or limit the device does not provide
     */
    public VkProgram get(String resourcePath) {
        if (resourcePath == null || resourcePath.isEmpty()) {
            throw new IllegalArgumentException("SPIR-V resource path must not be empty");
        }
        VkProgram program = programs.get(resourcePath);
        if (program != null) {
            return program;
        }
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("program cache is closed");
            }
            program = programs.get(resourcePath);
            if (program == null) {
                program = load(resourcePath);
                programs.put(resourcePath, program);
            }
            return program;
        }
    }

    public int size() {
        return programs.size();
    }

    /** The descriptor strategy every program of this cache is created for. */
    public VkDescriptorStrategy descriptorStrategy() {
        return descriptorStrategy;
    }

    /** The physical device the programs are validated against. */
    public VkPhysicalDeviceInfo deviceInfo() {
        return info;
    }

    /** {@code maxSubgroupSize}: a subgroup module needs {@code local_x} to be a multiple of it. */
    public int maxSubgroupSize() {
        return info.maxSubgroupSize();
    }

    /** True when pipelines may request {@code REQUIRE_FULL_SUBGROUPS} ({@code computeFullSubgroups} enabled). */
    public boolean fullSubgroupsEnabled() {
        return enabled.computeFullSubgroups();
    }

    /** True when {@code resourcePath} names a subgroup-reduction module. */
    public static boolean isSubgroupVariant(String resourcePath) {
        return resourcePath.endsWith(SUBGROUP_VARIANT_SUFFIX);
    }

    /** The plain module of a subgroup-reduction module ({@code X__sg.spv} to {@code X.spv}). */
    public static String plainVariantOf(String resourcePath) {
        if (!isSubgroupVariant(resourcePath)) {
            throw new IllegalArgumentException(resourcePath + " is not a subgroup variant");
        }
        return resourcePath.substring(0, resourcePath.length() - SUBGROUP_VARIANT_SUFFIX.length()) + SPIRV_SUFFIX;
    }

    /** Destroys every program; idempotent. The device must be idle. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        List<RuntimeException> failures = new ArrayList<>();
        for (VkProgram p : programs.values()) {
            try {
                p.close();
            } catch (RuntimeException e) {
                failures.add(e);
            }
        }
        programs.clear();
        if (!failures.isEmpty()) {
            RuntimeException first = failures.get(0);
            for (int i = 1; i < failures.size(); i++) {
                first.addSuppressed(failures.get(i));
            }
            throw first;
        }
    }

    private VkProgram load(String resourcePath) {
        byte[] bytes = readResource(resourcePath);
        SpirvModuleInfo module;
        try {
            module = SpirvReflector.reflect(bytes);
        } catch (IllegalArgumentException e) {
            throw new UnsupportedKernelException("SPIR-V module classpath:" + resourcePath + " is malformed: "
                    + e.getMessage(), e);
        } catch (UnsupportedKernelException e) {
            throw new UnsupportedKernelException("SPIR-V module classpath:" + resourcePath + ": " + e.getMessage(), e);
        }
        String reason = unsupportedReason(module);
        if (reason != null) {
            throw new UnsupportedKernelException("SPIR-V module classpath:" + resourcePath
                    + " cannot run on Vulkan device " + info.name() + ": " + reason);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length);
        try {
            code.put(bytes).flip();
            VkProgram program = new VkProgram(device, resourcePath, code, module, descriptorStrategy, pipelineCache);
            LOGGER.debug("Vulkan program {}: {}", resourcePath, module);
            return program;
        } finally {
            MemoryUtil.memFree(code);
        }
    }

    private byte[] readResource(String resourcePath) {
        try (InputStream in = classLoader.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new UnsupportedKernelException("SPIR-V module classpath:" + resourcePath + " not found");
            }
            return in.readAllBytes();
        } catch (IOException e) {
            throw new UnsupportedKernelException("SPIR-V module classpath:" + resourcePath + " could not be read: "
                    + e.getMessage(), e);
        }
    }

    /** Why the device cannot run {@code module}, or null when it can. */
    private String unsupportedReason(SpirvModuleInfo module) {
        for (int cap : module.capabilities()) {
            String missing = missingFeature(cap);
            if (missing != null) {
                return "SPIR-V capability " + cap + " needs " + missing;
            }
        }
        if (module.bindingCount() > info.maxPerStageDescriptorStorageBuffers()) {
            return module.bindingCount() + " storage buffers exceed maxPerStageDescriptorStorageBuffers "
                    + info.maxPerStageDescriptorStorageBuffers();
        }
        if (module.pushConstantSize() > info.maxPushConstantsSize()) {
            return "push constants of " + module.pushConstantSize() + " bytes exceed maxPushConstantsSize "
                    + info.maxPushConstantsSize();
        }
        return descriptorStrategy.unsupportedReason(module);
    }

    /** The device feature a capability needs that is not enabled, or null. */
    private String missingFeature(int cap) {
        switch (cap) {
            case SpirvModuleInfo.CAP_FLOAT16:
                return need(VkPhysicalDeviceInfo.Feature.SHADER_FLOAT16);
            case SpirvModuleInfo.CAP_FLOAT64:
                return need(VkPhysicalDeviceInfo.Feature.SHADER_FLOAT64);
            case SpirvModuleInfo.CAP_INT64:
                return need(VkPhysicalDeviceInfo.Feature.SHADER_INT64);
            case SpirvModuleInfo.CAP_INT16:
                return need(VkPhysicalDeviceInfo.Feature.SHADER_INT16);
            case SpirvModuleInfo.CAP_INT8:
                return need(VkPhysicalDeviceInfo.Feature.SHADER_INT8);
            case SpirvModuleInfo.CAP_STORAGE_BUFFER_16BIT_ACCESS:
                return need(VkPhysicalDeviceInfo.Feature.STORAGE_BUFFER_16BIT_ACCESS);
            case CAP_STORAGE_UNIFORM_16:
                return need(VkPhysicalDeviceInfo.Feature.UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS);
            case SpirvModuleInfo.CAP_STORAGE_BUFFER_8BIT_ACCESS:
                return need(VkPhysicalDeviceInfo.Feature.STORAGE_BUFFER_8BIT_ACCESS);
            case CAP_UNIFORM_AND_STORAGE_8:
                return need(VkPhysicalDeviceInfo.Feature.UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS);
            case CAP_STORAGE_PUSH_CONSTANT_16:
                return "storagePushConstant16, which the backend does not enable";
            case CAP_STORAGE_PUSH_CONSTANT_8:
                return "storagePushConstant8, which the backend does not enable";
            case CAP_VULKAN_MEMORY_MODEL:
                return "vulkanMemoryModel, which the backend does not enable";
            default:
                break;
        }
        if (cap >= CAP_GROUP_NON_UNIFORM_FIRST && cap <= CAP_GROUP_NON_UNIFORM_LAST) {
            int operation = 1 << (cap - CAP_GROUP_NON_UNIFORM_FIRST);
            if ((info.subgroupSupportedStages() & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
                return "subgroup operations in compute shaders";
            }
            if ((info.subgroupSupportedOperations() & operation) == 0) {
                return "subgroup operation bit 0x" + Integer.toHexString(operation);
            }
            return null;
        }
        if (cap >= CAP_DOT_PRODUCT_INPUT_ALL && cap <= CAP_DOT_PRODUCT) {
            return need(VkPhysicalDeviceInfo.Feature.SHADER_INTEGER_DOT_PRODUCT);
        }
        return null;
    }

    private String need(VkPhysicalDeviceInfo.Feature feature) {
        return enabled.isEnabled(feature) ? null : "device feature " + feature;
    }
}

package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSpecializationInfo;
import org.lwjgl.vulkan.VkSpecializationMapEntry;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import static org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.nvkCmdPushConstants;
import static org.lwjgl.vulkan.VK10.vkCreateComputePipelines;
import static org.lwjgl.vulkan.VK10.vkCreateDescriptorSetLayout;
import static org.lwjgl.vulkan.VK10.vkCreatePipelineLayout;
import static org.lwjgl.vulkan.VK10.vkCreateShaderModule;
import static org.lwjgl.vulkan.VK10.vkDestroyDescriptorSetLayout;
import static org.lwjgl.vulkan.VK10.vkDestroyPipeline;
import static org.lwjgl.vulkan.VK10.vkDestroyPipelineLayout;
import static org.lwjgl.vulkan.VK10.vkDestroyShaderModule;
import static org.lwjgl.vulkan.VK13.VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;

/**
 * One reflected SPIR-V module with its descriptor-set and pipeline layouts and
 * the compute pipelines created for each (local size, specialisation) pair.
 *
 * <p>The descriptor-set layout has one {@code STORAGE_BUFFER} binding (compute stage) per binding
 * present in the module, created with the descriptor strategy's flags. The pipeline layout has that
 * one set and, when the module has a push-constant block, one compute range
 * {@code [0, pushConstantSize)}. Pipelines are cached by local size, applied specialisation constants
 * and the full-subgroups flag; lookups are lock-free and creation is serialised per program.</p>
 */
public final class VkProgram implements AutoCloseable {

    private final VkDevice device;
    private final String resourcePath;
    private final SpirvModuleInfo info;
    private final VkDescriptorStrategy descriptorStrategy;
    private final VkPipelineCache pipelineCache;
    private final long shaderModule;
    private final long descriptorSetLayout;
    private final long pipelineLayout;
    private final ConcurrentHashMap<PipelineKey, Long> pipelines = new ConcurrentHashMap<>();
    private volatile boolean closed;

    /**
     * Creates the shader module and layouts; the caller has validated the module against the device.
     *
     * @param code SPIR-V words in native memory, position 0 and limit the code size
     * @throws GpuBackendException     when a Vulkan object cannot be created
     * @throws GpuOutOfMemoryException when the driver runs out of memory
     */
    VkProgram(VkDevice device, String resourcePath, ByteBuffer code, SpirvModuleInfo info,
              VkDescriptorStrategy descriptorStrategy, VkPipelineCache pipelineCache) {
        this.device = device;
        this.resourcePath = resourcePath;
        this.info = info;
        this.descriptorStrategy = descriptorStrategy;
        this.pipelineCache = pipelineCache;
        long module = VK_NULL_HANDLE;
        long setLayout = VK_NULL_HANDLE;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handle = stack.mallocLong(1);
            VkShaderModuleCreateInfo moduleInfo = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(code);
            VkErrors.check(vkCreateShaderModule(device, moduleInfo, null, handle), "vkCreateShaderModule");
            module = handle.get(0);

            List<SpirvModuleInfo.BindingInfo> bindings = info.bindings();
            VkDescriptorSetLayoutBinding.Buffer layoutBindings = VkDescriptorSetLayoutBinding.calloc(bindings.size(), stack);
            for (int i = 0; i < bindings.size(); i++) {
                layoutBindings.get(i)
                        .binding(bindings.get(i).binding())
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1)
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            }
            VkDescriptorSetLayoutCreateInfo setInfo = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
                    .flags(descriptorStrategy.setLayoutCreateFlags())
                    .pBindings(layoutBindings);
            VkErrors.check(vkCreateDescriptorSetLayout(device, setInfo, null, handle), "vkCreateDescriptorSetLayout");
            setLayout = handle.get(0);

            VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                    .pSetLayouts(stack.longs(setLayout));
            if (info.pushConstantSize() > 0) {
                VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, stack);
                range.get(0).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(info.pushConstantSize());
                layoutInfo.pPushConstantRanges(range);
            }
            VkErrors.check(vkCreatePipelineLayout(device, layoutInfo, null, handle), "vkCreatePipelineLayout");
            this.pipelineLayout = handle.get(0);
        } catch (RuntimeException e) {
            if (setLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, setLayout, null);
            }
            if (module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, module, null);
            }
            throw e;
        }
        this.shaderModule = module;
        this.descriptorSetLayout = setLayout;
    }

    /** Classpath resource the module was loaded from. */
    public String resourcePath() {
        return resourcePath;
    }

    public SpirvModuleInfo info() {
        return info;
    }

    public long shaderModule() {
        return shaderModule;
    }

    public long descriptorSetLayout() {
        return descriptorSetLayout;
    }

    public long pipelineLayout() {
        return pipelineLayout;
    }

    /** The strategy that binds this program's descriptor set. */
    public VkDescriptorStrategy descriptorStrategy() {
        return descriptorStrategy;
    }

    /** Number of storage-buffer bindings present in the module (see {@link SpirvModuleInfo#bindings()}). */
    public int bindingCount() {
        return info.bindingCount();
    }

    public int pushConstantSize() {
        return info.pushConstantSize();
    }

    /** Number of pipelines created so far. */
    public int pipelineCount() {
        return pipelines.size();
    }

    /**
     * The pipeline for this local size and these specialisation constants, created on first use.
     * Only SpecIds declared by the module are applied.
     *
     * @param local3               local size per dimension (three entries); a fixed dimension must
     *                             equal the module's value
     * @param specs                specialisation constants by SpecId (local size ids are set from {@code local3})
     * @param requireFullSubgroups set {@code REQUIRE_FULL_SUBGROUPS} on the shader stage
     * @throws IllegalArgumentException when the local size is invalid for this module
     * @throws GpuOutOfMemoryException  when pipeline creation runs out of host or device memory
     * @throws GpuBackendException      when pipeline creation fails otherwise
     */
    public long pipeline(int[] local3, SortedMap<Integer, Integer> specs, boolean requireFullSubgroups) {
        PipelineKey key = pipelineKey(local3, specs, requireFullSubgroups);
        Long pipeline = pipelines.get(key);
        if (pipeline != null) {
            return pipeline;
        }
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("program " + resourcePath + " is closed");
            }
            pipeline = pipelines.get(key);
            if (pipeline == null) {
                pipeline = createPipeline(key.applied, requireFullSubgroups);
                pipelines.put(key, pipeline);
            }
            return pipeline;
        }
    }

    /**
     * The pipeline {@link #pipeline} returns for these arguments when it already exists, or null when
     * {@link #pipeline} would have to create it. Never creates a pipeline.
     *
     * @throws IllegalArgumentException when the local size is invalid for this module
     */
    public Long existingPipeline(int[] local3, SortedMap<Integer, Integer> specs, boolean requireFullSubgroups) {
        return pipelines.get(pipelineKey(local3, specs, requireFullSubgroups));
    }

    /** The cache key of a pipeline: the local size, the applied specialisation constants and the subgroup flag. */
    private PipelineKey pipelineKey(int[] local3, SortedMap<Integer, Integer> specs, boolean requireFullSubgroups) {
        if (local3 == null || local3.length != 3) {
            throw new IllegalArgumentException(resourcePath + ": local size must have three entries");
        }
        TreeMap<Integer, Integer> applied = new TreeMap<>();
        for (int d = 0; d < 3; d++) {
            if (local3[d] < 1) {
                throw new IllegalArgumentException(resourcePath + ": local size " + Arrays.toString(local3)
                        + " must be positive");
            }
            int specId = info.localSizeSpecId(d);
            if (specId >= 0) {
                applied.put(specId, local3[d]);
            } else if (local3[d] != info.localSize(d)) {
                throw new IllegalArgumentException(resourcePath + ": local size dimension " + d + " is fixed at "
                        + info.localSize(d) + " by the module, requested " + local3[d]);
            }
        }
        if (specs != null) {
            for (Map.Entry<Integer, Integer> e : specs.entrySet()) {
                int id = e.getKey();
                if (info.declaresSpecId(id) && !info.isLocalSizeSpecId(id)) {
                    applied.put(id, e.getValue());
                }
            }
        }
        return new PipelineKey(local3, applied, requireFullSubgroups);
    }

    /** Records {@code args}' descriptors through this program's strategy. */
    public void bindDescriptors(VkCommandBuffer cb, VkArgState args, VkDescriptorStrategy.Frame frame) {
        descriptorStrategy.bind(cb, this, args, frame);
    }

    /** Records {@code vkCmdPushConstants} with {@code args}' push image; nothing when the module has no push block. */
    public void pushConstants(VkCommandBuffer cb, VkArgState args) {
        int size = info.pushConstantSize();
        if (size == 0) {
            return;
        }
        if (args.pushSize() != size) {
            throw new IllegalArgumentException(args.kernelName() + ": push image of " + args.pushSize()
                    + " bytes does not match the " + size + "-byte push range of " + resourcePath);
        }
        nvkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size,
                MemoryUtil.memAddress(args.pushImage()));
    }

    /** Destroys the pipelines, layouts and module; idempotent. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (Long p : pipelines.values()) {
            vkDestroyPipeline(device, p, null);
        }
        pipelines.clear();
        vkDestroyPipelineLayout(device, pipelineLayout, null);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, null);
        vkDestroyShaderModule(device, shaderModule, null);
    }

    private long createPipeline(SortedMap<Integer, Integer> applied, boolean requireFullSubgroups) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            List<Integer> ids = new ArrayList<>(applied.keySet());
            int dataSize = 0;
            for (int id : ids) {
                dataSize += info.specConstantSize(id);
            }
            VkSpecializationInfo specInfo = null;
            if (!ids.isEmpty()) {
                VkSpecializationMapEntry.Buffer entries = VkSpecializationMapEntry.calloc(ids.size(), stack);
                ByteBuffer data = stack.malloc(dataSize).order(ByteOrder.nativeOrder());
                int offset = 0;
                for (int i = 0; i < ids.size(); i++) {
                    int id = ids.get(i);
                    int size = info.specConstantSize(id);
                    entries.get(i).constantID(id).offset(offset).size(size);
                    if (size == SpirvModuleInfo.SPEC_SIZE_64) {
                        data.putLong(offset, applied.get(id));
                    } else {
                        data.putInt(offset, applied.get(id));
                    }
                    offset += size;
                }
                specInfo = VkSpecializationInfo.calloc(stack).pMapEntries(entries).pData(data);
            }
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                    .flags(requireFullSubgroups ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0)
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT)
                    .module(shaderModule)
                    .pName(stack.UTF8(info.entryPoint()))
                    .pSpecializationInfo(specInfo);
            VkComputePipelineCreateInfo.Buffer create = VkComputePipelineCreateInfo.calloc(1, stack);
            create.get(0)
                    .sType(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO)
                    .stage(stage)
                    .layout(pipelineLayout)
                    .basePipelineHandle(VK_NULL_HANDLE)
                    .basePipelineIndex(-1);
            LongBuffer pPipeline = stack.mallocLong(1);
            VkErrors.check(vkCreateComputePipelines(device, pipelineCache.handle(), create, null, pPipeline),
                    "vkCreateComputePipelines for " + resourcePath);
            return pPipeline.get(0);
        }
    }

    /** Cache key of one pipeline: local size, applied specialisation constants and the subgroup flag. */
    private static final class PipelineKey {
        /** Specialisation constants the pipeline is created with, local-size ids included; not part of equality. */
        final SortedMap<Integer, Integer> applied;
        private final int[] values;
        private final int hash;

        PipelineKey(int[] local3, SortedMap<Integer, Integer> applied, boolean requireFullSubgroups) {
            this.applied = applied;
            int[] v = new int[4 + 2 * applied.size()];
            v[0] = local3[0];
            v[1] = local3[1];
            v[2] = local3[2];
            v[3] = requireFullSubgroups ? 1 : 0;
            int i = 4;
            for (Map.Entry<Integer, Integer> e : applied.entrySet()) {
                v[i++] = e.getKey();
                v[i++] = e.getValue();
            }
            this.values = v;
            this.hash = Arrays.hashCode(v);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof PipelineKey && Arrays.equals(values, ((PipelineKey) o).values);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}

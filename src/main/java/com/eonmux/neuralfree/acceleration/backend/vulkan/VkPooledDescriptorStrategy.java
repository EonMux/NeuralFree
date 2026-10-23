package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDevice;

import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

import static org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
import static org.lwjgl.vulkan.VK10.VK_ERROR_FRAGMENTED_POOL;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.nvkAllocateDescriptorSets;
import static org.lwjgl.vulkan.VK10.nvkCmdBindDescriptorSets;
import static org.lwjgl.vulkan.VK10.nvkUpdateDescriptorSets;
import static org.lwjgl.vulkan.VK10.vkCreateDescriptorPool;
import static org.lwjgl.vulkan.VK10.vkDestroyDescriptorPool;
import static org.lwjgl.vulkan.VK10.vkResetDescriptorPool;
import static org.lwjgl.vulkan.VK11.VK_ERROR_OUT_OF_POOL_MEMORY;

/**
 * Allocates descriptor sets from growing per-batch pools, reuses a set while the
 * bound buffers are unchanged, and resets the pools when the batch retires.
 *
 * <p>Each frame owns a list of pools of {@value #SETS_PER_POOL} sets and
 * {@value #SETS_PER_POOL}&nbsp;&times;&nbsp;{@value #DESCRIPTORS_PER_SET} storage-buffer
 * descriptors. A dispatch allocates one set from the current pool, moving to the next pool (created
 * on demand) when the current one is exhausted, writes one descriptor per binding present in the
 * module, and binds it. When the same argument state is dispatched again in the same batch with
 * the same program and unchanged buffers ({@link VkArgState#bindingVersion()}), the set allocated
 * for it is bound again without an update. Retiring the batch resets every pool of the frame; the
 * pools are kept for the next batch.</p>
 */
public final class VkPooledDescriptorStrategy implements VkDescriptorStrategy {

    /** Descriptor sets per pool. */
    public static final int SETS_PER_POOL = 1024;
    /** Average storage-buffer descriptors per set that a pool is sized for. */
    public static final int DESCRIPTORS_PER_SET = 16;

    private final VkDevice device;
    private final long maxStorageBufferRange;
    private final Set<PoolFrame> liveFrames = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    /**
     * @param maxStorageBufferRange {@code maxStorageBufferRange}, widened unsigned
     */
    public VkPooledDescriptorStrategy(VkDevice device, long maxStorageBufferRange) {
        if (device == null) {
            throw new IllegalArgumentException("VkDevice must not be null");
        }
        if (maxStorageBufferRange <= 0) {
            throw new IllegalArgumentException("maxStorageBufferRange must be positive, got " + maxStorageBufferRange);
        }
        this.device = device;
        this.maxStorageBufferRange = maxStorageBufferRange;
    }

    @Override
    public boolean supports(SpirvModuleInfo info) {
        return true;
    }

    @Override
    public String unsupportedReason(SpirvModuleInfo info) {
        return null;
    }

    @Override
    public int setLayoutCreateFlags() {
        return 0;
    }

    @Override
    public Frame newFrame() {
        PoolFrame frame = new PoolFrame();
        synchronized (liveFrames) {
            if (closed) {
                frame.free();
                throw new IllegalStateException("pooled descriptor strategy is closed");
            }
            liveFrames.add(frame);
        }
        return frame;
    }

    @Override
    public void bind(VkCommandBuffer cb, VkProgram program, VkArgState args, Frame frame) {
        if (program.bindingCount() == 0) {
            return;
        }
        PoolFrame f = cast(frame);
        long set = f.reusableSet(args, program);
        if (set == VK_NULL_HANDLE) {
            set = f.allocate(program.descriptorSetLayout());
            int n = f.scratch.fill(program, args, set, maxStorageBufferRange);
            nvkUpdateDescriptorSets(device, n, f.scratch.writesAddress(), 0, MemoryUtil.NULL);
            f.remember(args, program, set);
        }
        MemoryUtil.memPutLong(f.setSlot, set);
        nvkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipelineLayout(), 0, 1, f.setSlot,
                0, MemoryUtil.NULL);
    }

    @Override
    public void reset(Frame frame) {
        PoolFrame f = cast(frame);
        for (int i = 0; i <= f.current && i < f.poolCount; i++) {
            VkErrors.check(vkResetDescriptorPool(device, f.pools[i], 0), "vkResetDescriptorPool");
        }
        f.current = 0;
        f.forgetSets();
    }

    @Override
    public void destroy(Frame frame) {
        PoolFrame f = cast(frame);
        synchronized (liveFrames) {
            liveFrames.remove(f);
        }
        f.free();
    }

    @Override
    public void close() {
        synchronized (liveFrames) {
            for (PoolFrame f : liveFrames) {
                f.free();
            }
            liveFrames.clear();
            closed = true;
        }
    }

    private static PoolFrame cast(Frame frame) {
        if (!(frame instanceof PoolFrame)) {
            throw new IllegalArgumentException("frame " + frame + " was not created by a pooled descriptor strategy");
        }
        return (PoolFrame) frame;
    }

    private long createPool() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(1, stack);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(SETS_PER_POOL * DESCRIPTORS_PER_SET);
            VkDescriptorPoolCreateInfo info = VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
                    .maxSets(SETS_PER_POOL)
                    .pPoolSizes(sizes);
            LongBuffer pPool = stack.mallocLong(1);
            VkErrors.check(vkCreateDescriptorPool(device, info, null, pPool), "vkCreateDescriptorPool");
            return pPool.get(0);
        }
    }

    /** The set last allocated for one argument state in this frame. */
    private static final class CachedSet {
        VkProgram program;
        long bindingVersion;
        long set;
    }

    /** Pools, native allocation scratch and set cache of one command batch. */
    private final class PoolFrame implements Frame {
        final WriteScratch scratch = new WriteScratch();
        final VkDescriptorSetAllocateInfo allocInfo = VkDescriptorSetAllocateInfo.calloc();
        /** Native slots: [0] the set layout passed to allocation, [1] the allocated or bound set. */
        final LongBuffer slots = MemoryUtil.memAllocLong(2);
        final long layoutSlot = MemoryUtil.memAddress(slots, 0);
        final long setSlot = MemoryUtil.memAddress(slots, 1);
        final Map<VkArgState, CachedSet> cache = new IdentityHashMap<>();
        final ArrayDeque<CachedSet> spare = new ArrayDeque<>();
        long[] pools = new long[4];
        int poolCount;
        int current;
        boolean freed;

        PoolFrame() {
            allocInfo.sType(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
            VkDescriptorSetAllocateInfo.npSetLayouts(allocInfo.address(), MemoryUtil.memLongBuffer(layoutSlot, 1));
        }

        long reusableSet(VkArgState args, VkProgram program) {
            CachedSet c = cache.get(args);
            return c != null && c.program == program && c.bindingVersion == args.bindingVersion()
                    ? c.set : VK_NULL_HANDLE;
        }

        void remember(VkArgState args, VkProgram program, long set) {
            CachedSet c = cache.get(args);
            if (c == null) {
                c = spare.isEmpty() ? new CachedSet() : spare.pop();
                cache.put(args, c);
            }
            c.program = program;
            c.bindingVersion = args.bindingVersion();
            c.set = set;
        }

        void forgetSets() {
            for (CachedSet c : cache.values()) {
                c.program = null;
                spare.push(c);
            }
            cache.clear();
        }

        long allocate(long setLayout) {
            MemoryUtil.memPutLong(layoutSlot, setLayout);
            boolean freshPool = false;
            while (true) {
                if (current == poolCount) {
                    if (poolCount == pools.length) {
                        pools = Arrays.copyOf(pools, poolCount * 2);
                    }
                    pools[poolCount++] = createPool();
                    freshPool = true;
                }
                VkDescriptorSetAllocateInfo.ndescriptorPool(allocInfo.address(), pools[current]);
                int result = nvkAllocateDescriptorSets(device, allocInfo.address(), setSlot);
                if (result == VK_SUCCESS) {
                    return MemoryUtil.memGetLong(setSlot);
                }
                if ((result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) && !freshPool) {
                    current++;
                    continue;
                }
                throw VkErrors.translate(result, "vkAllocateDescriptorSets from a "
                        + (freshPool ? "new" : "partly used") + " descriptor pool");
            }
        }

        void free() {
            if (freed) {
                return;
            }
            freed = true;
            for (int i = 0; i < poolCount; i++) {
                vkDestroyDescriptorPool(device, pools[i], null);
            }
            poolCount = 0;
            current = 0;
            scratch.free();
            allocInfo.free();
            MemoryUtil.memFree(slots);
            cache.clear();
            spare.clear();
        }
    }
}

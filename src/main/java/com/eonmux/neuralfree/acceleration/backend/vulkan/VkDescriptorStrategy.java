package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDescriptorBufferInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.util.List;

import static org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

/**
 * How storage-buffer descriptors reach a dispatch: pushed into the command
 * buffer ({@link VkPushDescriptorStrategy}) or allocated from per-batch pools
 * ({@link VkPooledDescriptorStrategy}).
 *
 * <p>One strategy serves every program of a backend: each program's descriptor-set layout is
 * created with {@link #setLayoutCreateFlags()}, and a program the strategy does not
 * {@link #supports support} is rejected when it is loaded. A {@link Frame} belongs to one command
 * batch of one queue, so frames are used by one thread at a time; the strategy itself is shared by
 * every queue of the backend and is thread-safe.</p>
 */
public interface VkDescriptorStrategy extends AutoCloseable {

    /** Per-batch descriptor state, reset when the batch retires. */
    interface Frame {
    }

    /**
     * The strategy for a device: push descriptors when {@code VK_KHR_push_descriptor} is enabled,
     * otherwise per-batch descriptor pools.
     */
    static VkDescriptorStrategy select(VkDevice device, VkPhysicalDeviceInfo info, VkDeviceFeatures.Enabled enabled) {
        if (enabled.pushDescriptors() && info.maxPushDescriptors() > 0) {
            return new VkPushDescriptorStrategy(device, info.maxPushDescriptors(), info.maxStorageBufferRange());
        }
        return new VkPooledDescriptorStrategy(device, info.maxStorageBufferRange());
    }

    /** True when this strategy can bind the descriptor set of a module with this interface. */
    boolean supports(SpirvModuleInfo info);

    /** Why {@link #supports} rejects {@code info}; null when it is supported. */
    String unsupportedReason(SpirvModuleInfo info);

    /** {@code VkDescriptorSetLayoutCreateFlags} the program's set layout must be created with. */
    int setLayoutCreateFlags();

    /** Creates the state for one command batch. */
    Frame newFrame();

    /**
     * Records the descriptors of {@code args}' current buffers for {@code program} into {@code cb};
     * only bindings present in the module are written. Does nothing when the module has no bindings.
     *
     * @throws IllegalStateException when a present binding has no buffer
     */
    void bind(VkCommandBuffer cb, VkProgram program, VkArgState args, Frame frame);

    /** Makes a retired batch's frame reusable. */
    void reset(Frame frame);

    /** Destroys a frame's resources; idempotent. */
    void destroy(Frame frame);

    /** Destroys every frame not yet destroyed and the strategy's own resources; idempotent. */
    @Override
    void close();

    /**
     * Native {@code VkWriteDescriptorSet} and {@code VkDescriptorBufferInfo} arrays reused for every
     * dispatch of one frame, so recording descriptors allocates nothing on the Java heap. Write
     * {@code i} permanently points at buffer info {@code i}; a dispatch fills the first {@code n}
     * entries with {@link #fill}. Not thread-safe; owned by one frame.
     */
    final class WriteScratch {

        private static final int INITIAL_CAPACITY = 16;

        private VkWriteDescriptorSet.Buffer writes;
        private VkDescriptorBufferInfo.Buffer infos;
        private int capacity;

        public WriteScratch() {
            allocate(INITIAL_CAPACITY);
        }

        /**
         * Fills one write per binding present in {@code program} from {@code args}' buffers.
         *
         * @param dstSet   descriptor set to write, or 0 for push descriptors
         * @param maxRange largest descriptor range, {@code maxStorageBufferRange}
         * @return the number of writes filled
         * @throws IllegalStateException when a present binding has no buffer
         */
        public int fill(VkProgram program, VkArgState args, long dstSet, long maxRange) {
            List<SpirvModuleInfo.BindingInfo> bindings = program.info().bindings();
            int n = bindings.size();
            if (n > capacity) {
                MemoryUtil.memFree(writes);
                MemoryUtil.memFree(infos);
                allocate(Math.max(n, capacity * 2));
            }
            VkGpuBuffer[] buffers = args.bindings();
            long writeBase = writes.address();
            long infoBase = infos.address();
            for (int i = 0; i < n; i++) {
                int binding = bindings.get(i).binding();
                VkGpuBuffer buffer = binding < buffers.length ? buffers[binding] : null;
                if (buffer == null) {
                    throw new IllegalStateException(args.kernelName() + ": binding " + binding
                            + " of " + program.resourcePath() + " has no buffer");
                }
                long info = infoBase + (long) i * VkDescriptorBufferInfo.SIZEOF;
                VkDescriptorBufferInfo.nbuffer(info, buffer.vkBuffer());
                VkDescriptorBufferInfo.noffset(info, buffer.offset());
                VkDescriptorBufferInfo.nrange(info, Math.min(buffer.range(), maxRange));
                long write = writeBase + (long) i * VkWriteDescriptorSet.SIZEOF;
                VkWriteDescriptorSet.ndstSet(write, dstSet);
                VkWriteDescriptorSet.ndstBinding(write, binding);
            }
            return n;
        }

        /** Address of the first write. */
        public long writesAddress() {
            return writes.address();
        }

        /** Frees the native arrays; idempotent. */
        public void free() {
            if (writes != null) {
                MemoryUtil.memFree(writes);
                MemoryUtil.memFree(infos);
                writes = null;
                infos = null;
                capacity = 0;
            }
        }

        private void allocate(int n) {
            writes = VkWriteDescriptorSet.calloc(n);
            infos = VkDescriptorBufferInfo.calloc(n);
            for (int i = 0; i < n; i++) {
                VkWriteDescriptorSet w = writes.get(i);
                w.sType(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET)
                        .dstArrayElement(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1);
                VkWriteDescriptorSet.npBufferInfo(w.address(), infos.slice(i, 1));
            }
            capacity = n;
        }
    }
}

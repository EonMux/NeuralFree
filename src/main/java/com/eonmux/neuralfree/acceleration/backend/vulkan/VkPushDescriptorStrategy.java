package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

import static org.lwjgl.vulkan.KHRPushDescriptor.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
import static org.lwjgl.vulkan.KHRPushDescriptor.nvkCmdPushDescriptorSetKHR;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE;

/**
 * Binds descriptors with {@code vkCmdPushDescriptorSetKHR}; no descriptor pools.
 *
 * <p>Each dispatch pushes one {@code VkDescriptorBufferInfo} per binding present in the module:
 * the buffer, its offset, and its range capped at {@code maxStorageBufferRange}. A module with
 * more bindings than {@code maxPushDescriptors} is not supported. Frames only hold the native
 * write scratch.</p>
 */
public final class VkPushDescriptorStrategy implements VkDescriptorStrategy {

    private final VkDevice device;
    private final int maxPushDescriptors;
    private final long maxStorageBufferRange;
    private final Set<PushFrame> liveFrames = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    /**
     * @param maxPushDescriptors    {@code VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors}
     * @param maxStorageBufferRange {@code maxStorageBufferRange}, widened unsigned
     */
    public VkPushDescriptorStrategy(VkDevice device, int maxPushDescriptors, long maxStorageBufferRange) {
        if (device == null) {
            throw new IllegalArgumentException("VkDevice must not be null");
        }
        if (maxPushDescriptors <= 0) {
            throw new IllegalArgumentException("maxPushDescriptors must be positive, got " + maxPushDescriptors);
        }
        if (maxStorageBufferRange <= 0) {
            throw new IllegalArgumentException("maxStorageBufferRange must be positive, got " + maxStorageBufferRange);
        }
        this.device = device;
        this.maxPushDescriptors = maxPushDescriptors;
        this.maxStorageBufferRange = maxStorageBufferRange;
    }

    public VkDevice device() {
        return device;
    }

    @Override
    public boolean supports(SpirvModuleInfo info) {
        return info.bindingCount() <= maxPushDescriptors;
    }

    @Override
    public String unsupportedReason(SpirvModuleInfo info) {
        return supports(info) ? null : "the module has " + info.bindingCount()
                + " storage-buffer bindings but the device pushes at most " + maxPushDescriptors + " descriptors";
    }

    @Override
    public int setLayoutCreateFlags() {
        return VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    @Override
    public Frame newFrame() {
        PushFrame frame = new PushFrame();
        synchronized (liveFrames) {
            if (closed) {
                frame.scratch.free();
                throw new IllegalStateException("push descriptor strategy is closed");
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
        PushFrame f = cast(frame);
        int n = f.scratch.fill(program, args, 0L, maxStorageBufferRange);
        nvkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipelineLayout(), 0, n,
                f.scratch.writesAddress());
    }

    @Override
    public void reset(Frame frame) {
        cast(frame);
    }

    @Override
    public void destroy(Frame frame) {
        PushFrame f = cast(frame);
        synchronized (liveFrames) {
            liveFrames.remove(f);
        }
        f.scratch.free();
    }

    @Override
    public void close() {
        synchronized (liveFrames) {
            for (PushFrame f : liveFrames) {
                f.scratch.free();
            }
            liveFrames.clear();
            closed = true;
        }
    }

    private static PushFrame cast(Frame frame) {
        if (!(frame instanceof PushFrame)) {
            throw new IllegalArgumentException("frame " + frame + " was not created by a push descriptor strategy");
        }
        return (PushFrame) frame;
    }

    private static final class PushFrame implements Frame {
        final WriteScratch scratch = new WriteScratch();
    }
}

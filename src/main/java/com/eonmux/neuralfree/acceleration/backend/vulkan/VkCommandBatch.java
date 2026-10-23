package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkMemoryBarrier;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.lwjgl.vulkan.VK10.VK_ACCESS_HOST_READ_BIT;
import static org.lwjgl.vulkan.VK10.VK_ACCESS_SHADER_READ_BIT;
import static org.lwjgl.vulkan.VK10.VK_ACCESS_SHADER_WRITE_BIT;
import static org.lwjgl.vulkan.VK10.VK_ACCESS_TRANSFER_READ_BIT;
import static org.lwjgl.vulkan.VK10.VK_ACCESS_TRANSFER_WRITE_BIT;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_HOST_BIT;
import static org.lwjgl.vulkan.VK10.VK_PIPELINE_STAGE_TRANSFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
import static org.lwjgl.vulkan.VK10.nvkCmdCopyBuffer;
import static org.lwjgl.vulkan.VK10.nvkCmdPipelineBarrier;
import static org.lwjgl.vulkan.VK10.vkBeginCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkCmdBindPipeline;
import static org.lwjgl.vulkan.VK10.vkCmdDispatch;
import static org.lwjgl.vulkan.VK10.vkCmdFillBuffer;
import static org.lwjgl.vulkan.VK10.vkEndCommandBuffer;
import static org.lwjgl.vulkan.VK10.vkResetCommandBuffer;

/**
 * One recording unit of a compute queue: a primary command buffer with its descriptor frame, the
 * timeline-semaphore waits its submission carries, the readbacks copied out when it retires, and
 * the timeline value it signals.
 *
 * <p>Recording starts with a global memory barrier, which orders the batch after every command
 * submitted earlier on the queue, so hazard tracking restarts with each batch. When the batch holds
 * readbacks, {@link #end()} adds a transfer-write to host-read barrier so the host sees the staged
 * bytes once the timeline value is reached. On a transfer-only queue family the barriers use the
 * transfer stage only.</p>
 *
 * <p>Recording helpers keep native structs per batch, so recording allocates nothing on the Java
 * heap. The batch also remembers the last bound pipeline and descriptor binding, so consecutive
 * dispatches of the same kernel with unchanged buffers skip redundant binds.</p>
 */
@NotThreadSafe("Owned by one compute queue; used under the queue's lock.")
public final class VkCommandBatch implements AutoCloseable {

    /** A readback whose host copy-out runs when the batch retires. */
    public static final class PendingRead {

        private final VkStagingRing.Slice slice;
        private final byte[] bytes;
        private final float[] floats;
        private final int dstIndex;

        private PendingRead(VkStagingRing.Slice slice, byte[] bytes, float[] floats, int dstIndex) {
            if (slice == null) {
                throw new IllegalArgumentException("readback slice must not be null");
            }
            this.slice = slice;
            this.bytes = bytes;
            this.floats = floats;
            this.dstIndex = dstIndex;
        }

        /**
         * A readback of the whole slice into {@code dst[dstOffset ..]}.
         *
         * @throws IllegalArgumentException when the slice does not fit the array
         */
        public static PendingRead ofBytes(VkStagingRing.Slice slice, byte[] dst, int dstOffset) {
            if (dst == null) {
                throw new IllegalArgumentException("readback destination must not be null");
            }
            if (dstOffset < 0 || dstOffset + slice.size() > dst.length) {
                throw new IllegalArgumentException("readback of " + slice.size() + " bytes at index " + dstOffset
                        + " exceeds the destination length " + dst.length);
            }
            return new PendingRead(slice, dst, null, dstOffset);
        }

        /**
         * A readback of the whole slice (a multiple of 4 bytes) into {@code dst[dstFloatOffset ..]}.
         *
         * @throws IllegalArgumentException when the slice size is not a multiple of 4 or does not fit the array
         */
        public static PendingRead ofFloats(VkStagingRing.Slice slice, float[] dst, int dstFloatOffset) {
            if (dst == null) {
                throw new IllegalArgumentException("readback destination must not be null");
            }
            if ((slice.size() & 3) != 0) {
                throw new IllegalArgumentException("float readback of " + slice.size()
                        + " bytes is not a multiple of 4");
            }
            if (dstFloatOffset < 0 || dstFloatOffset + slice.size() / Float.BYTES > dst.length) {
                throw new IllegalArgumentException("readback of " + slice.size() / Float.BYTES + " floats at index "
                        + dstFloatOffset + " exceeds the destination length " + dst.length);
            }
            return new PendingRead(slice, null, dst, dstFloatOffset);
        }

        /** The staging span the device copies into. */
        public VkStagingRing.Slice slice() {
            return slice;
        }

        /** Makes the staged bytes visible to the host and copies them into the destination array. */
        public void copyOut() {
            slice.invalidate();
            ByteBuffer staged = slice.mapped().duplicate().order(ByteOrder.LITTLE_ENDIAN);
            staged.clear();
            int size = (int) slice.size();
            if (bytes != null) {
                staged.get(bytes, dstIndex, size);
            } else {
                staged.asFloatBuffer().get(floats, dstIndex, size / Float.BYTES);
            }
        }
    }

    private final VkCommandBuffer cb;
    private final VkDescriptorStrategy.Frame frame;
    private final int barrierStages;
    private final VkCommandBufferBeginInfo beginInfo;
    private final VkMemoryBarrier.Buffer globalBarrier;
    private final VkMemoryBarrier.Buffer hostBarrier;
    private final VkBufferCopy.Buffer region;

    private final List<PendingRead> pendingReads = new ArrayList<>();
    private VkComputeQueue[] waitProducers = new VkComputeQueue[4];
    private long[] waitValues = new long[4];
    private int waitCount;
    private boolean recording;
    private int commandCount;
    private int dispatchCount;
    private long signalValue;
    private long boundPipeline = VK_NULL_HANDLE;
    private VkProgram boundProgram;
    private VkArgState boundArgs;
    private long boundBindingVersion;
    private boolean closed;

    /**
     * @param commandBuffer a primary command buffer from a pool created with
     *                      {@code RESET_COMMAND_BUFFER}; the pool owns it
     * @param frame         descriptor state of this batch, owned by the queue's strategy
     * @param computeStages true when the queue family runs compute shaders; false for a transfer-only family
     */
    public VkCommandBatch(VkCommandBuffer commandBuffer, VkDescriptorStrategy.Frame frame, boolean computeStages) {
        if (commandBuffer == null) {
            throw new IllegalArgumentException("command buffer must not be null");
        }
        if (frame == null) {
            throw new IllegalArgumentException("descriptor frame must not be null");
        }
        this.cb = commandBuffer;
        this.frame = frame;
        this.barrierStages = computeStages
                ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
                : VK_PIPELINE_STAGE_TRANSFER_BIT;
        int srcAccess = computeStages
                ? VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                : VK_ACCESS_TRANSFER_WRITE_BIT;
        int dstAccess = computeStages
                ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                : VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        this.beginInfo = VkCommandBufferBeginInfo.calloc()
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
                .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        this.globalBarrier = VkMemoryBarrier.calloc(1);
        this.globalBarrier.get(0).sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                .srcAccessMask(srcAccess)
                .dstAccessMask(dstAccess);
        this.hostBarrier = VkMemoryBarrier.calloc(1);
        this.hostBarrier.get(0).sType(VK_STRUCTURE_TYPE_MEMORY_BARRIER)
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
        this.region = VkBufferCopy.calloc(1);
    }

    public VkCommandBuffer commandBuffer() {
        return cb;
    }

    public VkDescriptorStrategy.Frame descriptorFrame() {
        return frame;
    }

    /**
     * Begins recording (one-time submit) and records the initial global barrier.
     *
     * @throws IllegalStateException    when the batch is already recording
     * @throws GpuOutOfMemoryException when {@code vkBeginCommandBuffer} reports exhausted host or device memory
     * @throws GpuBackendException     when {@code vkBeginCommandBuffer} fails otherwise
     */
    public void begin() {
        requireOpen();
        if (recording) {
            throw new IllegalStateException("command batch is already recording");
        }
        VkErrors.check(vkBeginCommandBuffer(cb, beginInfo), "vkBeginCommandBuffer");
        recording = true;
        barrier();
    }

    /**
     * Ends recording; adds the transfer-to-host barrier when the batch holds readbacks.
     *
     * @throws GpuOutOfMemoryException when {@code vkEndCommandBuffer} reports exhausted host or device memory
     * @throws GpuBackendException     when {@code vkEndCommandBuffer} fails otherwise
     */
    public void end() {
        requireRecording();
        if (!pendingReads.isEmpty()) {
            nvkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                    1, hostBarrier.address(), 0, MemoryUtil.NULL, 0, MemoryUtil.NULL);
        }
        recording = false;
        VkErrors.check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    }

    public boolean isRecording() {
        return recording;
    }

    /** True when the batch holds neither commands nor waits, so submitting it would do nothing. */
    public boolean isEmpty() {
        return commandCount == 0 && waitCount == 0;
    }

    /** Number of commands recorded (copies, fills and dispatches). */
    public int commandCount() {
        return commandCount;
    }

    public int dispatchCount() {
        return dispatchCount;
    }

    /**
     * Records the global memory barrier (shader and transfer writes before shader and transfer reads
     * and writes). The queue resets its hazard tracker after it.
     */
    public void barrier() {
        requireRecording();
        nvkCmdPipelineBarrier(cb, barrierStages, barrierStages, 0, 1, globalBarrier.address(),
                0, MemoryUtil.NULL, 0, MemoryUtil.NULL);
    }

    /** Records {@code vkCmdCopyBuffer} of one region. */
    public void copy(long srcBuffer, long srcOffset, long dstBuffer, long dstOffset, long size) {
        requireRecording();
        region.get(0).srcOffset(srcOffset).dstOffset(dstOffset).size(size);
        nvkCmdCopyBuffer(cb, srcBuffer, dstBuffer, 1, region.address());
        commandCount++;
    }

    /** Records {@code vkCmdFillBuffer}; offset and size are multiples of 4. */
    public void fill(long buffer, long offset, long size, int pattern) {
        requireRecording();
        vkCmdFillBuffer(cb, buffer, offset, size, pattern);
        commandCount++;
    }

    /** Binds a compute pipeline unless it is already bound in this batch. */
    public void bindPipeline(long pipeline) {
        requireRecording();
        if (pipeline != boundPipeline) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            boundPipeline = pipeline;
        }
    }

    /**
     * Records the descriptors of {@code args} for {@code program}, unless the previous dispatch of this
     * batch bound the same program and argument state with the same buffers.
     */
    public void bindDescriptors(VkProgram program, VkArgState args) {
        requireRecording();
        long version = args.bindingVersion();
        if (program == boundProgram && args == boundArgs && version == boundBindingVersion) {
            return;
        }
        program.bindDescriptors(cb, args, frame);
        boundProgram = program;
        boundArgs = args;
        boundBindingVersion = version;
    }

    /** Records {@code vkCmdDispatch}. */
    public void dispatch(int groupsX, int groupsY, int groupsZ) {
        requireRecording();
        vkCmdDispatch(cb, groupsX, groupsY, groupsZ);
        commandCount++;
        dispatchCount++;
    }

    /** Adds a readback copied out when the batch retires. */
    public void addPendingRead(PendingRead read) {
        if (read == null) {
            throw new IllegalArgumentException("pending read must not be null");
        }
        pendingReads.add(read);
    }

    /** True when the batch holds readbacks. */
    public boolean hasPendingReads() {
        return !pendingReads.isEmpty();
    }

    /** Unmodifiable view of the pending readbacks. */
    public List<PendingRead> pendingReads() {
        return Collections.unmodifiableList(pendingReads);
    }

    /**
     * Adds a wait on the timeline of {@code producer} reaching {@code value} before the batch
     * executes; a second wait on the same producer keeps the larger value. The producer is kept, not
     * its semaphore handle, so the submission can drop waits a closed producer already satisfied.
     */
    public void addWait(VkComputeQueue producer, long value) {
        if (producer == null) {
            throw new IllegalArgumentException("wait producer queue must not be null");
        }
        for (int i = 0; i < waitCount; i++) {
            if (waitProducers[i] == producer) {
                waitValues[i] = Math.max(waitValues[i], value);
                return;
            }
        }
        if (waitCount == waitProducers.length) {
            waitProducers = Arrays.copyOf(waitProducers, waitCount * 2);
            waitValues = Arrays.copyOf(waitValues, waitCount * 2);
        }
        waitProducers[waitCount] = producer;
        waitValues[waitCount] = value;
        waitCount++;
    }

    public int waitCount() {
        return waitCount;
    }

    /** The queues whose timelines the batch waits on (copy, {@link #waitCount()} entries). */
    public VkComputeQueue[] waitProducers() {
        return Arrays.copyOf(waitProducers, waitCount);
    }

    /** The wait values, parallel to {@link #waitProducers()} (copy). */
    public long[] waitValues() {
        return Arrays.copyOf(waitValues, waitCount);
    }

    /** The timeline value the batch signals; 0 until it is submitted. */
    public long signalValue() {
        return signalValue;
    }

    public void setSignalValue(long value) {
        this.signalValue = value;
    }

    /**
     * Resets the command buffer and clears every list and counter for reuse. The batch must not be
     * pending on the device.
     *
     * @throws GpuOutOfMemoryException when {@code vkResetCommandBuffer} reports exhausted device memory
     * @throws GpuBackendException     when {@code vkResetCommandBuffer} fails otherwise
     */
    public void reset() {
        requireOpen();
        recording = false;
        VkErrors.check(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
        pendingReads.clear();
        Arrays.fill(waitProducers, 0, waitCount, null);
        waitCount = 0;
        commandCount = 0;
        dispatchCount = 0;
        signalValue = 0;
        boundPipeline = VK_NULL_HANDLE;
        boundProgram = null;
        boundArgs = null;
        boundBindingVersion = 0;
    }

    /**
     * Frees the batch's native structs; idempotent. The command buffer is freed with its pool and the
     * descriptor frame by its strategy.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        recording = false;
        pendingReads.clear();
        beginInfo.free();
        globalBarrier.free();
        hostBarrier.free();
        region.free();
    }

    @Override
    public String toString() {
        return "VkCommandBatch{commands=" + commandCount + ", dispatches=" + dispatchCount + ", waits=" + waitCount
                + ", reads=" + pendingReads.size() + ", signal=" + signalValue + (recording ? ", recording" : "")
                + '}';
    }

    private void requireRecording() {
        requireOpen();
        if (!recording) {
            throw new IllegalStateException("command batch is not recording");
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("command batch is closed");
        }
    }
}

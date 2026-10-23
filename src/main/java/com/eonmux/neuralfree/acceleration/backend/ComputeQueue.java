package com.eonmux.neuralfree.acceleration.backend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * An in-order command queue of one {@link ComputeBackend}.
 *
 * <p><b>Dispatch.</b></p>
 * <ul>
 *   <li>{@code globalWorkSize[d]} is the total number of work-items in dimension {@code d}
 *       and {@code localWorkSize[d]} the work-group size (CUDA {@code blockDim}, Vulkan
 *       {@code local_size}); {@code 1 <= global.length <= 3} and
 *       {@code local.length == global.length}.</li>
 *   <li>Backends that launch by group count use {@code groups[d] = ceil(global[d] / local[d])}
 *       ({@link WorkSizes#groupCounts}). OpenCL requires {@code global % local == 0}; CUDA and
 *       Vulkan round up, so kernels bounds-check their tail.</li>
 *   <li>{@code localWorkSize == null} lets the backend choose: OpenCL passes null to the driver,
 *       CUDA and Vulkan use {@link WorkSizes#defaultLocal}, which divides {@code global[0]}.</li>
 *   <li>Vulkan folds a one-dimensional group count above {@code maxComputeWorkGroupCount[0]}
 *       into two dimensions; kernels recover the linear group id.</li>
 * </ul>
 *
 * <p><b>Arguments.</b></p>
 * <ul>
 *   <li>{@code Object[] args} in declared order. Supported: {@link GpuBuffer}, {@code Integer},
 *       {@code Float}, {@code Boolean} (a 4-byte 0/1) and {@code int[1]}/{@code float[1]} holders.
 *       {@code Long}, {@code Double} and other primitive arrays (passed by value) remain for the
 *       legacy ML callers only. {@code null} is rejected with {@link IllegalArgumentException}.</li>
 *   <li>Snapshot semantics: a value is captured when it is set; mutating an {@code int[1]}
 *       holder afterwards has no effect until the argument is set again.</li>
 *   <li>Arguments are sticky on the {@link KernelHandle}, not on the queue.
 *       {@code enqueueKernel(..., args)} replaces them all, {@link #setKernelArg} replaces one, and
 *       {@link #enqueueKernelPreBound} launches the stored arguments on any queue of the same backend.</li>
 * </ul>
 *
 * <p><b>Events and transfers.</b></p>
 * <ul>
 *   <li>An event may be waited on by any queue of the same backend; an event of another backend
 *       is rejected with {@link IllegalArgumentException}. Queues never retain the
 *       {@code waitOnEvents} array. An event must not be used as a wait after its {@code close()}.</li>
 *   <li>Writes copy the host data before returning, so the caller may reuse the array or
 *       ByteBuffer immediately. The ByteBuffer overload reads bytes {@code [0, size)} by absolute
 *       index and never changes the caller's position or limit.</li>
 *   <li>Read destinations are valid only after {@link #finish()} on the issuing queue, or after
 *       {@link GpuEvent#awaitCompletion()} on the read's event.</li>
 * </ul>
 *
 * <p><b>Ownership and threading.</b></p>
 * <ul>
 *   <li>Queues returned by {@link ComputeBackend#computeQueue()} and
 *       {@link ComputeBackend#transferQueue()} are owned by the backend: their {@link #close()}
 *       is equivalent to {@link #finish()} and only {@link ComputeBackend#close()} releases them.
 *       Queues from {@link ComputeBackend#createComputeQueue()} are owned by the caller.</li>
 *   <li>One thread at a time per queue.</li>
 *   <li>A {@link QueueType#TRANSFER} queue may reject kernel dispatches with
 *       {@link UnsupportedOperationException}.</li>
 * </ul>
 */
public interface ComputeQueue extends AutoCloseable {

    /** Largest host staging array the portable {@link #enqueueFillBuffer} fallback allocates. */
    int FILL_FALLBACK_CHUNK_BYTES = 1 << 20;

    enum QueueType {
        COMPUTE,
        TRANSFER
    }

    QueueType getQueueType();

    GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                 byte[] srcData, GpuEvent... waitOnEvents);

    GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                 float[] srcData, GpuEvent... waitOnEvents);

    /**
     * Enqueues a write of bytes {@code [0, sizeInBytes)} of {@code srcData}, read by absolute
     * index; the buffer's position and limit are left unchanged.
     *
     * <p>Default implementation copies to a byte[] and delegates. Backends should
     * override this to read the direct buffer natively.</p>
     */
    default GpuEvent enqueueWriteBuffer(GpuBuffer dstBuffer, long dstOffset, long sizeInBytes,
                                         ByteBuffer srcData, GpuEvent... waitOnEvents) {
        if (srcData == null) {
            throw new IllegalArgumentException("source ByteBuffer must not be null");
        }
        if (sizeInBytes < 0 || sizeInBytes > srcData.capacity()) {
            throw new IllegalArgumentException("write of " + sizeInBytes + " bytes exceeds the source capacity of "
                    + srcData.capacity() + " bytes");
        }
        byte[] arr = new byte[(int) sizeInBytes];
        ByteBuffer view = srcData.duplicate();
        view.clear();
        view.get(arr, 0, (int) sizeInBytes);
        return enqueueWriteBuffer(dstBuffer, dstOffset, sizeInBytes, arr, waitOnEvents);
    }

    GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                byte[] dstData, GpuEvent... waitOnEvents);

    GpuEvent enqueueReadBuffer(GpuBuffer srcBuffer, long srcOffset, long sizeInBytes,
                                float[] dstData, GpuEvent... waitOnEvents);

    GpuEvent enqueueKernel(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                           Object[] args, GpuEvent... waitOnEvents);

    /**
     * Enqueues a kernel without creating a GPU event for the dispatch.
     * Use this for kernels whose completion does not need to be tracked.
     * Avoids the overhead of event allocation and prevents event leaks.
     *
     * <p>Default implementation calls enqueueKernel and immediately closes the event.</p>
     */
    default void enqueueKernelNoEvent(KernelHandle kernel, long[] globalWorkSize, long[] localWorkSize,
                                       Object[] args, GpuEvent... waitOnEvents) {
        GpuEvent evt = enqueueKernel(kernel, globalWorkSize, localWorkSize, args, waitOnEvents);
        if (evt != null) evt.close();
    }

    /**
     * Enqueues a device-to-device buffer copy. Data stays on the GPU — no PCIe transfer.
     * Uses the OpenCL clEnqueueCopyBuffer / CUDA cuMemcpyDtoD path.
     */
    default void enqueueCopyBuffer(GpuBuffer src, long srcOffset, GpuBuffer dst, long dstOffset,
                                    long sizeInBytes, GpuEvent... waitOnEvents) {
        throw new UnsupportedOperationException("enqueueCopyBuffer not implemented");
    }

    /**
     * Sets a single kernel argument on a pre-bound kernel handle.
     * Only the specified argument index is updated; all others remain as previously set.
     */
    default void setKernelArg(KernelHandle kernel, int argIndex, Object arg) {
        throw new UnsupportedOperationException("setKernelArg not implemented");
    }

    /**
     * Enqueues a kernel dispatch WITHOUT setting any arguments.
     * All arguments must have been pre-bound via {@link #setKernelArg} or a prior
     * {@link #enqueueKernel} call on the same kernel handle.
     * No event is created for the dispatch.
     */
    default void enqueueKernelPreBound(KernelHandle kernel, long[] globalWorkSize,
                                        long[] localWorkSize, GpuEvent... waitOnEvents) {
        throw new UnsupportedOperationException("enqueueKernelPreBound not implemented");
    }

    /**
     * Enqueues a pre-bound kernel dispatch and returns a GPU event for synchronization.
     * Used for cross-queue dependency tracking in multi-queue dispatch.
     */
    default GpuEvent enqueueKernelPreBoundWithEvent(KernelHandle kernel, long[] globalWorkSize,
                                                     long[] localWorkSize, GpuEvent... waitOnEvents) {
        throw new UnsupportedOperationException("enqueueKernelPreBoundWithEvent not implemented");
    }

    /**
     * Fills {@code [dstOffset, dstOffset + sizeInBytes)} of {@code dst} with a repeating 32-bit
     * {@code pattern}, stored little-endian. Offset and size must be multiples of 4.
     *
     * <p>The real backends override this with a device-side fill. The portable default issues
     * chunked {@link #enqueueWriteBuffer(GpuBuffer, long, long, byte[], GpuEvent...)} calls from a
     * pattern array of at most {@link #FILL_FALLBACK_CHUNK_BYTES} bytes, each array exactly as long
     * as its chunk; the waits apply to the first chunk and in-order execution orders the rest.</p>
     *
     * @throws IllegalArgumentException when the range is negative, misaligned or outside {@code dst}
     */
    default void enqueueFillBuffer(GpuBuffer dst, long dstOffset, long sizeInBytes, int pattern,
                                   GpuEvent... waitOnEvents) {
        if (dst == null) {
            throw new IllegalArgumentException("fill destination must not be null");
        }
        if (dstOffset < 0 || sizeInBytes < 0 || (dstOffset & 3) != 0 || (sizeInBytes & 3) != 0) {
            throw new IllegalArgumentException("fill offset " + dstOffset + " and size " + sizeInBytes
                    + " must be non-negative multiples of 4");
        }
        if (dstOffset + sizeInBytes > dst.sizeInBytes()) {
            throw new IllegalArgumentException("fill range [" + dstOffset + ", " + (dstOffset + sizeInBytes)
                    + ") exceeds the buffer size of " + dst.sizeInBytes() + " bytes");
        }
        if (sizeInBytes == 0) {
            return;
        }
        int chunk = (int) Math.min(sizeInBytes, FILL_FALLBACK_CHUNK_BYTES);
        byte[] patternBytes = new byte[chunk];
        ByteBuffer fill = ByteBuffer.wrap(patternBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < chunk; i += 4) {
            fill.putInt(i, pattern);
        }
        long written = 0;
        while (written < sizeInBytes) {
            int n = (int) Math.min(chunk, sizeInBytes - written);
            byte[] src = n == chunk ? patternBytes : Arrays.copyOf(patternBytes, n);
            GpuEvent evt = written == 0
                    ? enqueueWriteBuffer(dst, dstOffset, n, src, waitOnEvents)
                    : enqueueWriteBuffer(dst, dstOffset + written, n, src);
            if (evt != null) {
                evt.close();
            }
            written += n;
        }
    }

    /**
     * Returns an event that completes when all work enqueued on this queue so far, and the
     * given waits, have completed.
     */
    default GpuEvent enqueueMarker(GpuEvent... waitOnEvents) {
        throw new UnsupportedOperationException("enqueueMarker not implemented");
    }

    void flush();

    void finish();

    void close();
}

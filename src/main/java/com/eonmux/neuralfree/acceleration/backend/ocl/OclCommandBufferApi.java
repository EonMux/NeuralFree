package com.eonmux.neuralfree.acceleration.backend.ocl;

import java.util.List;

/**
 * The {@code cl_khr_command_buffer} entry points, isolated behind an interface so the queue's
 * recording logic does not depend on how the extension functions are reached (JOCL does not bind
 * them; {@link JnaOclCommandBufferApi} calls them through JNA).
 *
 * <p>Every command takes the sync point of the command recorded before it, so the recorded
 * sequence keeps the dispatch order of the in-order queue it was captured from. The returned sync
 * point identifies the new command and is the wait of the next one; {@link #NO_SYNC_POINT} means
 * "no predecessor". Buffers, kernels and queues are native handles; the caller keeps them open
 * for the lifetime of the command buffer.</p>
 */
interface OclCommandBufferApi {

    /** The sync point value that names no command. */
    int NO_SYNC_POINT = -1;

    /** One argument value of a mutable dispatch, as the raw bytes {@code clSetKernelArg} took. */
    final class MutableArg {
        final int index;
        final byte[] value;

        MutableArg(int index, byte[] value) {
            this.index = index;
            this.value = value;
        }
    }

    /** One recorded dispatch whose arguments a replay updates in place. */
    final class MutableDispatch {
        final long mutableCommand;
        final List<MutableArg> arguments;

        MutableDispatch(long mutableCommand, List<MutableArg> arguments) {
            this.mutableCommand = mutableCommand;
            this.arguments = arguments;
        }
    }

    /** One recorded command: its sync point and, for a mutable dispatch, its mutable handle. */
    final class Recorded {
        final int syncPoint;
        final long mutableCommand;

        Recorded(int syncPoint, long mutableCommand) {
            this.syncPoint = syncPoint;
            this.mutableCommand = mutableCommand;
        }
    }

    /** True when the command buffer may be enqueued again while an earlier submission is pending. */
    boolean simultaneousUse();

    /** True when recorded dispatches expose mutable handles and their arguments can be updated. */
    boolean mutableArguments();

    /**
     * Creates a command buffer for one queue.
     *
     * @param simultaneousUse create it for repeated submission
     * @param mutable         record dispatches with mutable handles
     * @return the native command buffer
     */
    long createCommandBuffer(long queue, boolean simultaneousUse, boolean mutable);

    /** Finalizes a recorded command buffer ({@code clFinalizeCommandBufferKHR}). */
    void finalizeCommandBuffer(long commandBuffer);

    /** Releases a command buffer ({@code clReleaseCommandBufferKHR}). */
    void releaseCommandBuffer(long commandBuffer);

    /**
     * Enqueues a finalized command buffer on {@code queue}, after {@code waitEvents}.
     * The command buffer's completion is ordered after the enqueue on the in-order queue.
     */
    void enqueueCommandBuffer(long commandBuffer, long queue, long[] waitEvents);

    /**
     * Records a kernel dispatch.
     *
     * @param waitSyncPoint the predecessor's sync point, or {@link #NO_SYNC_POINT}
     * @return the new sync point and mutable handle
     */
    Recorded commandNDRangeKernel(long commandBuffer, long queue, long kernel, int workDim,
                                  long[] globalWorkSize, long[] localWorkSize, int waitSyncPoint);

    /** Records a device-to-device buffer copy, after {@code waitSyncPoint}. */
    int commandCopyBuffer(long commandBuffer, long queue, long src, long dst, long srcOffset,
                          long dstOffset, long size, int waitSyncPoint);

    /** Records a 32-bit fill of {@code dst}, after {@code waitSyncPoint}. */
    int commandFillBuffer(long commandBuffer, long queue, long dst, int pattern, long offset,
                          long size, int waitSyncPoint);

    /**
     * Replaces the arguments of recorded mutable dispatches in one
     * {@code clUpdateMutableCommandsKHR} call. The command buffer must not be pending.
     */
    void updateMutableArguments(long commandBuffer, List<MutableDispatch> updates);
}

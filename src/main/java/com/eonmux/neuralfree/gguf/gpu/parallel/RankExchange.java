package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

/**
 * Copies between the rank devices of a tensor-parallel split. {@link #replay} sends one rank's input rows to another
 * rank, device to device when the queues support peer copies and through host memory otherwise, and waits for the
 * copy so the receiving rank's kernels read the whole input.
 */
final class RankExchange {

    private RankExchange() {
    }

    /**
     * Copies {@code floats} floats from {@code src} element {@code srcOffset} to the start of {@code dst} and waits
     * for the destination queue.
     */
    static void replay(ComputeQueue src, GpuBuffer srcBuffer, long srcOffset, ComputeQueue dst, GpuBuffer dstBuffer,
                       long floats) {
        replay(src, srcBuffer, srcOffset, dst, dstBuffer, 0L, floats);
    }

    /**
     * Copies {@code floats} floats from {@code src} element {@code srcOffset} to {@code dst} element
     * {@code dstOffset} and waits for the destination queue.
     */
    static void replay(ComputeQueue src, GpuBuffer srcBuffer, long srcOffset, ComputeQueue dst, GpuBuffer dstBuffer,
                       long dstOffset, long floats) {
        long bytes = floats * Float.BYTES;
        if (bytes <= 0L) {
            return;
        }
        long srcBytes = srcOffset * Float.BYTES;
        long dstBytes = dstOffset * Float.BYTES;
        if (src == dst) {
            src.enqueueCopyBuffer(srcBuffer, srcBytes, dstBuffer, dstBytes, bytes);
            src.finish();
            return;
        }
        if (src.supportsPeerCopy(dst)) {
            dst.finish();
            src.enqueuePeerCopy(dst, srcBuffer, srcBytes, dstBuffer, dstBytes, bytes);
            src.finish();
            return;
        }
        float[] staged = new float[(int) Math.min(Integer.MAX_VALUE, floats)];
        try (GpuEvent read = src.enqueueReadBuffer(srcBuffer, srcBytes, bytes, staged)) {
            src.finish();
        }
        try (GpuEvent write = dst.enqueueWriteBuffer(dstBuffer, dstBytes, bytes, staged)) {
            dst.finish();
        }
    }

    /**
     * Copies {@code bytes} bytes from {@code srcBuffer} byte {@code srcByteOffset} to {@code dstBuffer} byte
     * {@code dstByteOffset} and waits for the destination queue: the form for whole cache rows and weight shards,
     * whose byte sizes are not float counts.
     */
    static void replayBytes(ComputeQueue src, GpuBuffer srcBuffer, long srcByteOffset, ComputeQueue dst,
                            GpuBuffer dstBuffer, long dstByteOffset, long bytes) {
        if (bytes <= 0L) {
            return;
        }
        if (src == dst) {
            src.enqueueCopyBuffer(srcBuffer, srcByteOffset, dstBuffer, dstByteOffset, bytes);
            src.finish();
            return;
        }
        if (src.supportsPeerCopy(dst)) {
            dst.finish();
            src.enqueuePeerCopy(dst, srcBuffer, srcByteOffset, dstBuffer, dstByteOffset, bytes);
            src.finish();
            return;
        }
        byte[] staged = new byte[(int) bytes];
        try (GpuEvent read = src.enqueueReadBuffer(srcBuffer, srcByteOffset, bytes, staged)) {
            src.finish();
        }
        try (GpuEvent write = dst.enqueueWriteBuffer(dstBuffer, dstByteOffset, bytes, staged)) {
            dst.finish();
        }
    }
}

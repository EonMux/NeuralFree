package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;

import java.util.ArrayList;
import java.util.List;

/**
 * Copies between the rank devices of a tensor-parallel split. {@link #replay} sends one rank's input rows to another
 * rank, device to device when the queues support peer copies and through host memory otherwise, and waits for the
 * copy so the receiving rank's kernels read the whole input. {@link #replayAll} sends one input to every peer rank
 * and waits for the source queue once.
 */
final class RankExchange {

    /**
     * Largest host staging a forward thread keeps between exchanges, in floats (4 MiB): every decode exchange and
     * most prefill chunks fit it; a larger exchange takes an array of its own, so no thread holds more than this.
     */
    static final int KEPT_STAGING_FLOATS = 1 << 20;
    /** Host staging of the exchanges that cross through host memory, one array per forward thread. */
    private static final ThreadLocal<float[]> STAGING = new ThreadLocal<>();

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
        float[] staged = staging(floats);
        try (GpuEvent read = src.enqueueReadBuffer(srcBuffer, srcBytes, bytes, staged)) {
            src.finish();
        }
        try (GpuEvent write = dst.enqueueWriteBuffer(dstBuffer, dstBytes, bytes, staged)) {
            dst.finish();
        }
    }

    /**
     * Sends {@code inFloats} floats of {@code in} (on {@code src}) to the input slot of every peer rank
     * {@code 1 .. ranks - 1} of {@code group} ({@link #replayAll}), sizing those slots first, and the peers' partial
     * slots to {@code partialFloats} when it is positive; the peers' parts can then be enqueued before rank 0's and
     * run beside it.
     */
    static void replayToPeers(TensorParallelGroup group, int ranks, ComputeQueue src, ScratchSlot in, long inFloats,
                              long partialFloats) {
        List<ComputeQueue> queues = new ArrayList<>(Math.max(0, ranks - 1));
        List<GpuBuffer> inputs = new ArrayList<>(Math.max(0, ranks - 1));
        for (int rank = 1; rank < ranks; rank++) {
            ScratchSlot peerIn = group.inputSlot(rank);
            peerIn.require(inFloats);
            if (partialFloats > 0) {
                group.partialSlot(rank).require(partialFloats);
            }
            queues.add(group.rankQueue(rank));
            inputs.add(peerIn.buffer());
        }
        replayAll(src, in.buffer(), in.offset(), queues, inputs, inFloats);
    }

    /**
     * Copies {@code floats} floats from {@code src} element {@code srcOffset} to element 0 of every
     * {@code dstBuffers[i]}, on {@code dsts[i]}'s device. The source queue is finished once for all of them: a peer
     * copy runs on it after its destination queue finished, and a host copy reads the source once and writes it on
     * each destination queue, whose later work runs after the write in queue order. Every destination's kernels
     * enqueued afterwards read the whole input.
     *
     * @throws IllegalArgumentException when the destination lists differ in length
     */
    static void replayAll(ComputeQueue src, GpuBuffer srcBuffer, long srcOffset, List<ComputeQueue> dsts,
                          List<GpuBuffer> dstBuffers, long floats) {
        if (dsts.size() != dstBuffers.size()) {
            throw new IllegalArgumentException(dsts.size() + " destination queues for " + dstBuffers.size()
                    + " destination buffers");
        }
        long bytes = floats * Float.BYTES;
        if (bytes <= 0L || dsts.isEmpty()) {
            return;
        }
        long srcBytes = srcOffset * Float.BYTES;
        float[] staged = null;
        for (int i = 0; i < dsts.size(); i++) {
            ComputeQueue dst = dsts.get(i);
            GpuBuffer dstBuffer = dstBuffers.get(i);
            if (dst == src) {
                src.enqueueCopyBuffer(srcBuffer, srcBytes, dstBuffer, 0L, bytes);
            } else if (src.supportsPeerCopy(dst)) {
                dst.finish();
                src.enqueuePeerCopy(dst, srcBuffer, srcBytes, dstBuffer, 0L, bytes);
            } else {
                if (staged == null) {
                    staged = staging(floats);
                    try (GpuEvent read = src.enqueueReadBuffer(srcBuffer, srcBytes, bytes, staged)) {
                        src.finish();
                    }
                }
                try (GpuEvent write = dst.enqueueWriteBuffer(dstBuffer, 0L, bytes, staged)) {
                    // the write copied the staged rows at enqueue; the destination's kernels follow it in queue order
                }
            }
        }
        src.finish();
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

    /**
     * A host staging array of at least {@code floats} floats: the calling thread's own, grown on demand up to
     * {@link #KEPT_STAGING_FLOATS} and kept for its next exchange, or a new array for a larger exchange. A host copy
     * reads into it and writes from it, and a write copies its source at enqueue, so the array is free again once the
     * exchange returns.
     *
     * @throws IllegalArgumentException when {@code floats} exceeds one array
     */
    static float[] staging(long floats) {
        if (floats > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException(floats + " floats exceed one host staging array");
        }
        if (floats > KEPT_STAGING_FLOATS) {
            return new float[(int) floats];
        }
        float[] held = STAGING.get();
        if (held == null || held.length < floats) {
            held = new float[(int) floats];
            STAGING.set(held);
        }
        return held;
    }
}

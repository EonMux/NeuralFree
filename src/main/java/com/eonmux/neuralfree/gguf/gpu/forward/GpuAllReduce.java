package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Blocking, in-process FP32 sum all-reduce over device buffer views. Rank zero accumulates in rank order using
 * {@link GpuOps#add}, then broadcasts the sum. Copies use the existing queue's device or peer-copy path when
 * available; otherwise they stage through host memory. All participating queues are finished before and after the
 * operation. The caller owns the buffers, queues, operations and root-device temporary, and must not use them
 * concurrently with the collective.
 */
public final class GpuAllReduce {
    /** Maximum staging and root temporary size, in floats. */
    public static final int CHUNK_FLOATS = 1 << 18;

    private GpuAllReduce() {
    }

    /** A rank's queue and its contiguous FP32 buffer view, with offset measured in floats. */
    public static final class Participant {
        private final ComputeQueue queue;
        private final GpuBuffer buffer;
        private final int offset;

        public Participant(ComputeQueue queue, GpuBuffer buffer, int offset) {
            this.queue = Objects.requireNonNull(queue, "queue");
            this.buffer = Objects.requireNonNull(buffer, "buffer");
            if (offset < 0) {
                throw new IllegalArgumentException("offset must be non-negative: " + offset);
            }
            this.offset = offset;
        }
    }

    /**
     * Sums {@code count} floats on every rank and replaces each rank's view with the sum. The first participant
     * must use {@code ops.queue()}; {@code temporary} belongs to that queue's backend, starts at element zero and
     * holds at least {@code min(count, CHUNK_FLOATS)} floats. Participant views and the temporary must not overlap.
     *
     * @throws IllegalArgumentException when a view, count, queue or temporary is invalid
     */
    public static void sum(GpuOps ops, GpuBuffer temporary, List<Participant> participants, int count) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(temporary, "temporary");
        Objects.requireNonNull(participants, "participants");
        List<Participant> ranks = new ArrayList<>(participants);
        validate(ops, temporary, ranks, count);
        for (Participant rank : ranks) {
            rank.queue.finish();
        }
        for (int offset = 0; offset < count;) {
            int chunk = Math.min(CHUNK_FLOATS, count - offset);
            reduceChunk(ops, temporary, ranks, offset, chunk);
            offset += chunk;
        }
        for (Participant rank : ranks) {
            rank.queue.finish();
        }
    }

    private static void reduceChunk(GpuOps ops, GpuBuffer temporary, List<Participant> ranks, int offset, int count) {
        Participant root = ranks.get(0);
        for (int i = 1; i < ranks.size(); i++) {
            copy(ranks.get(i), offset, root.queue, temporary, 0, count);
            ops.add(root.buffer, root.offset + offset, temporary, 0, root.buffer, root.offset + offset, count);
        }
        root.queue.finish();
        for (int i = 1; i < ranks.size(); i++) {
            Participant rank = ranks.get(i);
            copy(root, offset, rank.queue, rank.buffer, rank.offset + offset, count);
        }
    }

    private static void validate(GpuOps ops, GpuBuffer temporary, List<Participant> ranks, int count) {
        if (count < 1) {
            throw new IllegalArgumentException("count must be positive: " + count);
        }
        if (ranks.isEmpty()) {
            throw new IllegalArgumentException("participants must not be empty");
        }
        if ((long) Math.min(count, CHUNK_FLOATS) * Float.BYTES > temporary.sizeInBytes()) {
            throw new IllegalArgumentException("temporary is too small for count " + count);
        }
        for (int i = 0; i < ranks.size(); i++) {
            Participant p = Objects.requireNonNull(ranks.get(i), "participants[" + i + "]");
            long end = (long) p.offset + count;
            if (end > Integer.MAX_VALUE || end * Float.BYTES > p.buffer.sizeInBytes()) {
                throw new IllegalArgumentException("participants[" + i + "] view exceeds its buffer or int range");
            }
            if (p.buffer == temporary) {
                throw new IllegalArgumentException("temporary must be distinct from participants[" + i + "]");
            }
            for (int j = 0; j < i; j++) {
                Participant other = ranks.get(j);
                if (p.buffer == other.buffer && p.offset < (long) other.offset + count
                        && other.offset < end) {
                    throw new IllegalArgumentException("participants[" + i + "] overlaps participants[" + j + "]");
                }
            }
        }
        if (ranks.get(0).queue != ops.queue()) {
            throw new IllegalArgumentException("participants[0] queue must be ops.queue()");
        }
    }

    /** Copies between queues only after previous uses of the destination have completed. */
    private static void copy(Participant src, int srcOffset, ComputeQueue dstQueue, GpuBuffer dst, int dstOffset,
                             int count) {
        long bytes = (long) count * Float.BYTES;
        long srcBytes = ((long) src.offset + srcOffset) * Float.BYTES;
        long dstBytes = (long) dstOffset * Float.BYTES;
        if (src.queue == dstQueue) {
            src.queue.enqueueCopyBuffer(src.buffer, srcBytes, dst, dstBytes, bytes);
        } else if (src.queue.supportsPeerCopy(dstQueue)) {
            dstQueue.finish();
            src.queue.enqueuePeerCopy(dstQueue, src.buffer, srcBytes, dst, dstBytes, bytes);
            src.queue.finish();
        } else {
            float[] staged = new float[count];
            try (GpuEvent read = src.queue.enqueueReadBuffer(src.buffer, srcBytes, bytes, staged)) {
                src.queue.finish();
            }
            try (GpuEvent write = dstQueue.enqueueWriteBuffer(dst, dstBytes, bytes, staged)) {
                dstQueue.finish();
            }
        }
    }
}

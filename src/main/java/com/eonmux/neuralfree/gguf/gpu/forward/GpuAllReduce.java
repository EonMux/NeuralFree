package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * In-process FP32 sum collectives over device buffer views. Rank zero accumulates in rank order using
 * {@link GpuOps#add}. {@link #sum} is the blocking all-reduce: it finishes every participating queue before and
 * after, and broadcasts the sum to every rank. {@link #reduceToRoot} leaves the sum on rank zero only, for the
 * tensor-parallel blocks whose peers' partials are scratch: it waits only for what each copy needs and leaves rank
 * zero's queue running. Copies use the existing queue's device or peer-copy path when available; otherwise they
 * stage through one host array per call. The caller owns the buffers, queues, operations and root-device temporary,
 * and must not use them concurrently with the collective.
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
        float[] staging = new float[Math.min(CHUNK_FLOATS, count)];
        for (int offset = 0; offset < count;) {
            int chunk = Math.min(CHUNK_FLOATS, count - offset);
            accumulateChunk(ops, temporary, ranks, offset, chunk, staging);
            ranks.get(0).queue.finish();
            Participant root = ranks.get(0);
            for (int i = 1; i < ranks.size(); i++) {
                Participant rank = ranks.get(i);
                copy(root, offset, rank.queue, rank.buffer, rank.offset + offset, chunk, staging);
            }
            offset += chunk;
        }
        for (Participant rank : ranks) {
            rank.queue.finish();
        }
    }

    /**
     * Adds every rank's {@code count} floats into rank zero's view, which ends up holding the sum; the other ranks'
     * views are left as they were. Rank zero's queue is not finished: work enqueued on it afterwards reads the sum.
     * Each peer's view is read after the work its queue holds, so a peer's queue is finished only when its copy
     * crosses queues. The first participant must use {@code ops.queue()}; {@code temporary} belongs to that queue's
     * backend, starts at element zero and holds at least {@code min(count, CHUNK_FLOATS)} floats. Participant views
     * and the temporary must not overlap.
     *
     * @throws IllegalArgumentException when a view, count, queue or temporary is invalid
     */
    public static void reduceToRoot(GpuOps ops, GpuBuffer temporary, List<Participant> participants, int count) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(temporary, "temporary");
        Objects.requireNonNull(participants, "participants");
        List<Participant> ranks = new ArrayList<>(participants);
        validate(ops, temporary, ranks, count);
        if (ranks.size() == 1) {
            return;
        }
        float[] staging = new float[Math.min(CHUNK_FLOATS, count)];
        for (int offset = 0; offset < count;) {
            int chunk = Math.min(CHUNK_FLOATS, count - offset);
            accumulateChunk(ops, temporary, ranks, offset, chunk, staging);
            offset += chunk;
        }
    }

    /** Adds chunk {@code [offset, offset + count)} of every peer into rank zero's view, through the temporary. */
    private static void accumulateChunk(GpuOps ops, GpuBuffer temporary, List<Participant> ranks, int offset,
                                        int count, float[] staging) {
        Participant root = ranks.get(0);
        for (int i = 1; i < ranks.size(); i++) {
            copy(ranks.get(i), offset, root.queue, temporary, 0, count, staging);
            ops.add(root.buffer, root.offset + offset, temporary, 0, root.buffer, root.offset + offset, count);
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

    /**
     * Copies {@code count} floats of {@code src}'s view to {@code dst} on {@code dstQueue}: on one queue in queue
     * order; by a peer copy after {@code dstQueue}'s earlier uses of the destination completed, finished before the
     * destination is read; otherwise through {@code staging}, read once {@code src}'s queue finished and written on
     * {@code dstQueue} in its order (a write copies its source at enqueue).
     */
    private static void copy(Participant src, int srcOffset, ComputeQueue dstQueue, GpuBuffer dst, int dstOffset,
                             int count, float[] staging) {
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
            try (GpuEvent read = src.queue.enqueueReadBuffer(src.buffer, srcBytes, bytes, staging)) {
                src.queue.finish();
            }
            try (GpuEvent write = dstQueue.enqueueWriteBuffer(dst, dstBytes, bytes, staging)) {
                // the write copied the staging array at enqueue; queue order covers the destination's later reads
            }
        }
    }
}

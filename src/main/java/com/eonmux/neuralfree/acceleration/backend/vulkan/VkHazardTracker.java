package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;

import java.util.Arrays;

/**
 * Tracks the buffer ranges read and written since the last barrier of a command batch, so a
 * barrier is recorded only for read-after-write, write-after-read and write-after-write hazards.
 *
 * <p>An access is a byte range {@code [start, end)} of one buffer, identified by its
 * {@link VkGpuBuffer#trackingId()} (ids are positive; 0 means "no access"). Dispatches access
 * whole buffers ({@code [0, }{@link #WHOLE}{@code )}): their read and write sets come from
 * reflection, a binding the module may read is a read and one it may write is a write, and the same
 * buffer bound twice is one access per kind. Transfers access exact ranges: a copy reads its source
 * range and writes its destination range, a fill writes its range. Two accesses conflict when
 * they name the same buffer and their ranges overlap; read-read never conflicts.</p>
 *
 * <p>After {@link #MAX_PENDING} accesses of one kind the tracker reports a hazard for the next
 * command regardless of overlap, so the linear scans stay short; the resulting barrier is
 * conservative and always correct. Adjacent accesses of one buffer are merged, so a write split
 * into consecutive chunks occupies one entry.</p>
 */
@NotThreadSafe("Owned by one compute queue; used under the queue's lock.")
public final class VkHazardTracker {

    /** End of a whole-buffer range. */
    public static final long WHOLE = Long.MAX_VALUE;
    /** Pending accesses of one kind after which every new command needs a barrier. */
    public static final int MAX_PENDING = 1024;

    private static final int INITIAL_CAPACITY = 32;

    private final Accesses reads = new Accesses();
    private final Accesses writes = new Accesses();

    /**
     * True when a command reading the whole buffers {@code readIds} and writing the whole buffers
     * {@code writeIds} conflicts with the pending accesses.
     */
    public boolean needsBarrier(long[] readIds, long[] writeIds) {
        int nr = readIds == null ? 0 : readIds.length;
        int nw = writeIds == null ? 0 : writeIds.length;
        if (reads.count + nr > MAX_PENDING || writes.count + nw > MAX_PENDING) {
            return nr + nw > 0;
        }
        for (int i = 0; i < nr; i++) {
            if (writes.overlaps(readIds[i], 0, WHOLE)) {
                return true;
            }
        }
        for (int i = 0; i < nw; i++) {
            long id = writeIds[i];
            if (writes.overlaps(id, 0, WHOLE) || reads.overlaps(id, 0, WHOLE)) {
                return true;
            }
        }
        return false;
    }

    /** Adds a dispatch's whole-buffer accesses to the pending sets. */
    public void record(long[] readIds, long[] writeIds) {
        if (readIds != null) {
            for (long id : readIds) {
                reads.add(id, 0, WHOLE);
            }
        }
        if (writeIds != null) {
            for (long id : writeIds) {
                writes.add(id, 0, WHOLE);
            }
        }
    }

    /**
     * True when a transfer reading {@code [readOffset, readOffset + readSize)} of buffer {@code readId}
     * and writing {@code [writeOffset, writeOffset + writeSize)} of buffer {@code writeId} conflicts
     * with the pending accesses. An id of 0 means the transfer has no access of that kind.
     */
    public boolean needsBarrier(long readId, long readOffset, long readSize,
                                long writeId, long writeOffset, long writeSize) {
        boolean hasRead = readId != 0 && readSize > 0;
        boolean hasWrite = writeId != 0 && writeSize > 0;
        if ((hasRead && reads.count >= MAX_PENDING) || (hasWrite && writes.count >= MAX_PENDING)) {
            return true;
        }
        if (hasRead && writes.overlaps(readId, readOffset, readOffset + readSize)) {
            return true;
        }
        if (hasWrite) {
            long end = writeOffset + writeSize;
            return writes.overlaps(writeId, writeOffset, end) || reads.overlaps(writeId, writeOffset, end);
        }
        return false;
    }

    /** Adds a transfer's range accesses to the pending sets; an id of 0 means no access of that kind. */
    public void record(long readId, long readOffset, long readSize, long writeId, long writeOffset, long writeSize) {
        if (readId != 0 && readSize > 0) {
            reads.add(readId, readOffset, readOffset + readSize);
        }
        if (writeId != 0 && writeSize > 0) {
            writes.add(writeId, writeOffset, writeOffset + writeSize);
        }
    }

    /** Clears the pending sets after a barrier or at the start of a batch. */
    public void reset() {
        reads.count = 0;
        writes.count = 0;
    }

    /** Number of pending read accesses. */
    public int pendingReads() {
        return reads.count;
    }

    /** Number of pending write accesses. */
    public int pendingWrites() {
        return writes.count;
    }

    @Override
    public String toString() {
        return "VkHazardTracker{reads=" + reads.count + ", writes=" + writes.count + '}';
    }

    /** Growable parallel arrays of (id, start, end) ranges. */
    private static final class Accesses {
        long[] ids = new long[INITIAL_CAPACITY];
        long[] starts = new long[INITIAL_CAPACITY];
        long[] ends = new long[INITIAL_CAPACITY];
        int count;

        boolean overlaps(long id, long start, long end) {
            for (int i = 0; i < count; i++) {
                if (ids[i] == id && starts[i] < end && start < ends[i]) {
                    return true;
                }
            }
            return false;
        }

        void add(long id, long start, long end) {
            for (int i = count - 1; i >= 0; i--) {
                if (ids[i] != id) {
                    continue;
                }
                if (starts[i] <= start && end <= ends[i]) {
                    return;
                }
                if (start <= ends[i] && starts[i] <= end) {
                    starts[i] = Math.min(starts[i], start);
                    ends[i] = Math.max(ends[i], end);
                    return;
                }
            }
            if (count == ids.length) {
                int capacity = ids.length * 2;
                ids = Arrays.copyOf(ids, capacity);
                starts = Arrays.copyOf(starts, capacity);
                ends = Arrays.copyOf(ends, capacity);
            }
            ids[count] = id;
            starts[count] = start;
            ends[count] = end;
            count++;
        }
    }
}

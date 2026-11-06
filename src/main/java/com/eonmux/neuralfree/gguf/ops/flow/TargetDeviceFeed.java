package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuHiddenFeed;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The NextN drafter's {@link HiddenTap} on a GPU target whose device the drafter shares ({@link GpuMtpHead} over
 * the target's own context): every tap point's rows are captured on the device ({@link GpuHiddenFeed}) — one
 * device-to-device copy per ubatch chunk, enqueued on the target's queue before the call's finish — and the drafter
 * drains device batches whose rows never reach the host. Rows a host path delivers anyway (a batched call whose
 * rows land at their call rows, a foreign queue) are uploaded into this feed's slots, so every batch is a device
 * batch.
 *
 * <p>Batches queue until the drafter drains them; a call that starts at or before an earlier call's start replaces
 * it (a recurrent target restored its state and replayed the accepted tokens). The slots are named temporaries of
 * the shared scratch, handed out with a reference count: a slot stays out of the pool while a pending batch, the
 * drafter's row state or a deferred catch-up row reads it. Not thread-safe; the forward passes of the target and
 * its drafter alternate on one thread.
 */
final class TargetDeviceFeed implements HiddenTap, GpuHiddenFeed {

    /** The device rows of one ubatch chunk of a call: the feed's rows and the count they hold. */
    static final class Chunk {
        final MtpHead.Rows rows;
        final int nRows;
        /** Host staging of a chunk captured on another device that has no peer copy; null otherwise. */
        final float[] staged;
        /** The read of {@link #staged} on the capturing queue; null when nothing is staged. */
        final GpuEvent stagedRead;

        Chunk(MtpHead.Rows rows, int nRows) {
            this(rows, nRows, null, null);
        }

        Chunk(MtpHead.Rows rows, int nRows, float[] staged, GpuEvent stagedRead) {
            this.rows = rows;
            this.nRows = nRows;
            this.staged = staged;
            this.stagedRead = stagedRead;
        }

        @Override
        public String toString() {
            return rows + " x " + nRows;
        }
    }

    /** One target forward call's device rows: positions {@code pos0 ..}, its tokens and the chunks in call order. */
    static final class DeviceBatch {
        final int pos0;
        final int[] tokens;
        final List<Chunk> chunks;
        final int nRows;

        DeviceBatch(int pos0, int[] tokens, List<Chunk> chunks, int nRows) {
            this.pos0 = pos0;
            this.tokens = tokens;
            this.chunks = chunks;
            this.nRows = nRows;
        }

        int size() {
            return tokens.length;
        }

        @Override
        public String toString() {
            return "DeviceBatch{pos0=" + pos0 + ", n=" + tokens.length + "}";
        }
    }

    /** A device row: the rows it lives in and its index there. */
    static final class Located {
        final MtpHead.Rows rows;
        final int row;

        Located(MtpHead.Rows rows, int row) {
            this.rows = rows;
            this.row = row;
        }
    }

    private final MtpHead head;
    private final ModelMemory memory;
    private final int point;
    private final int width;
    /** The queue the head's rows live on (the head's device). */
    private final ComputeQueue queue;
    /** The queue that captures the tap rows (the last trunk layer's device). */
    private final ComputeQueue source;
    /** True when the source rows cross to the head's device by a device-to-device peer copy. */
    private final boolean peer;    private final List<MtpHead.Rows> pool = new ArrayList<>();
    private final Map<MtpHead.Rows, Integer> refs = new HashMap<>();
    private final List<DeviceBatch> pending = new ArrayList<>();
    private List<Chunk> current = new ArrayList<>();
    private int currentRows;
    private int slotsMade;

    TargetDeviceFeed(MtpHead head, ModelMemory memory, int point, int width) {
        this(head, memory, point, width, head.gpuContext() == null ? null : head.gpuContext().queue());
    }

    /**
     * A feed whose tap rows are captured on {@code source} and kept on the head's device. When the two are the same
     * queue the rows copy device-to-device; otherwise the source queue copies them directly when it can reach the
     * head's device (a peer copy), and stages them through the host when it cannot.
     */
    TargetDeviceFeed(MtpHead head, ModelMemory memory, int point, int width, ComputeQueue source) {
        if (head.gpuContext() == null) {
            throw new IllegalArgumentException("a device feed needs a head on the target's device");
        }
        if (width < 1) {
            throw new IllegalArgumentException("the device feed needs a positive width, got " + width);
        }
        this.head = head;
        this.memory = Objects.requireNonNull(memory, "target memory");
        this.point = point;
        this.width = width;
        this.queue = head.gpuContext().queue();
        this.source = Objects.requireNonNull(source, "capture queue");
        this.peer = source != queue && source.supportsPeerCopy(queue);
    }

    /** The device row state the drafter keeps across calls: the pending row, the bridge row, the last call's rows. */
    DeviceRows deviceRows() {
        return new DeviceRows();
    }

    @Override
    public boolean wants(int layer) {
        return layer == point;
    }

    @Override
    public boolean wantsDevice(int layer, ComputeQueue forwardQueue) {
        return layer == point && forwardQueue == source;
    }

    private void checkTap(int layer, int tapWidth) {
        if (layer != point) {
            throw new IllegalStateException("tap point " + layer + " was not requested (" + point + ")");
        }
        if (tapWidth != width) {
            throw new IllegalStateException("tap point " + layer + " delivers rows of " + tapWidth
                    + " floats; the drafter reads " + width);
        }
    }

    @Override
    public void captureDevice(int layer, GpuBuffer rows, long rowOffset, int nRows, int tapWidth) {
        checkTap(layer, tapWidth);
        MtpHead.Rows slot = take();
        MtpHead.Region dst = head.region(slot);
        long bytes = (long) nRows * width * Float.BYTES;
        long srcBytes = rowOffset * Float.BYTES;
        if (source == queue) {
            source.enqueueCopyBuffer(rows, srcBytes, dst.buffer, dst.offset * Float.BYTES, bytes);
            current.add(new Chunk(slot, nRows));
        } else if (peer) {
            source.enqueuePeerCopy(queue, rows, srcBytes, dst.buffer, dst.offset * Float.BYTES, bytes);
            current.add(new Chunk(slot, nRows));
        } else {
            float[] staged = new float[Math.multiplyExact(nRows, width)];
            GpuEvent read = source.enqueueReadBuffer(rows, srcBytes, bytes, staged);
            current.add(new Chunk(slot, nRows, staged, read));
        }
        currentRows += nRows;
    }

    @Override
    public void onHiddenDevice(int layer, int nRows, int tapWidth) {
        checkTap(layer, tapWidth);
        if (nRows != currentRows) {
            throw new IllegalStateException("device feed holds " + currentRows + " rows, the call delivered "
                    + nRows);
        }
        complete();
    }

    /** The host delivery of a call whose rows could not be captured on the device: uploaded into the slots. */
    @Override
    public void onHidden(int layer, float[] rows, int nRows, int tapWidth) {
        checkTap(layer, tapWidth);
        if (currentRows != 0) {
            throw new IllegalStateException("device feed holds " + currentRows + " rows of an undelivered call");
        }
        if (nRows > memory.sequence().nPast()) {
            throw new IllegalStateException("a forward call of " + nRows + " rows left "
                    + memory.sequence().nPast() + " stored positions");
        }
        int off = 0;
        while (off < nRows) {
            int n = Math.min(ubatch(), nRows - off);
            MtpHead.Rows slot = take();
            head.write(slot, Arrays.copyOfRange(rows, off * width, (off + n) * width), n);
            current.add(new Chunk(slot, n));
            currentRows += n;
            off += n;
        }
        complete();
    }

    private void complete() {
        flushStaged();
        int n = currentRows;
        int nPast = memory.sequence().nPast();
        if (nPast < n) {
            throw new IllegalStateException("a forward call of " + n + " rows left " + nPast + " stored positions");
        }
        int pos0 = nPast - n;
        int[] tokens = new int[n];
        for (int t = 0; t < n; t++) {
            tokens[t] = memory.sequence().token(pos0 + t);
        }
        pending.add(new DeviceBatch(pos0, tokens, new ArrayList<>(current), n));
        current = new ArrayList<>();
        currentRows = 0;
    }

    /**
     * Writes the chunks a source device without a peer copy staged in host memory into the head's slots, once their
     * reads completed. Called after the call's queues finished (the split backend finishes every device before it
     * delivers); the final {@code finish} makes the rows visible to the head before it drains them.
     */
    private void flushStaged() {
        boolean any = false;
        for (Chunk c : current) {
            if (c.staged == null) {
                continue;
            }
            any = true;
            if (c.stagedRead != null) {
                try {
                    c.stagedRead.awaitCompletion();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted while staging a device feed's rows", e);
                } finally {
                    c.stagedRead.close();
                }
            }
            MtpHead.Region dst = head.region(c.rows);
            queue.enqueueWriteBuffer(dst.buffer, dst.offset * Float.BYTES,
                    (long) c.staged.length * Float.BYTES, c.staged);
        }
        if (any) {
            queue.finish();
        }
    }

    private int ubatch() {
        return head.ubatch();
    }

    private MtpHead.Rows take() {
        MtpHead.Rows r = pool.isEmpty() ? head.rows("flow.tap." + slotsMade++, width) : pool.remove(pool.size() - 1);
        refs.put(r, 1);
        return r;
    }

    /** Takes another reference of a chunk (a deferred catch-up row or the drafter's row state now reads it). */
    void retain(Chunk c) {
        refs.merge(c.rows, 1, Integer::sum);
    }

    /** Drops one reference of a chunk; its rows return to the pool with the last one. */
    void release(Chunk c) {
        Integer left = refs.merge(c.rows, -1, Integer::sum);
        if (left == null || left < 0) {
            throw new IllegalStateException(c.rows + " released more references than it holds");
        }
        if (left == 0) {
            pool.add(c.rows);
        }
    }

    /** Drops the drafter's reference of every chunk of a drained batch. */
    void releaseBatch(DeviceBatch b) {
        for (Chunk c : b.chunks) {
            release(c);
        }
    }

    private void releaseChunks(List<Chunk> chunks) {
        for (Chunk c : chunks) {
            release(c);
        }
    }

    /**
     * Removes and returns the collected batches, oldest first, without those a later call replaced: a call that
     * starts at or before an earlier call's start rewrote all of its positions (a recurrent target restored its
     * state and replayed the accepted tokens). The dropped batches' rows return to the pool.
     */
    List<DeviceBatch> drain() {
        if (pending.isEmpty()) {
            return Collections.emptyList();
        }
        List<DeviceBatch> out = new ArrayList<>(pending.size());
        int earliestLater = Integer.MAX_VALUE;
        for (int i = pending.size() - 1; i >= 0; i--) {
            DeviceBatch b = pending.get(i);
            if (b.pos0 < earliestLater) {
                out.add(b);
            } else {
                releaseBatch(b);
            }
            earliestLater = Math.min(earliestLater, b.pos0);
        }
        pending.clear();
        Collections.reverse(out);
        return out;
    }

    /** Drops every collected batch and any partial call; their rows return to the pool. */
    void clear() {
        for (DeviceBatch b : pending) {
            releaseBatch(b);
        }
        pending.clear();
        releaseChunks(current);
        current = new ArrayList<>();
        currentRows = 0;
    }

    @Override
    public String toString() {
        return "TargetDeviceFeed{point=" + point + ", width=" + width + ", pending=" + pending.size()
                + ", slots=" + slotsMade + "}";
    }

    /** The drafter's device-side row state across calls: the pending row, the bridge row, the last call's rows. */
    final class DeviceRows {

        private final MtpHead.Rows pendingRow = head.rows("flow.tap.pending", width);
        private final MtpHead.Rows bridgeRow = head.rows("flow.tap.bridge", width);
        private List<Chunk> last = Collections.emptyList();
        private int lastPos0 = -1;
        private int lastRows;
        private int pendingPos = -1;
        private int bridgePos = -1;

        private DeviceRows() {
            // the slots never return to the pool, and start at zeros like the host path's pending row
            refs.put(pendingRow, 1);
            refs.put(bridgeRow, 1);
            zero(pendingRow);
            zero(bridgeRow);
        }

        /** The rows the pending row lives in (its row 0). */
        MtpHead.Rows pending() {
            return pendingRow;
        }

        /** Position of the pending row; -1 when there is none. */
        int pendingPos() {
            return pendingPos;
        }

        boolean hasPending() {
            return pendingPos >= 0;
        }

        /** The device row of position {@code pos}, or null when none is kept: the last call's rows, then pending. */
        Located at(int pos) {
            if (pos < 0) {
                return null;
            }
            if (pos >= lastPos0 && pos < lastPos0 + lastRows) {
                return locate(last, pos - lastPos0);
            }
            if (pos == pendingPos) {
                return new Located(pendingRow, 0);
            }
            if (pos == bridgePos) {
                return new Located(bridgeRow, 0);
            }
            return null;
        }

        /**
         * Makes a call's rows the last call's rows: the row before {@code pos0} moves to the bridge row, the last
         * row becomes the pending row.
         */
        void record(List<Chunk> chunks, int pos0, int nRows) {
            Located before = at(pos0 - 1);
            if (before != null) {
                head.copyRows(before.rows, before.row, bridgeRow, 0, 1);
                bridgePos = pos0 - 1;
            }
            retainAll(chunks);
            releaseChunks(last);
            last = List.copyOf(chunks);
            lastPos0 = pos0;
            lastRows = nRows;
            copyRow(chunks, nRows - 1, pendingRow, 0);
            pendingPos = pos0 + nRows - 1;
        }

        /** Makes row {@code min(n, rows - 1)} of the last call pending; nothing when no call was recorded. */
        void accept(int n) {
            if (last.isEmpty()) {
                return;
            }
            int i = Math.min(n, lastRows - 1);
            copyRow(last, i, pendingRow, 0);
            pendingPos = lastPos0 + i;
        }

        /** Drops the last call's rows and the positions; the pending and bridge rows go back to zeros. */
        void clear() {
            releaseChunks(last);
            last = Collections.emptyList();
            lastPos0 = -1;
            lastRows = 0;
            pendingPos = -1;
            bridgePos = -1;
            zero(pendingRow);
            zero(bridgeRow);
        }

        private void zero(MtpHead.Rows row) {
            head.write(row, new float[width], 1);
        }

        private void retainAll(List<Chunk> chunks) {
            for (Chunk c : chunks) {
                retain(c);
            }
        }

        /** The device row {@code globalRow} of a call's chunks. */
        private Located locate(List<Chunk> chunks, int globalRow) {
            int base = 0;
            for (Chunk c : chunks) {
                if (globalRow < base + c.nRows) {
                    return new Located(c.rows, globalRow - base);
                }
                base += c.nRows;
            }
            throw new IllegalStateException("row " + globalRow + " outside the call's " + base + " rows");
        }

        private void copyRow(List<Chunk> chunks, int globalRow, MtpHead.Rows dst, int dstRow) {
            Located at = locate(chunks, globalRow);
            head.copyRows(at.rows, at.row, dst, dstRow, 1);
        }
    }
}

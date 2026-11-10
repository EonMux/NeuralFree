package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

import java.util.Arrays;

/**
 * The per-sequence {@code kv_pos} buffers of one KV group ({@code GpuStepInputs.sequenceKvPos}): buffer {@code seq}
 * holds the mirror's position at each cell of sequence {@code seq}'s slots and -1 at every other row of the pool.
 * The buffers persist across steps, and {@link #update} patches only the rows whose value changed since the device
 * last received them: the cells the step appended, the cells whose position a removal, trim or shift changed, and
 * the cells that left the sequence (back to -1). The host keeps each sequence's last uploaded cells and values, so
 * the patch costs the sequence's slots on the host and its changed rows on the device, and a step that changed
 * nothing writes nothing. The patch runs inside the trunk, which therefore never enters a decode replay
 * ({@code GpuStepInputs.pagedTrunkWrites}). Not thread-safe.
 */
final class SequenceKvPos {

    private final GpuBuffer[] buffers;
    private final int rows;
    /** Per sequence: the cells the device buffer holds a position at, and those positions. */
    private final int[][] heldCells;
    private final int[][] heldValues;
    private final int[] heldCount;
    /** Per row: the update that last named it a cell of the sequence, and the position it takes. */
    private final int[] stamp;
    private final int[] value;
    /** Per row: the update that found the device holding a position there. */
    private final int[] heldStamp;
    private final int[] cells;
    private final KvPosDirtyRows changed;
    private final int[] runs = new int[2 * KvPosDirtyRows.MAX_RUNS];
    private final int[] scratch;
    private int generation;

    /** Buffers of {@code rows} ints each, whose device contents the caller filled with -1. */
    SequenceKvPos(GpuBuffer[] buffers, int rows, int maxCells) {
        this.buffers = buffers.clone();
        this.rows = rows;
        int n = buffers.length;
        this.heldCells = new int[n][0];
        this.heldValues = new int[n][0];
        this.heldCount = new int[n];
        this.stamp = new int[rows];
        this.value = new int[rows];
        this.heldStamp = new int[rows];
        this.cells = new int[maxCells];
        this.changed = new KvPosDirtyRows(rows);
        this.scratch = new int[rows];
    }

    GpuBuffer buffer(int seq) {
        return buffers[seq];
    }

    /**
     * Brings sequence {@code seq}'s buffer to its cells in {@code step} (group {@code group}) at the positions of
     * {@code mirror}, writing the changed rows alone.
     *
     * @return the rows written
     */
    int update(PagedStep step, int group, int seq, GroupPositions mirror, DevicePageTables.IntWrites writes) {
        if (++generation == 0) {
            Arrays.fill(stamp, 0);
            Arrays.fill(heldStamp, 0);
            generation = 1;
        }
        int slots = step.seqCells(group, seq, cells);
        for (int j = 0; j < slots; j++) {
            int cell = cells[j];
            stamp[cell] = generation;
            value[cell] = mirror.position(cell);
        }
        int[] oldCells = heldCells[seq];
        int[] oldValues = heldValues[seq];
        for (int i = 0; i < heldCount[seq]; i++) {
            int cell = oldCells[i];
            heldStamp[cell] = generation;
            if (desired(cell) != oldValues[i]) {
                changed.mark(cell);
            }
        }
        for (int j = 0; j < slots; j++) {
            int cell = cells[j];
            if (heldStamp[cell] != generation && value[cell] != -1) {
                changed.mark(cell);
            }
        }
        int written = writeChanged(seq, writes);
        hold(seq, slots);
        return written;
    }

    /** The value row {@code row} takes after the current update: the sequence's position there, or -1. */
    private int desired(int row) {
        return stamp[row] == generation ? value[row] : -1;
    }

    /** Writes the marked rows' new values (the sequence's position, or -1) in the runs the marks form. */
    private int writeChanged(int seq, DevicePageTables.IntWrites writes) {
        int n = changed.runs(runs);
        int written = 0;
        for (int r = 0; r < n; r++) {
            int from = runs[2 * r];
            int to = runs[2 * r + 1];
            for (int row = from; row < to; row++) {
                scratch[row - from] = desired(row);
            }
            writes.write(buffers[seq], from, scratch, 0, to - from);
            written += to - from;
        }
        changed.clearAll();
        return written;
    }

    /** Records the update's cells and values as what the device buffer of {@code seq} now holds. */
    private void hold(int seq, int slots) {
        if (heldCells[seq].length < slots) {
            int cap = Math.max(slots, heldCells[seq].length * 2);
            heldCells[seq] = new int[cap];
            heldValues[seq] = new int[cap];
        }
        int n = 0;
        for (int j = 0; j < slots; j++) {
            int cell = cells[j];
            if (value[cell] != -1) {
                heldCells[seq][n] = cell;
                heldValues[seq][n] = value[cell];
                n++;
            }
        }
        heldCount[seq] = n;
    }

    /** The rows of each buffer. */
    int rows() {
        return rows;
    }

    /** Releases the buffers. */
    void close() {
        for (GpuBuffer b : buffers) {
            b.close();
        }
    }
}

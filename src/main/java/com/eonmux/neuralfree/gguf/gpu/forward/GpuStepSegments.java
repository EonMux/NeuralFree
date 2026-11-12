package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.BatchSegments;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * The sequence runs of the running packed memoryless step, read by the attention mixers of the step: the driver
 * sets them around the step ({@link #begin}, {@link #end}) and every softmax attention launches once per run over the
 * run's own rows. The flash kernels read a launch's query positions launch-local ({@code q_pos[t]} for its row
 * {@code t}), so a run's launch binds {@link #queryPositions()}, the positions {@code 0 .. ubatch - 1}: a packed run
 * numbers its rows from 0 ({@link BatchSegments}), and its keys keep their own positions in the step's position
 * buffer.
 *
 * <p>Created with the forward context and shared by every context derived from it; the driver that runs the steps
 * closes it. Not thread-safe: one forward pass at a time.</p>
 */
public final class GpuStepSegments implements AutoCloseable {

    private BatchSegments current;
    private GpuBuffer queryPositions;
    private int queryRows;

    /** The running step's runs, or null outside a packed step. */
    public BatchSegments current() {
        return current;
    }

    /**
     * The device positions {@code 0 .. rows - 1} a run's launch reads as its query positions.
     *
     * @throws IllegalStateException outside a packed step
     */
    public GpuBuffer queryPositions() {
        if (current == null) {
            throw new IllegalStateException("no packed memoryless step is running");
        }
        return queryPositions;
    }

    /**
     * Starts a packed step over {@code segments}: uploads the run-local query positions once (for {@code ubatch}
     * rows) and publishes the runs to the step's mixers.
     *
     * @throws IllegalStateException    when a packed step is already running
     * @throws IllegalArgumentException when a run is longer than {@code ubatch}
     */
    public void begin(BatchSegments segments, int ubatch, ComputeBackend backend, ComputeQueue queue) {
        Objects.requireNonNull(segments, "segments");
        if (current != null) {
            throw new IllegalStateException("a packed memoryless step is already running");
        }
        if (segments.maxLength() > ubatch) {
            throw new IllegalArgumentException("a run of " + segments.maxLength() + " rows exceeds the ubatch "
                    + ubatch);
        }
        if (queryPositions == null || queryRows < ubatch) {
            upload(ubatch, backend, queue);
        }
        current = segments;
    }

    /** Ends the running packed step; a no-op outside one. */
    public void end() {
        current = null;
    }

    private void upload(int rows, ComputeBackend backend, ComputeQueue queue) {
        GpuBuffer fresh = backend.allocateBuffer((long) rows * Integer.BYTES, true);
        try {
            ByteBuffer staging = ByteBuffer.allocateDirect(rows * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int t = 0; t < rows; t++) {
                staging.putInt(t * Integer.BYTES, t);
            }
            GpuEvent e = queue.enqueueWriteBuffer(fresh, 0, (long) rows * Integer.BYTES, staging);
            if (e != null) {
                e.close();
            }
            queue.finish();
        } catch (RuntimeException ex) {
            fresh.close();
            throw ex;
        }
        if (queryPositions != null) {
            queryPositions.close();
        }
        queryPositions = fresh;
        queryRows = rows;
    }

    /** Releases the query positions. Idempotent. */
    @Override
    public void close() {
        current = null;
        if (queryPositions != null) {
            queryPositions.close();
            queryPositions = null;
            queryRows = 0;
        }
    }
}

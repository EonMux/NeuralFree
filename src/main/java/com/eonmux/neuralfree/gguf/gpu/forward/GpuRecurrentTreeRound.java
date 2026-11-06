package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.engine.RecurrentTreeRound;

import java.util.Arrays;

/**
 * The GPU backend's {@link RecurrentTreeRound}, the twin of {@code CpuRecurrentTreeRound}: three device buffers —
 * per verify row one conv-state and one scan-state stack slot (sized for the widest recurrent layer, every layer
 * indexing its own stride), and per recurrent layer one captured mixer-input row per verify row — plus the replay
 * output row. The tree step seeds a node's stack slot from its parent's slot with a device copy on the step's
 * queue, so the committed state (row 0 of the layer's recurrent memory) is never written during the verify, and
 * the stateful dispatches re-point their state arguments at the stack slots for the step's duration. The step's
 * rows ascend, and a node's parent always precedes it, so every seed copies a slot the step already wrote. The
 * replay runs the layers' decode paths from the captured input rows; the queue drains before the round frees its
 * buffers.
 */
public final class GpuRecurrentTreeRound implements RecurrentTreeRound {

    private final GpuForwardBackend owner;
    private final GpuOps ops;
    private final ComputeQueue queue;
    private final int rows;
    private final int realRows;
    private final int maxWidth;
    private final GpuBuffer convStack;
    private final GpuBuffer scanStack;
    private final GpuBuffer capture;
    private final GpuBuffer replayOut;
    private final ScratchSlot replayOutSlot;
    private final int[] layerIl;
    private final int[] layerCapBase;
    private final int[] layerWidth;
    private long[] mask;
    private boolean committed;
    private boolean closed;

    GpuRecurrentTreeRound(GpuForwardBackend owner, GpuOps ops, int rows, int realRows, int maxWidth,
                          long maxConvFloats, long maxScanFloats, long captureFloats, int[] layerIl,
                          int[] layerCapBase, int[] layerWidth) {
        ComputeBackend backend = owner.context().backend();
        this.owner = owner;
        this.ops = ops;
        this.queue = ops.queue();
        this.rows = rows;
        this.realRows = realRows;
        this.maxWidth = maxWidth;
        this.layerIl = layerIl;
        this.layerCapBase = layerCapBase;
        this.layerWidth = layerWidth;
        this.convStack = backend.allocateBuffer(Math.multiplyExact(rows, maxConvFloats) * Float.BYTES);
        this.scanStack = backend.allocateBuffer(Math.multiplyExact(rows, maxScanFloats) * Float.BYTES);
        this.capture = backend.allocateBuffer(captureFloats * Float.BYTES);
        this.replayOut = backend.allocateBuffer((long) maxWidth * Float.BYTES);
        this.replayOutSlot = ScratchSlot.of("tree.replayOut", replayOut, 0, maxWidth);
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int realRows() {
        return realRows;
    }

    @Override
    public long[] mask() {
        if (mask == null) {
            throw new IllegalStateException("the round runs no verify step");
        }
        return mask;
    }

    /** The queue the tree step's seed copies and the replay's kernels ride. */
    public ComputeQueue queue() {
        return queue;
    }

    /** The conv-state stack buffer; row {@code i}'s slot sits at {@code i * slotFloats} floats of the layer's stride. */
    public GpuBuffer convStack() {
        return convStack;
    }

    /** The scan-state stack buffer; row {@code i}'s slot sits at {@code i * slotFloats} floats of the layer's stride. */
    public GpuBuffer scanStack() {
        return scanStack;
    }

    /**
     * Installs the verify step's per-row ancestor bitmask, padding rows zeroed, and checks the shape the per-row
     * stack relies on (see {@code CpuRecurrentTreeRound#stepTreeMask}).
     *
     * @throws IllegalArgumentException when the mask does not shape a verify step
     */
    public void stepTreeMask(long[] raw) {
        if (closed) {
            throw new IllegalStateException("the round is closed");
        }
        if (mask != null) {
            throw new IllegalStateException("the round runs one verify step");
        }
        if (raw == null || raw.length != rows) {
            throw new IllegalArgumentException("the tree step's mask holds "
                    + (raw == null ? 0 : raw.length) + " rows, the round runs " + rows);
        }
        long[] normalized = Arrays.copyOf(raw, rows);
        if (normalized[0] != 1L) {
            throw new IllegalArgumentException("the tree step's row 0 is not the root (mask " + normalized[0] + ")");
        }
        for (int row = 1; row < rows; row++) {
            if (row >= realRows) {
                if (normalized[row] != -1L) {
                    throw new IllegalArgumentException("the tree step's padding row " + row + " carries mask "
                            + normalized[row] + ", the padded verify rows hold -1");
                }
                normalized[row] = 0;
                continue;
            }
            long bits = normalized[row];
            if (bits == 0 || (bits & (1L << row)) == 0) {
                throw new IllegalArgumentException("the tree step's row " + row + " does not name itself (mask "
                        + bits + ")");
            }
            long ancestors = bits ^ (1L << row);
            if (ancestors == 0) {
                throw new IllegalArgumentException("the tree step's row " + row + " names no ancestor");
            }
            int parent = 63 - Long.numberOfLeadingZeros(ancestors);
            int depth = RecurrentTreeRound.depth(bits);
            if (parent >= row || RecurrentTreeRound.depth(normalized[parent]) != depth - 1) {
                throw new IllegalArgumentException("the tree step's row " + row + " of depth " + depth
                        + " does not descend from row " + parent + " at depth " + (depth - 1));
            }
        }
        this.mask = normalized;
    }

    /** Copies the step's first {@code n} input rows of layer {@code il} into the layer's capture slice. */
    public void captureInput(int il, ScratchSlot in, int n) {
        int slice = slice(il);
        int width = layerWidth[slice];
        in.require((long) n * width);
        ops.copy2d(in.buffer(), in.offset(), width, capture, layerCapBase[slice], width, n, width);
    }

    /** The captured input row of layer {@code il} and verify row {@code r}, a view the layer's decode reads. */
    public ScratchSlot replayInput(int il, int r) {
        int slice = slice(il);
        return ScratchSlot.of("tree.capture." + il, capture,
                (long) layerCapBase[slice] + (long) r * layerWidth[slice], layerWidth[slice]);
    }

    /** The output row the replay writes (nobody consumes it: the replay advances state only). */
    public ScratchSlot replayOutput() {
        return replayOutSlot;
    }

    private int slice(int il) {
        if (mask == null || committed || closed) {
            throw new IllegalStateException("the round is " + (closed ? "closed" : mask == null
                    ? "not in a verify step" : "committed"));
        }
        for (int i = 0; i < layerIl.length; i++) {
            if (layerIl[i] == il) {
                return i;
            }
        }
        throw new IllegalArgumentException("layer " + il + " keeps no recurrent tree state");
    }

    @Override
    public void commit(int[] replayRows) {
        if (mask == null) {
            throw new IllegalStateException("the round runs no verify step");
        }
        if (committed) {
            throw new IllegalStateException("the round already committed");
        }
        if (replayRows == null || replayRows.length == 0 || replayRows[0] != 0) {
            throw new IllegalArgumentException("the replay starts at the pending row 0, got "
                    + Arrays.toString(replayRows));
        }
        for (int r : replayRows) {
            if (r < 0 || r >= realRows) {
                throw new IllegalArgumentException("the replay row " + r + " is outside the step's "
                        + realRows + " computed rows");
            }
        }
        owner.replayRecurrentTree(this, replayRows);
        committed = true;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        // the replay's copies and kernels read the capture and write the state through the queue: drain it before
        // the buffers leave
        queue.finish();
        convStack.close();
        scanStack.close();
        capture.close();
        replayOut.close();
        owner.closeTreeRound(this);
    }
}

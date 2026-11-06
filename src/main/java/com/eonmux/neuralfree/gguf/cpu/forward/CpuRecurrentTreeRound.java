package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.engine.RecurrentTreeRound;

import java.util.Arrays;

/**
 * The heap backend's {@link RecurrentTreeRound}: per verify row one conv-state and one scan-state stack slot
 * (sized for the widest recurrent layer, every layer indexing its own stride), and per recurrent layer one
 * captured mixer-input row per verify row — the commit inputs of the accepted-path replay. The tree step seeds a
 * node's stack slot from its parent's slot, so the committed state (slot 0 of the layer's recurrent memory) is
 * never written during the verify; {@link #commit} replays the accepted rows through the layers' decode paths
 * against the committed state. The step's rows ascend, and a node's parent always precedes it, so every seed
 * copies a slot the step already wrote.
 */
public final class CpuRecurrentTreeRound implements RecurrentTreeRound {

    private final CpuForwardBackend owner;
    private final int rows;
    private final int realRows;
    private final int maxWidth;
    private final int maxConv;
    private final int maxScan;
    private final int[] layerIl;
    private final int[] layerCapBase;
    private final int[] layerWidth;
    private final float[] convStack;
    private final float[] scanStack;
    private final float[] capture;
    private final float[] replayIn;
    private final float[] replayOut;
    private long[] mask;
    private boolean committed;
    private boolean closed;

    CpuRecurrentTreeRound(CpuForwardBackend owner, int rows, int realRows, int maxWidth, int maxConv, int maxScan,
                          long captureFloats, int[] layerIl, int[] layerCapBase, int[] layerWidth) {
        this.owner = owner;
        this.rows = rows;
        this.realRows = realRows;
        this.maxWidth = maxWidth;
        this.maxConv = maxConv;
        this.maxScan = maxScan;
        this.layerIl = layerIl;
        this.layerCapBase = layerCapBase;
        this.layerWidth = layerWidth;
        this.convStack = new float[Math.multiplyExact(rows, maxConv)];
        this.scanStack = new float[Math.multiplyExact(rows, maxScan)];
        this.capture = new float[Math.toIntExact(captureFloats)];
        this.replayIn = new float[maxWidth];
        this.replayOut = new float[maxWidth];
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

    /** The conv-state stack: row {@code i}'s slot at {@code i * slotFloats} of the layer's own stride. */
    public float[] convStack() {
        return convStack;
    }

    /** The scan-state stack: row {@code i}'s slot at {@code i * slotFloats} of the layer's own stride. */
    public float[] scanStack() {
        return scanStack;
    }

    /**
     * Installs the verify step's per-row ancestor bitmask, padding rows zeroed, and checks the shape the per-row
     * stack relies on: row 0 is the root, every node names itself and its ancestors, its immediate parent (the
     * largest row its mask names) precedes it at exactly one depth less — so the step's ascending walk seeds a
     * node's slot from a parent slot that is already written.
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
    public void captureInput(int il, float[] in, int n) {
        int slice = slice(il);
        int width = layerWidth[slice];
        long need = (long) n * width;
        if (in.length < need) {
            throw new IllegalArgumentException("the tree step's input holds " + in.length + " floats, layer " + il
                    + " captures " + need);
        }
        System.arraycopy(in, 0, capture, layerCapBase[slice], (int) need);
    }

    /**
     * The captured input row of layer {@code il} and verify row {@code r}, copied to a dense row the layer's
     * decode step reads from element 0.
     */
    public float[] replayInput(int il, int r) {
        int slice = slice(il);
        System.arraycopy(capture, layerCapBase[slice] + r * layerWidth[slice], replayIn, 0, layerWidth[slice]);
        return replayIn;
    }

    /** The output row the replay writes (nobody consumes it: the replay advances state only). */
    public float[] replayOutput() {
        return replayOut;
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
        owner.closeTreeRound(this);
    }
}

package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.util.Arrays;
import java.util.function.IntUnaryOperator;

/**
 * The drafter's own KV cache, one sequence. Each layer stores K (and V, unless V is read from the K row as in the
 * DSV4 backbone) per position: the rows injected from the target features at the target's positions, then the
 * rows of the noise block while it is drafted. A full-attention layer holds one row per position; a sliding-window
 * layer holds a ring of {@link #ringRows} rows, row {@code pos % rows}, enough for every query of a block and for
 * the rollback of one verified block. Each row records the position it holds; {@link #trim} forgets every position
 * from a bound on, as llama.cpp's {@code seq_rm(seq, pos, -1)} on the draft context (server-context.cpp spec
 * flow). A quantized cache carries llama.cpp's Hadamard rotation ({@link RotationPlan}, llama-kv-cache.cpp:324-360):
 * the stacks rotate Q and the stored K rows by {@link #rotationK()}, the stored V rows by {@link #rotationV()}, and
 * rotate the attention output back (llama-graph.cpp:3498-3560, and :3593-3631 for a cache whose V is its K row).
 * Not thread-safe.
 */
final class DflashKvCache {

    private final CpuKvLayer[] layers;
    private final int[][] rowPos;
    private final boolean[] ring;
    private final KvWindow[] windows;
    private final int[] rowEnd;
    private final int capacity;
    private final HadamardSpec rotK;
    private final HadamardSpec rotV;

    /**
     * @param capacity   positions a full-attention layer can hold (the target context plus one block)
     * @param blockSize  noise block length
     * @param nSwa       window of the sliding-window layers
     * @param swa        per layer: sliding-window attention
     * @param causal     block-internal attention is causal
     * @param keyOnly    V is read from the K row (one head of {@code headDim}); otherwise separate K and V
     * @param nHeadKv    per layer KV heads
     * @param headDim    K and V head size
     * @param dtype      storage type of the rows
     * @param rotate     apply the Hadamard rotation of quantized rows ({@code neuralfree.kv.rotation})
     */
    DflashKvCache(int capacity, int blockSize, int nSwa, boolean[] swa, boolean causal, boolean keyOnly,
                  int[] nHeadKv, int headDim, KvDType dtype, boolean rotate) {
        if (capacity <= 0 || blockSize <= 0) {
            throw new IllegalArgumentException("dflash KV cache needs positive capacity and block, got " + capacity
                    + "/" + blockSize);
        }
        final int nLayer = swa.length;
        this.capacity = capacity;
        RotationPlan plan = RotationPlan.of(dtype, dtype, headDim, keyOnly ? 0 : headDim, headDim, headDim, rotate,
                false);
        this.rotK = plan.rotateK() ? HadamardSpec.rotation(plan.nRotK()) : null;
        this.rotV = plan.rotateV() ? HadamardSpec.rotation(plan.nRotV()) : null;
        this.layers = new CpuKvLayer[nLayer];
        this.rowPos = new int[nLayer][];
        this.ring = new boolean[nLayer];
        this.windows = new KvWindow[nLayer];
        this.rowEnd = new int[nLayer];
        for (int il = 0; il < nLayer; il++) {
            final boolean windowed = swa[il] && nSwa > 0;
            windows[il] = KvWindow.of(windowed ? SwaType.STANDARD : SwaType.NONE, windowed ? nSwa : 0, causal);
            final int rows = layerRows(capacity, blockSize, nSwa, swa[il]);
            ring[il] = windowed && rows < capacity;
            KvLayout layout = layerLayout(rows, keyOnly, nHeadKv[il], headDim, dtype, windows[il]);
            layers[il] = CpuKvLayer.allocate(il, layout, -1, keyOnly ? 1 : nHeadKv[il], headDim, headDim);
            rowPos[il] = new int[rows];
            Arrays.fill(rowPos[il], -1);
        }
    }

    /**
     * Bytes a cache built with these arguments holds: every layer's K and V rows ({@link KvLayout#totalBytes()})
     * and its per-row position record.
     *
     * @throws IllegalArgumentException when the capacity or block is not positive
     */
    static long bytes(int capacity, int blockSize, int nSwa, boolean[] swa, boolean keyOnly, int[] nHeadKv,
                      int headDim, KvDType dtype) {
        if (capacity <= 0 || blockSize <= 0) {
            throw new IllegalArgumentException("dflash KV cache needs positive capacity and block, got " + capacity
                    + "/" + blockSize);
        }
        long total = 0;
        for (int il = 0; il < swa.length; il++) {
            final int rows = layerRows(capacity, blockSize, nSwa, swa[il]);
            KvLayout layout = layerLayout(rows, keyOnly, nHeadKv[il], headDim, dtype, KvWindow.FULL_CAUSAL);
            total = Math.addExact(total, Math.addExact(layout.totalBytes(), (long) rows * Integer.BYTES));
        }
        return total;
    }

    private static int layerRows(int capacity, int blockSize, int nSwa, boolean swa) {
        return swa && nSwa > 0 ? ringRows(capacity, nSwa, blockSize) : capacity;
    }

    /** Layout of one layer: a single key-only head, or {@code nHeadKv} heads of separate K and V. */
    private static KvLayout layerLayout(int rows, boolean keyOnly, int nHeadKv, int headDim, KvDType dtype,
                                        KvWindow window) {
        if (keyOnly) {
            return KvLayout.latent(rows, false, headDim, headDim, dtype, window);
        }
        final int width = nHeadKv * headDim;
        return KvLayout.of(rows, false, width, width, dtype, dtype, window);
    }

    /**
     * Ring size of a sliding-window layer: the {@code nSwa + blockSize - 1} positions a block's queries see, plus
     * one more block so that the rows a rolled-back verification overwrote lie outside every later window; capped
     * at the full capacity.
     */
    static int ringRows(int capacity, int nSwa, int blockSize) {
        long need = (long) nSwa + 2L * blockSize;
        long padded = (need + KvLayout.RING_PAD - 1) / KvLayout.RING_PAD * KvLayout.RING_PAD;
        return (int) Math.min(capacity, padded);
    }

    int nLayer() {
        return layers.length;
    }

    /** Positions a full-attention layer can hold. */
    int capacity() {
        return capacity;
    }

    KvAccessor layer(int il) {
        return layers[il];
    }

    /** Rotation of Q and the stored K rows, or null when K is stored unrotated. */
    HadamardSpec rotationK() {
        return rotK;
    }

    /** Rotation of the stored V rows and back-rotation of the attention output, or null without one. */
    HadamardSpec rotationV() {
        return rotV;
    }

    /** Visibility rule of layer {@code il}: its window and the block's causal flag. */
    KvWindow window(int il) {
        return windows[il];
    }

    /** Rows of layer {@code il}. */
    int rows(int il) {
        return rowPos[il].length;
    }

    /**
     * End of the rows of layer {@code il} that may hold a position: every row of a ring, else one past the
     * highest stored position (rows from there on are empty).
     */
    int rowEnd(int il) {
        return ring[il] ? rowPos[il].length : rowEnd[il];
    }

    /** Position held by each row of layer {@code il}, -1 for an empty row. */
    IntUnaryOperator rowPositions(int il) {
        final int[] p = rowPos[il];
        return r -> p[r];
    }

    /**
     * Stores the K row (and the V row unless V aliases K) of position {@code pos} in layer {@code il}.
     *
     * @throws IllegalStateException when a full-attention layer has no row for {@code pos}
     */
    void write(int il, int pos, float[] k, int kOff, float[] v, int vOff) {
        final int row = rowOf(il, pos);
        layers[il].writeK(row, k, kOff);
        if (v != null) {
            layers[il].writeV(row, v, vOff);
        }
        rowPos[il][row] = pos;
        rowEnd[il] = Math.max(rowEnd[il], row + 1);
    }

    /** True when positions {@code [0, pos + n)} fit the full-attention layers. */
    boolean fits(int pos, int n) {
        return pos >= 0 && (long) pos + n <= capacity;
    }

    /** Forgets every stored position {@code >= pos}. */
    void trim(int pos) {
        for (int il = 0; il < rowPos.length; il++) {
            final int[] p = rowPos[il];
            final int end = rowEnd(il);
            // a full layer stores position r in row r, so only rows from pos on can hold a trimmed position
            for (int r = ring[il] ? 0 : Math.max(0, pos); r < end; r++) {
                if (p[r] >= pos) {
                    p[r] = -1;
                }
            }
            if (!ring[il]) {
                rowEnd[il] = Math.min(rowEnd[il], Math.max(0, pos));
            }
        }
    }

    /** Forgets every position. */
    void clear() {
        trim(0);
    }

    private int rowOf(int il, int pos) {
        if (pos < 0) {
            throw new IllegalArgumentException("KV position " + pos + " is negative");
        }
        if (ring[il]) {
            return pos % rowPos[il].length;
        }
        if (pos >= rowPos[il].length) {
            throw new IllegalStateException("dflash draft KV of layer " + il + " holds " + rowPos[il].length
                    + " positions, position " + pos + " does not fit");
        }
        return pos;
    }
}

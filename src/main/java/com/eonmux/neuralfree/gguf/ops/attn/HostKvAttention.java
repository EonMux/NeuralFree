package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.KvInt8Rows;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * Contiguous CPU attention over a host KV cache and projected Q, K and V rows. The caller applies the layer's
 * normalization, positional encoding and cache rotations before this stage, and updates the cache's position
 * metadata before running it. A shared-cache layer supplies neither K nor V. An int8 cache with the fused path
 * selected attends through {@link CpuInt8Attention}; every other cache through {@link CpuAttentionCore}.
 */
final class HostKvAttention {
    private final KvAccessor kv;
    private final IntUnaryOperator rowPosition;
    private final int qDim;
    private final int kDim;
    private final int vDim;
    private final int outDim;
    private final int headDimK;
    private final int headDimV;
    private final float[] kScale;
    private final float[] vScale;
    private final KvInt8Rows int8;

    HostKvAttention(KvAccessor kv, IntUnaryOperator rowPosition, int qDim, int kDim, int vDim, int outDim,
                    int headDimK, int headDimV, float[] kScale, float[] vScale) {
        this(kv, rowPosition, qDim, kDim, vDim, outDim, headDimK, headDimV, kScale, vScale, null);
    }

    /**
     * @param int8 raw int8 view of the cache for the fused attention, or null to dequantize every row
     *             ({@link CpuAttentionCore})
     */
    HostKvAttention(KvAccessor kv, IntUnaryOperator rowPosition, int qDim, int kDim, int vDim, int outDim,
                    int headDimK, int headDimV, float[] kScale, float[] vScale, KvInt8Rows int8) {
        this.kv = Objects.requireNonNull(kv, "kv");
        this.rowPosition = Objects.requireNonNull(rowPosition, "rowPosition");
        this.qDim = qDim;
        this.kDim = kDim;
        this.vDim = vDim;
        this.outDim = outDim;
        this.headDimK = headDimK;
        this.headDimV = headDimV;
        this.kScale = kScale;
        this.vScale = vScale;
        this.int8 = int8;
    }

    /** Stores projected K/V when present and attends to the cache. K/V are scaled in place for scaled caches. */
    void run(CpuAttentionCore.Params params, int pos0, int n, int[] pos, float[] q, float[] k, float[] v,
             float[] out) {
        run(params, pos0, n, pos, pos, q, k, v, out, null);
    }

    /**
     * {@link #run} with the step's store and query positions separated (a tree verify's rows rope at their depth
     * positions but store at their row slots) and the step's tree mask: the fresh rows (the step's tree nodes at
     * the cache rows {@code [pos0, pos0 + n)}) are the last {@code tree.freshRows()} rows of the key range, and
     * each query row sees the fresh rows its bitmask names.
     */
    void run(CpuAttentionCore.Params params, int pos0, int n, int[] pos, int[] storeAt, float[] q, float[] k,
             float[] v, float[] out, CpuAttentionCore.TreeMask tree) {
        if (k != null) {
            store(n, storeAt, k, Objects.requireNonNull(v, "v"));
        } else if (v != null) {
            throw new IllegalArgumentException("V without K");
        }
        int keys = Math.min(kv.layout().rows(), pos0 + n);
        if (tree != null && keys < tree.freshRows()) {
            throw new IllegalArgumentException("a tree mask covers " + tree.freshRows() + " fresh rows, the cache"
                    + " holds " + keys + " key rows");
        }
        /* A tree verify's fresh-row visibility is the mask, which the fused int8 kernel takes no part in; the
         * dequantized core below serves the same cache. */
        if (int8 != null && tree == null && params.sinkTokens() == 0) {
            CpuInt8Attention.attendRows(params, n, q, 0, qDim, pos, 0, int8, rowPosition, 0, keys, out, 0, outDim);
            return;
        }
        if (tree == null) {
            CpuAttentionCore.attendRows(params, n, q, 0, qDim, pos, 0, kv, rowPosition, 0, keys, out, 0, outDim);
        } else {
            CpuAttentionCore.attendRows(params, n, q, 0, qDim, pos, 0, kv, rowPosition, 0, keys, out, 0, outDim,
                    tree);
        }
    }

    void store(int n, int[] pos, float[] k, float[] v) {
        KvLayout layout = kv.layout();
        for (int t = 0; t < n; t++) {
            int row = layout.rowOf(pos[t]);
            divideByScales(k, t * kDim, kDim, headDimK, kScale);
            divideByScales(v, t * vDim, vDim, headDimV, vScale);
            kv.writeK(row, k, t * kDim);
            kv.writeV(row, v, t * vDim);
        }
    }

    /** Divides a projected row by its per-head cache scales before storing. */
    static void divideByScales(float[] row, int rowOff, int width, int headDim, float[] scale) {
        if (scale == null) {
            return;
        }
        for (int h = 0; h < width / headDim; h++) {
            int base = rowOff + h * headDim;
            for (int d = 0; d < headDim; d++) {
                row[base + d] /= scale[h];
            }
        }
    }
}

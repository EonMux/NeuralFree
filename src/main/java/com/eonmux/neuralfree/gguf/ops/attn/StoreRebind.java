package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

/**
 * One cache store's rebind points: the prepared step, the argument slots its {@code cell_idx} and
 * {@code dst} buffers sit in, and the buffers they hold when bound. The shadow store swaps the two buffers
 * for the rerun and restores them.
 */
final class StoreRebind {

    final TokenStep step;
    final int cellArg;
    final GpuBuffer cell;
    final int dstArg;
    final GpuBuffer dst;
    /** Whether the store writes the V row ({@code false} writes K). */
    final boolean vSide;

    StoreRebind(TokenStep step, int cellArg, GpuBuffer cell, int dstArg, GpuBuffer dst, boolean vSide) {
        this.step = step;
        this.cellArg = cellArg;
        this.cell = cell;
        this.dstArg = dstArg;
        this.dst = dst;
        this.vSide = vSide;
    }
}

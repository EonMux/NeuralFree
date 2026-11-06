package com.eonmux.neuralfree.gguf.mm;

/**
 * The blocks and norms of one tower of a CLIP dual encoder over the input rows the tower assembled: the host tower
 * ({@link CpuVitTower}) or the device one ({@link GpuDualTowers}). Package-private executor seam of
 * {@link ClipTextTower} and {@link ClipImageTower}.
 */
@FunctionalInterface
interface TowerBlocks {

    /** The output rows of {@code n} input rows {@code [n][n_embd]}; {@code rows} may be overwritten. */
    float[] forward(float[] rows, int n);
}

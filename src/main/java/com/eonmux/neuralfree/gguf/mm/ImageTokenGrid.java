package com.eonmux.neuralfree.gguf.mm;

/**
 * The token grid of one prepared tile, the position layout the decoder gives its image tokens (llama.cpp
 * {@code clip_n_output_tokens_x/y}, tools/mtmd/clip.cpp): the M-RoPE models lay the tokens on the merged patch
 * grid, every other projector on one row holding all of them.
 */
public enum ImageTokenGrid {

    /** One row of all the tile's tokens. */
    ROW,
    /** The merged patch grid, {@code width / (patch * m)} columns by {@code height / (patch * m)} rows (floored). */
    MERGED;

    /**
     * The token columns of a {@code width x height} tile counting {@code tokens} image tokens.
     *
     * @throws IllegalArgumentException when the tile is empty
     */
    public int columns(int width, int height, ImageTokenGeometry geometry, int tokens) {
        requireTile(width, height);
        return this == MERGED ? width / geometry.mergedPatchSize() : tokens;
    }

    /**
     * The token rows of a {@code width x height} tile.
     *
     * @throws IllegalArgumentException when the tile is empty
     */
    public int rows(int width, int height, ImageTokenGeometry geometry) {
        requireTile(width, height);
        return this == MERGED ? height / geometry.mergedPatchSize() : 1;
    }

    private static void requireTile(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("tile " + width + "x" + height + " is empty");
        }
    }
}

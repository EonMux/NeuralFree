package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The output of one {@link ImagePreprocessor} run, the {@code mtmd_image_preproc_out} plus the token
 * computation of llama.cpp (tools/mtmd/mtmd-image.cpp, mtmd.cpp): the overview tile (the downscaled image,
 * present for the llava-uhd families), the slice tiles, the slice grid they came from and the order the tiles
 * encode in, and the image token counts per tile and in total ({@code clip_n_output_tokens}). The overview's count
 * comes from {@link ImageProcessorParams#tokensFor(ImageSize, boolean)} with the overview flag set, which adds the
 * DeepSeek-OCR view separator. An anyres group (Granite Vision) encodes its overview and tiles as one image whose
 * tile grid is cropped to the {@link #anyresOriginal() original size}'s aspect ratio, so its token count is that
 * of the assembled image ({@link ImageProcessorParams#anyresTokens}).
 *
 * <p>Immutable; the tiles are the {@link ImageTile}s the preprocessor produced.</p>
 */
public final class PreparedImages {

    private final ImageProcessorParams params;
    private final ImageTile overview;
    private final List<ImageTile> tiles;
    private final int gridX;
    private final int gridY;
    private final boolean overviewFirst;
    private final ImageSize anyresOriginal;

    private PreparedImages(Builder b) {
        this.params = b.params;
        this.overview = b.overview;
        this.tiles = Collections.unmodifiableList(new ArrayList<>(b.tiles));
        this.gridX = b.gridX;
        this.gridY = b.gridY;
        this.overviewFirst = b.overviewFirst;
        this.anyresOriginal = b.anyresOriginal;
    }

    /** The overview tile, or null when the preprocessor produces none. */
    public ImageTile overview() { return overview; }

    /** Whether an overview tile is present. */
    public boolean hasOverview() { return overview != null; }

    /** The slice tiles, without the overview. */
    public List<ImageTile> tiles() { return tiles; }

    /** The slice grid width ({@code preproc_out.grid_x}); 0 when there are no slices. */
    public int gridX() { return gridX; }

    /** The slice grid height ({@code preproc_out.grid_y}); 0 when there are no slices. */
    public int gridY() { return gridY; }

    /** Whether the overview encodes before the slices; false when the slices come first. */
    public boolean overviewFirst() { return overviewFirst; }

    /** Whether the group is one anyres image: the overview and the tile grid encode and assemble together. */
    public boolean isAnyres() { return !anyresOriginal.isEmpty(); }

    /** The original image size an anyres group crops its tile grid to; {@link ImageSize#EMPTY} otherwise. */
    public ImageSize anyresOriginal() { return anyresOriginal; }

    /** Every tile in encode order: the overview first or last per {@link #overviewFirst}. */
    public List<ImageTile> encodedTiles() {
        return encodeOrder(overview, tiles, overviewFirst);
    }

    /** Whether the tile at {@code index} of {@link #encodedTiles()} is the overview. */
    public boolean isOverview(int index) {
        return overviewIndex(overview, tiles.size(), overviewFirst) == index;
    }

    /**
     * The image token count of the tile at {@code index} of {@link #encodedTiles()}
     * ({@code clip_n_output_tokens}); for an anyres group, the tile's own tokens before the grid is cropped.
     *
     * @throws IndexOutOfBoundsException when the index is outside the encoded tiles
     */
    public int tokensFor(int index) {
        List<ImageTile> all = encodedTiles();
        return params.tokensFor(all.get(index).size(), isOverview(index));
    }

    /**
     * The total image token count of the whole group ({@code mtmd_image_tokens::n_tokens}); an anyres group counts
     * the assembled image.
     */
    public int totalTokens() {
        if (isAnyres()) {
            return params.anyresTokens(overview.size(), gridX, gridY, anyresOriginal);
        }
        int total = 0;
        int n = encodedTiles().size();
        for (int i = 0; i < n; i++) {
            total += tokensFor(i);
        }
        return total;
    }

    /**
     * The token grid of the group ({@code mtmd_image_tokens::nx/ny}): the grid of the last encoded tile, the
     * position layout the decoder uses for M-RoPE models; one row of all tokens for an anyres group.
     */
    public ImageSize tokenGrid() {
        if (isAnyres()) {
            return new ImageSize(totalTokens(), 1);
        }
        List<ImageTile> all = encodedTiles();
        if (all.isEmpty()) {
            throw new IllegalStateException("no tiles were prepared");
        }
        return params.tokenGridFor(all.get(all.size() - 1).size());
    }

    /**
     * Every tile as the normalized floats the vision encoder consumes, in encode order: each entry is the
     * interleaved R, G, B pixels of the tile with {@code (v / 255 - mean) / std} per channel
     * ({@code clip_image_f32}).
     */
    public List<float[]> normalized() {
        List<ImageTile> all = encodedTiles();
        List<float[]> out = new ArrayList<>(all.size());
        for (ImageTile tile : all) {
            out.add(tile.normalized(params.mean(), params.std()));
        }
        return out;
    }

    /** Builder of an immutable {@link PreparedImages}. */
    public static final class Builder {

        private final ImageProcessorParams params;
        private ImageTile overview;
        private List<ImageTile> tiles = new ArrayList<>();
        private int gridX = 0;
        private int gridY = 0;
        private boolean overviewFirst = true;
        private ImageSize anyresOriginal = ImageSize.EMPTY;

        /** A new builder over the params that define the token computation. */
        public Builder(ImageProcessorParams params) {
            this.params = Objects.requireNonNull(params, "params");
            this.overviewFirst = params.overviewFirst();
        }

        /** Sets the overview tile. */
        public Builder overview(ImageTile overview) {
            this.overview = overview;
            return this;
        }

        /** Replaces the slice tiles. */
        public Builder tiles(List<? extends ImageTile> tiles) {
            this.tiles = tiles == null ? new ArrayList<>() : new ArrayList<>(tiles);
            return this;
        }

        /** Appends one slice tile. */
        public Builder tile(ImageTile tile) {
            this.tiles.add(tile);
            return this;
        }

        /** Sets the slice grid. */
        public Builder grid(int gridX, int gridY) {
            this.gridX = gridX;
            this.gridY = gridY;
            return this;
        }

        /** Sets whether the overview encodes before the slices; defaults to the params' ordering. */
        public Builder overviewFirst(boolean overviewFirst) {
            this.overviewFirst = overviewFirst;
            return this;
        }

        /**
         * Marks the group as one anyres image of an {@code original} image: the overview first, then the
         * {@link #grid} tiles in raster order, assembled with the grid cropped to the original's aspect ratio.
         */
        public Builder anyres(ImageSize original) {
            this.anyresOriginal = Objects.requireNonNull(original, "original");
            return this;
        }

        /** Builds the validated group.
         *
         * @throws IllegalArgumentException when no tile was prepared, an anyres group lacks its overview or its grid
         *                                  of tiles, or a token count does not fit an int
         */
        public PreparedImages build() {
            if (overview == null && tiles.isEmpty()) {
                throw new IllegalArgumentException("an image group needs an overview or at least one tile");
            }
            if (!anyresOriginal.isEmpty()) {
                if (overview == null || !overviewFirst || gridX <= 0 || gridY <= 0
                        || tiles.size() != gridX * gridY) {
                    throw new IllegalArgumentException("an anyres group needs the overview first and a "
                            + gridX + "x" + gridY + " grid of tiles, got " + tiles.size() + " tiles"
                            + (overview == null ? " and no overview" : ""));
                }
                params.anyresTokens(overview.size(), gridX, gridY, anyresOriginal);
                return new PreparedImages(this);
            }
            long total = 0;
            List<ImageTile> all = encodeOrder(overview, tiles, overviewFirst);
            int overviewAt = overviewIndex(overview, tiles.size(), overviewFirst);
            for (int i = 0; i < all.size(); i++) {
                total += params.tokensFor(all.get(i).size(), i == overviewAt);
                if (total > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("the image group needs more than " + Integer.MAX_VALUE
                            + " tokens");
                }
            }
            return new PreparedImages(this);
        }
    }

    /** The tiles in encode order: the overview first or last. */
    private static List<ImageTile> encodeOrder(ImageTile overview, List<ImageTile> tiles, boolean overviewFirst) {
        List<ImageTile> out = new ArrayList<>(tiles.size() + 1);
        if (overviewFirst && overview != null) {
            out.add(overview);
        }
        out.addAll(tiles);
        if (!overviewFirst && overview != null) {
            out.add(overview);
        }
        return out;
    }

    /** The overview's index in encode order, or -1 without an overview. */
    private static int overviewIndex(ImageTile overview, int tileCount, boolean overviewFirst) {
        if (overview == null) {
            return -1;
        }
        return overviewFirst ? 0 : tileCount;
    }
}

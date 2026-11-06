package com.eonmux.neuralfree.gguf.image;

/**
 * The Cohere2 vision preprocessor ({@code mtmd_image_preprocessor_cohere2v}, tools/mtmd/mtmd-image.cpp): the
 * tile grid is the whole-tile factorization of the tile count that needs the least downscale (or the least
 * upscale, when every grid must shrink), the refined image is exactly the grid in tiles, a single 1x1 grid
 * skips the slices, and the tiles encode before the overview.
 */
public final class Cohere2vPreprocessor extends LlavaUhdPreprocessor {

    /** Creates the preprocessor over the params. */
    public Cohere2vPreprocessor(ImageProcessorParams params) {
        super(params);
    }

    @Override
    protected SliceInstructions sliceInstructions(ImageSize original) {
        int tile = params().imageSize();
        int maxTiles = params().maxTiles();

        // pick the grid with the least upscale; if all grids need downscale, pick the one with the least
        // downscale; grids are visited by tile count, then by width, the same order the reference uses for ties
        double bestDown = -1.0;
        double bestUp = Double.MAX_VALUE;
        ImageSize gridDown = new ImageSize(1, 1);
        ImageSize gridUp = ImageSize.EMPTY;
        for (int n = 1; n <= maxTiles; n++) {
            for (int w = 1; w <= n; w++) {
                if (n % w != 0) {
                    continue;
                }
                ImageSize g = new ImageSize(w, n / w);
                double scale = Math.min((double) (g.width() * tile) / original.width(),
                        (double) (g.height() * tile) / original.height());
                if (scale < 1.0) {
                    if (scale > bestDown) {
                        bestDown = scale;
                        gridDown = g;
                    }
                } else if (scale < bestUp) {
                    bestUp = scale;
                    gridUp = g;
                }
            }
        }
        ImageSize grid = gridUp.isEmpty() ? gridDown : gridUp;

        SliceInstructions inst = new SliceInstructions();
        inst.overviewSize = new ImageSize(tile, tile);
        inst.refinedSize = new ImageSize(tile * grid.width(), tile * grid.height());
        inst.gridSize = grid;
        if (grid.width() * grid.height() > 1) {
            for (int y = 0; y < grid.height(); y++) {
                for (int x = 0; x < grid.width(); x++) {
                    inst.slices.add(new SliceRect(x * tile, y * tile, tile, tile));
                }
            }
        }
        return inst;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        SliceInstructions inst = sliceInstructions(image.size());
        SliceOutput sliced = sliceImage(image, inst);
        PreparedImages.Builder b = new PreparedImages.Builder(params())
                .overview(sliced.overview)
                .tiles(sliced.slices);
        // the output carries the grid only when slices exist (mtmd-image.cpp cohere2v preprocess)
        if (!sliced.slices.isEmpty()) {
            b.grid(inst.gridSize.width(), inst.gridSize.height());
        }
        return b.build();
    }
}

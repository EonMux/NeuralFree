package com.eonmux.neuralfree.gguf.image;

/**
 * The LFM2-VL and d1-omni preprocessor ({@code mtmd_image_preprocessor_lfm2}, tools/mtmd/mtmd-image.cpp; the
 * transformers {@code Lfm2VlImageProcessorFast}): the image resizes to its smart-resize size, the multiple of
 * {@code patch * merge} nearest to it within the pixel limits. When that size would exceed the pixel limit by more
 * than the tolerance, the image instead stretches to a grid of {@code tileSize} tiles, whose shape is the tile
 * count in {@code [minTiles, maxTiles]} closest to the aspect ratio, and the smart-resized image becomes the
 * thumbnail encoded after the tiles. A single image without tiles has no thumbnail.
 */
public final class Lfm2Preprocessor extends LlavaUhdPreprocessor {

    /** Creates the preprocessor over the params. */
    public Lfm2Preprocessor(ImageProcessorParams params) {
        super(params);
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        SliceInstructions inst = sliceInstructions(image.size());
        SliceOutput sliced = sliceImage(image, inst);
        if (!inst.slices.isEmpty()) {
            return withOverview(inst, sliced);
        }
        // one tile: the resized image encodes as the image itself, not as a thumbnail
        return new PreparedImages.Builder(params()).tile(sliced.overview).build();
    }

    @Override
    protected SliceInstructions sliceInstructions(ImageSize original) {
        SliceInstructions inst = new SliceInstructions();
        inst.overviewSize = smartResize(original);
        if (!shouldTile(original)) {
            return inst;
        }
        ImageSize grid = gridLayout(original);
        int tile = params().tileSize();
        inst.gridSize = grid;
        inst.refinedSize = new ImageSize(tile * grid.width(), tile * grid.height());
        for (int row = 0; row < grid.height(); row++) {
            for (int col = 0; col < grid.width(); col++) {
                inst.slices.add(new SliceRect(col * tile, row * tile, tile, tile));
            }
        }
        return inst;
    }

    /**
     * Whether the image tiles: the tile cap allows a grid and the area of its sides rounded to the merged patch
     * (half to even) exceeds {@code maxPixels * maxPixelsTolerance} ({@code should_tile}).
     */
    boolean shouldTile(ImageSize original) {
        int align = params().patchSize() * params().mergeSize();
        int hBar = Math.max(params().patchSize(), roundToAlign(original.height(), align));
        int wBar = Math.max(params().patchSize(), roundToAlign(original.width(), align));
        return params().maxTiles() > 1
                && (double) hBar * wBar > (double) params().maxPixels() * params().maxPixelsTolerance();
    }

    /**
     * The single-image size ({@code Lfm2VlImageProcessor.smart_resize}): the sides rounded half to even to the
     * merged patch, scaled down onto the pixel limit when above it or up when below the minimum.
     */
    ImageSize smartResize(ImageSize original) {
        int align = params().patchSize() * params().mergeSize();
        double height = original.height();
        double width = original.width();
        int hBar = Math.max(align, roundToAlign(height, align));
        int wBar = Math.max(align, roundToAlign(width, align));
        if ((double) hBar * wBar > params().maxPixels()) {
            double beta = Math.sqrt(height * width / params().maxPixels());
            hBar = Math.max(align, (int) Math.floor(height / beta / align) * align);
            wBar = Math.max(align, (int) Math.floor(width / beta / align) * align);
        } else if ((double) hBar * wBar < params().minPixels()) {
            double beta = Math.sqrt(params().minPixels() / (height * width));
            hBar = (int) Math.ceil(height * beta / align) * align;
            wBar = (int) Math.ceil(width * beta / align) * align;
        }
        return new ImageSize(wBar, hBar);
    }

    /**
     * The tile grid ({@code get_grid_layout}): among the grids of {@code [minTiles, maxTiles]} tiles, ordered by
     * tile count, the one whose aspect ratio is closest to the image's; on an exact tie a later grid wins when the
     * image covers more than half of its pixel area.
     */
    ImageSize gridLayout(ImageSize original) {
        double aspectRatio = (double) original.width() / original.height();
        double area = (double) original.width() * original.height();
        double tile = params().tileSize();
        double bestDiff = Double.MAX_VALUE;
        ImageSize best = new ImageSize(1, 1);
        for (ImageSize ratio : ImageOps.tileGrids(params().minTiles(), params().maxTiles())) {
            double diff = Math.abs(aspectRatio - (double) ratio.width() / ratio.height());
            if (diff < bestDiff) {
                bestDiff = diff;
                best = ratio;
            } else if (diff == bestDiff && area > 0.5 * tile * tile * ratio.width() * ratio.height()) {
                best = ratio;
            }
        }
        return best;
    }

    /** {@code round(x / align) * align} with Python's half-to-even rounding. */
    private static int roundToAlign(double x, int align) {
        return (int) Math.rint(x / align) * align;
    }
}

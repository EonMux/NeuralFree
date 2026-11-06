package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

/**
 * The DeepSeek-OCR preprocessor ({@code mtmd_image_preprocessor_deepseekocr}, tools/mtmd/mtmd-image.cpp). The
 * global view is the image fitted into an {@code image_size} square (bicubic, centered on the pad color, the
 * fitted sides rounded to nearest); it encodes last and carries the view separator. An image with a side above
 * {@code tileSize} also stretches (bicubic) to the grid of {@code [minTiles, maxTiles]} tiles closest to its
 * aspect ratio. DeepSeek-OCR 2 encodes the tiles one by one; DeepSeek-OCR fuses each tile row into one image,
 * its tiles stacked top to bottom, so its grid is one column of rows.
 */
public final class DeepseekOcrPreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;
    private final boolean fuseRows;

    /**
     * Creates the preprocessor over the params.
     *
     * @throws IllegalArgumentException when the params carry no tile size
     */
    public DeepseekOcrPreprocessor(ImageProcessorParams params) {
        if (params.tileSize() <= 0) {
            throw new IllegalArgumentException("the '" + params.projectorName()
                    + "' preprocessor needs a positive tile size");
        }
        this.params = params;
        this.fuseRows = params.preprocessing().family() == VisionPreprocessing.Family.DEEPSEEK_OCR;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        int base = params.imageSize();
        int tile = params.tileSize();
        PreparedImages.Builder out = new PreparedImages.Builder(params).overview(ImageOps.resize(image,
                new ImageSize(base, base), ResizeAlgo.BICUBIC, PadStyle.NEAREST, params.padColor()));
        if (image.width() <= tile && image.height() <= tile) {
            return out.build();
        }
        ImageSize grid = gridLayout(image.size());
        RasterImage refined = ImageOps.resize(image, new ImageSize(tile * grid.width(), tile * grid.height()),
                ResizeAlgo.BICUBIC, PadStyle.NONE, params.padColor());
        for (int row = 0; row < grid.height(); row++) {
            if (fuseRows) {
                out.tile(fusedRow(refined, row, grid.width(), tile));
            } else {
                for (int col = 0; col < grid.width(); col++) {
                    out.tile(ImageOps.crop(refined, col * tile, row * tile, tile, tile));
                }
            }
        }
        return out.grid(fuseRows ? 1 : grid.width(), grid.height()).build();
    }

    /** The tiles of one row stacked top to bottom: {@code tile} wide, {@code tile * cols} high. */
    private static RasterImage fusedRow(RasterImage refined, int row, int cols, int tile) {
        byte[] src = refined.rgbArray();
        byte[] out = new byte[tile * tile * cols * 3];
        for (int col = 0; col < cols; col++) {
            for (int py = 0; py < tile; py++) {
                int from = ((row * tile + py) * refined.width() + col * tile) * 3;
                System.arraycopy(src, from, out, (col * tile + py) * tile * 3, tile * 3);
            }
        }
        return RasterImage.wrap(tile, tile * cols, out);
    }

    /**
     * The tile grid ({@code find_closest_aspect_ratio}, in float precision): the grid of the target ratios closest
     * to the aspect ratio; on an exact tie a later grid wins when the image covers more than half of its area.
     */
    ImageSize gridLayout(ImageSize original) {
        float aspectRatio = (float) original.width() / original.height();
        float area = (float) ((long) original.width() * original.height());
        int tile = params.tileSize();
        float bestDiff = Float.MAX_VALUE;
        ImageSize best = new ImageSize(1, 1);
        for (ImageSize ratio : ImageOps.tileGrids(params.minTiles(), params.maxTiles())) {
            float diff = Math.abs(aspectRatio - (float) ratio.width() / ratio.height());
            if (diff < bestDiff) {
                bestDiff = diff;
                best = ratio;
            } else if (diff == bestDiff
                    && area > 0.5f * (float) ((long) tile * tile * ratio.width() * ratio.height())) {
                best = ratio;
            }
        }
        return best;
    }
}

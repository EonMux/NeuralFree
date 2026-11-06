package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.List;

/**
 * The Granite Vision 4.1 preprocessor ({@code mtmd_image_preprocessor_granite}, tools/mtmd/mtmd-image.cpp), the
 * llava-next "anyres" layout: every image, small ones too, gets the grid pinpoint with the most effective
 * resolution and the least waste ({@code select_best_resolution}); the overview is the image stretched to one
 * {@code image_size} square, the refined image the aspect-preserving resize to that pinpoint padded with black, cut
 * into its {@code gridX x gridY} whole {@code image_size} tiles in raster order. The group is one anyres image
 * ({@link PreparedImages#isAnyres()}): the encoder assembles the overview and the tile grid cropped to the
 * original aspect ratio.
 */
public final class Granite4AnyresPreprocessor extends LlavaUhdPreprocessor {

    /** The largest tile count per side: the reference stacks the tiles into one image of bounded height. */
    static final int MAX_GRID_SIDE = 1024;

    /** Creates the preprocessor over params that carry the grid pinpoints. */
    public Granite4AnyresPreprocessor(ImageProcessorParams params) {
        super(params);
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        ImageProcessorParams p = params();
        ImageSize original = image.size();
        int tile = p.imageSize();
        ImageSize refinedSize = selectBestResolution(original, p.gridPinpoints());
        int gridX = refinedSize.width() / tile;
        int gridY = refinedSize.height() / tile;
        if (gridX <= 0 || gridY <= 0 || gridX > MAX_GRID_SIDE || gridY > MAX_GRID_SIDE) {
            throw new IllegalArgumentException("the anyres resolution " + refinedSize.width() + "x"
                    + refinedSize.height() + " chosen for a " + original.width() + "x" + original.height()
                    + " image holds a " + gridX + "x" + gridY + " grid of " + tile + "-pixel tiles, outside 1.."
                    + MAX_GRID_SIDE + " per side");
        }
        RasterImage overview = ImageOps.resize(image, new ImageSize(tile, tile), p.overviewResizeAlgo(),
                p.overviewPadStyle(), p.padColor());
        RasterImage refined = ImageOps.resize(image, refinedSize, p.refinedResizeAlgo(), p.refinedPadStyle(),
                p.refinedPadColor());
        List<RasterImage> tiles = new ArrayList<>(gridX * gridY);
        for (int ty = 0; ty < gridY; ty++) {
            for (int tx = 0; tx < gridX; tx++) {
                tiles.add(ImageOps.crop(refined, tx * tile, ty * tile, tile, tile));
            }
        }
        return new PreparedImages.Builder(p)
                .overview(overview)
                .tiles(tiles)
                .grid(gridX, gridY)
                .overviewFirst(true)
                .anyres(original)
                .build();
    }
}

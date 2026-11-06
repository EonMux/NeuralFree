package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.List;

/**
 * The Idefics3 / SmolVLM preprocessor ({@code mtmd_image_preprocessor_idefics3}, tools/mtmd/mtmd-image.cpp; the
 * transformers {@code Idefics3ImageProcessor}): the image is cut into {@code image_size} tiles plus an overview
 * tile, the tiles in row-major order first and the overview last. Two paths size the tiled image:
 *
 * <ul>
 *   <li>with a preprocessor longest edge ({@code clip.vision.preproc_image_size}) and no pixel limits: the longest
 *   side resizes to that edge with the aspect ratio kept (the short side truncated, then rounded up to even), the
 *   result stretches up to whole tiles, and the overview is that tiled image shrunk to one tile;</li>
 *   <li>otherwise (old files without the edge, or a token budget): the llava-uhd slicing over the smart resize
 *   aligned to {@code image_size}, under the longest edge and the pixel limits.</li>
 * </ul>
 */
public final class Idefics3Preprocessor extends LlavaUhdPreprocessor {

    /** Creates the preprocessor over the params. */
    public Idefics3Preprocessor(ImageProcessorParams params) {
        super(params);
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        if (!usesLongestEdge()) {
            return super.prepare(image);
        }
        ImageProcessorParams p = params();
        int tile = p.imageSize();
        RasterImage resized = ImageOps.resize(image, longestEdgeSize(image.size()), p.resizeAlgo(), PadStyle.NONE,
                p.padColor());
        ImageSize grid = tileGrid(resized.size());
        RasterImage refined = ImageOps.resize(resized, new ImageSize(grid.width() * tile, grid.height() * tile),
                p.resizeAlgo(), PadStyle.NONE, p.padColor());
        RasterImage overview = ImageOps.resize(refined, tile, p.resizeAlgo(), PadStyle.NONE, p.padColor());
        List<RasterImage> tiles = new ArrayList<>(grid.width() * grid.height());
        for (int y = 0; y < grid.height(); y++) {
            for (int x = 0; x < grid.width(); x++) {
                tiles.add(ImageOps.crop(refined, x * tile, y * tile, tile, tile));
            }
        }
        return new PreparedImages.Builder(p)
                .overview(overview)
                .tiles(tiles)
                .grid(grid.width(), grid.height())
                .build();
    }

    /** Whether the longest-edge path applies: a longest edge is set and neither pixel limit is. */
    public boolean usesLongestEdge() {
        ImageProcessorParams p = params();
        return p.longestEdge() > 0 && p.minPixels() <= 0 && p.maxPixels() <= 0;
    }

    /**
     * The aspect-preserving size whose longest side is the preprocessor longest edge; the short side truncates,
     * then rounds up to even.
     *
     * @throws IllegalArgumentException when no longest edge is set, or the short side truncates to 0
     */
    public ImageSize longestEdgeSize(ImageSize original) {
        int longest = params().longestEdge();
        if (longest <= 0) {
            throw new IllegalArgumentException("the idefics3 longest-edge resize needs metadata key"
                    + " 'clip.vision.preproc_image_size' to be positive, got " + longest);
        }
        double aspectRatio = (double) original.width() / original.height();
        int width;
        int height;
        if (original.width() >= original.height()) {
            width = longest;
            height = (int) (longest / aspectRatio);
            height += height % 2;
        } else {
            height = longest;
            width = (int) (longest * aspectRatio);
            width += width % 2;
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("image " + original.width() + "x" + original.height()
                    + " is too elongated for the idefics3 longest edge " + longest + ": it resizes to "
                    + width + "x" + height);
        }
        return new ImageSize(width, height);
    }

    /** The tile grid covering {@code size}: each side rounded up to whole {@code image_size} tiles. */
    public ImageSize tileGrid(ImageSize size) {
        int tile = params().imageSize();
        return new ImageSize((size.width() + tile - 1) / tile, (size.height() + tile - 1) / tile);
    }

    /**
     * The slice plan of the smart-resize path: the image resizes to the smart resize aligned to
     * {@code image_size} (under the longest edge and the pixel limits), then cuts into {@code image_size} tiles;
     * the overview is the whole image at one tile.
     */
    @Override
    protected SliceInstructions sliceInstructions(ImageSize original) {
        ImageProcessorParams p = params();
        int tile = p.imageSize();
        ImageSize refined = ImageOps.smartResize(original, SmartResizeOptions.builder(tile)
                .minPixels(p.minPixels())
                .maxPixels(p.maxPixels())
                .longestEdge(p.longestEdge())
                .build());
        SliceInstructions inst = new SliceInstructions();
        inst.overviewSize = new ImageSize(tile, tile);
        inst.refinedSize = refined;
        inst.gridSize = tileGrid(refined);
        for (int y = 0; y < refined.height(); y += tile) {
            for (int x = 0; x < refined.width(); x += tile) {
                inst.slices.add(new SliceRect(x, y, Math.min(tile, refined.width() - x),
                        Math.min(tile, refined.height() - y)));
            }
        }
        return inst;
    }
}

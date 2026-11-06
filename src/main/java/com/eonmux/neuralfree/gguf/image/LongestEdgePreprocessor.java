package com.eonmux.neuralfree.gguf.image;

/**
 * The longest-edge preprocessor ({@code mtmd_image_preprocessor_longest_edge}, tools/mtmd/mtmd-image.cpp): the
 * longest side caps at the preprocessor longest edge (Pixtral floors the sides before the align rounding), the
 * smart resize aligns to the merged patch size and the pixel limits apply; {@code minPixels} may push an edge
 * back above the longest edge, as llama.cpp orders the options.
 */
public final class LongestEdgePreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public LongestEdgePreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        ImageSize target = targetSize(image.size());
        RasterImage resized = ImageOps.resize(image, target, params.resizeAlgo(), params.padStyle(),
                params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    /** The target size ({@code get_target_size}). */
    public ImageSize targetSize(ImageSize original) {
        if (params.longestEdge() <= 0) {
            throw new IllegalArgumentException("the longest-edge preprocessor needs a positive longestEdge");
        }
        ImageSize size = original;
        if (params.floorBeforeAlign()) {
            double ratio = (double) Math.max(size.width(), size.height()) / params.longestEdge();
            if (ratio > 1.0) {
                int width = (int) Math.floor(size.width() / ratio);
                int height = (int) Math.floor(size.height() / ratio);
                if (width <= 0 || height <= 0) {
                    throw new IllegalArgumentException("image dimensions must be positive after resizing");
                }
                size = new ImageSize(width, height);
            }
        }
        int longest = Math.min(params.longestEdge(), Math.max(size.width(), size.height()));
        return ImageOps.smartResize(size, SmartResizeOptions.builder(params.patchSize() * params.mergeSize())
                .minPixels(Math.max(0, params.minPixels()))
                .maxPixels(Math.max(0, params.maxPixels()))
                .longestEdge(longest)
                .roundHalfEven(params.roundHalfEven())
                .build());
    }
}

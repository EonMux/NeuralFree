package com.eonmux.neuralfree.gguf.image;

/**
 * The dynamic-size preprocessor ({@code mtmd_image_preprocessor_dyn_size}, tools/mtmd/mtmd-image.cpp): the
 * whole image resizes once to a target size that keeps the aspect ratio, aligns both sides to the merged patch
 * size and fits the pixel limits — the transformers "smart resize" the Qwen2-VL, MiniCPM-V 4.6, Phi-4, Kimi-VL
 * and GLM-4V processors use. One tile.
 */
public final class DynSizePreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public DynSizePreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        ImageSize target = targetSize(image.size());
        RasterImage resized = ImageOps.resize(image, target, params.resizeAlgo(), params.padStyle(),
                params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    /**
     * The target size: the sides align to {@code patchSize * mergeSize} and the area fits
     * {@code [minPixels, maxPixels]} ({@code get_target_size}).
     */
    public ImageSize targetSize(ImageSize original) {
        return ImageOps.smartResize(original, SmartResizeOptions.builder(params.patchSize() * params.mergeSize())
                .minPixels(params.minPixels())
                .maxPixels(params.maxPixels())
                .roundHalfEven(params.roundHalfEven())
                .build());
    }
}

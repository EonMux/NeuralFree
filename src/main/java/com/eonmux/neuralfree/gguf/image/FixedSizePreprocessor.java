package com.eonmux.neuralfree.gguf.image;

/**
 * The fixed-size preprocessor ({@code mtmd_image_preprocessor_fixed_size}, tools/mtmd/mtmd-image.cpp): every
 * input resizes to the square CLIP image size, aspect-preserving with the pad style when the projector pads
 * (LLaVA-1.5, Gemma 3) and stretched when it does not (YASA2). One tile, whose patch grid is the whole token
 * count.
 */
public final class FixedSizePreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public FixedSizePreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        RasterImage resized = ImageOps.resize(image, params.imageSize(), params.resizeAlgo(),
                params.padStyle(), params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }
}

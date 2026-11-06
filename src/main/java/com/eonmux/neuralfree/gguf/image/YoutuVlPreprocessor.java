package com.eonmux.neuralfree.gguf.image;

/**
 * The Youtu-VL preprocessor ({@code mtmd_image_preprocessor_youtuvl}, tools/mtmd/mtmd-image.cpp): starting at
 * scale 1 and stepping down by 0.02, each side scales and rounds up to {@code patch * merge} (at least one merged
 * patch) until the patch count fits {@code maxPixels / patch^2} (256 patches without a limit); the image then
 * resizes to that size. One tile.
 */
public final class YoutuVlPreprocessor implements ImagePreprocessor {

    private static final float SCALE_STEP = 0.02f;
    private static final int DEFAULT_MAX_PATCHES = 256;

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public YoutuVlPreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        RasterImage resized = ImageOps.resize(image, targetSize(image.size()), params.resizeAlgo(),
                params.padStyle(), params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    /** The target size of the linear scale search. */
    ImageSize targetSize(ImageSize original) {
        int patch = params.patchSize();
        int align = patch * params.mergeSize();
        long maxPatches = params.maxPixels() > 0 ? params.maxPixels() / ((long) patch * patch) : DEFAULT_MAX_PATCHES;
        float scale = 1.0f;
        int targetHeight = original.height();
        int targetWidth = original.width();
        while (scale > 0.0f) {
            targetHeight = scaledSide(scale, original.height(), align);
            targetWidth = scaledSide(scale, original.width(), align);
            long patches = (long) (targetHeight / patch) * (targetWidth / patch);
            if (patches > maxPatches) {
                scale -= SCALE_STEP;
            } else {
                break;
            }
        }
        return new ImageSize(targetWidth, targetHeight);
    }

    private static int scaledSide(float scale, int size, int align) {
        float scaled = size * scale;
        return Math.max(align, (int) Math.ceil(scaled / align) * align);
    }
}

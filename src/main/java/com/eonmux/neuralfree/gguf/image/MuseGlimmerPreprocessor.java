package com.eonmux.neuralfree.gguf.image;

/**
 * The Muse Glimmer preprocessor ({@code mtmd_image_preprocessor_muse_glimmer}, tools/mtmd/mtmd-image.cpp; the
 * transformers {@code get_aspect_ratio_preserving_size}): the merged patch grid of the image, scaled down onto
 * the token cap {@code maxPixels / (patch * merge)^2} when above it, picks among the floor and ceiling of each
 * side the grid within the cap closest to the aspect ratio (ties to more tokens); the image stretches to it.
 * One tile.
 */
public final class MuseGlimmerPreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /**
     * Creates the preprocessor over the params.
     *
     * @throws IllegalArgumentException when the params carry no max pixel limit
     */
    public MuseGlimmerPreprocessor(ImageProcessorParams params) {
        if (params.maxPixels() <= 0) {
            throw new IllegalArgumentException("the '" + params.projectorName()
                    + "' preprocessor needs a positive max pixel limit");
        }
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        RasterImage resized = ImageOps.resize(image, targetSize(image.size()), params.resizeAlgo(), PadStyle.NONE,
                new int[] {0, 0, 0});
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    /** The stretched size: the chosen merged patch grid in pixels. */
    ImageSize targetSize(ImageSize original) {
        int patchHw = params.patchSize() * params.mergeSize();
        long maxTokens = params.maxPixels() / params.mergedPatchArea();
        double nph = (double) original.height() / patchHw;
        double npw = (double) original.width() / patchHw;
        double ratio = nph > 0.0 ? npw / nph : 1.0;
        if (nph * npw > (double) maxTokens) {
            nph = Math.sqrt((double) maxTokens / ratio);
            npw = nph * ratio;
        }
        int[] hs = {(int) Math.floor(nph), (int) Math.ceil(nph)};
        int[] ws = {(int) Math.floor(npw), (int) Math.ceil(npw)};
        double targetAr = (double) original.height() / original.width();
        int bestH = -1;
        int bestW = -1;
        double bestD = 0.0;
        for (int h : hs) {
            for (int w : ws) {
                if (h < 1 || w < 1 || (long) h * w > maxTokens) {
                    continue;
                }
                double d = Math.abs((double) h / w - targetAr);
                if (bestH < 0 || d < bestD || (d == bestD && (long) h * w > (long) bestH * bestW)) {
                    bestH = h;
                    bestW = w;
                    bestD = d;
                }
            }
        }
        if (bestH < 0) {
            // no candidate fits under the cap: round and clamp
            bestH = Math.max(1, (int) Math.round(nph));
            bestW = Math.max(1, (int) Math.round(npw));
        }
        return new ImageSize(bestW * patchHw, bestH * patchHw);
    }
}

package com.eonmux.neuralfree.gguf.image;

/**
 * The LFM3-VL preprocessor ({@code mtmd_image_preprocessor_lfm3vl}, tools/mtmd/mtmd-image.cpp): the
 * dynamic-size pipeline sized exactly as transformers' Qwen2-VL {@code smart_resize}, in double precision with
 * Python's half-to-even rounding before the pixel checks; each side is at least one merged patch. One tile,
 * stretched to the size. A video's frames take the reference video processor's own size and pixel limits; a frame
 * the decode already sized only snaps to the merged patch grid.
 */
public final class Lfm3vlPreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public Lfm3vlPreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    /** The pixel limits of the reference video processor ({@code Lfm3VlVideoProcessor}). */
    private static final long VIDEO_MIN_PIXELS = 4096;
    private static final long VIDEO_MAX_PIXELS = 25165824;
    /** The temporal patch size the reference video processor spans its size check over. */
    private static final int VIDEO_TEMPORAL_FACTOR = 2;

    @Override
    public PreparedImages prepare(RasterImage image) {
        RasterImage resized = ImageOps.resize(image, targetSize(image.size()), params.resizeAlgo(),
                params.padStyle(), params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    @Override
    public ImageSize videoFrameSize(int nFrames, int width, int height) {
        if (nFrames <= 0) {
            throw new IllegalArgumentException("a video frame size needs a positive frame count, got " + nFrames);
        }
        return videoSmartResize(width, height, factor(), VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, nFrames,
                VIDEO_TEMPORAL_FACTOR);
    }

    /**
     * The frame of a video that the decode already sized: each side snaps to the merged patch multiple, the size
     * the reference preprocessor keeps ({@code preprocess_video_frame}).
     */
    @Override
    public PreparedImages prepareVideoFrame(RasterImage frame) {
        int f = factor();
        ImageSize size = new ImageSize(Math.max(f, (int) Math.rint((double) frame.width() / f) * f),
                Math.max(f, (int) Math.rint((double) frame.height() / f) * f));
        RasterImage resized = ImageOps.resize(frame, size, params.resizeAlgo(), params.padStyle(),
                params.padColor());
        return new PreparedImages.Builder(params).tile(resized).build();
    }

    /** The merged patch side both the image and the video target sizes snap to. */
    private int factor() {
        return params.patchSize() * params.mergeSize();
    }

    /** The target size ({@code smart_resize} of one frame). */
    ImageSize targetSize(ImageSize original) {
        return smartResize(original.width(), original.height(), factor(), params.minPixels(), params.maxPixels());
    }

    /**
     * transformers' Qwen2-VL {@code smart_resize} of one frame: both sides round half to even to {@code factor};
     * an area above {@code maxPixels} floors onto it, one below {@code minPixels} ceils onto it.
     */
    static ImageSize smartResize(int width, int height, int factor, long minPixels, long maxPixels) {
        double h = height;
        double w = width;
        long hBar = (long) Math.rint(h / factor) * factor;
        long wBar = (long) Math.rint(w / factor) * factor;
        if (hBar * wBar > maxPixels) {
            double beta = Math.sqrt(h * w / maxPixels);
            hBar = (long) Math.floor(h / beta / factor) * factor;
            wBar = (long) Math.floor(w / beta / factor) * factor;
        } else if (hBar * wBar < minPixels) {
            double beta = Math.sqrt(minPixels / (h * w));
            hBar = (long) Math.ceil(h * beta / factor) * factor;
            wBar = (long) Math.ceil(w * beta / factor) * factor;
        }
        return new ImageSize((int) Math.max(factor, wBar), (int) Math.max(factor, hBar));
    }

    /**
     * The reference video frame size ({@code mtmd_image_preprocessor_lfm3vl::smart_resize} with a temporal
     * factor): edges below one merged patch first scale up, the area check spans the frame count rounded to whole
     * temporal groups, and both sides snap to the merged patch multiple.
     */
    static ImageSize videoSmartResize(int width, int height, int factor, long minPixels, long maxPixels,
                                      int nFrames, int temporalFactor) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("a video frame size needs a positive size, got " + width + "x"
                    + height);
        }
        if (temporalFactor <= 0) {
            throw new IllegalArgumentException("the temporal factor must be positive, got " + temporalFactor);
        }
        double h = height;
        double w = width;
        if (temporalFactor > 1 && (h < factor || w < factor)) {
            double scale = Math.max(factor / h, factor / w);
            h = (int) (h * scale);
            w = (int) (w * scale);
        }
        long tBar = (long) Math.rint((double) nFrames / temporalFactor) * temporalFactor;
        int hBar = (int) Math.rint(h / factor) * factor;
        int wBar = (int) Math.rint(w / factor) * factor;
        if (tBar * hBar * wBar > maxPixels) {
            double beta = Math.sqrt(nFrames * h * w / maxPixels);
            hBar = (int) Math.floor(h / beta / factor) * factor;
            wBar = (int) Math.floor(w / beta / factor) * factor;
        } else if (tBar * hBar * wBar < minPixels) {
            double beta = Math.sqrt(minPixels / (nFrames * h * w));
            hBar = (int) Math.ceil(h * beta / factor) * factor;
            wBar = (int) Math.ceil(w * beta / factor) * factor;
        }
        return new ImageSize((int) Math.max(factor, wBar), (int) Math.max(factor, hBar));
    }
}

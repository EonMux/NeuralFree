package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.List;

/**
 * The Step3-VL preprocessor ({@code mtmd_image_preprocessor_step3vl}, tools/mtmd/mtmd-image.cpp; stepfun-ai's
 * {@code processing_step3.py}). A thin extreme strip pads to a black square, and an image whose longest side
 * exceeds the longest edge shrinks to it (Pillow bilinear). The overview is the prepared image at
 * {@code image_size}. A large or elongated image also cuts into square windows (504 pixels, or the short side of
 * a small elongated image), the window grid covering the image rounded to whole windows; each window, black past
 * the image, becomes a 504 tile. The overview and the windows resample straight to normalized floats (a
 * half-pixel-centered bilinear over the normalized samples), so they are {@link FloatImage} tiles. The windows
 * encode before the overview.
 */
public final class Step3vlPreprocessor implements ImagePreprocessor {

    /** The side every window tile resizes to ({@code default_image_crop_size}). */
    static final int CROP_SIZE = 504;

    private static final float SMALL_ASPECT_RATIO_LIMIT = 1.5f;
    private static final float WIDE_ASPECT_RATIO_LIMIT = 4.0f;
    private static final float CROP_ROUNDING_THRESHOLD = 0.2f;

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public Step3vlPreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        RasterImage prepared = prepareImage(image);
        Plan plan = plan(prepared.size());
        PreparedImages.Builder out = new PreparedImages.Builder(params)
                .overview(bilinearToFloat(prepared, params.imageSize(), params.imageSize(), params.mean(),
                        params.std()));
        if (plan.window <= 0) {
            return out.build();
        }
        RasterImage forCrop = prepared;
        if (!plan.refinedSize.equals(prepared.size())) {
            forCrop = ImageOps.resize(prepared, plan.refinedSize, ResizeAlgo.BILINEAR, PadStyle.NONE,
                    new int[] {0, 0, 0});
        }
        for (int y : plan.ys) {
            for (int x : plan.xs) {
                RasterImage window = cropWithBlackPadding(forCrop, x, y, plan.window, plan.window);
                out.tile(bilinearToFloat(window, CROP_SIZE, CROP_SIZE, params.mean(), params.std()));
            }
        }
        return out.grid(plan.xs.size(), plan.ys.size()).build();
    }

    /**
     * The image before the windows ({@code prepare_image}): a strip under 32 pixels thick with an aspect ratio
     * beyond 4 pads to a black square (the image at the top-left), then a longest side above the longest edge
     * scales down to it, the sides floored.
     */
    RasterImage prepareImage(RasterImage image) {
        RasterImage resized = image;
        float aspectRatio = (float) image.width() / image.height();
        if (Math.min(image.width(), image.height()) < 32
                && (aspectRatio > WIDE_ASPECT_RATIO_LIMIT || aspectRatio < 1.0f / WIDE_ASPECT_RATIO_LIMIT)) {
            int square = Math.max(image.width(), image.height());
            resized = ImageOps.composite(RasterImage.filled(square, square, 0, 0, 0), image, 0, 0);
        }
        int longest = Math.max(resized.width(), resized.height());
        if (longest > params.longestEdge()) {
            float scale = (float) params.longestEdge() / longest;
            ImageSize size = new ImageSize(Math.max(1, (int) Math.floor(resized.width() * scale)),
                    Math.max(1, (int) Math.floor(resized.height() * scale)));
            resized = ImageOps.resize(resized, size, ResizeAlgo.BILINEAR, PadStyle.NONE, new int[] {0, 0, 0});
        }
        return resized;
    }

    /** The window plan of a prepared size ({@code build_slice_instructions}); no window when it is 0. */
    Plan plan(ImageSize prepared) {
        int window = windowSize(Math.max(prepared.width(), prepared.height()),
                Math.min(prepared.width(), prepared.height()));
        if (window <= 0) {
            return new Plan(0, ImageSize.EMPTY, new ArrayList<>(), new ArrayList<>());
        }
        int cropWidth = cropExtent(prepared.width(), window);
        int cropHeight = cropExtent(prepared.height(), window);
        return new Plan(window, new ImageSize(cropWidth, cropHeight), starts(cropWidth, window),
                starts(cropHeight, window));
    }

    /**
     * The window side ({@code determine_window_size}): up to {@code image_size}, the short side for an image
     * longer than 1.5:1 and none otherwise; above, 504, or the short side up to 504 beyond 4:1.
     */
    private int windowSize(int longer, int shorter) {
        float aspectRatio = (float) longer / shorter;
        if (longer <= params.imageSize()) {
            return aspectRatio > SMALL_ASPECT_RATIO_LIMIT ? shorter : 0;
        }
        return aspectRatio > WIDE_ASPECT_RATIO_LIMIT ? Math.min(shorter, CROP_SIZE) : CROP_SIZE;
    }

    /** The side rounded to whole windows, a fraction above 0.2 rounding up ({@code calc_crop_extent}). */
    static int cropExtent(int length, int window) {
        float ratio = (float) length / window;
        if (ratio < 1.0f) {
            return length;
        }
        float decimal = ratio - (float) Math.floor(ratio);
        int rounded = decimal > CROP_ROUNDING_THRESHOLD ? (int) Math.floor(ratio) + 1 : (int) Math.floor(ratio);
        return window * rounded;
    }

    /** The window starts along a side, the last pulled back inside it ({@code calc_grid}). */
    static List<Integer> starts(int length, int window) {
        int n = length <= window ? 1 : (int) Math.ceil((float) (length - window) / window + 1.0f);
        List<Integer> starts = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            starts.add(window * i);
        }
        if (n > 1 && starts.get(n - 1) + window > length) {
            starts.set(n - 1, length - window);
        }
        return starts;
    }

    /** The {@code w x h} rectangle at {@code (x, y)}, black where it leaves the image. */
    static RasterImage cropWithBlackPadding(RasterImage image, int x, int y, int w, int h) {
        byte[] out = new byte[w * h * 3];
        int x0 = Math.max(0, x);
        int y0 = Math.max(0, y);
        int x1 = Math.min(image.width(), x + w);
        int y1 = Math.min(image.height(), y + h);
        if (x0 < x1 && y0 < y1) {
            byte[] src = image.rgbArray();
            for (int sy = y0; sy < y1; sy++) {
                System.arraycopy(src, (sy * image.width() + x0) * 3, out, ((sy - y) * w + (x0 - x)) * 3,
                        (x1 - x0) * 3);
            }
        }
        return RasterImage.wrap(w, h, out);
    }

    /**
     * The half-pixel-centered bilinear resample to normalized floats ({@code img_u8_resize_bilinear_to_f32}):
     * each target pixel interpolates the normalized samples of its four nearest source pixels, the edges clamped;
     * a same-size target only normalizes.
     */
    static FloatImage bilinearToFloat(RasterImage src, int targetWidth, int targetHeight, float[] mean,
                                      float[] std) {
        if (src.width() == targetWidth && src.height() == targetHeight) {
            return FloatImage.wrap(targetWidth, targetHeight, src.normalized(mean, std), mean.clone(), std.clone());
        }
        float[] out = new float[targetWidth * targetHeight * 3];
        byte[] rgb = src.rgbArray();
        int width = src.width();
        float scaleX = (float) width / targetWidth;
        float scaleY = (float) src.height() / targetHeight;
        for (int y = 0; y < targetHeight; y++) {
            float srcY = (y + 0.5f) * scaleY - 0.5f;
            int y0Floor = (int) Math.floor(srcY);
            int row0 = clamp(y0Floor, src.height()) * width;
            int row1 = clamp(y0Floor + 1, src.height()) * width;
            float ly = srcY - y0Floor;
            for (int x = 0; x < targetWidth; x++) {
                float srcX = (x + 0.5f) * scaleX - 0.5f;
                int x0Floor = (int) Math.floor(srcX);
                int x0 = clamp(x0Floor, width);
                int x1 = clamp(x0Floor + 1, width);
                float lx = srcX - x0Floor;
                int dst = (y * targetWidth + x) * 3;
                for (int c = 0; c < 3; c++) {
                    float v00 = normalize(rgb[(row0 + x0) * 3 + c], mean[c], std[c]);
                    float v01 = normalize(rgb[(row0 + x1) * 3 + c], mean[c], std[c]);
                    float v10 = normalize(rgb[(row1 + x0) * 3 + c], mean[c], std[c]);
                    float v11 = normalize(rgb[(row1 + x1) * 3 + c], mean[c], std[c]);
                    float top = v00 + (v01 - v00) * lx;
                    float bottom = v10 + (v11 - v10) * lx;
                    out[dst + c] = top + (bottom - top) * ly;
                }
            }
        }
        return FloatImage.wrap(targetWidth, targetHeight, out, mean.clone(), std.clone());
    }

    private static float normalize(byte sample, float mean, float std) {
        return ((sample & 0xFF) / 255.0f - mean) / std;
    }

    private static int clamp(int v, int size) {
        return Math.max(0, Math.min(v, size - 1));
    }

    /** The window plan: the window side, the size the image resizes to before cropping, the window starts. */
    static final class Plan {
        final int window;
        final ImageSize refinedSize;
        final List<Integer> xs;
        final List<Integer> ys;

        Plan(int window, ImageSize refinedSize, List<Integer> xs, List<Integer> ys) {
            this.window = window;
            this.refinedSize = refinedSize;
            this.xs = xs;
            this.ys = ys;
        }
    }
}

package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Image operations shared by every preprocessor, the {@code img_tool} of llama.cpp
 * (tools/mtmd/mtmd-image.cpp): aspect-ratio-preserving resize onto a padded canvas, crop, and the
 * smart resize that computes a dynamic-resolution target size ({@code calc_size_preserved_ratio}, the
 * transformers {@code smart_resize} vLLM's Qwen2-VL processor uses).
 */
public final class ImageOps {

    private ImageOps() {
    }

    /**
     * Resizes {@code src} to {@code target}. With {@link PadStyle#NONE} the image stretches; with the other
     * styles it scales to fit inside the target, sits at its center and the border fills with {@code padColor}.
     *
     * @throws IllegalArgumentException when a target side is out of the {@code [1, 65536]} range of the resample
     */
    public static RasterImage resize(RasterImage src, ImageSize target, ResizeAlgo algo, PadStyle padStyle,
                                     int[] padColor) {
        requirePadColor(padColor);
        if (padStyle == PadStyle.NONE || target.equals(src.size())) {
            return ImageResample.resample(src, target.width(), target.height(), algo);
        }

        float scaleW = (float) target.width() / src.width();
        float scaleH = (float) target.height() / src.height();
        float scale = Math.min(scaleW, scaleH);

        int newWidth;
        int newHeight;
        if (padStyle == PadStyle.NEAREST) {
            newWidth = Math.min((int) Math.round(src.width() * scale), target.width());
            newHeight = Math.min((int) Math.round(src.height() * scale), target.height());
        } else {
            newWidth = Math.min((int) Math.ceil(src.width() * scale), target.width());
            newHeight = Math.min((int) Math.ceil(src.height() * scale), target.height());
        }

        RasterImage fitted = ImageResample.resample(src, newWidth, newHeight, algo);
        RasterImage canvas = RasterImage.filled(target.width(), target.height(),
                padColor[0], padColor[1], padColor[2]);
        int offsetX;
        int offsetY;
        if (padStyle == PadStyle.NEAREST) {
            // std::round of the float half: an odd border puts the extra pixel on the leading side
            offsetX = Math.round((target.width() - newWidth) / 2.0f);
            offsetY = Math.round((target.height() - newHeight) / 2.0f);
        } else {
            offsetX = (target.width() - newWidth) / 2;
            offsetY = (target.height() - newHeight) / 2;
        }
        return composite(canvas, fitted, offsetX, offsetY);
    }

    /** {@link #resize(RasterImage, ImageSize, ResizeAlgo, PadStyle, int[])} onto a square canvas. */
    public static RasterImage resize(RasterImage src, int size, ResizeAlgo algo, PadStyle padStyle, int[] padColor) {
        return resize(src, new ImageSize(size, size), algo, padStyle, padColor);
    }

    /**
     * The sub-image at {@code (x, y)} of size {@code w} x {@code h} (mtmd-image.cpp {@code img_tool::crop}).
     *
     * @throws IllegalArgumentException when the rectangle is empty or reaches outside the source
     */
    public static RasterImage crop(RasterImage image, int x, int y, int w, int h) {
        if (x < 0 || y < 0 || w <= 0 || h <= 0) {
            throw new IllegalArgumentException("crop rectangle must be positive, got x=" + x + " y=" + y
                    + " w=" + w + " h=" + h);
        }
        if (x + w > image.width() || y + h > image.height()) {
            throw new IllegalArgumentException("crop rectangle " + w + "x" + h + " at (" + x + ", " + y
                    + ") reaches outside the " + image.width() + "x" + image.height() + " image");
        }
        byte[] out = new byte[w * h * 3];
        byte[] src = image.rgbArray();
        for (int row = 0; row < h; row++) {
            System.arraycopy(src, ((y + row) * image.width() + x) * 3, out, row * w * 3, w * 3);
        }
        return RasterImage.wrap(w, h, out);
    }

    /**
     * {@code src} drawn into {@code canvas} at the offset, pixels outside the canvas skipped
     * (mtmd-image.cpp {@code img_tool::composite}).
     */
    public static RasterImage composite(RasterImage canvas, RasterImage src, int offsetX, int offsetY) {
        byte[] dst = canvas.rgbArray();
        byte[] srcPixels = src.rgbArray();
        int dstWidth = canvas.width();
        int dstHeight = canvas.height();
        for (int y = 0; y < src.height(); y++) {
            int dy = y + offsetY;
            if (dy < 0 || dy >= dstHeight) {
                continue;
            }
            for (int x = 0; x < src.width(); x++) {
                int dx = x + offsetX;
                if (dx < 0 || dx >= dstWidth) {
                    continue;
                }
                int srcI = (y * src.width() + x) * 3;
                int dstI = (dy * dstWidth + dx) * 3;
                dst[dstI] = srcPixels[srcI];
                dst[dstI + 1] = srcPixels[srcI + 1];
                dst[dstI + 2] = srcPixels[srcI + 2];
            }
        }
        return RasterImage.wrap(dstWidth, dstHeight, dst);
    }

    /**
     * The target size of an aspect-ratio-preserving resize whose sides are multiples of
     * {@code opts.alignSize()} (mtmd-image.cpp {@code img_tool::calc_size_preserved_ratio}): the sides round to
     * the multiple (or ceil under a longest edge), then the pixel area is pulled into the
     * {@code [minPixels, maxPixels]} window by a symmetric scale, each side floored or ceiled onto the multiple.
     * Returns {@link ImageSize#EMPTY} for a non-positive input side.
     */
    public static ImageSize smartResize(ImageSize input, SmartResizeOptions opts) {
        if (opts.alignSize() <= 0) {
            throw new IllegalArgumentException("alignSize must be positive, got " + opts.alignSize());
        }
        int width = input.width();
        int height = input.height();
        if (width <= 0 || height <= 0) {
            return ImageSize.EMPTY;
        }

        int wBar;
        int hBar;
        if (opts.longestEdge() > 0) {
            float scale = Math.min((float) opts.longestEdge() / width, (float) opts.longestEdge() / height);
            wBar = ceilByFactor(width * scale, opts);
            hBar = ceilByFactor(height * scale, opts);
        } else {
            // always align up first
            wBar = Math.max(opts.alignSize(), roundByFactor(width, opts));
            hBar = Math.max(opts.alignSize(), roundByFactor(height, opts));
        }

        if (opts.maxPixels() > 0 && (long) hBar * wBar > opts.maxPixels()) {
            float beta = (float) Math.sqrt((double) ((float) height * width / opts.maxPixels()));
            hBar = Math.max(opts.alignSize(), floorByFactor(height / beta, opts));
            wBar = Math.max(opts.alignSize(), floorByFactor(width / beta, opts));
        } else if (opts.minPixels() > 0 && (long) hBar * wBar < opts.minPixels()) {
            float beta = (float) Math.sqrt((double) (opts.minPixels() / ((float) height * width)));
            hBar = ceilByFactor(height * beta, opts);
            wBar = ceilByFactor(width * beta, opts);
        }

        return new ImageSize(wBar, hBar);
    }

    /**
     * Every {@code w x h} tile grid with {@code minTiles <= w * h <= maxTiles}, in first-seen order sorted stably
     * by tile count: the {@code get_target_ratios} of the LFM2 and DeepSeek-OCR preprocessors, Python's
     * {@code sorted} over the InternVL-style candidate set.
     */
    public static List<ImageSize> tileGrids(int minTiles, int maxTiles) {
        List<ImageSize> grids = new ArrayList<>();
        for (int n = minTiles; n <= maxTiles; n++) {
            for (int w = 1; w <= n; w++) {
                for (int h = 1; h <= n; h++) {
                    ImageSize grid = new ImageSize(w, h);
                    if (w * h >= minTiles && w * h <= maxTiles && !grids.contains(grid)) {
                        grids.add(grid);
                    }
                }
            }
        }
        grids.sort(Comparator.comparingInt(ImageSize::area));
        return grids;
    }

    /** Rounds to the nearest multiple of the align size; half to even when {@code roundHalfEven} is set. */
    private static int roundByFactor(float x, SmartResizeOptions opts) {
        float q = x / opts.alignSize();
        double rounded = opts.roundHalfEven() ? Math.rint(q) : Math.round(q);
        return (int) rounded * opts.alignSize();
    }

    private static int ceilByFactor(float x, SmartResizeOptions opts) {
        return (int) Math.ceil(x / opts.alignSize()) * opts.alignSize();
    }

    private static int floorByFactor(float x, SmartResizeOptions opts) {
        return (int) Math.floor(x / opts.alignSize()) * opts.alignSize();
    }

    private static void requirePadColor(int[] padColor) {
        if (padColor == null || padColor.length != 3) {
            throw new IllegalArgumentException("padColor must hold 3 values, got "
                    + (padColor == null ? "null" : padColor.length));
        }
    }
}

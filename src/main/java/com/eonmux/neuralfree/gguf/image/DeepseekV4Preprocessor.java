package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

/**
 * The DeepSeek-V4 and V4.1 preprocessors ({@code mtmd_image_preprocessor_deepseek4v} and
 * {@code mtmd_image_preprocessor_deepseek41v}, tools/mtmd/mtmd-image.cpp; {@code load_image} of the models'
 * {@code inference/image_processor.py}). Both size the image so its token block fits the cap
 * ({@code maxPixels} over the merged patch area): an image under the min pixels scales up first, the patch-aligned
 * size stands when its block fits, otherwise the solver picks the largest aspect-preserving size that does.
 *
 * <p>V4 keeps an aspect ratio of at most 8 (a wider image is planned as 8:1 and stretched), reserves 3 tokens of
 * the cap for the lead pads that align the image in the prompt, rounds the merged rows to an even count, and pads
 * the fitted image centered on gray. V4.1 counts the reading-order block, fits the image with Python's half-even
 * rounding, centers it on gray, and rounds the normalized samples to bfloat16 (round to nearest even), as the
 * model's {@code load_image} does: its tile is a {@link FloatImage}.</p>
 */
public final class DeepseekV4Preprocessor implements ImagePreprocessor {

    /** The DeepSeek-V4 aspect ratio limit ({@code vision_max_wh_ratio}). */
    static final int V4_MAX_WH_RATIO = 8;

    /** The lead pads the DeepSeek-V4 cap reserves ({@code COMPRESS_PAD_TO - 1}). */
    static final int V4_LEAD_PAD_RESERVE = 3;

    private final ImageProcessorParams params;
    private final boolean v41;

    /**
     * Creates the preprocessor over the params.
     *
     * @throws IllegalArgumentException when the params carry no token cap
     */
    public DeepseekV4Preprocessor(ImageProcessorParams params) {
        this.params = params;
        this.v41 = params.preprocessing().family() == VisionPreprocessing.Family.DEEPSEEK41V;
        if (maxTokens() <= 2) {
            throw new IllegalArgumentException("the '" + params.projectorName() + "' preprocessor needs a token"
                    + " block cap above 2, got " + maxTokens());
        }
    }

    /** The token block cap: {@code image_max_pixels} is {@code dsv4_max_n_token} times the merged patch area. */
    private int maxTokens() {
        long cell = (long) params.patchSize() * params.mergeSize();
        return (int) Math.min(Integer.MAX_VALUE, params.maxPixels() / (cell * cell));
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        ImageTile out = v41 ? bf16(prepare41(image)) : prepare4(image);
        return new PreparedImages.Builder(params).tile(out).build();
    }

    /** The normalized samples of {@code image} rounded to bfloat16. */
    private FloatImage bf16(RasterImage image) {
        float[] samples = image.normalized(params.mean(), params.std());
        for (int i = 0; i < samples.length; i++) {
            samples[i] = Float.intBitsToFloat(KvBlockCodec.fp32ToBf16(samples[i]) << 16);
        }
        return FloatImage.wrap(image.width(), image.height(), samples, params.mean(), params.std());
    }

    // ------------------------------------------------------------------ deepseek4v

    private RasterImage prepare4(RasterImage image) {
        int p = params.patchSize();
        int r = params.mergeSize();
        long width = image.width();
        long height = image.height();
        if (width > height * V4_MAX_WH_RATIO) {
            width = height * V4_MAX_WH_RATIO;
        }
        long minPixels = params.minPixels();
        if (minPixels > 0 && width * height < minPixels) {
            double up = Math.sqrt((double) minPixels / ((double) width * height));
            width = (long) (width * up);
            height = (long) (height * up);
        }
        int[] best = {align((int) width, p), align((int) height, p)};
        safeResize((int) height, (int) width, best, p, r, maxTokens());
        ImageSize target = new ImageSize(best[0], best[1]);
        if (image.width() >= (long) V4_MAX_WH_RATIO * image.height()) {
            return ImageOps.resize(image, target, params.resizeAlgo(), PadStyle.NONE, params.padColor());
        }
        return ImageOps.resize(image, target, params.resizeAlgo(), PadStyle.NEAREST, params.padColor());
    }

    /** The V4 token block of a {@code bestWidth x bestHeight} image without lead pads. */
    static int tokens4(int bestHeight, int bestWidth, int p, int r) {
        int h = ((bestHeight / p) + r - 1) / r;
        int w = ((bestWidth / p) + r - 1) / r;
        int rows = h + h % 2;
        int rowLen = w + 1;
        int padLast = (rows / 2 * rowLen) % 2 * 2;
        return 1 + rows * rowLen + padLast + 1;
    }

    /** {@code safe_resize}: shrinks {@code best = {width, height}} until the block fits, keeping room for pads. */
    private static void safeResize(int height, int width, int[] best, int p, int r, int maxTokens) {
        int cap = maxTokens - V4_LEAD_PAD_RESERVE;
        int budget = cap;
        while (tokens4(best[1], best[0], p, r) > cap) {
            solveResizeRatio(height, width, p, r, budget, best);
            budget--;
        }
    }

    /** {@code solve_resize_ratio}: the largest size whose block fits {@code maxTokens}. */
    private static void solveResizeRatio(int height, int width, int p, int r, int maxTokens, int[] best) {
        double ratio = (double) height / width;
        double maxWf = Math.sqrt((maxTokens - 2) / ratio + 0.25) - 0.5;
        double maxHf = maxWf * ratio;
        if (maxWf < 1.0) {
            int maxW = 1;
            int maxH = (maxTokens - 2) / (maxW + 1);
            if (maxH % 2 == 1) {
                maxH -= 1;
            }
            best[0] = maxW * p * r;
            best[1] = maxH * p * r;
        } else if (maxHf < 2.0) {
            int maxH = 2;
            int maxW = Math.max(((maxTokens - 2) / maxH) - 1, 2);
            best[0] = maxW * p * r;
            best[1] = maxH * p * r;
        } else {
            int maxWi = (int) Math.floor(maxWf);
            int maxHi = (int) Math.floor(maxHf);
            if (maxHi % 2 == 1) {
                maxHi -= 1;
            }
            double beta = Math.min((double) maxWi * p * r / width, (double) maxHi * p * r / height);
            best[0] = (int) Math.floor(width * beta / p) * p;
            best[1] = (int) Math.floor(height * beta / p) * p;
        }
    }

    // ------------------------------------------------------------------ deepseek41v

    private RasterImage prepare41(RasterImage image) {
        int[] best = planGrid41(image.width(), image.height(), params.patchSize(), params.mergeSize(), maxTokens(),
                params.minPixels());
        int bestWidth = best[0];
        int bestHeight = best[1];
        int newW = bestWidth;
        int newH = bestHeight;
        double imRatio = (double) image.width() / image.height();
        double destRatio = (double) bestWidth / bestHeight;
        if (imRatio != destRatio) {
            if (imRatio > destRatio) {
                newH = (int) Math.rint((double) image.height() / image.width() * bestWidth);
            } else {
                newW = (int) Math.rint((double) image.width() / image.height() * bestHeight);
            }
        }
        RasterImage resized = ImageOps.resize(image, new ImageSize(newW, newH), params.resizeAlgo(), PadStyle.NONE,
                params.padColor());
        if (newW == bestWidth && newH == bestHeight) {
            return resized;
        }
        int[] c = params.padColor();
        RasterImage canvas = RasterImage.filled(bestWidth, bestHeight, c[0], c[1], c[2]);
        int offX = (int) Math.rint((bestWidth - newW) * 0.5);
        int offY = (int) Math.rint((bestHeight - newH) * 0.5);
        return ImageOps.composite(canvas, resized, offX, offY);
    }

    /** The V4.1 reading-order block of a {@code bestWidth x bestHeight} image. */
    static int tokens41(int bestHeight, int bestWidth, int p, int r) {
        int h = ((bestHeight / p) + r - 1) / r;
        int w = ((bestWidth / p) + r - 1) / r;
        return h * (w + 1) + 2;
    }

    /** {@code plan_grid}: the {@code {width, height}} the V4.1 image takes. */
    static int[] planGrid41(int width, int height, int p, int r, int maxTokens, long minPixels) {
        long w = width;
        long h = height;
        if (minPixels > 0 && w * h > 0 && w * h < minPixels) {
            double ratio = Math.sqrt((double) minPixels / ((double) w * h));
            w = (long) (w * ratio);
            h = (long) (h * ratio);
        }
        int bestWidth = align((int) w, p);
        int bestHeight = align((int) h, p);
        if (tokens41(bestHeight, bestWidth, p, r) <= maxTokens) {
            return new int[]{bestWidth, bestHeight};
        }
        double ratio = (double) h / w;
        double maxWf = Math.sqrt((maxTokens - 2) / ratio + 0.25) - 0.5;
        double maxHf = maxWf * ratio;
        int cell = p * r;
        if (maxWf < 1.0) {
            bestHeight = (maxTokens - 2) / 2 * cell;
            bestWidth = cell;
        } else if (maxHf < 1.0) {
            bestHeight = cell;
            bestWidth = (maxTokens - 3) * cell;
        } else {
            double beta = Math.min(Math.floor(maxWf) * cell / w, Math.floor(maxHf) * cell / h);
            bestHeight = (int) Math.floor(h * beta / p) * p;
            bestWidth = (int) Math.floor(w * beta / p) * p;
        }
        if (tokens41(bestHeight, bestWidth, p, r) > maxTokens) {
            throw new IllegalStateException("the deepseek41v grid plan " + bestWidth + "x" + bestHeight
                    + " exceeds the token cap " + maxTokens);
        }
        return new int[]{bestWidth, bestHeight};
    }

    private static int align(int x, int p) {
        return (x + p - 1) / p * p;
    }
}

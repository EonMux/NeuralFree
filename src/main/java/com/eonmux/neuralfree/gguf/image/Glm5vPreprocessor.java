package com.eonmux.neuralfree.gguf.image;

/**
 * The GLM-5V preprocessor ({@code mtmd_image_preprocessor_glm5v}, tools/mtmd/mtmd-image.cpp). The canvas is the
 * image's sides rounded up to {@code patch * merge}. Below {@code minPixels} the sides first scale up onto it;
 * above {@code maxPixels} the canvas is the largest one whose height search (the width following
 * proportionally) fits the budget. The image scales to fit the canvas, never up unless it is below the minimum,
 * and sits at its top-left with black on the right and bottom. One tile.
 */
public final class Glm5vPreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /**
     * Creates the preprocessor over the params.
     *
     * @throws IllegalArgumentException when the params carry no pixel limits
     */
    public Glm5vPreprocessor(ImageProcessorParams params) {
        if (params.minPixels() <= 0 || params.maxPixels() <= 0) {
            throw new IllegalArgumentException("the '" + params.projectorName()
                    + "' preprocessor needs positive min and max pixel limits");
        }
        this.params = params;
    }

    @Override
    public PreparedImages prepare(RasterImage image) {
        ImageSize canvasSize = canvasSize(image.size());
        ImageSize content = contentSize(image.size(), canvasSize);
        RasterImage resized = ImageOps.resize(image, content, params.resizeAlgo(), PadStyle.NONE,
                new int[] {0, 0, 0});
        RasterImage canvas = ImageOps.composite(
                RasterImage.filled(canvasSize.width(), canvasSize.height(), 0, 0, 0), resized, 0, 0);
        return new PreparedImages.Builder(params).tile(canvas).build();
    }

    /** The aligned canvas within the pixel budget. */
    ImageSize canvasSize(ImageSize original) {
        long factor = (long) params.patchSize() * params.mergeSize();
        long minPx = params.minPixels();
        long maxPx = params.maxPixels();
        long height = original.height();
        long width = original.width();
        long canvasH = align(height, factor);
        long canvasW = align(width, factor);
        if (canvasH * canvasW < minPx) {
            double scale = Math.sqrt((double) minPx / (double) (height * width));
            canvasH = align(Math.max(1, (long) Math.ceil(height * scale)), factor);
            canvasW = align(Math.max(1, (long) Math.ceil(width * scale)), factor);
        }
        if (canvasH * canvasW > maxPx) {
            // the largest content height whose aligned canvas fits the budget
            long lo = 1;
            long hi = height;
            long bestH = factor;
            long bestW = factor;
            while (lo <= hi) {
                long ch = (lo + hi) / 2;
                long cw = Math.max(1, width * ch / height);
                long ah = align(ch, factor);
                long aw = align(cw, factor);
                if (ah * aw <= maxPx) {
                    bestH = ah;
                    bestW = aw;
                    lo = ch + 1;
                } else {
                    hi = ch - 1;
                }
            }
            canvasH = bestH;
            canvasW = bestW;
        }
        return new ImageSize(toSide(canvasW), toSide(canvasH));
    }

    /** The image's size on the canvas: fitted, never upscaled unless the image is below the minimum. */
    ImageSize contentSize(ImageSize original, ImageSize canvas) {
        long height = original.height();
        long width = original.width();
        double scale = Math.min((double) canvas.height() / height, (double) canvas.width() / width);
        if (height * width >= params.minPixels()) {
            scale = Math.min(1.0, scale);
        }
        long contentH = Math.max(1, Math.min(canvas.height(), (long) Math.floor(height * scale)));
        long contentW = Math.max(1, Math.min(canvas.width(), (long) Math.floor(width * scale)));
        return new ImageSize((int) contentW, (int) contentH);
    }

    private static long align(long v, long factor) {
        return (v + factor - 1) / factor * factor;
    }

    private static int toSide(long side) {
        if (side > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the canvas side " + side + " does not fit an int");
        }
        return (int) side;
    }
}

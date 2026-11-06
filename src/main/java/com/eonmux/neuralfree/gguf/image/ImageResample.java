package com.eonmux.neuralfree.gguf.image;

/**
 * Pillow-compatible separable resampling, a port of {@code img_tool::resize_pillow}
 * (tools/mtmd/mtmd-image.cpp), itself adapted from Pillow's {@code src/libImaging/Resample.c} and shared with the
 * image resize of llama.cpp and vLLM: a horizontal pass, then a vertical pass, each with pre-computed normalized
 * filter coefficients and fixed-point integer arithmetic (22 fractional bits).
 *
 * <p>Output equality with the references holds per pixel: the weights are IEEE doubles reduced to the same
 * fixed-point integers, with Pillow's truncate-toward-zero rounding, and the accumulation clamps to
 * {@code [0, 255]} at the end of each pass. The torchvision variants ({@link ResizeAlgo#int16Weights()}) round the
 * weights to int16 with as many fractional bits as the largest weight allows, then scale them back to the 22-bit
 * precision, which gives torchvision's output in the same passes.</p>
 */
final class ImageResample {

    /** Fixed-point precision: 32 (int) - 8 (uint8 pixel) - 2 (accumulation headroom). */
    private static final int PRECISION_BITS = 32 - 8 - 2;

    private ImageResample() {
    }

    /** Resamples {@code src} to {@code targetWidth} x {@code targetHeight} with the filter of {@code algo}. */
    static RasterImage resample(RasterImage src, int targetWidth, int targetHeight, ResizeAlgo algo) {
        if (targetWidth <= 0 || targetWidth > 65536 || targetHeight <= 0 || targetHeight > 65536) {
            throw new IllegalArgumentException("resize target " + targetWidth + "x" + targetHeight
                    + " is out of range (max 65536)");
        }
        int srcWidth = src.width();
        int srcHeight = src.height();
        if (targetWidth == srcWidth && targetHeight == srcHeight) {
            return src; // no resize needed
        }

        boolean needHorizontal = targetWidth != srcWidth;
        boolean needVertical = targetHeight != srcHeight;

        int[] boundsH = new int[0];
        int[] weightsH = new int[0];
        int[] boundsV = new int[0];
        int[] weightsV = new int[0];
        int ksizeH = 0;
        int ksizeV = 0;
        if (needHorizontal) {
            ksizeH = precomputeWeights(srcWidth, targetWidth, algo, boundsH = new int[targetWidth * 2],
                    weightsH = new int[targetWidth * ceilKsize(srcWidth, targetWidth, algo)]);
        }
        if (needVertical) {
            ksizeV = precomputeWeights(srcHeight, targetHeight, algo, boundsV = new int[targetHeight * 2],
                    weightsV = new int[targetHeight * ceilKsize(srcHeight, targetHeight, algo)]);
        }

        byte[] out;
        byte[] in = src.rgbArray();
        if (needHorizontal && needVertical) {
            byte[] temp = resampleHorizontal(in, srcWidth, srcHeight, targetWidth, ksizeH, boundsH, weightsH);
            out = resampleVertical(temp, targetWidth, srcHeight, targetHeight, ksizeV, boundsV, weightsV);
        } else if (needHorizontal) {
            out = resampleHorizontal(in, srcWidth, srcHeight, targetWidth, ksizeH, boundsH, weightsH);
        } else {
            out = resampleVertical(in, srcWidth, srcHeight, targetHeight, ksizeV, boundsV, weightsV);
        }
        return RasterImage.wrap(targetWidth, targetHeight, out);
    }

    /** Upper bound of the kernel size, so the weight arrays can be sized once. */
    private static int ceilKsize(int inSize, int outSize, ResizeAlgo algo) {
        double scale = Math.max((double) inSize / outSize, 1.0);
        return (int) Math.ceil(algoSupport(algo.kernel()) * scale) * 2 + 1;
    }

    private static double algoSupport(ResizeAlgo algo) {
        switch (algo) {
            case BILINEAR: return 1.0;
            case BICUBIC: return 2.0;
            case LANCZOS: return 3.0;
            default: throw new IllegalArgumentException("unsupported resize algo: " + algo);
        }
    }

    /** The filter weight for a distance {@code x} from the pixel center (mtmd-image.cpp resample_filter). */
    private static double filter(ResizeAlgo algo, double x) {
        if (algo == ResizeAlgo.LANCZOS) {
            if (x >= -3.0 && x < 3.0) {
                return sinc(x) * sinc(x / 3.0);
            }
            return 0.0;
        }
        if (x < 0.0) {
            x = -x;
        }
        if (algo == ResizeAlgo.BILINEAR) {
            return x < 1.0 ? 1.0 - x : 0.0;
        }
        // Pillow (and PyTorch with antialias) bicubic with a = -0.5
        final double a = -0.5;
        if (x < 1.0) {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        }
        if (x < 2.0) {
            return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        }
        return 0.0;
    }

    private static double sinc(double v) {
        if (v == 0.0) {
            return 1.0;
        }
        double piV = v * Math.PI;
        return Math.sin(piV) / piV;
    }

    private static int clip8(int val) {
        if (val < 0) return 0;
        if (val > 255) return 255;
        return val;
    }

    /**
     * Precomputes the normalized fixed-point filter coefficients for one dimension into {@code bounds}
     * ({@code [xx*2] = first input pixel, [xx*2+1] = count}) and {@code weights}
     * ({@code [xx*ksize + x]}), and returns the kernel size; the port of the lambda of the same name in
     * mtmd-image.cpp.
     */
    private static int precomputeWeights(int inSize, int outSize, ResizeAlgo algo, int[] bounds, int[] weights) {
        if (inSize <= 0 || outSize <= 0) {
            throw new IllegalArgumentException("resample sizes must be positive, got " + inSize + " -> " + outSize);
        }
        ResizeAlgo kernel = algo.kernel();
        double support = algoSupport(kernel);
        // For upsampling (scale < 1) the filter keeps its sharpness; for downsampling it widens to avoid aliasing.
        double filterscale = Math.max((double) inSize / outSize, 1.0);
        support *= filterscale;
        int ksize = (int) Math.ceil(support) * 2 + 1;

        double[] preWeights = new double[outSize * ksize];
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * ((double) inSize / outSize);
            double ww = 0.0;
            double ss = 1.0 / filterscale;

            int xmin = (int) (center - support + 0.5);
            if (xmin < 0) {
                xmin = 0;
            }
            int xmax = (int) (center + support + 0.5);
            if (xmax > inSize) {
                xmax = inSize;
            }
            xmax -= xmin;

            for (int x = 0; x < xmax; x++) {
                double w = filter(kernel, (x + xmin - center + 0.5) * ss);
                preWeights[xx * ksize + x] = w;
                ww += w;
            }
            for (int x = 0; x < xmax; x++) {
                if (ww != 0.0) {
                    preWeights[xx * ksize + x] /= ww;
                }
            }
            for (int x = xmax; x < ksize; x++) {
                preWeights[xx * ksize + x] = 0.0;
            }

            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }

        int precision = algo.int16Weights() ? int16Precision(preWeights) : PRECISION_BITS;
        // Pillow adds +/- 0.5 and truncates toward zero; std::round would round twice.
        double fxpScale = Math.pow(2.0, precision);
        int rescale = 1 << (PRECISION_BITS - precision);
        for (int i = 0; i < outSize * ksize; i++) {
            double w = preWeights[i];
            long rounded = (long) (w * fxpScale + (w < 0 ? -0.5 : 0.5));
            weights[i] = (int) rounded * rescale;
        }
        return ksize;
    }

    /** The fractional bits of torchvision's int16 weights: the most that keep the largest weight below 2^15. */
    private static int int16Precision(double[] preWeights) {
        double weightMax = Double.NEGATIVE_INFINITY;
        for (double w : preWeights) {
            weightMax = Math.max(weightMax, w);
        }
        int precision = 0;
        while (precision < PRECISION_BITS) {
            if ((int) (0.5 + weightMax * (1 << (precision + 1))) >= (1 << 15)) {
                break;
            }
            precision++;
        }
        return precision;
    }

    /** The horizontal pass: resizes width from {@code inNx} to {@code outNx}, keeping every row. */
    private static byte[] resampleHorizontal(byte[] src, int inNx, int inNy, int outNx,
                                             int ksize, int[] bounds, int[] weights) {
        byte[] out = new byte[outNx * inNy * 3];
        int bias = 1 << (PRECISION_BITS - 1);
        for (int yy = 0; yy < inNy; yy++) {
            int srcRow = yy * inNx * 3;
            int dstRow = yy * outNx * 3;
            for (int xx = 0; xx < outNx; xx++) {
                int xmin = bounds[xx * 2];
                int xcnt = bounds[xx * 2 + 1];
                int k = xx * ksize;
                int p = srcRow + xmin * 3;

                int ss0 = bias;
                int ss1 = bias;
                int ss2 = bias;
                for (int x = 0; x < xcnt; x++) {
                    int w = weights[k + x];
                    ss0 += (src[p] & 0xFF) * w;
                    ss1 += (src[p + 1] & 0xFF) * w;
                    ss2 += (src[p + 2] & 0xFF) * w;
                    p += 3;
                }
                out[dstRow + xx * 3] = (byte) clip8(ss0 >> PRECISION_BITS);
                out[dstRow + xx * 3 + 1] = (byte) clip8(ss1 >> PRECISION_BITS);
                out[dstRow + xx * 3 + 2] = (byte) clip8(ss2 >> PRECISION_BITS);
            }
        }
        return out;
    }

    /** The vertical pass: resizes height from {@code inNy} to {@code outNy}, keeping every column. */
    private static byte[] resampleVertical(byte[] src, int inNx, int inNy, int outNy,
                                           int ksize, int[] bounds, int[] weights) {
        int rowElems = inNx * 3;
        byte[] out = new byte[rowElems * outNy];
        int[] acc = new int[rowElems];
        int bias = 1 << (PRECISION_BITS - 1);
        for (int yy = 0; yy < outNy; yy++) {
            int ymin = bounds[yy * 2];
            int ycnt = bounds[yy * 2 + 1];
            int k = yy * ksize;

            for (int i = 0; i < rowElems; i++) {
                acc[i] = bias;
            }
            for (int y = 0; y < ycnt; y++) {
                int srcRow = (ymin + y) * rowElems;
                int w = weights[k + y];
                for (int i = 0; i < rowElems; i++) {
                    acc[i] += (src[srcRow + i] & 0xFF) * w;
                }
            }
            int dstRow = yy * rowElems;
            for (int i = 0; i < rowElems; i++) {
                out[dstRow + i] = (byte) clip8(acc[i] >> PRECISION_BITS);
            }
        }
        return out;
    }
}

package com.eonmux.neuralfree.tensor;

/**
 * The FLOAT64 host path of the convolution, pooling and batch-norm forwards ({@link CpuFloat64} has the contract):
 * the loops, output sizes and window rules of {@code CpuConvPool} and {@code CpuNorm.batchNorm}, with every value and
 * accumulator a Java {@code double}. The callers validated every argument as for the FLOAT32 path.
 */
final class CpuFloat64Conv {

    private CpuFloat64Conv() {
    }

    // ------------------------------------------------------------------ convolutions

    /** {@code CpuConvPool.conv2d}: input {@code [B, C, H, W]}, weight {@code [OC, C/groups, KH, KW]}. */
    static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int[] kernel, int[] stride, int[] padding,
                         int[] dilation, int groups) {
        Shape s = input.shape();
        int batch = s.dim(0);
        int inC = s.dim(1);
        int inH = s.dim(2);
        int inW = s.dim(3);
        int outC = weight.shape().dim(0);
        int outH = (inH + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1;
        int outW = (inW + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1;
        double[] x = input.toDoubleArray();
        double[] w = weight.toDoubleArray();
        double[] b = bias != null ? bias.toDoubleArray() : null;
        Shape outShape = Shape.of(batch, outC, outH, outW);
        double[] out = new double[CpuShapeOps.hostLength(outShape.numElements())];
        int cg = inC / groups;
        int ocg = outC / groups;
        for (int n = 0; n < batch; n++) {
            for (int oc = 0; oc < outC; oc++) {
                int g = oc / ocg;
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = b != null ? b[oc] : 0.0;
                        for (int ic = 0; ic < cg; ic++) {
                            int plane = (n * inC + g * cg + ic) * inH;
                            int taps = (oc * cg + ic) * kernel[0];
                            for (int kh = 0; kh < kernel[0]; kh++) {
                                int ih = oh * stride[0] + kh * dilation[0] - padding[0];
                                if (ih < 0 || ih >= inH) {
                                    continue;
                                }
                                for (int kw = 0; kw < kernel[1]; kw++) {
                                    int iw = ow * stride[1] + kw * dilation[1] - padding[1];
                                    if (iw >= 0 && iw < inW) {
                                        sum += x[(plane + ih) * inW + iw] * w[(taps + kh) * kernel[1] + kw];
                                    }
                                }
                            }
                        }
                        out[((n * outC + oc) * outH + oh) * outW + ow] = sum;
                    }
                }
            }
        }
        return CpuFloat64.doubles(input.backend(), outShape, out);
    }

    /** {@code CpuConvPool.conv1d}: input {@code [B, C, L]}, weight {@code [OC, C/groups, K]}. */
    static Tensor conv1d(Tensor input, Tensor weight, Tensor bias, int kernel, int stride, int padding,
                         int dilation, int groups) {
        int batch = input.shape().dim(0);
        int inC = input.shape().dim(1);
        int inL = input.shape().dim(2);
        try (Tensor x4 = input.reshape(batch, inC, 1, inL);
             Tensor w4 = weight.reshape(weight.shape().dim(0), weight.shape().dim(1), 1, kernel);
             Tensor out4 = conv2d(x4, w4, bias, new int[]{1, kernel}, new int[]{1, stride}, new int[]{0, padding},
                     new int[]{1, dilation}, groups)) {
            Shape s = out4.shape();
            return CpuFloat64.doubles(input.backend(), Shape.of(s.dim(0), s.dim(1), s.dim(3)),
                    out4.toDoubleArray());
        }
    }

    /**
     * {@code CpuConvPool.convTranspose2d} in gather form: input {@code [B, C, H, W]}, weight
     * {@code [C, OC/groups, KH, KW]}; output {@code (oh, ow)} receives input {@code (ih, iw)} through tap
     * {@code (kh, kw)} when {@code oh + padding - kh * dilation == ih * stride} (and likewise for the width).
     */
    static Tensor convTranspose2d(Tensor input, Tensor weight, Tensor bias, int[] kernel, int[] stride,
                                  int[] padding, int[] outputPadding, int[] dilation, int groups) {
        Shape s = input.shape();
        int batch = s.dim(0);
        int inC = s.dim(1);
        int inH = s.dim(2);
        int inW = s.dim(3);
        int outC = weight.shape().dim(1) * groups;
        int outH = (inH - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + outputPadding[0] + 1;
        int outW = (inW - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + outputPadding[1] + 1;
        double[] x = input.toDoubleArray();
        double[] w = weight.toDoubleArray();
        double[] b = bias != null ? bias.toDoubleArray() : null;
        Shape outShape = Shape.of(batch, outC, outH, outW);
        double[] out = new double[CpuShapeOps.hostLength(outShape.numElements())];
        int cg = inC / groups;
        int ocg = outC / groups;
        for (int n = 0; n < batch; n++) {
            for (int oc = 0; oc < outC; oc++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = b != null ? b[oc] : 0.0;
                        for (int ic = 0; ic < cg; ic++) {
                            int c = (oc / ocg) * cg + ic;
                            sum += transposedTaps(x, w, n, c, inC, inH, inW, (c * ocg + oc % ocg) * kernel[0], oh,
                                    ow, kernel, stride, padding, dilation);
                        }
                        out[((n * outC + oc) * outH + oh) * outW + ow] = sum;
                    }
                }
            }
        }
        return CpuFloat64.doubles(input.backend(), outShape, out);
    }

    /** The sum over the taps of input channel {@code c} that reach output {@code (oh, ow)} of a transposed conv. */
    private static double transposedTaps(double[] x, double[] w, int n, int c, int inC, int inH, int inW,
                                         int taps, int oh, int ow, int[] kernel, int[] stride, int[] padding,
                                         int[] dilation) {
        double sum = 0.0;
        for (int kh = 0; kh < kernel[0]; kh++) {
            int th = oh + padding[0] - kh * dilation[0];
            if (th < 0 || th % stride[0] != 0 || th / stride[0] >= inH) {
                continue;
            }
            int ih = th / stride[0];
            for (int kw = 0; kw < kernel[1]; kw++) {
                int tw = ow + padding[1] - kw * dilation[1];
                if (tw < 0 || tw % stride[1] != 0 || tw / stride[1] >= inW) {
                    continue;
                }
                sum += x[((n * inC + c) * inH + ih) * inW + tw / stride[1]] * w[(taps + kh) * kernel[1] + kw];
            }
        }
        return sum;
    }

    // ------------------------------------------------------------------ pooling

    /** {@code CpuConvPool.maxPool2d}: {@code Math.max} over the in-bounds taps from {@code -inf}. */
    static Tensor maxPool2d(Tensor input, int[] kernel, int[] stride, int[] padding, int[] dilation,
                            boolean ceilMode) {
        Shape s = input.shape();
        int planes = s.dim(0) * s.dim(1);
        int inH = s.dim(2);
        int inW = s.dim(3);
        int outH = ConvOps.poolOutputSize(inH, kernel[0], stride[0], padding[0], dilation[0], ceilMode);
        int outW = ConvOps.poolOutputSize(inW, kernel[1], stride[1], padding[1], dilation[1], ceilMode);
        double[] x = input.toDoubleArray();
        Shape outShape = Shape.of(s.dim(0), s.dim(1), outH, outW);
        double[] out = new double[CpuShapeOps.hostLength(outShape.numElements())];
        for (int p = 0; p < planes; p++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int kh = 0; kh < kernel[0]; kh++) {
                        int ih = oh * stride[0] + kh * dilation[0] - padding[0];
                        for (int kw = 0; kw < kernel[1]; kw++) {
                            int iw = ow * stride[1] + kw * dilation[1] - padding[1];
                            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                max = Math.max(max, x[(p * inH + ih) * inW + iw]);
                            }
                        }
                    }
                    out[(p * outH + oh) * outW + ow] = max;
                }
            }
        }
        return CpuFloat64.doubles(input.backend(), outShape, out);
    }

    /** {@code CpuConvPool.avgPool2d}: PyTorch's {@code avg_pool2d} windows and divisors. */
    static Tensor avgPool2d(Tensor input, int[] kernel, int[] stride, int[] padding, boolean ceilMode,
                            boolean countIncludePad) {
        Shape s = input.shape();
        int planes = s.dim(0) * s.dim(1);
        int inH = s.dim(2);
        int inW = s.dim(3);
        int outH = ConvOps.poolOutputSize(inH, kernel[0], stride[0], padding[0], 1, ceilMode);
        int outW = ConvOps.poolOutputSize(inW, kernel[1], stride[1], padding[1], 1, ceilMode);
        double[] x = input.toDoubleArray();
        Shape outShape = Shape.of(s.dim(0), s.dim(1), outH, outW);
        double[] out = new double[CpuShapeOps.hostLength(outShape.numElements())];
        for (int p = 0; p < planes; p++) {
            for (int oh = 0; oh < outH; oh++) {
                int hs = oh * stride[0] - padding[0];
                int he = Math.min(hs + kernel[0], inH + padding[0]);
                for (int ow = 0; ow < outW; ow++) {
                    int ws = ow * stride[1] - padding[1];
                    int we = Math.min(ws + kernel[1], inW + padding[1]);
                    int h0 = Math.max(hs, 0);
                    int h1 = Math.min(he, inH);
                    int w0 = Math.max(ws, 0);
                    int w1 = Math.min(we, inW);
                    if (h0 >= h1 || w0 >= w1) {
                        continue;
                    }
                    double divisor = countIncludePad ? (he - hs) * (we - ws) : (h1 - h0) * (w1 - w0);
                    out[(p * outH + oh) * outW + ow] = windowSum(x, p, inH, inW, h0, h1, w0, w1) / divisor;
                }
            }
        }
        return CpuFloat64.doubles(input.backend(), outShape, out);
    }

    /** {@code CpuConvPool.adaptiveAvgPool2d}: the windows of {@link ConvOps#adaptiveStart} and {@code adaptiveEnd}. */
    static Tensor adaptiveAvgPool2d(Tensor input, int outH, int outW) {
        Shape s = input.shape();
        int planes = s.dim(0) * s.dim(1);
        int inH = s.dim(2);
        int inW = s.dim(3);
        double[] x = input.toDoubleArray();
        Shape outShape = Shape.of(s.dim(0), s.dim(1), outH, outW);
        double[] out = new double[CpuShapeOps.hostLength(outShape.numElements())];
        for (int p = 0; p < planes; p++) {
            for (int oh = 0; oh < outH; oh++) {
                int h0 = ConvOps.adaptiveStart(oh, inH, outH);
                int h1 = ConvOps.adaptiveEnd(oh, inH, outH);
                for (int ow = 0; ow < outW; ow++) {
                    int w0 = ConvOps.adaptiveStart(ow, inW, outW);
                    int w1 = ConvOps.adaptiveEnd(ow, inW, outW);
                    long count = (long) (h1 - h0) * (w1 - w0);
                    out[(p * outH + oh) * outW + ow] = count > 0 ? windowSum(x, p, inH, inW, h0, h1, w0, w1) / count
                            : 0.0;
                }
            }
        }
        return CpuFloat64.doubles(input.backend(), outShape, out);
    }

    /** The row-major sum of plane {@code p}'s elements in {@code [h0, h1) x [w0, w1)}. */
    private static double windowSum(double[] x, int p, int inH, int inW, int h0, int h1, int w0, int w1) {
        double sum = 0.0;
        for (int ih = h0; ih < h1; ih++) {
            for (int iw = w0; iw < w1; iw++) {
                sum += x[(p * inH + ih) * inW + iw];
            }
        }
        return sum;
    }

    // ------------------------------------------------------------------ batch norm

    /**
     * {@code CpuNorm.batchNorm} for an input {@code [B, C, *]}: in training mode the batch mean and biased variance
     * normalize the input and the running statistics, when given, are updated as {@code running_mean = (1 -
     * momentum) * running_mean + momentum * mean} and {@code running_var = (1 - momentum) * running_var + momentum *
     * var * n / (n - 1)}, written back with {@link GpuOptimizerOps#storeDoubles}; in evaluation mode the running
     * statistics normalize it. The caller checked the batch-size rule.
     */
    static Tensor batchNorm(Tensor input, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                            boolean training, float momentum, float epsilon) {
        Shape s = input.shape();
        int batch = s.dim(0);
        int channels = s.dim(1);
        int spatial = 1;
        for (int i = 2; i < s.rank(); i++) {
            spatial *= s.dim(i);
        }
        double[] x = input.toDoubleArray();
        double[] out = new double[x.length];
        double[] runMean = runningMean == null ? null : runningMean.toDoubleArray();
        double[] runVar = runningVar == null ? null : runningVar.toDoubleArray();
        double[] g = gamma != null ? gamma.toDoubleArray() : null;
        double[] b = gamma != null ? beta.toDoubleArray() : null;
        long count = (long) batch * spatial;
        for (int c = 0; c < channels; c++) {
            double mean;
            double variance;
            if (training) {
                mean = channelSum(x, c, batch, channels, spatial, 0.0, false) / count;
                variance = channelSum(x, c, batch, channels, spatial, mean, true) / count;
                if (runMean != null) {
                    runMean[c] = (1 - (double) momentum) * runMean[c] + momentum * mean;
                    runVar[c] = (1 - (double) momentum) * runVar[c] + momentum * variance * count / (count - 1);
                }
            } else {
                mean = runMean[c];
                variance = runVar[c];
            }
            double std = Math.sqrt(variance + epsilon);
            for (int n = 0; n < batch; n++) {
                int base = (n * channels + c) * spatial;
                for (int i = 0; i < spatial; i++) {
                    double normalized = (x[base + i] - mean) / std;
                    out[base + i] = g != null ? g[c] * normalized + b[c] : normalized;
                }
            }
        }
        if (training && runMean != null) {
            GpuOptimizerOps.storeDoubles(runningMean, runMean);
            GpuOptimizerOps.storeDoubles(runningVar, runVar);
        }
        return CpuFloat64.doubles(input.backend(), s, out);
    }

    /** The sum of channel {@code c}'s values, or of their squared deviations from {@code mean} when {@code squared}. */
    private static double channelSum(double[] x, int c, int batch, int channels, int spatial, double mean,
                                     boolean squared) {
        double sum = 0.0;
        for (int n = 0; n < batch; n++) {
            int base = (n * channels + c) * spatial;
            for (int i = 0; i < spatial; i++) {
                double v = x[base + i] - mean;
                sum += squared ? v * v : x[base + i];
            }
        }
        return sum;
    }
}

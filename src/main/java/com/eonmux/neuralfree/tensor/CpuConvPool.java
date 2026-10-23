package com.eonmux.neuralfree.tensor;

/**
 * CPU reference of {@link ConvOps}: the host loops of {@code nn/layers/Conv2d}, {@code Conv1d},
 * {@code ConvTranspose2d}, {@code MaxPool2d}, {@code AvgPool2d} and {@code AdaptiveAvgPool2d}, in the modules' loop
 * order and Java float arithmetic, and the backward passes of {@link GpuConvGradOps}, in the order its kernels
 * document. Operands are read in row-major order ({@link Tensor#toFloatArray()}, so they must be FLOAT32) and each
 * result is uploaded as a new contiguous FLOAT32 tensor on the operands' backend with {@code requiresGrad} false.
 *
 * <p>Callers validate every argument first: {@link ConvOps} offers each op to {@link GpuConvOps} or
 * {@link GpuConvGradOps}, which check the shapes, parameters and backends and that every output size lies in
 * {@code [0, 2^31-1]} before they decline. The convolution output sizes below use the layers' {@code int} formulas;
 * two's-complement {@code int} arithmetic is exact modulo {@code 2^32}, so they equal the validated sizes. The
 * pooling output sizes come from {@link ConvOps#poolOutputSize} (PyTorch's rule), as the dispatchers compute them,
 * and the adaptive windows from {@link ConvOps#adaptiveStart} and {@link ConvOps#adaptiveEnd}. The backward
 * references take the output sizes from the validated gradient and output shapes.</p>
 */
final class CpuConvPool {

    private CpuConvPool() {
    }

    // ------------------------------------------------------------------ convolutions

    /** {@code nn/layers/Conv2d} forward: input {@code [B, C, H, W]}, weight {@code [OC, C/groups, KH, KW]}. */
    static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int[] kernelSize, int[] stride, int[] padding,
                         int[] dilation, int groups) {
        Shape inputShape = input.shape();
        int batch = inputShape.dim(0);
        int inChannels = inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outChannels = weight.shape().dim(0);
        boolean useBias = bias != null;

        int outH = (inH + 2 * padding[0] - dilation[0] * (kernelSize[0] - 1) - 1) / stride[0] + 1;
        int outW = (inW + 2 * padding[1] - dilation[1] * (kernelSize[1] - 1) - 1) / stride[1] + 1;

        float[] inputData = input.toFloatArray();
        float[] weightData = weight.toFloatArray();
        float[] biasData = useBias ? bias.toFloatArray() : null;

        Shape outputShape = Shape.of(batch, outChannels, outH, outW);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                int groupIdx = oc / outChannelsPerGroup;

                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = useBias ? biasData[oc] : 0.0f;

                        for (int ic = 0; ic < inChannelsPerGroup; ic++) {
                            int inputChannel = groupIdx * inChannelsPerGroup + ic;

                            for (int kh = 0; kh < kernelSize[0]; kh++) {
                                for (int kw = 0; kw < kernelSize[1]; kw++) {
                                    int ih = oh * stride[0] + kh * dilation[0] - padding[0];
                                    int iw = ow * stride[1] + kw * dilation[1] - padding[1];

                                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                        int inputIdx = b * inChannels * inH * inW +
                                                       inputChannel * inH * inW +
                                                       ih * inW + iw;

                                        int weightIdx = oc * inChannelsPerGroup * kernelSize[0] * kernelSize[1] +
                                                        ic * kernelSize[0] * kernelSize[1] +
                                                        kh * kernelSize[1] + kw;

                                        sum += inputData[inputIdx] * weightData[weightIdx];
                                    }
                                }
                            }
                        }

                        int outputIdx = b * outChannels * outH * outW +
                                        oc * outH * outW +
                                        oh * outW + ow;
                        outputData[outputIdx] = sum;
                    }
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    /** {@code nn/layers/Conv1d} forward: input {@code [B, C, L]}, weight {@code [OC, C/groups, K]}. */
    static Tensor conv1d(Tensor input, Tensor weight, Tensor bias, int kernelSize, int stride, int padding,
                         int dilation, int groups) {
        Shape inputShape = input.shape();
        int batch = inputShape.dim(0);
        int inChannels = inputShape.dim(1);
        int inL = inputShape.dim(2);
        int outChannels = weight.shape().dim(0);
        boolean useBias = bias != null;

        int outL = (inL + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1;

        float[] inputData = input.toFloatArray();
        float[] weightData = weight.toFloatArray();
        float[] biasData = useBias ? bias.toFloatArray() : null;

        Shape outputShape = Shape.of(batch, outChannels, outL);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                int groupIdx = oc / outChannelsPerGroup;

                for (int ol = 0; ol < outL; ol++) {
                    float sum = useBias ? biasData[oc] : 0.0f;

                    for (int ic = 0; ic < inChannelsPerGroup; ic++) {
                        int inputChannel = groupIdx * inChannelsPerGroup + ic;

                        for (int k = 0; k < kernelSize; k++) {
                            int il = ol * stride + k * dilation - padding;

                            if (il >= 0 && il < inL) {
                                int inputIdx = (b * inChannels + inputChannel) * inL + il;
                                int weightIdx = (oc * inChannelsPerGroup + ic) * kernelSize + k;
                                sum += inputData[inputIdx] * weightData[weightIdx];
                            }
                        }
                    }

                    outputData[(b * outChannels + oc) * outL + ol] = sum;
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    /**
     * {@code nn/layers/ConvTranspose2d} forward in gather form: input {@code [B, C, H, W]}, weight
     * {@code [C, OC/groups, KH, KW]}; output {@code (oh, ow)} receives input {@code (ih, iw)} through tap
     * {@code (kh, kw)} when {@code oh + padding - kh * dilation == ih * stride} (and likewise for the width).
     */
    static Tensor convTranspose2d(Tensor input, Tensor weight, Tensor bias, int[] kernelSize, int[] stride,
                                  int[] padding, int[] outputPadding, int[] dilation, int groups) {
        Shape inputShape = input.shape();
        int batch = inputShape.dim(0);
        int inChannels = inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outChannels = weight.shape().dim(1) * groups;
        boolean useBias = bias != null;

        int outH = (inH - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernelSize[0] - 1) + outputPadding[0] + 1;
        int outW = (inW - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernelSize[1] - 1) + outputPadding[1] + 1;

        float[] inputData = input.toFloatArray();
        float[] weightData = weight.toFloatArray();
        float[] biasData = useBias ? bias.toFloatArray() : null;

        Shape outputShape = Shape.of(batch, outChannels, outH, outW);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        int inChannelsPerGroup = inChannels / groups;
        int outChannelsPerGroup = outChannels / groups;

        for (int b = 0; b < batch; b++) {
            for (int oc = 0; oc < outChannels; oc++) {
                int groupIdx = oc / outChannelsPerGroup;
                int ocInGroup = oc % outChannelsPerGroup;

                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = useBias ? biasData[oc] : 0.0f;

                        for (int ic = 0; ic < inChannelsPerGroup; ic++) {
                            int inputChannel = groupIdx * inChannelsPerGroup + ic;

                            for (int kh = 0; kh < kernelSize[0]; kh++) {
                                int th = oh + padding[0] - kh * dilation[0];
                                if (th < 0 || th % stride[0] != 0 || th / stride[0] >= inH) {
                                    continue;
                                }
                                int ih = th / stride[0];

                                for (int kw = 0; kw < kernelSize[1]; kw++) {
                                    int tw = ow + padding[1] - kw * dilation[1];
                                    if (tw < 0 || tw % stride[1] != 0 || tw / stride[1] >= inW) {
                                        continue;
                                    }
                                    int iw = tw / stride[1];

                                    int inputIdx = ((b * inChannels + inputChannel) * inH + ih) * inW + iw;
                                    int weightIdx = ((inputChannel * outChannelsPerGroup + ocInGroup) * kernelSize[0]
                                                     + kh) * kernelSize[1] + kw;
                                    sum += inputData[inputIdx] * weightData[weightIdx];
                                }
                            }
                        }

                        outputData[((b * outChannels + oc) * outH + oh) * outW + ow] = sum;
                    }
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    // ------------------------------------------------------------------ pooling

    /**
     * {@code nn/layers/MaxPool2d} forward: {@code Math.max} over each window's in-bounds taps in {@code (kh, kw)}
     * order from {@code -inf}, so a window without input elements yields {@code -inf}.
     */
    static Tensor maxPool2d(Tensor input, int[] kernelSize, int[] stride, int[] padding, int[] dilation,
                            boolean ceilMode) {
        Shape inputShape = input.shape();
        int batch = inputShape.dim(0);
        int channels = inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outH = ConvOps.poolOutputSize(inH, kernelSize[0], stride[0], padding[0], dilation[0], ceilMode);
        int outW = ConvOps.poolOutputSize(inW, kernelSize[1], stride[1], padding[1], dilation[1], ceilMode);

        float[] inputData = input.toFloatArray();
        Shape outputShape = Shape.of(batch, channels, outH, outW);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float maxVal = Float.NEGATIVE_INFINITY;

                        for (int kh = 0; kh < kernelSize[0]; kh++) {
                            for (int kw = 0; kw < kernelSize[1]; kw++) {
                                int ih = oh * stride[0] + kh * dilation[0] - padding[0];
                                int iw = ow * stride[1] + kw * dilation[1] - padding[1];

                                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                                    int inputIdx = b * channels * inH * inW + c * inH * inW + ih * inW + iw;
                                    maxVal = Math.max(maxVal, inputData[inputIdx]);
                                }
                            }
                        }

                        int outputIdx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                        outputData[outputIdx] = maxVal;
                    }
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    /**
     * {@code nn/layers/AvgPool2d} forward (PyTorch {@code avg_pool2d}): window {@code (oh, ow)} spans rows
     * {@code [hs, he)} with {@code hs = oh*SH - PH} and {@code he = min(hs + KH, H + PH)} (columns likewise); its
     * in-bounds elements are summed in row-major order from 0 and divided once by {@link #avgDivisor}; a window
     * without input elements yields 0.
     */
    static Tensor avgPool2d(Tensor input, int[] kernelSize, int[] stride, int[] padding, boolean ceilMode,
                            boolean countIncludePad) {
        Shape inputShape = input.shape();
        int planes = inputShape.dim(0) * inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outH = ConvOps.poolOutputSize(inH, kernelSize[0], stride[0], padding[0], 1, ceilMode);
        int outW = ConvOps.poolOutputSize(inW, kernelSize[1], stride[1], padding[1], 1, ceilMode);

        float[] inputData = input.toFloatArray();
        Shape outputShape = Shape.of(inputShape.dim(0), inputShape.dim(1), outH, outW);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        for (int bc = 0; bc < planes; bc++) {
            int plane = bc * inH * inW;
            for (int oh = 0; oh < outH; oh++) {
                int hs = oh * stride[0] - padding[0];
                int he = Math.min(hs + kernelSize[0], inH + padding[0]);
                int h0 = Math.max(hs, 0);
                int h1 = Math.min(he, inH);
                for (int ow = 0; ow < outW; ow++) {
                    int ws = ow * stride[1] - padding[1];
                    int we = Math.min(ws + kernelSize[1], inW + padding[1]);
                    int w0 = Math.max(ws, 0);
                    int w1 = Math.min(we, inW);
                    int outputIdx = (bc * outH + oh) * outW + ow;
                    if (h0 >= h1 || w0 >= w1) {
                        outputData[outputIdx] = 0.0f;
                        continue;
                    }
                    float sum = 0.0f;
                    for (int ih = h0; ih < h1; ih++) {
                        for (int iw = w0; iw < w1; iw++) {
                            sum += inputData[plane + ih * inW + iw];
                        }
                    }
                    outputData[outputIdx] = sum / avgDivisor(hs, he, ws, we, h0, h1, w0, w1, countIncludePad);
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    /**
     * The divisor of an average-pool window with padded-extent bounds {@code [hs, he) x [ws, we)} and in-bounds
     * bounds {@code [h0, h1) x [w0, w1)}: {@code (he - hs)*(we - ws)} when {@code countIncludePad}, otherwise
     * {@code (h1 - h0)*(w1 - w0)}, as a float (PyTorch {@code divide_factor}).
     */
    private static float avgDivisor(int hs, int he, int ws, int we, int h0, int h1, int w0, int w1,
                                    boolean countIncludePad) {
        return countIncludePad ? (float) ((he - hs) * (we - ws)) : (float) ((h1 - h0) * (w1 - w0));
    }

    /**
     * {@code nn/layers/AdaptiveAvgPool2d} forward (PyTorch {@code adaptive_avg_pool2d}): output row {@code oh}
     * averages input rows {@code [floor(oh*H/outH), ceil((oh+1)*H/outH))} (columns likewise), summed in row-major
     * order from 0 and divided once by the window's element count; a window without input elements yields 0.
     */
    static Tensor adaptiveAvgPool2d(Tensor input, int outH, int outW) {
        Shape inputShape = input.shape();
        int batch = inputShape.dim(0);
        int channels = inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);

        float[] inputData = input.toFloatArray();
        Shape outputShape = Shape.of(batch, channels, outH, outW);
        float[] outputData = new float[CpuShapeOps.hostLength(outputShape.numElements())];

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    int hStart = ConvOps.adaptiveStart(oh, inH, outH);
                    int hEnd = ConvOps.adaptiveEnd(oh, inH, outH);

                    for (int ow = 0; ow < outW; ow++) {
                        int wStart = ConvOps.adaptiveStart(ow, inW, outW);
                        int wEnd = ConvOps.adaptiveEnd(ow, inW, outW);

                        float sum = 0.0f;
                        int count = 0;

                        for (int ih = hStart; ih < hEnd; ih++) {
                            for (int iw = wStart; iw < wEnd; iw++) {
                                int inputIdx = b * channels * inH * inW + c * inH * inW + ih * inW + iw;
                                sum += inputData[inputIdx];
                                count++;
                            }
                        }

                        int outputIdx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                        outputData[outputIdx] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }

        return CpuShapeOps.floats(input.backend(), outputShape, outputData, false);
    }

    // ------------------------------------------------------------------ convolution weight gradients

    /**
     * Weight gradient of {@link #conv2d}: {@code gw[oc, ic, kh, kw]} sums {@code gy[b, oc, oh, ow] * x[b, cin, ih, iw]}
     * over the in-bounds taps in ascending {@code (b, oh, ow)} order, {@code cin = g*Cg + ic}.
     */
    static Tensor conv2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                   int[] dilation, int groups) {
        Shape xs = x.shape();
        Shape gs = gy.shape();
        float[] gw = weightGrad(x.toFloatArray(), xs.dim(1), xs.dim(2), xs.dim(3),
                gy.toFloatArray(), gs.dim(1), gs.dim(2), gs.dim(3),
                xs.dim(0), weightShape, stride, padding, dilation, groups);
        return CpuShapeOps.floats(x.backend(), weightShape, gw, false);
    }

    /**
     * Weight gradient of {@link #convTranspose2d}: {@code gw[cin, ocg, kh, kw]} sums
     * {@code x[b, cin, ih, iw] * gy[b, g*OCg + ocg, oh, ow]} with {@code oh = ih*SH + kh*DH - PH} (and its W twin)
     * inside the output, in ascending {@code (b, ih, iw)} order: {@link #conv2dWeightGrad}'s sum with the forward
     * input in the role of the gradient and the output gradient in the role of the input.
     */
    static Tensor convTranspose2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                            int[] dilation, int groups) {
        Shape xs = x.shape();
        Shape gs = gy.shape();
        float[] gw = weightGrad(gy.toFloatArray(), gs.dim(1), gs.dim(2), gs.dim(3),
                x.toFloatArray(), xs.dim(1), xs.dim(2), xs.dim(3),
                xs.dim(0), weightShape, stride, padding, dilation, groups);
        return CpuShapeOps.floats(x.backend(), weightShape, gw, false);
    }

    /**
     * {@code gw[o, i, kh, kw] = sum over b, p, q ascending of g[b, o, p, q] * a[b, ac, p*SH + kh*DH - PH,
     * q*SW + kw*DW - PW]} over the in-bounds terms, with {@code ac = (o / (gC/groups)) * (aC/groups) + i} and
     * {@code gw} in the {@code [gC, aC/groups, KH, KW]} layout ({@code TR_Conv2dWeightGrad_F32}'s sum, with
     * {@code a} its x and {@code g} its gy).
     */
    private static float[] weightGrad(float[] a, int aC, int aH, int aW, float[] g, int gC, int gH, int gW,
                                      int batch, Shape weightShape, int[] stride, int[] padding, int[] dilation,
                                      int groups) {
        int kH = weightShape.dim(2);
        int kW = weightShape.dim(3);
        int aPerGroup = aC / groups;
        int gPerGroup = gC / groups;
        float[] gw = new float[CpuShapeOps.hostLength(weightShape.numElements())];
        for (int o = 0; o < gC; o++) {
            int firstChannel = (o / gPerGroup) * aPerGroup;
            for (int i = 0; i < aPerGroup; i++) {
                int ac = firstChannel + i;
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        float sum = 0.0f;
                        for (int b = 0; b < batch; b++) {
                            for (int p = 0; p < gH; p++) {
                                int ah = p * stride[0] + kh * dilation[0] - padding[0];
                                if (ah < 0 || ah >= aH) {
                                    continue;
                                }
                                for (int q = 0; q < gW; q++) {
                                    int aw = q * stride[1] + kw * dilation[1] - padding[1];
                                    if (aw < 0 || aw >= aW) {
                                        continue;
                                    }
                                    sum += g[((b * gC + o) * gH + p) * gW + q] * a[((b * aC + ac) * aH + ah) * aW + aw];
                                }
                            }
                        }
                        gw[((o * aPerGroup + i) * kH + kh) * kW + kw] = sum;
                    }
                }
            }
        }
        return gw;
    }

    // ------------------------------------------------------------------ pooling backward

    /**
     * Backward of {@link #maxPool2d}: for every window in ascending {@code (b, c, oh, ow)} order whose maximum
     * {@code m = y[b, c, oh, ow]} is not NaN, {@code gy[b, c, oh, ow]} is added to the first in-bounds tap in
     * {@code (kh, kw)} order whose value {@code == m}; every other element stays 0.
     */
    static Tensor maxPool2dBackward(Tensor x, Tensor y, Tensor gy, int[] kernelSize, int[] stride, int[] padding,
                                    int[] dilation) {
        Shape in = x.shape();
        int planes = in.dim(0) * in.dim(1);
        int inH = in.dim(2);
        int inW = in.dim(3);
        int outH = y.shape().dim(2);
        int outW = y.shape().dim(3);
        float[] inputData = x.toFloatArray();
        float[] maxData = y.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] gradInput = new float[CpuShapeOps.hostLength(in.numElements())];
        for (int bc = 0; bc < planes; bc++) {
            int plane = bc * inH * inW;
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int o = (bc * outH + oh) * outW + ow;
                    int tap = firstMax(inputData, plane, inH, inW, kernelSize, stride, padding, dilation, oh, ow,
                            maxData[o]);
                    if (tap >= 0) {
                        gradInput[plane + tap] += gradData[o];
                    }
                }
            }
        }
        return CpuShapeOps.floats(x.backend(), in, gradInput, false);
    }

    /**
     * The adjoint of {@link #maxPool2dBackward}: {@code out[b, c, oh, ow] = g[b, c, ih, iw]} at the tap that window
     * {@code (oh, ow)} routes its gradient to, and {@code +0} when it routes nowhere.
     */
    static Tensor maxPool2dSelect(Tensor x, Tensor y, Tensor g, int[] kernelSize, int[] stride, int[] padding,
                                  int[] dilation) {
        Shape in = x.shape();
        int planes = in.dim(0) * in.dim(1);
        int inH = in.dim(2);
        int inW = in.dim(3);
        int outH = y.shape().dim(2);
        int outW = y.shape().dim(3);
        float[] inputData = x.toFloatArray();
        float[] maxData = y.toFloatArray();
        float[] selectFrom = g.toFloatArray();
        float[] outputData = new float[CpuShapeOps.hostLength(y.shape().numElements())];
        for (int bc = 0; bc < planes; bc++) {
            int plane = bc * inH * inW;
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int o = (bc * outH + oh) * outW + ow;
                    int tap = firstMax(inputData, plane, inH, inW, kernelSize, stride, padding, dilation, oh, ow,
                            maxData[o]);
                    outputData[o] = tap >= 0 ? selectFrom[plane + tap] : 0.0f;
                }
            }
        }
        return CpuShapeOps.floats(x.backend(), y.shape(), outputData, false);
    }

    /**
     * The in-plane index {@code ih*W + iw} of the first in-bounds tap of window {@code (oh, ow)} in {@code (kh, kw)}
     * order whose value {@code == m} (Java float equality: {@code -0 == +0}, NaN equals nothing), or -1.
     */
    private static int firstMax(float[] x, int plane, int inH, int inW, int[] kernelSize, int[] stride,
                                int[] padding, int[] dilation, int oh, int ow, float m) {
        if (Float.isNaN(m)) {
            return -1;
        }
        for (int kh = 0; kh < kernelSize[0]; kh++) {
            int ih = oh * stride[0] + kh * dilation[0] - padding[0];
            if (ih < 0 || ih >= inH) {
                continue;
            }
            for (int kw = 0; kw < kernelSize[1]; kw++) {
                int iw = ow * stride[1] + kw * dilation[1] - padding[1];
                if (iw >= 0 && iw < inW && x[plane + ih * inW + iw] == m) {
                    return ih * inW + iw;
                }
            }
        }
        return -1;
    }

    /**
     * Backward of {@link #avgPool2d}: for every window in ascending {@code (b, c, oh, ow)} order with input elements,
     * {@code gy[b, c, oh, ow] / divisor} (the forward's {@link #avgDivisor}) is added to each of its in-bounds
     * elements.
     */
    static Tensor avgPool2dBackward(Tensor gy, Shape inputShape, int[] kernelSize, int[] stride, int[] padding,
                                    boolean countIncludePad) {
        int planes = inputShape.dim(0) * inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outH = gy.shape().dim(2);
        int outW = gy.shape().dim(3);
        float[] gradData = gy.toFloatArray();
        float[] gradInput = new float[CpuShapeOps.hostLength(inputShape.numElements())];
        for (int bc = 0; bc < planes; bc++) {
            int plane = bc * inH * inW;
            for (int oh = 0; oh < outH; oh++) {
                int hs = oh * stride[0] - padding[0];
                int he = Math.min(hs + kernelSize[0], inH + padding[0]);
                int h0 = Math.max(hs, 0);
                int h1 = Math.min(he, inH);
                for (int ow = 0; ow < outW; ow++) {
                    int ws = ow * stride[1] - padding[1];
                    int we = Math.min(ws + kernelSize[1], inW + padding[1]);
                    int w0 = Math.max(ws, 0);
                    int w1 = Math.min(we, inW);
                    if (h0 >= h1 || w0 >= w1) {
                        continue;
                    }
                    float term = gradData[(bc * outH + oh) * outW + ow]
                            / avgDivisor(hs, he, ws, we, h0, h1, w0, w1, countIncludePad);
                    for (int ih = h0; ih < h1; ih++) {
                        for (int iw = w0; iw < w1; iw++) {
                            gradInput[plane + ih * inW + iw] += term;
                        }
                    }
                }
            }
        }
        return CpuShapeOps.floats(gy.backend(), inputShape, gradInput, false);
    }

    /**
     * Backward of {@link #adaptiveAvgPool2d}: for every window in ascending {@code (b, c, oh, ow)} order,
     * {@code gy[b, c, oh, ow] / count} (the window's element count) is added to each of its elements; an element in
     * several overlapping windows sums their terms in that order.
     */
    static Tensor adaptiveAvgPool2dBackward(Tensor gy, Shape inputShape) {
        int planes = inputShape.dim(0) * inputShape.dim(1);
        int inH = inputShape.dim(2);
        int inW = inputShape.dim(3);
        int outH = gy.shape().dim(2);
        int outW = gy.shape().dim(3);
        float[] gradData = gy.toFloatArray();
        float[] gradInput = new float[CpuShapeOps.hostLength(inputShape.numElements())];
        for (int bc = 0; bc < planes; bc++) {
            int plane = bc * inH * inW;
            for (int oh = 0; oh < outH; oh++) {
                int hStart = ConvOps.adaptiveStart(oh, inH, outH);
                int hEnd = ConvOps.adaptiveEnd(oh, inH, outH);
                for (int ow = 0; ow < outW; ow++) {
                    int wStart = ConvOps.adaptiveStart(ow, inW, outW);
                    int wEnd = ConvOps.adaptiveEnd(ow, inW, outW);
                    int count = (hEnd - hStart) * (wEnd - wStart);
                    if (count == 0) {
                        continue;
                    }
                    float term = gradData[(bc * outH + oh) * outW + ow] / count;
                    for (int ih = hStart; ih < hEnd; ih++) {
                        for (int iw = wStart; iw < wEnd; iw++) {
                            gradInput[plane + ih * inW + iw] += term;
                        }
                    }
                }
            }
        }
        return CpuShapeOps.floats(gy.backend(), inputShape, gradInput, false);
    }
}

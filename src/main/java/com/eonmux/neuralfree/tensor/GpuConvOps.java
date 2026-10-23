package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;

/**
 * Convolution and pooling forward passes on the train kernels ({@code kernels/train/conv.cl}, {@code gemm.cl},
 * {@code pool.cl}) for NCHW FLOAT32 tensors. Results are new contiguous tensors on the input's backend with
 * {@code requiresGrad} false, as the layers' CPU paths produce; nothing is read back.
 *
 * <p><b>Dispatcher contract.</b> Every forward method returns {@code null} exactly when an operand is not FLOAT32
 * or the train library is unavailable on the input's backend ({@link TrainGpu#forBackend}); that is the caller's
 * CPU path. Invalid arguments throw {@link IllegalArgumentException} before that decision. Once the library is
 * available, kernel and ABI failures propagate. Operands may be any strided view: dense views (with any element
 * offset) are bound in place, other views are first copied contiguously on the device.</p>
 *
 * <p><b>Output sizes.</b> Convolutions use Java integer division (truncating toward zero), and a negative size
 * throws {@link IllegalArgumentException}. Max and average pooling use PyTorch's rule and argument checks,
 * {@link ConvOps#poolOutputSize}, which also throws {@link IllegalArgumentException}.</p>
 *
 * <p><b>Convolution algorithms.</b> A convolution runs either on the direct kernels ({@code TR_Conv2d_F32},
 * {@code TR_ConvTranspose2d_F32}: one output per work-item, a serial loop over its {@code (C/groups)*KH*KW} taps) or
 * as im2col + GEMM ({@code GpuIm2col}: the column matrices of chunks of images in a bounded scratch buffer, multiplied
 * with the weight by the strided GEMM; the transposed convolution multiplies its transposed weight with the input
 * into columns and gathers them back with {@code TR_Col2im_F32}). The choice depends only on the shapes: im2col +
 * GEMM when the tap loop has at least 16 taps and the columns of one image ({@code C*KH*KW*OH*OW} elements,
 * {@code OC*KH*KW*H*W} for a transposed convolution) fit 64 MiB, the direct kernels otherwise.</p>
 *
 * <p><b>Tolerance classes</b> against the CPU path. The reduction class is {@code |gpu - ref64| <= 1e-5 * sqrt(K) *
 * sum|terms| + K * FLT_MIN} against a double-precision reference with {@code K} terms; the division class is {@code
 * |gpu - ref| <= 4 ulp(ref)} or {@code <= FLT_MIN}; single-rounding is bit-identical for normal operands and results.
 * Convolutions are in the reduction class against a double reference with {@code (C/groups)*KH*KW} terms on both
 * algorithms (the direct kernels accumulate with {@code fma} in ic, kh, kw order starting from the bias; the GEMM path
 * sums the same products in the order {@code GpuIm2col} documents). A padding tap enters the GEMM path as a product
 * with an exact zero, so a non-finite weight at a tap that reads only padding gives NaN there where the direct kernel
 * skips the tap. Max pooling is bit-identical to {@code MaxPool2d} ({@code Math.max} semantics, first maximum); average
 * and adaptive average pooling are in the division class (the window sum is accumulated in the CPU order, then divided
 * once).</p>
 */
public final class GpuConvOps {

    private GpuConvOps() {
    }

    // ------------------------------------------------------------------ convolutions

    /**
     * 2-D convolution as {@code nn.layers.Conv2d}: {@code input [B, C, H, W]}, {@code weight [OC, C/groups, KH, KW]},
     * optional {@code bias} of {@code OC} elements; output {@code [B, OC, OH, OW]} with
     * {@code OH = (H + 2*PH - DH*(KH-1) - 1) / SH + 1} (and its W twin).
     *
     * @throws IllegalArgumentException on inconsistent shapes or parameters, operands on different backends, or a
     *                                  shape beyond the 32-bit kernel limits
     */
    public static Tensor conv2dForward(Tensor input, Tensor weight, Tensor bias, int[] kernelSize, int[] stride,
                                       int[] padding, int[] dilation, int groups) {
        return conv2dForward(input, weight, bias, kernelSize, stride, padding, dilation, groups,
                GpuIm2col.Strategy.AUTO);
    }

    /** {@link #conv2dForward} with an explicit algorithm choice. */
    static Tensor conv2dForward(Tensor input, Tensor weight, Tensor bias, int[] kernelSize, int[] stride,
                                int[] padding, int[] dilation, int groups, GpuIm2col.Strategy strategy) {
        requireOperands(input, weight, bias);
        requirePair(kernelSize, "kernelSize", 1);
        requirePair(stride, "stride", 1);
        requirePair(padding, "padding", 0);
        requirePair(dilation, "dilation", 1);
        requireRank(input.shape(), 4, "conv2d input [B, C, H, W]");
        requireRank(weight.shape(), 4, "conv2d weight [OC, C/groups, KH, KW]");
        int oc = weight.shape().dim(0);
        requireChannels(input.shape().dim(1), weight.shape().dim(1), oc, groups, "conv2d");
        requireKernelDims(weight, kernelSize, "conv2d");
        requireBias(bias, oc, "conv2d");
        Shape in = input.shape();
        int oh = extent("conv2d output height",
                convExtent((long) in.dim(2) + 2L * padding[0] - (long) dilation[0] * (kernelSize[0] - 1) - 1,
                        stride[0]));
        int ow = extent("conv2d output width",
                convExtent((long) in.dim(3) + 2L * padding[1] - (long) dilation[1] * (kernelSize[1] - 1) - 1,
                        stride[1]));
        ConvGeometry geo = new ConvGeometry(in.dim(0), in.dim(1), in.dim(2), in.dim(3), oc, oh, ow,
                kernelSize[0], kernelSize[1], stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
                groups);
        return convolve(geo, input, weight, bias, Shape.of(geo.b, oc, oh, ow), false, strategy);
    }

    /**
     * 1-D convolution as {@code nn.layers.Conv1d}: {@code input [B, C, L]}, {@code weight [OC, C/groups, K]},
     * optional {@code bias} of {@code OC} elements; output {@code [B, OC, OL]} with
     * {@code OL = (L + 2*padding - dilation*(K-1) - 1) / stride + 1}. Runs the 2-D kernel with height 1.
     *
     * @throws IllegalArgumentException as {@link #conv2dForward}
     */
    public static Tensor conv1dForward(Tensor input, Tensor weight, Tensor bias, int kernelSize, int stride,
                                       int padding, int dilation, int groups) {
        return conv1dForward(input, weight, bias, kernelSize, stride, padding, dilation, groups,
                GpuIm2col.Strategy.AUTO);
    }

    /** {@link #conv1dForward} with an explicit algorithm choice. */
    static Tensor conv1dForward(Tensor input, Tensor weight, Tensor bias, int kernelSize, int stride, int padding,
                                int dilation, int groups, GpuIm2col.Strategy strategy) {
        requireOperands(input, weight, bias);
        requirePositive(kernelSize, "kernelSize");
        requirePositive(stride, "stride");
        requirePositive(dilation, "dilation");
        if (padding < 0) {
            throw new IllegalArgumentException("padding must be non-negative: " + padding);
        }
        requireRank(input.shape(), 3, "conv1d input [B, C, L]");
        requireRank(weight.shape(), 3, "conv1d weight [OC, C/groups, K]");
        int oc = weight.shape().dim(0);
        requireChannels(input.shape().dim(1), weight.shape().dim(1), oc, groups, "conv1d");
        if (weight.shape().dim(2) != kernelSize) {
            throw new IllegalArgumentException("conv1d weight " + weight.shape() + " does not match kernelSize "
                    + kernelSize);
        }
        requireBias(bias, oc, "conv1d");
        Shape in = input.shape();
        int ol = extent("conv1d output length",
                convExtent((long) in.dim(2) + 2L * padding - (long) dilation * (kernelSize - 1) - 1, stride));
        ConvGeometry geo = new ConvGeometry(in.dim(0), in.dim(1), 1, in.dim(2), oc, 1, ol,
                1, kernelSize, 1, stride, 0, padding, 1, dilation, groups);
        return convolve(geo, input, weight, bias, Shape.of(geo.b, oc, ol), false, strategy);
    }

    /**
     * 2-D transposed convolution as {@code nn.layers.ConvTranspose2d}, in gather form (PyTorch semantics):
     * {@code input [B, C, H, W]}, {@code weight [C, OC/groups, KH, KW]}, optional {@code bias} of {@code OC}
     * elements; output {@code [B, OC, OH, OW]} with
     * {@code OH = (H-1)*SH - 2*PH + DH*(KH-1) + outputPadding[0] + 1} (and its W twin), where
     * {@code out[b, oc, oh, ow] = bias[oc] + sum over ic in the group, kh, kw of in[b, cin, ih, iw] * w[cin, oc % OCg,
     * kh, kw]} for every tap with {@code oh + PH - kh*DH = ih*SH} and {@code ow + PW - kw*DW = iw*SW} inside the input.
     *
     * @throws IllegalArgumentException as {@link #conv2dForward}, and when an {@code outputPadding[i]} is not smaller
     *                                  than {@code stride[i]} or {@code dilation[i]}
     *                                  ({@link ConvOps#requireTransposeOutputPadding})
     */
    public static Tensor convTranspose2dForward(Tensor input, Tensor weight, Tensor bias, int[] kernelSize,
                                                int[] stride, int[] padding, int[] outputPadding, int[] dilation,
                                                int groups) {
        return convTranspose2dForward(input, weight, bias, kernelSize, stride, padding, outputPadding, dilation,
                groups, GpuIm2col.Strategy.AUTO);
    }

    /** {@link #convTranspose2dForward} with an explicit algorithm choice. */
    static Tensor convTranspose2dForward(Tensor input, Tensor weight, Tensor bias, int[] kernelSize, int[] stride,
                                         int[] padding, int[] outputPadding, int[] dilation, int groups,
                                         GpuIm2col.Strategy strategy) {
        requireOperands(input, weight, bias);
        requirePair(kernelSize, "kernelSize", 1);
        requirePair(stride, "stride", 1);
        requirePair(padding, "padding", 0);
        requirePair(outputPadding, "outputPadding", 0);
        requirePair(dilation, "dilation", 1);
        ConvOps.requireTransposeOutputPadding(outputPadding, stride, dilation);
        requireRank(input.shape(), 4, "convTranspose2d input [B, C, H, W]");
        requireRank(weight.shape(), 4, "convTranspose2d weight [C, OC/groups, KH, KW]");
        if (groups < 1) {
            throw new IllegalArgumentException("groups must be positive: " + groups);
        }
        Shape in = input.shape();
        if (weight.shape().dim(0) != in.dim(1) || in.dim(1) % groups != 0) {
            throw new IllegalArgumentException("convTranspose2d input channels " + in.dim(1) + " must equal weight "
                    + weight.shape() + " dim 0 and be divisible by groups " + groups);
        }
        long ocLong = (long) weight.shape().dim(1) * groups;
        int oc = extent("convTranspose2d output channels", ocLong);
        requireKernelDims(weight, kernelSize, "convTranspose2d");
        requireBias(bias, oc, "convTranspose2d");
        int oh = extent("convTranspose2d output height", ((long) in.dim(2) - 1) * stride[0] - 2L * padding[0]
                + (long) dilation[0] * (kernelSize[0] - 1) + outputPadding[0] + 1);
        int ow = extent("convTranspose2d output width", ((long) in.dim(3) - 1) * stride[1] - 2L * padding[1]
                + (long) dilation[1] * (kernelSize[1] - 1) + outputPadding[1] + 1);
        ConvGeometry geo = new ConvGeometry(in.dim(0), in.dim(1), in.dim(2), in.dim(3), oc, oh, ow,
                kernelSize[0], kernelSize[1], stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
                groups);
        return convolve(geo, input, weight, bias, Shape.of(geo.b, oc, oh, ow), true, strategy);
    }

    // ------------------------------------------------------------------ pooling

    /**
     * Max pooling as {@code nn.layers.MaxPool2d} (PyTorch {@code max_pool2d}): {@code input [B, C, H, W]}, output
     * {@code [B, C, OH, OW]} with {@code OH = ConvOps.poolOutputSize(H, KH, SH, PH, DH, ceilMode)} (and its W twin).
     * A window without input elements yields {@code -inf}. Bit-identical to the CPU path.
     *
     * @throws IllegalArgumentException on invalid parameters, a rank other than 4, a padding above half the effective
     *                                  kernel, or an output size below 1
     */
    public static Tensor maxPool2dForward(Tensor input, int[] kernelSize, int[] stride, int[] padding,
                                          int[] dilation, boolean ceilMode) {
        Objects.requireNonNull(input, "input must not be null");
        requirePair(kernelSize, "kernelSize", 1);
        requirePair(stride, "stride", 1);
        requirePair(padding, "padding", 0);
        requirePair(dilation, "dilation", 1);
        requireRank(input.shape(), 4, "maxPool2d input [B, C, H, W]");
        Shape in = input.shape();
        int oh = ConvOps.poolOutputSize(in.dim(2), kernelSize[0], stride[0], padding[0], dilation[0], ceilMode);
        int ow = ConvOps.poolOutputSize(in.dim(3), kernelSize[1], stride[1], padding[1], dilation[1], ceilMode);
        TrainGpu g = floatGpu(input);
        if (g == null) {
            return null;
        }
        requireWindowFits(oh, stride[0], kernelSize[0], dilation[0], "maxPool2d height");
        requireWindowFits(ow, stride[1], kernelSize[1], dilation[1], "maxPool2d width");
        return pool(g, Kernels.MAX_POOL2D_F32, input, oh, ow, Float.NEGATIVE_INFINITY, view -> new Object[]{
                in.dim(0), in.dim(1), in.dim(2), in.dim(3), oh, ow, kernelSize[0], kernelSize[1],
                stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1], view.offset(), 0});
    }

    /**
     * Average pooling as {@code nn.layers.AvgPool2d} (PyTorch {@code avg_pool2d}): {@code input [B, C, H, W]}, output
     * {@code [B, C, OH, OW]} with {@code OH = ConvOps.poolOutputSize(H, KH, SH, PH, 1, ceilMode)} (and its W twin).
     * Window {@code (oh, ow)} spans rows {@code [hs, he)} with {@code hs = oh*SH - PH}, {@code he = min(hs + KH,
     * H + PH)} (columns likewise); each output is the sum of its in-bounds elements divided by
     * {@code (he - hs)*(we - ws)} when {@code countIncludePad}, otherwise by the in-bounds count, and 0 for a window
     * without input elements.
     *
     * @throws IllegalArgumentException on invalid parameters, a rank other than 4, a padding above half the kernel,
     *                                  or an output size below 1
     */
    public static Tensor avgPool2dForward(Tensor input, int[] kernelSize, int[] stride, int[] padding,
                                          boolean ceilMode, boolean countIncludePad) {
        Objects.requireNonNull(input, "input must not be null");
        requirePair(kernelSize, "kernelSize", 1);
        requirePair(stride, "stride", 1);
        requirePair(padding, "padding", 0);
        requireRank(input.shape(), 4, "avgPool2d input [B, C, H, W]");
        Shape in = input.shape();
        int oh = ConvOps.poolOutputSize(in.dim(2), kernelSize[0], stride[0], padding[0], 1, ceilMode);
        int ow = ConvOps.poolOutputSize(in.dim(3), kernelSize[1], stride[1], padding[1], 1, ceilMode);
        TrainGpu g = floatGpu(input);
        if (g == null) {
            return null;
        }
        requireAvgWindowFits(oh, stride[0], kernelSize[0], in.dim(2), padding[0], "avgPool2d height");
        requireAvgWindowFits(ow, stride[1], kernelSize[1], in.dim(3), padding[1], "avgPool2d width");
        requireIntSum((long) kernelSize[0] * kernelSize[1], 0L, "avgPool2d window size");
        return pool(g, Kernels.AVG_POOL2D_F32, input, oh, ow, 0.0f, view -> new Object[]{
                in.dim(0), in.dim(1), in.dim(2), in.dim(3), oh, ow, kernelSize[0], kernelSize[1],
                stride[0], stride[1], padding[0], padding[1], countIncludePad ? 1 : 0, view.offset(), 0});
    }

    /**
     * Adaptive average pooling as {@code nn.layers.AdaptiveAvgPool2d} (PyTorch {@code adaptive_avg_pool2d}):
     * {@code input [B, C, H, W]}, output {@code [B, C, outH, outW]}; output row {@code oh} averages input rows
     * {@code [floor(oh*H/outH), ceil((oh+1)*H/outH))} (columns likewise), so windows overlap when {@code outH} does
     * not divide {@code H}; a window without input elements (an empty plane) yields 0.
     *
     * @throws IllegalArgumentException when {@code outH} or {@code outW} is not positive, the rank is not 4, or
     *                                  {@code outH*H + outH - 1} or {@code outW*W + outW - 1} exceeds
     *                                  {@code 2^31-1}
     */
    public static Tensor adaptiveAvgPool2d(Tensor input, int outH, int outW) {
        Objects.requireNonNull(input, "input must not be null");
        requirePositive(outH, "outH");
        requirePositive(outW, "outW");
        requireRank(input.shape(), 4, "adaptiveAvgPool2d input [B, C, H, W]");
        Shape in = input.shape();
        requireAdaptiveBounds(outH, outW, in);
        TrainGpu g = floatGpu(input);
        if (g == null) {
            return null;
        }
        return pool(g, Kernels.ADAPTIVE_AVG_POOL2D_F32, input, outH, outW, 0.0f, view -> new Object[]{
                in.dim(0), in.dim(1), in.dim(2), in.dim(3), outH, outW, view.offset(), 0});
    }

    // ------------------------------------------------------------------ cache

    /** Releases the train context of every backend ({@link TrainGpu#releaseAll}). */
    public static void clearCache() {
        TrainGpu.releaseAll();
    }

    /** Releases the train context of one backend ({@link TrainGpu#release}). */
    public static void clearCache(ComputeBackend backend) {
        TrainGpu.release(backend);
    }

    // ------------------------------------------------------------------ launches

    /**
     * The convolution ({@code transposed}: the transposed convolution) of {@code geo} into a new {@code outShape}
     * tensor, on the algorithm {@code strategy} selects; null for the CPU path.
     */
    static Tensor convolve(ConvGeometry geo, Tensor input, Tensor weight, Tensor bias, Shape outShape,
                           boolean transposed, GpuIm2col.Strategy strategy) {
        if (input.dtype() != DType.FLOAT32 || weight.dtype() != DType.FLOAT32
                || (bias != null && bias.dtype() != DType.FLOAT32)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(input.backend());
        if (g == null) {
            return null;
        }
        if (transposed) {
            requireIntSum((long) geo.oh - 1, geo.ph, "convTranspose2d output row plus padding");
            requireIntSum((long) geo.ow - 1, geo.pw, "convTranspose2d output column plus padding");
            requireIntSum((long) (geo.kh - 1) * geo.dh, 0, "convTranspose2d kernel height span");
            requireIntSum((long) (geo.kw - 1) * geo.dw, 0, "convTranspose2d kernel width span");
        } else {
            requireWindowFits(geo.oh, geo.sh, geo.kh, geo.dh, "convolution height");
            requireWindowFits(geo.ow, geo.sw, geo.kw, geo.dw, "convolution width");
        }
        Tensor out = allocate(g, input.backend(), outShape);
        try {
            if (out.numElements() == 0) {
                return out;
            }
            if (input.numElements() == 0) {
                fillBias(g, out, bias, outShape);
                return out;
            }
            GpuIm2col.Plan plan = transposed ? GpuIm2col.planTransposed(geo, strategy)
                    : GpuIm2col.planForward(geo, strategy);
            try (Operand in = Operand.of(input); Operand w = Operand.of(weight);
                 Operand b = bias == null ? null : Operand.of(bias)) {
                GpuBuffer outBuffer = out.storage().gpuBuffer();
                if (plan != null && transposed) {
                    GpuIm2col.transposed(g, plan, geo, in.view, w.view, b == null ? null : b.view, outBuffer);
                    return out;
                }
                if (plan != null) {
                    if (bias != null) {
                        fillBias(g, out, bias, outShape);
                    }
                    GpuIm2col.forward(g, plan, geo, in.view, w.view, bias != null, outBuffer);
                    return out;
                }
                GpuBuffer biasBuffer = b == null ? g.placeholder() : b.view.buffer();
                g.launch1d(transposed ? Kernels.CONV_TRANSPOSE2D_F32 : Kernels.CONV2D_F32, out.numElements(),
                        geo.b, geo.c, geo.h, geo.w, geo.oc, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw,
                        geo.ph, geo.pw, geo.dh, geo.dw, geo.groups, b == null ? 0 : 1,
                        in.view.offset(), w.view.offset(), b == null ? 0 : b.view.offset(), 0,
                        in.view.buffer(), w.view.buffer(), biasBuffer, outBuffer);
            }
            return out;
        } catch (RuntimeException e) {
            out.close();
            throw e;
        }
    }

    /**
     * Writes the bias broadcast over each channel into every element of {@code out}, or zeros without bias: the
     * output of a convolution whose input has no elements, and the start value of the GEMM path's accumulation. The
     * reshaped bias is an internal temporary taken under {@link ComputationGraph#noGrad()}, so it never records,
     * whatever the caller's grad mode.
     */
    private static void fillBias(TrainGpu g, Tensor out, Tensor bias, Shape outShape) {
        if (bias == null) {
            g.fill(out.storage().gpuBuffer(), 0, out.numElements(), 0.0f);
            return;
        }
        int[] dims = new int[outShape.rank()];
        Arrays.fill(dims, 1);
        dims[1] = outShape.dim(1);
        try (ComputationGraph.NoGradScope untracked = ComputationGraph.noGrad();
             Tensor channels = bias.reshape(Shape.of(dims))) {
            GpuCoreOps.copyInto(channels, out);
        }
    }

    static Tensor pool(TrainGpu g, String kernel, Tensor input, int oh, int ow, float emptyInputValue,
                       Function<StridedView, Object[]> scalars) {
        Shape in = input.shape();
        Shape outShape = Shape.of(in.dim(0), in.dim(1), oh, ow);
        Tensor out = allocate(g, input.backend(), outShape);
        try {
            if (out.numElements() == 0) {
                return out;
            }
            if (input.numElements() == 0) {
                g.fill(out.storage().gpuBuffer(), 0, out.numElements(), emptyInputValue);
                return out;
            }
            try (Operand x = Operand.of(input)) {
                Object[] s = scalars.apply(x.view);
                Object[] args = Arrays.copyOf(s, s.length + 2);
                args[s.length] = x.view.buffer();
                args[s.length + 1] = out.storage().gpuBuffer();
                g.launch1d(kernel, out.numElements(), args);
            }
            return out;
        } catch (RuntimeException e) {
            out.close();
            throw e;
        }
    }

    static Tensor allocate(TrainGpu g, ComputeBackend backend, Shape shape) {
        long n = shape.numElements();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("output " + shape + " has " + n + " elements, beyond the 32-bit kernel"
                    + " limit " + Integer.MAX_VALUE);
        }
        return Tensor.fromStorage(backend, shape, DType.FLOAT32, g.allocate(n, DType.FLOAT32));
    }

    /** The train context for a FLOAT32 input, or null for the CPU path. */
    private static TrainGpu floatGpu(Tensor input) {
        return input.dtype() == DType.FLOAT32 ? TrainGpu.forBackend(input.backend()) : null;
    }

    // ------------------------------------------------------------------ validation

    /** The convolution output extent {@code span / stride + 1}, with Java integer division as the conv layers use. */
    static long convExtent(long span, int stride) {
        if (span < Integer.MIN_VALUE || span > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("output span " + span + " overflows the layer's int arithmetic");
        }
        return span / stride + 1;
    }

    /** {@code value} as an int extent, checked to lie in {@code [0, 2^31-1]}. */
    static int extent(String what, long value) {
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " must be in [0, " + Integer.MAX_VALUE + "], got " + value);
        }
        return (int) value;
    }

    /** The kernels compute {@code o*S + k*D} for every output index and tap in int. */
    private static void requireWindowFits(int out, int stride, int k, int dilation, String what) {
        if (out > 0) {
            requireIntSum((long) (out - 1) * stride, (long) (k - 1) * dilation, what + " window position");
        }
    }

    /**
     * The average-pool kernels compute each window's start {@code o*S - P}, its end {@code o*S - P + K} and the
     * padded edge {@code in + P} in int.
     */
    static void requireAvgWindowFits(int out, int stride, int k, int in, int padding, String what) {
        if (out > 0) {
            requireIntSum((long) (out - 1) * stride, k, what + " window end");
        }
        requireIntSum(in, padding, what + " padded extent");
    }

    /**
     * The adaptive kernels compute {@code o*in} and the ceil-end window bound {@code (o+1)*in + out - 1} in int,
     * for {@code o < out}.
     */
    static void requireAdaptiveBounds(int outH, int outW, Shape in) {
        if ((long) outH * in.dim(2) + outH - 1 > Integer.MAX_VALUE
                || (long) outW * in.dim(3) + outW - 1 > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("adaptiveAvgPool2d window bounds overflow int: output " + outH + "x"
                    + outW + " over input " + in);
        }
    }

    /** The kernels evaluate {@code a + b} in int, so the sum must not exceed {@code 2^31-1}. */
    static void requireIntSum(long a, long b, String what) {
        if (a + b > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + (a + b) + " exceeds the 32-bit kernel limit");
        }
    }

    private static void requireOperands(Tensor input, Tensor weight, Tensor bias) {
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(weight, "weight must not be null");
        if (weight.backend() != input.backend() || (bias != null && bias.backend() != input.backend())) {
            throw new IllegalArgumentException("input, weight and bias must live on the same backend");
        }
    }

    static void requireRank(Shape shape, int rank, String what) {
        if (shape.rank() != rank) {
            throw new IllegalArgumentException(what + " must have rank " + rank + ", got " + shape);
        }
    }

    private static void requireChannels(int c, int weightInChannels, int oc, int groups, String op) {
        if (groups < 1) {
            throw new IllegalArgumentException("groups must be positive: " + groups);
        }
        if (oc % groups != 0 || (long) weightInChannels * groups != c) {
            throw new IllegalArgumentException(op + ": input channels " + c + " must equal weight input channels "
                    + weightInChannels + " times groups " + groups + ", and output channels " + oc
                    + " must be divisible by groups");
        }
    }

    private static void requireKernelDims(Tensor weight, int[] kernelSize, String op) {
        if (weight.shape().dim(2) != kernelSize[0] || weight.shape().dim(3) != kernelSize[1]) {
            throw new IllegalArgumentException(op + " weight " + weight.shape() + " does not match kernelSize "
                    + Arrays.toString(kernelSize));
        }
    }

    private static void requireBias(Tensor bias, int oc, String op) {
        if (bias != null && bias.numElements() != oc) {
            throw new IllegalArgumentException(op + " bias " + bias.shape() + " must have " + oc + " elements");
        }
    }

    /** {@code values} is an {@code int[2]} whose entries are at least {@code min}. */
    static void requirePair(int[] values, String name, int min) {
        if (values == null || values.length != 2 || values[0] < min || values[1] < min) {
            throw new IllegalArgumentException(name + " must be int[2] with values >= " + min + ", got "
                    + Arrays.toString(values));
        }
    }

    private static void requirePositive(int value, String name) {
        if (value < 1) {
            throw new IllegalArgumentException(name + " must be positive: " + value);
        }
    }

    // ------------------------------------------------------------------ helpers

    /** The 16 shape scalars of the conv kernels; a 1-D convolution is the 2-D case with height 1. */
    static final class ConvGeometry {
        final int b, c, h, w, oc, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, groups;

        ConvGeometry(int b, int c, int h, int w, int oc, int oh, int ow, int kh, int kw, int sh, int sw, int ph,
                     int pw, int dh, int dw, int groups) {
            this.b = b;
            this.c = c;
            this.h = h;
            this.w = w;
            this.oc = oc;
            this.oh = oh;
            this.ow = ow;
            this.kh = kh;
            this.kw = kw;
            this.sh = sh;
            this.sw = sw;
            this.ph = ph;
            this.pw = pw;
            this.dh = dh;
            this.dw = dw;
            this.groups = groups;
        }
    }

    /**
     * A dense row-major view of an operand: the tensor's own buffer and element offset when it is dense, otherwise a
     * device contiguous copy that {@link #close} releases.
     */
    static final class Operand implements AutoCloseable {
        final Tensor staged;
        final StridedView view;

        private Operand(Tensor staged, StridedView view) {
            this.staged = staged;
            this.view = view;
        }

        static Operand of(Tensor t) {
            StridedView v = StridedView.of(t);
            if (v.contiguous()) {
                return new Operand(null, v);
            }
            Tensor copy = GpuCoreOps.contiguous(t);
            if (copy == null) {
                throw new IllegalStateException("device contiguous copy of a FLOAT32 " + t.shape()
                        + " view unavailable while the train library is available");
            }
            try {
                return new Operand(copy, StridedView.of(copy));
            } catch (RuntimeException e) {
                copy.close();
                throw e;
            }
        }

        @Override
        public void close() {
            if (staged != null) {
                staged.close();
            }
        }
    }
}

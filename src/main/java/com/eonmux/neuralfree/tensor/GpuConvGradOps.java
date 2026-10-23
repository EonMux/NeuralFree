package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.Objects;

/**
 * Backward passes of the convolutions and poolings of {@link GpuConvOps} on the train kernels
 * ({@code kernels/train/conv.cl} {@code TR_Conv2dWeightGrad_F32} or {@code TR_Im2col_F32} with {@code gemm.cl}
 * {@code TR_Gemm_F32}, {@code pool.cl} {@code TR_MaxPool2dBackward_F32}, {@code TR_AvgPool2dBackward_F32},
 * {@code TR_AdaptiveAvgPool2dBackward_F32}, {@code TR_MaxPool2dSelect_F32}) for
 * NCHW FLOAT32 tensors. Results are new contiguous tensors on the operands' backend with {@code requiresGrad} false;
 * nothing is read back. The input gradients of the convolutions are not here: they are the forward convolutions with
 * swapped roles ({@code ConvOps.convTranspose2d} for a convolution, {@code ConvOps.conv2d} for a transposed one).
 *
 * <p><b>Dispatcher contract.</b> Every method returns {@code null} exactly when an operand is not FLOAT32 or the train
 * library is unavailable on the operands' backend ({@link TrainGpu#forBackend}); that is the caller's CPU path
 * ({@code CpuConvPool}). Invalid arguments throw {@link IllegalArgumentException} before that decision, with the
 * forward's output-size formulas: every gradient and saved operand must have the shape the forward produced from the
 * given geometry. Once the library is available, kernel and ABI failures propagate. Operands may be any strided view:
 * dense views are bound in place, other views are first copied contiguously on the device.</p>
 *
 * <p><b>Launch rules.</b> A result without elements launches nothing. A result whose every element is a sum of zero
 * terms (an empty gradient or forward operand) is filled with 0 instead of launching. The weight gradients choose
 * their algorithm by shape alone, with the rule of {@link GpuConvOps} applied to the convolution in the kernel roles:
 * im2col + GEMM ({@code GpuIm2col}: per image the GEMM {@code gy_b x cols_b^T}, accumulated over the images in
 * ascending order) when its tap loop {@code (C/groups)*KH*KW} has at least 16 taps and the columns of one image fit
 * 64 MiB; otherwise {@code TR_Conv2dWeightGrad_F32}, which runs {@code nW * chunks} groups of 256 with
 * {@code chunks = (nW < 4096 && K > 32768) ? min(64, ceil(K / 32768)) : 1}, {@code K = B * OH * OW}; with
 * {@code chunks > 1} the {@code [nW, chunks]} partials are summed into the result by {@code TR_ReduceRowsSmall_F32}
 * ({@code TR_R_SUM}, divisor 1).</p>
 *
 * <p><b>Tolerance classes</b> against the CPU path. The reduction class is {@code |gpu - ref64| <= 1e-5 * sqrt(K) *
 * sum|terms| + K * FLT_MIN} against a double-precision reference with {@code K} terms; the division class is {@code
 * |gpu - ref| <= 4 ulp(ref)} or {@code <= FLT_MIN}; single-rounding is bit-identical for normal operands and results.
 * The weight gradients are in the reduction class with {@code K} terms against a double reference on both algorithms
 * (on the GEMM path a padding tap is a product with an exact zero, so a non-finite gradient at a tap whose input is
 * only padding gives NaN where the direct kernel skips it); the max-pool backward is single-rounding (the CPU's adds in
 * the CPU's order); {@code maxPool2dSelect} is bit-identical; the average-pool backward is in the reduction class with
 * {@code ceil(KH/SH) * ceil(KW/SW)} terms {@code gy / divisor} (division class when the windows do not overlap); the
 * adaptive average-pool backward is in the reduction class with at most {@code (floor((OH-1)/H) + 2) * (floor((OW-1)/W)
 * + 2)} terms {@code gy / count} (division class when every input element lies in one window, as when {@code OH}
 * divides {@code H} and {@code OW} divides {@code W}).</p>
 */
public final class GpuConvGradOps {

    /** Weight counts below this with more than {@link #CHUNK_TERMS} terms per weight are reduced in chunks. */
    private static final int CHUNK_MAX_WEIGHTS = 4096;
    private static final int CHUNK_TERMS = 32768;
    private static final int MAX_CHUNKS = 64;

    private GpuConvGradOps() {
    }

    // ------------------------------------------------------------------ convolution weight gradients

    /**
     * Weight gradient of {@code GpuConvOps.conv2dForward}: {@code gw[oc, ic, kh, kw] = sum over b, oh, ow of
     * gy[b, oc, oh, ow] * x[b, g*Cg + ic, oh*SH + kh*DH - PH, ow*SW + kw*DW - PW]} over the in-bounds taps, with
     * {@code g = oc / (OC/groups)} and {@code Cg = C/groups}.
     *
     * @param x           the forward input {@code [B, C, H, W]}
     * @param gy          the output gradient {@code [B, OC, OH, OW]}, OH and OW as the forward computes them
     * @param weightShape the weight shape {@code [OC, C/groups, KH, KW]}
     * @return a new {@code weightShape} tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException on inconsistent shapes or parameters, operands on different backends, or a
     *                                  shape beyond the 32-bit kernel limits
     */
    public static Tensor conv2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                          int[] dilation, int groups) {
        return conv2dWeightGrad(x, gy, weightShape, stride, padding, dilation, groups, GpuIm2col.Strategy.AUTO);
    }

    /** {@link #conv2dWeightGrad} with an explicit algorithm choice. */
    static Tensor conv2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                   int[] dilation, int groups, GpuIm2col.Strategy strategy) {
        ConvGradGeometry geo = ConvGradGeometry.conv2d(x, gy, weightShape, stride, padding, dilation, groups);
        return weightGrad(geo, x, gy, weightShape, strategy);
    }

    /**
     * Weight gradient of {@code GpuConvOps.convTranspose2dForward}: {@code gw[cin, ocg, kh, kw] = sum over b, ih, iw
     * of x[b, cin, ih, iw] * gy[b, g*OCg + ocg, ih*SH + kh*DH - PH, iw*SW + kw*DW - PW]} over the in-bounds output
     * positions, with {@code g = cin / (C/groups)}. It is {@code TR_Conv2dWeightGrad_F32} with the operands swapped
     * (the kernel's x is {@code gy}, its gy is {@code x}).
     *
     * @param x           the forward input {@code [B, C, H, W]}
     * @param gy          the output gradient {@code [B, OC, OH, OW]} with
     *                    {@code OH = (H-1)*SH - 2*PH + DH*(KH-1) + outputPadding[0] + 1} (and its W twin)
     * @param weightShape the weight shape {@code [C, OC/groups, KH, KW]}
     * @return a new {@code weightShape} tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException as {@link #conv2dWeightGrad}
     */
    public static Tensor convTranspose2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride,
                                                   int[] padding, int[] outputPadding, int[] dilation, int groups) {
        return convTranspose2dWeightGrad(x, gy, weightShape, stride, padding, outputPadding, dilation, groups,
                GpuIm2col.Strategy.AUTO);
    }

    /** {@link #convTranspose2dWeightGrad} with an explicit algorithm choice. */
    static Tensor convTranspose2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                            int[] outputPadding, int[] dilation, int groups,
                                            GpuIm2col.Strategy strategy) {
        ConvGradGeometry geo = ConvGradGeometry.convTranspose2d(x, gy, weightShape, stride, padding, outputPadding,
                dilation, groups);
        return weightGrad(geo, gy, x, weightShape, strategy);
    }

    /**
     * The convolution weight gradient over {@code geo} with kernel operands {@code kx} ({@code [B, C, H, W]} of the
     * geometry) and {@code kgy} ({@code [B, OC, OH, OW]}), into a new {@code weightShape} tensor, on the algorithm
     * {@code strategy} selects ({@code GpuIm2col.weightGrad} or {@code TR_Conv2dWeightGrad_F32}).
     */
    private static Tensor weightGrad(ConvGradGeometry geo, Tensor kx, Tensor kgy, Shape weightShape,
                                     GpuIm2col.Strategy strategy) {
        if (!allFloat32(kx, kgy)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(kx.backend());
        if (g == null) {
            return null;
        }
        int nW = requireKernelElements(weightShape, "weight gradient");
        if (nW > 0 && kx.numElements() > 0 && kgy.numElements() > 0) {
            GpuConvOps.requireIntSum((long) (geo.oh - 1) * geo.sh, (long) (geo.kh - 1) * geo.dh,
                    "weight gradient row position");
            GpuConvOps.requireIntSum((long) (geo.ow - 1) * geo.sw, (long) (geo.kw - 1) * geo.dw,
                    "weight gradient column position");
        }
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            return GpuNormOps.output(g, kx, weightShape, DType.FLOAT32, out -> {
                GpuBuffer gw = out.storage().gpuBuffer();
                if (nW == 0) {
                    return;
                }
                if (kx.numElements() == 0 || kgy.numElements() == 0) {
                    g.fill(gw, 0, nW, 0.0f);
                    return;
                }
                StridedView xv = s.dense(kx);
                StridedView gv = s.dense(kgy);
                GpuConvOps.ConvGeometry conv = geo.convGeometry();
                GpuIm2col.Plan plan = GpuIm2col.planWeightGrad(conv, strategy);
                if (plan != null) {
                    GpuIm2col.weightGrad(g, plan, conv, xv, gv, gw);
                    return;
                }
                long k = (long) geo.b * geo.oh * geo.ow;
                int chunks = weightGradChunks(nW, k);
                GpuBuffer target = chunks == 1 ? gw : s.temp(nW * chunks);
                g.launchRows(Kernels.CONV2D_WEIGHT_GRAD_F32, (long) nW * chunks,
                        geo.b, geo.c, geo.h, geo.w, geo.oc, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw,
                        geo.ph, geo.pw, geo.dh, geo.dw, geo.groups, chunks, xv.offset(), gv.offset(), 0,
                        xv.buffer(), gv.buffer(), target);
                if (chunks > 1) {
                    GpuNormOps.reduceRowsOnce(g, TrainOps.TR_R_SUM, nW, chunks, target, 0, g.placeholder(), gw,
                            1.0f);
                }
            });
        }
    }

    /** The chunk count of the weight-gradient shape rule ({@code conv.cl} header), a function of the shape only. */
    static int weightGradChunks(int weights, long terms) {
        if (weights >= CHUNK_MAX_WEIGHTS || terms <= CHUNK_TERMS) {
            return 1;
        }
        return (int) Math.min(MAX_CHUNKS, (terms + CHUNK_TERMS - 1) / CHUNK_TERMS);
    }

    // ------------------------------------------------------------------ pooling

    /**
     * Max-pool backward: the gradient of window {@code (oh, ow)} goes to its first in-bounds tap in {@code (kh, kw)}
     * order whose value is Java-{@code ==} the window maximum {@code y[b, c, oh, ow]}; a NaN maximum routes nowhere.
     * Each input element sums the gradients routed to it in ascending {@code (oh, ow)} order from 0.
     *
     * @param x  the forward input {@code [B, C, H, W]}
     * @param y  the forward output {@code [B, C, OH, OW]} of {@code x} with this geometry
     * @param gy the output gradient, shaped as {@code y}
     * @return a new {@code [B, C, H, W]} tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException on invalid parameters, mismatched shapes or backends
     */
    public static Tensor maxPool2dBackward(Tensor x, Tensor y, Tensor gy, int[] kernelSize, int[] stride,
                                           int[] padding, int[] dilation, boolean ceilMode) {
        PoolGeometry geo = PoolGeometry.max(x, y, gy, kernelSize, stride, padding, dilation, ceilMode, "gy");
        if (!allFloat32(x, y, gy)) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(x.backend());
        if (g == null) {
            return null;
        }
        Shape outShape = x.shape();
        requireKernelElements(outShape, "maxPool2d input gradient");
        boolean launch = outShape.numElements() > 0 && gy.numElements() > 0;
        if (launch) {
            geo.requireBackwardFits();
        }
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            return GpuNormOps.output(g, x, outShape, DType.FLOAT32, out -> {
                GpuBuffer gx = out.storage().gpuBuffer();
                if (!launch) {
                    g.fill(gx, 0, out.numElements(), 0.0f);
                    return;
                }
                StridedView xv = s.dense(x);
                StridedView yv = s.dense(y);
                StridedView gv = s.dense(gy);
                g.launch1d(Kernels.MAX_POOL2D_BACKWARD_F32, out.numElements(),
                        geo.b, geo.c, geo.h, geo.w, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw, geo.ph, geo.pw,
                        geo.dh, geo.dw, xv.offset(), yv.offset(), gv.offset(), 0,
                        xv.buffer(), yv.buffer(), gv.buffer(), gx);
            });
        }
    }

    /**
     * The adjoint of {@link #maxPool2dBackward} in its gradient: {@code out[b, c, oh, ow] = g[b, c, ih, iw]} at the
     * tap that {@link #maxPool2dBackward} routes window {@code (oh, ow)} to, and {@code +0} when it routes nowhere.
     *
     * @param x the forward input {@code [B, C, H, W]}
     * @param y the forward output {@code [B, C, OH, OW]} of {@code x} with this geometry
     * @param g a tensor shaped as {@code x}
     * @return a new tensor shaped as {@code y}, or null (dispatcher contract)
     * @throws IllegalArgumentException on invalid parameters, mismatched shapes or backends
     */
    public static Tensor maxPool2dSelect(Tensor x, Tensor y, Tensor g, int[] kernelSize, int[] stride, int[] padding,
                                         int[] dilation, boolean ceilMode) {
        PoolGeometry geo = PoolGeometry.max(x, y, null, kernelSize, stride, padding, dilation, ceilMode, "g");
        Objects.requireNonNull(g, "g must not be null");
        requireSameBackend(x, g);
        requireShape(g, x.shape(), "maxPool2dSelect g");
        if (!allFloat32(x, y, g)) {
            return null;
        }
        TrainGpu gpu = TrainGpu.forBackend(x.backend());
        if (gpu == null) {
            return null;
        }
        Shape outShape = y.shape();
        requireKernelElements(outShape, "maxPool2dSelect output");
        boolean launch = outShape.numElements() > 0 && x.numElements() > 0;
        if (launch) {
            geo.requireBackwardFits();
        }
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(gpu)) {
            return GpuNormOps.output(gpu, x, outShape, DType.FLOAT32, out -> {
                GpuBuffer selected = out.storage().gpuBuffer();
                if (!launch) {
                    gpu.fill(selected, 0, out.numElements(), 0.0f);
                    return;
                }
                StridedView xv = s.dense(x);
                StridedView yv = s.dense(y);
                StridedView gv = s.dense(g);
                gpu.launch1d(Kernels.MAX_POOL2D_SELECT_F32, out.numElements(),
                        geo.b, geo.c, geo.h, geo.w, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw, geo.ph, geo.pw,
                        geo.dh, geo.dw, xv.offset(), yv.offset(), gv.offset(), 0,
                        xv.buffer(), yv.buffer(), gv.buffer(), selected);
            });
        }
    }

    /**
     * Average-pool backward: each input element sums {@code gy[b, c, oh, ow] / divisor(oh, ow)} over the windows that
     * cover it, in ascending {@code (oh, ow)} order from 0, with the forward's divisor: {@code (he - hs)*(we - ws)}
     * for the window clipped to the padded extent ({@code hs = oh*SH - PH}, {@code he = min(hs + KH, H + PH)}) when
     * {@code countIncludePad}, else the window's in-bounds element count.
     *
     * @param gy         the output gradient {@code [B, C, OH, OW]}
     * @param inputShape the forward input shape {@code [B, C, H, W]}
     * @return a new {@code inputShape} tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException on invalid parameters or a gradient of the wrong shape
     */
    public static Tensor avgPool2dBackward(Tensor gy, Shape inputShape, int[] kernelSize, int[] stride,
                                           int[] padding, boolean ceilMode, boolean countIncludePad) {
        PoolGeometry geo = PoolGeometry.avg(gy, inputShape, kernelSize, stride, padding, ceilMode);
        if (gy.dtype() != DType.FLOAT32) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(gy.backend());
        if (g == null) {
            return null;
        }
        requireKernelElements(inputShape, "avgPool2d input gradient");
        boolean launch = inputShape.numElements() > 0 && gy.numElements() > 0;
        if (launch) {
            geo.requireBackwardFits();
            GpuConvOps.requireAvgWindowFits(geo.oh, geo.sh, geo.kh, geo.h, geo.ph, "avgPool2d height");
            GpuConvOps.requireAvgWindowFits(geo.ow, geo.sw, geo.kw, geo.w, geo.pw, "avgPool2d width");
        }
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            return GpuNormOps.output(g, gy, inputShape, DType.FLOAT32, out -> {
                GpuBuffer gx = out.storage().gpuBuffer();
                if (!launch) {
                    g.fill(gx, 0, out.numElements(), 0.0f);
                    return;
                }
                StridedView gv = s.dense(gy);
                g.launch1d(Kernels.AVG_POOL2D_BACKWARD_F32, out.numElements(),
                        geo.b, geo.c, geo.h, geo.w, geo.oh, geo.ow, geo.kh, geo.kw, geo.sh, geo.sw, geo.ph, geo.pw,
                        countIncludePad ? 1 : 0, gv.offset(), 0, gv.buffer(), gx);
            });
        }
    }

    /**
     * Adaptive average-pool backward: {@code gx[b, c, ih, iw]} sums {@code gy[b, c, oh, ow] / count(oh, ow)} over
     * every window {@code (oh, ow)} holding {@code (ih, iw)} (the forward's windows
     * {@code [floor(oh*H/OH), ceil((oh+1)*H/OH))} overlap when {@code OH} does not divide {@code H}), in ascending
     * {@code (oh, ow)} order from 0.
     *
     * @param gy         the output gradient {@code [B, C, outH, outW]}, {@code outH, outW >= 1}
     * @param inputShape the forward input shape {@code [B, C, H, W]}
     * @return a new {@code inputShape} tensor, or null (dispatcher contract)
     * @throws IllegalArgumentException on a gradient of the wrong shape or window bounds
     *                                  ({@code OH*H + OH - 1}, {@code OW*W + OW - 1}) beyond {@code 2^31-1}
     */
    public static Tensor adaptiveAvgPool2dBackward(Tensor gy, Shape inputShape) {
        PoolGeometry geo = PoolGeometry.adaptive(gy, inputShape);
        if (gy.dtype() != DType.FLOAT32) {
            return null;
        }
        TrainGpu g = TrainGpu.forBackend(gy.backend());
        if (g == null) {
            return null;
        }
        requireKernelElements(inputShape, "adaptiveAvgPool2d input gradient");
        try (GpuNormOps.Staging s = new GpuNormOps.Staging(g)) {
            return GpuNormOps.output(g, gy, inputShape, DType.FLOAT32, out -> {
                if (out.numElements() == 0) {
                    return;
                }
                StridedView gv = s.dense(gy);
                g.launch1d(Kernels.ADAPTIVE_AVG_POOL2D_BACKWARD_F32, out.numElements(),
                        geo.b, geo.c, geo.h, geo.w, geo.oh, geo.ow, gv.offset(), 0,
                        gv.buffer(), out.storage().gpuBuffer());
            });
        }
    }

    // ------------------------------------------------------------------ geometry

    /**
     * The 16 shape scalars of {@code TR_Conv2dWeightGrad_F32} in the kernel's roles: x {@code [B, C, H, W]}, gy
     * {@code [B, OC, OH, OW]}, weight {@code [OC, C/groups, KH, KW]}.
     */
    private static final class ConvGradGeometry {
        final int b, c, h, w, oc, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, groups;

        private ConvGradGeometry(int b, int c, int h, int w, int oc, int oh, int ow, int kh, int kw, int[] stride,
                                 int[] padding, int[] dilation, int groups) {
            this.b = b;
            this.c = c;
            this.h = h;
            this.w = w;
            this.oc = oc;
            this.oh = oh;
            this.ow = ow;
            this.kh = kh;
            this.kw = kw;
            this.sh = stride[0];
            this.sw = stride[1];
            this.ph = padding[0];
            this.pw = padding[1];
            this.dh = dilation[0];
            this.dw = dilation[1];
            this.groups = groups;
        }

        static ConvGradGeometry conv2d(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                       int[] dilation, int groups) {
            requireConvOperands(x, gy, weightShape, stride, padding, dilation, groups, "conv2dWeightGrad");
            Shape in = x.shape();
            int oc = weightShape.dim(0);
            if (oc % groups != 0 || (long) weightShape.dim(1) * groups != in.dim(1)) {
                throw new IllegalArgumentException("conv2dWeightGrad: input channels " + in.dim(1) + " must equal "
                        + "weight " + weightShape + " input channels times groups " + groups + ", and output "
                        + "channels must be divisible by groups");
            }
            int kh = weightShape.dim(2);
            int kw = weightShape.dim(3);
            int oh = GpuConvOps.extent("conv2dWeightGrad output height",
                    GpuConvOps.convExtent((long) in.dim(2) + 2L * padding[0] - (long) dilation[0] * (kh - 1) - 1,
                            stride[0]));
            int ow = GpuConvOps.extent("conv2dWeightGrad output width",
                    GpuConvOps.convExtent((long) in.dim(3) + 2L * padding[1] - (long) dilation[1] * (kw - 1) - 1,
                            stride[1]));
            requireShape(gy, Shape.of(in.dim(0), oc, oh, ow), "conv2dWeightGrad gy");
            return new ConvGradGeometry(in.dim(0), in.dim(1), in.dim(2), in.dim(3), oc, oh, ow, kh, kw, stride,
                    padding, dilation, groups);
        }

        /** This geometry in the shape of the convolution kernels' geometry (the kernel roles are kept). */
        GpuConvOps.ConvGeometry convGeometry() {
            return new GpuConvOps.ConvGeometry(b, c, h, w, oc, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, groups);
        }

        /** The transposed convolution's weight gradient in the kernel's roles (x := gy, gy := the input). */
        static ConvGradGeometry convTranspose2d(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                                int[] outputPadding, int[] dilation, int groups) {
            requireConvOperands(x, gy, weightShape, stride, padding, dilation, groups, "convTranspose2dWeightGrad");
            GpuConvOps.requirePair(outputPadding, "outputPadding", 0);
            Shape in = x.shape();
            if (weightShape.dim(0) != in.dim(1) || in.dim(1) % groups != 0) {
                throw new IllegalArgumentException("convTranspose2dWeightGrad: input channels " + in.dim(1)
                        + " must equal weight " + weightShape + " dim 0 and be divisible by groups " + groups);
            }
            int oc = GpuConvOps.extent("convTranspose2dWeightGrad output channels",
                    (long) weightShape.dim(1) * groups);
            int kh = weightShape.dim(2);
            int kw = weightShape.dim(3);
            int oh = GpuConvOps.extent("convTranspose2dWeightGrad output height", ((long) in.dim(2) - 1) * stride[0]
                    - 2L * padding[0] + (long) dilation[0] * (kh - 1) + outputPadding[0] + 1);
            int ow = GpuConvOps.extent("convTranspose2dWeightGrad output width", ((long) in.dim(3) - 1) * stride[1]
                    - 2L * padding[1] + (long) dilation[1] * (kw - 1) + outputPadding[1] + 1);
            requireShape(gy, Shape.of(in.dim(0), oc, oh, ow), "convTranspose2dWeightGrad gy");
            return new ConvGradGeometry(in.dim(0), oc, oh, ow, in.dim(1), in.dim(2), in.dim(3), kh, kw, stride,
                    padding, dilation, groups);
        }

        private static void requireConvOperands(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                                int[] dilation, int groups, String op) {
            Objects.requireNonNull(x, "x must not be null");
            Objects.requireNonNull(gy, "gy must not be null");
            Objects.requireNonNull(weightShape, "weightShape must not be null");
            requireSameBackend(x, gy);
            GpuConvOps.requirePair(stride, "stride", 1);
            GpuConvOps.requirePair(padding, "padding", 0);
            GpuConvOps.requirePair(dilation, "dilation", 1);
            if (groups < 1) {
                throw new IllegalArgumentException("groups must be positive: " + groups);
            }
            GpuConvOps.requireRank(x.shape(), 4, op + " x [B, C, H, W]");
            GpuConvOps.requireRank(gy.shape(), 4, op + " gy");
            GpuConvOps.requireRank(weightShape, 4, op + " weight shape");
            if (weightShape.dim(2) < 1 || weightShape.dim(3) < 1) {
                throw new IllegalArgumentException(op + " weight shape " + weightShape + " needs a kernel of at "
                        + "least 1x1");
            }
        }
    }

    /** The 14 shape scalars of the pooling kernels; {@code DH = DW = 1} for the average and adaptive pools. */
    private static final class PoolGeometry {
        final int b, c, h, w, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw;

        private PoolGeometry(Shape in, int oh, int ow, int[] kernelSize, int[] stride, int[] padding,
                             int[] dilation) {
            this.b = in.dim(0);
            this.c = in.dim(1);
            this.h = in.dim(2);
            this.w = in.dim(3);
            this.oh = oh;
            this.ow = ow;
            this.kh = kernelSize[0];
            this.kw = kernelSize[1];
            this.sh = stride[0];
            this.sw = stride[1];
            this.ph = padding[0];
            this.pw = padding[1];
            this.dh = dilation[0];
            this.dw = dilation[1];
        }

        /** The max-pool geometry of {@code x}; {@code y} (and {@code gy} when given) must be its output shape. */
        static PoolGeometry max(Tensor x, Tensor y, Tensor gy, int[] kernelSize, int[] stride, int[] padding,
                                int[] dilation, boolean ceilMode, String gradName) {
            Objects.requireNonNull(x, "x must not be null");
            Objects.requireNonNull(y, "y must not be null");
            GpuConvOps.requirePair(kernelSize, "kernelSize", 1);
            GpuConvOps.requirePair(stride, "stride", 1);
            GpuConvOps.requirePair(padding, "padding", 0);
            GpuConvOps.requirePair(dilation, "dilation", 1);
            GpuConvOps.requireRank(x.shape(), 4, "maxPool2d input [B, C, H, W]");
            requireSameBackend(x, y);
            Shape in = x.shape();
            int oh = ConvOps.poolOutputSize(in.dim(2), kernelSize[0], stride[0], padding[0], dilation[0], ceilMode);
            int ow = ConvOps.poolOutputSize(in.dim(3), kernelSize[1], stride[1], padding[1], dilation[1], ceilMode);
            Shape outShape = Shape.of(in.dim(0), in.dim(1), oh, ow);
            requireShape(y, outShape, "maxPool2d output y");
            if (gy != null) {
                requireSameBackend(x, gy);
                requireShape(gy, outShape, "maxPool2d " + gradName);
            }
            return new PoolGeometry(in, oh, ow, kernelSize, stride, padding, dilation);
        }

        static PoolGeometry avg(Tensor gy, Shape inputShape, int[] kernelSize, int[] stride, int[] padding,
                                boolean ceilMode) {
            Objects.requireNonNull(gy, "gy must not be null");
            Objects.requireNonNull(inputShape, "inputShape must not be null");
            GpuConvOps.requirePair(kernelSize, "kernelSize", 1);
            GpuConvOps.requirePair(stride, "stride", 1);
            GpuConvOps.requirePair(padding, "padding", 0);
            GpuConvOps.requireRank(inputShape, 4, "avgPool2d input shape [B, C, H, W]");
            int oh = ConvOps.poolOutputSize(inputShape.dim(2), kernelSize[0], stride[0], padding[0], 1, ceilMode);
            int ow = ConvOps.poolOutputSize(inputShape.dim(3), kernelSize[1], stride[1], padding[1], 1, ceilMode);
            requireShape(gy, Shape.of(inputShape.dim(0), inputShape.dim(1), oh, ow), "avgPool2d gy");
            return new PoolGeometry(inputShape, oh, ow, kernelSize, stride, padding, new int[]{1, 1});
        }

        static PoolGeometry adaptive(Tensor gy, Shape inputShape) {
            Objects.requireNonNull(gy, "gy must not be null");
            Objects.requireNonNull(inputShape, "inputShape must not be null");
            GpuConvOps.requireRank(inputShape, 4, "adaptiveAvgPool2d input shape [B, C, H, W]");
            GpuConvOps.requireRank(gy.shape(), 4, "adaptiveAvgPool2d gy");
            int oh = gy.shape().dim(2);
            int ow = gy.shape().dim(3);
            if (oh < 1 || ow < 1) {
                throw new IllegalArgumentException("adaptiveAvgPool2d gy " + gy.shape() + " needs a positive output "
                        + "size");
            }
            requireShape(gy, Shape.of(inputShape.dim(0), inputShape.dim(1), oh, ow), "adaptiveAvgPool2d gy");
            GpuConvOps.requireAdaptiveBounds(oh, ow, inputShape);
            return new PoolGeometry(inputShape, oh, ow, new int[]{1, 1}, new int[]{1, 1}, new int[]{0, 0},
                    new int[]{1, 1});
        }

        /**
         * The backward kernels compute {@code o*S + k*D} for every window and tap, {@code i + P}, and the window tap
         * index {@code kh*KW + kw} (max pool) or tap count {@code KH*KW} (average pool) in int.
         */
        void requireBackwardFits() {
            GpuConvOps.requireIntSum((long) (oh - 1) * sh, (long) (kh - 1) * dh, "pooling window row position");
            GpuConvOps.requireIntSum((long) (ow - 1) * sw, (long) (kw - 1) * dw, "pooling window column position");
            GpuConvOps.requireIntSum((long) h - 1, ph, "pooling input row plus padding");
            GpuConvOps.requireIntSum((long) w - 1, pw, "pooling input column plus padding");
            GpuConvOps.requireIntSum((long) kh * kw, 0L, "pooling window tap count " + kh + "x" + kw);
        }
    }

    // ------------------------------------------------------------------ validation

    private static int requireKernelElements(Shape shape, String what) {
        long n = shape.numElements();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " " + shape + " has " + n + " elements, beyond the 32-bit "
                    + "kernel limit " + Integer.MAX_VALUE);
        }
        return (int) n;
    }

    private static void requireSameBackend(Tensor a, Tensor b) {
        if (a.backend() != b.backend()) {
            throw new IllegalArgumentException("operands must live on the same backend");
        }
    }

    private static void requireShape(Tensor t, Shape expected, String what) {
        if (!t.shape().equals(expected)) {
            throw new IllegalArgumentException(what + " " + t.shape() + " must have the forward's shape "
                    + expected);
        }
    }

    private static boolean allFloat32(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t.dtype() != DType.FLOAT32) {
                return false;
            }
        }
        return true;
    }
}

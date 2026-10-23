package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ConvGradFunctions;

import java.util.Arrays;
import java.util.Collections;

/**
 * Convolution and pooling ops for the nn/layers modules on NCHW tensors, and their backward primitives. Each method
 * offers the work to {@link GpuConvOps} (forward) or {@link GpuConvGradOps} (backward), which validate every argument
 * (throwing {@link IllegalArgumentException} whether or not the train library is available) and compute on the
 * device; when they decline (the train library is unavailable on the operands' backend, or an operand is not FLOAT32)
 * the CPU reference runs ({@code CpuConvPool}, the layers' host loops and their backward). Results are new contiguous
 * FLOAT32 tensors on the input's backend.
 *
 * <p><b>Recording.</b> The six forward ops record one autograd node exactly when grad mode is on and a tensor operand
 * requires grad ({@link Autograd#needsGrad(Tensor, Tensor, Tensor)}); their result then requires grad and carries the
 * gradient function of {@link ConvGradFunctions}, otherwise it does not require grad. The arguments are
 * validated by the raw computation before anything is recorded, so errors are the same with and without grad. The
 * backward primitives ({@link #conv2dWeightGrad}, {@link #convTranspose2dWeightGrad}, {@link #maxPool2dBackward},
 * {@link #maxPool2dSelect}, {@link #avgPool2dBackward}, {@link #adaptiveAvgPool2dBackward}) record only inside a
 * create-graph pass ({@link Autograd#isCreatingGraph()}) when a differentiable operand requires grad, so the
 * gradients of a create-graph backward can be differentiated again; their gradient functions are the second-order
 * classes of {@link ConvGradFunctions}. Everywhere else they never record and their results do not require grad. The
 * differentiable operands of a primitive are those it is not piecewise constant in: both operands of the weight
 * gradients, and only the gradient operand of the pooling primitives (the forward input and output of max pooling
 * select the routing, like an index tensor, and get no gradient).</p>
 *
 * <p>Both paths have one contract: the argument checks and output sizes of the layers (convolutions: Java integer
 * division, a negative size throws; pooling: PyTorch's window semantics, {@link #poolOutputSize}), and results within
 * the tolerance class {@link GpuConvOps} and {@link GpuConvGradOps} document ({@linkplain
 * com.eonmux.neuralfree.tensor.gpu numerical contract}): convolutions and weight gradients in the reduction class, max
 * pooling and its backward bit-identical or single-rounding, average and adaptive average pooling in the division class
 * and their backward in the division (or, for overlapping windows, reduction) class. The CPU path reads every operand
 * to the host, so its operands must be FLOAT32.</p>
 */
public final class ConvOps {

    private ConvOps() {
    }

    // ------------------------------------------------------------------ convolutions

    /**
     * 2-D convolution ({@code nn/layers/Conv2d}): {@code x [B, C, H, W]}, {@code w [OC, C/groups, KH, KW]},
     * optional {@code bias} of {@code OC} elements; output {@code [B, OC, OH, OW]} with
     * {@code OH = (H + 2*PH - DH*(KH-1) - 1) / SH + 1} (and its W twin). Records
     * {@link ConvGradFunctions.Conv2dGrad}.
     *
     * @throws IllegalArgumentException on inconsistent shapes or parameters, or operands on different backends
     */
    public static Tensor conv2d(Tensor x, Tensor w, Tensor bias, int[] kernel, int[] stride, int[] padding,
                                int[] dilation, int groups) {
        if (!Autograd.needsGrad(x, w, bias)) {
            return conv2dRaw(x, w, bias, kernel, stride, padding, dilation, groups);
        }
        Tensor raw = Autograd.untracked(() -> conv2dRaw(x, w, bias, kernel, stride, padding, dilation, groups));
        return Autograd.record(raw, Arrays.asList(x, w, bias), out -> {
            ConvGradFunctions.Conv2dGrad f = new ConvGradFunctions.Conv2dGrad(x.shape(), w.shape(), shapeOf(bias),
                    stride, padding, dilation, groups, x.requiresGrad(), w.requiresGrad(), requiresGrad(bias));
            f.saveForBackward(x, w);
            return f;
        });
    }

    /**
     * 1-D convolution ({@code nn/layers/Conv1d}): {@code x [B, C, L]}, {@code w [OC, C/groups, K]}, optional
     * {@code bias} of {@code OC} elements; output {@code [B, OC, OL]} with
     * {@code OL = (L + 2*padding - dilation*(K-1) - 1) / stride + 1}. Records {@link ConvGradFunctions.Conv1dGrad}.
     *
     * @throws IllegalArgumentException as {@link #conv2d}
     */
    public static Tensor conv1d(Tensor x, Tensor w, Tensor bias, int kernel, int stride, int padding, int dilation,
                                int groups) {
        if (!Autograd.needsGrad(x, w, bias)) {
            return conv1dRaw(x, w, bias, kernel, stride, padding, dilation, groups);
        }
        Tensor raw = Autograd.untracked(() -> conv1dRaw(x, w, bias, kernel, stride, padding, dilation, groups));
        return Autograd.record(raw, Arrays.asList(x, w, bias), out -> {
            ConvGradFunctions.Conv1dGrad f = new ConvGradFunctions.Conv1dGrad(x.shape(), w.shape(), shapeOf(bias),
                    stride, padding, dilation, groups, x.requiresGrad(), w.requiresGrad(), requiresGrad(bias));
            f.saveForBackward(x, w);
            return f;
        });
    }

    /**
     * 2-D transposed convolution ({@code nn/layers/ConvTranspose2d}, PyTorch semantics): {@code x [B, C, H, W]},
     * {@code w [C, OC/groups, KH, KW]}, optional {@code bias} of {@code OC} elements; output {@code [B, OC, OH, OW]}
     * with {@code OH = (H-1)*SH - 2*PH + DH*(KH-1) + outputPadding[0] + 1} (and its W twin). Records
     * {@link ConvGradFunctions.ConvTranspose2dGrad}.
     *
     * @throws IllegalArgumentException as {@link #conv2d}, and when an output padding is not smaller than its stride
     *                                  or its dilation ({@link #requireTransposeOutputPadding})
     */
    public static Tensor convTranspose2d(Tensor x, Tensor w, Tensor bias, int[] kernel, int[] stride, int[] padding,
                                         int[] outputPadding, int[] dilation, int groups) {
        requireTransposeOutputPadding(outputPadding, stride, dilation);
        if (!Autograd.needsGrad(x, w, bias)) {
            return convTranspose2dRaw(x, w, bias, kernel, stride, padding, outputPadding, dilation, groups);
        }
        Tensor raw = Autograd.untracked(
                () -> convTranspose2dRaw(x, w, bias, kernel, stride, padding, outputPadding, dilation, groups));
        return Autograd.record(raw, Arrays.asList(x, w, bias), out -> {
            ConvGradFunctions.ConvTranspose2dGrad f = new ConvGradFunctions.ConvTranspose2dGrad(x.shape(), w.shape(),
                    shapeOf(bias), stride, padding, outputPadding, dilation, groups, x.requiresGrad(),
                    w.requiresGrad(), requiresGrad(bias));
            f.saveForBackward(x, w);
            return f;
        });
    }

    // ------------------------------------------------------------------ pooling

    /**
     * Max pooling ({@code nn/layers/MaxPool2d}, PyTorch {@code max_pool2d}): {@code x [B, C, H, W]}, output
     * {@code [B, C, OH, OW]} with {@code OH = poolOutputSize(H, KH, SH, PH, DH, ceilMode)} (and its W twin); window
     * {@code (oh, ow)} holds the taps {@code (oh*SH - PH + kh*DH, ow*SW - PW + kw*DW)}, and the output is
     * {@code Math.max} over its in-bounds taps in {@code (kh, kw)} order, {@code -inf} for a window without input
     * elements. Records {@link ConvGradFunctions.MaxPool2dGrad}, which saves the input and the output.
     *
     * @throws IllegalArgumentException on invalid parameters, a rank other than 4, a padding above half the effective
     *                                  kernel, or an output size below 1 ({@link #poolOutputSize})
     */
    public static Tensor maxPool2d(Tensor x, int[] kernel, int[] stride, int[] padding, int[] dilation,
                                   boolean ceilMode) {
        if (!Autograd.needsGrad(x)) {
            return maxPool2dRaw(x, kernel, stride, padding, dilation, ceilMode);
        }
        Tensor raw = Autograd.untracked(() -> maxPool2dRaw(x, kernel, stride, padding, dilation, ceilMode));
        return Autograd.record(raw, Collections.singletonList(x), out -> {
            ConvGradFunctions.MaxPool2dGrad f = new ConvGradFunctions.MaxPool2dGrad(kernel, stride, padding, dilation,
                    ceilMode);
            f.saveForBackward(x, out);
            return f;
        });
    }

    /**
     * Average pooling ({@code nn/layers/AvgPool2d}, PyTorch {@code avg_pool2d} without a divisor override):
     * {@code x [B, C, H, W]}, output {@code [B, C, OH, OW]} with {@code OH = poolOutputSize(H, KH, SH, PH, 1,
     * ceilMode)} (and its W twin). Window {@code (oh, ow)} starts at {@code hs = oh*SH - PH} and ends at
     * {@code he = min(hs + KH, H + PH)} (columns likewise), so it never reaches past the padded edge; the output is
     * the sum of the input elements in {@code [max(hs, 0), min(he, H)) x [max(ws, 0), min(we, W))} in row-major order
     * divided by {@code (he - hs)*(we - ws)} when {@code countIncludePad}, otherwise by the number of those elements,
     * and 0 for a window without input elements. Records {@link ConvGradFunctions.AvgPool2dGrad}.
     *
     * @throws IllegalArgumentException on invalid parameters, a rank other than 4, a padding above half the kernel,
     *                                  or an output size below 1 ({@link #poolOutputSize})
     */
    public static Tensor avgPool2d(Tensor x, int[] kernel, int[] stride, int[] padding, boolean ceilMode,
                                   boolean countIncludePad) {
        if (!Autograd.needsGrad(x)) {
            return avgPool2dRaw(x, kernel, stride, padding, ceilMode, countIncludePad);
        }
        Tensor raw = Autograd.untracked(() -> avgPool2dRaw(x, kernel, stride, padding, ceilMode, countIncludePad));
        return Autograd.record(raw, Collections.singletonList(x), out -> new ConvGradFunctions.AvgPool2dGrad(
                x.shape(), kernel, stride, padding, ceilMode, countIncludePad));
    }

    /**
     * Adaptive average pooling ({@code nn/layers/AdaptiveAvgPool2d}, PyTorch {@code adaptive_avg_pool2d}):
     * {@code x [B, C, H, W]}, output {@code [B, C, outH, outW]}; output row {@code oh} averages input rows
     * {@code [floor(oh*H/outH), ceil((oh+1)*H/outH))} (columns likewise). Neighbouring windows overlap when
     * {@code outH} does not divide {@code H}, and every window holds at least one row when {@code H >= 1}; a window
     * without input elements (an empty input plane) yields 0. Records {@link ConvGradFunctions.AdaptiveAvgPool2dGrad}.
     *
     * @throws IllegalArgumentException when {@code outH} or {@code outW} is not positive, the rank is not 4, or
     *                                  {@code outH*H + outH - 1} or {@code outW*W + outW - 1} exceeds {@code 2^31-1}
     */
    public static Tensor adaptiveAvgPool2d(Tensor x, int outH, int outW) {
        if (!Autograd.needsGrad(x)) {
            return adaptiveAvgPool2dRaw(x, outH, outW);
        }
        Tensor raw = Autograd.untracked(() -> adaptiveAvgPool2dRaw(x, outH, outW));
        return Autograd.record(raw, Collections.singletonList(x),
                out -> new ConvGradFunctions.AdaptiveAvgPool2dGrad(x.shape()));
    }

    // ------------------------------------------------------------------ pooling geometry

    /**
     * Output extent of one spatial dimension of {@link #maxPool2d} and {@link #avgPool2d} (which passes
     * {@code dilation = 1}), with PyTorch's argument checks ({@code pool2d_shape_check}) and output rule
     * ({@code pooling_output_shape}). With the effective window {@code E = dilation*(kernel-1) + 1}:
     * {@code out = floor((input + 2*padding - E) / stride) + 1}, or in ceil mode
     * {@code ceil((input + 2*padding - E) / stride) + 1}, minus one when the last window would start inside the
     * right padding ({@code (out-1)*stride >= input + padding}), so every window starts inside the input or the left
     * padding. Divisions round toward negative infinity.
     *
     * @param input    the input extent, {@code >= 0}
     * @param kernel   the kernel extent, {@code >= 1}
     * @param stride   the stride, {@code >= 1}
     * @param padding  the padding on each side, in {@code [0, E/2]} (integer division)
     * @param dilation the tap spacing, {@code >= 1}
     * @param ceilMode whether the division rounds up
     * @return the output extent, in {@code [1, 2^31-1]}
     * @throws IllegalArgumentException when an argument is outside its range or the output extent is below 1
     */
    public static int poolOutputSize(int input, int kernel, int stride, int padding, int dilation,
                                     boolean ceilMode) {
        if (input < 0 || kernel < 1 || stride < 1 || padding < 0 || dilation < 1) {
            throw new IllegalArgumentException("pooling needs input >= 0, kernel >= 1, stride >= 1, padding >= 0 and "
                    + "dilation >= 1, got input " + input + ", kernel " + kernel + ", stride " + stride
                    + ", padding " + padding + ", dilation " + dilation);
        }
        long window = (long) dilation * (kernel - 1) + 1;
        if (padding > window / 2) {
            throw new IllegalArgumentException("pooling padding " + padding + " must be at most half of the "
                    + "effective kernel size " + window + " (kernel " + kernel + ", dilation " + dilation + ")");
        }
        long span = (long) input + 2L * padding - window;
        long out = (ceilMode ? -Math.floorDiv(-span, stride) : Math.floorDiv(span, stride)) + 1;
        if (ceilMode && (out - 1) * stride >= (long) input + padding) {
            out--;
        }
        if (out < 1 || out > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("pooling output size " + out + " must be in [1, " + Integer.MAX_VALUE
                    + "]: input " + input + ", kernel " + kernel + ", stride " + stride + ", padding " + padding
                    + ", dilation " + dilation + ", ceilMode " + ceilMode);
        }
        return (int) out;
    }

    /**
     * First input index {@code floor(o*in/out)} of adaptive window {@code o} of {@code out} over {@code in} elements.
     * Callers guarantee {@code out*in} fits in int.
     */
    static int adaptiveStart(int o, int in, int out) {
        return (o * in) / out;
    }

    /**
     * End (exclusive) {@code ceil((o+1)*in/out)} of adaptive window {@code o} of {@code out} over {@code in}
     * elements. Callers guarantee {@code out*in + out - 1} fits in int.
     */
    static int adaptiveEnd(int o, int in, int out) {
        return ((o + 1) * in + out - 1) / out;
    }

    // ------------------------------------------------------------------ backward primitives

    /**
     * Weight gradient of {@link #conv2d}: {@code gw[oc, ic, kh, kw] = sum over b, oh, ow of gy[b, oc, oh, ow] *
     * x[b, g*Cg + ic, oh*SH + kh*DH - PH, ow*SW + kw*DW - PW]} (in-bounds taps). Records
     * {@link ConvGradFunctions.Conv2dWeightGradGrad} inside a create-graph pass only.
     *
     * @param x           the forward input {@code [B, C, H, W]}
     * @param gy          the output gradient {@code [B, OC, OH, OW]}
     * @param weightShape the weight shape {@code [OC, C/groups, KH, KW]}, which also gives the kernel size
     * @return a new FLOAT32 tensor of {@code weightShape}
     * @throws IllegalArgumentException on shapes or parameters inconsistent with the forward
     */
    public static Tensor conv2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                          int[] dilation, int groups) {
        if (!recordsBackward(x, gy)) {
            return conv2dWeightGradRaw(x, gy, weightShape, stride, padding, dilation, groups);
        }
        Tensor raw = Autograd.untracked(
                () -> conv2dWeightGradRaw(x, gy, weightShape, stride, padding, dilation, groups));
        return Autograd.record(raw, Arrays.asList(x, gy), out -> {
            ConvGradFunctions.Conv2dWeightGradGrad f = new ConvGradFunctions.Conv2dWeightGradGrad(x.shape(),
                    weightShape, stride, padding, dilation, groups, x.requiresGrad(), gy.requiresGrad());
            f.saveForBackward(x, gy);
            return f;
        });
    }

    /**
     * Weight gradient of {@link #convTranspose2d}: {@code gw[cin, ocg, kh, kw] = sum over b, ih, iw of
     * x[b, cin, ih, iw] * gy[b, g*OCg + ocg, ih*SH + kh*DH - PH, iw*SW + kw*DW - PW]} (output positions inside
     * {@code gy}). Records {@link ConvGradFunctions.ConvTranspose2dWeightGradGrad} inside a create-graph pass only.
     *
     * @param x           the forward input {@code [B, C, H, W]}
     * @param gy          the output gradient {@code [B, OC, OH, OW]}, OH and OW as the forward computes them
     * @param weightShape the weight shape {@code [C, OC/groups, KH, KW]}, which also gives the kernel size
     * @return a new FLOAT32 tensor of {@code weightShape}
     * @throws IllegalArgumentException on shapes or parameters inconsistent with the forward
     */
    public static Tensor convTranspose2dWeightGrad(Tensor x, Tensor gy, Shape weightShape, int[] stride,
                                                   int[] padding, int[] outputPadding, int[] dilation, int groups) {
        if (!recordsBackward(x, gy)) {
            return convTranspose2dWeightGradRaw(x, gy, weightShape, stride, padding, outputPadding, dilation, groups);
        }
        Tensor raw = Autograd.untracked(() -> convTranspose2dWeightGradRaw(x, gy, weightShape, stride, padding,
                outputPadding, dilation, groups));
        return Autograd.record(raw, Arrays.asList(x, gy), out -> {
            ConvGradFunctions.ConvTranspose2dWeightGradGrad f = new ConvGradFunctions.ConvTranspose2dWeightGradGrad(
                    x.shape(), weightShape, stride, padding, outputPadding, dilation, groups, x.requiresGrad(),
                    gy.requiresGrad());
            f.saveForBackward(x, gy);
            return f;
        });
    }

    /**
     * Input gradient of {@link #maxPool2d}: the gradient of each window goes to its first in-bounds tap in
     * {@code (kh, kw)} order whose value is Java-{@code ==} the window maximum {@code y}; a NaN maximum routes nowhere.
     * Records {@link ConvGradFunctions.MaxPool2dBackwardGrad} over {@code gy} inside a create-graph pass only.
     *
     * @param x  the forward input {@code [B, C, H, W]}
     * @param y  the forward output of {@code x} with this geometry
     * @param gy the output gradient, shaped as {@code y}
     * @return a new FLOAT32 tensor shaped as {@code x}
     * @throws IllegalArgumentException on invalid parameters, or operands whose shapes are not the forward's
     */
    public static Tensor maxPool2dBackward(Tensor x, Tensor y, Tensor gy, int[] kernel, int[] stride, int[] padding,
                                           int[] dilation, boolean ceilMode) {
        if (!recordsBackward(gy)) {
            return maxPool2dBackwardRaw(x, y, gy, kernel, stride, padding, dilation, ceilMode);
        }
        Tensor raw = Autograd.untracked(
                () -> maxPool2dBackwardRaw(x, y, gy, kernel, stride, padding, dilation, ceilMode));
        return Autograd.record(raw, Collections.singletonList(gy), out -> {
            ConvGradFunctions.MaxPool2dBackwardGrad f = new ConvGradFunctions.MaxPool2dBackwardGrad(kernel, stride,
                    padding, dilation, ceilMode);
            f.saveForBackward(x, y);
            return f;
        });
    }

    /**
     * The adjoint of {@link #maxPool2dBackward} in its gradient: {@code out[b, c, oh, ow] = g[b, c, ih, iw]} at the
     * tap window {@code (oh, ow)} routes its gradient to, {@code +0} when it routes nowhere. Records
     * {@link ConvGradFunctions.MaxPool2dSelectGrad} over {@code g} inside a create-graph pass only.
     *
     * @param x the forward input {@code [B, C, H, W]}
     * @param y the forward output of {@code x} with this geometry
     * @param g a tensor shaped as {@code x}
     * @return a new FLOAT32 tensor shaped as {@code y}
     * @throws IllegalArgumentException on invalid parameters, or operands whose shapes are not the forward's
     */
    public static Tensor maxPool2dSelect(Tensor x, Tensor y, Tensor g, int[] kernel, int[] stride, int[] padding,
                                         int[] dilation, boolean ceilMode) {
        if (!recordsBackward(g)) {
            return maxPool2dSelectRaw(x, y, g, kernel, stride, padding, dilation, ceilMode);
        }
        Tensor raw = Autograd.untracked(
                () -> maxPool2dSelectRaw(x, y, g, kernel, stride, padding, dilation, ceilMode));
        return Autograd.record(raw, Collections.singletonList(g), out -> {
            ConvGradFunctions.MaxPool2dSelectGrad f = new ConvGradFunctions.MaxPool2dSelectGrad(kernel, stride,
                    padding, dilation, ceilMode);
            f.saveForBackward(x, y);
            return f;
        });
    }

    /**
     * Input gradient of {@link #avgPool2d}: each input element sums {@code gy / divisor} over the windows covering
     * it in ascending {@code (oh, ow)} order, with the forward's per-window divisor. Records
     * {@link ConvGradFunctions.AvgPool2dBackwardGrad} inside a create-graph pass only.
     *
     * @param gy         the output gradient {@code [B, C, OH, OW]}
     * @param inputShape the forward input shape {@code [B, C, H, W]}
     * @return a new FLOAT32 tensor of {@code inputShape} on {@code gy}'s backend
     * @throws IllegalArgumentException on invalid parameters or a gradient whose shape is not the forward's
     */
    public static Tensor avgPool2dBackward(Tensor gy, Shape inputShape, int[] kernel, int[] stride, int[] padding,
                                           boolean ceilMode, boolean countIncludePad) {
        if (!recordsBackward(gy)) {
            return avgPool2dBackwardRaw(gy, inputShape, kernel, stride, padding, ceilMode, countIncludePad);
        }
        Tensor raw = Autograd.untracked(
                () -> avgPool2dBackwardRaw(gy, inputShape, kernel, stride, padding, ceilMode, countIncludePad));
        return Autograd.record(raw, Collections.singletonList(gy), out -> new ConvGradFunctions.AvgPool2dBackwardGrad(
                inputShape, kernel, stride, padding, ceilMode, countIncludePad));
    }

    /**
     * Input gradient of {@link #adaptiveAvgPool2d}: each input element sums {@code gy / count} over every window
     * holding it (one or more, since the windows may overlap) in ascending {@code (oh, ow)} order, with each window's
     * element count. Records {@link ConvGradFunctions.AdaptiveAvgPool2dBackwardGrad} inside a create-graph pass
     * only.
     *
     * @param gy         the output gradient {@code [B, C, outH, outW]}
     * @param inputShape the forward input shape {@code [B, C, H, W]}
     * @return a new FLOAT32 tensor of {@code inputShape} on {@code gy}'s backend
     * @throws IllegalArgumentException on a gradient whose shape is not the forward's
     */
    public static Tensor adaptiveAvgPool2dBackward(Tensor gy, Shape inputShape) {
        if (!recordsBackward(gy)) {
            return adaptiveAvgPool2dBackwardRaw(gy, inputShape);
        }
        Tensor raw = Autograd.untracked(() -> adaptiveAvgPool2dBackwardRaw(gy, inputShape));
        return Autograd.record(raw, Collections.singletonList(gy),
                out -> new ConvGradFunctions.AdaptiveAvgPool2dBackwardGrad(inputShape, gy.shape()));
    }

    // ------------------------------------------------------------------ raw computations

    private static Tensor conv2dRaw(Tensor x, Tensor w, Tensor bias, int[] kernel, int[] stride, int[] padding,
                                    int[] dilation, int groups) {
        Tensor gpu = GpuConvOps.conv2dForward(x, w, bias, kernel, stride, padding, dilation, groups);
        return gpu != null ? gpu : CpuConvPool.conv2d(x, w, bias, kernel, stride, padding, dilation, groups);
    }

    /**
     * PyTorch's transposed-convolution rule: every {@code outputPadding[i]} is smaller than {@code stride[i]} or
     * {@code dilation[i]}, so the padding only adds rows the strided or dilated taps can reach. Pairs that are not
     * {@code int[2]} are reported by the argument checks of the computation.
     *
     * @throws IllegalArgumentException when {@code outputPadding[i] >= stride[i]} and {@code >= dilation[i]}
     */
    public static void requireTransposeOutputPadding(int[] outputPadding, int[] stride, int[] dilation) {
        if (outputPadding == null || stride == null || dilation == null || outputPadding.length != 2
                || stride.length != 2 || dilation.length != 2) {
            return;
        }
        for (int i = 0; i < 2; i++) {
            if (outputPadding[i] >= stride[i] && outputPadding[i] >= dilation[i]) {
                throw new IllegalArgumentException("output padding must be smaller than either stride or dilation, "
                        + "got outputPadding " + Arrays.toString(outputPadding) + ", stride "
                        + Arrays.toString(stride) + ", dilation " + Arrays.toString(dilation));
            }
        }
    }

    private static Tensor conv1dRaw(Tensor x, Tensor w, Tensor bias, int kernel, int stride, int padding,
                                    int dilation, int groups) {
        Tensor gpu = GpuConvOps.conv1dForward(x, w, bias, kernel, stride, padding, dilation, groups);
        return gpu != null ? gpu : CpuConvPool.conv1d(x, w, bias, kernel, stride, padding, dilation, groups);
    }

    private static Tensor convTranspose2dRaw(Tensor x, Tensor w, Tensor bias, int[] kernel, int[] stride,
                                             int[] padding, int[] outputPadding, int[] dilation, int groups) {
        Tensor gpu = GpuConvOps.convTranspose2dForward(x, w, bias, kernel, stride, padding, outputPadding, dilation,
                groups);
        return gpu != null ? gpu
                : CpuConvPool.convTranspose2d(x, w, bias, kernel, stride, padding, outputPadding, dilation, groups);
    }

    private static Tensor maxPool2dRaw(Tensor x, int[] kernel, int[] stride, int[] padding, int[] dilation,
                                       boolean ceilMode) {
        Tensor gpu = GpuConvOps.maxPool2dForward(x, kernel, stride, padding, dilation, ceilMode);
        return gpu != null ? gpu : CpuConvPool.maxPool2d(x, kernel, stride, padding, dilation, ceilMode);
    }

    private static Tensor avgPool2dRaw(Tensor x, int[] kernel, int[] stride, int[] padding, boolean ceilMode,
                                       boolean countIncludePad) {
        Tensor gpu = GpuConvOps.avgPool2dForward(x, kernel, stride, padding, ceilMode, countIncludePad);
        return gpu != null ? gpu : CpuConvPool.avgPool2d(x, kernel, stride, padding, ceilMode, countIncludePad);
    }

    private static Tensor adaptiveAvgPool2dRaw(Tensor x, int outH, int outW) {
        Tensor gpu = GpuConvOps.adaptiveAvgPool2d(x, outH, outW);
        return gpu != null ? gpu : CpuConvPool.adaptiveAvgPool2d(x, outH, outW);
    }

    private static Tensor conv2dWeightGradRaw(Tensor x, Tensor gy, Shape weightShape, int[] stride, int[] padding,
                                              int[] dilation, int groups) {
        Tensor gpu = GpuConvGradOps.conv2dWeightGrad(x, gy, weightShape, stride, padding, dilation, groups);
        return gpu != null ? gpu
                : CpuConvPool.conv2dWeightGrad(x, gy, weightShape, stride, padding, dilation, groups);
    }

    private static Tensor convTranspose2dWeightGradRaw(Tensor x, Tensor gy, Shape weightShape, int[] stride,
                                                       int[] padding, int[] outputPadding, int[] dilation,
                                                       int groups) {
        Tensor gpu = GpuConvGradOps.convTranspose2dWeightGrad(x, gy, weightShape, stride, padding, outputPadding,
                dilation, groups);
        return gpu != null ? gpu
                : CpuConvPool.convTranspose2dWeightGrad(x, gy, weightShape, stride, padding, dilation, groups);
    }

    private static Tensor maxPool2dBackwardRaw(Tensor x, Tensor y, Tensor gy, int[] kernel, int[] stride,
                                               int[] padding, int[] dilation, boolean ceilMode) {
        Tensor gpu = GpuConvGradOps.maxPool2dBackward(x, y, gy, kernel, stride, padding, dilation, ceilMode);
        return gpu != null ? gpu : CpuConvPool.maxPool2dBackward(x, y, gy, kernel, stride, padding, dilation);
    }

    private static Tensor maxPool2dSelectRaw(Tensor x, Tensor y, Tensor g, int[] kernel, int[] stride, int[] padding,
                                             int[] dilation, boolean ceilMode) {
        Tensor gpu = GpuConvGradOps.maxPool2dSelect(x, y, g, kernel, stride, padding, dilation, ceilMode);
        return gpu != null ? gpu : CpuConvPool.maxPool2dSelect(x, y, g, kernel, stride, padding, dilation);
    }

    private static Tensor avgPool2dBackwardRaw(Tensor gy, Shape inputShape, int[] kernel, int[] stride,
                                               int[] padding, boolean ceilMode, boolean countIncludePad) {
        Tensor gpu = GpuConvGradOps.avgPool2dBackward(gy, inputShape, kernel, stride, padding, ceilMode,
                countIncludePad);
        return gpu != null ? gpu
                : CpuConvPool.avgPool2dBackward(gy, inputShape, kernel, stride, padding, countIncludePad);
    }

    private static Tensor adaptiveAvgPool2dBackwardRaw(Tensor gy, Shape inputShape) {
        Tensor gpu = GpuConvGradOps.adaptiveAvgPool2dBackward(gy, inputShape);
        return gpu != null ? gpu : CpuConvPool.adaptiveAvgPool2dBackward(gy, inputShape);
    }

    /**
     * Whether a backward primitive records: inside a create-graph pass, with grad mode on and a differentiable
     * operand requiring grad (null operands are ignored).
     */
    private static boolean recordsBackward(Tensor a) {
        return Autograd.isCreatingGraph() && Autograd.needsGrad(a);
    }

    private static boolean recordsBackward(Tensor a, Tensor b) {
        return Autograd.isCreatingGraph() && Autograd.needsGrad(a, b);
    }

    private static Shape shapeOf(Tensor t) {
        return t == null ? null : t.shape();
    }

    private static boolean requiresGrad(Tensor t) {
        return t != null && t.requiresGrad();
    }
}

package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.ConvOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Backward functions of the convolution and pooling ops of {@link ConvOps}. Every backward is a composition of
 * {@link ConvOps} and {@link TensorOps} calls and tensor views, so it runs on the device when the train library is
 * available on the gradient's backend and on the CPU reference path otherwise, and follows the ownership contract of
 * {@link GradFunctions}: {@code gradOutput} is never closed, every temporary is closed, also when the backward throws,
 * and each returned gradient is a new tensor owned by the caller.
 *
 * <p>Input gradients of the convolutions are the forward convolutions with swapped roles, through the public
 * {@link ConvOps#convTranspose2d} and {@link ConvOps#conv2d}; weight gradients use the backward primitives
 * {@link ConvOps#conv2dWeightGrad} and {@link ConvOps#convTranspose2dWeightGrad}; bias gradients are
 * {@code TensorOps.sum(gy, 0, 2, 3)} ({@code 0, 2} for the 1-D convolution). Pooling gradients use the pooling
 * backward primitives. Every gradient has the shape of its input.</p>
 *
 * <p><b>Create graph.</b> Every class supports it ({@code supportsCreateGraph()} is true): each backward is composed
 * of recording forwards, recording views and {@link TensorOps} reductions, and of the {@link ConvOps} backward
 * primitives, which record inside a create-graph pass with the second-order classes of this file as their gradient
 * functions:</p>
 * <ul>
 *   <li>{@code conv2dWeightGrad(x, gy)} is bilinear: {@code d/dx = convTranspose2d(gy, g2)} (the conv input gradient
 *       with {@code g2} as the weight) and {@code d/dgy = conv2d(x, g2)} ({@link Conv2dWeightGradGrad});
 *       {@code convTranspose2dWeightGrad(x, gy)} likewise with the roles swapped
 *       ({@link ConvTranspose2dWeightGradGrad});</li>
 *   <li>the pooling backwards are linear in {@code gy} and piecewise constant in the forward tensors:
 *       {@code d/dgy maxPool2dBackward = maxPool2dSelect} and back ({@link MaxPool2dBackwardGrad},
 *       {@link MaxPool2dSelectGrad}), {@code d/dgy avgPool2dBackward = avgPool2d} ({@link AvgPool2dBackwardGrad}),
 *       {@code d/dgy adaptiveAvgPool2dBackward = adaptiveAvgPool2d} ({@link AdaptiveAvgPool2dBackwardGrad}).</li>
 * </ul>
 * <p>No backward detaches a tensor that may require grad (rule G4); the max-pooling routing operands are read without a
 * path to their origin because they are not differentiated.</p>
 */
public final class ConvGradFunctions {

    private ConvGradFunctions() {
    }

    // ------------------------------------------------------------------ convolutions

    /**
     * {@code conv2d(x, w, bias)}, recorded with the inputs {@code (x, w, bias)}:
     * {@code gx = convTranspose2d(gy, w, null, K, S, P, outPad, D, groups)} with
     * {@code outPad = (H + 2P - D(K-1) - 1) mod S} per spatial dim, which makes its output exactly {@code H x W}
     * (when that span is negative the forward output has one row, the transposed output has {@code H - span} rows and
     * is cut to the first {@code H}); {@code gw = conv2dWeightGrad(x, gy, w.shape, S, P, D, groups)};
     * {@code gb = sum(gy, 0, 2, 3)} in the bias's shape. Saved: the input and the weight. Tolerance: reduction class
     * ({@code K = OC/groups * KH * KW} terms for gx, {@code B * OH * OW} for gw and gb).
     */
    public static class Conv2dGrad extends GradFunctions.SavingGradFn {
        private final ConvGeometry geo;
        private final boolean needInput;
        private final boolean needWeight;
        private final boolean needBias;

        /**
         * @param biasShape the bias shape, or null without a bias
         * @param needInput whether {@code x} requires a gradient (likewise {@code needWeight}, {@code needBias})
         */
        public Conv2dGrad(Shape inputShape, Shape weightShape, Shape biasShape, int[] stride, int[] padding,
                          int[] dilation, int groups, boolean needInput, boolean needWeight, boolean needBias) {
            this.geo = ConvGeometry.of(inputShape, weightShape, biasShape, stride, padding, null, dilation, groups,
                    false);
            this.needInput = needInput;
            this.needWeight = needWeight;
            this.needBias = needBias && biasShape != null;
        }

        /** Saves the forward input and weight, which must have the constructor's shapes. */
        public void saveForBackward(Tensor input, Tensor weight) {
            geo.requireOperands(input, weight);
            save(input, weight);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            geo.requireGradOutput(gradOutput);
            return gradients(
                    () -> needInput ? conv2dInputGrad(gradOutput, saved(1), geo) : null,
                    () -> needWeight ? ConvOps.conv2dWeightGrad(saved(0), gradOutput, geo.weightShape, geo.stride,
                            geo.padding, geo.dilation, geo.groups) : null,
                    () -> needBias ? biasGrad(gradOutput, geo.biasShape, 0, 2, 3) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code conv1d(x, w, bias)}, recorded with the inputs {@code (x, w, bias)}: {@link Conv2dGrad}'s formulas on the
     * {@code unsqueeze(2)} views {@code x [B, C, 1, L]}, {@code w [OC, C/groups, 1, K]} and {@code gy [B, OC, 1, OL]}
     * with {@code H = KH = SH = DH = 1} and {@code PH = 0}, squeezed back; {@code gb = sum(gy, 0, 2)}. Saved: the
     * input and the weight. Tolerance: as {@link Conv2dGrad}.
     */
    public static class Conv1dGrad extends GradFunctions.SavingGradFn {
        private final ConvGeometry geo;
        private final Shape inputShape;
        private final Shape weightShape;
        private final boolean needInput;
        private final boolean needWeight;
        private final boolean needBias;

        /**
         * @param inputShape  the input shape {@code [B, C, L]}
         * @param weightShape the weight shape {@code [OC, C/groups, K]}
         * @param biasShape   the bias shape, or null without a bias
         */
        public Conv1dGrad(Shape inputShape, Shape weightShape, Shape biasShape, int stride, int padding, int dilation,
                          int groups, boolean needInput, boolean needWeight, boolean needBias) {
            this.inputShape = requireRank(inputShape, 3, "conv1d input shape");
            this.weightShape = requireRank(weightShape, 3, "conv1d weight shape");
            this.geo = ConvGeometry.of(unsqueezed(inputShape), unsqueezed(weightShape), biasShape,
                    new int[]{1, stride}, new int[]{0, padding}, null, new int[]{1, dilation}, groups, false);
            this.needInput = needInput;
            this.needWeight = needWeight;
            this.needBias = needBias && biasShape != null;
        }

        /** Saves the forward input and weight, which must have the constructor's shapes. */
        public void saveForBackward(Tensor input, Tensor weight) {
            requireShape(input, inputShape, "conv1d input");
            requireShape(weight, weightShape, "conv1d weight");
            save(input, weight);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            if (gradOutput.shape().rank() != 3) {
                throw new IllegalArgumentException("conv1d gradient " + gradOutput.shape() + " must have rank 3");
            }
            try (Tensor gy = gradOutput.unsqueeze(2)) {
                geo.requireGradOutput(gy);
                return gradients(
                        () -> needInput ? inputGrad(gy) : null,
                        () -> needWeight ? weightGrad(gy) : null,
                        () -> needBias ? biasGrad(gradOutput, geo.biasShape, 0, 2) : null);
            }
        }

        private Tensor inputGrad(Tensor gy) {
            try (Tensor w = saved(1).unsqueeze(2)) {
                return squeezeHeight(conv2dInputGrad(gy, w, geo));
            }
        }

        private Tensor weightGrad(Tensor gy) {
            try (Tensor x = saved(0).unsqueeze(2)) {
                return squeezeHeight(ConvOps.conv2dWeightGrad(x, gy, geo.weightShape, geo.stride, geo.padding,
                        geo.dilation, geo.groups));
            }
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }

        private static Shape unsqueezed(Shape shape) {
            return Shape.of(shape.dim(0), shape.dim(1), 1, shape.dim(2));
        }

        /** {@code t} {@code [N, M, 1, L]} as {@code [N, M, L]}; {@code t} is closed. */
        private static Tensor squeezeHeight(Tensor t) {
            try {
                return t.squeeze(2);
            } finally {
                t.close();
            }
        }
    }

    /**
     * {@code convTranspose2d(x, w, bias)} with output padding {@code op}, recorded with the inputs
     * {@code (x, w, bias)}: {@code gx = conv2d(gy, w, null, K, S, P, D, groups)}, whose output has
     * {@code H + floor(op/S)} rows (and the W twin); when that exceeds the input, the rows and columns past
     * {@code H x W} (which belong to input positions that do not exist) are cut by slice views and a contiguous copy,
     * so gx always has the input's shape; {@code gw = convTranspose2dWeightGrad(x, gy, w.shape, ...)};
     * {@code gb = sum(gy, 0, 2, 3)} in the bias's shape. Saved: the input and the weight. Tolerance: reduction class
     * ({@code K = C/groups * KH * KW} terms for gx, {@code B * H * W} for gw, {@code B * OH * OW} for gb).
     */
    public static class ConvTranspose2dGrad extends GradFunctions.SavingGradFn {
        private final ConvGeometry geo;
        private final boolean needInput;
        private final boolean needWeight;
        private final boolean needBias;

        /**
         * @param biasShape the bias shape, or null without a bias
         * @param needInput whether {@code x} requires a gradient (likewise {@code needWeight}, {@code needBias})
         */
        public ConvTranspose2dGrad(Shape inputShape, Shape weightShape, Shape biasShape, int[] stride, int[] padding,
                                   int[] outputPadding, int[] dilation, int groups, boolean needInput,
                                   boolean needWeight, boolean needBias) {
            this.geo = ConvGeometry.of(inputShape, weightShape, biasShape, stride, padding,
                    Objects.requireNonNull(outputPadding, "outputPadding"), dilation, groups, true);
            this.needInput = needInput;
            this.needWeight = needWeight;
            this.needBias = needBias && biasShape != null;
        }

        /** Saves the forward input and weight, which must have the constructor's shapes. */
        public void saveForBackward(Tensor input, Tensor weight) {
            geo.requireOperands(input, weight);
            save(input, weight);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            geo.requireGradOutput(gradOutput);
            return gradients(
                    () -> needInput ? convTranspose2dInputGrad(gradOutput, saved(1), geo) : null,
                    () -> needWeight ? ConvOps.convTranspose2dWeightGrad(saved(0), gradOutput, geo.weightShape,
                            geo.stride, geo.padding, geo.outputPadding, geo.dilation, geo.groups) : null,
                    () -> needBias ? biasGrad(gradOutput, geo.biasShape, 0, 2, 3) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ pooling

    /**
     * {@code maxPool2d(x)}: {@code gx = maxPool2dBackward(x, y, gy, geometry)}, the gradient of each window routed to
     * its first in-bounds tap in {@code (kh, kw)} order whose value equals the window maximum (a NaN maximum routes
     * nowhere), summed over overlapping windows in ascending {@code (oh, ow)} order. Saved: the input and the output.
     * They only select the routing, which is piecewise constant, so a create-graph pass reads them without a path to
     * their origin and differentiates the result in {@code gy} only ({@link MaxPool2dBackwardGrad}). Tolerance:
     * single-rounding.
     */
    public static class MaxPool2dGrad extends GradFunctions.SavingGradFn {
        private final PoolGeometry pool;

        public MaxPool2dGrad(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
            this.pool = new PoolGeometry(kernel, stride, padding, dilation, ceilMode);
        }

        /** Saves the forward input and its output. */
        public void saveForBackward(Tensor input, Tensor output) {
            save(input, output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(ConvOps.maxPool2dBackward(routing(this, 0), routing(this, 1),
                    gradOutput, pool.kernel, pool.stride, pool.padding, pool.dilation, pool.ceilMode));
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code avgPool2d(x)}: {@code gx = avgPool2dBackward(gy, x.shape, geometry)}, each input element summing
     * {@code gy / divisor} over the windows covering it with the forward's per-window divisors (PyTorch's: the window
     * size clipped to the padded extent with {@code countIncludePad}, otherwise its in-bounds element count). Saved:
     * nothing. Tolerance: reduction class with {@code ceil(KH/SH) * ceil(KW/SW)} terms (division class when the
     * windows do not overlap).
     */
    public static class AvgPool2dGrad implements GradientFunction {
        private final Shape inputShape;
        private final int[] kernel;
        private final int[] stride;
        private final int[] padding;
        private final boolean ceilMode;
        private final boolean countIncludePad;

        public AvgPool2dGrad(Shape inputShape, int[] kernel, int[] stride, int[] padding, boolean ceilMode,
                             boolean countIncludePad) {
            this.inputShape = requireRank(inputShape, 4, "avgPool2d input shape");
            this.kernel = pair(kernel, "kernel");
            this.stride = pair(stride, "stride");
            this.padding = pair(padding, "padding");
            this.ceilMode = ceilMode;
            this.countIncludePad = countIncludePad;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(ConvOps.avgPool2dBackward(gradOutput, inputShape, kernel, stride,
                    padding, ceilMode, countIncludePad));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code adaptiveAvgPool2d(x)}: {@code gx = adaptiveAvgPool2dBackward(gy, x.shape)}, each input element summing
     * {@code gy / count} over every window holding it (PyTorch's windows {@code [floor(o*H/outH), ceil((o+1)*H/outH))}
     * overlap when the output size does not divide the input size) in ascending {@code (oh, ow)} order. Saved:
     * nothing. Tolerance: reduction class with at most {@code (floor((outH-1)/H) + 2) * (floor((outW-1)/W) + 2)}
     * terms (division class when every element lies in one window).
     */
    public static class AdaptiveAvgPool2dGrad implements GradientFunction {
        private final Shape inputShape;

        public AdaptiveAvgPool2dGrad(Shape inputShape) {
            this.inputShape = requireRank(inputShape, 4, "adaptiveAvgPool2d input shape");
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(ConvOps.adaptiveAvgPool2dBackward(gradOutput, inputShape));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ second order (backward primitives)

    /**
     * {@code conv2dWeightGrad(x, gy)}, recorded inside a create-graph pass with the inputs {@code (x, gy)}. The weight
     * gradient is bilinear in its operands, so for an incoming gradient {@code g2} of the weight's shape:
     * {@code gx = convTranspose2d(gy, g2, null, K, S, P, outPad, D, groups)} cut to the input (the input gradient of
     * {@link Conv2dGrad} with {@code g2} as the weight) and {@code ggy = conv2d(x, g2, null, K, S, P, D, groups)}.
     * Saved: the input and the output gradient. Tolerance: reduction class ({@code K = OC/groups * KH * KW} terms for
     * gx, {@code C/groups * KH * KW} for ggy).
     */
    public static class Conv2dWeightGradGrad extends GradFunctions.SavingGradFn {
        private final ConvGeometry geo;
        private final boolean needInput;
        private final boolean needGradOutput;

        /**
         * @param needInput      whether {@code x} requires a gradient
         * @param needGradOutput whether {@code gy} requires a gradient
         */
        public Conv2dWeightGradGrad(Shape inputShape, Shape weightShape, int[] stride, int[] padding, int[] dilation,
                                    int groups, boolean needInput, boolean needGradOutput) {
            this.geo = ConvGeometry.of(inputShape, weightShape, null, stride, padding, null, dilation, groups, false);
            this.needInput = needInput;
            this.needGradOutput = needGradOutput;
        }

        /** Saves the primitive's input and output gradient, which must have the forward's shapes. */
        public void saveForBackward(Tensor input, Tensor gradOutput) {
            requireShape(input, geo.inputShape, "convolution input");
            geo.requireGradOutput(gradOutput);
            save(input, gradOutput);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            requireShape(gradOutput, geo.weightShape, "weight-gradient gradient");
            return gradients(
                    () -> needInput ? conv2dInputGrad(saved(1), gradOutput, geo) : null,
                    () -> needGradOutput ? ConvOps.conv2d(saved(0), gradOutput, null, geo.kernel(), geo.stride,
                            geo.padding, geo.dilation, geo.groups) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code convTranspose2dWeightGrad(x, gy)}, recorded inside a create-graph pass with the inputs {@code (x, gy)}.
     * Bilinear like {@link Conv2dWeightGradGrad}, with the roles of the convolution and its transpose swapped: for an
     * incoming gradient {@code g2} of the weight's shape, {@code gx = conv2d(gy, g2, null, K, S, P, D, groups)} cut to
     * the input (the input gradient of {@link ConvTranspose2dGrad} with {@code g2} as the weight) and
     * {@code ggy = convTranspose2d(x, g2, null, K, S, P, op, D, groups)}. Saved: the input and the output gradient.
     * Tolerance: reduction class ({@code K = C/groups * KH * KW} terms for gx, {@code OC/groups * KH * KW} for ggy at
     * most).
     */
    public static class ConvTranspose2dWeightGradGrad extends GradFunctions.SavingGradFn {
        private final ConvGeometry geo;
        private final boolean needInput;
        private final boolean needGradOutput;

        /**
         * @param needInput      whether {@code x} requires a gradient
         * @param needGradOutput whether {@code gy} requires a gradient
         */
        public ConvTranspose2dWeightGradGrad(Shape inputShape, Shape weightShape, int[] stride, int[] padding,
                                             int[] outputPadding, int[] dilation, int groups, boolean needInput,
                                             boolean needGradOutput) {
            this.geo = ConvGeometry.of(inputShape, weightShape, null, stride, padding,
                    Objects.requireNonNull(outputPadding, "outputPadding"), dilation, groups, true);
            this.needInput = needInput;
            this.needGradOutput = needGradOutput;
        }

        /** Saves the primitive's input and output gradient, which must have the forward's shapes. */
        public void saveForBackward(Tensor input, Tensor gradOutput) {
            requireShape(input, geo.inputShape, "transposed convolution input");
            geo.requireGradOutput(gradOutput);
            save(input, gradOutput);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            requireShape(gradOutput, geo.weightShape, "weight-gradient gradient");
            return gradients(
                    () -> needInput ? convTranspose2dInputGrad(saved(1), gradOutput, geo) : null,
                    () -> needGradOutput ? ConvOps.convTranspose2d(saved(0), gradOutput, null, geo.kernel(),
                            geo.stride, geo.padding, geo.outputPadding, geo.dilation, geo.groups) : null);
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code maxPool2dBackward(x, y, gy)}, recorded inside a create-graph pass with the input {@code gy} only: the
     * primitive is linear in {@code gy} and piecewise constant in {@code x} and {@code y}, which only select the tap
     * each window routes to. For an incoming gradient {@code g2} of the input's shape,
     * {@code ggy = maxPool2dSelect(x, y, g2)}, the adjoint routing. Saved: the forward input and output (routing
     * operands, read without a path to their origin). Tolerance: bit-identical.
     */
    public static class MaxPool2dBackwardGrad extends GradFunctions.SavingGradFn {
        private final PoolGeometry pool;

        public MaxPool2dBackwardGrad(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
            this.pool = new PoolGeometry(kernel, stride, padding, dilation, ceilMode);
        }

        /** Saves the forward input and output the primitive routed with. */
        public void saveForBackward(Tensor input, Tensor output) {
            save(input, output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(ConvOps.maxPool2dSelect(routing(this, 0), routing(this, 1), gradOutput,
                    pool.kernel, pool.stride, pool.padding, pool.dilation, pool.ceilMode));
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code maxPool2dSelect(x, y, g)}, recorded inside a create-graph pass with the input {@code g} only (linear in
     * {@code g}, piecewise constant in {@code x} and {@code y}). For an incoming gradient {@code g3} of the forward
     * output's shape, {@code gg = maxPool2dBackward(x, y, g3)}, the adjoint routing. Saved: the forward input and
     * output (routing operands, read without a path to their origin). Tolerance: single-rounding.
     */
    public static class MaxPool2dSelectGrad extends GradFunctions.SavingGradFn {
        private final PoolGeometry pool;

        public MaxPool2dSelectGrad(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
            this.pool = new PoolGeometry(kernel, stride, padding, dilation, ceilMode);
        }

        /** Saves the forward input and output the primitive routed with. */
        public void saveForBackward(Tensor input, Tensor output) {
            save(input, output);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            return Collections.singletonList(ConvOps.maxPool2dBackward(routing(this, 0), routing(this, 1), gradOutput,
                    pool.kernel, pool.stride, pool.padding, pool.dilation, pool.ceilMode));
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code avgPool2dBackward(gy)}, recorded inside a create-graph pass with the input {@code gy}. The primitive is
     * linear in {@code gy} and is the adjoint of {@link ConvOps#avgPool2d} with the same windows and counts, so for an
     * incoming gradient {@code g2} of the forward input's shape {@code ggy = avgPool2d(g2, geometry)}. Saved: nothing.
     * Tolerance: division class (reduction class over the window when it holds several taps).
     */
    public static class AvgPool2dBackwardGrad implements GradientFunction {
        private final Shape inputShape;
        private final int[] kernel;
        private final int[] stride;
        private final int[] padding;
        private final boolean ceilMode;
        private final boolean countIncludePad;

        public AvgPool2dBackwardGrad(Shape inputShape, int[] kernel, int[] stride, int[] padding, boolean ceilMode,
                                     boolean countIncludePad) {
            this.inputShape = requireRank(inputShape, 4, "avgPool2d input shape");
            this.kernel = pair(kernel, "kernel");
            this.stride = pair(stride, "stride");
            this.padding = pair(padding, "padding");
            this.ceilMode = ceilMode;
            this.countIncludePad = countIncludePad;
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            requireShape(gradOutput, inputShape, "avgPool2d input-gradient gradient");
            return Collections.singletonList(ConvOps.avgPool2d(gradOutput, kernel, stride, padding, ceilMode,
                    countIncludePad));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code adaptiveAvgPool2dBackward(gy)}, recorded inside a create-graph pass with the input {@code gy}. The
     * primitive is linear in {@code gy} and is the adjoint of {@link ConvOps#adaptiveAvgPool2d}, so for an incoming
     * gradient {@code g2} of the forward input's shape {@code ggy = adaptiveAvgPool2d(g2, outH, outW)}. Saved:
     * nothing. Tolerance: division class (reduction class over the window when it holds several taps).
     */
    public static class AdaptiveAvgPool2dBackwardGrad implements GradientFunction {
        private final Shape inputShape;
        private final int outH;
        private final int outW;

        /**
         * @param inputShape      the forward input shape {@code [B, C, H, W]}
         * @param gradOutputShape the shape of the primitive's output gradient {@code [B, C, outH, outW]}
         */
        public AdaptiveAvgPool2dBackwardGrad(Shape inputShape, Shape gradOutputShape) {
            this.inputShape = requireRank(inputShape, 4, "adaptiveAvgPool2d input shape");
            Shape out = requireRank(gradOutputShape, 4, "adaptiveAvgPool2d output gradient shape");
            this.outH = out.dim(2);
            this.outW = out.dim(3);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            requireShape(gradOutput, inputShape, "adaptiveAvgPool2d input-gradient gradient");
            return Collections.singletonList(ConvOps.adaptiveAvgPool2d(gradOutput, outH, outW));
        }

        @Override
        public List<Tensor> savedTensors() {
            return Collections.emptyList();
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ shared pieces

    /** The immutable window parameters of a recorded max pooling (copied). */
    private static final class PoolGeometry {
        final int[] kernel;
        final int[] stride;
        final int[] padding;
        final int[] dilation;
        final boolean ceilMode;

        PoolGeometry(int[] kernel, int[] stride, int[] padding, int[] dilation, boolean ceilMode) {
            this.kernel = pair(kernel, "kernel");
            this.stride = pair(stride, "stride");
            this.padding = pair(padding, "padding");
            this.dilation = pair(dilation, "dilation");
            this.ceilMode = ceilMode;
        }
    }

    /**
     * The immutable geometry of a recorded 2-D convolution: operand shapes, the parameters (copied) and the forward
     * output shape, computed with the forward's formula from validated arguments.
     */
    private static final class ConvGeometry {
        final Shape inputShape;
        final Shape weightShape;
        final Shape biasShape;
        final Shape outputShape;
        final int[] stride;
        final int[] padding;
        final int[] outputPadding;
        final int[] dilation;
        final int groups;

        private ConvGeometry(Shape inputShape, Shape weightShape, Shape biasShape, Shape outputShape, int[] stride,
                             int[] padding, int[] outputPadding, int[] dilation, int groups) {
            this.inputShape = inputShape;
            this.weightShape = weightShape;
            this.biasShape = biasShape;
            this.outputShape = outputShape;
            this.stride = stride;
            this.padding = padding;
            this.outputPadding = outputPadding;
            this.dilation = dilation;
            this.groups = groups;
        }

        static ConvGeometry of(Shape inputShape, Shape weightShape, Shape biasShape, int[] stride, int[] padding,
                               int[] outputPadding, int[] dilation, int groups, boolean transposed) {
            Shape in = requireRank(inputShape, 4, "convolution input shape");
            Shape w = requireRank(weightShape, 4, "convolution weight shape");
            int[] s = pair(stride, "stride");
            int[] p = pair(padding, "padding");
            int[] d = pair(dilation, "dilation");
            int[] op = outputPadding == null ? null : pair(outputPadding, "outputPadding");
            if (groups < 1) {
                throw new IllegalArgumentException("groups must be positive: " + groups);
            }
            Shape out;
            if (transposed) {
                out = Shape.of(in.dim(0), w.dim(1) * groups,
                        transposedExtent(in.dim(2), w.dim(2), s[0], p[0], op[0], d[0]),
                        transposedExtent(in.dim(3), w.dim(3), s[1], p[1], op[1], d[1]));
            } else {
                out = Shape.of(in.dim(0), w.dim(0), convExtent(in.dim(2), w.dim(2), s[0], p[0], d[0]),
                        convExtent(in.dim(3), w.dim(3), s[1], p[1], d[1]));
            }
            return new ConvGeometry(in, w, biasShape, out, s, p, op, d, groups);
        }

        int[] kernel() {
            return new int[]{weightShape.dim(2), weightShape.dim(3)};
        }

        void requireOperands(Tensor input, Tensor weight) {
            requireShape(input, inputShape, "convolution input");
            requireShape(weight, weightShape, "convolution weight");
        }

        void requireGradOutput(Tensor gradOutput) {
            requireShape(gradOutput, outputShape, "convolution output gradient");
        }

        private static int convExtent(int size, int k, int s, int p, int d) {
            return (int) (((long) size + 2L * p - (long) d * (k - 1) - 1) / s + 1);
        }

        private static int transposedExtent(int size, int k, int s, int p, int op, int d) {
            return (int) (((long) size - 1) * s - 2L * p + (long) d * (k - 1) + op + 1);
        }
    }

    /**
     * The input gradient of a 2-D convolution of {@code geo}: {@code convTranspose2d(gy, w)} with the output padding
     * that restores the input extent, cut to the input when the forward span was negative. Zeros when the input or
     * the gradient has no elements (no output position reads an input element).
     */
    private static Tensor conv2dInputGrad(Tensor gy, Tensor w, ConvGeometry geo) {
        Shape in = geo.inputShape;
        if (in.numElements() == 0 || gy.numElements() == 0) {
            return Tensor.zeros(gy.backend(), in);
        }
        int[] kernel = geo.kernel();
        int[] outputPadding = {
                inputGradOutputPadding(in.dim(2), kernel[0], geo.stride[0], geo.padding[0], geo.dilation[0],
                        gy.shape().dim(2)),
                inputGradOutputPadding(in.dim(3), kernel[1], geo.stride[1], geo.padding[1], geo.dilation[1],
                        gy.shape().dim(3))};
        Tensor full = ConvOps.convTranspose2d(gy, w, null, kernel, geo.stride, geo.padding, outputPadding,
                geo.dilation, geo.groups);
        return cropToInput(full, in);
    }

    /**
     * The output padding that gives {@code convTranspose2d} over {@code out} rows exactly {@code size} rows:
     * {@code span - (out - 1) * s} with {@code span = size + 2p - d(k-1) - 1}, which is {@code span mod s} for
     * {@code span >= 0}. A negative span (the forward's truncating division gave {@code out = 1}) yields 0; the
     * transposed output then has {@code size - span} rows, cut to {@code size} by {@link #cropToInput}.
     */
    private static int inputGradOutputPadding(int size, int k, int s, int p, int d, int out) {
        long span = (long) size + 2L * p - (long) d * (k - 1) - 1;
        return (int) Math.max(0L, span - (long) (out - 1) * s);
    }

    /**
     * The input gradient of a 2-D transposed convolution of {@code geo}: {@code conv2d(gy, w)}, whose output has
     * {@code H + floor(op/S)} rows (and the W twin), cut to the input when that exceeds it. Zeros when the input or the
     * gradient has no elements.
     */
    private static Tensor convTranspose2dInputGrad(Tensor gy, Tensor w, ConvGeometry geo) {
        if (geo.inputShape.numElements() == 0 || gy.numElements() == 0) {
            return Tensor.zeros(gy.backend(), geo.inputShape);
        }
        Tensor full = ConvOps.conv2d(gy, w, null, geo.kernel(), geo.stride, geo.padding, geo.dilation, geo.groups);
        return cropToInput(full, geo.inputShape);
    }

    /**
     * Saved tensor {@code i} of {@code fn} as saved, without the path to its origin that {@code saved(i)} gives inside
     * a create-graph pass: for the max-pooling routing operands, which the backward does not differentiate. Borrowed.
     */
    private static Tensor routing(GradFunctions.SavingGradFn fn, int i) {
        List<Tensor> saved = fn.savedTensors();
        if (i >= saved.size()) {
            throw new IllegalStateException(fn.getClass().getSimpleName() + " has no saved tensor " + i
                    + ": nothing was saved, or the saved tensors were released after an earlier backward");
        }
        return saved.get(i);
    }

    /**
     * {@code full} cut to the first {@code target.dim(2)} rows and {@code target.dim(3)} columns and made contiguous,
     * or {@code full} itself when it already has {@code target}'s shape; {@code full} is handed over.
     */
    private static Tensor cropToInput(Tensor full, Shape target) {
        Tensor current = full;
        try {
            if (current.shape().equals(target)) {
                return current;
            }
            if (current.shape().dim(0) != target.dim(0) || current.shape().dim(1) != target.dim(1)
                    || current.shape().dim(2) < target.dim(2) || current.shape().dim(3) < target.dim(3)) {
                throw new IllegalStateException("input gradient " + current.shape() + " cannot be cut to the input "
                        + target);
            }
            if (current.shape().dim(2) > target.dim(2)) {
                current = replace(current, current.slice(2, 0, target.dim(2)));
            }
            if (current.shape().dim(3) > target.dim(3)) {
                current = replace(current, current.slice(3, 0, target.dim(3)));
            }
            Tensor dense = current.contiguous();
            if (dense != current) {
                current = replace(current, dense);
            }
            return current;
        } catch (RuntimeException e) {
            throw closeAfterFailure(Collections.singletonList(current), e);
        }
    }

    /** Closes {@code previous} and returns {@code next}; when closing fails, {@code next} is closed too. */
    private static Tensor replace(Tensor previous, Tensor next) {
        try {
            previous.close();
        } catch (RuntimeException e) {
            throw closeAfterFailure(Collections.singletonList(next), e);
        }
        return next;
    }

    /** {@code sum(gy, dims)} in the bias's shape (a view when the shapes differ). */
    private static Tensor biasGrad(Tensor gy, Shape biasShape, int... dims) {
        Tensor sum = TensorOps.sum(gy, dims);
        if (sum.shape().equals(biasShape)) {
            return sum;
        }
        try {
            return sum.reshape(biasShape);
        } finally {
            sum.close();
        }
    }

    /** The gradients in input order; when one throws, those already computed are closed. */
    @SafeVarargs
    private static List<Tensor> gradients(Supplier<Tensor>... parts) {
        List<Tensor> out = new ArrayList<>(parts.length);
        try {
            for (Supplier<Tensor> part : parts) {
                out.add(part.get());
            }
            return out;
        } catch (RuntimeException e) {
            throw closeAfterFailure(out, e);
        }
    }

    private static int[] pair(int[] values, String name) {
        if (values == null || values.length != 2) {
            throw new IllegalArgumentException(name + " must be int[2], got " + Arrays.toString(values));
        }
        return values.clone();
    }

    private static Shape requireRank(Shape shape, int rank, String what) {
        Objects.requireNonNull(shape, what);
        if (shape.rank() != rank) {
            throw new IllegalArgumentException(what + " must have rank " + rank + ", got " + shape);
        }
        return shape;
    }

    private static void requireShape(Tensor t, Shape expected, String what) {
        Objects.requireNonNull(t, what);
        if (!t.shape().equals(expected)) {
            throw new IllegalArgumentException(what + " " + t.shape() + " does not match the recorded shape "
                    + expected);
        }
    }

    /** Closes the non-null tensors, attaching any close failure to {@code failure}, and returns it. */
    private static RuntimeException closeAfterFailure(List<Tensor> tensors, RuntimeException failure) {
        for (Tensor t : tensors) {
            if (t != null) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    failure.addSuppressed(e);
                }
            }
        }
        return failure;
    }
}

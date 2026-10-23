package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.GradFunctions;
import com.eonmux.neuralfree.autograd.GradientFunction;
import com.eonmux.neuralfree.autograd.IndexGradFunctions;
import com.eonmux.neuralfree.autograd.SecondOrderGradFunctions;
import com.eonmux.neuralfree.autograd.SelectionGradFunctions;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Tensor operations. Each method first offers the work to the train-library dispatcher of its family
 * ({@link GpuTensorOps}, {@link GpuReduceOps}, {@link GpuMatmul}, {@link GpuNormOps}, {@link GpuCoreOps},
 * {@link GpuIndexGradOps}); the dispatcher computes on the device, keeps the result there and never reads it back.
 * When the dispatcher declines (the train library is unavailable on the tensors' backend, or a dtype is not one it
 * supports), the CPU reference runs: {@code CpuElementwise}, {@code CpuReductions}, {@code CpuLinalg},
 * {@code CpuShapeOps} and {@code CpuIndexing}, which read the operands in row-major order, compute in Java float
 * arithmetic and upload a new contiguous result to the operands' backend.
 *
 * <p>Both paths have one contract: the same shapes, dtypes and argument checks, and results within the tolerance
 * class each dispatcher documents ({@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}). An out-of-range
 * gather, scatter or row index throws {@link IndexOutOfBoundsException} at the call on the CPU path and at the
 * backend's next host synchronization on the device path.</p>
 *
 * <p><b>Autograd.</b> A differentiable operation records one node ({@link Autograd#record}) exactly when grad mode is
 * on and one of its differentiable tensor operands requires grad ({@link Autograd#needsGrad}); its result then
 * requires grad and carries the node's gradient function ({@code GradFunctions}, {@code SelectionGradFunctions},
 * {@code IndexGradFunctions}). Otherwise the result does not require grad. The raw computation of a recorded
 * operation runs with grad mode off ({@link Autograd#untracked}), and argument checks run before the recording
 * decision, so errors are the same with and without grad. Index tensors, the condition of {@link #where}, the mask of
 * {@link #maskScale} and the CSR arrays of {@link #segmentReduce} are not differentiable operands: they never make a
 * result require grad and get no gradient. The non-differentiable operations ({@link #round}, {@link #floor},
 * {@link #argmax}, {@link #argmin}, {@link #eq}, {@link #gt}, {@link #lt}) never record and return results that do not
 * require grad. The backward primitives ({@link #activationBackward}, {@link #softmaxBackward},
 * {@link #logSoftmaxBackward}, {@link #maxBackward}, {@link #indexAddRows}, {@link #segmentReduceBackward}) record only
 * inside a create-graph pass ({@link Autograd#isCreatingGraph()}), with their {@code SecondOrderGradFunctions}, when
 * an operand they depend on differentiably requires grad; otherwise their results do not require grad. The in-place
 * operations refuse a destination that requires grad while grad mode is on.</p>
 */
public final class TensorOps {

    private TensorOps() {
    }

    // ------------------------------------------------------------------ binary (broadcasting)

    /** {@code a + b} over the broadcast shape. Recorded with {@code AddGrad}. */
    public static Tensor add(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuTensorOps.add(a, b), () -> CpuElementwise.add(a, b)),
                out -> new GradFunctions.AddGrad(a.shape(), b.shape(), a.requiresGrad(), b.requiresGrad()));
    }

    /** {@code a - b} over the broadcast shape. Recorded with {@code SubGrad}. */
    public static Tensor sub(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuTensorOps.sub(a, b), () -> CpuElementwise.sub(a, b)),
                out -> new GradFunctions.SubGrad(a.shape(), b.shape(), a.requiresGrad(), b.requiresGrad()));
    }

    /** {@code a * b} over the broadcast shape. Recorded with {@code MulGrad}, saving both operands. */
    public static Tensor mul(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuTensorOps.mul(a, b), () -> CpuElementwise.mul(a, b)),
                out -> saving(new GradFunctions.MulGrad(a.shape(), b.shape(), a.requiresGrad(), b.requiresGrad()),
                        f -> f.saveForBackward(a, b)));
    }

    /** {@code a / b} over the broadcast shape. Recorded with {@code DivGrad}, saving both operands. */
    public static Tensor div(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuTensorOps.div(a, b), () -> CpuElementwise.div(a, b)),
                out -> saving(new GradFunctions.DivGrad(a.shape(), b.shape(), a.requiresGrad(), b.requiresGrad()),
                        f -> f.saveForBackward(a, b)));
    }

    // ------------------------------------------------------------------ unary and activations

    public static Tensor pow(Tensor base, float exp) {
        Objects.requireNonNull(base, "Base tensor cannot be null");
        return recordUnary(base, () -> orCpu(GpuTensorOps.pow(base, exp), () -> CpuElementwise.pow(base, exp)),
                out -> saving(new GradFunctions.PowGrad(exp), f -> f.saveForBackward(base)));
    }

    public static Tensor neg(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.neg(a), () -> CpuElementwise.neg(a)),
                out -> new GradFunctions.NegGrad());
    }

    public static Tensor abs(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.abs(a), () -> CpuElementwise.abs(a)),
                out -> saving(new GradFunctions.AbsGrad(), f -> f.saveForBackward(a)));
    }

    public static Tensor sqrt(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.sqrt(a), () -> CpuElementwise.sqrt(a)),
                out -> saving(new GradFunctions.SqrtGrad(), f -> f.saveForBackward(out)));
    }

    public static Tensor exp(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.exp(a), () -> CpuElementwise.exp(a)),
                out -> saving(new GradFunctions.ExpGrad(), f -> f.saveForBackward(out)));
    }

    public static Tensor log(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.log(a), () -> CpuElementwise.log(a)),
                out -> saving(new GradFunctions.LogGrad(), f -> f.saveForBackward(a)));
    }

    /** {@code (float) Math.sin(x)} per element. Recorded with {@code SinGrad}. */
    public static Tensor sin(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.sin(a), () -> CpuElementwise.sin(a)),
                out -> saving(new GradFunctions.SinGrad(), f -> f.saveForBackward(a)));
    }

    /** {@code (float) Math.cos(x)} per element. Recorded with {@code CosGrad}. */
    public static Tensor cos(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.cos(a), () -> CpuElementwise.cos(a)),
                out -> saving(new GradFunctions.CosGrad(), f -> f.saveForBackward(a)));
    }

    /** {@code Math.max(min, Math.min(max, x))} per element. */
    public static Tensor clamp(Tensor a, float min, float max) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.clamp(a, min, max), () -> CpuElementwise.clamp(a, min, max)),
                out -> saving(new GradFunctions.ClampGrad(min, max), f -> f.saveForBackward(a)));
    }

    public static Tensor add(Tensor a, float scalar) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.addScalar(a, scalar), () -> CpuElementwise.addScalar(a,
                scalar)), out -> new GradFunctions.AddScalarGrad(a.shape()));
    }

    public static Tensor mul(Tensor a, float scalar) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.mulScalar(a, scalar), () -> CpuElementwise.mulScalar(a,
                scalar)), out -> new GradFunctions.MulScalarGrad(scalar));
    }

    public static Tensor relu(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.relu(a), () -> CpuElementwise.relu(a)),
                out -> saving(new GradFunctions.ReluGrad(), f -> f.saveForBackward(a)));
    }

    public static Tensor gelu(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.gelu(a), () -> CpuElementwise.gelu(a)),
                out -> saving(new GradFunctions.GeluGrad(), f -> f.saveForBackward(a)));
    }

    public static Tensor silu(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.silu(a), () -> CpuElementwise.silu(a)),
                out -> saving(new GradFunctions.SiluGrad(), f -> f.saveForBackward(a)));
    }

    public static Tensor sigmoid(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.sigmoid(a), () -> CpuElementwise.sigmoid(a)),
                out -> saving(new GradFunctions.SigmoidGrad(), f -> f.saveForBackward(out)));
    }

    public static Tensor tanh(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.tanh(a), () -> CpuElementwise.tanh(a)),
                out -> saving(new GradFunctions.TanhGrad(), f -> f.saveForBackward(out)));
    }

    public static Tensor leakyRelu(Tensor a, float negativeSlope) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.leakyRelu(a, negativeSlope),
                () -> CpuElementwise.leakyRelu(a, negativeSlope)),
                out -> saving(new GradFunctions.LeakyReluGrad(negativeSlope), f -> f.saveForBackward(a)));
    }

    public static Tensor elu(Tensor a, float alpha) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.elu(a, alpha), () -> CpuElementwise.elu(a, alpha)),
                out -> saving(new GradFunctions.EluGrad(alpha), f -> f.saveForBackward(a, out)));
    }

    public static Tensor softplus(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.softplus(a), () -> CpuElementwise.softplus(a)),
                out -> saving(new GradFunctions.SoftplusGrad(), f -> f.saveForBackward(a)));
    }

    public static Tensor mish(Tensor a) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuTensorOps.mish(a), () -> CpuElementwise.mish(a)),
                out -> saving(new GradFunctions.MishGrad(), f -> f.saveForBackward(a)));
    }

    /**
     * {@code Math.round(x)} per element, kept as a float ({@code floor(x)}, plus 1 when the fraction is at least
     * 0.5; {@code -0}, NaN and the infinities unchanged). A cast to INT32 then gives Java's {@code int} result. Not
     * differentiable: never recorded.
     */
    public static Tensor round(Tensor a) {
        Tensor gpu = GpuTensorOps.round(check(a));
        return gpu != null ? gpu : CpuElementwise.round(a);
    }

    /** {@code (float) Math.floor(x)} per element. Not differentiable: never recorded. */
    public static Tensor floor(Tensor a) {
        Tensor gpu = GpuTensorOps.floor(check(a));
        return gpu != null ? gpu : CpuElementwise.floor(a);
    }

    /**
     * {@code first * sigmoid(second)}, where {@code first} and {@code second} are the halves of {@code dim}.
     *
     * @throws IllegalArgumentException when the size of {@code dim} is odd
     */
    public static Tensor glu(Tensor a, int dim) {
        int d = CpuShapeOps.normalizeDim(dim, check(a).shape().rank());
        Shape outShape = CpuElementwise.gatedShape(a.shape(), d, "GLU");
        return recordUnary(a, () -> orCpu(gatedOnDevice(a, d, outShape, false), () -> CpuElementwise.glu(a, d)),
                out -> saving(new GradFunctions.GluGrad(d), f -> f.saveForBackward(a)));
    }

    /**
     * {@code first * gelu(second)}, where {@code first} and {@code second} are the halves of {@code dim}.
     *
     * @throws IllegalArgumentException when the size of {@code dim} is odd
     */
    public static Tensor geglu(Tensor a, int dim) {
        int d = CpuShapeOps.normalizeDim(dim, check(a).shape().rank());
        Shape outShape = CpuElementwise.gatedShape(a.shape(), d, "GeGLU");
        return recordUnary(a, () -> orCpu(gatedOnDevice(a, d, outShape, true), () -> CpuElementwise.geglu(a, d)),
                out -> saving(new GradFunctions.GegluGrad(d), f -> f.saveForBackward(a)));
    }

    /**
     * {@code gx = gy * f'} for the forward function of {@code op}, from the forward input {@code x} and/or output
     * {@code y} as {@link ElementwiseOp.Backward#readsX()} and {@link ElementwiseOp.Backward#readsY()} state (an
     * operand the op does not read may be null), with the forward parameters {@code p0} and {@code p1} (LEAKY_RELU
     * slope, ELU alpha, CLAMP bounds). The formulas are those of the {@code GradFunctions} activation classes. A
     * backward primitive: recorded only in a create-graph pass, with {@code ActivationBackwardGrad}.
     *
     * @throws IllegalArgumentException when the tensors live on different backends or a used shape differs from
     *                                  {@code gy}'s
     */
    public static Tensor activationBackward(ElementwiseOp.Backward op, Tensor x, Tensor y, Tensor gy, float p0,
                                            float p1) {
        Objects.requireNonNull(op, "op must not be null");
        return primitive(() -> orCpu(GpuTensorOps.unaryBackward(op, x, y, gy, p0, p1),
                () -> CpuElementwise.activationBackward(op, x, y, gy, p0, p1)),
                SecondOrderGradFunctions.activationOperands(op, x, y, gy),
                out -> SecondOrderGradFunctions.activationBackward(op, x, y, gy, p0, p1));
    }

    /**
     * Dropout apply: {@code mask == +-0 ? +0 : (x * mask) * scale} per element, in that evaluation order on both
     * paths ({@code kernels/train/eltwise.cl} {@code TR_MaskScale_F32}). A dropped element is {@code +0} even for a
     * non-finite {@code x}; a kept element is {@code x} times the keep scale whether the mask holds the scale and
     * {@code scale} is 1, or the mask holds 1 and {@code scale} is the keep scale. Single-rounding class. Recorded
     * with {@code IndexGradFunctions.MaskScaleGrad} when {@code x} needs grad; the mask is a constant operand.
     *
     * @throws IllegalArgumentException when the tensors live on different backends or their shapes differ
     */
    public static Tensor maskScale(Tensor x, Tensor mask, float scale) {
        checkPair(x, mask);
        if (!x.shape().equals(mask.shape())) {
            throw new IllegalArgumentException("Mask shape " + mask.shape() + " does not match input shape "
                    + x.shape());
        }
        Supplier<Tensor> raw = () -> orCpu(GpuTensorOps.maskScale(x, mask, scale),
                () -> CpuElementwise.maskScale(x, mask, scale));
        return !Autograd.needsGrad(x) ? raw.get() : recordOver(raw, Arrays.asList(x, mask),
                out -> saving(new IndexGradFunctions.MaskScaleGrad(scale), f -> f.saveForBackward(mask)));
    }

    // ------------------------------------------------------------------ softmax family

    /** Recorded with {@code SoftmaxGrad}, saving the output. */
    public static Tensor softmax(Tensor a, int dim) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.softmax(a, dim), () -> CpuReductions.softmax(a, dim)),
                out -> saving(new GradFunctions.SoftmaxGrad(dim), f -> f.saveForBackward(out)));
    }

    /** Recorded with {@code LogSoftmaxGrad}, saving the output. */
    public static Tensor logSoftmax(Tensor a, int dim) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.logSoftmax(a, dim), () -> CpuReductions.logSoftmax(a, dim)),
                out -> saving(new GradFunctions.LogSoftmaxGrad(dim), f -> f.saveForBackward(out)));
    }

    /**
     * Softmax along {@code dim} where {@code -inf} entries are masked out: they are left out of the sum and get
     * exactly 0, and a row of only {@code -inf} gives all 0 ({@code MultiHeadAttention}'s attention softmax).
     * Recorded with {@code SoftmaxGrad} saving the output, whose {@code y * (gy - sum(y * gy))} is 0 at the masked
     * entries and on all-masked rows.
     */
    public static Tensor maskedSoftmax(Tensor a, int dim) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.maskedSoftmax(a, dim),
                () -> CpuReductions.maskedSoftmax(a, dim)),
                out -> saving(new GradFunctions.SoftmaxGrad(dim), f -> f.saveForBackward(out)));
    }

    /**
     * Gradient of {@link #softmax} from its output {@code y}: {@code gx = y * (gy - sum(y * gy))} along {@code dim}.
     * A backward primitive: recorded only in a create-graph pass, with {@code SoftmaxBackwardGrad}.
     */
    public static Tensor softmaxBackward(Tensor y, Tensor gy, int dim) {
        return softmaxBackward(y, gy, dim, false);
    }

    /**
     * Gradient of {@link #logSoftmax} from its output {@code y}: {@code gx = gy - exp(y) * sum(gy)} along {@code dim}.
     * A backward primitive: recorded only in a create-graph pass, with {@code LogSoftmaxBackwardGrad}.
     */
    public static Tensor logSoftmaxBackward(Tensor y, Tensor gy, int dim) {
        return softmaxBackward(y, gy, dim, true);
    }

    private static Tensor softmaxBackward(Tensor y, Tensor gy, int dim, boolean log) {
        checkPair(y, gy);
        return primitive(() -> orCpu(GpuReduceOps.softmaxBackward(y, gy, dim, log),
                () -> CpuReductions.softmaxBackward(y, gy, dim, log)), Arrays.asList(y, gy),
                out -> SecondOrderGradFunctions.softmaxBackward(y, gy, dim, log));
    }

    // ------------------------------------------------------------------ matrix products

    /**
     * {@code a @ b} with vector/matrix rank rules up to rank 2 and broadcast batch dims above. Recorded with
     * {@code MatmulGrad}, saving both operands.
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuMatmul.matmul(a, b), () -> CpuLinalg.matmul(a, b)),
                out -> saving(new GradFunctions.MatmulGrad(a.requiresGrad(), b.requiresGrad()),
                        f -> f.saveForBackward(a, b)));
    }

    /** {@code a @ b} for two rank-3 operands with equal batch. Recorded with {@code BmmGrad}, saving both operands. */
    public static Tensor bmm(Tensor a, Tensor b) {
        checkPair(a, b);
        return recordBinary(a, b, () -> orCpu(GpuMatmul.bmm(a, b), () -> CpuLinalg.bmm(a, b)),
                out -> saving(new GradFunctions.BmmGrad(a.requiresGrad(), b.requiresGrad()),
                        f -> f.saveForBackward(a, b)));
    }

    // ------------------------------------------------------------------ reductions

    /** Sum over {@code dims} (reduced dims removed); no dims gives a scalar. Recorded with {@code SumGrad}. */
    public static Tensor sum(Tensor a, int... dims) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.sum(a, dims), () -> CpuReductions.sum(a, dims)),
                out -> new GradFunctions.SumGrad(a.shape(), dims));
    }

    /** Mean over {@code dims} (reduced dims removed); no dims gives a scalar. Recorded with {@code MeanGrad}. */
    public static Tensor mean(Tensor a, int... dims) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.mean(a, dims), () -> CpuReductions.mean(a, dims)),
                out -> new GradFunctions.MeanGrad(a.shape(), dims));
    }

    /** Maximum along {@code dim}. Recorded with {@code MaxGrad}, saving the input and the output. */
    public static Tensor max(Tensor a, int dim) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.max(a, dim), () -> CpuReductions.max(a, dim)),
                out -> saving(new GradFunctions.MaxGrad(dim, a.shape()), f -> f.saveForBackward(a, out)));
    }

    /** Minimum along {@code dim}. Recorded with {@code MinGrad}, saving the input and the output. */
    public static Tensor min(Tensor a, int dim) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.min(a, dim), () -> CpuReductions.min(a, dim)),
                out -> saving(new SelectionGradFunctions.MinGrad(dim, a.shape()), f -> f.saveForBackward(a, out)));
    }

    /** INT32 index of the first strict maximum along {@code dim} (NaN never selected; 0 when none). Never recorded. */
    public static Tensor argmax(Tensor a, int dim) {
        Tensor gpu = GpuReduceOps.argmax(check(a), dim);
        return gpu != null ? gpu : CpuReductions.argmax(a, dim);
    }

    /** INT32 index of the first strict minimum along {@code dim} (NaN never selected; 0 when none). Never recorded. */
    public static Tensor argmin(Tensor a, int dim) {
        Tensor gpu = GpuReduceOps.argmin(check(a), dim);
        return gpu != null ? gpu : CpuReductions.argmin(a, dim);
    }

    /**
     * Population variance over {@code dims}, for any choice of dims; no dims gives a scalar. Recorded with
     * {@code VarGrad}, saving the input.
     */
    public static Tensor var(Tensor a, int... dims) {
        check(a);
        return recordUnary(a, () -> orCpu(GpuReduceOps.var(a, dims), () -> CpuReductions.var(a, dims)),
                out -> saving(new SelectionGradFunctions.VarGrad(a.shape(), dims), f -> f.saveForBackward(a)));
    }

    /**
     * Gradient of {@link #max}: zeros of {@code x}'s shape except {@code gy[row]} at the first index along
     * {@code dim} whose value equals {@code maxOut[row]}; {@code maxOut} and {@code gy} hold one value per reduced
     * row (any shape with that element count, such as the {@link #max} result). A backward primitive: recorded only
     * in a create-graph pass, with {@code MaxBackwardGrad} over {@code gy} (the routing is piecewise constant).
     */
    public static Tensor maxBackward(Tensor x, Tensor maxOut, Tensor gy, int dim) {
        Objects.requireNonNull(maxOut, "max output must not be null");
        checkPair(check(x), gy);
        return primitive(() -> orCpu(GpuReduceOps.maxBackward(x, maxOut, gy, dim),
                () -> CpuReductions.maxBackward(x, maxOut, gy, dim)), Collections.singletonList(gy),
                out -> SecondOrderGradFunctions.maxBackward(x, maxOut, gy, dim));
    }

    /**
     * {@code t} summed down to {@code target}, the shape of an operand that was broadcast to {@code t}'s shape (the
     * prepended dims and the dims where {@code target} has 1 are summed); a {@code target} that only rearranges
     * {@code t}'s elements gives a copy. Always a new tensor. Recorded with {@code SumToShapeGrad}.
     *
     * @throws IllegalArgumentException when {@code target} neither broadcasts to {@code t}'s shape nor has its
     *                                  element count
     */
    public static Tensor sumToShape(Tensor t, Shape target) {
        Objects.requireNonNull(target, "target shape must not be null");
        check(t);
        return recordUnary(t, () -> orCpu(GpuReduceOps.sumToShape(t, target), () -> CpuReductions.sumToShape(t,
                target)), out -> new SelectionGradFunctions.SumToShapeGrad(t.shape(), target));
    }

    /**
     * Per-segment aggregation of the rows of {@code src} over a CSR edge list (graph message aggregation); see
     * {@link GpuReduceOps#segmentReduce} for the layout, the empty value and the CSR precondition. The CSR arrays
     * are INT32, or INT64 narrowed by Java {@code (int)}. The forward accepts a source row named by several edges and
     * computes it identically on both paths; it never checks the precondition of the backward. Recorded with
     * {@code SegmentReduceGrad} when {@code src} needs grad; its backward ({@link #segmentReduceBackward}) requires
     * each source row at most once in {@code csrEdges}, which the CSRs of {@code nn/graph/GraphData} (permutations of
     * the edge and node ids) satisfy.
     */
    public static Tensor segmentReduce(Tensor src, Tensor csrPtr, Tensor csrEdges, int numNodes,
                                       GpuReduceOps.SegmentOp op, float emptyValue) {
        Objects.requireNonNull(src, "source must not be null");
        Objects.requireNonNull(csrPtr, "CSR offsets must not be null");
        Objects.requireNonNull(csrEdges, "CSR edges must not be null");
        Objects.requireNonNull(op, "segment op must not be null");
        Supplier<Tensor> raw = () -> segmentReduceRaw(src, csrPtr, csrEdges, numNodes, op, emptyValue);
        return !Autograd.needsGrad(src) ? raw.get() : recordOver(raw, Arrays.asList(src, csrPtr, csrEdges),
                out -> saving(new IndexGradFunctions.SegmentReduceGrad(op),
                        f -> f.saveForBackward(src, csrPtr, csrEdges, out)));
    }

    private static Tensor segmentReduceRaw(Tensor src, Tensor csrPtr, Tensor csrEdges, int numNodes,
                                           GpuReduceOps.SegmentOp op, float emptyValue) {
        Tensor gpu;
        try (Tensor ptr32 = deviceInt32(csrPtr, src);
             Tensor edges32 = deviceInt32(csrEdges, src)) {
            gpu = GpuReduceOps.segmentReduce(src, ptr32 != null ? ptr32 : csrPtr,
                    edges32 != null ? edges32 : csrEdges, numNodes, op, emptyValue);
        }
        return gpu != null ? gpu : CpuIndexing.segmentReduce(src, csrPtr, csrEdges, numNodes, op, emptyValue);
    }

    /**
     * Gradient of {@link #segmentReduce} with respect to {@code src}: a FLOAT32 tensor of {@code src}'s shape, zero
     * except at the source rows the segments name. For each segment with edges and feature {@code f}
     * ({@code g = gy[t, f]}): SUM gives {@code g} to every source row of the segment, MEAN {@code g / (float) count},
     * MAX gives {@code g} to the first source row, in ascending edge order, whose value is Java {@code ==}
     * {@code out[t, f]} ({@code out} the forward result; a NaN result routes nowhere). Tolerance: SUM and MAX
     * bit-identical, MEAN division. A backward primitive: recorded only in a create-graph pass, with
     * {@code SegmentReduceBackwardGrad} over {@code gy}.
     *
     * <p>Precondition: each source row appears at most once in {@code csrEdges} (over the edge slots the segments
     * own), so every gradient element has one writer. The CSRs of {@code nn/graph/GraphData} satisfy it. The CPU
     * reference validates it and throws {@link IllegalArgumentException}; the device does not re-check it.</p>
     *
     * @param out the forward result, required for MAX (ignored, and may be null, otherwise)
     * @throws IllegalArgumentException when the operands do not match the forward's shapes or backends, or the CPU
     *                                  path finds the CSR arrays invalid or a source row repeated
     */
    public static Tensor segmentReduceBackward(Tensor src, Tensor csrPtr, Tensor csrEdges, Tensor out, Tensor gy,
                                               GpuReduceOps.SegmentOp op) {
        return primitive(() -> orCpu(GpuIndexGradOps.segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op),
                () -> CpuIndexing.segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op)),
                Collections.singletonList(gy),
                o -> SecondOrderGradFunctions.segmentReduceBackward(src, csrPtr, csrEdges, out, gy, op));
    }

    // ------------------------------------------------------------------ compare and select

    /** BOOL {@code a == b} over the broadcast shape (Java float semantics). Never recorded. */
    public static Tensor eq(Tensor a, Tensor b) {
        checkPair(a, b);
        Tensor gpu = GpuTensorOps.eq(a, b);
        return gpu != null ? gpu : CpuElementwise.eq(a, b);
    }

    /** BOOL {@code a > b} over the broadcast shape (Java float semantics). Never recorded. */
    public static Tensor gt(Tensor a, Tensor b) {
        checkPair(a, b);
        Tensor gpu = GpuTensorOps.gt(a, b);
        return gpu != null ? gpu : CpuElementwise.gt(a, b);
    }

    /** BOOL {@code a < b} over the broadcast shape (Java float semantics). Never recorded. */
    public static Tensor lt(Tensor a, Tensor b) {
        checkPair(a, b);
        Tensor gpu = GpuTensorOps.lt(a, b);
        return gpu != null ? gpu : CpuElementwise.lt(a, b);
    }

    /**
     * {@code condition != 0 ? a : b} per element, for a FLOAT32 condition ({@code != 0.0f}) or a BOOL condition
     * (non-zero byte); all three tensors have one shape. Recorded with {@code WhereGrad} (saving the condition) when
     * {@code a} or {@code b} needs grad; the condition is a constant operand.
     */
    public static Tensor where(Tensor condition, Tensor a, Tensor b) {
        Objects.requireNonNull(condition, "Condition tensor cannot be null");
        checkPair(a, b);
        CpuShapeOps.requireSameBackend(condition, a);
        Supplier<Tensor> raw = () -> orCpu(GpuTensorOps.where(condition, a, b),
                () -> CpuElementwise.where(condition, a, b));
        return !Autograd.needsGrad(a, b) ? raw.get() : recordOver(raw, Arrays.asList(condition, a, b),
                out -> saving(new SelectionGradFunctions.WhereGrad(a.requiresGrad(), b.requiresGrad()),
                        f -> f.saveForBackward(condition)));
    }

    // ------------------------------------------------------------------ gather, scatter, row selection

    /**
     * {@code out[c] = input[c with c[dim] = index[c]]} over the index shape, for INT32 indices or INT64 indices
     * narrowed by Java {@code (int)}. Recorded with {@code GatherGrad} (saving the index) when {@code input} needs
     * grad.
     */
    public static Tensor gather(Tensor input, int dim, Tensor index) {
        Objects.requireNonNull(input, "Input tensor cannot be null");
        Objects.requireNonNull(index, "Index tensor cannot be null");
        Supplier<Tensor> raw = () -> {
            Tensor gpu;
            try (Tensor index32 = deviceInt32(index, input)) {
                gpu = GpuTensorOps.gather(input, dim, index32 != null ? index32 : index);
            }
            return gpu != null ? gpu : CpuIndexing.gather(input, dim, index);
        };
        return !Autograd.needsGrad(input) ? raw.get() : recordOver(raw, Arrays.asList(input, index),
                out -> saving(new GradFunctions.GatherGrad(dim, input.shape()), f -> f.saveForBackward(index)));
    }

    /**
     * A copy of {@code input} with {@code src} written at the indexed positions; the last write wins. Indices as for
     * {@link #gather}. Recorded with {@code ScatterGrad} (saving the index) when {@code input} or {@code src} needs
     * grad.
     */
    public static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src) {
        return scatter(input, dim, index, src, false);
    }

    /**
     * A copy of {@code input} with {@code src} added at the indexed positions, in row-major index order. Indices as
     * for {@link #gather}. Recorded with {@code ScatterAddGrad} (saving the index) when {@code input} or {@code src}
     * needs grad.
     */
    public static Tensor scatterAdd(Tensor input, int dim, Tensor index, Tensor src) {
        return scatter(input, dim, index, src, true);
    }

    private static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src, boolean add) {
        Objects.requireNonNull(input, "Input tensor cannot be null");
        Objects.requireNonNull(index, "Index tensor cannot be null");
        Objects.requireNonNull(src, "Source tensor cannot be null");
        Supplier<Tensor> raw = () -> {
            Tensor gpu;
            try (Tensor index32 = deviceInt32(index, input, src)) {
                Tensor idx = index32 != null ? index32 : index;
                gpu = add ? GpuTensorOps.scatterAdd(input, dim, idx, src) : GpuTensorOps.scatter(input, dim, idx, src);
            }
            return gpu != null ? gpu : CpuIndexing.scatter(input, dim, index, src, add);
        };
        if (!Autograd.needsGrad(input, src)) {
            return raw.get();
        }
        boolean needInput = input.requiresGrad();
        boolean needSrc = src.requiresGrad();
        return recordOver(raw, Arrays.asList(input, index, src), out -> add
                ? saving(new SelectionGradFunctions.ScatterAddGrad(dim, needInput, needSrc),
                        f -> f.saveForBackward(index, src))
                : saving(new GradFunctions.ScatterGrad(dim, needInput, needSrc), f -> f.saveForBackward(index, src)));
    }

    /**
     * Rows of a {@code [rows, dim]} table selected by INT32 or INT64 indices: {@code out[i, :] = weight[indices[i], :]}
     * with shape {@code indices.shape + [dim]}. With {@code clamp} indices are clamped into the table; without it an
     * out-of-range index throws {@link IndexOutOfBoundsException}. Recorded as
     * {@link #indexSelectRows(Tensor, Tensor, boolean, int)} without a padding row.
     */
    public static Tensor indexSelectRows(Tensor weight, Tensor indices, boolean clamp) {
        return indexSelectRows(weight, indices, clamp, -1);
    }

    /**
     * {@link #indexSelectRows(Tensor, Tensor, boolean)} whose backward leaves the table row {@code paddingIdx} without
     * gradient (Embedding's padding row; negative for none). The forward copies that row like any other. Recorded
     * with {@code IndexSelectRowsGrad} (saving the indices) when {@code weight} needs grad.
     *
     * @throws IllegalArgumentException when {@code paddingIdx} is not below the table's row count, and as
     *                                  {@link GpuNormOps#indexSelectRows}
     */
    public static Tensor indexSelectRows(Tensor weight, Tensor indices, boolean clamp, int paddingIdx) {
        Objects.requireNonNull(weight, "weight must not be null");
        Objects.requireNonNull(indices, "indices must not be null");
        if (paddingIdx >= 0 && weight.shape().rank() == 2 && paddingIdx >= weight.shape().dim(0)) {
            throw new IllegalArgumentException("paddingIdx " + paddingIdx + " must be below the "
                    + weight.shape().dim(0) + " table rows (negative for none)");
        }
        Supplier<Tensor> raw = () -> orCpu(GpuNormOps.indexSelectRows(weight, indices, clamp),
                () -> CpuIndexing.indexSelectRows(weight, indices, clamp));
        return !Autograd.needsGrad(weight) ? raw.get() : recordOver(raw, Arrays.asList(weight, indices),
                out -> saving(new IndexGradFunctions.IndexSelectRowsGrad(weight.shape(), clamp, paddingIdx),
                        f -> f.saveForBackward(indices)));
    }

    /**
     * Gradient of {@link #indexSelectRows}: a new {@code [rows, dim]} FLOAT32 table with
     * {@code gw[k, :] = sum of gy[i, :]} over the {@code i} (ascending) whose index {@code k = indices[i]} (clamped
     * into the table with {@code clamp}) names row {@code k}; out-of-range indices without {@code clamp} and row
     * {@code skipRow} get nothing. Single-rounding class (same-order float sums from 0). A backward primitive:
     * recorded only in a create-graph pass, with {@code IndexAddRowsGrad} over {@code gy}.
     *
     * @param gy      FLOAT32 gradient {@code [n, dim]}, one row per index
     * @param indices n INT32 or INT64 indices of any shape
     * @param rows    the table's row count
     * @param clamp   whether the forward clamped indices into the table
     * @param skipRow the row that gets no gradient, negative for none
     * @throws IllegalArgumentException      when the shapes, dtypes or backends do not match, {@code rows} is negative
     *                                       or {@code skipRow} is not below {@code rows}
     * @throws UnsupportedOperationException when the indices are neither INT32 nor INT64
     */
    public static Tensor indexAddRows(Tensor gy, Tensor indices, int rows, boolean clamp, int skipRow) {
        return primitive(() -> orCpu(GpuIndexGradOps.indexAddRows(gy, indices, rows, clamp, skipRow),
                () -> CpuIndexing.indexAddRows(gy, indices, rows, clamp, skipRow)), Collections.singletonList(gy),
                out -> SecondOrderGradFunctions.indexAddRows(gy, indices, rows, clamp, skipRow));
    }

    // ------------------------------------------------------------------ cat, split, stack

    /** Concatenation along {@code dim} as a new contiguous tensor. Recorded with {@code CatGrad}. */
    public static Tensor cat(List<Tensor> tensors, int dim) {
        int d = CpuShapeOps.normalizeDim(dim, first(tensors, "concatenate").shape().rank());
        Shape resultShape = CpuShapeOps.catShape(tensors, d);
        Supplier<Tensor> raw = () -> catRaw(tensors, d, resultShape);
        return !Autograd.needsGrad(tensors) ? raw.get() : recordOver(raw, tensors,
                out -> new GradFunctions.CatGrad(d, splitPoints(tensors, d), requiresGradFlags(tensors)));
    }

    private static Tensor catRaw(List<Tensor> tensors, int d, Shape resultShape) {
        if (!onDevice(tensors)) {
            return CpuShapeOps.cat(tensors, d, resultShape);
        }
        Tensor out = deviceOutput(tensors.get(0), resultShape);
        try {
            int offset = 0;
            for (Tensor t : tensors) {
                int size = t.shape().dim(d);
                if (size > 0) {
                    try (Tensor slot = out.slice(d, offset, offset + size)) {
                        GpuCoreOps.copyInto(t, slot);
                    }
                }
                offset += size;
            }
            return out;
        } catch (RuntimeException e) {
            out.close();
            throw e;
        }
    }

    /**
     * {@code chunks} equal parts of {@code a} along {@code dim}, each a fresh contiguous copy. Each part is recorded
     * with its own {@code SliceGrad} (zero-size parts included).
     */
    public static List<Tensor> split(Tensor a, int chunks, int dim) {
        int d = CpuShapeOps.normalizeDim(dim, check(a).shape().rank());
        int dimSize = a.shape().dim(d);
        if (chunks <= 0) {
            throw new IllegalArgumentException("Number of chunks must be positive, got " + chunks);
        }
        if (dimSize % chunks != 0) {
            throw new IllegalArgumentException("Dimension size " + dimSize + " is not divisible by " + chunks);
        }
        int chunkSize = dimSize / chunks;
        if (!Autograd.needsGrad(a)) {
            return splitRaw(a, chunks, d, chunkSize);
        }
        List<Tensor> parts = Autograd.untracked(() -> splitRaw(a, chunks, d, chunkSize));
        Shape shape = a.shape();
        return Autograd.recordEach(parts, Collections.singletonList(a),
                i -> new GradFunctions.SliceGrad(shape, d, i * chunkSize, (i + 1) * chunkSize));
    }

    private static List<Tensor> splitRaw(Tensor a, int chunks, int d, int chunkSize) {
        if (!GpuTensorOps.shouldUseGpu(a)) {
            return CpuShapeOps.split(a, chunks, d);
        }
        List<Tensor> parts = new ArrayList<>(chunks);
        try {
            for (int i = 0; i < chunks; i++) {
                parts.add(chunkSize == 0 ? deviceZeros(a, a.shape()) : contiguousSlice(a, d, i * chunkSize,
                        (i + 1) * chunkSize));
            }
        } catch (RuntimeException e) {
            throw CpuShapeOps.closeAll(parts, e);
        }
        return parts;
    }

    /**
     * The tensors, which share one shape, stacked along a new axis {@code dim} (in {@code [-rank-1, rank]}). Recorded
     * with {@code StackGrad}.
     */
    public static Tensor stack(List<Tensor> tensors, int dim) {
        int rank = first(tensors, "stack").shape().rank();
        int d = dim < 0 ? rank + dim + 1 : dim;
        if (d < 0 || d > rank) {
            throw new IndexOutOfBoundsException("Dimension " + dim + " out of bounds for stack (valid: 0 to " + rank
                    + ")");
        }
        Shape resultShape = CpuShapeOps.stackShape(tensors, d);
        Supplier<Tensor> raw = () -> stackRaw(tensors, d, resultShape);
        return !Autograd.needsGrad(tensors) ? raw.get() : recordOver(raw, tensors,
                out -> new GradFunctions.StackGrad(d, tensors.size(), requiresGradFlags(tensors)));
    }

    private static Tensor stackRaw(List<Tensor> tensors, int d, Shape resultShape) {
        if (!onDevice(tensors)) {
            return CpuShapeOps.stack(tensors, d, resultShape);
        }
        Tensor out = deviceOutput(tensors.get(0), resultShape);
        try {
            for (int i = 0; i < tensors.size(); i++) {
                try (Tensor slot = out.slice(d, i, i + 1);
                     Tensor src = tensors.get(i).unsqueeze(d)) {
                    GpuCoreOps.copyInto(src, slot);
                }
            }
            return out;
        } catch (RuntimeException e) {
            out.close();
            throw e;
        }
    }

    // ------------------------------------------------------------------ in place

    /**
     * {@code dst += alpha * src} in place, element by element; {@code dst} may be any view that addresses each
     * element once (offset slices and transposes included), {@code src} any view of {@code dst}'s shape. Never
     * recorded.
     *
     * @throws IllegalArgumentException when the shapes or backends differ or {@code dst} addresses an element twice
     * @throws IllegalStateException    when {@code dst} requires grad while grad mode is on
     */
    public static void addInPlace(Tensor dst, Tensor src, float alpha) {
        Objects.requireNonNull(dst, "Destination tensor cannot be null");
        Objects.requireNonNull(src, "Source tensor cannot be null");
        if (!dst.shape().equals(src.shape())) {
            throw new IllegalArgumentException("Tensors must have the same shape for in-place add");
        }
        CpuShapeOps.requireSameBackend(dst, src);
        GpuTensorOps.requireWritable(dst);
        if (!GpuTensorOps.axpbyInPlace(dst, alpha, src, 1.0f)) {
            CpuElementwise.addInPlace(dst, src, alpha);
        }
    }

    /**
     * {@code dst *= scalar} in place; {@code dst} may be any view that addresses each element once. Never recorded.
     *
     * @throws IllegalStateException when {@code dst} requires grad while grad mode is on
     */
    public static void mulInPlace(Tensor dst, float scalar) {
        Objects.requireNonNull(dst, "Tensor cannot be null");
        GpuTensorOps.requireWritable(dst);
        if (!GpuTensorOps.scaleInPlace(dst, scalar)) {
            CpuElementwise.mulInPlace(dst, scalar);
        }
    }

    /**
     * Sets every element of a FLOAT32 tensor to {@code +0.0f} in place; {@code dst} may be any view that addresses
     * each element once. Never recorded.
     *
     * @throws UnsupportedOperationException when {@code dst} is not FLOAT32
     * @throws IllegalStateException         when {@code dst} requires grad while grad mode is on
     */
    public static void zeroInPlace(Tensor dst) {
        Objects.requireNonNull(dst, "Tensor cannot be null");
        if (dst.dtype() != DType.FLOAT32) {
            throw new UnsupportedOperationException("zeroInPlace() only supported for FLOAT32 tensors, got: "
                    + dst.dtype());
        }
        GpuTensorOps.requireWritable(dst);
        TrainGpu g = TrainGpu.forBackend(dst.backend());
        if (g == null) {
            CpuElementwise.zeroInPlace(dst);
        } else if (dst.isDense()) {
            StridedView view = StridedView.of(dst);
            g.fill(view.buffer(), view.offset(), view.numElements(), 0.0f);
        } else {
            try (Tensor zero = deviceZeros(dst, Shape.scalar())) {
                GpuCoreOps.copyInto(zero, dst);
            }
        }
    }

    // ------------------------------------------------------------------ recording

    /**
     * The facade recording pattern for one differentiable operand {@code a}: without grad the raw result, else the raw
     * result computed untracked and recorded as one node.
     */
    private static Tensor recordUnary(Tensor a, Supplier<Tensor> raw, Function<Tensor, GradientFunction> gradFn) {
        return !Autograd.needsGrad(a) ? raw.get() : recordOver(raw, Collections.singletonList(a), gradFn);
    }

    /** As {@link #recordUnary} for the differentiable operands {@code a} and {@code b}. */
    private static Tensor recordBinary(Tensor a, Tensor b, Supplier<Tensor> raw,
                                       Function<Tensor, GradientFunction> gradFn) {
        return !Autograd.needsGrad(a, b) ? raw.get() : recordOver(raw, Arrays.asList(a, b), gradFn);
    }

    /**
     * A backward primitive: {@code raw}, recorded over its differentiable operands
     * {@code inputs} with the second-order function {@code gradFn} builds only inside a create-graph pass and when
     * one of them requires grad.
     */
    private static Tensor primitive(Supplier<Tensor> raw, List<Tensor> inputs,
                                    Function<Tensor, GradientFunction> gradFn) {
        return Autograd.isCreatingGraph() && Autograd.needsGrad(inputs) ? recordOver(raw, inputs, gradFn) : raw.get();
    }

    /** {@code raw} computed with grad mode off, then recorded over {@code inputs} (the caller checked needsGrad). */
    private static Tensor recordOver(Supplier<Tensor> raw, List<Tensor> inputs,
                                     Function<Tensor, GradientFunction> gradFn) {
        return Autograd.record(Autograd.untracked(raw), inputs, gradFn);
    }

    /** {@code fn} after {@code save} stored its tensors (one {@code saveForBackward} call, atomic on failure). */
    private static <F extends GradientFunction> F saving(F fn, Consumer<F> save) {
        save.accept(fn);
        return fn;
    }

    /** {@code gpu}, the dispatcher's result, or the CPU reference when the dispatcher declined (null). */
    private static Tensor orCpu(Tensor gpu, Supplier<Tensor> cpu) {
        return gpu != null ? gpu : cpu.get();
    }

    /** Cumulative sizes along {@code d}: the end of each tensor's range in the concatenation. */
    private static int[] splitPoints(List<Tensor> tensors, int d) {
        int[] points = new int[tensors.size()];
        int cumulative = 0;
        for (int i = 0; i < points.length; i++) {
            cumulative += tensors.get(i).shape().dim(d);
            points[i] = cumulative;
        }
        return points;
    }

    private static boolean[] requiresGradFlags(List<Tensor> tensors) {
        boolean[] flags = new boolean[tensors.size()];
        for (int i = 0; i < flags.length; i++) {
            flags[i] = tensors.get(i).requiresGrad();
        }
        return flags;
    }

    // ------------------------------------------------------------------ device compositions

    /** GLU ({@code gelu == false}) or GeGLU of the halves of {@code dim} from slice views; null for the CPU path. */
    private static Tensor gatedOnDevice(Tensor a, int dim, Shape outShape, boolean gelu) {
        if (!GpuTensorOps.shouldUseGpu(a)) {
            return null;
        }
        int half = outShape.dim(dim);
        if (half == 0) {
            return deviceZeros(a, outShape);
        }
        try (Tensor first = a.slice(dim, 0, half);
             Tensor second = a.slice(dim, half, 2L * half);
             Tensor gate = requireDevice(gelu ? GpuTensorOps.gelu(second) : GpuTensorOps.sigmoid(second))) {
            return requireDevice(GpuTensorOps.mul(first, gate));
        }
    }

    /** A fresh contiguous device copy of {@code a} sliced to {@code [start, end)} along {@code dim}. */
    private static Tensor contiguousSlice(Tensor a, int dim, int start, int end) {
        try (Tensor view = a.slice(dim, start, end)) {
            return requireDevice(GpuCoreOps.contiguous(view));
        }
    }

    /**
     * An INT64 {@code index} narrowed to INT32 on the device with Java {@code (int)} (the low word, the same values
     * the CPU path reads through {@link Tensor#toIntArray()}), so the dispatcher, which binds INT32 indices only, can
     * take it. Null, leaving {@code index} as it is, when it is not INT64, a value tensor is not FLOAT32 or lives on
     * another backend, or the train library is unavailable. The caller closes the result.
     */
    private static Tensor deviceInt32(Tensor index, Tensor... values) {
        if (index.dtype() != DType.INT64) {
            return null;
        }
        for (Tensor v : values) {
            if (v.dtype() != DType.FLOAT32 || v.backend() != index.backend()) {
                return null;
            }
        }
        return GpuCoreOps.cast(index, DType.INT32);
    }

    /** Whether every tensor is FLOAT32 and the train library is available on their (shared) backend. */
    private static boolean onDevice(List<Tensor> tensors) {
        for (Tensor t : tensors) {
            if (t.dtype() != DType.FLOAT32) {
                return false;
            }
        }
        return TrainGpu.forBackend(tensors.get(0).backend()) != null;
    }

    /** A new uninitialized contiguous FLOAT32 device tensor of {@code shape} on {@code like}'s backend. */
    private static Tensor deviceOutput(Tensor like, Shape shape) {
        TrainGpu g = TrainGpu.forBackend(like.backend());
        if (g == null) {
            throw unavailable(like);
        }
        return Tensor.fromStorage(like.backend(), shape, DType.FLOAT32, g.allocate(shape.numElements(),
                DType.FLOAT32));
    }

    /** A zero-filled FLOAT32 device tensor of {@code shape} on {@code like}'s backend. */
    private static Tensor deviceZeros(Tensor like, Shape shape) {
        Tensor zeros = GpuCoreOps.zeros(like.backend(), shape, DType.FLOAT32);
        if (zeros == null) {
            throw unavailable(like);
        }
        return zeros;
    }

    /** {@code t}, which a dispatcher returned after the library was found available. */
    private static Tensor requireDevice(Tensor t) {
        if (t == null) {
            throw new IllegalStateException("the train library became unavailable during a device composition");
        }
        return t;
    }

    private static IllegalStateException unavailable(Tensor like) {
        return new IllegalStateException("train kernels unavailable on " + like.backend().backendType() + ": "
                + TrainGpu.availability(like.backend()).reason());
    }

    // ------------------------------------------------------------------ argument checks

    private static Tensor check(Tensor a) {
        return Objects.requireNonNull(a, "Tensor cannot be null");
    }

    private static void checkPair(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        CpuShapeOps.requireSameBackend(a, b);
    }

    private static Tensor first(List<Tensor> tensors, String op) {
        Objects.requireNonNull(tensors, "Tensor list cannot be null");
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Cannot " + op + " empty tensor list");
        }
        for (Tensor t : tensors) {
            Objects.requireNonNull(t, "Tensor list cannot contain null");
        }
        return tensors.get(0);
    }
}

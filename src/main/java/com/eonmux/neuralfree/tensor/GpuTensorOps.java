package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Backward;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Binary;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Compare;
import com.eonmux.neuralfree.tensor.gpu.ElementwiseOp.Unary;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Elementwise, compare, select, gather/scatter, activation-backward and dropout-apply ops on the train kernels
 * ({@code kernels/train/eltwise.cl}). Results are new contiguous tensors that stay on the device; inputs may be any
 * view (offsets, transposes, broadcast strides), and nothing is read back.
 *
 * <p><b>Dispatch.</b> A method returns {@code null} (an in-place method {@code false}) exactly when the train library
 * is unavailable on the tensors' backend ({@link TrainGpu#forBackend}) or a dtype is not the one documented (FLOAT32
 * values, INT32 indices, FLOAT32 or BOOL conditions); that is the caller's CPU path. Once the library is available,
 * argument errors throw like the CPU path and kernel failures propagate. Results never require grad, whatever the
 * inputs' flags; only the recording facades mark a result as requiring grad.</p>
 *
 * <p><b>In-place writes.</b> {@link #axpbyInPlace} and {@link #scaleInPlace} throw {@link IllegalStateException},
 * before touching anything, when the destination requires grad while grad mode is enabled; parameter updates, EMA and
 * target-network copies run under {@link ComputationGraph#noGrad()}.</p>
 *
 * <p><b>Index errors.</b> An out-of-range gather or scatter index is detected on the device: the element is written
 * as 0 (gather) or skipped (scatter), and {@link IndexOutOfBoundsException} is thrown at the next host
 * synchronization of the backend (a readback, {@link TrainGpu#checkErrors}).</p>
 *
 * <p><b>Tolerance against the CPU path:</b></p>
 * <ul>
 *   <li>bit-identical: neg, abs, relu, clamp, round, floor, identity, compare, where, gather, scatter (a NaN result
 *   of round or floor is NaN, possibly with another payload, since devices may canonicalize NaN);</li>
 *   <li>single rounding (bit-identical for normal operands and results; a result below {@code Float.MIN_NORMAL} may
 *   flush to a signed zero): add, sub, mul, addScalar, mulScalar, leakyRelu, axpby, scale, maskScale, scatterAdd,
 *   and the relu, leaky-relu, abs, clamp, sigmoid, tanh and elu backwards (arithmetic on the given {@code y});</li>
 *   <li>division and square root ({@code |gpu - ref| <= 4 ulp(ref)} or {@code <= Float.MIN_NORMAL}): div, sqrt;</li>
 *   <li>transcendental ({@code |gpu - ref| <= 1e-5 * max(1, |ref|)}): exp, log, sin, cos, sigmoid, tanh, gelu, silu,
 *   elu, softplus, mish, pow, and the other backwards. sin and cos are the OpenCL and CUDA builtins (within 4 and 2
 *   ulp) and, on Vulkan, an explicit range reduction and polynomial within {@code 2^-22} of the exact value; all
 *   three hold for every finite argument, so the class holds over the whole float range.</li>
 * </ul>
 */
public final class GpuTensorOps {

    /** Message of the in-place guard {@link #requireWritable}. */
    static final String IN_PLACE_WITH_GRAD = "in-place write to a tensor that requires grad while grad mode is "
            + "enabled; wrap it in ComputationGraph.noGrad()";

    private GpuTensorOps() {
    }

    // ------------------------------------------------------------------ availability

    /** Whether {@code a} is FLOAT32 and the train library is available on its backend. */
    public static boolean shouldUseGpu(Tensor a) {
        return a != null && a.dtype() == DType.FLOAT32 && TrainGpu.forBackend(a.backend()) != null;
    }

    /** Whether both tensors are FLOAT32 on one backend and the train library is available there. */
    public static boolean shouldUseGpu(Tensor a, Tensor b) {
        return b != null && a != null && a.backend() == b.backend() && b.dtype() == DType.FLOAT32 && shouldUseGpu(a);
    }

    /** Releases the train context of every backend ({@link TrainGpu#releaseAll}). */
    public static void clearCache() {
        TrainGpu.releaseAll();
    }

    /** Releases the train context of one backend ({@link TrainGpu#release}). */
    public static void clearCache(ComputeBackend backend) {
        TrainGpu.release(backend);
    }

    // ------------------------------------------------------------------ unary ops

    public static Tensor relu(Tensor a) {
        return unary(Unary.RELU, a, 0.0f, 0.0f);
    }

    public static Tensor gelu(Tensor a) {
        return unary(Unary.GELU, a, 0.0f, 0.0f);
    }

    public static Tensor silu(Tensor a) {
        return unary(Unary.SILU, a, 0.0f, 0.0f);
    }

    public static Tensor sigmoid(Tensor a) {
        return unary(Unary.SIGMOID, a, 0.0f, 0.0f);
    }

    public static Tensor tanh(Tensor a) {
        return unary(Unary.TANH, a, 0.0f, 0.0f);
    }

    public static Tensor exp(Tensor a) {
        return unary(Unary.EXP, a, 0.0f, 0.0f);
    }

    public static Tensor log(Tensor a) {
        return unary(Unary.LOG, a, 0.0f, 0.0f);
    }

    /** {@code (float) Math.sin(x)} per element, accurate over the whole float range on every backend. */
    public static Tensor sin(Tensor a) {
        return unary(Unary.SIN, a, 0.0f, 0.0f);
    }

    /** {@code (float) Math.cos(x)} per element, accurate over the whole float range on every backend. */
    public static Tensor cos(Tensor a) {
        return unary(Unary.COS, a, 0.0f, 0.0f);
    }

    public static Tensor sqrt(Tensor a) {
        return unary(Unary.SQRT, a, 0.0f, 0.0f);
    }

    public static Tensor neg(Tensor a) {
        return unary(Unary.NEG, a, 0.0f, 0.0f);
    }

    public static Tensor abs(Tensor a) {
        return unary(Unary.ABS, a, 0.0f, 0.0f);
    }

    public static Tensor addScalar(Tensor a, float scalar) {
        return unary(Unary.ADD_SCALAR, a, scalar, 0.0f);
    }

    public static Tensor mulScalar(Tensor a, float scalar) {
        return unary(Unary.MUL_SCALAR, a, scalar, 0.0f);
    }

    /** {@code (float) Math.pow(x, exponent)} per element, with every {@code Math.pow} special case. */
    public static Tensor pow(Tensor a, float exponent) {
        return unary(Unary.POW, a, exponent, 0.0f);
    }

    /** {@code Math.max(min, Math.min(max, x))} per element (NaN bounds and inputs as Java gives them). */
    public static Tensor clamp(Tensor a, float min, float max) {
        return unary(Unary.CLAMP, a, min, max);
    }

    public static Tensor leakyRelu(Tensor a, float negativeSlope) {
        return unary(Unary.LEAKY_RELU, a, negativeSlope, 0.0f);
    }

    public static Tensor elu(Tensor a, float alpha) {
        return unary(Unary.ELU, a, alpha, 0.0f);
    }

    public static Tensor softplus(Tensor a) {
        return unary(Unary.SOFTPLUS, a, 0.0f, 0.0f);
    }

    public static Tensor mish(Tensor a) {
        return unary(Unary.MISH, a, 0.0f, 0.0f);
    }

    /** {@code Math.round(x)} kept as a float; a cast to INT32 then gives Java's {@code int} result. */
    public static Tensor round(Tensor a) {
        return unary(Unary.ROUND, a, 0.0f, 0.0f);
    }

    /** {@code (float) Math.floor(x)} per element. */
    public static Tensor floor(Tensor a) {
        return unary(Unary.FLOOR, a, 0.0f, 0.0f);
    }

    /**
     * {@code op} applied to every element of {@code a}, with the parameters {@link Unary} documents; null when
     * {@code a} is not FLOAT32 or the library is unavailable.
     */
    public static Tensor unary(Unary op, Tensor a, float p0, float p1) {
        Objects.requireNonNull(op, "op must not be null");
        TrainGpu g = floatGpu(a);
        if (g == null) {
            return null;
        }
        Tensor out = newTensor(g, a.shape(), DType.FLOAT32);
        return build(g, out, temps -> launchUnary(g, op, StridedView.of(a), StridedView.of(out), p0, p1));
    }

    // ------------------------------------------------------------------ binary ops (broadcasting)

    public static Tensor add(Tensor a, Tensor b) {
        return binary(Binary.ADD, a, b);
    }

    public static Tensor sub(Tensor a, Tensor b) {
        return binary(Binary.SUB, a, b);
    }

    public static Tensor mul(Tensor a, Tensor b) {
        return binary(Binary.MUL, a, b);
    }

    public static Tensor div(Tensor a, Tensor b) {
        return binary(Binary.DIV, a, b);
    }

    /** {@code a op b} over the broadcast shape ({@link BroadcastEngine}); the shape check throws like the CPU path. */
    private static Tensor binary(Binary op, Tensor a, Tensor b) {
        Objects.requireNonNull(op, "op must not be null");
        TrainGpu g = floatGpu(a, b);
        if (g == null) {
            return null;
        }
        Shape shape = BroadcastEngine.broadcastShape(a.shape(), b.shape());
        Tensor out = newTensor(g, shape, DType.FLOAT32);
        return build(g, out, temps -> {
            StridedView[][] p = split(StridedView.broadcast(a, shape), StridedView.broadcast(b, shape),
                    StridedView.of(out));
            for (StridedView[] v : p) {
                int n = v[0].numElements();
                g.launch1d(Kernels.BINARY_F32, n, args(op.id(), n, v[0].rank(), v[0].dimArgs(), v[0].strideArgs(),
                        v[1].strideArgs(), v[0].offset(), v[1].offset(), v[2].offset(), v[0].buffer(), v[1].buffer(),
                        v[2].buffer()));
            }
        });
    }

    // ------------------------------------------------------------------ in-place ops

    /**
     * {@code y = alpha * x + beta * y} in place (two roundings and a sum, no contraction; when {@code beta} is
     * {@code +-0}, {@code y} is not read, so a NaN in {@code y} does not propagate). {@code y} may be any view that
     * addresses each element once; {@code x} may be any view, including one of {@code y}'s buffer. Returns false when
     * a tensor is not FLOAT32 or the library is unavailable.
     *
     * @throws IllegalArgumentException when the tensors live on different backends, their shapes differ, or {@code y}
     *                                  addresses an element more than once
     * @throws IllegalStateException    when {@code y} requires grad while grad mode is enabled (on every backend)
     */
    public static boolean axpbyInPlace(Tensor y, float alpha, Tensor x, float beta) {
        TrainGpu g = floatGpu(y, x);
        requireWritable(y);
        if (g == null) {
            return false;
        }
        if (!y.shape().equals(x.shape())) {
            throw new IllegalArgumentException("Tensors must have the same shape for in-place add");
        }
        requireDistinctElements(y);
        run(g, temps -> {
            Tensor target = y.isDense() ? y : temps.own(denseCopy(y));
            StridedView tv = StridedView.of(target);
            StridedView xv = StridedView.of(temps.dense(x));
            if (xv.buffer() == tv.buffer() && xv.offset() != tv.offset()) {
                xv = StridedView.of(temps.own(denseCopy(x)));
            }
            int n = tv.numElements();
            g.launch1d(Kernels.AXPBY_F32, n, n, alpha, beta, xv.offset(), tv.offset(), xv.buffer(), tv.buffer());
            if (target != y) {
                GpuCoreOps.copyInto(target, y);
            }
        });
        return true;
    }

    /**
     * {@code y = y * alpha} in place; {@code y} may be any view that addresses each element once. Returns false when
     * {@code y} is not FLOAT32 or the library is unavailable.
     *
     * @throws IllegalArgumentException when {@code y} addresses an element more than once
     * @throws IllegalStateException    when {@code y} requires grad while grad mode is enabled (on every backend)
     */
    public static boolean scaleInPlace(Tensor y, float alpha) {
        TrainGpu g = floatGpu(y);
        requireWritable(y);
        if (g == null) {
            return false;
        }
        requireDistinctElements(y);
        run(g, temps -> {
            if (y.isDense()) {
                StridedView v = StridedView.of(y);
                g.launch1d(Kernels.SCALE_F32, v.numElements(), v.numElements(), alpha, v.offset(), v.buffer());
                return;
            }
            Tensor scaled = temps.own(newTensor(g, y.shape(), DType.FLOAT32));
            launchUnary(g, Unary.MUL_SCALAR, StridedView.of(y), StridedView.of(scaled), alpha, 0.0f);
            GpuCoreOps.copyInto(scaled, y);
        });
        return true;
    }

    // ------------------------------------------------------------------ compare and select

    /** BOOL {@code a == b} over the broadcast shape (Java float semantics), or null for the CPU path. */
    public static Tensor eq(Tensor a, Tensor b) {
        return compare(Compare.EQ, a, b);
    }

    /** BOOL {@code a > b} over the broadcast shape (Java float semantics), or null for the CPU path. */
    public static Tensor gt(Tensor a, Tensor b) {
        return compare(Compare.GT, a, b);
    }

    /** BOOL {@code a < b} over the broadcast shape (Java float semantics), or null for the CPU path. */
    public static Tensor lt(Tensor a, Tensor b) {
        return compare(Compare.LT, a, b);
    }

    /** BOOL {@code a op b} over the broadcast shape (Java float comparisons: NaN false, -0 equal to +0). */
    private static Tensor compare(Compare op, Tensor a, Tensor b) {
        Objects.requireNonNull(op, "op must not be null");
        TrainGpu g = floatGpu(a, b);
        if (g == null) {
            return null;
        }
        Shape shape = BroadcastEngine.broadcastShape(a.shape(), b.shape());
        Tensor out = newTensor(g, shape, DType.BOOL);
        return build(g, out, temps -> {
            StridedView[] v = StridedView.coalesce(StridedView.broadcast(a, shape), StridedView.broadcast(b, shape));
            if (v[0].rank() > StridedView.MAX_RANK) {
                v = StridedView.coalesce(StridedView.of(temps.materialize(g, a, shape)),
                        StridedView.of(temps.materialize(g, b, shape)));
            }
            int n = v[0].numElements();
            g.launch1d(Kernels.COMPARE_F32, (n + 3L) / 4, args(op.id(), n, v[0].rank(), v[0].dimArgs(),
                    v[0].strideArgs(), v[1].strideArgs(), v[0].offset(), v[1].offset(), v[0].buffer(), v[1].buffer(),
                    out.storage().gpuBuffer()));
        });
    }

    /**
     * {@code cond != 0 ? a : b} per element, with a FLOAT32 condition ({@code != 0.0f}: NaN selects {@code a}, +-0
     * selects {@code b}) or a BOOL condition (non-zero byte). All three tensors have one shape, as the CPU path
     * requires; any of them may be a strided view. Null when {@code a} or {@code b} is not FLOAT32, the condition is
     * neither FLOAT32 nor BOOL, or the library is unavailable.
     *
     * @throws IllegalArgumentException when the tensors live on different backends or their shapes differ
     */
    public static Tensor where(Tensor cond, Tensor a, Tensor b) {
        Objects.requireNonNull(cond, "Condition tensor cannot be null");
        TrainGpu g = floatGpu(a, b);
        CpuShapeOps.requireSameBackend(cond, a);
        if (g == null || (cond.dtype() != DType.FLOAT32 && cond.dtype() != DType.BOOL)) {
            return null;
        }
        if (!cond.shape().equals(a.shape()) || !cond.shape().equals(b.shape())) {
            throw new IllegalArgumentException("All tensors must have the same shape for where");
        }
        int kind = cond.dtype() == DType.BOOL ? TrainOps.TR_WHERE_COND_BOOL : TrainOps.TR_WHERE_COND_F32;
        Tensor out = newTensor(g, cond.shape(), DType.FLOAT32);
        return build(g, out, temps -> {
            StridedView[][] p = split(StridedView.of(cond), StridedView.of(a), StridedView.of(b), StridedView.of(out));
            Tensor staging = p.length == 1 ? out : temps.own(newTensor(g, Shape.of(p[0][0].numElements()),
                    DType.FLOAT32));
            GpuBuffer part = staging.storage().gpuBuffer();
            for (StridedView[] v : p) {
                int n = v[0].numElements();
                g.launch1d(Kernels.WHERE_F32, n, args(kind, n, v[0].rank(), v[0].dimArgs(), v[0].strideArgs(),
                        v[1].strideArgs(), v[2].strideArgs(), v[0].offset(), v[1].offset(), v[2].offset(),
                        v[0].buffer(), v[1].buffer(), v[2].buffer(), part));
                if (p.length > 1) {
                    g.launch1d(Kernels.COPY_F32, n, n, 0, v[3].offset(), part, v[3].buffer());
                }
            }
        });
    }

    // ------------------------------------------------------------------ gather and scatter

    /**
     * {@code out[c] = input[c with c[dim] = index[c]]} over the index shape ({@code TensorOps.gather}); null when the
     * input is not FLOAT32, the index is not INT32, or the library is unavailable. An index outside
     * {@code [0, input.dim(dim))} raises {@link IndexOutOfBoundsException} at the next synchronization.
     *
     * @throws IllegalArgumentException  when the tensors live on different backends, their ranks differ, or an index
     *                                   dim other than {@code dim} exceeds the input's
     * @throws IndexOutOfBoundsException when {@code dim} is outside {@code [-rank, rank)}
     */
    public static Tensor gather(Tensor input, int dim, Tensor index) {
        Objects.requireNonNull(input, "Input tensor cannot be null");
        Objects.requireNonNull(index, "Index tensor cannot be null");
        CpuShapeOps.requireSameBackend(input, index);
        TrainGpu g = indexGpu(input, index);
        if (g == null) {
            return null;
        }
        int d = CpuShapeOps.checkIndexShapes(input.shape(), dim, index.shape());
        Tensor out = newTensor(g, index.shape(), DType.FLOAT32);
        if (index.numElements() == 0) {
            return out;
        }
        return build(g, out, temps -> {
            int rank = index.shape().rank();
            int n = (int) index.numElements();
            if (rank <= StridedView.MAX_RANK) {
                launchGather(g, d, n, index.shape().dims(), input.shape().dims(), StridedView.of(temps.dense(input)),
                        StridedView.of(temps.dense(index)), out, d);
                return;
            }
            Tensor cropped = temps.dense(temps.crop(input, d, index.shape()));
            launchGather(g, 1, n, merged(index.shape(), d, index.shape().dim(d)),
                    merged(index.shape(), d, input.shape().dim(d)), StridedView.of(cropped),
                    StridedView.of(temps.dense(index)), out, d);
        });
    }

    /**
     * A copy of {@code input} with {@code out[c with c[dim] = index[c]] = src[c]} for every index element in row-major
     * order, so the last write wins ({@code TensorOps.scatter}); null when the input or source is not FLOAT32, the
     * index is not INT32, or the library is unavailable. An index outside {@code [0, input.dim(dim))} is skipped and
     * raises {@link IndexOutOfBoundsException} at the next synchronization.
     *
     * @throws IllegalArgumentException  when the tensors live on different backends, the index and source shapes
     *                                   differ, the ranks differ, or an index dim other than {@code dim} exceeds the
     *                                   input's
     * @throws IndexOutOfBoundsException when {@code dim} is outside {@code [-rank, rank)}
     */
    public static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src) {
        return scatter(input, dim, index, src, TrainOps.TR_SCATTER_OVERWRITE);
    }

    /**
     * A copy of {@code input} with {@code out[c with c[dim] = index[c]] += src[c]}, accumulated in row-major index
     * order (the order of {@code GradFunctions.GatherGrad}); otherwise as {@link #scatter}.
     */
    public static Tensor scatterAdd(Tensor input, int dim, Tensor index, Tensor src) {
        return scatter(input, dim, index, src, TrainOps.TR_SCATTER_ADD);
    }

    private static Tensor scatter(Tensor input, int dim, Tensor index, Tensor src, int mode) {
        Objects.requireNonNull(input, "Input tensor cannot be null");
        Objects.requireNonNull(index, "Index tensor cannot be null");
        Objects.requireNonNull(src, "Source tensor cannot be null");
        CpuShapeOps.requireSameBackend(input, index);
        CpuShapeOps.requireSameBackend(input, src);
        TrainGpu g = indexGpu(input, index);
        if (g == null || src.dtype() != DType.FLOAT32) {
            return null;
        }
        if (!index.shape().equals(src.shape())) {
            throw new IllegalArgumentException("Index and source must have the same shape");
        }
        int d = CpuShapeOps.checkIndexShapes(input.shape(), dim, index.shape());
        Tensor out = denseCopy(input);
        if (index.numElements() == 0) {
            return out;
        }
        return build(g, out, temps -> {
            int rank = index.shape().rank();
            StridedView iv = StridedView.of(temps.dense(index));
            StridedView sv = StridedView.of(temps.dense(src));
            if (rank <= StridedView.MAX_RANK) {
                launchScatter(g, d, mode, index.shape().dims(), input.shape().dims(), iv, sv, out, d);
                return;
            }
            Tensor region = temps.crop(out, d, index.shape());
            boolean direct = region.isDense() && StridedView.of(region).offset() == 0;
            Tensor target = direct ? region : temps.own(denseCopy(region));
            launchScatter(g, 1, mode, merged(index.shape(), d, index.shape().dim(d)),
                    merged(index.shape(), d, input.shape().dim(d)), iv, sv, target, d);
            if (!direct) {
                GpuCoreOps.copyInto(target, region);
            }
        });
    }

    // ------------------------------------------------------------------ activation backward and dropout apply

    /**
     * {@code gx = gy * f'} for the forward function of {@code op}, from the forward input {@code x} and/or output
     * {@code y} as {@link Backward#readsX()} and {@link Backward#readsY()} state (an operand the op does not read may
     * be null and is ignored), with the forward parameters {@code p0} and {@code p1}. Null when a used tensor is not
     * FLOAT32 or the library is unavailable.
     *
     * @throws NullPointerException     when a tensor the op reads is null
     * @throws IllegalArgumentException when the tensors live on different backends or a used shape differs from
     *                                  {@code gy}'s
     */
    public static Tensor unaryBackward(Backward op, Tensor x, Tensor y, Tensor gy, float p0, float p1) {
        Objects.requireNonNull(op, "op must not be null");
        Objects.requireNonNull(gy, "Gradient tensor cannot be null");
        Tensor xs = op.readsX() ? Objects.requireNonNull(x, "x must not be null for " + op) : null;
        Tensor ys = op.readsY() ? Objects.requireNonNull(y, "y must not be null for " + op) : null;
        for (Tensor t : new Tensor[]{xs, ys}) {
            if (t != null) {
                CpuShapeOps.requireSameBackend(t, gy);
                if (!t.shape().equals(gy.shape())) {
                    throw new IllegalArgumentException("Saved tensor shape " + t.shape() + " does not match gradient "
                            + "shape " + gy.shape());
                }
            }
        }
        boolean floats = (xs == null || xs.dtype() == DType.FLOAT32) && (ys == null || ys.dtype() == DType.FLOAT32);
        TrainGpu g = floats ? floatGpu(gy) : null;
        if (g == null) {
            return null;
        }
        Tensor out = newTensor(g, gy.shape(), DType.FLOAT32);
        return build(g, out, temps -> {
            StridedView gv = StridedView.of(temps.dense(gy));
            StridedView xv = xs == null ? null : StridedView.of(temps.dense(xs));
            StridedView yv = ys == null ? null : StridedView.of(temps.dense(ys));
            int n = gv.numElements();
            g.launch1d(Kernels.UNARY_BACKWARD_F32, n, op.id(), n, p0, p1, xv == null ? 0 : xv.offset(),
                    yv == null ? 0 : yv.offset(), gv.offset(), 0, xv == null ? g.placeholder() : xv.buffer(),
                    yv == null ? g.placeholder() : yv.buffer(), gv.buffer(), out.storage().gpuBuffer());
        });
    }

    /**
     * Dropout apply: {@code mask == +-0 ? +0 : (x * mask) * scale} per element, so a dropped element is 0 even for a
     * non-finite {@code x}. Null when a tensor is not FLOAT32 or the library is unavailable.
     *
     * @throws IllegalArgumentException when the tensors live on different backends or their shapes differ
     */
    public static Tensor maskScale(Tensor x, Tensor mask, float scale) {
        TrainGpu g = floatGpu(x, mask);
        if (g == null) {
            return null;
        }
        if (!x.shape().equals(mask.shape())) {
            throw new IllegalArgumentException("Mask shape " + mask.shape() + " does not match input shape "
                    + x.shape());
        }
        Tensor out = newTensor(g, x.shape(), DType.FLOAT32);
        return build(g, out, temps -> {
            StridedView xv = StridedView.of(temps.dense(x));
            StridedView mv = StridedView.of(temps.dense(mask));
            int n = xv.numElements();
            g.launch1d(Kernels.MASK_SCALE_F32, n, n, scale, xv.offset(), mv.offset(), 0, xv.buffer(), mv.buffer(),
                    out.storage().gpuBuffer());
        });
    }

    // ------------------------------------------------------------------ launches

    /** TR_Unary_F32 from view {@code in} into view {@code out} of the same dims; one launch per leading index. */
    private static void launchUnary(TrainGpu g, Unary op, StridedView in, StridedView out, float p0, float p1) {
        for (StridedView[] v : split(in, out)) {
            int n = v[0].numElements();
            g.launch1d(Kernels.UNARY_F32, n, args(op.id(), n, v[0].rank(), v[0].dimArgs(), v[0].strideArgs(),
                    v[0].offset(), v[1].offset(), p0, p1, v[0].buffer(), v[1].buffer()));
        }
    }

    /**
     * TR_Gather_F32 over index dims {@code d} (rank <= 6) and input dims {@code id}; out is contiguous at 0.
     * {@code errDim} is the caller's dimension, which the deferred out-of-range error reports; it differs from
     * {@code dim} when the dimensions were merged.
     */
    private static void launchGather(TrainGpu g, int dim, int n, int[] d, int[] id, StridedView in, StridedView idx,
                                     Tensor out, int errDim) {
        g.launch1d(Kernels.GATHER_F32, n, args(dim, n, d.length, slots(d), slots(id), in.offset(), idx.offset(),
                TrainGpu.ERROR_SLOT_GATHER, errDim, in.buffer(), idx.buffer(), out.storage().gpuBuffer(),
                g.errors()));
    }

    /**
     * TR_Scatter_F32 over index dims {@code d} (rank <= 6) into {@code out} of dims {@code od}, which is dense at
     * element 0 of its buffer and pre-filled; one work-item per fiber (every coordinate except {@code dim}).
     * {@code errDim} is the caller's dimension, which the deferred out-of-range error reports; it differs from
     * {@code dim} when the dimensions were merged.
     */
    private static void launchScatter(TrainGpu g, int dim, int mode, int[] d, int[] od, StridedView idx,
                                      StridedView src, Tensor out, int errDim) {
        long fibers = 1;
        for (int k = 0; k < d.length; k++) {
            fibers *= k == dim ? 1 : d[k];
        }
        g.launch1d(Kernels.SCATTER_F32, fibers, args(dim, mode, (int) fibers, d.length, slots(d), slots(od),
                idx.offset(), src.offset(), TrainGpu.ERROR_SLOT_SCATTER, errDim, idx.buffer(), src.buffer(),
                out.storage().gpuBuffer(), g.errors()));
    }

    // ------------------------------------------------------------------ argument checks

    /** The context when {@code a} is FLOAT32 and the library is available, else null. */
    private static TrainGpu floatGpu(Tensor a) {
        Objects.requireNonNull(a, "Tensor cannot be null");
        return a.dtype() == DType.FLOAT32 ? TrainGpu.forBackend(a.backend()) : null;
    }

    /** The context when both tensors are FLOAT32 and the library is available, else null. */
    private static TrainGpu floatGpu(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "Tensor a cannot be null");
        Objects.requireNonNull(b, "Tensor b cannot be null");
        CpuShapeOps.requireSameBackend(a, b);
        return b.dtype() == DType.FLOAT32 ? floatGpu(a) : null;
    }

    /** The context when the values are FLOAT32, the index INT32 and the library available, else null. */
    private static TrainGpu indexGpu(Tensor values, Tensor index) {
        return index.dtype() == DType.INT32 ? floatGpu(values) : null;
    }

    /**
     * The in-place guard: throws {@link IllegalStateException} when {@code dst} requires grad while
     * grad mode is enabled, since the write would change a tensor the autograd graph may read. The grad-mode check
     * (one ThreadLocal read) runs only for a destination that requires grad.
     */
    static void requireWritable(Tensor dst) {
        if (dst.requiresGrad() && ComputationGraph.isGradEnabled()) {
            throw new IllegalStateException(IN_PLACE_WITH_GRAD);
        }
    }

    private static void requireDistinctElements(Tensor t) {
        for (int k = 0; k < t.shape().rank(); k++) {
            if (t.shape().dim(k) > 1 && t.strides().stride(k) == 0) {
                throw new IllegalArgumentException("in-place destination " + t.shape() + " with strides " + t.strides()
                        + " addresses an element more than once");
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    /**
     * A new contiguous tensor with fresh storage (byte size rounded up to 4, exact length) that does not require
     * grad.
     */
    private static Tensor newTensor(TrainGpu g, Shape shape, DType dtype) {
        TensorStorage storage = g.allocate(shape.numElements(), dtype);
        return Tensor.wrap(shape, dtype, Strides.contiguous(shape), storage, g.backend(), false);
    }

    /** A new contiguous device copy of a FLOAT32 or INT32 tensor whose backend has the library. */
    private static Tensor denseCopy(Tensor t) {
        Tensor copy = GpuCoreOps.contiguous(t);
        if (copy == null) {
            throw new IllegalStateException("device copy of a " + t.dtype() + " tensor is unavailable on "
                    + t.backend().backendType());
        }
        return copy;
    }

    /** Index dims merged to {@code [before dim, middle, after dim]}, the rank-3 form of a rank above 6. */
    private static int[] merged(Shape index, int dim, int middle) {
        long before = 1;
        long after = 1;
        for (int k = 0; k < index.rank(); k++) {
            if (k < dim) {
                before *= index.dim(k);
            } else if (k > dim) {
                after *= index.dim(k);
            }
        }
        return new int[]{(int) before, middle, (int) after};
    }

    /**
     * Views of identical dims, coalesced jointly and split to rank 6 or less: one row per launch, holding the views
     * in argument order. The parts of a contiguous view are contiguous blocks at their offsets.
     */
    private static StridedView[][] split(StridedView... views) {
        StridedView[] merged = StridedView.coalesce(views);
        StridedView[][] byView = new StridedView[merged.length][];
        for (int k = 0; k < merged.length; k++) {
            byView[k] = merged[k].splitLeading();
        }
        StridedView[][] rows = new StridedView[byView[0].length][merged.length];
        for (int i = 0; i < rows.length; i++) {
            for (int k = 0; k < merged.length; k++) {
                rows[i][k] = byView[k][i];
            }
        }
        return rows;
    }

    /** The six dim slots of a kernel: {@code dims} padded with 1. */
    private static Object[] slots(int[] dims) {
        Object[] out = new Object[StridedView.MAX_RANK];
        for (int k = 0; k < out.length; k++) {
            out[k] = k < dims.length ? dims[k] : 1;
        }
        return out;
    }

    /** Kernel arguments in order, with every {@code Object[]} part flattened in place. */
    private static Object[] args(Object... parts) {
        List<Object> out = new ArrayList<>(40);
        for (Object p : parts) {
            if (p instanceof Object[]) {
                out.addAll(Arrays.asList((Object[]) p));
            } else {
                out.add(p);
            }
        }
        return out.toArray();
    }

    /** Runs {@code body} under the backend lock, closing its temporaries afterwards. */
    private static void run(TrainGpu g, Body body) {
        g.exclusive(() -> {
            try (Temps temps = new Temps()) {
                body.run(temps);
            }
            return null;
        });
    }

    /** {@link #run} that returns {@code out}, closing it when the sequence fails. */
    private static Tensor build(TrainGpu g, Tensor out, Body body) {
        try {
            run(g, body);
            return out;
        } catch (RuntimeException e) {
            throw closeAfter(out, e);
        }
    }

    /** Closes {@code t} and returns {@code failure}, with a close failure added to it (or returned when it is null). */
    private static RuntimeException closeAfter(Tensor t, RuntimeException failure) {
        try {
            t.close();
        } catch (RuntimeException e) {
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
        }
        return failure;
    }

    /** A launch sequence over temporaries. */
    private interface Body {
        void run(Temps temps);
    }

    /**
     * Temporaries of one op: views and device copies, closed when the op's launches are enqueued (the backends defer
     * releasing buffers that pending kernels use).
     */
    private static final class Temps implements AutoCloseable {
        private final List<Tensor> owned = new ArrayList<>();

        Tensor own(Tensor t) {
            owned.add(t);
            return t;
        }

        /** {@code t} when it is dense (at any offset), else an owned contiguous copy. */
        Tensor dense(Tensor t) {
            return t.isDense() ? t : own(denseCopy(t));
        }

        /** An owned contiguous copy of {@code t} broadcast to {@code shape}. */
        Tensor materialize(TrainGpu g, Tensor t, Shape shape) {
            Tensor m = own(newTensor(g, shape, t.dtype()));
            GpuCoreOps.copyInto(t, m);
            return m;
        }

        /**
         * {@code t} sliced to {@code [0, index.dim(k))} on every dim {@code k != dim} (a view sharing its storage).
         * The slices are internal temporaries taken under {@link ComputationGraph#noGrad()}, so they never record,
         * whatever the caller's grad mode.
         */
        Tensor crop(Tensor t, int dim, Shape index) {
            Tensor view = t;
            try (ComputationGraph.NoGradScope untracked = ComputationGraph.noGrad()) {
                for (int k = 0; k < index.rank(); k++) {
                    if (k != dim && index.dim(k) < t.shape().dim(k)) {
                        view = own(view.slice(k, 0, index.dim(k)));
                    }
                }
            }
            return view;
        }

        @Override
        public void close() {
            RuntimeException failure = null;
            for (int i = owned.size() - 1; i >= 0; i--) {
                failure = closeAfter(owned.get(i), failure);
            }
            owned.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }
}

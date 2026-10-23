package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.gpu.StridedView;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainOps;
import com.eonmux.neuralfree.tensor.gpu.TrainOps.Kernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.function.IntPredicate;

/**
 * Optimizer steps, gradient clipping and loss-scale unscaling on the train kernels ({@code kernels/train/optim.cl}),
 * with parameters, gradients and optimizer state resident on the device.
 *
 * <p><b>Contract.</b> Every method returns {@code false} ({@link #unscaleAndCheck}: an empty result) exactly when the
 * train library is unavailable on the tensors' backend ({@link TrainGpu#forBackend}) or a tensor is not FLOAT32; that
 * is the caller's CPU path. Once the library is available, kernel and argument failures propagate. Updates are in
 * place on the given tensors. Every tensor of one call lives on one backend.</p>
 *
 * <p><b>In-place guard.</b> The optimizer steps and {@link #storeFloats} throw
 * {@link IllegalStateException}, on every backend and before anything is written, when a tensor they write requires
 * grad while grad mode is enabled: the optimizers run their steps under {@code ComputationGraph.noGrad()}, as every
 * other parameter write does. The gradient-list methods (clipping, unscaling) rewrite gradient slots outside the
 * graph and are not guarded.</p>
 *
 * <p><b>Views.</b> A dense view (row-major, any element offset) is bound directly. Any other strided view is staged
 * through a contiguous device copy and, when written, copied back with {@link GpuCoreOps#copyInto}. A written tensor
 * must not address an element twice, and the written tensors of one step must address disjoint elements. A gradient
 * that shares elements with a written tensor is staged first, so every step reads the gradient as it was before the
 * step, as the CPU loops do.</p>
 *
 * <p><b>Numerics.</b> Optimizer steps and the clip-by-norm coefficient follow the formulas of the CPU loops
 * operation by operation and fall in the reduction class ({@code |gpu - ref64| <= 1e-5 * sqrt(K) * sum|terms| + K *
 * FLT_MIN} against a double-precision reference with {@code K} terms); clip-by-value is bit-identical
 * (Java {@code Math.max}/{@code Math.min}); unscaling multiplies by {@code invScale} and is exact for a power-of-two
 * scale.</p>
 *
 * <p><b>Ordering.</b> Each call enqueues its launches on the compute queue as one {@link TrainGpu#exclusive}
 * sequence, without a host synchronization; {@link #unscaleAndCheck} ends with its single flag readback, a documented
 * sync point.</p>
 */
public final class GpuOptimizerOps {

    /** Scratch slots of this dispatcher's sequences (valid only inside one exclusive sequence). */
    private static final int SLOT_NORMS = 0;
    private static final int SLOT_COEF = 1;
    private static final int SLOT_PARTIAL = 2;
    private static final int SLOT_LAMB_UPDATE = 3;
    private static final int SLOT_FLAG = 4;
    private static final int SLOT_UNUSED_STATE = 5;

    /** One reduced row: sequential kernel up to this many columns. */
    private static final int SMALL_COLS = 64;
    /** One reduced row above this many columns is reduced in chunks. */
    private static final int PARTIAL_MIN_COLS = 65536;
    private static final int COLS_PER_CHUNK = 8192;
    private static final int MAX_CHUNKS = 1024;

    private GpuOptimizerOps() {
    }

    // ------------------------------------------------------------------ hyperparameters

    /** SGD step hyperparameters ({@code optim/SGD.java}); momentum is applied when {@code momentum > 0}. */
    public static final class SgdParams {
        final float lr;
        final float momentum;
        final float weightDecay;
        final boolean nesterov;

        public SgdParams(float lr, float momentum, float weightDecay, boolean nesterov) {
            this.lr = lr;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
            this.nesterov = nesterov;
        }

        boolean hasMomentum() {
            return momentum > 0f;
        }
    }

    /**
     * Adam or AdamW step hyperparameters, built by {@link #adam} or {@link #adamW}, which fix the bias-correction
     * contract of each mode.
     */
    public static final class AdamParams {
        final int mode;
        final float lr;
        final float beta1;
        final float beta2;
        final float eps;
        final float weightDecay;
        final float bc1;
        final float bc2;

        private AdamParams(int mode, float lr, float beta1, float beta2, float eps, float weightDecay, float bc1,
                           float bc2) {
            this.mode = mode;
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.weightDecay = weightDecay;
            this.bc1 = bc1;
            this.bc2 = bc2;
        }

        /**
         * Adam with L2 decay ({@code optim/Adam.java}): {@code stepSize = lr / (1 - beta1^t)} as Adam computes it, and
         * {@code bc2 = 1 - beta2^t}. The kernel's first-moment correction is then 1.
         */
        public static AdamParams adam(float stepSize, float beta1, float beta2, float eps, float weightDecay,
                                      float bc2) {
            return new AdamParams(TrainOps.TR_ADAM_MODE_ADAM, stepSize, beta1, beta2, eps, weightDecay, 1.0f, bc2);
        }

        /**
         * AdamW with decoupled decay ({@code optim/AdamW.java}): the raw learning rate, {@code bc1 = 1 - beta1^t} and
         * {@code bc2 = 1 - beta2^t}; the kernel takes the step size {@code lr / bc1}.
         */
        public static AdamParams adamW(float lr, float beta1, float beta2, float eps, float weightDecay, float bc1,
                                       float bc2) {
            return new AdamParams(TrainOps.TR_ADAM_MODE_ADAMW, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
        }
    }

    /** RMSprop step hyperparameters ({@code optim/RMSprop.java}); momentum is applied when {@code momentum > 0}. */
    public static final class RmspropParams {
        final float lr;
        final float alpha;
        final float eps;
        final float weightDecay;
        final float momentum;
        final boolean centered;

        public RmspropParams(float lr, float alpha, float eps, float weightDecay, float momentum, boolean centered) {
            this.lr = lr;
            this.alpha = alpha;
            this.eps = eps;
            this.weightDecay = weightDecay;
            this.momentum = momentum;
            this.centered = centered;
        }

        boolean hasMomentum() {
            return momentum > 0f;
        }
    }

    /** Adagrad step hyperparameters ({@code optim/Adagrad.java}). */
    public static final class AdagradParams {
        final float lr;
        final float eps;
        final float weightDecay;

        public AdagradParams(float lr, float eps, float weightDecay) {
            this.lr = lr;
            this.eps = eps;
            this.weightDecay = weightDecay;
        }
    }

    /**
     * LAMB step hyperparameters ({@code optim/LAMB.java}): {@code bc1 = 1 - beta1^t} and {@code bc2 = 1 - beta2^t}
     * with bias correction, {@code bc1 = bc2 = 1} without it.
     */
    public static final class LambParams {
        final float lr;
        final float beta1;
        final float beta2;
        final float eps;
        final float weightDecay;
        final float bc1;
        final float bc2;

        public LambParams(float lr, float beta1, float beta2, float eps, float weightDecay, float bc1, float bc2) {
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.weightDecay = weightDecay;
            this.bc1 = bc1;
            this.bc2 = bc2;
        }
    }

    // ------------------------------------------------------------------ availability and caches

    /**
     * Whether the optimizer steps run on the device for this parameter: the train library is available on its
     * backend and the parameter (and its gradient, when set) is FLOAT32.
     */
    public static boolean shouldUseGpu(Parameter param) {
        if (param == null || param.tensor() == null || param.tensor().dtype() != DType.FLOAT32) {
            return false;
        }
        Tensor grad = param.grad();
        if (grad != null && grad.dtype() != DType.FLOAT32) {
            return false;
        }
        return TrainGpu.forBackend(param.backend()) != null;
    }

    /** Releases the train context of every backend ({@link TrainGpu#releaseAll}). */
    public static void clearCache() {
        TrainGpu.releaseAll();
    }

    /** Releases the train context of one backend ({@link TrainGpu#release}). */
    public static void clearCache(ComputeBackend backend) {
        TrainGpu.release(backend);
    }

    // ------------------------------------------------------------------ optimizer steps

    /**
     * One SGD step on {@code p} ({@code optim/SGD.java}); {@code velocityOrNull} is the momentum buffer, required
     * when {@code momentum > 0} and ignored otherwise.
     *
     * @throws IllegalArgumentException when momentum is used without a velocity buffer, or as described in the class
     *                                  contract (element counts, backends, overlapping written tensors)
     */
    public static boolean sgdStep(Tensor p, Tensor g, Tensor velocityOrNull, SgdParams h) {
        Objects.requireNonNull(h, "SGD parameters must not be null");
        requireOperands(p, g);
        Tensor v = h.hasMomentum() ? requireState(velocityOrNull, "velocity (momentum > 0)") : null;
        requireWritable(p, v);
        TrainGpu gpu = context(p, g, v);
        if (gpu == null) {
            return false;
        }
        int n = elementCount(p, g, v);
        if (n == 0) {
            return true;
        }
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] w = st.bindWritten(p, v);
                Operand pg = st.bindGradient(g, w);
                GpuBuffer vBuf = v == null ? unusedState(gpu) : w[1].buffer;
                int vOff = v == null ? 0 : w[1].offset;
                gpu.launch1d(Kernels.SGD_STEP_F32, n, n, h.lr, h.momentum, h.weightDecay, h.nesterov ? 1 : 0,
                        h.hasMomentum() ? 1 : 0, w[0].offset, pg.offset, vOff, w[0].buffer, pg.buffer, vBuf);
                st.writeBack(w);
            }
            return null;
        });
        return true;
    }

    /** One Adam or AdamW step on {@code p} with first and second moments {@code m} and {@code v}. */
    public static boolean adamStep(Tensor p, Tensor g, Tensor m, Tensor v, AdamParams h) {
        Objects.requireNonNull(h, "Adam parameters must not be null");
        requireOperands(p, g);
        requireState(m, "first moment");
        requireState(v, "second moment");
        requireWritable(p, m, v);
        TrainGpu gpu = context(p, g, m, v);
        if (gpu == null) {
            return false;
        }
        int n = elementCount(p, g, m, v);
        if (n == 0) {
            return true;
        }
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] w = st.bindWritten(p, m, v);
                Operand pg = st.bindGradient(g, w);
                gpu.launch1d(Kernels.ADAM_STEP_F32, n, n, h.mode, h.lr, h.beta1, h.beta2, h.eps, h.weightDecay,
                        h.bc1, h.bc2, w[0].offset, pg.offset, w[1].offset, w[2].offset,
                        w[0].buffer, pg.buffer, w[1].buffer, w[2].buffer);
                st.writeBack(w);
            }
            return null;
        });
        return true;
    }

    /**
     * One RMSprop step on {@code p} with the squared-gradient average {@code sq}; {@code bufOrNull} (momentum buffer)
     * is required when {@code momentum > 0} and {@code gavgOrNull} (gradient average) when centered, and each is
     * ignored otherwise.
     */
    public static boolean rmspropStep(Tensor p, Tensor g, Tensor sq, Tensor bufOrNull, Tensor gavgOrNull,
                                      RmspropParams h) {
        Objects.requireNonNull(h, "RMSprop parameters must not be null");
        requireOperands(p, g);
        requireState(sq, "squared-gradient average");
        Tensor buf = h.hasMomentum() ? requireState(bufOrNull, "momentum buffer (momentum > 0)") : null;
        Tensor gavg = h.centered ? requireState(gavgOrNull, "gradient average (centered)") : null;
        requireWritable(p, sq, buf, gavg);
        TrainGpu gpu = context(p, g, sq, buf, gavg);
        if (gpu == null) {
            return false;
        }
        int n = elementCount(p, g, sq, buf, gavg);
        if (n == 0) {
            return true;
        }
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] w = st.bindWritten(p, sq, buf, gavg);
                Operand pg = st.bindGradient(g, w);
                GpuBuffer unused = buf == null || gavg == null ? unusedState(gpu) : null;
                gpu.launch1d(Kernels.RMSPROP_STEP_F32, n, n, h.lr, h.alpha, h.eps, h.weightDecay, h.momentum,
                        h.centered ? 1 : 0, h.hasMomentum() ? 1 : 0, w[0].offset, pg.offset, w[1].offset,
                        buf == null ? 0 : w[2].offset, gavg == null ? 0 : w[3].offset,
                        w[0].buffer, pg.buffer, w[1].buffer, buf == null ? unused : w[2].buffer,
                        gavg == null ? unused : w[3].buffer);
                st.writeBack(w);
            }
            return null;
        });
        return true;
    }

    /** One Adagrad step on {@code p} with the accumulator {@code acc} ({@code optim/Adagrad.java}). */
    public static boolean adagradStep(Tensor p, Tensor g, Tensor acc, AdagradParams h) {
        Objects.requireNonNull(h, "Adagrad parameters must not be null");
        requireOperands(p, g);
        requireState(acc, "accumulator");
        requireWritable(p, acc);
        TrainGpu gpu = context(p, g, acc);
        if (gpu == null) {
            return false;
        }
        int n = elementCount(p, g, acc);
        if (n == 0) {
            return true;
        }
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] w = st.bindWritten(p, acc);
                Operand pg = st.bindGradient(g, w);
                gpu.launch1d(Kernels.ADAGRAD_STEP_F32, n, n, h.lr, h.eps, h.weightDecay, w[0].offset, pg.offset,
                        w[1].offset, w[0].buffer, pg.buffer, w[1].buffer);
                st.writeBack(w);
            }
            return null;
        });
        return true;
    }

    /**
     * One LAMB step on {@code p} with moments {@code m} and {@code v} ({@code optim/LAMB.java}): the update and the
     * moments (TR_LambUpdate), the norms of the old parameter and of the update (TR_ReduceRows SUMSQ), then the
     * trust-ratio step (TR_LambApply). The update and the norms live in scratch.
     */
    public static boolean lambStep(Tensor p, Tensor g, Tensor m, Tensor v, LambParams h) {
        Objects.requireNonNull(h, "LAMB parameters must not be null");
        requireOperands(p, g);
        requireState(m, "first moment");
        requireState(v, "second moment");
        requireWritable(p, m, v);
        TrainGpu gpu = context(p, g, m, v);
        if (gpu == null) {
            return false;
        }
        int n = elementCount(p, g, m, v);
        if (n == 0) {
            return true;
        }
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] w = st.bindWritten(p, m, v);
                Operand pg = st.bindGradient(g, w);
                GpuBuffer u = gpu.scratch(SLOT_LAMB_UPDATE, (long) n * Float.BYTES);
                GpuBuffer norms = gpu.scratch(SLOT_NORMS, 2L * Float.BYTES);
                gpu.launch1d(Kernels.LAMB_UPDATE_F32, n, n, h.beta1, h.beta2, h.eps, h.weightDecay, h.bc1, h.bc2,
                        w[0].offset, pg.offset, w[1].offset, w[2].offset, 0,
                        w[0].buffer, pg.buffer, w[1].buffer, w[2].buffer, u);
                sumOfSquares(gpu, w[0].buffer, w[0].offset, n, norms, 0);
                sumOfSquares(gpu, u, 0, n, norms, 1);
                gpu.launch1d(Kernels.LAMB_APPLY_F32, n, n, h.lr, h.eps, w[0].offset, 0, 0, w[0].buffer, u, norms);
                st.writeBack(w);
            }
            return null;
        });
        return true;
    }

    // ------------------------------------------------------------------ gradient lists

    /**
     * Clips gradients by their global L2 norm in place ({@code training/GradientClipping.java}): the norm
     * is taken over the list as given, so a gradient listed twice counts twice, as the CPU loop over parameters
     * does; the scale {@code maxNorm / (norm + 1e-6f)}, applied when {@code norm > maxNorm}, reaches each distinct
     * view once. The norm never leaves the device. An empty list needs no work and returns true.
     *
     * @throws IllegalArgumentException when two listed gradients share elements without being the same view, or a
     *                                  gradient addresses an element more than once (a broadcast view)
     */
    public static boolean clipGradNorm(List<Tensor> grads, float maxNorm) {
        GradientSet set = GradientSet.of(grads);
        if (set.isEmpty()) {
            return true;
        }
        TrainGpu gpu = set.context();
        if (gpu == null) {
            return false;
        }
        set.resolveViews();
        int count = grads.size();
        gpu.exclusive(() -> {
            GpuBuffer norms = gpu.scratch(SLOT_NORMS, (long) count * Float.BYTES);
            GpuBuffer coef = gpu.scratch(SLOT_COEF, 2L * Float.BYTES);
            try (Staging st = new Staging()) {
                Operand[] bound = st.bindRegions(set);
                for (int i = 0; i < count; i++) {
                    Operand o = bound[set.regionOf[i]];
                    sumOfSquares(gpu, o.buffer, o.offset, set.regions.get(set.regionOf[i]).elements, norms, i);
                }
                gpu.launch1d(Kernels.GRAD_NORM_COEF_F32, 1, count, maxNorm, 0, 0, norms, coef);
                for (int r = 0; r < bound.length; r++) {
                    int n = set.regions.get(r).elements;
                    gpu.launch1d(Kernels.SCALE_BY_DEVICE_F32, n, n, bound[r].offset, 0, bound[r].buffer, coef);
                }
                st.writeBack(bound);
            }
            return null;
        });
        return true;
    }

    /**
     * Clamps every gradient element to {@code [-maxValue, maxValue]} in place, {@code Math.max(-maxValue,
     * Math.min(maxValue, x))} (NaN stays NaN), once per distinct view. An empty list returns true.
     *
     * @throws IllegalArgumentException when {@code maxValue} is negative or NaN, two listed gradients share
     *                                  elements without being the same view, or a gradient addresses an element
     *                                  more than once (a broadcast view)
     */
    public static boolean clipGradValue(List<Tensor> grads, float maxValue) {
        if (!(maxValue >= 0f)) {
            throw new IllegalArgumentException("maxValue must be non-negative, got " + maxValue);
        }
        GradientSet set = GradientSet.of(grads);
        if (set.isEmpty()) {
            return true;
        }
        TrainGpu gpu = set.context();
        if (gpu == null) {
            return false;
        }
        set.resolveViews();
        gpu.exclusive(() -> {
            try (Staging st = new Staging()) {
                Operand[] bound = st.bindRegions(set);
                for (int r = 0; r < bound.length; r++) {
                    int n = set.regions.get(r).elements;
                    Operand o = bound[r];
                    gpu.launch1d(Kernels.UNARY_F32, n, TrainOps.TR_E_CLAMP, n, 1, n, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            o.offset, o.offset, -maxValue, maxValue, o.buffer, o.buffer);
                }
                st.writeBack(bound);
            }
            return null;
        });
        return true;
    }

    /**
     * Multiplies every gradient element by {@code invScale} in place, once per distinct view, and reports whether any
     * result is non-finite (the mixed-precision overflow check, {@code precision/MixedPrecisionTrainer.java}). The
     * CPU path divides by the loss scale; {@code invScale = 1 / scale} gives the same quotient for a power-of-two
     * scale. One int is read back.
     *
     * @return empty when the library is unavailable or a gradient is not FLOAT32 (nothing was changed), otherwise 1
     *         when a non-finite value was produced and 0 when not; 0 for an empty list
     * @throws IllegalArgumentException when two listed gradients share elements without being the same view, or a
     *                                  gradient addresses an element more than once (a broadcast view)
     */
    public static OptionalInt unscaleAndCheck(List<Tensor> grads, float invScale) {
        GradientSet set = GradientSet.of(grads);
        if (set.isEmpty()) {
            return OptionalInt.of(0);
        }
        TrainGpu gpu = set.context();
        if (gpu == null) {
            return OptionalInt.empty();
        }
        set.resolveViews();
        int flag = gpu.exclusive(() -> {
            GpuBuffer flagBuf = gpu.scratch(SLOT_FLAG, Integer.BYTES);
            gpu.fill(flagBuf, 0, 1, 0.0f);
            try (Staging st = new Staging()) {
                Operand[] bound = st.bindRegions(set);
                for (int r = 0; r < bound.length; r++) {
                    int n = set.regions.get(r).elements;
                    gpu.launch1d(Kernels.UNSCALE_CHECK_F32, n, n, invScale, bound[r].offset, 0, bound[r].buffer,
                            flagBuf);
                }
                st.writeBack(bound);
            }
            return gpu.readFlag(flagBuf, 0);
        });
        return OptionalInt.of(flag != 0 ? 1 : 0);
    }

    /**
     * Whether two tensors are the same view of one buffer (equal buffer, element offset, and dims and strides after
     * coalescing), which the gradient-list methods treat as one gradient.
     *
     * @throws IllegalArgumentException when a tensor exceeds the 32-bit kernel limit ({@link StridedView#of})
     */
    public static boolean sameView(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "first tensor must not be null");
        Objects.requireNonNull(b, "second tensor must not be null");
        return Region.of(a).sameView(Region.of(b));
    }

    /**
     * Whether two tensors address a common element of one buffer; exact for strided views, so interleaved disjoint
     * views do not share elements. Two different views that share elements are rejected by the gradient-list methods.
     *
     * @throws IllegalArgumentException when a tensor exceeds the 32-bit kernel limit ({@link StridedView#of})
     */
    public static boolean sharesElements(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "first tensor must not be null");
        Objects.requireNonNull(b, "second tensor must not be null");
        return Region.of(a).overlaps(Region.of(b));
    }

    // ------------------------------------------------------------------ CPU path support

    /**
     * Writes {@code values}, in row-major order, into the FLOAT32 elements {@code target} addresses, through
     * {@link TensorStorage#copyFromHostFloat(float[], long, ComputeQueue)} on the backend's transfer queue, which
     * respects the storage offset and orders the write after pending train work: one write for a dense view,
     * one per run of consecutive elements otherwise. The optimizers' CPU path stores parameters and state with it.
     *
     * @throws IllegalArgumentException when the dtype is not FLOAT32, the lengths differ, or the view addresses an
     *                                  element more than once
     * @throws IllegalStateException    when {@code target} requires grad while grad mode is enabled
     */
    public static void storeFloats(Tensor target, float[] values) {
        Objects.requireNonNull(target, "target must not be null");
        Objects.requireNonNull(values, "values must not be null");
        GpuTensorOps.requireWritable(target);
        if (target.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("storeFloats needs a FLOAT32 tensor, got " + target.dtype());
        }
        if (values.length != target.numElements()) {
            throw new IllegalArgumentException(values.length + " values for a tensor of " + target.numElements()
                    + " elements " + target.shape());
        }
        if (values.length == 0) {
            return;
        }
        ComputeQueue queue = target.backend().transferQueue();
        TensorStorage storage = target.storage();
        Strides strides = target.strides();
        if (target.isDense()) {
            closeEvent(storage.copyFromHostFloat(values, strides.offset(), queue));
            return;
        }
        int rank = target.shape().rank();
        int[] dims = target.shape().dims();
        for (int k = 0; k < rank; k++) {
            if (dims[k] > 1 && strides.stride(k) == 0) {
                throw new IllegalArgumentException("view " + target.shape() + " with strides " + strides
                        + " addresses an element more than once");
            }
        }
        int[] idx = new int[rank];
        long address = strides.offset();
        int runStart = 0;
        long runAddress = address;
        for (int i = 1; i <= values.length; i++) {
            if (i < values.length) {
                for (int k = rank - 1; k >= 0; k--) {
                    if (++idx[k] < dims[k]) {
                        address += strides.stride(k);
                        break;
                    }
                    address -= (long) (dims[k] - 1) * strides.stride(k);
                    idx[k] = 0;
                }
                if (address == runAddress + (i - runStart)) {
                    continue;
                }
            }
            float[] run = Arrays.copyOfRange(values, runStart, i);
            closeEvent(storage.copyFromHostFloat(run, runAddress, queue));
            runStart = i;
            runAddress = address;
        }
    }

    // ------------------------------------------------------------------ internals

    private static void requireOperands(Tensor p, Tensor g) {
        Objects.requireNonNull(p, "parameter must not be null");
        Objects.requireNonNull(g, "gradient must not be null");
    }

    /** The in-place guard ({@link GpuTensorOps#requireWritable}) over the written tensors of a step; nulls skipped. */
    private static void requireWritable(Tensor... written) {
        for (Tensor t : written) {
            if (t != null) {
                GpuTensorOps.requireWritable(t);
            }
        }
    }

    private static Tensor requireState(Tensor t, String what) {
        if (t == null) {
            throw new IllegalArgumentException("the " + what + " tensor must not be null");
        }
        return t;
    }

    /**
     * The train context of the tensors' backend, or null when it is unavailable or a tensor is not FLOAT32. Null
     * entries are skipped.
     *
     * @throws IllegalArgumentException when the tensors live on different backends
     */
    private static TrainGpu context(Tensor... tensors) {
        ComputeBackend backend = null;
        boolean float32 = true;
        for (Tensor t : tensors) {
            if (t == null) {
                continue;
            }
            if (backend == null) {
                backend = t.backend();
            } else if (t.backend() != backend) {
                throw new IllegalArgumentException("optimizer tensors live on different backends: " + backend
                        .backendType() + " and " + t.backend().backendType());
            }
            float32 &= t.dtype() == DType.FLOAT32;
        }
        return float32 ? TrainGpu.forBackend(backend) : null;
    }

    /**
     * The common element count of the parameter and its gradient and state (null entries skipped).
     *
     * @throws IllegalArgumentException when the counts differ or exceed the 32-bit kernel limit
     */
    private static int elementCount(Tensor p, Tensor... others) {
        long n = p.numElements();
        for (Tensor t : others) {
            if (t != null && t.numElements() != n) {
                throw new IllegalArgumentException("tensor " + t.shape() + " has " + t.numElements()
                        + " elements, the parameter " + p.shape() + " has " + n);
            }
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("parameter of " + n + " elements exceeds the 32-bit kernel limit");
        }
        return (int) n;
    }

    /** A writable stand-in for a state buffer the kernel does not access; never read or written. */
    private static GpuBuffer unusedState(TrainGpu gpu) {
        return gpu.scratch(SLOT_UNUSED_STATE, Float.BYTES);
    }

    /**
     * {@code out[outOff] = sum of x * x} over the {@code n} elements at {@code in[inOff]}, as one row of TR_ReduceRows
     * SUMSQ, the variant chosen by the shape alone: the sequential kernel up to {@value #SMALL_COLS} elements, one
     * group up to {@value #PARTIAL_MIN_COLS}, otherwise chunk partials summed by a second pass; an empty run stores
     * the sum's identity 0 without binding {@code in}. {@code in} and {@code out} are different buffers.
     */
    private static void sumOfSquares(TrainGpu gpu, GpuBuffer in, int inOff, int n, GpuBuffer out, int outOff) {
        if (n == 0) {
            gpu.fill(out, outOff, 1, 0.0f);
            return;
        }
        GpuBuffer mean = gpu.placeholder();
        if (n <= SMALL_COLS) {
            gpu.launch1d(Kernels.REDUCE_ROWS_SMALL_F32, 1, TrainOps.TR_R_SUMSQ, 1, n, inOff, 0, outOff, 1.0f,
                    in, mean, out);
            return;
        }
        if (n <= PARTIAL_MIN_COLS) {
            gpu.launchRows(Kernels.REDUCE_ROWS_F32, 1, TrainOps.TR_R_SUMSQ, 1, n, inOff, 0, outOff, 1.0f,
                    in, mean, out);
            return;
        }
        int chunks = Math.min(MAX_CHUNKS, (n + COLS_PER_CHUNK - 1) / COLS_PER_CHUNK);
        GpuBuffer partial = gpu.scratch(SLOT_PARTIAL, (long) chunks * Float.BYTES);
        gpu.launchRows(Kernels.REDUCE_ROWS_PARTIAL_F32, chunks, TrainOps.TR_R_SUMSQ, 1, n, chunks, inOff, 0,
                in, mean, partial);
        if (chunks <= SMALL_COLS) {
            gpu.launch1d(Kernels.REDUCE_ROWS_SMALL_F32, 1, TrainOps.TR_R_SUM, 1, chunks, 0, 0, outOff, 1.0f,
                    partial, mean, out);
        } else {
            gpu.launchRows(Kernels.REDUCE_ROWS_F32, 1, TrainOps.TR_R_SUM, 1, chunks, 0, 0, outOff, 1.0f,
                    partial, mean, out);
        }
    }

    private static void closeEvent(GpuEvent event) {
        if (event != null) {
            event.close();
        }
    }

    /** A tensor bound to kernel arguments: a buffer and element offset, of the view itself or of a staged copy. */
    private static final class Operand {
        final Tensor view;
        final GpuBuffer buffer;
        final int offset;
        final Tensor staged;

        Operand(Tensor view, GpuBuffer buffer, int offset, Tensor staged) {
            this.view = view;
            this.buffer = buffer;
            this.offset = offset;
            this.staged = staged;
        }
    }

    /** The staged copies of one sequence; runs under the backend lock and closes every copy it made. */
    private static final class Staging implements AutoCloseable {
        private final List<Tensor> copies = new ArrayList<>();

        /**
         * Binds the written tensors of a step (null entries give null operands).
         *
         * @throws IllegalArgumentException when one addresses an element twice or two share an element
         */
        Operand[] bindWritten(Tensor... written) {
            Region[] regions = new Region[written.length];
            for (int i = 0; i < written.length; i++) {
                if (written[i] == null) {
                    continue;
                }
                regions[i] = Region.of(written[i]);
                regions[i].requireDistinctElements();
                for (int j = 0; j < i; j++) {
                    if (regions[j] != null && regions[i].overlaps(regions[j])) {
                        throw new IllegalArgumentException("tensors updated by one optimizer step share elements: "
                                + written[j].shape() + " and " + written[i].shape());
                    }
                }
            }
            Operand[] out = new Operand[written.length];
            for (int i = 0; i < written.length; i++) {
                out[i] = written[i] == null ? null : bind(written[i], regions[i], false);
            }
            return out;
        }

        /** Binds a gradient read by a step; staged when it is not dense or shares elements with a written tensor. */
        Operand bindGradient(Tensor g, Operand[] written) {
            Region r = Region.of(g);
            boolean stage = false;
            for (Operand w : written) {
                stage |= w != null && r.overlaps(Region.of(w.view));
            }
            return bind(g, r, stage);
        }

        /** Binds each distinct view of a gradient set, in region order. */
        Operand[] bindRegions(GradientSet set) {
            Operand[] out = new Operand[set.regions.size()];
            for (int r = 0; r < out.length; r++) {
                out[r] = bind(set.views.get(r), set.regions.get(r), false);
            }
            return out;
        }

        /**
         * Copies the staged copies of written operands back into their views (null entries skipped). Runs under
         * {@link ComputationGraph#noGrad()}: the steps applied the in-place guard to these views before anything was
         * written, and the gradient-list methods are unguarded, so a staged write-back behaves as a direct write.
         */
        void writeBack(Operand[] written) {
            try (ComputationGraph.NoGradScope untracked = ComputationGraph.noGrad()) {
                for (Operand o : written) {
                    if (o != null && o.staged != null) {
                        GpuCoreOps.copyInto(o.staged, o.view);
                    }
                }
            }
        }

        private Operand bind(Tensor t, Region r, boolean forceCopy) {
            if (r.dense && !forceCopy) {
                return new Operand(t, r.buffer, r.offset, null);
            }
            Tensor copy = GpuCoreOps.contiguous(t);
            if (copy == null) {
                throw new IllegalStateException("contiguous device copy unavailable for a FLOAT32 tensor on "
                        + t.backend().backendType());
            }
            copies.add(copy);
            StridedView v = StridedView.of(copy);
            return new Operand(t, v.buffer(), v.offset(), copy);
        }

        @Override
        public void close() {
            RuntimeException failure = null;
            for (Tensor c : copies) {
                try {
                    c.close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            copies.clear();
            if (failure != null) {
                throw failure;
            }
        }
    }

    /**
     * A gradient list and, after {@link #resolveViews}, its distinct views: {@code regions.get(regionOf[i])} is the
     * view of {@code grads.get(i)}, and {@code views.get(r)} the first listed tensor of region {@code r}.
     */
    private static final class GradientSet {
        final List<Tensor> grads;
        final List<Region> regions = new ArrayList<>();
        final List<Tensor> views = new ArrayList<>();
        final int[] regionOf;
        private final boolean float32;

        private GradientSet(List<Tensor> grads, boolean float32) {
            this.grads = grads;
            this.regionOf = new int[grads.size()];
            this.float32 = float32;
        }

        /**
         * @throws NullPointerException     when the list or an entry is null
         * @throws IllegalArgumentException when the entries live on different backends
         */
        static GradientSet of(List<Tensor> grads) {
            Objects.requireNonNull(grads, "gradient list must not be null");
            ComputeBackend backend = null;
            boolean float32 = true;
            for (int i = 0; i < grads.size(); i++) {
                Tensor t = Objects.requireNonNull(grads.get(i), "gradient " + i + " must not be null");
                if (backend == null) {
                    backend = t.backend();
                } else if (t.backend() != backend) {
                    throw new IllegalArgumentException("gradients live on different backends: "
                            + backend.backendType() + " and " + t.backend().backendType());
                }
                float32 &= t.dtype() == DType.FLOAT32;
            }
            return new GradientSet(grads, float32);
        }

        boolean isEmpty() {
            return grads.isEmpty();
        }

        /** The train context, or null when it is unavailable or a gradient is not FLOAT32. */
        TrainGpu context() {
            return float32 ? TrainGpu.forBackend(grads.get(0).backend()) : null;
        }

        /**
         * Groups the entries by view: equal views (same buffer, offset, coalesced dims and strides) share a region.
         *
         * @throws IllegalArgumentException when two entries share elements without being the same view, or an entry
         *                                  addresses an element twice
         */
        void resolveViews() {
            for (int i = 0; i < grads.size(); i++) {
                regionOf[i] = add(grads.get(i), i);
            }
        }

        private int add(Tensor t, int index) {
            Region region = Region.of(t);
            region.requireDistinctElements();
            for (int r = 0; r < regions.size(); r++) {
                Region known = regions.get(r);
                if (known.sameView(region)) {
                    return r;
                }
                if (known.overlaps(region)) {
                    throw new IllegalArgumentException("gradient " + index + " " + t.shape() + " shares elements with "
                            + "an earlier gradient " + views.get(r).shape() + " without being the same view; give it "
                            + "its own copy first");
                }
            }
            regions.add(region);
            views.add(t);
            return regions.size() - 1;
        }
    }

    /** The elements one view addresses in its buffer, with adjacent dims coalesced; identity is (buffer, view). */
    private static final class Region {
        final GpuBuffer buffer;
        final int offset;
        final int[] dims;
        final int[] strides;
        final int elements;
        final boolean dense;
        /** Highest element address (equal to {@code offset} for an empty view). */
        final long last;

        private Region(StridedView v) {
            this.buffer = v.buffer();
            this.offset = v.offset();
            this.dims = new int[v.rank()];
            this.strides = new int[v.rank()];
            long top = v.offset();
            for (int k = 0; k < dims.length; k++) {
                dims[k] = v.dim(k);
                strides[k] = v.stride(k);
                if (dims[k] > 1) {
                    top += (long) (dims[k] - 1) * strides[k];
                }
            }
            this.elements = v.numElements();
            this.dense = v.contiguous();
            this.last = top;
        }

        static Region of(Tensor t) {
            return new Region(StridedView.coalesce(StridedView.of(t))[0]);
        }

        boolean sameView(Region o) {
            return buffer == o.buffer && offset == o.offset && Arrays.equals(dims, o.dims)
                    && Arrays.equals(strides, o.strides);
        }

        void requireDistinctElements() {
            for (int k = 0; k < dims.length; k++) {
                if (dims[k] > 1 && strides[k] == 0) {
                    throw new IllegalArgumentException("a view with dims " + Arrays.toString(dims) + " and strides "
                            + Arrays.toString(strides) + " addresses an element more than once and cannot be updated");
                }
            }
        }

        /** Whether the two views address a common element (exact, also for interleaved strided views). */
        boolean overlaps(Region o) {
            if (buffer != o.buffer || elements == 0 || o.elements == 0) {
                return false;
            }
            long lo = Math.max(offset, o.offset);
            long hi = Math.min(last, o.last);
            if (lo > hi) {
                return false;
            }
            if (dense && o.dense) {
                return true;
            }
            BitSet marks = new BitSet((int) (hi - lo + 1));
            forEachAddress(a -> {
                if (a >= lo && a <= hi) {
                    marks.set((int) (a - lo));
                }
                return true;
            });
            boolean[] hit = {false};
            o.forEachAddress(a -> {
                if (a >= lo && a <= hi && marks.get((int) (a - lo))) {
                    hit[0] = true;
                    return false;
                }
                return true;
            });
            return hit[0];
        }

        /** Visits the element addresses in row-major order until {@code visitor} returns false. */
        private void forEachAddress(IntPredicate visitor) {
            int[] idx = new int[dims.length];
            long address = offset;
            for (int c = 0; c < elements; c++) {
                if (!visitor.test((int) address)) {
                    return;
                }
                for (int k = dims.length - 1; k >= 0; k--) {
                    if (++idx[k] < dims[k]) {
                        address += strides[k];
                        break;
                    }
                    address -= (long) (dims[k] - 1) * strides[k];
                    idx[k] = 0;
                }
            }
        }
    }
}

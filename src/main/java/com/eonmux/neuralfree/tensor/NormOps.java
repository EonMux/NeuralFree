package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.NormGradFunctions;

import java.util.Arrays;
import java.util.Objects;

/**
 * Normalization ops for the nn/norm modules: layer, RMS, batch and group normalization. Each method validates its
 * arguments, then offers the work to {@link GpuNormOps} when every operand lives on the input's backend, and runs
 * the CPU reference ({@code CpuNorm}, the modules' host loops) when the dispatcher declines (the train library is
 * unavailable on that backend, or an operand is not FLOAT32) or an operand lives on another backend (a cross-backend
 * transfer). Results are new contiguous FLOAT32 tensors on the input's backend; they require grad
 * exactly when the call was recorded (see Autograd below).
 *
 * <p>Both paths have one contract: the same shapes, argument checks and running-statistic updates, and results
 * within the reduction tolerance class of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract} ({@link
 * GpuNormOps} documents the device numerics). The CPU path reads every operand to the host, so its operands must be
 * FLOAT32.</p>
 *
 * <p><b>Autograd.</b> Each normalization records one node when grad mode is on and the input or an affine parameter
 * requires grad ({@link Autograd#needsGrad}); its gradient function ({@link NormGradFunctions}) calls the backward
 * primitive of the same normalization, and inside a create-graph pass computes the same formulas as a recorded
 * composition of {@link TensorOps} calls instead, so second-order gradients flow through every normalization. The
 * backward primitives ({@link #layerNormBackward}, {@link #rmsNormBackward}, {@link #batchNormBackward},
 * {@link #groupNormBackward}) are public, never record (also inside a create-graph pass), and run on
 * {@link GpuNormGradOps} with {@code CpuNorm} as their CPU path, under the same routing rule as the forwards (the
 * operands they read must live on the input's backend for the device path). They recompute the forward's statistics
 * with the forward's own launches (CPU loops), so the normalized values they use are the forward's whenever the
 * forward and its backward take the same path, and return {@link Gradients}.</p>
 */
public final class NormOps {

    private NormOps() {
    }

    /**
     * Layer normalization over the trailing {@code normalizedSize} elements of each row ({@code nn/norm/LayerNorm}):
     * {@code mean}, biased variance, {@code sdev = sqrt(var + eps)}, {@code y = (x - mean) / sdev}, then
     * {@code y = gamma * y} when {@code gamma} is given and {@code y = y + beta} when {@code beta} is given.
     *
     * @param x              input whose element count is a multiple of {@code normalizedSize}
     * @param normalizedSize product of the normalized trailing dims, positive
     * @param gamma          scale of {@code normalizedSize} elements (row-major within the row), or null
     * @param beta           shift of {@code normalizedSize} elements, or null
     * @param eps            added to the variance before the square root
     * @return a new tensor of {@code x}'s shape
     * @throws IllegalArgumentException for a non-positive size, a size that does not divide the input, a parameter of
     *                                  the wrong length, or an input above {@code 2^31-1} elements
     */
    public static Tensor layerNorm(Tensor x, int normalizedSize, Tensor gamma, Tensor beta, float eps) {
        Objects.requireNonNull(x, "input must not be null");
        if (normalizedSize <= 0) {
            throw new IllegalArgumentException("normalizedSize must be positive, got " + normalizedSize);
        }
        if (x.numElements() % normalizedSize != 0) {
            throw new IllegalArgumentException("input " + x.shape() + " does not split into rows of "
                    + normalizedSize + " elements");
        }
        requireKernelSize(x.shape());
        requireElements(gamma, normalizedSize, "gamma");
        requireElements(beta, normalizedSize, "beta");
        if (!Autograd.needsGrad(x, gamma, beta)) {
            return layerNormRaw(x, normalizedSize, gamma, beta, eps);
        }
        Tensor raw = Autograd.untracked(() -> layerNormRaw(x, normalizedSize, gamma, beta, eps));
        return Autograd.record(raw, Arrays.asList(x, gamma, beta), out -> {
            NormGradFunctions.LayerNormGrad f = new NormGradFunctions.LayerNormGrad(normalizedSize, eps,
                    x.requiresGrad(), requires(gamma), requires(beta) ? beta.shape() : null);
            f.saveForBackward(x, gamma);
            return f;
        });
    }

    private static Tensor layerNormRaw(Tensor x, int normalizedSize, Tensor gamma, Tensor beta, float eps) {
        if (onInputBackend(x, gamma, beta)) {
            Tensor gpu = GpuNormOps.layerNorm(x, normalizedSize, gamma, beta, eps);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.layerNorm(x, normalizedSize, gamma, beta, eps);
    }

    /**
     * RMS normalization over the last dim ({@code nn/norm/RMSNorm}): {@code rms = sqrt(sum(x^2) / n + eps)},
     * {@code y = x / rms}, then {@code y = y * weight} when {@code weight} is given.
     *
     * @param x      input of rank at least 1
     * @param weight scale with as many elements as the last dim, or null
     * @param eps    added to the mean square before the square root
     * @return a new tensor of {@code x}'s shape
     * @throws IllegalArgumentException for a rank-0 input, a weight of the wrong length, or an input above
     *                                  {@code 2^31-1} elements
     */
    public static Tensor rmsNorm(Tensor x, Tensor weight, float eps) {
        Objects.requireNonNull(x, "input must not be null");
        if (x.shape().rank() < 1) {
            throw new IllegalArgumentException("rmsNorm needs an input of rank at least 1, got " + x.shape());
        }
        requireKernelSize(x.shape());
        requireElements(weight, x.shape().dim(x.shape().rank() - 1), "weight");
        if (!Autograd.needsGrad(x, weight)) {
            return rmsNormRaw(x, weight, eps);
        }
        Tensor raw = Autograd.untracked(() -> rmsNormRaw(x, weight, eps));
        return Autograd.record(raw, Arrays.asList(x, weight), out -> {
            NormGradFunctions.RMSNormGrad f = new NormGradFunctions.RMSNormGrad(eps, x.requiresGrad(),
                    requires(weight));
            f.saveForBackward(x, weight);
            return f;
        });
    }

    private static Tensor rmsNormRaw(Tensor x, Tensor weight, float eps) {
        if (onInputBackend(x, weight)) {
            Tensor gpu = GpuNormOps.rmsNorm(x, weight, eps);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.rmsNorm(x, weight, eps);
    }

    /**
     * Batch normalization of an input {@code [B, C, *]} ({@code nn/norm/BatchNorm1d} on {@code [B, C]} and
     * {@code [B, C, L]}, {@code BatchNorm2d} on {@code [B, C, H, W]}). In training mode the per-channel batch mean and
     * biased variance over the {@code n = B * prod(*)} elements of each channel normalize the input, and the running
     * statistics, when given, are updated in place as {@code running_mean = (1 - momentum) * running_mean + momentum *
     * mean} and {@code running_var = (1 - momentum) * running_var + momentum * var * n / (n - 1)}: the running variance
     * blends in the unbiased batch variance ({@link CpuNorm#runningVarWeight}) as PyTorch does, on the device or on
     * the host as the modules do; the update runs inside
     * {@link Autograd#untracked} and is never part of a recorded computation. In evaluation mode the running
     * statistics normalize the input and nothing is updated. The affine step is {@code gamma * y + beta}. A recorded
     * evaluation-mode call saves contiguous copies of the running statistics, so a later in-place update of the
     * buffers does not change its backward; a training-mode backward recomputes the batch statistics from the input.
     *
     * @param x           input of rank at least 2, channels on dim 1
     * @param gamma       per-channel scale, or null; given exactly when {@code beta} is
     * @param beta        per-channel shift, or null
     * @param runningMean per-channel running mean: updated in training mode (or null there, given exactly when
     *                    {@code runningVar} is), required in evaluation mode
     * @param runningVar  per-channel running variance, as {@code runningMean}
     * @param training    whether to normalize with the batch statistics and update the running ones
     * @param momentum    weight of the batch statistic in the running update
     * @param eps         added to the variance before the square root
     * @return a new tensor of {@code x}'s shape
     * @throws IllegalArgumentException for an input of rank below 2 or above {@code 2^31-1} elements, unpaired or
     *                                  mis-sized per-channel tensors, a running statistic whose view addresses an
     *                                  element twice in training mode, or, in training mode, a single value per
     *                                  channel ({@code B * prod(*) == 1}, {@link CpuNorm#requireBatchStatistics})
     * @throws NullPointerException     for a missing running statistic in evaluation mode
     */
    public static Tensor batchNorm(Tensor x, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                                   boolean training, float momentum, float eps) {
        GpuNormOps.ChannelGeometry geo = GpuNormOps.ChannelGeometry.of(x, 0);
        requireAffine(gamma, beta, geo.channels);
        if (training) {
            CpuNorm.requireBatchStatistics(x.shape(), geo.batch, geo.spatial);
            requirePair(runningMean, runningVar, "runningMean", "runningVar");
        } else {
            Objects.requireNonNull(runningMean, "runningMean must not be null");
            Objects.requireNonNull(runningVar, "runningVar must not be null");
        }
        requireElements(runningMean, geo.channels, "runningMean");
        requireElements(runningVar, geo.channels, "runningVar");
        if (training) {
            requireWritable(runningMean, "runningMean");
            requireWritable(runningVar, "runningVar");
        }
        if (!Autograd.needsGrad(x, gamma, beta)) {
            return batchNormRaw(x, gamma, beta, runningMean, runningVar, training, momentum, eps);
        }
        Tensor raw = Autograd.untracked(
                () -> batchNormRaw(x, gamma, beta, runningMean, runningVar, training, momentum, eps));
        return Autograd.record(raw, Arrays.asList(x, gamma, beta), out -> {
            NormGradFunctions.BatchNormGrad f = new NormGradFunctions.BatchNormGrad(training, eps,
                    x.requiresGrad(), requires(gamma), requires(beta) ? beta.shape() : null);
            if (training) {
                f.saveForBackward(x, gamma, null, null);
            } else {
                try (Tensor mean = copyOf(runningMean); Tensor var = copyOf(runningVar)) {
                    f.saveForBackward(x, gamma, mean, var);
                }
            }
            return f;
        });
    }

    private static Tensor batchNormRaw(Tensor x, Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                                       boolean training, float momentum, float eps) {
        boolean device = onInputBackend(x, gamma, beta, runningMean, runningVar);
        if (!training) {
            Tensor gpu = device ? GpuNormOps.batchNormEval(x, gamma, beta, runningMean, runningVar, eps) : null;
            return gpu != null ? gpu
                    : CpuNorm.batchNorm(x, gamma, beta, runningMean, runningVar, false, momentum, eps);
        }
        return Autograd.untracked(() -> {
            Tensor gpu = device
                    ? GpuNormOps.batchNormTrain(x, gamma, beta, runningMean, runningVar, momentum, eps) : null;
            return gpu != null ? gpu
                    : CpuNorm.batchNorm(x, gamma, beta, runningMean, runningVar, true, momentum, eps);
        });
    }

    /**
     * Group normalization of an input {@code [B, C, *]} with {@code groups} groups of {@code C / groups} channels
     * ({@code nn/norm/GroupNorm}): the mean and biased variance of each (batch, group) over its
     * {@code (C / groups) * prod(*)} elements normalize them, followed by the per-channel affine step
     * {@code gamma * y + beta}.
     *
     * @param x      input of rank at least 2, channels on dim 1
     * @param groups number of groups, positive and dividing the channel count
     * @param gamma  per-channel scale, or null; given exactly when {@code beta} is
     * @param beta   per-channel shift, or null
     * @param eps    added to the variance before the square root
     * @return a new tensor of {@code x}'s shape
     * @throws IllegalArgumentException for an input of rank below 2 or above {@code 2^31-1} elements, a group count
     *                                  that is not positive or does not divide the channels, or unpaired or
     *                                  mis-sized per-channel tensors
     */
    public static Tensor groupNorm(Tensor x, int groups, Tensor gamma, Tensor beta, float eps) {
        if (groups <= 0) {
            throw new IllegalArgumentException("groups must be positive, got " + groups);
        }
        GpuNormOps.ChannelGeometry geo = GpuNormOps.ChannelGeometry.of(x, groups);
        requireAffine(gamma, beta, geo.channels);
        if (!Autograd.needsGrad(x, gamma, beta)) {
            return groupNormRaw(x, groups, gamma, beta, eps);
        }
        Tensor raw = Autograd.untracked(() -> groupNormRaw(x, groups, gamma, beta, eps));
        return Autograd.record(raw, Arrays.asList(x, gamma, beta), out -> {
            NormGradFunctions.GroupNormGrad f = new NormGradFunctions.GroupNormGrad(groups, eps,
                    x.requiresGrad(), requires(gamma), requires(beta) ? beta.shape() : null);
            f.saveForBackward(x, gamma);
            return f;
        });
    }

    private static Tensor groupNormRaw(Tensor x, int groups, Tensor gamma, Tensor beta, float eps) {
        if (onInputBackend(x, gamma, beta)) {
            Tensor gpu = GpuNormOps.groupNorm(x, groups, gamma, beta, eps);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.groupNorm(x, groups, gamma, beta, eps);
    }

    /** The default {@code eps} of {@link #l2Normalize(Tensor, int)}, PyTorch's {@code F.normalize} default. */
    public static final float L2_NORMALIZE_EPS = 1e-12f;

    /** {@link #l2Normalize(Tensor, int, float)} with {@link #L2_NORMALIZE_EPS}. */
    public static Tensor l2Normalize(Tensor x, int dim) {
        return l2Normalize(x, dim, L2_NORMALIZE_EPS);
    }

    /**
     * {@code torch.nn.functional.normalize(x, p=2, dim, eps)}: {@code x / max(||x||_2, eps)} with the norm taken along
     * {@code dim} (negative counts from the end) and broadcast back over it. Computed as
     * {@code x / sqrt(max(sum(x^2, dim), eps^2))}, a recorded {@link TensorOps} composition: the gradient flows
     * through the norm where it exceeds {@code eps}, and a row whose norm does not (an all-zero row included) gets
     * the gradient of {@code x / eps}, without a division by zero.
     *
     * @return a new tensor of {@code x}'s shape
     * @throws IllegalArgumentException  when {@code eps} is not positive or {@code eps^2} is not a positive finite
     *                                   float
     * @throws IndexOutOfBoundsException when {@code dim} is outside {@code [-rank, rank)}
     */
    public static Tensor l2Normalize(Tensor x, int dim, float eps) {
        Objects.requireNonNull(x, "x cannot be null");
        float floor = eps * eps;
        if (!(eps > 0f) || !(floor > 0f) || Float.isInfinite(floor)) {
            throw new IllegalArgumentException("eps must be positive with a positive finite float square, got " + eps);
        }
        int d = CpuShapeOps.normalizeDim(dim, x.shape().rank());
        try (Tensor squares = TensorOps.mul(x, x);
             Tensor sumSquares = TensorOps.sum(squares, d);
             Tensor floored = TensorOps.clamp(sumSquares, floor, Float.POSITIVE_INFINITY);
             Tensor norm = TensorOps.sqrt(floored);
             Tensor broadcastNorm = norm.unsqueeze(d)) {
            return TensorOps.div(x, broadcastNorm);
        }
    }

    // ------------------------------------------------------------------ backward primitives

    /**
     * Gradients of {@link #layerNorm} with respect to the input, gamma and beta, given the gradient {@code gy} of its
     * output. Per row: {@code xhat = (x - mean) / sdev} with the forward's statistics, {@code d = gy * gamma} (or
     * {@code gy} without gamma), {@code gx = ((d - mean(d)) - xhat * mean(d * xhat)) / sdev}; per column
     * {@code ggamma = sum_rows gy * xhat} and {@code gbeta = sum_rows gy}. Never recorded. Tolerance: reduction class,
     * K = the row length for {@code gx} and the row count for the affine gradients.
     *
     * @param gy             gradient of the forward's output, of {@code x}'s shape
     * @param x              the forward's input
     * @param normalizedSize the forward's row length, positive
     * @param gamma          the forward's gamma, or null
     * @param eps            the forward's epsilon
     * @param needInput      whether to compute {@code gx}
     * @param needGamma      whether to compute the gamma gradient (in gamma's shape); requires gamma
     * @param betaShape      the shape of the forward's beta when its gradient is wanted, else null (the beta gradient
     *                       does not depend on beta's values)
     * @return the requested gradients, owned by the caller; null entries for the gradients not requested
     * @throws IllegalArgumentException for a gradient whose shape differs from the input, the forward's argument
     *                                  errors, a gamma gradient without gamma, or a beta shape of the wrong size
     */
    public static Gradients layerNormBackward(Tensor gy, Tensor x, int normalizedSize, Tensor gamma, float eps,
                                              boolean needInput, boolean needGamma, Shape betaShape) {
        GpuNormGradOps.checkLayerNorm(gy, x, normalizedSize, gamma, needGamma, betaShape);
        if (!needInput && !needGamma && betaShape == null) {
            return new Gradients(null, null, null);
        }
        if (onInputBackend(x, gy, gamma)) {
            Gradients gpu = GpuNormGradOps.layerNormBackward(gy, x, normalizedSize, gamma, eps, needInput, needGamma,
                    betaShape);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.layerNormBackward(gy, x, normalizedSize, gamma, eps, needInput, needGamma, betaShape);
    }

    /**
     * Gradients of {@link #rmsNorm} with respect to the input and the weight. Per row: {@code xhat = x / rms} with
     * the forward's {@code rms}, {@code d = gy * weight} (or {@code gy}), {@code gx = (d - xhat * mean(d * xhat)) /
     * rms}; per column {@code gweight = sum_rows gy * xhat}. Never recorded. Tolerance: reduction class.
     *
     * @param needWeight whether to compute the weight gradient (in the weight's shape); requires the weight
     * @return the requested gradients ({@link Gradients#bias()} is always null), owned by the caller
     * @throws IllegalArgumentException for a gradient whose shape differs from the input, the forward's argument
     *                                  errors, or a weight gradient without a weight
     */
    public static Gradients rmsNormBackward(Tensor gy, Tensor x, Tensor weight, float eps, boolean needInput,
                                            boolean needWeight) {
        GpuNormGradOps.checkRmsNorm(gy, x, weight, needWeight);
        if (!needInput && !needWeight) {
            return new Gradients(null, null, null);
        }
        if (onInputBackend(x, gy, weight)) {
            Gradients gpu = GpuNormGradOps.rmsNormBackward(gy, x, weight, eps, needInput, needWeight);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.rmsNormBackward(gy, x, weight, eps, needInput, needWeight);
    }

    /**
     * Gradients of {@link #batchNorm} with respect to the input, gamma and beta. Training mode recomputes the batch
     * statistics from {@code x} (with the forward's statistics launch, never updating running statistics) and gives
     * {@code gx = gamma * ((gy - s1 / n) - xhat * (s2 / n)) / sdev} with {@code s1 = sum(gy)},
     * {@code s2 = sum(gy * xhat)} over the {@code n = B * prod(*)} elements of each channel. Evaluation mode uses
     * {@code runningMean} and {@code runningVar} as constants: {@code gx = gy * gamma / sdev}. In both modes
     * {@code gbeta = s1} and {@code ggamma = s2} per channel. Never recorded. Tolerance: reduction class, K = n.
     *
     * @param runningMean the statistics the evaluation-mode forward normalized with; read only when
     *                    {@code training} is false
     * @param runningVar  as {@code runningMean}
     * @return the requested gradients, owned by the caller
     * @throws IllegalArgumentException for a gradient whose shape differs from the input, the forward's argument
     *                                  errors, a gamma gradient without gamma, a beta shape of the wrong size, or more
     *                                  than {@code 2^30 - 1} channels
     * @throws NullPointerException     for a missing running statistic in evaluation mode
     */
    public static Gradients batchNormBackward(Tensor gy, Tensor x, Tensor gamma, Tensor runningMean,
                                              Tensor runningVar, boolean training, float eps, boolean needInput,
                                              boolean needGamma, Shape betaShape) {
        GpuNormGradOps.checkBatchNorm(gy, x, gamma, runningMean, runningVar, training, needGamma, betaShape);
        if (!needInput && !needGamma && betaShape == null) {
            return new Gradients(null, null, null);
        }
        boolean device = training ? onInputBackend(x, gy, gamma) : onInputBackend(x, gy, gamma, runningMean,
                runningVar);
        if (device) {
            Gradients gpu = GpuNormGradOps.batchNormBackward(gy, x, gamma, runningMean, runningVar, training, eps,
                    needInput, needGamma, betaShape);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.batchNormBackward(gy, x, gamma, runningMean, runningVar, training, eps, needInput, needGamma,
                betaShape);
    }

    /**
     * Gradients of {@link #groupNorm} with respect to the input, gamma and beta. Per (batch, group) with the
     * forward's statistics: {@code d = gy * gamma[c]} (or {@code gy}), {@code gx = ((d - s1 / n) - xhat * (s2 / n)) /
     * sdev} with {@code s1 = sum(d)}, {@code s2 = sum(d * xhat)} over the group's {@code n} elements; per channel
     * {@code ggamma = sum gy * xhat} and {@code gbeta = sum gy} over the batch. Never recorded. Tolerance: reduction
     * class, K = the longer of the group size and the per-channel batch.
     *
     * @return the requested gradients, owned by the caller
     * @throws IllegalArgumentException for a gradient whose shape differs from the input, the forward's argument
     *                                  errors, a gamma gradient without gamma, a beta shape of the wrong size, or more
     *                                  than {@code 2^30 - 1} channels or (batch, group) rows
     */
    public static Gradients groupNormBackward(Tensor gy, Tensor x, int groups, Tensor gamma, float eps,
                                              boolean needInput, boolean needGamma, Shape betaShape) {
        GpuNormGradOps.checkGroupNorm(gy, x, groups, gamma, needGamma, betaShape);
        if (!needInput && !needGamma && betaShape == null) {
            return new Gradients(null, null, null);
        }
        if (onInputBackend(x, gy, gamma)) {
            Gradients gpu = GpuNormGradOps.groupNormBackward(gy, x, groups, gamma, eps, needInput, needGamma,
                    betaShape);
            if (gpu != null) {
                return gpu;
            }
        }
        return CpuNorm.groupNormBackward(gy, x, groups, gamma, eps, needInput, needGamma, betaShape);
    }

    /**
     * The result of a normalization backward primitive: the gradients with respect to the input, the scale (gamma or
     * the RMS weight) and the shift (beta), each a new contiguous FLOAT32 tensor or null when it was not requested.
     * The holder owns them until they are handed on: {@link #close()} closes every gradient it holds, and a caller that
     * hands the gradients to another owner does not close it.
     */
    public static final class Gradients implements AutoCloseable {
        private final Tensor input;
        private final Tensor weight;
        private final Tensor bias;

        Gradients(Tensor input, Tensor weight, Tensor bias) {
            this.input = input;
            this.weight = weight;
            this.bias = bias;
        }

        /** Gradient with respect to the input, in its shape, or null. */
        public Tensor input() {
            return input;
        }

        /** Gradient with respect to the scale (gamma, RMS weight), in its shape, or null. */
        public Tensor weight() {
            return weight;
        }

        /** Gradient with respect to the shift (beta), in the requested shape, or null. */
        public Tensor bias() {
            return bias;
        }

        /** Closes every gradient held; closing again has no further effect. */
        @Override
        public void close() {
            RuntimeException failure = null;
            for (Tensor t : new Tensor[]{input, weight, bias}) {
                if (t == null) {
                    continue;
                }
                try {
                    t.close();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private static boolean requires(Tensor t) {
        return t != null && t.requiresGrad();
    }

    /**
     * A new contiguous copy of a statistic that owns its storage: on the device when the train library is available,
     * else through the host. It never requires grad.
     */
    private static Tensor copyOf(Tensor t) {
        Tensor device = GpuCoreOps.contiguous(t);
        return device != null ? device : CpuShapeOps.floats(t.backend(), t.shape(), t.toFloatArray(), false);
    }

    // ------------------------------------------------------------------ validation

    /** Whether every non-null operand lives on the input's backend, which the dispatcher requires. */
    private static boolean onInputBackend(Tensor x, Tensor... operands) {
        for (Tensor t : operands) {
            if (t != null && t.backend() != x.backend()) {
                return false;
            }
        }
        return true;
    }

    private static void requireElements(Tensor p, long elements, String name) {
        if (p != null && p.numElements() != elements) {
            throw new IllegalArgumentException(name + " must have " + elements + " elements, got shape " + p.shape());
        }
    }

    /** The train kernel ABI and the host arrays address at most {@code 2^31-1} elements. */
    private static void requireKernelSize(Shape shape) {
        if (shape.numElements() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("input " + shape + " exceeds the 2^31-1 elements of the train kernel "
                    + "ABI");
        }
    }

    private static void requirePair(Tensor a, Tensor b, String aName, String bName) {
        if ((a == null) != (b == null)) {
            throw new IllegalArgumentException(aName + " and " + bName + " must both be given or both be null");
        }
    }

    private static void requireAffine(Tensor gamma, Tensor beta, int channels) {
        requirePair(gamma, beta, "gamma", "beta");
        requireElements(gamma, channels, "gamma");
        requireElements(beta, channels, "beta");
    }

    /** An in-place target must address each of its elements once (no stride-0 dim of size above 1). */
    private static void requireWritable(Tensor t, String name) {
        if (t == null) {
            return;
        }
        long[] strides = t.strides().strides();
        for (int k = 0; k < strides.length; k++) {
            if (strides[k] == 0 && t.shape().dim(k) > 1) {
                throw new IllegalArgumentException(name + " view " + t.shape() + " with strides " + t.strides()
                        + " addresses an element more than once and cannot be updated in place");
            }
        }
    }
}

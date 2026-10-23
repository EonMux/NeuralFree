package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Backward functions of the {@link NormOps} normalizations: layer, RMS, batch and group normalization. Each one is
 * recorded over the inputs {@code (x, gamma, beta)} ({@code (x, weight)} for RMSNorm, with null entries for absent
 * parameters) and calls the matching {@link NormOps} backward primitive, which runs on the device when the train
 * library is available on the input's backend and on the CPU reference path otherwise; nothing is read to the host
 * on the device path. The primitive recomputes the forward's statistics from the saved input with the forward's own
 * launches, so the normalized values are the forward's.
 *
 * <p><b>Ownership</b> follows {@link GradientFunction}: {@code gradOutput} is never closed, and every returned
 * gradient is a new tensor owned by the caller. The need flags captured at construction select which gradients are
 * computed; the others are null. Saved tensors are detached handles released by {@code clearSavedTensors()}.</p>
 *
 * <p><b>Create graph.</b> Every class supports it ({@code supportsCreateGraph()} is true). The fused backward
 * primitives never record, so inside a create-graph pass ({@link Autograd#isCreatingGraph()}), when the incoming
 * gradient or a saved tensor requires grad, a backward computes the same formulas as a composition of recording
 * {@link TensorOps} calls and views instead: the statistics are recomputed from the saved input
 * (mean, biased variance, {@code sdev = sqrt(var + eps)}), so the gradients it returns are differentiable in the
 * incoming gradient, the input and the scale. The saved tensors are read through {@code saved(i)}, which gives them
 * their path to the forward tensors. The composed gradients agree with the fused primitive within the reduction
 * tolerance class (K = the longest reduction); an empty input takes the fused path, whose zero gradients are
 * constants.</p>
 */
public final class NormGradFunctions {

    private NormGradFunctions() {
    }

    /**
     * {@code NormOps.layerNorm(x, normalizedSize, gamma, beta, eps)}: per row {@code xhat = (x - mean) / sdev},
     * {@code d = gy * gamma} (or {@code gy}), {@code gx = ((d - mean(d)) - xhat * mean(d * xhat)) / sdev}; per column
     * {@code ggamma = sum_rows gy * xhat}, {@code gbeta = sum_rows gy} ({@link NormOps#layerNormBackward}). Saved: the
     * input and gamma (when given). Tolerance: reduction class, K = the row length for {@code gx}, the row count for
     * the affine gradients.
     */
    public static class LayerNormGrad extends GradFunctions.SavingGradFn {
        private final int normalizedSize;
        private final float eps;
        private final boolean needInput;
        private final boolean needGamma;
        private final Shape betaShape;
        private boolean hasGamma;

        /**
         * @param needInput whether the input requires a gradient
         * @param needGamma whether gamma requires a gradient
         * @param betaShape beta's shape when beta requires a gradient, else null
         */
        public LayerNormGrad(int normalizedSize, float eps, boolean needInput, boolean needGamma, Shape betaShape) {
            if (normalizedSize <= 0) {
                throw new IllegalArgumentException("normalizedSize must be positive, got " + normalizedSize);
            }
            this.normalizedSize = normalizedSize;
            this.eps = eps;
            this.needInput = needInput;
            this.needGamma = needGamma;
            this.betaShape = betaShape;
        }

        /** Saves the forward's input and gamma ({@code null} when the forward had none). */
        public void saveForBackward(Tensor x, Tensor gamma) {
            requireScaleForGradient(gamma, needGamma, "gamma");
            hasGamma = gamma != null;
            saveAll(this, x, gamma);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            Tensor gamma = hasGamma ? saved(1) : null;
            if (composes(gradOutput, x, gamma)) {
                return layerNormComposed(gradOutput, x, normalizedSize, gamma, eps, needInput, needGamma, betaShape);
            }
            NormOps.Gradients g = NormOps.layerNormBackward(gradOutput, x, normalizedSize, gamma, eps, needInput,
                    needGamma, betaShape);
            return Arrays.asList(g.input(), g.weight(), g.bias());
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code NormOps.rmsNorm(x, weight, eps)}: per row {@code xhat = x / rms}, {@code d = gy * weight} (or
     * {@code gy}), {@code gx = (d - xhat * mean(d * xhat)) / rms}; per column {@code gweight = sum_rows gy * xhat}
     * ({@link NormOps#rmsNormBackward}). Saved: the input and the weight (when given). Tolerance: reduction class,
     * K = the row length for {@code gx}, the row count for the weight gradient.
     */
    public static class RMSNormGrad extends GradFunctions.SavingGradFn {
        private final float eps;
        private final boolean needInput;
        private final boolean needWeight;
        private boolean hasWeight;

        /**
         * @param needInput  whether the input requires a gradient
         * @param needWeight whether the weight requires a gradient
         */
        public RMSNormGrad(float eps, boolean needInput, boolean needWeight) {
            this.eps = eps;
            this.needInput = needInput;
            this.needWeight = needWeight;
        }

        /** Saves the forward's input and weight ({@code null} when the forward had none). */
        public void saveForBackward(Tensor x, Tensor weight) {
            requireScaleForGradient(weight, needWeight, "weight");
            hasWeight = weight != null;
            saveAll(this, x, weight);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            Tensor weight = hasWeight ? saved(1) : null;
            if (composes(gradOutput, x, weight)) {
                return rmsNormComposed(gradOutput, x, weight, eps, needInput, needWeight);
            }
            NormOps.Gradients g = NormOps.rmsNormBackward(gradOutput, x, weight, eps, needInput, needWeight);
            return Arrays.asList(g.input(), g.weight());
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code NormOps.batchNorm(x, gamma, beta, runningMean, runningVar, training, momentum, eps)}
     * ({@link NormOps#batchNormBackward}). Training mode: the batch statistics are recomputed from the input (never
     * updating the running statistics), {@code gx = gamma * ((gy - s1 / n) - xhat * (s2 / n)) / sdev} with
     * {@code s1 = sum(gy)}, {@code s2 = sum(gy * xhat)} per channel over its {@code n = B * prod(*)} elements.
     * Evaluation mode: the running statistics are constants, {@code gx = gy * gamma / sdev}. Both modes:
     * {@code gbeta = s1}, {@code ggamma = s2}. Saved: the input, gamma (when given) and, in evaluation mode,
     * contiguous copies of the running mean and variance taken at the forward (rule G1: the module updates its
     * buffers in place). Tolerance: reduction class, K = n.
     */
    public static class BatchNormGrad extends GradFunctions.SavingGradFn {
        private final boolean training;
        private final float eps;
        private final boolean needInput;
        private final boolean needGamma;
        private final Shape betaShape;
        private boolean hasGamma;

        /**
         * @param training  whether the forward normalized with the batch statistics
         * @param needInput whether the input requires a gradient
         * @param needGamma whether gamma requires a gradient
         * @param betaShape beta's shape when beta requires a gradient, else null
         */
        public BatchNormGrad(boolean training, float eps, boolean needInput, boolean needGamma, Shape betaShape) {
            this.training = training;
            this.eps = eps;
            this.needInput = needInput;
            this.needGamma = needGamma;
            this.betaShape = betaShape;
        }

        /**
         * Saves the forward's input, gamma ({@code null} when the forward had none) and, in evaluation mode, copies of
         * the running statistics that no one writes afterwards ({@code null} in training mode).
         *
         * @throws IllegalArgumentException when the statistics are given in training mode or missing in evaluation
         *                                  mode
         */
        public void saveForBackward(Tensor x, Tensor gamma, Tensor meanCopy, Tensor varCopy) {
            requireScaleForGradient(gamma, needGamma, "gamma");
            boolean statistics = meanCopy != null && varCopy != null;
            if (training ? meanCopy != null || varCopy != null : !statistics) {
                throw new IllegalArgumentException(training
                        ? "a training-mode BatchNorm backward recomputes its statistics; save none"
                        : "an evaluation-mode BatchNorm backward needs copies of both running statistics");
            }
            hasGamma = gamma != null;
            if (training) {
                saveAll(this, x, gamma);
            } else {
                saveAll(this, x, gamma, meanCopy, varCopy);
            }
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            Tensor gamma = hasGamma ? saved(1) : null;
            int next = hasGamma ? 2 : 1;
            Tensor mean = training ? null : saved(next);
            Tensor var = training ? null : saved(next + 1);
            if (composes(gradOutput, x, gamma)) {
                return batchNormComposed(gradOutput, x, gamma, mean, var, training, eps, needInput, needGamma,
                        betaShape);
            }
            NormOps.Gradients g = NormOps.batchNormBackward(gradOutput, x, gamma, mean, var, training, eps,
                    needInput, needGamma, betaShape);
            return Arrays.asList(g.input(), g.weight(), g.bias());
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    /**
     * {@code NormOps.groupNorm(x, groups, gamma, beta, eps)}: per (batch, group) with the forward's statistics,
     * {@code d = gy * gamma[c]} (or {@code gy}), {@code gx = ((d - s1 / n) - xhat * (s2 / n)) / sdev} with
     * {@code s1 = sum(d)}, {@code s2 = sum(d * xhat)} over the group's {@code n} elements; per channel
     * {@code ggamma = sum gy * xhat}, {@code gbeta = sum gy} over the batch ({@link NormOps#groupNormBackward}).
     * Saved: the input and gamma (when given). Tolerance: reduction class, K = the longer of the group size and the
     * per-channel batch.
     */
    public static class GroupNormGrad extends GradFunctions.SavingGradFn {
        private final int groups;
        private final float eps;
        private final boolean needInput;
        private final boolean needGamma;
        private final Shape betaShape;
        private boolean hasGamma;

        /**
         * @param needInput whether the input requires a gradient
         * @param needGamma whether gamma requires a gradient
         * @param betaShape beta's shape when beta requires a gradient, else null
         */
        public GroupNormGrad(int groups, float eps, boolean needInput, boolean needGamma, Shape betaShape) {
            if (groups <= 0) {
                throw new IllegalArgumentException("groups must be positive, got " + groups);
            }
            this.groups = groups;
            this.eps = eps;
            this.needInput = needInput;
            this.needGamma = needGamma;
            this.betaShape = betaShape;
        }

        /** Saves the forward's input and gamma ({@code null} when the forward had none). */
        public void saveForBackward(Tensor x, Tensor gamma) {
            requireScaleForGradient(gamma, needGamma, "gamma");
            hasGamma = gamma != null;
            saveAll(this, x, gamma);
        }

        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor x = saved(0);
            Tensor gamma = hasGamma ? saved(1) : null;
            if (composes(gradOutput, x, gamma)) {
                return groupNormComposed(gradOutput, x, groups, gamma, eps, needInput, needGamma, betaShape);
            }
            NormOps.Gradients g = NormOps.groupNormBackward(gradOutput, x, groups, gamma, eps, needInput,
                    needGamma, betaShape);
            return Arrays.asList(g.input(), g.weight(), g.bias());
        }

        @Override
        public boolean supportsCreateGraph() {
            return true;
        }
    }

    // ------------------------------------------------------------------ create-graph compositions

    /**
     * Whether a backward takes its recorded composition: inside a create-graph pass, for a non-empty input, when the
     * incoming gradient, the input or the scale requires grad (with grad mode on, which a create-graph pass sets).
     */
    private static boolean composes(Tensor gradOutput, Tensor x, Tensor scale) {
        return Autograd.isCreatingGraph() && x.numElements() > 0 && Autograd.needsGrad(gradOutput, x, scale);
    }

    /** {@link LayerNormGrad}'s formulas on the rows {@code [R, N]} of the input, recorded. */
    private static List<Tensor> layerNormComposed(Tensor gy, Tensor x, int n, Tensor gamma, float eps,
                                                  boolean needInput, boolean needGamma, Shape betaShape) {
        requireGradientShape(gy, x);
        int rows = Math.toIntExact(x.numElements() / n);
        Shape rowShape = Shape.of(rows, n);
        Shape statShape = Shape.of(rows, 1);
        try (Composition c = new Composition()) {
            Tensor xr = c.temp(x.reshape(rowShape));
            Tensor gyr = c.temp(gy.reshape(rowShape));
            Tensor xc = c.temp(TensorOps.sub(xr, c.meanKept(xr, statShape, 1)));
            Tensor sdev = c.sdev(c.meanKept(c.temp(TensorOps.mul(xc, xc)), statShape, 1), eps);
            Tensor xhat = c.temp(TensorOps.div(xc, sdev));
            Tensor gGamma = needGamma ? c.result(c.sumTo(c.temp(TensorOps.mul(gyr, xhat)), gamma.shape(), 0))
                    : null;
            Tensor gBeta = betaShape != null ? c.result(c.sumTo(gyr, betaShape, 0)) : null;
            Tensor gx = null;
            if (needInput) {
                Tensor d = gamma == null ? gyr : c.temp(TensorOps.mul(gyr, c.temp(gamma.reshape(Shape.of(n)))));
                gx = c.result(c.temp(TensorOps.div(c.centered(d, xhat, statShape, 1), sdev)).reshape(x.shape()));
            }
            return c.finish(gx, gGamma, gBeta);
        }
    }

    /** {@link RMSNormGrad}'s formulas on the rows {@code [R, N]} of the input (N = the last dim), recorded. */
    private static List<Tensor> rmsNormComposed(Tensor gy, Tensor x, Tensor weight, float eps, boolean needInput,
                                                boolean needWeight) {
        requireGradientShape(gy, x);
        int n = x.shape().dim(x.shape().rank() - 1);
        int rows = Math.toIntExact(x.numElements() / n);
        Shape rowShape = Shape.of(rows, n);
        Shape statShape = Shape.of(rows, 1);
        try (Composition c = new Composition()) {
            Tensor xr = c.temp(x.reshape(rowShape));
            Tensor gyr = c.temp(gy.reshape(rowShape));
            Tensor rms = c.sdev(c.meanKept(c.temp(TensorOps.mul(xr, xr)), statShape, 1), eps);
            Tensor xhat = c.temp(TensorOps.div(xr, rms));
            Tensor gWeight = needWeight ? c.result(c.sumTo(c.temp(TensorOps.mul(gyr, xhat)), weight.shape(), 0))
                    : null;
            Tensor gx = null;
            if (needInput) {
                Tensor d = weight == null ? gyr : c.temp(TensorOps.mul(gyr, c.temp(weight.reshape(Shape.of(n)))));
                Tensor projection = c.temp(TensorOps.mul(xhat,
                        c.meanKept(c.temp(TensorOps.mul(d, xhat)), statShape, 1)));
                Tensor numerator = c.temp(TensorOps.sub(d, projection));
                gx = c.result(c.temp(TensorOps.div(numerator, rms)).reshape(x.shape()));
            }
            return c.finish(gx, gWeight);
        }
    }

    /**
     * {@link BatchNormGrad}'s formulas on the input viewed as {@code [B, C, S]} ({@code S = prod(*)}), recorded. In
     * training mode the batch statistics are recomputed from the input over dims 0 and 2; in evaluation mode the saved
     * running-statistic copies are constants.
     */
    private static List<Tensor> batchNormComposed(Tensor gy, Tensor x, Tensor gamma, Tensor runningMean,
                                                  Tensor runningVar, boolean training, float eps, boolean needInput,
                                                  boolean needGamma, Shape betaShape) {
        requireGradientShape(gy, x);
        int batch = x.shape().dim(0);
        int channels = x.shape().dim(1);
        int spatial = Math.toIntExact(x.numElements() / ((long) batch * channels));
        Shape cube = Shape.of(batch, channels, spatial);
        Shape channelShape = Shape.of(1, channels, 1);
        try (Composition c = new Composition()) {
            Tensor xr = c.temp(x.reshape(cube));
            Tensor gyr = c.temp(gy.reshape(cube));
            Tensor mean = training ? c.meanKept(xr, channelShape, 0, 2) : c.temp(runningMean.reshape(channelShape));
            Tensor xc = c.temp(TensorOps.sub(xr, mean));
            Tensor var = training ? c.meanKept(c.temp(TensorOps.mul(xc, xc)), channelShape, 0, 2)
                    : c.temp(runningVar.reshape(channelShape));
            Tensor sdev = c.sdev(var, eps);
            Tensor xhat = c.temp(TensorOps.div(xc, sdev));
            Tensor gGamma = needGamma ? c.result(c.sumTo(c.temp(TensorOps.mul(gyr, xhat)), gamma.shape(), 0, 2))
                    : null;
            Tensor gBeta = betaShape != null ? c.result(c.sumTo(gyr, betaShape, 0, 2)) : null;
            Tensor gx = null;
            if (needInput) {
                Tensor inner = training ? c.centered(gyr, xhat, channelShape, 0, 2) : gyr;
                Tensor scaled = gamma == null ? inner
                        : c.temp(TensorOps.mul(inner, c.temp(gamma.reshape(channelShape))));
                gx = c.result(c.temp(TensorOps.div(scaled, sdev)).reshape(x.shape()));
            }
            return c.finish(gx, gGamma, gBeta);
        }
    }

    /**
     * {@link GroupNormGrad}'s formulas, recorded: the statistics and the input gradient on the input viewed as
     * {@code [B, G, (C/G) * S]}, the affine gradients on {@code [B, C, S]} ({@code S = prod(*)}).
     */
    private static List<Tensor> groupNormComposed(Tensor gy, Tensor x, int groups, Tensor gamma, float eps,
                                                  boolean needInput, boolean needGamma, Shape betaShape) {
        requireGradientShape(gy, x);
        int batch = x.shape().dim(0);
        int channels = x.shape().dim(1);
        int spatial = Math.toIntExact(x.numElements() / ((long) batch * channels));
        Shape grouped = Shape.of(batch, groups, Math.toIntExact(x.numElements() / ((long) batch * groups)));
        Shape statShape = Shape.of(batch, groups, 1);
        Shape cube = Shape.of(batch, channels, spatial);
        Shape channelShape = Shape.of(1, channels, 1);
        try (Composition c = new Composition()) {
            Tensor xg = c.temp(x.reshape(grouped));
            Tensor xc = c.temp(TensorOps.sub(xg, c.meanKept(xg, statShape, 2)));
            Tensor sdev = c.sdev(c.meanKept(c.temp(TensorOps.mul(xc, xc)), statShape, 2), eps);
            Tensor xhat = c.temp(TensorOps.div(xc, sdev));
            Tensor gyc = c.temp(gy.reshape(cube));
            Tensor gGamma = null;
            if (needGamma) {
                Tensor xhatC = c.temp(xhat.reshape(cube));
                gGamma = c.result(c.sumTo(c.temp(TensorOps.mul(gyc, xhatC)), gamma.shape(), 0, 2));
            }
            Tensor gBeta = betaShape != null ? c.result(c.sumTo(gyc, betaShape, 0, 2)) : null;
            Tensor gx = null;
            if (needInput) {
                Tensor d = gamma == null ? gyc : c.temp(TensorOps.mul(gyc, c.temp(gamma.reshape(channelShape))));
                Tensor dg = c.temp(d.reshape(grouped));
                gx = c.result(c.temp(TensorOps.div(c.centered(dg, xhat, statShape, 2), sdev)).reshape(x.shape()));
            }
            return c.finish(gx, gGamma, gBeta);
        }
    }

    private static void requireGradientShape(Tensor gy, Tensor x) {
        if (!gy.shape().equals(x.shape())) {
            throw new IllegalArgumentException("normalization gradient " + gy.shape() + " does not match the input "
                    + x.shape());
        }
    }

    /**
     * The tensors of one recorded composition. Temporaries are closed when it closes; results are handed to the
     * caller by {@link #finish}, and closed with the temporaries when the composition ends without finishing or a
     * temporary fails to close. Closing a temporary never invalidates a result: views share reference-counted storage
     * and the recorded nodes keep their own handles of what they saved.
     */
    private static final class Composition implements AutoCloseable {
        private final List<Tensor> temps = new ArrayList<>();
        private final List<Tensor> results = new ArrayList<>();
        private boolean finished;

        Tensor temp(Tensor t) {
            temps.add(t);
            return t;
        }

        Tensor result(Tensor t) {
            results.add(t);
            return t;
        }

        /** {@code mean(t, dims)} reshaped to {@code kept} (the reduced dims kept as size 1). */
        Tensor meanKept(Tensor t, Shape kept, int... dims) {
            return temp(temp(TensorOps.mean(t, dims)).reshape(kept));
        }

        /** {@code sum(t, dims)} reshaped to {@code target}; the caller owns the result. */
        Tensor sumTo(Tensor t, Shape target, int... dims) {
            return temp(TensorOps.sum(t, dims)).reshape(target);
        }

        /** {@code sqrt(var + eps)}. */
        Tensor sdev(Tensor var, float eps) {
            return temp(TensorOps.sqrt(temp(TensorOps.add(var, eps))));
        }

        /** {@code d - mean(d) - xhat * mean(d * xhat)} with the means over {@code dims}, kept as {@code kept}. */
        Tensor centered(Tensor d, Tensor xhat, Shape kept, int... dims) {
            Tensor shifted = temp(TensorOps.sub(d, meanKept(d, kept, dims)));
            Tensor projection = temp(TensorOps.mul(xhat, meanKept(temp(TensorOps.mul(d, xhat)), kept, dims)));
            return temp(TensorOps.sub(shifted, projection));
        }

        /** The gradients in input order (null entries allowed), now owned by the caller. */
        List<Tensor> finish(Tensor... gradients) {
            finished = true;
            return Arrays.asList(gradients);
        }

        @Override
        public void close() {
            RuntimeException failure = closeEach(temps, null);
            if (!finished || failure != null) {
                failure = closeEach(results, failure);
            }
            if (failure != null) {
                throw failure;
            }
        }

        private static RuntimeException closeEach(List<Tensor> tensors, RuntimeException failure) {
            RuntimeException result = failure;
            for (int i = tensors.size() - 1; i >= 0; i--) {
                try {
                    tensors.get(i).close();
                } catch (RuntimeException e) {
                    if (result == null) {
                        result = e;
                    } else {
                        result.addSuppressed(e);
                    }
                }
            }
            tensors.clear();
            return result;
        }
    }

    // ------------------------------------------------------------------ helpers

    /** A gradient for the scale parameter needs the parameter: the primitive returns it in the parameter's shape. */
    private static void requireScaleForGradient(Tensor scale, boolean need, String name) {
        if (need && scale == null) {
            throw new IllegalArgumentException("a " + name + " gradient needs " + name);
        }
    }

    /** Saves the non-null tensors in order; the input must be given. */
    private static void saveAll(GradFunctions.SavingGradFn fn, Tensor x, Tensor... rest) {
        Objects.requireNonNull(x, "the saved input must not be null");
        List<Tensor> tensors = new ArrayList<>(rest.length + 1);
        tensors.add(x);
        for (Tensor t : rest) {
            if (t != null) {
                tensors.add(t);
            }
        }
        fn.save(tensors.toArray(new Tensor[0]));
    }
}

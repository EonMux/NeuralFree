package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;

/** Norm operands: a weight and bias, or neither for a weightless norm. */
final class Norm {
    static final Norm WEIGHTLESS = new Norm(null, null);

    final GpuWeight weight;
    final GpuWeight bias;
    /** Epsilon of this norm; NaN uses the layer's norm epsilon. */
    final float eps;
    /** {@code norm(x * weight)} instead of {@code norm(x) * weight} (muse-spark post-FFN norm). */
    final boolean weightFirst;
    /** Equal row slices normalized on their own before the full-width weight; 1 for a plain row norm. */
    final int groups;

    Norm(GpuWeight weight, GpuWeight bias) {
        this(weight, bias, Float.NaN, false, 1);
    }

    private Norm(GpuWeight weight, GpuWeight bias, float eps, boolean weightFirst, int groups) {
        this.weight = weight;
        this.bias = bias;
        this.eps = eps;
        this.weightFirst = weightFirst;
        this.groups = groups;
    }

    /** This norm applied with {@code eps}. */
    Norm withEps(float e) {
        return new Norm(weight, bias, e, weightFirst, groups);
    }

    /**
     * This norm over {@code g} equal slices of each row, each normalized on its own, then multiplied by the
     * full-width weight (k2-horizon.cpp:124-135).
     *
     * @throws IllegalArgumentException when the weight is not one single-buffer F32 vector of {@code width}
     *                                  values, a bias is present, or {@code width} is not a multiple of {@code g}
     */
    Norm grouped(String arch, int il, int width, int g) {
        if (g == 1) {
            return this;
        }
        if (weightFirst || weight == null || bias != null || weight.type() != GGMLType.F32
                || !weight.isSingleShard() || weight.rows() != 1 || weight.cols() != width || width % g != 0) {
            throw new IllegalArgumentException(arch + " layer " + il + ": a norm over " + g + " groups needs a"
                    + " single-buffer F32 weight vector of " + width + " values, a width divisible by the groups"
                    + " and no bias, got " + weight);
        }
        return new Norm(weight, null, eps, false, g);
    }

    /**
     * This norm with its weight applied before the normalization.
     *
     * @throws IllegalArgumentException when the weight is not one single-buffer F32 vector without a bias
     */
    Norm weightFirst(String arch, int il, int width) {
        if (weight == null || bias != null || weight.type() != GGMLType.F32 || !weight.isSingleShard()
                || weight.rows() != 1 || weight.cols() != width) {
            throw new IllegalArgumentException(arch + " layer " + il + ": a norm applied to the weighted input"
                    + " needs a single-buffer F32 weight vector of " + width + " values and no bias, got " + weight);
        }
        return new Norm(weight, null, eps, true, 1);
    }
}

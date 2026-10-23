package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Mean squared error: {@code (prediction - target)^2} per element of the broadcast shape, then the reduction.
 * Composed from {@link TensorOps} (sub, mul), so it runs on the device of the operands; per element it is in the
 * single-rounding class, and a SUM or MEAN result in the reduction class ({@linkplain com.eonmux.neuralfree.tensor.gpu
 * numerical contract}).
 */
public final class MSELoss implements LossFunction {

    private final Reduction reduction;

    public MSELoss() {
        this(Reduction.MEAN);
    }

    public MSELoss(Reduction reduction) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
    }

    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Tensor squared;
        try (Tensor diff = TensorOps.sub(prediction, target)) {
            squared = TensorOps.mul(diff, diff);
        }
        return LossTensors.reduce(squared, reduction);
    }

    @Override
    public String name() {
        return "MSELoss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    @Override
    public String toString() {
        return "MSELoss{reduction=" + reduction + "}";
    }
}

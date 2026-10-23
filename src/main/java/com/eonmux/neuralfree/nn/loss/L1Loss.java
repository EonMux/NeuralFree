package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Mean absolute error: {@code |prediction - target|} per element of the broadcast shape, then the reduction.
 * Composed from {@link TensorOps} (sub, abs), so it runs on the device of the operands; per element it is in the
 * single-rounding class, and a SUM or MEAN result in the reduction class ({@linkplain com.eonmux.neuralfree.tensor.gpu
 * numerical contract}).
 */
public final class L1Loss implements LossFunction {

    private final Reduction reduction;

    public L1Loss() {
        this(Reduction.MEAN);
    }

    public L1Loss(Reduction reduction) {
        this.reduction = Objects.requireNonNull(reduction, "Reduction cannot be null");
    }

    @Override
    public Tensor compute(Tensor prediction, Tensor target) {
        Objects.requireNonNull(prediction, "Prediction cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        Tensor absDiff;
        try (Tensor diff = TensorOps.sub(prediction, target)) {
            absDiff = TensorOps.abs(diff);
        }
        return LossTensors.reduce(absDiff, reduction);
    }

    @Override
    public String name() {
        return "L1Loss";
    }

    @Override
    public Reduction reduction() {
        return reduction;
    }

    @Override
    public String toString() {
        return "L1Loss{reduction=" + reduction + "}";
    }
}

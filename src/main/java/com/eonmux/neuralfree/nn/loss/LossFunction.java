package com.eonmux.neuralfree.nn.loss;

import com.eonmux.neuralfree.tensor.Tensor;

public interface LossFunction {

    Tensor compute(Tensor prediction, Tensor target);

    String name();

    default Reduction reduction() {
        return Reduction.MEAN;
    }

    enum Reduction {
        NONE,
        MEAN,
        SUM
    }
}

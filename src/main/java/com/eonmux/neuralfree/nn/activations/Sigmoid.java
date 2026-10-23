package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class Sigmoid extends Module {

    public Sigmoid() {
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.sigmoid(input);
    }

    @Override
    public String toString() {
        return "Sigmoid{}";
    }
}

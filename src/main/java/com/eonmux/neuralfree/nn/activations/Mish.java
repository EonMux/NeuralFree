package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class Mish extends Module {

    public Mish() {
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.mish(input);
    }

    @Override
    public String toString() {
        return "Mish{}";
    }
}

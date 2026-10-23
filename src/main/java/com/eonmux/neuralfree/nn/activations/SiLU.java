package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class SiLU extends Module {

    public SiLU() {
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.silu(input);
    }

    @Override
    public String toString() {
        return "SiLU{}";
    }
}

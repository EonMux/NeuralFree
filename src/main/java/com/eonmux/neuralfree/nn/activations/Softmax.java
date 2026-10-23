package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class Softmax extends Module {
    private final int dim;

    public Softmax() {
        this(-1);
    }

    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.softmax(input, dim);
    }

    public int dim() {
        return dim;
    }

    @Override
    public String toString() {
        return "Softmax{dim=" + dim + "}";
    }
}

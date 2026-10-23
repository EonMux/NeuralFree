package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class ELU extends Module {
    private final float alpha;

    public ELU() {
        this(1.0f);
    }

    public ELU(float alpha) {
        if (alpha <= 0.0f) {
            throw new IllegalArgumentException("alpha must be positive: " + alpha);
        }
        this.alpha = alpha;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.elu(input, alpha);
    }

    public float alpha() {
        return alpha;
    }

    @Override
    public String toString() {
        return "ELU{alpha=" + alpha + "}";
    }
}

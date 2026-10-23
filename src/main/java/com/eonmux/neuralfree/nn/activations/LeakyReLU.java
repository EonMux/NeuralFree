package com.eonmux.neuralfree.nn.activations;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

public final class LeakyReLU extends Module {
    private final float negativeSlope;

    public LeakyReLU() {
        this(0.01f);
    }

    public LeakyReLU(float negativeSlope) {
        if (negativeSlope < 0.0f) {
            throw new IllegalArgumentException("negativeSlope must be non-negative: " + negativeSlope);
        }
        this.negativeSlope = negativeSlope;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        return TensorOps.leakyRelu(input, negativeSlope);
    }

    public float negativeSlope() {
        return negativeSlope;
    }

    @Override
    public String toString() {
        return "LeakyReLU{negativeSlope=" + negativeSlope + "}";
    }
}

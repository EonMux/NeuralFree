package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;
import java.util.Random;

/**
 * Pre-norm transformer encoder block of the multimodal encoders ({@link TextEncoder}, {@link VisionEncoder},
 * {@link AudioEncoder}): {@code r = x + outProj(attn(norm1(x)))}, then {@code r + fc2(gelu(fc1(norm2(r))))} with a
 * feed-forward width of {@code 4 * hiddenDim}. The child modules are registered as {@code qProj}, {@code kProj},
 * {@code vProj}, {@code outProj}, {@code fc1}, {@code fc2}, {@code norm1} and {@code norm2}.
 *
 * <p>Self-attention is {@link ScaledDotProductAttention} without a mask and GELU is {@link TensorOps#gelu} (the tanh
 * approximation), so the block runs on the device when the train kernels are available. Intermediates are released
 * before {@link #forward} returns; the input is never closed.</p>
 */
final class EncoderBlock extends Module {
    private final int hiddenDim;
    private final int numHeads;

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear outProj;
    private final Linear fc1;
    private final Linear fc2;
    private final LayerNorm norm1;
    private final LayerNorm norm2;

    /**
     * Creates and initializes the block on {@code backend}, drawing the projection weights from {@code random}.
     *
     * @throws IllegalArgumentException when {@code hiddenDim} is not divisible by {@code numHeads}
     */
    EncoderBlock(ComputeBackend backend, int hiddenDim, int numHeads, Random random) {
        Objects.requireNonNull(backend, "backend cannot be null");
        Objects.requireNonNull(random, "random cannot be null");
        ScaledDotProductAttention.requireDivisible(hiddenDim, numHeads);
        this.hiddenDim = hiddenDim;
        this.numHeads = numHeads;

        qProj = registerModule("qProj", linear(backend, hiddenDim, hiddenDim, random));
        kProj = registerModule("kProj", linear(backend, hiddenDim, hiddenDim, random));
        vProj = registerModule("vProj", linear(backend, hiddenDim, hiddenDim, random));
        outProj = registerModule("outProj", linear(backend, hiddenDim, hiddenDim, random));
        fc1 = registerModule("fc1", linear(backend, hiddenDim, hiddenDim * 4, random));
        fc2 = registerModule("fc2", linear(backend, hiddenDim * 4, hiddenDim, random));

        norm1 = registerModule("norm1", layerNorm(backend, hiddenDim));
        norm2 = registerModule("norm2", layerNorm(backend, hiddenDim));
    }

    /** One block over a {@code [batch, length, hiddenDim]} input; returns a new tensor of the same shape. */
    @Override
    public Tensor forward(Tensor x) {
        Objects.requireNonNull(x, "input cannot be null");
        Tensor residual;
        try (Tensor normed = norm1.forward(x);
             Tensor attn = selfAttention(normed)) {
            residual = TensorOps.add(x, attn);
        }
        try (Tensor r = residual;
             Tensor normed = norm2.forward(r);
             Tensor ff = feedForward(normed)) {
            return TensorOps.add(r, ff);
        }
    }

    private Tensor selfAttention(Tensor x) {
        try (Tensor q = qProj.forward(x);
             Tensor k = kProj.forward(x);
             Tensor v = vProj.forward(x);
             Tensor context = ScaledDotProductAttention.attend(q, k, v, numHeads, null, false)) {
            return outProj.forward(context);
        }
    }

    private Tensor feedForward(Tensor x) {
        try (Tensor expanded = fc1.forward(x);
             Tensor activated = TensorOps.gelu(expanded)) {
            return fc2.forward(activated);
        }
    }

    private static Linear linear(ComputeBackend backend, int in, int out, Random random) {
        Linear layer = new Linear(in, out);
        layer.initialize(backend, random);
        return layer;
    }

    private static LayerNorm layerNorm(ComputeBackend backend, int dim) {
        LayerNorm norm = new LayerNorm(new int[]{dim});
        norm.initialize(backend);
        return norm;
    }

    @Override
    public String toString() {
        return "EncoderBlock{hiddenDim=" + hiddenDim + ", numHeads=" + numHeads + '}';
    }
}

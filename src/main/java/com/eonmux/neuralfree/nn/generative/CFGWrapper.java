package com.eonmux.neuralfree.nn.generative;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Classifier-free guidance: {@code scale * conditional(x) - (scale - 1) * unconditional(x)}, a {@link TensorOps}
 * composition recorded like any forward pass when grad mode is on and a model parameter requires grad (sampling
 * loops call it under {@link com.eonmux.neuralfree.autograd.ComputationGraph#noGrad()}). The model outputs are
 * closed after use unless a model returns its input.
 *
 * <p>The wrapped models are registered children, {@value #CONDITIONAL} and (when it is a separate module)
 * {@value #UNCONDITIONAL}, so {@link #parameters()}, {@link #stateDict()} (keys {@code conditional.<key>} and
 * {@code unconditional.<key>}), {@link #to}, {@link #train(boolean)}, {@link #zeroGrad()} and {@link #close()} reach
 * them; a model shared by both roles is registered and listed once. The wrapper takes the models over and closes them
 * on {@link #close()}.</p>
 */
public final class CFGWrapper extends Module {

    /** Registered name of the conditional model. */
    public static final String CONDITIONAL = "conditional";
    /** Registered name of the unconditional model when it is not the conditional one. */
    public static final String UNCONDITIONAL = "unconditional";

    private final Module conditionalModel;
    private final Module unconditionalModel;
    private final float guidanceScale;
    private boolean cfgClosed = false;

    public CFGWrapper(Module conditionalModel, Module unconditionalModel, float guidanceScale) {
        Objects.requireNonNull(conditionalModel, "Conditional model cannot be null");
        Objects.requireNonNull(unconditionalModel, "Unconditional model cannot be null");
        if (!(guidanceScale >= 1.0f)) {
            throw new IllegalArgumentException("guidanceScale must be >= 1.0, got: " + guidanceScale);
        }
        this.conditionalModel = registerModule(CONDITIONAL, conditionalModel);
        this.unconditionalModel = unconditionalModel == conditionalModel
            ? conditionalModel
            : registerModule(UNCONDITIONAL, unconditionalModel);
        this.guidanceScale = guidanceScale;
    }

    public CFGWrapper(Module model, float guidanceScale) {
        this(model, model, guidanceScale);
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();

        Tensor conditionalOutput = conditionalModel.forward(input);

        if (conditionalModel == unconditionalModel) {
            return conditionalOutput;
        }

        return guided(input, conditionalOutput, guidanceScale);
    }

    public Tensor forwardWithGuidance(Tensor input, float scale) {
        ensureOpen();

        if (scale == 1.0f) {
            return conditionalModel.forward(input);
        }

        return guided(input, conditionalModel.forward(input), scale);
    }

    /**
     * {@code scale * conditionalOutput - (scale - 1) * unconditional(input)}; {@code conditionalOutput} (taken over)
     * and the unconditional output are closed unless they are {@code input}.
     */
    private Tensor guided(Tensor input, Tensor conditionalOutput, float scale) {
        Tensor unconditionalOutput = null;
        try {
            unconditionalOutput = unconditionalModel.forward(input);
            try (Tensor scaledConditional = TensorOps.mul(conditionalOutput, scale);
                 Tensor scaledUnconditional = TensorOps.mul(unconditionalOutput, scale - 1.0f)) {
                return TensorOps.sub(scaledConditional, scaledUnconditional);
            }
        } finally {
            release(conditionalOutput, input);
            release(unconditionalOutput, input);
        }
    }

    /** Closes {@code t} unless it is null or {@code keep}. */
    private static void release(Tensor t, Tensor keep) {
        if (t != null && t != keep) {
            t.close();
        }
    }

    public Tensor forwardConditional(Tensor input) {
        ensureOpen();
        return conditionalModel.forward(input);
    }

    public Tensor forwardUnconditional(Tensor input) {
        ensureOpen();
        return unconditionalModel.forward(input);
    }

    public Module conditionalModel() {
        return conditionalModel;
    }

    public Module unconditionalModel() {
        return unconditionalModel;
    }

    public float guidanceScale() {
        return guidanceScale;
    }

    public boolean isSharedModel() {
        return conditionalModel == unconditionalModel;
    }

    private void ensureOpen() {
        if (cfgClosed) {
            throw new IllegalStateException("CFGWrapper has been closed");
        }
    }

    /** Closes the wrapper and, as registered children, the wrapped models; idempotent. */
    @Override
    public void close() {
        cfgClosed = true;
        super.close();
    }

    @Override
    public String toString() {
        return "CFGWrapper{guidanceScale=" + guidanceScale
            + ", sharedModel=" + isSharedModel()
            + ", closed=" + cfgClosed + "}";
    }
}

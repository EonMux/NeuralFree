package com.eonmux.neuralfree.gguf.train;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The optimizer of a finetune run: AdamW (the default of {@code llama-finetune}) or SGD with L2 weight decay, over the
 * host float arrays of a {@link Trainer}.
 *
 * <p>The update is the one {@code llama-finetune} uses and the one the tensor optimizers of
 * {@code com.eonmux.neuralfree.optim} compute: AdamW keeps a first and a second moment per weight, corrects them by
 * {@code 1 - beta^t}, applies the decoupled decay {@code w *= 1 - lr * wd}, then
 * {@code w -= lr * mhat / (sqrt(vhat) + eps)} with {@code beta1 = 0.9}, {@code beta2 = 0.999} and
 * {@code eps = 1e-8}; SGD applies {@code w -= lr * (grad + wd * w)}. The moment buffers are this optimizer's own;
 * the weights and gradients are the model's parameters, so a step mutates exactly the arrays a later save writes to
 * GGUF.</p>
 *
 * <p>The learning rate is passed to every {@link #step(float)} because the run schedules it per epoch
 * ({@link FinetuneSchedule}); the weight decay is fixed for the run. Not thread-safe.</p>
 */
public final class FinetuneOptimizer {

    /** The optimizers {@code llama-finetune} offers ({@code -opt}). */
    public enum Kind {
        /** Decoupled AdamW, {@code llama-finetune}'s default. */
        ADAMW("adamw"),
        /** Plain SGD with L2 weight decay. */
        SGD("sgd");

        private final String flag;

        Kind(String flag) {
            this.flag = flag;
        }

        /** The name {@code -opt} takes. */
        public String flag() {
            return flag;
        }

        /** The kind named by {@code -opt}, or null when the name is not an optimizer of the run. */
        public static Kind parse(String name) {
            for (Kind k : values()) {
                if (k.flag.equalsIgnoreCase(name)) {
                    return k;
                }
            }
            return null;
        }

        /** The names {@code -opt} accepts, for an error message. */
        public static String names() {
            StringBuilder sb = new StringBuilder();
            for (Kind k : values()) {
                if (sb.length() > 0) {
                    sb.append(", ");
                }
                sb.append(k.flag);
            }
            return sb.toString();
        }
    }

    /** AdamW's first moment decay. */
    static final float BETA1 = 0.9f;
    /** AdamW's second moment decay. */
    static final float BETA2 = 0.999f;
    /** AdamW's denominator floor. */
    static final float EPS = 1e-8f;

    private final Kind kind;
    private final float weightDecay;
    private final List<TrainableModel.Parameter> parameters;
    private final List<float[]> firstMoments;
    private final List<float[]> secondMoments;
    private int stepCount;

    /**
     * An optimizer over every trainable tensor of {@code model}.
     *
     * @param weightDecay the decay coefficient, zero or more
     */
    public FinetuneOptimizer(Trainer model, Kind kind, float weightDecay) {
        Objects.requireNonNull(model, "model");
        this.kind = Objects.requireNonNull(kind, "kind");
        if (!(weightDecay >= 0f) || !Float.isFinite(weightDecay)) {
            throw new IllegalArgumentException("the weight decay must be finite and non-negative, not "
                    + weightDecay);
        }
        this.weightDecay = weightDecay;
        this.parameters = Collections.unmodifiableList(new ArrayList<>(model.parameters()));
        List<float[]> first = new ArrayList<>(parameters.size());
        List<float[]> second = new ArrayList<>(parameters.size());
        for (TrainableModel.Parameter p : parameters) {
            if (kind == Kind.ADAMW) {
                first.add(new float[p.size()]);
                second.add(new float[p.size()]);
            }
        }
        this.firstMoments = Collections.unmodifiableList(first);
        this.secondMoments = Collections.unmodifiableList(second);
    }

    /** The optimizer of the run. */
    public Kind kind() {
        return kind;
    }

    /** The weight decay every step applies. */
    public float weightDecay() {
        return weightDecay;
    }

    /** The steps taken so far; AdamW's bias correction uses it. */
    public int stepCount() {
        return stepCount;
    }

    /** The parameters the optimizer steps, in the model's order. */
    public List<TrainableModel.Parameter> parameters() {
        return parameters;
    }

    /**
     * One update of every parameter from its accumulated gradient.
     *
     * @param learningRate the learning rate of this step, positive and finite
     * @throws IllegalArgumentException when the learning rate is not positive and finite
     */
    public void step(float learningRate) {
        if (!(learningRate > 0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("the learning rate must be positive and finite, not " + learningRate);
        }
        if (parameters.isEmpty()) {
            return;
        }
        stepCount++;
        float biasCorrection1 = 1f - (float) Math.pow(BETA1, stepCount);
        float biasCorrection2 = 1f - (float) Math.pow(BETA2, stepCount);
        float stepSize = learningRate / biasCorrection1;
        for (int i = 0; i < parameters.size(); i++) {
            TrainableModel.Parameter p = parameters.get(i);
            float[] w = p.value();
            float[] g = p.grad();
            if (kind == Kind.ADAMW) {
                adamw(w, g, firstMoments.get(i), secondMoments.get(i), learningRate, stepSize, biasCorrection2);
            } else {
                sgd(w, g, learningRate);
            }
        }
    }

    private void adamw(float[] w, float[] g, float[] m, float[] v, float lr, float stepSize, float biasCorrection2) {
        for (int i = 0; i < w.length; i++) {
            float grad = g[i];
            if (weightDecay != 0f) {
                w[i] -= lr * weightDecay * w[i];
            }
            m[i] = BETA1 * m[i] + (1f - BETA1) * grad;
            v[i] = BETA2 * v[i] + (1f - BETA2) * grad * grad;
            float vHat = v[i] / biasCorrection2;
            w[i] -= stepSize * m[i] / ((float) Math.sqrt(vHat) + EPS);
        }
    }

    private void sgd(float[] w, float[] g, float lr) {
        for (int i = 0; i < w.length; i++) {
            w[i] -= lr * (g[i] + weightDecay * w[i]);
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s(wd=%g, steps=%d, %d tensors)", kind.flag(), weightDecay, stepCount,
                parameters.size());
    }
}

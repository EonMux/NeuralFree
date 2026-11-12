package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The optimizer of a finetune run: AdamW (the default of {@code llama-finetune}) or SGD with L2 weight decay, over the
 * parameters of a {@link Trainer}.
 *
 * <p>The update is the one {@code llama-finetune} uses and the one the tensor optimizers of
 * {@code com.eonmux.neuralfree.optim} compute: AdamW keeps a first and a second moment per weight, corrects them by
 * {@code 1 - beta^t}, applies the decoupled decay {@code w *= 1 - lr * wd}, then
 * {@code w -= lr * mhat / (sqrt(vhat) + eps)} with {@code beta1 = 0.9}, {@code beta2 = 0.999} and
 * {@code eps = 1e-8}; SGD applies {@code w -= lr * (grad + wd * w)}.</p>
 *
 * <p>This class owns the schedule state (the step count and its bias corrections); the moments and the arithmetic
 * are an {@link Update} the trainer provides ({@link Trainer#newUpdate}), so they live where the trainer keeps its
 * parameters: host arrays for the host trainer, device tensors for the device trainer. A step therefore mutates
 * exactly the values a later save writes to GGUF, wherever they are held.</p>
 *
 * <p>The learning rate is passed to every {@link #step(float)} because the run schedules it per epoch
 * ({@link FinetuneRunner.Config#learningRate}); the weight decay is fixed for the run. {@link #close()} releases the update's state.
 * Not thread-safe.</p>
 */
public final class FinetuneOptimizer implements AutoCloseable {

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

    /** Where an {@link Update} keeps the moments and runs the arithmetic. */
    public enum Placement {
        /** Host float arrays, updated on the engine's CPU pool. */
        HOST,
        /** Device tensors, updated by the {@code train} kernels. */
        DEVICE
    }

    /**
     * The state and arithmetic of one optimizer's updates over a trainer's parameters, in the formulas of the class
     * description.
     */
    public interface Update extends AutoCloseable {

        /**
         * One update of every parameter from its accumulated gradient: AdamW with the step size
         * {@code learningRate / biasCorrection1} and {@code vhat = v / biasCorrection2}, or SGD with
         * {@code learningRate} (which ignores the corrections).
         */
        void apply(float learningRate, float biasCorrection1, float biasCorrection2);

        /** Where the state lives. */
        Placement placement();

        /** Releases the state; later calls do nothing. */
        @Override
        void close();
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
    private final Update update;
    private int stepCount;

    /**
     * An optimizer over every trainable tensor of {@code model}, its state placed by {@link Trainer#newUpdate}.
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
        this.update = Objects.requireNonNull(model.newUpdate(kind, weightDecay), "the trainer's update");
    }

    /**
     * The host update over {@code parameters}: moments in host arrays and the element loops split by range over the
     * engine's CPU pool, which leaves every element's arithmetic, and so every bit, as a sequential loop computes it.
     */
    static Update hostUpdate(List<TrainableModel.Parameter> parameters, Kind kind, float weightDecay) {
        return new HostUpdate(parameters, kind, weightDecay);
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

    /** Where the optimizer's moments live and its arithmetic runs. */
    public Placement placement() {
        return update.placement();
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
        update.apply(learningRate, biasCorrection1, biasCorrection2);
    }

    /** Releases the optimizer's state (device moments, for a device update). */
    @Override
    public void close() {
        update.close();
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "%s(wd=%g, steps=%d, %d tensors, %s)", kind.flag(), weightDecay, stepCount,
                parameters.size(), update.placement().name().toLowerCase(Locale.ROOT));
    }

    /** {@link #hostUpdate}. */
    private static final class HostUpdate implements Update {
        /** Elements one pool task updates at least. */
        private static final int GRAIN = 1 << 15;

        private final List<TrainableModel.Parameter> parameters;
        private final Kind kind;
        private final float weightDecay;
        private final List<float[]> firstMoments;
        private final List<float[]> secondMoments;

        HostUpdate(List<TrainableModel.Parameter> parameters, Kind kind, float weightDecay) {
            this.parameters = Collections.unmodifiableList(new ArrayList<>(parameters));
            this.kind = kind;
            this.weightDecay = weightDecay;
            List<float[]> first = new ArrayList<>(parameters.size());
            List<float[]> second = new ArrayList<>(parameters.size());
            if (kind == Kind.ADAMW) {
                for (TrainableModel.Parameter p : parameters) {
                    first.add(new float[p.size()]);
                    second.add(new float[p.size()]);
                }
            }
            this.firstMoments = Collections.unmodifiableList(first);
            this.secondMoments = Collections.unmodifiableList(second);
        }

        @Override
        public void apply(float learningRate, float biasCorrection1, float biasCorrection2) {
            float stepSize = learningRate / biasCorrection1;
            for (int i = 0; i < parameters.size(); i++) {
                TrainableModel.Parameter p = parameters.get(i);
                float[] w = p.value();
                float[] g = p.grad();
                if (kind == Kind.ADAMW) {
                    float[] m = firstMoments.get(i);
                    float[] v = secondMoments.get(i);
                    CpuParallel.parallelFor(w.length, GRAIN, (from, to) ->
                            adamw(w, g, m, v, from, to, learningRate, stepSize, biasCorrection2));
                } else {
                    CpuParallel.parallelFor(w.length, GRAIN, (from, to) -> sgd(w, g, from, to, learningRate));
                }
            }
        }

        private void adamw(float[] w, float[] g, float[] m, float[] v, int from, int to, float lr, float stepSize,
                           float biasCorrection2) {
            for (int i = from; i < to; i++) {
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

        private void sgd(float[] w, float[] g, int from, int to, float lr) {
            for (int i = from; i < to; i++) {
                w[i] -= lr * (g[i] + weightDecay * w[i]);
            }
        }

        @Override
        public Placement placement() {
            return Placement.HOST;
        }

        @Override
        public void close() {
        }
    }
}

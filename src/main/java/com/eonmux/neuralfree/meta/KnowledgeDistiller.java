package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.KLDivLoss;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Knowledge distillation (Hinton et al.) of a student from a frozen teacher, computed on the student's backend.
 *
 * <p><b>Loss.</b> Both outputs are read as a batch of logits over their last dimension: an output of rank 2 or more
 * holds {@code B = dim(0)} samples, and an output of rank 0 or 1 is one sample ({@code B = 1}). With
 * {@code T = temperature}, the teacher's soft targets {@code t = softmax(teacherLogits / T)} and the student's
 * log-probabilities {@code log s = log_softmax(studentLogits / T)}, both along the last dimension,</p>
 * <pre>
 *   distillation = T^2 * (1 / B) * sum over all elements of t * (log t - log s)
 *   total        = alpha * distillation + (1 - alpha) * hardTargetLoss(studentOutput, target)
 * </pre>
 * <p>The distillation term is the forward KL divergence from teacher to student that PyTorch code writes as
 * {@code T * T * kl_div(log_softmax(s / T, -1), softmax(t / T, -1), reduction='batchmean')}; it is computed with
 * {@link KLDivLoss} ({@code batchmean}, probability targets), so an entry with {@code t == 0} contributes 0
 * ({@code xlogy}), and nothing is clamped. The student's log-probabilities come from {@link TensorOps#logSoftmax}, so
 * they stay finite for finite logits. The distillation term's gradient with respect to the student's logits is
 * {@code (T / B) * (softmax(studentLogits / T) - t)} (each row of {@code t} sums to 1); the {@code T^2} factor keeps
 * it on the scale of the hard-target gradient as the temperature changes. The teacher's and the student's outputs
 * must have the same shape.</p>
 *
 * <p><b>Training.</b> The student's log-probabilities, the divergence and the total are built with the recording
 * {@link TensorOps}. When the student's output requires grad the total is backpropagated, together with the auxiliary
 * loss terms the student's forward pass leaves pending ({@link Module#takeAuxiliaryLoss()}); the pending terms of the
 * teacher's and of evaluation forward passes are dropped ({@link Module#clearAuxiliaryLoss()}). Each student parameter
 * with a gradient then takes the SGD step {@code w -= learningRate * g} in place, without recording, and its gradient
 * is released. Loss values (without auxiliary terms) are read back as scalars for the result;
 * nothing else leaves the device.</p>
 *
 * <p>The teacher is frozen: the constructor sets it to evaluation mode and turns off its parameters' gradients
 * ({@link Module#setRequiresGrad(boolean)}), and its forward passes and soft targets are computed under
 * {@link ComputationGraph#noGrad()}, so the targets are constants. Each training step runs inside a
 * {@link ComputationGraph.StepScope}, so it leaves no autograd node behind.</p>
 */
public final class KnowledgeDistiller implements AutoCloseable {
    /** PyTorch {@code kl_div(..., reduction='batchmean')} with probability targets. */
    private static final KLDivLoss BATCH_MEAN_KL = new KLDivLoss();

    private final Module teacher;
    private final Module student;
    private final float temperature;
    private final float alpha;
    private final LossFunction hardTargetLoss;
    private final AtomicBoolean closed;

    public KnowledgeDistiller(Module teacher, Module student, float temperature, 
                              float alpha, LossFunction hardTargetLoss) {
        Objects.requireNonNull(teacher, "Teacher model cannot be null");
        Objects.requireNonNull(student, "Student model cannot be null");
        Objects.requireNonNull(hardTargetLoss, "Hard target loss function cannot be null");
        
        if (temperature <= 0) {
            throw new IllegalArgumentException("Temperature must be positive");
        }
        if (alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1");
        }

        requireDisjointParameters(teacher, student);

        this.teacher = teacher;
        this.student = student;
        this.temperature = temperature;
        this.alpha = alpha;
        this.hardTargetLoss = hardTargetLoss;
        this.closed = new AtomicBoolean(false);
        
        teacher.eval();
        teacher.setRequiresGrad(false);
    }

    /**
     * @throws IllegalArgumentException when the teacher and the student share a parameter: freezing the teacher would
     *                                  freeze the student as well
     */
    private static void requireDisjointParameters(Module teacher, Module student) {
        Set<Parameter> studentParams = Collections.newSetFromMap(new IdentityHashMap<>());
        studentParams.addAll(student.parameters());
        for (Map.Entry<String, Parameter> entry : teacher.namedParameters().entrySet()) {
            if (studentParams.contains(entry.getValue())) {
                throw new IllegalArgumentException("Teacher parameter '" + entry.getKey() + "' is also a student "
                    + "parameter; the teacher is frozen, so it must not share parameters with the student");
            }
        }
    }

    /** {@code steps} distillation steps on one input batch; one loss entry per step. */
    public DistillationResult train(Tensor input, Tensor target, float learningRate, int steps) {
        ensureOpen();
        
        if (input == null || target == null) {
            throw new IllegalArgumentException("Input and target cannot be null");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        if (steps < 1) {
            throw new IllegalArgumentException("Steps must be at least 1");
        }

        List<Float> losses = new ArrayList<>();
        List<Float> distillLosses = new ArrayList<>();
        List<Float> hardLosses = new ArrayList<>();

        try (SoftTargets softTargets = softTargets(input)) {
            for (int step = 0; step < steps; step++) {
                float[] stepLosses = step(input, target, softTargets, learningRate);
                distillLosses.add(stepLosses[0]);
                hardLosses.add(stepLosses[1]);
                losses.add(stepLosses[2]);
            }
        }

        return new DistillationResult(student, losses, distillLosses, hardLosses);
    }

    /** {@code epochs} passes over the input batches, one step per batch; one mean loss entry per epoch. */
    public DistillationResult trainWithBatch(List<Tensor> inputs, List<Tensor> targets, 
                                             float learningRate, int epochs) {
        ensureOpen();
        
        if (inputs == null || inputs.isEmpty() || targets == null || targets.isEmpty()) {
            throw new IllegalArgumentException("Inputs and targets cannot be null or empty");
        }
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size, got " + inputs.size()
                + " and " + targets.size());
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        if (epochs < 1) {
            throw new IllegalArgumentException("Epochs must be at least 1");
        }

        List<Float> losses = new ArrayList<>();
        List<Float> distillLosses = new ArrayList<>();
        List<Float> hardLosses = new ArrayList<>();

        List<SoftTargets> softTargets = new ArrayList<>();
        try {
            for (Tensor input : inputs) {
                softTargets.add(softTargets(input));
            }

            for (int epoch = 0; epoch < epochs; epoch++) {
                float epochTotalLoss = 0.0f;
                float epochDistillLoss = 0.0f;
                float epochHardLoss = 0.0f;

                for (int i = 0; i < inputs.size(); i++) {
                    float[] stepLosses = step(inputs.get(i), targets.get(i), softTargets.get(i), learningRate);
                    epochDistillLoss += stepLosses[0];
                    epochHardLoss += stepLosses[1];
                    epochTotalLoss += stepLosses[2];
                }

                int batchSize = inputs.size();
                losses.add(epochTotalLoss / batchSize);
                distillLosses.add(epochDistillLoss / batchSize);
                hardLosses.add(epochHardLoss / batchSize);
            }
        } finally {
            for (SoftTargets targetsOfBatch : softTargets) {
                targetsOfBatch.close();
            }
        }

        return new DistillationResult(student, losses, distillLosses, hardLosses);
    }

    /**
     * The teacher's soft targets for {@code input}: {@code softmax(teacherLogits / T)} over the last dimension, in
     * evaluation mode and without recording, so they are a constant that does not require grad. Auxiliary loss terms
     * the teacher's forward pass leaves pending are dropped.
     */
    private SoftTargets softTargets(Tensor input) {
        teacher.eval();
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            Tensor output = teacher.forward(input);
            try {
                teacher.clearAuxiliaryLoss();
                return new SoftTargets(output.shape(), withTemperature(output, false));
            } finally {
                if (output != input) {
                    output.close();
                }
            }
        }
    }

    /**
     * One student step: forward, distillation and hard-target losses, backward of the total plus the auxiliary loss
     * terms the student's forward pass left pending, SGD update. Returns the distillation, hard-target and total loss
     * values (without auxiliary terms). The step runs inside a {@link ComputationGraph.StepScope}; its
     * results are released whether or not it completes: the graph they were recorded in, and any other node the step
     * recorded, is released before they are closed ({@link Trainer.Temporaries}); the student's own tensors are
     * spared.
     *
     * @throws IllegalArgumentException when the student's output shape differs from the teacher's
     */
    private float[] step(Tensor input, Tensor target, SoftTargets softTargets, float learningRate) {
        student.train();
        student.zeroGrad();

        float distillLoss;
        float hardLoss;
        try (ComputationGraph.StepScope scope = ComputationGraph.current().step();
             Trainer.Temporaries temps = new Trainer.Temporaries(student)) {
            Tensor studentOutput = MAML.TaskLearner.keepResult(temps, student.forward(input), input, target);
            Tensor auxiliary = temps.keep(student.takeAuxiliaryLoss());
            softTargets.requireShapeOf(studentOutput);
            Tensor distill = temps.keep(distillation(studentOutput, softTargets.probabilities));
            Tensor hard = MAML.TaskLearner.keepResult(temps, hardTargetLoss.compute(studentOutput, target), input,
                target);
            distillLoss = MAML.TaskLearner.scalarValue(distill);
            hardLoss = MAML.TaskLearner.scalarValue(hard);
            backwardTotal(distill, hard, auxiliary);
        }

        applyGradients(learningRate);
        return new float[]{distillLoss, hardLoss, alpha * distillLoss + (1 - alpha) * hardLoss};
    }

    /**
     * Backpropagates {@code alpha * distill + (1 - alpha) * hard}, plus the student's pending auxiliary loss terms
     * {@code auxiliary} when not null, built with autograd ops, when it requires grad.
     */
    private void backwardTotal(Tensor distill, Tensor hard, Tensor auxiliary) {
        List<Tensor> made = new ArrayList<>();
        try {
            Tensor weightedDistill = keep(made, TensorOps.mul(distill, alpha));
            Tensor weightedHard = keep(made, TensorOps.mul(hard, 1 - alpha));
            Tensor total = keep(made, TensorOps.add(weightedDistill, weightedHard));
            Tensor objective = MAML.TaskLearner.withAuxiliary(total, auxiliary, t -> keep(made, t));
            if (objective.requiresGrad()) {
                objective.backward();
            }
        } finally {
            closeReversed(made);
        }
    }

    /** Adds {@code t} to {@code made} and returns it. */
    private static Tensor keep(List<Tensor> made, Tensor t) {
        made.add(t);
        return t;
    }

    /** Closes {@code made} from the last tensor to the first. */
    private static void closeReversed(List<Tensor> made) {
        for (int i = made.size() - 1; i >= 0; i--) {
            made.get(i).close();
        }
    }

    /**
     * {@code T^2 * kl_div(log_softmax(studentLogits / T), teacherProbs, reduction='batchmean')} as a rank-0 tensor,
     * recorded when {@code studentLogits} requires grad; {@code teacherProbs} are the teacher's soft targets for the
     * same batch, in the batch shape of {@link #withTemperature}.
     */
    private Tensor distillation(Tensor studentLogits, Tensor teacherProbs) {
        try (Tensor studentLogProbs = withTemperature(studentLogits, true);
             Tensor divergence = BATCH_MEAN_KL.compute(studentLogProbs, teacherProbs)) {
            return TensorOps.mul(divergence, temperature * temperature);
        }
    }

    /**
     * {@code log_softmax(logits / T)} when {@code log}, otherwise {@code softmax(logits / T)}, along the last
     * dimension, as a new tensor recorded when {@code logits} requires grad. A logits tensor of rank 0 or 1 is one
     * sample: the result has shape {@code [1, n]}, so the {@code batchmean} divisor of {@link #distillation} is 1;
     * otherwise it has the logits' shape. The division is skipped at {@code T = 1}, where it is exact.
     */
    private Tensor withTemperature(Tensor logits, boolean log) {
        List<Tensor> made = new ArrayList<>();
        try {
            Tensor batch = logits.shape().rank() >= 2 ? logits
                : keep(made, logits.reshape(Shape.of(1, Math.toIntExact(logits.numElements()))));
            Tensor scaled = batch;
            if (temperature != 1.0f) {
                Tensor divisor = keep(made, MAML.TaskLearner.scalarOn(batch.backend(), temperature));
                scaled = keep(made, TensorOps.div(batch, divisor));
            }
            int lastDim = scaled.shape().rank() - 1;
            return log ? TensorOps.logSoftmax(scaled, lastDim) : TensorOps.softmax(scaled, lastDim);
        } finally {
            closeReversed(made);
        }
    }

    /**
     * {@code w -= learningRate * g} in place for each student parameter with a gradient, each parameter once, without
     * recording (an optimizer step); the gradients are then closed and cleared.
     */
    private void applyGradients(float learningRate) {
        Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Parameter> unique = new ArrayList<>();
        for (Parameter param : student.parameters()) {
            if (seen.add(param)) {
                unique.add(param);
            }
        }
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (Parameter param : unique) {
                Tensor grad = param.grad();
                if (grad != null) {
                    TensorOps.addInPlace(param.tensor(), grad, -learningRate);
                }
            }
        } finally {
            MAML.TaskLearner.releaseGradients(unique);
        }
    }

    /**
     * The student's hard-target loss on one batch, in evaluation mode and without recording; auxiliary loss terms the
     * forward pass leaves pending are dropped ({@link Module#clearAuxiliaryLoss()}).
     */
    public float evaluate(Tensor input, Tensor target) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");

        student.eval();
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            Tensor output = student.forward(input);
            try {
                student.clearAuxiliaryLoss();
                Tensor loss = hardTargetLoss.compute(output, target);
                try {
                    return MAML.TaskLearner.scalarValue(loss);
                } finally {
                    if (loss != output && loss != target) {
                        loss.close();
                    }
                }
            } finally {
                if (output != input) {
                    output.close();
                }
            }
        }
    }

    public Module teacher() {
        ensureOpen();
        return teacher;
    }

    public Module student() {
        ensureOpen();
        return student;
    }

    public float temperature() {
        return temperature;
    }

    public float alpha() {
        return alpha;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KnowledgeDistiller has been closed");
        }
    }

    /** Marks this distiller closed; it holds no tensors between calls and owns neither model. */
    @Override
    public void close() {
        closed.set(true);
    }

    @Override
    public String toString() {
        return "KnowledgeDistiller{" +
            "temperature=" + temperature +
            ", alpha=" + alpha +
            ", closed=" + closed.get() +
            '}';
    }

    /**
     * The teacher's soft targets for one input batch: the probabilities {@code softmax(teacherLogits / T)} in the
     * batch shape of {@code withTemperature} (never requiring grad), and the shape of the teacher output they were
     * computed from, which the student's output must match.
     */
    private static final class SoftTargets implements AutoCloseable {
        private final Shape teacherOutputShape;
        private final Tensor probabilities;

        SoftTargets(Shape teacherOutputShape, Tensor probabilities) {
            this.teacherOutputShape = teacherOutputShape;
            this.probabilities = probabilities;
        }

        /** @throws IllegalArgumentException when {@code studentOutput}'s shape differs from the teacher output's */
        void requireShapeOf(Tensor studentOutput) {
            if (!teacherOutputShape.equals(studentOutput.shape())) {
                throw new IllegalArgumentException("Student output shape " + studentOutput.shape()
                    + " differs from the teacher output shape " + teacherOutputShape
                    + "; distillation pairs the two distributions element by element");
            }
        }

        @Override
        public void close() {
            probabilities.close();
        }
    }

    public static final class DistillationResult {
        private final Module student;
        private final List<Float> totalLosses;
        private final List<Float> distillationLosses;
        private final List<Float> hardTargetLosses;

        public DistillationResult(Module student, List<Float> totalLosses,
                                  List<Float> distillationLosses, List<Float> hardTargetLosses) {
            this.student = student;
            this.totalLosses = new ArrayList<>(totalLosses);
            this.distillationLosses = new ArrayList<>(distillationLosses);
            this.hardTargetLosses = new ArrayList<>(hardTargetLosses);
        }

        public Module student() {
            return student;
        }

        public List<Float> totalLosses() {
            return new ArrayList<>(totalLosses);
        }

        public List<Float> distillationLosses() {
            return new ArrayList<>(distillationLosses);
        }

        public List<Float> hardTargetLosses() {
            return new ArrayList<>(hardTargetLosses);
        }

        public float finalTotalLoss() {
            if (totalLosses.isEmpty()) {
                return Float.NaN;
            }
            return totalLosses.get(totalLosses.size() - 1);
        }

        public float finalDistillationLoss() {
            if (distillationLosses.isEmpty()) {
                return Float.NaN;
            }
            return distillationLosses.get(distillationLosses.size() - 1);
        }

        public float finalHardTargetLoss() {
            if (hardTargetLosses.isEmpty()) {
                return Float.NaN;
            }
            return hardTargetLosses.get(hardTargetLosses.size() - 1);
        }

        @Override
        public String toString() {
            return "DistillationResult{" +
                "iterations=" + totalLosses.size() +
                ", finalTotalLoss=" + finalTotalLoss() +
                ", finalDistillationLoss=" + finalDistillationLoss() +
                ", finalHardTargetLoss=" + finalHardTargetLoss() +
                '}';
        }
    }
}

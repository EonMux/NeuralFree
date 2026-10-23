package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.Autograd;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Trainer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.UnaryOperator;

/**
 * Model-Agnostic Meta-Learning on one model whose weights stay on its backend, first order (FOMAML) or second order.
 *
 * <p>For each task of a meta-iteration the weights are adapted by {@code innerSteps} SGD steps on the mean
 * support-set loss ({@code w' = w - innerLearningRate * g}); the mean query-set loss and its gradient are taken at the
 * adapted weights. After all tasks the meta update is
 * {@code w -= outerLearningRate * (sum of task query gradients / taskCount)}, applied in place without recording.
 * Gradients come from autograd: a parameter that receives no gradient contributes nothing. The differentiated loss of
 * each pair includes the auxiliary loss terms the model's forward pass leaves pending
 * ({@link Module#takeAuxiliaryLoss()}, such as a mixture of experts' load-balancing term); the reported loss values
 * are those of the loss function alone. Weights and gradients never leave the device; loss values are read back as
 * scalars for the result. Every meta-iteration runs inside one {@link ComputationGraph.StepScope},
 * so it leaves no autograd node behind.</p>
 *
 * <p><b>First order</b> ({@code firstOrder = true}). The weights are adapted in place, one backward pass per support
 * pair, and restored from a device snapshot after the task; the query gradient is taken at the adapted weights and
 * treated as the gradient with respect to the original weights. See {@link TaskLearner} for the per-task steps.</p>
 *
 * <p><b>Second order</b> ({@code firstOrder = false}). The adaptation is differentiated through, as PyTorch's MAML
 * with {@code create_graph=True}: inner step {@code k} runs the support forward passes with the trainable parameters
 * substituted by the adapted tensors {@code theta(k-1)} ({@link Module#substitute}; the parameter leaves themselves for
 * {@code k = 1}), takes {@code g = Autograd.grad(loss, theta(k-1), null, true, true)} and records
 * {@code theta(k) = theta(k-1) - innerLearningRate * g}; the query loss runs with {@code theta(K)} substituted, and its
 * backward pass reaches the parameter leaves through the recorded inner updates, Hessian-vector terms included. The
 * model's weights are never written during a task. Frozen parameters are not adapted. Every operation of the model and
 * the loss must support create-graph ({@code GradientFunction.supportsCreateGraph()}); otherwise the inner gradient
 * throws {@link UnsupportedOperationException} naming the operation. A module that substitutes its own parameter
 * during the forward pass, such as spectral normalization, nests its scope inside MAML's
 * ({@link Parameter#substitute}): it reads the adapted tensor and its transformation of it is differentiated through
 * like any other operation.</p>
 *
 * <p>The meta update and the inner updates of the first-order path write parameters in place and run under
 * {@link ComputationGraph#noGrad()}, after every substitution scope is closed.</p>
 */
public final class MAML implements AutoCloseable {
    private final Module model;
    private final float innerLearningRate;
    private final float outerLearningRate;
    private final int innerSteps;
    private final boolean firstOrder;
    private final LossFunction lossFunction;
    private final TaskLearner learner;
    private final AtomicBoolean closed;

    public MAML(Module model, float innerLearningRate, float outerLearningRate,
                int innerSteps, LossFunction lossFunction) {
        this(model, innerLearningRate, outerLearningRate, innerSteps, lossFunction, true);
    }

    /** MAML with first-order ({@code firstOrder = true}) or second-order adaptation (see the class description). */
    public MAML(Module model, float innerLearningRate, float outerLearningRate,
                int innerSteps, LossFunction lossFunction, boolean firstOrder) {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(lossFunction, "LossFunction cannot be null");

        if (innerLearningRate <= 0) {
            throw new IllegalArgumentException("Inner learning rate must be positive");
        }
        if (outerLearningRate <= 0) {
            throw new IllegalArgumentException("Outer learning rate must be positive");
        }
        if (innerSteps < 1) {
            throw new IllegalArgumentException("Inner steps must be at least 1");
        }

        this.model = model;
        this.innerLearningRate = innerLearningRate;
        this.outerLearningRate = outerLearningRate;
        this.innerSteps = innerSteps;
        this.firstOrder = firstOrder;
        this.lossFunction = lossFunction;
        this.learner = new TaskLearner(model, lossFunction, innerSteps, innerLearningRate);
        this.closed = new AtomicBoolean(false);
    }

    /**
     * Runs {@code metaIterations} meta-updates over all tasks. Each iteration records the mean query loss at the
     * adapted weights over the tasks, and the query loss of the first task at the updated weights as the validation
     * loss. On failure the weights are restored to their values at the start of the failing iteration.
     */
    public MAMLResult metaTrain(List<Task> tasks, int metaIterations) {
        ensureOpen();

        if (tasks == null || tasks.isEmpty()) {
            throw new IllegalArgumentException("Tasks list cannot be null or empty");
        }
        if (metaIterations < 1) {
            throw new IllegalArgumentException("Meta iterations must be at least 1");
        }

        List<Float> metaLosses = new ArrayList<>();
        List<Float> validationLosses = new ArrayList<>();

        for (int iter = 0; iter < metaIterations; iter++) {
            model.train();
            float totalMetaLoss;
            try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
                totalMetaLoss = metaStep(tasks);
            }
            metaLosses.add(totalMetaLoss / tasks.size());
            validationLosses.add(learner.evaluate(tasks.get(0)));
        }

        return new MAMLResult(model, metaLosses, validationLosses);
    }

    /**
     * One meta-update over {@code tasks}: the task query gradients are summed per parameter and their mean applied
     * with the outer learning rate. Returns the sum of the task query losses.
     */
    private float metaStep(List<Task> tasks) {
        float totalMetaLoss = 0.0f;
        Map<String, Tensor> start = TaskLearner.snapshot(model);
        Map<String, Tensor> metaGradients = new LinkedHashMap<>();
        try {
            for (Task task : tasks) {
                if (firstOrder) {
                    learner.adaptInPlace(task);
                    totalMetaLoss += learner.queryLossAndGradients(task, metaGradients);
                    TaskLearner.restore(model, start);
                } else {
                    totalMetaLoss += learner.secondOrderQueryLossAndGradients(task, metaGradients);
                }
            }
            TaskLearner.applyMean(model, metaGradients, tasks.size(), outerLearningRate);
        } catch (RuntimeException e) {
            TaskLearner.restoreAfterFailure(model, start, e);
            throw e;
        } finally {
            TaskLearner.closeAll(start, metaGradients);
        }
        return totalMetaLoss;
    }

    /**
     * The model's weights after {@code innerSteps} adaptation steps on the task's support set, by parameter name, as
     * new device tensors the caller owns that do not require grad; the model's own weights are left unchanged. The
     * steps are the in-place SGD steps of the first-order path in both modes (the adapted values of the second-order
     * path are the same SGD iterates); they run inside a {@link ComputationGraph.StepScope}.
     */
    public Map<String, Tensor> adapt(Task task) {
        ensureOpen();
        Objects.requireNonNull(task, "Task cannot be null");
        Map<String, Tensor> start = TaskLearner.snapshot(model);
        Map<String, Tensor> adapted = new LinkedHashMap<>();
        try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
            learner.adaptInPlace(task);
            adapted.putAll(TaskLearner.snapshot(model));
            TaskLearner.restore(model, start);
            return adapted;
        } catch (RuntimeException e) {
            TaskLearner.restoreAfterFailure(model, start, e);
            TaskLearner.closeAll(adapted);
            throw e;
        } finally {
            TaskLearner.closeAll(start);
        }
    }

    /**
     * The mean query loss of the task at the current weights, without recording autograd nodes (0 without query
     * samples).
     */
    public float evaluate(Task task) {
        ensureOpen();
        Objects.requireNonNull(task, "Task cannot be null");
        return learner.evaluate(task);
    }

    public Module model() {
        ensureOpen();
        return model;
    }

    public float innerLearningRate() {
        return innerLearningRate;
    }

    public float outerLearningRate() {
        return outerLearningRate;
    }

    public int innerSteps() {
        return innerSteps;
    }

    public boolean isFirstOrder() {
        return firstOrder;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("MAML has been closed");
        }
    }

    /** Marks this learner closed; it holds no tensors between calls and does not own the model. */
    @Override
    public void close() {
        closed.set(true);
    }

    @Override
    public String toString() {
        return "MAML{" +
            "innerLearningRate=" + innerLearningRate +
            ", outerLearningRate=" + outerLearningRate +
            ", innerSteps=" + innerSteps +
            ", firstOrder=" + firstOrder +
            ", closed=" + closed.get() +
            '}';
    }

    public static final class Task {
        private final List<Tensor> supportInputs;
        private final List<Tensor> supportTargets;
        private final List<Tensor> queryInputs;
        private final List<Tensor> queryTargets;

        public Task(List<Tensor> supportInputs, List<Tensor> supportTargets,
                   List<Tensor> queryInputs, List<Tensor> queryTargets) {
            Objects.requireNonNull(supportInputs, "Support inputs cannot be null");
            Objects.requireNonNull(supportTargets, "Support targets cannot be null");
            Objects.requireNonNull(queryInputs, "Query inputs cannot be null");
            Objects.requireNonNull(queryTargets, "Query targets cannot be null");

            this.supportInputs = new ArrayList<>(supportInputs);
            this.supportTargets = new ArrayList<>(supportTargets);
            this.queryInputs = new ArrayList<>(queryInputs);
            this.queryTargets = new ArrayList<>(queryTargets);
        }

        public List<Tensor> supportInputs() {
            return new ArrayList<>(supportInputs);
        }

        public List<Tensor> supportTargets() {
            return new ArrayList<>(supportTargets);
        }

        public List<Tensor> queryInputs() {
            return new ArrayList<>(queryInputs);
        }

        public List<Tensor> queryTargets() {
            return new ArrayList<>(queryTargets);
        }

        public int supportSize() {
            return supportInputs.size();
        }

        public int querySize() {
            return queryInputs.size();
        }

        @Override
        public String toString() {
            return "Task{supportSize=" + supportSize() + ", querySize=" + querySize() + '}';
        }
    }

    public static final class MAMLResult {
        private final Module model;
        private final List<Float> metaLosses;
        private final List<Float> validationLosses;

        public MAMLResult(Module model, List<Float> metaLosses, List<Float> validationLosses) {
            this.model = model;
            this.metaLosses = new ArrayList<>(metaLosses);
            this.validationLosses = new ArrayList<>(validationLosses);
        }

        public Module model() {
            return model;
        }

        public List<Float> metaLosses() {
            return new ArrayList<>(metaLosses);
        }

        public List<Float> validationLosses() {
            return new ArrayList<>(validationLosses);
        }

        public float finalMetaLoss() {
            if (metaLosses.isEmpty()) {
                return Float.NaN;
            }
            return metaLosses.get(metaLosses.size() - 1);
        }

        public float finalValidationLoss() {
            if (validationLosses.isEmpty()) {
                return Float.NaN;
            }
            return validationLosses.get(validationLosses.size() - 1);
        }

        @Override
        public String toString() {
            return "MAMLResult{" +
                "iterations=" + metaLosses.size() +
                ", finalMetaLoss=" + finalMetaLoss() +
                ", finalValidationLoss=" + finalValidationLoss() +
                '}';
        }
    }

    /**
     * The per-task steps of MAML on one model (first order, shared with {@link FederatedMetaLearner}, and second
     * order), and the device weight snapshots, restores and averaged updates both use. Parameters are addressed by
     * name, each {@link Parameter} once even when it is registered under several names. Sums and means are FLOAT32
     * device tensors built with {@link TensorOps}; only loss values are read back. The in-place
     * weight updates run under {@link ComputationGraph#noGrad()} and must not run inside a substitution scope.
     */
    static final class TaskLearner {
        private final Module model;
        private final LossFunction lossFunction;
        private final int innerSteps;
        private final float innerLearningRate;

        TaskLearner(Module model, LossFunction lossFunction, int innerSteps, float innerLearningRate) {
            this.model = model;
            this.lossFunction = lossFunction;
            this.innerSteps = innerSteps;
            this.innerLearningRate = innerLearningRate;
        }

        /**
         * {@code innerSteps} SGD steps in place, each on the mean gradient over the task's support pairs:
         * {@code w -= innerLearningRate * (sum / n)}. A task without support pairs leaves the weights unchanged.
         */
        void adaptInPlace(Task task) {
            List<Tensor> inputs = task.supportInputs();
            List<Tensor> targets = task.supportTargets();
            int n = Math.min(inputs.size(), targets.size());
            if (n == 0) {
                return;
            }
            for (int step = 0; step < innerSteps; step++) {
                Map<String, Tensor> sums = new LinkedHashMap<>();
                try {
                    accumulate(inputs, targets, n, sums);
                    applyMean(model, sums, n, innerLearningRate);
                } finally {
                    closeAll(sums);
                }
            }
        }

        /**
         * The mean loss over the task's query pairs at the current weights; adds each parameter's mean query gradient
         * into {@code metaGradients} (by name, starting from zeros). Returns 0 and adds nothing without query pairs.
         */
        float queryLossAndGradients(Task task, Map<String, Tensor> metaGradients) {
            List<Tensor> inputs = task.queryInputs();
            List<Tensor> targets = task.queryTargets();
            int n = Math.min(inputs.size(), targets.size());
            if (n == 0) {
                return 0.0f;
            }
            Map<String, Tensor> sums = new LinkedHashMap<>();
            try {
                float total = accumulate(inputs, targets, n, sums);
                Map<String, Parameter> params = parameters(model);
                for (Map.Entry<String, Tensor> entry : sums.entrySet()) {
                    Tensor mean = mean(entry.getValue(), n);
                    try {
                        addInto(metaGradients, entry.getKey(), params.get(entry.getKey()), mean);
                    } finally {
                        if (mean != entry.getValue()) {
                            mean.close();
                        }
                    }
                }
                return total / n;
            } finally {
                closeAll(sums);
            }
        }

        /**
         * Second-order MAML for one task (see {@link MAML}): the mean query loss at the recorded adapted weights
         * {@code theta(K)}, backpropagated to the parameter leaves through the inner updates; each parameter's
         * gradient is added into {@code metaGradients} (by name, starting from zeros) and then released. Returns the
         * query loss value, or 0 and adds nothing without query pairs. The model's weights are not written. The task
         * runs inside a {@link ComputationGraph.StepScope}; the tensors it computes are closed after the backward
         * pass, also on failure.
         */
        float secondOrderQueryLossAndGradients(Task task, Map<String, Tensor> metaGradients) {
            List<Tensor> inputs = task.queryInputs();
            List<Tensor> targets = task.queryTargets();
            int n = Math.min(inputs.size(), targets.size());
            if (n == 0) {
                return 0.0f;
            }
            Map<String, Parameter> params = parameters(model);
            model.zeroGrad();
            try (ComputationGraph.StepScope scope = ComputationGraph.current().step();
                 TaskTensors temps = new TaskTensors(model, task)) {
                Map<Parameter, Tensor> adapted = adaptRecorded(task, trainable(params.values()), temps);
                MeanLoss loss = meanLoss(inputs, targets, n, substitutes(adapted), temps);
                float value = scalarValue(loss.value);
                if (loss.objective.requiresGrad()) {
                    loss.objective.backward();
                }
                for (Map.Entry<String, Parameter> entry : params.entrySet()) {
                    Tensor grad = entry.getValue().grad();
                    if (grad != null) {
                        addInto(metaGradients, entry.getKey(), entry.getValue(), grad);
                    }
                }
                return value;
            } finally {
                releaseGradients(params.values());
            }
        }

        /**
         * The recorded inner loop of second-order MAML: starting from the parameter leaves {@code theta(0)}, step
         * {@code k} takes the mean support loss with {@code theta(k-1)} substituted, its gradients
         * {@code g = Autograd.grad(loss, theta(k-1), null, true, true)} and the recorded update
         * {@code theta(k) = theta(k-1) - innerLearningRate * g} (unchanged where {@code g} is null). Returns
         * {@code theta(innerSteps)} per trainable parameter. Without support pairs, without trainable parameters, or
         * when the support loss does not require grad (grad mode off), the leaves are returned unchanged.
         */
        private Map<Parameter, Tensor> adaptRecorded(Task task, List<Parameter> trainable, TaskTensors temps) {
            Map<Parameter, Tensor> theta = new LinkedHashMap<>();
            for (Parameter param : trainable) {
                theta.put(param, param.tensor());
            }
            List<Tensor> inputs = task.supportInputs();
            List<Tensor> targets = task.supportTargets();
            int n = Math.min(inputs.size(), targets.size());
            if (n == 0 || trainable.isEmpty()) {
                return theta;
            }
            for (int step = 0; step < innerSteps; step++) {
                Tensor loss = meanLoss(inputs, targets, n, substitutes(theta), temps).objective;
                if (!loss.requiresGrad()) {
                    return theta;
                }
                List<Tensor> current = new ArrayList<>(theta.values());
                List<Tensor> grads = Autograd.grad(loss, current, null, true, true);
                for (Tensor g : grads) {
                    temps.keep(g);
                }
                Map<Parameter, Tensor> next = new LinkedHashMap<>();
                int i = 0;
                for (Parameter param : theta.keySet()) {
                    Tensor previous = current.get(i);
                    Tensor g = grads.get(i++);
                    if (g == null) {
                        next.put(param, previous);
                    } else {
                        Tensor update = temps.keep(TensorOps.mul(g, innerLearningRate));
                        next.put(param, temps.keep(TensorOps.sub(previous, update)));
                    }
                }
                theta = next;
            }
            return theta;
        }

        /**
         * Over the first {@code n} pairs, computed with {@code substitutes} in place of their parameters
         * ({@link Module#substitute}; none when empty): the objective, the mean of each pair's loss summed over its
         * elements (a scalar loss as it is) plus the auxiliary loss terms that pair's forward pass left pending
         * ({@link Module#takeAuxiliaryLoss()}), recorded in the caller's grad mode; and the mean of the losses alone,
         * for the reported value (computed without recording when auxiliary terms were added, else the objective
         * itself). Every tensor computed is kept in {@code temps}.
         */
        private MeanLoss meanLoss(List<Tensor> inputs, List<Tensor> targets, int n,
                                  Map<Parameter, Tensor> substitutes, TaskTensors temps) {
            Tensor total = null;
            List<Tensor> lossTerms = new ArrayList<>(n);
            boolean auxiliaryAdded = false;
            try (Parameter.Substitution scope = substitutes.isEmpty() ? null : model.substitute(substitutes)) {
                for (int i = 0; i < n; i++) {
                    Tensor output = temps.keep(model.forward(inputs.get(i)));
                    Tensor loss = temps.keep(lossFunction.compute(output, targets.get(i)));
                    Tensor auxiliary = temps.keep(model.takeAuxiliaryLoss());
                    if (loss.numElements() == 0) {
                        throw new IllegalArgumentException("Loss tensor of shape " + loss.shape() + " has no elements");
                    }
                    Tensor lossTerm = loss.numElements() == 1 ? loss : temps.keep(TensorOps.sum(loss));
                    Tensor term = withAuxiliary(lossTerm, auxiliary, temps::keep);
                    auxiliaryAdded |= auxiliary != null;
                    lossTerms.add(lossTerm);
                    total = total == null ? term : temps.keep(TensorOps.add(total, term));
                }
            }
            Tensor objective = meanOf(total, n, temps);
            if (!auxiliaryAdded) {
                return new MeanLoss(objective, objective);
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor lossTotal = lossTerms.get(0);
                for (int i = 1; i < lossTerms.size(); i++) {
                    lossTotal = temps.keep(TensorOps.add(lossTotal, lossTerms.get(i)));
                }
                return new MeanLoss(objective, meanOf(lossTotal, n, temps));
            }
        }

        /** {@code total / n} divided by a device scalar ({@code total} itself when {@code n} is 1), kept in temps. */
        private static Tensor meanOf(Tensor total, int n, TaskTensors temps) {
            if (n == 1) {
                return total;
            }
            Tensor divisor = temps.keep(scalarOn(total.backend(), n));
            return temps.keep(TensorOps.div(total, divisor));
        }

        /** The objective of {@link #meanLoss} and the tensor holding its reported value. */
        private static final class MeanLoss {
            final Tensor objective;
            final Tensor value;

            MeanLoss(Tensor objective, Tensor value) {
                this.objective = objective;
                this.value = value;
            }
        }

        /** The entries of {@code theta} whose tensor is not the parameter's own leaf. */
        private static Map<Parameter, Tensor> substitutes(Map<Parameter, Tensor> theta) {
            Map<Parameter, Tensor> result = new LinkedHashMap<>();
            for (Map.Entry<Parameter, Tensor> entry : theta.entrySet()) {
                if (entry.getValue() != entry.getKey().tensor()) {
                    result.put(entry.getKey(), entry.getValue());
                }
            }
            return result;
        }

        /** The trainable parameters among {@code params} ({@link Parameter#requiresGrad()}), in order. */
        private static List<Parameter> trainable(Collection<Parameter> params) {
            List<Parameter> result = new ArrayList<>();
            for (Parameter param : params) {
                if (param.requiresGrad()) {
                    result.add(param);
                }
            }
            return result;
        }

        /** The mean loss over the task's query pairs at the current weights, without recording (0 without pairs). */
        float evaluate(Task task) {
            List<Tensor> inputs = task.queryInputs();
            List<Tensor> targets = task.queryTargets();
            int n = Math.min(inputs.size(), targets.size());
            if (n == 0) {
                return 0.0f;
            }
            float total = 0.0f;
            try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
                for (int i = 0; i < n; i++) {
                    total += lossValue(inputs.get(i), targets.get(i), false);
                }
            }
            return total / n;
        }

        /**
         * The sum of the loss values over the first {@code n} pairs. Gradients the model holds on entry are cleared
         * and closed. Each pair then runs forward, loss and backward, every
         * non-null parameter gradient is added into {@code sums}, and the gradients the backward produced are closed
         * and cleared ({@link #releaseGradients}), also when a pair fails.
         */
        private float accumulate(List<Tensor> inputs, List<Tensor> targets, int n, Map<String, Tensor> sums) {
            Map<String, Parameter> params = parameters(model);
            float total = 0.0f;
            model.zeroGrad();
            try {
                for (int i = 0; i < n; i++) {
                    total += lossValue(inputs.get(i), targets.get(i), true);
                    for (Map.Entry<String, Parameter> entry : params.entrySet()) {
                        Tensor grad = entry.getValue().grad();
                        if (grad != null) {
                            addInto(sums, entry.getKey(), entry.getValue(), grad);
                        }
                    }
                    releaseGradients(params.values());
                }
            } catch (RuntimeException e) {
                try {
                    releaseGradients(params.values());
                } catch (RuntimeException releaseFailure) {
                    e.addSuppressed(releaseFailure);
                }
                throw e;
            }
            return total;
        }

        /** {@code result}, kept in {@code temps} for closing unless it is the caller's input or target. */
        static Tensor keepResult(Trainer.Temporaries temps, Tensor result, Tensor input, Tensor target) {
            return result == input || result == target ? result : temps.keep(result);
        }

        /**
         * Clears the gradients of {@code params}, which closes them ({@link Parameter#zeroGrad()}). Every parameter is
         * tried; the first failure is thrown with the later ones suppressed.
         */
        static void releaseGradients(Collection<Parameter> params) {
            RuntimeException failure = null;
            for (Parameter param : params) {
                try {
                    param.zeroGrad();
                } catch (RuntimeException e) {
                    if (failure == null) {
                        failure = e;
                    } else {
                        failure.addSuppressed(e);
                    }
                }
            }
            if (failure != null) {
                throw failure;
            }
        }

        /**
         * The loss value of one pair, read back (the loss alone, without auxiliary terms); with {@code backward}, the
         * loss plus the auxiliary loss terms the forward pass left pending ({@link Module#takeAuxiliaryLoss()}) is
         * backpropagated when it requires grad. Without {@code backward} the pending terms are discarded. The output,
         * the loss and the auxiliary terms are closed after the nodes recorded for the pair are released, whether or
         * not the computation completes ({@link Trainer.Temporaries}); the model's own tensors are spared.
         */
        private float lossValue(Tensor input, Tensor target, boolean backward) {
            try (Trainer.Temporaries temps = new Trainer.Temporaries(model)) {
                Tensor output = keepResult(temps, model.forward(input), input, target);
                Tensor loss = keepResult(temps, lossFunction.compute(output, target), input,
                    target);
                Tensor auxiliary = temps.keep(model.takeAuxiliaryLoss());
                float value = scalarValue(loss);
                if (backward) {
                    Tensor objective = withAuxiliary(loss, auxiliary, temps::keep);
                    if (objective.requiresGrad()) {
                        objective.backward();
                    }
                }
                return value;
            }
        }

        /**
         * {@code loss} plus {@code auxiliary} ({@link TensorOps#add}, recorded in grad mode), the loss summed first
         * when it has several elements (a backward pass seeded with ones differentiates that sum); {@code loss} itself
         * when {@code auxiliary} is null. Every tensor made here is handed to {@code keep}.
         */
        static Tensor withAuxiliary(Tensor loss, Tensor auxiliary, UnaryOperator<Tensor> keep) {
            if (auxiliary == null) {
                return loss;
            }
            Tensor total = loss.numElements() == 1 ? loss : keep.apply(TensorOps.sum(loss));
            return keep.apply(TensorOps.add(total, auxiliary));
        }

        /** The model's parameters by name in registration order, each {@link Parameter} once (first name kept). */
        static Map<String, Parameter> parameters(Module model) {
            Map<String, Parameter> result = new LinkedHashMap<>();
            Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<>());
            for (Map.Entry<String, Parameter> entry : model.namedParameters().entrySet()) {
                if (seen.add(entry.getValue())) {
                    result.put(entry.getKey(), entry.getValue());
                }
            }
            return result;
        }

        /** New copies of the model's weights by name ({@link Parameter#copyTensor()}); the caller closes them. */
        static Map<String, Tensor> snapshot(Module model) {
            Map<String, Tensor> copies = new LinkedHashMap<>();
            try {
                for (Map.Entry<String, Parameter> entry : parameters(model).entrySet()) {
                    copies.put(entry.getKey(), entry.getValue().copyTensor());
                }
            } catch (RuntimeException e) {
                closeAll(copies);
                throw e;
            }
            return copies;
        }

        /** Writes each snapshot back into the parameter of its name, in place ({@link Parameter#copyFrom}). */
        static void restore(Module model, Map<String, Tensor> snapshot) {
            Map<String, Parameter> params = parameters(model);
            for (Map.Entry<String, Tensor> entry : snapshot.entrySet()) {
                Parameter param = params.get(entry.getKey());
                if (param == null) {
                    throw new IllegalStateException("Parameter '" + entry.getKey() + "' is no longer registered");
                }
                param.copyFrom(entry.getValue());
            }
        }

        /** {@link #restore} after {@code failure}; a failure of the restore is added to it as suppressed. */
        static void restoreAfterFailure(Module model, Map<String, Tensor> snapshot, RuntimeException failure) {
            try {
                restore(model, snapshot);
            } catch (RuntimeException restoreFailure) {
                failure.addSuppressed(restoreFailure);
            }
        }

        /** For each named sum, {@code w -= learningRate * (sum / count)} in place on the parameter of that name. */
        static void applyMean(Module model, Map<String, Tensor> sums, int count, float learningRate) {
            Map<String, Parameter> params = parameters(model);
            for (Map.Entry<String, Tensor> entry : sums.entrySet()) {
                applyMean(params.get(entry.getKey()), entry.getKey(), entry.getValue(), count, learningRate);
            }
        }

        /**
         * {@code w -= learningRate * (sum / count)} in place on {@code param}, registered as {@code name}, under
         * {@link ComputationGraph#noGrad()} (an optimizer step).
         */
        static void applyMean(Parameter param, String name, Tensor sum, int count, float learningRate) {
            if (param == null) {
                throw new IllegalStateException("Parameter '" + name + "' is no longer registered");
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                Tensor mean = mean(sum, count);
                try {
                    TensorOps.addInPlace(param.tensor(), mean, -learningRate);
                } finally {
                    if (mean != sum) {
                        mean.close();
                    }
                }
            }
        }

        /**
         * {@code sums[name] += value} without recording; a missing entry starts as zeros of the parameter's shape on
         * its backend.
         */
        static void addInto(Map<String, Tensor> sums, String name, Parameter param, Tensor value) {
            Tensor sum = sums.get(name);
            if (sum == null) {
                sum = Tensor.zeros(param.backend(), param.shape());
                sums.put(name, sum);
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                TensorOps.addInPlace(sum, value, 1.0f);
            }
        }

        /**
         * {@code sum / count} as a new tensor, divided by a device scalar as the host loops divided; {@code sum}
         * itself when {@code count} is 1.
         */
        static Tensor mean(Tensor sum, int count) {
            if (count == 1) {
                return sum;
            }
            try (Tensor divisor = scalarOn(sum.backend(), count)) {
                return TensorOps.div(sum, divisor);
            }
        }

        /**
         * A rank-0 FLOAT32 tensor holding {@code value} (not {@code -0.0f}) on {@code backend}: {@code 0 + value} over
         * a zero-filled scalar ({@link Tensor#zeros}, filled on the device when the train kernels are available, so no
         * host transfer is made). Never requires grad.
         */
        static Tensor scalarOn(ComputeBackend backend, float value) {
            try (Tensor zero = Tensor.zeros(backend, Shape.scalar(), DType.FLOAT32)) {
                return TensorOps.add(zero, value);
            }
        }

        /**
         * The value of a loss tensor (its first element when it holds several), read back to the host.
         *
         * @throws IllegalArgumentException when the loss has no elements
         */
        static float scalarValue(Tensor loss) {
            long n = loss.numElements();
            if (n == 1) {
                return loss.toFloatArray()[0];
            }
            if (n == 0) {
                throw new IllegalArgumentException("Loss tensor of shape " + loss.shape() + " has no elements");
            }
            try (Tensor flat = loss.reshape(Shape.of(Math.toIntExact(n)));
                 Tensor first = flat.slice(0, 0, 1)) {
                return first.toFloatArray()[0];
            }
        }

        /** Closes every tensor of the maps and clears them. */
        @SafeVarargs
        static void closeAll(Map<String, Tensor>... maps) {
            for (Map<String, Tensor> map : maps) {
                for (Tensor t : map.values()) {
                    t.close();
                }
                map.clear();
            }
        }

        /**
         * The tensors one second-order task computes (model outputs, losses, sums, inner gradients and adapted
         * weights), all closed together by {@link #close()} once the task's outer backward pass has run, never before:
         * the inner graphs must stay intact until then. The task's own inputs and targets and the model's parameter
         * tensors and buffers are never closed. Closing is idempotent, so outputs the backward pass or the enclosing
         * {@link ComputationGraph.StepScope} also close are unaffected.
         */
        private static final class TaskTensors implements AutoCloseable {
            private final Module model;
            private final Set<Tensor> callers = Collections.newSetFromMap(new IdentityHashMap<>());
            private final List<Tensor> tensors = new ArrayList<>();

            TaskTensors(Module model, Task task) {
                this.model = model;
                callers.addAll(task.supportInputs());
                callers.addAll(task.supportTargets());
                callers.addAll(task.queryInputs());
                callers.addAll(task.queryTargets());
            }

            /** Keeps {@code t} for closing (null and the task's own tensors are ignored) and returns it. */
            Tensor keep(Tensor t) {
                if (t != null && !callers.contains(t)) {
                    tensors.add(t);
                }
                return t;
            }

            /** Closes the kept tensors in reverse order; the first failure is thrown with later ones suppressed. */
            @Override
            public void close() {
                Set<Tensor> held = Collections.newSetFromMap(new IdentityHashMap<>());
                for (Parameter param : model.parameters()) {
                    held.add(param.tensor());
                }
                held.addAll(model.namedBuffers().values());
                RuntimeException failure = null;
                for (int i = tensors.size() - 1; i >= 0; i--) {
                    Tensor t = tensors.get(i);
                    if (held.contains(t)) {
                        continue;
                    }
                    try {
                        t.close();
                    } catch (RuntimeException e) {
                        if (failure == null) {
                            failure = e;
                        } else {
                            failure.addSuppressed(e);
                        }
                    }
                }
                tensors.clear();
                if (failure != null) {
                    throw failure;
                }
            }
        }
    }
}

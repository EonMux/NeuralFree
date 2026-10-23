package com.eonmux.neuralfree.meta;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Federated first-order MAML: each round a random fraction of the clients computes a meta-gradient on its own tasks
 * from the current global weights, and the global model takes one step on the mean of those meta-gradients.
 *
 * <p>Clients share the global model and its backend. A client's meta-gradient is the mean over its tasks of the
 * {@link MAML} task step: adapt in place by {@code innerSteps} SGD steps on the mean support gradient, take the mean
 * query gradient at the adapted weights, restore the global weights from a device snapshot. The round update is
 * {@code w -= metaLearningRate * (sum of client meta-gradients / clients that produced one)} per parameter. The client
 * losses are recorded and backpropagated in the caller's grad mode; the in-place adaptation steps, the gradient sums
 * and the round update run under {@link ComputationGraph#noGrad()}, and every round runs inside one
 * {@link ComputationGraph.StepScope}, so it leaves no autograd node behind. Weights and gradients stay on the device;
 * loss values are read back as scalars. Client selection draws from an unseeded
 * {@link Random}.</p>
 */
public final class FederatedMetaLearner implements AutoCloseable {
    private final Module globalModel;
    private final int numClients;
    private final int numRounds;
    private final int innerSteps;
    private final float metaLearningRate;
    private final float clientFraction;
    private final Random random;
    private final MAML.TaskLearner learner;
    private final AtomicBoolean closed;

    public FederatedMetaLearner(Module globalModel, LossFunction lossFunction,
                                int numClients, int numRounds, int innerSteps,
                                float innerLearningRate, float metaLearningRate) {
        this(globalModel, lossFunction, numClients, numRounds, innerSteps,
            innerLearningRate, metaLearningRate, 0.1f);
    }

    public FederatedMetaLearner(Module globalModel, LossFunction lossFunction,
                                int numClients, int numRounds, int innerSteps,
                                float innerLearningRate, float metaLearningRate,
                                float clientFraction) {
        Objects.requireNonNull(globalModel, "Global model cannot be null");
        Objects.requireNonNull(lossFunction, "Loss function cannot be null");
        
        if (numClients < 1) {
            throw new IllegalArgumentException("Number of clients must be at least 1");
        }
        if (numRounds < 1) {
            throw new IllegalArgumentException("Number of rounds must be at least 1");
        }
        if (innerSteps < 1) {
            throw new IllegalArgumentException("Inner steps must be at least 1");
        }
        if (innerLearningRate <= 0) {
            throw new IllegalArgumentException("Inner learning rate must be positive");
        }
        if (metaLearningRate <= 0) {
            throw new IllegalArgumentException("Meta learning rate must be positive");
        }
        if (clientFraction <= 0 || clientFraction > 1) {
            throw new IllegalArgumentException("Client fraction must be between 0 and 1");
        }

        this.globalModel = globalModel;
        this.numClients = numClients;
        this.numRounds = numRounds;
        this.innerSteps = innerSteps;
        this.metaLearningRate = metaLearningRate;
        this.clientFraction = clientFraction;
        this.random = new Random();
        this.learner = new MAML.TaskLearner(globalModel, lossFunction, innerSteps, innerLearningRate);
        this.closed = new AtomicBoolean(false);
    }

    /**
     * Runs {@code numRounds} rounds. {@code clientTasks.get(i)} holds client {@code i}'s tasks; a selected client
     * without tasks (or beyond the list) is skipped. Each round records the mean over participating clients of their
     * mean query loss at the adapted weights, and, as {@link FederatedMetaResult#roundAccuracies()}, the evaluation
     * loss of the updated global model ({@link #evaluateGlobalModel}).
     */
    public FederatedMetaResult train(List<List<MAML.Task>> clientTasks) {
        ensureOpen();
        
        if (clientTasks == null || clientTasks.isEmpty()) {
            throw new IllegalArgumentException("Client tasks cannot be null or empty");
        }

        List<Float> roundLosses = new ArrayList<>();
        List<Float> roundAccuracies = new ArrayList<>();

        for (int round = 0; round < numRounds; round++) {
            globalModel.train();
            float totalRoundLoss = 0.0f;
            int evaluatedClients = 0;
            Map<String, Tensor> global = MAML.TaskLearner.snapshot(globalModel);
            Map<String, Tensor> roundGradients = new LinkedHashMap<>();
            Map<String, Integer> gradCounts = new HashMap<>();
            try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
                for (int clientId : selectClients()) {
                    if (clientId >= clientTasks.size() || clientTasks.get(clientId).isEmpty()) {
                        continue;
                    }
                    totalRoundLoss += clientUpdate(clientTasks.get(clientId), global, roundGradients, gradCounts);
                    evaluatedClients++;
                }
                applyMetaUpdate(roundGradients, gradCounts);
            } catch (RuntimeException e) {
                MAML.TaskLearner.restoreAfterFailure(globalModel, global, e);
                throw e;
            } finally {
                MAML.TaskLearner.closeAll(global, roundGradients);
            }

            roundLosses.add(evaluatedClients > 0 ? totalRoundLoss / evaluatedClients : 0.0f);
            roundAccuracies.add(evaluateGlobalModel(clientTasks));
        }

        return new FederatedMetaResult(globalModel, roundLosses, roundAccuracies);
    }

    /**
     * One client's meta-gradient from the global weights: the mean over its tasks of the query gradient at the
     * adapted weights, added into {@code roundGradients} (one count per parameter in {@code gradCounts}). The global
     * weights are restored after each task. Returns the client's mean query loss.
     */
    private float clientUpdate(List<MAML.Task> tasks, Map<String, Tensor> global, Map<String, Tensor> roundGradients,
                               Map<String, Integer> gradCounts) {
        Map<String, Tensor> clientGradients = new LinkedHashMap<>();
        try {
            float totalLoss = 0.0f;
            for (MAML.Task task : tasks) {
                learner.adaptInPlace(task);
                totalLoss += learner.queryLossAndGradients(task, clientGradients);
                MAML.TaskLearner.restore(globalModel, global);
            }
            Map<String, Parameter> params = MAML.TaskLearner.parameters(globalModel);
            for (Map.Entry<String, Tensor> entry : clientGradients.entrySet()) {
                Tensor mean = MAML.TaskLearner.mean(entry.getValue(), tasks.size());
                try {
                    MAML.TaskLearner.addInto(roundGradients, entry.getKey(), params.get(entry.getKey()), mean);
                } finally {
                    if (mean != entry.getValue()) {
                        mean.close();
                    }
                }
                gradCounts.merge(entry.getKey(), 1, Integer::sum);
            }
            return totalLoss / tasks.size();
        } finally {
            MAML.TaskLearner.closeAll(clientGradients);
        }
    }

    private List<Integer> selectClients() {
        List<Integer> selected = new ArrayList<>();
        int numSelected = Math.max(1, (int) (numClients * clientFraction));

        List<Integer> available = new ArrayList<>();
        for (int i = 0; i < numClients; i++) {
            available.add(i);
        }

        for (int i = 0; i < numSelected && !available.isEmpty(); i++) {
            int idx = random.nextInt(available.size());
            selected.add(available.remove(idx));
        }

        return selected;
    }

    /**
     * {@code w -= metaLearningRate * (sum / count)} per parameter, with the per-parameter client count, in place and
     * without recording ({@link MAML.TaskLearner#applyMean}).
     */
    private void applyMetaUpdate(Map<String, Tensor> roundGradients, Map<String, Integer> gradCounts) {
        Map<String, Parameter> params = MAML.TaskLearner.parameters(globalModel);
        for (Map.Entry<String, Tensor> entry : roundGradients.entrySet()) {
            MAML.TaskLearner.applyMean(params.get(entry.getKey()), entry.getKey(), entry.getValue(),
                gradCounts.get(entry.getKey()), metaLearningRate);
        }
        globalModel.zeroGrad();
    }

    /**
     * The global model's mean query loss over the first task of each of the first {@code numClients} clients that
     * have tasks, in evaluation mode and without recording (0 when no client has tasks); the training mode is
     * restored afterwards.
     */
    private float evaluateGlobalModel(List<List<MAML.Task>> clientTasks) {
        float totalLoss = 0.0f;
        int count = 0;
        boolean wasTraining = globalModel.isTraining();
        globalModel.eval();
        try {
            for (int i = 0; i < Math.min(numClients, clientTasks.size()); i++) {
                List<MAML.Task> tasks = clientTasks.get(i);
                if (tasks.isEmpty()) {
                    continue;
                }
                totalLoss += learner.evaluate(tasks.get(0));
                count++;
            }
        } finally {
            globalModel.train(wasTraining);
        }
        return count > 0 ? totalLoss / count : 0.0f;
    }

    /**
     * The global weights by parameter name as host arrays: an export API that returns Java arrays.
     */
    public Map<String, float[]> getGlobalWeights() {
        ensureOpen();
        
        Map<String, float[]> weights = new HashMap<>();
        Map<String, Parameter> params = globalModel.namedParameters();
        
        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            weights.put(entry.getKey(), entry.getValue().tensor().toFloatArray());
        }
        
        return weights;
    }

    public Module globalModel() {
        ensureOpen();
        return globalModel;
    }

    public int numClients() {
        return numClients;
    }

    public int numRounds() {
        return numRounds;
    }

    public float clientFraction() {
        return clientFraction;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("FederatedMetaLearner has been closed");
        }
    }

    /** Marks this learner closed; it holds no tensors between calls and does not own the global model. */
    @Override
    public void close() {
        closed.set(true);
    }

    @Override
    public String toString() {
        return "FederatedMetaLearner{" +
            "numClients=" + numClients +
            ", numRounds=" + numRounds +
            ", innerSteps=" + innerSteps +
            ", clientFraction=" + clientFraction +
            ", closed=" + closed.get() +
            '}';
    }

    public static final class FederatedMetaResult {
        private final Module model;
        private final List<Float> roundLosses;
        private final List<Float> roundAccuracies;

        public FederatedMetaResult(Module model, List<Float> roundLosses, 
                                   List<Float> roundAccuracies) {
            this.model = model;
            this.roundLosses = new ArrayList<>(roundLosses);
            this.roundAccuracies = new ArrayList<>(roundAccuracies);
        }

        public Module model() {
            return model;
        }

        public List<Float> roundLosses() {
            return new ArrayList<>(roundLosses);
        }

        public List<Float> roundAccuracies() {
            return new ArrayList<>(roundAccuracies);
        }

        public float finalLoss() {
            if (roundLosses.isEmpty()) {
                return Float.NaN;
            }
            return roundLosses.get(roundLosses.size() - 1);
        }

        public float finalAccuracy() {
            if (roundAccuracies.isEmpty()) {
                return Float.NaN;
            }
            return roundAccuracies.get(roundAccuracies.size() - 1);
        }

        public int numRounds() {
            return roundLosses.size();
        }

        @Override
        public String toString() {
            return "FederatedMetaResult{" +
                "rounds=" + numRounds() +
                ", finalLoss=" + finalLoss() +
                ", finalAccuracy=" + finalAccuracy() +
                '}';
        }
    }
}

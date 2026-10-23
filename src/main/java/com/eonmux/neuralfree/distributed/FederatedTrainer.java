package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.TensorValues;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Federated averaging (FedAvg) and FedProx over simulated clients that share the global model and its backend.
 *
 * <p>Each round, every selected client starts from the global state (the weights and the persistent buffers, such
 * as running statistics), trains locally for {@code localEpochs} optimizer steps on its data with the
 * mean-squared-error loss plus the auxiliary loss terms the forward pass leaves pending
 * ({@link Module#takeAuxiliaryLoss()}; recorded by {@link TensorOps} in the caller's grad mode and backpropagated to
 * the model's parameters), and contributes its resulting state; the global weights and persistent buffers become the
 * means of the clients' (FedAvg over the state dict). FedProx adds {@code proximalMu * (w - w_global)} to each
 * gradient before the step (the gradient of {@code proximalMu / 2 * ||w - w_global||^2}). The global weights are
 * snapshotted on the device, client weights are summed on the device ({@code TensorOps.addInPlace}) and divided by the
 * client count. The persistent buffers, listed once per {@link #train} call ({@link Module#stateDict()} keys), are
 * small: they are snapshotted, summed and averaged on the host and written back with
 * {@link Module#loadStateDict(Map, boolean)}. Apart from them only the evaluation losses are read back. Each local step
 * runs inside a {@link ComputationGraph.StepScope}, so it leaves no autograd node behind; the proximal term, the weight
 * averaging, the restores and the evaluation run under {@link ComputationGraph#noGrad()}. The optimizer runs with
 * {@code learningRate} ({@link Optimizer#setLearningRate}) and its internal state carries across clients. The trainer
 * does not own the global model.</p>
 */
public final class FederatedTrainer implements AutoCloseable {
    private final Module globalModel;
    private final Optimizer optimizer;
    private final int numClients;
    private final int numRounds;
    private final int localEpochs;
    private final float learningRate;
    private final FederatedStrategy strategy;
    private final float proximalMu;
    private final Random random;
    private final AtomicBoolean closed;
    private final float clientFraction;

    public FederatedTrainer(Module globalModel, Optimizer optimizer, int numClients, int numRounds, int localEpochs, float learningRate) {
        this(globalModel, optimizer, numClients, numRounds, localEpochs, learningRate, FederatedStrategy.FED_AVG, 0.0f);
    }

    public FederatedTrainer(Module globalModel, Optimizer optimizer, int numClients, int numRounds, 
                           int localEpochs, float learningRate, FederatedStrategy strategy, float proximalMu) {
        if (globalModel == null) {
            throw new IllegalArgumentException("Global model cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("Optimizer cannot be null");
        }
        if (numClients < 1) {
            throw new IllegalArgumentException("Number of clients must be at least 1");
        }
        if (numRounds < 1) {
            throw new IllegalArgumentException("Number of rounds must be at least 1");
        }
        if (localEpochs < 1) {
            throw new IllegalArgumentException("Local epochs must be at least 1");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        if (strategy == null) {
            throw new IllegalArgumentException("Strategy cannot be null");
        }
        
        this.globalModel = globalModel;
        this.optimizer = optimizer;
        this.numClients = numClients;
        this.numRounds = numRounds;
        this.localEpochs = localEpochs;
        this.learningRate = learningRate;
        this.strategy = strategy;
        this.proximalMu = proximalMu;
        this.random = new Random();
        this.closed = new AtomicBoolean(false);
        this.clientFraction = 1.0f;
    }

    /**
     * Runs {@code numRounds} rounds; {@code clientData.get(i)} and {@code clientLabels.get(i)} are client
     * {@code i}'s inputs and targets (labels with the output's element count). The result holds the global model's
     * mean evaluation loss over all clients after round 0 and every {@code max(1, numRounds / 10)}-th round.
     */
    public FederatedResult train(List<Tensor> clientData, List<Tensor> clientLabels) {
        ensureOpen();
        
        if (clientData == null || clientLabels == null) {
            throw new IllegalArgumentException("Client data and labels cannot be null");
        }
        if (clientData.size() != numClients || clientLabels.size() != numClients) {
            throw new IllegalArgumentException("Client data and labels must have size equal to numClients");
        }
        optimizer.setLearningRate(learningRate);
        
        List<Float> roundLosses = new ArrayList<>();
        List<String> bufferPaths = PersistentBuffers.paths(globalModel);

        for (int round = 0; round < numRounds; round++) {
            runRound(clientData, clientLabels, bufferPaths);
            if (round % Math.max(1, numRounds / 10) == 0) {
                roundLosses.add(evaluateGlobalModel(clientData, clientLabels));
            }
        }
        
        return new FederatedResult(globalModel, roundLosses);
    }

    /**
     * One round: every selected client trains from the global state and the global state becomes the clients' mean;
     * on failure the global state is restored.
     */
    private void runRound(List<Tensor> clientData, List<Tensor> clientLabels, List<String> bufferPaths) {
        Map<String, Object> globalBuffers = PersistentBuffers.read(globalModel, bufferPaths);
        Map<Parameter, Tensor> global = snapshot();
        Map<Parameter, Tensor> clientSums = new LinkedHashMap<>();
        Map<String, float[]> bufferSums = new LinkedHashMap<>();
        try {
            List<Integer> selectedClients = selectClients();
            for (int clientId : selectedClients) {
                restore(global, globalBuffers);
                for (int epoch = 0; epoch < localEpochs; epoch++) {
                    trainClientLocally(clientData.get(clientId), clientLabels.get(clientId), global);
                }
                addClientWeights(clientSums);
                addClientBuffers(bufferSums, bufferPaths);
            }
            applyFedAvg(clientSums, bufferSums, selectedClients.size());
        } catch (RuntimeException e) {
            try {
                restore(global, globalBuffers);
            } catch (RuntimeException restoreFailure) {
                e.addSuppressed(restoreFailure);
            }
            throw e;
        } finally {
            closeAll(global);
            closeAll(clientSums);
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
     * One local optimizer step: zeroed gradients, forward, mean-squared-error loss plus the auxiliary loss terms the
     * forward pass left pending ({@link Module#takeAuxiliaryLoss()}) and backward inside a
     * {@link ComputationGraph.StepScope}, the FedProx term when enabled, then {@link Optimizer#step()} without
     * recording; the gradients are zeroed afterwards, also on failure, which closes them
     * ({@link Parameter#zeroGrad()}).
     */
    private void trainClientLocally(Tensor data, Tensor labels, Map<Parameter, Tensor> global) {
        globalModel.train();
        optimizer.zeroGrad();
        try {
            try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
                Tensor output = globalModel.forward(data);
                try (Tensor auxiliary = globalModel.takeAuxiliaryLoss()) {
                    mseLoss(output, labels, auxiliary, true);
                } finally {
                    if (output != data && !holdsTensor(output)) {
                        output.close();
                    }
                }
            }

            if (strategy == FederatedStrategy.FED_PROX && proximalMu != 0.0f) {
                applyProximalTerm(global);
            }
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                optimizer.step();
            }
        } catch (RuntimeException e) {
            try {
                optimizer.zeroGrad();
            } catch (RuntimeException releaseFailure) {
                e.addSuppressed(releaseFailure);
            }
            throw e;
        }
        optimizer.zeroGrad();
    }

    /**
     * {@code mean((output - labels)^2)} over the elements, recorded by {@link TensorOps}. With {@code backward} the
     * loss plus {@code auxiliary} (when not null) is backpropagated, when it requires grad, before the intermediates
     * are closed, and NaN is returned; without it the loss value is read back and returned.
     *
     * @throws IllegalArgumentException when the labels' element count differs from the output's
     */
    private static float mseLoss(Tensor output, Tensor labels, Tensor auxiliary, boolean backward) {
        if (labels.numElements() != output.numElements()) {
            throw new IllegalArgumentException("Labels of shape " + labels.shape() + " do not match output of shape "
                + output.shape());
        }
        try (Tensor target = labels.shape().equals(output.shape()) ? labels.detach() : labels.reshape(output.shape());
             Tensor diff = TensorOps.sub(output, target);
             Tensor squared = TensorOps.mul(diff, diff);
             Tensor loss = TensorOps.mean(squared);
             Tensor objective = backward && auxiliary != null ? TensorOps.add(loss, auxiliary) : null) {
            if (backward) {
                Tensor root = objective == null ? loss : objective;
                if (root.requiresGrad()) {
                    root.backward();
                }
                return Float.NaN;
            }
            return TensorValues.toFloats(loss)[0];
        }
    }

    /**
     * FedProx: each parameter with a gradient gets {@code g + proximalMu * (w - w_global)} as its gradient, without
     * recording (the proximal term edits gradients after the backward pass; it is not part of the graph). The new
     * gradients are computed out of place from every current gradient before any is installed, so gradients that
     * share storage between parameters (a backward pass hands one gradient to both operands of a same-shape sum) each
     * receive only their own parameter's term.
     */
    private void applyProximalTerm(Map<Parameter, Tensor> global) {
        Map<Parameter, Tensor> updated = new LinkedHashMap<>();
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (Map.Entry<Parameter, Tensor> entry : global.entrySet()) {
                Tensor grad = entry.getKey().grad();
                if (grad != null) {
                    try (Tensor drift = TensorOps.sub(entry.getKey().tensor(), entry.getValue());
                         Tensor term = TensorOps.mul(drift, proximalMu)) {
                        updated.put(entry.getKey(), TensorOps.add(grad, term));
                    }
                }
            }
        } catch (RuntimeException e) {
            closeAll(updated, e);
            throw e;
        }
        installGradients(updated);
    }

    /**
     * Makes each value of {@code gradients} its parameter's gradient ({@link Parameter#setGrad}); on failure the
     * gradients not yet installed are closed.
     */
    private static void installGradients(Map<Parameter, Tensor> gradients) {
        Iterator<Map.Entry<Parameter, Tensor>> pending = gradients.entrySet().iterator();
        while (pending.hasNext()) {
            Map.Entry<Parameter, Tensor> entry = pending.next();
            try {
                entry.getKey().setGrad(entry.getValue());
            } catch (RuntimeException e) {
                closeAll(gradients, e);
                throw e;
            }
            pending.remove();
        }
    }

    /** Adds the current (client) weights into {@code sums}, which start as zeros, without recording. */
    private void addClientWeights(Map<Parameter, Tensor> sums) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (Parameter param : uniqueParameters()) {
                Tensor sum = sums.get(param);
                if (sum == null) {
                    sum = Tensor.zeros(param.backend(), param.shape());
                    sums.put(param, sum);
                }
                TensorOps.addInPlace(sum, param.tensor(), 1.0f);
            }
        }
    }

    /** Adds the current (client) values of the persistent buffers at {@code paths} into {@code sums}, on the host. */
    private void addClientBuffers(Map<String, float[]> sums, List<String> paths) {
        Map<String, Object> values = PersistentBuffers.read(globalModel, paths);
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            float[] value = (float[]) entry.getValue();
            float[] sum = sums.get(entry.getKey());
            if (sum == null) {
                sums.put(entry.getKey(), value);
            } else {
                for (int i = 0; i < sum.length; i++) {
                    sum[i] += value[i];
                }
            }
        }
    }

    /**
     * The global weights become {@code sum / clientCount} (division by a device scalar), without recording, and the
     * persistent buffers {@code bufferSums / clientCount}; gradients cleared.
     */
    private void applyFedAvg(Map<Parameter, Tensor> sums, Map<String, float[]> bufferSums, int clientCount) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            for (Map.Entry<Parameter, Tensor> entry : sums.entrySet()) {
                if (clientCount == 1) {
                    entry.getKey().copyFrom(entry.getValue());
                    continue;
                }
                try (Tensor divisor = scalarOn(entry.getValue().backend(), clientCount);
                     Tensor mean = TensorOps.div(entry.getValue(), divisor)) {
                    entry.getKey().copyFrom(mean);
                }
            }
        }
        if (!bufferSums.isEmpty()) {
            Map<String, Object> means = new LinkedHashMap<>();
            for (Map.Entry<String, float[]> entry : bufferSums.entrySet()) {
                float[] mean = entry.getValue();
                for (int i = 0; i < mean.length; i++) {
                    mean[i] /= clientCount;
                }
                means.put(entry.getKey(), mean);
            }
            globalModel.loadStateDict(means, false);
        }
        globalModel.zeroGrad();
    }

    /**
     * The global model's mean evaluation loss over all clients, in evaluation mode and without recording; auxiliary
     * loss terms the forward passes leave pending (constants without recording) are dropped
     * ({@link Module#clearAuxiliaryLoss()}).
     */
    private float evaluateGlobalModel(List<Tensor> clientData, List<Tensor> clientLabels) {
        boolean wasTraining = globalModel.isTraining();
        globalModel.eval();
        float totalLoss = 0.0f;
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad()) {
            for (int i = 0; i < numClients; i++) {
                Tensor data = clientData.get(i);
                Tensor output = globalModel.forward(data);
                try {
                    globalModel.clearAuxiliaryLoss();
                    totalLoss += mseLoss(output, clientLabels.get(i), null, false);
                } finally {
                    if (output != data && !holdsTensor(output)) {
                        output.close();
                    }
                }
            }
        } finally {
            globalModel.train(wasTraining);
        }
        return totalLoss / numClients;
    }

    /** Whether {@code t} is a parameter tensor or a buffer of the global model, which a forward pass may return. */
    private boolean holdsTensor(Tensor t) {
        for (Parameter param : globalModel.parameters()) {
            if (param.tensor() == t) {
                return true;
            }
        }
        for (Tensor buffer : globalModel.namedBuffers().values()) {
            if (buffer == t) {
                return true;
            }
        }
        return false;
    }

    /** The global model's parameters, each once, in registration order. */
    private List<Parameter> uniqueParameters() {
        Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Parameter> result = new ArrayList<>();
        for (Parameter param : globalModel.parameters()) {
            if (seen.add(param)) {
                result.add(param);
            }
        }
        return result;
    }

    /** New device copies of the global weights ({@link Parameter#copyTensor()}); the caller closes them. */
    private Map<Parameter, Tensor> snapshot() {
        Map<Parameter, Tensor> copies = new LinkedHashMap<>();
        try {
            for (Parameter param : uniqueParameters()) {
                copies.put(param, param.copyTensor());
            }
        } catch (RuntimeException e) {
            closeAll(copies);
            throw e;
        }
        return copies;
    }

    /**
     * Writes each snapshot back into its parameter, in place and without recording ({@link Parameter#copyFrom}), and
     * the buffer values back into the persistent buffers ({@link Module#loadStateDict(Map, boolean)}).
     */
    private void restore(Map<Parameter, Tensor> snapshot, Map<String, Object> bufferValues) {
        for (Map.Entry<Parameter, Tensor> entry : snapshot.entrySet()) {
            entry.getKey().copyFrom(entry.getValue());
        }
        if (!bufferValues.isEmpty()) {
            globalModel.loadStateDict(bufferValues, false);
        }
    }

    private static void closeAll(Map<Parameter, Tensor> tensors) {
        for (Tensor t : tensors.values()) {
            t.close();
        }
        tensors.clear();
    }

    /** {@link #closeAll(Map)}, adding each close failure to {@code failure} as suppressed. */
    private static void closeAll(Map<Parameter, Tensor> tensors, RuntimeException failure) {
        for (Tensor t : tensors.values()) {
            try {
                t.close();
            } catch (RuntimeException closeFailure) {
                failure.addSuppressed(closeFailure);
            }
        }
        tensors.clear();
    }

    /**
     * A rank-0 FLOAT32 tensor holding {@code value} on {@code backend}: {@code 0 + value} over a zero-filled scalar
     * ({@link Tensor#zeros}, filled on the device when the train kernels are available, so no host transfer is made).
     */
    private static Tensor scalarOn(ComputeBackend backend, float value) {
        try (Tensor zero = Tensor.zeros(backend, Shape.scalar(), DType.FLOAT32)) {
            return TensorOps.add(zero, value);
        }
    }

    public Module globalModel() {
        ensureOpen();
        return globalModel;
    }

    public int numClients() {
        ensureOpen();
        return numClients;
    }

    public int numRounds() {
        ensureOpen();
        return numRounds;
    }

    public FederatedStrategy strategy() {
        ensureOpen();
        return strategy;
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("FederatedTrainer has been closed");
        }
    }

    /** Marks this trainer closed; it holds no tensors between calls and does not own the global model. */
    @Override
    public void close() {
        closed.set(true);
    }

    public enum FederatedStrategy {
        FED_AVG,
        FED_PROX
    }

    public static class FederatedResult {
        public final Module model;
        public final List<Float> roundLosses;

        public FederatedResult(Module model, List<Float> roundLosses) {
            this.model = model;
            this.roundLosses = new ArrayList<>(roundLosses);
        }

        public float finalLoss() {
            if (roundLosses.isEmpty()) {
                return Float.NaN;
            }
            return roundLosses.get(roundLosses.size() - 1);
        }
    }
}

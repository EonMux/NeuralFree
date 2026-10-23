package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.LRScheduler;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Mini-batch training loop: collate, forward, loss, backward, then at each accumulation boundary the mean of the
 * accumulated gradients, optional gradient clipping ({@link GradientClipping}), the optimizer step and zeroing the
 * gradients. Batches are collated and evaluation accuracy is counted on the tensors' backend; the loss and accuracy
 * scalars are the only per-batch readbacks (logged metrics and the early-stopping decision).
 *
 * <p><b>Training step.</b> The model's forward pass and the loss record the autograd graph. The step back-propagates
 * the loss plus the auxiliary loss terms the forward pass left pending ({@link Module#takeAuxiliaryLoss()}, such as a
 * mixture-of-experts load-balance term or a VAE's KL term); the logged loss is the loss alone. Terms left pending by
 * forward passes outside the step are discarded before its forward pass. The gradients reach the parameters the
 * optimizer updates ({@link #appliedParameters}); after the first backward pass of each {@link #train} call the
 * trainer checks that at least one of them that requires grad received a gradient, and throws
 * {@link IllegalStateException} with {@link #NOT_CONNECTED} otherwise, as it does for a loss that does not require
 * grad.</p>
 *
 * <p><b>Accumulation boundary.</b> With {@link TrainerConfig#accumulationSteps()} {@code > 1} the gradients of
 * several batches are summed in the parameters' gradient slots; the boundary multiplies every gradient by
 * {@code 1 / n}, {@code n} being the number of batches back-propagated since the previous step (the last group of an
 * epoch and a group cut short by {@link #requestStop()} included), so the step applies their mean. Clipping then acts
 * on the mean. The boundary runs under {@link ComputationGraph#noGrad()}.</p>
 *
 * <p><b>Tensor lifetime.</b> Device buffers have no GC-based release, so every step releases what it created: the
 * collated batch, the prediction and the loss are closed once the step no longer needs them ({@link Temporaries}),
 * the backward pass releases its graph ({@link ComputationGraph#backward(Tensor)}), and {@link Temporaries} removes
 * every node the step recorded that the pass did not reach, closing their outputs. After each batch the calling
 * thread's graph is back at its size before the batch. The gradients are closed when the optimizer zeroes them at the
 * accumulation boundary ({@link Parameter#zeroGrad()}; scaling and clipping on the host close the gradients they
 * replace). The batch's samples are closed with the step's temporaries ({@link Temporaries#keepSamples}), which
 * releases the copies a loader made for them. A pass over a loader that ends before its last batch (a stop request or
 * a failure) is ended with {@link #endPass}, which closes that pass alone and releases the batches it loaded ahead,
 * leaving other passes over the same loader running; a stop request is honoured before the next batch is fetched.
 * Evaluation, accuracy counting and {@link #predict} run without recording autograd operations, and drop the
 * auxiliary terms their forward passes leave pending ({@link Module#clearAuxiliaryLoss()}).</p>
 *
 * <p><b>Progress and checkpoints.</b> Each run counts its steps from 0 ({@link #currentStep()}, the count
 * {@link Callback#onBatchEnd} receives and {@link TrainingResult#totalSteps()} reports). An epoch's training loss is
 * the mean of its batch losses, or NaN when it trained no batch (a stop requested before its first batch), so that
 * early stopping and checkpoint selection never take it for an improvement. With
 * {@link TrainerConfig#verbose()} the trainer writes its own progress log ({@link ProgressLog}). A configuration with
 * {@link TrainerConfig#checkpointEveryNEpochs()} positive needs a {@link ModelCheckpoint} callback, which then saves
 * at that interval; the constructor checks it ({@link TrainerConfig#validateCallbacks}).</p>
 */
public final class Trainer implements TrainingContext, AutoCloseable {
    /** The message of the check that a training loss reaches a trainable parameter. */
    public static final String NOT_CONNECTED = "loss is not connected to any trainable parameter";

    private final Module model;
    private final Optimizer optimizer;
    private final LossFunction lossFunction;
    private final DataLoader<TensorDataset.Sample> trainLoader;
    private final DataLoader<TensorDataset.Sample> valLoader;
    private final List<Callback> callbacks;
    private final LRScheduler lrScheduler;
    private final TrainerConfig config;
    private final ProgressLog progressLog;

    private int currentEpoch = 0;
    private int currentStep = 0;
    private float currentLoss = 0.0f;
    private volatile boolean closed = false;
    private volatile boolean shouldStop = false;
    private boolean connectionVerified = false;

    public Trainer(Module model, Optimizer optimizer, LossFunction lossFunction,
                   DataLoader<TensorDataset.Sample> trainLoader) {
        this(model, optimizer, lossFunction, trainLoader, null, new ArrayList<>(), null, TrainerConfig.defaults());
    }

    /**
     * A trainer of {@code model}; {@code valLoader}, {@code callbacks} and {@code lrScheduler} may be null for none.
     *
     * @throws NullPointerException     when the model, optimizer, loss function, training loader or configuration is
     *                                  null
     * @throws IllegalArgumentException when the callbacks cannot carry out the configuration
     *                                  ({@link TrainerConfig#validateCallbacks})
     */
    public Trainer(Module model, Optimizer optimizer, LossFunction lossFunction,
                   DataLoader<TensorDataset.Sample> trainLoader,
                   DataLoader<TensorDataset.Sample> valLoader,
                   List<Callback> callbacks,
                   LRScheduler lrScheduler,
                   TrainerConfig config) {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        Objects.requireNonNull(lossFunction, "LossFunction cannot be null");
        Objects.requireNonNull(trainLoader, "TrainLoader cannot be null");
        Objects.requireNonNull(config, "Config cannot be null");

        this.model = model;
        this.optimizer = optimizer;
        this.lossFunction = lossFunction;
        this.trainLoader = trainLoader;
        this.valLoader = valLoader;
        this.callbacks = callbacks != null ? new ArrayList<>(callbacks) : new ArrayList<>();
        this.lrScheduler = lrScheduler;
        this.config = config;
        config.validateCallbacks(this.callbacks);
        this.progressLog = new ProgressLog(Trainer.class, config, this.callbacks);
    }

    public TrainingResult train(int epochs) {
        ensureOpen();
        if (epochs < 1) {
            throw new IllegalArgumentException("Epochs must be at least 1, got: " + epochs);
        }

        long startTime = System.currentTimeMillis();
        TrainingResult.Builder resultBuilder = TrainingResult.builder();
        resultBuilder.epochs(epochs);

        shouldStop = false;
        connectionVerified = false;
        currentStep = 0;
        model.train();

        for (Callback callback : callbacks) {
            callback.onTrainBegin(this);
        }

        Float bestValLoss = null;
        int bestEpoch = -1;

        for (int epoch = 0; epoch < epochs && !shouldStop; epoch++) {
            currentEpoch = epoch;

            for (Callback callback : callbacks) {
                callback.onEpochBegin(epoch, epochs);
            }

            float epochLoss = trainEpoch(epoch, epochs);

            Float valLoss = null;
            if (valLoader != null && (epoch + 1) % config.evalEveryNEpochs() == 0) {
                for (Callback callback : callbacks) {
                    callback.onValidationBegin(epoch);
                }
                EvalResult evalResult = evaluate(valLoader);
                valLoss = evalResult.loss();
                for (Callback callback : callbacks) {
                    callback.onValidationEnd(epoch, evalResult);
                }

                if (bestValLoss == null || valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    bestEpoch = epoch;
                }
            }

            resultBuilder.addTrainLoss(epochLoss);
            if (valLoss != null) {
                resultBuilder.addValLoss(valLoss);
            }

            if (lrScheduler != null) {
                lrScheduler.stepEpoch();
                resultBuilder.addLearningRate(optimizer.getLearningRate());
            }

            progressLog.epoch(epoch, epochs, epochLoss, valLoss);
            for (Callback callback : callbacks) {
                callback.onEpochEnd(epoch, epochLoss, valLoss);
            }
        }

        long endTime = System.currentTimeMillis();
        resultBuilder.trainingTimeMs(endTime - startTime);
        resultBuilder.totalSteps(currentStep);
        resultBuilder.bestValLoss(bestValLoss);
        resultBuilder.bestEpoch(bestEpoch);

        TrainingResult result = resultBuilder.build();

        for (Callback callback : callbacks) {
            callback.onTrainEnd(result);
        }

        return result;
    }

    private float trainEpoch(int epoch, int epochs) {
        float totalLoss = 0.0f;
        int batchCount = 0;
        int accumulationCounter = 0;
        int totalBatches = trainLoader.numBatches();

        DataLoader.Pass<TensorDataset.Sample> pass = trainLoader.pass();
        try {
            while (!shouldStop && pass.hasNext()) {
                List<TensorDataset.Sample> batch = pass.next();
                notifyBatchBegin(batch, totalBatches);

                float batchLoss = trainBatch(batch);
                totalLoss += batchLoss;
                batchCount++;
                accumulationCounter++;

                if (accumulationCounter >= config.accumulationSteps()) {
                    applyAccumulatedGradients(accumulationCounter);
                    accumulationCounter = 0;
                }

                boolean logged = progressLog.logsStep(currentStep + 1);
                float learningRate = logged ? optimizer.getLearningRate() : Float.NaN;
                if (lrScheduler != null) {
                    lrScheduler.step();
                }

                currentStep++;
                if (logged) {
                    progressLog.step(epoch, epochs, currentStep, batchLoss, learningRate);
                }

                for (Callback callback : callbacks) {
                    callback.onBatchEnd(currentStep, batchLoss);
                }
            }
        } catch (RuntimeException | Error e) {
            endPass(pass, e);
            throw e;
        }
        endPass(pass, null);

        if (accumulationCounter > 0) {
            applyAccumulatedGradients(accumulationCounter);
        }

        return batchCount > 0 ? totalLoss / batchCount : Float.NaN;
    }

    /**
     * Runs the {@link Callback#onBatchBegin} callbacks for a fetched batch; when one fails, the batch's samples are
     * closed ({@link #closeSamples}) before the failure propagates, since no step will take them.
     */
    private void notifyBatchBegin(List<TensorDataset.Sample> batch, int totalBatches) {
        try {
            for (Callback callback : callbacks) {
                callback.onBatchBegin(currentStep, totalBatches);
            }
        } catch (RuntimeException | Error e) {
            closeSamples(batch, e);
            throw e;
        }
    }

    /**
     * Ends a loader pass that its consumer leaves: when the pass has batches left it is closed
     * ({@link DataLoader.Pass#close()}), which stops its loading ahead and releases the batches it loaded but did not
     * hand out; other passes over the same loader are not affected. A pass that handed out every batch is left as it
     * is. With a {@code primary} failure (the reason the consumer leaves), a failure while closing is added to it as
     * suppressed; otherwise it is thrown.
     */
    public static void endPass(DataLoader.Pass<?> pass, Throwable primary) {
        Objects.requireNonNull(pass, "Pass cannot be null");
        try {
            if (pass.hasNext()) {
                pass.close();
            }
        } catch (RuntimeException e) {
            if (primary == null) {
                throw e;
            }
            primary.addSuppressed(e);
        }
    }

    /**
     * Closes the samples of a batch no step will take ({@link TensorDataset.Sample#close()}, which releases only the
     * copies a loader made); every sample is tried and each failure is added to {@code primary} as suppressed.
     */
    public static void closeSamples(List<TensorDataset.Sample> batch, Throwable primary) {
        for (TensorDataset.Sample sample : batch) {
            try {
                sample.close();
            } catch (RuntimeException e) {
                primary.addSuppressed(e);
            }
        }
    }

    /**
     * The accumulation boundary over {@link #appliedParameters}, under {@link ComputationGraph#noGrad()}: the mean of
     * the gradients of the {@code batches} batches back-propagated since the previous step (each gradient multiplied
     * by {@code 1 / batches}, {@link GradientClipping#scale}), optional clipping of the mean, the optimizer step, then
     * zeroing the gradients, which closes them ({@link Parameter#zeroGrad()}).
     */
    private void applyAccumulatedGradients(int batches) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            List<Parameter> params = appliedParameters(optimizer);
            if (batches > 1) {
                GradientClipping.scale(params, 1.0f / batches);
            }
            if (config.gradientClippingEnabled()) {
                GradientClipping.clip(params, config);
            }
            optimizer.step();
            optimizer.zeroGrad();
        }
    }

    /**
     * Forward and backward pass of one batch: terms left pending before the batch are dropped
     * ({@link Module#clearAuxiliaryLoss()}), then the loss plus the batch's auxiliary loss terms is back-propagated
     * ({@link #withAuxiliaryLoss}, {@link #backwardToParameters}); the batch, prediction, loss and auxiliary tensors
     * are closed and every node the batch recorded is removed before returning.
     *
     * @return the loss without the auxiliary terms, read back for logging
     * @throws IllegalStateException with {@link #NOT_CONNECTED} when the loss does not require grad, or when the first
     *                               backward pass of a {@link #train} call gives none of the optimizer's trainable
     *                               parameters a gradient
     */
    private float trainBatch(List<TensorDataset.Sample> batch) {
        currentLoss = 0.0f;

        try (Temporaries step = new Temporaries(model)) {
            step.keepSamples(batch);
            model.clearAuxiliaryLoss();
            Tensor batchInput = step.keep(collateInputs(batch));
            Tensor batchTarget = step.keep(collateTargets(batch));

            Tensor prediction = step.keep(model.forward(batchInput));
            Tensor loss = step.keep(lossFunction.compute(prediction, batchTarget));

            float lossValue = loss.toFloatArray()[0];
            currentLoss = lossValue;

            Tensor objective = withAuxiliaryLoss(model, loss, step);
            backwardToParameters(objective, appliedParameters(optimizer), !connectionVerified);
            connectionVerified = true;

            return lossValue;
        }
    }

    /**
     * The parameters a training step updates: {@code optimizer}'s parameters, each {@link Parameter} object once, in
     * first-listed order. The accumulation mean, loss-scale unscaling and clipping act on these gradients.
     */
    public static List<Parameter> appliedParameters(Optimizer optimizer) {
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        Set<Parameter> seen = identitySet();
        List<Parameter> params = new ArrayList<>();
        for (Parameter p : optimizer.parameters()) {
            if (seen.add(p)) {
                params.add(p);
            }
        }
        return params;
    }

    /**
     * {@code loss} plus the auxiliary loss terms {@code model}'s forward pass left pending
     * ({@link Module#takeAuxiliaryLoss()}), added with {@link TensorOps#add} (recorded in grad mode), or {@code loss}
     * itself when no term is pending. A loss of more than one element is summed first ({@link TensorOps#sum}), since
     * a backward pass seeded with ones differentiates that sum. The taken term and every tensor made here are kept in
     * {@code step} for closing.
     */
    public static Tensor withAuxiliaryLoss(Module model, Tensor loss, Temporaries step) {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(loss, "Loss cannot be null");
        Objects.requireNonNull(step, "Temporaries cannot be null");
        Tensor auxiliary = step.keep(model.takeAuxiliaryLoss());
        if (auxiliary == null) {
            return loss;
        }
        Tensor total = loss.numElements() == 1 ? loss : step.keep(TensorOps.sum(loss));
        return step.keep(TensorOps.add(total, auxiliary));
    }

    /**
     * Back-propagates {@code objective}, the loss of a training step ({@link Tensor#backward()}). With
     * {@code verify}, the gradient slots of the parameters of {@code params} that require grad are compared before
     * and after the pass: at least one must hold a new gradient afterwards (a slot the pass filled or accumulated
     * into).
     *
     * @throws IllegalStateException with {@link #NOT_CONNECTED} when {@code objective} does not require grad, or, with
     *                               {@code verify}, when no parameter of {@code params} that requires grad received a
     *                               gradient (the pass ran; the caller's step closes what it filled)
     */
    public static void backwardToParameters(Tensor objective, List<Parameter> params, boolean verify) {
        Objects.requireNonNull(objective, "Objective cannot be null");
        Objects.requireNonNull(params, "Parameters cannot be null");
        if (!objective.requiresGrad()) {
            throw new IllegalStateException(NOT_CONNECTED);
        }
        if (!verify) {
            objective.backward();
            return;
        }
        List<Parameter> trainable = new ArrayList<>(params.size());
        List<Tensor> before = new ArrayList<>(params.size());
        for (Parameter p : params) {
            if (p.requiresGrad()) {
                trainable.add(p);
                before.add(p.grad());
            }
        }
        objective.backward();
        for (int i = 0; i < trainable.size(); i++) {
            Tensor grad = trainable.get(i).grad();
            if (grad != null && grad != before.get(i)) {
                return;
            }
        }
        throw new IllegalStateException(NOT_CONNECTED);
    }

    /**
     * Mean loss and accuracy of the model over {@code loader}, in evaluation mode and without recording autograd
     * operations; each batch's tensors, and the auxiliary loss terms its forward pass leaves pending, are closed after
     * use, and a pass a failure leaves is ended ({@link #endPass}). The loss is the loss function's value alone.
     * Accuracy is the fraction of samples {@link #countCorrect} counts as correct; it is left out of the result when a
     * batch's prediction and target have no accuracy definition there. The model is back in training mode afterwards.
     */
    public EvalResult evaluate(DataLoader<TensorDataset.Sample> loader) {
        ensureOpen();
        Objects.requireNonNull(loader, "Loader cannot be null");

        float totalLoss = 0.0f;
        int totalSamples = 0;
        int correctPredictions = 0;
        boolean accuracyDefined = true;

        DataLoader.Pass<TensorDataset.Sample> pass = loader.pass();
        model.eval();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            while (pass.hasNext()) {
                List<TensorDataset.Sample> batch = pass.next();
                try (Temporaries step = new Temporaries(model)) {
                    step.keepSamples(batch);
                    Tensor batchInput = step.keep(collateInputs(batch));
                    Tensor batchTarget = step.keep(collateTargets(batch));

                    Tensor prediction = step.keep(model.forward(batchInput));
                    model.clearAuxiliaryLoss();
                    Tensor loss = step.keep(lossFunction.compute(prediction, batchTarget));

                    float lossValue = loss.toFloatArray()[0];
                    int batchSize = batch.size();

                    totalLoss += lossValue * batchSize;
                    totalSamples += batchSize;

                    if (accuracyDefined) {
                        OptionalInt correct = countCorrect(prediction, batchTarget, batchSize);
                        if (correct.isPresent()) {
                            correctPredictions += correct.getAsInt();
                        } else {
                            accuracyDefined = false;
                        }
                    }
                }
            }
        } catch (RuntimeException | Error e) {
            endPass(pass, e);
            throw e;
        } finally {
            model.train();
        }

        float avgLoss = totalSamples > 0 ? totalLoss / totalSamples : 0.0f;
        EvalResult.Builder result = EvalResult.builder()
            .loss(avgLoss)
            .numSamples(totalSamples);
        if (accuracyDefined) {
            result.accuracy(totalSamples > 0 ? (float) correctPredictions / totalSamples : 0.0f);
        }
        return result.build();
    }

    /**
     * The inputs of a batch stacked along a new leading batch dimension ({@link TensorOps#stack}, a device copy when
     * the train kernels are available).
     *
     * @throws IllegalArgumentException when the batch is empty or the inputs differ in shape
     */
    public static Tensor collateInputs(List<TensorDataset.Sample> batch) {
        return collate(batch, true);
    }

    /**
     * The targets of a batch stacked along a new leading batch dimension, as {@link #collateInputs} does for inputs.
     *
     * @throws IllegalArgumentException when the batch is empty or the targets differ in shape
     */
    public static Tensor collateTargets(List<TensorDataset.Sample> batch) {
        return collate(batch, false);
    }

    private static Tensor collate(List<TensorDataset.Sample> batch, boolean inputs) {
        Objects.requireNonNull(batch, "Batch cannot be null");
        if (batch.isEmpty()) {
            throw new IllegalArgumentException("Batch cannot be empty");
        }
        List<Tensor> parts = new ArrayList<>(batch.size());
        for (TensorDataset.Sample sample : batch) {
            parts.add(inputs ? sample.input() : sample.target());
        }
        return TensorOps.stack(parts, 0);
    }

    /**
     * The number of correctly classified samples in one batch, counted on the tensors' backend with one scalar
     * readback. The prediction is viewed as {@code [batchSize, classes]} and each sample's predicted class is
     * {@link TensorOps#argmax} over the classes: the first maximum, never NaN, class 0 when no value exceeds
     * negative infinity. A sample is correct when its target, viewed as {@code [batchSize, classes]}, holds a value
     * within 0.5 of 1 at the predicted class (one-hot or probability targets), or when its target, viewed as
     * {@code [batchSize]}, equals the predicted class index (class-index targets). A prediction of one value per
     * sample has no accuracy definition: the argmax over a single class is always class 0, whatever the value.
     *
     * Runs under {@link ComputationGraph#noGrad()}: nothing is recorded, also for a prediction that requires grad.
     *
     * @return the count, or empty when accuracy is not defined for these tensors: either is not FLOAT32, the
     *         prediction is empty, not a whole number of elements per sample or one element per sample, or the
     *         target has neither {@code batchSize * classes} nor {@code batchSize} elements
     */
    public static OptionalInt countCorrect(Tensor prediction, Tensor target, int batchSize) {
        Objects.requireNonNull(prediction, "Prediction cannot be null");
        Objects.requireNonNull(target, "Target cannot be null");
        long n = prediction.numElements();
        if (batchSize <= 0 || n == 0 || n % batchSize != 0 || prediction.dtype() != DType.FLOAT32
            || target.dtype() != DType.FLOAT32) {
            return OptionalInt.empty();
        }
        int classes = (int) (n / batchSize);
        if (classes == 1) {
            return OptionalInt.empty();
        }
        boolean oneHot = target.numElements() == n;
        boolean classIndex = target.numElements() == batchSize;
        if (!oneHot && !classIndex) {
            return OptionalInt.empty();
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Temporaries temps = new Temporaries()) {
            Tensor rows = temps.keep(prediction.reshape(batchSize, classes));
            Tensor predicted = temps.keep(TensorOps.argmax(rows, 1));
            Tensor hits;
            if (oneHot) {
                Tensor targetRows = temps.keep(target.reshape(batchSize, classes));
                Tensor index = temps.keep(predicted.reshape(batchSize, 1));
                Tensor picked = temps.keep(temps.keep(TensorOps.gather(targetRows, 1, index)).reshape(batchSize));
                Tensor distance = temps.keep(TensorOps.abs(temps.keep(TensorOps.add(picked, -1.0f))));
                Tensor half = temps.keep(Tensor.fromArray(prediction.backend(), new float[]{0.5f}));
                hits = temps.keep(temps.keep(TensorOps.lt(distance, half)).to(DType.FLOAT32));
            } else {
                Tensor targetIndex = temps.keep(target.reshape(batchSize));
                Tensor predictedIndex = temps.keep(predicted.to(DType.FLOAT32));
                hits = temps.keep(temps.keep(TensorOps.eq(targetIndex, predictedIndex)).to(DType.FLOAT32));
            }
            Tensor count = temps.keep(TensorOps.sum(hits));
            return OptionalInt.of((int) count.toFloatArray()[0]);
        }
    }

    /**
     * Tensors created for one computation, closed together in reverse order of keeping when it no longer needs them,
     * and the autograd step of that computation: creating the temporaries opens a {@link ComputationGraph.StepScope}
     * on the calling thread's graph, and closing them (on the same thread) ends it.
     *
     * <p>{@link #close()} first releases the autograd nodes each kept tensor was computed from
     * ({@link ComputationGraph#release(Tensor)}), then closes the step scope, which removes every node recorded since
     * the temporaries were created that is still in the graph (a forward graph no backward pass reached, a discarded
     * branch, nodes a stray consumer kept alive) and closes their outputs, so afterwards the graph is back at its size
     * when the temporaries were created. A result of the computation that must outlive the temporaries is therefore
     * either a tensor no node produced or a {@link Tensor#detach()} handle. A gradient a kept tensor received (one that
     * requires grad without a recorded node, such as a loss computed without recording) is cleared and closed with it.
     * A tensor kept several times is closed once. With an owner module, the tensors that module holds (its parameters'
     * tensors and its buffers, one of which a forward pass may return) are spared: neither released nor closed. Batch
     * samples kept with {@link #keepSamples} are closed last. Failures are collected: the first is thrown with the
     * others suppressed.</p>
     */
    public static final class Temporaries implements AutoCloseable {
        private final Module owner;
        private final List<Tensor> tensors = new ArrayList<>();
        private final ComputationGraph graph = ComputationGraph.current();
        private final ComputationGraph.StepScope scope = graph.step();
        private final List<TensorDataset.Sample> samples = new ArrayList<>();

        /** Temporaries that are all closed. */
        public Temporaries() {
            this(null);
        }

        /** Temporaries of a computation over {@code owner} (null for none), whose own tensors are spared. */
        public Temporaries(Module owner) {
            this.owner = owner;
        }

        /**
         * Keeps the samples of a consumed batch for closing after the tensors: only the tensors a {@link DataLoader}
         * created for the batch (copies on its backend) are released; a dataset's own tensors stay open
         * ({@link TensorDataset.Sample#close()}).
         */
        public void keepSamples(List<TensorDataset.Sample> batch) {
            samples.addAll(batch);
        }

        /** Keeps {@code t} for closing and returns it; null is returned unchanged. */
        public Tensor keep(Tensor t) {
            if (t != null) {
                tensors.add(t);
            }
            return t;
        }

        @Override
        public void close() {
            Set<Tensor> spared = ownerTensors();
            List<Tensor> toClose = new ArrayList<>(tensors.size());
            for (int i = tensors.size() - 1; i >= 0; i--) {
                Tensor t = tensors.get(i);
                if (!spared.contains(t)) {
                    toClose.add(t);
                }
            }
            tensors.clear();
            RuntimeException failure = null;
            for (Tensor t : toClose) {
                try {
                    graph.release(t);
                } catch (RuntimeException e) {
                    failure = addFailure(failure, e);
                }
            }
            for (Tensor t : spared) {
                scope.spare(t);
            }
            try {
                scope.close();
            } catch (RuntimeException e) {
                failure = addFailure(failure, e);
            }
            List<Tensor> gradients = new ArrayList<>();
            for (Tensor t : toClose) {
                Tensor g = t.grad();
                if (g != null) {
                    t.setGrad(null);
                    gradients.add(g);
                }
            }
            try {
                closeAll(gradients);
            } catch (RuntimeException e) {
                failure = addFailure(failure, e);
            }
            try {
                closeAll(toClose);
            } catch (RuntimeException e) {
                failure = addFailure(failure, e);
            }
            for (TensorDataset.Sample sample : samples) {
                try {
                    sample.close();
                } catch (RuntimeException e) {
                    failure = addFailure(failure, e);
                }
            }
            samples.clear();
            if (failure != null) {
                throw failure;
            }
        }

        private Set<Tensor> ownerTensors() {
            Set<Tensor> owned = identitySet();
            if (owner != null) {
                for (Parameter p : owner.parameters()) {
                    owned.add(p.tensor());
                }
                owned.addAll(owner.namedBuffers().values());
            }
            return owned;
        }
    }

    /**
     * The progress log a trainer writes when its {@link TrainerConfig#verbose()} is set, through the project logger
     * ({@link LoggingUtil}) under the trainer's class name, at INFO:
     * <ul>
     *   <li>a step line after every batch whose step count (the count {@link Callback#onBatchEnd} receives) is a
     *       multiple of {@link TrainerConfig#logEveryNSteps()}:
     *       {@code epoch <e>/<epochs> step <s>: loss <loss>, lr <learning rate>}, with the batch's loss as
     *       {@link Callback#onBatchEnd} receives it and the learning rate the batch trained with (read before the
     *       scheduler's per-batch step);</li>
     *   <li>an epoch line at the end of every epoch, before {@link Callback#onEpochEnd}:
     *       {@code epoch <e>/<epochs>: loss <training loss>[, val_loss <validation loss>]}, the validation loss when
     *       validation ran in that epoch.</li>
     * </ul>
     * Epochs are counted from 1 and values are printed with six significant digits.
     *
     * <p>A {@link ProgressLogger} callback whose {@link ProgressLogger#verbose()} is set reports the same progress, so
     * the log is off for a trainer whose callbacks include one: the progress is reported once, by the callback. It is
     * also off without {@code verbose}, and writes nothing while the logger's INFO level is disabled.</p>
     */
    public static final class ProgressLog {
        private final Logger logger;
        private final int logEveryNSteps;
        private final boolean enabled;

        /**
         * The progress log of a trainer of class {@code trainerClass} running with {@code config} and
         * {@code callbacks}.
         */
        public ProgressLog(Class<?> trainerClass, TrainerConfig config, List<? extends Callback> callbacks) {
            Objects.requireNonNull(trainerClass, "Trainer class cannot be null");
            Objects.requireNonNull(config, "Config cannot be null");
            Objects.requireNonNull(callbacks, "Callbacks cannot be null");
            this.logger = LoggingUtil.getLogger(trainerClass);
            this.logEveryNSteps = config.logEveryNSteps();
            this.enabled = config.verbose() && !hasVerboseProgressLogger(callbacks);
        }

        private static boolean hasVerboseProgressLogger(List<? extends Callback> callbacks) {
            for (Callback callback : callbacks) {
                if (callback instanceof ProgressLogger && ((ProgressLogger) callback).verbose()) {
                    return true;
                }
            }
            return false;
        }

        /** Whether the log writes at all (see the class description). */
        public boolean enabled() {
            return enabled;
        }

        /** Whether the batch that brings the step count to {@code step} gets a step line. */
        public boolean logsStep(int step) {
            return enabled && step % logEveryNSteps == 0 && logger.isInfoEnabled();
        }

        /** Writes the step line of step {@code step} in epoch {@code epoch} (from 0) when {@link #logsStep} says so. */
        public void step(int epoch, int epochs, int step, float loss, float learningRate) {
            if (logsStep(step)) {
                logger.info(String.format(Locale.ROOT, "epoch %d/%d step %d: loss %.6g, lr %.6g",
                    epoch + 1, epochs, step, loss, learningRate));
            }
        }

        /** Writes the epoch line of epoch {@code epoch} (from 0); {@code valLoss} is null when no validation ran. */
        public void epoch(int epoch, int epochs, float trainLoss, Float valLoss) {
            if (!enabled || !logger.isInfoEnabled()) {
                return;
            }
            String line = String.format(Locale.ROOT, "epoch %d/%d: loss %.6g", epoch + 1, epochs, trainLoss);
            if (valLoss != null) {
                line += String.format(Locale.ROOT, ", val_loss %.6g", valLoss);
            }
            logger.info(line);
        }
    }

    private static <T> Set<T> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }

    private static RuntimeException addFailure(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    /** Closes every tensor; the first failure is thrown once all were tried, with later ones suppressed. */
    private static void closeAll(List<Tensor> tensors) {
        RuntimeException failure = null;
        for (Tensor t : tensors) {
            try {
                t.close();
            } catch (RuntimeException e) {
                failure = addFailure(failure, e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * The model's output for {@code input} in evaluation mode, computed without recording autograd operations. The
     * result belongs to the caller; the auxiliary loss terms the forward pass leaves pending are closed; the model's
     * previous mode is restored.
     */
    public Tensor predict(Tensor input) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");

        boolean wasTraining = model.isTraining();
        model.eval();

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            Tensor output = model.forward(input);
            try {
                model.clearAuxiliaryLoss();
            } catch (RuntimeException e) {
                if (output != input) {
                    closeAfterFailure(output, e);
                }
                throw e;
            }
            return output;
        } finally {
            if (wasTraining) {
                model.train();
            }
        }
    }

    private static void closeAfterFailure(Tensor t, RuntimeException failure) {
        try {
            t.close();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    @Override
    public void requestStop() {
        this.shouldStop = true;
    }

    @Override
    public Module model() {
        ensureOpen();
        return model;
    }

    @Override
    public Optimizer optimizer() {
        ensureOpen();
        return optimizer;
    }

    @Override
    public LossFunction lossFunction() {
        ensureOpen();
        return lossFunction;
    }

    @Override
    public TrainerConfig config() {
        return config;
    }

    /** The loader of the training batches. */
    public DataLoader<TensorDataset.Sample> trainLoader() {
        return trainLoader;
    }

    /** The loader of the validation batches, or null when the trainer does not validate. */
    public DataLoader<TensorDataset.Sample> valLoader() {
        return valLoader;
    }

    /** The callbacks the trainer runs, in order; the list cannot be modified. */
    public List<Callback> callbacks() {
        return Collections.unmodifiableList(callbacks);
    }

    /** The learning-rate scheduler stepped after each batch and each epoch, or null for none. */
    public LRScheduler lrScheduler() {
        return lrScheduler;
    }

    public int currentEpoch() {
        return currentEpoch;
    }

    /**
     * The step count of the current run, or of the last one: the number of batches it has trained, the count
     * {@link Callback#onBatchEnd} receives. Every run starts again from 0.
     */
    public int currentStep() {
        return currentStep;
    }

    public float currentLoss() {
        return currentLoss;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Trainer has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
        }
    }

    @Override
    public String toString() {
        return "Trainer{" +
            "model=" + model.moduleName() +
            ", optimizer=" + optimizer.getClass().getSimpleName() +
            ", loss=" + lossFunction.name() +
            ", currentEpoch=" + currentEpoch +
            ", currentStep=" + currentStep +
            ", closed=" + closed +
            '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Module model;
        private Optimizer optimizer;
        private LossFunction lossFunction;
        private DataLoader<TensorDataset.Sample> trainLoader;
        private DataLoader<TensorDataset.Sample> valLoader;
        private List<Callback> callbacks = new ArrayList<>();
        private LRScheduler lrScheduler;
        private TrainerConfig config = TrainerConfig.defaults();

        public Builder model(Module model) {
            this.model = model;
            return this;
        }

        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Builder lossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder trainLoader(DataLoader<TensorDataset.Sample> trainLoader) {
            this.trainLoader = trainLoader;
            return this;
        }

        public Builder valLoader(DataLoader<TensorDataset.Sample> valLoader) {
            this.valLoader = valLoader;
            return this;
        }

        public Builder callbacks(List<Callback> callbacks) {
            this.callbacks = new ArrayList<>(callbacks);
            return this;
        }

        public Builder addCallback(Callback callback) {
            this.callbacks.add(callback);
            return this;
        }

        public Builder lrScheduler(LRScheduler lrScheduler) {
            this.lrScheduler = lrScheduler;
            return this;
        }

        public Builder config(TrainerConfig config) {
            this.config = config;
            return this;
        }

        public Trainer build() {
            Objects.requireNonNull(model, "Model cannot be null");
            Objects.requireNonNull(optimizer, "Optimizer cannot be null");
            Objects.requireNonNull(lossFunction, "LossFunction cannot be null");
            Objects.requireNonNull(trainLoader, "TrainLoader cannot be null");

            return new Trainer(model, optimizer, lossFunction, trainLoader, valLoader,
                callbacks, lrScheduler, config);
        }
    }
}

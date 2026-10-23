package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.LRScheduler;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.training.Callback;
import com.eonmux.neuralfree.training.EvalResult;
import com.eonmux.neuralfree.training.GradientClipping;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;
import com.eonmux.neuralfree.training.TrainingContext;
import com.eonmux.neuralfree.training.TrainingResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Training with loss scaling and reduced-precision emulation.
 *
 * <p><b>Precision.</b> Every framework op computes in FLOAT32, so FP16 and BF16 modes emulate the reduced format: the
 * model input is rounded to FLOAT16 or BFLOAT16 (round to nearest even) and widened back to FLOAT32 before the forward
 * pass, on the device when the train kernels are available. Parameters, gradients and optimizer state stay FLOAT32.</p>
 *
 * <p><b>Training step.</b> As in {@link Trainer}: the backward pass starts from the loss plus the auxiliary loss
 * terms the forward pass left pending ({@link Trainer#withAuxiliaryLoss}), multiplied by the current loss scale with
 * the recorded {@link TensorOps#mul}; the logged loss and the overflow decision use the loss alone. The gradients reach
 * the optimizer's parameters ({@link Trainer#appliedParameters}); the first backward pass of each {@link #train} call
 * must give one of them that requires grad a gradient, and a loss that does not require grad is refused, both with
 * {@link Trainer#NOT_CONNECTED}.</p>
 *
 * <p><b>Loss scaling and accumulation.</b> At each accumulation boundary, under {@link ComputationGraph#noGrad()}, the
 * gradients are first multiplied by {@code 1 / (scale * n)} and checked for non-finite values
 * ({@link GpuOptimizerOps#unscaleAndCheck} on the device, one flag readback), {@code n} being the number of batches
 * back-propagated since the previous step or overflow: the unscaled gradient is the mean of the accumulated batch
 * gradients, as {@link Trainer} applies it. They are then clipped as true gradients ({@link GradientClipping}), the
 * optimizer steps and the gradients are zeroed.</p>
 *
 * <p><b>Overflow.</b> A batch overflows when its loss is non-finite or, at the accumulation boundary, any unscaled
 * gradient is non-finite. On overflow the accumulated gradients are zeroed, the step is skipped and the count
 * {@code n} restarts; with dynamic scaling the scale is divided by the scale factor (not below the minimum), otherwise
 * a {@link RuntimeException} is thrown. With dynamic scaling, after {@code lossScaleWindow} overflow-free batches the
 * scale is multiplied by the factor (not above the maximum) once the next optimizer step has completed, so every
 * gradient of one accumulation window is computed and unscaled with the same scale.</p>
 *
 * <p><b>Learning-rate schedule.</b> As in {@link Trainer}, the scheduler steps after every batch
 * ({@link LRScheduler#step()}), whether or not the batch overflowed or ended an accumulation window, and after every
 * epoch ({@link LRScheduler#stepEpoch()}); the {@link TrainingResult} records the learning rate once per epoch, after
 * the epoch step.</p>
 *
 * <p><b>Tensor lifetime.</b> As in {@link Trainer}: each batch's collated, rounded, prediction, loss, auxiliary and
 * scaled-loss tensors are closed once the batch no longer needs them, and every node the batch recorded is removed
 * ({@link Trainer.Temporaries}); the backward pass releases its graph, and gradients are closed when they are zeroed
 * ({@link Parameter#zeroGrad()}); unscaling and clipping on the host close the gradients they replace. Evaluation and
 * {@link #predict} run without recording autograd operations and drop the auxiliary terms their forward passes
 * leave pending ({@link Module#clearAuxiliaryLoss()}). A training or evaluation pass that a failure or a stop request
 * leaves is ended ({@link Trainer#endPass}), so the batches its loader prepared ahead are released.</p>
 *
 * <p><b>Callbacks.</b> The callbacks run at the points {@link Trainer} runs them, with this trainer as their
 * {@link TrainingContext}: {@link Callback#onTrainBegin} and {@link Callback#onTrainEnd} around the run,
 * {@link Callback#onEpochBegin} and {@link Callback#onEpochEnd} around each epoch, {@link Callback#onBatchBegin} and
 * {@link Callback#onBatchEnd} around each batch (also a batch whose loss overflowed, with its non-finite loss), and
 * {@link Callback#onValidationBegin} and {@link Callback#onValidationEnd} around each validation. A
 * {@link #requestStop()} (such as {@link com.eonmux.neuralfree.training.EarlyStopping}'s) is honoured before the next
 * batch is fetched and before the next epoch starts.</p>
 *
 * <p><b>Progress and checkpoints.</b> As in {@link Trainer}: with {@link TrainerConfig#verbose()} the trainer writes
 * the progress log of {@link Trainer.ProgressLog} under this class's name (a step line of a batch whose loss
 * overflowed shows its non-finite loss). An epoch's training loss is the mean of its finite batch losses, or NaN
 * when it has none (every batch overflowed, or a stop was requested before its first batch), so that early stopping
 * and checkpoint selection never take it for an improvement. A configuration with
 * {@link TrainerConfig#checkpointEveryNEpochs()} positive needs a
 * {@link com.eonmux.neuralfree.training.ModelCheckpoint} callback, which then saves at that interval
 * ({@link TrainerConfig#validateCallbacks}, checked when the trainer is built).</p>
 *
 * <p><b>Base trainer.</b> Built with {@link Builder#baseTrainer}, the trainer takes its components from that
 * {@link Trainer}: the model, optimizer, loss function, training and validation loaders, callbacks, learning-rate
 * scheduler and {@link TrainerConfig} (accumulation, clipping, evaluation interval, progress log, checkpoint
 * interval). It then trains, evaluates and predicts with them exactly as one built from the same components:
 * reduced-precision input, dynamic loss scaling, unscaling before clipping and the overflow skip all apply, and the
 * callbacks receive this trainer as their {@link TrainingContext}, so a stop request (such as
 * {@link com.eonmux.neuralfree.training.EarlyStopping}'s) ends this trainer's run. The base trainer itself is not
 * run; it shares those components, so the two must not train at the same time.</p>
 *
 * <p><b>Closing.</b> {@link #close()} ends the use of this trainer only. It owns none of its components (the caller
 * created the model, optimizer, loaders and callbacks) and does not close them, nor a base trainer, which stays usable
 * by its owner.</p>
 */
public final class MixedPrecisionTrainer implements TrainingContext, AutoCloseable {
    public enum PrecisionMode {
        FP32,
        FP16,
        BF16
    }

    private final Module model;
    private final Optimizer optimizer;
    private final LossFunction lossFunction;
    private final DataLoader<TensorDataset.Sample> trainLoader;
    private final DataLoader<TensorDataset.Sample> valLoader;
    private final List<Callback> callbacks;
    private final LRScheduler lrScheduler;
    private final TrainerConfig config;
    private final PrecisionMode precisionMode;
    private final boolean dynamicLossScaling;
    private final float initialLossScale;
    private final float lossScaleFactor;
    private final int lossScaleWindow;
    private final float minLossScale;
    private final float maxLossScale;
    private final Trainer.ProgressLog progressLog;

    private float currentLossScale;
    private int goodSteps = 0;
    private int currentStep = 0;
    private boolean connectionVerified = false;
    private volatile boolean shouldStop = false;
    private volatile boolean closed = false;

    private MixedPrecisionTrainer(Builder builder, Components components) {
        this.model = components.model;
        this.optimizer = components.optimizer;
        this.lossFunction = components.lossFunction;
        this.trainLoader = components.trainLoader;
        this.valLoader = components.valLoader;
        this.callbacks = components.callbacks;
        this.lrScheduler = components.lrScheduler;
        this.config = components.config;
        this.precisionMode = builder.precisionMode != null ? builder.precisionMode : PrecisionMode.FP16;
        this.dynamicLossScaling = builder.dynamicLossScaling;
        this.initialLossScale = builder.initialLossScale;
        this.lossScaleFactor = builder.lossScaleFactor;
        this.lossScaleWindow = builder.lossScaleWindow;
        this.minLossScale = builder.minLossScale;
        this.maxLossScale = builder.maxLossScale;
        this.currentLossScale = initialLossScale;
        this.progressLog = new Trainer.ProgressLog(MixedPrecisionTrainer.class, config, callbacks);
    }

    public TrainingResult train(int epochs) {
        ensureOpen();
        if (epochs < 1) {
            throw new IllegalArgumentException("Epochs must be at least 1, got: " + epochs);
        }
        return trainWithMixedPrecision(epochs);
    }

    private TrainingResult trainWithMixedPrecision(int epochs) {
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
            for (Callback callback : callbacks) {
                callback.onEpochBegin(epoch, epochs);
            }

            float epochLoss = trainEpochMixedPrecision(epoch, epochs);

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

    private float trainEpochMixedPrecision(int epoch, int epochs) {
        float totalLoss = 0.0f;
        int batchCount = 0;
        int accumulationCounter = 0;
        int totalBatches = trainLoader.numBatches();

        DataLoader.Pass<TensorDataset.Sample> pass = trainLoader.pass();
        try {
            while (!shouldStop && pass.hasNext()) {
                List<TensorDataset.Sample> batch = pass.next();
                notifyBatchBegin(batch, totalBatches);

                float batchLoss = trainBatchMixedPrecision(batch);

                if (Float.isFinite(batchLoss)) {
                    totalLoss += batchLoss;
                    batchCount++;
                    accumulationCounter++;
                    if (dynamicLossScaling) {
                        goodSteps++;
                    }

                    if (accumulationCounter >= config.accumulationSteps()) {
                        applyAccumulatedGradients(accumulationCounter);
                        accumulationCounter = 0;
                    }
                } else {
                    handleOverflow("Loss overflow detected without dynamic loss scaling");
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
            Trainer.endPass(pass, e);
            throw e;
        }
        Trainer.endPass(pass, null);

        if (accumulationCounter > 0) {
            applyAccumulatedGradients(accumulationCounter);
        }

        return batchCount > 0 ? totalLoss / batchCount : Float.NaN;
    }

    /**
     * Runs the {@link Callback#onBatchBegin} callbacks for a fetched batch; when one fails, the batch's samples are
     * closed ({@link Trainer#closeSamples}) before the failure propagates, since no step will take them.
     */
    private void notifyBatchBegin(List<TensorDataset.Sample> batch, int totalBatches) {
        try {
            for (Callback callback : callbacks) {
                callback.onBatchBegin(currentStep, totalBatches);
            }
        } catch (RuntimeException | Error e) {
            Trainer.closeSamples(batch, e);
            throw e;
        }
    }

    /**
     * Forward and scaled backward pass of one batch; the batch tensors, the rounded input, the prediction, the loss,
     * the auxiliary terms and the scaled loss are closed before returning (the collated input as soon as its rounded
     * copy exists), and every node the batch recorded is removed.
     *
     * @return the unscaled loss without the auxiliary terms, read back for logging and the overflow decision
     * @throws IllegalStateException with {@link Trainer#NOT_CONNECTED} as {@link Trainer#backwardToParameters} throws
     */
    private float trainBatchMixedPrecision(List<TensorDataset.Sample> batch) {
        try (Trainer.Temporaries step = new Trainer.Temporaries(model)) {
            step.keepSamples(batch);
            model.clearAuxiliaryLoss();
            Tensor batchInput = step.keep(Trainer.collateInputs(batch));
            Tensor batchTarget = step.keep(Trainer.collateTargets(batch));

            Tensor input = step.keep(castToPrecision(batchInput));
            if (input != batchInput) {
                batchInput.close();
            }

            Tensor prediction = step.keep(model.forward(input));
            Tensor loss = step.keep(lossFunction.compute(prediction, batchTarget));

            float lossValue = loss.toFloatArray()[0];

            Tensor objective = Trainer.withAuxiliaryLoss(model, loss, step);
            Tensor scaledLoss = step.keep(scaleLoss(objective, currentLossScale));
            Trainer.backwardToParameters(scaledLoss, Trainer.appliedParameters(optimizer), !connectionVerified);
            connectionVerified = true;

            return lossValue;
        }
    }

    /**
     * The accumulation boundary over {@link Trainer#appliedParameters}, under {@link ComputationGraph#noGrad()}:
     * unscale the gradients of the {@code batches} batches back-propagated since the previous step or overflow to
     * their mean and check them, then on overflow skip the step ({@link #handleOverflow}); otherwise clip the true
     * gradients, step, zero and close the gradients (with every gradient unscaling or clipping replaced) and grow the
     * scale when its window is complete.
     */
    private void applyAccumulatedGradients(int batches) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            List<Parameter> params = Trainer.appliedParameters(optimizer);
            if (unscaleGradientsAndCheckOverflow(params, batches)) {
                handleOverflow("Gradient overflow detected without dynamic loss scaling");
                return;
            }
            if (config.gradientClippingEnabled()) {
                GradientClipping.clip(params, config);
            }
            optimizer.step();
            optimizer.zeroGrad();
        }
        if (dynamicLossScaling && goodSteps >= lossScaleWindow) {
            increaseLossScale();
            goodSteps = 0;
        }
    }

    /**
     * {@code tensor} rounded through the reduced format of the precision mode and widened back to FLOAT32 (the
     * precision emulation of the class description); the tensor itself in FP32 mode or when it is not FLOAT32.
     */
    private Tensor castToPrecision(Tensor tensor) {
        if (precisionMode == PrecisionMode.FP32 || tensor.dtype() != DType.FLOAT32) {
            return tensor;
        }
        
        DType reduced = precisionMode == PrecisionMode.BF16 ? DType.BFLOAT16 : DType.FLOAT16;
        Tensor rounded = tensor.to(reduced);
        try {
            return rounded.to(DType.FLOAT32);
        } finally {
            rounded.close();
        }
    }

    /**
     * The loss multiplied by {@code scale} with {@link TensorOps#mul}, recorded for autograd so its backward pass
     * scales every gradient.
     */
    private Tensor scaleLoss(Tensor loss, float scale) {
        return TensorOps.mul(loss, scale);
    }

    /**
     * Multiplies every gradient of {@code params} by {@code 1 / (scale * batches)}, the same factor on both paths
     * (one rounding when the scale is a power of two), and reports whether any result is non-finite: on the device in
     * place ({@link GpuOptimizerOps#unscaleAndCheck}, gradients shared between parameters prepared by
     * {@link GradientClipping#deviceGradients}), else on the host with a new gradient tensor per gradient slot. A slot
     * is unscaled once: a parameter whose current gradient this call installed (listed twice, or sharing its tensor
     * with an earlier parameter) is skipped. Every gradient is read before any is replaced, since replacing a
     * parameter's gradient closes the old tensor ({@link Parameter#setGrad}), which another parameter may also hold.
     */
    private boolean unscaleGradientsAndCheckOverflow(List<Parameter> params, int batches) {
        float invScale = 1.0f / (currentLossScale * batches);
        List<Tensor> deviceGrads = GradientClipping.deviceGradients(params);
        if (deviceGrads != null) {
            OptionalInt flag = GpuOptimizerOps.unscaleAndCheck(deviceGrads, invScale);
            if (flag.isPresent()) {
                return flag.getAsInt() != 0;
            }
        }
        Map<Tensor, float[]> values = new IdentityHashMap<>();
        for (Parameter param : params) {
            Tensor grad = param.grad();
            if (grad != null) {
                values.computeIfAbsent(grad, Tensor::toFloatArray);
            }
        }
        boolean overflow = false;
        Set<Tensor> installed = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Parameter param : params) {
            Tensor grad = param.grad();
            if (grad == null || installed.contains(grad)) {
                continue;
            }
            float[] gradData = values.get(grad).clone();
            for (int i = 0; i < gradData.length; i++) {
                gradData[i] *= invScale;
                overflow |= !Float.isFinite(gradData[i]);
            }
            Tensor next;
            try (Tensor flat = Tensor.fromArray(grad.backend(), gradData)) {
                next = flat.reshape(grad.shape());
            }
            try {
                param.setGrad(next);
            } catch (RuntimeException e) {
                next.close();
                throw e;
            }
            installed.add(next);
        }
        return overflow;
    }

    /**
     * The overflow rule: zero and close the accumulated gradients ({@link Parameter#zeroGrad()}), then reduce the loss
     * scale with dynamic scaling, else throw.
     *
     * @throws RuntimeException with {@code message} when dynamic loss scaling is off
     */
    private void handleOverflow(String message) {
        optimizer.zeroGrad();
        if (!dynamicLossScaling) {
            throw new RuntimeException(message);
        }
        handleLossOverflow();
    }

    private void handleLossOverflow() {
        currentLossScale = Math.max(minLossScale, currentLossScale / lossScaleFactor);
        goodSteps = 0;
    }

    private void increaseLossScale() {
        currentLossScale = Math.min(maxLossScale, currentLossScale * lossScaleFactor);
    }

    /**
     * Mean loss and accuracy over {@code loader} in evaluation mode, with the input rounded as in training and without
     * recording autograd operations; each batch's tensors, and the auxiliary loss terms its forward pass leaves
     * pending, are closed after use; the loss is the loss function's value alone. Accuracy as in
     * {@link Trainer#evaluate} ({@link Trainer#countCorrect}). The model is back in training mode afterwards.
     */
    public EvalResult evaluate(DataLoader<TensorDataset.Sample> loader) {
        ensureOpen();
        Objects.requireNonNull(loader, "Loader cannot be null");

        DataLoader.Pass<TensorDataset.Sample> pass = loader.pass();
        model.eval();

        float totalLoss = 0.0f;
        int totalSamples = 0;
        int correctPredictions = 0;
        boolean accuracyDefined = true;

        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            while (pass.hasNext()) {
                List<TensorDataset.Sample> batch = pass.next();
                try (Trainer.Temporaries step = new Trainer.Temporaries(model)) {
                    step.keepSamples(batch);
                    Tensor batchInput = step.keep(Trainer.collateInputs(batch));
                    Tensor batchTarget = step.keep(Trainer.collateTargets(batch));

                    Tensor input = step.keep(castToPrecision(batchInput));
                    if (input != batchInput) {
                        batchInput.close();
                    }

                    Tensor prediction = step.keep(model.forward(input));
                    model.clearAuxiliaryLoss();
                    Tensor loss = step.keep(lossFunction.compute(prediction, batchTarget));

                    float lossValue = loss.toFloatArray()[0];
                    int batchSize = batch.size();

                    totalLoss += lossValue * batchSize;
                    totalSamples += batchSize;

                    if (accuracyDefined) {
                        OptionalInt correct = Trainer.countCorrect(prediction, batchTarget, batchSize);
                        if (correct.isPresent()) {
                            correctPredictions += correct.getAsInt();
                        } else {
                            accuracyDefined = false;
                        }
                    }
                }
            }
        } catch (RuntimeException | Error e) {
            Trainer.endPass(pass, e);
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
     * The model's output for {@code input} rounded as in training, in evaluation mode and without recording autograd
     * operations. The result belongs to the caller; the rounded copy of the input is closed unless the model returned
     * it, and the auxiliary loss terms the forward pass leaves pending are closed. The model's previous mode is
     * restored.
     */
    public Tensor predict(Tensor input) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");

        boolean wasTraining = model.isTraining();
        model.eval();

        Tensor rounded = null;
        Tensor output = null;
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            rounded = castToPrecision(input);
            output = model.forward(rounded);
            model.clearAuxiliaryLoss();
            return output;
        } catch (RuntimeException e) {
            if (output != null && output != input) {
                try {
                    output.close();
                } catch (RuntimeException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        } finally {
            if (rounded != null && rounded != input && rounded != output) {
                rounded.close();
            }
            if (wasTraining) {
                model.train();
            }
        }
    }

    public float currentLossScale() {
        return currentLossScale;
    }

    public PrecisionMode precisionMode() {
        return precisionMode;
    }

    /** The model being trained (with a base trainer, that trainer's model). */
    @Override
    public Module model() {
        ensureOpen();
        return model;
    }

    /** The optimizer (with a base trainer, that trainer's optimizer). */
    @Override
    public Optimizer optimizer() {
        ensureOpen();
        return optimizer;
    }

    /** The loss function (with a base trainer, that trainer's loss function). */
    @Override
    public LossFunction lossFunction() {
        ensureOpen();
        return lossFunction;
    }

    /** The training configuration (with a base trainer, that trainer's configuration). */
    @Override
    public TrainerConfig config() {
        return config;
    }

    /**
     * Ends the current {@link #train} run of this trainer early, as {@link Trainer#requestStop()} does: no further
     * batch is fetched and no further epoch starts.
     */
    @Override
    public void requestStop() {
        shouldStop = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("MixedPrecisionTrainer has been closed");
        }
    }

    /**
     * Marks this trainer closed: later calls to {@link #train}, {@link #evaluate}, {@link #predict}, {@link #model()},
     * {@link #optimizer()} and {@link #lossFunction()} throw {@link IllegalStateException}. Nothing else is closed:
     * neither the components nor a base trainer (see the class description).
     */
    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "MixedPrecisionTrainer{" +
            "model=" + model.moduleName() +
            ", precisionMode=" + precisionMode +
            ", currentLossScale=" + currentLossScale +
            ", dynamicLossScaling=" + dynamicLossScaling +
            ", closed=" + closed +
            '}';
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Trainer baseTrainer;
        private Module model;
        private Optimizer optimizer;
        private LossFunction lossFunction;
        private DataLoader<TensorDataset.Sample> trainLoader;
        private DataLoader<TensorDataset.Sample> valLoader;
        private List<Callback> callbacks;
        private LRScheduler lrScheduler;
        private TrainerConfig config;
        private PrecisionMode precisionMode = PrecisionMode.FP16;
        private boolean dynamicLossScaling = true;
        private float initialLossScale = 65536.0f;
        private float lossScaleFactor = 2.0f;
        private int lossScaleWindow = 2000;
        private float minLossScale = 1.0f;
        private float maxLossScale = 32768.0f * 32768.0f;

        /**
         * Takes the model, optimizer, loss function, training and validation loaders, callbacks, learning-rate
         * scheduler and configuration of {@code baseTrainer} (read when {@link #build()} runs); the mixed-precision
         * settings of this builder apply to them. A builder with a base trainer takes none of those components
         * itself: {@link #build()} rejects them.
         */
        public Builder baseTrainer(Trainer baseTrainer) {
            this.baseTrainer = baseTrainer;
            return this;
        }

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

        /** Replaces the callbacks with a copy of {@code callbacks} (none when null). */
        public Builder callbacks(List<Callback> callbacks) {
            if (callbacks != null) {
                for (Callback callback : callbacks) {
                    Objects.requireNonNull(callback, "Callback cannot be null");
                }
            }
            this.callbacks = callbacks != null ? new ArrayList<>(callbacks) : null;
            return this;
        }

        public Builder addCallback(Callback callback) {
            Objects.requireNonNull(callback, "Callback cannot be null");
            if (this.callbacks == null) {
                this.callbacks = new ArrayList<>();
            }
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

        public Builder precisionMode(PrecisionMode precisionMode) {
            this.precisionMode = precisionMode;
            return this;
        }

        public Builder dynamicLossScaling(boolean dynamicLossScaling) {
            this.dynamicLossScaling = dynamicLossScaling;
            return this;
        }

        public Builder initialLossScale(float initialLossScale) {
            if (initialLossScale <= 0) {
                throw new IllegalArgumentException("Initial loss scale must be positive, got: " + initialLossScale);
            }
            this.initialLossScale = initialLossScale;
            return this;
        }

        public Builder lossScaleFactor(float lossScaleFactor) {
            if (lossScaleFactor <= 1.0f) {
                throw new IllegalArgumentException("Loss scale factor must be > 1.0, got: " + lossScaleFactor);
            }
            this.lossScaleFactor = lossScaleFactor;
            return this;
        }

        public Builder lossScaleWindow(int lossScaleWindow) {
            if (lossScaleWindow < 1) {
                throw new IllegalArgumentException("Loss scale window must be at least 1, got: " + lossScaleWindow);
            }
            this.lossScaleWindow = lossScaleWindow;
            return this;
        }

        public Builder minLossScale(float minLossScale) {
            if (minLossScale <= 0) {
                throw new IllegalArgumentException("Min loss scale must be positive, got: " + minLossScale);
            }
            this.minLossScale = minLossScale;
            return this;
        }

        public Builder maxLossScale(float maxLossScale) {
            if (maxLossScale <= 0) {
                throw new IllegalArgumentException("Max loss scale must be positive, got: " + maxLossScale);
            }
            this.maxLossScale = maxLossScale;
            return this;
        }

        /**
         * The trainer, with its components taken from the base trainer when one is set, else from this builder (no
         * callbacks, no scheduler and {@link TrainerConfig#defaults()} when those are not set).
         *
         * @throws NullPointerException     without a base trainer, when the model, optimizer, loss function or
         *                                  training loader is not set
         * @throws IllegalStateException    with a base trainer, when this builder also sets a component the base
         *                                  trainer supplies (model, optimizer, loss function, a loader, callbacks, a
         *                                  scheduler or a configuration), or when the base trainer has been closed
         * @throws IllegalArgumentException when the callbacks cannot carry out the configuration
         *                                  ({@link TrainerConfig#validateCallbacks})
         */
        public MixedPrecisionTrainer build() {
            if (baseTrainer == null) {
                return new MixedPrecisionTrainer(this, Components.fromBuilder(this));
            }
            List<String> conflicts = componentsSet();
            if (!conflicts.isEmpty()) {
                throw new IllegalStateException("MixedPrecisionTrainer with a base trainer takes its components from "
                    + "that trainer; remove the builder's " + String.join(", ", conflicts)
                    + " or set them on the base Trainer");
            }
            return new MixedPrecisionTrainer(this, Components.fromTrainer(baseTrainer));
        }

        /** The names of the components this builder sets itself, in builder order. */
        private List<String> componentsSet() {
            List<String> set = new ArrayList<>();
            if (model != null) {
                set.add("model");
            }
            if (optimizer != null) {
                set.add("optimizer");
            }
            if (lossFunction != null) {
                set.add("lossFunction");
            }
            if (trainLoader != null) {
                set.add("trainLoader");
            }
            if (valLoader != null) {
                set.add("valLoader");
            }
            if (callbacks != null && !callbacks.isEmpty()) {
                set.add("callbacks");
            }
            if (lrScheduler != null) {
                set.add("lrScheduler");
            }
            if (config != null) {
                set.add("config");
            }
            return set;
        }
    }

    /** The components a trainer runs on, resolved once when it is built. */
    private static final class Components {
        final Module model;
        final Optimizer optimizer;
        final LossFunction lossFunction;
        final DataLoader<TensorDataset.Sample> trainLoader;
        final DataLoader<TensorDataset.Sample> valLoader;
        final List<Callback> callbacks;
        final LRScheduler lrScheduler;
        final TrainerConfig config;

        private Components(Module model, Optimizer optimizer, LossFunction lossFunction,
                           DataLoader<TensorDataset.Sample> trainLoader, DataLoader<TensorDataset.Sample> valLoader,
                           List<Callback> callbacks, LRScheduler lrScheduler, TrainerConfig config) {
            this.model = Objects.requireNonNull(model, "Model cannot be null");
            this.optimizer = Objects.requireNonNull(optimizer, "Optimizer cannot be null");
            this.lossFunction = Objects.requireNonNull(lossFunction, "LossFunction cannot be null");
            this.trainLoader = Objects.requireNonNull(trainLoader, "TrainLoader cannot be null");
            this.valLoader = valLoader;
            this.callbacks = callbacks != null
                ? Collections.unmodifiableList(new ArrayList<>(callbacks))
                : Collections.emptyList();
            this.lrScheduler = lrScheduler;
            this.config = config != null ? config : TrainerConfig.defaults();
            this.config.validateCallbacks(this.callbacks);
        }

        static Components fromBuilder(Builder b) {
            return new Components(b.model, b.optimizer, b.lossFunction, b.trainLoader, b.valLoader, b.callbacks,
                b.lrScheduler, b.config);
        }

        /** @throws IllegalStateException when {@code base} has been closed ({@link Trainer#model()}) */
        static Components fromTrainer(Trainer base) {
            return new Components(base.model(), base.optimizer(), base.lossFunction(), base.trainLoader(),
                base.valLoader(), base.callbacks(), base.lrScheduler(), base.config());
        }
    }
}

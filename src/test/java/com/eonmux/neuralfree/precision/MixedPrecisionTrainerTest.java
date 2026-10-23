package com.eonmux.neuralfree.precision;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.optim.StepLR;
import com.eonmux.neuralfree.precision.MixedPrecisionTrainer.PrecisionMode;
import com.eonmux.neuralfree.tensor.HalfFloats;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.training.Callback;
import com.eonmux.neuralfree.training.EarlyStopping;
import com.eonmux.neuralfree.training.EvalResult;
import com.eonmux.neuralfree.training.LogCapture;
import com.eonmux.neuralfree.training.ModelCheckpoint;
import com.eonmux.neuralfree.training.ProgressLogger;
import com.eonmux.neuralfree.training.Trainer;
import com.eonmux.neuralfree.training.TrainerConfig;
import com.eonmux.neuralfree.training.TrainingContext;
import com.eonmux.neuralfree.training.TrainingResult;
import org.apache.logging.log4j.Level;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link MixedPrecisionTrainer}: reduced-precision emulation of the input, the boundary order (unscale and check, then
 * clip the true gradients, then step), the overflow rule for non-finite losses and gradients, loss-scale growth after
 * the step, evaluation accuracy, and the callback hooks with a stop request, on the host path and on each GPU API;
 * built from a base {@link Trainer}, the same loop runs on that trainer's components, configuration and callbacks.
 * Every value in these scenarios is exact in float, so both paths are compared bit for bit.
 */
@RunWith(Enclosed.class)
public class MixedPrecisionTrainerTest {

    /** {@code x * w} with one parameter {@code w} of shape {@code [1, n]}. */
    static final class ScaleModel extends Module {
        private final Parameter w;

        ScaleModel(ComputeBackend backend, float[] init) {
            Tensor flat = Tensor.fromArray(backend, init);
            try {
                this.w = registerParameter("w", new Parameter(flat.reshape(1, init.length)));
            } finally {
                flat.close();
            }
        }

        float[] weights() {
            return w.tensor().toFloatArray();
        }

        Parameter weight() {
            return w;
        }

        @Override
        public Tensor forward(Tensor input) {
            return TensorOps.mul(input, w.tensor());
        }
    }

    /** Returns its input. */
    static final class IdentityModel extends Module {
        @Override
        public Tensor forward(Tensor input) {
            return input;
        }
    }

    /** The sum of the prediction; the target is not used. */
    static final class SumLoss implements LossFunction {
        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            return TensorOps.sum(prediction);
        }

        @Override
        public String name() {
            return "sum";
        }
    }

    static DataLoader<TensorDataset.Sample> loader(ComputeBackend backend, float[][] inputs, float[][] targets) {
        return loader(backend, inputs, targets, inputs.length);
    }

    static DataLoader<TensorDataset.Sample> loader(ComputeBackend backend, float[][] inputs, float[][] targets,
                                                   int batchSize) {
        Tensor[] in = new Tensor[inputs.length];
        Tensor[] out = new Tensor[targets.length];
        for (int i = 0; i < inputs.length; i++) {
            in[i] = Tensor.fromArray(backend, inputs[i]);
            out[i] = Tensor.fromArray(backend, targets[i]);
        }
        return new DataLoader<>(new TensorDataset(in, out), batchSize, false);
    }

    static MixedPrecisionTrainer.Builder trainer(ComputeBackend backend, Module model, float[] input) {
        return MixedPrecisionTrainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.5f))
            .lossFunction(new SumLoss())
            .trainLoader(loader(backend, new float[][]{input}, new float[][]{{0f}}));
    }

    /**
     * A builder over a base {@link Trainer} of {@code model} with SGD (learning rate 0.5), {@link SumLoss}, one batch
     * holding {@code input} and {@code config}.
     */
    static MixedPrecisionTrainer.Builder viaBase(ComputeBackend backend, Module model, float[] input,
                                                 TrainerConfig config) {
        Trainer base = new Trainer(model, new SGD(model.parameters(), 0.5f), new SumLoss(),
            loader(backend, new float[][]{input}, new float[][]{{0f}}), null, null, null, config);
        return MixedPrecisionTrainer.builder().baseTrainer(base);
    }

    // ------------------------------------------------------------------ scenarios, run on any backend

    /** Clipping sees the unscaled gradient: clip(1024 * g / 1024) and not clip(1024 * g) / 1024. */
    static void clipsTrueGradients(ComputeBackend backend) {
        ScaleModel model = new ScaleModel(backend, new float[]{1f, 1f});
        MixedPrecisionTrainer t = trainer(backend, model, new float[]{3f, -4f})
            .precisionMode(PrecisionMode.FP16)
            .initialLossScale(1024f)
            .config(TrainerConfig.builder().gradientClipValue(1f).build())
            .build();
        t.train(1);
        TrainParity.assertBitIdentical("w - 0.5 * clip([3, -4], 1)", new float[]{0.5f, 1.5f}, model.weights());
        assertEquals(1024f, t.currentLossScale(), 0f);
        assertNull("gradients are zeroed after the step", model.weight().grad());
    }

    /** The scale grows after the step, so the step's gradients are unscaled with the scale they were made with. */
    static void growsTheScaleAfterTheStep(ComputeBackend backend) {
        ScaleModel model = new ScaleModel(backend, new float[]{1f, 1f});
        MixedPrecisionTrainer t = trainer(backend, model, new float[]{3f, -4f})
            .precisionMode(PrecisionMode.FP32)
            .initialLossScale(1024f)
            .lossScaleWindow(1)
            .build();
        t.train(1);
        TrainParity.assertBitIdentical("w - 0.5 * [3, -4]", new float[]{-0.5f, 3f}, model.weights());
        assertEquals(2048f, t.currentLossScale(), 0f);
    }

    /** A finite loss whose scaled gradient overflows: the step is skipped and the scale halves. */
    static void skipsTheStepOnGradientOverflow(ComputeBackend backend) {
        float[] w = {1e-3f, 1f};
        ScaleModel model = new ScaleModel(backend, w);
        MixedPrecisionTrainer t = trainer(backend, model, new float[]{1e35f, 0f})
            .precisionMode(PrecisionMode.FP32)
            .build();
        t.train(1);
        TrainParity.assertBitIdentical("parameters unchanged", w, model.weights());
        assertEquals(32768f, t.currentLossScale(), 0f);
        assertNull("gradients are zeroed on overflow", model.weight().grad());
    }

    /**
     * A non-finite loss: the step is skipped and the scale halves; the epoch, whose only batch overflowed, has no
     * training loss (NaN).
     */
    static void skipsTheStepOnLossOverflow(ComputeBackend backend) {
        float[] w = {1f, 1f};
        ScaleModel model = new ScaleModel(backend, w);
        MixedPrecisionTrainer t = trainer(backend, model, new float[]{3e38f, 3e38f})
            .precisionMode(PrecisionMode.FP32)
            .build();
        TrainingResult result = t.train(1);
        TrainParity.assertBitIdentical("parameters unchanged", w, model.weights());
        assertEquals(32768f, t.currentLossScale(), 0f);
        assertNull("gradients are zeroed on overflow", model.weight().grad());
        assertTrue("the epoch loss of an epoch without a finite batch loss is NaN",
            Float.isNaN(result.finalTrainLoss()));
    }

    /** Without dynamic scaling either overflow throws. */
    static void throwsOnOverflowWithoutDynamicScaling(ComputeBackend backend) {
        float[][] inputs = {{1e35f, 0f}, {3e38f, 3e38f}};
        float[][] weights = {{1e-3f, 1f}, {1f, 1f}};
        String[] messages = {"Gradient overflow", "Loss overflow"};
        for (int i = 0; i < inputs.length; i++) {
            ScaleModel model = new ScaleModel(backend, weights[i]);
            MixedPrecisionTrainer t = trainer(backend, model, inputs[i])
                .precisionMode(PrecisionMode.FP32)
                .dynamicLossScaling(false)
                .build();
            try {
                t.train(1);
                fail("expected an overflow exception for input " + i);
            } catch (RuntimeException e) {
                assertTrue(e.getMessage(), e.getMessage().contains(messages[i]));
            }
        }
    }

    /**
     * Built from a base trainer, the loss is scaled: the scaled gradient of a finite loss overflows, so the step is
     * skipped and the scale halves, where the base trainer alone would step to {@code w - 0.5 * [1e35, 0]}.
     */
    static void baseTrainerSkipsTheStepOnGradientOverflow(ComputeBackend backend) {
        float[] w = {1e-3f, 1f};
        ScaleModel model = new ScaleModel(backend, w);
        MixedPrecisionTrainer t = viaBase(backend, model, new float[]{1e35f, 0f}, TrainerConfig.defaults())
            .precisionMode(PrecisionMode.FP32)
            .build();
        t.train(1);
        TrainParity.assertBitIdentical("parameters unchanged", w, model.weights());
        assertEquals(32768f, t.currentLossScale(), 0f);
        assertNull("gradients are zeroed on overflow", model.weight().grad());
    }

    /**
     * Built from a base trainer, its configuration's clipping acts on the unscaled gradient, and the scale grows after
     * the step (the mixed-precision loop ran).
     */
    static void baseTrainerClipsWithItsConfig(ComputeBackend backend) {
        ScaleModel model = new ScaleModel(backend, new float[]{1f, 1f});
        TrainerConfig config = TrainerConfig.builder().gradientClipValue(1f).build();
        MixedPrecisionTrainer t = viaBase(backend, model, new float[]{3f, -4f}, config)
            .precisionMode(PrecisionMode.FP16)
            .initialLossScale(1024f)
            .lossScaleWindow(1)
            .build();
        assertSame(config, t.config());
        t.train(1);
        TrainParity.assertBitIdentical("w - 0.5 * clip([3, -4], 1)", new float[]{0.5f, 1.5f}, model.weights());
        assertEquals(2048f, t.currentLossScale(), 0f);
        assertNull("gradients are zeroed after the step", model.weight().grad());
    }

    /** FP16 and BF16 round the input through the reduced format; FP32 passes it through. */
    static void emulatesReducedPrecision(ComputeBackend backend) {
        float[] values = {1.0001f, 0.1f, 1e5f, -2.5f, 1.01f, 65519f};
        float[] half = new float[values.length];
        float[] bfloat = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            half[i] = HalfFloats.halfToFloat(HalfFloats.floatToHalf(values[i]));
            bfloat[i] = HalfFloats.bfloat16ToFloat(HalfFloats.floatToBFloat16(values[i]));
        }
        assertEquals(1.0f, half[0], 0f);
        assertEquals(Float.POSITIVE_INFINITY, half[2], 0f);
        assertEquals(1.0078125f, bfloat[4], 0f);
        assertEquals(99840f, bfloat[2], 0f);

        Tensor input = Tensor.fromArray(backend, values);
        try {
            TrainParity.assertBitIdentical("FP16", half, predict(backend, PrecisionMode.FP16, input));
            TrainParity.assertBitIdentical("BF16", bfloat, predict(backend, PrecisionMode.BF16, input));
            MixedPrecisionTrainer fp32 = trainer(backend, new IdentityModel(), new float[]{0f})
                .precisionMode(PrecisionMode.FP32)
                .build();
            assertSame(input, fp32.predict(input));
        } finally {
            input.close();
        }
    }

    private static float[] predict(ComputeBackend backend, PrecisionMode mode, Tensor input) {
        MixedPrecisionTrainer t = trainer(backend, new IdentityModel(), new float[]{0f}).precisionMode(mode).build();
        Tensor out = t.predict(input);
        try {
            return out.toFloatArray();
        } finally {
            out.close();
        }
    }

    /**
     * Accuracy for one-hot targets and class-index targets (one per sample); a single output per sample and a target
     * layout that matches neither rule have no accuracy definition.
     */
    static void countsAccuracy(ComputeBackend backend) {
        float[][] scores = {{0.1f, 0.7f, 0.2f}, {0.5f, 0.2f, 0.3f}, {0.2f, 0.2f, 0.6f}};
        assertAccuracy(backend, scores, new float[][]{{0f, 1f, 0f}, {0f, 0f, 1f}, {0f, 0f, 1f}}, 2f / 3f);
        assertAccuracy(backend, scores, new float[][]{{1f}, {2f}, {2f}}, 2f / 3f);
        assertAccuracy(backend, new float[][]{{0.3f}, {0.9f}, {-1f}}, new float[][]{{1f}, {0f}, {0.4f}}, null);
        assertAccuracy(backend, scores, new float[][]{{1f, 0f}, {1f, 0f}, {1f, 0f}}, null);
    }

    private static void assertAccuracy(ComputeBackend backend, float[][] predictions, float[][] targets,
                                       Float expected) {
        MixedPrecisionTrainer t = trainer(backend, new IdentityModel(), new float[]{0f})
            .precisionMode(PrecisionMode.FP32)
            .build();
        EvalResult result = t.evaluate(loader(backend, predictions, targets));
        assertEquals(predictions.length, result.numSamples());
        if (expected == null) {
            assertNull("accuracy is not defined for these tensors", result.metric("accuracy"));
        } else {
            Float accuracy = result.metric("accuracy");
            assertEquals(expected.floatValue(), accuracy == null ? Float.NaN : accuracy.floatValue(), 0f);
        }
    }

    /** Records every hook it receives and requests a stop at the end of batch {@code stopAtStep}. */
    static final class RecordingCallback implements Callback {
        final List<String> events = new ArrayList<>();
        private final int stopAtStep;
        TrainingContext context;

        RecordingCallback(int stopAtStep) {
            this.stopAtStep = stopAtStep;
        }

        @Override
        public void onTrainBegin(TrainingContext trainer) {
            context = trainer;
            events.add("trainBegin");
        }

        @Override
        public void onTrainEnd(TrainingResult result) {
            events.add("trainEnd " + result.totalSteps());
        }

        @Override
        public void onEpochBegin(int epoch, int totalEpochs) {
            events.add("epochBegin " + epoch + "/" + totalEpochs);
        }

        @Override
        public void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
            events.add("epochEnd " + epoch + " " + trainLoss + " " + valLoss);
        }

        @Override
        public void onBatchBegin(int batch, int totalBatches) {
            events.add("batchBegin " + batch + "/" + totalBatches);
        }

        @Override
        public void onBatchEnd(int batch, float loss) {
            events.add("batchEnd " + batch + " " + loss);
            if (batch == stopAtStep) {
                context.requestStop();
            }
        }

        @Override
        public void onValidationBegin(int epoch) {
            events.add("validationBegin " + epoch);
        }

        @Override
        public void onValidationEnd(int epoch, EvalResult result) {
            events.add("validationEnd " + epoch + " " + result.loss());
        }
    }

    /**
     * The hooks run in {@link Trainer}'s order with the trainer as context, and a stop requested after the second of
     * three batches ends the run before the third batch and the second epoch: {@code x * w} with {@code x = [1, 1]},
     * so each step subtracts {@code 0.5 * [1, 1]} from {@code w = [1, 1]}.
     */
    static void runsCallbacksAndHonoursStop(ComputeBackend backend) {
        ScaleModel model = new ScaleModel(backend, new float[]{1f, 1f});
        RecordingCallback recorder = new RecordingCallback(2);
        MixedPrecisionTrainer t = MixedPrecisionTrainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.5f))
            .lossFunction(new SumLoss())
            .trainLoader(callbackTrainLoader(backend))
            .valLoader(callbackValLoader(backend))
            .precisionMode(PrecisionMode.FP32)
            .addCallback(recorder)
            .build();
        assertCallbackRun(t, recorder, model);
    }

    /**
     * Built from a base trainer, the callbacks of that trainer run as in {@link #runsCallbacksAndHonoursStop}, with
     * the mixed-precision trainer as their context, and its validation loader is used.
     */
    static void baseTrainerRunsItsCallbacks(ComputeBackend backend) {
        ScaleModel model = new ScaleModel(backend, new float[]{1f, 1f});
        RecordingCallback recorder = new RecordingCallback(2);
        Trainer base = new Trainer(model, new SGD(model.parameters(), 0.5f), new SumLoss(),
            callbackTrainLoader(backend), callbackValLoader(backend), Collections.singletonList(recorder), null,
            TrainerConfig.defaults());
        MixedPrecisionTrainer t = MixedPrecisionTrainer.builder()
            .baseTrainer(base)
            .precisionMode(PrecisionMode.FP32)
            .build();
        assertCallbackRun(t, recorder, model);
        assertNotSame("the base trainer is not the context", base, recorder.context);
    }

    private static DataLoader<TensorDataset.Sample> callbackTrainLoader(ComputeBackend backend) {
        float[][] inputs = {{1f, 1f}, {1f, 1f}, {1f, 1f}};
        float[][] targets = {{0f}, {0f}, {0f}};
        return loader(backend, inputs, targets, 1);
    }

    private static DataLoader<TensorDataset.Sample> callbackValLoader(ComputeBackend backend) {
        return loader(backend, new float[][]{{2f, 4f}}, new float[][]{{0f}});
    }

    private static void assertCallbackRun(MixedPrecisionTrainer t, RecordingCallback recorder, ScaleModel model) {
        TrainingResult result = t.train(3);

        assertSame("the callbacks' context is the trainer", t, recorder.context);
        assertEquals(Arrays.asList(
            "trainBegin",
            "epochBegin 0/3",
            "batchBegin 0/3", "batchEnd 1 2.0",
            "batchBegin 1/3", "batchEnd 2 1.0",
            "validationBegin 0", "validationEnd 0 0.0",
            "epochEnd 0 1.5 0.0",
            "trainEnd 2"), recorder.events);
        TrainParity.assertBitIdentical("two steps of w - 0.5 * [1, 1]", new float[]{0f, 0f}, model.weights());
        assertEquals(2, result.totalSteps());
        assertEquals(1, result.trainLosses().size());
    }

    // ------------------------------------------------------------------ configuration settings

    private static final float[][] PROGRESS_INPUTS = {{1f, 2f, -3f}, {-0.5f, 0.25f, 1.5f}, {2f, -1f, 0.5f}};

    /**
     * A trainer of {@code x * w} with {@code w = [0.5, -1, 2]}, SGD (rate 0.5) and {@link SumLoss} over the three
     * {@link #PROGRESS_INPUTS} batches, in FP32 mode: the losses are {@code -7.5}, {@code 4.75} and {@code 4} in the
     * first epoch and strictly decrease from epoch to epoch.
     */
    static MixedPrecisionTrainer.Builder progressTrainer(ComputeBackend backend, TrainerConfig config) {
        ScaleModel model = new ScaleModel(backend, new float[]{0.5f, -1f, 2f});
        return MixedPrecisionTrainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.5f))
            .lossFunction(new SumLoss())
            .trainLoader(loader(backend, PROGRESS_INPUTS, new float[][]{{0f}, {0f}, {0f}}, 1))
            .precisionMode(PrecisionMode.FP32)
            .config(config);
    }

    /** The progress log of {@link Trainer.ProgressLog}, written under this trainer's class name. */
    static void logsProgressAsTheTrainerDoes(ComputeBackend backend) {
        try (LogCapture log = new LogCapture(MixedPrecisionTrainer.class)) {
            progressTrainer(backend, TrainerConfig.builder().logEveryNSteps(2).build()).build().train(1);
            assertEquals(Arrays.asList(
                "epoch 1/1 step 2: loss 4.75000, lr 0.500000",
                "epoch 1/1: loss 0.416667"), log.messages(Level.INFO, "epoch "));
        }
        try (LogCapture log = new LogCapture(MixedPrecisionTrainer.class)) {
            progressTrainer(backend, TrainerConfig.builder().logEveryNSteps(1).verbose(false).build()).build()
                .train(1);
            progressTrainer(backend, TrainerConfig.builder().logEveryNSteps(1).build())
                .addCallback(new ProgressLogger(1, 1, false, true)).build().train(1);
            assertEquals("off without verbose and beside a verbose ProgressLogger", Collections.emptyList(),
                log.messages(Level.INFO, "epoch "));
        }
    }

    /**
     * {@code checkpointEveryNEpochs = 2} saves epochs 1 and 3 through the checkpoint callback besides its best epoch
     * (mode "max" over a decreasing training loss: epoch 0), and is refused without a checkpoint callback.
     */
    static void savesCheckpointsEveryNEpochs(ComputeBackend backend) {
        TrainerConfig config = TrainerConfig.builder().checkpointEveryNEpochs(2).verbose(false).build();
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().saveBestOnly(ModelCheckpoint.MODE_MAX).build();
        progressTrainer(backend, config).addCallback(checkpoint).build().train(5);
        List<Integer> saved = new ArrayList<>();
        for (int epoch = 0; epoch < 5; epoch++) {
            if (checkpoint.getCheckpoint(epoch) != null) {
                saved.add(epoch);
            }
        }
        assertEquals(Arrays.asList(0, 1, 3), saved);

        try {
            progressTrainer(backend, config).build();
            fail("a checkpoint interval without a ModelCheckpoint callback must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("checkpointEveryNEpochs"));
        }
    }

    /**
     * EarlyStopping reads this trainer's configuration and model: without a validation loader it warns once and the
     * run goes on; monitoring the training loss with restore, the weights of the best epoch come back.
     */
    static void earlyStoppingUsesTheContext(ComputeBackend backend) {
        try (LogCapture log = new LogCapture(EarlyStopping.class)) {
            EarlyStopping unavailable = EarlyStopping.builder().patience(1).build();
            TrainingResult result = progressTrainer(backend, TrainerConfig.builder().verbose(false).build())
                .addCallback(unavailable).build().train(3);
            assertEquals(3, result.trainLosses().size());
            assertEquals(1, log.messages(Level.WARN).size());
        }
        EarlyStopping stopping = EarlyStopping.builder().monitor(EarlyStopping.MONITOR_LOSS)
            .mode(EarlyStopping.MODE_MAX).patience(1).restoreBestWeights(true).build();
        MixedPrecisionTrainer t = progressTrainer(backend, TrainerConfig.builder().verbose(false).build())
            .addCallback(stopping).build();
        TrainingResult result = t.train(5);
        assertEquals("epoch 1 does not raise the loss, so patience 1 stops there", 2, result.trainLosses().size());
        float[] afterEpoch0 = {0.5f, -1f, 2f};
        for (float[] x : PROGRESS_INPUTS) {
            for (int i = 0; i < x.length; i++) {
                afterEpoch0[i] -= 0.5f * x[i];
            }
        }
        TrainParity.assertBitIdentical("the weights at the end of epoch 0", afterEpoch0,
            ((ScaleModel) t.model()).weights());
    }

    // ------------------------------------------------------------------ runners

    /** The host path: a backend without train kernels. */
    public static class HostPath {
        private ComputeBackend host;

        @Before
        public void setUp() {
            host = TrainParity.hostBackend();
        }

        @Test
        public void clipsTrueGradients() {
            MixedPrecisionTrainerTest.clipsTrueGradients(host);
        }

        @Test
        public void growsTheScaleAfterTheStep() {
            MixedPrecisionTrainerTest.growsTheScaleAfterTheStep(host);
        }

        @Test
        public void skipsTheStepOnGradientOverflow() {
            MixedPrecisionTrainerTest.skipsTheStepOnGradientOverflow(host);
        }

        @Test
        public void skipsTheStepOnLossOverflow() {
            MixedPrecisionTrainerTest.skipsTheStepOnLossOverflow(host);
        }

        @Test
        public void throwsOnOverflowWithoutDynamicScaling() {
            MixedPrecisionTrainerTest.throwsOnOverflowWithoutDynamicScaling(host);
        }

        @Test
        public void emulatesReducedPrecision() {
            MixedPrecisionTrainerTest.emulatesReducedPrecision(host);
        }

        @Test
        public void countsAccuracy() {
            MixedPrecisionTrainerTest.countsAccuracy(host);
        }

        @Test
        public void runsCallbacksAndHonoursStop() {
            MixedPrecisionTrainerTest.runsCallbacksAndHonoursStop(host);
        }

        @Test
        public void baseTrainerSkipsTheStepOnGradientOverflow() {
            MixedPrecisionTrainerTest.baseTrainerSkipsTheStepOnGradientOverflow(host);
        }

        @Test
        public void baseTrainerClipsWithItsConfig() {
            MixedPrecisionTrainerTest.baseTrainerClipsWithItsConfig(host);
        }

        @Test
        public void baseTrainerRunsItsCallbacks() {
            MixedPrecisionTrainerTest.baseTrainerRunsItsCallbacks(host);
        }

        @Test
        public void logsProgressAsTheTrainerDoes() {
            MixedPrecisionTrainerTest.logsProgressAsTheTrainerDoes(host);
        }

        @Test
        public void savesCheckpointsEveryNEpochs() {
            MixedPrecisionTrainerTest.savesCheckpointsEveryNEpochs(host);
        }

        @Test
        public void earlyStoppingUsesTheContext() {
            MixedPrecisionTrainerTest.earlyStoppingUsesTheContext(host);
        }

        /** Every component the base trainer supplies is refused when the builder sets it too, naming it. */
        @Test
        public void rejectsComponentsBesideABaseTrainer() {
            ScaleModel model = new ScaleModel(host, new float[]{1f});
            SGD optimizer = new SGD(model.parameters(), 0.5f);
            Map<String, Consumer<MixedPrecisionTrainer.Builder>> settings = new LinkedHashMap<>();
            settings.put("model", b -> b.model(model));
            settings.put("optimizer", b -> b.optimizer(optimizer));
            settings.put("lossFunction", b -> b.lossFunction(new SumLoss()));
            settings.put("trainLoader", b -> b.trainLoader(loader(host, new float[][]{{1f}}, new float[][]{{0f}})));
            settings.put("valLoader", b -> b.valLoader(loader(host, new float[][]{{1f}}, new float[][]{{0f}})));
            settings.put("callbacks", b -> b.addCallback(new RecordingCallback(1)));
            settings.put("lrScheduler", b -> b.lrScheduler(new StepLR(optimizer, 1)));
            settings.put("config", b -> b.config(TrainerConfig.defaults()));
            for (Map.Entry<String, Consumer<MixedPrecisionTrainer.Builder>> setting : settings.entrySet()) {
                MixedPrecisionTrainer.Builder builder = viaBase(host, model, new float[]{1f}, TrainerConfig.defaults());
                setting.getValue().accept(builder);
                try {
                    builder.build();
                    fail("expected the builder's " + setting.getKey() + " to be rejected beside a base trainer");
                } catch (IllegalStateException e) {
                    assertTrue(e.getMessage(), e.getMessage().contains(setting.getKey()));
                }
            }
        }

        @Test(expected = IllegalStateException.class)
        public void rejectsAClosedBaseTrainer() {
            ScaleModel model = new ScaleModel(host, new float[]{1f});
            Trainer base = new Trainer(model, new SGD(model.parameters(), 0.5f), new SumLoss(),
                loader(host, new float[][]{{1f}}, new float[][]{{0f}}));
            base.close();
            MixedPrecisionTrainer.builder().baseTrainer(base).build();
        }

        /** The components are the base trainer's; closing the trainer leaves the base trainer open. */
        @Test
        public void takesTheComponentsOfTheBaseTrainer() {
            ScaleModel model = new ScaleModel(host, new float[]{1f});
            SGD optimizer = new SGD(model.parameters(), 0.5f);
            SumLoss loss = new SumLoss();
            TrainerConfig config = TrainerConfig.builder().accumulationSteps(2).build();
            Trainer base = new Trainer(model, optimizer, loss, loader(host, new float[][]{{1f}}, new float[][]{{0f}}),
                null, null, null, config);
            MixedPrecisionTrainer t = MixedPrecisionTrainer.builder().baseTrainer(base).build();
            assertSame(model, t.model());
            assertSame(optimizer, t.optimizer());
            assertSame(loss, t.lossFunction());
            assertSame(config, t.config());
            assertEquals(PrecisionMode.FP16, t.precisionMode());

            t.close();
            assertSame("the base trainer stays open", model, base.model());
            try {
                t.model();
                fail("expected the closed trainer to refuse its accessors");
            } catch (IllegalStateException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("closed"));
            }
        }
    }

    /** The device path on each GPU API with the train kernels available. */
    @RunWith(Parameterized.class)
    public static class Device {

        @Parameterized.Parameters(name = "{0}")
        public static Collection<Object[]> data() {
            return TrainParity.apis();
        }

        private final TrainParity.Api api;
        private TrainParity.Session session;
        private ComputeBackend gpu;

        public Device(TrainParity.Api api) {
            this.api = api;
        }

        @Before
        public void setUp() {
            session = TrainParity.open(api);
            gpu = session.backend();
        }

        @After
        public void tearDown() {
            session.close();
        }

        @Test
        public void clipsTrueGradients() {
            MixedPrecisionTrainerTest.clipsTrueGradients(gpu);
        }

        @Test
        public void growsTheScaleAfterTheStep() {
            MixedPrecisionTrainerTest.growsTheScaleAfterTheStep(gpu);
        }

        @Test
        public void skipsTheStepOnGradientOverflow() {
            MixedPrecisionTrainerTest.skipsTheStepOnGradientOverflow(gpu);
        }

        @Test
        public void skipsTheStepOnLossOverflow() {
            MixedPrecisionTrainerTest.skipsTheStepOnLossOverflow(gpu);
        }

        @Test
        public void throwsOnOverflowWithoutDynamicScaling() {
            MixedPrecisionTrainerTest.throwsOnOverflowWithoutDynamicScaling(gpu);
        }

        @Test
        public void emulatesReducedPrecision() {
            MixedPrecisionTrainerTest.emulatesReducedPrecision(gpu);
        }

        @Test
        public void countsAccuracy() {
            MixedPrecisionTrainerTest.countsAccuracy(gpu);
        }

        @Test
        public void runsCallbacksAndHonoursStop() {
            MixedPrecisionTrainerTest.runsCallbacksAndHonoursStop(gpu);
        }

        @Test
        public void baseTrainerSkipsTheStepOnGradientOverflow() {
            MixedPrecisionTrainerTest.baseTrainerSkipsTheStepOnGradientOverflow(gpu);
        }

        @Test
        public void baseTrainerClipsWithItsConfig() {
            MixedPrecisionTrainerTest.baseTrainerClipsWithItsConfig(gpu);
        }

        @Test
        public void baseTrainerRunsItsCallbacks() {
            MixedPrecisionTrainerTest.baseTrainerRunsItsCallbacks(gpu);
        }
    }
}

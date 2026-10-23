package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.training.TrainerTest.SumLoss;
import org.apache.logging.log4j.Level;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EarlyStopping} with Keras semantics, run by {@link Trainer} on {@link TrainParity#hostBackend()}. The model is
 * {@code y = x * w} with {@code w0 = [0.5, -1, 2]} and a persistent buffer counting its training-mode forward passes;
 * the loss is the sum, the optimizer SGD with rate 0.5, and each epoch trains one batch {@code x = [1, 2, -3]}, so
 * after epoch {@code e} the weights are {@code w0 - 0.5 (e + 1) x}: {@code [0, -2, 3.5]}, {@code [-0.5, -3, 5]},
 * {@code [-1, -4, 6.5]}. The validation sample {@code [-1, -1, 0]} gives the validation losses 2, 3.5 and 5 after
 * epochs 0, 1 and 2: the best epoch is 0. The training losses (before each step) are -7.5, -14.5, -21.5, -28.5.
 */
public class EarlyStoppingTest {
    private static final float[] W0 = {0.5f, -1.0f, 2.0f};
    private static final float[] TRAIN_X = {1.0f, 2.0f, -3.0f};
    private static final float[] VAL_X = {-1.0f, -1.0f, 0.0f};
    private static final float[] AFTER_EPOCH_0 = {0.0f, -2.0f, 3.5f};
    private static final float[] AFTER_EPOCH_2 = {-1.0f, -4.0f, 6.5f};

    private ComputeBackend host;
    private TensorDataset train;
    private TensorDataset validation;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        host = TrainParity.hostBackend();
        train = dataset(host, TRAIN_X);
        validation = dataset(host, VAL_X);
    }

    @After
    public void tearDown() {
        validation.close();
        train.close();
        host.close();
    }

    // ------------------------------------------------------------------ restoring the best weights

    /** Stopped after patience 2 at epoch 2: parameters and buffer return to the end of epoch 0. */
    @Test
    public void restoresTheParametersAndBuffersOfTheBestEpochWhenStopping() {
        try (CountingModel model = new CountingModel(host);
             LogCapture log = new LogCapture(EarlyStopping.class)) {
            EarlyStopping stopping = EarlyStopping.builder().patience(2).restoreBestWeights(true).build();

            TrainingResult result = trainer(model, stopping, true).train(5);

            assertEquals("stopped at the end of epoch 2", 3, result.trainLosses().size());
            assertEquals(2, stopping.stoppedEpoch());
            assertEquals(0, stopping.bestEpoch());
            assertEquals(2.0f, stopping.bestLoss(), 0f);
            assertArrayEquals("the parameters of epoch 0", AFTER_EPOCH_0, model.weights(), 0f);
            assertEquals("the buffer of epoch 0", 1.0f, model.batches(), 0f);
            assertEquals(Collections.singletonList("Restoring model weights from the end of the best epoch: 1"),
                log.messages(Level.INFO, "Restoring"));
            assertEquals(1, log.messages(Level.INFO, "Epoch 3: early stopping").size());
            assertEquals(0, ComputationGraph.current().size());
        }
    }

    /** A run that ends without stopping also restores the best epoch, as Keras' {@code on_train_end} does. */
    @Test
    public void restoresTheBestEpochWhenTheRunEndsWithoutStopping() {
        try (CountingModel model = new CountingModel(host)) {
            EarlyStopping stopping = EarlyStopping.builder().patience(10).restoreBestWeights(true).build();

            trainer(model, stopping, true).train(3);

            assertEquals(-1, stopping.stoppedEpoch());
            assertArrayEquals(AFTER_EPOCH_0, model.weights(), 0f);
            assertEquals(1.0f, model.batches(), 0f);
        }
    }

    @Test
    public void withoutRestoreTheLastWeightsStay() {
        try (CountingModel model = new CountingModel(host)) {
            EarlyStopping stopping = new EarlyStopping(2);

            trainer(model, stopping, true).train(5);

            assertEquals(2, stopping.stoppedEpoch());
            assertArrayEquals(AFTER_EPOCH_2, model.weights(), 0f);
            assertEquals(3.0f, model.batches(), 0f);
        }
    }

    // ------------------------------------------------------------------ the monitored value

    /**
     * Without a validation loader the validation loss is never available: one WARN per run, training runs every
     * epoch, nothing improves and nothing is restored.
     */
    @Test
    public void anUnavailableValidationLossWarnsOncePerRunAndDoesNotStop() {
        try (CountingModel model = new CountingModel(host);
             LogCapture log = new LogCapture(EarlyStopping.class)) {
            EarlyStopping stopping = EarlyStopping.builder().patience(1).restoreBestWeights(true).build();
            Trainer trainer = trainer(model, stopping, false);

            TrainingResult result = trainer.train(3);

            assertEquals("every epoch ran", 3, result.trainLosses().size());
            assertEquals(-1, stopping.stoppedEpoch());
            assertNull(stopping.bestLoss());
            assertEquals(0, stopping.waitCount());
            assertArrayEquals("nothing was restored", AFTER_EPOCH_2, model.weights(), 0f);
            List<String> warnings = log.messages(Level.WARN);
            assertEquals(1, warnings.size());
            assertTrue(warnings.get(0), warnings.get(0).startsWith(
                "Early stopping conditioned on metric 'val_loss' which is not available"));

            trainer.train(1);
            assertEquals("a new run warns again", 2, log.messages(Level.WARN).size());
        }
    }

    /** An epoch that does not validate because of the evaluation interval is skipped without a warning. */
    @Test
    public void anEpochWithoutScheduledValidationIsSkippedSilently() {
        try (CountingModel model = new CountingModel(host);
             LogCapture log = new LogCapture(EarlyStopping.class)) {
            EarlyStopping stopping = new EarlyStopping(1);
            Trainer trainer = Trainer.builder()
                .model(model)
                .optimizer(new SGD(model.parameters(), 0.5f))
                .lossFunction(new SumLoss())
                .trainLoader(new DataLoader<>(train, 1, false))
                .valLoader(new DataLoader<>(validation, 1, false))
                .addCallback(stopping)
                .config(TrainerConfig.builder().evalEveryNEpochs(2).verbose(false).build())
                .build();

            trainer.train(2);

            assertEquals(Collections.emptyList(), log.messages(Level.WARN));
            assertEquals("epoch 1 validated: x_v . [-0.5, -3, 5] = 3.5", 1, stopping.bestEpoch());
            assertEquals(3.5f, stopping.bestLoss(), 0f);
        }
    }

    /**
     * Monitoring the training loss with minDelta 10: epoch 1 lowers it by 7 (from -7.5 to -14.5), less than minDelta,
     * so it does not improve and patience 1 stops the run; with minDelta 0 every epoch improves.
     */
    @Test
    public void monitorsTheTrainingLossWithMinDelta() {
        try (CountingModel model = new CountingModel(host)) {
            EarlyStopping stopping = EarlyStopping.builder().monitor(EarlyStopping.MONITOR_LOSS).patience(1)
                .minDelta(10f).build();

            TrainingResult result = trainer(model, stopping, false).train(4);

            assertEquals(2, result.trainLosses().size());
            assertEquals(1, stopping.stoppedEpoch());
            assertEquals(-7.5f, stopping.bestLoss(), 0f);
        }
        try (CountingModel model = new CountingModel(host)) {
            EarlyStopping stopping = EarlyStopping.builder().monitor(EarlyStopping.MONITOR_LOSS).patience(1).build();

            trainer(model, stopping, false).train(4);

            assertEquals(-1, stopping.stoppedEpoch());
            assertEquals(3, stopping.bestEpoch());
            assertEquals(-28.5f, stopping.bestLoss(), 0f);
        }
    }

    // ------------------------------------------------------------------ callback rules without a trainer

    /** NaN never improves, and the patience can only stop the run after the first epoch. */
    @Test
    public void nanNeverImprovesAndTheFirstEpochNeverStops() {
        EarlyStopping stopping = new EarlyStopping(1);
        stopping.onTrainBegin(null);

        stopping.onEpochEnd(0, 0f, Float.NaN);
        assertEquals(1, stopping.waitCount());
        assertEquals(-1, stopping.stoppedEpoch());
        assertNull(stopping.bestLoss());

        stopping.onEpochEnd(1, 0f, Float.NaN);
        assertEquals(1, stopping.stoppedEpoch());
    }

    /** In mode "max" with minDelta, a value must exceed the best by more than minDelta. */
    @Test
    public void maxModeAppliesMinDeltaAboveTheBest() {
        EarlyStopping stopping = new EarlyStopping(5, 0.5f, EarlyStopping.MODE_MAX, false);
        stopping.onTrainBegin(null);

        stopping.onEpochEnd(0, 0f, 1.0f);
        stopping.onEpochEnd(1, 0f, 1.5f);
        assertEquals("1.5 - 0.5 is not above 1", 0, stopping.bestEpoch());
        assertEquals(1, stopping.waitCount());
        stopping.onEpochEnd(2, 0f, 1.75f);
        assertEquals(2, stopping.bestEpoch());
        assertEquals(0, stopping.waitCount());
    }

    @Test
    public void restoringNeedsATrainer() {
        EarlyStopping stopping = EarlyStopping.builder().restoreBestWeights(true).build();
        stopping.onTrainBegin(null);
        assertThrows(IllegalStateException.class, () -> stopping.onEpochEnd(0, 0f, 1f));
    }

    @Test
    public void settingsAndDefaults() {
        EarlyStopping defaults = EarlyStopping.builder().build();
        assertEquals(EarlyStopping.MONITOR_VAL_LOSS, defaults.monitor());
        assertEquals(10, defaults.patience());
        assertEquals(0f, defaults.minDelta(), 0f);
        assertEquals(EarlyStopping.MODE_MIN, defaults.mode());
        assertFalse(defaults.restoreBestWeights());
        assertEquals("the legacy constructors monitor the validation loss", EarlyStopping.MONITOR_VAL_LOSS,
            new EarlyStopping(5, 0.1f, "max", true).monitor());
        assertTrue(defaults.toString().contains("monitor='val_loss'"));

        assertThrows(IllegalArgumentException.class, () -> EarlyStopping.builder().monitor("accuracy").build());
        assertThrows(IllegalArgumentException.class, () -> EarlyStopping.builder().minDelta(Float.NaN).build());
        assertThrows(IllegalArgumentException.class, () -> EarlyStopping.builder().minDelta(-1f).build());
        assertThrows(IllegalArgumentException.class, () -> EarlyStopping.builder().patience(0).build());
        assertThrows(IllegalArgumentException.class, () -> EarlyStopping.builder().mode("auto").build());
        assertThrows(NullPointerException.class, () -> EarlyStopping.builder().monitor(null));
    }

    // ------------------------------------------------------------------ fixtures

    private Trainer trainer(CountingModel model, EarlyStopping stopping, boolean validate) {
        Trainer.Builder builder = Trainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), 0.5f))
            .lossFunction(new SumLoss())
            .trainLoader(new DataLoader<>(train, 1, false))
            .addCallback(stopping)
            .config(TrainerConfig.builder().verbose(false).build());
        if (validate) {
            builder.valLoader(new DataLoader<>(validation, 1, false));
        }
        return builder.build();
    }

    private static TensorDataset dataset(ComputeBackend backend, float[] x) {
        return new TensorDataset(new Tensor[]{Tensor.fromArray(backend, x)},
            new Tensor[]{Tensor.fromArray(backend, new float[]{0.0f})});
    }

    /**
     * {@code x * w} with the parameter {@code w = w0} and the persistent buffer {@code batches}, which every
     * training-mode forward pass increments (state that, like BatchNorm's running statistics, the optimizer does
     * not update).
     */
    static final class CountingModel extends Module {
        private final Parameter w;

        CountingModel(ComputeBackend backend) {
            this.w = registerParameter("w", new Parameter(Tensor.fromArray(backend, W0)));
            registerBuffer("batches", Tensor.fromArray(backend, new float[]{0.0f}));
        }

        float[] weights() {
            return w.tensor().toFloatArray();
        }

        float batches() {
            return buffer("batches").toFloatArray()[0];
        }

        @Override
        public Tensor forward(Tensor input) {
            if (isTraining()) {
                try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                    Tensor counter = buffer("batches");
                    GpuOptimizerOps.storeFloats(counter, new float[]{counter.toFloatArray()[0] + 1.0f});
                }
            }
            return TensorOps.mul(input, w.tensor());
        }
    }
}

package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.optim.LRScheduler;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import com.eonmux.neuralfree.training.TrainerTest.ScaleModel;
import com.eonmux.neuralfree.training.TrainerTest.SumLoss;
import org.apache.logging.log4j.Level;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@link TrainerConfig} settings {@link Trainer} carries out: the progress log ({@code verbose},
 * {@code logEveryNSteps}, {@link Trainer.ProgressLog}) and the checkpoint interval ({@code checkpointEveryNEpochs})
 * with its {@link ModelCheckpoint} callback. On {@link TrainParity#hostBackend()} with the model {@code y = x * w} and
 * the sum loss, so every logged loss is known exactly: with {@code w0 = [0.5, -1, 2]}, the three rows of
 * {@link #XS} and SGD, the first batch's loss is {@code -7.5}, the step leaves {@code w1 = w0 - lr0 * XS[0]}, and the
 * second batch's loss is {@code XS[1] . w1 = 4.75}.
 */
public class TrainerSettingsTest {
    static final float[] W0 = {0.5f, -1.0f, 2.0f};
    static final float[][] XS = {{1.0f, 2.0f, -3.0f}, {-0.5f, 0.25f, 1.5f}, {2.0f, -1.0f, 0.5f}};

    private ComputeBackend host;
    private TensorDataset data;

    @Before
    public void setUp() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
        host = TrainParity.hostBackend();
        data = dataset(host);
    }

    @After
    public void tearDown() {
        data.close();
        host.close();
    }

    // ------------------------------------------------------------------ progress log

    /**
     * Step lines every second step with the learning rate the batch trained with (the scheduler halves it after the
     * first batch, then sets {@code 0.5 / (1 + steps)}), and an epoch line per epoch: the first epoch's losses are
     * {@code -7.5}, {@code 4.75} and {@code XS[2] . (w1 - 0.25 * XS[1]) = 3.875}, mean {@code 0.375}.
     */
    @Test
    public void verboseLogsEveryNStepsAndEveryEpoch() {
        try (ScaleModel model = new ScaleModel(host, W0, false);
             LogCapture log = new LogCapture(Trainer.class)) {
            SGD sgd = new SGD(model.parameters(), 0.5f);
            Trainer trainer = builder(model, sgd)
                .lrScheduler(new InverseStepLR(sgd))
                .config(TrainerConfig.builder().logEveryNSteps(2).build())
                .build();

            trainer.train(2);

            List<String> steps = log.messages(Level.INFO, "epoch ");
            assertEquals(Arrays.asList(
                "epoch 1/2 step 2: loss 4.75000, lr 0.250000",
                "epoch 1/2: loss 0.375000"), steps.subList(0, 2));
            assertEquals("steps 2, 4, 6 and two epoch lines", 5, steps.size());
            assertTrue(steps.get(2), steps.get(2).startsWith("epoch 2/2 step 4: loss "));
            assertTrue(steps.get(2), steps.get(2).endsWith(", lr 0.125000"));
            assertTrue(steps.get(3), steps.get(3).startsWith("epoch 2/2 step 6: loss "));
            assertTrue(steps.get(4), steps.get(4).startsWith("epoch 2/2: loss "));
            assertEquals(0, ComputationGraph.current().size());
        }
    }

    /** A validation loss joins the epoch line of an epoch that validated. */
    @Test
    public void theEpochLineCarriesTheValidationLoss() {
        try (ScaleModel model = new ScaleModel(host, W0, false);
             LogCapture log = new LogCapture(Trainer.class)) {
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .valLoader(new DataLoader<>(data, 3, false))
                .config(TrainerConfig.builder().logEveryNSteps(100).build())
                .build();

            trainer.train(1);

            List<String> lines = log.messages(Level.INFO, "epoch ");
            assertEquals(1, lines.size());
            assertTrue(lines.get(0), lines.get(0).matches("epoch 1/1: loss \\S+, val_loss \\S+"));
        }
    }

    @Test
    public void withoutVerboseNothingIsLogged() {
        try (ScaleModel model = new ScaleModel(host, W0, false);
             LogCapture log = new LogCapture(Trainer.class)) {
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .config(TrainerConfig.builder().logEveryNSteps(1).verbose(false).build())
                .build();

            trainer.train(1);

            assertEquals(Collections.emptyList(), log.messages(Level.INFO, "epoch "));
            assertFalse(trainer.config().verbose());
        }
    }

    /** A verbose ProgressLogger reports the progress itself, so the trainer's log stays off; a silent one does not. */
    @Test
    public void aVerboseProgressLoggerReplacesTheTrainersLog() {
        try (ScaleModel model = new ScaleModel(host, W0, false);
             LogCapture log = new LogCapture(Trainer.class)) {
            TrainerConfig config = TrainerConfig.builder().logEveryNSteps(1).build();
            Trainer withLogger = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(new ProgressLogger(1, 1, false, true))
                .config(config)
                .build();
            withLogger.train(1);
            assertEquals(Collections.emptyList(), log.messages(Level.INFO, "epoch "));

            Trainer withSilentLogger = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(new ProgressLogger(1, 1, false, false))
                .config(config)
                .build();
            withSilentLogger.train(1);
            assertEquals("three step lines and one epoch line", 4, log.messages(Level.INFO, "epoch ").size());
        }
    }

    /** Every run counts its steps from the start, as the step lines, the result and {@code currentStep()} show. */
    @Test
    public void everyRunCountsItsStepsAfresh() {
        try (ScaleModel model = new ScaleModel(host, W0, false);
             LogCapture log = new LogCapture(Trainer.class)) {
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .config(TrainerConfig.builder().logEveryNSteps(3).build())
                .build();

            assertEquals(3, trainer.train(1).totalSteps());
            assertEquals(3, trainer.train(1).totalSteps());

            assertEquals(3, trainer.currentStep());
            List<String> steps = log.messages(Level.INFO, "epoch 1/1 step ");
            assertEquals(2, steps.size());
            assertTrue(steps.get(0), steps.get(0).startsWith("epoch 1/1 step 3: loss "));
            assertTrue(steps.get(1), steps.get(1).startsWith("epoch 1/1 step 3: loss "));
        }
    }

    @Test
    public void theProgressLogFollowsTheConfiguration() {
        List<Callback> none = Collections.emptyList();
        TrainerConfig every3 = TrainerConfig.builder().logEveryNSteps(3).build();
        Trainer.ProgressLog log = new Trainer.ProgressLog(Trainer.class, every3, none);
        try (LogCapture capture = new LogCapture(Trainer.class)) {
            assertTrue(log.enabled());
            assertFalse(log.logsStep(2));
            assertTrue(log.logsStep(3));
            assertTrue(log.logsStep(6));
        }
        Trainer.ProgressLog silent = new Trainer.ProgressLog(Trainer.class,
            TrainerConfig.builder().verbose(false).build(), none);
        assertFalse(silent.enabled());
        assertFalse(silent.logsStep(10));
    }

    // ------------------------------------------------------------------ checkpoint interval

    /**
     * With {@code checkpointEveryNEpochs = 2} the checkpoint callback saves epochs 1 and 3 besides its best epoch: in
     * mode "max" the monitored training loss (strictly decreasing under SGD) improves only in epoch 0.
     */
    @Test
    public void checkpointEveryNEpochsSavesThroughTheCheckpointCallback() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            ModelCheckpoint checkpoint = ModelCheckpoint.builder().saveBestOnly(ModelCheckpoint.MODE_MAX).build();
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(checkpoint)
                .config(TrainerConfig.builder().checkpointEveryNEpochs(2).verbose(false).build())
                .build();

            trainer.train(5);

            assertEquals(Arrays.asList(0, 1, 3), savedEpochs(checkpoint, 5));
            assertEquals(0, checkpoint.bestEpoch());
            assertEquals("the checkpoint's own interval is unchanged", 0, checkpoint.saveEveryNEpochs());
        }
    }

    /** Without the trainer's interval the checkpoint keeps its own: only the best epoch is saved. */
    @Test
    public void withoutTheIntervalOnlyTheCheckpointsOwnRulesApply() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            ModelCheckpoint checkpoint = ModelCheckpoint.builder().saveBestOnly(ModelCheckpoint.MODE_MAX).build();
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(checkpoint)
                .config(TrainerConfig.builder().verbose(false).build())
                .build();

            trainer.train(4);

            assertEquals(Collections.singletonList(0), savedEpochs(checkpoint, 4));
        }
    }

    /** An equal interval on the checkpoint itself is accepted and saves the same epochs. */
    @Test
    public void anEqualCheckpointIntervalIsAccepted() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            ModelCheckpoint checkpoint = ModelCheckpoint.builder().saveBestOnly(ModelCheckpoint.MODE_MAX)
                .saveEveryNEpochs(2).build();
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(checkpoint)
                .config(TrainerConfig.builder().checkpointEveryNEpochs(2).verbose(false).build())
                .build();

            trainer.train(4);

            assertEquals(Arrays.asList(0, 1, 3), savedEpochs(checkpoint, 4));
        }
    }

    @Test
    public void checkpointEveryNEpochsWithoutACheckpointCallbackIsRefused() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            TrainerConfig config = TrainerConfig.builder().checkpointEveryNEpochs(2).build();
            try {
                builder(model, new SGD(model.parameters(), 0.5f))
                    .addCallback(new ProgressLogger(1, 1, false, false))
                    .config(config)
                    .build();
                fail("a checkpoint interval without a ModelCheckpoint callback must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("checkpointEveryNEpochs"));
                assertTrue(expected.getMessage(), expected.getMessage().contains("ModelCheckpoint"));
            }
            try {
                new Trainer(model, new SGD(model.parameters(), 0.5f), new SumLoss(),
                    new DataLoader<>(data, 1, false), null, null, null, config);
                fail("the constructor applies the same check");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("checkpointEveryNEpochs"));
            }
        }
    }

    @Test
    public void aConflictingCheckpointIntervalIsRefused() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            ModelCheckpoint everyThird = ModelCheckpoint.builder().saveEveryNEpochs(3).build();
            try {
                builder(model, new SGD(model.parameters(), 0.5f))
                    .addCallback(everyThird)
                    .config(TrainerConfig.builder().checkpointEveryNEpochs(2).build())
                    .build();
                fail("a checkpoint interval that differs from the trainer's must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("conflicts"));
            }
        }
    }

    /** A checkpoint started by a trainer of another interval directly (not through a trainer build) refuses it too. */
    @Test
    public void theCheckpointRefusesAConflictingTrainerWhenARunBegins() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(ModelCheckpoint.builder().build())
                .config(TrainerConfig.builder().checkpointEveryNEpochs(2).verbose(false).build())
                .build();
            ModelCheckpoint everyThird = ModelCheckpoint.builder().saveEveryNEpochs(3).build();
            try {
                everyThird.onTrainBegin(trainer);
                fail("expected the conflicting interval to be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("conflicts"));
            }
        }
    }

    @Test
    public void theConfigurationIsTheCallbacksContext() {
        try (ScaleModel model = new ScaleModel(host, W0, false)) {
            TrainerConfig config = TrainerConfig.builder().verbose(false).build();
            List<TrainerConfig> seen = new ArrayList<>();
            Trainer trainer = builder(model, new SGD(model.parameters(), 0.5f))
                .addCallback(new Callback() {
                    @Override
                    public void onTrainBegin(TrainingContext context) {
                        seen.add(context.config());
                    }
                })
                .config(config)
                .build();
            trainer.train(1);
            assertEquals(Collections.singletonList(config), seen);
        }
    }

    // ------------------------------------------------------------------ fixtures

    /** The epochs of {@code 0 .. epochs - 1} the checkpoint holds, in order. */
    static List<Integer> savedEpochs(ModelCheckpoint checkpoint, int epochs) {
        List<Integer> saved = new ArrayList<>();
        for (int e = 0; e < epochs; e++) {
            if (checkpoint.getCheckpoint(e) != null) {
                assertNotNull("an in-memory checkpoint holds the model state",
                    checkpoint.getCheckpoint(e).get("modelState"));
                saved.add(e);
            }
        }
        if (saved.isEmpty()) {
            assertNull(checkpoint.getBestCheckpoint());
        }
        return saved;
    }

    private Trainer.Builder builder(ScaleModel model, Optimizer optimizer) {
        return Trainer.builder()
            .model(model)
            .optimizer(optimizer)
            .lossFunction(new SumLoss())
            .trainLoader(new DataLoader<>(data, 1, false));
    }

    /** The rows of {@link #XS} as samples of shape {@code [3]}, with unused targets. */
    static TensorDataset dataset(ComputeBackend backend) {
        Tensor[] inputs = new Tensor[XS.length];
        Tensor[] targets = new Tensor[XS.length];
        for (int i = 0; i < XS.length; i++) {
            inputs[i] = Tensor.fromArray(backend, XS[i]);
            targets[i] = Tensor.fromArray(backend, new float[]{0.0f});
        }
        return new TensorDataset(inputs, targets);
    }

    /** The initial rate divided by {@code 1 + steps}: 0.5, 0.25, 0.1667, 0.125, ... for an initial rate of 0.5. */
    static final class InverseStepLR extends LRScheduler {
        InverseStepLR(Optimizer optimizer) {
            super(optimizer);
        }

        @Override
        public float getLR() {
            return initialLr / (1 + stepCount);
        }
    }
}

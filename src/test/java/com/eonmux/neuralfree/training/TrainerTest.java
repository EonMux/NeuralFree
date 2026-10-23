package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.Dataset;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.Before;
import org.junit.Test;

import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * {@link Trainer} and its configuration and callbacks. The training-step tests run on {@link TrainParity#hostBackend()}
 * with a one-parameter model {@code y = x * w} whose gradients are known in closed form: the sum loss gives
 * {@code dL/dw = x}, and the optional auxiliary term {@code sum(w * w)} adds {@code 2w}.
 */
public class TrainerTest {

    @Before
    public void resetGraph() {
        ComputationGraph.current().clear();
        ComputationGraph.setGradEnabled(true);
    }

    @Test
    public void testTrainerConfigDefaults() {
        TrainerConfig config = TrainerConfig.defaults();
        
        assertEquals(0.0f, config.gradientClipNorm(), 0.001f);
        assertEquals(0.0f, config.gradientClipValue(), 0.001f);
        assertEquals(1, config.accumulationSteps());
        assertEquals(10, config.logEveryNSteps());
        assertEquals(1, config.evalEveryNEpochs());
        assertEquals(0, config.checkpointEveryNEpochs());
        assertTrue(config.verbose());
        assertFalse(config.gradientClippingEnabled());
    }

    @Test
    public void testTrainerConfigBuilder() {
        TrainerConfig config = TrainerConfig.builder()
            .gradientClipNorm(1.0f)
            .gradientClipValue(5.0f)
            .accumulationSteps(4)
            .logEveryNSteps(100)
            .evalEveryNEpochs(5)
            .checkpointEveryNEpochs(10)
            .verbose(false)
            .build();
        
        assertEquals(1.0f, config.gradientClipNorm(), 0.001f);
        assertEquals(5.0f, config.gradientClipValue(), 0.001f);
        assertEquals(4, config.accumulationSteps());
        assertEquals(100, config.logEveryNSteps());
        assertEquals(5, config.evalEveryNEpochs());
        assertEquals(10, config.checkpointEveryNEpochs());
        assertFalse(config.verbose());
        assertTrue(config.gradientClippingEnabled());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainerConfigInvalidGradientClipNorm() {
        TrainerConfig.builder().gradientClipNorm(-1.0f).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainerConfigInvalidGradientClipValue() {
        TrainerConfig.builder().gradientClipValue(-1.0f).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainerConfigInvalidAccumulationSteps() {
        TrainerConfig.builder().accumulationSteps(0).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainerConfigInvalidLogEveryNSteps() {
        TrainerConfig.builder().logEveryNSteps(0).build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTrainerConfigInvalidEvalEveryNEpochs() {
        TrainerConfig.builder().evalEveryNEpochs(0).build();
    }

    @Test
    public void testTrainingResultBuilder() {
        TrainingResult result = TrainingResult.builder()
            .epochs(10)
            .totalSteps(100)
            .trainingTimeMs(5000)
            .bestValLoss(0.1f)
            .bestEpoch(5)
            .build();
        
        assertEquals(10, result.epochs());
        assertEquals(100, result.totalSteps());
        assertEquals(5000, result.trainingTimeMs());
        assertEquals(0.1f, result.bestValLoss(), 0.001f);
        assertEquals(5, result.bestEpoch());
    }

    @Test
    public void testTrainingResultLossHistory() {
        TrainingResult.Builder builder = TrainingResult.builder();
        builder.addTrainLoss(1.0f);
        builder.addTrainLoss(0.8f);
        builder.addTrainLoss(0.6f);
        builder.addValLoss(0.9f);
        builder.addValLoss(0.7f);
        
        TrainingResult result = builder.build();
        
        assertEquals(3, result.trainLosses().size());
        assertEquals(2, result.valLosses().size());
        assertEquals(0.6f, result.finalTrainLoss(), 0.001f);
        assertEquals(0.7f, result.finalValLoss(), 0.001f);
    }

    @Test
    public void testEvalResultBuilder() {
        EvalResult result = EvalResult.builder()
            .loss(0.5f)
            .numSamples(100)
            .accuracy(0.95f)
            .metric("precision", 0.92f)
            .metric("recall", 0.88f)
            .build();
        
        assertEquals(0.5f, result.loss(), 0.001f);
        assertEquals(100, result.numSamples());
        assertEquals(0.95f, result.accuracy(), 0.001f);
        assertEquals(0.92f, result.metric("precision"), 0.001f);
        assertEquals(0.88f, result.metric("recall"), 0.001f);
    }

    @Test
    public void testEarlyStoppingDefault() {
        EarlyStopping es = new EarlyStopping();
        assertEquals(10, es.patience());
        assertEquals(0.0f, es.minDelta(), 0.001f);
        assertEquals("min", es.mode());
        assertFalse(es.restoreBestWeights());
    }

    @Test
    public void testEarlyStoppingCustom() {
        EarlyStopping es = new EarlyStopping(5, 0.01f, "max", true);
        assertEquals(5, es.patience());
        assertEquals(0.01f, es.minDelta(), 0.001f);
        assertEquals("max", es.mode());
        assertTrue(es.restoreBestWeights());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEarlyStoppingInvalidPatience() {
        new EarlyStopping(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testEarlyStoppingInvalidMode() {
        new EarlyStopping(10, 0.0f, "invalid", false);
    }

    @Test
    public void testEarlyStoppingTriggers() {
        EarlyStopping es = new EarlyStopping(3);
        
        es.onTrainBegin(null);
        assertNull(es.bestLoss());
        
        es.onEpochEnd(0, 0.5f, 0.5f);
        assertEquals(0.5f, es.bestLoss(), 0.001f);
        assertEquals(0, es.waitCount());
        
        es.onEpochEnd(1, 0.5f, 0.6f);
        assertEquals(1, es.waitCount());
        
        es.onEpochEnd(2, 0.5f, 0.7f);
        assertEquals(2, es.waitCount());
        
        es.onEpochEnd(3, 0.5f, 0.8f);
        assertEquals(3, es.waitCount());
    }

    @Test
    public void testModelCheckpointConfiguration() {
        ModelCheckpoint checkpoint = new ModelCheckpoint("checkpoints", "min", 0, false);
        
        assertEquals("checkpoints", checkpoint.checkpointDir());
        assertEquals("min", checkpoint.saveBestOnly());
        assertEquals(0, checkpoint.saveEveryNEpochs());
        assertFalse(checkpoint.saveOptimizerState());
        
        assertEquals(0, checkpoint.checkpointCount());
        assertNull(checkpoint.bestLoss());
        assertEquals(-1, checkpoint.bestEpoch());
    }

    @Test
    public void testModelCheckpointSaveEveryNEpochsConfig() {
        ModelCheckpoint checkpoint = new ModelCheckpoint("checkpoints", "none", 2, true);
        
        assertEquals("none", checkpoint.saveBestOnly());
        assertEquals(2, checkpoint.saveEveryNEpochs());
        assertTrue(checkpoint.saveOptimizerState());
    }

    @Test
    public void testMetricsTracker() {
        MetricsTracker tracker = new MetricsTracker();
        
        tracker.onTrainBegin(null);
        
        tracker.onEpochEnd(0, 0.5f, 0.6f);
        tracker.onValidationEnd(0, EvalResult.builder().loss(0.6f).accuracy(0.8f).build());
        
        tracker.onEpochEnd(1, 0.4f, 0.5f);
        tracker.onValidationEnd(1, EvalResult.builder().loss(0.5f).accuracy(0.9f).build());
        
        assertEquals(2, tracker.epochCount());
        assertTrue(tracker.hasValidationData());
        assertEquals(0.4f, tracker.bestTrainLoss(), 0.001f);
        assertEquals(0.5f, tracker.bestValLoss(), 0.001f);
        assertEquals(0.9f, tracker.bestAccuracy(), 0.001f);
        assertEquals(0.4f, tracker.latestTrainLoss(), 0.001f);
        assertEquals(0.5f, tracker.latestValLoss(), 0.001f);
        assertEquals(0.9f, tracker.latestAccuracy(), 0.001f);
    }

    @Test
    public void testMetricsTrackerSkipsAnUndefinedAccuracy() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.onTrainBegin(null);

        tracker.onEpochEnd(0, 0.5f, 0.6f);
        tracker.onValidationEnd(0, EvalResult.builder().loss(0.6f).build());

        assertTrue(tracker.accuracyHistory().isEmpty());
        assertEquals(1, tracker.valLossHistory().size());
    }

    @Test
    public void testProgressLoggerDefault() {
        ProgressLogger logger = new ProgressLogger();
        assertEquals(1, logger.logEveryNEpochs());
        assertEquals(10, logger.logEveryNSteps());
        assertTrue(logger.logMetrics());
        assertTrue(logger.verbose());
    }

    @Test
    public void testProgressLoggerCustom() {
        ProgressLogger logger = new ProgressLogger(5, 100, false, false);
        assertEquals(5, logger.logEveryNEpochs());
        assertEquals(100, logger.logEveryNSteps());
        assertFalse(logger.logMetrics());
        assertFalse(logger.verbose());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testProgressLoggerInvalidLogEveryNEpochs() {
        new ProgressLogger(0, 10, true, true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testProgressLoggerInvalidLogEveryNSteps() {
        new ProgressLogger(1, 0, true, true);
    }

    @Test
    public void testCallbackName() {
        assertEquals("EarlyStopping", new EarlyStopping().name());
        assertEquals("ModelCheckpoint", new ModelCheckpoint().name());
        assertEquals("ProgressLogger", new ProgressLogger().name());
        assertEquals("MetricsTracker", new MetricsTracker().name());
    }

    @Test(expected = NullPointerException.class)
    public void testTrainerNullModel() {
        Trainer.builder().build();
    }

    @Test
    public void testTrainerConfigGradientClippingEnabled() {
        assertFalse(TrainerConfig.defaults().gradientClippingEnabled());
        
        TrainerConfig configWithNorm = TrainerConfig.builder()
            .gradientClipNorm(1.0f)
            .build();
        assertTrue(configWithNorm.gradientClippingEnabled());
        
        TrainerConfig configWithValue = TrainerConfig.builder()
            .gradientClipValue(1.0f)
            .build();
        assertTrue(configWithValue.gradientClippingEnabled());
    }

    @Test
    public void testTrainingResultToString() {
        TrainingResult result = TrainingResult.builder()
            .epochs(5)
            .totalSteps(50)
            .trainingTimeMs(1000)
            .addTrainLoss(0.5f)
            .build();
        
        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("epochs=5"));
        assertTrue(str.contains("totalSteps=50"));
    }

    @Test
    public void testEvalResultToString() {
        EvalResult result = EvalResult.builder()
            .loss(0.5f)
            .numSamples(100)
            .accuracy(0.95f)
            .build();
        
        String str = result.toString();
        assertNotNull(str);
        assertTrue(str.contains("loss=0.5"));
        assertTrue(str.contains("numSamples=100"));
    }

    @Test
    public void testTrainerConfigToString() {
        TrainerConfig config = TrainerConfig.defaults();
        String str = config.toString();
        assertNotNull(str);
        assertTrue(str.contains("TrainerConfig"));
    }

    @Test
    public void testEarlyStoppingToString() {
        EarlyStopping es = new EarlyStopping(5);
        String str = es.toString();
        assertNotNull(str);
        assertTrue(str.contains("EarlyStopping"));
        assertTrue(str.contains("patience=5"));
    }

    @Test
    public void testModelCheckpointToString() {
        ModelCheckpoint checkpoint = new ModelCheckpoint("checkpoints");
        String str = checkpoint.toString();
        assertNotNull(str);
        assertTrue(str.contains("ModelCheckpoint"));
    }

    @Test
    public void testMetricsTrackerToString() {
        MetricsTracker tracker = new MetricsTracker();
        String str = tracker.toString();
        assertNotNull(str);
        assertTrue(str.contains("MetricsTracker"));
    }

    @Test
    public void testProgressLoggerToString() {
        ProgressLogger logger = new ProgressLogger();
        String str = logger.toString();
        assertNotNull(str);
        assertTrue(str.contains("ProgressLogger"));
    }

    @Test
    public void testTrainingResultWithNoValLosses() {
        TrainingResult result = TrainingResult.builder()
            .addTrainLoss(0.5f)
            .epochs(1)
            .build();
        
        assertNull(result.finalValLoss());
        assertNull(result.bestValLoss());
    }

    @Test
    public void testTrainingResultFinalTrainLossNoLosses() {
        TrainingResult result = TrainingResult.builder().epochs(1).build();
        try {
            result.finalTrainLoss();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("No training losses"));
        }
    }

    @Test
    public void testMetricsTrackerNoValidationData() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.onTrainBegin(null);
        tracker.onEpochEnd(0, 0.5f, null);
        
        assertFalse(tracker.hasValidationData());
    }

    @Test
    public void testMetricsTrackerBestMethodsNoData() {
        MetricsTracker tracker = new MetricsTracker();
        tracker.onTrainBegin(null);
        
        try {
            tracker.bestValLoss();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("No val losses"));
        }
        
        try {
            tracker.bestAccuracy();
            fail("Expected IllegalStateException");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("No accuracy"));
        }
    }

    // ------------------------------------------------------------------ training step

    private static final float[] W0 = {0.5f, -1.0f, 2.0f};
    private static final float[][] XS = {{1.0f, 2.0f, -3.0f}, {-0.5f, 0.25f, 1.5f}, {2.0f, -1.0f, 0.5f}};

    @Test
    public void accumulatedGradientsAreAveragedIncludingTheTailGroup() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, false);
             TensorDataset data = dataset(host, XS.length)) {
            float lr = 0.5f;
            Trainer trainer = trainer(model, data, new SumLoss(), lr, 2);

            trainer.train(1);

            float[] expected = W0.clone();
            for (int i = 0; i < expected.length; i++) {
                float mean = (XS[0][i] + XS[1][i]) * 0.5f;
                expected[i] -= lr * mean;
                expected[i] -= lr * XS[2][i];
            }
            assertArrayEquals("first group of two averaged, the tail batch alone", expected,
                model.weight().tensor().toFloatArray(), 1e-6f);
            assertNull("gradients zeroed after the last step", model.weight().grad());
            assertEquals("no node left after the epoch", 0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void auxiliaryLossReachesTheParametersButNotTheLoggedLoss() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, true);
             TensorDataset data = dataset(host, 1)) {
            float lr = 0.25f;
            Trainer trainer = trainer(model, data, new SumLoss(), lr, 1);

            trainer.train(1);

            float mainLoss = 0.0f;
            float[] expected = W0.clone();
            for (int i = 0; i < expected.length; i++) {
                mainLoss += XS[0][i] * W0[i];
                expected[i] -= lr * (XS[0][i] + 2.0f * W0[i]);
            }
            assertEquals("the logged loss is the loss function's value", mainLoss, trainer.currentLoss(), 1e-6f);
            assertArrayEquals("the step applies the gradient of loss + auxiliary term", expected,
                model.weight().tensor().toFloatArray(), 1e-6f);
            assertNull("the auxiliary term was taken", model.takeAuxiliaryLoss());
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void evaluationAndPredictionDiscardTheAuxiliaryTerms() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, true);
             TensorDataset data = dataset(host, 2);
             Tensor input = Tensor.fromArray(host, XS[0])) {
            Trainer trainer = trainer(model, data, new SumLoss(), 0.1f, 1);

            trainer.evaluate(new DataLoader<>(data, 1, false));
            assertNull("evaluate takes and closes the terms", model.takeAuxiliaryLoss());
            try (Tensor output = trainer.predict(input)) {
                assertFalse("prediction records nothing", output.requiresGrad());
            }
            assertNull("predict takes and closes the terms", model.takeAuxiliaryLoss());
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void lossThatDoesNotRequireGradIsRefused() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, false);
             TensorDataset data = dataset(host, 1)) {
            Trainer trainer = trainer(model, data, new DetachedSumLoss(), 0.1f, 1);
            assertRefused(trainer);
            assertArrayEquals("no step was taken", W0, model.weight().tensor().toFloatArray(), 0.0f);
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void frozenModelIsRefused() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, false);
             TensorDataset data = dataset(host, 1)) {
            model.setRequiresGrad(false);
            Trainer trainer = trainer(model, data, new SumLoss(), 0.1f, 1);
            assertRefused(trainer);
            assertArrayEquals("no step was taken", W0, model.weight().tensor().toFloatArray(), 0.0f);
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void countCorrectRecordsNothingInGradMode() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor raw = Tensor.fromArray(host, new float[]{0.1f, 0.9f, 0.8f, 0.2f});
             Tensor logits = raw.withRequiresGrad(true);
             Tensor target = Tensor.fromArray(host, new float[]{1.0f, 1.0f})) {
            int before = ComputationGraph.current().size();
            OptionalInt correct = Trainer.countCorrect(logits, target, 2);
            assertTrue(correct.isPresent());
            assertEquals("sample 0 predicts class 1, sample 1 class 0", 1, correct.getAsInt());
            assertEquals("countCorrect records no node", before, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    @Test
    public void countCorrectLeavesASingleOutputPerSampleUndefined() {
        ComputeBackend host = TrainParity.hostBackend();
        try (Tensor logits = Tensor.fromArray(host, new float[]{0.3f, 0.9f, -1.0f});
             Tensor target = Tensor.fromArray(host, new float[]{1.0f, 0.0f, 0.0f})) {
            assertFalse("argmax over one class carries no prediction",
                Trainer.countCorrect(logits, target, 3).isPresent());
        } finally {
            host.close();
        }
    }

    @Test
    public void aStopRequestFetchesNoFurtherBatch() {
        ComputeBackend host = TrainParity.hostBackend();
        try (ScaleModel model = new ScaleModel(host, W0, false);
             TensorDataset data = dataset(host, 3)) {
            AtomicInteger fetched = new AtomicInteger();
            Dataset<TensorDataset.Sample> counted = new Dataset<TensorDataset.Sample>() {
                @Override
                public int size() {
                    return data.size();
                }

                @Override
                public TensorDataset.Sample get(int index) {
                    fetched.incrementAndGet();
                    return data.get(index);
                }
            };
            Trainer trainer = Trainer.builder()
                .model(model)
                .optimizer(new SGD(model.parameters(), 0.1f))
                .lossFunction(new SumLoss())
                .trainLoader(new DataLoader<>(counted, 1, false))
                .addCallback(new Callback() {
                    private TrainingContext owner;

                    @Override
                    public void onTrainBegin(TrainingContext t) {
                        owner = t;
                    }

                    @Override
                    public void onBatchEnd(int batch, float loss) {
                        owner.requestStop();
                    }
                })
                .config(TrainerConfig.builder().verbose(false).build())
                .build();

            trainer.train(2);

            assertEquals("one step ran", 1, trainer.currentStep());
            assertEquals("the stop request is honoured before the next batch is loaded", 1, fetched.get());
            assertEquals(0, ComputationGraph.current().size());
        } finally {
            host.close();
        }
    }

    private static void assertRefused(Trainer trainer) {
        try {
            trainer.train(1);
            fail("a loss that reaches no trainable parameter must be refused");
        } catch (IllegalStateException expected) {
            assertEquals(Trainer.NOT_CONNECTED, expected.getMessage());
        }
    }

    private static Trainer trainer(ScaleModel model, TensorDataset data, LossFunction loss, float lr,
                                   int accumulationSteps) {
        return Trainer.builder()
            .model(model)
            .optimizer(new SGD(model.parameters(), lr))
            .lossFunction(loss)
            .trainLoader(new DataLoader<>(data, 1, false))
            .config(TrainerConfig.builder().accumulationSteps(accumulationSteps).verbose(false).build())
            .build();
    }

    /** The first {@code n} rows of {@link #XS} as samples of shape {@code [3]}, with unused targets. */
    private static TensorDataset dataset(ComputeBackend backend, int n) {
        Tensor[] inputs = new Tensor[n];
        Tensor[] targets = new Tensor[n];
        for (int i = 0; i < n; i++) {
            inputs[i] = Tensor.fromArray(backend, XS[i]);
            targets[i] = Tensor.fromArray(backend, new float[]{0.0f});
        }
        return new TensorDataset(inputs, targets);
    }

    /**
     * {@code x * w} with one parameter {@code w} of shape {@code [3]}; with {@code auxiliary}, every forward pass also
     * leaves the auxiliary loss term {@code sum(w * w)} pending.
     */
    static final class ScaleModel extends Module {
        private final Parameter w;
        private final boolean auxiliary;

        ScaleModel(ComputeBackend backend, float[] init, boolean auxiliary) {
            this.w = registerParameter("w", new Parameter(Tensor.fromArray(backend, init)));
            this.auxiliary = auxiliary;
        }

        Parameter weight() {
            return w;
        }

        @Override
        public Tensor forward(Tensor input) {
            if (auxiliary) {
                try (Tensor squares = TensorOps.mul(w.tensor(), w.tensor())) {
                    setAuxiliaryLoss(TensorOps.sum(squares));
                }
            }
            return TensorOps.mul(input, w.tensor());
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

    /** The sum of the prediction computed without recording: it does not require grad. */
    static final class DetachedSumLoss implements LossFunction {
        @Override
        public Tensor compute(Tensor prediction, Tensor target) {
            try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
                return TensorOps.sum(prediction);
            }
        }

        @Override
        public String name() {
            return "detached sum";
        }
    }
}

package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.data.DataLoader;
import com.eonmux.neuralfree.data.TensorDataset;
import com.eonmux.neuralfree.io.ModelSerializer;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.loss.MSELoss;
import com.eonmux.neuralfree.nn.norm.BatchNorm1d;
import com.eonmux.neuralfree.optim.Adam;
import com.eonmux.neuralfree.optim.Optimizer;
import com.eonmux.neuralfree.optim.SGD;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.gpu.TrainGpu;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ModelCheckpoint} keeps the model's whole state (parameters and BatchNorm running statistics) and the
 * optimizer's state either in memory, as host copies that later training does not change, or in atomically written
 * files with metadata; it honours the mode, periodic and keep settings (deleting superseded files), and
 * {@link ModelCheckpoint#restore(int)} and {@link ModelCheckpoint#restoreBest()} load a checkpoint back, all or
 * nothing, also from a file an earlier run wrote. Runs on the host reference backend; no training step runs, the
 * callback is driven directly.
 */
public class ModelCheckpointTest {
    private static final float[] FIRST_MOMENT = {0.1f, -0.2f};

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private ComputeBackend host;
    private BatchNorm1d model;
    private Adam adam;
    private TensorDataset data;
    private DataLoader<TensorDataset.Sample> loader;
    private Trainer trainer;

    @Before
    public void setUp() {
        host = TrainParity.hostBackend();
        model = new BatchNorm1d(2);
        model.initialize(host);
        adam = new Adam(model.parameters(), 0.01f);
        data = new TensorDataset(new Tensor[]{Tensor.fromArray(host, new float[]{1f, 2f})},
                new Tensor[]{Tensor.fromArray(host, new float[]{0f, 0f})});
        loader = new DataLoader<>(data, 1, false);
        trainer = trainerFor(model, adam);
    }

    @After
    public void tearDown() {
        trainer.close();
        loader.close();
        data.close();
        adam.close();
        model.close();
        TrainGpu.release(host);
        host.close();
    }

    @Test
    @SuppressWarnings("unchecked")
    public void snapshotsHoldRunningStatisticsAndOptimizerStateWithoutValidation() {
        ModelCheckpoint checkpoint = new ModelCheckpoint(null, "min", 0, true);
        checkpoint.onTrainBegin(trainer);
        updateRunningStatistics();
        setFirstMoment(FIRST_MOMENT);

        checkpoint.onEpochEnd(0, 1.5f, null);

        Map<String, Object> saved = checkpoint.getCheckpoint(0);
        assertNull(saved.get("valLoss"));
        assertEquals(Float.valueOf(1.5f), saved.get("monitored"));
        assertFalse("in-memory checkpoints have no file", saved.containsKey("path"));
        Map<String, Object> modelState = (Map<String, Object>) saved.get("modelState");
        assertTrue(modelState.keySet().containsAll(Arrays.asList("weight", "bias", "running_mean",
                "running_var")));
        float[] mean = model.runningMean().toFloatArray();
        assertArrayEquals(mean, (float[]) modelState.get("running_mean"), 0f);

        overwriteState();

        assertArrayEquals("the snapshot does not follow the model", mean,
                (float[]) modelState.get("running_mean"), 0f);
        Map<String, float[]> optimizerState = (Map<String, float[]>) saved.get("optimizerState");
        assertArrayEquals("the snapshot does not follow the optimizer", FIRST_MOMENT,
                optimizerState.get("moment_0"), 0f);
    }

    @Test
    public void restoreLoadsTheModelAndOptimizerStateBack() {
        ModelCheckpoint checkpoint = new ModelCheckpoint(null, "none", 0, true);
        checkpoint.onTrainBegin(trainer);
        updateRunningStatistics();
        setFirstMoment(FIRST_MOMENT);
        checkpoint.onEpochEnd(0, 1.5f, 1.25f);
        float[] mean = model.runningMean().toFloatArray();
        float[] var = model.runningVar().toFloatArray();
        Tensor runningMean = model.runningMean();

        overwriteState();
        checkpoint.restore(0);

        assertSame("a FLOAT32 buffer is restored in place", runningMean, model.runningMean());
        assertArrayEquals(mean, model.runningMean().toFloatArray(), 0f);
        assertArrayEquals(var, model.runningVar().toFloatArray(), 0f);
        assertArrayEquals(new float[]{1f, 1f}, model.weight().tensor().toFloatArray(), 0f);
        assertArrayEquals(FIRST_MOMENT, firstMoment(adam), 0f);
    }

    @Test
    public void restoreRefusesAnEpochWithoutACheckpoint() {
        ModelCheckpoint checkpoint = new ModelCheckpoint(null, "min", 0, false);
        assertThrows(IllegalStateException.class, () -> checkpoint.restore(0));
        checkpoint.onTrainBegin(trainer);
        checkpoint.onEpochEnd(0, 2f, null);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> checkpoint.restore(1));
        assertTrue(e.getMessage(), e.getMessage().contains("epoch 1"));
    }

    @Test
    public void inMemoryCheckpointsHonourKeep() {
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().saveBestOnly("none").keep(2).build();
        checkpoint.onTrainBegin(trainer);
        for (int epoch = 0; epoch < 4; epoch++) {
            checkpoint.onEpochEnd(epoch, 1f, null);
        }
        assertEquals(2, checkpoint.checkpointCount());
        assertNull(checkpoint.getCheckpoint(1));
        assertNotNull(checkpoint.getCheckpoint(2));
        assertNotNull(checkpoint.getCheckpoint(3));
    }

    @Test
    public void restoreBestLoadsTheBestEpochAndRefusesWithoutOne() {
        ModelCheckpoint checkpoint = new ModelCheckpoint(null, "min", 0, false);
        checkpoint.onTrainBegin(trainer);
        assertThrows(IllegalStateException.class, checkpoint::restoreBest);
        updateRunningStatistics();
        float[] mean = model.runningMean().toFloatArray();
        checkpoint.onEpochEnd(0, 2f, null);
        overwriteState();
        checkpoint.onEpochEnd(1, 3f, null);
        assertEquals(1, checkpoint.checkpointCount());

        checkpoint.restoreBest();

        assertArrayEquals(mean, model.runningMean().toFloatArray(), 0f);
        ModelCheckpoint everyEpoch = new ModelCheckpoint(null, "none", 0, false);
        everyEpoch.onTrainBegin(trainer);
        everyEpoch.onEpochEnd(0, 1f, null);
        IllegalStateException e = assertThrows(IllegalStateException.class, everyEpoch::restoreBest);
        assertTrue(e.getMessage(), e.getMessage().contains("'none'"));
    }

    @Test
    public void aNaNMonitoredValueNeverBecomesTheBest() {
        ModelCheckpoint checkpoint = new ModelCheckpoint(null, "min", 0, false);
        checkpoint.onTrainBegin(trainer);
        checkpoint.onEpochEnd(0, Float.NaN, null);
        assertEquals(0, checkpoint.checkpointCount());
        assertThrows(IllegalStateException.class, checkpoint::restoreBest);

        checkpoint.onEpochEnd(1, 2f, null);

        assertEquals(1, checkpoint.bestEpoch());
        assertEquals(Float.valueOf(2f), checkpoint.bestLoss());
        assertNotNull(checkpoint.getCheckpoint(1));
    }

    @Test
    public void diskCheckpointIsWrittenAtomicallyWithMetadataAndRestored() throws IOException {
        Path dir = folder.getRoot().toPath().resolve("nested").resolve("checkpoints");
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().directory(dir).saveBestOnly("none")
                .saveOptimizerState(true).build();
        checkpoint.onTrainBegin(trainer);
        updateRunningStatistics();
        setFirstMoment(FIRST_MOMENT);

        checkpoint.onEpochEnd(3, 1.5f, 1.25f);

        Path file = checkpoint.checkpointFile(3);
        assertEquals(dir.resolve("checkpoint-epoch0003.nfc"), file);
        assertEquals("no temporary file is left", Collections.singletonList("checkpoint-epoch0003.nfc"),
                fileNames(dir));
        Map<String, Object> kept = checkpoint.getCheckpoint(3);
        assertEquals(file, kept.get("path"));
        assertFalse("the state lives in the file", kept.containsKey("modelState"));

        Map<String, Object> metadata = ModelCheckpoint.readMetadata(file);
        assertEquals(Integer.valueOf(3), metadata.get("epoch"));
        assertEquals(Float.valueOf(1.5f), metadata.get("trainLoss"));
        assertEquals(Float.valueOf(1.25f), metadata.get("valLoss"));
        assertEquals(Float.valueOf(1.25f), metadata.get("monitored"));
        assertEquals("none", metadata.get("mode"));
        assertEquals(Boolean.TRUE, metadata.get("hasOptimizerState"));
        assertEquals(new ArrayList<>(model.namedParameters().keySet()), metadata.get("parameterNames"));
        assertEquals(new ArrayList<>(model.namedBuffers().keySet()), metadata.get("bufferNames"));
        assertTrue("the file is a ModelSerializer checkpoint",
                ModelSerializer.inspect(file).parameters().keySet().containsAll(Arrays.asList("weight", "bias")));

        float[] mean = model.runningMean().toFloatArray();
        float[] var = model.runningVar().toFloatArray();
        overwriteState();
        checkpoint.restore(3);

        assertArrayEquals(mean, model.runningMean().toFloatArray(), 0f);
        assertArrayEquals(var, model.runningVar().toFloatArray(), 0f);
        assertArrayEquals(new float[]{1f, 1f}, model.weight().tensor().toFloatArray(), 0f);
        assertArrayEquals(FIRST_MOMENT, firstMoment(adam), 0f);
    }

    @Test
    public void bestOnlyWithKeepOneDeletesSupersededFiles() throws IOException {
        Path dir = folder.newFolder("best").toPath();
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().directory(dir).prefix("run").keep(1).build();
        checkpoint.onTrainBegin(trainer);
        float[] losses = {3f, 2f, 2.5f, 1f};
        for (int epoch = 0; epoch < losses.length; epoch++) {
            checkpoint.onEpochEnd(epoch, 9f, losses[epoch]);
        }
        assertEquals(Collections.singletonList("run-epoch0003.nfc"), fileNames(dir));
        assertEquals(1, checkpoint.checkpointCount());
        assertEquals(3, checkpoint.bestEpoch());
        assertEquals(Float.valueOf(1f), checkpoint.bestLoss());
        assertNull(checkpoint.getCheckpoint(1));
        assertSame(checkpoint.getCheckpoint(3), checkpoint.getBestCheckpoint());
    }

    @Test
    public void everyEpochModeKeepsTheLastK() throws IOException {
        Path dir = folder.newFolder("last").toPath();
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().directory(dir).saveBestOnly("none").keep(2).build();
        checkpoint.onTrainBegin(trainer);
        for (int epoch = 0; epoch < 4; epoch++) {
            checkpoint.onEpochEnd(epoch, 1f, null);
        }
        assertEquals(Arrays.asList("checkpoint-epoch0002.nfc", "checkpoint-epoch0003.nfc"), fileNames(dir));
        assertEquals(2, checkpoint.checkpointCount());
    }

    @Test
    public void keepNeverDropsTheBestCheckpoint() throws IOException {
        Path dir = folder.newFolder("periodic").toPath();
        ModelCheckpoint checkpoint = ModelCheckpoint.builder().directory(dir).saveBestOnly("min")
                .saveEveryNEpochs(1).keep(1).build();
        checkpoint.onTrainBegin(trainer);
        float[] losses = {1f, 2f, 3f};
        for (int epoch = 0; epoch < losses.length; epoch++) {
            checkpoint.onEpochEnd(epoch, losses[epoch], null);
        }
        assertEquals(Arrays.asList("checkpoint-epoch0000.nfc", "checkpoint-epoch0002.nfc"), fileNames(dir));
        assertEquals(0, checkpoint.bestEpoch());
    }

    @Test
    public void restoreReadsAFileAnEarlierRunWrote() throws IOException {
        Path dir = folder.newFolder("resume").toPath();
        ModelCheckpoint first = new ModelCheckpoint(dir.toString(), "none", 0, true);
        first.onTrainBegin(trainer);
        updateRunningStatistics();
        setFirstMoment(FIRST_MOMENT);
        first.onEpochEnd(0, 1f, null);
        float[] mean = model.runningMean().toFloatArray();
        overwriteState();

        ModelCheckpoint second = new ModelCheckpoint(dir.toString(), "min", 0, true);
        second.onTrainBegin(trainer);
        assertNull("the new run kept nothing", second.getCheckpoint(0));
        second.restore(0);

        assertArrayEquals(mean, model.runningMean().toFloatArray(), 0f);
        assertArrayEquals(FIRST_MOMENT, firstMoment(adam), 0f);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> second.restore(1));
        assertTrue(e.getMessage(), e.getMessage().contains("epoch 1")
                && e.getMessage().contains(second.checkpointFile(1).toString()));
    }

    @Test
    public void refusedOptimizerStateRollsTheModelBack() throws IOException {
        Path dir = folder.newFolder("rollback").toPath();
        ModelCheckpoint writer = new ModelCheckpoint(dir.toString(), "none", 0, true);
        writer.onTrainBegin(trainer);
        updateRunningStatistics();
        writer.onEpochEnd(0, 1f, null);

        BatchNorm1d other = new BatchNorm1d(2);
        other.initialize(host);
        Adam weightOnly = new Adam(Collections.singletonList(other.weight()), 0.01f);
        Trainer otherTrainer = trainerFor(other, weightOnly);
        try {
            float[] mean = other.runningMean().toFloatArray();
            ModelCheckpoint reader = new ModelCheckpoint(dir.toString());
            reader.onTrainBegin(otherTrainer);

            UncheckedIOException e = assertThrows(UncheckedIOException.class, () -> reader.restore(0));

            assertTrue(e.getMessage(), e.getMessage().contains(reader.checkpointFile(0).toString()));
            assertArrayEquals("the model is put back", mean, other.runningMean().toFloatArray(), 0f);
        } finally {
            otherTrainer.close();
            weightOnly.close();
            other.close();
        }
    }

    @Test
    public void restoreRefusesAFileOfAnotherModel() throws IOException {
        Path dir = folder.newFolder("mismatch").toPath();
        ModelCheckpoint writer = new ModelCheckpoint(dir.toString(), "none", 0, false);
        writer.onTrainBegin(trainer);
        writer.onEpochEnd(0, 1f, null);

        Linear linear = new Linear(2, 2);
        linear.initialize(host);
        Adam linearAdam = new Adam(linear.parameters(), 0.01f);
        Trainer linearTrainer = trainerFor(linear, linearAdam);
        try {
            ModelCheckpoint reader = new ModelCheckpoint(dir.toString());
            reader.onTrainBegin(linearTrainer);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> reader.restore(0));
            assertTrue(e.getMessage(), e.getMessage().contains("unexpected buffers"));
        } finally {
            linearTrainer.close();
            linearAdam.close();
            linear.close();
        }
    }

    @Test
    public void restoreRefusesOptimizerStateOfAnotherOptimizerClass() throws IOException {
        Path dir = folder.newFolder("optimizer").toPath();
        ModelCheckpoint writer = new ModelCheckpoint(dir.toString(), "none", 0, true);
        writer.onTrainBegin(trainer);
        writer.onEpochEnd(0, 1f, null);
        assertEquals(Adam.class.getName(), ModelCheckpoint.readMetadata(writer.checkpointFile(0)).get("optimizerType"));

        BatchNorm1d other = new BatchNorm1d(2);
        other.initialize(host);
        SGD sgd = new SGD(other.parameters(), 0.01f);
        Trainer sgdTrainer = trainerFor(other, sgd);
        try {
            ModelCheckpoint reader = new ModelCheckpoint(dir.toString());
            reader.onTrainBegin(sgdTrainer);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> reader.restore(0));
            assertTrue(e.getMessage(), e.getMessage().contains(Adam.class.getName()));
        } finally {
            sgdTrainer.close();
            sgd.close();
            other.close();
        }
    }

    @Test
    public void ioFailuresPropagateWithThePath() throws IOException {
        Path notADirectory = folder.newFile("plain").toPath();
        ModelCheckpoint checkpoint = new ModelCheckpoint(notADirectory.toString());
        checkpoint.onTrainBegin(trainer);
        UncheckedIOException write = assertThrows(UncheckedIOException.class,
                () -> checkpoint.onEpochEnd(0, 1f, null));
        assertTrue(write.getMessage(), write.getMessage().contains(notADirectory.toString()));
        assertEquals("a failed save records no best epoch", -1, checkpoint.bestEpoch());
        assertEquals(0, checkpoint.checkpointCount());

        Path dir = folder.newFolder("corrupt").toPath();
        ModelCheckpoint reader = new ModelCheckpoint(dir.toString());
        reader.onTrainBegin(trainer);
        Path corrupt = reader.checkpointFile(5);
        Files.write(corrupt, "not a checkpoint".getBytes(StandardCharsets.UTF_8));
        UncheckedIOException read = assertThrows(UncheckedIOException.class, () -> reader.restore(5));
        assertTrue(read.getMessage(), read.getMessage().contains(corrupt.toString()));
    }

    @Test
    public void readMetadataRefusesANameCountBeyondTheMetadataWithAnIOException() throws IOException {
        Properties metadata = new Properties();
        metadata.setProperty("format_version", "1");
        metadata.setProperty("epoch", "0");
        metadata.setProperty("train_loss", "1.0");
        metadata.setProperty("monitored", "1.0");
        metadata.setProperty("mode", "min");
        metadata.setProperty("optimizer_state", "false");
        metadata.setProperty("parameter.count", Integer.toString(Integer.MAX_VALUE));
        metadata.setProperty("buffer.count", "0");
        Path file = folder.newFile("huge-count.nfc").toPath();
        try (ZipOutputStream zip = new ZipOutputStream(Files.newOutputStream(file))) {
            zip.putNextEntry(new ZipEntry(ModelCheckpoint.METADATA_ENTRY));
            metadata.store(zip, null);
            zip.closeEntry();
        }

        IOException e = assertThrows(IOException.class, () -> ModelCheckpoint.readMetadata(file));

        assertTrue(e.getMessage(), e.getMessage().contains(file.toString())
                && e.getMessage().contains("parameter.count"));
    }

    @Test
    public void invalidSettingsAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> new ModelCheckpoint(null, "lowest", 0, false));
        assertThrows(IllegalArgumentException.class, () -> new ModelCheckpoint(null, "min", -1, false));
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.builder().keep(-1).build());
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.builder().prefix("a/b").build());
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.builder().prefix("").build());
        assertThrows(IllegalStateException.class, () -> new ModelCheckpoint().checkpointFile(0));
    }

    private Trainer trainerFor(Module trained, Optimizer optimizer) {
        return Trainer.builder()
                .model(trained)
                .optimizer(optimizer)
                .lossFunction(new MSELoss())
                .trainLoader(loader)
                .config(TrainerConfig.builder().verbose(false).build())
                .build();
    }

    /** One training-mode forward pass, which moves the running statistics away from zeros and ones. */
    private void updateRunningStatistics() {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor flat = Tensor.fromArray(host, new float[]{1f, 2f, 3f, 6f});
             Tensor in = flat.reshape(Shape.matrix(2, 2));
             Tensor out = model.forward(in)) {
            assertEquals(2, out.shape().dim(0));
        }
    }

    private void setFirstMoment(float[] values) {
        Map<String, Tensor> state = new HashMap<>();
        try (Tensor m = Tensor.fromArray(host, values)) {
            state.put("moment_0", m);
            adam.loadStateDict(state);
        }
    }

    /** The first moment of the first parameter, read from a state dict that is closed afterwards. */
    private static float[] firstMoment(Optimizer optimizer) {
        Map<String, Object> state = optimizer.stateDict();
        try {
            return ((Tensor) state.get("moment_0")).toFloatArray();
        } finally {
            Optimizer.closeStateDict(state);
        }
    }

    /** Overwrites every parameter and persistent buffer with nines, and the first moment with other values. */
    private void overwriteState() {
        Map<String, Object> state = new HashMap<>();
        for (Map.Entry<String, Object> entry : model.stateDict().entrySet()) {
            float[] nines = new float[((float[]) entry.getValue()).length];
            Arrays.fill(nines, 9f);
            state.put(entry.getKey(), nines);
        }
        model.loadStateDict(state, true);
        setFirstMoment(new float[]{9f, 9f});
    }

    private static List<String> fileNames(Path dir) throws IOException {
        try (Stream<Path> files = Files.list(dir)) {
            return files.map(file -> file.getFileName().toString()).sorted().collect(Collectors.toList());
        }
    }
}

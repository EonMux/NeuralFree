package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.gguf.train.FinetuneDataset;
import com.eonmux.neuralfree.gguf.train.TrainableModel;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool finetune}: its command line, its refusals, and a whole run over the tiny F32 llama fixture whose
 * written model carries a lower loss on the corpus it was trained on.
 */
public class FinetuneCommandTest {

    private static final int CTX = 16;
    private static final String CORPUS = "the quick brown fox jumps over the lazy dog and the slow turtle "
            + "the quick brown fox jumps over the lazy dog and the slow turtle "
            + "the quick brown fox jumps over the lazy dog and the slow turtle";

    @Rule
    public final TemporaryFolder folder = new TemporaryFolder();

    private static Path modelFile;
    private static Path quantizedFile;
    private static Path f16File;

    @BeforeClass
    public static void write() throws IOException {
        modelFile = SyntheticModels.llama().writeTempFile();
        quantizedFile = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.Q8_0).build()).writeTempFile();
        f16File = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.F16).build()).writeTempFile();
    }

    @AfterClass
    public static void close() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
        }
        if (quantizedFile != null) {
            Files.deleteIfExists(quantizedFile);
        }
        if (f16File != null) {
            Files.deleteIfExists(f16File);
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(CTX).ubatch(CTX).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static Path corpus() throws IOException {
        Path file = Files.createTempFile("corpus", ".txt");
        Files.write(file, CORPUS.getBytes(StandardCharsets.UTF_8));
        file.toFile().deleteOnExit();
        return file;
    }

    @Test
    public void theFlagsDefaultToLlamaFinetunesValues() throws Exception {
        FinetuneCommand.Options o = FinetuneCommand.parse(new String[] {"-m", "m.gguf", "-f", "c.txt", "-o",
                "out.gguf"});
        assertEquals("m.gguf", o.model.toString());
        assertEquals("c.txt", o.corpus.toString());
        assertEquals("out.gguf", o.output.toString());
        assertEquals(FinetuneCommand.DEFAULT_CTX, o.ctx);
        assertEquals(-1, o.batch);
        assertEquals(0, o.gpuLayers);
        assertEquals(FinetuneCommand.DEFAULT_EPOCHS, o.epochs);
        assertEquals(FinetuneCommand.DEFAULT_LR0, o.lr0, 0f);
        assertEquals(FinetuneCommand.DEFAULT_LR_MIN, o.lrMin, 0f);
        assertEquals(FinetuneCommand.DEFAULT_DECAY_EPOCHS, o.decayEpochs, 0f);
        assertEquals(FinetuneCommand.DEFAULT_WEIGHT_DECAY, o.weightDecay, 0f);
        assertEquals(FinetuneCommand.DEFAULT_VAL_SPLIT, o.valSplit, 0.0);
        assertEquals(com.eonmux.neuralfree.gguf.train.FinetuneOptimizer.Kind.ADAMW, o.optimizer);
    }

    @Test
    public void theLongFlagsSetTheSameValues() throws Exception {
        FinetuneCommand.Options o = FinetuneCommand.parse(new String[] {"--model", "m.gguf", "--file", "c.txt",
                "--output", "out.gguf", "--ctx", "32", "--batch", "32", "--optimizer", "sgd", "--learning-rate",
                "0.5", "--learning-rate-min", "0.1", "-decay-epochs", "2", "--weight-decay", "0.25", "--epochs",
                "3", "--val-split", "0.1"});
        assertEquals(32, o.ctx);
        assertEquals(32, o.batch);
        assertEquals(com.eonmux.neuralfree.gguf.train.FinetuneOptimizer.Kind.SGD, o.optimizer);
        assertEquals(0.5f, o.lr0, 0f);
        assertEquals(0.1f, o.lrMin, 0f);
        assertEquals(2f, o.decayEpochs, 0f);
        assertEquals(0.25f, o.weightDecay, 0f);
        assertEquals(3, o.epochs);
        assertEquals(0.1, o.valSplit, 0.0);
    }

    @Test
    public void helpPrintsTheUsageAndStops() throws Exception {
        assertEquals(null, FinetuneCommand.parse(new String[] {"--help"}));
    }

    @Test
    public void theContradictoryOrUnknownFlagsAreRefusedByName() {
        assertRefused(new String[] {"-m", "m", "-f", "c"}, "-o/--output");
        assertRefused(new String[] {"-f", "c", "-o", "o"}, "-m/--model");
        assertRefused(new String[] {"-m", "m", "-o", "o"}, "-f/--file");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-b", "8", "-c", "16"}, "-b 8");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-ngl", "-1"}, "-ngl");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-opt", "lamb"}, "lamb");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "--val-split", "1.0"}, "--val-split");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-lr0", "0.001", "-lr-min", "0.01"},
                "-lr-min");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-c", "2"}, "-c");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-lr0", "0"}, "-lr0");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-lr-min", "0"}, "-lr-min");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-wd", "-1"}, "-wd");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-decay-epochs", "-1"}, "-decay-epochs");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "-epochs", "0"}, "-epochs");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "--nonsense"}, "--nonsense");
        assertRefused(new String[] {"-m", "m", "-f", "c", "-o", "o", "extra"}, "unexpected argument");
    }

    private static void assertRefused(String[] args, String fragment) {
        try {
            FinetuneCommand.parse(args);
            throw new AssertionError("the command line " + Arrays.toString(args) + " was accepted");
        } catch (FinetuneCommand.UsageException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(fragment));
        }
    }

    @Test
    public void aRunTrainsTheCorpusAndWritesAModelWithALowerLoss() throws Exception {
        Path out = folder.getRoot().toPath().resolve("trained.gguf");
        Path corpus = corpus();
        double before = meanLoss(modelFile, corpus);
        FinetuneCommand.dispatch(new String[] {"-m", modelFile.toString(), "-f", corpus.toString(), "-o",
                out.toString(), "-c", Integer.toString(CTX), "-epochs", "3", "-lr0", "0.05", "-lr-min", "0.05",
                "-wd", "0", "--val-split", "0"}, config());
        assertTrue("the run wrote no model", Files.exists(out));
        double after = meanLoss(out, corpus);
        assertTrue("the loss of the written model " + after + " is not below the original " + before, after < before);
        assertFalse("the run left every tensor of the model unchanged", sameModel(modelFile, out));
    }

    @Test
    public void aPartialGpuOffloadAndAGpuRequestWithoutABackendAreRefusedByName() throws Exception {
        Path out = folder.getRoot().toPath().resolve("ngl.gguf");
        assertDispatchRefused(new String[] {"-m", modelFile.toString(), "-f", corpus().toString(), "-o",
                out.toString(), "-ngl", "1"}, "-ngl");
        assertDispatchRefused(new String[] {"-m", modelFile.toString(), "-f", corpus().toString(), "-o",
                out.toString(), "-ngl", "2"}, "-ngl needs a GPU backend");
        assertFalse(Files.exists(out));
    }

    private static void assertDispatchRefused(String[] args, String fragment) throws IOException {
        try {
            FinetuneCommand.dispatch(args, config());
            throw new AssertionError("the command line was accepted");
        } catch (FinetuneCommand.UsageException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(fragment));
        }
    }

    @Test
    public void aGpuRunOverEveryTrunkLayerTrainsAndWritesTheModel() throws Exception {
        try (com.eonmux.neuralfree.tensor.gpu.TrainParity.Session session =
                     com.eonmux.neuralfree.tensor.gpu.TrainParity.open(
                             com.eonmux.neuralfree.tensor.gpu.TrainParity.Api.OPENCL)) {
            assertTrue(session != null);
        }
        Path out = folder.getRoot().toPath().resolve("trained-gpu.gguf");
        Path corpus = corpus();
        double before = meanLoss(modelFile, corpus);
        EngineConfig gpu = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.builder().nCtx(CTX).ubatch(CTX).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        FinetuneCommand.dispatch(new String[] {"-m", modelFile.toString(), "-f", corpus.toString(), "-o",
                out.toString(), "-c", Integer.toString(CTX), "-ngl", "2", "-epochs", "3", "-lr0", "0.05",
                "-lr-min", "0.05", "-wd", "0", "--val-split", "0"}, gpu);
        assertTrue("the GPU run wrote no model", Files.exists(out));
        assertTrue("the GPU run did not lower the loss", meanLoss(out, corpus) < before);
        assertFalse("the GPU run left every tensor unchanged", sameModel(modelFile, out));
    }

    @Test
    public void aRunOverF16WeightsWritesAModelThatStaysF16() throws Exception {
        Path out = folder.getRoot().toPath().resolve("trained-f16.gguf");
        Path corpus = corpus();
        double before = meanLoss(f16File, corpus);
        FinetuneCommand.dispatch(new String[] {"-m", f16File.toString(), "-f", corpus.toString(), "-o",
                out.toString(), "-c", Integer.toString(CTX), "-epochs", "3", "-lr0", "0.05", "-lr-min", "0.05",
                "-wd", "0", "--val-split", "0"}, config());
        assertTrue("the run wrote no model", Files.exists(out));
        assertTrue("the F16 run did not lower the loss", meanLoss(out, corpus) < before);
        try (ModelSource source = ModelSources.open(out.toString())) {
            assertEquals(GGMLType.F16, tensor(source, "blk.0.attn_q.weight").type());
        }
    }

    private static GGUFTensorInfo tensor(ModelSource source, String name) {
        for (GGUFTensorInfo info : source.getTensors()) {
            if (info.name().equals(name)) {
                return info;
            }
        }
        throw new AssertionError("no tensor " + name);
    }

    /** The mean cross entropy of every window of the corpus, the loss the run trains. */
    private static double meanLoss(Path model, Path corpus) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             TrainableModel train = TrainableModel.open(m, CTX)) {
            String text = new String(Files.readAllBytes(corpus), StandardCharsets.UTF_8);
            FinetuneDataset data = FinetuneDataset.ofText(text, m.tokenizer(), CTX, 0.0);
            double sum = 0.0;
            for (FinetuneDataset.Window window : data.training()) {
                sum += train.forward(window.tokens(), window.labels()).loss();
            }
            return sum / data.training().size();
        }
    }

    /** Whether two files store the same bytes for every tensor. */
    private static boolean sameModel(Path a, Path b) throws IOException {
        try (ModelSource x = ModelSources.open(a.toString()); ModelSource y = ModelSources.open(b.toString());
             WeightStore sx = x.createMappedStore(StoreUse.RESIDENT_UPLOAD);
             WeightStore sy = y.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            for (int i = 0; i < x.getTensors().size(); i++) {
                long size = x.getTensors().get(i).sizeInBytes();
                byte[] ba = new byte[(int) size];
                byte[] bb = new byte[(int) size];
                sx.readTensor(x.getTensors().get(i).offset(), ba.length, ba, 0);
                sy.readTensor(y.getTensors().get(i).offset(), bb.length, bb, 0);
                if (!Arrays.equals(ba, bb)) {
                    return false;
                }
            }
            return true;
        }
    }

    @Test
    public void anExistingOutputFileIsNotOverwritten() throws Exception {
        Path out = folder.newFile("existing.gguf").toPath();
        try {
            FinetuneCommand.dispatch(new String[] {"-m", modelFile.toString(), "-f", corpus().toString(), "-o",
                    out.toString()}, config());
            throw new AssertionError("an existing output file was overwritten");
        } catch (FinetuneCommand.UsageException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("already exists"));
        }
    }

    @Test
    public void aMissingCorpusIsRefused() throws Exception {
        try {
            FinetuneCommand.dispatch(new String[] {"-m", modelFile.toString(), "-f", "/nonexistent/corpus.txt",
                    "-o", folder.getRoot().toPath().resolve("out.gguf").toString()}, config());
            throw new AssertionError("a missing corpus was accepted");
        } catch (FinetuneCommand.UsageException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("not a readable file"));
        }
    }

    @Test
    public void aModelTheTrainerCannotRepresentIsRefusedByName() throws Exception {
        Path out = folder.getRoot().toPath().resolve("quantized.gguf");
        try {
            FinetuneCommand.dispatch(new String[] {"-m", quantizedFile.toString(), "-f", corpus().toString(),
                    "-o", out.toString(), "-c", Integer.toString(CTX)}, config());
            throw new AssertionError("a quantized model was trained");
        } catch (IOException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("F32"));
        }
        assertFalse(Files.exists(out));
    }

    @Test
    public void aCorpusTooShortForTheContextIsRefused() throws Exception {
        Path shortCorpus = Files.createTempFile("short", ".txt");
        Files.write(shortCorpus, "hi".getBytes(StandardCharsets.UTF_8));
        shortCorpus.toFile().deleteOnExit();
        try {
            FinetuneCommand.dispatch(new String[] {"-m", modelFile.toString(), "-f", shortCorpus.toString(),
                    "-o", folder.getRoot().toPath().resolve("short.gguf").toString(), "-c",
                    Integer.toString(CTX)}, config());
            throw new AssertionError("a corpus shorter than a window was trained");
        } catch (IOException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("tokenizes to"));
        }
    }
}

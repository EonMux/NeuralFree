package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.gpu.forward.DecodeStepBreakdown;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool bench} end to end on the CPU engine, over the synthetic llama model: a sweep of pp and tg
 * shapes with repetitions produces one Markdown row per run with the reference's statistics, the JSON and
 * JSONL printers carry the raw samples, the progress lines land on stderr, and every unsupported or invalid
 * input is refused by name. The runs stay small; the timing asserts only that the samples exist.
 */
public class BenchCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theSweepRunsEveryShapeAndPrintsOneRowPerRun() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        ByteArrayOutputStream errBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-n", "4", "-r", "2", "-o", "md"),
                print(outBytes), print(errBytes));
        List<String> lines = table(outBytes);
        assertEquals("two shapes, two rows", 4, lines.size());
        assertTrue("the header carries the fixed columns: " + lines.get(0),
                lines.get(0).contains(" model ") && lines.get(0).contains(" size ")
                        && lines.get(0).contains(" params ") && lines.get(0).contains(" test ")
                        && lines.get(0).contains(" t/s "));
        assertTrue("the test column is present: " + lines.get(0), lines.get(0).contains(" test |"));
        assertTrue("pp8 lands first: " + lines.get(2), lines.get(2).contains(" pp8 "));
        assertTrue("tg4 lands second: " + lines.get(3), lines.get(3).contains(" tg4 "));
        assertTrue("every t/s cell has the avg ± stddev shape: " + lines.get(2),
                lines.get(2).matches(".*\\d+\\.\\d{2} ± \\d+\\.\\d{2} \\|"));
        assertTrue("the run itself stays silent: stderr holds " + errBytes.size(), errBytes.size() == 0);
    }

    /**
     * The table lines of a Markdown run, after the header's first line, which names the CPU kernels this JVM runs
     * and why ({@link CpuSimd#status()}), and the blank line below it.
     */
    private static List<String> table(ByteArrayOutputStream out) {
        List<String> lines = Arrays.asList(out.toString(StandardCharsets.UTF_8).split("\n"));
        assertEquals("the header names the CPU kernels: " + lines, "CPU kernels: " + CpuSimd.status(), lines.get(0));
        assertEquals("a blank line separates the table: " + lines, "", lines.get(1));
        return lines.subList(2, lines.size());
    }

    @Test
    public void theDepthPrefillsBeforeTheTimedRun() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "0", "-pg", "4,2", "-d", "2", "-r", "1",
                "-o", "md"), print(outBytes), print(new ByteArrayOutputStream()));
        List<String> lines = table(outBytes);
        assertEquals(3, lines.size());
        assertTrue("the pair shape names its depth: " + lines.get(2), lines.get(2).contains(" pp4+tg2 @ d2 "));
    }

    @Test
    public void theMachineFormatsCarryTheRawSamples() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream json = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "4", "-r", "3", "-o", "json"),
                print(json), print(new ByteArrayOutputStream()));
        String s = json.toString(StandardCharsets.UTF_8);
        assertTrue("the array opens and closes: " + s, s.startsWith("[\n") && s.trim().endsWith("]"));
        assertEquals("one object per run", 1, s.split("\\{").length - 1);
        assertTrue("three repetitions, three samples: " + s, s.contains("\"samples_ns\": [ "));
        assertTrue("the samples feed the statistics: " + s, s.contains("\"avg_ns\": "));

        ByteArrayOutputStream jsonl = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "4", "-r", "2", "-o", "jsonl"),
                print(jsonl), print(new ByteArrayOutputStream()));
        List<String> lines = Arrays.asList(jsonl.toString(StandardCharsets.UTF_8).split("\n"));
        assertEquals("one line per run", 1, lines.size());
        assertTrue(lines.get(0).startsWith("{") && lines.get(0).endsWith("}"));

        ByteArrayOutputStream csv = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-n", "0", "-r", "1", "-o", "csv"),
                print(csv), print(new ByteArrayOutputStream()));
        List<String> csvLines = Arrays.asList(csv.toString(StandardCharsets.UTF_8).split("\n"));
        assertEquals(2, csvLines.size());
        assertTrue("the row is all quoted: " + csvLines.get(1), csvLines.get(1).startsWith("\""));

        ByteArrayOutputStream sql = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "4", "-r", "1", "-o", "sql"),
                print(sql), print(new ByteArrayOutputStream()));
        String ddl = sql.toString(StandardCharsets.UTF_8);
        assertTrue(ddl.startsWith("CREATE TABLE IF NOT EXISTS llama_bench ("));
        assertTrue(ddl.contains("INSERT INTO llama_bench"));
    }

    @Test
    public void theProgressLinesLandOnStderr() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        ByteArrayOutputStream errBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-n", "4", "-r", "2", "--progress", "-o", "md"),
                print(outBytes), print(errBytes));
        List<String> lines = Arrays.asList(errBytes.toString(StandardCharsets.UTF_8).split("\n"));
        assertTrue("each run announces itself: " + lines, lines.contains("nftool bench: benchmark 1/2: starting")
                && lines.contains("nftool bench: benchmark 2/2: starting"));
        assertTrue("the warmup announces itself: " + lines,
                lines.contains("nftool bench: benchmark 1/2: warmup prompt run"));
        assertTrue("each repetition announces itself: " + lines,
                lines.contains("nftool bench: benchmark 1/2: run 1/2"));
    }

    /**
     * The generation warmup decodes the test's tokens up to {@link BenchCommand#WARMUP_DECODES}, enough for a decode
     * replay to capture and verify its trunk before the first timed repetition.
     */
    @Test
    public void theGenerationWarmupDecodesUpToItsCap() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream errBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "3," + (BenchCommand.WARMUP_DECODES + 4), "-r",
                "1", "--progress", "-o", "none"), print(new ByteArrayOutputStream()), print(errBytes));
        List<String> lines = Arrays.asList(errBytes.toString(StandardCharsets.UTF_8).split("\n"));
        assertTrue("a short generation warms up with all of its tokens: " + lines,
                lines.contains("nftool bench: benchmark 1/2: warmup generation run of 3 tokens"));
        assertTrue("a long one with the cap: " + lines, lines.contains("nftool bench: benchmark 2/2: warmup generation"
                + " run of " + BenchCommand.WARMUP_DECODES + " tokens"));
    }

    /**
     * The warmup decodes each of its tokens in turn from position 0, the first the given one and the rest drawn from
     * the generator below the vocab, and runs the test's token count up to {@link BenchCommand#WARMUP_DECODES}.
     */
    @Test
    public void theWarmupDecodesItsCountFromPositionZero() throws Exception {
        assertEquals(3, BenchCommand.warmupDecodes(3));
        assertEquals(BenchCommand.WARMUP_DECODES, BenchCommand.warmupDecodes(BenchCommand.WARMUP_DECODES + 4));
        List<Integer> tokens = new ArrayList<>();
        List<Integer> positions = new ArrayList<>();
        int vocab = 50;
        BenchCommand.decodeWarmup(5, 7, new Random(3L), vocab, (token, pos) -> {
            tokens.add(token);
            positions.add(pos);
        });
        assertEquals(Arrays.asList(0, 1, 2, 3, 4), positions);
        assertEquals("the first decode takes the given token", Integer.valueOf(7), tokens.get(0));
        Random draws = new Random(3L);
        for (int i = 1; i < tokens.size(); i++) {
            assertEquals("decode " + i, Integer.valueOf(draws.nextInt(vocab)), tokens.get(i));
        }
    }

    @Test
    public void noWarmupSkipsTheWarmupPass() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream errBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-r", "1", "--no-warmup", "--progress", "-o", "none"),
                print(new ByteArrayOutputStream()), print(errBytes));
        String err = errBytes.toString(StandardCharsets.UTF_8);
        assertTrue("only the run announces itself: " + err, err.contains(": run 1/1"));
        assertTrue("no warmup line: " + err, !err.contains("warmup"));
        assertTrue("the CPU kernel choice and its reason lead the progress: " + err,
                err.startsWith("nftool bench: CPU kernels: " + CpuSimd.status() + "\n"));
    }

    @Test
    public void helpPrintsTheUsageAndRunsNothing() throws Exception {
        assertNull("help runs no sweep", BenchCommand.parse(new String[]{"--help"}));
        assertTrue("the usage documents the test parameters: " + BenchCommand.USAGE,
                BenchCommand.USAGE.contains("test parameters"));
        assertTrue("the usage documents the sweep flags: " + BenchCommand.USAGE,
                BenchCommand.USAGE.contains("-pg") && BenchCommand.USAGE.contains("--repetitions"));
        for (String flag : new String[]{"-dev, --device", "-ncmoe, --n-cpu-moe", "-ot, --override-tensor",
                "-lm, --load-mode", "-embd, --embeddings", "-hf", "-hff", "-hft", "-t, --threads",
                "--numa", "-nopo", "-fitt", "-fitc"}) {
            assertTrue("the usage documents " + flag, BenchCommand.USAGE.contains(flag));
        }
        assertTrue("the refused sentence names the unsupported flags: " + BenchCommand.USAGE,
                BenchCommand.USAGE.contains("Unsupported here:"));
    }

    @Test
    public void theHuggingFaceTokenIsValidatedByName() {
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "-hft", "not-a-token"}));
        assertTrue("the refusal names the flag: " + e.getMessage(), e.getMessage().contains("-hft"));
    }

    @Test
    public void theSplitAndKvOffloadFlagsParseIntoTheGrid() throws Exception {
        BenchGrid grid = BenchCommand.parse(new String[]{"-m", "x.gguf", "-sm", "none,layer", "-mg", "0-1",
                "-ts", "1/1", "-nkvo", "0,1"});
        assertEquals(2, grid.splitModes.size());
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, grid.splitModes.get(1));
        assertEquals(2, grid.mainGpus.size());
        assertEquals(1, grid.tensorSplits.size());
        assertEquals(2, grid.noKvOffload.size());
        assertTrue(grid.noKvOffload.get(1).booleanValue());
        BenchGrid repeatedGpu = BenchCommand.parse(new String[]{"-m", "x.gguf", "-mg", "0", "-mg", "1"});
        assertEquals("llama-bench's -mg assigns rather than appends", 1, repeatedGpu.mainGpus.size());
        assertEquals(1, repeatedGpu.mainGpus.get(0).intValue());
        assertTrue("the usage documents the new sweep flags: " + BenchCommand.USAGE,
                BenchCommand.USAGE.contains("-sm, --split-mode") && BenchCommand.USAGE.contains("-ts, --tensor-split")
                        && BenchCommand.USAGE.contains("-nkvo, --no-kv-offload")
                        && BenchCommand.USAGE.contains("-mg, --main-gpu"));
    }

    @Test
    public void aKvOffloadSweepRunsOnTheCpuEngine() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-n", "0", "-nkvo", "0,1", "-r", "1", "-o", "md"),
                print(outBytes), print(new ByteArrayOutputStream()));
        List<String> lines = table(outBytes);
        assertEquals("two KV-offload values, two rows", 4, lines.size());
        assertTrue("the varying KV offload gains a column: " + lines.get(0), lines.get(0).contains(" nkvo |"));
    }

    @Test
    public void theRefusalsNameTheirFlag() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        BenchGrid threaded = BenchCommand.parse(new String[]{"-m", "x.gguf", "-t", "4"});
        assertEquals("one thread count sizes the pool", 4, threaded.threads);
        assertTrue(threaded.threadsGiven);
        BenchCommand.UsageException threadList = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "-t", "4,8"}));
        assertTrue("the thread list refusal names its flag and reason: " + threadList.getMessage(),
                threadList.getMessage().contains("-t") && threadList.getMessage().contains("one thread count"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchCommand.parse(new String[]{"-fa", "on"}));
        BenchGrid overridden = BenchCommand.parse(new String[]{"-m", "x.gguf", "-ot", "a=CPU,b=CUDA0"});
        assertEquals("two comma-separated groups, two sweep members", 2, overridden.tensorOverrides.size());
        assertEquals("cpu", overridden.tensorOverrides.get(0).overrides().get(0).target());
        assertEquals("cuda", overridden.tensorOverrides.get(1).overrides().get(0).target());
        BenchGrid hipAndMetal = BenchCommand.parse(new String[]{"-m", "x.gguf", "-ot", "a=ROCm0,b=Metal0"});
        assertEquals("llama.cpp's ROCm buffer maps onto hip", "hip",
                hipAndMetal.tensorOverrides.get(0).overrides().get(0).target());
        assertEquals("llama.cpp's Metal buffer maps onto metal", "metal",
                hipAndMetal.tensorOverrides.get(1).overrides().get(0).target());
        BenchGrid accelerator = BenchCommand.parse(new String[]{"-m", "x.gguf", "-ot", "a=SYCL0"});
        assertEquals("an accelerator API's buffer is that API's placement target", "sycl",
                accelerator.tensorOverrides.get(0).overrides().get(0).target());
        BenchCommand.UsageException bufferType = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "-ot", "a=CUDA_Host"}));
        assertTrue("the buffer-type refusal names it: " + bufferType.getMessage(),
                bufferType.getMessage().contains("CUDA_Host"));
        assertEquals("the tensor split mode parses",
                com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.TENSOR,
                BenchCommand.parse(new String[]{"-m", "x.gguf", "-sm", "tensor"}).splitModes.get(0));
        BenchCommand.UsageException split = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-sm", "bogus"}));
        assertTrue("the split-mode refusal names its flag: " + split.getMessage(),
                split.getMessage().contains("-sm"));
        BenchCommand.UsageException kv = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-nkvo", "yes"}));
        assertTrue("the KV-offload refusal names its flag: " + kv.getMessage(),
                kv.getMessage().contains("-nkvo"));
        assertEquals("long flags accept underscores like the reference normalizes them", 8,
                BenchCommand.parse(new String[]{"-m", "x.gguf", "--n_prompt", "8"}).nPrompt.get(0).intValue());
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"--no-such-flag", "x"}));
        assertTrue("the unknown option is named: " + e.getMessage(), e.getMessage().contains("--no-such-flag"));
        BenchCommand.UsageException fmt = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(args(model, "-o", "xml")));
        assertTrue("the bad format is named: " + fmt.getMessage(), fmt.getMessage().contains("-o"));
        BenchCommand.UsageException reps = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(args(model, "-r", "0")));
        assertTrue("the repetitions refusal names the flag: " + reps.getMessage(), reps.getMessage().contains("-r"));
        BenchCommand.UsageException delay = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(args(model, "--delay", "-1")));
        assertTrue("the delay refusal names the flag: " + delay.getMessage(),
                delay.getMessage().contains("--delay"));
    }

    @Test
    public void theEmbeddingModeTimesAnEmbeddingModelsPromptPass() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-embd", "1", "-p", "8", "-n", "0", "-r", "2", "-o", "md"),
                print(outBytes), print(new ByteArrayOutputStream()));
        List<String> lines = table(outBytes);
        assertEquals("header, rule and one run", 3, lines.size());
        assertTrue("the embedding mode gains its column: " + lines.get(0), lines.get(0).contains(" embd |"));
        assertTrue("the run names its shape: " + lines.get(2), lines.get(2).contains(" pp8 "));
        assertTrue("the t/s cell carries avg ± stddev: " + lines.get(2),
                lines.get(2).matches(".*\\d+\\.\\d{2} ± \\d+\\.\\d{2} \\|"));
    }

    @Test
    public void theLoadModeSweepAppliesAndRestoresTheReadProperty() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        String key = com.eonmux.neuralfree.gguf.TensorReadMode.PROP_READ_MODE;
        String prior = System.getProperty(key);
        try {
            for (String mode : new String[]{"none", "mmap", "mlock", "mmap+mlock", "dio"}) {
                BenchCommand.bench(args(model, "-ngl", "0", "-p", "8", "-n", "0", "-lm", mode, "-r", "1",
                                "-o", "none"),
                        print(new ByteArrayOutputStream()), print(new ByteArrayOutputStream()));
                assertEquals("the " + mode + " sweep leaves the process property as it found it",
                        prior, System.getProperty(key));
            }
        } finally {
            if (prior == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, prior);
            }
        }
    }

    @Test
    public void theNumaFlagSetsThePlacementPropertyAndRefusesUnknownStrategies() throws Exception {
        String key = com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy.PROPERTY;
        String prior = System.getProperty(key);
        try {
            BenchCommand.parse(new String[]{"-m", "x.gguf", "--numa", "isolate"});
            assertEquals("isolate", System.getProperty(key));
            BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class,
                    () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "--numa", "mirror"}));
            assertTrue("the refusal names the flag: " + e.getMessage(),
                    e.getMessage().startsWith("--numa") && e.getMessage().contains("distribute, isolate"));
        } finally {
            if (prior == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, prior);
            }
        }
    }

    @Test
    public void theEmbeddingModeRunsThePagedHiddenRowAndDecodePasses() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
        BenchCommand.bench(args(model, "-ngl", "0", "-embd", "1", "-p", "8", "-n", "4", "-r", "1", "-o", "csv"),
                print(outBytes), print(new ByteArrayOutputStream()));
        List<String> csv = Arrays.asList(outBytes.toString(StandardCharsets.UTF_8).split("\n"));
        assertEquals("header and two shapes", 3, csv.size());
        assertTrue("the prompt row carries the embedding flag and its shape: " + csv.get(1),
                csv.get(1).contains("\"auto\",\"0.00\",\"none\",\"auto\",\"1\",\"logits\",\"8\",\"0\","));
        assertTrue("the generation shape follows: " + csv.get(2), csv.get(2).contains("\"0\",\"4\","));
    }

    @Test
    public void aModelWithoutADecoderLoopIsRefusedByName() throws Exception {
        Path model = SyntheticModels.bert().writeTempFile();
        BenchCommand.UsageException grid = null;
        try {
            BenchCommand.bench(args(model, "-ngl", "0", "-n", "4", "-r", "1", "-o", "none"),
                    print(new ByteArrayOutputStream()), print(new ByteArrayOutputStream()));
        } catch (IOException e) {
            assertTrue("the refusal names the architecture: " + e.getMessage(), e.getMessage().contains("bert"));
            return;
        }
        throw new AssertionError("the embedding model ran a decode bench");
    }

    @Test
    public void theMeasurementFlagsParse() throws Exception {
        BenchGrid plain = BenchCommand.parse(new String[]{"-m", "x.gguf"});
        assertEquals(KernelTiming.Mode.OFF, plain.kernels);
        assertTrue(!plain.hostBreakdown);
        assertEquals(BenchReadout.Kind.LOGITS, plain.readout.kind());
        assertEquals(KernelTiming.Mode.DISPATCH, BenchCommand.parse(new String[]{"-m", "x.gguf", "--kernels"}).kernels);
        assertEquals(KernelTiming.Mode.STEP,
                BenchCommand.parse(new String[]{"-m", "x.gguf", "--kernels=step"}).kernels);
        assertTrue(BenchCommand.parse(new String[]{"-m", "x.gguf", "--host-breakdown"}).hostBreakdown);
        assertEquals(BenchReadout.Kind.ARGMAX,
                BenchCommand.parse(new String[]{"-m", "x.gguf", "--tg-readout", "argmax"}).readout.kind());
        BenchReadout greedy = BenchCommand.parse(new String[]{"-m", "x.gguf", "--tg-readout", "greedy"}).readout;
        assertEquals(BenchReadout.Kind.GREEDY, greedy.kind());
        assertTrue("greedy decodes each greedy id as the next token", greedy.feedsBack());
        assertEquals("greedy", greedy.toString());
        BenchReadout topK = BenchCommand.parse(new String[]{"-m", "x.gguf", "--tg-readout", "TopK=40"}).readout;
        assertEquals(BenchReadout.Kind.TOPK, topK.kind());
        assertEquals(40, topK.topK());
        assertEquals("topk=40", topK.toString());
        for (String bad : new String[]{"topk=0", "topk=1025", "topk=x", "probs"}) {
            BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class,
                    () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "--tg-readout", bad}));
            assertTrue(e.getMessage(), e.getMessage().contains("--tg-readout"));
        }
        assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(new String[]{"-m", "x.gguf", "--kernels=all"}));
    }

    @Test
    public void everyReadoutRunsTheGenerationAndTheBreakdownNotesACpuEngine() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        for (String readout : new String[]{"logits", "argmax", "greedy", "topk=8"}) {
            ByteArrayOutputStream outBytes = new ByteArrayOutputStream();
            ByteArrayOutputStream errBytes = new ByteArrayOutputStream();
            BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "4", "-r", "1", "-o", "md",
                    "--host-breakdown", "--kernels=step", "--tg-readout", readout), print(outBytes), print(errBytes));
            List<String> lines = table(outBytes);
            assertEquals(readout + ": one generation row", 3, lines.size());
            assertTrue(lines.get(2), lines.get(2).contains(" tg4 "));
            assertEquals(readout + ": the table names a readout other than llama-bench's", !readout.equals("logits"),
                    lines.get(0).contains(" readout ") && lines.get(2).contains(" " + readout + " "));
            ByteArrayOutputStream jsonl = new ByteArrayOutputStream();
            BenchCommand.bench(args(model, "-ngl", "0", "-p", "0", "-n", "4", "-r", "1", "-o", "jsonl",
                    "--tg-readout", readout), print(jsonl), print(new ByteArrayOutputStream()));
            assertTrue("the row records its readout: " + jsonl,
                    jsonl.toString(StandardCharsets.UTF_8).contains("\"tg_readout\": \"" + readout + "\""));
            String err = errBytes.toString(StandardCharsets.UTF_8);
            assertTrue(err, err.contains("decode host breakdown (" + readout + " readout)"));
            assertTrue(err, err.contains("(no one-token decode step ran on the GPU engine"));
            assertTrue(err, err.contains("(the backend timed no dispatch; --kernels needs the GPU engine)"));
            assertNull("the run restores the breakdown property", System.getProperty(DecodeStepBreakdown.PROPERTY));
            assertNull("the run restores the profile property", System.getProperty(KernelTiming.PROFILE_PROPERTY));
        }
    }

    private static String[] args(Path model, String... flags) {
        List<String> args = new java.util.ArrayList<>();
        args.add("-m");
        args.add(model.toString());
        args.addAll(Arrays.asList(flags));
        return args.toArray(new String[0]);
    }

    private static PrintStream print(ByteArrayOutputStream bytes) {
        return new PrintStream(bytes, true, StandardCharsets.UTF_8);
    }
}

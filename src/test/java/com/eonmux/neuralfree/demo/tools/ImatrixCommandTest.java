package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool imatrix} against the llama.cpp behavior: the flag refusals, the GPU engine's placement flags, the
 * corpus window plan, the ask filter, the combine-only save of {@code --in-file} matrices (GGUF and legacy {@code .dat}), the statistics
 * report, and the collection runs over a synthetic model — window accounting, the frequency copies, the
 * {@code --in-file} chunk seed and the {@code --process-output} head collection.
 */
public class ImatrixCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ flags

    private static ImatrixCommand.Params parse(String... args) throws ImatrixCommand.UsageException {
        return ImatrixCommand.Params.parse(args);
    }

    private static void assertFlag(String line, String fragment) throws Exception {
        String[] args = line.split(" ");
        try {
            parse(args);
            throw new AssertionError("the line '" + line + "' parsed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue("the refusal '" + e.getMessage() + "' names the flag", e.getMessage().contains(fragment));
        }
    }

    @Test
    public void parseRefusesInvalidInputByName() throws Exception {
        List<String> lines = List.of(
                "--chunk-size 512",            // removed with the n_ctx chunking; named refusal
                "--output-format json",
                "--chunks 0",
                "--ctx-size 0",
                "--file",                      // a flag with no value
                "--in-file missing.bin",
                "--file x.txt",                // calibration text without a model
                "--show-statistics");          // no --in-file to describe
        Iterator<String> cases = lines.iterator();
        assertFlag(cases.next(), "unknown option --chunk-size");
        assertFlag(cases.next(), "--output-format needs gguf or dat");
        assertFlag(cases.next(), "--chunks");
        assertFlag(cases.next(), "--ctx-size");
        assertFlag(cases.next(), "--file needs a value");
        assertFlag(cases.next(), "--in-file missing.bin is not a readable file");
        assertFlag(cases.next(), "needs a model path");
        assertFlag(cases.next(), "--show-statistics needs an --in-file");
        try {
            parse("a.gguf", "b.gguf");
            throw new AssertionError("two model paths parsed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("too many file arguments"));
        }
        try {
            parse();
            throw new AssertionError("an empty command line parsed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--file"));
        }
    }

    @Test
    public void parseDefaultsMatchLlamaCppAndHelpReturnsNull() throws Exception {
        assertNull(parse("--help"));
        Path model = tmp.getRoot().toPath().resolve("m.gguf");
        Files.write(model, new byte[]{1});
        Path f = tmp.newFile("c.txt").toPath();
        ImatrixCommand.Params p = parse(model.toString(), "--file", f.toString(), "--no-ppl");
        assertEquals(model, p.model);
        assertEquals("imatrix.gguf", p.outFile.toString());
        assertEquals(512, p.ctxSize);
        assertEquals(-1, p.chunks);
        assertEquals(0, p.fromChunk);
        assertEquals(10, p.outputFrequency);
        assertEquals(0, p.saveFrequency);
        assertEquals(false, p.processOutput);
        assertEquals(false, p.nextn);
        assertEquals(false, p.ppl);
        assertEquals(false, p.legacyFormat);
        ImatrixCommand.Params dat = parse(model.toString(), "--file", f.toString(), "--output-format", "dat",
                "--chunks", "4", "--from-chunk", "2", "--save-frequency", "8", "--output-frequency", "3",
                "--process-output", "--nextn", "--ppl", "--parse-special",
                "--out-file", "m.imatrix", "--in-file", f.toString());
        assertEquals(true, dat.legacyFormat);
        assertEquals(4, dat.chunks);
        assertEquals(2, dat.fromChunk);
        assertEquals(8, dat.saveFrequency);
        assertEquals(3, dat.outputFrequency);
        assertEquals(true, dat.processOutput);
        assertEquals(true, dat.nextn);
        assertEquals(true, dat.ppl);
        assertEquals(true, dat.parseSpecial);
        assertEquals("m.imatrix", dat.outFile.toString());
    }

    @Test
    public void placementFlagsReachTheEngineConfigAndRefuseByName() throws Exception {
        Path model = tmp.getRoot().toPath().resolve("p.gguf");
        Files.write(model, new byte[]{1});
        Path f = tmp.newFile("p.txt").toPath();
        ImatrixCommand.Params p = parse(model.toString(), "--file", f.toString(), "--n-cpu-moe", "3",
                "-ot", "ffn_.*_exps=cpu", "--cpu-moe");
        assertEquals(List.of("--n-cpu-moe", "3", "-ot", "ffn_.*_exps=cpu", "--cpu-moe"), p.placement);
        EngineConfig cfg = p.placed(EngineConfig.builder().build());
        assertEquals(3, cfg.nCpuMoe());
        assertTrue(cfg.cpuMoe());
        assertEquals(1, cfg.tensorOverrides().size());
        assertTrue("no placement flags leave the configuration as given",
                parse(model.toString(), "--file", f.toString()).placement.isEmpty());

        String base = model + " --file " + f + " ";
        assertFlag(base + "--n-cpu-moe x", "--n-cpu-moe");
        assertFlag(base + "-ot ffn_up=disk", "-ot");
        assertFlag(base + "--moe-cache sometimes", "--moe-cache");
        assertFlag(base + "--moe-cache-mib", "--moe-cache-mib");
    }

    // ------------------------------------------------------------------ window plan

    @Test
    public void chunkWindowDropsLeadsAndCapsTheCount() throws Exception {
        assertArrayEquals(new int[]{0, 2}, ImatrixCommand.chunkWindow(100, 50, 0, -1));
        assertArrayEquals(new int[]{0, 3}, ImatrixCommand.chunkWindow(200, 50, 0, 3));
        assertArrayEquals(new int[]{50, 3}, ImatrixCommand.chunkWindow(201, 50, 1, -1));
        assertArrayEquals(new int[]{100, 1}, ImatrixCommand.chunkWindow(400, 50, 2, 1));
        try {
            ImatrixCommand.chunkWindow(150, 50, 1, -1);
            throw new AssertionError("(from + 2) * ctx == total passed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--from-chunk 1"));
        }
        try {
            ImatrixCommand.chunkWindow(99, 50, 0, -1);
            throw new AssertionError("a corpus under two windows passed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("at least 100 tokens"));
        }
    }

    // ------------------------------------------------------------------ ask filter

    @Test
    public void theAskFilterFollowsTheReferenceTensors() {
        for (String blk : new String[]{"blk.0.attn_q.weight", "blk.3.ffn_gate_exps.weight",
                "blk.7.nextn.fc.weight"}) {
            assertTrue(blk, ImatrixCommand.collectable(blk, false, false));
        }
        assertTrue(ImatrixCommand.collectable("output.weight", true, false));
        assertTrue(ImatrixCommand.collectable("token_embd.weight", true, false));
        assertTrue(ImatrixCommand.collectable("nextn.post_projection.weight", true, false));
        assertTrue(ImatrixCommand.collectable("nextn.pre_projection.weight", false, true));
        assertTrue(!ImatrixCommand.collectable("output.weight", false, false));
        assertTrue(!ImatrixCommand.collectable("token_embd.weight", false, true));
        assertTrue(!ImatrixCommand.collectable("nextn.pre_projection.weight", true, false));
    }

    // ------------------------------------------------------------------ combine only

    @Test
    public void combineOnlyMergesGgufAndLegacyMatrices() throws Exception {
        Path dir = tmp.getRoot().toPath();
        ImatrixFile.writeGguf(dir.resolve("a.gguf"),
                List.of(new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{1, 2}, new long[]{10})),
                List.of("cal-a.txt"), 3, 4);
        ImatrixFile.writeLegacy(dir.resolve("b.dat"),
                List.of(new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 2, new float[]{3, 4}, new long[]{8})),
                2, 4, List.of("cal-b.txt"), null);
        Path out = dir.resolve("out.gguf");
        Path written = ImatrixCommand.collect(new String[]{"--in-file", dir.resolve("a.gguf").toString(),
                "--in-file", dir.resolve("b.dat").toString(), "--out-file", out.toString(), "--ctx-size", "4"});
        assertEquals(out, written);
        ImatrixFile.Read read = ImatrixFile.read(out);
        assertEquals(1, read.entries().size());
        // the gguf sums [1, 2] and counts 10, plus the legacy sums and count scaled by the chunk size 4
        assertEquals(4.0f, read.entries().get(0).sumSq()[0], 1e-4f);
        assertEquals(6.0f, read.entries().get(0).sumSq()[1], 1e-4f);
        assertEquals(18L, read.entries().get(0).count(0));
        // the chunk counter seeds from the largest scaled count (18) with half-up rounding
        assertEquals(5, read.chunkCount());
        assertEquals(4, read.chunkSize());
    }

    // ------------------------------------------------------------------ statistics

    @Test
    public void showStatisticsPrintsTheReportAndWritesNothing() throws Exception {
        Path dir = tmp.getRoot().toPath();
        Path matrix = dir.resolve("m.gguf");
        ImatrixFile.writeGguf(matrix,
                List.of(new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 4,
                        new float[]{4, 8, 12, 16}, new long[]{4})),
                List.of("c.txt"), 1, 8);
        PrintStream out = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8.name()));
        try {
            Path written = ImatrixCommand.collect(new String[]{"--show-statistics", "--in-file", matrix.toString()});
            assertNull(written);
        } finally {
            System.setOut(out);
        }
        String report = new String(captured.toByteArray(), StandardCharsets.UTF_8);
        assertTrue(report, report.contains("blk.0.attn_q.weight"));
        assertTrue(report, report.contains("2.5000")); // the mean of the per-column means {1, 2, 3, 4}
        assertTrue(report, report.contains("Computing tensor statistics"));
    }

    // ------------------------------------------------------------------ collection runs

    /** The synthetic llama model file, opened once to size the corpus by its tokenizer. */
    private Path modelFile() throws IOException {
        return SyntheticModels.llama().writeTempFile();
    }

    /** A calibration text tokenizing to at least {@code minTokens} tokens under the model's tokenizer. */
    private Path corpus(GGUFModel m, int minTokens) throws IOException {
        Path f = tmp.newFile("corpus.txt").toPath();
        StringBuilder text = new StringBuilder();
        while (m.tokenizer().encode(text + "the quick cat", true, false).length < minTokens) {
            text.append("the quick cat\n");
        }
        text.append("the quick cat\n");
        Files.write(f, text.toString().getBytes(StandardCharsets.UTF_8));
        return f;
    }

    private EngineConfig config(int ctxSize) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(ctxSize).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    @Test
    public void corpusRunChunksTheWholeCorpusAndSavesFrequencies() throws Exception {
        Path model = modelFile();
        Path out = tmp.getRoot().toPath().resolve("imatrix.gguf");
        int ctx = 64;
        try (GGUFModel m = new GGUFModel(model.toString(), config(ctx))) {
            Path corpus = corpus(m, 3 * ctx);
            ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                    "--ctx-size", String.valueOf(ctx), "--batch-size", String.valueOf(ctx), "--chunks", "2",
                    "--no-ppl", "--output-frequency", "1", "--save-frequency", "1", "--out-file", out.toString()});
        }
        ImatrixFile.Read read = ImatrixFile.read(out);
        assertEquals("the two windows count as two chunks", 2, read.chunkCount());
        assertEquals(ctx, read.chunkSize());
        ImatrixFile.Entry attn = entry(read, "blk.0.attn_q.weight");
        assertNotNull("the block projections were collected", attn);
        assertEquals("two windows of 64 tokens feed the projections", 128L, attn.count(0));
        assertTrue("token_embd is a lookup, never a matmul", entry(read, "token_embd.weight") == null);
        assertTrue("output.weight stays out without --process-output", entry(read, "output.weight") == null);
        assertTrue("the corpus file is the dataset", read.datasets().contains(
                tmp.getRoot().toPath().resolve("corpus.txt").toString()));
        // the save-frequency copies keep the state at chunks 1 and 2
        ImatrixFile.Read at1 = ImatrixFile.read(tmp.getRoot().toPath().resolve("imatrix.gguf.at_1"));
        assertEquals(1, at1.chunkCount());
        ImatrixFile.Entry at1Attn = entry(at1, "blk.0.attn_q.weight");
        assertEquals(64L, at1Attn.count(0));
        assertTrue(Files.exists(tmp.getRoot().toPath().resolve("imatrix.gguf.at_2")));
    }

    @Test
    public void corpusRunCombinesAnInFileSeedIntoTheChunkCounter() throws Exception {
        Path model = modelFile();
        Path out1 = tmp.getRoot().toPath().resolve("first.gguf");
        Path out2 = tmp.getRoot().toPath().resolve("second.gguf");
        int ctx = 64;
        Path corpus;
        try (GGUFModel m = new GGUFModel(model.toString(), config(ctx))) {
            corpus = corpus(m, 2 * ctx);
        }
        ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                "--ctx-size", String.valueOf(ctx), "--chunks", "2", "--no-ppl", "--out-file", out1.toString()});
        ImatrixFile.Read first = ImatrixFile.read(out1);
        assertEquals(2, first.chunkCount());
        long afterTwoChunks = entry(first, "blk.0.attn_q.weight").count(0);
        ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                "--ctx-size", String.valueOf(ctx), "--chunks", "1", "--no-ppl",
                "--in-file", out1.toString(), "--out-file", out2.toString()});
        ImatrixFile.Read second = ImatrixFile.read(out2);
        // the combined matrix seeds the chunk counter at 2, the run adds 1
        assertEquals(3, second.chunkCount());
        assertEquals("the collected rows add to the loaded ones", afterTwoChunks + ctx,
                entry(second, "blk.0.attn_q.weight").count(0));
        assertEquals("both dataset lists carry the corpus", 1, second.datasets().size());
    }

    @Test
    public void processOutputCollectsTheHeadRows() throws Exception {
        Path model = modelFile();
        Path out = tmp.getRoot().toPath().resolve("head.gguf");
        int ctx = 64;
        Path corpus;
        try (GGUFModel m = new GGUFModel(model.toString(), config(ctx))) {
            corpus = corpus(m, 2 * ctx); // the two-window minimum even for a one-chunk run
        }
        ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                "--ctx-size", String.valueOf(ctx), "--chunks", "1", "--no-ppl", "--process-output",
                "--out-file", out.toString()});
        ImatrixFile.Read read = ImatrixFile.read(out);
        assertEquals("every window row ran the head", (long) ctx, entry(read, "output.weight").count(0));
    }

    @Test
    public void theDefaultRunReportsTheCorpusPerplexity() throws Exception {
        Path model = modelFile();
        Path out = tmp.getRoot().toPath().resolve("ppl.gguf");
        int ctx = 64;
        Path corpus;
        try (GGUFModel m = new GGUFModel(model.toString(), config(ctx))) {
            corpus = corpus(m, 2 * ctx);
        }
        PrintStream stdout = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8.name()));
        try {
            ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                    "--ctx-size", String.valueOf(ctx), "--chunks", "2",
                    "--out-file", out.toString()});
        } finally {
            System.setOut(stdout);
        }
        String run = new String(captured.toByteArray(), StandardCharsets.UTF_8);
        assertTrue("the run prints the final PPL estimate", run.contains("Final estimate: PPL = "));
        assertTrue("the run prints a per-chunk estimate", run.contains("[1]"));
        ImatrixFile.Read read = ImatrixFile.read(out);
        assertEquals(2, read.chunkCount());
    }

    @Test
    public void aShortCorpusIsRefusedByName() throws Exception {
        Path model = modelFile();
        int ctx = 64;
        Path corpus;
        try (GGUFModel m = new GGUFModel(model.toString(), config(ctx))) {
            corpus = corpus(m, ctx); // under the 2 * ctx minimum
        }
        try {
            ImatrixCommand.collect(new String[]{model.toString(), "--file", corpus.toString(),
                    "--ctx-size", String.valueOf(ctx)});
            throw new AssertionError("a corpus under two windows collected");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("at least " + (2 * ctx) + " tokens"));
        }
    }

    private static ImatrixFile.Entry entry(ImatrixFile.Read read, String tensor) {
        for (ImatrixFile.Entry e : read.entries()) {
            if (e.tensor().equals(tensor)) {
                return e;
            }
        }
        return null;
    }
}

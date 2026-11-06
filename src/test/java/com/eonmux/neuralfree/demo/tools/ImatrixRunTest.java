package com.eonmux.neuralfree.demo.tools;

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
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The collection runs of {@link ImatrixRun} against llama.cpp's {@code compute_imatrix}: the {@code n_batch} /
 * {@code n_ctx} multi-sequence windows and sub-batches collect the same statistics as one whole-context decode per
 * chunk, decodes below 16 rows collect only the routed experts (the reference's ask), the batch-size rules refuse by
 * flag, and {@code --nextn} replays the NextN head of a synthetic GLM4-MoE model over the trunk's hidden rows,
 * recording {@code imatrix.n_layer_nextn}.
 */
public class ImatrixRunTest {

    private static final int CTX = 32;
    private static final String LINE = "the quick cat\n";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A corpus of {@code lines} repetitions of one line, enough for several windows of {@link #CTX}. */
    private Path corpus(int lines) throws IOException {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < lines; i++) {
            text.append(LINE);
        }
        Path f = tmp.newFile("corpus-" + lines + ".txt").toPath();
        Files.write(f, text.toString().getBytes(StandardCharsets.UTF_8));
        return f;
    }

    private Path collect(Path model, Path corpus, String name, String... extra) throws Exception {
        Path out = tmp.getRoot().toPath().resolve(name);
        List<String> args = new ArrayList<>();
        args.add(model.toString());
        args.add("--file");
        args.add(corpus.toString());
        args.add("--ctx-size");
        args.add(String.valueOf(CTX));
        args.add("--out-file");
        args.add(out.toString());
        for (String e : extra) {
            args.add(e);
        }
        PrintStream stdout = System.out;
        System.setOut(new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8.name()));
        try {
            assertEquals(out, ImatrixCommand.collect(args.toArray(new String[0])));
        } finally {
            System.setOut(stdout);
        }
        return out;
    }

    // ------------------------------------------------------------------ multi-sequence windows

    @Test
    public void parallelSequencesAndSubBatchesCollectTheSameStatistics() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path corpus = corpus(80);
        ImatrixFile.Read single = ImatrixFile.read(collect(model, corpus, "single.gguf",
                "--chunks", "4", "--batch-size", String.valueOf(CTX), "--process-output"));
        ImatrixFile.Read parallel = ImatrixFile.read(collect(model, corpus, "parallel.gguf",
                "--chunks", "4", "--batch-size", String.valueOf(4 * CTX), "--process-output"));
        ImatrixFile.Read split = ImatrixFile.read(collect(model, corpus, "split.gguf",
                "--chunks", "4", "--batch-size", String.valueOf(CTX / 2), "--process-output"));
        assertEquals(4, single.chunkCount());
        assertEquals("four sequences per decode still count four chunks", 4, parallel.chunkCount());
        assertEquals(4, split.chunkCount());
        assertEquals((long) 4 * CTX, entry(single, "blk.0.attn_q.weight").count(0));
        assertEquals("every row of every sequence ran the head", (long) 4 * CTX,
                entry(parallel, "output.weight").count(0));
        ImatrixParity.assertSameStatistics("4 sequences per decode", single, parallel, 1e-3f);
        ImatrixParity.assertSameStatistics("2 batches per chunk", single, split, 1e-3f);
    }

    @Test
    public void decodesBelowSixteenRowsCollectOnlyTheRoutedExperts() throws Exception {
        // llama.cpp collect_imatrix: a MUL_MAT with src1->ne[1] < 16 is declined, every MUL_MAT_ID is collected
        Path model = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe(), 1).writeTempFile();
        Path corpus = corpus(80);
        int chunks = 2;
        ImatrixFile.Read small = ImatrixFile.read(collect(model, corpus, "small.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", "8", "--no-ppl"));
        ImatrixFile.Read full = ImatrixFile.read(collect(model, corpus, "full.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", "16", "--no-ppl"));
        long pairs = (long) chunks * CTX * SyntheticModels.Config.qwen3moe().nExpertUsed();
        for (ImatrixFile.Entry e : small.entries()) {
            assertTrue("only routed experts below 16 rows: " + e.tensor(), e.tensor().endsWith("_exps.weight"));
            assertEquals(e.tensor() + ": every routed pair", pairs, total(e.counts()));
        }
        assertNotNull(entry(small, "blk.1.ffn_up_exps.weight"));
        assertNull(entry(small, "blk.0.attn_q.weight"));
        assertEquals("16-row decodes collect the dense products", (long) chunks * CTX,
                entry(full, "blk.0.attn_q.weight").count(0));
        assertEquals(pairs, total(entry(full, "blk.1.ffn_up_exps.weight").counts()));
    }

    @Test
    public void aGateRunPerKvHeadCountsEachTokenOnce() throws Exception {
        // limite's value-embedding gate is one MUL_MAT over the attention input (models/limite.cpp)
        Path model = SyntheticModels.limite().writeTempFile();
        int chunks = 2;
        ImatrixFile.Read r = ImatrixFile.read(collect(model, corpus(80), "limite.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", String.valueOf(CTX), "--no-ppl"));
        ImatrixFile.Entry gate = entry(r, "blk.0.attn_ve_gate.weight");
        assertNotNull("the value-embedding gate was collected", gate);
        assertEquals((long) chunks * CTX, gate.count(0));
        assertEquals(gate.count(0), entry(r, "blk.0.attn_q.weight").count(0));
    }

    private static long total(long[] counts) {
        long n = 0;
        for (long c : counts) {
            n += c;
        }
        return n;
    }

    @Test
    public void theBatchSizeRulesRefuseByFlag() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path corpus = corpus(4);
        assertSequences(1, model, corpus, "--batch-size", "16");
        assertSequences(1, model, corpus, "--batch-size", String.valueOf(CTX));
        assertSequences(4, model, corpus, "--batch-size", String.valueOf(4 * CTX));
        assertRefused(model, corpus, "must be below --ctx-size " + CTX + " or a multiple of it",
                "--batch-size", String.valueOf(CTX + 8));
        assertRefused(model, corpus, "'--nextn' and '--model-draft' need a single sequence per batch",
                "--batch-size", String.valueOf(2 * CTX), "--nextn");
        assertRefused(model, corpus, "more than the 256", "--batch-size", String.valueOf(257 * CTX));
        try {
            ImatrixCommand.Params.parse(new String[]{model.toString(), "--file", corpus.toString(), "-ngl", "-2"});
            fail("-ngl -2 parsed");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("-ngl needs auto, all or a layer count"));
        }
    }

    private static ImatrixCommand.Params params(Path model, Path corpus, String... extra)
            throws ImatrixCommand.UsageException {
        List<String> args = new ArrayList<>();
        args.add(model.toString());
        args.add("--file");
        args.add(corpus.toString());
        args.add("--ctx-size");
        args.add(String.valueOf(CTX));
        for (String e : extra) {
            args.add(e);
        }
        return ImatrixCommand.Params.parse(args.toArray(new String[0]));
    }

    private static void assertSequences(int expected, Path model, Path corpus, String... extra) throws Exception {
        assertEquals(expected, ImatrixRun.sequences(params(model, corpus, extra)));
    }

    private static void assertRefused(Path model, Path corpus, String fragment, String... extra) throws Exception {
        try {
            ImatrixRun.sequences(params(model, corpus, extra));
            fail(String.join(" ", extra) + " was accepted");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(fragment));
        }
    }

    // ------------------------------------------------------------------ NextN replay

    @Test
    public void nextnReplaysTheHeadOverTheTrunkRows() throws Exception {
        Path model = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe(), 1).writeTempFile();
        Path corpus = corpus(80);
        int chunks = 2;
        int nextnBlock = SyntheticModels.Config.qwen3moe().nLayer();
        String eh = "blk." + nextnBlock + ".nextn.eh_proj.weight";
        String headQ = "blk." + nextnBlock + ".attn_q.weight";

        ImatrixFile.Read plain = ImatrixFile.read(collect(model, corpus, "plain.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", String.valueOf(CTX), "--no-ppl"));
        assertNull("the head runs only under --nextn", entry(plain, eh));
        assertEquals(0, plain.nLayerNextn());

        ImatrixFile.Read whole = ImatrixFile.read(collect(model, corpus, "nextn.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", String.valueOf(CTX), "--no-ppl", "--nextn"));
        assertEquals("the file records the replayed NextN block", 1, whole.nLayerNextn());
        assertEquals("the trunk runs every row", (long) chunks * CTX, entry(whole, "blk.0.attn_q.weight").count(0));
        ImatrixFile.Entry ehProj = entry(whole, eh);
        assertNotNull("the NextN projection was collected", ehProj);
        assertEquals("the head pairs token[p] with h[p - 1]: n_ctx - 1 rows per chunk", (long) chunks * (CTX - 1),
                ehProj.count(0));
        assertEquals("the projection reads [embedding, hidden]", 2 * SyntheticModels.Config.qwen3moe().nEmbd(),
                ehProj.cols());
        assertEquals((long) chunks * (CTX - 1), entry(whole, headQ).count(0));
        ImatrixParity.assertSameStatistics("the trunk under --nextn", plain, without(whole, nextnBlock), 1e-4f);

        // two 16-token sub-batches per chunk: the first replays 15 pairs (no previous row for its first token),
        // fewer than the 16 rows a dense product needs, so only the second sub-batch's 16 pairs are collected
        ImatrixFile.Read split = ImatrixFile.read(collect(model, corpus, "nextn-split.gguf",
                "--chunks", String.valueOf(chunks), "--batch-size", String.valueOf(CTX / 2), "--no-ppl", "--nextn"));
        ImatrixParity.assertSameStatistics("the trunk across sub-batches", without(whole, nextnBlock),
                without(split, nextnBlock), 1e-3f);
        assertEquals("the second sub-batch's pairs", (long) chunks * (CTX / 2), entry(split, eh).count(0));
        assertEquals((long) chunks * (CTX / 2), entry(split, headQ).count(0));

        String report = showStatistics(tmp.getRoot().toPath().resolve("nextn.gguf"));
        assertTrue(report, report.contains("mtp0"));
    }

    @Test
    public void aModelDraftNextToOwnNextnLayersIsRefused() throws Exception {
        Path model = SyntheticModels.glm4moe(SyntheticModels.Config.qwen3moe(), 1).writeTempFile();
        Path corpus = corpus(40);
        try {
            collect(model, corpus, "draft.gguf", "--batch-size", String.valueOf(CTX), "--model-draft",
                    model.toString());
            fail("a target with NextN layers took a draft");
        } catch (ImatrixCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("already includes NextN layers"));
        }
    }

    private static String showStatistics(Path matrix) throws Exception {
        PrintStream stdout = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8.name()));
        try {
            ImatrixCommand.collect(new String[]{"--show-statistics", "--in-file", matrix.toString()});
        } finally {
            System.setOut(stdout);
        }
        return new String(captured.toByteArray(), StandardCharsets.UTF_8);
    }

    /** {@code read} without the tensors of block {@code blk}. */
    private static ImatrixFile.Read without(ImatrixFile.Read read, int blk) throws IOException {
        List<ImatrixFile.Entry> kept = new ArrayList<>();
        for (ImatrixFile.Entry e : read.entries()) {
            if (!e.tensor().startsWith("blk." + blk + ".")) {
                kept.add(e);
            }
        }
        Path f = Files.createTempFile("imatrix-trunk", ".gguf");
        try {
            ImatrixFile.writeGguf(f, kept, read.datasets(), read.chunkCount(), read.chunkSize());
            return ImatrixFile.read(f);
        } finally {
            Files.deleteIfExists(f);
        }
    }

    static ImatrixFile.Entry entry(ImatrixFile.Read read, String tensor) {
        for (ImatrixFile.Entry e : read.entries()) {
            if (e.tensor().equals(tensor)) {
                return e;
            }
        }
        return null;
    }
}

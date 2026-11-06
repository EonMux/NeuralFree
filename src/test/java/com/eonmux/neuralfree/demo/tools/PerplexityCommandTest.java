package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool perplexity} over a running engine (the synthetic llama fixture on the CPU backend): the
 * chunked perplexity scores the second half of every chunk and reports the final estimate, the strided run
 * scores the trailing rows, the saved log-probabilities compare against the same model with vanishing KL in
 * the KL-divergence run (the self-comparison llama.cpp's quantization audits degenerate to), the command
 * line refuses every unsupported combination by name, and the whole surface dispatches from
 * {@code nftool perplexity}.
 */
public class PerplexityCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config(int nCtx, int ubatch) {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(ubatch)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A prompt of at least {@code minTokens} tokens over the fixture's byte-fallback SPM vocabulary. */
    private static String text(GGUFTokenizer tok, int minTokens) {
        String sentence = "The quick brown fox jumps over the lazy dog near the river bank.";
        StringBuilder sb = new StringBuilder();
        while (tok.encode(sb.toString()).length < minTokens) {
            sb.append(sentence).append(' ');
        }
        return sb.toString();
    }

    private static PerplexityCommand.Options options(String prompt) {
        PerplexityCommand.Options o = new PerplexityCommand.Options();
        o.promptText = prompt;
        o.promptBytes = prompt.getBytes(StandardCharsets.UTF_8);
        return o;
    }

    @Test
    public void thePlainRunScoresTheSecondHalfOfEveryChunk() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(32, 8))) {
            int nCtx = m.forwardBackend().memory().plan().nCtx();
            String prompt = text(m.tokenizer(), 8 * nCtx);
            int expectedChunks = m.tokenizer().encode(prompt).length / nCtx;
            PerplexityCommand.Options o = options(prompt);
            o.ctxSize = nCtx;
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            double ppl = PerplexityCommand.perplexity(m, o, out, err);
            assertTrue("a perplexity above 1: " + ppl, ppl > 1.0 && ppl < 1e6);
            assertEquals("one running value per chunk", expectedChunks, occurrences(out.toString(), "["));
            assertTrue(out.toString(), out.toString().endsWith(",\n"));
            assertTrue(err.toString(), err.toString().contains("final estimate: PPL ="));
            // the run is deterministic on one engine
            StringBuilder out2 = new StringBuilder();
            StringBuilder err2 = new StringBuilder();
            double again = PerplexityCommand.perplexity(m, o, out2, err2);
            assertEquals(ppl, again, 0.0);
            assertEquals(out.toString(), out2.toString());
        }
    }

    @Test
    public void theStridedRunScoresTheTrailingRowsOfEveryChunk() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // llama.cpp widens the context by half the stride before loading; the test opens the widened context
        try (GGUFModel m = new GGUFModel(model.toString(), config(36, 8))) {
            PerplexityCommand.Options o = options(text(m.tokenizer(), 2 * 36));
            o.ctxSize = 36;
            o.pplStride = 8;
            o.chunks = 2;
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            double ppl = PerplexityCommand.perplexity(m, o, out, err);
            assertTrue("a finite strided perplexity: " + ppl, ppl > 1.0 && ppl < 1e6);
            assertTrue("one comma-separated running value per chunk, llama.cpp's type 0 line: " + out,
                    out.toString().matches("\\[1\\][\\d.]+,\\[2\\][\\d.]+,\n"));
        }
    }

    @Test
    public void aStrideBeyondTheContextIsRefusedByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(32, 8))) {
            PerplexityCommand.Options o = options(text(m.tokenizer(), 2 * 40));
            o.ctxSize = 32;
            o.pplStride = 40;
            IOException e = assertThrows(IOException.class,
                    () -> PerplexityCommand.perplexity(m, o, new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("--ppl-stride"));
        }
    }

    @Test
    public void theSavedLogitsCompareAgainstTheSameModelWithVanishingKl() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path base = tmp.newFile("base.bin").toPath();
        int nCtx = 32;
        int nChunk;
        try (GGUFModel m = new GGUFModel(model.toString(), config(nCtx, 8))) {
            String prompt = text(m.tokenizer(), 7 * nCtx);
            nChunk = m.tokenizer().encode(prompt).length / nCtx;
            assertTrue("enough chunks for the statistics block", nChunk * (nCtx / 2 - 1) >= 100);
            PerplexityCommand.Options save = options(prompt);
            save.ctxSize = nCtx;
            save.logitsPath = base;
            PerplexityCommand.perplexity(m, save, new StringBuilder(), new StringBuilder());
        }
        int nToken = nCtx - 1 - nCtx / 2;
        try (LogitsFile.In in = new LogitsFile.In(base)) {
            assertEquals("the file holds the context of the run", nCtx, in.nCtx());
            assertEquals("the file holds the model's vocabulary", 320, in.nVocab());
            assertEquals(nChunk, in.nChunk());
            assertEquals(nChunk * nCtx, in.tokens().length);
            assertEquals("the saved tokens carry the BOS the tokenizer added", 1, in.tokens()[0]);
        }
        try (GGUFModel m = new GGUFModel(model.toString(), config(nCtx, 8))) {
            PerplexityCommand.Options kl = options("");
            kl.ctxSize = nCtx;
            kl.logitsPath = base;
            kl.klDivergence = true;
            StringBuilder out = new StringBuilder();
            PerplexityCommand.klDivergence(m, kl, out, new StringBuilder());
            String report = out.toString();
            assertTrue(report, report.contains("ln(PPL(Q)/PPL(base))"));
            List<double[]> rows = tableRows(report);
            assertEquals("one table row per chunk", nChunk, rows.size());
            for (double[] row : rows) {
                assertEquals("the same model diverges by its quantization only: " + row[5], 0.0, row[5], 1e-3);
                assertEquals("the log perplexity ratio vanishes: " + row[3], 0.0, row[3], 1e-6);
                assertEquals("the top token always agrees: " + row[9], 100.0, row[9], 1e-9);
            }
            assertTrue("the statistics blocks print over a hundred rows", report.contains(
                    "====== Perplexity statistics ======"));
            assertTrue(report.contains("====== KL divergence statistics ======"));
            assertTrue(report.contains("====== Token probability statistics ======"));
            assertTrue(report, report.contains("Mean    KLD"));
            assertTrue(report, report.contains("Same top p: 100.000"));
        }
    }

    @Test
    public void theKlRunRefusesMissingMismatchedAndBrokenBasesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(32, 8))) {
            PerplexityCommand.Options noBase = options("");
            noBase.klDivergence = true;
            IOException none = assertThrows(IOException.class,
                    () -> PerplexityCommand.klDivergence(m, noBase, new StringBuilder(), new StringBuilder()));
            assertTrue(none.getMessage(), none.getMessage().contains("--kl-divergence-base"));

            Path wrongVocab = tmp.newFile("wrong-vocab.bin").toPath();
            try (LogitsFile.Out out = new LogitsFile.Out(wrongVocab, 32)) {
                out.writeRun(4, 1, new int[32], 32);
                out.writeRows(new short[PerplexityMath.rowLength(4) * 15], 15, PerplexityMath.rowLength(4));
            }
            PerplexityCommand.Options vocab = options("");
            vocab.klDivergence = true;
            vocab.logitsPath = wrongVocab;
            IOException mismatch = assertThrows(IOException.class,
                    () -> PerplexityCommand.klDivergence(m, vocab, new StringBuilder(), new StringBuilder()));
            assertTrue(mismatch.getMessage(), mismatch.getMessage().contains("inconsistent vocabulary"));

            Path bigCtx = tmp.newFile("big-ctx.bin").toPath();
            try (LogitsFile.Out out = new LogitsFile.Out(bigCtx, 64)) {
                out.writeRun(320, 1, new int[64], 64);
                out.writeRows(new short[PerplexityMath.rowLength(320) * 31], 31, PerplexityMath.rowLength(320));
            }
            PerplexityCommand.Options wide = options("");
            wide.klDivergence = true;
            wide.logitsPath = bigCtx;
            IOException tooWide = assertThrows(IOException.class,
                    () -> PerplexityCommand.klDivergence(m, wide, new StringBuilder(), new StringBuilder()));
            assertTrue(tooWide.getMessage(), tooWide.getMessage().contains("--ctx-size"));

            Path corrupt = tmp.newFile("corrupt.bin").toPath();
            Files.write(corrupt, "not a base file".getBytes(StandardCharsets.US_ASCII));
            PerplexityCommand.Options broken = options("");
            broken.klDivergence = true;
            broken.logitsPath = corrupt;
            IOException bad = assertThrows(IOException.class,
                    () -> PerplexityCommand.klDivergence(m, broken, new StringBuilder(), new StringBuilder()));
            assertTrue(bad.getMessage(), bad.getMessage().contains("does not look like"));
        }
    }

    @Test
    public void aShortPromptIsRefusedWithTheCounts() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config(32, 8))) {
            PerplexityCommand.Options o = options("too short by far");
            IOException e = assertThrows(IOException.class,
                    () -> PerplexityCommand.perplexity(m, o, new StringBuilder(), new StringBuilder()));
            assertTrue(e.getMessage(), e.getMessage().contains("you need at least 64 tokens"));
            assertTrue(e.getMessage(), e.getMessage().contains("tokenizes to only"));
        }
    }

    @Test
    public void theCommandLineRefusesEveryUnsupportedCombinationByName() {
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--hellaswag", "--winogrande"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--kl-divergence", "--hellaswag"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--ppl-stride", "4", "--hellaswag"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--ppl-stride", "4",
                        "--save-all-logits", "b.bin"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--chunks", "0"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--ppl-output-type", "2"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--ppl-stride", "-1"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "--no-such-flag"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"--hellaswag"}));
        assertThrows(PerplexityCommand.UsageException.class,
                () -> PerplexityCommand.parse(new String[] {"m.gguf", "-f", "/no/such/file.txt"}));
    }

    @Test
    public void theStridedCommandLineWidensTheContextBeforeLoading() throws Exception {
        PerplexityCommand.Options o = PerplexityCommand.parse(new String[] {"m.gguf",
                "--ctx-size", "32", "--ppl-stride", "8", "-b", "16"});
        assertEquals(32, o.ctxSize);
        assertEquals(8, o.pplStride);
        assertEquals("the parse keeps the flag for the loader", 16, o.batchSize);
        PerplexityCommand.Options plain = PerplexityCommand.parse(new String[] {"m.gguf"});
        assertEquals(PerplexityCommand.DEFAULT_CTX_SIZE, plain.ctxSize);
        assertEquals(PerplexityCommand.DEFAULT_HELASWAG_TASKS, plain.hellaswagTasks);
    }

    @Test
    public void theLauncherDispatchesTheBenchmarkFromTheCommandLine() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        Path data = tmp.newFile("hellaswag.txt").toPath();
        Files.write(data, hellaswagTasks(2).getBytes(StandardCharsets.UTF_8));
        PrintStream realOut = System.out;
        PrintStream realErr = System.err;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured, true, "UTF-8"));
            System.setErr(new PrintStream(new ByteArrayOutputStream(), true, "UTF-8"));
            PerplexityCommand.dispatch(new String[] {model.toString(), "--hellaswag",
                    "--hellaswag-tasks", "2", "-f", data.toString(), "-c", "64", "-b", "8"},
                    config(64, 8));
        } finally {
            System.setOut(realOut);
            System.setErr(realErr);
        }
        String stdout = captured.toString("UTF-8");
        assertTrue(stdout, stdout.contains("task\tacc_norm\t95% confidence interval"));
        assertEquals("one line per scored task", 2, occurrences(stdout, "%\t["));
    }

    // ------------------------------------------------------------------ helpers

    private static int occurrences(String s, String token) {
        int count = 0;
        int at = 0;
        while ((at = s.indexOf(token, at)) >= 0) {
            count++;
            at += token.length();
        }
        return count;
    }

    /**
     * The numeric columns of the KL table, one array per row line: chunk, PPL ± , ratio ± , KLD ± ,
     * Δp RMS ± , top p ± — the KLD mean at index 5 and the top agreement at index 9. The report's other
     * lines carry fewer numbers and drop out.
     */
    private static List<double[]> tableRows(String report) {
        List<double[]> rows = new ArrayList<>();
        for (String line : report.split("\n")) {
            List<Double> numbers = new ArrayList<>();
            for (String token : line.trim().split("[\\s±%]+")) {
                if (token.isEmpty()) {
                    continue;
                }
                try {
                    numbers.add(Double.parseDouble(token));
                } catch (NumberFormatException e) {
                    // the header words and the statistics labels are not rows
                }
            }
            if (numbers.size() >= 10) {
                double[] row = new double[numbers.size()];
                for (int i = 0; i < row.length; i++) {
                    row[i] = numbers.get(i);
                }
                rows.add(row);
            }
        }
        return rows;
    }

    /** A HellaSwag datafile of {@code n} tasks whose four endings share the context's prefix. */
    private static String hellaswagTasks(int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append("He said\n");
            sb.append(i % 4).append('\n');
            sb.append("yes today\n");
            sb.append("no today\n");
            sb.append("yes tomorrow\n");
            sb.append("no tomorrow\n");
        }
        return sb.toString();
    }
}

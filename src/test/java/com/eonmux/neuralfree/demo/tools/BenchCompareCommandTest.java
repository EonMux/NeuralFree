package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.demo.tools.compare.BenchComparison;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool bench-compare} over result files in the layouts the bench tools write: rows keyed by their
 * configuration, sides pooled across files, Welch's p-value over the raw samples, the verdict against the threshold
 * in the metric's better direction, and the {@code --fail-on-regress} exit status.
 */
public class BenchCompareCommandTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static String benchRow(String model, int nPrompt, int nGen, double... samplesTs) {
        return benchRow(model, "", nPrompt, nGen, samplesTs);
    }

    /** A bench row with the JSON members {@code extra} (each followed by a comma) after its model fields. */
    private static String benchRow(String model, String extra, int nPrompt, int nGen, double... samplesTs) {
        double sum = 0;
        StringBuilder ts = new StringBuilder();
        StringBuilder ns = new StringBuilder();
        for (double s : samplesTs) {
            sum += s;
            ts.append(ts.length() == 0 ? "" : ", ").append(s);
            ns.append(ns.length() == 0 ? "" : ", ").append((long) (1e9 * (nPrompt + nGen) / s));
        }
        double avg = sum / samplesTs.length;
        return "{\"build_commit\": \"abc\", \"model_filename\": \"" + model + "\", \"model_type\": \"qwen3 0.6B\","
                + " \"backends\": \"CUDA\", " + extra + " \"n_batch\": 2048, \"n_ubatch\": 512, \"n_gpu_layers\": 99,"
                + " \"type_k\": \"f16\", \"type_v\": \"f16\", \"embeddings\": false, \"n_prompt\": " + nPrompt
                + ", \"n_gen\": " + nGen + ", \"n_depth\": 0, \"avg_ns\": 1, \"stddev_ns\": 0, \"avg_ts\": " + avg
                + ", \"stddev_ts\": 0.0, \"samples_ns\": [" + ns + "], \"samples_ts\": [" + ts + "]}";
    }

    private Path file(String name, String text) throws IOException {
        Path f = tmp.getRoot().toPath().resolve(name);
        Files.write(f, text.getBytes(StandardCharsets.UTF_8));
        return f;
    }

    private static final class Run {
        final int status;
        final String out;

        Run(int status, String out) {
            this.status = status;
            this.out = out;
        }
    }

    private static Run run(String... args) throws Exception {
        BenchCompareCommand.Args parsed = BenchCompareCommand.parse(args);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        int status;
        try (PrintStream out = new PrintStream(bytes, true, "UTF-8")) {
            status = BenchCompareCommand.compare(parsed, out);
        }
        return new Run(status, bytes.toString("UTF-8"));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> entries(Run run) {
        Map<String, Object> report = (Map<String, Object>) Json.parse(run.out);
        return (List<Map<String, Object>>) report.get("entries");
    }

    @Test
    public void aSignificantDropPastTheThresholdFailsTheGate() throws Exception {
        Path base = file("base.json", "[" + benchRow("m.gguf", 0, 128, 480, 482, 479, 481, 483) + ",\n"
                + benchRow("m.gguf", 512, 0, 9000, 9010, 8990) + "]");
        Path next = file("new.json", "[" + benchRow("m.gguf", 0, 128, 440, 441, 439, 442, 438) + ",\n"
                + benchRow("m.gguf", 512, 0, 9005, 9000, 8995) + "]");
        Run gated = run("--fail-on-regress", "-o", "json", base.toString(), next.toString());
        assertEquals(BenchCompareCommand.REGRESSION_EXIT, gated.status);
        List<Map<String, Object>> entries = entries(gated);
        assertEquals(2, entries.size());
        Map<String, Object> tg = entries.get(0);
        assertEquals("tg128", tg.get("test"));
        assertEquals("regression", tg.get("verdict"));
        assertEquals(5L, tg.get("base_samples"));
        assertEquals(440.0 / 481.0, (Double) tg.get("ratio"), 1e-12);
        assertTrue((Double) tg.get("p") < 1e-6);
        assertEquals("unchanged", entries.get(1).get("verdict"));
        assertEquals(0, run(base.toString(), next.toString()).status);
    }

    @Test
    public void aDropInsideTheNoiseIsNotSignificant() throws Exception {
        Path base = file("base.jsonl", benchRow("m.gguf", 0, 128, 400, 520, 450) + "\n");
        Path next = file("new.jsonl", benchRow("m.gguf", 0, 128, 380, 500, 430) + "\n");
        Run r = run("--fail-on-regress", "-o", "json", base.toString(), next.toString());
        assertEquals(0, r.status);
        assertEquals("not significant", entries(r).get(0).get("verdict"));
    }

    @Test
    public void theFilesOfASidePoolTheirSamples() throws Exception {
        Path b1 = file("b1.jsonl", benchRow("m.gguf", 0, 128, 480, 481));
        Path b2 = file("b2.jsonl", benchRow("m.gguf", 0, 128, 482, 479));
        Path n1 = file("n1.jsonl", benchRow("m.gguf", 0, 128, 500, 501));
        Path n2 = file("n2.jsonl", benchRow("m.gguf", 0, 128, 502, 499));
        Run r = run("-o", "json", b1 + "," + b2, n1 + "," + n2);
        Map<String, Object> e = entries(r).get(0);
        assertEquals(4L, e.get("base_samples"));
        assertEquals(4L, e.get("new_samples"));
        assertEquals(480.5, (Double) e.get("base"), 1e-12);
        assertEquals(500.5, (Double) e.get("new"), 1e-12);
        assertEquals("improvement", e.get("verdict"));
    }

    @Test
    public void aModelPathMatchesItsFileNameAndUnmatchedTestsAreListed() throws Exception {
        Path base = file("base.jsonl", benchRow("/models/a/m.gguf", 0, 128, 100, 101) + "\n"
                + benchRow("/models/a/m.gguf", 0, 256, 99, 98));
        Path next = file("new.jsonl", benchRow("m.gguf", 0, 128, 100, 101) + "\n"
                + benchRow("m.gguf", 0, 64, 103, 104));
        Run md = run(base.toString(), next.toString());
        assertTrue(md.out, md.out.contains("| bench | tg128 |"));
        assertTrue(md.out, md.out.contains("only in base: bench tg256"));
        assertTrue(md.out, md.out.contains("only in new: bench tg64"));
    }

    @Test
    public void aLatencyMetricImprovesDownward() throws Exception {
        String base = "{\"date\": \"20260101-000000\", \"backend\": \"openai\", \"endpoint_type\": \"openai\","
                + " \"model_id\": \"qwen\", \"num_prompts\": 4, \"request_rate\": \"inf\", \"burstiness\": 1.0,"
                + " \"max_concurrency\": null, \"duration\": 2.0, \"completed\": 4, \"output_throughput\": 100.0,"
                + " \"mean_ttft_ms\": 50.0, \"ttfts\": [0.049, 0.050, 0.051, 0.050]}";
        String next = base.replace("\"mean_ttft_ms\": 50.0", "\"mean_ttft_ms\": 40.0")
                .replace("[0.049, 0.050, 0.051, 0.050]", "[0.039, 0.040, 0.041, 0.040]");
        Run r = run("--metric", "mean_ttft_ms", "-o", "json", file("b.json", base).toString(),
                file("n.json", next).toString());
        Map<String, Object> e = entries(r).get(0);
        assertEquals(false, e.get("higher_is_better"));
        assertEquals(20.0, (Double) e.get("gain_pct"), 1e-9);
        assertEquals("the samples are the per-request seconds in milliseconds", 50.0, (Double) e.get("base"), 1e-9);
        assertEquals("improvement", e.get("verdict"));
    }

    @Test
    public void batchedBenchCellsMatchAcrossFlagSpellingsAndCompareOneValuePerRow() throws Exception {
        String cell = "{\"n_kv_max\": 4096, \"n_batch\": 2048, \"n_ubatch\": 512, \"flash_attn\": 0,"
                + " \"is_pp_shared\": 0, \"is_tg_separate\": 0, \"n_gpu_layers\": 99, \"n_threads\": 8,"
                + " \"n_threads_batch\": 8, \"pp\": 128, \"tg\": 128, \"pl\": %d, \"n_kv\": 1024, \"t_pp\": 0.1,"
                + " \"speed_pp\": 5000.0, \"t_tg\": 1.0, \"speed_tg\": %s, \"t\": 1.1, \"speed\": %s}";
        Path base = file("b.jsonl", String.format(cell, 4, "400.0", "900.0") + "\n"
                + String.format(cell, 8, "700.0", "1500.0"));
        String nftoolCell = cell.replace("\"is_pp_shared\": 0, \"is_tg_separate\": 0",
                "\"is_pp_shared\": false, \"is_tg_separate\": false");
        Path next = file("n.jsonl", String.format(nftoolCell, 4, "420.0", "950.0") + "\n"
                + String.format(nftoolCell, 8, "600.0", "1400.0"));
        Run r = run("--metric", "speed_tg", "--fail-on-regress", "-o", "json", base.toString(), next.toString());
        List<Map<String, Object>> entries = entries(r);
        assertEquals("pp128 tg128 pl4", entries.get(0).get("test"));
        assertEquals("one value per row leaves no t-test", null, entries.get(0).get("p"));
        assertEquals("improvement", entries.get(0).get("verdict"));
        assertEquals("regression", entries.get(1).get("verdict"));
        assertEquals(BenchCompareCommand.REGRESSION_EXIT, r.status);
    }

    @Test
    public void anObjectThatIsNoBenchResultIsRefusedWithItsPosition() throws Exception {
        Path base = file("b.jsonl", benchRow("m.gguf", 0, 128, 1, 2) + "\n{\"hello\": 1}\n");
        Path next = file("n.jsonl", benchRow("m.gguf", 0, 128, 1, 2));
        IOException e = assertThrows(IOException.class, () -> run(base.toString(), next.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("b.jsonl:2"));
    }

    @Test
    public void sidesWithoutACommonTestAreAnError() throws Exception {
        Path base = file("b.jsonl", benchRow("m.gguf", 0, 128, 1, 2));
        Path next = file("n.jsonl", benchRow("other.gguf", 0, 128, 1, 2));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> run(base.toString(), next.toString()));
        assertTrue(e.getMessage(), e.getMessage().contains("share no test"));
    }

    @Test
    public void theCommandLineIsValidated() throws Exception {
        assertNull(BenchCompareCommand.parse(new String[]{"--help"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"only-one.json"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"-o", "csv", "a", "b"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"--threshold", "-1", "a", "b"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"--alpha", "1.5", "a", "b"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"--bogus", "a", "b"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"--key", "a,,b", "a", "b"}));
        assertThrows(BenchCompareCommand.UsageException.class,
                () -> BenchCompareCommand.parse(new String[]{"a,", "b"}));
    }

    @Test
    public void aCustomKeyMergesTheRowsItDoesNotName() throws Exception {
        Path base = file("b.jsonl", benchRow("m.gguf", 0, 128, 480, 481) + "\n" + benchRow("m.gguf", 0, 256, 470, 471));
        Path next = file("n.jsonl", benchRow("m.gguf", 0, 128, 480, 481) + "\n" + benchRow("m.gguf", 0, 256, 470, 471));
        Run r = run("--key", "model_filename", "-o", "json", base.toString(), next.toString());
        List<Map<String, Object>> entries = entries(r);
        assertEquals(1, entries.size());
        assertEquals(4L, entries.get(0).get("base_samples"));
        assertEquals(BenchComparison.Verdict.UNCHANGED.label(), entries.get(0).get("verdict"));
        Run md = run("--key", "model_filename", base.toString(), next.toString());
        assertTrue("the merge of different test shapes is reported: " + md.out,
                md.out.contains("pooled: base bench tg128 {model_filename=m.gguf}") && md.out.contains(
                        "pools rows that differ in n_gen: '128' (" + base + ":1) '256' (" + base + ":2)"));
        assertTrue(md.out, md.out.contains("pooled: new bench tg128"));
    }

    @Test
    public void rowsOfDifferentThreadCountsOrSplitModesDoNotPool() throws Exception {
        Path base = file("b.jsonl", benchRow("m.gguf", "\"n_threads\": 4,", 0, 128, 50, 50.5) + "\n"
                + benchRow("m.gguf", "\"n_threads\": 16,", 0, 128, 150, 150.5));
        Path next = file("n.jsonl", benchRow("m.gguf", "\"n_threads\": 4,", 0, 128, 50, 50.5));
        Run r = run("-o", "json", base.toString(), next.toString());
        List<Map<String, Object>> entries = entries(r);
        assertEquals(1, entries.size());
        assertEquals("the 4-thread rows compare alone", 1.0, (Double) entries.get(0).get("ratio"), 1e-12);
        assertEquals(BenchComparison.Verdict.UNCHANGED.label(), entries.get(0).get("verdict"));
        Run md = run(base.toString(), next.toString());
        assertTrue(md.out, md.out.contains("only in base: bench tg128") && md.out.contains("n_threads=16"));
        for (String field : new String[]{"\"n_cpu_moe\": %d,", "\"split_mode\": \"%s\",",
                "\"flash_attn\": %d,", "\"tg_readout\": \"%s\","}) {
            boolean text = field.contains("%s");
            Path b = file("b2.jsonl", benchRow("m.gguf", String.format(field, text ? "one" : 1), 0, 128, 50, 51));
            Path n = file("n2.jsonl", benchRow("m.gguf", String.format(field, text ? "two" : 2), 0, 128, 50, 51));
            IllegalArgumentException e = assertThrows(field, IllegalArgumentException.class,
                    () -> run(b.toString(), n.toString()));
            assertTrue(e.getMessage(), e.getMessage().contains("share no test"));
        }
    }

    @Test
    public void aRowWithoutTheReadoutKeysAsTheLogitsReadout() throws Exception {
        Path llama = file("llama.jsonl", benchRow("m.gguf", 0, 128, 600, 601, 602));
        Path logits = file("logits.jsonl", benchRow("m.gguf", "\"tg_readout\": \"logits\",", 0, 128, 540, 541));
        Path argmax = file("argmax.jsonl", benchRow("m.gguf", "\"tg_readout\": \"argmax\",", 0, 128, 545, 546));
        Run r = run("-o", "json", llama.toString(), logits + "," + argmax);
        List<Map<String, Object>> entries = entries(r);
        assertEquals("only the logits readout pairs with llama-bench's row", 1, entries.size());
        assertEquals(540.5, (Double) entries.get(0).get("new"), 1e-12);
        Run md = run(llama.toString(), logits + "," + argmax);
        assertTrue(md.out, md.out.contains("only in new: bench tg128") && md.out.contains("tg_readout=argmax"));
    }
}

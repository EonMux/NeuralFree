package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The printers of {@code nftool bench} against llama.cpp's five formats: the all-quoted CSV rows, the JSON
 * array with the raw samples, the JSONL objects, the Markdown table whose columns are only the parameters
 * that vary and whose cells carry the {@code pp}/{@code tg} names and the {@code avg ± stddev} t/s, and the
 * SQL DDL plus INSERTs. One finished {@link BenchTest} feeds them all; its samples {1 ms, 3 ms} over 64
 * tokens average 42666.67 t/s with a 30169.89 spread.
 */
public class BenchPrintersTest {

    private static BenchGrid grid(String... flags) throws BenchCommand.UsageException {
        return BenchCommand.parse(flags);
    }

    private static BenchTest test() {
        BenchInstance inst = new BenchInstance("m.gguf", 64, 0, 8, 1024, 128, KvDType.Q8_0, KvDType.F16, 0);
        return new BenchTest(inst, "llama tiny", 1024, 128, "CPU", 72, BenchReadout.LOGITS,
                new long[]{1_000_000L, 3_000_000L});
    }

    @Test
    public void csvQuotesEveryValueAndNamesEveryField() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.CSV, out, g);
        p.printHeader(g);
        p.printTest(test());
        List<String> lines = lines(out);
        assertEquals("model_filename,model_type,model_size,model_n_params,backends,"
                + "n_batch,n_ubatch,n_threads,n_ctx,type_k,type_v,n_gpu_layers,n_cpu_moe,"
                + "split_mode,main_gpu,no_kv_offload,devices,tensor_split,tensor_buft_overrides,"
                + "load_mode,embeddings,tg_readout,n_prompt,n_gen,n_depth,test_time,"
                + "avg_ns,stddev_ns,avg_ts,stddev_ts", lines.get(0));
        String row = lines.get(1);
        assertTrue("the filename leads the row: " + row, row.startsWith("\"m.gguf\",\"llama tiny\",\"1024\",\"128\","
                + "\"CPU\",\"1024\",\"128\",\"" + test().inst.nThreads + "\",\"72\",\"q8_0\",\"f16\",\"0\",\"0\","
                + "\"none\",\"0\",\"0\",\"auto\",\"0.00\",\"none\",\"auto\",\"0\",\"logits\",\"64\",\"0\",\"8\","));
        assertTrue("the row carries the mean and the spread: " + row,
                row.contains("\",\"2000000\",\"1414214\",\"42666.666667\","));
        assertEquals(2, lines.size());
    }

    @Test
    public void csvEscapesQuotesByDoublingThem() {
        assertEquals("\"a\"\"b\"", BenchPrinters.escapeCsv("a\"b"));
        assertEquals("\"plain\"", BenchPrinters.escapeCsv("plain"));
    }

    @Test
    public void jsonGathersTheTestsWithTheirSamples() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.JSON, out, g);
        p.printHeader(g);
        p.printTest(test());
        p.printTest(test());
        p.printFooter();
        List<String> lines = lines(out);
        assertEquals("[", lines.get(0));
        assertEquals("  {", lines.get(1));
        assertTrue(lines.get(2).startsWith("    \"model_filename\": \"m.gguf\","));
        assertTrue("strings stay quoted: " + lines.get(11), lines.get(11).startsWith("    \"type_k\": \"q8_0\","));
        assertTrue("booleans stay bare: " + lines.get(17), lines.get(17).startsWith("    \"no_kv_offload\": 0,"));
        assertTrue("strings stay quoted: " + lines.get(23), lines.get(23).startsWith("    \"tg_readout\": \"logits\","));
        assertTrue("ints stay bare: " + lines.get(24), lines.get(24).startsWith("    \"n_prompt\": 64,"));
        assertTrue(lines.get(lines.size() - 4).startsWith("    \"samples_ns\": [ 1000000, 3000000 ],"));
        assertTrue(lines.get(lines.size() - 3).startsWith("    \"samples_ts\": [ "));
        assertEquals("  }", lines.get(lines.size() - 2));
        assertEquals("]", lines.get(lines.size() - 1));
        long objects = out.toString().chars().filter(c -> c == '{').count();
        assertEquals(2, objects);
    }

    @Test
    public void jsonlWritesOneObjectPerLine() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.JSONL, out, grid("-m", "m.gguf"));
        p.printHeader(grid("-m", "m.gguf"));
        p.printTest(test());
        List<String> lines = lines(out);
        assertEquals(1, lines.size());
        String line = lines.get(0);
        assertTrue("the object opens with the first field: " + line,
                line.startsWith("{\"model_filename\": \"m.gguf\", "));
        assertTrue("the samples close the object: " + line,
                line.contains("\"samples_ns\": [ 1000000, 3000000 ],")
                        && line.contains("\"samples_ts\": [ ") && line.endsWith("}"));
    }

    @Test
    public void markdownPrintsOnlyTheColumnsThatVary() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf", "-ctk", "q8_0");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.MARKDOWN, out, g);
        p.printHeader(g);
        p.printTest(test());
        String header = lines(out).get(0);
        assertTrue("the varying cache type gains a column: " + header, header.contains(" type_k |"));
        assertTrue("the depth lives in the test cell, not a column: " + header, !header.contains(" n_depth "));
        assertTrue(header.contains(" test |"));
        assertTrue(header.contains(" t/s |"));
        assertTrue("the model column leads, left-aligned: " + header, header.startsWith("| model"));
        String row = lines(out).get(2);
        assertTrue("the test cell names the shape and depth: " + row, row.contains(" pp64 @ d8 "));
        assertTrue("the t/s cell carries avg ± stddev: " + row, row.contains("42666.67 ± 30169.89"));
    }

    @Test
    public void markdownKeepsTheDefaultSweepLean() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf", "-ngl", "0");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.MARKDOWN, out, g);
        p.printHeader(g);
        String header = lines(out).get(0);
        for (String column : new String[]{"ngl", "n_batch", "n_ubatch", "type_k", "type_v"}) {
            assertTrue("the default " + column + " is no column: " + header, !header.contains(" " + column + " |"));
        }
    }

    @Test
    public void markdownShowsTheSweptSplitAndKvOffloadColumns() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf", "-sm", "none,layer", "-mg", "0,1", "-ts", "3/1", "-nkvo", "0,1");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.MARKDOWN, out, g);
        p.printHeader(g);
        com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode layer =
                com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER;
        BenchInstance inst = new BenchInstance("m.gguf", 64, 0, 0, 1024, 128, KvDType.F16, KvDType.F16, 0,
                true, layer, 1, new double[]{3.0, 1.0});
        p.printTest(new BenchTest(inst, "llama tiny", 1024, 128, "CPU", 64, BenchReadout.LOGITS,
                new long[]{1_000_000L}));
        List<String> lines = lines(out);
        String header = lines.get(0);
        assertTrue("the main GPU gains a column: " + header, header.contains(" main_gpu |"));
        assertTrue("the split mode gains its abbreviation: " + header, header.contains(" sm |"));
        assertTrue("the KV offload gains its abbreviation: " + header, header.contains(" nkvo |"));
        assertTrue("the tensor split gains its abbreviation: " + header, header.contains("| ts "));
        String row = lines.get(2);
        assertTrue("the row carries the layer split: " + row, row.contains(" layer "));
        assertTrue("the row carries the main GPU: " + row, row.contains(" 1 "));
        assertTrue("the row carries the host KV flag: " + row, row.contains(" 1 "));
        assertTrue("the row carries the shares: " + row, row.contains(" 3.00/1.00 "));
    }

    @Test
    public void markdownShowsTheDevicePlacementLoadModeAndEmbeddingColumns() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchGrid g = grid("-m", "m.gguf", "-t", "4", "-ncmoe", "0,2", "-dev", "CUDA0,none",
                "-ot", "a=CPU", "-lm", "none", "-embd", "1");
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.MARKDOWN, out, g);
        p.printHeader(g);
        String header = lines(out).get(0);
        for (String column : new String[]{" threads |", " n_cpu_moe |", " dev ", " ot ", " lm |", " embd |"}) {
            assertTrue("the swept parameter gains its column: " + header, header.contains(column));
        }
    }

    @Test
    public void sqlCreatesTheTableAndInsertsTheRows() throws Exception {
        StringBuilder out = new StringBuilder();
        BenchPrinters.BenchPrinter p = BenchPrinters.create(BenchOutput.SQL, out, grid("-m", "m.gguf"));
        p.printHeader(grid("-m", "m.gguf"));
        p.printTest(test());
        List<String> lines = lines(out);
        assertEquals("CREATE TABLE IF NOT EXISTS llama_bench (", lines.get(0));
        assertEquals("  model_filename TEXT,", lines.get(1));
        assertEquals("  n_threads INTEGER,", lines.get(8));
        assertEquals("  no_kv_offload INTEGER,", lines.get(16));
        assertEquals("  devices TEXT,", lines.get(17));
        assertEquals("  tensor_buft_overrides TEXT,", lines.get(19));
        assertEquals("  embeddings INTEGER,", lines.get(21));
        assertEquals("  tg_readout TEXT,", lines.get(22));
        assertEquals("  avg_ns INTEGER,", lines.get(27));
        assertEquals("  avg_ts REAL,", lines.get(29));
        assertEquals("  stddev_ts REAL", lines.get(30));
        assertEquals(");", lines.get(31));
        assertTrue("the insert lists every field: " + lines.get(33),
                lines.get(33).startsWith("INSERT INTO llama_bench (model_filename, model_type, model_size,"
                        + " model_n_params, backends, n_batch, n_ubatch, n_threads, n_ctx, type_k, type_v,"
                        + " n_gpu_layers, n_cpu_moe, split_mode, main_gpu, no_kv_offload, devices, tensor_split,"
                        + " tensor_buft_overrides, load_mode, embeddings, tg_readout,"
                        + " n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts) VALUES ("));
        assertTrue("every value is single-quoted: " + lines.get(33),
                lines.get(33).contains("'m.gguf', 'llama tiny', '1024',"));
    }

    @Test
    public void noneHasNoPrinterAndJsonEscapesItsStrings() throws Exception {
        assertNull(BenchPrinters.create(BenchOutput.NONE, new StringBuilder(), grid("-m", "m.gguf")));
        assertEquals("say \\\"hi\\\"", BenchPrinters.escapeJson("say \"hi\""));
        assertEquals("tab \\u0009 here", BenchPrinters.escapeJson("tab \t here"));
        assertEquals("\"f16\"", BenchPrinters.jsonValue("type_k", "f16"));
        assertEquals("128", BenchPrinters.jsonValue("n_ubatch", "128"));
    }

    /** The output's lines, without the empty artifact of the trailing newline. */
    private static List<String> lines(StringBuilder out) {
        return Arrays.asList(out.toString().split("\n"));
    }
}

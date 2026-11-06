package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool batched-bench} over the synthetic two-layer llama: the {@code -npp x -ntg x -npl} grid
 * reports one llama.cpp table row per cell with the reference's column layout and the reference's speed
 * formulas, {@code -pps} shares one prompt through the page copy (its {@code n_kv} is the unified-KV
 * formula {@code pp + pl*tg}), {@code -tgs} runs, {@code --output-format jsonl} prints the reference's
 * keys, a cell that does not fit is skipped with a standard-error note and the parser refuses every bad
 * flag by name. The load flags LLMDemo shares ({@code --backend}, {@code -ngl}, {@code -fa}, the KV types) reach
 * the engine configuration over the bench's paged memory, and with a GPU present a small grid runs on the GPU
 * backend's paged device memory.
 */
public class BatchedBenchCommandTest {

    /** The CPU engine the table and refusal runs open on. */
    private static final EngineConfig CPU = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();

    private String savedBackend;

    @Before
    public void clearBackendProperty() {
        savedBackend = System.getProperty(EngineConfig.PROP_BACKEND);
        System.clearProperty(EngineConfig.PROP_BACKEND);
    }

    @After
    public void restoreBackendProperty() {
        if (savedBackend == null) {
            System.clearProperty(EngineConfig.PROP_BACKEND);
        } else {
            System.setProperty(EngineConfig.PROP_BACKEND, savedBackend);
        }
    }

    private static Path model() throws IOException {
        return SyntheticModels.llama().writeTempFile();
    }

    private static BatchedBenchCommand.Plan plan(Path model, int[] npp, int[] ntg, int[] npl,
            boolean pps, boolean tgs, boolean jsonl, int ctx) {
        return new BatchedBenchCommand.Plan(model, CPU, list(npp), list(ntg), list(npl),
                pps, tgs, jsonl, ctx, 8, 1L);
    }

    private static List<Integer> list(int[] values) {
        List<Integer> out = new ArrayList<>();
        for (int v : values) {
            out.add(v);
        }
        return out;
    }

    /** The cells of a table run: one parsed row per cell, {pp, tg, pl, nKv, t_pp, s_pp, t_tg, s_tg, t, s}. */
    private static List<double[]> rows(String table) {
        List<double[]> out = new ArrayList<>();
        for (String line : table.split("\n")) {
            if (!line.startsWith("|") || line.contains("PP") || line.contains("-")) {
                continue;
            }
            String[] parts = line.split("\\|");
            double[] row = new double[parts.length - 1];
            for (int i = 1; i < parts.length; i++) {
                row[i - 1] = Double.parseDouble(parts[i].trim());
            }
            out.add(row);
        }
        return out;
    }

    @Test
    public void theMatrixReportsOneReferenceRowPerCell() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(plan(model,
                new int[] {4, 8}, new int[] {3}, new int[] {1, 2}, false, false, false, 64), out, err);
        assertTrue("the run skips nothing: " + err, err.length() == 0);
        String table = out.toString();
        assertTrue("the header names the backend: " + table, table.contains("is_tg_separate = false, backend = cpu\n"));
        assertTrue("the reference header: " + table,
                table.contains("|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s |"
                        + " S_TG t/s |      T s |    S t/s |\n"
                        + "|-------|--------|------|--------|----------|----------|----------|----------|"
                        + "----------|----------|\n"));
        double[][] expected = {{4, 3, 1, 7}, {4, 3, 2, 14}, {8, 3, 1, 11}, {8, 3, 2, 22}};
        List<double[]> parsed = rows(table);
        assertEquals("one row per grid cell", expected.length, parsed.size());
        for (int i = 0; i < expected.length; i++) {
            double[] row = parsed.get(i);
            assertEquals("pp", expected[i][0], row[0], 0);
            assertEquals("tg", expected[i][1], row[1], 0);
            assertEquals("pl", expected[i][2], row[2], 0);
            assertEquals("n_kv = pl*(pp+tg)", expected[i][3], row[3], 0);
            assertEquals("the printed row is the measured cell", (int) row[3], measured.get(i).nKv);
            assertRowSpeeds(measured.get(i), false);
        }
    }

    /**
     * The cell's speeds carry the reference's formulas: (pl*)pp/t_pp (one prompt under -pps), pl*tg/t_tg and the
     * total over t = t_pp + t_tg.
     */
    private static void assertRowSpeeds(BatchedBenchCommand.BenchRow row, boolean ppShared) {
        double ppTokens = ppShared ? row.pp : (double) row.pl * row.pp;
        double tgTokens = (double) row.pl * row.tg;
        assertTrue(row.tPP > 0);
        assertEquals("speed_pp", ppTokens / row.tPP, row.speedPP, 1e-9 * row.speedPP);
        assertTrue(row.tTG > 0);
        assertEquals("speed_tg = pl*tg/t_tg", tgTokens / row.tTG, row.speedTG, 1e-9 * row.speedTG);
        assertEquals("t = t_pp + t_tg", row.tPP + row.tTG, row.t, 1e-12);
        assertEquals("speed", (ppTokens + tgTokens) / row.t, row.speed, 1e-9 * row.speed);
    }

    @Test
    public void ppsSharesOnePromptThroughThePageCopy() throws IOException {
        Path model = model();
        StringBuilder shared = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(plan(model, new int[] {8},
                new int[] {3}, new int[] {2}, true, false, false, 64), shared, new StringBuilder());
        StringBuilder separate = new StringBuilder();
        BatchedBenchCommand.execute(plan(model, new int[] {8}, new int[] {3}, new int[] {2},
                false, false, false, 64), separate, new StringBuilder());
        // llama.cpp's unified-KV accounting: pp + pl*tg rows instead of pl*(pp+tg)
        assertEquals(14, rows(shared.toString()).get(0)[3], 0);
        assertEquals(22, rows(separate.toString()).get(0)[3], 0);
        assertRowSpeeds(measured.get(0), true);
    }

    @Test
    public void ppsAdmitsACellTheSeparatePromptsDoNotFit() throws IOException {
        Path model = model();
        // n_kv_max 128: pl=3 separate prompts need 3*(40+4)=132 rows, one shared prompt 40+3*4=52
        StringBuilder separateErr = new StringBuilder();
        StringBuilder separate = new StringBuilder();
        BatchedBenchCommand.execute(plan(model, new int[] {40}, new int[] {4}, new int[] {2, 3},
                false, false, false, 128), separate, separateErr);
        assertEquals("only pl=2 fits", 1, rows(separate.toString()).size());
        assertTrue(separateErr.toString(), separateErr.toString().contains("pp=40 tg=4 pl=3"));

        StringBuilder sharedErr = new StringBuilder();
        StringBuilder shared = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(plan(model, new int[] {40},
                new int[] {4}, new int[] {2, 3}, true, false, false, 128), shared, sharedErr);
        List<double[]> parsed = rows(shared.toString());
        assertEquals("both cells fit: " + sharedErr, 2, parsed.size());
        assertEquals(48, parsed.get(0)[3], 0);
        assertEquals(52, parsed.get(1)[3], 0);
        for (BatchedBenchCommand.BenchRow row : measured) {
            assertRowSpeeds(row, true);
        }
    }

    @Test
    public void aCellThatRunsThePagedPoolsOutIsSkippedAndTheGridContinues() throws IOException {
        Path model = model();
        // n_kv_max 32 holds two 16-row pages: the shared 12-row prompt takes one, and the first decode of each
        // of three sequences privatizes it, which needs two more pages than the pools hold
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(plan(model, new int[] {12},
                new int[] {2}, new int[] {3, 1}, true, false, false, 32), out, err);
        assertTrue(err.toString(), err.toString().contains("skipped pp=12 tg=2 pl=3: n_kv 18 ran the paged pools out"));
        assertEquals("the next cell runs on the reset memory", 1, measured.size());
        assertEquals(1, measured.get(0).pl);
        assertEquals(14, measured.get(0).nKv);
    }

    @Test
    public void theDefaultContextIsTheTrainedOne() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        BatchedBenchCommand.execute(plan(model, new int[] {4}, new int[] {2}, new int[] {1},
                false, false, true, BatchedBenchCommand.DEFAULT_CTX), out, new StringBuilder());
        String line = out.toString().trim();
        assertTrue("the synthetic llama trains 256 positions: " + line,
                line.startsWith("{\"n_kv_max\": 256,"));
    }

    @Test
    public void tgSeparateRunsAndReportsTheSameCells() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(plan(model, new int[] {4},
                new int[] {2}, new int[] {2}, false, true, false, 64), out, new StringBuilder());
        assertEquals(12, rows(out.toString()).get(0)[3], 0);
        assertRowSpeeds(measured.get(0), false);
    }

    @Test
    public void jsonlCarriesTheReferenceKeysPerCell() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        BatchedBenchCommand.execute(plan(model, new int[] {4}, new int[] {3}, new int[] {1},
                false, false, true, 64), out, new StringBuilder());
        String line = out.toString().trim();
        assertTrue(line, line.startsWith("{\"n_kv_max\": 64, \"n_ubatch\": 8, \"parallel\": 1,"
                + " \"is_pp_shared\": false, \"is_tg_separate\": false, \"pp\": 4, \"tg\": 3, \"pl\": 1,"
                + " \"n_kv\": 7,"));
        assertTrue(line, line.contains("\"t_pp\": ") && line.contains("\"speed_pp\": ")
                && line.contains("\"t_tg\": ") && line.contains("\"speed_tg\": ")
                && line.contains("\"t\": ") && line.contains("\"speed\": "));
    }

    @Test
    public void aCellThatDoesNotFitIsSkippedWithANoteAndTheGridContinues() throws IOException {
        Path model = model();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        BatchedBenchCommand.execute(plan(model, new int[] {50, 10}, new int[] {20}, new int[] {1},
                false, false, false, 64), out, err);
        List<double[]> parsed = rows(out.toString());
        assertEquals("only the fitting cell reports", 1, parsed.size());
        assertEquals(30, parsed.get(0)[3], 0);
        String note = err.toString();
        assertTrue(note, note.contains("pp=50 tg=20 pl=1"));
        assertTrue(note, note.contains("n_kv 70") && note.contains("n_kv_max 64"));
        assertFalse("the skipped cell prints no table row", out.toString().contains("|   50 "));
    }

    @Test
    public void parseDefaultsAndFlagsFollowLlamaCpp() throws Exception {
        BatchedBenchCommand.Plan p = BatchedBenchCommand.parse(new String[] {"m.gguf",
                "-npp", "4,8", "-ntg", "3", "-npl", "1"});
        assertEquals("m.gguf", p.model.toString());
        assertEquals("llama.cpp's n_ctx default: the trained context", 0, p.ctx);
        assertEquals(512, p.ubatch);
        assertEquals(1L, p.seed);
        assertFalse(p.ppShared);
        assertFalse(p.tgSeparate);
        assertFalse(p.jsonl);
        assertEquals(1, p.maxPl);

        BatchedBenchCommand.Plan full = BatchedBenchCommand.parse(new String[] {"m.gguf",
                "-pps", "-tgs", "--output-format", "jsonl", "-c", "128", "-ub", "16", "-s", "7",
                "-npp", "4", "-ntg", "3", "-npl", "1,2"});
        assertTrue(full.ppShared);
        assertTrue(full.tgSeparate);
        assertTrue(full.jsonl);
        assertEquals(128, full.ctx);
        assertEquals(16, full.ubatch);
        assertEquals(7L, full.seed);
        assertEquals(2, full.maxPl);

        // the lists append, as llama.cpp's repeatable flags do
        BatchedBenchCommand.Plan appended = BatchedBenchCommand.parse(new String[] {"m.gguf",
                "-npp", "4", "-npp", "8", "-ntg", "3", "-npl", "1"});
        assertEquals(2, appended.npp.size());
        assertEquals(8, (int) appended.npp.get(1));
    }

    @Test
    public void theLoadFlagsReachTheEngineConfigOverThePagedMemory() throws Exception {
        BatchedBenchCommand.Plan defaults = BatchedBenchCommand.parse(new String[] {"m.gguf",
                "-npp", "4", "-ntg", "3", "-npl", "1,4"});
        EngineConfig auto = defaults.engineConfig();
        assertEquals("LLMDemo's default: the best GPU backend", EngineConfig.BackendMode.AUTO, auto.backend());
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, auto.gpuLayers());
        assertTrue("the bench's memory is paged", auto.memory().kvPaged());
        assertEquals("one sequence per parallel slot of the largest -npl", 4, auto.memory().parallel());

        BatchedBenchCommand.Plan p = BatchedBenchCommand.parse(new String[] {"m.gguf", "--backend", "opencl",
                "-ngl", "all", "--device", "RTX", "-ot", "ffn_down=cpu", "--cpu-moe", "-fa", "on", "-ctk", "q8_0",
                "-ctv", "q4_0", "-c", "128", "-ub", "16", "-npp", "4", "-ntg", "3", "-npl", "2"});
        EngineConfig cfg = p.engineConfig();
        assertEquals(EngineConfig.BackendMode.OPENCL, cfg.backend());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, cfg.gpuLayers());
        assertEquals("RTX", cfg.deviceFilter());
        assertEquals(1, cfg.tensorOverrides().size());
        assertTrue(cfg.cpuMoe());
        MemoryConfig m = cfg.memory();
        assertEquals(KvDType.Q8_0, m.typeK());
        assertEquals(KvDType.Q4_0, m.typeV());
        assertEquals(128, m.nCtx());
        assertEquals(16, m.ubatch());
        assertEquals(2, m.parallel());
        assertTrue(m.kvPaged());
        assertEquals("the bench's own flags still parse", 128, p.ctx);

        assertEquals(12, BatchedBenchCommand.parse(new String[] {"m.gguf", "-ngl", "12",
                "-npp", "4", "-ntg", "3", "-npl", "1"}).engineConfig().gpuLayers());
        assertEquals(EngineConfig.BackendMode.CPU, BatchedBenchCommand.parse(new String[] {"m.gguf", "--cpu",
                "-npp", "4", "-ntg", "3", "-npl", "1"}).engineConfig().backend());
        System.setProperty(EngineConfig.PROP_BACKEND, "cpu");
        assertEquals("neuralfree.backend is the base, as for LLMDemo", EngineConfig.BackendMode.CPU,
                BatchedBenchCommand.parse(new String[] {"m.gguf", "-npp", "4", "-ntg", "3", "-npl", "1"})
                        .engineConfig().backend());
    }

    @Test
    public void aSmallGridRunsOnTheGpuPagedMemory() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path model = model();
        BatchedBenchCommand.Plan p = BatchedBenchCommand.parse(new String[] {model.toString(),
                "--backend", "opencl", "-ngl", "all", "-fa", "auto", "-c", "64", "-ub", "8",
                "-npp", "4,8", "-ntg", "3", "-npl", "1,2"});
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> measured = BatchedBenchCommand.execute(p, out, err);
        String table = out.toString();
        assertTrue("the model opened on the GPU backend: " + table + err,
                table.contains("parallel = 2, is_pp_shared = false, is_tg_separate = false, backend = opencl\n"));
        assertTrue("the run skips nothing: " + err, err.length() == 0);
        long[] nKv = {7, 14, 11, 22};
        assertEquals(nKv.length, measured.size());
        for (int i = 0; i < nKv.length; i++) {
            assertEquals("n_kv = pl*(pp+tg)", nKv[i], measured.get(i).nKv);
            assertRowSpeeds(measured.get(i), false);
        }

        StringBuilder shared = new StringBuilder();
        List<BatchedBenchCommand.BenchRow> pps = BatchedBenchCommand.execute(BatchedBenchCommand.parse(
                new String[] {model.toString(), "--backend", "opencl", "-c", "64", "-ub", "8", "-pps",
                        "-npp", "8", "-ntg", "3", "-npl", "2"}), shared, err);
        assertTrue(shared.toString(), shared.toString().contains("backend = opencl"));
        assertEquals("the device page copy shares the prompt: pp + pl*tg rows", 14, pps.get(0).nKv);
        assertRowSpeeds(pps.get(0), true);
    }

    @Test
    public void parseRefusesBadInputByName() throws Exception {
        assertUsage(new String[] {}, "needs a model path");
        assertUsage(new String[] {"m.gguf"}, "-npp");
        assertUsage(new String[] {"m.gguf", "-npp", "4"}, "-ntg");
        assertUsage(new String[] {"m.gguf", "-npp", "4", "-ntg", "3"}, "-npl");
        assertUsage(new String[] {"m.gguf", "--frobozz"}, "--frobozz");
        assertUsage(new String[] {"m.gguf", "-npp", "0", "-ntg", "3", "-npl", "1"}, "positive");
        assertUsage(new String[] {"m.gguf", "-npp", "x", "-ntg", "3", "-npl", "1"}, "integers");
        assertUsage(new String[] {"m.gguf", "--output-format", "yaml", "-npp", "4", "-ntg", "3",
                "-npl", "1"}, "md or jsonl");
        assertUsage(new String[] {"m.gguf", "-npp"}, "needs a value");
        assertUsage(new String[] {"m.gguf", "other.gguf", "-npp", "4", "-ntg", "3", "-npl", "1"},
                "too many file arguments");
        assertUsage(new String[] {"m.gguf", "-fa", "off", "-npp", "4", "-ntg", "3", "-npl", "1"},
                "-fa off is not supported");
        assertUsage(new String[] {"m.gguf", "-ngl", "some", "-npp", "4", "-ntg", "3", "-npl", "1"}, "-ngl");
        assertUsage(new String[] {"m.gguf", "-ctk", "q6_k", "-npp", "4", "-ntg", "3", "-npl", "1"}, "-ctk");
        assertUsage(new String[] {"m.gguf", "--backend", "tpu", "-npp", "4", "-ntg", "3", "-npl", "1"},
                "--backend");
    }

    private static void assertUsage(String[] args, String messagePart) {
        BatchedBenchCommand.UsageException e = null;
        try {
            BatchedBenchCommand.parse(args);
        } catch (BatchedBenchCommand.UsageException ex) {
            e = ex;
        }
        assertTrue(messagePart, e != null && e.getMessage().contains(messagePart));
    }
}

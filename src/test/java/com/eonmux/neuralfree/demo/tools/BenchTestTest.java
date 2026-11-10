package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * One finished {@code nftool bench} measurement: llama.cpp's {@code test} statistics — the arithmetic mean,
 * the sample standard deviation {@code sqrt(sq_sum/(n-1) - mean^2 n/(n-1))}, the per-sample
 * {@code 1e9 n_tokens / t} throughputs — the field list and its types, and the value row the flat formats
 * print.
 */
public class BenchTestTest {

    private static BenchInstance inst() {
        return new BenchInstance("m.gguf", 64, 32, 8, 2048, 128, KvDType.Q8_0, KvDType.F16, 0);
    }

    @Test
    public void theStatisticsFollowTheReferenceFormulas() {
        BenchTest t = new BenchTest(inst(), "llama tiny", 1024, 128, "CPU", 104, BenchReadout.LOGITS,
                new long[]{1000, 3000});
        assertEquals(2000, t.avgNs());
        // sqrt((1e6 + 9e6)/1 - 2000^2 * 2/1) = sqrt(2e6).
        assertEquals(1414, t.stdevNs());
        // Throughputs 1e9*96/1000 = 9.6e7 and 1e9*96/3000 = 3.2e7; mean 6.4e7.
        assertEquals(6.4e7, t.avgTs(), 1e-6);
        double[] ts = t.getTs();
        assertEquals(9.6e7, ts[0], 1e-6);
        assertEquals(3.2e7, ts[1], 1e-6);
        assertEquals(Math.sqrt((9.6e7 * 9.6e7 + 3.2e7 * 3.2e7) - 6.4e7 * 6.4e7 * 2), t.stdevTs(), 1e3);
    }

    @Test
    public void oneSampleHasNoDeviation() {
        BenchTest t = new BenchTest(inst(), "llama tiny", 1024, 128, "CPU", 104, BenchReadout.LOGITS,
                new long[]{500});
        assertEquals(500, t.avgNs());
        assertEquals(0, t.stdevNs());
        assertEquals(1.92e8, t.avgTs(), 1e-6);
        assertEquals(0.0, t.stdevTs(), 0.0);
    }

    @Test
    public void theFieldsKeepTheReferenceOrderAndTypes() {
        String[] fields = BenchTest.fields();
        assertEquals("model_filename", fields[0]);
        assertEquals("model_type", fields[1]);
        assertEquals("model_size", fields[2]);
        assertEquals("model_n_params", fields[3]);
        assertEquals("backends", fields[4]);
        assertEquals("n_batch", fields[5]);
        assertEquals("n_ubatch", fields[6]);
        assertEquals("n_threads", fields[7]);
        assertEquals("n_ctx", fields[8]);
        assertEquals("type_k", fields[9]);
        assertEquals("type_v", fields[10]);
        assertEquals("n_gpu_layers", fields[11]);
        assertEquals("n_cpu_moe", fields[12]);
        assertEquals("split_mode", fields[13]);
        assertEquals("main_gpu", fields[14]);
        assertEquals("no_kv_offload", fields[15]);
        assertEquals("devices", fields[16]);
        assertEquals("tensor_split", fields[17]);
        assertEquals("tensor_buft_overrides", fields[18]);
        assertEquals("load_mode", fields[19]);
        assertEquals("embeddings", fields[20]);
        assertEquals("tg_readout", fields[21]);
        assertEquals("n_prompt", fields[22]);
        assertEquals("n_gen", fields[23]);
        assertEquals("n_depth", fields[24]);
        assertEquals("test_time", fields[25]);
        assertEquals("avg_ns", fields[26]);
        assertEquals("stddev_ns", fields[27]);
        assertEquals("avg_ts", fields[28]);
        assertEquals("stddev_ts", fields[29]);
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("model_type"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("type_k"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("split_mode"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("devices"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("tensor_split"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("tensor_buft_overrides"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("load_mode"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("tg_readout"));
        assertEquals(BenchTest.FieldType.STRING, BenchTest.fieldType("test_time"));
        assertEquals(BenchTest.FieldType.BOOL, BenchTest.fieldType("no_kv_offload"));
        assertEquals(BenchTest.FieldType.BOOL, BenchTest.fieldType("embeddings"));
        assertEquals(BenchTest.FieldType.INT, BenchTest.fieldType("main_gpu"));
        assertEquals(BenchTest.FieldType.INT, BenchTest.fieldType("n_cpu_moe"));
        assertEquals(BenchTest.FieldType.INT, BenchTest.fieldType("n_threads"));
        assertEquals(BenchTest.FieldType.INT, BenchTest.fieldType("avg_ns"));
        assertEquals(BenchTest.FieldType.INT, BenchTest.fieldType("n_ctx"));
        assertEquals(BenchTest.FieldType.FLOAT, BenchTest.fieldType("avg_ts"));
        assertEquals(BenchTest.FieldType.FLOAT, BenchTest.fieldType("stddev_ts"));
    }

    @Test
    public void theValueRowCarriesTheRunAndItsStatistics() {
        BenchTest t = new BenchTest(inst(), "llama tiny", 1024, 128, "CPU", 104, BenchReadout.LOGITS,
                new long[]{1_000_000_000L, 3_000_000_000L});
        String[] v = t.values();
        assertEquals(BenchTest.fields().length, v.length);
        assertEquals("m.gguf", v[0]);
        assertEquals("llama tiny", v[1]);
        assertEquals("1024", v[2]);
        assertEquals("128", v[3]);
        assertEquals("CPU", v[4]);
        assertEquals("2048", v[5]);
        assertEquals("128", v[6]);
        assertEquals(Integer.toString(inst().nThreads), v[7]);
        assertEquals("104", v[8]);
        assertEquals("q8_0", v[9]);
        assertEquals("f16", v[10]);
        assertEquals("0", v[11]);
        assertEquals("0", v[12]);
        assertEquals("none", v[13]);
        assertEquals("0", v[14]);
        assertEquals("0", v[15]);
        assertEquals("auto", v[16]);
        assertEquals("0.00", v[17]);
        assertEquals("none", v[18]);
        assertEquals("auto", v[19]);
        assertEquals("0", v[20]);
        assertEquals("logits", v[21]);
        assertEquals("64", v[22]);
        assertEquals("32", v[23]);
        assertEquals("8", v[24]);
        assertTrue("the test time is RFC 3339 UTC: " + v[25], v[25].matches("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z"));
        // samples 1e9 and 3e9: mean 2e9, stdev sqrt(2e18) rounded; the throughputs 96 and 32 average to 64.
        assertEquals("2000000000", v[26]);
        assertEquals("1414213562", v[27]);
        assertEquals("64.000000", v[28]);
        assertTrue("avg and stddev t/s share the flat double format: " + v[29], v[29].matches("\\d+\\.\\d{6}"));
        assertEquals(v.length, t.map().size());
        assertEquals("64", t.map().get("n_prompt"));
    }

    @Test
    public void theValueRowCarriesTheSweptSplitAndKvOffloadSettings() {
        BenchInstance inst = new BenchInstance("m.gguf", 64, 0, 0, 2048, 128, KvDType.F16, KvDType.F16, -1,
                true, com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, 1,
                new double[]{3.0, 1.0});
        assertEquals("layer", inst.splitMode.llamaName());
        assertEquals("3.00/1.00", inst.tensorSplitString());
        BenchTest t = new BenchTest(inst, "llama tiny", 1024, 128, "CPU", 64, BenchReadout.LOGITS,
                new long[]{1000});
        assertEquals("layer", t.values()[13]);
        assertEquals("1", t.values()[14]);
        assertEquals("1", t.values()[15]);
        assertEquals("auto", t.values()[16]);
        assertEquals("3.00/1.00", t.values()[17]);
        assertEquals("3.00/1.00", t.map().get("tensor_split"));
    }

    @Test
    public void theRowRecordsTheReadoutItsGeneratedTokensUsed() throws Exception {
        BenchReadout argmax = BenchReadout.parse("argmax", "--tg-readout");
        BenchReadout topK = BenchReadout.parse("topk=40", "--tg-readout");
        assertEquals("argmax", new BenchTest(inst(), "llama tiny", 1024, 128, "CPU", 104, argmax,
                new long[]{1000}).map().get("tg_readout"));
        assertEquals("topk=40", new BenchTest(inst(), "llama tiny", 1024, 128, "CPU", 104, topK,
                new long[]{1000}).map().get("tg_readout"));
        BenchInstance promptOnly = new BenchInstance("m.gguf", 64, 0, 0, 2048, 128, KvDType.F16, KvDType.F16, 0);
        assertEquals("a test without generated tokens reads the prompt pass's logits", "logits",
                new BenchTest(promptOnly, "llama tiny", 1024, 128, "CPU", 104, argmax, new long[]{1000}).map()
                        .get("tg_readout"));
        BenchInstance embedding = new BenchInstance("m.gguf", 64, 32, 0, 2048, 128, KvDType.F16, KvDType.F16, 0,
                false, com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.NONE, 0, new double[0],
                BenchGrid.defaultThreads(), com.eonmux.neuralfree.gguf.engine.EngineConfig.N_CPU_LAYERS_UNSET, "",
                BenchTensorOverrides.NONE, BenchLoadMode.AUTO, true);
        assertEquals("the embedding mode's decodes read the logits row", "logits",
                new BenchTest(embedding, "llama tiny", 1024, 128, "CPU", 104, topK, new long[]{1000}).map()
                        .get("tg_readout"));
    }

    @Test
    public void equalTensorSharesPrintTheReferenceDefault() {
        BenchInstance inst = new BenchInstance("m.gguf", 64, 0, 0, 2048, 128, KvDType.F16, KvDType.F16, -1);
        assertEquals("0.00", inst.tensorSplitString());
    }

    @Test
    public void theInstanceAppliesItsFlagsToTheEngineConfiguration() throws Exception {
        com.eonmux.neuralfree.gguf.engine.EngineConfig base =
                com.eonmux.neuralfree.gguf.engine.EngineConfig.builder().build();
        com.eonmux.neuralfree.gguf.memory.MemoryConfig mc =
                com.eonmux.neuralfree.gguf.memory.MemoryConfig.builder().nCtx(64).ubatch(128).build();
        BenchInstance gpu = new BenchInstance("m.gguf", 64, 0, 0, 2048, 128, KvDType.F16, KvDType.F16, -1,
                true, com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, 1,
                new double[]{3.0, 1.0});
        com.eonmux.neuralfree.gguf.engine.EngineConfig cfg = gpu.engineConfig(base, mc);
        assertTrue("--no-kv-offload reaches the engine", !cfg.kvOffload());
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertEquals(1, cfg.mainGpu());
        assertEquals(2, cfg.tensorSplit().length);
        assertEquals(3.0, cfg.tensorSplit()[0], 0.0);
        assertEquals(1.0, cfg.tensorSplit()[1], 0.0);
        assertEquals("the GPU engine is the backend when ngl is not 0",
                com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.AUTO, cfg.backend());

        BenchInstance cpu = new BenchInstance("m.gguf", 64, 0, 0, 2048, 128, KvDType.F16, KvDType.F16, 0);
        assertEquals("ngl 0 selects the CPU engine",
                com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.CPU, cpu.engineConfig(base, mc).backend());
    }

    @Test
    public void theTestNamesFollowTheReference() {
        assertEquals("pp512", new BenchInstance("m", 512, 0, 0, 2048, 512, KvDType.F16, KvDType.F16, 0).testName());
        assertEquals("tg128", new BenchInstance("m", 0, 128, 0, 2048, 512, KvDType.F16, KvDType.F16, 0).testName());
        assertEquals("pp64+tg32", new BenchInstance("m", 64, 32, 0, 2048, 512, KvDType.F16, KvDType.F16, 0).testName());
        assertEquals("pp64+tg32 @ d8",
                new BenchInstance("m", 64, 32, 8, 2048, 512, KvDType.F16, KvDType.F16, 0).testName());
        assertEquals("tg16 @ d4",
                new BenchInstance("m", 0, 16, 4, 2048, 512, KvDType.F16, KvDType.F16, 0).testName());
    }
}

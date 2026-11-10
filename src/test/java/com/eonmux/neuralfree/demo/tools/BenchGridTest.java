package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.cpu.numa.CpuThreadPolicy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The sweep of {@code nftool bench}: the value ranges llama.cpp's {@code parse_int_range} accepts, the
 * cross-product order of {@code get_cmd_params_instances} (engine settings outermost, then the depth, then
 * the pp, tg and pg shapes) and the refusals the reference's parse raises for the settings this engine does
 * not carry.
 */
public class BenchGridTest {

    @Test
    public void theRangeParseWalksListsScalesAndMultiples() throws Exception {
        assertEquals(List.of(512), BenchGrid.parseIntRange("512", "-p"));
        assertEquals(List.of(0, 128, 256), BenchGrid.parseIntRange("0,128,256", "-p"));
        assertEquals(List.of(64, 128, 192, 256), BenchGrid.parseIntRange("64-256+64", "-p"));
        assertEquals(List.of(8, 16, 32, 64), BenchGrid.parseIntRange("8-64*2", "-p"));
        assertEquals(List.of(4, 5, 6, 7), BenchGrid.parseIntRange("4-7", "-d"));
        assertEquals(List.of(0), BenchGrid.parseIntRange("0", "-n"));
    }

    @Test
    public void theRangeParseRefusesTheShapesTheReferenceRefuses() {
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("8-", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("-8", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("8..16", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("16-8", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("8-16+0", "-p"));
        assertThrows(BenchCommand.UsageException.class, () -> BenchGrid.parseIntRange("8-16-2", "-p"));
    }

    @Test
    public void theDefaultSweepIsOnePromptTestOneGenerationTest() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf");
        List<BenchInstance> runs = grid.instances();
        assertEquals(2, runs.size());
        assertEquals(512, runs.get(0).nPrompt);
        assertEquals(0, runs.get(0).nGen);
        assertEquals(2048, runs.get(0).nBatch);
        assertEquals(512, runs.get(0).nUbatch);
        assertEquals(KvDType.F16, runs.get(0).typeK);
        assertEquals(-1, runs.get(0).ngl);
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.NONE, runs.get(0).splitMode);
        assertEquals(0, runs.get(0).mainGpu);
        assertTrue("the KV cache stays on the device by default", !runs.get(0).noKvOffload);
        assertEquals("0.00", runs.get(0).tensorSplitString());
        assertEquals(0, runs.get(1).nPrompt);
        assertEquals(128, runs.get(1).nGen);
        assertEquals("pp512", runs.get(0).testName());
        assertEquals("tg128", runs.get(1).testName());
    }

    @Test
    public void theCrossProductExpandsInTheReferenceOrder() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-p", "64,128", "-n", "32", "-d", "8", "-ub", "64,128",
                "-ctv", "f32", "-ngl", "0");
        List<BenchInstance> runs = grid.instances();
        // The ubatch crosses every shape; the order is ubatch outermost, then pp, tg.
        assertEquals(6, runs.size());
        assertEquals(64, runs.get(0).nPrompt);
        assertEquals(8, runs.get(0).nDepth);
        assertEquals(64, runs.get(0).nUbatch);
        assertEquals("pp64 @ d8", runs.get(0).testName());
        assertEquals(128, runs.get(1).nPrompt);
        assertEquals(64, runs.get(1).nUbatch);
        assertEquals(0, runs.get(2).nPrompt);
        assertEquals(32, runs.get(2).nGen);
        assertEquals("tg32 @ d8", runs.get(2).testName());
        assertEquals(128, runs.get(3).nUbatch);
        assertEquals(64, runs.get(3).nPrompt);
        assertEquals(KvDType.F32, runs.get(5).typeV);
        assertEquals(KvDType.F16, runs.get(5).typeK);
        assertEquals(0, runs.get(5).ngl);
    }

    @Test
    public void thePairShapesLandAfterTheSingles() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-n", "16", "-pg", "64,32", "-pg", "0,8", "-p", "128");
        List<BenchInstance> runs = grid.instances();
        // A pair with a non-zero side is a real test: 0,8 runs as a plain generation shape.
        assertEquals(4, runs.size());
        assertEquals("pp128", runs.get(0).testName());
        assertEquals("tg16", runs.get(1).testName());
        assertEquals("pp64+tg32", runs.get(2).testName());
        assertEquals("tg8", runs.get(3).testName());
        // A 0,0 pair is no test, like the reference skips it; the defaults still fill the sweep.
        BenchGrid zeroPair = parse("-m", "m.gguf", "-pg", "0,0");
        assertEquals(2, zeroPair.instances().size());
        // All-zero shapes give nothing to run; the sweep refuses instead of printing empty tables.
        BenchGrid allZero = parse("-m", "m.gguf", "-p", "0", "-n", "0");
        assertThrows(BenchCommand.UsageException.class, allZero::instances);
    }

    @Test
    public void theSplitAndKvOffloadFlagsExpandInTheReferenceOrder() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-p", "0", "-n", "8", "-sm", "layer", "-mg", "0,1",
                "-ts", "1/1,3/1", "-nkvo", "0,1");
        List<BenchInstance> runs = grid.instances();
        // main_gpu outermost, then tensor_split, then no_kv_offload, as get_cmd_params_instances orders them.
        assertEquals(8, runs.size());
        com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode layer =
                com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER;
        assertEquals(layer, runs.get(0).splitMode);
        assertEquals(0, runs.get(0).mainGpu);
        assertEquals("1.00/1.00", runs.get(0).tensorSplitString());
        assertTrue(!runs.get(0).noKvOffload);
        assertEquals("1.00/1.00", runs.get(1).tensorSplitString());
        assertTrue("the second run keeps the KV caches in host RAM", runs.get(1).noKvOffload);
        assertEquals("3.00/1.00", runs.get(2).tensorSplitString());
        assertEquals(1, runs.get(4).mainGpu);
        assertEquals("3.00/1.00", runs.get(7).tensorSplitString());
        assertTrue(runs.get(7).noKvOffload);
    }

    @Test
    public void aTensorSplitAloneRunsALayerSplit() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-ts", "3;1");
        List<BenchInstance> runs = grid.instances();
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, runs.get(0).splitMode);
        assertEquals("3.00/1.00", runs.get(0).tensorSplitString());
    }

    @Test
    public void theSplitAndKvOffloadRefusalsNameTheirFlag() throws Exception {
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.ROW,
                parse("-m", "m.gguf", "-sm", "row").instances().get(0).splitMode);
        assertEquals(com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.TENSOR,
                parse("-m", "m.gguf", "-sm", "tensor").instances().get(0).splitMode);
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-sm", "bogus"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-ts", "0/1"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-ts", "abc"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-ts", "1/0/1"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-nkvo", "2"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-mg", "-1"));
        BenchCommand.UsageException none = assertThrows(BenchCommand.UsageException.class,
                () -> parse("-m", "m.gguf", "-sm", "none", "-ts", "1/1").instances());
        assertTrue("the tensor split refusal names its flag: " + none.getMessage(),
                none.getMessage().contains("-ts") || none.getMessage().contains("--tensor-split"));
        BenchCommand.UsageException cpu = assertThrows(BenchCommand.UsageException.class,
                () -> parse("-m", "m.gguf", "-ngl", "0", "-sm", "layer").instances());
        assertTrue("the CPU layer-split refusal names the setting: " + cpu.getMessage(),
                cpu.getMessage().contains("split"));
    }

    @Test
    public void theRefusalsNameTheirFlag() throws Exception {
        assertThrows(BenchCommand.UsageException.class, () -> parse().instances());
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-pg", "64"));
        assertThrows(BenchCommand.UsageException.class, () -> parse("-m", "m.gguf", "-pg", "64,32,8"));
        BenchGrid smallBatch = parse("-m", "m.gguf", "-b", "64", "-ub", "512");
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class, smallBatch::instances);
        assertTrue("the refusal names both flags: " + e.getMessage(),
                e.getMessage().contains("-b/--batch-size") && e.getMessage().contains("-ub/--ubatch-size"));
    }

    @Test
    public void theRunsGroupByTheirReloadKey() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-p", "64,128", "-ub", "64");
        assertEquals(1, grid.groups(grid.instances()).size());
        BenchGrid two = parse("-m", "m.gguf", "-p", "64", "-ub", "64,128", "-n", "32");
        assertEquals(2, two.groups(two.instances()).size());
        BenchGrid twoModels = parse("-m", "a.gguf,b.gguf", "-n", "8");
        assertEquals(2, twoModels.groups(twoModels.instances()).size());
        BenchGrid twoKv = parse("-m", "m.gguf", "-n", "8", "-nkvo", "0,1");
        assertEquals(2, twoKv.groups(twoKv.instances()).size());
        BenchGrid twoGpus = parse("-m", "m.gguf", "-n", "8", "-sm", "none", "-mg", "0,1");
        assertEquals("the main GPU changes what a load reads", 2, twoGpus.groups(twoGpus.instances()).size());
        BenchGrid twoSplits = parse("-m", "m.gguf", "-n", "8", "-ts", "1/1,3/1");
        assertEquals(2, twoSplits.groups(twoSplits.instances()).size());
    }

    @Test
    public void theDevicePlacementLoadModeAndEmbeddingFlagsParseIntoTheGrid() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-dev", "CUDA0/CUDA1,none,auto", "-ncmoe", "0,2",
                "-ot", "a=CPU;b=Vulkan1,c=OpenCL2", "-lm", "none,dio", "-embd", "0,1");
        assertEquals(3, grid.devices.size());
        assertEquals("CUDA0,CUDA1", grid.devices.get(0));
        assertEquals("none", grid.devices.get(1));
        assertEquals("", grid.devices.get(2));
        assertEquals(2, grid.nCpuMoe.size());
        assertEquals(2, grid.tensorOverrides.size());
        assertEquals("a=CPU;b=Vulkan1", grid.tensorOverrides.get(0).text());
        assertEquals("c=OpenCL2", grid.tensorOverrides.get(1).text());
        assertEquals("cpu", grid.tensorOverrides.get(0).overrides().get(0).target());
        assertEquals("vulkan", grid.tensorOverrides.get(0).overrides().get(1).target());
        assertEquals("opencl", grid.tensorOverrides.get(1).overrides().get(0).target());
        assertEquals(2, grid.loadModes.size());
        assertEquals(BenchLoadMode.NONE, grid.loadModes.get(0));
        assertEquals(BenchLoadMode.DIRECT_IO, grid.loadModes.get(1));
        assertEquals(2, grid.embeddings.size());
        assertTrue("the new sweeps show their columns", grid.devicesGiven && grid.overridesGiven
                && grid.loadModeGiven);
    }

    @Test
    public void theNewLoadFlagsReachTheEngineConfiguration() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-p", "0", "-n", "8", "-dev", "CUDA0/CUDA1,none",
                "-ncmoe", "2", "-ot", "blk\\.0=CPU", "-lm", "none");
        List<BenchInstance> runs = grid.instances();
        assertEquals("one run per device member", 2, runs.size());
        BenchInstance gpu = runs.get(0);
        assertEquals("CUDA0/CUDA1", gpu.devicesString());
        assertEquals("a two-device list splits by layer, as llama.cpp's default mode does",
                com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode.LAYER, gpu.splitMode);
        com.eonmux.neuralfree.gguf.engine.EngineConfig cfg = gpu.engineConfig(
                com.eonmux.neuralfree.gguf.engine.EngineConfig.builder().build(),
                com.eonmux.neuralfree.gguf.memory.MemoryConfig.builder().nCtx(64).ubatch(16).build());
        assertEquals(2, cfg.nCpuMoe());
        assertEquals(1, cfg.tensorOverrides().size());
        assertEquals("blk\\.0", cfg.tensorOverrides().get(0).pattern());
        assertTrue("CPU keeps the matched tensors on the host", cfg.tensorOverrides().get(0).host());
        assertEquals(2, cfg.gpuDevices().size());
        BenchInstance cpu = runs.get(1);
        assertEquals("none", cpu.devicesString());
        com.eonmux.neuralfree.gguf.engine.EngineConfig cpuCfg = cpu.engineConfig(
                com.eonmux.neuralfree.gguf.engine.EngineConfig.builder().build(),
                com.eonmux.neuralfree.gguf.memory.MemoryConfig.builder().nCtx(64).ubatch(16).build());
        assertEquals("dev none runs the CPU engine", com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode.CPU,
                cpuCfg.backend());
    }

    @Test
    public void theNewLoadSettingsSplitTheModelLoadGroups() throws Exception {
        BenchGrid expert = parse("-m", "m.gguf", "-n", "8", "-ncmoe", "0,2");
        assertEquals(2, expert.groups(expert.instances()).size());
        BenchGrid devices = parse("-m", "m.gguf", "-n", "8", "-dev", "CUDA0,none");
        assertEquals(2, devices.groups(devices.instances()).size());
        BenchGrid overrides = parse("-m", "m.gguf", "-n", "8", "-ot", "a=CPU,");
        assertEquals(2, overrides.groups(overrides.instances()).size());
        BenchGrid loads = parse("-m", "m.gguf", "-n", "8", "-lm", "none,dio");
        assertEquals(2, loads.groups(loads.instances()).size());
        BenchGrid embd = parse("-m", "m.gguf", "-n", "8", "-embd", "0,1");
        assertEquals("the embedding pass needs its own paged load", 2, embd.groups(embd.instances()).size());
    }

    @Test
    public void theEmbeddingModeRefusesADepthPrefill() throws Exception {
        BenchGrid grid = parse("-m", "m.gguf", "-n", "0", "-p", "8", "-d", "4", "-embd", "1");
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class, grid::instances);
        assertTrue("the refusal names both flags: " + e.getMessage(),
                e.getMessage().contains("-d") && e.getMessage().contains("-embd"));
    }

    @Test
    public void theNewRefusalsNameTheirFlagAndReason() throws Exception {
        assertRefused("nopo", "-m", "m.gguf", "-nopo", "1");
        assertRefused("fitt", "-m", "m.gguf", "-fitt", "1024");
        assertRefused("fitc", "-m", "m.gguf", "-fitc", "512");
        assertEquals("mlock is an accepted load mode", 1,
                parse("-m", "m.gguf", "-lm", "mlock").loadModes.size());
        assertEquals("mmap+mlock is an accepted load mode", 1,
                parse("-m", "m.gguf", "-lm", "mmap+mlock").loadModes.size());
        assertRefused("CUDA_Host", "-m", "m.gguf", "-ot", "a=CUDA_Host");
        assertRefused("one thread count", "-m", "m.gguf", "-t", "4,8");
        assertRefused("one thread count", "-m", "m.gguf", "-t", "4-8");
    }

    @Test
    public void theHuggingFaceFlagsParseAndPair() throws Exception {
        BenchGrid grid = parse("-hf", "org/model:Q4_K_M", "-hff", "model.gguf");
        assertEquals(1, grid.hfRepos.size());
        assertEquals("org/model:Q4_K_M", grid.hfRepos.get(0));
        assertEquals(1, grid.hfFiles.size());
        assertEquals("model.gguf", grid.hfFiles.get(0));
        BenchGrid unpaired = parse("-hff", "model.gguf");
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class, unpaired::instances);
        assertTrue("the pairing refusal names the flags: " + e.getMessage(),
                e.getMessage().contains("-hff") && e.getMessage().contains("-hf"));
    }

    private static void assertRefused(String reason, String... args) throws Exception {
        BenchCommand.UsageException e = assertThrows(BenchCommand.UsageException.class,
                () -> BenchCommand.parse(args));
        assertTrue("the refusal names '" + reason + "': " + e.getMessage(), e.getMessage().contains(reason));
    }

    /** Parses flags the way {@code BenchCommand} does, so the tests exercise the same path. */
    private static BenchGrid parse(String... args) throws BenchCommand.UsageException {
        return BenchCommand.parse(args);
    }

    @Test
    public void theDefaultThreadCountFollowsThePoolAndFallsBackToTheDefaultPolicy() {
        String threads = System.getProperty(CpuParallel.THREADS_PROPERTY);
        String policy = System.getProperty(CpuThreadPolicy.PROPERTY);
        try {
            System.setProperty(CpuParallel.THREADS_PROPERTY, "3");
            assertEquals(3, BenchGrid.defaultThreads());
            System.clearProperty(CpuParallel.THREADS_PROPERTY);
            System.setProperty(CpuThreadPolicy.PROPERTY, "logical");
            assertEquals(CpuThreadPolicy.host().threads(CpuThreadPolicy.Policy.LOGICAL), BenchGrid.defaultThreads());
            int fallback = CpuThreadPolicy.host().threads(CpuThreadPolicy.DEFAULT_POLICY);
            System.setProperty(CpuThreadPolicy.PROPERTY, "fastest");
            assertEquals("an invalid policy reports the default policy's count", fallback, BenchGrid.defaultThreads());
            System.clearProperty(CpuThreadPolicy.PROPERTY);
            System.setProperty(CpuParallel.THREADS_PROPERTY, "many");
            assertEquals("an invalid count reports the default policy's count", fallback, BenchGrid.defaultThreads());
        } finally {
            restore(CpuParallel.THREADS_PROPERTY, threads);
            restore(CpuThreadPolicy.PROPERTY, policy);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}

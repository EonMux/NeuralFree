package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineLoadFlags}: the load flags LLMDemo and the batched tools share consume exactly their own arguments,
 * start from the {@code -Dneuralfree.*} properties, take llama.cpp's {@code -fa} spellings (refusing {@code off} by
 * name), and {@link EngineLoadFlags#pagedConfig} lays a tool's paged memory over the load settings.
 */
public class EngineLoadFlagsTest {

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

    private static EngineConfig parse(String... args) throws EngineLoadFlags.FlagException {
        EngineLoadFlags load = EngineLoadFlags.fromSystemProperties();
        for (int i = 0; i < args.length; i++) {
            int last = load.parse(args, i);
            assertTrue("a load flag: " + args[i], last >= i);
            i = last;
        }
        return load.build();
    }

    @Test
    public void rpcAddsRemoteDevicesToALayerSplit() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--device", "0", "--rpc", "box:50052@k");
        assertEquals(EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertEquals(1, cfg.rpcServers().size());
        assertEquals("box", cfg.rpcServers().get(0).host());
        assertEquals(50052, cfg.rpcServers().get(0).port());

        assertThrows(EngineLoadFlags.FlagException.class, () -> parse("--cpu", "--rpc", "box:50052@k"));
        assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--device", "none", "--rpc", "box:50052@k"));
        assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--rpc", "box:50052@k", "--device", "none"));
        assertThrows(EngineLoadFlags.FlagException.class, () -> parse("--rpc", "box:50052@k,"));
    }

    @Test
    public void theOffloadFlagsParseIntoTheConfiguration() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--cpu-offload-mb", "256", "--cpu-offload-params",
                "experts,ffn_gate_exps", "--stream-params", "ffn_up_exps");

        assertEquals(256L, cfg.cpuOffloadMb());
        assertEquals(256L * 1024 * 1024, cfg.cpuOffloadBytes());
        assertEquals(Arrays.asList("experts", "ffn_gate_exps"), cfg.cpuOffloadParams());
        assertEquals(Collections.singletonList("ffn_up_exps"), cfg.streamParams());
    }

    @Test
    public void anEmptyOffloadParameterSegmentIsRefusedByTheFlag() {
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--cpu-offload-mb", "8", "--stream-params", "experts,,ffn_gate_exps"));

        assertTrue(e.getMessage(), e.getMessage().contains("--stream-params"));
    }

    @Test
    public void thePoolFlagParsesBesideTheExpertCacheAndTheDeviceFlags() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--device", "0,1", "--moe-cache", "on", "--moe-cache-pool");
        assertTrue("two devices select the layer split the pool needs", cfg.split());
        assertTrue(cfg.moeCache().enabled());
        assertTrue(cfg.moeCachePool());
    }

    @Test
    public void theEplbFlagsParseIntoTheConfiguration() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                "--eplb-profile", "/tmp/load.bin", "--eplb-save", "/tmp/save.bin", "--eplb-policy", "default");

        assertTrue(cfg.expertParallel());
        assertTrue(cfg.eplb().enabled());
        assertTrue("the flag enables the runtime rearrangement, as vLLM's EPLB does", cfg.eplb().rebalance());
        assertEquals("/tmp/load.bin", cfg.eplb().profile().toString());
        assertEquals("/tmp/save.bin", cfg.eplb().save().toString());
        assertEquals(EplbConfig.Policy.DEFAULT, cfg.eplb().policy());
    }

    @Test
    public void enableEplbAloneImpliesRuntimeRebalancing() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb");

        assertTrue(cfg.eplb().enabled());
        assertTrue(cfg.eplb().rebalance());
        assertNull(cfg.eplb().profile());
        assertNull(cfg.eplb().save());
    }

    @Test
    public void theEplbConfigJsonParsesIntoTheConfiguration() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                "--eplb-config", "{\"num_redundant_experts\": 32, \"window_size\": 1000, \"step_interval\": 3000}");

        assertTrue(cfg.eplb().enabled());
        assertTrue(cfg.eplb().rebalance());
        assertEquals(32, cfg.eplb().numRedundant());
        assertEquals(1000, cfg.eplb().windowSize());
        assertEquals(3000, cfg.eplb().stepInterval());
    }

    @Test
    public void theEplbConfigJsonPolicyYieldsToALaterPolicyFlag() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                "--eplb-config", "{\"policy\": \"hierarchical\"}");
        assertEquals(EplbConfig.Policy.HIERARCHICAL, cfg.eplb().policy());

        EngineConfig laterFlagWins = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                "--eplb-config", "{\"policy\": \"hierarchical\"}", "--eplb-policy", "default");
        assertEquals(EplbConfig.Policy.DEFAULT, laterFlagWins.eplb().policy());
    }

    @Test
    public void theEplbConfigJsonAcceptsTheAlwaysOnAndSynchronousSettings() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                "--eplb-config", "{\"log_balancedness\": true, \"log_balancedness_interval\": 2,"
                        + " \"use_async\": false, \"enable_migration_batching\": false, \"communicator\": null}");

        assertTrue(cfg.eplb().enabled());
    }

    @Test
    public void theEplbConfigJsonKeysTheEngineCannotRunAreRefusedByName() {
        for (String[] json : new String[][]{
                new String[]{"{\"use_async\": true}", "use_async"},
                new String[]{"{\"communicator\": \"nixl\"}", "communicator"},
                new String[]{"{\"enable_migration_batching\": true}", "enable_migration_batching"},
                new String[]{"{\"window_size\": 0}", "window_size"},
                new String[]{"{\"step_interval\": \"soon\"}", "step_interval"},
                new String[]{"{\"num_redundant_experts\": -1}", "num_redundant_experts"},
                new String[]{"{\"log_balancedness_interval\": 0}", "log_balancedness_interval"},
                new String[]{"{\"log_balancedness\": \"yes\"}", "log_balancedness"},
                new String[]{"{\"expert_placement_strategy\": \"round_robin\"}", "expert_placement_strategy"}}) {
            EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb",
                            "--eplb-config", json[0]));
            assertTrue(e.getMessage(), e.getMessage().contains("--eplb-config"));
            assertTrue(e.getMessage(), e.getMessage().contains(json[1]));
        }
    }

    @Test
    public void theEplbConfigJsonNeedsTheBalancerEnabled() {
        EngineLoadFlags.FlagException off = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--split-mode", "tensor", "--enable-expert-parallel", "--eplb-config",
                        "{\"num_redundant_experts\": 32, \"window_size\": 1000, \"step_interval\": 3000}"));
        assertTrue(off.getMessage(), off.getMessage().contains("--eplb-config"));
        assertTrue(off.getMessage(), off.getMessage().contains("--enable-eplb"));

        EngineLoadFlags.FlagException disabled = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--split-mode", "tensor", "--enable-expert-parallel", "--enable-eplb", "--no-eplb",
                        "--eplb-config", "{\"policy\": \"default\"}"));
        assertTrue(disabled.getMessage(), disabled.getMessage().contains("--eplb-config"));
    }

    @Test
    public void theEplbFlagsAreRefusedWithoutExpertParallel() {
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--split-mode", "tensor", "--enable-eplb", "--eplb-profile", "/tmp/load.bin"));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_EXPERT_PARALLEL));
    }

    @Test
    public void anUnknownEplbPolicyIsRefusedByTheFlag() {
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--split-mode", "tensor", "--enable-expert-parallel", "--eplb-policy", "fastest"));

        assertTrue(e.getMessage(), e.getMessage().contains("--eplb-policy"));
        assertTrue(e.getMessage(), e.getMessage().contains("fastest"));
    }

    @Test
    public void parseReturnsTheIndexOfTheFlagsLastArgumentOrMinusOne() throws EngineLoadFlags.FlagException {
        EngineLoadFlags load = EngineLoadFlags.fromSystemProperties();
        String[] args = {"-npp", "4", "-ngl", "all", "--cpu-moe", "-ctk", "q8_0"};
        assertEquals("not a load flag", -1, load.parse(args, 0));
        assertEquals("a flag with a value ends at the value", 3, load.parse(args, 2));
        assertEquals("a switch ends at itself", 4, load.parse(args, 4));
        assertEquals(6, load.parse(args, 5));
        EngineConfig cfg = load.build();
        assertEquals(EngineConfig.GPU_LAYERS_ALL, cfg.gpuLayers());
        assertTrue(cfg.cpuMoe());
        assertEquals(KvDType.Q8_0, cfg.memory().typeK());
        assertEquals("-ctv untouched", KvDType.F16, cfg.memory().typeV());
    }

    @Test
    public void theDefaultsAreLlmDemosAndTheBackendPropertyIsTheBase() throws EngineLoadFlags.FlagException {
        EngineConfig defaults = parse();
        assertEquals("the best GPU backend, as LLMDemo", EngineConfig.BackendMode.AUTO, defaults.backend());
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, defaults.gpuLayers());
        System.setProperty(EngineConfig.PROP_BACKEND, "cpu");
        assertEquals(EngineConfig.BackendMode.CPU, parse().backend());
        assertEquals("a flag overrides the property", EngineConfig.BackendMode.OPENCL,
                parse("--backend", "opencl").backend());
        System.setProperty(EngineConfig.PROP_BACKEND, "bogus");
        assertThrows(EngineLoadFlags.FlagException.class, EngineLoadFlags::fromSystemProperties);
    }

    @Test
    public void theKvOffloadFlagsReachTheEngine() throws EngineLoadFlags.FlagException {
        assertTrue("the KV caches stay on the device by default", parse().kvOffload());
        assertFalse(parse("-nkvo").kvOffload());
        assertFalse(parse("--no-kv-offload").kvOffload());
        assertTrue("the later flag wins", parse("-nkvo", "-kvo").kvOffload());
        assertTrue(parse("-kvo").kvOffload());
        String saved = System.getProperty(EngineConfig.PROP_KV_OFFLOAD);
        try {
            System.setProperty(EngineConfig.PROP_KV_OFFLOAD, "false");
            assertFalse("the property is the base", parse().kvOffload());
            assertTrue(parse("--kv-offload").kvOffload());
        } finally {
            if (saved == null) {
                System.clearProperty(EngineConfig.PROP_KV_OFFLOAD);
            } else {
                System.setProperty(EngineConfig.PROP_KV_OFFLOAD, saved);
            }
        }
    }

    @Test
    public void theBackendAndPlacementFlagsReachTheEngine() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--backend", "cuda", "-ngl", "12", "--device", "RTX", "-ot", "ffn.*=cpu",
                "--n-cpu-moe", "3", "--kv-type", "q4_0");
        assertEquals(EngineConfig.BackendMode.CUDA, cfg.backend());
        assertEquals(12, cfg.gpuLayers());
        assertEquals("RTX", cfg.deviceFilter());
        assertEquals("cpu", cfg.tensorOverrides().get(0).target());
        assertEquals(3, cfg.nCpuMoe());
        assertEquals(KvDType.Q4_0, cfg.memory().typeK());
        assertEquals(KvDType.Q4_0, cfg.memory().typeV());
        assertEquals(EngineConfig.BackendMode.CPU, parse("--cpu").backend());
        assertEquals(EngineConfig.BackendMode.AUTO, parse("--cpu", "--gpu").backend());
    }

    /** A TurboQuant preset names a key and a value stream together, so only {@code --kv-type} accepts it. */
    @Test
    public void aTurboQuantPresetSetsBothCaches() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--kv-type", "turboquant_3bit_nc");
        assertEquals(KvDType.TQ_3BIT_KEYS, cfg.memory().typeK());
        assertEquals(KvDType.TQ_3BIT_VALUES, cfg.memory().typeV());
        EngineLoadFlags.FlagException split = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-ctk", "turboquant_k8v4"));
        assertTrue(split.getMessage(), split.getMessage().contains(MemoryConfig.PROP_KV_TYPE));
    }

    @Test
    public void flashAttentionTakesLlamaCppsSpellingsAndRefusesOff() throws EngineLoadFlags.FlagException {
        for (String on : new String[] {"on", "enabled", "true", "1", "auto", "-1", "AUTO"}) {
            parse("-fa", on);
            parse("--flash-attn", on);
        }
        for (String off : new String[] {"off", "disabled", "false", "0"}) {
            EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("-fa", off));
            assertTrue(e.getMessage(), e.getMessage().startsWith("-fa " + off + " is not supported"));
        }
        EngineLoadFlags.FlagException bad = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--flash-attn", "sometimes"));
        assertTrue(bad.getMessage(), bad.getMessage().contains("--flash-attn") && bad.getMessage().contains("on, off"));
        EngineLoadFlags.FlagException missing = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-fa"));
        assertEquals("-fa needs a value", missing.getMessage());
    }

    @Test
    public void invalidValuesAreRefusedNamingTheFlag() {
        assertMessage("-ngl", "-ngl", "bogus");
        assertMessage("-ctv", "-ctv", "q6_k");
        assertMessage("--backend", "--backend", "tpu");
        assertMessage("-ot", "-ot", "nopattern");
        assertMessage("--n-cpu-moe", "--n-cpu-moe", "-1");
    }

    @Test
    public void deviceSelectionNamesTheFlagItWasGiven() throws EngineLoadFlags.FlagException {
        EngineLoadFlags.FlagException mixed = assertThrows(EngineLoadFlags.FlagException.class,
                () -> EngineLoadFlags.deviceSelection("-dev", "none,0"));
        assertTrue(mixed.getMessage(), mixed.getMessage().startsWith("-dev none,0"));
        EngineLoadFlags.FlagException empty = assertThrows(EngineLoadFlags.FlagException.class,
                () -> EngineLoadFlags.deviceSelection("--device", "0,,1"));
        assertTrue(empty.getMessage(), empty.getMessage().startsWith("--device 0,,1"));
        assertEquals("the selectors are trimmed and joined", "0,1",
                EngineLoadFlags.deviceSelection("--device", "0, 1"));
        assertEquals("none", EngineLoadFlags.deviceSelection("-device", "none"));
        assertNull("auto is the engine default", EngineLoadFlags.deviceSelection("-dev", "auto"));
    }

    @Test
    public void theMoeCacheTailSkipFlagKeepsTheOtherCacheSettings() throws EngineLoadFlags.FlagException {
        EngineConfig cfg = parse("--moe-cache", "on", "--moe-cache-tail-skip", "7");
        assertEquals(7, cfg.moeCache().tailSkipRank());
        assertEquals(MoeCacheConfig.Mode.ON, cfg.moeCache().mode());

        // changing the mode keeps the rank, and the rank flag keeps the mode
        assertEquals(7, parse("--moe-cache-tail-skip", "7", "--moe-cache", "auto").moeCache().tailSkipRank());
        assertEquals(MoeCacheConfig.Mode.ON,
                parse("--moe-cache", "on", "--moe-cache-tail-skip", "7").moeCache().mode());
        assertEquals(0, parse("--moe-cache-tail-skip", "off").moeCache().tailSkipRank());

        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--moe-cache-tail-skip", "12"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--moe-cache-tail-skip"));
    }

    @Test
    public void theNumaAndLoadModeFlagsSetTheProcessPlacementAndPin() throws EngineLoadFlags.FlagException {
        String priorNuma = System.getProperty(NumaStrategy.PROPERTY);
        String priorMode = System.getProperty(TensorReadMode.PROP_READ_MODE);
        try {
            System.clearProperty(NumaStrategy.PROPERTY);
            System.clearProperty(TensorReadMode.PROP_READ_MODE);

            EngineConfig cfg = parse("--numa", "isolate", "--mlock");
            assertEquals(NumaStrategy.ISOLATE, cfg.numaStrategy());
            assertEquals("the pool reads the property", "isolate", System.getProperty(NumaStrategy.PROPERTY));
            assertEquals("mmap+mlock over the default mmap", "mmap+mlock",
                    System.getProperty(TensorReadMode.PROP_READ_MODE));
            assertEquals("--numa off disables placement", NumaStrategy.DISABLED,
                    parse("--numa", "off").numaStrategy());

            parse("-lm", "none");
            assertEquals("read", System.getProperty(TensorReadMode.PROP_READ_MODE));
            parse("--mlock");
            assertEquals("--mlock after --no-mmap's mode pins the read buffers", "mlock",
                    System.getProperty(TensorReadMode.PROP_READ_MODE));
            parse("-lm", "auto");
            assertEquals("auto leaves the configured mode", "mlock",
                    System.getProperty(TensorReadMode.PROP_READ_MODE));
            parse("-lm", "mmap+mlock");
            assertEquals("mmap+mlock", System.getProperty(TensorReadMode.PROP_READ_MODE));

            assertMessage("--numa", "--numa", "mirror");
            assertMessage("-lm", "-lm", "mmap+mlock+dio");

            parse("-lm", "dio");
            EngineLoadFlags.FlagException direct = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("--mlock"));
            assertTrue(direct.getMessage(), direct.getMessage().startsWith("--mlock")
                    && direct.getMessage().contains("direct I/O bypasses the page cache"));

            System.setProperty(TensorReadMode.PROP_READ_MODE, "lazy");
            EngineLoadFlags.FlagException lazy = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("--mlock"));
            assertTrue(lazy.getMessage(), lazy.getMessage().contains("lazily read tensors"));
        } finally {
            restore(NumaStrategy.PROPERTY, priorNuma);
            restore(TensorReadMode.PROP_READ_MODE, priorMode);
        }
    }

    private static void restore(String key, String prior) {
        if (prior == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, prior);
        }
    }

    private static void assertMessage(String flag, String... args) {
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class, () -> parse(args));
        assertTrue(e.getMessage(), e.getMessage().contains(flag));
    }

    @Test
    public void pagedConfigLaysTheToolsMemoryOverTheLoadSettings() throws EngineLoadFlags.FlagException {
        EngineConfig load = parse("--backend", "opencl", "-ngl", "all", "-ctk", "q8_0");
        EngineConfig cfg = EngineLoadFlags.pagedConfig(load, 128, 16, 4);
        MemoryConfig m = cfg.memory();
        assertEquals(EngineConfig.BackendMode.OPENCL, cfg.backend());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, cfg.gpuLayers());
        assertEquals(128, m.nCtx());
        assertEquals(16, m.ubatch());
        assertEquals(4, m.parallel());
        assertTrue(m.kvPaged());
        assertEquals("the load's KV types stay", KvDType.Q8_0, m.typeK());
        assertFalse("the load configuration is unchanged", load.memory().kvPaged());
        assertThrows(IllegalArgumentException.class,
                () -> EngineLoadFlags.pagedConfig(load, 128, 16, MemoryConfig.MAX_PARALLEL + 1));
    }

    @Test
    public void theSplitFlagsAreSharedWithTheBatchedTools() throws EngineLoadFlags.FlagException {
        EngineLoadFlags load = EngineLoadFlags.fromSystemProperties();
        String[] args = {"-dev", "0,1", "-ts", "3/1", "-mg", "1", "-npp", "4"};
        assertEquals(1, load.parse(args, 0));
        assertEquals(3, load.parse(args, 2));
        assertEquals(5, load.parse(args, 4));
        assertEquals("not a load flag", -1, load.parse(args, 6));
        EngineConfig cfg = load.build();
        assertTrue("a device list splits by layer", cfg.split());
        assertEquals(java.util.Arrays.asList("0", "1"), cfg.gpuDevices());
        assertEquals(3.0, cfg.tensorSplit()[0], 0.0);
        assertEquals(1, cfg.mainGpu());
        EngineConfig paged = EngineLoadFlags.pagedConfig(cfg, 128, 16, 4);
        assertTrue("the paged tool configuration keeps the split", paged.split());
        assertEquals("one device keeps its single-GPU meaning", "RTX", parse("--device", "RTX").deviceFilter());
        assertEquals(EngineConfig.BackendMode.CPU, parse("-dev", "none").backend());
        assertMessage("--tensor-split", "--tensor-split", "3,1");
        assertMessage("--main-gpu 2 is beyond", "-dev", "0,1", "-mg", "2");
    }

    @Test
    public void theTensorSplitModeTakesTheFlagsAndRefusesATensorSplit() throws EngineLoadFlags.FlagException {
        EngineConfig tensor = parse("--device", "0,1", "--split-mode", "tensor");
        assertTrue(tensor.tensorMode());
        assertTrue("a tensor split is not a layer split", !tensor.split());
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--split-mode", "tensor", "--tensor-split", "2,2"));
        assertTrue(e.getMessage(), e.getMessage().contains("--tensor-split"));
        assertTrue(e.getMessage(), e.getMessage().contains("--split-mode"));
        assertMessage("-sm tensorish", "-sm", "tensorish");
    }

    @Test
    public void theRowSplitModeTakesTensorSplitShares() throws EngineLoadFlags.FlagException {
        EngineConfig row = parse("--device", "0,1", "--split-mode", "row", "--tensor-split", "3,1");
        assertTrue(row.rowSplit());
        assertEquals(3.0, row.tensorSplit()[0], 0.0);
        assertEquals(1.0, row.tensorSplit()[1], 0.0);
        EngineLoadFlags.FlagException e = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("--device", "0,1,2", "--split-mode", "row", "--tensor-split", "3,1"));
        assertTrue(e.getMessage(), e.getMessage().contains("--tensor-split"));
        assertTrue(e.getMessage(), e.getMessage().contains("3 GPU(s)"));
    }

    @Test
    public void theContextParallelFlagsParseIntoTheConfiguration() throws EngineLoadFlags.FlagException {
        EngineConfig decode = parse("-dcp", "2");
        assertEquals(EngineConfig.SplitMode.CONTEXT, decode.splitMode());
        assertEquals(Arrays.asList("0", "1"), decode.gpuDevices());

        EngineConfig prefill = parse("-pcp", "2");
        assertEquals(EngineConfig.SplitMode.CONTEXT, prefill.splitMode());
        assertEquals(2, prefill.prefillContextParallelSize());

        EngineLoadFlags.FlagException unequal = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-dcp", "2", "-pcp", "3"));
        assertTrue(unequal.getMessage(), unequal.getMessage().contains("--prefill-context-parallel-size")
                && unequal.getMessage().contains("must equal 2"));
        EngineLoadFlags.FlagException besideLayer = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-dcp", "2", "-sm", "layer"));
        assertTrue(besideLayer.getMessage(), besideLayer.getMessage().contains("dcp-within-tp"));
        EngineLoadFlags.FlagException bad = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-dcp", "x"));
        assertTrue(bad.getMessage(), bad.getMessage().contains("-dcp"));

        EngineLoadFlags.FlagException contiguous = assertThrows(EngineLoadFlags.FlagException.class,
                () -> parse("-dcp", "2", "--dcp-kv-cache-interleave-size", "16"));
        assertTrue(contiguous.getMessage(), contiguous.getMessage().contains("contiguous"));

        String priorParallel = System.getProperty(MemoryConfig.PROP_PARALLEL);
        try {
            System.setProperty(MemoryConfig.PROP_PARALLEL, "2");
            EngineConfig page = parse("-dcp", "2", "--dcp-kv-cache-interleave-size", "16");
            assertTrue(page.memory().paged());
            assertEquals(16, page.memory().kvPageSize());
            EngineLoadFlags.FlagException wrong = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("-dcp", "2", "--dcp-kv-cache-interleave-size", "8"));
            assertTrue(wrong.getMessage(), wrong.getMessage().contains("page size 16"));
            EngineLoadFlags.FlagException paged = assertThrows(EngineLoadFlags.FlagException.class,
                    () -> parse("-pcp", "2"));
            assertTrue(paged.getMessage(), paged.getMessage().contains("paged"));
        } finally {
            restore(MemoryConfig.PROP_PARALLEL, priorParallel);
        }
    }

    @Test
    public void theUsageSectionDescribesTheRowAndTensorSplitModes() {
        String usage = String.join("\n", com.eonmux.neuralfree.demo.SplitFlags.usage());
        assertTrue(usage, usage.contains("row (each GPU holds a"));
        assertTrue(usage, usage.contains("slice of every dense feed-forward and an all-reduce sums the"));
        assertTrue(usage, usage.contains("tensor (vLLM tensor parallelism"));
        assertTrue(usage, !usage.contains("tensor is refused"));
    }

    @Test
    public void theUsageSectionListsEverySharedFlag() {
        String usage = EngineLoadFlags.usageSection("load:");
        assertTrue(usage.startsWith("load:\n"));
        for (String flag : new String[] {"--backend", "--device", "-ngl", "-ot", "--cpu-moe", "-fa", "--kv-type",
                "-ctk", "-ctv", "--threads", "--numa", "-lm, --load-mode", "--mlock", "--split-mode",
                "--tensor-split", "--main-gpu", "--decode-context-parallel-size",
                "--prefill-context-parallel-size", "--dcp-kv-cache-interleave-size"}) {
            assertTrue(flag + " in " + usage, usage.contains(flag));
        }
    }

    @Test
    public void theBackendUsageLineNamesEveryBackendMode() {
        String backendLine = null;
        for (String line : EngineLoadFlags.backendUsage()) {
            if (line.contains("--backend")) {
                backendLine = line;
                break;
            }
        }
        assertTrue("the usage has a --backend line: " + EngineLoadFlags.backendUsage(), backendLine != null);
        for (EngineConfig.BackendMode mode : EngineConfig.BackendMode.values()) {
            assertTrue("the --backend usage line names " + mode + ": " + backendLine,
                    backendLine.contains(mode.name().toLowerCase(java.util.Locale.ROOT)));
        }
    }
}

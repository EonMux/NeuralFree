package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Arrays;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Property and builder parsing of the multi-GPU settings ({@code neuralfree.split.mode}, {@code
 * neuralfree.tensor.split}, {@code neuralfree.gpu.devices}, {@code neuralfree.main.gpu}); {@code tensor} selects
 * the tensor-parallel mode, {@code neuralfree.expert.parallel} its whole-expert placement, {@code context} the
 * decode-context parallelism, {@code neuralfree.prefill_context_parallel_size} its prefill query split, and
 * {@code row} the feed-forward row split.
 */
public class EngineConfigSplitTest {

    @Test
    public void defaultsRunOneDevice() {
        EngineConfig cfg = EngineConfig.fromProperties(new Properties());
        assertEquals(EngineConfig.SplitMode.NONE, cfg.splitMode());
        assertTrue(cfg.gpuDevices().isEmpty());
        assertArrayEquals(new double[0], cfg.tensorSplit(), 0.0);
        assertEquals(0, cfg.mainGpu());
        assertFalse(cfg.split());
    }

    @Test
    public void layerModeParsesFromProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertTrue(cfg.split());
    }

    @Test
    public void noneModeAndBlankValueStaySingleDevice() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "none");
        assertEquals(EngineConfig.SplitMode.NONE, EngineConfig.fromProperties(p).splitMode());
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "  ");
        assertEquals(EngineConfig.SplitMode.NONE, EngineConfig.fromProperties(p).splitMode());
    }

    @Test
    public void rowModeSelectsTheFeedForwardRowSplit() {
        for (String mode : new String[]{"row", "ROW"}) {
            Properties p = new Properties();
            p.setProperty(EngineConfig.PROP_SPLIT_MODE, mode);
            EngineConfig cfg = EngineConfig.fromProperties(p);
            assertEquals(mode, EngineConfig.SplitMode.ROW, cfg.splitMode());
            assertTrue(mode, cfg.rowSplit());
            assertFalse("a row split is not a layer split", cfg.split());
        }
        assertEquals(EngineConfig.SplitMode.ROW, EngineConfig.builder()
                .splitMode(EngineConfig.SplitMode.ROW).build().splitMode());
    }

    @Test
    public void tensorModeIsAccepted() {
        for (String mode : new String[]{"tensor", "TENSOR"}) {
            Properties p = new Properties();
            p.setProperty(EngineConfig.PROP_SPLIT_MODE, mode);
            EngineConfig cfg = EngineConfig.fromProperties(p);
            assertSame("tensor selects vLLM tensor parallelism", EngineConfig.SplitMode.TENSOR, cfg.splitMode());
            assertTrue(mode, cfg.tensorMode());
            assertTrue(mode, cfg.tensorParallel());
            assertFalse(mode, cfg.rowSplit());
            assertFalse(mode, cfg.split());
        }
    }

    @Test
    public void contextModeIsAccepted() {
        for (String mode : new String[]{"context", "CONTEXT"}) {
            Properties p = new Properties();
            p.setProperty(EngineConfig.PROP_SPLIT_MODE, mode);
            EngineConfig cfg = EngineConfig.fromProperties(p);
            assertSame("context selects vLLM decode context parallelism", EngineConfig.SplitMode.CONTEXT,
                    cfg.splitMode());
            assertTrue(mode, cfg.contextMode());
            assertFalse(mode, cfg.tensorMode());
            assertFalse(mode, cfg.tensorParallel());
            assertFalse(mode, cfg.rowSplit());
            assertFalse(mode, cfg.split());
        }
        assertSame(EngineConfig.SplitMode.CONTEXT, EngineConfig.builder()
                .splitMode(EngineConfig.SplitMode.CONTEXT).build().splitMode());
    }

    @Test
    public void contextModeOnTheCpuBackendIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).splitMode(EngineConfig.SplitMode.CONTEXT).build());
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_BACKEND + "=cpu"));
    }

    @Test
    public void prefillContextParallelParsesFromProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "context");
        p.setProperty(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL, "2");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(2, cfg.prefillContextParallelSize());
        assertTrue(cfg.prefillContextParallel());
    }

    @Test
    public void prefillContextParallelDefaultsToOneWholePrefill() {
        EngineConfig cfg = EngineConfig.fromProperties(new Properties());
        assertEquals(1, cfg.prefillContextParallelSize());
        assertFalse(cfg.prefillContextParallel());
    }

    @Test
    public void prefillContextParallelOutsideAContextSplitIsRefusedByName() {
        for (EngineConfig.SplitMode mode : new EngineConfig.SplitMode[]{EngineConfig.SplitMode.NONE,
                EngineConfig.SplitMode.LAYER, EngineConfig.SplitMode.ROW, EngineConfig.SplitMode.TENSOR}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                    .splitMode(mode).prefillContextParallelSize(2).build());
            assertTrue(mode + " build", e.getMessage().contains(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL));
            assertTrue(mode + " build", e.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
        }
    }

    @Test
    public void prefillContextParallelBelowOneIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .splitMode(EngineConfig.SplitMode.CONTEXT).prefillContextParallelSize(0).build());
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL));
    }

    @Test
    public void unknownSplitModeIsRefusedByName() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "pipeline");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.split.mode"));
    }

    @Test
    public void rowSplitOnTheCpuBackendIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).splitMode(EngineConfig.SplitMode.ROW).build());
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_BACKEND + "=cpu"));
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "row");
        p.setProperty(EngineConfig.PROP_BACKEND, "cpu");
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
    }

    @Test
    public void deviceListParsesAndTrims() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, " cuda0 , NVIDIA GeForce ");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(Arrays.asList("cuda0", "NVIDIA GeForce"), cfg.gpuDevices());
    }

    @Test
    public void emptyDeviceEntryIsRefusedByName() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, "a,,b");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.gpu.devices"));
    }

    @Test
    public void tensorSplitParsesShares() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_SPLIT, "3, 1");
        assertArrayEquals(new double[]{3, 1}, EngineConfig.fromProperties(p).tensorSplit(), 1e-12);
    }

    @Test
    public void nonNumericShareIsRefusedByName() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_SPLIT, "3,x");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.tensor.split"));
    }

    @Test
    public void nonPositiveShareIsRefusedByName() {
        for (String value : new String[]{"0,1", "-1,1", "NaN,1"}) {
            Properties p = new Properties();
            p.setProperty(EngineConfig.PROP_TENSOR_SPLIT, value);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> EngineConfig.fromProperties(p));
            assertTrue(value, e.getMessage().contains("neuralfree.tensor.split"));
        }
    }

    @Test
    public void negativeMainGpuIsRefusedByName() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_MAIN_GPU, "-1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.main.gpu"));
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder().mainGpu(-2));
    }

    @Test
    public void builderRoundTripsTheSplitSettings() {
        EngineConfig cfg = EngineConfig.builder()
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("cuda0")
                .addGpuDevice("cuda1")
                .tensorSplit(new double[]{2, 1})
                .mainGpu(1)
                .build();
        assertEquals(EngineConfig.SplitMode.LAYER, cfg.splitMode());
        assertEquals(Arrays.asList("cuda0", "cuda1"), cfg.gpuDevices());
        assertArrayEquals(new double[]{2, 1}, cfg.tensorSplit(), 1e-12);
        assertEquals(1, cfg.mainGpu());
        assertTrue(cfg.split());
        EngineConfig copy = cfg.toBuilder().build();
        assertEquals(cfg.splitMode(), copy.splitMode());
        assertEquals(cfg.gpuDevices(), copy.gpuDevices());
        assertArrayEquals(cfg.tensorSplit(), copy.tensorSplit(), 1e-12);
        assertEquals(cfg.mainGpu(), copy.mainGpu());
        assertTrue(copy.toString().contains("split=layer"));
        assertTrue(copy.toString().contains("mainGpu=1"));
    }

    @Test
    public void builderRejectsBadShares() {
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder().tensorSplit(new double[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().tensorSplit(new double[]{Double.POSITIVE_INFINITY}));
        assertThrows(NullPointerException.class, () -> EngineConfig.builder().tensorSplit(null));
        assertThrows(NullPointerException.class, () -> EngineConfig.builder().splitMode(null));
    }

    @Test
    public void layerSplitOnTheCpuBackendIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU).splitMode(EngineConfig.SplitMode.LAYER).build());
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_BACKEND + "=cpu"));
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        p.setProperty(EngineConfig.PROP_BACKEND, "cpu");
        assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
        assertFalse(EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build().split());
    }

    @Test
    public void gpuVramLimitIsReadFromItsPropertyAndKeptByToBuilder() {
        assertEquals(EngineConfig.GPU_VRAM_UNLIMITED, EngineConfig.fromProperties(new Properties()).gpuVramLimitMb());
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_GPU_VRAM_LIMIT, "512");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(512L, cfg.gpuVramLimitMb());
        assertEquals(512L * 1024 * 1024, cfg.gpuVramLimitBytes());
        assertEquals(512L, cfg.toBuilder().build().gpuVramLimitMb());
        assertTrue(cfg.toString(), cfg.toString().contains("gpuVramLimitMb=512"));
    }

    @Test
    public void negativeOrMalformedGpuVramLimitIsRefusedByName() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_GPU_VRAM_LIMIT, "-1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_GPU_VRAM_LIMIT));
        p.setProperty(EngineConfig.PROP_GPU_VRAM_LIMIT, "lots");
        e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_GPU_VRAM_LIMIT));
    }

    @Test
    public void mainGpuBeyondTheDeviceListWithoutASplitIsRefusedFromProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "none");
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, "0,1");
        p.setProperty(EngineConfig.PROP_MAIN_GPU, "1");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(1, cfg.mainGpu());
        assertTrue(cfg.picksMainGpu());
        p.setProperty(EngineConfig.PROP_MAIN_GPU, "2");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_MAIN_GPU));
    }

    @Test
    public void expertParallelParsesOnlyWithATensorSplit() {
        assertFalse(EngineConfig.fromProperties(new Properties()).expertParallel());
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        p.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        assertTrue(EngineConfig.fromProperties(p).expertParallel());

        Properties row = new Properties();
        row.setProperty(EngineConfig.PROP_SPLIT_MODE, "row");
        row.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        IllegalArgumentException split = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(row));
        assertTrue(split.getMessage(), split.getMessage().contains(EngineConfig.PROP_EXPERT_PARALLEL));

        Properties offload = new Properties();
        offload.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        offload.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        offload.setProperty(EngineConfig.PROP_CPU_MOE, "true");
        IllegalArgumentException experts = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(offload));
        assertTrue(experts.getMessage(), experts.getMessage().contains(EngineConfig.PROP_EXPERT_PARALLEL));
    }

    @Test
    public void thePooledExpertCacheNeedsTheCacheAndALayerSplit() {
        Properties off = new Properties();
        off.setProperty(EngineConfig.PROP_MOE_CACHE_POOL, "true");
        IllegalArgumentException noCache = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(off));
        assertTrue(noCache.getMessage(), noCache.getMessage().contains(EngineConfig.PROP_MOE_CACHE_POOL));

        Properties row = new Properties();
        row.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        row.setProperty(EngineConfig.PROP_SPLIT_MODE, "row");
        row.setProperty(EngineConfig.PROP_MOE_CACHE_POOL, "true");
        IllegalArgumentException split = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(row));
        assertTrue(split.getMessage(), split.getMessage().contains(EngineConfig.PROP_MOE_CACHE_POOL));

        Properties layer = new Properties();
        layer.setProperty(MoeCacheConfig.PROP_MOE_CACHE, "on");
        layer.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        layer.setProperty(EngineConfig.PROP_MOE_CACHE_POOL, "true");
        EngineConfig cfg = EngineConfig.fromProperties(layer);
        assertTrue(cfg.moeCachePool());
        assertTrue(cfg.toString(), cfg.toString().contains("moeCachePool"));
        assertFalse("the pool is off by default", EngineConfig.fromProperties(new Properties()).moeCachePool());
    }

    @Test
    public void eplbParsesWithExpertParallelAndAProfileOrSave() {
        assertTrue(EngineConfig.fromProperties(new Properties()).eplb().isDefault());
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        p.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        p.setProperty(EplbConfig.PROP_EPLB, "on");
        p.setProperty(EplbConfig.PROP_EPLB_PROFILE, "/tmp/load.bin");
        p.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        EplbConfig eplb = EngineConfig.fromProperties(p).eplb();
        assertTrue(eplb.enabled());
        assertEquals("/tmp/load.bin", eplb.profile().toString());
        assertEquals("/tmp/next.bin", eplb.save().toString());
        assertEquals(EplbConfig.Policy.DEFAULT, eplb.policy());

        Properties saveOnly = new Properties();
        saveOnly.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        saveOnly.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        saveOnly.setProperty(EplbConfig.PROP_EPLB, "true");
        saveOnly.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        assertTrue(EngineConfig.fromProperties(saveOnly).eplb().enabled());
    }

    @Test
    public void eplbParsesTheRebalancingProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        p.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        p.setProperty(EplbConfig.PROP_EPLB, "on");
        p.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        p.setProperty(EplbConfig.PROP_EPLB_POLICY, "hierarchical");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_REDUNDANT, "2");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_GROUPS, "2");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_NODES, "2");
        p.setProperty(EplbConfig.PROP_EPLB_REBALANCE, "on");
        p.setProperty(EplbConfig.PROP_EPLB_STEP_INTERVAL, "64");
        p.setProperty(EplbConfig.PROP_EPLB_WINDOW_SIZE, "4");
        EplbConfig eplb = EngineConfig.fromProperties(p).eplb();
        assertEquals(EplbConfig.Policy.HIERARCHICAL, eplb.policy());
        assertEquals(2, eplb.numRedundant());
        assertEquals(2, eplb.numGroups());
        assertEquals(2, eplb.numNodes());
        assertTrue(eplb.rebalance());
        assertEquals(64, eplb.stepInterval());
        assertEquals(4, eplb.windowSize());

        Properties plain = new Properties();
        plain.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        plain.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        plain.setProperty(EplbConfig.PROP_EPLB, "on");
        plain.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        EplbConfig defaults = EngineConfig.fromProperties(plain).eplb();
        assertEquals(EplbConfig.Policy.DEFAULT, defaults.policy());
        assertEquals(0, defaults.numRedundant());
        assertEquals(0, defaults.numGroups());
        assertEquals(1, defaults.numNodes());
        assertFalse(defaults.rebalance());
        assertEquals(1000, defaults.stepInterval());
        assertEquals(8, defaults.windowSize());
    }

    @Test
    public void eplbRefusesOutOfRangeCountsAndRebalancingWithoutTheBalancer() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        p.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        p.setProperty(EplbConfig.PROP_EPLB, "on");
        p.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_REDUNDANT, "-1");
        IllegalArgumentException redundant = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(redundant.getMessage(), redundant.getMessage().contains(EplbConfig.PROP_EPLB_NUM_REDUNDANT));

        p.setProperty(EplbConfig.PROP_EPLB_NUM_REDUNDANT, "0");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_GROUPS, "-2");
        IllegalArgumentException groups = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(groups.getMessage(), groups.getMessage().contains(EplbConfig.PROP_EPLB_NUM_GROUPS));

        p.setProperty(EplbConfig.PROP_EPLB_NUM_GROUPS, "0");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_NODES, "0");
        IllegalArgumentException nodes = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(nodes.getMessage(), nodes.getMessage().contains(EplbConfig.PROP_EPLB_NUM_NODES));

        p.setProperty(EplbConfig.PROP_EPLB_NUM_NODES, "1");
        p.setProperty(EplbConfig.PROP_EPLB_STEP_INTERVAL, "0");
        IllegalArgumentException interval = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(interval.getMessage(), interval.getMessage().contains(EplbConfig.PROP_EPLB_STEP_INTERVAL));

        p.setProperty(EplbConfig.PROP_EPLB_STEP_INTERVAL, "16");
        p.setProperty(EplbConfig.PROP_EPLB_WINDOW_SIZE, "0");
        IllegalArgumentException window = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(window.getMessage(), window.getMessage().contains(EplbConfig.PROP_EPLB_WINDOW_SIZE));

        p.setProperty(EplbConfig.PROP_EPLB_WINDOW_SIZE, "4");
        p.remove(EplbConfig.PROP_EPLB_SAVE);
        p.setProperty(EplbConfig.PROP_EPLB_REBALANCE, "on");
        p.setProperty(EplbConfig.PROP_EPLB, "off");
        IllegalArgumentException off = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(off.getMessage(), off.getMessage().contains(EplbConfig.PROP_EPLB_REBALANCE));

        p.setProperty(EplbConfig.PROP_EPLB_REBALANCE, "off");
        p.setProperty(EplbConfig.PROP_EPLB_NUM_REDUNDANT, "2");
        IllegalArgumentException redundantOff = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(redundantOff.getMessage(), redundantOff.getMessage().contains(EplbConfig.PROP_EPLB_NUM_REDUNDANT));
    }

    @Test
    public void eplbNeedsExpertParallelAndAProfileOrASave() {
        Properties noParallel = new Properties();
        noParallel.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        noParallel.setProperty(EplbConfig.PROP_EPLB, "on");
        noParallel.setProperty(EplbConfig.PROP_EPLB_PROFILE, "/tmp/load.bin");
        IllegalArgumentException parallel = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(noParallel));
        assertTrue(parallel.getMessage(), parallel.getMessage().contains(EngineConfig.PROP_EXPERT_PARALLEL));

        Properties nothing = new Properties();
        nothing.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        nothing.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        nothing.setProperty(EplbConfig.PROP_EPLB, "on");
        IllegalArgumentException paths = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(nothing));
        assertTrue(paths.getMessage(), paths.getMessage().contains(EplbConfig.PROP_EPLB_PROFILE));
        assertTrue(paths.getMessage(), paths.getMessage().contains(EplbConfig.PROP_EPLB_SAVE));
    }

    @Test
    public void eplbRefusesAPathWithTheBalancerOffAndAnUnknownPolicy() {
        Properties off = new Properties();
        off.setProperty(EplbConfig.PROP_EPLB_PROFILE, "/tmp/load.bin");
        IllegalArgumentException disabled = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(off));
        assertTrue(disabled.getMessage(), disabled.getMessage().contains(EplbConfig.PROP_EPLB_PROFILE));

        Properties policy = new Properties();
        policy.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        policy.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        policy.setProperty(EplbConfig.PROP_EPLB, "on");
        policy.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        policy.setProperty(EplbConfig.PROP_EPLB_POLICY, "fastest");
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(policy));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains(EplbConfig.PROP_EPLB_POLICY));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("fastest"));
    }

    @Test
    public void eplbRefusesASecondExpertProfileWriter() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "tensor");
        p.setProperty(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        p.setProperty(EplbConfig.PROP_EPLB, "on");
        p.setProperty(EplbConfig.PROP_EPLB_SAVE, "/tmp/next.bin");
        p.setProperty("neuralfree.moe.cache.ranking.save", "/tmp/ranking.bin");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().contains(EplbConfig.PROP_EPLB_SAVE));
    }

    @Test
    public void aLayerSplitOverOneDeviceIndexPicksThatDeviceFromProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, "1");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertTrue(cfg.split());
        assertTrue(cfg.picksMainGpu());
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, "GeForce");
        assertFalse(EngineConfig.fromProperties(p).picksMainGpu());
        p.setProperty(EngineConfig.PROP_GPU_DEVICES, "0,1");
        p.setProperty(EngineConfig.PROP_MAIN_GPU, "1");
        assertFalse("a split over two devices keeps them both", EngineConfig.fromProperties(p).picksMainGpu());
    }
}

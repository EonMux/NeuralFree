package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The runtime expert rearrangement over a synthetic tensor split: the routed-row window records exactly the steps the
 * next boundary reads, a boundary re-balances the layers from the window's loads and migrates the changed layouts to
 * the migrator (hot rows on one expert move its replicas onto both ranks, an idle boundary re-packs to the zero-load
 * layout), and a plan without a split routed-expert layer is refused. Hand-worked {@code EplbPolicy} results. The plan
 * needs the kernel library's tensor plans, so the cases run on the OpenCL backend and are skipped without a device.
 */
public class EplbRebalancerTest {

    private static final int RANKS = 2;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GGUFFileParser parser;
    private WeightStore store;
    private ModelSpec spec;
    private GpuWeightUploader uploader;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        MemoryConfig mem = MemoryConfig.defaults().toBuilder().ubatch(3).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        uploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "qwen3moe",
                GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
    }

    @After
    public void close() throws IOException {
        try {
            uploader.close();
            store.close();
            main.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    /** The load-time placement of {100,100,1,1} with two redundant slots: {0,0,2} on rank 0 and {1,1,3} on rank 1. */
    private TpMoePlan plan() {
        EplbPlacement placement = EplbPlacement.of(spec, RANKS, loads(100, 100, 1, 1),
                EplbPlacement.Options.of(EplbConfig.Policy.DEFAULT, 2, 0, 1));
        return TpMoePlan.of(spec, RANKS, uploader, true, placement, true);
    }

    /** A rebalancer rearranging every 2 steps over a 2 step window, collecting the migrated layouts. */
    private EplbRebalancer rebalancer(TpMoePlan plan, EplbConfig cfg, List<Map<Integer, int[]>> migrations) {
        EplbRebalancer rebalancer = EplbRebalancer.open(cfg, spec, RANKS, plan, migrations::add);
        assertTrue("the configuration rearranges", rebalancer != null);
        return rebalancer;
    }

    private static List<RankedExpert> loads(int e0, int e1, int e2, int e3) {
        return Arrays.asList(new RankedExpert(0, 0, e0), new RankedExpert(0, 1, e1),
                new RankedExpert(0, 2, e2), new RankedExpert(0, 3, e3));
    }

    @Test
    public void aBoundaryRearrangesFromTheWindowAndAnIdleBoundaryRePacks() {
        TpMoePlan plan = plan();
        int layerId = plan.layers().get(0).weightLayer();
        List<Map<Integer, int[]>> migrations = new ArrayList<>();
        EplbConfig cfg = EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 2, 1, 1, true, 2, 2);
        EplbRebalancer rebalancer = rebalancer(plan, cfg, migrations);

        assertTrue("the first two steps are within the window of the first boundary",
                rebalancer.window().recording());
        rebalancer.window().note(layerId, new int[]{0, 0}, 2);
        rebalancer.afterStep();
        assertEquals("no rearrangement before the boundary", 1, rebalancer.steps());
        assertTrue(migrations.isEmpty());
        rebalancer.window().note(layerId, new int[]{0, 0, 0}, 3);
        rebalancer.afterStep();

        assertEquals(2, rebalancer.steps());
        assertEquals("the boundary rearranged once", 1, migrations.size());
        assertArrayEquals("the hot expert's three replicas spread over both ranks and the staying experts keep"
                        + " their slots",
                new int[]{0, 0, 3, 1, 0, 2}, migrations.get(0).get(layerId));
        assertEquals("the layouts are the migrated ones", 1, rebalancer.layouts().size());
        assertArrayEquals(migrations.get(0).get(layerId), rebalancer.layouts().get(layerId));
        assertTrue("the next two steps are recorded again", rebalancer.window().recording());
        assertTrue("the window is empty after the boundary", rebalancer.window().isEmpty());

        // An empty window at the next boundary re-balances to the zero-load layout: the idle packing gives every
        // expert one slot in order, which the slot preservation keeps where the rank agrees.
        rebalancer.afterStep();
        rebalancer.afterStep();
        assertEquals(4, rebalancer.steps());
        assertEquals("the idle boundary migrated the zero-load layout", 2, migrations.size());
        assertArrayEquals(new int[]{0, 1, 2, 3, 0, 0}, migrations.get(1).get(layerId));
        assertArrayEquals(migrations.get(1).get(layerId), rebalancer.layouts().get(layerId));
    }

    @Test
    public void theWindowOnlyRecordsTheStepsTheNextBoundaryReads() {
        TpMoePlan plan = plan();
        int layerId = plan.layers().get(0).weightLayer();
        List<Map<Integer, int[]>> migrations = new ArrayList<>();
        EplbConfig cfg = EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 2, 1, 1, true, 4, 2);
        EplbRebalancer rebalancer = rebalancer(plan, cfg, migrations);

        assertFalse("the first steps are outside the window of the fourth step", rebalancer.window().recording());
        rebalancer.window().note(layerId, new int[]{0, 1, 2, 3}, 4);
        rebalancer.afterStep();
        assertFalse(rebalancer.window().recording());
        rebalancer.window().note(layerId, new int[]{0, 1, 2, 3}, 4);
        rebalancer.afterStep();
        assertTrue("from two steps before the boundary the window records", rebalancer.window().recording());
        rebalancer.window().note(layerId, new int[]{2, 2}, 2);
        rebalancer.afterStep();
        assertTrue(rebalancer.window().recording());
        rebalancer.window().note(layerId, new int[]{2, 2}, 2);
        rebalancer.afterStep();

        assertEquals(4, rebalancer.steps());
        assertEquals("only the two recorded steps reached the rearrangement", 1, migrations.size());
        assertArrayEquals("four routed rows on expert 2 give it three replicas spread over both ranks; the kept"
                        + " slots stay",
                new int[]{2, 3, 2, 1, 2, 0}, migrations.get(0).get(layerId));
        assertFalse("the steps after the boundary are outside the next window", rebalancer.window().recording());
        rebalancer.afterStep();
        assertFalse(rebalancer.window().recording());
    }

    @Test
    public void aSplitWithoutARoutedExpertLayerIsRefused() throws IOException {
        Path dense = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build())
                .writeTempFile();
        GGUFFileParser denseParser = new GGUFFileParser(dense.toString());
        WeightStore denseStore = denseParser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        GpuWeightUploader denseUploader = new GpuWeightUploader(main, queue, denseParser.getTensors(), denseStore,
                "llama", GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels)).build());
        try {
            ModelSpec denseSpec = ModelSpecBuilder.build(GgufKv.of(denseParser), TensorCatalog.of(denseParser),
                    MemoryConfig.defaults().toBuilder().ubatch(3).build());
            TpMoePlan plan = TpMoePlan.of(denseSpec, RANKS, denseUploader, true, null, true);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> EplbRebalancer.open(
                            EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 0, 1, 1, true, 2, 2),
                            denseSpec, RANKS, plan, layouts -> { }));
            assertTrue(e.getMessage(), e.getMessage().contains(EplbConfig.PROP_EPLB_REBALANCE));
            assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_EXPERT_PARALLEL));
        } finally {
            denseUploader.close();
            denseStore.close();
            denseParser.close();
            Files.deleteIfExists(dense);
        }
    }

    @Test
    public void aRebalancerWithoutRearrangementOrAnEmptyPlanIsNotOpened() {
        EplbConfig off = EplbConfig.of(true, null, null, EplbConfig.Policy.DEFAULT, 0, 1, 1, false, 2, 2);
        org.junit.Assert.assertNull("rebalance=off opens nothing",
                EplbRebalancer.open(off, spec, RANKS, plan(), layouts -> { }));
    }
}

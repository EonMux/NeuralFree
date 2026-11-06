package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HostMappedPlan} over a 4-layer synthetic llama: the budget takes trunk tensors in weight-layer and file
 * order, the parameter selection narrows the offload to matching tensors, a tensor a placement keeps in RAM is
 * never a candidate, and a segment naming nothing or a budget below the first tensor is refused by the property
 * that caused it.
 */
public class HostMappedPlanTest {

    private static final int N_LAYER = 2;

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;
    private LayerWeightSet.Options options;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER).nEmbd(256)
                .nFf(2048).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        backend = FakeComputeBackend.create();
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        TensorCatalog catalog = TensorCatalog.of(parser);
        GgufKv kv = GgufKv.of(parser);
        spec = ModelSpecBuilder.build(kv, catalog, MemoryConfig.defaults());
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "llama",
                GpuWeightUploader.config((type, use) -> true).build());
        options = LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name()))
                .maxShardBytes(LayerWeightSet.Options.maxShardBytes(backend.capabilities()))
                .build();
    }

    @After
    public void close() throws IOException {
        try {
            uploader.close();
            store.close();
            parser.close();
            backend.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    private static EngineConfig offload(long mb) {
        return EngineConfig.builder().cpuOffloadMb(mb).build();
    }

    private List<LayerWeightSet> trunk() {
        return LayerWeightSet.forTrunk(spec, options);
    }

    private static long trunkBytes(List<LayerWeightSet> sets) {
        long total = 0L;
        for (LayerWeightSet set : sets) {
            total += set.uploadBytes();
        }
        return total;
    }

    @Test
    public void noBudgetPlansNothing() {
        HostMappedPlan plan = HostMappedPlan.plan(offload(0), spec, options, name -> true);

        assertTrue(plan.isEmpty());
        assertEquals(0L, plan.bytes());
        assertEquals(0, plan.tensorCount());
        assertTrue(plan.sets().isEmpty());
    }

    @Test
    public void aBudgetCoveringTheModelOffloadsEveryTrunkTensor() {
        List<LayerWeightSet> trunk = trunk();
        long mb = (trunkBytes(trunk) + (1 << 20) - 1) >> 20;

        HostMappedPlan plan = HostMappedPlan.plan(offload(mb), spec, options, name -> true);

        assertEquals(named(trunk), plan.names());
        assertEquals(trunk.size(), plan.sets().size());
        assertEquals(trunkBytes(trunk), plan.bytes());
    }

    @Test
    public void aSmallBudgetTakesWholeTensorsInWeightLayerOrder() {
        List<LayerWeightSet> trunk = trunk();
        assertTrue("the fixture layer must exceed the 1 MiB budget", trunk.get(0).uploadBytes() > (1 << 20));

        HostMappedPlan plan = HostMappedPlan.plan(offload(1), spec, options, name -> true);

        assertFalse(plan.isEmpty());
        assertEquals(1, plan.sets().size());
        LayerWeightSet first = plan.sets().get(0);
        assertEquals(0, first.weightLayer());
        assertTrue(plan.bytes() <= 1 << 20);
        List<String> fileOrder = new ArrayList<>(trunk.get(0).names());
        List<String> chosen = new ArrayList<>(first.names());
        assertEquals(fileOrder.subList(0, chosen.size()), chosen);
    }

    @Test
    public void theParameterSelectionNarrowsTheOffload() {
        EngineConfig cfg = EngineConfig.builder().cpuOffloadMb(4096)
                .cpuOffloadParams(Collections.singletonList("ffn_gate.weight")).build();

        HostMappedPlan plan = HostMappedPlan.plan(cfg, spec, options, name -> true);

        assertEquals(N_LAYER, plan.tensorCount());
        for (String name : plan.names()) {
            assertTrue(name, name.endsWith("ffn_gate.weight"));
        }
        assertFalse(plan.contains("blk.0.ffn_up.weight"));
    }

    @Test
    public void tensorsAPlacementKeepsInRamAreNotCandidates() {
        EngineConfig cfg = EngineConfig.builder().cpuOffloadMb(4096)
                .cpuOffloadParams(Collections.singletonList("ffn_gate.weight")).build();

        HostMappedPlan plan = HostMappedPlan.plan(cfg, spec, options, name -> !name.startsWith("blk.0."));

        assertEquals(1, plan.tensorCount());
        assertFalse(plan.contains("blk.0.ffn_gate.weight"));
        assertTrue(plan.contains("blk.1.ffn_gate.weight"));
    }

    @Test
    public void aSegmentThatMatchesNoTensorIsRefusedByName() {
        EngineConfig cfg = EngineConfig.builder().cpuOffloadMb(64)
                .cpuOffloadParams(Collections.singletonList("not_a_tensor")).build();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> HostMappedPlan.plan(cfg, spec, options, name -> true));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_PARAMS));
        assertTrue(e.getMessage(), e.getMessage().contains("not_a_tensor"));
    }

    @Test
    public void aBudgetBelowTheFirstSelectedTensorIsRefusedByName() {
        EngineConfig cfg = EngineConfig.builder().cpuOffloadMb(1)
                .cpuOffloadParams(Collections.singletonList("ffn_gate.weight")).build();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> HostMappedPlan.plan(cfg, spec, options, name -> true));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_MB));
    }

    @Test
    public void aModelThatKeepsEveryTrunkTensorInRamIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> HostMappedPlan.plan(offload(4096), spec, options, name -> false));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_MB));
    }

    private static Set<String> named(List<LayerWeightSet> sets) {
        Set<String> out = new HashSet<>();
        for (LayerWeightSet set : sets) {
            out.addAll(set.names());
        }
        return out;
    }

}

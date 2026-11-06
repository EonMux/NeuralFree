package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The split plan of a feed-forward row split over a synthetic llama: every dense trunk layer splits on the hidden
 * dimension, a {@code n_ff} that does not divide into the rank shards is refused with the reason, a routed-MoE model
 * keeps no layer and is refused, and the tensor and layer lookups cover only the projections the plan splits.
 */
public class FfnRowSplitPlanTest {

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        backend = FakeComputeBackend.create();
        store = parser.createMappedStore(com.eonmux.neuralfree.gguf.StoreUse.RESIDENT_UPLOAD);
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "llama",
                GpuWeightUploader.config((type, use) -> true).build());
    }

    @After
    public void close() throws IOException {
        try {
            uploader.close();
            store.close();
            backend.close();
            parser.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void everyDenseTrunkLayerSplitsOnTheHiddenDimension() {
        FfnRowSplitPlan plan = FfnRowSplitPlan.of(spec, 2, uploader);
        assertEquals(2, plan.ranks());
        assertEquals(4, plan.layers().size());
        assertTrue(plan.ineligible().isEmpty());
        for (int il = 0; il < 4; il++) {
            FfnRowSplitPlan.Layer layer = plan.layers().get(il);
            assertEquals("blk." + il, layer.stack() + "." + layer.weightLayer());
            assertEquals(il, layer.il());
            assertEquals(64, layer.nIn());
            assertEquals(64, layer.nOut());
            assertEquals(128, layer.nFf());
            assertEquals(64, layer.shard());
            assertEquals(0, layer.rowStart(0));
            assertEquals(64, layer.rowStart(1));
            assertEquals("ffn_up.weight", layer.up());
            assertEquals("ffn_gate.weight", layer.gate());
            assertEquals("ffn_down.weight", layer.down());
            assertFalse("llama keeps its gate separate", layer.fusedUp());
            assertNull("llama has no FFN biases or scales", layer.upBias());
            assertNull(layer.gateBias());
            assertNull(layer.downBias());
            assertNull(layer.downScale());
            assertNull(layer.actScales());
            assertEquals("blk." + il + ".ffn_up.weight", layer.tensor(layer.up()));
        }
    }

    @Test
    public void onlyTheSplitTensorsAndLayersLookup() {
        FfnRowSplitPlan plan = FfnRowSplitPlan.of(spec, 2, uploader);
        for (String suffix : new String[]{"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
            assertTrue(suffix, plan.splits("blk.2." + suffix));
            assertNotNull(suffix, plan.layerOf("blk.2." + suffix));
        }
        assertFalse("attention stays whole", plan.splits("blk.2.attn_q.weight"));
        assertFalse("the output head stays whole", plan.splits("output.weight"));
        assertFalse(plan.splits("blk.9.ffn_up.weight"));
        assertNull(plan.layerOf("blk.2.attn_q.weight"));
        LayerSpec l1 = spec.layers().get(1);
        assertNotNull(plan.layerOf(l1));
        assertEquals(1, plan.layerOf(l1).weightLayer());
    }

    @Test
    public void anUndividedHiddenWidthIsRefusedWithTheReason() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> FfnRowSplitPlan.of(spec, 3, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("every llama layer is ineligible"));
        assertTrue(e.getMessage(), e.getMessage().contains("n_ff 128"));
        assertTrue(e.getMessage(), e.getMessage().contains("3 rank shard(s)"));
    }

    @Test
    public void fewerThanTwoRanksAreRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> FfnRowSplitPlan.of(spec, 1, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("at least two ranks"));
        assertThrows(NullPointerException.class, () -> FfnRowSplitPlan.of(null, 2, uploader));
        assertThrows(NullPointerException.class, () -> FfnRowSplitPlan.of(spec, 2, null));
    }

    @Test
    public void aRoutedMoeModelKeepsNoLayerAndIsRefused() throws IOException {
        Path moe = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).nEmbd(64)
                .nHead(4).nHeadKv(2).headDim(16).nRot(16).nExpert(4).nExpertUsed(2).nFfExp(32)
                .nCtxTrain(256).build()).writeTempFile();
        try (GGUFFileParser moeParser = new GGUFFileParser(moe.toString());
             WeightStore moeStore = moeParser.createMappedStore(com.eonmux.neuralfree.gguf.StoreUse.RESIDENT_UPLOAD);
             FakeComputeBackend moeBackend = FakeComputeBackend.create()) {
            ModelSpec moeSpec = ModelSpecBuilder.build(GgufKv.of(moeParser), TensorCatalog.of(moeParser),
                    MemoryConfig.defaults());
            GpuWeightUploader moeUploader = new GpuWeightUploader(moeBackend, moeBackend.computeQueue(),
                    moeParser.getTensors(), moeStore, "qwen3moe",
                    GpuWeightUploader.config((type, use) -> true).build());
            try {
                UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                        () -> FfnRowSplitPlan.of(moeSpec, 2, moeUploader));
                assertTrue(e.getMessage(), e.getMessage().contains("every qwen3moe layer is ineligible"));
                assertTrue(e.getMessage(), e.getMessage().contains("no hidden-dimension split"));
            } finally {
                moeUploader.close();
            }
        } finally {
            Files.deleteIfExists(moe);
        }
    }

    @Test
    public void theTensorSplitSharesSizeTheRankShards() {
        FfnRowSplitPlan plan = FfnRowSplitPlan.of(spec, 2, uploader, new double[]{3, 1});
        for (FfnRowSplitPlan.Layer layer : plan.layers()) {
            assertEquals(0, layer.rowStart(0));
            assertEquals(96, layer.shard(0));
            assertEquals(96, layer.rowStart(1));
            assertEquals(32, layer.shard(1));
            assertEquals(96, layer.maxShard());
            assertFalse("3:1 is not even", layer.even());
        }
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> FfnRowSplitPlan.of(spec, 2, uploader, new double[]{1, 1, 1}));
        assertTrue(e.getMessage(), e.getMessage().contains("3 share(s)"));
        assertTrue(e.getMessage(), e.getMessage().contains("2 rank(s)"));
    }

    @Test
    public void aShareThatLeavesARankEmptyIsRefusedWithTheReason() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> FfnRowSplitPlan.of(spec, 2, uploader, new double[]{0.001, 1000}));
        assertTrue(e.getMessage(), e.getMessage().contains("tensor split shares leave a rank without"));
    }

    @Test
    public void theLayersKeepModelOrder() {
        List<FfnRowSplitPlan.Layer> layers = FfnRowSplitPlan.of(spec, 2, uploader).layers();
        for (int i = 1; i < layers.size(); i++) {
            assertTrue(layers.get(i - 1).il() < layers.get(i).il());
        }
    }
}

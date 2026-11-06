package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
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
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The attention split plan of a tensor-parallel load over synthetic models: a llama's GQA attention divides its query
 * and KV heads over the ranks with the matching output columns, a head count that does not divide is refused by name,
 * an MLA model is refused by name, and the tensor and layer lookups cover only the split attention.
 */
public class TpAttentionPlanTest {

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        backend = FakeComputeBackend.create();
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
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
    public void everyAttentionLayerDividesItsHeadsOverTheRanks() {
        TpAttentionPlan plan = TpAttentionPlan.of(spec, 2, uploader);
        assertEquals(2, plan.ranks());
        assertEquals(2, plan.layers().size());
        for (int il = 0; il < 2; il++) {
            TpAttentionPlan.Layer layer = plan.layers().get(il);
            assertEquals(il, layer.il());
            assertEquals(4, layer.nHead());
            assertEquals(2, layer.nHeadKv());
            assertEquals(2, layer.shardHeads());
            assertEquals(1, layer.shardKvHeads());
            assertEquals(64, layer.qProjWidth());
            assertEquals(32, layer.kDim());
            assertEquals(32, layer.vDim());
            assertEquals(64, layer.outDim());
            assertEquals(32, layer.outColumns());
            assertEquals(0, layer.outColumnStart(0));
            assertEquals(32, layer.outColumnStart(1));
            assertFalse(layer.fused());
            assertNotNull(layer.shardOf("attn_q.weight"));
            assertNotNull(layer.shardOf("attn_k.weight"));
            assertNotNull(layer.shardOf("attn_v.weight"));
            assertNotNull(layer.shardOf("attn_output.weight"));
            assertEquals(TpAttentionPlan.Shard.OUT_COLUMNS, layer.shardOf("attn_output.weight"));
            assertEquals("blk." + il + ".attn_q.weight", layer.tensor("attn_q.weight"));
            assertEquals(2, layer.shardSpec(spec.layers().get(il).attention(), 1).nHead());
            assertEquals(1, layer.shardSpec(spec.layers().get(il).attention(), 1).nHeadKv());
        }
    }

    @Test
    public void onlyTheAttentionTensorsAndLayersLookup() {
        TpAttentionPlan plan = TpAttentionPlan.of(spec, 2, uploader);
        assertTrue(plan.splits("blk.1.attn_q.weight"));
        assertTrue(plan.splits("blk.1.attn_output.weight"));
        assertFalse("the feed-forward is the other half of the split", plan.splits("blk.1.ffn_up.weight"));
        assertFalse(plan.splits("output.weight"));
        assertNull(plan.layerOf("blk.1.ffn_up.weight"));
        LayerSpec layer = spec.layers().get(1);
        assertNotNull(plan.layerOf(layer));
        assertTrue(plan.splits(layer));
    }

    @Test
    public void anUndividedHeadCountIsRefusedByName() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> TpAttentionPlan.of(spec, 3, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("n_head 4 does not divide into 3 rank(s)"));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0"));
    }

    @Test
    public void aGqaHeadCountBelowTheRankCountIsRefusedByName() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> TpAttentionPlan.of(spec, 4, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("n_head_kv 2 does not divide into 4 rank(s)"));
        assertTrue(e.getMessage(), e.getMessage().contains("GQA/MQA has fewer KV heads than ranks"));
    }

    @Test
    public void anMlaModelIsRefusedByName() throws IOException {
        Path mla = SyntheticModels.mistralLarge4(
                SyntheticModels.Config.mistralLarge4().toBuilder().nLayer(2).build()).writeTempFile();
        try (GGUFFileParser mlaParser = new GGUFFileParser(mla.toString());
             WeightStore mlaStore = mlaParser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
             FakeComputeBackend mlaBackend = FakeComputeBackend.create()) {
            ModelSpec mlaSpec = ModelSpecBuilder.build(GgufKv.of(mlaParser), TensorCatalog.of(mlaParser),
                    MemoryConfig.defaults());
            GpuWeightUploader mlaUploader = new GpuWeightUploader(mlaBackend, mlaBackend.computeQueue(),
                    mlaParser.getTensors(), mlaStore, "deepseek2",
                    GpuWeightUploader.config((type, use) -> true).build());
            try {
                UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                        () -> TpAttentionPlan.of(mlaSpec, 2, mlaUploader));
                assertTrue(e.getMessage(), e.getMessage().contains("MLA"));
            } finally {
                mlaUploader.close();
            }
        } finally {
            Files.deleteIfExists(mla);
        }
    }

    @Test
    public void fewerThanTwoRanksAreRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TpAttentionPlan.of(spec, 1, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("at least two ranks"));
        assertThrows(NullPointerException.class, () -> TpAttentionPlan.of(null, 2, uploader));
        assertThrows(NullPointerException.class, () -> TpAttentionPlan.of(spec, 2, null));
    }

    @Test
    public void theLayersKeepModelOrder() {
        List<TpAttentionPlan.Layer> layers = TpAttentionPlan.of(spec, 2, uploader).layers();
        for (int i = 1; i < layers.size(); i++) {
            assertTrue(layers.get(i - 1).il() < layers.get(i).il());
        }
    }

    @Test
    public void aContextPlanKeepsEveryHeadAndUploadsEveryTensorWhole() {
        TpAttentionPlan plan = TpAttentionPlan.context(spec, 3, uploader);
        assertTrue(plan.contextParallel());
        assertFalse(TpAttentionPlan.of(spec, 2, uploader).contextParallel());
        assertEquals(2, plan.layers().size());
        for (TpAttentionPlan.Layer layer : plan.layers()) {
            assertEquals(4, layer.nHead());
            assertEquals(2, layer.nHeadKv());
            assertEquals("every rank keeps every head", 4, layer.shardHeads());
            assertEquals(2, layer.shardKvHeads());
            assertEquals(4, layer.shardSpec(spec.layers().get(layer.il()).attention(), 1).nHead());
            for (String suffix : layer.tensors().keySet()) {
                assertEquals("context uploads the whole tensor", TpAttentionPlan.Shard.WHOLE,
                        layer.shardOf(suffix));
            }
            assertNotNull(layer.shardOf("attn_q.weight"));
            assertNotNull(layer.shardOf("attn_output.weight"));
        }
    }

    @Test
    public void aContextPlanDividesThePositionsWithoutAHeadDivisibilityCheck() {
        TpAttentionPlan plan = TpAttentionPlan.context(spec, 3, uploader);
        assertEquals(0, plan.contextStart(0, 10));
        assertEquals(3, plan.contextEnd(0, 10));
        assertEquals(3, plan.contextStart(1, 10));
        assertEquals(6, plan.contextEnd(1, 10));
        assertEquals(6, plan.contextStart(2, 10));
        assertEquals(10, plan.contextEnd(2, 10));
        assertEquals(0, plan.contextStart(0, 8));
        assertEquals(2, plan.contextEnd(0, 8));
        assertEquals(8, plan.contextEnd(2, 8));
        assertThrows(IllegalArgumentException.class, () -> plan.contextStart(3, 10));
        assertThrows(IllegalStateException.class,
                () -> TpAttentionPlan.of(spec, 2, uploader).contextStart(0, 10));
    }

    @Test
    public void aContextPlanRecordsAnIneligibleAttentionLayerAndKeepsTheRest() {
        ModelSpec withSinks = withSinkTokenOn(spec, 1);
        TpAttentionPlan plan = TpAttentionPlan.context(withSinks, 2, uploader);
        assertEquals("the eligible layer keeps its context split", 1, plan.layers().size());
        assertEquals(0, plan.layers().get(0).il());
        assertNull(plan.layerOf(withSinks.layers().get(1)));
        assertEquals(1, plan.skipped().size());
        String skipped = plan.skipped().get(0);
        assertTrue(skipped, skipped.startsWith("blk.1"));
        assertTrue(skipped, skipped.contains("sinks"));
    }

    @Test
    public void aTensorPlanRecordsNothingSkipped() {
        assertTrue(TpAttentionPlan.of(spec, 2, uploader).skipped().isEmpty());
        assertTrue(TpAttentionPlan.context(spec, 2, uploader).skipped().isEmpty());
    }

    @Test
    public void aContextPlanRefusesInvalidArguments() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TpAttentionPlan.context(spec, 1, uploader));
        assertTrue(e.getMessage(), e.getMessage().contains("context split"));
        assertTrue(e.getMessage(), e.getMessage().contains("at least two ranks"));
        assertThrows(NullPointerException.class, () -> TpAttentionPlan.context(null, 2, uploader));
        assertThrows(NullPointerException.class, () -> TpAttentionPlan.context(spec, 2, null));
    }

    /** A copy of {@code spec} whose layer {@code il} declares a learned sink token. */
    private static ModelSpec withSinkTokenOn(ModelSpec of, int il) {
        ModelSpec.Builder builder = ModelSpec.builder(of.hparams(), of.tensors())
                .schedule(of.schedule()).embedding(of.embedding()).head(of.head());
        for (LayerSpec layer : of.layers()) {
            builder.layer(layer.il() == il
                    ? LayerSpec.builder(layer)
                            .attention(layer.attention().toBuilder().sinkTokens(1).build()).build()
                    : layer);
        }
        return builder.build();
    }
}

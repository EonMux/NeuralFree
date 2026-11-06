package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The memory plan of the synthetic mistral-large4 (layer 1 reads layer 0's latent cache through
 * {@code attention.kv_source_map}, both layers carry a full lightning indexer): as in llama.cpp, KV sharing covers
 * the MLA cache only and every indexer owns its keys (llama-kv-cache-dsa.cpp:44-51), so layer 1 plans as SHARED_KV
 * with an indexer cache of its own, on the contiguous and the paged plan, and the CPU memory gives it its own indexer
 * rows. The forward mixers run the model: the direct {@code attn_q} path, SSMax, the layer-normed indexer queries and
 * the LeRoPE factors are part of the DSA mixers.
 */
public class MistralLarge4PlanTest {

    private static ModelSpec spec(Path path) throws IOException {
        GGUFFileParser parser = new GGUFFileParser(path.toString());
        try {
            return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        } finally {
            parser.close();
        }
    }

    private static void assertSharedLayerOwnsItsIndexer(MemoryPlan plan) {
        LayerMemoryPlan owner = plan.layer(0);
        LayerMemoryPlan shared = plan.layer(1);
        assertEquals(LayerMemoryPlan.Kind.KV, owner.kind());
        assertEquals(LayerMemoryPlan.Kind.SHARED_KV, shared.kind());
        assertEquals(0, shared.sharedFrom().layer());
        assertNotNull("the owner keeps its indexer keys", owner.indexer());
        assertNotNull("the KV-sharing layer keeps indexer keys of its own", shared.indexer());
        assertEquals(owner.kv().rows(), shared.indexer().rows());
        assertEquals(owner.indexer().width(), shared.indexer().width());
        assertTrue("the shared layer's indexer bytes count in the plan", shared.sideCacheBytes() > 0);
    }

    @Test
    public void aKvSharingLayerPlansItsOwnIndexerCache() throws IOException {
        Path path = SyntheticModels.mistralLarge4().writeTempFile();
        try {
            ModelSpec spec = spec(path);
            MemoryConfig f32 = MemoryConfig.builder().typeK(KvDType.F32).typeV(KvDType.F32).build();
            MemoryPlan contiguous = MemoryPlanner.planForContext(spec, f32, 64);
            assertSharedLayerOwnsItsIndexer(contiguous);
            MemoryPlan paged = MemoryPlanner.planForContext(spec, MemoryConfig.builder().typeK(KvDType.F32)
                    .typeV(KvDType.F32).ubatch(8).parallel(2).build(), 64);
            assertSharedLayerOwnsItsIndexer(paged);

            try (HeapModelMemory memory = HeapModelMemory.allocate(contiguous, spec, null)) {
                assertSame("layer 1 reads layer 0's latent rows", memory.kv(0), memory.kv(1));
                assertNotNull(memory.indexer(1));
                assertNotSame("layer 1 writes its own indexer keys", memory.indexer(0), memory.indexer(1));
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void aNoPeLayerRotatesNoIndexerAndARopedLayerKeepsTheLearnedFactors() throws IOException {
        Path path = SyntheticModels.mistralLarge4NoPe(SyntheticModels.Config.mistralLarge4()).writeTempFile();
        try {
            ModelSpec spec = spec(path);
            MemoryPlan plan = MemoryPlanner.planForContext(spec, MemoryConfig.builder().typeK(KvDType.F32)
                    .typeV(KvDType.F32).build(), 64);
            SparseBlocks.IndexerConfig roped = SparseBlocks.IndexerConfig.forLayer(spec.layer(0),
                    plan.layer(0).indexer().rotation(), KvWindow.FULL_CAUSAL, spec.hparams().normEps());
            assertNotNull("the roped layer rotates the indexer q/k", roped.rope());
            assertSame("the indexer rotation reads the LeRoPE factors", spec.layer(0).rope().freqFactors(),
                    roped.rope().freqFactors());
            SparseBlocks.IndexerConfig noPe = SparseBlocks.IndexerConfig.forLayer(spec.layer(1),
                    plan.layer(1).indexer().rotation(), KvWindow.FULL_CAUSAL, spec.hparams().normEps());
            assertNull("a NoPE layer rotates no indexer q/k", noPe.rope());
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void theDsaMixersRunTheModel() throws IOException {
        Path path = SyntheticModels.mistralLarge4().writeTempFile();
        try (GGUFModel model = new GGUFModel(path.toString(), EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(64).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build())) {
            float[] logits = model.forwardBackend().prefill(new int[]{2, 3, 5, 7}, 0);
            assertEquals(model.vocabSize(), logits.length);
            for (float v : logits) {
                assertTrue("non-finite logit " + v, Float.isFinite(v));
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }
}

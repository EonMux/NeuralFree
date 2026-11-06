package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
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
import java.util.Collections;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The selective streaming tier ({@code neuralfree.stream.params}, vLLM's prefetch {@code offload_params}) of
 * {@link GpuLoadPlacement} over the synthetic qwen3next: a parameter segment restricts the streamed sets to the
 * matching tensors of the streamed layers while the rest stay resident, and a segment that matches no device
 * tensor is refused by the property it came from.
 */
public class SelectiveStreamPlacementTest {

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.qwen3next().writeTempFile();
        parser = new GGUFFileParser(file.toString());
        backend = FakeComputeBackend.create();
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        TensorCatalog catalog = TensorCatalog.of(parser);
        GgufKv kv = GgufKv.of(parser);
        spec = ModelSpecBuilder.build(kv, catalog, MemoryConfig.defaults());
        InputTransformLoader.Result transforms = InputTransformLoader.load(parser, kv, catalog,
                EngineConfig.defaults().activationQuant());
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "qwen3next",
                GpuWeightUploader.config((type, use) -> true).transforms(transforms).build());
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

    /** A placement over the fixture whose weights cannot stay resident beside the training context. */
    private GpuLoadPlacement placement(long usable) {
        return new GpuLoadPlacement(parser, spec.tensors(), null, uploader, backend, ContextFitter.Companion.NONE,
                () -> usable, false, name -> false);
    }

    private long usableForcingStreaming() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        long weightBytes = uploader.plannedBytes(weights::onDevice);
        long contextBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), spec.hparams().nCtxTrain());
        return weightBytes + contextBytes - 1;
    }

    @Test
    public void aStreamParameterRestrictsTheStreamedSets() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .streamParams(Collections.singletonList("ffn_gate_exps")).build();
        WeightPlacement weights = WeightPlacement.of(spec, false);

        GpuLoadPlacement.Placement p = placement(usableForcingStreaming())
                .place(spec, MemoryConfig.defaults(), cfg, weights, null);

        assertFalse(p.streamed.isEmpty());
        assertSame(weights, p.weights);
        assertTrue(p.nCtx > 0);
        for (LayerWeightSet set : p.streamed) {
            for (String name : set.names()) {
                assertTrue(name, name.endsWith("ffn_gate_exps.weight"));
            }
        }
        int tensors = 0;
        for (LayerWeightSet set : p.streamed) {
            tensors += set.names().size();
        }
        assertTrue("one ffn_gate_exps tensor per streamed layer", tensors >= 1);
    }

    @Test
    public void aStreamSelectionKeepsTheAutoFitOnTheStreamingTier() {
        EngineConfig cfg = EngineConfig.builder().streamParams(Collections.singletonList("ffn_gate_exps")).build();
        WeightPlacement weights = WeightPlacement.of(spec, false);

        GpuLoadPlacement.Placement p = placement(usableForcingStreaming())
                .place(spec, MemoryConfig.defaults(), cfg, weights, null);

        assertFalse("the selection asks for streaming, so the fit does not split whole layers to the host",
                p.streamed.isEmpty());
        assertSame(weights, p.weights);
        for (LayerWeightSet set : p.streamed) {
            for (String name : set.names()) {
                assertTrue(name, name.endsWith("ffn_gate_exps.weight"));
            }
        }
    }

    @Test
    public void aStreamSelectionIsRefusedBesideAnExplicitHybridOffload() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(1)
                .streamParams(Collections.singletonList("ffn_gate_exps")).build();
        WeightPlacement weights = WeightPlacement.of(spec, false);
        LayerOffload requested = LayerOffload.requested(spec, cfg, parser.getTensors());

        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> placement(usableForcingStreaming())
                        .place(spec, MemoryConfig.defaults(), cfg, weights, requested));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_STREAM_PARAMS));
    }

    @Test
    public void aSegmentThatMatchesNoDeviceTensorIsRefusedByName() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .streamParams(Collections.singletonList("not_a_tensor")).build();

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> placement(usableForcingStreaming()).place(spec, MemoryConfig.defaults(), cfg,
                        WeightPlacement.of(spec, false), null));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_STREAM_PARAMS));
        assertTrue(e.getMessage(), e.getMessage().contains("not_a_tensor"));
    }
}

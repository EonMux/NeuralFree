package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.HostMappedPlan;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The placement fit of a GPU load over a planned (never uploaded) 4-layer synthetic llama: every weight stays resident
 * at the training context when the VRAM holds it; a requested {@code -ngl} split is honored with its host layers off
 * the device; an automatic {@code -ngl} offloads a suffix of trunk layers when the weights do not fit; with every
 * layer kept on the device the trunk streams instead; and weights that cannot fit even streamed are refused. With the
 * MoE expert cache configured, weights that fit stay resident, while weights that do not are refused by name instead of
 * offloading or streaming trunk layers.
 */
public class GpuLoadPlacementTest {

    private static final int N_LAYER = 4;

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;
    private TensorCatalog catalog;
    private InputTransformLoader.Result transforms;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYER).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        backend = FakeComputeBackend.create();
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        catalog = TensorCatalog.of(parser);
        GgufKv kv = GgufKv.of(parser);
        spec = ModelSpecBuilder.build(kv, catalog, MemoryConfig.defaults());
        transforms = InputTransformLoader.load(parser, kv, catalog, EngineConfig.defaults().activationQuant());
        uploader = new GpuWeightUploader(backend, backend.computeQueue(), parser.getTensors(), store, "llama",
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

    @Test
    public void everyWeightStaysResidentAtTheTrainingContextWhenTheVramHoldsIt() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        GpuLoadPlacement.Placement p = placement(1L << 30).place(spec, MemoryConfig.defaults(),
                EngineConfig.defaults(), weights, null);

        assertEquals(spec.hparams().nCtxTrain(), p.nCtx);
        assertTrue(p.streamed.isEmpty());
        assertNull(p.split);
        assertSame(weights, p.weights);
    }

    @Test
    public void aRequestedSplitIsHonoredWithItsHostLayersOffTheDevice() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(3).build();
        LayerOffload requested = LayerOffload.requested(spec, cfg, parser.getTensors());
        WeightPlacement weights = WeightPlacement.of(spec, false);
        GpuLoadPlacement.Placement p = placement(1L << 30).place(spec, MemoryConfig.defaults(), cfg, weights,
                requested);

        assertSame(requested, p.split);
        assertTrue(p.streamed.isEmpty());
        assertNotSame(weights, p.weights);
        assertFalse(p.weights.onDevice("blk.0.attn_q.weight"));
        assertTrue(p.weights.onDevice("blk." + (N_LAYER - 1) + ".attn_q.weight"));
    }

    @Test
    public void anAutomaticSplitOffloadsASuffixOfTrunkLayersWhenTheWeightsDoNotFit() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        long weightBytes = uploader.plannedBytes(weights::onDevice);
        GpuLoadPlacement.Placement p = placement(weightBytes * 3 / 4).place(spec, MemoryConfig.defaults(),
                EngineConfig.defaults(), weights, null);

        assertTrue(p.split != null);
        assertTrue(p.split.nGpuLayers() + " layers offloaded", p.split.nGpuLayers() >= 1
                && p.split.nGpuLayers() < N_LAYER);
        assertTrue(p.streamed.isEmpty());
        assertFalse(p.weights.onDevice("blk.0.attn_q.weight"));
        assertTrue(p.nCtx > 0);
    }

    @Test
    public void theTrunkStreamsWhenEveryLayerStaysOnTheDevice() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        long weightBytes = uploader.plannedBytes(weights::onDevice);
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL).build();
        // one byte short of the resident weights beside the training context, which is the context floor here
        long contextBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), spec.hparams().nCtxTrain());
        GpuLoadPlacement.Placement p = placement(weightBytes + contextBytes - 1).place(spec,
                MemoryConfig.defaults(), cfg, weights, null);

        assertNull(p.split);
        assertSame(weights, p.weights);
        assertFalse(p.streamed.isEmpty());
        for (LayerWeightSet s : p.streamed) {
            assertTrue(s.weightLayer() >= 0 && s.weightLayer() < N_LAYER);
        }
        assertTrue(p.nCtx > 0);
    }

    @Test
    public void weightsThatDoNotFitEvenStreamedAreRefused() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL).build();
        try {
            placement(1024L).place(spec, MemoryConfig.defaults(), cfg, WeightPlacement.of(spec, false), null);
            fail("1 KiB of VRAM cannot hold the model");
        } catch (GpuOutOfMemoryException | ContextFitter.ContextBudgetException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("llama"));
        }
    }

    @Test
    public void theMoeCacheKeepsAFittingLoadResident() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        GpuLoadPlacement.Placement p = placement(1L << 30, true).place(spec, MemoryConfig.defaults(),
                EngineConfig.defaults(), weights, null);

        assertEquals(spec.hparams().nCtxTrain(), p.nCtx);
        assertTrue(p.streamed.isEmpty());
        assertNull(p.split);
        assertSame(weights, p.weights);
    }

    @Test
    public void theMoeCacheRefusesToOffloadOrStreamWeightsThatDoNotFit() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        long weightBytes = uploader.plannedBytes(weights::onDevice);
        // without the cache this budget offloads a suffix of trunk layers (see the automatic split test above)
        try {
            placement(weightBytes * 3 / 4, true).place(spec, MemoryConfig.defaults(), EngineConfig.defaults(),
                    weights, null);
            fail("the MoE expert cache never offloads nor streams trunk layers");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE));
            assertTrue(expected.getMessage(), expected.getMessage().contains("llama"));
        }
    }

    @Test
    public void theHostMappedTierLeavesTheDeviceFit() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        long weightBytes = uploader.plannedBytes(weights::onDevice);
        HostMappedPlan plan = HostMappedPlan.plan(EngineConfig.builder().cpuOffloadMb(1).build(), spec,
                LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name())).build(),
                weights::onDeviceName);
        assertFalse(plan.isEmpty());
        long contextBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), spec.hparams().nCtxTrain());

        GpuLoadPlacement.Placement p = placement(weightBytes - plan.bytes() + contextBytes, plan::contains)
                .place(spec, MemoryConfig.defaults(), EngineConfig.builder().cpuOffloadMb(1).build(), weights, null);

        assertNull(p.split);
        assertTrue(p.streamed.isEmpty());
        assertTrue(p.nCtx > 0);
    }

    @Test
    public void aContextFloorAndAHostMappedTierCombine() {
        WeightPlacement weights = WeightPlacement.of(spec, false);
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL).cpuOffloadMb(1)
                .cpuOffloadParams(Collections.singletonList("ffn_up")).build();
        HostMappedPlan plan = HostMappedPlan.plan(cfg, spec,
                LayerWeightSet.Options.builder(t -> uploader.deviceType(t.name())).build(),
                weights::onDeviceName);
        assertFalse(plan.isEmpty());
        long deviceWeightBytes = uploader.plannedBytes(t -> weights.onDevice(t) && !plan.contains(t.name()));
        // one byte short of the device weights beside the training context, so the floor forces streaming
        long usable = deviceWeightBytes
                + ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), spec.hparams().nCtxTrain()) - 1;

        GpuLoadPlacement.Placement p = new GpuLoadPlacement(parser, catalog, transforms, uploader, backend,
                ContextFitter.Companion.NONE, () -> usable, false, spec.hparams().nCtxTrain(), plan::contains)
                .place(spec, MemoryConfig.defaults(), cfg, weights, null);

        assertNull(p.split);
        assertFalse(p.streamed.isEmpty());
        for (LayerWeightSet s : p.streamed) {
            for (String name : s.names()) {
                assertFalse(name + " is host-mapped and must not stream", plan.contains(name));
            }
        }
        assertTrue(p.nCtx > 0);
    }

    @Test
    public void theHostMappedTierIsRefusedBesideALayerSplit() {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(3).cpuOffloadMb(8).build();
        LayerOffload requested = LayerOffload.requested(spec, cfg, parser.getTensors());

        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> placement(1L << 30).place(spec, MemoryConfig.defaults(), cfg, WeightPlacement.of(spec, false),
                        requested));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_MB));
    }

    private GpuLoadPlacement placement(long usable) {
        return placement(usable, false);
    }

    private GpuLoadPlacement placement(long usable, java.util.function.Predicate<String> offloaded) {
        return new GpuLoadPlacement(parser, catalog, transforms, uploader, backend, ContextFitter.Companion.NONE,
                () -> usable, false, offloaded);
    }

    private GpuLoadPlacement placement(long usable, boolean moeCache) {
        return new GpuLoadPlacement(parser, catalog, transforms, uploader, backend, ContextFitter.Companion.NONE,
                () -> usable, moeCache, name -> false);
    }
}

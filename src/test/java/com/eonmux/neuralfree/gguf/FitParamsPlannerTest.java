package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The parameter fit over a planned (never uploaded) synthetic model: weights that hold beside the training
 * context fit as resident ({@code -ngl all}); weights that do not leave the device over a hybrid split whose
 * {@code -ngl} count the arguments repeat; with every layer kept on the device the trunk streams and the
 * arguments still say all layers on the device; a user context is kept; and {@code --fit-ctx} lets the fit
 * reduce the context instead of streaming the weights. A margin the free VRAM cannot absorb refuses the fit
 * by name, and the expert override line of a MoE overflow names the overflowing layers exactly as {@code
 * --n-cpu-moe} derives them.
 */
public class FitParamsPlannerTest {

    private static final int N_LAYER = 4;

    private Path file;
    private GGUFFileParser parser;
    private FakeComputeBackend backend;
    private WeightStore store;
    private GpuWeightUploader uploader;
    private ModelSpec spec;
    private TensorCatalog catalog;
    private InputTransformLoader.Result transforms;
    private long weightBytes;
    private int train;

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
        weightBytes = uploader.plannedBytes(WeightPlacement.of(spec, false)::onDevice);
        train = spec.hparams().nCtxTrain();
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
    public void weightsThatHoldBesideTheTrainingContextFitResident() throws IOException {
        long ctxBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), train);
        FitParams fit = plan(weightBytes + ctxBytes + 1024, null, 0);

        assertEquals(FitParams.Mode.RESIDENT, fit.mode());
        assertEquals(train, fit.nCtx());
        assertEquals("-c " + train + " -ngl all", fit.arguments());
    }

    @Test
    public void weightsThatDoNotHoldLeaveTheDeviceOverASplitTheArgumentsRepeat() throws IOException {
        FitParams fit = plan(weightBytes * 3 / 4, null, 0);

        assertEquals(FitParams.Mode.HYBRID, fit.mode());
        assertTrue(fit.gpuLayers() >= 1 && fit.gpuLayers() < N_LAYER);
        assertEquals("-c " + train + " -ngl " + (fit.gpuLayers() + 1), fit.arguments());
        assertTrue(fit.hostWeightBytes() > 0);
    }

    @Test
    public void aHostKvSplitRepeatsTheNoKvOffloadFlag() throws IOException {
        EngineConfig cfg = EngineConfig.builder().kvOffload(false).build();
        long ctxBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), train);
        FitParams fit = plan(weightBytes + ctxBytes + 1024, cfg, 0);

        assertEquals(FitParams.Mode.HYBRID, fit.mode());
        assertEquals("-c " + train + " -ngl " + (N_LAYER + 1) + " -nkvo", fit.arguments());
    }

    @Test
    public void theStreamingLadderPrintsAllLayersOnTheDevice() throws IOException {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL).build();
        long ctxBytes = ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), train);
        FitParams fit = plan(weightBytes + ctxBytes - 1, cfg, 0);

        assertTrue(fit.mode() + " streams", fit.mode() == FitParams.Mode.PARTIAL
                || fit.mode() == FitParams.Mode.STREAMED);
        assertTrue(fit.streamedLayers() >= 1);
        assertEquals("-c " + train + " -ngl all", fit.arguments());
    }

    /**
     * {@code --fit-ctx} lowers the floor of the automatic context: with a budget that holds the weights beside
     * a context of 512 but not beside the 2048-token training context, the fit with a 512 floor keeps every
     * weight on the device, while the default floor (the training context) streams the trunk instead.
     */
    @Test
    public void aFitContextFloorKeepsTheWeightsInsteadOfStreamingThem() throws IOException {
        EngineConfig cfg = EngineConfig.builder().gpuLayers(EngineConfig.GPU_LAYERS_ALL).build();
        MemoryConfig mem = cfg.memory();
        Path file2048 = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nCtxTrain(2048).build()).writeTempFile();
        try (GGUFFileParser parser2048 = new GGUFFileParser(file2048.toString())) {
            ModelSpec spec2048 = ModelSpecBuilder.build(GgufKv.of(parser2048), TensorCatalog.of(parser2048),
                    MemoryConfig.defaults());
            try (WeightStore store2048 = parser2048.createMappedStore(StoreUse.RESIDENT_UPLOAD);
                 GpuWeightUploader uploader2048 = new GpuWeightUploader(backend, backend.computeQueue(),
                         parser2048.getTensors(), store2048, "llama",
                         GpuWeightUploader.config((type, use) -> true).transforms(transforms).build())) {
                long weights = uploader2048.plannedBytes(WeightPlacement.of(spec2048, false)::onDevice);
                long usable = weights + ContextFitter.requiredBytes(spec2048, mem, 512);
                FitParams floored = new FitParamsPlanner.Builder(parser2048, backend, cfg)
                        .uploader(uploader2048)
                        .drafter(ContextFitter.Companion.NONE)
                        .usableVram(() -> usable)
                        .contextFloor(512)
                        .plan();
                assertEquals(FitParams.Mode.RESIDENT, floored.mode());
                assertEquals(512, floored.nCtx());
                assertEquals("-c 512 -ngl all", floored.arguments());
                // the default floor is the training context, which the budget cannot hold
                FitParams unfloored = new FitParamsPlanner.Builder(parser2048, backend, cfg)
                        .uploader(uploader2048)
                        .drafter(ContextFitter.Companion.NONE)
                        .usableVram(() -> usable)
                        .plan();
                assertEquals(FitParams.Mode.STREAMED, unfloored.mode());
                assertTrue(unfloored.streamedLayers() >= 1);
                assertTrue(unfloored.nCtx() >= 512);
                assertTrue(unfloored.arguments(), unfloored.arguments().endsWith("-ngl all"));
            }
        } finally {
            Files.deleteIfExists(file2048);
        }
    }

    @Test
    public void aUserContextIsKeptInsteadOfFitted() throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .memory(MemoryConfig.defaults().toBuilder().nCtx(128).build()).build();
        long ctxBytes = ContextFitter.requiredBytes(spec, cfg.memory(), 128);
        FitParams fit = plan(weightBytes + ctxBytes + 1024, cfg, 0);

        assertEquals(FitParams.Mode.RESIDENT, fit.mode());
        assertEquals(128, fit.nCtx());
        assertEquals("-c 128 -ngl all", fit.arguments());
    }

    @Test
    public void aMarginTheFreeVramCannotAbsorbRefusesTheFit() throws IOException {
        try {
            // no injected budget: the planner measures the free VRAM of the fake device, then applies the margin
            new FitParamsPlanner.Builder(parser, backend, EngineConfig.defaults())
                    .uploader(uploader)
                    .drafter(ContextFitter.Companion.NONE)
                    .marginBytes(Long.MAX_VALUE)
                    .plan();
            fail("a margin above the free VRAM leaves nothing to fit into");
        } catch (GpuOutOfMemoryException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("llama"));
        }
    }

    @Test
    public void theExpertOverridesNameTheOverflowingLayers() {
        assertEquals("-ot 'blk\\." + (N_LAYER - 1) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps=cpu'",
                FitParams.expertOverrides(N_LAYER - 1, N_LAYER));
        assertEquals("-ot 'blk\\.1\\.ffn_(up|down|gate|gate_up)_(ch|)exps=cpu,"
                + "blk\\.2\\.ffn_(up|down|gate|gate_up)_(ch|)exps=cpu'", FitParams.expertOverrides(1, 3));
        assertEquals("", FitParams.expertOverrides(N_LAYER, N_LAYER));
    }

    /** Plans the open 4-layer llama against an injected budget, as the fit-params tool plans its model. */
    private FitParams plan(long usable, EngineConfig cfg, long marginBytes) throws IOException {
        return new FitParamsPlanner.Builder(parser, backend, cfg != null ? cfg : EngineConfig.defaults())
                .uploader(uploader)
                .drafter(ContextFitter.Companion.NONE)
                .usableVram(() -> usable)
                .marginBytes(marginBytes)
                .plan();
    }
}

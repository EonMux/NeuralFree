package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.testutil.LoraAdapterFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Per-request LoRA slots on the GPU: the load reserves {@code neuralfree.lora.max} worst-case slots of
 * {@code neuralfree.lora.max.rank} over the model's adaptable products in its VRAM fit ({@code neuralfree.lora.enable},
 * vLLM's {@code --enable-lora}), and refuses per-request loads without it by name; startup adapters are rescaled or
 * disabled per request, an activated LoRA applies from its invocation, and the hybrid layer offload
 * ({@code n_gpu_layers}) and the layer split over two devices, with and without host layers, serve per-request
 * adapters too — each matching the same placement loaded with that adapter set alone.
 */
public class GpuLoraSlotsTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float SCALE_A = 0.75f;
    private static final float SCALE_B = 0.5f;
    private static final int N_NEW = 5;
    private static final int[] PROMPT = {11, 12, 13, 77, 91, 14, 15, 16};
    private static final int[] INVOCATION = {77, 91};
    private static final int INVOCATION_START = 3;
    private static final float LOGITS_TOL = 1e-3f;

    @BeforeClass
    public static void requireGpu() {
        GpuTestUtil.assumeOpenCLAvailable();
    }

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(48).ubatch(4).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).memory(mc.build())
                .loraConfig(LoraConfig.builder().enabled(true).maxAdapters(2).maxRank(16).build());
    }

    private static GenerationRequest request(LoraSelection lora) {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding())
                .ignoreEos(true).lora(lora).build();
    }

    private static LoraSelection only(String name, float scale) {
        return LoraSelection.builder().add(name, scale).build();
    }

    /** The greedy tokens of {@code PROMPT} through the single-sequence loop of a load of {@code cfg}. */
    private static int[] reference(Path model, EngineConfig cfg) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            return GenerationLoop.forModel(m, cfg).generate(PROMPT, request(null), null).tokens();
        }
    }

    // ------------------------------------------------------------------ the reservation

    @Test
    public void theSlotReservationCoversMaxRankOnEveryAdaptableProduct() {
        List<long[]> shapes = Arrays.asList(new long[]{4, 8, 1}, new long[]{4, 8, 3});
        // dense: rank * (in + out) + one ubatch of zero rows; experts: rank * (in + out) per expert
        assertEquals(4L * ((2 * 12 + 5) + 2 * 12 * 3), LoraAttachment.slotReserveBytes(shapes, 2, 5));
        assertThrows(IllegalArgumentException.class, () -> LoraAttachment.slotReserveBytes(shapes, 0, 5));
        assertThrows(IllegalArgumentException.class,
                () -> LoraAttachment.slotReserveBytes(Arrays.asList(new long[]{4, 0, 1}), 2, 5));
    }

    @Test
    public void theReservationEntersTheVramFit() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig.Builder limited = config(1).gpuVramLimitMb(64);
        try (GGUFModelGPU fits = new GGUFModelGPU(model.toString(), null, limited.build())) {
            assertNotNull("two slots fit the 64 MiB budget", fits.loraRegistry());
        }
        // 4096 slots of rank 64 reserve about 2.8 GiB, past the 64 MiB the load may allocate
        EngineConfig crowded = limited.loraConfig(LoraConfig.builder().enabled(true).maxAdapters(4096)
                .maxRank(64).build()).build();
        GpuOutOfMemoryException e = assertThrows(GpuOutOfMemoryException.class,
                () -> new GGUFModelGPU(model.toString(), null, crowded).close());
        assertTrue(e.getMessage(), e.getMessage().contains("do not fit"));
        EngineConfig off = limited.loraConfig(LoraConfig.builder().enabled(false).maxAdapters(4096)
                .maxRank(64).build()).build();
        try (GGUFModelGPU fits = new GGUFModelGPU(model.toString(), null, off)) {
            assertNotNull("without the switch nothing is reserved", fits.loraRegistry());
        }
    }

    @Test
    public void perRequestLoadsWithoutTheSwitchAreRefusedByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        EngineConfig cfg = config(1).loraConfig(LoraConfig.builder().enabled(false).build()).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> m.loraRegistry().load("a", a.toString(), SCALE_A));
            assertTrue(e.getMessage(), e.getMessage().contains(LoraConfig.PROP_ENABLE));
            assertEquals(0, m.loraRegistry().list().size());
        }
    }

    // ------------------------------------------------------------------ startup adapters and aLoRA

    @Test
    public void requestsRescaleDisableOrReplaceTheStartupAdapterOnTheDevice() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        Path b = LoraAdapterFiles.llama(5000, RANK, ALPHA);
        int[] refA = reference(model, config(1).addLora(a.toString(), SCALE_A).build());
        int[] refBare = reference(model, config(1).build());
        int[] refQuarter = reference(model, config(1).addLora(a.toString(), 0.25f).build());
        int[] refB = reference(model, config(1).addLora(b.toString(), SCALE_B).build());
        assertFalse("the adapter must change the tokens", Arrays.equals(refA, refBare));

        EngineConfig cfg = config(4).addLora(a.toString(), SCALE_A).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            LoraRegistry registry = m.loraRegistry();
            String name = registry.list().get(0).name();
            registry.load("b", b.toString(), SCALE_B);
            BatchSequence sA = engine.submit(PROMPT, request(null), null);
            BatchSequence sBare = engine.submit(PROMPT, request(only(name, 0f)), null);
            BatchSequence sQuarter = engine.submit(PROMPT, request(only(name, 0.25f)), null);
            BatchSequence sB = engine.submit(PROMPT, request(LoraSelection.of("b")), null);
            engine.drain();
            assertTokens(refA, sA);
            assertTokens(refBare, sBare);
            assertTokens(refQuarter, sQuarter);
            assertTokens(refB, sB);
        }
    }

    /**
     * A startup adapter whose rank exceeds {@code neuralfree.lora.max.rank} rescales on the device: its slot runs
     * the resident slot-0 factors at the request's factor of the startup scale and uploads no copy, so the budget
     * that bounds a rank-16 slot copy does not refuse the request. A rescale of the rank-4 startup set below
     * exercises the same path at a rank the copy would have fit.
     */
    @Test
    public void aStartupAdapterAboveMaxRankRescalesOnTheDeviceWithoutACopy() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // alpha 64 over rank 32 keeps the effective scale in the rank-4 range (alpha / rank)
        Path high = LoraAdapterFiles.llama(7000, 32, 64.0f);
        int[] refHalf = reference(model, config(1).addLora(high.toString(), 0.5f).build());
        int[] refStartup = reference(model, config(1).addLora(high.toString(), SCALE_A).build());
        int[] refBare = reference(model, config(1).build());
        assertFalse("the rescale must change the tokens", Arrays.equals(refHalf, refBare));

        EngineConfig cfg = config(4).addLora(high.toString(), SCALE_A).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            String name = m.loraRegistry().list().get(0).name();
            assertTrue("the rescale key needs no slot budget", m.loraRegistry().isRescale(
                    m.loraRegistry().resolve(only(name, 0.5f))));
            BatchSequence s = engine.submit(PROMPT, request(only(name, 0.5f)), null);
            BatchSequence def = engine.submit(PROMPT, request(null), null);
            engine.drain();
            assertTokens(refHalf, s);
            assertTokens(refStartup, def);
        }
    }

    @Test
    public void anActivatedLoraAppliesFromItsInvocationOnTheDevice() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path alora = LoraAdapterFiles.activated(3000, RANK, ALPHA, INVOCATION);
        int[] refInvoked;
        int[] refAlways;
        EngineConfig refCfg = config(1).addLora(alora.toString(), SCALE_A).build();
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), null, refCfg)) {
            refInvoked = greedyFrom(one, INVOCATION_START);
            refAlways = greedyFrom(one, 0);
        }
        assertFalse("the invocation start must change the tokens", Arrays.equals(refInvoked, refAlways));

        EngineConfig cfg = config(2).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            m.loraRegistry().load("al", alora.toString(), SCALE_A);
            BatchSequence s = engine.submit(PROMPT, request(LoraSelection.of("al")), null);
            engine.drain();
            assertTokens(refInvoked, s);
        }
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), null, refCfg)) {
            int[] loop = GenerationLoop.forModel(one, refCfg).generate(PROMPT, request(null), null).tokens();
            assertPrefix(refInvoked, loop);
        }
    }

    /**
     * Greedy tokens of a load whose slot 0 holds the adapter: the prompt rows before {@code start} in the bare slot,
     * the rest and the decode in slot 0.
     */
    private static int[] greedyFrom(EngineModel m, int start) {
        LoraRegistry registry = m.loraRegistry();
        ForwardBackend f = m.forwardBackend();
        f.reset();
        if (start > 0) {
            registry.useSlot(registry.bareSlot());
            f.prefill(Arrays.copyOfRange(PROMPT, 0, start), 0);
        }
        registry.useSlot(0);
        float[] logits = f.prefill(Arrays.copyOfRange(PROMPT, start, PROMPT.length), start);
        int[] out = new int[N_NEW];
        for (int i = 0; i < N_NEW; i++) {
            out[i] = ForwardBackend.argmax(logits);
            if (i + 1 < N_NEW) {
                logits = f.decode(out[i], PROMPT.length + i);
            }
        }
        f.reset();
        return out;
    }

    // ------------------------------------------------------------------ hybrid and split placements

    @Test
    public void theHybridOffloadServesPerRequestAdaptersOnBothSides() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path a = adapterOf(4, 1000);
        assertPlacementServes(model, a, config(1).gpuLayers(2));
    }

    @Test
    public void theLayerSplitServesPerRequestAdaptersOnEveryStage() throws IOException {
        Assume.assumeTrue("the split needs two OpenCL devices", openClDeviceCount() >= 2);
        Path model = SyntheticModels.llama().writeTempFile();
        Path a = LoraAdapterFiles.llama(1000, RANK, ALPHA);
        try {
            assertPlacementServes(model, a, config(1).addGpuDevice("0").addGpuDevice("1")
                    .splitMode(EngineConfig.SplitMode.LAYER));
        } catch (IllegalArgumentException e) {
            // the second OpenCL device can vanish between the assume and the load; that is the machine, not the
            // adapters
            assumeSelection(e);
        }
    }

    @Test
    public void theLayerSplitWithHostLayersServesPerRequestAdaptersOnTheStagesAndTheHost() throws IOException {
        Assume.assumeTrue("the split needs two OpenCL devices", openClDeviceCount() >= 2);
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path a = adapterOf(4, 1000);
        try {
            assertPlacementServes(model, a, config(1).addGpuDevice("0").addGpuDevice("1")
                    .splitMode(EngineConfig.SplitMode.LAYER).gpuLayers(3));
        } catch (IllegalArgumentException e) {
            assumeSelection(e);
        }
    }

    /** Skips when the refusal is the selected devices vanishing under the load; rethrows anything else. */
    private static void assumeSelection(IllegalArgumentException e) {
        Assume.assumeTrue("the selected devices vanished under the load: " + e.getMessage(),
                e.getMessage() != null && e.getMessage().contains("names no device"));
        throw e;
    }

    /**
     * On the placement of {@code placed}: the logits of a pass in the per-request adapter's slot equal those of the
     * placement loaded with it at startup, a pass in the bare slot of a startup load equals the bare placement, and
     * the single-sequence loop generates the same tokens per request.
     */
    private static void assertPlacementServes(Path model, Path adapter, EngineConfig.Builder placed)
            throws IOException {
        EngineConfig startup = placed.build().toBuilder().addLora(adapter.toString(), SCALE_A).build();
        float[] refAdapted;
        float[] refBare;
        int[] tokensAdapted;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, startup)) {
            refAdapted = prefillLogits(m);
            tokensAdapted = GenerationLoop.forModel(m, startup).generate(PROMPT, request(null), null).tokens();
            LoraRegistry registry = m.loraRegistry();
            registry.useSlot(registry.bareSlot());
            float[] disabled = prefillLogits(m);
            registry.useSlot(0);
            try (GGUFModelGPU bare = new GGUFModelGPU(model.toString(), null, placed.build())) {
                refBare = prefillLogits(bare);
            }
            assertTrue("the adapter must change the logits, max diff " + maxDiff(refAdapted, refBare),
                    maxDiff(refAdapted, refBare) > 100 * LOGITS_TOL);
            assertArrayEquals("the bare slot of a startup load is the bare placement", refBare, disabled,
                    LOGITS_TOL);
            String name = registry.list().get(0).name();
            int[] off = GenerationLoop.forModel(m, startup).generate(PROMPT, request(only(name, 0f)), null)
                    .tokens();
            int[] bareTokens;
            try (GGUFModelGPU bare = new GGUFModelGPU(model.toString(), null, placed.build())) {
                bareTokens = GenerationLoop.forModel(bare, placed.build()).generate(PROMPT, request(null), null)
                        .tokens();
            }
            assertPrefix(bareTokens, off);
        }

        EngineConfig perRequest = placed.build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, perRequest)) {
            LoraRegistry registry = m.loraRegistry();
            assertNotNull("the placement has a registry", registry);
            registry.load("a", adapter.toString(), SCALE_A);
            int slot = registry.acquire(registry.resolve(LoraSelection.of("a")));
            registry.useSlot(slot);
            assertArrayEquals("a pass in the adapter's slot", refAdapted, prefillLogits(m), LOGITS_TOL);
            registry.useSlot(0);
            assertArrayEquals("a pass in slot 0", refBare, prefillLogits(m), LOGITS_TOL);
            registry.release(slot);
            GenerationLoop loop = GenerationLoop.forModel(m, perRequest);
            assertPrefix(tokensAdapted, loop.generate(PROMPT, request(LoraSelection.of("a")), null).tokens());
            assertPrefix(tokensAdapted, loop.generate(PROMPT, request(LoraSelection.of("a")), null).tokens());
        }
    }

    private static float[] prefillLogits(EngineModel m) {
        ForwardBackend f = m.forwardBackend();
        f.reset();
        float[] logits = f.prefill(PROMPT, 0).clone();
        f.reset();
        return logits;
    }

    private static float maxDiff(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    /** An adapter of every projection of a {@code nLayer}-layer synthetic llama. */
    private static Path adapterOf(int nLayer, long seed) throws IOException {
        return LoraAdapterFiles.llama(SyntheticModels.Config.llama().toBuilder().nLayer(nLayer).build(), seed, RANK,
                ALPHA);
    }

    private static int openClDeviceCount() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            return hw.getCLDevices().size();
        } finally {
            hw.shutdown();
        }
    }

    private static void assertTokens(int[] reference, BatchSequence s) {
        assertEquals("expected DONE but was " + s.state() + (s.error() == null ? "" : ": " + s.error()),
                BatchSequence.State.DONE, s.state());
        assertPrefix(reference, s.result().tokens());
    }

    private static void assertPrefix(int[] reference, int[] got) {
        assertEquals("every request generates its budget (ignore_eos)", N_NEW, got.length);
        assertArrayEquals(reference, got);
    }
}

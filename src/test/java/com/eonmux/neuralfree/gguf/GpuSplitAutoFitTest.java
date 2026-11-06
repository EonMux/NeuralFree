package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.SplitFixtures.assertParity;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static com.eonmux.neuralfree.gguf.SplitFixtures.run;
import static com.eonmux.neuralfree.gguf.SplitFixtures.split;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The automatic fit of a two-device layer split on the OpenCL devices of the machine, under a
 * {@code neuralfree.gpu.vram.limit} cap on every device: under {@code -ngl auto}, a layout that does not stay
 * resident moves the routed experts of the overflowing layers to RAM and then the first trunk layers to the host,
 * before it streams any, as a one-GPU load does. Without {@code --tensor-split}, the fit fills each GPU back to front
 * from its own device bytes and rewrites the shares; an explicit {@code --tensor-split} is kept. Each layout matches
 * the one-device load of the same layout, and {@code -ngl all} still refuses the load; the per-request LoRA slots a
 * load reserves ({@code neuralfree.lora.enable}) count in every layout the fit tries. A layer split over a single
 * listed device runs on that device.
 */
public class GpuSplitAutoFitTest {

    private static final int LAYERS = 8;
    /** A user context whose KV caches do not fit the capped GPUs beside their streaming slots. */
    private static final int LONG_CONTEXT = 4096;
    private static final int TRAIN_CONTEXT = 8192;
    private static final long CAP_MB = 12;
    /** A cap under which each GPU holds more than one layer at the long context without LoRA slots. */
    private static final long RESERVE_CAP_MB = 20;
    /** A cap under which the MoE model's GPUs hold neither their whole spans nor their dense weights and caches. */
    private static final long MOE_CAP_MB = 10;

    private final List<Path> files = new ArrayList<>();

    @Before
    public void requireTwoDevices() {
        SplitFixtures.assumeTwoDevices();
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    /** Eight llama layers of about 3.8 MiB (F32) whose K and V take 512 bytes per position each. */
    private Path llama() throws IOException {
        return written(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(LAYERS).nEmbd(256)
                .nHead(4).nHeadKv(2).headDim(64).nRot(64).nFf(1024).nCtxTrain(TRAIN_CONTEXT).build()));
    }

    /** Eight qwen3moe layers whose routed experts (8 of width 256, about 6 MiB) dwarf their dense weights. */
    private Path qwen3moe() throws IOException {
        return written(SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(LAYERS)
                .nEmbd(256).nHead(4).nHeadKv(2).headDim(64).nRot(64).nExpert(8).nExpertUsed(2).nFfExp(256)
                .nCtxTrain(TRAIN_CONTEXT).build()));
    }

    private static MemoryConfig context(int nCtx) {
        return MemoryConfig.defaults().toBuilder().nCtx(nCtx).ubatch(4).build();
    }

    /** {@code -ngl} of the one-device load that keeps the same first {@code hostLayers} layers on the host. */
    private static int gpuLayersBeside(int hostLayers) {
        return LAYERS - hostLayers + 1;
    }

    /**
     * Loads {@code setting}: its first layers on the host, every GPU span resident after them, and the logits and
     * greedy tokens of the one-device load {@code ref} of the same layout. Returns the host layers.
     */
    private static int assertHostLayoutMatches(Path model, EngineConfig.Builder setting, EngineConfig.Builder ref)
            throws IOException {
        SplitFixtures.Run got;
        int host;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, setting.build())) {
            assertEquals(2, m.deviceCount());
            host = m.hostLayerCount();
            assertSpansFollow(m, host);
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNull("device " + d + " holds its span resident beside the host layers", m.streamingOf(d));
            }
            got = new SplitFixtures.Run(SplitFixtures.logits(m), SplitFixtures.greedy(m),
                    m.forwardBackend().toString());
        }
        assertTrue("-ngl auto moved the first layers to the host: " + got.backend, host > 0);
        assertTrue(got.backend, got.backend.contains("host layers [0, " + host + ")"));
        SplitFixtures.Run reference = run(model, ref.gpuLayers(gpuLayersBeside(host)).build());
        assertTrue(reference.backend, reference.backend.contains("Hybrid"));
        assertParity(got.backend, reference, got);
        return host;
    }

    /** The GPU spans cover the layers after the {@code host} prefix, in pipeline order, each with a layer. */
    private static void assertSpansFollow(GGUFModelGPU m, int host) {
        int next = host;
        for (int d = 0; d < m.deviceCount(); d++) {
            assertEquals("device " + d + " starts where the previous span ends", next, m.firstLayerOf(d));
            assertTrue("device " + d + " runs a layer", m.layerCountOf(d) > 0);
            next += m.layerCountOf(d);
        }
        assertEquals(LAYERS, next);
    }

    // ------------------------------------------------------------------ -ngl auto

    @Test
    public void aContextTheStreamedSpansCannotHoldMovesTheFirstLayersToTheHost() throws IOException {
        Path model = llama();
        EngineConfig.Builder capped = split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(CAP_MB);
        int host = assertHostLayoutMatches(model, capped, oneDevice().memory(context(LONG_CONTEXT)));
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, capped.build())) {
            assertEquals(LONG_CONTEXT, m.memoryPlan().nCtx());
        }
        RuntimeException all = assertThrows(RuntimeException.class, () -> new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(CAP_MB)
                        .gpuLayers(EngineConfig.GPU_LAYERS_ALL).build()).close());
        assertTrue("-ngl all keeps every layer on the GPUs and does not fit: " + all,
                all instanceof GpuOutOfMemoryException || all instanceof ContextFitter.ContextBudgetException);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).build())) {
            assertEquals("without the cap every layer stays on the GPUs", 0, m.hostLayerCount());
        }
        assertTrue("every GPU keeps a layer", host <= LAYERS - 2);
    }

    @Test
    public void tensorSplitSharesSpreadTheLayersLeftOnTheGpus() throws IOException {
        Path model = llama();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(CAP_MB).tensorSplit(new double[] {1, 3})
                        .build())) {
            int host = m.hostLayerCount();
            assertTrue(m.forwardBackend().toString(), host > 0);
            assertSpansFollow(m, host);
            int left = LAYERS - host;
            assertTrue("the first GPU takes about a quarter of the " + left + " GPU layers: " + m.layerCountOf(0),
                    Math.abs(m.layerCountOf(0) - left / 4.0) <= 1.0);
            assertTrue(m.layerCountOf(1) >= m.layerCountOf(0));
        }
    }

    @Test
    public void withoutTensorSplitTheFillSizesEachSpanFromItsOwnDevice() throws IOException {
        Path model = llama();
        try (GGUFModelGPU first = new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(CAP_MB).build())) {
            assertTrue(first.hostLayerCount() > 0);
            assertSpansFollow(first, first.hostLayerCount());
            assertTrue("the main GPU also holds the embedding and the output head, so the fill gives it fewer"
                            + " layers than the other device: " + first.layerCountOf(0) + " vs "
                            + first.layerCountOf(1), first.layerCountOf(1) > first.layerCountOf(0));
            double[] rewritten = first.engineConfig().tensorSplit();
            assertEquals("the fit rewrites --tensor-split from what each GPU holds", 2, rewritten.length);
            assertEquals(first.layerCountOf(0), (int) rewritten[0]);
            assertEquals(first.layerCountOf(1), (int) rewritten[1]);
        }
        try (GGUFModelGPU last = new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(CAP_MB).mainGpu(1).build())) {
            assertTrue(last.hostLayerCount() > 0);
            assertSpansFollow(last, last.hostLayerCount());
            assertTrue("with the main GPU last, the device without the model-level tensors takes more: "
                            + last.layerCountOf(0) + " vs " + last.layerCountOf(1),
                    last.layerCountOf(0) > last.layerCountOf(1));
        }
    }

    @Test
    public void aCapStreamingCouldCoverStillMovesLayersToTheHostOnOneGpuAndASplit() throws IOException {
        Path model = llama();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                split().memory(context(64)).gpuVramLimitMb(CAP_MB).build())) {
            assertTrue("the split moves layers to the host rather than streaming: " + m.forwardBackend(),
                    m.hostLayerCount() > 0);
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNull("device " + d + " holds its span resident beside the host layers", m.streamingOf(d));
            }
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                oneDevice().memory(context(64)).gpuVramLimitMb(CAP_MB).build())) {
            assertTrue("one GPU moves layers to the host rather than streaming: " + m.forwardBackend(),
                    m.forwardBackend().toString().contains("Hybrid"));
            assertNull("the offloaded one-device load does not stream", m.streamingOf(0));
        }
    }

    @Test
    public void anAutomaticContextBelowTheFloorWhenStreamedMovesLayersToTheHost() throws IOException {
        Path model = llama();
        EngineConfig.Builder capped = split().memory(context(0)).gpuVramLimitMb(CAP_MB);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, capped.build())) {
            assertTrue("the host takes layers so the context reaches the floor: " + m.forwardBackend(),
                    m.hostLayerCount() > 0);
            assertSpansFollow(m, m.hostLayerCount());
            assertTrue("context " + m.memoryPlan().nCtx(), m.memoryPlan().nCtx() >= 4096);
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                capped.gpuLayers(EngineConfig.GPU_LAYERS_ALL).build())) {
            assertEquals("-ngl all streams instead", 0, m.hostLayerCount());
            assertTrue("context " + m.memoryPlan().nCtx(), m.memoryPlan().nCtx() < 4096);
        }
    }

    @Test
    public void theLoraSlotReservationMovesMoreLayersToTheHost() throws IOException {
        Path model = llama();
        int plain;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(RESERVE_CAP_MB).build())) {
            plain = m.hostLayerCount();
        }
        assertTrue("without the reservation the GPUs hold more than one layer each: " + plain + " host layers",
                plain < LAYERS - 2);
        EngineConfig.Builder reserved = split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(RESERVE_CAP_MB)
                .loraConfig(LoraConfig.builder().enabled(true).maxAdapters(4).maxRank(64).build());
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, reserved.build())) {
            int host = m.hostLayerCount();
            assertSpansFollow(m, host);
            assertTrue("the slots each GPU reserves for its layers leave room for fewer of them: " + host
                    + " host layers with the reservation, " + plain + " without", host > plain);
            assertEquals(LONG_CONTEXT, m.memoryPlan().nCtx());
            assertTrue("the split serves per-request adapters", m.loraRegistry() != null);
        }
    }

    // ------------------------------------------------------------------ --fit

    @Test
    public void theOverflowingExpertsMoveToRamBeforeWholeLayers() throws IOException {
        Path model = qwen3moe();
        EngineConfig.Builder fitted = split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(MOE_CAP_MB).fit(true);
        int host = assertHostLayoutMatches(model, fitted,
                oneDevice().memory(context(LONG_CONTEXT)).fit(true));
        EngineConfig.Builder plain = split().memory(context(LONG_CONTEXT)).gpuVramLimitMb(MOE_CAP_MB);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, plain.build())) {
            assertEquals("without --fit the automatic split still moves the overflowing experts to RAM before a"
                    + " whole layer: " + m.forwardBackend(), host, m.hostLayerCount());
            assertSpansFollow(m, host);
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNull("device " + d + " holds its span resident", m.streamingOf(d));
            }
            SplitFixtures.Run reference = run(model, oneDevice().memory(context(LONG_CONTEXT)).fit(true)
                    .gpuLayers(gpuLayersBeside(host)).build());
            assertParity("experts in RAM without --fit", reference, new SplitFixtures.Run(SplitFixtures.logits(m),
                    SplitFixtures.greedy(m), m.forwardBackend().toString()));
        }
    }

    // ------------------------------------------------------------------ one listed device

    @Test
    public void aLayerSplitOverOneListedDeviceRunsOnThatDevice() throws IOException {
        Path model = llama();
        String expected;
        String other;
        SplitFixtures.Run reference;
        try (GGUFModelGPU direct = new GGUFModelGPU(model.toString(), null, oneDevice("1").build())) {
            expected = direct.backend().capabilities().getDeviceName();
            reference = new SplitFixtures.Run(SplitFixtures.logits(direct), SplitFixtures.greedy(direct), "");
        }
        try (GGUFModelGPU first = new GGUFModelGPU(model.toString(), null, oneDevice("0").build())) {
            other = first.backend().capabilities().getDeviceName();
        }
        assertNotEquals("the two listed indices name different devices", expected, other);
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                oneDevice("1").splitMode(EngineConfig.SplitMode.LAYER).build())) {
            assertTrue(m.engineConfig().split());
            assertEquals(1, m.deviceCount());
            assertEquals("--split-mode layer --device 1 runs on device 1", expected,
                    m.backend().capabilities().getDeviceName());
            assertParity("-sm layer -dev 1", reference, new SplitFixtures.Run(SplitFixtures.logits(m),
                    SplitFixtures.greedy(m), m.forwardBackend().toString()));
        }
    }
}

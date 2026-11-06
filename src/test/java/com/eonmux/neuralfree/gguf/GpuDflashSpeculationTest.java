package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDrafter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * A {@code dflash} draft of a GPU target drafting on the target's device ({@link DflashDrafter#onDevice}, llama.cpp
 * {@code --spec-draft-backend-sampling}): fed the features of the real target forward calls it drafts what the host
 * drafter drafts, with the greedy readout on the device and with {@code p_min} reading the logits back, and a
 * speculative session keeps the plain greedy tokens. {@code neuralfree.draft.dflash.device=false} drafts on the
 * host, and a quantized draft cache drafts on the device when the draft's head size is a multiple of 32 (a wider
 * draft), refusing the synthetic 16 head by name. Skipped when OpenCL is unavailable.
 */
public class GpuDflashSpeculationTest {

    private static final int TOKENS = 20;
    private static final int DRAFT = 3;
    private static final int[] CONTINUATION = {77, 3, 150, 9};

    private Path model;
    private Path draft;
    private Path wideDraft;

    @Before
    public void writeModels() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.llama().writeTempFile();
        draft = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(DRAFT + 1, 1, 2)).writeTempFile();
        wideDraft = SyntheticDflash.dflash(SyntheticModels.Config.llama().toBuilder().headDim(32).nRot(32).build(),
                SyntheticDflash.Options.of(DRAFT + 1, 1, 2)).writeTempFile();
    }

    @After
    public void deleteModels() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
        if (draft != null) {
            Files.deleteIfExists(draft);
        }
        if (wideDraft != null) {
            Files.deleteIfExists(wideDraft);
        }
    }

    @Test
    public void theDeviceDrafterDraftsWhatTheHostDrafterDrafts() throws IOException {
        List<int[]> device = drafts(base(true));
        List<int[]> host = drafts(base(false));
        assertEquals(host.size(), device.size());
        for (int i = 0; i < host.size(); i++) {
            assertArrayEquals("draft " + i, host.get(i), device.get(i));
            assertEquals("draft " + i + " is a full block", DRAFT, device.get(i).length);
        }
    }

    @Test
    public void aPMinDraftReadsTheDeviceLogitsLikeTheHost() throws IOException {
        List<int[]> device = drafts(base(true).draftPMin(0.01f));
        List<int[]> host = drafts(base(false).draftPMin(0.01f));
        for (int i = 0; i < host.size(); i++) {
            assertArrayEquals("draft " + i, host.get(i), device.get(i));
        }
    }

    @Test
    public void speculationOnTheDeviceKeepsThePlainGreedyTokens() throws IOException {
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            reference = greedy(m);
        }
        SpeculativeDecoder.Stats device = assertGreedy(base(true), reference, true);
        SpeculativeDecoder.Stats host = assertGreedy(base(false), reference, false);
        assertEquals("the same drafts are drafted and accepted", host.drafted(), device.drafted());
        assertEquals("the same drafts are drafted and accepted", host.accepted(), device.accepted());
        assertTrue(String.valueOf(device), device.drafted() > 0);
    }

    /** A target with a layer on the host keeps its device context, which the drafter drafts on. */
    @Test
    public void aHybridTargetDraftsOnItsDevice() throws IOException {
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().gpuLayers(1).build())) {
            reference = greedy(m);
        }
        SpeculativeDecoder.Stats device = assertGreedy(base(true).gpuLayers(1), reference, true);
        SpeculativeDecoder.Stats host = assertGreedy(base(false).gpuLayers(1), reference, false);
        assertEquals("the same drafts are accepted", host.accepted(), device.accepted());
    }

    /** A full-GPU target's rows reach the drafter on the device; a hybrid target reads them back. */
    @Test
    public void theTargetRowsArriveOnTheDrafterDevice() throws IOException {
        EngineConfig full = base(true).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, full);
             SpeculativeSession s = LLMModelFactory.speculate(target, full)) {
            assertTrue(((DflashDrafter) s.drafter()).readsTargetRowsOnDevice());
        }
        EngineConfig hybrid = base(true).gpuLayers(1).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, hybrid);
             SpeculativeSession s = LLMModelFactory.speculate(target, hybrid)) {
            assertFalse(((DflashDrafter) s.drafter()).readsTargetRowsOnDevice());
        }
    }

    /**
     * A seeded temperature chain verifies the same on the device as on the host: the draw over the device's top
     * candidates is the chain's draw over the whole row, so the tokens, drafts and acceptances match.
     */
    @Test
    public void aSeededTemperatureChainVerifiesTheSameOnTheDevice() throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(TOKENS)
                .sampler(SamplerConfig.builder().temperature(0.9f).topK(40).topP(0.95f).minP(0.05f).typicalP(1f)
                        .penaltyLastN(0).seed(2024L).build()).build();
        SpeculativeDecoder.Stats device = temperatureSession(base(true), request);
        SpeculativeDecoder.Stats host = temperatureSession(base(false).speculativeOptions(
                SpeculativeOptions.builder().dflashOnDevice(false).verifyOnDevice(false).build()), request);
        assertEquals("the same drafts are drafted and accepted", host.drafted(), device.drafted());
        assertEquals("the same drafts are drafted and accepted", host.accepted(), device.accepted());
    }

    /** One temperature generation of a session, as its stats. */
    private SpeculativeDecoder.Stats temperatureSession(EngineConfig.Builder b, GenerationRequest request)
            throws IOException {
        EngineConfig cfg = b.build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            s.generate(PROMPT, request, null);
            return s.lastStats();
        }
    }

    /** The quantized draft cache runs on the device when the draft's head size is a multiple of 32. */
    @Test
    public void aQuantizedDraftCacheDraftsOnTheDevice() throws IOException {
        EngineConfig cfg = base(true).draftModelPath(wideDraft.toString()).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            assertTrue(((DflashDrafter) s.drafter()).onDevice());
        }
    }

    /** The drafts over a quantized draft cache match the host cache's within its quantization. */
    @Test
    public void aQuantizedDraftCacheDraftsWhatTheHostDrafts() throws IOException {
        EngineConfig.Builder device = base(true).draftModelPath(wideDraft.toString())
                .memory(MemoryConfig.builder().draftTypeK(KvDType.Q8_0).draftTypeV(KvDType.Q8_0).build());
        EngineConfig.Builder host = base(false).draftModelPath(wideDraft.toString())
                .memory(MemoryConfig.builder().draftTypeK(KvDType.Q8_0).draftTypeV(KvDType.Q8_0).build());
        List<int[]> devDrafts = drafts(device);
        List<int[]> hostDrafts = drafts(host);
        assertEquals(hostDrafts.size(), devDrafts.size());
        for (int i = 0; i < hostDrafts.size(); i++) {
            assertArrayEquals("draft " + i, hostDrafts.get(i), devDrafts.get(i));
        }
    }

    /**
     * A watermarked speculative generation on a device target: the drafter's keyed draw reads the block's logits
     * back (the device readout carries only ids), the target verifies by rejection under the target key, and the
     * output is detectable.
     */
    @Test
    public void aWatermarkedDflashSpeculationOnTheDeviceIsDetectable() throws IOException {
        com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark =
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig.builder(55L)
                        .algorithm(com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm.DUAL_KEY_GUMBEL)
                        .contextWidth(4)
                        .deduplicateContexts(
                                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkContextScope.NONE)
                        .build();
        EngineConfig.Builder builder = base(true).speculativeOptions(SpeculativeOptions.builder()
                .verifyMode(com.eonmux.neuralfree.gguf.engine.draft.VerifyMode.REJECTION).build());
        EngineConfig cfg = builder.build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(40)
                .sampler(SamplerConfig.builder().temperature(0.8f).topK(0).topP(1f).minP(0f).penaltyLastN(0)
                        .seed(2024L).watermark(watermark).build())
                .build();
        int[] tokens;
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            assertTrue("the drafter runs on the device", ((DflashDrafter) s.drafter()).onDevice());
            tokens = s.generate(PROMPT, request, null).tokens();
            assertTrue("the watermarked drafter proposed", s.lastStats().drafted() > 0);
        }
        assertTrue("the watermarked speculative output is detected",
                com.eonmux.neuralfree.gguf.engine.watermark.WatermarkDetector.create(watermark)
                        .detect(tokens).isWatermarked());
    }

    private EngineConfig.Builder base(boolean onDevice) {
        return oneDevice().draftModelPath(draft.toString()).draftMax(DRAFT)
                .speculativeOptions(SpeculativeOptions.builder().dflashOnDevice(onDevice).build());
    }

    /**
     * The drafts after the prompt and after each continuation token, the target's forward calls feeding the drafter
     * as a session's do.
     */
    private List<int[]> drafts(EngineConfig.Builder b) throws IOException {
        EngineConfig cfg = b.build();
        List<int[]> out = new ArrayList<>();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            DraftModel d = s.drafter();
            assertEquals(cfg.speculativeOptions().dflashOnDevice(), ((DflashDrafter) d).onDevice());
            ForwardBackend f = target.forwardBackend();
            f.reset();
            d.reset();
            f.prefill(PROMPT, 0);
            out.add(d.draft(PROMPT[PROMPT.length - 1], DRAFT));
            int pos = PROMPT.length;
            for (int token : CONTINUATION) {
                f.decode(token, pos++);
                out.add(d.draft(token, DRAFT));
            }
            int[] verify = Arrays.copyOf(CONTINUATION, 3);
            f.forwardRows(verify, pos, null);
            f.rollback(2);
            out.add(d.draft(verify[0], DRAFT));
            f.reset();
            d.reset();
        }
        return out;
    }

    private SpeculativeDecoder.Stats assertGreedy(EngineConfig.Builder b, int[] reference, boolean onDevice)
            throws IOException {
        EngineConfig cfg = b.build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            assertEquals(onDevice, ((DflashDrafter) s.drafter()).onDevice());
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            assertArrayEquals(reference, s.generate(PROMPT, request, null).tokens());
            return s.lastStats();
        }
    }

    private static int[] greedy(GGUFModelGPU m) {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(TOKENS)
                .sampler(m.engineConfig().legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
        return m.generate(PROMPT, request, null).tokens();
    }
}

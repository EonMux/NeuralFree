package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.flow.NextnDraftModel;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticMtpArchModels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * Speculative decoding over the NextN / MTP heads of the vLLM-only MTP architectures: EXAONE 4.5, EXAONE-MoE,
 * BailingMoE V2, Dots3-Note and dense Nemotron-H. Each synthetic GGUF target drafts through its own appended blocks
 * and keeps the plain greedy tokens (verification is lossless); a multi-block EXAONE 4.5 head cycles its blocks, so a
 * draft longer than the block count still runs.
 */
public class NextnMtpArchSpeculationTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26, 3, 64};
    private static final int GREEDY_TOKENS = 8;

    @Test
    public void exaone45KeepsThePlainGreedyTokens() throws IOException {
        assertGreedyParity(SyntheticMtpArchModels.exaone4Mtp(exaone4Config(), 1), 3);
    }

    @Test
    public void exaone45CyclesItsNextnBlocks() throws IOException {
        assertTrue("exaone4 cycles", NextnDraftModel.cyclesBlocks(Arch.EXAONE4));
        assertCycles(SyntheticMtpArchModels.exaone4Mtp(exaone4Config(), 2), 2, 5);
    }

    @Test
    public void exaoneMoeKeepsThePlainGreedyTokens() throws IOException {
        assertGreedyParity(SyntheticMtpArchModels.exaoneMoeMtp(exaoneMoeConfig(), 1), 3);
    }

    @Test
    public void bailingMoe2KeepsThePlainGreedyTokens() throws IOException {
        assertGreedyParity(SyntheticMtpArchModels.bailingMoe2Mtp(bailingConfig(), 1), 3);
    }

    @Test
    public void dots3NoteKeepsThePlainGreedyTokens() throws IOException {
        assertGreedyParity(SyntheticMtpArchModels.dots3NoteMtp(dots3Config(), 1), 3);
    }

    @Test
    public void denseNemotronHKeepsThePlainGreedyTokens() throws IOException {
        assertGreedyParity(SyntheticMtpArchModels.nemotronHDenseMtp(nemotronConfig(), 1), 3);
    }

    private void assertGreedyParity(GgufTestWriter writer, int draftMax) throws IOException {
        Path model = write(writer);
        try (EngineModel m = LLMModelFactory.open(model.toString(), base().build())) {
            int[] plain = m.generate(PROMPT, greedy(GREEDY_TOKENS), null).tokens();
            m.resetKVCache();

            EngineConfig cfg = base().mtp(true).draftMax(draftMax).build();
            try (SpeculativeSession s = LLMModelFactory.speculate(m, cfg)) {
                GenerationResult r = s.generate(PROMPT, greedy(GREEDY_TOKENS), null);
                assertArrayEquals("MTP greedy over the target's own blocks", plain, r.tokens());
                assertTrue("the drafter proposed tokens: " + s.lastStats(), s.lastStats().drafted() > 0);
            }
        }
    }

    /** A cycling head drafts past its block count: each step runs the next block (vLLM's {@code spec_step_idx % n}). */
    private void assertCycles(GgufTestWriter writer, int blocks, int draftMax) throws IOException {
        Path model = write(writer);
        try (EngineModel m = LLMModelFactory.open(model.toString(), base().build());
             SpeculativeSession s = LLMModelFactory.speculate(m, base().mtp(true).draftMax(draftMax).build())) {
            m.forwardBackend().forwardRows(PROMPT, 0, null);
            assertTrue("a cycling head is not clamped to its block count: " + s.draftMax(), s.draftMax() > blocks);
            int[] draft = s.drafter().draft(PROMPT[PROMPT.length - 1], draftMax);
            assertTrue("a cycling head with " + s.drafter() + " drafts " + draft.length + " tokens",
                    draft.length > blocks);
        }
    }

    private Path write(GgufTestWriter writer) throws IOException {
        return writer.writeTempFile();
    }

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static GenerationRequest greedy(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    private static SyntheticModels.Config exaone4Config() {
        return SyntheticModels.Config.qwen2().toBuilder().nLayer(2).build();
    }

    private static SyntheticModels.Config exaoneMoeConfig() {
        return SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).slidingWindow(32).build();
    }

    private static SyntheticModels.Config bailingConfig() {
        return SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build();
    }

    private static SyntheticModels.Config dots3Config() {
        return new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(1).headDim(32).nRot(16).nFf(128)
                .nExpert(4).nExpertUsed(2).nFfExp(32).normEps(1e-6f).build();
    }

    private static SyntheticModels.Config nemotronConfig() {
        return SyntheticModels.Config.llama().toBuilder().nLayer(2).build();
    }
}

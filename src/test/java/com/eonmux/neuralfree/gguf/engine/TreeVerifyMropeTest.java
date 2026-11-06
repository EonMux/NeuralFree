package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The tree verify of a multi-section (M-RoPE) model: qwen2vl ropes every token from four position sections
 * (temporal, height, width, extra), so the verify batch cannot carry one position per row — its rows carry the
 * section-major positions of their depth positions, the pending row roped at {@code pos} and a node roped at
 * {@code pos + depth} in every section ({@code MropeTextPositions#position}). The forced two-level proposal makes
 * every round accept the second root's path, so the commit moves its rows down onto the consecutive positions and
 * the generation must stay lossless against the target-only greedy loop: the ropes, the mask and the compaction
 * all ride the four-section batch, on the paged memory and the plain contiguous memory alike.
 */
public class TreeVerifyMropeTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 12;
    private static final int PARALLEL = 3;
    private static final int DRAFT_MAX = 3;
    /** The biased tokens: 6 wins every draw, 5 stays visible as the runner-up (both roots propose them). */
    private static final int LO_TOKEN = 5;
    private static final int HI_TOKEN = 6;
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5};

    private static Path modelFile;
    private static int nVocab;

    @BeforeClass
    public static void writeModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(4).build();
        modelFile = SyntheticModels.qwen2vl(c).writeTempFile();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                        .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build())) {
            nVocab = model.spec().hparams().nVocab();
            assertTrue("the qwen2vl model ropes from the four sections",
                    com.eonmux.neuralfree.gguf.ops.rope.RopeMath
                            .positionsPerToken(model.spec().hparams().ropeType()) > 1);
        }
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
        }
    }

    /**
     * A drafter proposing the forced two-level tree — root {@code 5}, root {@code 6}, and {@code 6} again as the
     * second root's depth-2 child — with no target interaction. It records every round's acceptance, so the test
     * can tell a real tree round (both levels kept, the commit moving real rows) from a fallback.
     */
    private static final class ForcedTreeDrafter implements TreeDraftModel {

        final List<Integer> acceptedPerRound = new ArrayList<>();

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int beginGeneration(int reused) {
            return 0;
        }

        @Override
        public int[] draft(int lastToken, int n) {
            return new int[0];
        }

        @Override
        public void accept(int nAccepted) {
        }

        @Override
        public int branches() {
            return 2;
        }

        @Override
        public boolean singlePassVerify() {
            return true;
        }

        @Override
        public void acceptTree(int nAccepted) {
            acceptedPerRound.add(nAccepted);
        }

        @Override
        public void featureTapSuspended(boolean suspended) {
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature,
                                   SplittableRandom rng) {
            return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                                   SplittableRandom rng) {
            List<DraftTree.Node> nodes = new ArrayList<>();
            nodes.add(new DraftTree.Node(LO_TOKEN, -1, 1, 0));
            nodes.add(new DraftTree.Node(HI_TOKEN, -1, 1, 0));
            nodes.add(new DraftTree.Node(HI_TOKEN, 1, 2, 0));
            return new DraftTree(nodes.subList(0, Math.min(nodes.size(), Math.max(0, maxRows))));
        }
    }

    private static EngineConfig treeConfig(boolean paged) {
        MemoryConfig.Builder m = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            m.parallel(PARALLEL).kvUnified(true);
        }
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(m.build())
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX).build())
                .build();
    }

    private static EngineConfig plainConfig() {
        MemoryConfig m = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(m).build();
    }

    /** The greedy request whose biases force every draw to {@code HI_TOKEN}, runner-up {@code LO_TOKEN}. */
    private static GenerationRequest forced() {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding());
        for (int t = 0; t < nVocab; t++) {
            if (t == HI_TOKEN) {
                b.logitBias(t, 50.0f);
            } else if (t == LO_TOKEN) {
                b.logitBias(t, 20.0f);
            } else {
                b.logitBias(t, -100.0f);
            }
        }
        return b.build();
    }

    @Test
    public void theMropeTreeVerifyStaysLosslessOnPagedMemory() throws IOException {
        check(true);
    }

    /** The same round over the plain contiguous single-sequence memory. */
    @Test
    public void theMropeTreeVerifyStaysLosslessOnContiguousMemory() throws IOException {
        check(false);
    }

    private void check(boolean paged) throws IOException {
        GenerationRequest request = forced();
        int[] expectedTokens;
        try (EngineModel plain = LLMModelFactory.open(modelFile.toString(), plainConfig())) {
            expectedTokens = GenerationLoop.forModel(plain, plainConfig())
                    .generate(PROMPT, request, (TokenCallback) null).tokens();
        }
        EngineConfig treeCfg = treeConfig(paged);
        int[] actualTokens;
        ForcedTreeDrafter drafter = new ForcedTreeDrafter();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, treeCfg);
            actualTokens = decoder.generate(PROMPT, request, (TokenCallback) null).tokens();
        }
        assertArrayEquals("the four-section verify commits the plain decode's tokens (" + (paged ? "paged"
                : "contiguous") + ")", expectedTokens, actualTokens);
        assertTrue("the forced rounds kept the full two-level path, so the compaction moved real rows: "
                + drafter.acceptedPerRound, drafter.acceptedPerRound.contains(2));
    }
}

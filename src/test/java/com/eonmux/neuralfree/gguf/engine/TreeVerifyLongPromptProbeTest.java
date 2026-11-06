package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Throwaway probe: a paged device tree verify whose prompt prefills through several ubatches. */
public class TreeVerifyLongPromptProbeTest {

    private static final int N_CTX = 1024;
    private static final int N_NEW = 48;
    private static final int PARALLEL = 2;
    private static final int DRAFT_MAX = 8;
    private static final int LO_TOKEN = 5;
    private static final int HI_TOKEN = 6;

    private static final class ForcedTreeDrafter implements TreeDraftModel {
        @Override public void bindTarget(EngineModel target) { }
        @Override public int beginGeneration(int reused) { return 0; }
        @Override public int[] draft(int lastToken, int n) { return new int[0]; }
        @Override public void accept(int nAccepted) { }
        @Override public int branches() { return 2; }
        @Override public boolean singlePassVerify() { return true; }
        @Override public void acceptTree(int nAccepted) { }
        @Override public void featureTapSuspended(boolean suspended) { }
        @Override public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature,
                                             SplittableRandom rng) {
            return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
        }
        @Override public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows,
                                             float temperature, SplittableRandom rng) {
            List<DraftTree.Node> nodes = new ArrayList<>();
            nodes.add(new DraftTree.Node(HI_TOKEN, -1, 1, 0));
            for (int d = 2; d <= 8; d++) {
                nodes.add(new DraftTree.Node(HI_TOKEN, d - 2, d, 0));
            }
            nodes.add(new DraftTree.Node(LO_TOKEN, -1, 1, 1));
            return new DraftTree(nodes.subList(0, Math.min(nodes.size(), Math.max(0, maxRows))));
        }
    }

    @Test
    public void probe() throws IOException {
        Assume.assumeTrue(GpuTestUtil.isCudaAvailable());
        Path modelFile = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build())
                .writeTempFile();
        Path drafterFile = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build())
                .writeTempFile();
        modelFile.toFile().deleteOnExit();
        drafterFile.toFile().deleteOnExit();
        int[] prompt = new int[65];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = 4 + (i % 5);
        }
        MemoryConfig paged = MemoryConfig.builder().nCtx(N_CTX).ubatch(64)
                .parallel(PARALLEL).kvUnified(true).build();
        EngineConfig treeCfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CUDA)
                .memory(paged).draftMax(DRAFT_MAX).draftModelPath(drafterFile.toString())
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(16)
                        .pSplit(0.1).build())
                .build();
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding());
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            int nVocab = model.spec().hparams().nVocab();
            for (int t = 0; t < nVocab; t++) {
                b.logitBias(t, t == HI_TOKEN ? 50.0f : (t == LO_TOKEN ? 20.0f : -100.0f));
            }
            try (SpeculativeSession session = LLMModelFactory.speculate(model, treeCfg)) {
                GenerationResult r = session.generate(prompt, b.build(), (TokenCallback) null);
                System.out.println("probe tokens " + r.tokens().length + " stop " + r.stopReason());
            }
        }
    }
}

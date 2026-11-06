package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The fixed-width tree verify round captured as a replayed decode-step graph ({@code neuralfree.gpu.chainReplay},
 * warmup zero so the first round captures): the verify forward's shape never changes — {@code 1 + width} rows,
 * whatever the proposal drafts — so the step trunk records once and replays through the whole generation, and a
 * compaction commit's row moves dispatch outside the captured trunk without closing it (the capture keeps the
 * kernel dispatches only; the uploads, the mask, the appends and the row moves run between replays). The test
 * walks a seeded-random-tree drafter's generation on the GPU with the capture on and asserts the run stays
 * lossless against the target's plain greedy loop; on CUDA it further asserts the step replay still holds a
 * captured body graph at the end — a capture the commits had invalidated would instead report warmup steps —
 * and on OpenCL, whose queue may record no command graphs, it demands the capture only when the stats show one.
 * The GPU cases are skipped when no CUDA or OpenCL device is available.
 */
public class TreeVerifyReplayTest {

    private static final String CHAIN_REPLAY = "neuralfree.gpu.chainReplay";
    private static final String WARMUP = "neuralfree.gpu.graphWarmupSteps";
    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int WIDTH = 8;
    private static final long SEED = 20261106L;
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};

    private static Path modelFile;
    private static int nVocab;

    @BeforeClass
    public static void writeModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(4).build();
        modelFile = SyntheticModels.llama(c).writeTempFile();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                        .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build())) {
            nVocab = model.spec().hparams().nVocab();
        }
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
        }
    }

    @Before
    public void captureFromTheFirstRound() {
        System.setProperty(CHAIN_REPLAY, "true");
        System.setProperty(WARMUP, "0");
    }

    @After
    public void restoreReplayProperties() {
        System.clearProperty(CHAIN_REPLAY);
        System.clearProperty(WARMUP);
    }

    /**
     * The random-tree drafter of {@link TreeVerifyDeviceAcceptTest}: a chain of a calibration run's greedy tokens
     * fronted by one or two decoy roots (some repeating the chain's next token), so every round verifies a
     * branching proposal over the fixed width and the accepted paths' commits move real rows down.
     */
    private static final class RandomChainTreeDrafter implements TreeDraftModel {
        private final int[] chain;
        private final List<Integer> acceptedPerRound = new ArrayList<>();
        private int cursor = 1; // the generation's pre-loop sample emitted the chain's first token

        RandomChainTreeDrafter(int[] chain) {
            this.chain = chain;
        }

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int beginGeneration(int reused) {
            cursor = 1;
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
            return WIDTH;
        }

        @Override
        public boolean singlePassVerify() {
            return true;
        }

        @Override
        public void acceptTree(int nAccepted) {
            acceptedPerRound.add(nAccepted);
            cursor += 1 + nAccepted;
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature,
                                   SplittableRandom rng) {
            return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                                   SplittableRandom rng) {
            int budget = Math.min(Math.min(maxTokens, branches), maxRows);
            List<DraftTree.Node> nodes = new ArrayList<>();
            int decoys = 1 + rng.nextInt(2);
            for (int i = 0; i < decoys && nodes.size() < budget; i++) {
                int token = rng.nextBoolean() && cursor < chain.length ? chain[cursor] : rng.nextInt(nVocab);
                nodes.add(new DraftTree.Node(token, -1, 1, 0));
            }
            int parent = -1;
            int depth = 1;
            while (nodes.size() < budget && cursor + depth - 1 < chain.length) {
                nodes.add(new DraftTree.Node(chain[cursor + depth - 1], parent, depth, 0));
                parent = nodes.size() - 1;
                depth++;
            }
            if (!nodes.isEmpty() && nodes.size() < budget && rng.nextBoolean()) {
                int p = rng.nextInt(nodes.size());
                nodes.add(new DraftTree.Node(rng.nextInt(nVocab), p, nodes.get(p).depth() + 1, 0));
            }
            return new DraftTree(nodes);
        }
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(N_CTX).ubatch(16).parallel(2).kvUnified(true)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    /** A greedy request at a fixed seed: every run's draft stream draws the same trees. */
    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding().toBuilder().seed(SEED).build()).build();
    }

    private static int[] calibration(EngineConfig.BackendMode backend) throws IOException {
        EngineConfig plain = EngineConfig.builder().backend(backend)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), plain)) {
            return GenerationLoop.forModel(model, plain).generate(PROMPT, greedy(), (TokenCallback) null).tokens();
        }
    }

    private void theCapturedRoundsStayLossless(EngineConfig.BackendMode backend) throws IOException {
        int[] chain = calibration(backend);
        RandomChainTreeDrafter drafter = new RandomChainTreeDrafter(chain);
        EngineConfig cfg = EngineConfig.builder().backend(backend).memory(memory())
                .draftMax(WIDTH)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(WIDTH).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.builder(model.forwardBackend(), drafter,
                    model.tokenizer())
                    .draftMax(WIDTH)
                    .treeNodes(WIDTH)
                    .ropeType(model.spec().hparams().ropeType())
                    .textPositions(model.spec().hparams().mropeTextPositions())
                    .verifyOnDevice(false)
                    .build();
            GenerationResult r = decoder.generate(PROMPT, greedy(), (TokenCallback) null);
            assertEquals(N_NEW, r.tokens().length);
            assertEquals(StopReason.MAX_TOKENS, r.stopReason());
            assertArrayEquals("the replayed verify rounds commit the plain greedy chain", chain, r.tokens());
            assertTrue("the rounds accepted through the calibrated chain: " + drafter.acceptedPerRound,
                    drafter.acceptedPerRound.stream().mapToInt(Integer::intValue).sum() >= 4);
            String stats = ((GpuForwardBackend) model.forwardBackend()).decodeReplayStats();
            // the second half of the stats is the step replay (the first, the chain decode's position replay the
            // verify batches keep invalidated)
            String stepStats = stats.substring(stats.lastIndexOf("GpuDecodeReplay{"));
            if (backend == EngineConfig.BackendMode.CUDA) {
                assertTrue("the step replay holds the captured verify trunk after the commits' compaction: " + stats,
                        stepStats.contains(", body replay"));
                assertFalse("the capture never rebuilt mid-generation: " + stats,
                        stepStats.contains(" warmup steps"));
            } else if (stepStats.contains("body replay")) {
                // an OpenCL queue that records command graphs holds the capture through the commits too
                assertFalse("the capture never rebuilt mid-generation: " + stats,
                        stepStats.contains(" warmup steps"));
            }
        }
    }

    /** The captured rounds on CUDA (the backend the chain replay is on by default). */
    @Test
    public void theCapturedRoundsStayLosslessOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        theCapturedRoundsStayLossless(EngineConfig.BackendMode.CUDA);
    }

    /** The captured rounds on OpenCL through the GLSL graph path. */
    @Test
    public void theCapturedRoundsStayLosslessOnOpenCl() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        theCapturedRoundsStayLossless(EngineConfig.BackendMode.OPENCL);
    }
}

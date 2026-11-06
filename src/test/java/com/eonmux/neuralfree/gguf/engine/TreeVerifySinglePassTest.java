package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
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
 * The single-pass round's accepted commit, deterministic: a test drafter proposes four fixed tokens as depth-1
 * roots, and the request's logit bias bans every other id (the specials included), pinning the target's sampling
 * support to exactly those four, so every round's sample matches one root — the round accepts one drafted token and
 * commits it. The verify runs as one forward of the one target sequence: the pending token plus the four roots are
 * five rows of a single batch under the tree attention mask (the width of {@code neuralfree.draft.tree.nodes},
 * shorter trees padded), each row storing at its slot and roping at its depth position, and the commit compacts:
 * the accepted path's rows move onto the consecutive positions and the rows above leave, without a re-decode.
 * Every committed token
 * equals the target-only greedy loop's under the same bias, which bites the whole machinery: the tree mask's
 * visibility, the positions, the acceptance walk's draw count and the commit choreography, on the paged memory of
 * {@code neuralfree.parallel} sequences and on the plain contiguous single-sequence memory alike. The request
 * carries no {@code minTokens}: its masking stage would count the chain as constrained, and a constrained request
 * narrows the proposal to a linear one.
 */
public class TreeVerifySinglePassTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int PARALLEL = 4;
    private static final int DRAFT_MAX = 4;
    /** The tail rows the test drafter asks the target to replay at each generation's start. */
    private static final int REPLAY_ROWS = 2;
    private static final int[] ROOT_TOKENS = {5, 6, 7, 8};

    private static Path modelFile;

    @BeforeClass
    public static void writeModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(4).build();
        modelFile = SyntheticModels.llama(c).writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (modelFile != null) {
            Files.deleteIfExists(modelFile);
        }
    }

    /**
     * A drafter that proposes the fixed roots as depth-1 leaves and records the rounds' acceptance, the injection
     * suspensions and the generations' replay asks: each generation claims {@code REPLAY_ROWS} stored rows, so the
     * target re-evaluates that tail through the replay window and the fixed-width verify's commit leaves a
     * consistent drafter view behind.
     */
    private static final class FixedRootsDrafter implements TreeDraftModel {
        private final List<Integer> acceptedPerRound = new ArrayList<>();
        private final List<Integer> replayPerGeneration = new ArrayList<>();
        private int suspensions;

        @Override
        public void bindTarget(EngineModel target) {
        }

        @Override
        public int beginGeneration(int reused) {
            int replay = Math.min(reused, REPLAY_ROWS);
            replayPerGeneration.add(replay);
            return replay;
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
            return ROOT_TOKENS.length;
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
            if (suspended) {
                suspensions++;
            }
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature,
                                   SplittableRandom rng) {
            return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
        }

        @Override
        public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                                   SplittableRandom rng) {
            int n = Math.min(Math.min(maxTokens, branches), ROOT_TOKENS.length);
            n = Math.min(n, Math.max(0, maxRows));
            List<DraftTree.Node> nodes = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                nodes.add(new DraftTree.Node(ROOT_TOKENS[i], -1, 1, 0));
            }
            return new DraftTree(nodes);
        }
    }

    private static EngineConfig config(boolean paged) {
        return config(paged, EngineConfig.BackendMode.CPU);
    }

    private static EngineConfig config(boolean paged, EngineConfig.BackendMode backend) {
        MemoryConfig.Builder memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (paged) {
            memory.parallel(PARALLEL).kvUnified(true);
        }
        return EngineConfig.builder().backend(backend).memory(memory.build())
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX).build())
                .build();
    }

    /**
     * The greedy request whose sampling support the logit bias pins to the drafter's roots: every other id — the
     * end-of-generation token included, so no {@code minTokens} stage is needed — sits at a {@code -100} bias, and
     * the target cannot sample a token no root proposes, so every round matches exactly one.
     */
    private static GenerationRequest greedy(int nVocab) {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding());
        for (int t = 0; t < nVocab; t++) {
            if (t < ROOT_TOKENS[0] || t > ROOT_TOKENS[ROOT_TOKENS.length - 1]) {
                b.logitBias(t, -100.0f);
            }
        }
        return b.build();
    }

    /** The target-only loop over the same model under the same biased request. */
    private static GenerationResult plain(int[] prompt, GenerationRequest request) throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            return GenerationLoop.forModel(model, cfg).generate(prompt, request, (TokenCallback) null);
        }
    }

    private static int[] prompt() {
        return new int[] {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
    }

    private static int nVocab() throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                        .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(8)
                                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build())) {
            return model.spec().hparams().nVocab();
        }
    }

    @Test
    public void aFixedRootTreeAcceptsEveryRoundAndStaysLosslessOnPagedMemory() throws IOException {
        aFixedRootTreeAcceptsEveryRoundAndStaysLossless(config(true));
    }

    /** The same round over the plain contiguous single-sequence memory: no page table, the same committed tokens. */
    @Test
    public void aFixedRootTreeAcceptsEveryRoundAndStaysLosslessOnContiguousMemory() throws IOException {
        aFixedRootTreeAcceptsEveryRoundAndStaysLossless(config(false));
    }

    /**
     * The same round on each available GPU backend: the fixed roots make every proposal branch, so the masked
     * verify's one forward runs on the device with its one-section rope (the rows' depth positions) riding the
     * paged step of a single-section rope model, and the round stays lossless.
     */
    @Test
    public void theFixedRootTreeStaysLosslessOnGpuBackends() throws IOException {
        for (EngineConfig.BackendMode backend : new EngineConfig.BackendMode[] {EngineConfig.BackendMode.CUDA,
                EngineConfig.BackendMode.OPENCL}) {
            switch (backend) {
                case CUDA:
                    GpuTestUtil.assumeCudaAvailable();
                    break;
                case OPENCL:
                    GpuTestUtil.assumeOpenCLAvailable();
                    break;
                default:
                    Assume.assumeTrue("no device for " + backend, false);
            }
            aFixedRootTreeAcceptsEveryRoundAndStaysLossless(config(true, backend));
        }
    }

    /** The biased greedy request under {@code cfg}: tree rounds accept one root, the tokens stay the plain ones. */
    private void aFixedRootTreeAcceptsEveryRoundAndStaysLossless(EngineConfig cfg) throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(nVocab());
        GenerationResult expected = plain(prompt, request);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            FixedRootsDrafter drafter = new FixedRootsDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, cfg);
            GenerationResult actual = decoder.generate(prompt, request, (TokenCallback) null);
            assertArrayEquals("every committed token is the target-only greedy token", expected.tokens(),
                    actual.tokens());
            assertEquals("stop reason", expected.stopReason(), actual.stopReason());
            assertEquals("the generation runs to the budget", N_NEW, actual.tokens().length);
            assertTrue("tree rounds ran: " + drafter.acceptedPerRound, drafter.acceptedPerRound.size() >= 4);
            List<Integer> perRound = drafter.acceptedPerRound;
            for (int i = 0; i < perRound.size() - 1; i++) {
                assertEquals("every round but a budget-terminated last one accepts exactly one root: " + perRound,
                        1, (int) perRound.get(i));
            }
            int last = perRound.get(perRound.size() - 1);
            assertTrue("the last round accepts a root or ends on the budget's own sample: " + perRound,
                    last == 0 || last == 1);
            assertEquals("the tree batch suspended the injection each round: " + drafter.suspensions,
                    perRound.size(), drafter.suspensions);
        }
    }

    /**
     * The replay capture across tree rounds: each generation's start asks the target to re-evaluate the drafter's
     * stored tail (the same row count for a deterministic run), the lossless tokens prove the replay window landed
     * the rows where a plain prefill would have, and the tree rounds' compaction commit leaves a
     * consistent drafter view behind.
     */
    @Test
    public void repeatedGenerationsRestartFromThePromptThroughTheReplayWindow() throws IOException {
        int[] prompt = prompt();
        GenerationRequest request = greedy(nVocab());
        EngineConfig cfg = config(true);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            FixedRootsDrafter drafter = new FixedRootsDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, cfg);
            int[] first = decoder.generate(prompt, request, (TokenCallback) null).tokens();
            int roundsAfterFirst = drafter.acceptedPerRound.size();
            int[] second = decoder.generate(prompt, request, (TokenCallback) null).tokens();
            assertArrayEquals("each generation resets the memory and replays identically", first, second);
            assertTrue("the second generation ran its own rounds: " + drafter.acceptedPerRound,
                    drafter.acceptedPerRound.size() > roundsAfterFirst);
            assertEquals("one beginGeneration per generation: " + drafter.replayPerGeneration, 2,
                    drafter.replayPerGeneration.size());
            assertEquals("each generation replays the drafter's stored tail", REPLAY_ROWS,
                    (int) drafter.replayPerGeneration.get(0));
            assertEquals("the second generation replays the same window", REPLAY_ROWS,
                    (int) drafter.replayPerGeneration.get(1));
        }
    }
}

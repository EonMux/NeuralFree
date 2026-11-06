package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The single-pass tree verify over targets with recurrent layers ({@code neuralfree.draft.tree.recurrent}): the
 * synthetic qwen3next hybrid (gated-delta-net layers, one attention layer, routed MoE), the synthetic Mamba-2
 * and Mamba-1 (pure recurrent state), the plamo2 hybrid (Mamba layers and fused-QKV attention) and the kimi-linear
 * hybrid (KDA layers, MLA attention, routed MoE) run the fixed-root rounds of {@code TreeVerifySinglePassTest} — a
 * test drafter proposes four fixed tokens as depth-1 roots and the request's logit bias bans every other id, so
 * every round's sample matches one root and the round accepts exactly it. The verify runs as one masked forward of the one target
 * sequence: the recurrent layers step their per-node state through the round's depth-indexed stack (the committed
 * state stays unwritten) and capture their mixer inputs, and the commit replays the accepted path from the
 * unchanged committed state. Every committed token equals the target-only greedy loop's under the same bias, which
 * bites the whole machinery: a wrong replay corrupts the next round's pending row and the streams diverge. Two
 * unit-level checks pin the pieces: the masked forward's per-row logits equal the sequential per-path decodes of a
 * fresh model, and the committed state after a generation continues exactly like a model that decoded the accepted
 * tokens one by one. Disabling the property refuses a recurrent target by name.
 */
public class RecurrentTreeVerifyTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int PARALLEL = 4;
    private static final int DRAFT_MAX = 4;
    private static final int[] ROOT_TOKENS = {5, 6, 7, 8};
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};

    private static Path hybridFile;
    private static Path mamba2File;
    private static Path mamba1File;
    private static Path plamo2File;
    private static Path kdaFile;

    @BeforeClass
    public static void writeModels() throws IOException {
        hybridFile = SyntheticModels.qwen3next().writeTempFile();
        mamba2File = SyntheticModels.mamba2().writeTempFile();
        mamba1File = SyntheticModels.mamba().writeTempFile();
        plamo2File = SyntheticModels.plamo2().writeTempFile();
        kdaFile = SyntheticModels.kimiLinear().writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        Files.deleteIfExists(hybridFile);
        Files.deleteIfExists(mamba2File);
        Files.deleteIfExists(mamba1File);
        Files.deleteIfExists(plamo2File);
        Files.deleteIfExists(kdaFile);
    }

    /**
     * A drafter that proposes the fixed roots as depth-1 leaves and records the rounds' acceptance, as
     * {@code TreeVerifySinglePassTest} runs it.
     */
    private static final class FixedRootsDrafter implements TreeDraftModel {
        private final List<Integer> acceptedPerRound = new ArrayList<>();

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

    private static EngineConfig plainConfig() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
    }

    /** {@code plainConfig()} whose memory can snapshot its recurrent state, for per-path reference decodes. */
    private static EngineConfig snapshotConfig() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).stateSnapshots(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
    }

    /**
     * The greedy request whose sampling support the logit bias pins to the drafter's roots: every other id — the
     * end-of-generation token included — sits at a {@code -100} bias, so the target cannot sample a token no root
     * proposes and every round matches exactly one.
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

    private static int nVocab(Path modelFile) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), plainConfig())) {
            return model.spec().hparams().nVocab();
        }
    }

    /** The target-only greedy loop over the same model under the same biased request. */
    private static GenerationResult plain(Path modelFile, int[] prompt, GenerationRequest request) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), plainConfig())) {
            return GenerationLoop.forModel(model, plainConfig()).generate(prompt, request, (TokenCallback) null);
        }
    }

    private static void assertSameTokens(GenerationResult expected, GenerationResult actual, String what) {
        assertArrayEquals(what + ": every committed token is the target-only greedy token", expected.tokens(),
                actual.tokens());
        assertEquals(what + ": stop reason", expected.stopReason(), actual.stopReason());
        assertEquals(what + ": the generation runs to the budget", N_NEW, actual.tokens().length);
    }

    private void assertRoundsRan(FixedRootsDrafter drafter) {
        List<Integer> perRound = drafter.acceptedPerRound;
        assertTrue("tree rounds ran: " + perRound, perRound.size() >= 4);
        for (int i = 0; i < perRound.size() - 1; i++) {
            assertEquals("every round but a budget-terminated last one accepts exactly one root: " + perRound,
                    1, (int) perRound.get(i));
        }
    }

    /** The forced generation under {@code cfg}: the tree rounds' committed tokens equal the plain loop's. */
    private void assertLossless(Path modelFile, EngineConfig cfg, GenerationRequest request,
                                GenerationResult expected) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), cfg)) {
            FixedRootsDrafter drafter = new FixedRootsDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, cfg);
            GenerationResult actual = decoder.generate(PROMPT, request, (TokenCallback) null);
            assertSameTokens(expected, actual, cfg.backend() + " " + (cfg.memory().parallel() > 1 ? "paged"
                    : "contiguous"));
            assertRoundsRan(drafter);
        }
    }

    // ================================================================== token forcing

    @Test
    public void theFixedRootTreeStaysLosslessOnTheHybridOverCpuMemory() throws IOException {
        GenerationRequest request = greedy(nVocab(hybridFile));
        GenerationResult expected = plain(hybridFile, PROMPT, request);
        assertLossless(hybridFile, config(true, EngineConfig.BackendMode.CPU), request, expected);
        assertLossless(hybridFile, config(false, EngineConfig.BackendMode.CPU), request, expected);
    }

    @Test
    public void theFixedRootTreeStaysLosslessOnTheMamba2OverCpuMemory() throws IOException {
        GenerationRequest request = greedy(nVocab(mamba2File));
        GenerationResult expected = plain(mamba2File, PROMPT, request);
        assertLossless(mamba2File, config(true, EngineConfig.BackendMode.CPU), request, expected);
        assertLossless(mamba2File, config(false, EngineConfig.BackendMode.CPU), request, expected);
    }

    @Test
    public void theFixedRootTreeStaysLosslessOnTheMamba1OverCpuMemory() throws IOException {
        GenerationRequest request = greedy(nVocab(mamba1File));
        GenerationResult expected = plain(mamba1File, PROMPT, request);
        assertLossless(mamba1File, config(true, EngineConfig.BackendMode.CPU), request, expected);
        assertLossless(mamba1File, config(false, EngineConfig.BackendMode.CPU), request, expected);
    }

    @Test
    public void theFixedRootTreeStaysLosslessOnPlamo2OverCpuMemory() throws IOException {
        GenerationRequest request = greedy(nVocab(plamo2File));
        GenerationResult expected = plain(plamo2File, PROMPT, request);
        assertLossless(plamo2File, config(true, EngineConfig.BackendMode.CPU), request, expected);
        assertLossless(plamo2File, config(false, EngineConfig.BackendMode.CPU), request, expected);
    }

    @Test
    public void theFixedRootTreeStaysLosslessOnKdaOverCpuMemory() throws IOException {
        GenerationRequest request = greedy(nVocab(kdaFile));
        GenerationResult expected = plain(kdaFile, PROMPT, request);
        assertLossless(kdaFile, config(true, EngineConfig.BackendMode.CPU), request, expected);
        assertLossless(kdaFile, config(false, EngineConfig.BackendMode.CPU), request, expected);
    }

    /**
     * The same rounds on each available GPU backend, every covered family alike: the recurrent layers' tree steps
     * run the device stack copies and the state-arg swaps of the round, and the commits stay lossless.
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
            for (Path modelFile : new Path[] {hybridFile, mamba2File, mamba1File, plamo2File, kdaFile}) {
                GenerationRequest request = greedy(nVocab(modelFile));
                GenerationResult expected = plain(modelFile, PROMPT, request);
                assertLossless(modelFile, config(true, backend), request, expected);
            }
        }
    }

    // ================================================================== per-node outputs

    /**
     * The masked verify forward's rows equal the sequential per-path decodes of a fresh model: the pending row's
     * logits are the target's continuation of the committed prefix, and a node's row is the logits after decoding
     * exactly its root path, one token at a time — the tree step's stack made every node's recurrent state the
     * state along its own path, and the tree mask made its attention see only that path.
     */
    @Test
    public void theVerifyRowsEqualTheSequentialPathDecodes() throws IOException {
        try (EngineModel model = LLMModelFactory.open(hybridFile.toString(),
                config(false, EngineConfig.BackendMode.CPU));
             EngineModel reference = LLMModelFactory.open(hybridFile.toString(), snapshotConfig())) {
            ForwardBackend backend = model.forwardBackend();
            // the committed prefix: the prompt prefill, then the pending token the proposal roots from
            backend.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1});
            int token = 9;
            int pos = PROMPT.length;
            // rows: 0 the pending token, 1 and 2 the depth-1 roots, 3 the child of node 0, 4 the child of node 1
            DraftTree proposal = new DraftTree(Arrays.asList(
                    new DraftTree.Node(ROOT_TOKENS[0], -1, 1, 0),
                    new DraftTree.Node(ROOT_TOKENS[1], -1, 1, 0),
                    new DraftTree.Node(4, 0, 2, 0),
                    new DraftTree.Node(ROOT_TOKENS[3], 1, 2, 0)));
            int width = DRAFT_MAX;
            float[][] rowOut;
            try (RecurrentTreeRound round = backend.openRecurrentTreeRound(width + 1,
                    proposal.nodes().size() + 1)) {
                rowOut = backend.forwardBatch(verifyBatch(proposal, token, pos, width));
            }
            // the sequential per-path references: decode the pending token and each root path, then rewind the
            // snapshot to the prefill for the second chain
            ForwardBackend ref = reference.forwardBackend();
            ModelMemory refMemory = ref.memory();
            ref.forwardRows(PROMPT, 0, new int[]{PROMPT.length - 1});
            refMemory.snapshot();
            float[] row0 = ref.forwardRows(new int[]{token}, pos, new int[]{0})[0];
            float[] row1 = ref.forwardRows(new int[]{ROOT_TOKENS[0]}, pos + 1, new int[]{0})[0];
            float[] row3 = ref.forwardRows(new int[]{4}, pos + 2, new int[]{0})[0];
            refMemory.restore();
            // the second chain re-decodes the pending token from the rewound state: its logits come out the same
            float[] row0Again = ref.forwardRows(new int[]{token}, pos, new int[]{0})[0];
            float[] row2 = ref.forwardRows(new int[]{ROOT_TOKENS[1]}, pos + 1, new int[]{0})[0];
            float[] row4 = ref.forwardRows(new int[]{ROOT_TOKENS[3]}, pos + 2, new int[]{0})[0];
            assertArrayEquals("row 0 continues the committed prefix", row0, rowOut[0], 0f);
            assertArrayEquals("the rewound pending decode repeats its logits", row0Again, rowOut[0], 0f);
            assertArrayEquals("row 1 is the decode of the first root", row1, rowOut[1], 0f);
            assertArrayEquals("row 2 is the decode of the second root", row2, rowOut[2], 0f);
            assertArrayEquals("row 3 is the decode along the first root's path", row3, rowOut[3], 0f);
            assertArrayEquals("row 4 is the decode along the second root's path", row4, rowOut[4], 0f);
        }
    }

    /**
     * The verify batch of a full-width proposal after the pending {@code token} at {@code pos}, the exact round's
     * shape: row 0 stores the pending token, row {@code i + 1} node {@code i} at {@code pos + i + 1}, every row
     * ropes at {@code pos +} its depth and its ancestor mask names the pending row (the rows attend the pending
     * token they were drafted from, and the recurrent stack's depth-0 slot is its state).
     */
    private static ForwardBatch verifyBatch(DraftTree proposal, int token, int pos, int width) {
        List<DraftTree.Node> nodes = proposal.nodes();
        int rows = width + 1;
        int[][] rope = new int[1][rows];
        long[] mask = new long[rows];
        ForwardBatch.Builder batch = ForwardBatch.builder(rows);
        mask[0] = 1L;
        rope[0][0] = pos;
        batch.add(token, pos, 0, true);
        for (int i = 0; i < nodes.size(); i++) {
            DraftTree.Node n = nodes.get(i);
            int row = i + 1;
            long bits = (1L << row) | 1L;
            for (int p = n.parent(); p >= 0; p = nodes.get(p).parent()) {
                bits |= 1L << (p + 1);
            }
            mask[row] = bits;
            rope[0][row] = pos + n.depth();
            batch.add(n.token(), pos + row, 0, true);
        }
        batch.ropePositions(rope);
        batch.treeMask(mask);
        return batch.build();
    }


    // ================================================================== committed state

    /**
     * The state a generation leaves continues exactly like the accepted tokens' own: after the forced generation,
     * a further decode of the speculative model's memory matches a fresh model that decodes the prompt and the
     * stored committed tokens one by one (the last emitted token's row the budget cut off is not in memory yet) —
     * the commit's accepted-path replay left the recurrent state where a sequential decode would have.
     */
    @Test
    public void theCommittedStateContinuesLikeTheAcceptedPathReplay() throws IOException {
        GenerationRequest request = greedy(nVocab(hybridFile));
        int[] committed;
        float[] continued;
        int nPast;
        try (EngineModel model = LLMModelFactory.open(hybridFile.toString(),
                config(false, EngineConfig.BackendMode.CPU))) {
            FixedRootsDrafter drafter = new FixedRootsDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, cfg());
            GenerationResult result = decoder.generate(PROMPT, request, (TokenCallback) null);
            assertRoundsRan(drafter);
            committed = result.tokens();
            nPast = model.forwardBackend().memory().sequences().sequence(0).nPast();
            int stored = nPast - PROMPT.length;
            assertTrue("the rounds committed accepted tokens into the memory: " + stored, stored >= 1);
            assertTrue("the memory holds no token the generation did not emit: " + stored + " of "
                    + committed.length, stored <= committed.length);
            continued = model.forwardBackend().forwardRows(new int[]{9}, nPast, new int[]{0})[0];
            committed = Arrays.copyOf(committed, stored);
        }
        try (EngineModel reference = LLMModelFactory.open(hybridFile.toString(), plainConfig())) {
            ForwardBackend ref = reference.forwardBackend();
            int p = 0;
            for (int t : PROMPT) {
                ref.forwardRows(new int[]{t}, p++, new int[]{0});
            }
            for (int t : committed) {
                ref.forwardRows(new int[]{t}, p++, new int[]{0});
            }
            assertArrayEquals("the committed state equals the accepted tokens' own", ref.forwardRows(new int[]{9}, p,
                    new int[]{0})[0], continued, 0f);
        }
    }

    private static EngineConfig cfg() {
        return config(false, EngineConfig.BackendMode.CPU);
    }

    // ================================================================== refusal

    /** Opting the recurrent tree verify out refuses a recurrent target by name instead of running unsafe rounds. */
    @Test
    public void optingOutRefusesARecurrentTargetByName() throws IOException {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX)
                        .treeRecurrent(false).build())
                .build();
        try (EngineModel model = LLMModelFactory.open(hybridFile.toString(), cfg)) {
            FixedRootsDrafter drafter = new FixedRootsDrafter();
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, cfg);
            try {
                decoder.generate(PROMPT, greedy(model.spec().hparams().nVocab()), (TokenCallback) null);
                fail("the disabled recurrent tree verify must refuse the generation");
            } catch (UnsupportedOperationException e) {
                assertTrue("the refusal names the property: " + e.getMessage(),
                        e.getMessage().contains(SpeculativeOptions.PROP_TREE_RECURRENT));
            }
        }
    }
}

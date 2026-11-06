package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tree verify inside the batch engine: a scripted {@link BatchDrafter} answers {@link BatchDrafter#draftTreeBatch}
 * with branched proposals — the target's own greedy continuation as the best chain plus a wrong sibling the walk
 * must reject — and the engine verifies every sequence's tree in one masked forward, walking the rows through each
 * sequence's own sampling chain. Every finished sequence carries the same tokens as the same prompt run plain,
 * one at a time, through {@link GenerationLoop}, on CPU and on the OpenCL GPU (whose multi-segment masked step
 * uploads each segment's relativized tree mask and rope positions).
 */
public class BatchEngineTreeVerifyTest {

    private static final int N_TOKENS = 24;
    private static final int N_NEW = 8;

    private static EngineConfig.Builder config(int nCtx) {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        return EngineConfig.builder().memory(mc);
    }

    /** Greedy decoding over a bounded budget: the same prompt yields the same tokens on every path. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    /** A prompt whose token n-grams repeat, so a scripted drafter can follow the stream round after round. */
    private static int[] tokens() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i % 5) * 7;
        }
        return t;
    }

    /** The reference: the same prompt and request through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, int nCtx, int[] prompt, GenerationRequest request)
            throws IOException {
        MemoryConfig mc = MemoryConfig.builder().nCtx(nCtx).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    /** DONE with the reference result, or fails naming the engine error and where it was thrown. */
    private static void assertDone(BatchSequence sequence) {
        assertTrue("expected DONE but was " + sequence.state()
                        + (sequence.error() == null ? "" : ": " + sequence.error() + " at "
                                + Arrays.stream(sequence.error().getStackTrace()).limit(10)
                                        .map(StackTraceElement::toString)
                                        .collect(java.util.stream.Collectors.joining("\n  "))),
                sequence.state() == BatchSequence.State.DONE);
    }

    /** Tokens, text and stop reason of a batch result match the sequential reference exactly. */
    private static void assertResultsMatch(GenerationResult reference, GenerationResult batched) {
        assertEquals("stop reason", reference.stopReason(), batched.stopReason());
        assertEquals("token count", reference.tokenCount(), batched.tokenCount());
        assertArrayEquals("token ids", reference.tokens(), batched.tokens());
        assertEquals("text", reference.text(), batched.text());
    }

    private static SpeculativeOptions treeOptions() {
        return SpeculativeOptions.builder().treeVerify(true).build();
    }

    @Test
    public void aLongPromptChunksIntoTheMaskedVerifyStepsBesideATreeRound() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // an 8-row ubatch: the 60-token prompt's chunks span several steps, and once the short sequence decodes
        // with an armed tree those steps carry both the tree rows and the admitted chunk rows
        EngineConfig cfg = config(96).build();
        GenerationRequest request = request(N_NEW);
        int[] shortPrompt = tokens();
        int[] longPrompt = new int[60];
        for (int i = 0; i < longPrompt.length; i++) {
            longPrompt[i] = 4 + (i % 5) * 7;
        }
        GenerationResult refShort = reference(model, 96, shortPrompt, request);
        GenerationResult refLong = reference(model, 96, longPrompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(refShort.tokens(), false);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()))) {
            BatchSequence a = engine.submit(shortPrompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(longPrompt, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(refShort, a.result());
            assertResultsMatch(refLong, b.result());
            // the short sequence's tree rounds ran while the long prompt was still admitting its chunks
            assertTrue("the drafter proposed trees", drafter.proposals > 0);
            assertTrue(engine.stats().specAccepted() > 0);
        }
    }

    @Test
    public void treeProposalsVerifyThroughTheTargetSampleAndKeepTheReferenceResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(ref.tokens(), false);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            // both sequences ran branched rounds: the full budget arms every proposed node
            assertTrue("the drafter proposed trees", drafter.proposals > 0);
            assertTrue("the proposals were branched", drafter.branched > 0);
            assertTrue(drafter.accepted > 0);
            BatchStats stats = engine.stats();
            assertEquals("the engine counted the armed nodes", drafter.proposedNodes, stats.specDrafted());
            // the walks accept the armed chain nodes; the armed decoys (one per branched proposal) never accept,
            // and the walks' own samples carry each round forward
            assertTrue("the walks accepted the armed chain nodes", stats.specAccepted() > 0);
            assertTrue("the armed decoys never accept", stats.specAccepted() < stats.specDrafted());
            assertTrue("the proposals were verified in rounds", stats.specRounds() > 0);
        }
    }

    @Test
    public void aRowBudgetCutsTheTreesRaggedlyAndStaysLossless() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(ref.tokens(), false);
        SpeculativeOptions options = SpeculativeOptions.builder().treeVerify(true).treeRows("4").build();

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, options))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            // the step's four-row budget over two decoding sequences arms a ragged split, three nodes and one,
            // and the cut prefixes verify as losslessly as the full trees
            assertTrue(drafter.proposals > 0);
            BatchStats stats = engine.stats();
            assertTrue("the budget cut nodes away", stats.specDrafted() < drafter.proposedNodes);
            // the walks accept the armed chain nodes; the armed decoys never accept
            assertTrue("the walks accepted the armed chain nodes", stats.specAccepted() > 0);
            assertTrue("the armed decoys never accept", stats.specAccepted() < stats.specDrafted());
        }
    }

    @Test
    public void wrongTreesRejectWithoutChangingTheResult() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(ref.tokens(), true);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            // every round rejected at its first tree row: the rollback left the plain result
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            assertTrue("the drafter proposed trees", drafter.proposedNodes > 0);
            assertEquals("the walks accepted nothing", 0, drafter.accepted);
            BatchStats stats = engine.stats();
            assertEquals(drafter.proposedNodes, stats.specDrafted());
            assertEquals(0, stats.specAccepted());
        }
    }

    @Test
    public void theDraftModelBuildsTheTreesAndKeepsTheReferenceResult() throws IOException {
        draftModelTreesOn(EngineConfig.BackendMode.CPU);
    }

    @Test
    public void theDraftModelBuildsTheTreesOnTheOpenCLGpuToo() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        draftModelTreesOn(EngineConfig.BackendMode.OPENCL);
    }

    private static void draftModelTreesOn(EngineConfig.BackendMode mode) throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // the target drafts for itself over two branches a node: its masked draft forwards build the trees the
        // budget cuts, and its own greedy picks verify nearly whole
        MemoryConfig mc = MemoryConfig.builder().nCtx(96).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc).backend(mode)
                .draftType(EngineConfig.DraftType.MODEL).draftModelPath(model.toString())
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeBranches(2).build())
                .build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 96, prompt, request);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg)) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            BatchStats stats = engine.stats();
            assertTrue("the draft model drafted trees", stats.specDrafted() > 0);
            assertTrue("the rounds verified", stats.specRounds() > 0);
            assertTrue("the walks accepted the greedy chains", stats.specAccepted() > 0);
            assertTrue("accepted cannot exceed drafted", stats.specAccepted() <= stats.specDrafted());
        }
    }

    @Test
    public void treeProposalsKeepTheReferenceResultOnTheOpenCLGpu() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config(64).backend(EngineConfig.BackendMode.OPENCL).build();
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, 64, prompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(ref.tokens(), false);

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            BatchStats stats = engine.stats();
            assertEquals("the full budget armed every proposed node", drafter.proposedNodes,
                    stats.specDrafted());
            assertTrue("the walks accepted the armed chain nodes on the GPU too", stats.specAccepted() > 0);
            assertTrue("the armed decoys never accept", stats.specAccepted() < stats.specDrafted());
        }
    }

    /**
     * Drafts each request a tree whose best chain is the reference's greedy continuation and whose sibling under
     * the root carries a wrong token (a decoy the walk must reject without ever sampling it): in the true mode the
     * walk accepts the whole chain, with {@code wrong} the chain's tokens differ from the target's own next tokens
     * and every round rejects at its first tree row. The cursor follows the sequence's stream through
     * {@code accept}.
     */
    private static final class ScriptedTreeDrafter implements BatchDrafter {

        private final int[] continuation;
        private final boolean wrong;
        private final int[] cursor = new int[8];
        private int proposals;
        private int proposedNodes;
        private int branched;
        private int accepted;

        ScriptedTreeDrafter(int[] continuation, boolean wrong) {
            this.continuation = continuation;
            this.wrong = wrong;
        }

        @Override
        public boolean proposesTrees() {
            return true;
        }

        /** Unused: the tree verify routes every proposal through {@link #draftTreeBatch}. */
        @Override
        public int[] draft(int seq, int lastToken, int n) {
            return new int[0];
        }

        @Override
        public TreeProposal[] draftTreeBatch(List<Request> requests) {
            TreeProposal[] out = new TreeProposal[requests.size()];
            for (int i = 0; i < requests.size(); i++) {
                Request r = requests.get(i);
                // re-anchor the cursor on the token the target just sampled (the stream is the reference's)
                while (cursor[r.seq()] < continuation.length && continuation[cursor[r.seq()]] != r.lastToken()) {
                    cursor[r.seq()]++;
                }
                int n = Math.min(r.limit(), continuation.length - cursor[r.seq()] - 1);
                out[i] = tree(r.seq(), n);
            }
            return out;
        }

        /**
         * The request's tree of at most {@code limit} nodes: the chain of the next stream tokens scored
         * {@code -0.1, -0.2, -0.3, ...}, and once two nodes fit the decoy sibling under the root scored between
         * the root's chain child and the chain's second node — the node list stays parents-first (the decoy
         * shifts the chain's list indices: chain node {@code i >= 2} sits at index {@code i + 1} and points at
         * {@code i}) and the scores descend along it, so the budget's cut stays a prefix of the true chain.
         */
        private TreeProposal tree(int seq, int limit) {
            proposals++;
            int t = Math.min(limit, continuation.length - cursor[seq] - 1);
            List<DraftTree.Node> nodes = new ArrayList<>();
            List<Double> scores = new ArrayList<>();
            int truth0 = t > 0 ? continuation[cursor[seq] + 1] : 0;
            int chain = t >= 2 ? t - 1 : t; // a branched proposal spends one of its nodes on the decoy
            for (int i = 0; i < chain; i++) {
                int truth = continuation[cursor[seq] + 1 + i];
                int parent = i == 0 ? -1 : (i == 1 ? 0 : i);
                nodes.add(new DraftTree.Node(wrong ? flip(truth) : truth, parent, i + 1, 0));
                scores.add(-(i + 1) * 0.1);
                if (i == 0 && t >= 2) {
                    // the decoy never carries the token the walk looks for at the root
                    nodes.add(new DraftTree.Node(flip(truth0), 0, 2, 0));
                    scores.add(-0.15);
                    branched++;
                }
            }
            proposedNodes += nodes.size();
            double[] pathScores = new double[scores.size()];
            for (int i = 0; i < pathScores.length; i++) {
                pathScores[i] = scores.get(i);
            }
            return new TreeProposal(new DraftTree(nodes), pathScores);
        }

        private static int flip(int token) {
            return token == 4 ? 5 : 4;
        }

        @Override
        public void accept(int seq, int nAccepted) {
            accepted += nAccepted;
            cursor[seq] += 1 + nAccepted;
        }
    }
}

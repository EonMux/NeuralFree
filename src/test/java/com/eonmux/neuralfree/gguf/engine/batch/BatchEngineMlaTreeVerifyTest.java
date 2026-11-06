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
import static org.junit.Assert.fail;

/**
 * Tree verify over the batch engine's MLA layers: a deepseek2's masked step carries one segment per decoding
 * sequence, whose rows rope at their depth positions under their ancestor bits — the CPU slices each segment's
 * rows out of the shared mask ({@code CpuMlaMixer}), the GPU uploads each segment's relativized mask rows and
 * packs its fresh-row count into the launch flags ({@code GpuMlaMixer}). The branched scripted trees verify in
 * one masked forward per step beside the other sequence's rows, and every finished sequence carries the same
 * tokens as the same prompt run plain, one at a time, through {@link GenerationLoop}, on CPU and on the GPUs.
 * An MLA draft model builds its trees over the same masked launches. The layers whose attention takes no tree
 * mask keep refusing by name (a deepseek32's DSA indexer).
 */
public class BatchEngineMlaTreeVerifyTest {

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

    /** Both sequences verify branched trees on the MLA layers and land on the reference tokens. */
    private static void verifyOn(EngineConfig cfg, Path model, int nCtx) throws IOException {
        GenerationRequest request = request(N_NEW);
        int[] prompt = tokens();
        GenerationResult ref = reference(model, nCtx, prompt, request);
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(ref.tokens());

        try (GGUFModel two = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()))) {
            BatchSequence a = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertDone(a);
            assertDone(b);
            assertResultsMatch(ref, a.result());
            assertResultsMatch(ref, b.result());
            // both sequences ran branched rounds over the MLA layers' masked step
            assertTrue("the drafter proposed trees", drafter.proposals > 0);
            assertTrue("the proposals were branched", drafter.branched > 0);
            BatchStats stats = engine.stats();
            assertEquals("the engine counted the armed nodes", drafter.proposedNodes, stats.specDrafted());
            assertTrue("the walks accepted the armed chain nodes", stats.specAccepted() > 0);
            assertTrue("the armed decoys never accept", stats.specAccepted() < stats.specDrafted());
        }
    }

    @Test
    public void mlaTreeProposalsVerifyAndKeepTheReferenceResult() throws IOException {
        Path model = SyntheticModels.deepseek2().writeTempFile();
        verifyOn(config(64).build(), model, 64);
    }

    @Test
    public void mlaTreeProposalsVerifyOnTheOpenCLGpu() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = SyntheticModels.deepseek2().writeTempFile();
        verifyOn(config(64).backend(EngineConfig.BackendMode.OPENCL).build(), model, 64);
    }

    @Test
    public void mlaTreeProposalsVerifyOnTheCudaGpu() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        Path model = SyntheticModels.deepseek2().writeTempFile();
        verifyOn(config(64).backend(EngineConfig.BackendMode.CUDA).build(), model, 64);
    }

    /** An MLA draft model builds the trees its masked draft forwards run over, and the rounds verify losslessly. */
    @Test
    public void theMlaDraftModelBuildsTheTrees() throws IOException {
        Path model = SyntheticModels.deepseek2().writeTempFile();
        // the target drafts for itself over two branches a node: the draft model's masked forwards build the
        // trees the budget cuts, through the same masked MLA launches the target's verify step runs
        MemoryConfig mc = MemoryConfig.builder().nCtx(96).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc)
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

    /** The DSA indexer layers of a deepseek32 refuse the masked batch by name. */
    @Test
    public void theDsaLayersRefuseTheMaskedBatchByName() throws IOException {
        Path model = SyntheticModels.deepseek32().writeTempFile();
        EngineConfig cfg = config(64).build();
        ScriptedTreeDrafter drafter = new ScriptedTreeDrafter(new int[0]);
        try (GGUFModel two = new GGUFModel(model.toString(), cfg)) {
            BatchEngine.forModel(two, cfg, drafter, BatchSpec.of(3, 0, treeOptions()));
            fail("a deepseek32's DSA layers must refuse a tree-masked batch");
        } catch (UnsupportedOperationException e) {
            assertTrue("the refusal names DSA: " + e.getMessage(), e.getMessage().contains("DSA"));
            assertTrue("the refusal names the property: " + e.getMessage(),
                    e.getMessage().contains(SpeculativeOptions.PROP_TREE));
        }
    }

    /**
     * Drafts each request a tree whose best chain is the reference's greedy continuation and whose sibling under
     * the root carries a wrong token (a decoy the walk must reject without ever sampling it): the walk accepts
     * the whole chain and the decoy never carries the token the walk looks for at the root. The cursor follows
     * the sequence's stream through {@code accept}.
     */
    private static final class ScriptedTreeDrafter implements BatchDrafter {

        private final int[] continuation;
        private final int[] cursor = new int[8];
        private int proposals;
        private int proposedNodes;
        private int branched;
        private int accepted;

        ScriptedTreeDrafter(int[] continuation) {
            this.continuation = continuation;
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
                nodes.add(new DraftTree.Node(truth, parent, i + 1, 0));
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

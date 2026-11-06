package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A separate draft model's multi-branch proposal over a recurrent target: the draft model (a small model sharing
 * the target's vocabulary but not its architecture) runs one draft sequence per target sequence on its own paged
 * memory, splits a branch whenever a secondary candidate outranks {@code neuralfree.draft.p.split} (0 here, so every
 * positive candidate splits), and the target verifies the proposal in one masked forward over its recurrent layers.
 * Every committed token equals the target-only greedy loop's under the same logit bias, on the paged memory of the
 * target and on each GPU backend, which bites the whole machinery: the drafter's prefix re-alignments after a
 * round's commit (the copies the splits made leave the draft's other sequences to drop), the commit's accepted-path
 * replay into the committed state, and the next round's proposal from that state.
 */
public class DraftModelTreeRecurrentTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int PARALLEL = 4;
    private static final int DRAFT_MAX = 4;
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};

    private static Path hybridFile;
    private static Path mamba2File;
    private static Path kdaFile;
    private static Path qwen2DraftFile;
    private static Path mambaDraftFile;
    private static Path plamo2DraftFile;

    @BeforeClass
    public static void writeModels() throws IOException {
        hybridFile = SyntheticModels.qwen3next().writeTempFile();
        mamba2File = SyntheticModels.mamba2().writeTempFile();
        kdaFile = SyntheticModels.kimiLinear().writeTempFile();
        // each draft shares its target's vocabulary (the factory requires it) but is another architecture:
        // a dense qwen2 transformer drafts for the hybrid, Mamba-1 for mamba2, PLaMo-2 for the kimi-linear model
        qwen2DraftFile = SyntheticModels.qwen2Mtp(SyntheticModels.Config.qwen2(), 0).writeTempFile();
        mambaDraftFile = SyntheticModels.mamba().writeTempFile();
        plamo2DraftFile = SyntheticModels.plamo2(
                SyntheticModels.Config.plamo2().toBuilder().nLayer(2).seed(61L).build()).writeTempFile();
    }

    @AfterClass
    public static void deleteModels() throws IOException {
        for (Path f : new Path[] {hybridFile, mamba2File, kdaFile, qwen2DraftFile, mambaDraftFile, plamo2DraftFile}) {
            if (f != null) {
                Files.deleteIfExists(f);
            }
        }
    }

    /** The draft model paired with {@code targetFile} by {@link #writeModels}. */
    private static Path draftFor(Path targetFile) {
        if (targetFile.equals(hybridFile)) {
            return qwen2DraftFile;
        }
        return targetFile.equals(mamba2File) ? mambaDraftFile : plamo2DraftFile;
    }

    /** The target's memory: paged unified, one sequence per draft branch. */
    private static EngineConfig targetConfig(EngineConfig.BackendMode backend) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(PARALLEL).kvUnified(true).build();
        return EngineConfig.builder().backend(backend).memory(memory)
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX)
                        .pSplit(0.0).build())
                .build();
    }

    /** The session's configuration: the target's with the paired draft model's path. */
    private static EngineConfig sessionConfig(Path targetFile, EngineConfig.BackendMode backend) {
        return targetConfig(backend).toBuilder().draftModelPath(draftFor(targetFile).toString()).build();
    }

    private static EngineConfig plainConfig() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).ubatch(16)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
    }

    /**
     * The greedy request whose sampling support the logit bias pins to four token ids, so the target cannot confirm
     * a proposed path its own distribution would never take.
     */
    private static GenerationRequest greedy(int nVocab) {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding());
        for (int t = 0; t < nVocab; t++) {
            if (t < 5 || t > 8) {
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
    private static GenerationResult plain(Path modelFile, GenerationRequest request) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), plainConfig())) {
            return GenerationLoop.forModel(model, plainConfig()).generate(PROMPT, request, (TokenCallback) null);
        }
    }

    /** The forced generation of one target under {@code backend}: the committed tokens equal the plain loop's. */
    private static void assertLossless(Path targetFile, EngineConfig.BackendMode backend) throws IOException {
        GenerationRequest request = greedy(nVocab(targetFile));
        GenerationResult expected = plain(targetFile, request);
        try (EngineModel model = LLMModelFactory.open(targetFile.toString(), targetConfig(backend));
             SpeculativeSession session = LLMModelFactory.speculate(model, sessionConfig(targetFile, backend))) {
            GenerationResult actual = session.generate(PROMPT, request, (TokenCallback) null);
            assertArrayEquals(backend + " " + targetFile.getFileName()
                    + ": every committed token is the target-only greedy token", expected.tokens(), actual.tokens());
            assertEquals(backend + ": stop reason", expected.stopReason(), actual.stopReason());
            assertEquals(backend + ": the generation runs to the budget", N_NEW, actual.tokens().length);
            assertTrue(backend + " " + targetFile.getFileName() + ": the draft model proposed tokens: "
                    + session.lastStats(), session.lastStats().drafted() > 0);
        }
    }

    // ================================================================== losslessness

    @Test
    public void theSeparateDraftModelTreeStaysLosslessOnCpuMemory() throws IOException {
        for (Path targetFile : new Path[] {hybridFile, mamba2File, kdaFile}) {
            assertLossless(targetFile, EngineConfig.BackendMode.CPU);
        }
    }

    /** The same rounds on each available GPU backend: the draft model opens beside the target on the device. */
    @Test
    public void theSeparateDraftModelTreeStaysLosslessOnGpuBackends() throws IOException {
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
            for (Path targetFile : new Path[] {hybridFile, mamba2File, kdaFile}) {
                assertLossless(targetFile, backend);
            }
        }
    }

    // ================================================================== the drafter's own sequences

    /**
     * The branching proposal and the drafter's own memory across rounds, driven directly: the drafter runs one
     * draft sequence per target sequence and its proposal splits roots while secondary candidates rank over
     * {@code pSplit} (0 here); after the target commits an accepted token, the next proposal re-aligns the draft's
     * first sequence by the common prefix (the committed token stays) and empties the sequences the splits copied
     * into, so the second tree starts from the committed state.
     */
    @Test
    public void theDraftModelBranchesAndRealignsItsSequencesAcrossRounds() throws IOException {
        try (EngineModel target = LLMModelFactory.open(hybridFile.toString(),
                targetConfig(EngineConfig.BackendMode.CPU));
             EngineModel draft = LLMModelFactory.open(qwen2DraftFile.toString(),
                     targetConfig(EngineConfig.BackendMode.CPU))) {
            ModelDrafter drafter = new ModelDrafter(draft, 0.0f, false, 0.0);
            drafter.bindTarget(target);
            assertEquals("one draft sequence per target sequence", PARALLEL, drafter.branches());
            ForwardBackend targetBackend = target.forwardBackend();
            ForwardBatch.Builder prefill = ForwardBatch.builder(PROMPT.length);
            for (int i = 0; i < PROMPT.length; i++) {
                prefill.add(PROMPT[i], i, 0, i == PROMPT.length - 1);
            }
            targetBackend.forwardBatch(prefill.build());
            int pos = PROMPT.length;
            DraftTree first = drafter.draftTree(9, DRAFT_MAX, PARALLEL, DRAFT_MAX, 0f, new SplittableRandom(7L));
            assertTrue("the first proposal drafted: " + first.nodes().size(), first.nodes().size() >= 1);
            assertTrue("the proposal branched into sibling roots: " + first.nodes(), siblingRoots(first) >= 2);
            // the target commits the pending token's continuation; the draft's copied sequences leave
            targetBackend.forwardBatch(ForwardBatch.builder(1).add(5, pos, 0, true).build());
            DraftTree second = drafter.draftTree(5, DRAFT_MAX, PARALLEL, DRAFT_MAX, 0f, new SplittableRandom(8L));
            assertTrue("the second proposal drafted from the committed state: " + second.nodes().size(),
                    second.nodes().size() >= 1);
            SequenceMemory draftSeqs = draft.forwardBackend().memory().sequences();
            SequenceState targetSeq = targetBackend.memory().sequences().sequence(0);
            int committed = targetSeq.nPast();
            for (int i = 0; i < committed; i++) {
                assertEquals("the draft's first sequence carries the target's committed prefix", targetSeq.token(i),
                        draftSeqs.sequence(0).token(i));
            }
            assertEquals("the draft's first sequence decodes the pending token at the target's position", 5,
                    draftSeqs.sequence(0).token(committed));
            for (int s = 1; s < PARALLEL; s++) {
                SequenceState branchSeq = draftSeqs.sequence(s);
                int stored = branchSeq.nPast();
                if (stored > 0) {
                    assertTrue("a draft branch sequence holds at least the committed prefix and its own proposal",
                            stored >= committed + 1);
                    for (int i = 0; i < committed; i++) {
                        assertEquals("draft branch sequence " + s + " carries the committed prefix",
                                targetSeq.token(i), branchSeq.token(i));
                    }
                }
            }
        }
    }

    /** The nodes of {@code tree} whose parent is the pending token itself: one per branch the proposal split. */
    private static int siblingRoots(DraftTree tree) {
        List<DraftTree.Node> nodes = tree.nodes();
        int roots = 0;
        for (DraftTree.Node n : nodes) {
            if (n.parent() < 0) {
                roots++;
            }
        }
        return roots;
    }
}

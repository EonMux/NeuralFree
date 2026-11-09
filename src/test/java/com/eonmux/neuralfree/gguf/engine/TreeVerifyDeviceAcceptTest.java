package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The single-pass round's acceptance walked on the device ({@code LLM_TreeAccept} over the verify rows' argmax ids
 * instead of a logits readback), against the same round walked on the host over the same batch. The drafter
 * proposes seeded random trees — a chain of a calibration run's greedy tokens fronted by random decoy roots, some
 * repeating the chain's next token so several roots match one sample and the walk keeps the smallest index — so
 * every round's walk matches through real frontiers, and the accepted path often sits above the batch's leftmost
 * rows, whose commit gathers real rows down. The request carries no logit bias (an argmax-only sampler is what the
 * device path serves), and the runs stay lossless against the target's plain greedy loop. The device and the host
 * walks see the same floats — the batch and its argmax ids are identical, only where the walk runs differs — so
 * their tokens and per-round acceptance match exactly.
 */
public class TreeVerifyDeviceAcceptTest {

    private static final int N_CTX = 256;
    private static final int N_NEW = 16;
    private static final int PARALLEL = 2;
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

    /**
     * A drafter whose every proposal is a seeded random tree: a chain of the calibration run's greedy tokens (which
     * the round then matches, exercising deep acceptance through real frontiers) fronted by one or two random
     * decoy roots — some repeating the chain's next token, so several roots match one sample and the walk keeps
     * the smallest index — and sometimes a random child off a random node. The fronted decoys make every proposal
     * branching and put the chain above the batch's leftmost rows, so an accepted path's commit gathers real rows
     * down. The drafter advances its chain cursor by the tokens each round emitted (the pending sample plus the
     * accepted nodes), so the next round proposes the calibration run's continuation, and records every round's
     * acceptance.
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
            int budget = Math.min(Math.min(maxTokens, branches), maxRows);
            List<DraftTree.Node> nodes = new ArrayList<>();
            int decoys = 1 + rng.nextInt(2);
            for (int i = 0; i < decoys && nodes.size() < budget; i++) {
                int token = rng.nextBoolean() ? chainAt(cursor) : rng.nextInt(nVocab);
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

        private int chainAt(int i) {
            return i < chain.length ? chain[i] : rngFallback();
        }

        private int rngFallback() {
            return 4 + (cursor * 7) % Math.max(1, nVocab - 4);
        }
    }

    private static MemoryConfig.Builder memory() {
        return MemoryConfig.builder().nCtx(N_CTX).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32);
    }

    private static EngineConfig treeConfig(EngineConfig.BackendMode backend, boolean paged) {
        MemoryConfig.Builder m = memory();
        if (paged) {
            m.parallel(PARALLEL).kvUnified(true);
        }
        return EngineConfig.builder().backend(backend).memory(m.build())
                .draftMax(WIDTH)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(WIDTH).build())
                .build();
    }

    private static EngineConfig plainConfig(EngineConfig.BackendMode backend) {
        return EngineConfig.builder().backend(backend).memory(memory().build()).build();
    }

    /** A greedy request at a fixed seed: every run's draft stream draws the same trees. */
    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding().toBuilder().seed(SEED).build()).build();
    }

    /** The target's plain greedy tokens over {@code backend}: the chain the drafter proposes. */
    private static int[] calibration(EngineConfig.BackendMode backend) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), plainConfig(backend))) {
            return GenerationLoop.forModel(model, plainConfig(backend))
                    .generate(PROMPT, greedy(), (TokenCallback) null).tokens();
        }
    }

    private static final class Run {
        final int[] tokens;
        final List<Integer> acceptedPerRound;

        Run(int[] tokens, List<Integer> acceptedPerRound) {
            this.tokens = tokens;
            this.acceptedPerRound = acceptedPerRound;
        }
    }

    /** One tree-verify generation over {@code backend}, the acceptance walked on the host or on the device. */
    private static Run run(EngineConfig.BackendMode backend, boolean paged, boolean device) throws IOException {
        int[] chain = calibration(backend);
        RandomChainTreeDrafter drafter = new RandomChainTreeDrafter(chain);
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeConfig(backend, paged))) {
            SpeculativeDecoder decoder = SpeculativeDecoder.builder(model.forwardBackend(), drafter,
                    model.tokenizer())
                    .draftMax(WIDTH)
                    .treeNodes(WIDTH)
                    .ropeType(model.spec().hparams().ropeType())
                    .textPositions(model.spec().hparams().mropeTextPositions())
                    .verifyOnDevice(device)
                    .build();
            GenerationResult r = decoder.generate(PROMPT, greedy(), (TokenCallback) null);
            assertEquals("the generation runs to its budget", N_NEW, r.tokens().length);
            assertEquals(StopReason.MAX_TOKENS, r.stopReason());
            return new Run(r.tokens(), drafter.acceptedPerRound);
        }
    }

    /**
     * The host walk on the CPU: the calibrated rounds accept through real frontiers, and the run stays lossless
     * against the plain greedy loop — the walk semantics, the cursor and the random trees behave before the device
     * variants run.
     */
    @Test
    public void theHostWalkStaysLosslessWithRandomTreesOnCpu() throws IOException {
        int[] chain = calibration(EngineConfig.BackendMode.CPU);
        Run run = run(EngineConfig.BackendMode.CPU, false, false);
        assertArrayEquals("the random-tree rounds commit the plain greedy chain", chain, run.tokens);
        assertTrue("the rounds accepted through the calibrated chain: " + run.acceptedPerRound,
                run.acceptedPerRound.stream().mapToInt(Integer::intValue).sum() >= 4);
    }

    /**
     * Device walk against host walk on CUDA: the same batch, the same floats — the tokens and the per-round
     * acceptance match exactly, the gate is on, and both stay lossless against the plain greedy loop.
     */
    @Test
    public void theDeviceWalkMatchesTheHostWalkOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        theDeviceWalkMatchesTheHostWalk(EngineConfig.BackendMode.CUDA);
    }

    /** The same device-walk parity on OpenCL: the GLSL twin's walk over the same batch. */
    @Test
    public void theDeviceWalkMatchesTheHostWalkOnOpenCl() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        theDeviceWalkMatchesTheHostWalk(EngineConfig.BackendMode.OPENCL);
    }

    /** The same device-walk parity on Vulkan: the GLSL twin's walk over the same batch. */
    @Test
    public void theDeviceWalkMatchesTheHostWalkOnVulkan() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        theDeviceWalkMatchesTheHostWalk(EngineConfig.BackendMode.VULKAN);
    }

    private void theDeviceWalkMatchesTheHostWalk(EngineConfig.BackendMode backend) throws IOException {
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeConfig(backend, true))) {
            assertTrue("the backend walks the acceptance on the device",
                    ((GpuForwardBackend) model.forwardBackend()).treeAcceptOnDevice());
        }
        Run device = run(backend, true, true);
        Run deviceRepeat = run(backend, true, true);
        Run host = run(backend, true, false);
        assertArrayEquals("the device walk is deterministic across runs", device.tokens, deviceRepeat.tokens);
        assertEquals("the device walk's acceptance is deterministic", device.acceptedPerRound,
                deviceRepeat.acceptedPerRound);
        assertArrayEquals("the device walk commits the host walk's tokens", host.tokens, device.tokens);
        assertEquals("the device walk accepts what the host walk accepts per round", host.acceptedPerRound,
                device.acceptedPerRound);
        assertFalse("some round matched a node on the device: " + device.acceptedPerRound,
                device.acceptedPerRound.stream().mapToInt(Integer::intValue).allMatch(a -> a == 0));
        assertArrayEquals("the device walk stays lossless against the plain greedy loop",
                calibration(backend), device.tokens);
    }
}

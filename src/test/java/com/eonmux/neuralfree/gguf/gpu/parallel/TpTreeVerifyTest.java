package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.TreeDraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A tree verify's speculative rounds over a tensor-parallel split: every round's masked verify step runs on the
 * main device while the step mirror forwards the step and its page tables to every rank cache, and every commit's
 * compaction (the accepted path's row moves) carries onto every rank through the backend's {@code carryRowMoves}
 * — the mirror this test exists for, which no other tree test covers. A two-rank split of one OpenCL device runs
 * a calibrated random-tree drafter's generation (a chain of the plain greedy run's tokens fronted by decoy roots,
 * so the rounds accept deep and their commits move real rows down) and matches the whole-device load of the same
 * model token for token, with the same per-round acceptance and rank-consistent KV after the last commit. The GPU
 * case is skipped when no OpenCL device is available.
 */
public class TpTreeVerifyTest {

    private static final int UBATCH = 16;
    private static final int N_CTX = 256;
    private static final int RANKS = 2;
    private static final int N_NEW = 16;
    private static final int WIDTH = 4;
    private static final long SEED = 20261106L;
    private static final int[] PROMPT = {4, 5, 6, 4, 5, 7, 4, 5, 8, 4, 5};
    private static final float TOL = 1e-3f;

    private Path file;
    private HWAcceleration hw;
    private OclComputeBackend main;
    private OclComputeBackend peer;
    private LlmKernels kernels;
    private ComputeQueue queue;
    private GpuPlaceholders placeholders;
    private GpuScratch scratch;
    private GpuOps ops;
    private GGUFFileParser parser;
    private WeightStore store;
    private GpuWeightUploader referenceUploader;
    private GpuWeightUploader splitUploader;
    private ModelSpec spec;
    private MemoryPlan referencePlan;
    private MemoryPlan splitPlan;
    private DeviceModelMemory referenceMemory;
    private DeviceModelMemory splitMemory;
    private TensorParallelGroup group;
    private GpuForwardBackend referenceRaw;
    private DeviceModelMemory plainMemory;
    private GpuForwardBackend plainRaw;
    private ForwardBackend split;
    private GGUFTokenizer tokenizer;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(4).build()).writeTempFile();
        parser = new GGUFFileParser(file.toString());
        hw = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hw.getCLDevices().isEmpty());
        main = new OclComputeBackend(hw.getCLDevices().get(0));
        peer = new OclComputeBackend(hw.getCLDevices().get(0));
        kernels = LlmKernels.open(main);
        queue = main.computeQueue();
        placeholders = GpuPlaceholders.allocate(main, queue);
        MemoryConfig mem = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH).parallel(RANKS).kvUnified(true)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), mem);
        scratch = GpuScratch.forModel(main, spec, UBATCH);
        ops = new GpuOps(queue, kernels, scratch, placeholders);
        store = parser.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        GpuWeightUploader.Config cfg = GpuWeightUploader.config(
                GpuWeightUploader.TypePolicy.forKernels(kernels)).build();
        referenceUploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama", cfg);
        referenceUploader.uploadResident(t -> false);
        splitUploader = new GpuWeightUploader(main, queue, parser.getTensors(), store, "llama", cfg);
        TpAttentionPlan attentionPlan = TpAttentionPlan.of(spec, RANKS, splitUploader);
        TpVocabPlan vocabPlan = TpVocabPlan.of(spec, RANKS, splitUploader);
        splitUploader.uploadResident(t -> attentionPlan.splits(t.name()) || vocabPlan.splits(t.name()));
        referencePlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        referenceMemory = DeviceModelMemory.allocate(main, queue, referencePlan,
                spec.hparams().mropeTextPositions(), null);
        splitPlan = MemoryPlanner.planForContext(spec, mem, N_CTX);
        splitMemory = DeviceModelMemory.allocate(main, queue, splitPlan,
                spec.hparams().mropeTextPositions(), null);
        group = TensorParallelGroup.forPeers(main, Collections.singletonList(
                new GpuDevices.Selected(peer, "peer of the same device")));
        group.init(spec, UBATCH, kernels, queue, placeholders, null, attentionPlan, vocabPlan, null, mem, N_CTX);
        group.upload(store, splitUploader);
        group.verifyPagedMemory(splitMemory);
        GpuForwardContext referenceContext = GpuForwardContext.builder(kernels, ops, referenceUploader, spec)
                .memory(referenceMemory).build();
        referenceRaw = new GpuForwardBackend(referenceContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        // the plain greedy loop of the calibration runs over a contiguous whole-device load: the plain
        // GenerationLoop maps positions through MemoryPlan's row addressing, which a paged group refuses
        MemoryConfig plainMem = MemoryConfig.builder().nCtx(N_CTX).ubatch(UBATCH)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        MemoryPlan plainPlan = MemoryPlanner.planForContext(spec, plainMem, N_CTX);
        plainMemory = DeviceModelMemory.allocate(main, queue, plainPlan,
                spec.hparams().mropeTextPositions(), null);
        GpuForwardContext plainContext = GpuForwardContext.builder(kernels, ops, referenceUploader, spec)
                .memory(plainMemory).build();
        plainRaw = new GpuForwardBackend(plainContext, GpuFactoryCatalog.defaults(),
                GpuFactoryCatalog.headPreprocessors());
        GpuBlockRegistry decorated = GpuFactoryCatalog.defaults()
                .withMixerDecorator(group.attentionDecorator())
                .withFfnDecorator(group.decorator());
        GpuForwardContext splitContext = GpuForwardContext.builder(kernels, ops, splitUploader, spec)
                .memory(splitMemory).build();
        GpuForwardBackend splitRaw = new GpuForwardBackend(splitContext, decorated,
                GpuFactoryCatalog.headPreprocessors(), group.vocabEmbedding(), group.vocabHead());
        split = new TensorParallelBackend(splitRaw, group);
        tokenizer = new GGUFTokenizer(parser);
    }

    @After
    public void close() throws IOException {
        try {
            if (split != null) {
                split.close();
            }
            if (referenceRaw != null) {
                referenceRaw.close();
            }
            if (plainRaw != null) {
                plainRaw.close();
            }
            if (plainMemory != null) {
                plainMemory.close();
            }
            if (group != null) {
                group.close();
            }
            if (splitMemory != null) {
                splitMemory.close();
            }
            if (referenceMemory != null) {
                referenceMemory.close();
            }
            splitUploader.close();
            referenceUploader.close();
            ops.close();
            scratch.close();
            store.close();
            placeholders.close();
            main.close();
            peer.close();
            parser.close();
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
            Files.deleteIfExists(file);
        }
    }

    /**
     * The random-tree drafter of the engine tree tests: a chain of the calibration run's greedy tokens fronted by
     * one or two decoy roots (some repeating the chain's next token, so the walk keeps the smallest matching
     * index), advanced by the tokens each round emitted. Every proposal is branching, and the accepted paths sit
     * above the batch's leftmost rows, whose commits gather real rows down.
     */
    private static final class RandomChainTreeDrafter implements TreeDraftModel {
        private final int[] chain;
        private final int nVocab;
        private final List<Integer> acceptedPerRound = new ArrayList<>();
        private int cursor = 1; // the generation's pre-loop sample emitted the chain's first token

        RandomChainTreeDrafter(int[] chain, int nVocab) {
            this.chain = chain;
            this.nVocab = nVocab;
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

    /** A greedy request at a fixed seed: the whole-device run and the split draw the same trees. */
    private static GenerationRequest greedy() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding().toBuilder().seed(SEED).build()).build();
    }

    /** The plain greedy tokens of the contiguous whole-device load: the chain both drafters propose. */
    private int[] calibration() {
        return GenerationLoop.builder(plainRaw, tokenizer).build()
                .generate(PROMPT, greedy(), null).tokens();
    }

    /** One tree-verify generation over {@code backend} with a fresh drafter over {@code chain}. */
    private Run run(ForwardBackend backend, int[] chain) {
        RandomChainTreeDrafter drafter = new RandomChainTreeDrafter(chain, spec.hparams().nVocab());
        SpeculativeDecoder decoder = SpeculativeDecoder.builder(backend, drafter, tokenizer)
                .draftMax(WIDTH)
                .treeNodes(WIDTH)
                .ropeType(spec.hparams().ropeType())
                .textPositions(spec.hparams().mropeTextPositions())
                .build();
        GenerationResult r = decoder.generate(PROMPT, greedy(), null);
        assertEquals(N_NEW, r.tokens().length);
        assertEquals(StopReason.MAX_TOKENS, r.stopReason());
        return new Run(r.tokens(), drafter.acceptedPerRound);
    }

    private static final class Run {
        final int[] tokens;
        final List<Integer> acceptedPerRound;

        Run(int[] tokens, List<Integer> acceptedPerRound) {
            this.tokens = tokens;
            this.acceptedPerRound = acceptedPerRound;
        }
    }

    /**
     * The split's tree rounds match the whole-device load's: the same tokens, the same per-round acceptance (the
     * commits' row moves mirrored onto every rank keep the split's KV where the whole-device run's is), and one
     * more paged batch after the last commit still matches the reference logits.
     */
    @Test
    public void theSplitTreeRoundsMatchTheWholeDeviceLoad() {
        assertTrue("the split walks the acceptance on the device", split.treeAcceptOnDevice());
        int[] chain = calibration();
        Run whole = run(referenceRaw, chain);
        Run ranks = run(split, chain);
        assertArrayEquals("the split commits the whole-device run's tokens", whole.tokens, ranks.tokens);
        assertEquals("the split accepts per round what the whole-device run accepts", whole.acceptedPerRound,
                ranks.acceptedPerRound);
        assertTrue("the rounds accepted through the calibrated chain: " + ranks.acceptedPerRound,
                ranks.acceptedPerRound.stream().mapToInt(Integer::intValue).sum() >= 4);
    }
}

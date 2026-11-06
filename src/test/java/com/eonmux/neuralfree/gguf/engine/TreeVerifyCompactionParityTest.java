package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The compaction commit's KV parity, per cache type: a forced two-level proposal (two roots, the second root's
 * child) whose logit biases make every greedy draw the second root's token, so every round accepts a path that is
 * NOT the batch's leftmost branch — its rows sit at slots 2 and 3 and the commit MOVES them down onto positions
 * {@code pos+1} and {@code pos+2} (an identity move would exercise nothing). After the generation, the target's
 * K/V rows are read back per layer and compared against the same model decoding the very same tokens without
 * speculation: a committed row must hold what the row of its position holds after a plain decode. The rows come
 * from differently shaped forwards (a width+1 batch versus one decode row), so float rounding differs at the last
 * bits and block-quantized caches re-quantize those bits — the comparison bounds the deviation relative to the
 * values' scale, which a wrong move, a stale row or an off-by-one position overshoots by orders of magnitude.
 * Every cache type the parity runs on — per-element, FP16, the block quantizations and the per-token-head int8 —
 * must pass it, on the paged memory of parallel sequences (the tree verify's device shape) and on the plain
 * contiguous memory alike. On the GPU backends the rows are read back from the device buffers through the
 * compute queue, whose paged fresh-window rule makes a slot cursor above the live rows misalign the verify
 * batch's stored positions against the slots the attention addresses — the parity holds the commit to the same
 * bytes a decode would have written, paged or contiguous, on CUDA and OpenCL alike.
 */
public class TreeVerifyCompactionParityTest {

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
        // head dim 32: the block-quantized K/V caches (q4_0, q5_0, q8_0) address 32-value blocks, which a
        // 16-value head would split
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(4).nEmbd(128).headDim(32)
                .nRot(32).build();
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
     * A drafter proposing the forced two-level tree — root {@code 5}, root {@code 6}, and {@code 6} again as the
     * second root's depth-2 child — with no target interaction (no hidden rows, no injection to suspend). It
     * records every round's acceptance, so the test can tell a real tree round (both levels kept) from a fallback.
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

    /**
     * A drafter whose every proposal fronts eight never-accepted {@code LO_TOKEN} decoy roots onto the forced
     * eight-level {@code HI_TOKEN} chain, so the chain's rows sit at the proposal's last eight rows and every
     * round's commit gathers all eight of them down — the deep moves whose dead sources strand the tail pages.
     */
    private static final class DeepForcedTreeDrafter implements TreeDraftModel {

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
            int depth = Math.min(8, maxTokens);
            for (int i = 0; i < 7 && nodes.size() < maxRows; i++) {
                nodes.add(new DraftTree.Node(LO_TOKEN, -1, 1, 0));
            }
            int parent = -1;
            for (int d = 1; d <= depth && nodes.size() < maxRows; d++) {
                nodes.add(new DraftTree.Node(HI_TOKEN, parent, d, 0));
                parent = nodes.size() - 1;
            }
            if (nodes.size() < maxRows) {
                // the last row keeps a live decoy on the tail page, so the truncation's walk sees the page as
                // whole-leaving with gathered-away rows below its one live row
                nodes.add(new DraftTree.Node(LO_TOKEN, -1, 1, 0));
            }
            return new DraftTree(nodes);
        }
    }

    private static MemoryConfig.Builder memory(KvDType type) {
        return MemoryConfig.builder().nCtx(N_CTX).ubatch(16).typeK(type).typeV(type);
    }

    private static EngineConfig treeConfig(KvDType type, boolean paged) {
        MemoryConfig.Builder m = memory(type);
        if (paged) {
            m.parallel(PARALLEL).kvUnified(true);
        }
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(m.build())
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX).build())
                .build();
    }

    /**
     * The reference loop's config: always the plain contiguous memory. The single-sequence GenerationLoop prefills
     * through the contiguous rows alone — a paged {@code kvUnified} memory maps positions through the page table,
     * which that path does not use — so the paged parity compares the paged tree run against the same decode on the
     * contiguous memory, whose committed bytes the tree run must reproduce either way.
     */
    private static EngineConfig plainConfig(KvDType type) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory(type).build()).build();
    }

    /** The greedy request whose biases force every draw to {@code HI_TOKEN}, runner-up {@code LO_TOKEN}. */
    private static GenerationRequest forced() {
        return forced(N_NEW);
    }

    private static GenerationRequest forced(int maxNewTokens) {
        GenerationRequest.Builder b = GenerationRequest.builder().maxNewTokens(maxNewTokens)
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
    public void theCompactedKvMatchesAPlainDecodeOnPagedMemoryAcrossCacheTypes() throws IOException {
        for (KvDType type : new KvDType[] {KvDType.F32, KvDType.F16, KvDType.Q8_0, KvDType.Q4_0, KvDType.Q5_0,
                KvDType.INT8_PER_TOKEN_HEAD}) {
            checkParity(type, true);
        }
    }

    /** The same parity over the plain contiguous single-sequence memory: the host row moves, no page table. */
    @Test
    public void theCompactedKvMatchesAPlainDecodeOnContiguousMemory() throws IOException {
        for (KvDType type : new KvDType[] {KvDType.F32, KvDType.F16, KvDType.Q8_0, KvDType.Q4_0, KvDType.Q5_0,
                KvDType.INT8_PER_TOKEN_HEAD}) {
            checkParity(type, false);
        }
    }

    /**
     * A deep acceptance whose gathered sources deaden every own row of a tail page: the forced eight-level chain
     * sits at the proposal's last eight rows, so the commit's eight gathers empty the page past the verify batch's
     * page boundary and the truncation must pull the append cursor back with the page. The parity compares the
     * committed rows against the plain decode over the paged memory, whose page list the walk reshapes.
     */
    @Test
    public void theDeepCompactionKeepsThePagedPagesAcrossABoundary() throws IOException {
        KvDType type = KvDType.F32;
        GenerationRequest request = forced(24);
        EngineConfig plainCfg = plainConfig(type);
        int[] expectedTokens;
        KvSnapshot expected;
        try (EngineModel plain = LLMModelFactory.open(modelFile.toString(), plainCfg)) {
            GenerationResult r = GenerationLoop.forModel(plain, plainCfg)
                    .generate(PROMPT, request, (TokenCallback) null);
            expectedTokens = r.tokens();
            expected = KvSnapshot.of(plain);
        }
        MemoryConfig m = MemoryConfig.builder().nCtx(N_CTX).ubatch(32).typeK(type).typeV(type)
                .parallel(PARALLEL).kvUnified(true).build();
        EngineConfig treeCfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(m)
                .draftMax(8)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(16).build())
                .build();
        int[] actualTokens;
        KvSnapshot actual;
        DeepForcedTreeDrafter drafter = new DeepForcedTreeDrafter();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, treeCfg);
            GenerationResult r = decoder.generate(PROMPT, request, (TokenCallback) null);
            actualTokens = r.tokens();
            actual = KvSnapshot.of(model);
        }
        assertArrayEquals("the deep forced tree commits the plain decode's tokens", expectedTokens, actualTokens);
        assertEquals("every token committed, so both caches hold the same rows", expected.nPast, actual.nPast);
        assertTrue("every round accepted the full eight-level chain: " + drafter.acceptedPerRound,
                drafter.acceptedPerRound.stream().filter(a -> a == 8).count() >= 2);
        actual.assertMatches(expected, type + " deep paged");
    }

    private void checkParity(KvDType type, boolean paged) throws IOException {
        GenerationRequest request = forced();
        EngineConfig plainCfg = plainConfig(type);
        int[] expectedTokens;
        KvSnapshot expected;
        try (EngineModel plain = LLMModelFactory.open(modelFile.toString(), plainCfg)) {
            GenerationResult r = GenerationLoop.forModel(plain, plainCfg)
                    .generate(PROMPT, request, (TokenCallback) null);
            expectedTokens = r.tokens();
            expected = KvSnapshot.of(plain);
        }
        EngineConfig treeCfg = treeConfig(type, paged);
        int[] actualTokens;
        KvSnapshot actual;
        ForcedTreeDrafter drafter = new ForcedTreeDrafter();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, treeCfg);
            GenerationResult r = decoder.generate(PROMPT, request, (TokenCallback) null);
            actualTokens = r.tokens();
            actual = KvSnapshot.of(model);
        }
        assertArrayEquals("the forced tree commits the plain decode's tokens (" + type + ", "
                + (paged ? "paged" : "contiguous") + ")", expectedTokens, actualTokens);
        assertEquals("every token committed, so both caches hold the same rows (" + type + ")",
                expected.nPast, actual.nPast);
        assertTrue("the forced rounds kept the full two-level path, so the compaction moved real rows: "
                + drafter.acceptedPerRound, drafter.acceptedPerRound.contains(2));
        actual.assertMatches(expected, type + (paged ? " paged" : " contiguous"));
    }

    /**
     * The device parity on CUDA: both paged forms (the parallel unified sequences the verify session shapes, and
     * the single-sequence {@code kvPaged} memory) commit rows the device's contiguous plain decode reproduces. The
     * contiguous form's own compaction runs on the host memory — the tests above it — because a device backend
     * refuses the masked multi-row verify batch off the paged sequences ({@code batchRefusal}).
     */
    @Test
    public void theCompactedDeviceKvMatchesAPlainDecodeOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        for (boolean parallelSequences : new boolean[] {true, false}) {
            checkParityOnDevice(EngineConfig.BackendMode.CUDA, parallelSequences);
        }
    }

    /** The same device parity on OpenCL: the tree-masked step runs the GLSL twin's flash kernels. */
    @Test
    public void theCompactedDeviceKvMatchesAPlainDecodeOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        for (boolean parallelSequences : new boolean[] {true, false}) {
            checkParityOnDevice(EngineConfig.BackendMode.OPENCL, parallelSequences);
        }
    }

    /**
     * The deep boundary-crossing compaction on the device's paged memory: the forced eight-level chain gathers
     * across the verify batch's page boundary, whose moved-away rows strand dead slots under the append cursor —
     * the geometry whose drift misaligned the GPU verify batch's stored positions against the attention's slots.
     */
    @Test
    public void theDeepCompactionKeepsTheDevicePagesOnCuda() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        checkDeepParityOnDevice(EngineConfig.BackendMode.CUDA);
    }

    /** The same deep device parity on OpenCL. */
    @Test
    public void theDeepCompactionKeepsTheDevicePagesOnOpenCL() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        checkDeepParityOnDevice(EngineConfig.BackendMode.OPENCL);
    }

    private void checkParityOnDevice(EngineConfig.BackendMode backend, boolean parallelSequences) throws IOException {
        GenerationRequest request = forced();
        EngineConfig plainCfg = EngineConfig.builder().backend(backend).memory(memory(KvDType.F32).build()).build();
        int[] expectedTokens;
        KvSnapshot expected;
        try (EngineModel plain = LLMModelFactory.open(modelFile.toString(), plainCfg)) {
            GenerationResult r = GenerationLoop.forModel(plain, plainCfg)
                    .generate(PROMPT, request, (TokenCallback) null);
            expectedTokens = r.tokens();
            expected = KvSnapshot.of(plain);
        }
        MemoryConfig.Builder m = memory(KvDType.F32);
        if (parallelSequences) {
            m.parallel(PARALLEL).kvUnified(true);
        } else {
            m.kvPaged(true);
        }
        EngineConfig treeCfg = EngineConfig.builder().backend(backend).memory(m.build())
                .draftMax(DRAFT_MAX)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(DRAFT_MAX).build())
                .build();
        int[] actualTokens;
        KvSnapshot actual;
        ForcedTreeDrafter drafter = new ForcedTreeDrafter();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, treeCfg);
            GenerationResult r = decoder.generate(PROMPT, request, (TokenCallback) null);
            actualTokens = r.tokens();
            actual = KvSnapshot.of(model);
        }
        assertArrayEquals("the forced tree commits the plain decode's tokens (" + backend + ", "
                + (parallelSequences ? "paged parallel" : "paged single") + ")", expectedTokens, actualTokens);
        assertEquals("every token committed, so both caches hold the same rows (" + backend + ")",
                expected.nPast, actual.nPast);
        assertTrue("the forced rounds kept the full two-level path, so the compaction moved real rows: "
                + drafter.acceptedPerRound, drafter.acceptedPerRound.contains(2));
        actual.assertMatches(expected, backend + (parallelSequences ? " paged parallel" : " paged single"));
    }

    private void checkDeepParityOnDevice(EngineConfig.BackendMode backend) throws IOException {
        KvDType type = KvDType.F32;
        GenerationRequest request = forced(24);
        EngineConfig plainCfg = EngineConfig.builder().backend(backend).memory(memory(type).build()).build();
        int[] expectedTokens;
        KvSnapshot expected;
        try (EngineModel plain = LLMModelFactory.open(modelFile.toString(), plainCfg)) {
            GenerationResult r = GenerationLoop.forModel(plain, plainCfg)
                    .generate(PROMPT, request, (TokenCallback) null);
            expectedTokens = r.tokens();
            expected = KvSnapshot.of(plain);
        }
        MemoryConfig m = MemoryConfig.builder().nCtx(N_CTX).ubatch(32).typeK(type).typeV(type)
                .parallel(PARALLEL).kvUnified(true).build();
        EngineConfig treeCfg = EngineConfig.builder().backend(backend).memory(m)
                .draftMax(8)
                .speculativeOptions(SpeculativeOptions.builder().treeVerify(true).treeNodes(16).build())
                .build();
        int[] actualTokens;
        KvSnapshot actual;
        DeepForcedTreeDrafter drafter = new DeepForcedTreeDrafter();
        try (EngineModel model = LLMModelFactory.open(modelFile.toString(), treeCfg)) {
            SpeculativeDecoder decoder = SpeculativeDecoder.forModel(model, drafter, treeCfg);
            GenerationResult r = decoder.generate(PROMPT, request, (TokenCallback) null);
            actualTokens = r.tokens();
            actual = KvSnapshot.of(model);
        }
        assertArrayEquals("the deep forced tree commits the plain decode's tokens (" + backend + ")",
                expectedTokens, actualTokens);
        assertEquals("every token committed, so both caches hold the same rows", expected.nPast, actual.nPast);
        assertTrue("every round accepted the full eight-level chain: " + drafter.acceptedPerRound,
                drafter.acceptedPerRound.stream().filter(a -> a == 8).count() >= 2);
        actual.assertMatches(expected, backend + " deep paged");
    }

    /** The committed K/V rows of one engine's memory, read back per layer after its generation. */
    private static final class KvSnapshot {

        final int nPast;
        /** Per layer, per head: the dequantized K row of every position, then the V rows under the same key. */
        final List<Map<Integer, float[]>> keys;
        final List<Map<Integer, float[]>> values;
        final float maxAbs;

        private KvSnapshot(int nPast, List<Map<Integer, float[]>> keys, List<Map<Integer, float[]>> values,
                           float maxAbs) {
            this.nPast = nPast;
            this.keys = keys;
            this.values = values;
            this.maxAbs = maxAbs;
        }

        static KvSnapshot of(EngineModel model) {
            if (model.forwardBackend() instanceof GpuForwardBackend) {
                return ofDevice(model, (GpuForwardBackend) model.forwardBackend());
            }
            ModelMemory memory = model.forwardBackend().memory();
            CpuMemory cpu = (CpuMemory) memory;
            MemoryPlan plan = memory.plan();
            int nPast = memory.sequences().sequence(0).nPast();
            // the row of every position, per KV group
            List<Map<Integer, Integer>> rowsOfPos = new ArrayList<>();
            for (int g = 0; g < plan.groups().size(); g++) {
                Map<Integer, Integer> rows = new HashMap<>();
                for (int r = 0; r < plan.group(g).rows(); r++) {
                    int p = cpu.positionOfRow(g, r);
                    if (p >= 0) {
                        rows.put(p, r);
                    }
                }
                rowsOfPos.add(rows);
            }
            List<Map<Integer, float[]>> keys = new ArrayList<>();
            List<Map<Integer, float[]>> values = new ArrayList<>();
            float maxAbs = 0f;
            for (int il = 0; il < plan.layers().size(); il++) {
                KvAccessor kv = cpu.kv(il);
                int group = kv.group();
                int nK = kv.nHeadKv() * kv.headDimK();
                int nV = kv.nHeadKv() * kv.headDimV();
                Map<Integer, float[]> layerK = new HashMap<>();
                Map<Integer, float[]> layerV = new HashMap<>();
                for (int pos = 0; pos < nPast; pos++) {
                    Integer row = rowsOfPos.get(group).get(pos);
                    if (row == null) {
                        throw new AssertionError("layer " + il + " holds no row for position " + pos);
                    }
                    float[] k = new float[nK];
                    float[] v = new float[nV];
                    int offK = 0;
                    int offV = 0;
                    for (int h = 0; h < kv.nHeadKv(); h++) {
                        kv.readK(row, h, k, offK);
                        kv.readV(row, h, v, offV);
                        offK += kv.headDimK();
                        offV += kv.headDimV();
                    }
                    for (float x : k) {
                        maxAbs = Math.max(maxAbs, Math.abs(x));
                    }
                    for (float x : v) {
                        maxAbs = Math.max(maxAbs, Math.abs(x));
                    }
                    layerK.put(pos, k);
                    layerV.put(pos, v);
                }
                keys.add(layerK);
                values.add(layerV);
            }
            return new KvSnapshot(nPast, keys, values, maxAbs);
        }

        /**
         * The same rows read back from a device memory: every KV cell the step inputs map to a committed
         * position contributes its K and V row, read through the engine's compute queue (F32 caches, whose rows
         * are plain float vectors of the heads' width). The queue drains before the snapshot returns, so the
         * reads observe every write the generation queued.
         */
        private static KvSnapshot ofDevice(EngineModel model, GpuForwardBackend gpu) {
            DeviceModelMemory memory = (DeviceModelMemory) gpu.context().memory();
            ComputeQueue queue = gpu.context().queue();
            queue.finish();
            MemoryPlan plan = memory.plan();
            int nPast = memory.sequences().sequence(0).nPast();
            DeviceStepInputs steps = memory.stepInputs();
            List<Map<Integer, float[]>> keys = new ArrayList<>();
            List<Map<Integer, float[]>> values = new ArrayList<>();
            float maxAbs = 0f;
            for (int il = 0; il < plan.layers().size(); il++) {
                DeviceKvLayer kv = (DeviceKvLayer) memory.kv(il);
                KvLayout layout = kv.layout();
                int group = kv.group();
                int nK = (int) (layout.rowBytesK() / Integer.BYTES);
                int nV = (int) (layout.rowBytesV() / Integer.BYTES);
                Map<Integer, float[]> layerK = new HashMap<>();
                Map<Integer, float[]> layerV = new HashMap<>();
                for (int cell = 0; cell < layout.rows(); cell++) {
                    int pos = steps.positionOfRow(group, cell);
                    if (pos < 0 || pos >= nPast) {
                        continue;
                    }
                    float[] k = new float[nK];
                    queue.enqueueReadBuffer(kv.k(), (long) cell * layout.rowBytesK(), layout.rowBytesK(), k);
                    float[] v = new float[nV];
                    queue.enqueueReadBuffer(kv.v(), (long) cell * layout.rowBytesV(), layout.rowBytesV(), v);
                    queue.finish();
                    for (float x : k) {
                        maxAbs = Math.max(maxAbs, Math.abs(x));
                    }
                    for (float x : v) {
                        maxAbs = Math.max(maxAbs, Math.abs(x));
                    }
                    layerK.put(pos, k);
                    layerV.put(pos, v);
                }
                keys.add(layerK);
                values.add(layerV);
            }
            return new KvSnapshot(nPast, keys, values, maxAbs);
        }

        /**
         * Every committed position's rows match the reference's, within the rounding the differently shaped
         * forwards and the cache's own quantization leave: the bound scales with the values' magnitude, which a
         * structural error — a row left at its verify slot, a one-position-off move, a stale row — exceeds many
         * times over.
         */
        void assertMatches(KvSnapshot expected, String what) {
            assertEquals("layer count " + what, expected.keys.size(), keys.size());
            float tol = Math.max(2e-2f, 0.05f * Math.max(expected.maxAbs, maxAbs));
            for (int il = 0; il < expected.keys.size(); il++) {
                for (int pos = 0; pos < expected.nPast; pos++) {
                    assertClose(expected.keys.get(il).get(pos), keys.get(il).get(pos),
                            tol, what + " K layer " + il + " position " + pos);
                    assertClose(expected.values.get(il).get(pos), values.get(il).get(pos),
                            tol, what + " V layer " + il + " position " + pos);
                }
            }
        }

        private static void assertClose(float[] a, float[] b, float tol, String what) {
            assertEquals(what + " width", a.length, b.length);
            float worst = 0f;
            for (int i = 0; i < a.length; i++) {
                worst = Math.max(worst, Math.abs(a[i] - b[i]));
            }
            assertTrue(what + " deviates by " + worst + " (tolerance " + tol + ")", worst <= tol);
        }
    }
}

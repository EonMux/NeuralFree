package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The dispatch sequence of {@link GpuMoeFfn} over a {@link MoeExpertCache} on the host-backed fake backend, over a
 * synthetic qwen3next file whose experts are preloaded from a ranking: the cached experts run through the
 * indirection kernels ({@code F32_MatVecIdGateUpInd}, {@code F32_MatVecIdInd}) with the per-layer map and the
 * banks, a miss pair adds its row through a device write before the weighted sum, an all-miss step skips the
 * expert kernels entirely, and the prefill path sorts and aligns the pairs and runs the grouped GEMM over the map
 * ({@code F32_GroupedGemmInd}), or the per-pair {@code F32_BatchMatMulIdInd} without that kernel. With a
 * {@link MoeExpertPrefetch}, a prefill-sized step computes no host row and reads every expert through the prefill
 * map over the banks. The routed ids come from a router emulator, as the real readback delivers them.
 */
public class GpuCachedMoeFfnTest {

    private static final int N_EMBD = 64;
    private static final int N_EXPERT = 8;
    private static final int K = 2;
    private static final int N_FF_EXP = 32;
    private static final int UBATCH = 8;
    /** The decode sequence around the expert kernels: the shared expert's gated products precede the sum. */
    private static final String ROUTER_LOGITS = "Float_MatVec";
    /** The shared expert's products between the two weighted sums: dense decode rows or batched prefill rows. */
    private static List<String> shexp(String matvec) {
        return Arrays.asList(matvec, matvec, "LLM_GLU", matvec, matvec, "LLM_Unary");
    }
    private static final String WSUM = MoeKernels.WEIGHTED_SUM;
    private static final String GATE_UP_IND = MoeKernels.matVecIdGateUpInd(GGMLType.F32);
    private static final String MATVEC_ID_IND = MoeKernels.matVecIdInd(GGMLType.F32);
    private static final String BMM_ID_IND = MoeKernels.batchMatMulIdInd(GGMLType.F32);
    private static final String GG_IND = MoeKernels.groupedGemmInd(GGMLType.F32);

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private MoeExpertCache cache;
    private MoeExpertPrefetch prefetch;
    /** The heat recorder the block notes its routed ids into; null leaves the context without one. */
    private MoeExpertHeat heat;
    private CpuWeights cpuWeights;
    private Path modelPath;
    private GGUFFileParser parser;
    private ModelSpec spec;
    private LayerSpec layer;
    private GpuFfnBlock block;
    private WeightStore store;
    /** The ids the router emulator writes per step; set before every forward. */
    private int[] routedIds;

    /** One expert of gate or up, in bytes. */
    private static final int TENSOR_BYTES = N_EMBD * N_FF_EXP * Float.BYTES;
    private static final int DOWN_BYTES = N_FF_EXP * N_EMBD * Float.BYTES;

    @Before
    public void setUp() throws Exception {
        open();
    }

    /** Opens the fixtures over a fake backend without {@code unavailableKernels}. */
    private void open(String... unavailableKernels) throws Exception {
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1)
                .nEmbd(N_EMBD).nExpert(N_EXPERT).nExpertUsed(K).nFfExp(N_FF_EXP).nFfShexp(N_FF_EXP).build();
        this.modelPath = SyntheticModels.qwen3next(config).writeTempFile();
        this.parser = new GGUFFileParser(modelPath.toString());
        this.spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        this.layer = spec.layers().get(0);
        this.cpuWeights = CpuWeights.open(parser);


        this.routedIds = new int[K];
        backend = FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels(unavailableKernels)
                .emulate(MoeKernels.ROUTER, d -> {
                    int pairs = Math.min(routedIds.length, d.intArg(0) * d.intArg(2));
                    FakeGpuBuffer idsBuf = (FakeGpuBuffer) d.bufferArg(MoeKernels.ARG_ROUTER_IDS);
                    FakeGpuBuffer weightsBuf = (FakeGpuBuffer) d.bufferArg(MoeKernels.ARG_ROUTER_WEIGHTS);
                    for (int i = 0; i < pairs; i++) {
                        idsBuf.putInt(i, routedIds[i]);
                        weightsBuf.putFloat(i, 0.5f);
                    }
                })
                .build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);

        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        // the routed expert stacks stay in host RAM (the cache serves them); the shared expert stays resident
        put(TensorNames.weight(TensorNames.FFN_GATE_SHEXP), N_EMBD, N_FF_EXP);
        put(TensorNames.weight(TensorNames.FFN_UP_SHEXP), N_EMBD, N_FF_EXP);
        put(TensorNames.weight(TensorNames.FFN_DOWN_SHEXP), N_FF_EXP, N_EMBD);
        put(TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP), N_EMBD);

        this.store = emptyStore();
        cache = MoeExpertCache.open(backend, backend.computeQueue(), cacheData(),
                Collections.singletonList(plan()), 2L * (2L * TENSOR_BYTES + DOWN_BYTES), UBATCH * K,
                ranking(3, 5), "test");

        bindBlock();
    }

    /** Binds a fresh block over the current cache and prefetch pool. */
    private void bindBlock() {
        bindBlock(MoeTailSkip.OFF);
    }

    /** Binds a fresh block over the current cache and prefetch pool with the hosted tail-skip rank. */
    private void bindBlock(int tailSkipRank) {
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return tensors.get(name);
            }
        };
        GpuForwardContext ctx = GpuForwardContext.builder(ops.kernels(), ops, weights, spec)
                .moeExpertCache(cache)
                .hostWeights(cpuWeights)
                .moeExpertPrefetch(prefetch)
                .moeTailSkipRank(tailSkipRank)
                .heat(heat)
                .build();
        block = new GpuMoeFfn(com.eonmux.neuralfree.gguf.arch.FfnKind.MOE, layer);
        block.bind(ctx, layer);
    }

    /** Replaces the cache with one of a whole layer stack of slots and a one-zone prefetch pool. */
    private void rebindWithPrefetch() throws java.io.IOException {
        block.close();
        cache.close();
        cache = MoeExpertCache.open(backend, backend.computeQueue(), cacheData(),
                Collections.singletonList(plan()), (long) N_EXPERT * (2L * TENSOR_BYTES + DOWN_BYTES), UBATCH * K,
                ranking(3, 5), "test");
        prefetch = MoeExpertPrefetch.open(backend, cache, Collections.singletonList(plan()), 1, 0, "test");
        bindBlock();
    }

    @After
    public void tearDown() {
        RuntimeException failure = null;
        for (AutoCloseable c : new AutoCloseable[]{block, prefetch, cache, ops, placeholders, scratch, cpuWeights,
                parser, store}) {
            if (c == null) {
                continue;
            }
            try {
                c.close();
            } catch (Exception e) {
                failure = failure != null ? failure
                        : e instanceof RuntimeException ? (RuntimeException) e : new RuntimeException(e);
            }
        }
        try {
            if (modelPath != null) {
                Files.deleteIfExists(modelPath);
            }
        } catch (java.io.IOException e) {
            failure = failure != null ? failure : new RuntimeException(e);
        }
        if (failure != null) {
            throw failure;
        }
    }

    private static WeightStore emptyStore() {
        return new WeightStore() {
            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst) {
                Arrays.fill(dst, 0, size, (byte) 1);
            }

            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
                Arrays.fill(dst, dstOffset, dstOffset + size, (byte) 1);
            }

            @Override
            public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
                for (int i = 0; i < size; i++) {
                    dst.put((byte) 1);
                }
            }

            @Override
            public void close() {
            }
        };
    }

    private static MoeCacheRanking ranking(int first, int second) throws java.io.IOException {
        ByteBuffer buf = ByteBuffer.allocate(24 + 2 * 8).order(ByteOrder.LITTLE_ENDIAN);
        buf.put("STRP".getBytes(java.nio.charset.StandardCharsets.US_ASCII)).putInt(2).putInt(1).putInt(N_EXPERT)
                .putInt(2).putInt(2);
        buf.putShort((short) 0).putShort((short) first).putInt(9);
        buf.putShort((short) 0).putShort((short) second).putInt(4);
        return MoeCacheRanking.open(null, N_EXPERT,
                new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.PATH, tempFile(buf), null));
    }

    /** A file over the given bytes (the ranking reader reads files). */
    private static Path tempFile(ByteBuffer buf) throws java.io.IOException {
        Path p = Files.createTempFile("ranking", ".bin");
        Files.write(p, buf.array());
        p.toFile().deleteOnExit();
        return p;
    }

    private static String nameOf(String suffix) {
        return TensorNames.layerName(0, TensorNames.weight(suffix));
    }

    private void put(String suffix, long... ne) {
        String name = TensorNames.layerName(0, suffix);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        tensors.put(name, GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true)));
    }

    private MoeExpertCache.LayerPlan plan() {
        List<MoeExpertCache.TensorPlan> expertTensors = Arrays.asList(
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT,
                        GGMLType.F32));
        return new MoeExpertCache.LayerPlan(0, K, expertTensors);
    }

    private WeightDataProvider cacheData() {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_GATE_EXPS), new long[]{N_EMBD, N_FF_EXP, N_EXPERT},
                GGMLType.F32, 0));
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_UP_EXPS), new long[]{N_EMBD, N_FF_EXP, N_EXPERT},
                GGMLType.F32, N_EXPERT * TENSOR_BYTES));
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_DOWN_EXPS), new long[]{N_FF_EXP, N_EMBD, N_EXPERT},
                GGMLType.F32, 2L * N_EXPERT * TENSOR_BYTES));
        return new WeightDataProvider(store, infos,
                Collections.singletonList(Arrays.asList(nameOf(TensorNames.FFN_GATE_EXPS),
                        nameOf(TensorNames.FFN_UP_EXPS), nameOf(TensorNames.FFN_DOWN_EXPS))), 0, 0L);
    }

    private List<FakeDispatch> forward(int n, int[] ids) {
        this.routedIds = ids;
        backend.clearLog();
        block.forward(n, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    /** {@link #forward} over nonzero input rows, so the host-computed expert rows are distinguishable from zeros. */
    private List<FakeDispatch> forwardOnes(int n, int[] ids) {
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        float[] ones = new float[n * N_EMBD];
        Arrays.fill(ones, 1.0f);
        ((FakeGpuBuffer) in.buffer()).putFloats(in.offset(), ones);
        return forward(n, ids);
    }

    private static List<String> names(List<FakeDispatch> ds) {
        List<String> out = new ArrayList<>();
        for (FakeDispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(FakeDispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    private long rowWrites() {
        long n = 0;
        for (FakeOp op : backend.ops()) {
            if (op.kind() == FakeOpKind.WRITE && op.bytes() == (long) N_EMBD * Float.BYTES) {
                n++;
            }
        }
        return n;
    }

    @Test
    public void preloadedHitsRunTheIndirectionKernelsWithoutHostRows() {
        List<FakeDispatch> ds = forward(1, new int[]{3, 5});

        List<String> expected = new ArrayList<>(Arrays.asList(ROUTER_LOGITS, MoeKernels.ROUTER, GATE_UP_IND,
                MATVEC_ID_IND, WSUM));
        expected.addAll(shexp("Float_MatVec"));
        expected.add(WSUM);
        assertEquals(expected, names(ds));

        FakeDispatch gateUp = ds.get(2);
        assertEquals(MoeKernels.GU_IND_ARG_COUNT, gateUp.argCount());
        assertSame(cache.mapBuffer(0), gateUp.bufferArg(MoeKernels.ARG_GUI_MAP));
        assertSame(cache.bank(0, 0), gateUp.bufferArg(MoeKernels.ARG_GUI_GATE));
        assertSame(cache.bank(0, 1), gateUp.bufferArg(MoeKernels.ARG_GUI_UP));
        assertEquals(MoeKernels.SCALE_NONE, gateUp.intArg(MoeKernels.ARG_GU_HAS_SCALE));
        assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, K, 0, 0}, ints(gateUp, 0, 5));

        FakeDispatch down = ds.get(3);
        assertEquals(MoeKernels.ID_IND_ARG_COUNT, down.argCount());
        assertSame(cache.bank(0, 2), down.bufferArg(MoeKernels.ARG_IDI_WEIGHT));
        assertSame(cache.mapBuffer(0), down.bufferArg(MoeKernels.ARG_IDI_MAP));
        assertArrayEquals(new int[]{N_EMBD, N_FF_EXP, K, N_FF_EXP, 0, 0}, ints(down, 0, 6));

        FakeDispatch wsum = ds.get(4);
        assertSame(scratch.slot(ScratchSlots.BRANCH).buffer(), wsum.bufferArg(6));
        int idLocal = (int) gateUp.local()[0];
        assertArrayEquals(new long[]{(long) N_FF_EXP * idLocal, K}, gateUp.global());
        assertEquals(0, rowWrites());
    }

    @Test
    public void aMissPairAddsItsRowThroughADeviceWrite() {
        forward(1, new int[]{3, 5}); // fill the preloaded experts
        List<FakeDispatch> ds = forward(1, new int[]{3, 6}); // 6 is not in the cache: a miss

        List<String> expected = new ArrayList<>(Arrays.asList(ROUTER_LOGITS, MoeKernels.ROUTER, GATE_UP_IND,
                MATVEC_ID_IND, WSUM));
        expected.addAll(shexp("Float_MatVec"));
        expected.add(WSUM);
        assertEquals(expected, names(ds));
        assertEquals("one row write for the miss", 1, rowWrites());
    }

    @Test
    public void anAllMissStepSkipsTheExpertKernels() {
        forward(1, new int[]{3, 5});
        List<FakeDispatch> ds = forward(1, new int[]{1, 2});

        List<String> expected = new ArrayList<>(Arrays.asList(ROUTER_LOGITS, MoeKernels.ROUTER, WSUM));
        expected.addAll(shexp("Float_MatVec"));
        expected.add(WSUM);
        assertEquals(expected, names(ds));
        assertEquals(0, rowWrites());
        List<FakeOp> writes = rowWriteOps(FakeOpKind.WRITE);
        assertEquals("the two consecutive miss rows go up in one write", 1, writes.size());
        assertEquals(2L * N_EMBD * Float.BYTES, writes.get(0).bytes());
    }

    /** Miss pairs 1, 2, 3 and 5 of four tokens go up in two writes, each run's rows landing at its pairs. */
    @Test
    public void missRowsGoUpInOneWritePerRunOfConsecutivePairs() {
        forward(1, new int[]{3, 5});
        // pairs: 0 -> 3 (hit), 1 -> 1, 2 -> 2, 3 -> 4 (misses), 4 -> 5 (hit), 5 -> 6 (miss), 6 -> 3, 7 -> 5 (hits)
        forwardOnes(4, new int[]{3, 1, 2, 4, 5, 6, 3, 5});

        List<FakeOp> writes = new ArrayList<>(rowWriteOps(FakeOpKind.WRITE));
        writes.sort((a, b) -> Long.compare(a.offset(), b.offset()));
        assertEquals(2, writes.size());
        assertEquals(3L * N_EMBD * Float.BYTES, writes.get(0).bytes());
        assertEquals(1L * N_EMBD * Float.BYTES, writes.get(1).bytes());
        assertEquals("the second run starts 4 rows after the first", 4L * N_EMBD * Float.BYTES,
                writes.get(1).offset() - writes.get(0).offset());
        assertFalse("the host rows are computed", allZero(writes.get(0)));
        assertFalse("the run past the first row reads its own rows", allZero(writes.get(1)));
    }

    @Test
    public void runEndStopsAtTheFirstGapOrTheCount() {
        int[] pairs = {1, 2, 3, 5, 6, 9};
        assertEquals(3, GpuCachedMoePass.runEnd(pairs, 0, pairs.length));
        assertEquals(5, GpuCachedMoePass.runEnd(pairs, 3, pairs.length));
        assertEquals(6, GpuCachedMoePass.runEnd(pairs, 5, pairs.length));
        assertEquals("the count bounds the run", 2, GpuCachedMoePass.runEnd(pairs, 0, 2));
    }

    /** The expert-output row writes (or fills) of the last forward, one per run of host rows or dropped rows. */
    private List<FakeOp> rowWriteOps(FakeOpKind kind) {
        ScratchSlot rows = scratch.temp(GpuCachedMoePass.T_EXPERT_OUT_IND, 1);
        long start = 4L * rows.offset();
        List<FakeOp> out = new ArrayList<>();
        for (FakeOp op : backend.ops()) {
            if (op.kind() == kind && op.buffer() == rows.buffer() && op.offset() >= start
                    && op.offset() + op.bytes() <= start + 4L * UBATCH * K * N_EMBD) {
                out.add(op);
            }
        }
        return out;
    }

    private static boolean allZero(FakeOp write) {
        int n = (int) (write.bytes() / Float.BYTES);
        for (float f : write.buffer().getFloats(write.offset() / Float.BYTES, n)) {
            if (f != 0.0f) {
                return false;
            }
        }
        return true;
    }

    /** The tail-skip rule drops a miss that is only ever the token's second pick, and zeros its row. */
    @Test
    public void tailSkipZeroesTheRowOfAMissRoutedOnlyAtTheLastRank() {
        block.close();
        bindBlock(1);
        forward(1, new int[]{3, 6}); // 3 is a preloaded hit, 6 is a miss at rank 1

        assertEquals("the dropped miss computes no host row", 0, rowWriteOps(FakeOpKind.WRITE).size());
        List<FakeOp> fills = rowWriteOps(FakeOpKind.FILL);
        assertEquals("one device fill zeroes the dropped row", 1, fills.size());
        assertEquals((long) N_EMBD * Float.BYTES, fills.get(0).bytes());
        assertTrue("the dropped expert's row stays zero", allZero(fills.get(0)));
    }

    /** The rule keeps a miss some token routes at a better rank and drops the experts that are pure tail picks. */
    @Test
    public void tailSkipKeepsAMissSomeTokenRoutesAtAHigherRank() {
        block.close();
        bindBlock(1);
        // pairs: 0 -> expert 3 (hit), 1 -> 6 (rank 1), 2 -> 6 (rank 0), 3 -> 4 (rank 1)
        forwardOnes(2, new int[]{3, 6, 6, 4});

        List<FakeOp> writes = rowWriteOps(FakeOpKind.WRITE);
        assertEquals("the two consecutive 6 rows go up in one write", 1, writes.size());
        assertEquals(2L * N_EMBD * Float.BYTES, writes.get(0).bytes());
        assertFalse("the 6 rows are token 1's first pick and computed", allZero(writes.get(0)));
        List<FakeOp> fills = rowWriteOps(FakeOpKind.FILL);
        assertEquals("one zeroed row", 1, fills.size());
        assertEquals("pair 3 follows the 6 rows", writes.get(0).offset() + writes.get(0).bytes(),
                fills.get(0).offset());
        assertTrue("expert 4 is only ever the second pick, its row stays zero", allZero(fills.get(0)));
    }

    @Test
    public void prefillRunsTheGroupedGemmOverTheCacheMap() {
        int n = 5;
        List<FakeDispatch> ds = forward(n, preloadedPrefill(n));

        List<String> expected = new ArrayList<>(Arrays.asList("LLM_Batch_Float_MatMul_2D", MoeKernels.ROUTER,
                MoeKernels.BUCKETIZE, MoeKernels.ALIGN_BLOCKS, GG_IND, GG_IND, "LLM_GLU", GG_IND, WSUM));
        expected.addAll(shexp("LLM_Batch_Float_MatMul_2D"));
        expected.add(WSUM);
        assertEquals(expected, names(ds));
        int tiles = MoeKernels.groupedTiles(UBATCH * K, N_EXPERT);
        FakeDispatch align = ds.get(3);
        FakeDispatch gate = ds.get(4);
        assertEquals(MoeKernels.GG_IND_ARG_COUNT, gate.argCount());
        assertSame(cache.bank(0, 0), gate.bufferArg(MoeKernels.ARG_GGI_WEIGHT));
        assertSame(cache.bank(0, 1), ds.get(5).bufferArg(MoeKernels.ARG_GGI_WEIGHT));
        assertSame(cache.mapBuffer(0), gate.bufferArg(MoeKernels.ARG_GGI_MAP));
        assertSame(align.bufferArg(8), gate.bufferArg(MoeKernels.ARG_GGI_GEMM_TOKEN));
        assertSame(align.bufferArg(9), gate.bufferArg(MoeKernels.ARG_GGI_GEMM_SLOT));
        assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, tiles, K, N_EMBD, 0, 0, MoeKernels.SCALE_NONE},
                ints(gate, 0, 8));
        assertArrayEquals(MoeKernels.groupedGemmGlobal(N_FF_EXP, tiles), gate.global());
        FakeDispatch down = ds.get(7);
        assertArrayEquals(new int[]{N_EMBD, N_FF_EXP, tiles, 1, N_FF_EXP}, ints(down, 0, 5));
        assertSame(cache.bank(0, 2), down.bufferArg(MoeKernels.ARG_GGI_WEIGHT));
        assertSame(cache.mapBuffer(0), down.bufferArg(MoeKernels.ARG_GGI_MAP));
        assertEquals(0, rowWrites());
    }

    /** Without {@code F32_GroupedGemmInd} the prefill multiplies per pair over the unaligned sort. */
    @Test
    public void withoutTheMappedGroupedGemmPrefillRunsTheBatchedIndirectionKernels() throws Exception {
        tearDown();
        open(GG_IND);
        int n = 5;
        List<FakeDispatch> ds = forward(n, preloadedPrefill(n));

        List<String> expected = new ArrayList<>(Arrays.asList("LLM_Batch_Float_MatMul_2D", MoeKernels.ROUTER,
                MoeKernels.BUCKETIZE, BMM_ID_IND, BMM_ID_IND, "LLM_GLU", BMM_ID_IND, WSUM));
        expected.addAll(shexp("LLM_Batch_Float_MatMul_2D"));
        expected.add(WSUM);
        assertEquals(expected, names(ds));
        FakeDispatch gate = ds.get(3);
        assertSame(cache.bank(0, 0), gate.bufferArg(MoeKernels.ARG_BMI_WEIGHT));
        assertSame(cache.mapBuffer(0), gate.bufferArg(MoeKernels.ARG_BMI_MAP));
        assertEquals(n * K, gate.intArg(MoeKernels.ARG_BMM_N_ROWS));
        assertEquals(K, gate.intArg(3));
        FakeDispatch down = ds.get(6);
        assertEquals(1, down.intArg(3));
        assertSame(cache.bank(0, 2), down.bufferArg(MoeKernels.ARG_BMI_WEIGHT));
        assertTrue(down.global()[1] >= 1);
    }

    /** {@code n} tokens routed to the preloaded experts 3 and 5, so the step keeps the batched path over the cache. */
    private static int[] preloadedPrefill(int n) {
        int[] ids = new int[n * K];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % 2 == 0 ? 3 : 5;
        }
        return ids;
    }

    @Test
    public void theSecondPrefillStepReusesThePreparedDispatches() {
        int[] ids = {3, 5, 3, 5};
        forward(2, ids);
        int prepared = backend.preparedKernels().size();
        forward(2, ids);
        assertEquals("no kernel is prepared during the steps", prepared, backend.preparedKernels().size());
    }

    @Test
    public void aPrefetchedPrefillComputesNoHostRowAndReadsThePrefillMap() throws Exception {
        rebindWithPrefetch();
        int n = UBATCH;
        int[] ids = new int[n * K];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % N_EXPERT; // every expert demanded, most of them not cached
        }
        List<FakeDispatch> ds = forward(n, ids);

        String dense = ds.get(0).kernelName(); // the router logits' product, the shared expert's too
        List<String> expected = new ArrayList<>(Arrays.asList(dense, MoeKernels.ROUTER,
                MoeKernels.BUCKETIZE, MoeKernels.ALIGN_BLOCKS, GG_IND, GG_IND, "LLM_GLU", GG_IND, WSUM));
        expected.addAll(shexp(dense));
        expected.add(WSUM);
        assertEquals("one dispatch over every expert", expected, names(ds));
        assertEquals("no pair is computed on the host", 0, rowWrites());
        FakeGpuBuffer map = ds.get(4).bufferArg(MoeKernels.ARG_GGI_MAP);
        assertNotSame(cache.mapBuffer(0), map);
        for (int index : new int[]{4, 5, 7}) {
            assertSame(map, ds.get(index).bufferArg(MoeKernels.ARG_GGI_MAP));
        }
        assertSame(cache.bank(0, 0), ds.get(4).bufferArg(MoeKernels.ARG_GGI_WEIGHT));
        assertSame(cache.bank(0, 2), ds.get(7).bufferArg(MoeKernels.ARG_GGI_WEIGHT));
        // the one zone is the whole pool: every expert reads its own slot of the lent zone
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5, 6, 7}, map.getInts(0, N_EXPERT));
        assertEquals(N_EXPERT, prefetch.stepExperts());
    }

    @Test
    public void hostedAndPrefetchedStepsBothCountTheirRoutedIdsIntoTheHeat() throws Exception {
        heat = new MoeExpertHeat();
        rebindWithPrefetch();
        int[] prefill = new int[UBATCH * K];
        for (int i = 0; i < prefill.length; i++) {
            prefill[i] = i % N_EXPERT;
        }
        forward(UBATCH, prefill);
        assertTrue("the prefill step was prefetched", prefetch.lent(0));
        List<RankedExpert> afterPrefill = heat.snapshot();
        assertEquals(N_EXPERT, afterPrefill.size());
        for (RankedExpert e : afterPrefill) {
            assertEquals(e.toString(), 0, e.layer());
            assertEquals(e.toString(), (long) UBATCH * K / N_EXPERT, e.heat());
        }

        forward(1, new int[]{3, 5}); // a decode step keeps its misses on the host
        assertEquals(new RankedExpert(0, 3, (long) UBATCH * K / N_EXPERT + 1), heat.snapshot().get(0));
        long total = 0;
        for (RankedExpert e : heat.snapshot()) {
            total += e.heat();
        }
        assertEquals((long) UBATCH * K + K, total);
    }

    @Test
    public void aSmallBatchAfterAPrefetchedPrefillReadsTheCacheMapAgain() throws Exception {
        rebindWithPrefetch();
        int[] prefill = new int[UBATCH * K];
        for (int i = 0; i < prefill.length; i++) {
            prefill[i] = i % N_EXPERT;
        }
        forward(UBATCH, prefill);
        assertTrue(prefetch.lent(0));

        forward(2, new int[]{3, 5, 3, 5}); // gives the zone back; 3 and 5 refill the restored slots
        assertEquals("a small batch gives the zone back", false, prefetch.lent(0));
        List<FakeDispatch> ds = forward(2, new int[]{3, 5, 3, 5});

        FakeDispatch gate = ds.get(4);
        assertEquals(GG_IND, gate.kernelName());
        assertSame(cache.mapBuffer(0), gate.bufferArg(MoeKernels.ARG_GGI_MAP));
        assertSame(cache.mapBuffer(0), ds.get(7).bufferArg(MoeKernels.ARG_GGI_MAP));
    }
}

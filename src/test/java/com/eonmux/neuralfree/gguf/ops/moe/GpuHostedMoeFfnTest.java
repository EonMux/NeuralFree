package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link GpuMoeFfn} over a hosted layer on the fake backend, on a one-layer synthetic qwen3moe file whose routed
 * experts stay in RAM: with the router emulated, the block output without a cache is the exact routed contribution
 * of the host pairs through the weighted sum, a cache with the split dispatcher copies the routed experts the cost
 * model picks into the banks, their pairs run on the device and the misses upload their host rows before the
 * weighted sum; the unsupported layer kinds fail naming the layer.
 */
public class GpuHostedMoeFfnTest {

    private static final int N_EMBD = 64;
    private static final int N_EXPERT = 4;
    private static final int K = 2;
    private static final int N_FF_EXP = 32;
    private static final int UBATCH = 8;
    /** One expert of one expert tensor, in bytes (the file stores the stacks as F32). */
    private static final int TENSOR_BYTES = N_EMBD * N_FF_EXP * Float.BYTES;
    private static final String ROUTER = MoeKernels.ROUTER;
    private static final String WSUM = MoeKernels.WEIGHTED_SUM;
    private static final String ROUTER_LOGITS = "LLM_Batch_Float_MatMul_2D";
    private static final int ROUTER_ARG_IDS = MoeKernels.ARG_ROUTER_IDS;
    private static final int ROUTER_ARG_WEIGHTS = MoeKernels.ARG_ROUTER_WEIGHTS;

    /** A one-layer synthetic qwen3moe file with its host weights. */
    private static final class HostFile implements AutoCloseable {
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final LayerSpec layer;
        final ModelSpec spec;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;

        HostFile() throws Exception {
            SyntheticModels.Config c = SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build();
            this.path = SyntheticModels.qwen3moe(c).writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            this.weights = CpuWeights.open(parser);
            MoeSpec moe = MoeSpec.builder().experts(c.nExpert(), c.nExpertUsed(), c.nFfExp())
                    .gating(GatingFunc.SOFTMAX).normWeights(true).glu(GluOp.SWIGLU).build();
            this.layer = LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE)
                    .norm(NormSpec.builder(NormKind.RMS, c.normEps()).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, c.nEmbd()).build())
                    .moe(moe)
                    .build();
            ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN3MOE)).layerCounts(1, 0)
                    .nCtxTrain(256).nEmbd(c.nEmbd()).build();
            this.spec = ModelSpec.builder(hp, TensorCatalog.of(Arch.QWEN3MOE.ggufName(), Collections.emptyList()))
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, c.nEmbd()).build())
                    .head(HeadSpec.builder().build())
                    .layer(layer)
                    .build();
            this.gate = weights.layer(layer, TensorNames.weight(TensorNames.FFN_GATE_EXPS));
            this.up = weights.layer(layer, TensorNames.weight(TensorNames.FFN_UP_EXPS));
            this.down = weights.layer(layer, TensorNames.weight(TensorNames.FFN_DOWN_EXPS));
        }

        /** One expert's projection rows as floats (the file's F32 values). */
        float[] expertRows(CpuTensorRef t, int e, int rows) {
            float[] out = new float[rows * (int) t.cols()];
            t.decodeRows((long) e * rows, rows, out, 0);
            return out;
        }

        @Override
        public void close() throws Exception {
            try {
                weights.close();
            } finally {
                try {
                    parser.close();
                } finally {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    private HostFile hostFile;
    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private GpuForwardContext ctx;
    private com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock block;
    private MoeExpertCache cache;
    /** The resident tensors the block reads; the routed expert stacks stay out (the host holds them). */
    private final Map<String, GpuWeight> device = new TreeMap<>();
    /** The routed ids the router emulator writes per step; set before every forward. */
    private int[] routedIds;

    private void open() throws Exception {
        hostFile = new HostFile();
        routedIds = new int[K];
        backend = FakeComputeBackend.builder().strictLlmArgs(true)
                .emulate(ROUTER, GpuHostedMoeFfnTest::emulateRouter)
                .emulate(WSUM, GpuHostedMoeFfnTest::emulateWeightedSum)
                .build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        GpuWeight router = GpuWeight.of(TensorNames.layerName(0, TensorNames.weight(TensorNames.FFN_GATE_INP)),
                GGMLType.F32, new long[]{N_EMBD, N_EXPERT},
                backend.allocateBuffer((long) N_EMBD * N_EXPERT * 4, true));
        device.put(router.name(), router);
        this.ctx = context(null);
    }

    private GpuForwardContext context(MoeExpertCache cache) {
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return device.get(name);
            }
        };
        GpuForwardContext.Builder b = GpuForwardContext.builder(ops.kernels(), ops, weights, hostFile.spec)
                .hostWeights(hostFile.weights);
        if (cache != null) {
            b.moeExpertCache(cache);
        }
        return b.build();
    }

    @After
    public void tearDown() throws Exception {
        RuntimeException failure = null;
        for (AutoCloseable c : new AutoCloseable[]{block, cache, ops, placeholders, scratch, backend, hostFile}) {
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
        if (failure != null) {
            throw failure;
        }
    }

    /** The router emulator: token {@code t} routes to experts {@code t % 4} and {@code (t + 1) % 4}, 0.6 / 0.4. */
    private static void emulateRouter(FakeDispatch d) {
        int nTokens = d.intArg(0);
        int k = d.intArg(2);
        int[] idVals = new int[nTokens * k];
        float[] wVals = new float[nTokens * k];
        for (int t = 0; t < nTokens; t++) {
            idVals[t * k] = t % N_EXPERT;
            idVals[t * k + 1] = (t + 1) % N_EXPERT;
            wVals[t * k] = 0.6f;
            wVals[t * k + 1] = 0.4f;
        }
        writeInts(d.bufferArg(ROUTER_ARG_IDS), idVals);
        writeFloats(d.bufferArg(ROUTER_ARG_WEIGHTS), wVals);
    }

    /**
     * {@code LLM_MoE_WeightedSum(i n_tokens, i n_embd, i k, i flags, R expert_out, R weights, RWn dst)}: row
     * {@code t} of {@code dst} is the weighted sum of the expert rows of its {@code k} pairs (and the previous
     * value under the accumulate flag).
     */
    private static void emulateWeightedSum(FakeDispatch d) {
        int n = d.intArg(0);
        int outDim = d.intArg(1);
        int k = d.intArg(2);
        int flags = d.intArg(3);
        FakeGpuBuffer expertOut = (FakeGpuBuffer) d.bufferArg(4);
        FakeGpuBuffer weights = (FakeGpuBuffer) d.bufferArg(5);
        FakeGpuBuffer dst = (FakeGpuBuffer) d.bufferArg(6);
        float[] rows = expertOut.getFloats(0, (int) (expertOut.sizeInBytes() / Float.BYTES));
        float[] w = weights.getFloats(0, n * k);
        boolean accumulate = (flags & MoeKernels.WSUM_ACCUMULATE) != 0;
        float[] out = dst.getFloats(0, n * outDim);
        for (int t = 0; t < n; t++) {
            for (int e = 0; e < outDim; e++) {
                float acc = accumulate ? out[t * outDim + e] : 0f;
                for (int j = 0; j < k; j++) {
                    acc += rows[(t * k + j) * outDim + e] * w[t * k + j];
                }
                out[t * outDim + e] = acc;
            }
        }
        dst.putFloats(0, out);
    }

    private static void writeInts(GpuBuffer buffer, int[] values) {
        byte[] raw = new byte[values.length * 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().put(values);
        ((FakeGpuBuffer) buffer).write(0, raw, 0, raw.length);
    }

    private static void writeFloats(GpuBuffer buffer, float[] values) {
        byte[] raw = new byte[values.length * 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);
        ((FakeGpuBuffer) buffer).write(0, raw, 0, raw.length);
    }

    private static int[] readInts(GpuBuffer buffer, int n) {
        byte[] raw = new byte[n * 4];
        ((FakeGpuBuffer) buffer).read(0, raw, 0, raw.length);
        int[] out = new int[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }

    private static float[] readFloats(GpuBuffer buffer, int n) {
        byte[] raw = new byte[n * 4];
        ((FakeGpuBuffer) buffer).read(0, raw, 0, raw.length);
        float[] out = new float[n];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    /** The expert input rows of {@code n} tokens, deterministic per row. */
    private static float[] input(int n) {
        Random rnd = new Random(42L);
        float[] x = new float[n * N_EMBD];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() - 0.5f;
        }
        return x;
    }

    /** The routed ids and weights of the emulated router over {@code n} tokens. */
    private static int[] routedIds(int n) {
        int[] ids = new int[n * K];
        for (int t = 0; t < n; t++) {
            ids[t * K] = t % N_EXPERT;
            ids[t * K + 1] = (t + 1) % N_EXPERT;
        }
        return ids;
    }

    private static float[] routedWeights(int n) {
        float[] w = new float[n * K];
        for (int t = 0; t < n; t++) {
            w[t * K] = 0.6f;
            w[t * K + 1] = 0.4f;
        }
        return w;
    }

    /** The reference host contribution of the emulated routing over {@code x}. */
    private float[] reference(int n, float[] x) {
        return referenceOver(n * K, routedIds(n), routedWeights(n), x);
    }

    /** The contribution of {@code pairs} routed pairs over the token rows of {@code x}. */
    private float[] referenceOver(int pairs, int[] ids, float[] w, float[] x) {
        CpuMatVec mv = CpuMatVec.PLAIN;
        float[] gate = new float[N_FF_EXP];
        float[] upRow = new float[N_FF_EXP];
        float[] act = new float[N_FF_EXP];
        float[] pairOut = new float[N_EMBD];
        int tokens = (pairs + K - 1) / K;
        float[] out = new float[tokens * N_EMBD];
        for (int p = 0; p < pairs; p++) {
            int t = p / K;
            Arrays.fill(pairOut, 0f);
            mv.matvecExpert(hostFile.up, ids[p], N_EXPERT, x, t * N_EMBD, upRow, 0);
            mv.matvecExpert(hostFile.gate, ids[p], N_EXPERT, x, t * N_EMBD, gate, 0);
            CpuOps.gluRows(GluOp.SWIGLU, ActParams.NONE, 1, N_FF_EXP, gate, 0, N_FF_EXP, upRow, 0, N_FF_EXP,
                    act, 0, N_FF_EXP);
            mv.matvecExpert(hostFile.down, ids[p], N_EXPERT, act, 0, pairOut, 0);
            CpuOps.axpy(w[p], pairOut, 0, out, t * N_EMBD, N_EMBD);
        }
        return out;
    }

    private float[] runForward(int n) {
        block = new GpuMoeFfn(FfnKind.MOE, hostFile.layer);
        block.bind(ctx, hostFile.layer);
        com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        float[] x = input(n);
        writeFloats(in.buffer(), x);
        backend.clearLog();
        block.forward(n, in, out);
        return x;
    }

    private float[] branchValues(int n) {
        return readFloats(scratch.slot(ScratchSlots.BRANCH).buffer(), n * N_EMBD);
    }

    // ------------------------------------------------------------------ host only (no cache)

    @Test
    public void hostExpertsComputeTheExactRoutedContribution() throws Exception {
        open();
        int n = 5;
        float[] x = runForward(n);
        assertArrayEquals(reference(n, input(n)), branchValues(n), 2e-4f);
        // The hosted pass uploads the miss rows and the weighted sum covers the step.
        assertEquals(Arrays.asList(ROUTER_LOGITS, ROUTER, WSUM), backend.dispatchedKernelNames());
    }

    @Test
    public void everyTokenMayRouteToTheHostWithoutACache() throws Exception {
        open();
        int n = UBATCH;
        runForward(n);
        assertArrayEquals(reference(n, input(n)), branchValues(n), 2e-4f);
    }

    // ------------------------------------------------------------------ with a cache and the dispatcher

    @Test
    public void theDispatcherCopiesGainingExpertsAndTheHostComputesTheRest() throws Exception {
        open();
        // Two slots over four demanded experts: the demand is wider than the pool, so the dispatcher picks the
        // misses whose PCIe copy ends the MoE window sooner, and the host computes the remaining pairs.
        this.cache = MoeExpertCache.open(backend, backend.computeQueue(), cacheData(),
                Collections.singletonList(plan()), 2L * (2L * TENSOR_BYTES + TENSOR_BYTES), UBATCH * K, null,
                "test", new MoeSplitDispatcher(MoeSplitDispatcher.Model.of(8)));
        this.ctx = context(cache);
        int n = 6;
        float[] x = runForward(n);

        int[] map = readInts(cache.mapBuffer(0), N_EXPERT);
        List<Integer> migrated = new ArrayList<>();
        for (int e = 0; e < N_EXPERT; e++) {
            if (map[e] >= 0) {
                migrated.add(e);
            }
        }
        assertFalse("the dispatcher moved at least one expert", migrated.isEmpty());
        assertTrue("the move stays within the free slots", migrated.size() <= 2);

        // The moved slots hold the file's rows: each upload of the up cache equals one expert's rows.
        FakeGpuBuffer cacheUp = (FakeGpuBuffer) cache.bank(0, 1);
        Map<Integer, Integer> slotToExpert = new TreeMap<>();
        for (int[] slotAndCount : uploadsTo(cacheUp)) {
            int expert = expertOf(cacheUp, slotAndCount);
            assertTrue("cache slot " + slotAndCount[0] + " holds no expert of the file", expert >= 0);
            slotToExpert.put(slotAndCount[0], expert);
        }
        assertEquals("every moved expert uploaded exactly once", migrated.size(), slotToExpert.size());
        for (int e : migrated) {
            assertTrue("expert " + e + " uploaded", slotToExpert.containsValue(e));
            assertEquals("expert " + e + " runs from the slot its rows were uploaded to",
                    map[e], keyOf(slotToExpert, e));
        }

        // The branch holds the host contribution only: the device pairs of the migrated experts enter the
        // weighted sum with zero rows on the fake backend, as the (unemulated) id-matvec kernels leave them.
        int[] routed = routedIds(n);
        float[] routedW = routedWeights(n);
        float[] hostW = routedW.clone();
        for (int p = 0; p < n * K; p++) {
            if (map[routed[p]] >= 0) {
                hostW[p] = 0f;
            }
        }
        assertArrayEquals(referenceOver(n * K, routed, hostW, x), branchValues(n), 2e-4f);
    }

    private static int keyOf(Map<Integer, Integer> slotToExpert, int expert) {
        for (Map.Entry<Integer, Integer> e : slotToExpert.entrySet()) {
            if (e.getValue() == expert) {
                return e.getKey();
            }
        }
        return -1;
    }

    /** The WRITE ops into {@code buffer}, as (slot, byte count) pairs in upload order. */
    private List<int[]> uploadsTo(FakeGpuBuffer buffer) {
        List<int[]> out = new ArrayList<>();
        for (com.eonmux.neuralfree.gguf.testutil.FakeOp op : backend.ops()) {
            if (op.kind() == com.eonmux.neuralfree.gguf.testutil.FakeOpKind.WRITE
                    && op.buffer() == buffer) {
                out.add(new int[]{(int) (op.offset() / TENSOR_BYTES), (int) op.bytes()});
            }
        }
        return out;
    }

    /** Which expert's rows the upload at {@code slotAndCount} wrote into the up cache. */
    private int expertOf(FakeGpuBuffer cacheUp, int[] slotAndCount) {
        byte[] got = new byte[slotAndCount[1]];
        cacheUp.read((long) slotAndCount[0] * TENSOR_BYTES, got, 0, got.length);
        int expert = -1;
        for (int e = 0; e < N_EXPERT; e++) {
            if (Arrays.equals(got, bytesOf(hostFile.expertRows(hostFile.up, e, N_FF_EXP)))) {
                expert = e;
            }
        }
        return expert;
    }

    private static byte[] bytesOf(float[] values) {
        byte[] raw = new byte[values.length * 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values);
        return raw;
    }

    private static String nameOf(String suffix) {
        return TensorNames.layerName(0, TensorNames.weight(suffix));
    }

    private WeightDataProvider cacheData() {
        // the cache reads the file's expert rows through one store of the three stacks, F32, laid back to back
        long expertFloats = (long) N_FF_EXP * N_EMBD;
        byte[][] stacks = new byte[3][];
        stacks[0] = bytesOf(allExperts(hostFile.gate));
        stacks[1] = bytesOf(allExperts(hostFile.up));
        stacks[2] = bytesOf(allExperts(hostFile.down));
        byte[] all = new byte[3 * stacks[0].length];
        for (int i = 0; i < 3; i++) {
            System.arraycopy(stacks[i], 0, all, i * stacks[i].length, stacks[i].length);
        }
        WeightStore store = new WeightStore() {
            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst) {
                System.arraycopy(all, Math.toIntExact(tensorOffset), dst, 0, size);
            }

            @Override
            public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
                System.arraycopy(all, Math.toIntExact(tensorOffset), dst, dstOffset, size);
            }

            @Override
            public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
                dst.put(ByteBuffer.wrap(all, Math.toIntExact(tensorOffset), size));
            }

            @Override
            public void close() {
            }
        };
        List<GGUFTensorInfo> infos = new ArrayList<>();
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_GATE_EXPS),
                new long[]{N_EMBD, N_FF_EXP, N_EXPERT}, GGMLType.F32, 0));
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_UP_EXPS),
                new long[]{N_EMBD, N_FF_EXP, N_EXPERT}, GGMLType.F32, N_EXPERT * expertFloats * 4L));
        infos.add(new GGUFTensorInfo(nameOf(TensorNames.FFN_DOWN_EXPS),
                new long[]{N_FF_EXP, N_EMBD, N_EXPERT}, GGMLType.F32, 2L * N_EXPERT * expertFloats * 4L));
        return new WeightDataProvider(store, infos, Collections.singletonList(Arrays.asList(
                nameOf(TensorNames.FFN_GATE_EXPS), nameOf(TensorNames.FFN_UP_EXPS),
                nameOf(TensorNames.FFN_DOWN_EXPS))), 0, 0L);
    }

    /** Every row of one stack tensor, in expert order ({@code rowsPerExpert * cols} each). */
    private float[] allExperts(CpuTensorRef t) {
        int rows = (int) t.rows();
        float[] out = new float[Math.multiplyExact(rows, (int) t.cols())];
        t.decodeRows(0, rows, out, 0);
        return out;
    }

    private MoeExpertCache.LayerPlan plan() {
        List<MoeExpertCache.TensorPlan> tensors = Arrays.asList(
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan(nameOf(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT,
                        GGMLType.F32));
        return new MoeExpertCache.LayerPlan(0, K, tensors);
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void latentExpertsAreRefusedByLayer() {
        MoeSpec latent = MoeSpec.builder().experts(N_EXPERT, K, N_FF_EXP).gating(GatingFunc.SOFTMAX)
                .normWeights(true).glu(GluOp.SWIGLU).latentSize(16).build();
        LayerSpec l = LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE_LATENT)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .moe(latent)
                .build();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> GpuMoeFfn.checkSupported(specOf(l), l));
        assertTrue(e.getMessage().contains("layer 0"));
        assertTrue(e.getMessage().contains("latent"));
    }

    @Test
    public void hashRoutingIsRefused() {
        MoeSpec hash = MoeSpec.builder().experts(N_EXPERT, K, N_FF_EXP).gating(GatingFunc.SOFTMAX)
                .normWeights(true).glu(GluOp.SWIGLU).hashRouting(true).build();
        LayerSpec l = layerOf(hash);
        assertTrue(assertThrows(UnsupportedOperationException.class,
                () -> GpuMoeFfn.checkSupported(specOf(l), l)).getMessage().contains("hash"));
    }

    @Test
    public void weightBeforeFfnExpertsAreRefused() {
        MoeSpec weighted = MoeSpec.builder().experts(N_EXPERT, K, N_FF_EXP).gating(GatingFunc.SOFTMAX)
                .normWeights(true).glu(GluOp.SWIGLU).weightBeforeFfn(true).build();
        LayerSpec l = layerOf(weighted);
        assertTrue(assertThrows(UnsupportedOperationException.class,
                () -> GpuMoeFfn.checkSupported(specOf(l), l)).getMessage().contains("weight-before-FFN"));
    }

    @Test
    public void aHostedLayerWithoutHostTensorsFailsAtBindNamingTheTensor() throws Exception {
        open();
        GpuForwardContext noHost = GpuForwardContext.builder(ops.kernels(), ops,
                new GpuWeights() {
                    @Override
                    public GpuWeight find(String name) {
                        return device.get(name);
                    }
                }, hostFile.spec).build();
        GpuMoeFfn moe = new GpuMoeFfn(FfnKind.MOE, hostFile.layer);
        Exception e = assertThrows(Exception.class, () -> moe.bind(noHost, hostFile.layer));
        assertTrue(e.getMessage(), e.getMessage().contains("_exps.weight"));
    }

    private static LayerSpec layerOf(MoeSpec moe) {
        return LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .moe(moe)
                .build();
    }

    private static ModelSpec specOf(LayerSpec l) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN3MOE)).layerCounts(1, 0)
                .nCtxTrain(256).nEmbd(N_EMBD).build();
        return ModelSpec.builder(hp, TensorCatalog.of(Arch.QWEN3MOE.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }
}

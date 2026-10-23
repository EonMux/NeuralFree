package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnGating;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * Dispatch sequences of {@link GpuMoeFfn} on the host-only fake backend with strict LLM argument rules: the
 * qwen3moe decode step (router, fused gate-up id-matvec, down id-matvec, weighted sum straight into the branch),
 * its prefill step (bucketize then batched id-matmuls re-sized per step only), a qwen2moe sigmoid-gated shared
 * expert accumulated by a one-slot weighted sum, the cohere2moe halved (routed + shared) sum, the kolibri1 selection
 * by biased logits, expert tensors held in several device shards of whole experts (one id-matvec per shard with its
 * expert range, decode and prefill, and the separate gate/up fallback when the two tensors are split differently),
 * the separate one-token gate/up products a gate LoRA delta attached after bind runs without preparing any kernel,
 * and the missing-tensor error.
 */
public class GpuMoeFfnTest {

    private static final int N_EMBD = 64;
    private static final int N_EXPERT = 8;
    private static final int K = 2;
    private static final int N_FF_EXP = 32;
    private static final int N_FF_SHEXP = 48;
    private static final int UBATCH = 8;
    private static final String MATVEC = "Float_MatVec";
    private static final String BATCH_2D = "LLM_Batch_Float_MatMul_2D";
    private static final String GATE_UP = MoeKernels.matVecIdGateUp(GGMLType.F32);
    private static final String MATVEC_ID = MoeKernels.matVecId(GGMLType.F32);
    private static final String BMM_ID = MoeKernels.batchMatMulId(GGMLType.F32);

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private GpuFfnBlock block;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    @After
    public void tearDown() {
        if (block != null) {
            block.close();
        }
        if (ops != null) {
            ops.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    // ------------------------------------------------------------------ fixtures

    private GpuWeight put(String suffix, long... ne) {
        String name = TensorNames.layerName(0, suffix);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        GpuWeight w = GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true));
        tensors.put(name, w);
        return w;
    }

    /**
     * An F32 expert tensor {@code [cols, rows, sum(expertsPerShard)]} held in one device buffer per entry of
     * {@code expertsPerShard}, each holding that many whole experts.
     */
    private GpuWeight putSharded(String suffix, long cols, long rows, int... expertsPerShard) {
        String name = TensorNames.layerName(0, suffix);
        List<GpuBuffer> buffers = new ArrayList<>();
        long[] shardRows = new long[expertsPerShard.length];
        int experts = 0;
        for (int i = 0; i < expertsPerShard.length; i++) {
            shardRows[i] = expertsPerShard[i] * rows;
            buffers.add(backend.allocateBuffer(shardRows[i] * cols * Float.BYTES, true));
            experts += expertsPerShard[i];
        }
        GpuWeight w = GpuWeight.sharded(name, GGMLType.F32, new long[]{cols, rows, experts}, buffers, shardRows);
        tensors.put(name, w);
        return w;
    }

    /** Router and separate gate/up/down expert tensors in llama.cpp shapes. */
    private void putExperts() {
        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT);
    }

    private static MoeSpec.Builder qwen3moe() {
        return MoeSpec.builder().experts(N_EXPERT, K, N_FF_EXP).gating(GatingFunc.SOFTMAX).normWeights(true)
                .glu(GluOp.SWIGLU);
    }

    private static LayerSpec layer(MoeSpec moe) {
        return LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .moe(moe)
                .build();
    }

    private GpuForwardContext context(Arch arch, LayerSpec l) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nCtxTrain(4096)
                .nEmbd(N_EMBD).build();
        ModelSpec spec = ModelSpec.builder(hp, TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return tensors.get(name);
            }
        };
        return GpuForwardContext.builder(ops.kernels(), ops, weights, spec).build();
    }

    private GpuFfnBlock create(LayerSpec l) {
        for (GpuFfnFactory f : MoeBlocks.gpuFfns()) {
            if (f.kind() == l.ffnKind()) {
                return f.create(l);
            }
        }
        throw new AssertionError("no MoE factory for " + l.ffnKind());
    }

    private void bind(Arch arch, LayerSpec l) {
        block = create(l);
        block.bind(context(arch, l), l);
    }

    private List<Dispatch> run(int nTokens) {
        backend.clearLog();
        block.forward(nTokens, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private GpuBuffer normed() {
        return scratch.slot(ScratchSlots.NORMED).buffer();
    }

    private GpuBuffer branch() {
        return scratch.slot(ScratchSlots.BRANCH).buffer();
    }

    private GpuBuffer weight(String suffix) {
        return tensors.get(TensorNames.layerName(0, TensorNames.weight(suffix))).shard(0);
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    /** {@code {expert_base, expert_count}} of an id-matvec dispatch whose base is argument {@code baseArg}. */
    private static int[] expertRange(Dispatch d, int baseArg) {
        return ints(d, baseArg, baseArg + 2);
    }

    private static int[] ints(Dispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    // ------------------------------------------------------------------ qwen3moe decode

    @Test
    public void qwen3moeDecodeRoutesOnDeviceAndSumsIntoTheBranch() {
        open();
        putExperts();
        bind(Arch.QWEN3MOE, layer(qwen3moe().build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM),
                names(ds));
        Dispatch logits = ds.get(0);
        assertSame(weight(TensorNames.FFN_GATE_INP), logits.bufferArg(2));
        assertSame(normed(), logits.bufferArg(3));

        Dispatch router = ds.get(1);
        assertEquals(MoeKernels.ROUTER_ARG_COUNT, router.argCount());
        assertArrayEquals(new int[]{1, N_EXPERT, K, GatingFunc.SOFTMAX.code(), 1, 1, 1}, ints(router, 0, 7));
        assertSame(logits.bufferArg(4), router.bufferArg(11));
        assertEquals(1, router.global().length);
        assertEquals(router.local()[0], router.global()[0]);
        GpuBuffer ids = router.bufferArg(13);
        GpuBuffer weights = router.bufferArg(14);

        Dispatch gu = ds.get(2);
        assertEquals(MoeKernels.GU_ARG_COUNT, gu.argCount());
        assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, K, 0, 0, GluOp.SWIGLU.code()}, ints(gu, 0, 6));
        assertEquals(0, gu.intArg(MoeKernels.ARG_GU_FUSED));
        assertEquals(MoeKernels.SCALE_NONE, gu.intArg(MoeKernels.ARG_GU_HAS_SCALE));
        assertSame(ids, gu.bufferArg(MoeKernels.ARG_GU_SLOTS));
        assertSame(weight(TensorNames.FFN_GATE_EXPS), gu.bufferArg(MoeKernels.ARG_GU_GATE));
        assertSame(weight(TensorNames.FFN_UP_EXPS), gu.bufferArg(MoeKernels.ARG_GU_UP));
        assertSame(normed(), gu.bufferArg(MoeKernels.ARG_GU_INPUT));
        int idLocal = (int) gu.local()[0];
        assertArrayEquals(new long[]{(long) N_FF_EXP * idLocal, K}, gu.global());
        assertArrayEquals(new int[]{0, N_EXPERT}, expertRange(gu, MoeKernels.ARG_GU_EXPERT_BASE));
        GpuBuffer act = gu.bufferArg(MoeKernels.ARG_GU_OUTPUT);

        Dispatch down = ds.get(3);
        assertEquals(MoeKernels.ID_ARG_COUNT, down.argCount());
        assertArrayEquals(new int[]{N_EMBD, N_FF_EXP, K, N_FF_EXP, 0, 0}, ints(down, 0, 6));
        assertSame(ids, down.bufferArg(MoeKernels.ARG_ID_SLOTS));
        assertSame(weight(TensorNames.FFN_DOWN_EXPS), down.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(act, down.bufferArg(MoeKernels.ARG_ID_INPUT));
        assertArrayEquals(new int[]{0, N_EXPERT}, expertRange(down, MoeKernels.ARG_ID_EXPERT_BASE));
        GpuBuffer expertOut = down.bufferArg(MoeKernels.ARG_ID_OUTPUT);
        assertNotSame(branch(), expertOut);

        Dispatch wsum = ds.get(4);
        assertArrayEquals(new int[]{1, N_EMBD, K, 0}, ints(wsum, 0, 4));
        assertSame(expertOut, wsum.bufferArg(4));
        assertSame(weights, wsum.bufferArg(5));
        assertSame(branch(), wsum.bufferArg(6));
    }

    @Test
    public void gateDeltaAttachedAfterBindRunsTheSplitProductsPreparedAtBind() {
        open();
        putExperts();
        LayerSpec l = layer(qwen3moe().build());
        GpuForwardContext ctx = context(Arch.QWEN3MOE, l);
        block = create(l);
        block.bind(ctx, l);
        int preparedAtBind = backend.preparedKernels().size();
        List<int[]> calls = new ArrayList<>();
        ctx.projectionDeltas().registerExperts(
                GpuProjectionDeltas.key(l, TensorNames.weight(TensorNames.FFN_GATE_EXPS)),
                (n, k, ids, x, xOff, tokenRows, y, yOff) -> calls.add(new int[]{n, k}));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, MATVEC_ID, MATVEC_ID, ActKernels.GLU, MATVEC_ID,
                MoeKernels.WEIGHTED_SUM), names(ds));
        assertEquals("no kernel is prepared during the step", preparedAtBind, backend.preparedKernels().size());
        assertEquals(1, calls.size());
        assertArrayEquals(new int[]{1, K}, calls.get(0));
        Dispatch gateMm = ds.get(2);
        assertSame(weight(TensorNames.FFN_GATE_EXPS), gateMm.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(normed(), gateMm.bufferArg(MoeKernels.ARG_ID_INPUT));
        Dispatch upMm = ds.get(3);
        assertSame(weight(TensorNames.FFN_UP_EXPS), upMm.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(normed(), upMm.bufferArg(MoeKernels.ARG_ID_INPUT));
    }

    // ------------------------------------------------------------------ qwen3moe prefill

    @Test
    public void qwen3moePrefillBucketizesThenRunsBatchedIdMatmuls() {
        open();
        putExperts();
        bind(Arch.QWEN3MOE, layer(qwen3moe().build()));
        int n = 5;
        List<Dispatch> ds = run(n);

        assertEquals(Arrays.asList(BATCH_2D, MoeKernels.ROUTER, MoeKernels.BUCKETIZE, BMM_ID, BMM_ID,
                ActKernels.GLU, BMM_ID, MoeKernels.WEIGHTED_SUM), names(ds));
        Dispatch router = ds.get(1);
        assertEquals(n, router.intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(n * router.local()[0], router.global()[0]);

        Dispatch bucket = ds.get(2);
        assertArrayEquals(new int[]{n, K, N_EXPERT}, ints(bucket, 0, 3));
        assertSame(router.bufferArg(13), bucket.bufferArg(3));
        assertEquals(bucket.local()[0], bucket.global()[0]);
        GpuBuffer rowToken = bucket.bufferArg(6);
        GpuBuffer rowSlot = bucket.bufferArg(7);

        int pairs = n * K;
        long tiles = (pairs + MoeKernels.BMM_ROWS - 1) / MoeKernels.BMM_ROWS;
        Dispatch gate = ds.get(3);
        Dispatch up = ds.get(4);
        Dispatch down = ds.get(6);
        assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, pairs, K, 0, MoeKernels.SCALE_NONE}, ints(gate, 0, 6));
        assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, pairs, K, 0, MoeKernels.SCALE_NONE}, ints(up, 0, 6));
        assertArrayEquals(new int[]{N_EMBD, N_FF_EXP, pairs, 1, 0, MoeKernels.SCALE_NONE}, ints(down, 0, 6));
        for (Dispatch d : Arrays.asList(gate, up, down)) {
            assertSame(rowToken, d.bufferArg(MoeKernels.ARG_BMM_ROW_TOKEN));
            assertSame(rowSlot, d.bufferArg(MoeKernels.ARG_BMM_ROW_SLOT));
            assertEquals(tiles, d.global()[1]);
            assertArrayEquals(new int[]{0, N_EXPERT}, expertRange(d, MoeKernels.ARG_BMM_EXPERT_BASE));
        }
        assertSame(weight(TensorNames.FFN_GATE_EXPS), gate.bufferArg(MoeKernels.ARG_BMM_WEIGHT));
        assertSame(weight(TensorNames.FFN_UP_EXPS), up.bufferArg(MoeKernels.ARG_BMM_WEIGHT));
        assertSame(normed(), gate.bufferArg(MoeKernels.ARG_BMM_INPUT));
        assertSame(normed(), up.bufferArg(MoeKernels.ARG_BMM_INPUT));

        Dispatch glu = ds.get(5);
        assertEquals(pairs, glu.intArg(ActKernels.ARG_GLU_N_ROWS));
        assertSame(gate.bufferArg(MoeKernels.ARG_BMM_OUTPUT), glu.bufferArg(ActKernels.ARG_GLU_GATE));
        assertSame(up.bufferArg(MoeKernels.ARG_BMM_OUTPUT), glu.bufferArg(ActKernels.ARG_GLU_UP));
        assertSame(glu.bufferArg(ActKernels.ARG_GLU_OUT), down.bufferArg(MoeKernels.ARG_BMM_INPUT));

        Dispatch wsum = ds.get(7);
        assertArrayEquals(new int[]{n, N_EMBD, K, 0}, ints(wsum, 0, 4));
        assertSame(down.bufferArg(MoeKernels.ARG_BMM_OUTPUT), wsum.bufferArg(4));
        assertSame(branch(), wsum.bufferArg(6));

        List<Dispatch> again = run(1);
        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM),
                names(again));
        assertEquals(1, again.get(1).intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(1, again.get(4).intArg(MoeKernels.ARG_N_TOKENS));
    }

    // ------------------------------------------------------------------ qwen2moe gated shared expert

    @Test
    public void qwen2moeSharedExpertIsSigmoidGatedAndAccumulated() {
        open();
        putExperts();
        put(TensorNames.weight(TensorNames.FFN_GATE_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_UP_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_DOWN_SHEXP), N_FF_SHEXP, N_EMBD);
        put(TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP), N_EMBD);
        FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(N_FF_SHEXP)
                .glu(FfnGating.PAR, GluOp.SWIGLU).build();
        bind(Arch.QWEN2MOE, layer(qwen3moe().normWeights(false).sharedExpert(shexp, true).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM,
                MATVEC, MATVEC, ActKernels.GLU, MATVEC, MATVEC, ActKernels.UNARY, MoeKernels.WEIGHTED_SUM),
                names(ds));
        Dispatch shDown = ds.get(8);
        assertSame(weight(TensorNames.FFN_DOWN_SHEXP), shDown.bufferArg(2));
        Dispatch gateLogit = ds.get(9);
        assertSame(weight(TensorNames.FFN_GATE_INP_SHEXP), gateLogit.bufferArg(2));
        assertSame(normed(), gateLogit.bufferArg(3));
        Dispatch sigmoid = ds.get(10);
        assertEquals(ActOp.SIGMOID.code(), sigmoid.intArg(1));
        Dispatch acc = ds.get(11);
        assertArrayEquals(new int[]{1, N_EMBD, 1, MoeKernels.WSUM_ACCUMULATE}, ints(acc, 0, 4));
        assertSame(shDown.bufferArg(4), acc.bufferArg(4));
        assertSame(gateLogit.bufferArg(4), acc.bufferArg(5));
        assertSame(branch(), acc.bufferArg(6));
    }

    // ------------------------------------------------------------------ cohere2moe halved shared combine

    /**
     * cohere2moe adds the ungated shared expert to the routed output and halves the sum
     * (cohere2moe.cpp:241-243): an {@code LLM_Axpby_F32} add into the branch, then an in-place scale by 0.5.
     */
    @Test
    public void cohere2moeHalvesTheRoutedPlusSharedSum() {
        open();
        putExperts();
        put(TensorNames.weight(TensorNames.FFN_GATE_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_UP_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_DOWN_SHEXP), N_FF_SHEXP, N_EMBD);
        FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(N_FF_SHEXP)
                .glu(FfnGating.PAR, GluOp.SWIGLU).build();
        bind(Arch.COHERE2MOE, layer(qwen3moe().sharedExpert(shexp, false).sharedCombineScale(0.5f).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM,
                MATVEC, MATVEC, ActKernels.GLU, MATVEC, GpuOps.AXPBY, GpuOps.AXPBY), names(ds));
        Dispatch shDown = ds.get(8);
        Dispatch add = ds.get(9);
        assertEquals(N_EMBD, add.intArg(0));
        assertEquals(1.0f, add.floatArg(1), 0f);
        assertEquals(1.0f, add.floatArg(2), 0f);
        assertSame(shDown.bufferArg(4), add.bufferArg(6));
        assertSame(branch(), add.bufferArg(7));
        assertSame(branch(), add.bufferArg(8));
        Dispatch half = ds.get(10);
        assertEquals(N_EMBD, half.intArg(0));
        assertEquals(0.0f, half.floatArg(1), 0f);
        assertEquals(0.5f, half.floatArg(2), 0f);
        assertSame(branch(), half.bufferArg(7));
        assertSame(branch(), half.bufferArg(8));
    }

    /** Without a combine scale the ungated shared expert is added once and nothing scales the sum. */
    @Test
    public void ungatedSharedExpertWithoutCombineScaleIsOnlyAdded() {
        open();
        putExperts();
        put(TensorNames.weight(TensorNames.FFN_GATE_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_UP_SHEXP), N_EMBD, N_FF_SHEXP);
        put(TensorNames.weight(TensorNames.FFN_DOWN_SHEXP), N_FF_SHEXP, N_EMBD);
        FfnSpec shexp = FfnSpec.builder(FfnSpec.Role.SHARED_EXPERT).nFf(N_FF_SHEXP)
                .glu(FfnGating.PAR, GluOp.SWIGLU).build();
        bind(Arch.QWEN3MOE, layer(qwen3moe().sharedExpert(shexp, false).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM,
                MATVEC, MATVEC, ActKernels.GLU, MATVEC, GpuOps.AXPBY), names(ds));
    }

    // ------------------------------------------------------------------ kolibri1 biased-logit selection

    @Test
    public void kolibri1PicksByBiasedLogitsThenWeighsTheGivenIds() {
        open();
        putExperts();
        put(TensorNames.bias(TensorNames.FFN_EXP_PROBS_B), N_EXPERT);
        bind(Arch.KOLIBRI1, layer(qwen3moe().gating(GatingFunc.SIGMOID).probBias(true).build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, GpuOps.COPY2D, MoeKernels.ADD_ID, MoeKernels.ROUTER, MoeKernels.ROUTER,
                GATE_UP, MATVEC_ID, MoeKernels.WEIGHTED_SUM), names(ds));
        GpuBuffer logits = ds.get(0).bufferArg(4);
        Dispatch bias = ds.get(2);
        assertArrayEquals(new int[]{1, N_EXPERT}, ints(bias, 0, 2));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B))).shard(0),
                bias.bufferArg(3));
        GpuBuffer selection = bias.bufferArg(4);
        assertNotSame(logits, selection);

        Dispatch pick = ds.get(3);
        assertEquals(MoeKernels.SEL_RAW_LOGITS, pick.intArg(9));
        assertEquals(0, pick.intArg(8));
        assertSame(selection, pick.bufferArg(11));
        Dispatch weigh = ds.get(4);
        assertEquals(MoeKernels.SEL_PRESELECTED, weigh.intArg(9));
        assertEquals(GatingFunc.SIGMOID.code(), weigh.intArg(3));
        assertEquals(1, weigh.intArg(6));
        assertSame(logits, weigh.bufferArg(11));
        assertSame(pick.bufferArg(13), weigh.bufferArg(13));
        assertSame(weigh.bufferArg(13), ds.get(5).bufferArg(MoeKernels.ARG_GU_SLOTS));
        assertSame(weigh.bufferArg(14), ds.get(7).bufferArg(5));
    }

    // ------------------------------------------------------------------ sharded experts

    /** Experts 0-4 in the first shard, 5-7 in the second, for gate, up and down alike. */
    private static final int[] SHARDS = {5, 3};
    private static final int[][] RANGES = {{0, 5}, {5, 3}};

    @Test
    public void shardedExpertsDecodeWithOneIdMatvecPerShard() {
        open();
        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        GpuWeight gate = putSharded(TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, SHARDS);
        GpuWeight up = putSharded(TensorNames.weight(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, SHARDS);
        GpuWeight down = putSharded(TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, SHARDS);
        bind(Arch.QWEN3MOE, layer(qwen3moe().build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, GATE_UP, MATVEC_ID, MATVEC_ID,
                MoeKernels.WEIGHTED_SUM), names(ds));
        GpuBuffer ids = ds.get(1).bufferArg(13);
        GpuBuffer act = ds.get(2).bufferArg(MoeKernels.ARG_GU_OUTPUT);
        GpuBuffer expertOut = ds.get(4).bufferArg(MoeKernels.ARG_ID_OUTPUT);
        for (int i = 0; i < SHARDS.length; i++) {
            Dispatch gu = ds.get(2 + i);
            assertArrayEquals(new int[]{N_FF_EXP, N_EMBD, K, 0, 0, GluOp.SWIGLU.code()}, ints(gu, 0, 6));
            assertArrayEquals(RANGES[i], expertRange(gu, MoeKernels.ARG_GU_EXPERT_BASE));
            assertSame(gate.shard(i), gu.bufferArg(MoeKernels.ARG_GU_GATE));
            assertSame(up.shard(i), gu.bufferArg(MoeKernels.ARG_GU_UP));
            assertSame(ids, gu.bufferArg(MoeKernels.ARG_GU_SLOTS));
            assertSame(normed(), gu.bufferArg(MoeKernels.ARG_GU_INPUT));
            assertSame(act, gu.bufferArg(MoeKernels.ARG_GU_OUTPUT));

            Dispatch dn = ds.get(4 + i);
            assertArrayEquals(new int[]{N_EMBD, N_FF_EXP, K, N_FF_EXP, 0, 0}, ints(dn, 0, 6));
            assertArrayEquals(RANGES[i], expertRange(dn, MoeKernels.ARG_ID_EXPERT_BASE));
            assertSame(down.shard(i), dn.bufferArg(MoeKernels.ARG_ID_WEIGHT));
            assertSame(ids, dn.bufferArg(MoeKernels.ARG_ID_SLOTS));
            assertSame(act, dn.bufferArg(MoeKernels.ARG_ID_INPUT));
            assertSame(expertOut, dn.bufferArg(MoeKernels.ARG_ID_OUTPUT));
        }
        assertSame(expertOut, ds.get(6).bufferArg(4));
        assertSame(branch(), ds.get(6).bufferArg(6));
    }

    @Test
    public void shardedExpertsPrefillWithOneBatchedIdMatmulPerShard() {
        open();
        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        GpuWeight gate = putSharded(TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, SHARDS);
        GpuWeight up = putSharded(TensorNames.weight(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, SHARDS);
        GpuWeight down = putSharded(TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, SHARDS);
        bind(Arch.QWEN3MOE, layer(qwen3moe().build()));
        int n = 5;
        int pairs = n * K;
        List<Dispatch> ds = run(n);

        assertEquals(Arrays.asList(BATCH_2D, MoeKernels.ROUTER, MoeKernels.BUCKETIZE, BMM_ID, BMM_ID, BMM_ID, BMM_ID,
                ActKernels.GLU, BMM_ID, BMM_ID, MoeKernels.WEIGHTED_SUM), names(ds));
        Dispatch bucket = ds.get(2);
        GpuWeight[] tensorsInOrder = {gate, up, down};
        int[] first = {3, 5, 8};
        for (int t = 0; t < tensorsInOrder.length; t++) {
            GpuBuffer out = ds.get(first[t]).bufferArg(MoeKernels.ARG_BMM_OUTPUT);
            for (int i = 0; i < SHARDS.length; i++) {
                Dispatch d = ds.get(first[t] + i);
                assertEquals(pairs, d.intArg(MoeKernels.ARG_BMM_N_ROWS));
                assertArrayEquals(RANGES[i], expertRange(d, MoeKernels.ARG_BMM_EXPERT_BASE));
                assertSame(tensorsInOrder[t].shard(i), d.bufferArg(MoeKernels.ARG_BMM_WEIGHT));
                assertSame(bucket.bufferArg(6), d.bufferArg(MoeKernels.ARG_BMM_ROW_TOKEN));
                assertSame(bucket.bufferArg(7), d.bufferArg(MoeKernels.ARG_BMM_ROW_SLOT));
                assertSame(out, d.bufferArg(MoeKernels.ARG_BMM_OUTPUT));
            }
        }
        assertSame(ds.get(8).bufferArg(MoeKernels.ARG_BMM_OUTPUT), ds.get(10).bufferArg(4));

        List<Dispatch> again = run(1);
        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, GATE_UP, GATE_UP, MATVEC_ID, MATVEC_ID,
                MoeKernels.WEIGHTED_SUM), names(again));
    }

    /**
     * Gate split 5 + 3 but up in one buffer: no dispatch can bind a shard of each, so gate and up run as separate
     * id-matvecs over their own shards, followed by {@code LLM_GLU}.
     */
    @Test
    public void differentlyShardedGateAndUpFallBackToSeparateProducts() {
        open();
        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        GpuWeight gate = putSharded(TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, SHARDS);
        GpuWeight up = put(TensorNames.weight(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT);
        bind(Arch.QWEN3MOE, layer(qwen3moe().build()));
        List<Dispatch> ds = run(1);

        assertEquals(Arrays.asList(MATVEC, MoeKernels.ROUTER, MATVEC_ID, MATVEC_ID, MATVEC_ID, ActKernels.GLU,
                MATVEC_ID, MoeKernels.WEIGHTED_SUM), names(ds));
        for (int i = 0; i < SHARDS.length; i++) {
            Dispatch g = ds.get(2 + i);
            assertArrayEquals(RANGES[i], expertRange(g, MoeKernels.ARG_ID_EXPERT_BASE));
            assertSame(gate.shard(i), g.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        }
        Dispatch u = ds.get(4);
        assertArrayEquals(new int[]{0, N_EXPERT}, expertRange(u, MoeKernels.ARG_ID_EXPERT_BASE));
        assertSame(up.shard(0), u.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        Dispatch glu = ds.get(5);
        assertSame(ds.get(2).bufferArg(MoeKernels.ARG_ID_OUTPUT), glu.bufferArg(ActKernels.ARG_GLU_GATE));
        assertSame(u.bufferArg(MoeKernels.ARG_ID_OUTPUT), glu.bufferArg(ActKernels.ARG_GLU_UP));
    }

    // ------------------------------------------------------------------ errors

    @Test
    public void missingExpertTensorFailsAtBind() {
        open();
        put(TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT);
        LayerSpec l = layer(qwen3moe().build());
        block = create(l);
        GpuForwardContext ctx = context(Arch.QWEN3MOE, l);
        assertThrows(MissingTensorException.class, () -> block.bind(ctx, l));
    }

    @Test
    public void kindMismatchFailsAtCreate() {
        LayerSpec l = layer(qwen3moe().build());
        assertThrows(IllegalArgumentException.class, () -> new GpuMoeFfn(FfnKind.MOE_LATENT, l));
    }
}

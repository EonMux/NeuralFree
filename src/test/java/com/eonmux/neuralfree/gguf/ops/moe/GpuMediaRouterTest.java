package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnFactory;
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
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.ScratchWidths;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeDispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeOp;
import com.eonmux.neuralfree.gguf.testutil.FakeOpKind;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The embedding-row routing of {@link GpuMoeFfn} on the host-only fake backend ({@link GpuMediaRouter},
 * deepseek4.cpp:1290-1301): on a hash layer with {@code exp_probs_b_vl}, a step holding embedding rows (host step
 * token {@code -1}) runs a second {@code LLM_MoE_Router} over the same logits with the router's own top-k and
 * {@code exp_probs_b_vl}, then copies its ids and weights over the hash selection for each run of embedding rows,
 * before the experts read them; a step of token rows dispatches neither, and without {@code exp_probs_b_vl} the
 * second router has no bias.
 */
public class GpuMediaRouterTest {

    private static final int N_EMBD = 64;
    private static final int N_EXPERT = 8;
    private static final int K = 2;
    private static final int N_FF_EXP = 32;
    private static final int N_VOCAB = 16;
    private static final int UBATCH = 8;
    /** {@code LLM_MoE_Router} arguments: has_bias, sel_mode, per_group, logits, exp_probs_b, ids, weights. */
    private static final int ARG_HAS_BIAS = 8;
    private static final int ARG_SEL_MODE = 9;
    private static final int ARG_LOGITS = MoeKernels.ARG_ROUTER_LOGITS;
    private static final int ARG_BIAS = MoeKernels.ARG_ROUTER_EXP_PROBS_B;
    private static final int ARG_IDS = MoeKernels.ARG_ROUTER_IDS;
    private static final int ARG_WEIGHTS = MoeKernels.ARG_ROUTER_WEIGHTS;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private GpuFfnBlock block;

    private void open(boolean visionBias) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.TOKEN_IDS, ScratchWidths.rows(UBATCH, ScratchWidths.TOKEN_ID_INTS))
                .slot(ScratchSlots.X, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        put(GGMLType.F32, TensorNames.weight(TensorNames.FFN_GATE_INP), N_EMBD, N_EXPERT);
        put(GGMLType.F32, TensorNames.weight(TensorNames.FFN_GATE_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(GGMLType.F32, TensorNames.weight(TensorNames.FFN_UP_EXPS), N_EMBD, N_FF_EXP, N_EXPERT);
        put(GGMLType.F32, TensorNames.weight(TensorNames.FFN_DOWN_EXPS), N_FF_EXP, N_EMBD, N_EXPERT);
        put(GGMLType.I32, TensorNames.weight(TensorNames.FFN_GATE_TID2EID), K, N_VOCAB);
        if (visionBias) {
            put(GGMLType.F32, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B_VL), N_EXPERT);
        }
        MoeSpec moe = MoeSpec.builder().experts(N_EXPERT, K, N_FF_EXP).gating(GatingFunc.SOFTMAX).normWeights(true)
                .glu(GluOp.SWIGLU).hashRouting(true).mediaProbBias(visionBias).build();
        LayerSpec l = LayerSpec.builder(0, MixerKind.NONE, FfnKind.MOE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .moe(moe)
                .build();
        block = create(l);
        block.bind(context(l), l);
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

    private void put(GGMLType type, String suffix, long... ne) {
        String name = TensorNames.layerName(0, suffix);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        tensors.put(name, GpuWeight.of(name, type, ne, backend.allocateBuffer(n * 4, true)));
    }

    private GpuForwardContext context(LayerSpec l) {
        Arch arch = Arch.QWEN3MOE;
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

    private static GpuFfnBlock create(LayerSpec l) {
        for (GpuFfnFactory f : MoeBlocks.gpuFfns()) {
            if (f.kind() == l.ffnKind()) {
                return f.create(l);
            }
        }
        throw new AssertionError("no MoE factory for " + l.ffnKind());
    }

    /** One step over {@code stepTokens} (the host record the embedding stage keeps); the recorded operations. */
    private List<FakeOp> run(int... stepTokens) {
        scratch.recordStepTokens(stepTokens, stepTokens.length);
        backend.clearLog();
        block.forward(stepTokens.length, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.ops();
    }

    private static List<Integer> routerOps(List<FakeOp> ops) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < ops.size(); i++) {
            if (ops.get(i).kind() == FakeOpKind.DISPATCH
                    && ops.get(i).dispatch().kernelName().equals(MoeKernels.ROUTER)) {
                out.add(i);
            }
        }
        return out;
    }

    private static List<FakeOp> copies(List<FakeOp> ops) {
        List<FakeOp> out = new ArrayList<>();
        for (FakeOp op : ops) {
            if (op.kind() == FakeOpKind.COPY) {
                out.add(op);
            }
        }
        return out;
    }

    @Test
    public void embeddingRowsTakeTheRouterSelectionWithTheVisionBias() {
        open(true);
        List<FakeOp> ops = run(5, -1, -1, 7, -1);
        List<Integer> routers = routerOps(ops);
        assertEquals("the hash weights, then the embedding rows' own selection", 2, routers.size());
        FakeDispatch hash = ops.get(routers.get(0)).dispatch();
        FakeDispatch media = ops.get(routers.get(1)).dispatch();
        assertEquals(MoeKernels.SEL_PRESELECTED, hash.intArg(ARG_SEL_MODE));
        assertEquals("the router's own top-k", MoeSpec.SelectionMode.PROBS.code(), media.intArg(ARG_SEL_MODE));
        assertEquals(5, media.intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(1, media.intArg(ARG_HAS_BIAS));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B_VL))).shard(0),
                media.bufferArg(ARG_BIAS));
        assertSame("the same logits", hash.bufferArg(ARG_LOGITS), media.bufferArg(ARG_LOGITS));
        FakeGpuBuffer ids = hash.bufferArg(ARG_IDS);
        FakeGpuBuffer weights = hash.bufferArg(ARG_WEIGHTS);
        assertNotSame(ids, media.bufferArg(ARG_IDS));

        List<FakeOp> copies = copies(ops);
        assertEquals("ids and weights of the two runs of embedding rows", 4, copies.size());
        long rowBytes = (long) K * Float.BYTES;
        assertCopy(copies.get(0), media.bufferArg(ARG_IDS), ids, rowBytes, 2 * rowBytes);
        assertCopy(copies.get(1), media.bufferArg(ARG_WEIGHTS), weights, rowBytes, 2 * rowBytes);
        assertCopy(copies.get(2), media.bufferArg(ARG_IDS), ids, 4 * rowBytes, rowBytes);
        assertCopy(copies.get(3), media.bufferArg(ARG_WEIGHTS), weights, 4 * rowBytes, rowBytes);
        int lastCopy = ops.lastIndexOf(copies.get(3));
        assertTrue("the copies follow the second router", ops.indexOf(copies.get(0)) > routers.get(1));
        for (int i = lastCopy + 1; i < ops.size(); i++) {
            assertTrue("no router after the copies", ops.get(i).kind() != FakeOpKind.DISPATCH
                    || !ops.get(i).dispatch().kernelName().equals(MoeKernels.ROUTER));
        }
        assertTrue("the experts run after the copies", ops.size() > lastCopy + 1);
    }

    private static void assertCopy(FakeOp copy, FakeGpuBuffer src, FakeGpuBuffer dst, long at, long bytes) {
        assertSame(src, copy.source());
        assertSame(dst, copy.buffer());
        assertEquals("source offset", at, copy.sourceOffset());
        assertEquals("destination offset", at, copy.offset());
        assertEquals(bytes, copy.bytes());
    }

    @Test
    public void aStepOfTokenRowsKeepsTheHashSelection() {
        open(true);
        List<FakeOp> ops = run(5, 9, 7);
        assertEquals(1, routerOps(ops).size());
        assertEquals(0, copies(ops).size());
    }

    @Test
    public void withoutTheVisionBiasEmbeddingRowsSelectUnbiased() {
        open(false);
        List<FakeOp> ops = run(-1, -1, -1);
        List<Integer> routers = routerOps(ops);
        assertEquals(2, routers.size());
        FakeDispatch media = ops.get(routers.get(1)).dispatch();
        assertEquals("a hash layer has no exp_probs_b", 0, media.intArg(ARG_HAS_BIAS));
        List<FakeOp> copies = copies(ops);
        assertEquals("one run of every row", 2, copies.size());
        assertEquals(0L, copies.get(0).offset());
        assertEquals(3L * K * Float.BYTES, copies.get(0).bytes());
    }
}

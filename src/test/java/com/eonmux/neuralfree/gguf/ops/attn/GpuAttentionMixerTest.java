package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.FusedMode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOpsFixture;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.fused.FusedKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FopeSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of {@link GpuAttentionMixer} against the host-only fake backend with strict LLM argument rules,
 * recorded for a llama-shaped GQA layer (4 query heads, 2 KV heads, head dim 64, NORM RoPE over a real
 * {@link DeviceModelMemory}): the decode and prefill sequences of the mixer's own kernels with their sticky and
 * per-step arguments, the Q8_0 Hadamard rotation, sinks and softcap, the qwen3 QK norm, Q/K norms after RoPE, the
 * whole-projection LayerNorm of mpt and the BERT graph, suppressed norm and output biases, the softplus gates, the
 * MoVA value experts, the encoder path (bert: biases, no cache, non-causal), deci's linear replacement and the RoPE
 * context shift.
 */
public class GpuAttentionMixerTest {

    private static final int WIDTH = 256;
    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int HEAD_DIM = 64;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final float KQ_SCALE = 0.125f;

    /** The mixer's own kernels (projection GEMMs excluded), in the order a trace lists them. */
    private static final Set<String> OWN = own();

    private static Set<String> own() {
        Set<String> s = new HashSet<>(AttnBlocks.KERNELS);
        s.addAll(AttnBlocks.VALUE_EXPERT_KERNELS);
        return s;
    }

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuMixer mixer;
    private Map<String, GpuWeight> tensors;
    /** Fused-kernel selection of the next bind; the traces below check the unfused kernels. */
    private FusedMode fusedMode = FusedMode.OFF;

    @After
    public void tearDown() {
        closeRig();
    }

    private void closeRig() {
        if (mixer != null) {
            mixer.close();
            mixer = null;
        }
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
        if (memory != null) {
            memory.close();
            memory = null;
        }
        if (backend != null) {
            backend.close();
            backend = null;
        }
    }

    // ================================================================== fixtures

    private static AttentionSpec.Builder llamaAttention() {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .nRot(HEAD_DIM).kqScale(KQ_SCALE).kvOwnerLayer(0);
    }

    private static RopeSpec rope() {
        return RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(10000f).build();
    }

    private static LayerSpec.Builder layerBuilder(MixerKind kind, AttentionSpec attention) {
        return LayerSpec.builder(0, kind, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(attention.qkNorm()).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(attention);
    }

    private static ModelSpec model(Arch arch, LayerSpec l) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0)
                .nCtxTrain(512).nEmbd(WIDTH).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRotFull(HEAD_DIM)
                .normEps(1e-5f).normRmsEps(1e-6f);
        hb.nHead(0, N_HEAD).nHeadKv(0, l.attention().linearReplace() ? 0 : N_HEAD_KV);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private static GpuMixerFactory factory(MixerKind kind) {
        for (GpuMixerFactory f : AttnBlocks.gpuMixers()) {
            if (f.kind() == kind) {
                return f;
            }
        }
        throw new AssertionError("no GPU factory for " + kind);
    }

    private void llamaTensors(LayerSpec l, boolean biases) {
        tensors = new HashMap<>();
        matrix(l, TensorNames.ATTN_Q, WIDTH, Q_DIM);
        matrix(l, TensorNames.ATTN_K, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_V, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_OUT, Q_DIM, WIDTH);
        if (biases) {
            vector(l, TensorNames.bias(TensorNames.ATTN_Q), Q_DIM);
            vector(l, TensorNames.bias(TensorNames.ATTN_K), KV_DIM);
            vector(l, TensorNames.bias(TensorNames.ATTN_V), KV_DIM);
            vector(l, TensorNames.bias(TensorNames.ATTN_OUT), WIDTH);
        }
    }

    private void matrix(LayerSpec l, String base, int cols, int rows) {
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        tensors.put(n, GpuWeight.of(n, GGMLType.F32, new long[]{cols, rows},
                backend.allocateBuffer((long) cols * rows * Float.BYTES, true)));
    }

    private void expertTensor(LayerSpec l, String base, int cols, int rows, int nExpert) {
        String n = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(base));
        tensors.put(n, GpuWeight.of(n, GGMLType.F32, new long[]{cols, rows, nExpert},
                backend.allocateBuffer((long) cols * rows * nExpert * Float.BYTES, true)));
    }

    private void vector(LayerSpec l, String suffix, int n) {
        String name = TensorNames.layerName(l.stack(), l.weightLayer(), suffix);
        tensors.put(name, GpuWeight.of(name, GGMLType.F32, new long[]{n},
                backend.allocateBuffer((long) n * Float.BYTES, true)));
    }

    private ScratchSlot slot(String name) {
        return scratch.slot(name);
    }

    private DeviceStepInputs steps() {
        return (DeviceStepInputs) memory.stepInputs();
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    /** The mixer's own kernels of a trace, in order. */
    private static List<String> trace(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            if (OWN.contains(d.kernelName()) && !d.kernelName().equals(GpuOps.COPY2D)
                    && !d.kernelName().equals(GpuOps.AXPBY)) {
                out.add(d.kernelName());
            }
        }
        return out;
    }

    private static List<Dispatch> named(List<Dispatch> ds, String kernel) {
        List<Dispatch> out = new ArrayList<>();
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                out.add(d);
            }
        }
        return out;
    }

    /**
     * The matvec the 256-row Q and output projections launch: at this device's 256-work-item matvec work-group
     * they meet the two-row thread floor and pair their rows, where the 128-row K and V projections stay below it.
     */
    private static String pairedMatvec() {
        return GpuOps.matVecKernel(GGMLType.F32).replace("MatVec", "MatVec2");
    }

    private void decode(int pos) {
        steps().upload(pos, 1);
        mixer.decode(pos, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH));
    }

    private void prefill(int pos0, int n) {
        steps().upload(pos0, n);
        mixer.prefill(pos0, n, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH));
    }

    /** Creates the llama tensors on a fresh backend, then binds the mixer. */
    private void bindLlama(ModelSpec spec, KvDType kvType, boolean biases) {
        bindLlama(spec, kvType, kvType, biases);
    }

    /** Creates the llama tensors on a fresh backend, then binds the mixer over K and V caches of the given types. */
    private void bindLlama(ModelSpec spec, KvDType typeK, KvDType typeV, boolean biases) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        llamaTensors(spec.layers().get(0), biases);
        bindPrepared(spec, typeK, typeV);
    }

    private FlashAttnTiling decodePlan() {
        return FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, HEAD_DIM, HEAD_DIM, N_CTX, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()));
    }

    private FlashAttnTiling prefillPlan(int keyRows) {
        return FlashAttnTiling.prefill(N_HEAD, N_HEAD_KV, HEAD_DIM, HEAD_DIM, keyRows, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()));
    }

    private static ModelSpec llama(AttentionSpec attention) {
        return model(Arch.LLAMA, layerBuilder(MixerKind.ATTENTION, attention).rope(rope()).build());
    }

    // ================================================================== llama decode / prefill

    @Test
    public void llamaDecodeTrace() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        int pos = 5;
        List<Dispatch> ds = run(() -> decode(pos));
        assertEquals(Arrays.asList(RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, RopeKernels.kvStore(KvDType.F16),
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16)), trace(ds));

        GpuKvLayer kv = memory.kv(0);
        List<Dispatch> ropes = named(ds, RopeKernels.ROPE_EXT);
        for (int i = 0; i < 2; i++) {
            Dispatch r = ropes.get(i);
            assertEquals(1, r.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(1, r.intArg(RopeKernels.ARG_POS_STRIDE));
            assertTrue(r.bufferArgs().contains(steps().ropePos()));
            assertTrue(r.bufferArgs().contains(slot(i == 0 ? ScratchSlots.Q : ScratchSlots.K).buffer()));
        }
        List<Dispatch> stores = named(ds, RopeKernels.kvStore(KvDType.F16));
        for (int i = 0; i < 2; i++) {
            Dispatch s = stores.get(i);
            assertEquals(1, s.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(KV_DIM, s.intArg(1));
            assertSame(slot(i == 0 ? ScratchSlots.K : ScratchSlots.V).buffer(), s.bufferArg(5));
            assertSame(steps().cellIdx(kv.group()), s.bufferArg(6));
            assertSame(i == 0 ? kv.k() : kv.v(), s.bufferArg(7));
        }

        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        FlashAttnTiling plan = decodePlan();
        assertEquals(N_HEAD, fd.intArg(AttnKernels.ARG_FD_N_HEAD));
        assertEquals(N_HEAD_KV, fd.intArg(AttnKernels.ARG_FD_N_HEAD_KV));
        assertEquals(plan.gTile(), fd.intArg(AttnKernels.ARG_FD_G_TILE));
        assertEquals(1, fd.intArg(AttnKernels.ARG_FD_N_SPLIT));
        assertEquals(plan.decodeSplit(pos + 1).length(), fd.intArg(AttnKernels.ARG_FD_SPLIT_LEN));
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_KV_START));
        assertEquals(pos + 1, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(pos, fd.intArg(AttnKernels.ARG_FD_Q_POS));
        assertEquals(KV_DIM, fd.intArg(AttnKernels.ARG_FD_K_ROW_STRIDE));
        assertEquals(HEAD_DIM, fd.intArg(AttnKernels.ARG_FD_K_HEAD_STRIDE));
        assertEquals(KQ_SCALE, fd.floatArg(AttnKernels.ARG_FD_SCALE), 0f);
        assertEquals(0f, fd.floatArg(AttnKernels.ARG_FD_SOFTCAP), 0f);
        assertEquals(AttnKernels.FLAG_CAUSAL | AttnKernels.FLAG_DIRECT, fd.intArg(AttnKernels.ARG_FD_FLAGS));
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_Q_OFF));
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_OUT_OFF));
        assertSame(slot(ScratchSlots.Q).buffer(), fd.bufferArg(AttnKernels.ARG_FD_BUF_Q));
        assertSame(kv.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(kv.v(), fd.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertSame(steps().kvPos(kv.group()), fd.bufferArg(AttnKernels.ARG_FD_BUF_KV_POS));
        assertSame(placeholders.readOnlyFloats(), fd.bufferArg(AttnKernels.ARG_FD_BUF_SINKS));
        assertSame(placeholders.readOnlyFloats(), fd.bufferArg(AttnKernels.ARG_FD_BUF_REL_BIAS));
        assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), fd.bufferArg(AttnKernels.ARG_FD_BUF_OUT));
        assertTrue(Arrays.equals(plan.decodeGlobal(1), fd.global()));
    }

    @Test
    public void llamaPrefillTrace() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        int n = 4;
        List<Dispatch> ds = run(() -> prefill(0, n));
        FlashAttnTiling plan = prefillPlan(N_CTX);
        List<String> expected = new ArrayList<>(Arrays.asList(RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT,
                RopeKernels.kvStore(KvDType.F16), RopeKernels.kvStore(KvDType.F16)));
        if (plan.plan() == FlashAttnTiling.Plan.TILED) {
            expected.add(AttnKernels.flashAttn(KvDType.F16));
        } else {
            for (int t = 0; t < n; t++) {
                expected.add(AttnKernels.flashDecode(KvDType.F16));
            }
        }
        assertEquals(expected, trace(ds));
        for (Dispatch r : named(ds, RopeKernels.ROPE_EXT)) {
            assertEquals(n, r.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(n, r.intArg(RopeKernels.ARG_POS_STRIDE));
        }
        for (Dispatch s : named(ds, RopeKernels.kvStore(KvDType.F16))) {
            assertEquals(n, s.intArg(RopeKernels.ARG_N_TOKENS));
        }
        GpuKvLayer kv = memory.kv(0);
        if (plan.plan() == FlashAttnTiling.Plan.TILED) {
            Dispatch fa = named(ds, AttnKernels.flashAttn(KvDType.F16)).get(0);
            assertEquals(n, fa.intArg(AttnKernels.ARG_FA_N_TOKENS));
            assertEquals(plan.bq(), fa.intArg(AttnKernels.ARG_FA_BQ));
            assertEquals(plan.bk(), fa.intArg(AttnKernels.ARG_FA_BK));
            assertEquals(n, fa.intArg(AttnKernels.ARG_FA_KV_END));
            assertEquals(Q_DIM, fa.intArg(AttnKernels.ARG_FA_Q_ROW_STRIDE));
            assertEquals(Q_DIM, fa.intArg(AttnKernels.ARG_FA_OUT_ROW_STRIDE));
            assertEquals(KV_DIM, fa.intArg(AttnKernels.ARG_FA_K_ROW_STRIDE));
            assertEquals(AttnKernels.FLAG_CAUSAL | AttnKernels.FLAG_DIRECT, fa.intArg(AttnKernels.ARG_FA_FLAGS));
            assertEquals(1, fa.intArg(AttnKernels.ARG_FA_N_SPLIT));
            assertSame(steps().qPos(), fa.bufferArg(AttnKernels.ARG_FA_BUF_Q_POS));
            assertSame(steps().kvPos(kv.group()), fa.bufferArg(AttnKernels.ARG_FA_BUF_KV_POS));
            assertTrue(Arrays.equals(plan.prefillGlobal(n, 1), fa.global()));
        } else {
            List<Dispatch> rows = named(ds, AttnKernels.flashDecode(KvDType.F16));
            for (int t = 0; t < n; t++) {
                Dispatch r = rows.get(t);
                assertEquals(t, r.intArg(AttnKernels.ARG_FD_Q_POS));
                assertEquals(t + 1, r.intArg(AttnKernels.ARG_FD_KV_END));
                assertEquals(t * Q_DIM, r.intArg(AttnKernels.ARG_FD_Q_OFF));
                assertEquals(t * Q_DIM, r.intArg(AttnKernels.ARG_FD_OUT_OFF));
            }
        }
    }

    /**
     * The cache-write-only step launches only the K projection and its norm/RoPE, the store rotation and the store:
     * no Q RoPE, no attention and no output projection, and fewer kernels than the whole prefill.
     */
    @Test
    public void kvOnlyPrefillRunsTheStoreWithoutQueryOrAttention() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        int n = 4;
        List<Dispatch> whole = run(() -> prefill(0, n));
        List<Dispatch> only = run(() -> {
            steps().upload(0, n);
            mixer.prefillKV(0, n, slot(ScratchSlots.NORMED));
        });
        assertTrue("the cache-write-only step launches fewer kernels", only.size() < whole.size());
        assertEquals("one Q and one K RoPE in the whole step", 2,
                named(whole, RopeKernels.ROPE_EXT).size());
        assertEquals("the K RoPE alone in the cache-write-only step", 1,
                named(only, RopeKernels.ROPE_EXT).size());
        assertEquals("both stores run", 2, named(only, RopeKernels.kvStore(KvDType.F16)).size());
        for (Dispatch d : only) {
            String kernel = d.kernelName();
            assertFalse("no flash attention: " + kernel, kernel.equals(AttnKernels.flashAttn(KvDType.F16))
                    || kernel.equals(AttnKernels.flashDecode(KvDType.F16)));
        }
    }

    @Test
    public void decodeReusesPreparedDispatchesAcrossSteps() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        List<Dispatch> first = run(() -> decode(0));
        List<Dispatch> second = run(() -> decode(1));
        assertEquals(trace(first), trace(second));
        Dispatch fd = named(second, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertEquals(2, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(1, fd.intArg(AttnKernels.ARG_FD_Q_POS));
    }

    @Test
    public void longDecodeSplitsThroughCombine() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        int pos = N_CTX - 1;
        FlashAttnTiling.Split split = decodePlan().decodeSplit(N_CTX);
        int nSplit = split.count();
        List<Dispatch> ds = run(() -> decode(pos));
        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertEquals(nSplit, fd.intArg(AttnKernels.ARG_FD_N_SPLIT));
        assertEquals(split.length(), fd.intArg(AttnKernels.ARG_FD_SPLIT_LEN));
        assertEquals(N_CTX, fd.intArg(AttnKernels.ARG_FD_KV_END));
        List<Dispatch> combines = named(ds, AttnKernels.FLASH_COMBINE);
        assertEquals(nSplit == 1 ? 0 : 1, combines.size());
        if (nSplit > 1) {
            Dispatch c = combines.get(0);
            assertEquals(0, fd.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_DIRECT);
            assertEquals(N_HEAD, c.intArg(AttnKernels.ARG_FC_N_ROWS));
            assertEquals(nSplit, c.intArg(AttnKernels.ARG_FC_N_SPLIT));
            assertEquals(0, c.intArg(AttnKernels.ARG_FC_OUT_OFF));
            assertSame(fd.bufferArg(AttnKernels.ARG_FD_BUF_PARTIAL), c.bufferArg(AttnKernels.ARG_FC_BUF_PARTIAL));
        }
    }

    // ================================================================== variants

    @Test
    public void quantizedCacheRotatesBeforeStoreAndAfterAttention() {
        bindLlama(llama(llamaAttention().build()), KvDType.Q8_0, false);
        RotationPlan rot = memory.kv(0).rotation();
        List<Dispatch> ds = run(() -> decode(0));
        List<String> t = trace(ds);
        int before = (rot.rotateK() ? 2 : 0) + (rot.rotateV() ? 1 : 0);
        int after = rot.rotateV() ? 1 : 0;
        assertEquals(before + after, named(ds, HadamardSpec.KERNEL).size());
        int store = t.indexOf(RopeKernels.kvStore(KvDType.Q8_0));
        int flash = t.indexOf(AttnKernels.flashDecode(KvDType.Q8_0));
        assertTrue(store >= 0 && flash > store);
        for (int i = 0; i < t.size(); i++) {
            if (t.get(i).equals(HadamardSpec.KERNEL)) {
                assertTrue(i < store || i > flash);
            }
        }
        if (rot.rotateK()) {
            assertEquals(rot.nRotK(), named(ds, HadamardSpec.KERNEL).get(0).intArg(HadamardSpec.ARG_BLOCK));
        }
    }

    @Test
    public void mixedCacheTypesDecodeThroughTheMixedKernel() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, KvDType.Q8_0, false);
        GpuKvLayer kv = memory.kv(0);
        assertEquals(KvDType.F16, kv.layout().dtypeK());
        assertEquals(KvDType.Q8_0, kv.layout().dtypeV());
        RotationPlan rot = kv.rotation();
        assertFalse(rot.rotateK());
        int pos = 3;
        List<Dispatch> ds = run(() -> decode(pos));
        List<String> t = trace(ds);
        assertFalse(t.contains(AttnKernels.flashDecode(KvDType.F16)));
        assertFalse(t.contains(AttnKernels.flashDecode(KvDType.Q8_0)));
        List<Dispatch> storesK = named(ds, RopeKernels.kvStore(KvDType.F16));
        List<Dispatch> storesV = named(ds, RopeKernels.kvStore(KvDType.Q8_0));
        assertEquals(1, storesK.size());
        assertEquals(1, storesV.size());
        assertTrue(storesK.get(0).bufferArgs().contains(kv.k()));
        assertTrue(storesV.get(0).bufferArgs().contains(kv.v()));
        assertEquals(rot.rotateV() ? 2 : 0, named(ds, HadamardSpec.KERNEL).size());

        List<Dispatch> flash = named(ds, AttnKernels.FLASH_DECODE_MIXED);
        assertEquals(1, flash.size());
        Dispatch fd = flash.get(0);
        assertTrue(t.indexOf(RopeKernels.kvStore(KvDType.Q8_0)) < t.indexOf(AttnKernels.FLASH_DECODE_MIXED));
        assertEquals(AttnKernels.FD_MIXED_ARG_COUNT, fd.argCount());
        assertEquals(AttnKernels.kvTypes(KvDType.F16, KvDType.Q8_0), fd.intArg(AttnKernels.ARG_FD_KV_TYPES));
        assertEquals(pos, fd.intArg(AttnKernels.ARG_FD_Q_POS));
        assertEquals(pos + 1, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(KV_DIM, fd.intArg(AttnKernels.ARG_FD_K_ROW_STRIDE));
        assertEquals(KV_DIM, fd.intArg(AttnKernels.ARG_FD_V_ROW_STRIDE));
        assertSame(kv.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(kv.v(), fd.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertTrue(Arrays.equals(decodePlan().decodeGlobal(1), fd.global()));
    }

    @Test
    public void mixedCacheTypesPrefillThroughTheMixedKernel() {
        bindLlama(llama(llamaAttention().build()), KvDType.Q8_0, KvDType.BF16, false);
        int n = 4;
        List<Dispatch> ds = run(() -> prefill(0, n));
        assertEquals(1, named(ds, RopeKernels.kvStore(KvDType.Q8_0)).size());
        assertEquals(1, named(ds, RopeKernels.kvStore(KvDType.BF16)).size());
        int kvTypes = AttnKernels.kvTypes(KvDType.Q8_0, KvDType.BF16);
        FlashAttnTiling plan = prefillPlan(N_CTX);
        if (plan.plan() == FlashAttnTiling.Plan.TILED) {
            List<Dispatch> fa = named(ds, AttnKernels.FLASH_ATTN_MIXED);
            assertEquals(1, fa.size());
            assertEquals(AttnKernels.FA_MIXED_ARG_COUNT, fa.get(0).argCount());
            assertEquals(kvTypes, fa.get(0).intArg(AttnKernels.ARG_FA_KV_TYPES));
            assertEquals(n, fa.get(0).intArg(AttnKernels.ARG_FA_N_TOKENS));
            assertEquals(n, fa.get(0).intArg(AttnKernels.ARG_FA_KV_END));
            assertTrue(named(ds, AttnKernels.flashAttn(KvDType.Q8_0)).isEmpty());
        } else {
            List<Dispatch> rows = named(ds, AttnKernels.FLASH_DECODE_MIXED);
            assertEquals(n, rows.size());
            for (int t = 0; t < n; t++) {
                assertEquals(kvTypes, rows.get(t).intArg(AttnKernels.ARG_FD_KV_TYPES));
                assertEquals(t + 1, rows.get(t).intArg(AttnKernels.ARG_FD_KV_END));
            }
        }
    }

    @Test
    public void sinksAndSoftcapAreKernelArguments() {
        AttentionSpec a = llamaAttention().sinks(true).softcap(30f).build();
        ModelSpec spec = model(Arch.OPENAI_MOE, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, true);
        vector(l, TensorNames.weight(TensorNames.ATTN_SINKS), N_HEAD);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> decode(2));
        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertTrue((fd.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_SINKS) != 0);
        assertEquals(30f, fd.floatArg(AttnKernels.ARG_FD_SOFTCAP), 0f);
        String sinks = TensorNames.layerName(l.stack(), 0, TensorNames.weight(TensorNames.ATTN_SINKS));
        assertSame(tensors.get(sinks).shard(0), fd.bufferArg(AttnKernels.ARG_FD_BUF_SINKS));
        assertEquals(Arrays.asList(GpuAttentionMixer.BIAS_ADD, GpuAttentionMixer.BIAS_ADD,
                GpuAttentionMixer.BIAS_ADD, RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, RopeKernels.kvStore(KvDType.F16),
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16), GpuAttentionMixer.BIAS_ADD), trace(ds));
    }

    /**
     * One token of a qwen3 layer whose Q and K carry per-head RMS norms before RoPE and whose V carries the
     * weightless RMS norm. The fused path runs the Q norm, rotation and V norm as one multi-task launch and writes
     * the K row straight to the cache with its norm and rotation, where the unfused path launches five kernels.
     */
    @Test
    public void fusedQkNormDecodeLaunchesFewerKernels() {
        String mv = GpuOps.matVecKernel(GGMLType.F32);
        String wide = pairedMatvec();
        assertEquals(Arrays.asList(wide, mv, mv, NormKernels.ROW_NORM, NormKernels.ROW_NORM, NormKernels.ROW_NORM,
                RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, RopeKernels.kvStore(KvDType.F16),
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16), wide),
                qkNormDecode(FusedMode.OFF));
        assertEquals(Arrays.asList(wide, mv, mv, FusedKernels.MULTI_TASK, FusedKernels.NORM_ROPE_STORE,
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16), wide),
                qkNormDecode(FusedMode.ON));
    }

    /**
     * Without a V norm the fused K store leaves the multi-task launch with only the Q body, so the path keeps the
     * single {@code LLM_RmsNormRope} launch instead of wrapping one body in the persistent kernel.
     */
    @Test
    public void fusedQkNormWithoutVNormUsesTheSingleNormRope() {
        fusedMode = FusedMode.ON;
        AttentionSpec a = llamaAttention().qkNorm(QkNormKind.RMS_HEAD).build();
        ModelSpec spec = model(Arch.QWEN3, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), HEAD_DIM);
        vector(l, TensorNames.weight(TensorNames.ATTN_K_NORM), HEAD_DIM);
        bindPrepared(spec, KvDType.F16);
        List<String> t = new ArrayList<>();
        for (Dispatch d : run(() -> decode(2))) {
            t.add(d.kernelName());
        }
        String mv = GpuOps.matVecKernel(GGMLType.F32);
        assertEquals(Arrays.asList(pairedMatvec(), mv, mv, FusedKernels.RMS_NORM_ROPE,
                FusedKernels.NORM_ROPE_STORE, RopeKernels.kvStore(KvDType.F16),
                AttnKernels.flashDecode(KvDType.F16), pairedMatvec()), t);
    }

    /**
     * A layer without a QK norm rotates its K rows straight into the cache ({@code LLM_RopeStore}), so only the V
     * row keeps a separate store; the unfused path launches the K rotation and two stores.
     */
    @Test
    public void fusedRopeStoreSkipsTheSeparateKRotation() {
        fusedMode = FusedMode.ON;
        AttentionSpec a = llamaAttention().build();
        ModelSpec spec = model(Arch.LLAMA, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> decode(1));
        String mv = GpuOps.matVecKernel(GGMLType.F32);
        List<String> t = new ArrayList<>();
        for (Dispatch d : ds) {
            t.add(d.kernelName());
        }
        assertEquals(Arrays.asList(pairedMatvec(), mv, mv, RopeKernels.ROPE_EXT, FusedKernels.ROPE_STORE,
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16), pairedMatvec()), t);
        assertEquals("only the V row keeps a separate store",
                1, named(ds, RopeKernels.kvStore(KvDType.F16)).size());
    }

    /** The kernels of one qwen3 decode step with the given fused-kernel selection. */
    private List<String> qkNormDecode(FusedMode mode) {
        fusedMode = mode;
        AttentionSpec a = llamaAttention().qkNorm(QkNormKind.RMS_HEAD).vNorm(true).build();
        ModelSpec spec = model(Arch.QWEN3, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), HEAD_DIM);
        vector(l, TensorNames.weight(TensorNames.ATTN_K_NORM), HEAD_DIM);
        bindPrepared(spec, KvDType.F16);
        List<String> names = new ArrayList<>();
        for (Dispatch d : run(() -> decode(2))) {
            names.add(d.kernelName());
        }
        closeRig();
        return names;
    }

    @Test
    public void qwen3QkNormRunsPerHeadBeforeRope() {
        AttentionSpec a = llamaAttention().qkNorm(QkNormKind.RMS_HEAD).build();
        ModelSpec spec = model(Arch.QWEN3, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), HEAD_DIM);
        vector(l, TensorNames.weight(TensorNames.ATTN_K_NORM), HEAD_DIM);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> prefill(0, 3));
        List<String> t = trace(ds);
        assertEquals(Arrays.asList(NormKernels.ROW_NORM, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT,
                RopeKernels.ROPE_EXT), t.subList(0, 4));
        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        assertEquals(3 * N_HEAD, norms.get(0).intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(3 * N_HEAD_KV, norms.get(1).intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(3L * N_HEAD * ops.local(), norms.get(0).global()[0]);
    }

    @Test
    public void qkNormAfterRopeRunsOnTheRotatedHeads() {
        AttentionSpec a = llamaAttention().qkNorm(QkNormKind.RMS_HEAD).qkNormAfterRope(true).build();
        ModelSpec spec = model(Arch.HUNYUAN_MOE, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), HEAD_DIM);
        vector(l, TensorNames.weight(TensorNames.ATTN_K_NORM), HEAD_DIM);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> decode(1));
        assertEquals(Arrays.asList(RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, NormKernels.ROW_NORM,
                NormKernels.ROW_NORM, RopeKernels.kvStore(KvDType.F16), RopeKernels.kvStore(KvDType.F16),
                AttnKernels.flashDecode(KvDType.F16)), trace(ds));
        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        assertSame(slot(ScratchSlots.Q).buffer(), norms.get(0).bufferArg(NormKernels.ARG_ROW_NORM_X));
        assertSame(slot(ScratchSlots.K).buffer(), norms.get(1).bufferArg(NormKernels.ARG_ROW_NORM_X));
        assertEquals(N_HEAD_KV, norms.get(1).intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
    }

    @Test
    public void mptLayerNormSpansTheProjectionAndFollowsTheQueryNorm() {
        AttentionSpec a = llamaAttention().noRope(true).nRot(0).qkNorm(QkNormKind.LN_FULL).build();
        ModelSpec spec = model(Arch.MPT, layerBuilder(MixerKind.ATTENTION, a).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), Q_DIM);
        vector(l, TensorNames.bias(TensorNames.ATTN_Q_NORM), Q_DIM);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> prefill(0, 3));
        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        assertEquals(2, norms.size());
        Dispatch q = norms.get(0);
        assertEquals(3, q.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(Q_DIM, q.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(NormKind.LAYER.rowNormMode(), q.intArg(NormKernels.ARG_ROW_NORM_MODE));
        assertEquals(1e-5f, q.floatArg(NormKernels.ARG_ROW_NORM_EPS), 0f);
        assertEquals(NormKernels.FLAG_HAS_W | NormKernels.FLAG_HAS_B, q.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        Dispatch k = norms.get(1);
        assertEquals(3, k.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(KV_DIM, k.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(0, k.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        assertSame(slot(ScratchSlots.K).buffer(), k.bufferArg(NormKernels.ARG_ROW_NORM_X));
    }

    @Test
    public void bertGraphLayerNormRunsOnlyWhereItsWeightExists() {
        AttentionSpec a = llamaAttention().noRope(true).nRot(0).causal(false).usesMemory(false)
                .qkNorm(QkNormKind.LN_FULL).build();
        ModelSpec spec = model(Arch.JINA_BERT_V2, layerBuilder(MixerKind.ATTENTION, a).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        vector(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), Q_DIM);
        vector(l, TensorNames.bias(TensorNames.ATTN_Q_NORM), Q_DIM);
        bindPrepared(spec, null);
        List<Dispatch> ds = run(() -> mixer.prefill(0, 2, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        assertEquals(1, norms.size());
        assertEquals(Q_DIM, norms.get(0).intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertSame(slot(ScratchSlots.Q).buffer(), norms.get(0).bufferArg(NormKernels.ARG_ROW_NORM_X));
    }

    @Test
    public void qkNormBiasIsLeftOutWhenTheGraphDropsIt() {
        AttentionSpec a = llamaAttention().qkNorm(QkNormKind.RMS_HEAD).qkNormBias(false).build();
        ModelSpec spec = model(Arch.APERTUS, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        for (String base : new String[]{TensorNames.ATTN_Q_NORM, TensorNames.ATTN_K_NORM}) {
            vector(l, TensorNames.weight(base), HEAD_DIM);
            vector(l, TensorNames.bias(base), HEAD_DIM);
        }
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> norms = named(run(() -> decode(0)), NormKernels.ROW_NORM);
        assertEquals(2, norms.size());
        for (Dispatch d : norms) {
            assertEquals(NormKernels.FLAG_HAS_W, d.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
            assertSame(placeholders.readOnlyFloats(), d.bufferArg(NormKernels.ARG_ROW_NORM_B));
        }
    }

    @Test
    public void outputBiasIsLeftOutWhenTheGraphPassesNull() {
        bindLlama(llama(llamaAttention().outBias(false).build()), KvDType.F16, true);
        List<Dispatch> ds = run(() -> decode(0));
        assertEquals(Arrays.asList(GpuAttentionMixer.BIAS_ADD, GpuAttentionMixer.BIAS_ADD,
                GpuAttentionMixer.BIAS_ADD, RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, RopeKernels.kvStore(KvDType.F16),
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16)), trace(ds));
    }

    @Test
    public void lagunaPerHeadSoftplusGateReadsOneGatePerHead() {
        AttentionSpec a = llamaAttention().gate(AttnGateKind.PER_HEAD_SOFTPLUS).build();
        ModelSpec spec = model(Arch.LAGUNA, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        matrix(l, TensorNames.ATTN_GATE, WIDTH, N_HEAD);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> prefill(0, 2));
        Dispatch g = named(ds, AttnExtKernels.OUT_GATE).get(0);
        assertEquals(AttnExtMath.GATE_PER_HEAD_SOFTPLUS, g.intArg(AttnExtKernels.ARG_GATE_KIND));
        assertEquals(N_HEAD, g.intArg(AttnExtKernels.ARG_GATE_GATE_STRIDE));
        assertEquals(0, g.intArg(AttnExtKernels.ARG_GATE_HAS_BIAS));
        assertEquals(2, g.intArg(AttnExtKernels.ARG_GATE_N_TOKENS));
    }

    @Test
    public void k2HorizonBase2GateIsElementwise() {
        AttentionSpec a = llamaAttention().gate(AttnGateKind.SOFTPLUS_BASE2).build();
        ModelSpec spec = model(Arch.LLAMA, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        LayerSpec l = spec.layers().get(0);
        llamaTensors(l, false);
        matrix(l, TensorNames.ATTN_GATE, WIDTH, Q_DIM);
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> decode(0));
        List<String> t = trace(ds);
        assertEquals(AttnExtKernels.OUT_GATE, t.get(t.size() - 1));
        Dispatch g = named(ds, AttnExtKernels.OUT_GATE).get(0);
        assertEquals(AttnExtMath.GATE_SOFTPLUS_BASE2, g.intArg(AttnExtKernels.ARG_GATE_KIND));
        assertEquals(Q_DIM, g.intArg(AttnExtKernels.ARG_GATE_GATE_STRIDE));
    }

    private ModelSpec movaModel(String... unavailableKernels) {
        MoeSpec experts = MoeSpec.builder().experts(8, 2, KV_DIM).gating(GatingFunc.SIGMOID).probBias(true)
                .normWeights(true).weightsScale(1f).act(ActOp.SILU).build();
        AttentionSpec a = llamaAttention().valueExperts(experts).build();
        ModelSpec spec = model(Arch.LLAMA, layerBuilder(MixerKind.ATTENTION, a).rope(rope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels(unavailableKernels).build();
        LayerSpec l = spec.layers().get(0);
        tensors = new HashMap<>();
        matrix(l, TensorNames.ATTN_Q, WIDTH, Q_DIM);
        matrix(l, TensorNames.ATTN_K, WIDTH, KV_DIM);
        matrix(l, TensorNames.ATTN_OUT, Q_DIM, WIDTH);
        matrix(l, TensorNames.ATTN_V_GATE, WIDTH, 8);
        vector(l, TensorNames.bias(TensorNames.ATTN_V_GATE), 8);
        expertTensor(l, TensorNames.ATTN_V_EXPS, WIDTH, KV_DIM, 8);
        return spec;
    }

    @Test
    public void movaDecodeRoutesValueExpertsIntoV() {
        ModelSpec spec = movaModel();
        bindPrepared(spec, KvDType.F16);
        List<Dispatch> ds = run(() -> decode(0));
        assertEquals(Arrays.asList(MoeKernels.ROUTER, ActKernels.UNARY, MoeKernels.WEIGHTED_SUM,
                RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, RopeKernels.kvStore(KvDType.F16), RopeKernels.kvStore(KvDType.F16),
                AttnKernels.flashDecode(KvDType.F16)), trace(ds));
        assertEquals(1, named(ds, MoeKernels.matVecId(GGMLType.F32)).size());
        Dispatch router = named(ds, MoeKernels.ROUTER).get(0);
        assertEquals(1, router.intArg(MoeKernels.ARG_N_TOKENS));
        String bias = TensorNames.layerName(spec.layers().get(0).stack(), 0,
                TensorNames.bias(TensorNames.ATTN_V_GATE));
        assertTrue(router.bufferArgs().contains(tensors.get(bias).shard(0)));
        Dispatch wsum = named(ds, MoeKernels.WEIGHTED_SUM).get(0);
        assertTrue(wsum.bufferArgs().contains(slot(ScratchSlots.V).buffer()));
    }

    @Test
    public void movaPrefillBucketsTheTokenExpertPairs() {
        bindPrepared(movaModel(), KvDType.F16);
        int n = 3;
        List<Dispatch> ds = run(() -> prefill(0, n));
        List<String> t = trace(ds);
        assertEquals(Arrays.asList(MoeKernels.ROUTER, MoeKernels.BUCKETIZE, ActKernels.UNARY,
                MoeKernels.WEIGHTED_SUM), t.subList(0, 4));
        assertTrue(named(ds, MoeKernels.batchMatMulId(GGMLType.F32)).isEmpty());
        assertTrue(named(ds, MoeKernels.matVecId(GGMLType.F32)).isEmpty());
        assertEquals(n, named(ds, MoeKernels.ROUTER).get(0).intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(n, named(ds, MoeKernels.WEIGHTED_SUM).get(0).intArg(MoeKernels.ARG_N_TOKENS));
        Dispatch bucket = named(ds, MoeKernels.BUCKETIZE).get(0);
        Dispatch align = named(ds, MoeKernels.ALIGN_BLOCKS).get(0);
        List<Dispatch> gg = named(ds, MoeKernels.groupedGemm(GGMLType.F32));
        assertEquals(1, gg.size());
        Dispatch g = gg.get(0);
        int tiles = MoeKernels.groupedTiles(UBATCH * 2, 8);
        assertEquals(n, align.intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(tiles, g.intArg(MoeKernels.ARG_GG_N_TILES));
        assertEquals(KV_DIM, g.intArg(0));
        assertEquals(WIDTH, g.intArg(1));
        assertEquals("token rows", 2, g.intArg(MoeKernels.ARG_GG_X_DIV));
        assertEquals(WIDTH, g.intArg(MoeKernels.ARG_GG_X_STRIDE));
        assertEquals(MoeKernels.SCALE_NONE, g.intArg(MoeKernels.ARG_GG_HAS_SCALE));
        assertSame(bucket.bufferArg(3), named(ds, MoeKernels.ROUTER).get(0).bufferArg(MoeKernels.ARG_ROUTER_IDS));
        assertSame(align.bufferArg(8), g.bufferArg(MoeKernels.ARG_GG_GEMM_TOKEN));
        assertSame(align.bufferArg(9), g.bufferArg(MoeKernels.ARG_GG_GEMM_SLOT));
        assertTrue(ds.indexOf(bucket) < ds.indexOf(align) && ds.indexOf(align) < ds.indexOf(g));
        assertTrue(ds.indexOf(g) < ds.indexOf(named(ds, ActKernels.UNARY).get(0)));
    }

    /** Without the align kernel the value experts of a prefill keep the per-pair {@code {T}_BatchMatMulId}. */
    @Test
    public void movaPrefillWithoutTheAlignKernelKeepsBatchMatMulId() {
        bindPrepared(movaModel(MoeKernels.ALIGN_BLOCKS), KvDType.F16);
        int n = 3;
        List<Dispatch> ds = run(() -> prefill(0, n));
        assertTrue(named(ds, MoeKernels.groupedGemm(GGMLType.F32)).isEmpty());
        assertEquals(1, named(ds, MoeKernels.batchMatMulId(GGMLType.F32)).size());
        assertEquals(n * 2, named(ds, MoeKernels.batchMatMulId(GGMLType.F32)).get(0)
                .intArg(MoeKernels.ARG_BMM_N_ROWS));
    }

    @Test
    public void bertEncoderAttendsTheBatchWithoutCache() {
        AttentionSpec a = llamaAttention().noRope(true).nRot(0).causal(false).usesMemory(false).build();
        ModelSpec spec = model(Arch.BERT, layerBuilder(MixerKind.ATTENTION, a).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        llamaTensors(spec.layers().get(0), true);
        bindPrepared(spec, null);
        int n = 5;
        List<Dispatch> ds = run(() -> mixer.prefill(0, n, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        List<String> t = trace(ds);
        assertFalse(t.contains(RopeKernels.ROPE_EXT));
        assertFalse(t.contains(RopeKernels.kvStore(KvDType.F32)));
        assertEquals(4, named(ds, GpuAttentionMixer.BIAS_ADD).size());
        FlashAttnTiling plan = prefillPlan(UBATCH);
        if (plan.plan() == FlashAttnTiling.Plan.TILED) {
            Dispatch fa = named(ds, AttnKernels.flashAttn(KvDType.F32)).get(0);
            assertEquals(n, fa.intArg(AttnKernels.ARG_FA_KV_END));
            assertEquals(0, fa.intArg(AttnKernels.ARG_FA_FLAGS) & (AttnKernels.FLAG_CAUSAL | AttnKernels.FLAG_SINKS));
            assertSame(slot(ScratchSlots.K).buffer(), fa.bufferArg(AttnKernels.ARG_FA_BUF_K));
            assertSame(slot(ScratchSlots.POS).buffer(), fa.bufferArg(AttnKernels.ARG_FA_BUF_Q_POS));
            assertSame(slot(ScratchSlots.POS).buffer(), fa.bufferArg(AttnKernels.ARG_FA_BUF_KV_POS));
        } else {
            List<Dispatch> rows = named(ds, AttnKernels.flashDecode(KvDType.F32));
            assertEquals(n, rows.size());
            for (Dispatch r : rows) {
                assertEquals(n, r.intArg(AttnKernels.ARG_FD_KV_END));
                assertEquals(0, r.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_CAUSAL);
            }
        }
        assertThrows(UnsupportedOperationException.class, () -> mixer.shiftContext(0, 1));
    }

    @Test
    public void deciLinearReplaceIsOnlyTheOutputProjection() {
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(0).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(KQ_SCALE).kvOwnerLayer(0).linearReplace(true).build();
        ModelSpec spec = model(Arch.DECI, layerBuilder(MixerKind.ATTENTION_LINEAR_REPLACE, a).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        tensors = new HashMap<>();
        matrix(spec.layers().get(0), TensorNames.ATTN_OUT, WIDTH, WIDTH);
        bindPrepared(spec, null);
        List<Dispatch> ds = run(() -> mixer.prefill(0, 3, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        assertTrue(trace(ds).isEmpty());
        assertFalse(ds.isEmpty());
        mixer.shiftContext(0, 1);
    }

    @Test
    public void contextShiftRotatesTheOwnKCache() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        int n = 8;
        run(() -> prefill(0, n));
        memory.sequence().append(new int[n], 0, n);
        memory.shiftPositions(2, 3);
        List<Dispatch> ds = run(() -> mixer.shiftContext(2, 3));
        List<Dispatch> shifts = named(ds, RopeKernels.kvRopeShift(KvDType.F16));
        assertEquals(1, shifts.size());
        GpuKvLayer kv = memory.kv(0);
        assertTrue(shifts.get(0).bufferArgs().contains(memory.shiftDelta(kv.group())));
        assertTrue(shifts.get(0).bufferArgs().contains(kv.k()));
    }

    @Test
    public void contextShiftKernelIsPreparedAtBind() {
        bindLlama(llama(llamaAttention().build()), KvDType.F16, false);
        String kernel = RopeKernels.kvRopeShift(KvDType.F16);
        List<String> prepared = backend.preparedKernels();
        assertTrue(prepared.toString(), prepared.contains(kernel));
        assertNull(mixer.shiftRefusal());
    }

    @Test
    public void unavailableContextShiftKernelIsAShiftRefusalAtBind() {
        String kernel = RopeKernels.kvRopeShift(KvDType.F16);
        ModelSpec spec = llama(llamaAttention().build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).unavailableKernels(kernel).build();
        llamaTensors(spec.layers().get(0), false);
        bindPrepared(spec, KvDType.F16);

        String refusal = mixer.shiftRefusal();
        assertNotNull(refusal);
        assertTrue(refusal, refusal.contains(kernel));
        assertThrows(UnsupportedOperationException.class, () -> mixer.shiftContext(0, 1));
    }

    @Test
    public void factoriesCoverBothKindsAndRejectOthers() {
        assertEquals(3, AttnBlocks.gpuMixers().size());
        assertEquals(3, AttnBlocks.cpuMixers().size());
        assertTrue(AttnBlocks.gpuFfns().isEmpty());
        assertTrue(AttnBlocks.cpuFfns().isEmpty());
        assertEquals(1, AttnBlocks.gpuExtras().size());
        assertEquals(LayerSpec.Extras.INKLING_SCONV, AttnBlocks.gpuExtras().get(0).featureId());
        assertTrue(AttnBlocks.cpuExtras().isEmpty());
        LayerSpec plain = layerBuilder(MixerKind.ATTENTION, llamaAttention().build()).rope(rope()).build();
        assertThrows(IllegalArgumentException.class,
                () -> factory(MixerKind.ATTENTION_LINEAR_REPLACE).create(plain));
        assertTrue(factory(MixerKind.ATTENTION).create(plain) instanceof GpuAttentionMixer);
    }

    /** Binds a mixer on {@link #backend}, whose tensors the test already created. */
    private void bindPrepared(ModelSpec spec, KvDType kvType) {
        bindPrepared(spec, kvType, kvType);
    }

    /** Binds a mixer on {@link #backend} with a K cache of type {@code typeK} and a V cache of {@code typeV}. */
    private void bindPrepared(ModelSpec spec, KvDType typeK, KvDType typeV) {
        LayerSpec l = spec.layers().get(0);
        if (typeK != null) {
            MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(typeK).typeV(typeV).build();
            MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
            memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        }
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * WIDTH)
                .slot(ScratchSlots.Q, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.K, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.V, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.ATTN_OUT, (long) UBATCH * Q_DIM)
                .slot(ScratchSlots.POS, (long) UBATCH * 4)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = GpuOpsFixture.withFusedMode(backend.computeQueue(), kernels, scratch, placeholders, fusedMode);
        GpuForwardContext.Builder cb = GpuForwardContext.builder(kernels, ops, tensors::get, spec);
        if (memory != null) {
            cb.memory(memory);
        }
        mixer = factory(l.mixerKind()).create(l);
        mixer.bind(cb.build(), l);
    }

    // ================================================================== NTK frequency factors

    /**
     * A layer whose per-pair divisors the resolver computed (vLLM's fixed NTK scaling) rotates with them bound as
     * {@code ff}: the values ride a mixer-owned buffer both rotations share, uploaded once.
     */
    @Test
    public void computedFrequencyFactorsAreUploadedAsTheRopeDivisors() {
        float[] factors = RopeMath.ntkFactors(4f, Float.NaN, HEAD_DIM);
        RopeSpec ntk = RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(10000f)
                .computedFreqFactors(factors).build();
        ModelSpec spec = model(Arch.LLAMA,
                layerBuilder(MixerKind.ATTENTION, llamaAttention().build()).rope(ntk).build());
        bindLlama(spec, KvDType.F16, false);
        List<Dispatch> ds = run(() -> decode(5));
        List<Dispatch> ropes = named(ds, RopeKernels.ROPE_EXT);
        assertEquals(2, ropes.size());
        FakeGpuBuffer ff = (FakeGpuBuffer) ropes.get(0).bufferArg(21);
        assertSame("both rotations share the uploaded divisors", ff, ropes.get(1).bufferArg(21));
        for (Dispatch r : ropes) {
            assertEquals("has_ff", 1, r.intArg(8));
        }
        for (int i = 0; i < factors.length; i++) {
            assertEquals("divisor " + i, factors[i], ff.getFloat(i), 0f);
        }
    }

    // ================================================================== Fourier position embedding

    private static final int FOPE_DIM = 4;

    private static RopeSpec fopeRope() {
        float[] inv = RopeMath.fopeInvFreq(10000f, HEAD_DIM, FOPE_DIM, 512);
        TensorRef cos = TensorRef.of(TensorNames.weight(TensorNames.ROPE_COS_COEF), GGMLType.F32,
                FOPE_DIM, FOPE_DIM, N_HEAD_KV);
        TensorRef sin = TensorRef.of(TensorNames.weight(TensorNames.ROPE_SIN_COEF), GGMLType.F32,
                FOPE_DIM, FOPE_DIM, N_HEAD_KV);
        return RopeSpec.builder(RopeType.NEOX).nDims(HEAD_DIM).freqBase(10000f)
                .fope(FopeSpec.of(N_HEAD_KV, true, inv, cos, sin)).build();
    }

    /** The model-level coefficient matrices, read by every FoPE layer of the model. */
    private void fopeTensors() {
        for (String name : Arrays.asList(TensorNames.weight(TensorNames.ROPE_COS_COEF),
                TensorNames.weight(TensorNames.ROPE_SIN_COEF))) {
            tensors.put(name, GpuWeight.of(name, GGMLType.F32,
                    new long[]{FOPE_DIM, FOPE_DIM, N_HEAD_KV},
                    backend.allocateBuffer((long) FOPE_DIM * FOPE_DIM * N_HEAD_KV * Float.BYTES, true)));
        }
    }

    /**
     * A FoPE layer rotates Q and K with {@link RopeKernels#ROPE_FOPE} in decode and prefill alike: the learned
     * coefficients replace the analytic pairs, so the fused norm+rope and fused cache-store kernels stay off.
     */
    @Test
    public void fopeDecodeAndPrefillRotateWithTheCoefficientKernel() {
        ModelSpec spec = model(Arch.INTERN_S1_PRO,
                layerBuilder(MixerKind.ATTENTION, llamaAttention().build()).rope(fopeRope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        llamaTensors(spec.layers().get(0), false);
        fopeTensors();
        bindPrepared(spec, KvDType.F16);

        List<Dispatch> ds = run(() -> decode(5));
        assertEquals(Arrays.asList(RopeKernels.ROPE_FOPE, RopeKernels.ROPE_FOPE, RopeKernels.kvStore(KvDType.F16),
                RopeKernels.kvStore(KvDType.F16), AttnKernels.flashDecode(KvDType.F16)), trace(ds));
        checkFopeSteps(named(ds, RopeKernels.ROPE_FOPE), 1);

        backend.clearLog();
        int n = 4;
        List<Dispatch> pre = run(() -> prefill(0, n));
        assertEquals(n, named(pre, RopeKernels.ROPE_FOPE).get(0).intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(RopeKernels.ROPE_FOPE, trace(pre).get(0));
        assertEquals(RopeKernels.ROPE_FOPE, trace(pre).get(1));
        checkFopeSteps(named(pre, RopeKernels.ROPE_FOPE), n);
    }

    /** The learned cos/sin do not factor into a rotation of a position delta, so the shift is refused by name. */
    @Test
    public void aFourierLayerRefusesTheContextShiftWithoutBuildingTheShiftKernel() {
        ModelSpec spec = model(Arch.INTERN_S1_PRO,
                layerBuilder(MixerKind.ATTENTION, llamaAttention().build()).rope(fopeRope()).build());
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        llamaTensors(spec.layers().get(0), false);
        fopeTensors();
        bindPrepared(spec, KvDType.F16);

        String refusal = mixer.shiftRefusal();
        assertNotNull(refusal);
        assertTrue(refusal, refusal.contains("Fourier position embedding"));
        assertThrows(UnsupportedOperationException.class, () -> mixer.shiftContext(0, 1));
        assertFalse(backend.preparedKernels().contains(RopeKernels.kvRopeShift(KvDType.F16)));
    }

    private void checkFopeSteps(List<Dispatch> ropes, int nTokens) {
        assertEquals(2, ropes.size());
        int[] heads = {N_HEAD, N_HEAD_KV};
        GpuBuffer cos = tensors.get(TensorNames.weight(TensorNames.ROPE_COS_COEF)).shard(0);
        GpuBuffer sin = tensors.get(TensorNames.weight(TensorNames.ROPE_SIN_COEF)).shard(0);
        for (int i = 0; i < 2; i++) {
            Dispatch r = ropes.get(i);
            assertEquals(nTokens, r.intArg(RopeKernels.ARG_FOPE_N_TOKENS));
            assertEquals(heads[i], r.intArg(RopeKernels.ARG_FOPE_N_HEADS));
            assertEquals(HEAD_DIM, r.intArg(RopeKernels.ARG_FOPE_HEAD_STRIDE));
            assertEquals(i == 0 ? Q_DIM : KV_DIM, r.intArg(RopeKernels.ARG_FOPE_ROW_STRIDE));
            assertEquals(HEAD_DIM, r.intArg(RopeKernels.ARG_FOPE_N_DIMS));
            assertEquals(FOPE_DIM, r.intArg(RopeKernels.ARG_FOPE_N_INV));
            assertEquals(N_HEAD_KV, r.intArg(RopeKernels.ARG_FOPE_N_COEF_ROWS));
            assertEquals(1, r.intArg(RopeKernels.ARG_FOPE_SEP_HEAD));
            assertSame(steps().ropePos(), r.bufferArg(RopeKernels.ARG_FOPE_POS));
            assertSame(cos, r.bufferArg(RopeKernels.ARG_FOPE_COS));
            assertSame(sin, r.bufferArg(RopeKernels.ARG_FOPE_SIN));
            assertNotNull(r.bufferArg(RopeKernels.ARG_FOPE_INV));
            assertSame(slot(i == 0 ? ScratchSlots.Q : ScratchSlots.K).buffer(),
                    r.bufferArg(RopeKernels.ARG_FOPE_X));
        }
    }
}

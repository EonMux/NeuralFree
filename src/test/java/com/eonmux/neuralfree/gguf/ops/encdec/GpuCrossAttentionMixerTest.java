package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
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
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the T5 decoder cross-attention ({@link GpuCrossAttentionMixer}) against the host-only fake
 * backend with strict LLM argument rules: the cross K/V build of {@code encode}, the decode step (bidirectional
 * unbiased attention over the cross cache rows {@code [0, nEnc)}, no self-cache access), the prefill step, and the
 * Hadamard rotations of a quantized cross cache.
 */
public class GpuCrossAttentionMixerTest {

    private static final int WIDTH = 128;
    private static final int N_HEAD = 2;
    private static final int HEAD_DIM = 64;
    private static final int DIM = N_HEAD * HEAD_DIM;
    private static final int BUCKETS = 32;
    private static final int MAX_DISTANCE = 128;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final int N_ENC = 5;

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuCrossAttentionMixer mixer;
    private LayerSpec layer;
    private GpuBuffer encoderOutput;

    @After
    public void tearDown() {
        if (mixer != null) {
            mixer.close();
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
        if (memory != null) {
            memory.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    // ------------------------------------------------------------------ fixtures

    private void open(KvDType kvType) {
        open(kvType, kvType);
    }

    private void open(KvDType typeK, KvDType typeV) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        ModelSpec spec = t5Decoder();
        layer = spec.layers().get(0);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(typeK).typeV(typeV).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        long rows = UBATCH;
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.X, rows * WIDTH)
                .slot(ScratchSlots.NORMED, rows * WIDTH)
                .slot(ScratchSlots.BRANCH, rows * WIDTH)
                .slot(ScratchSlots.Q, rows * DIM)
                .slot(ScratchSlots.K, rows * DIM)
                .slot(ScratchSlots.V, rows * DIM)
                .slot(ScratchSlots.ATTN_OUT, rows * DIM)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        Map<String, GpuWeight> tensors = tensors();
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        mixer = (GpuCrossAttentionMixer) EncDecBlocks.gpuMixers().get(0).create(layer);
        mixer.bind(ctx, layer);
        encoderOutput = backend.allocateBuffer((long) UBATCH * WIDTH * Float.BYTES);
    }

    private static ModelSpec t5Decoder() {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.T5)).layerCounts(1, 0)
                .nCtxTrain(512).headKFull(HEAD_DIM).headVFull(HEAD_DIM).nRelAttnBuckets(BUCKETS);
        hb.nHead(0, N_HEAD).nHeadKv(0, N_HEAD);
        AttentionSpec self = attention().causal(true).biasMode(BiasMode.T5_CAUSAL).t5Buckets(BUCKETS)
                .t5MaxDistance(MAX_DISTANCE).t5BiasLayer(0).build();
        AttentionSpec cross = attention().causal(false).build();
        LayerSpec l = LayerSpec.builder(0, MixerKind.CROSS_ATTENTION_T5, FfnKind.NONE)
                .stack(TensorNames.STACK_DEC)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(self)
                .crossAttention(cross)
                .build();
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.T5.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, WIDTH).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private static AttentionSpec.Builder attention() {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(1.0f).kvOwnerLayer(0);
    }

    private Map<String, GpuWeight> tensors() {
        Map<String, GpuWeight> m = new HashMap<>();
        matrix(m, TensorNames.DEC_CROSS_ATTN_Q, WIDTH, DIM);
        matrix(m, TensorNames.DEC_CROSS_ATTN_K, WIDTH, DIM);
        matrix(m, TensorNames.DEC_CROSS_ATTN_V, WIDTH, DIM);
        matrix(m, TensorNames.DEC_CROSS_ATTN_OUT, DIM, WIDTH);
        return m;
    }

    private void matrix(Map<String, GpuWeight> m, String base, int cols, int rows) {
        String n = name(base);
        m.put(n, GpuWeight.of(n, GGMLType.F32, new long[]{cols, rows},
                backend.allocateBuffer((long) cols * rows * Float.BYTES, true)));
    }

    private static String name(String base) {
        return TensorNames.layerName(TensorNames.STACK_DEC, 0, TensorNames.weight(base));
    }

    private ScratchSlot slot(String name) {
        return scratch.slot(name);
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
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

    private static int indexOf(List<Dispatch> ds, Dispatch d) {
        for (int i = 0; i < ds.size(); i++) {
            if (ds.get(i) == d) {
                return i;
            }
        }
        throw new AssertionError(d.kernelName() + " is not in the trace");
    }

    private DeviceStepInputs steps() {
        return (DeviceStepInputs) memory.stepInputs();
    }

    // ------------------------------------------------------------------ encode

    @Test
    public void encodeStoresCrossKvOnceAtIdentityRows() {
        open(KvDType.F16);
        List<Dispatch> ds = run(() -> mixer.encode(encoderOutput, 0, N_ENC));
        List<Dispatch> stores = named(ds, RopeKernels.kvStore(KvDType.F16));
        assertEquals(2, stores.size());
        GpuKvLayer crossKv = memory.crossKv(0);
        for (int i = 0; i < 2; i++) {
            Dispatch s = stores.get(i);
            assertEquals(N_ENC, s.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(DIM, s.intArg(1));
            assertSame(slot(i == 0 ? ScratchSlots.K : ScratchSlots.V).buffer(), s.bufferArg(5));
            assertSame(steps().identityRows(), s.bufferArg(6));
            assertSame(i == 0 ? crossKv.k() : crossKv.v(), s.bufferArg(7));
            assertEquals(RopeKernels.roundUp((long) N_ENC * DIM, ops.local()), s.global()[0]);
        }
        assertTrue(named(ds, AttnKernels.flashDecode(KvDType.F16)).isEmpty());
        assertEquals(N_ENC, mixer.encodedTokens());
    }

    @Test
    public void encodeRejectsMoreTokensThanOneUbatch() {
        open(KvDType.F16);
        assertThrows(IllegalArgumentException.class, () -> mixer.encode(encoderOutput, 0, UBATCH + 1));
        assertThrows(IllegalArgumentException.class, () -> mixer.encode(slot(ScratchSlots.K).buffer(), 0, 1));
    }

    @Test
    public void stepBeforeEncodeFails() {
        open(KvDType.F16);
        steps().upload(0, 1);
        assertThrows(IllegalStateException.class,
                () -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
    }

    @Test
    public void stepRejectsAliasedViews() {
        open(KvDType.F16);
        mixer.encode(encoderOutput, 0, N_ENC);
        steps().upload(0, 1);
        assertThrows(IllegalArgumentException.class,
                () -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.NORMED)));
        assertThrows(IllegalArgumentException.class,
                () -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.ATTN_OUT)));
        assertThrows(IllegalArgumentException.class,
                () -> mixer.decode(0, slot(ScratchSlots.Q), slot(ScratchSlots.BRANCH)));
    }

    // ------------------------------------------------------------------ decode

    @Test
    public void decodeAttendsTheCrossCacheOnly() {
        open(KvDType.F16);
        mixer.encode(encoderOutput, 0, N_ENC);
        int pos = 3;
        steps().upload(pos, 1);
        List<Dispatch> ds = run(() -> mixer.decode(pos, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        assertTrue(named(ds, RopeKernels.kvStore(KvDType.F16)).isEmpty());
        assertTrue(named(ds, HadamardSpec.KERNEL).isEmpty());
        List<Dispatch> flash = named(ds, AttnKernels.flashDecode(KvDType.F16));
        assertEquals(1, flash.size());
        FlashAttnTiling plan = FlashAttnTiling.decode(N_HEAD, N_HEAD, HEAD_DIM, HEAD_DIM, N_CTX,
                kernels.faSmemFloats(), Math.max(1, backend.capabilities().getComputeUnits()));
        int nSplit = plan.decodeSplit(N_ENC).count();
        assertEquals(nSplit == 1 ? 0 : 1, named(ds, AttnKernels.FLASH_COMBINE).size());

        Dispatch cross = flash.get(0);
        GpuKvLayer crossKv = memory.crossKv(0);
        assertEquals(N_HEAD, cross.intArg(AttnKernels.ARG_FD_N_HEAD));
        assertEquals(nSplit, cross.intArg(AttnKernels.ARG_FD_N_SPLIT));
        assertEquals(0, cross.intArg(AttnKernels.ARG_FD_KV_START));
        assertEquals(N_ENC, cross.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(pos, cross.intArg(AttnKernels.ARG_FD_Q_POS));
        assertEquals(DIM, cross.intArg(AttnKernels.ARG_FD_K_ROW_STRIDE));
        assertEquals(HEAD_DIM, cross.intArg(AttnKernels.ARG_FD_K_HEAD_STRIDE));
        assertEquals(1.0f, cross.floatArg(AttnKernels.ARG_FD_SCALE), 0.0f);
        assertEquals(0, cross.intArg(AttnKernels.ARG_FD_CAUSAL));
        assertEquals(BiasMode.NONE.code(), cross.intArg(AttnKernels.ARG_FD_BIAS_MODE));
        assertEquals(0, cross.intArg(AttnKernels.ARG_FD_HAS_SINKS));
        assertEquals(nSplit == 1 ? 1 : 0, cross.intArg(AttnKernels.ARG_FD_DIRECT_OUT));
        assertSame(slot(ScratchSlots.Q).buffer(), cross.bufferArg(AttnKernels.ARG_FD_BUF_Q));
        assertSame(crossKv.k(), cross.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(crossKv.v(), cross.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertSame(steps().identityRows(), cross.bufferArg(AttnKernels.ARG_FD_BUF_KV_POS));
        assertSame(placeholders.readOnlyFloats(), cross.bufferArg(AttnKernels.ARG_FD_BUF_REL_BIAS));
        assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), cross.bufferArg(AttnKernels.ARG_FD_BUF_OUT));

        int at = indexOf(ds, cross);
        assertTrue(at > 0);
        assertTrue(ds.get(0).bufferArgs().contains(slot(ScratchSlots.NORMED).buffer()));
        assertTrue(ds.get(ds.size() - 1).bufferArgs().contains(slot(ScratchSlots.BRANCH).buffer()));
    }

    // ------------------------------------------------------------------ prefill

    @Test
    public void prefillAttendsEveryEncoderRowForEachQuery() {
        open(KvDType.F16);
        mixer.encode(encoderOutput, 0, N_ENC);
        int n = 4;
        steps().upload(0, n);
        FlashAttnTiling plan = FlashAttnTiling.prefill(N_HEAD, N_HEAD, HEAD_DIM, HEAD_DIM, N_CTX,
                kernels.faSmemFloats(), Math.max(1, backend.capabilities().getComputeUnits()));
        List<Dispatch> ds = run(() -> mixer.prefill(0, n, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        assertTrue(named(ds, RopeKernels.kvStore(KvDType.F16)).isEmpty());
        if (plan.plan() == FlashAttnTiling.Plan.TILED) {
            List<Dispatch> tiled = named(ds, AttnKernels.flashAttn(KvDType.F16));
            assertEquals(1, tiled.size());
            Dispatch cross = tiled.get(0);
            assertEquals(n, cross.intArg(AttnKernels.ARG_FA_N_TOKENS));
            assertEquals(0, cross.intArg(AttnKernels.ARG_FA_KV_START));
            assertEquals(N_ENC, cross.intArg(AttnKernels.ARG_FA_KV_END));
            assertEquals(DIM, cross.intArg(AttnKernels.ARG_FA_Q_ROW_STRIDE));
            assertEquals(DIM, cross.intArg(AttnKernels.ARG_FA_OUT_ROW_STRIDE));
            assertEquals(0, cross.intArg(AttnKernels.ARG_FA_CAUSAL));
            assertEquals(BiasMode.NONE.code(), cross.intArg(AttnKernels.ARG_FA_BIAS_MODE));
            assertSame(steps().qPos(), cross.bufferArg(AttnKernels.ARG_FA_BUF_Q_POS));
            assertSame(steps().identityRows(), cross.bufferArg(AttnKernels.ARG_FA_BUF_KV_POS));
            assertSame(placeholders.readOnlyFloats(), cross.bufferArg(AttnKernels.ARG_FA_BUF_REL_BIAS));
        } else {
            List<Dispatch> rows = named(ds, AttnKernels.flashDecode(KvDType.F16));
            assertEquals(n, rows.size());
            for (int t = 0; t < n; t++) {
                Dispatch row = rows.get(t);
                assertEquals(t, row.intArg(AttnKernels.ARG_FD_Q_POS));
                assertEquals(N_ENC, row.intArg(AttnKernels.ARG_FD_KV_END));
                assertEquals(t * DIM, row.intArg(AttnKernels.ARG_FD_Q_OFF));
                assertEquals(t * DIM, row.intArg(AttnKernels.ARG_FD_OUT_OFF));
            }
        }
    }

    // ------------------------------------------------------------------ rotated cache

    @Test
    public void quantizedCacheRotatesKeysValuesQueriesAndOutputs() {
        open(KvDType.Q8_0);
        List<Dispatch> enc = run(() -> mixer.encode(encoderOutput, 0, N_ENC));
        assertEquals(2, named(enc, HadamardSpec.KERNEL).size());
        assertEquals(2, named(enc, RopeKernels.kvStore(KvDType.Q8_0)).size());
        steps().upload(0, 1);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        List<Dispatch> rotations = named(ds, HadamardSpec.KERNEL);
        assertEquals(2, rotations.size());
        assertEquals(1, named(ds, AttnKernels.flashDecode(KvDType.Q8_0)).size());
        for (Dispatch h : rotations) {
            assertEquals(64, h.intArg(HadamardSpec.ARG_BLOCK));
        }
        Dispatch flash = named(ds, AttnKernels.flashDecode(KvDType.Q8_0)).get(0);
        assertTrue(indexOf(ds, rotations.get(0)) < indexOf(ds, flash));
        assertTrue(indexOf(ds, rotations.get(1)) > indexOf(ds, flash));
    }

    // ------------------------------------------------------------------ mixed K/V cache types

    @Test
    public void mixedCacheTypesStoreEachTypeAndUseTheMixedFlashKernel() {
        open(KvDType.F16, KvDType.Q8_0);
        GpuKvLayer cross = memory.crossKv(0);
        assertEquals(KvDType.F16, cross.layout().dtypeK());
        assertEquals(KvDType.Q8_0, cross.layout().dtypeV());
        List<Dispatch> enc = run(() -> mixer.encode(encoderOutput, 0, N_ENC));
        List<Dispatch> storesK = named(enc, RopeKernels.kvStore(KvDType.F16));
        List<Dispatch> storesV = named(enc, RopeKernels.kvStore(KvDType.Q8_0));
        assertEquals(1, storesK.size());
        assertEquals(1, storesV.size());
        assertTrue(storesK.get(0).bufferArgs().contains(cross.k()));
        assertTrue(storesV.get(0).bufferArgs().contains(cross.v()));
        assertEquals(cross.rotation().rotateV() ? 1 : 0, named(enc, HadamardSpec.KERNEL).size());

        steps().upload(0, 1);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        assertTrue(named(ds, AttnKernels.flashDecode(KvDType.F16)).isEmpty());
        assertTrue(named(ds, AttnKernels.flashDecode(KvDType.Q8_0)).isEmpty());
        List<Dispatch> flash = named(ds, AttnKernels.FLASH_DECODE_MIXED);
        assertEquals(1, flash.size());
        Dispatch fd = flash.get(0);
        assertEquals(AttnKernels.FD_MIXED_ARG_COUNT, fd.argCount());
        assertEquals(AttnKernels.kvTypes(KvDType.F16, KvDType.Q8_0), fd.intArg(AttnKernels.ARG_FD_KV_TYPES));
        assertSame(cross.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(cross.v(), fd.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertEquals(N_ENC, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_CAUSAL));
        assertEquals(cross.rotation().rotateV() ? 1 : 0, named(ds, HadamardSpec.KERNEL).size());
    }

    @Test
    public void factoryRejectsOtherMixerKinds() {
        LayerSpec attentionOnly = LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, WIDTH).build())
                .attention(attention().build())
                .build();
        assertThrows(IllegalArgumentException.class, () -> EncDecBlocks.gpuMixers().get(0).create(attentionOnly));
        assertEquals(MixerKind.CROSS_ATTENTION_T5, EncDecBlocks.gpuMixers().get(0).kind());
        assertEquals(MixerKind.CROSS_ATTENTION_T5, EncDecBlocks.cpuMixers().get(0).kind());
        assertTrue(EncDecBlocks.gpuFfns().isEmpty());
        assertTrue(EncDecBlocks.cpuExtras().isEmpty());
    }
}

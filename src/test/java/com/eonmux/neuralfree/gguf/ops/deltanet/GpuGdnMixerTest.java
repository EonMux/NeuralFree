package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.arch.KdaGateMode;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.ssm.SsmKernels;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import org.junit.After;
import org.junit.Test;

import java.lang.reflect.Proxy;
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
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the delta-rule family against the host-only fake backend with strict LLM argument rules:
 * the qwen35 gated DeltaNet (TILED heads, separate beta/alpha) for one token and a batch, the qwen3next legacy
 * {@code ssm_in} / fused {@code ssm_ba} variant (GROUPED heads), the memory-less path, a kimi-linear KDA layer and a
 * MiniMax-01 lightning layer. Each trace checks the kernel sequence and the kernel arguments with buffer identities.
 */
public class GpuGdnMixerTest {

    private static final int N_EMBD = 64;
    private static final int UBATCH = 8;
    private static final int SLOT_WIDTH = 256;
    private static final int D_CONV = 4;

    private static final String CONV = SsmKernels.SSM_CONV;
    private static final String GATE = DeltaNetKernels.GDN_GATE_PREP;
    private static final String SCAN = DeltaNetKernels.GDN_SCAN;
    private static final String NORM = GpuOps.ROW_NORM;
    private static final String GLU = GpuOps.GLU;
    private static final String MV = "Float_MatVec";

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private LlmKernels kernels;
    private GpuMixer mixer;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private GpuBuffer convState;
    private GpuBuffer scanState;
    private GpuBuffer stateRows;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        GpuScratch.Builder sb = GpuScratch.builder(backend, UBATCH);
        for (String s : Arrays.asList(ScratchSlots.NORMED, ScratchSlots.BRANCH)) {
            sb.slot(s, (long) UBATCH * N_EMBD);
        }
        for (String s : Arrays.asList(ScratchSlots.Q, ScratchSlots.K, ScratchSlots.V, ScratchSlots.ATTN_OUT)) {
            sb.slot(s, (long) UBATCH * SLOT_WIDTH);
        }
        scratch = sb.build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
    }

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
        if (backend != null) {
            backend.close();
        }
    }

    // ------------------------------------------------------------------ fixtures

    private void tensor(String name, long... ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        tensors.put(name, GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true)));
    }

    private GpuBuffer buf(String name) {
        return tensors.get(name).shard(0);
    }

    private static ModelSpec model(Arch arch, LayerSpec layer) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f);
        hb.isRecurrent(0, true);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }

    private static LayerSpec layer(MixerKind kind, RecurrentSpec r) {
        return LayerSpec.builder(0, kind, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .recurrent(r)
                .build();
    }

    /** A one-sequence device memory whose layer 0 holds the given state tensors. */
    private GpuMemory memory(LayerSpec layer) {
        List<LayerMemoryPlan.StateTensor> plan = LayerMemoryPlan.recurrentState(layer.recurrent());
        Map<String, GpuBuffer> buffers = new HashMap<>();
        for (LayerMemoryPlan.StateTensor t : plan) {
            buffers.put(t.name(), backend.allocateBuffer(t.bytes()));
        }
        convState = buffers.get(LayerMemoryPlan.STATE_CONV);
        scanState = buffers.get(LayerMemoryPlan.STATE_SCAN);
        stateRows = backend.allocateBuffer(4L * Integer.BYTES);
        GpuRecurrentState state = proxy(GpuRecurrentState.class, (name, args) -> {
            switch (name) {
                case "layer": return 0;
                case "tensors": return plan;
                case "tensor": return buffers.get((String) args[0]);
                default: return null;
            }
        });
        GpuStepInputs steps = proxy(GpuStepInputs.class, (name, args) -> "stateRows".equals(name) ? stateRows : null);
        return proxy(GpuMemory.class, (name, args) -> {
            switch (name) {
                case "recurrent": return state;
                case "stepInputs": return steps;
                default: throw new UnsupportedOperationException(name);
            }
        });
    }

    private interface Handler {
        Object call(String method, Object[] args);
    }

    private static <T> T proxy(Class<T> type, Handler h) {
        return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[]{type},
                (p, m, a) -> {
                    if (m.getDeclaringClass() == Object.class) {
                        return "toString".equals(m.getName()) ? type.getSimpleName() : m.invoke(h, a);
                    }
                    return h.call(m.getName(), a);
                }));
    }

    private GpuForwardContext context(ModelSpec spec, GpuMemory memory) {
        GpuWeights ws = tensors::get;
        GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, ws, spec);
        if (memory != null) {
            b.memory(memory);
        }
        return b.build();
    }

    private GpuMixer bind(MixerKind kind, Arch arch, LayerSpec layer, boolean withMemory) {
        GpuMixerFactory f = null;
        for (GpuMixerFactory c : DeltaNetBlocks.gpuMixers()) {
            if (c.kind() == kind) {
                f = c;
            }
        }
        mixer = f.create(layer);
        mixer.bind(context(model(arch, layer), withMemory ? memory(layer) : null), layer);
        return mixer;
    }

    private ScratchSlot slot(String name) {
        return scratch.slot(name);
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(Dispatch d, int from, int to) {
        int[] v = new int[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.intArg(i);
        }
        return v;
    }

    private long roundUp(long n) {
        int l = ops.local();
        return (n + l - 1) / l * l;
    }

    private int recurrenceLocal() {
        return DeltaNetKernels.recurrenceLocal(kernels.preferredMatvecLocal());
    }

    // ------------------------------------------------------------------ qwen35 (TILED, separate beta/alpha)

    private static final int NK = 2;
    private static final int NV = 4;
    private static final int HK = 16;
    private static final int HV = 16;
    private static final int KEY = NK * HK;
    private static final int VAL = NV * HV;
    private static final int CONV_CH = 2 * KEY + VAL;

    private LayerSpec qwen35Layer() {
        tensor("blk.0.attn_qkv.weight", N_EMBD, CONV_CH);
        tensor("blk.0.attn_gate.weight", N_EMBD, VAL);
        tensor("blk.0.ssm_alpha.weight", N_EMBD, NV);
        tensor("blk.0.ssm_beta.weight", N_EMBD, NV);
        tensor("blk.0.ssm_conv1d.weight", D_CONV, CONV_CH);
        tensor("blk.0.ssm_dt.bias", NV);
        tensor("blk.0.ssm_a", NV);
        tensor("blk.0.ssm_norm.weight", HV);
        tensor("blk.0.ssm_out.weight", VAL, N_EMBD);
        return layer(MixerKind.GDN, RecurrentSpec.builder(MixerKind.GDN).gdnHeads(NK, NV, HK, HV).dConv(D_CONV)
                .headMap(GdnHeadMap.TILED).build());
    }

    @Test
    public void qwen35DecodeTrace() {
        open();
        LayerSpec layer = qwen35Layer();
        bind(MixerKind.GDN, Arch.QWEN35, layer, true);
        ScratchSlot in = slot(ScratchSlots.NORMED);
        ScratchSlot out = slot(ScratchSlots.BRANCH);
        List<Dispatch> ds = run(() -> mixer.decode(7, in, out));

        assertEquals(Arrays.asList(MV, MV, MV, MV, CONV, NORM, GATE, SCAN, NORM, GLU, MV), names(ds));
        assertSame(buf("blk.0.attn_qkv.weight"), ds.get(0).bufferArg(2));
        assertSame(slot(ScratchSlots.Q).buffer(), ds.get(0).bufferArg(4));
        assertSame(slot(ScratchSlots.V).buffer(), ds.get(1).bufferArg(4));

        Dispatch conv = ds.get(4);
        assertArrayEquals(new int[]{CONV_CH, D_CONV, 1, SsmKernels.CONV_FLAG_SILU, CONV_CH, 0, CONV_CH, 0, 0,
                (D_CONV - 1) * CONV_CH}, ints(conv, 0, 10));
        assertSame(slot(ScratchSlots.Q).buffer(), conv.bufferArg(10));
        assertSame(buf("blk.0.ssm_conv1d.weight"), conv.bufferArg(11));
        assertSame(placeholders.readOnlyFloats(), conv.bufferArg(12));
        assertSame(convState, conv.bufferArg(13));
        assertSame(stateRows, conv.bufferArg(14));
        assertSame(slot(ScratchSlots.K).buffer(), conv.bufferArg(15));
        assertArrayEquals(new long[]{roundUp(CONV_CH)}, conv.global());

        Dispatch l2 = ds.get(5);
        assertArrayEquals(new long[]{2L * NK * ops.local()}, l2.global());
        assertTrue(l2.bufferArgs().contains(slot(ScratchSlots.K).buffer()));

        Dispatch gate = ds.get(6);
        assertArrayEquals(new int[]{1, NV, NK, DeltaNetKernels.LAYOUT_SEPARATE, -1, HV}, ints(gate, 0, 6));
        assertEquals(0.0f, gate.floatArg(DeltaNetKernels.GATE_ARG_LB), 0.0f);
        assertArrayEquals(new int[]{NV, 0, NV, 0}, ints(gate, 7, 11));
        assertSame(buf("blk.0.ssm_dt.bias"), gate.bufferArg(DeltaNetKernels.GATE_ARG_DT_BIAS));
        assertSame(buf("blk.0.ssm_a"), gate.bufferArg(DeltaNetKernels.GATE_ARG_SSM_A));
        assertNotSame(gate.bufferArg(DeltaNetKernels.GATE_ARG_A_IN), gate.bufferArg(DeltaNetKernels.GATE_ARG_B_IN));
        assertArrayEquals(new long[]{roundUp(NV)}, gate.global());

        Dispatch scan = ds.get(7);
        assertArrayEquals(new int[]{HK, HV, NV, NK, 1, GdnHeadMap.TILED.code(), DeltaNetKernels.GATE_MODE_PER_HEAD,
                CONV_CH, 0, CONV_CH, KEY, CONV_CH, 2 * KEY, VAL, 0}, ints(scan, 0, 15));
        assertEquals(0.25f, scan.floatArg(DeltaNetKernels.SCAN_ARG_SCALE), 0.0f);
        for (int i = DeltaNetKernels.SCAN_ARG_Q; i <= DeltaNetKernels.SCAN_ARG_V; i++) {
            assertSame(slot(ScratchSlots.K).buffer(), scan.bufferArg(i));
        }
        assertSame(gate.bufferArg(DeltaNetKernels.GATE_ARG_G_OUT), scan.bufferArg(DeltaNetKernels.SCAN_ARG_G));
        assertSame(gate.bufferArg(DeltaNetKernels.GATE_ARG_BETA_OUT), scan.bufferArg(DeltaNetKernels.SCAN_ARG_BETA));
        assertSame(scanState, scan.bufferArg(DeltaNetKernels.SCAN_ARG_STATE));
        assertSame(stateRows, scan.bufferArg(DeltaNetKernels.SCAN_ARG_STATE_ROWS));
        assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_OUT));
        int local = recurrenceLocal();
        assertArrayEquals(new long[]{DeltaNetKernels.recurrenceGlobal(NV, HV, local)}, scan.global());
        assertArrayEquals(new long[]{local}, scan.local());

        Dispatch rms = ds.get(8);
        assertArrayEquals(new long[]{(long) NV * ops.local()}, rms.global());
        assertTrue(rms.bufferArgs().contains(buf("blk.0.ssm_norm.weight")));
        Dispatch glu = ds.get(9);
        assertTrue(glu.bufferArgs().contains(slot(ScratchSlots.V).buffer()));
        assertTrue(glu.bufferArgs().contains(slot(ScratchSlots.ATTN_OUT).buffer()));
        assertSame(buf("blk.0.ssm_out.weight"), ds.get(10).bufferArg(2));
        assertSame(out.buffer(), ds.get(10).bufferArg(4));
    }

    @Test
    public void qwen35PrefillCompactsQkAndReusesPreparedDispatches() {
        open();
        LayerSpec layer = qwen35Layer();
        bind(MixerKind.GDN, Arch.QWEN35, layer, true);
        ScratchSlot in = slot(ScratchSlots.NORMED);
        ScratchSlot out = slot(ScratchSlots.BRANCH);
        int n = 5;
        List<Dispatch> ds = run(() -> mixer.prefill(0, n, in, out));

        List<String> kernelsRun = new ArrayList<>();
        List<Dispatch> core = new ArrayList<>();
        for (Dispatch d : ds) {
            if (!d.kernelName().contains("Float")) {
                kernelsRun.add(d.kernelName());
                core.add(d);
            }
        }
        assertEquals(Arrays.asList(CONV, GpuOps.COPY2D, NORM, GATE, SCAN, NORM, GLU), kernelsRun);
        assertEquals(n, core.get(0).intArg(SsmKernels.ARG_CONV_N_TOKENS));
        assertArrayEquals(new int[]{n, 2 * KEY, CONV_CH, 0, 2 * KEY}, ints(core.get(1), 0, 5));
        assertSame(slot(ScratchSlots.K).buffer(), core.get(1).bufferArg(6));
        GpuBuffer qk = core.get(1).bufferArg(7);
        assertArrayEquals(new long[]{(long) n * 2 * NK * ops.local()}, core.get(2).global());
        assertTrue(core.get(2).bufferArgs().contains(qk));

        Dispatch gate = core.get(3);
        assertEquals(n, gate.intArg(DeltaNetKernels.GATE_ARG_N_TOKENS));
        assertArrayEquals(new long[]{roundUp((long) n * NV)}, gate.global());

        Dispatch scan = core.get(4);
        assertEquals(n, scan.intArg(DeltaNetKernels.SCAN_ARG_N_TOKENS));
        assertArrayEquals(new int[]{2 * KEY, 0, 2 * KEY, KEY, CONV_CH, 2 * KEY, VAL, 0}, ints(scan, 7, 15));
        assertSame(qk, scan.bufferArg(DeltaNetKernels.SCAN_ARG_Q));
        assertSame(qk, scan.bufferArg(DeltaNetKernels.SCAN_ARG_K));
        assertSame(slot(ScratchSlots.K).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_V));
        assertArrayEquals(new long[]{(long) n * NV * ops.local()}, core.get(5).global());

        assertTrue(core.get(0).isPreBound() && gate.isPreBound() && scan.isPreBound());

        List<Dispatch> again = run(() -> mixer.decode(n, in, out));
        assertEquals(1, again.get(4).intArg(SsmKernels.ARG_CONV_N_TOKENS));
        assertEquals(1, again.get(6).intArg(DeltaNetKernels.GATE_ARG_N_TOKENS));
        assertArrayEquals(new long[]{roundUp(NV)}, again.get(6).global());
    }

    // ------------------------------------------------------------------ qwen3next (GROUPED, ssm_in, ssm_ba)

    @Test
    public void qwen3nextLegacySsmInAndFusedBetaAlphaDecodeTrace() {
        open();
        int r = NV / NK;
        int perKh = 2 * HK + 2 * r * HV;
        tensor("blk.0.ssm_in.weight", N_EMBD, (long) NK * perKh);
        tensor("blk.0.ssm_ba.weight", N_EMBD, 2L * NV);
        tensor("blk.0.ssm_conv1d.weight", D_CONV, CONV_CH);
        tensor("blk.0.ssm_dt.bias", NV);
        tensor("blk.0.ssm_a", NV);
        tensor("blk.0.ssm_norm.weight", HV);
        tensor("blk.0.ssm_out.weight", VAL, N_EMBD);
        LayerSpec layer = layer(MixerKind.GDN, RecurrentSpec.builder(MixerKind.GDN).gdnHeads(NK, NV, HK, HV)
                .dConv(D_CONV).headMap(GdnHeadMap.GROUPED).legacySsmIn(true).fusedBetaAlpha(true).build());
        bind(MixerKind.GDN, Arch.QWEN3NEXT, layer, true);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        String cp = GpuOps.COPY2D;
        assertEquals(Arrays.asList(MV, cp, cp, cp, cp, MV, CONV, NORM, GATE, SCAN, NORM, GLU, MV), names(ds));
        assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), ds.get(0).bufferArg(4));
        int vw = r * HV;
        assertArrayEquals(new int[]{NK, HK, perKh, 0, HK, 0}, ints(ds.get(1), 0, 6));
        assertArrayEquals(new int[]{NK, HK, perKh, HK, HK, KEY}, ints(ds.get(2), 0, 6));
        assertArrayEquals(new int[]{NK, vw, perKh, 2 * HK, vw, 2 * KEY}, ints(ds.get(3), 0, 6));
        assertArrayEquals(new int[]{NK, vw, perKh, 2 * HK + vw, vw, 0}, ints(ds.get(4), 0, 6));
        assertSame(slot(ScratchSlots.Q).buffer(), ds.get(3).bufferArg(7));
        assertSame(slot(ScratchSlots.V).buffer(), ds.get(4).bufferArg(7));

        Dispatch gate = ds.get(8);
        assertArrayEquals(new int[]{1, NV, NK, DeltaNetKernels.LAYOUT_FUSED_BA, -1}, ints(gate, 0, 5));
        assertArrayEquals(new int[]{2 * NV, 0, 2 * NV, 0}, ints(gate, 7, 11));
        assertSame(gate.bufferArg(DeltaNetKernels.GATE_ARG_A_IN), gate.bufferArg(DeltaNetKernels.GATE_ARG_B_IN));
        assertSame(ds.get(5).bufferArg(4), gate.bufferArg(DeltaNetKernels.GATE_ARG_A_IN));
        assertEquals(GdnHeadMap.GROUPED.code(), ds.get(9).intArg(DeltaNetKernels.SCAN_ARG_KV_HEAD_MODE));
    }

    @Test
    public void withoutMemoryStatesAreZeroedEveryStepAndRowZeroIsUsed() {
        open();
        LayerSpec layer = qwen35Layer();
        bind(MixerKind.GDN, Arch.CLEF, layer, false);
        backend.clearLog();
        mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH));

        List<FakeComputeBackend.Op> log = backend.ops();
        assertEquals(FakeComputeBackend.OpKind.FILL, log.get(0).kind());
        assertEquals(FakeComputeBackend.OpKind.FILL, log.get(1).kind());
        Dispatch conv = backend.dispatchesOf(CONV).get(0);
        Dispatch scan = backend.dispatchesOf(SCAN).get(0);
        assertSame(log.get(0).buffer(), conv.bufferArg(13));
        assertSame(log.get(1).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_STATE));
        assertSame(placeholders.readOnlyInts(), conv.bufferArg(14));
        assertSame(placeholders.readOnlyInts(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_STATE_ROWS));
    }

    @Test
    public void hyperConnectionLayerProjectsOneEmbeddingRow() {
        open();
        LayerSpec plain = qwen35Layer();
        LayerSpec layer = LayerSpec.builder(0, MixerKind.GDN, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.HYPER_CONNECTIONS, N_EMBD).hyperConnections(4, true)
                        .build())
                .recurrent(plain.recurrent())
                .build();
        bind(MixerKind.GDN, Arch.QWEN35, layer, true);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        assertEquals(MV, ds.get(0).kernelName());
        assertArrayEquals(new int[]{CONV_CH, N_EMBD}, ints(ds.get(0), 0, 2));
        assertArrayEquals(new int[]{N_EMBD, VAL}, ints(ds.get(ds.size() - 1), 0, 2));
    }

    @Test
    public void factoriesServeTheirKindOnly() {
        open();
        LayerSpec gdn = qwen35Layer();
        List<GpuMixerFactory> fs = DeltaNetBlocks.gpuMixers();
        assertEquals(Arrays.asList(MixerKind.GDN, MixerKind.KDA, MixerKind.LIGHTNING),
                Arrays.asList(fs.get(0).kind(), fs.get(1).kind(), fs.get(2).kind()));
        assertThrows(IllegalArgumentException.class, () -> fs.get(1).create(gdn));
        assertTrue(DeltaNetBlocks.gpuFfns().isEmpty());
        assertTrue(DeltaNetBlocks.gpuExtras().isEmpty());
    }

    // ------------------------------------------------------------------ kimi-linear KDA

    @Test
    public void kimiLinearKdaDecodeTrace() {
        open();
        int nH = 4;
        int d = 16;
        int dInner = nH * d;
        for (String s : Arrays.asList("attn_q", "attn_k", "attn_v")) {
            tensor("blk.0." + s + ".weight", N_EMBD, dInner);
        }
        for (String s : Arrays.asList("ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v")) {
            tensor("blk.0." + s + ".weight", D_CONV, 1, dInner);
        }
        tensor("blk.0.ssm_f_a.weight", N_EMBD, d);
        tensor("blk.0.ssm_f_b.weight", d, dInner);
        tensor("blk.0.ssm_beta.weight", N_EMBD, nH);
        tensor("blk.0.ssm_a", 1, nH);
        tensor("blk.0.ssm_dt.bias", dInner);
        tensor("blk.0.ssm_g_a.weight", N_EMBD, d);
        tensor("blk.0.ssm_g_b.weight", d, dInner);
        tensor("blk.0.ssm_norm.weight", d);
        tensor("blk.0.attn_output.weight", dInner, N_EMBD);
        LayerSpec layer = layer(MixerKind.KDA, RecurrentSpec.builder(MixerKind.KDA).nHead(nH).headDim(d)
                .dConv(D_CONV).kdaGateMode(KdaGateMode.LOWRANK).decayGate(RecurrentSpec.DecayGate.SOFTPLUS).build());
        bind(MixerKind.KDA, Arch.KIMI_LINEAR, layer, true);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        assertEquals(Arrays.asList(MV, CONV, MV, CONV, MV, CONV, MV, MV, MV, GATE, NORM, NORM, SCAN, NORM, MV, MV,
                GLU, MV), names(ds));
        String[] outs = {ScratchSlots.Q, ScratchSlots.K, ScratchSlots.V};
        String[] convW = {"blk.0.ssm_conv1d_q.weight", "blk.0.ssm_conv1d_k.weight", "blk.0.ssm_conv1d_v.weight"};
        for (int i = 0; i < 3; i++) {
            Dispatch c = ds.get(1 + 2 * i);
            assertArrayEquals(new int[]{dInner, D_CONV, 1, SsmKernels.CONV_FLAG_SILU, dInner, 0, dInner, 0,
                    i * dInner, 3 * dInner * (D_CONV - 1)}, ints(c, 0, 10));
            assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), c.bufferArg(10));
            assertSame(buf(convW[i]), c.bufferArg(11));
            assertSame(convState, c.bufferArg(13));
            assertSame(slot(outs[i]).buffer(), c.bufferArg(15));
        }
        Dispatch gate = ds.get(9);
        assertArrayEquals(new int[]{1, nH, nH, DeltaNetKernels.LAYOUT_SEPARATE,
                RecurrentSpec.DecayGate.SOFTPLUS.code(), d}, ints(gate, 0, 6));
        assertArrayEquals(new int[]{dInner, 0, nH, 0}, ints(gate, 7, 11));
        assertSame(slot(ScratchSlots.ATTN_OUT).buffer(), gate.bufferArg(DeltaNetKernels.GATE_ARG_A_IN));
        assertArrayEquals(new long[]{roundUp(dInner)}, gate.global());

        Dispatch scan = ds.get(12);
        assertArrayEquals(new int[]{d, d, nH, nH, 1, GdnHeadMap.TILED.code(), DeltaNetKernels.GATE_MODE_PER_KEY,
                dInner, 0, dInner, 0, dInner, 0, dInner, 0}, ints(scan, 0, 15));
        assertSame(slot(ScratchSlots.Q).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_Q));
        assertSame(slot(ScratchSlots.K).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_K));
        assertSame(slot(ScratchSlots.V).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_V));
        assertSame(slot(ScratchSlots.V).buffer(), scan.bufferArg(DeltaNetKernels.SCAN_ARG_OUT));
        assertSame(scanState, scan.bufferArg(DeltaNetKernels.SCAN_ARG_STATE));
        assertSame(buf("blk.0.attn_output.weight"), ds.get(17).bufferArg(2));
    }

    // ------------------------------------------------------------------ minimax-01 lightning

    @Test
    public void minimaxLightningDecodeTrace() {
        open();
        int nH = 4;
        int d = 16;
        int width = nH * d;
        tensor("blk.0.attn_qkv.weight", N_EMBD, 3L * width);
        tensor("blk.0.attn_gate.weight", N_EMBD, width);
        tensor("blk.0.attn_output.weight", width, N_EMBD);
        tensor("blk.0.attn_norm_2.weight", width);
        float[] slopes = {0.5f, 0.25f, 0.125f, 0.0625f};
        LayerSpec layer = layer(MixerKind.LIGHTNING, RecurrentSpec.builder(MixerKind.LIGHTNING).nHead(nH).headDim(d)
                .slopes(slopes).build());
        backend.clearLog();
        bind(MixerKind.LIGHTNING, Arch.MINIMAX_01, layer, true);
        List<FakeComputeBackend.Op> bindOps = backend.ops();
        FakeComputeBackend.Op upload = null;
        for (FakeComputeBackend.Op op : bindOps) {
            if (op.kind() == FakeComputeBackend.OpKind.WRITE) {
                upload = op;
            }
        }
        assertEquals(nH * Float.BYTES, upload.bytes());

        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        assertEquals(Arrays.asList(MV, GpuOps.UNARY, DeltaNetKernels.LIGHTNING_ATTN, NORM, MV, GLU, MV), names(ds));
        Dispatch la = ds.get(2);
        assertArrayEquals(new int[]{1, nH, d}, ints(la, 0, 3));
        assertEquals(1.0f, la.floatArg(DeltaNetKernels.LIGHTNING_ARG_LAYER_SCALE), 0.0f);
        assertSame(upload.buffer(), la.bufferArg(DeltaNetKernels.LIGHTNING_ARG_SLOPES));
        assertSame(slot(ScratchSlots.Q).buffer(), la.bufferArg(DeltaNetKernels.LIGHTNING_ARG_QKV));
        assertSame(scanState, la.bufferArg(DeltaNetKernels.LIGHTNING_ARG_STATE));
        assertSame(slot(ScratchSlots.K).buffer(), la.bufferArg(DeltaNetKernels.LIGHTNING_ARG_OUT));
        int local = recurrenceLocal();
        assertArrayEquals(new long[]{DeltaNetKernels.recurrenceGlobal(nH, d, local)}, la.global());
        assertArrayEquals(new long[]{ops.local()}, ds.get(3).global());
        assertTrue(ds.get(3).bufferArgs().contains(buf("blk.0.attn_norm_2.weight")));
    }
}

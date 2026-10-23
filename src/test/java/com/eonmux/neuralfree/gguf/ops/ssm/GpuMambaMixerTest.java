package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixerFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

/**
 * Dispatch traces of the SSM mixers on the host-only fake backend with strict LLM argument rules: the Mamba-2
 * decode / prefill / decode sequence with every SSM kernel argument and buffer identity,
 * the Mamba-1 sequence with weightless dt/B/C norms and dt compaction, the PLaMo-2 de-interleave, norms and
 * staged dt projection, the LFM2 causal and centered short convs, and a falcon-h1 layer whose attention branch is
 * left to the layer executor.
 */
public class GpuMambaMixerTest {

    private static final int UBATCH = 8;
    private static final int N_EMBD = 64;
    private static final int D_CONV = 4;
    private static final int D_INNER = 128;
    private static final int D_STATE = 16;
    private static final int N_GROUP = 2;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = D_INNER / N_HEAD;
    private static final int DT_RANK = 8;
    private static final int L_CACHE = 3;
    private static final int CONV_CH = D_INNER + 2 * N_GROUP * D_STATE;
    private static final int D_IN_PROJ = 2 * D_INNER + 2 * N_GROUP * D_STATE + N_HEAD;
    private static final int EL = SsmKernels.ELEMENTWISE_LOCAL;
    private static final String PROJ = "proj";
    private static final Set<String> NAMED = new HashSet<>(Arrays.asList(SsmKernels.SSM_CONV,
            SsmKernels.SHORTCONV_CENTERED, SsmKernels.SSM_SCAN, SsmKernels.MAMBA_GATE_OUT, NormKernels.ROW_NORM,
            NormKernels.GROUP_RMS, GpuOps.COPY2D, GpuOps.AXPBY, ActKernels.GLU));

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private DeviceModelMemory memory;
    private GpuMixer mixer;
    private final Map<String, GpuWeight> tensors = new HashMap<>();

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        long proj = (long) UBATCH * D_IN_PROJ;
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.Q, proj)
                .slot(ScratchSlots.K, proj)
                .slot(ScratchSlots.V, proj)
                .slot(ScratchSlots.ATTN_OUT, proj)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    @After
    public void tearDown() {
        if (mixer != null) {
            mixer.close();
        }
        if (memory != null) {
            memory.close();
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

    private static LayerSpec.Builder layer(MixerKind kind, RecurrentSpec r) {
        return LayerSpec.builder(0, kind, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .recurrent(r);
    }

    private static RecurrentSpec mamba2Spec() {
        return RecurrentSpec.builder(MixerKind.MAMBA2).nEmbd(N_EMBD).dConv(D_CONV).dInner(D_INNER).dState(D_STATE)
                .nGroup(N_GROUP).nHead(N_HEAD).headDim(HEAD_DIM).convBias(true).build();
    }

    private static RecurrentSpec mamba1Spec() {
        return RecurrentSpec.builder(MixerKind.MAMBA1).nEmbd(N_EMBD).dConv(D_CONV).dInner(D_INNER).dState(D_STATE)
                .dtRank(DT_RANK).dtBcRms(true).convBias(true).build();
    }

    private void mamba2Weights() {
        put(TensorNames.weight(TensorNames.SSM_IN), N_EMBD, D_IN_PROJ);
        put(TensorNames.weight(TensorNames.SSM_CONV1D), D_CONV, CONV_CH);
        put(TensorNames.bias(TensorNames.SSM_CONV1D), CONV_CH);
        put(TensorNames.bias(TensorNames.SSM_DT), N_HEAD);
        put(TensorNames.SSM_A, 1, N_HEAD);
        put(TensorNames.SSM_D, 1, N_HEAD);
        put(TensorNames.weight(TensorNames.SSM_NORM), D_INNER / N_GROUP, N_GROUP);
        put(TensorNames.weight(TensorNames.SSM_OUT), D_INNER, N_EMBD);
    }

    private void mamba1Weights() {
        put(TensorNames.weight(TensorNames.SSM_IN), N_EMBD, 2 * D_INNER);
        put(TensorNames.weight(TensorNames.SSM_CONV1D), D_CONV, D_INNER);
        put(TensorNames.bias(TensorNames.SSM_CONV1D), D_INNER);
        put(TensorNames.weight(TensorNames.SSM_X), D_INNER, DT_RANK + 2 * D_STATE);
        put(TensorNames.weight(TensorNames.SSM_DT), DT_RANK, D_INNER);
        put(TensorNames.bias(TensorNames.SSM_DT), D_INNER);
        put(TensorNames.SSM_A, D_STATE, D_INNER);
        put(TensorNames.SSM_D, D_INNER);
        put(TensorNames.weight(TensorNames.SSM_OUT), D_INNER, N_EMBD);
    }

    /** Context over a one-layer model; with recurrent memory planned and allocated when {@code withMemory}. */
    private GpuForwardContext context(Arch arch, LayerSpec l, boolean withMemory) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).ssmDConv(D_CONV).ssmDInner(D_INNER).ssmDState(D_STATE)
                .shortconvLCache(L_CACHE);
        if (l.mixerKind() == MixerKind.MAMBA2) {
            hb.ssmNGroup(N_GROUP).ssmDtRank(N_HEAD);
        } else if (l.mixerKind() == MixerKind.MAMBA1) {
            hb.ssmDtRank(DT_RANK).ssmDtBcRms(true);
        } else if (l.mixerKind() == MixerKind.PLAMO2_MAMBA) {
            hb.ssmDtRank(N_HEAD);
        }
        hb.isRecurrent(0, true);
        ModelSpec spec = ModelSpec.builder(hb.build(), TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
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
        GpuForwardContext.Builder b = GpuForwardContext.builder(ops.kernels(), ops, weights, spec);
        if (withMemory) {
            MemoryPlan plan = MemoryPlanner.planForContext(spec, MemoryConfig.builder().ubatch(UBATCH).build(), 64);
            memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
            b.memory(memory);
        }
        return b.build();
    }

    private GpuMixer create(List<GpuMixerFactory> factories, LayerSpec l) {
        for (GpuMixerFactory f : factories) {
            if (f.kind() == l.mixerKind()) {
                return f.create(l);
            }
        }
        throw new AssertionError("no factory for " + l.mixerKind());
    }

    private List<Dispatch> decode() {
        backend.clearLog();
        mixer.decode(0, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private List<Dispatch> prefill(int nTokens) {
        backend.clearLog();
        mixer.prefill(0, nTokens, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    /** Kernel names with each run of projection launches folded into {@value #PROJ}. */
    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            String n = NAMED.contains(d.kernelName()) ? d.kernelName() : PROJ;
            if (!(PROJ.equals(n) && !out.isEmpty() && PROJ.equals(out.get(out.size() - 1)))) {
                out.add(n);
            }
        }
        return out;
    }

    private static Dispatch only(List<Dispatch> ds, String kernel) {
        Dispatch found = null;
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                if (found != null) {
                    throw new AssertionError(kernel + " dispatched more than once");
                }
                found = d;
            }
        }
        if (found == null) {
            throw new AssertionError(kernel + " not dispatched");
        }
        return found;
    }

    private GpuBuffer slot(String name) {
        return scratch.slot(name).buffer();
    }

    // ------------------------------------------------------------------ Mamba-2

    @Test
    public void mamba2DecodePrefillDecodeTrace() {
        open();
        mamba2Weights();
        LayerSpec l = layer(MixerKind.MAMBA2, mamba2Spec()).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.MAMBA2, l, true), l);
        GpuBuffer convState = memory.recurrent(0).tensor(LayerMemoryPlan.STATE_CONV);
        GpuBuffer scanState = memory.recurrent(0).tensor(LayerMemoryPlan.STATE_SCAN);
        GpuBuffer stateRows = memory.stepInputs().stateRows();

        for (int t : new int[]{1, 5, 1}) {
            List<Dispatch> ds = t == 1 ? decode() : prefill(t);
            assertEquals(Arrays.asList(PROJ, SsmKernels.SSM_CONV, SsmKernels.SSM_SCAN, SsmKernels.MAMBA_GATE_OUT,
                    NormKernels.GROUP_RMS, PROJ), names(ds));
            assertMamba2Conv(only(ds, SsmKernels.SSM_CONV), t, convState, stateRows);
            assertMamba2Scan(only(ds, SsmKernels.SSM_SCAN), t, scanState, stateRows);
            assertMamba2Gate(only(ds, SsmKernels.MAMBA_GATE_OUT), t);

            Dispatch norm = only(ds, NormKernels.GROUP_RMS);
            assertEquals(t, norm.intArg(NormKernels.ARG_GROUP_RMS_N_TOKENS));
            assertEquals(D_INNER, norm.intArg(NormKernels.ARG_GROUP_RMS_D_INNER));
            assertEquals(N_GROUP, norm.intArg(NormKernels.ARG_GROUP_RMS_N_GROUP));
            assertEquals(D_INNER, norm.intArg(NormKernels.ARG_GROUP_RMS_X_STRIDE));
            assertSame(slot(ScratchSlots.V), norm.bufferArg(NormKernels.ARG_GROUP_RMS_X));
            assertSame(slot(ScratchSlots.V), norm.bufferArg(NormKernels.ARG_GROUP_RMS_Y));
            int local = (int) norm.local()[0];
            assertArrayEquals(new long[]{(long) t * N_GROUP * local}, norm.global());
        }
    }

    private void assertMamba2Conv(Dispatch d, int t, GpuBuffer convState, GpuBuffer stateRows) {
        assertEquals(SsmKernels.CONV_ARG_COUNT, d.argCount());
        assertEquals(CONV_CH, d.intArg(0));
        assertEquals(D_CONV, d.intArg(1));
        assertEquals(t, d.intArg(SsmKernels.ARG_CONV_N_TOKENS));
        assertEquals(SsmKernels.CONV_FLAG_BIAS | SsmKernels.CONV_FLAG_SILU, d.intArg(3));
        assertEquals(D_IN_PROJ, d.intArg(4));
        assertEquals(D_INNER, d.intArg(5));
        assertEquals(CONV_CH, d.intArg(6));
        assertEquals(0, d.intArg(7));
        assertEquals(0, d.intArg(8));
        assertEquals((D_CONV - 1) * CONV_CH, d.intArg(9));
        assertSame(slot(ScratchSlots.Q), d.bufferArg(10));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.weight(TensorNames.SSM_CONV1D))).shard(0),
                d.bufferArg(11));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.bias(TensorNames.SSM_CONV1D))).shard(0),
                d.bufferArg(12));
        assertSame(convState, d.bufferArg(13));
        assertSame(stateRows, d.bufferArg(14));
        assertSame(slot(ScratchSlots.K), d.bufferArg(15));
        assertArrayEquals(new long[]{SsmKernels.elementwiseGlobal(CONV_CH, EL)}, d.global());
        assertArrayEquals(new long[]{EL}, d.local());
    }

    private void assertMamba2Scan(Dispatch d, int t, GpuBuffer scanState, GpuBuffer stateRows) {
        SsmKernels.ScanGeometry geo = SsmKernels.ScanGeometry.of(D_STATE, HEAD_DIM, N_HEAD, N_GROUP);
        int gs = N_GROUP * D_STATE;
        assertEquals(SsmKernels.SCAN_ARG_COUNT, d.argCount());
        int[] scalars = {D_STATE, HEAD_DIM, N_HEAD, N_GROUP, t, 0, geo.channelsPerGroup(), CONV_CH, 0, D_IN_PROJ,
            2 * D_INNER + 2 * gs, CONV_CH, CONV_CH, D_INNER, D_INNER + gs, D_INNER, 1};
        for (int i = 0; i < scalars.length; i++) {
            assertEquals("scan arg " + i, scalars[i], d.intArg(i));
        }
        assertSame(slot(ScratchSlots.K), d.bufferArg(17));
        assertSame(slot(ScratchSlots.Q), d.bufferArg(18));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.bias(TensorNames.SSM_DT))).shard(0),
                d.bufferArg(19));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.SSM_A)).shard(0), d.bufferArg(20));
        assertSame(slot(ScratchSlots.K), d.bufferArg(21));
        assertSame(slot(ScratchSlots.K), d.bufferArg(22));
        assertSame(scanState, d.bufferArg(23));
        assertSame(stateRows, d.bufferArg(24));
        assertSame(slot(ScratchSlots.ATTN_OUT), d.bufferArg(25));
        assertArrayEquals(new long[]{geo.globalSize()}, d.global());
        assertArrayEquals(new long[]{geo.localSize()}, d.local());
    }

    private void assertMamba2Gate(Dispatch d, int t) {
        assertEquals(SsmKernels.GATE_ARG_COUNT, d.argCount());
        int[] scalars = {t * D_INNER, D_INNER, HEAD_DIM, CONV_CH, 0, 0, D_IN_PROJ, 0};
        for (int i = 0; i < scalars.length; i++) {
            assertEquals("gate arg " + i, scalars[i], d.intArg(i));
        }
        assertSame(slot(ScratchSlots.ATTN_OUT), d.bufferArg(8));
        assertSame(slot(ScratchSlots.K), d.bufferArg(9));
        assertSame(tensors.get(TensorNames.layerName(0, TensorNames.SSM_D)).shard(0), d.bufferArg(10));
        assertSame(slot(ScratchSlots.Q), d.bufferArg(11));
        assertSame(slot(ScratchSlots.V), d.bufferArg(12));
        assertArrayEquals(new long[]{SsmKernels.elementwiseGlobal((long) t * D_INNER, EL)}, d.global());
    }

    // ------------------------------------------------------------------ Mamba-1

    @Test
    public void mamba1NormsDtAndCompaction() {
        open();
        mamba1Weights();
        LayerSpec l = layer(MixerKind.MAMBA1, mamba1Spec()).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.MAMBA, l, true), l);
        int xdb = DT_RANK + 2 * D_STATE;

        List<Dispatch> one = decode();
        assertEquals(Arrays.asList(PROJ, SsmKernels.SSM_CONV, PROJ, NormKernels.ROW_NORM, NormKernels.ROW_NORM,
                NormKernels.ROW_NORM, PROJ, SsmKernels.SSM_SCAN, SsmKernels.MAMBA_GATE_OUT, PROJ), names(one));
        int[] fieldOff = {0, DT_RANK, DT_RANK + D_STATE};
        int[] fieldLen = {DT_RANK, D_STATE, D_STATE};
        int n = 0;
        for (Dispatch d : one) {
            if (d.kernelName().equals(NormKernels.ROW_NORM)) {
                assertEquals(1, d.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
                assertEquals(fieldLen[n], d.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
                assertEquals(xdb, d.intArg(NormKernels.ARG_ROW_NORM_ROW_STRIDE));
                assertEquals(0, d.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
                assertEquals(fieldOff[n], d.intArg(NormKernels.ARG_ROW_NORM_X_OFF));
                assertEquals(fieldOff[n], d.intArg(NormKernels.ARG_ROW_NORM_Y_OFF));
                assertSame(slot(ScratchSlots.V), d.bufferArg(NormKernels.ARG_ROW_NORM_X));
                n++;
            }
        }
        assertEquals(3, n);
        Dispatch scan = only(one, SsmKernels.SSM_SCAN);
        assertEquals(1, scan.intArg(5));
        assertEquals(D_INNER, scan.intArg(2));
        assertEquals(1, scan.intArg(1));
        assertEquals(DT_RANK, scan.intArg(13));
        assertEquals(DT_RANK + D_STATE, scan.intArg(14));
        assertSame(slot(ScratchSlots.V), scan.bufferArg(21));
        assertSame(slot(ScratchSlots.ATTN_OUT), scan.bufferArg(25));
        Dispatch gate = only(one, SsmKernels.MAMBA_GATE_OUT);
        assertEquals(1, gate.intArg(2));
        assertEquals(D_INNER, gate.intArg(7));
        assertSame(slot(ScratchSlots.V), gate.bufferArg(12));

        List<Dispatch> three = prefill(3);
        assertEquals(Arrays.asList(PROJ, SsmKernels.SSM_CONV, PROJ, NormKernels.ROW_NORM, NormKernels.ROW_NORM,
                NormKernels.ROW_NORM, GpuOps.COPY2D, PROJ, SsmKernels.SSM_SCAN, SsmKernels.MAMBA_GATE_OUT, PROJ),
                names(three));
        assertEquals(3, only(three, SsmKernels.SSM_SCAN).intArg(SsmKernels.ARG_SCAN_N_TOKENS));
        assertEquals(3 * D_INNER, only(three, SsmKernels.MAMBA_GATE_OUT).intArg(SsmKernels.ARG_GATE_N));
    }

    // ------------------------------------------------------------------ PLaMo-2

    @Test
    public void plamo2DeinterleaveNormsAndStagedDt() {
        open();
        int xdb = 2 * D_STATE + DT_RANK;
        put(TensorNames.weight(TensorNames.SSM_IN), N_EMBD, 2 * D_INNER);
        put(TensorNames.weight(TensorNames.SSM_CONV1D), D_CONV, D_INNER);
        put(TensorNames.weight(TensorNames.SSM_X), D_INNER, xdb);
        put(TensorNames.weight(TensorNames.SSM_DT), DT_RANK, N_HEAD);
        put(TensorNames.bias(TensorNames.SSM_DT), N_HEAD);
        put(TensorNames.SSM_A, N_HEAD);
        put(TensorNames.SSM_D, N_HEAD);
        put(TensorNames.weight(TensorNames.SSM_OUT), D_INNER, N_EMBD);
        put(TensorNames.SSM_DT_NORM, DT_RANK);
        put(TensorNames.SSM_B_NORM, D_STATE);
        put(TensorNames.SSM_C_NORM, D_STATE);
        RecurrentSpec r = RecurrentSpec.builder(MixerKind.PLAMO2_MAMBA).nEmbd(N_EMBD).dConv(D_CONV).dInner(D_INNER)
                .dState(D_STATE).nGroup(0).nHead(N_HEAD).headDim(HEAD_DIM).dtRank(DT_RANK).build();
        LayerSpec l = layer(MixerKind.PLAMO2_MAMBA, r).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.PLAMO2, l, true), l);
        GpuBuffer convState = memory.recurrent(0).tensor(LayerMemoryPlan.STATE_CONV);
        GpuBuffer scanState = memory.recurrent(0).tensor(LayerMemoryPlan.STATE_SCAN);
        GpuBuffer stateRows = memory.stepInputs().stateRows();
        SsmKernels.ScanGeometry geo = SsmKernels.ScanGeometry.of(D_STATE, HEAD_DIM, N_HEAD, 1);

        for (int t : new int[]{1, 3}) {
            List<Dispatch> ds = t == 1 ? decode() : prefill(t);
            assertEquals(Arrays.asList(PROJ, GpuOps.COPY2D, GpuOps.COPY2D, SsmKernels.SSM_CONV, PROJ,
                    NormKernels.ROW_NORM, NormKernels.ROW_NORM, NormKernels.ROW_NORM, GpuOps.COPY2D, PROJ,
                    SsmKernels.SSM_SCAN, SsmKernels.MAMBA_GATE_OUT, PROJ), names(ds));

            List<Dispatch> copies = new ArrayList<>();
            List<Dispatch> norms = new ArrayList<>();
            for (Dispatch d : ds) {
                if (d.kernelName().equals(GpuOps.COPY2D)) {
                    copies.add(d);
                } else if (d.kernelName().equals(NormKernels.ROW_NORM)) {
                    norms.add(d);
                }
            }
            int[][] split = {{HEAD_DIM, 0}, {0, 0}};
            GpuBuffer[] splitDst = {slot(ScratchSlots.K), slot(ScratchSlots.V)};
            for (int i = 0; i < 2; i++) {
                Dispatch c = copies.get(i);
                assertArrayEquals(new int[]{t * N_HEAD, HEAD_DIM, 2 * HEAD_DIM, split[i][0], HEAD_DIM, split[i][1]},
                        new int[]{c.intArg(0), c.intArg(1), c.intArg(2), c.intArg(3), c.intArg(4), c.intArg(5)});
                assertSame(slot(ScratchSlots.Q), c.bufferArg(6));
                assertSame(splitDst[i], c.bufferArg(7));
            }
            Dispatch dtCopy = copies.get(2);
            assertEquals(t, dtCopy.intArg(0));
            assertEquals(DT_RANK, dtCopy.intArg(1));
            if (t > 1) {
                assertEquals(xdb, dtCopy.intArg(2));
                assertSame(slot(ScratchSlots.K), dtCopy.bufferArg(7));
            }
            assertEquals(2 * D_STATE, dtCopy.intArg(3));
            assertSame(slot(ScratchSlots.Q), dtCopy.bufferArg(6));

            int[] normLen = {D_STATE, D_STATE, DT_RANK};
            int[] normOff = {0, D_STATE, 2 * D_STATE};
            for (int i = 0; i < 3; i++) {
                Dispatch nd = norms.get(i);
                assertEquals(t, nd.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
                assertEquals(normLen[i], nd.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
                assertEquals(xdb, nd.intArg(NormKernels.ARG_ROW_NORM_ROW_STRIDE));
                assertEquals(NormKernels.FLAG_HAS_W, nd.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
                assertEquals(normOff[i], nd.intArg(NormKernels.ARG_ROW_NORM_X_OFF));
                assertSame(slot(ScratchSlots.Q), nd.bufferArg(NormKernels.ARG_ROW_NORM_X));
                assertSame(slot(ScratchSlots.Q), nd.bufferArg(NormKernels.ARG_ROW_NORM_Y));
            }

            Dispatch conv = only(ds, SsmKernels.SSM_CONV);
            int[] convScalars = {D_INNER, D_CONV, t, SsmKernels.CONV_FLAG_SILU, D_INNER, 0, D_INNER, 0, 0,
                (D_CONV - 1) * D_INNER};
            for (int i = 0; i < convScalars.length; i++) {
                assertEquals("conv arg " + i, convScalars[i], conv.intArg(i));
            }
            assertSame(slot(ScratchSlots.K), conv.bufferArg(10));
            assertSame(placeholders.readOnlyFloats(), conv.bufferArg(12));
            assertSame(convState, conv.bufferArg(13));
            assertSame(stateRows, conv.bufferArg(14));
            assertSame(slot(ScratchSlots.ATTN_OUT), conv.bufferArg(15));

            Dispatch scan = only(ds, SsmKernels.SSM_SCAN);
            int[] scanScalars = {D_STATE, HEAD_DIM, N_HEAD, 1, t, 0, geo.channelsPerGroup(), D_INNER, 0, N_HEAD, 0,
                xdb, xdb, 0, D_STATE, D_INNER, 1};
            for (int i = 0; i < scanScalars.length; i++) {
                assertEquals("scan arg " + i, scanScalars[i], scan.intArg(i));
            }
            assertSame(slot(ScratchSlots.ATTN_OUT), scan.bufferArg(17));
            assertSame(slot(ScratchSlots.Q), scan.bufferArg(21));
            assertSame(slot(ScratchSlots.Q), scan.bufferArg(22));
            assertSame(scanState, scan.bufferArg(23));
            assertSame(slot(ScratchSlots.K), scan.bufferArg(25));

            Dispatch gate = only(ds, SsmKernels.MAMBA_GATE_OUT);
            int[] gateScalars = {t * D_INNER, D_INNER, HEAD_DIM, D_INNER, 0, 0, D_INNER, 0};
            for (int i = 0; i < gateScalars.length; i++) {
                assertEquals("gate arg " + i, gateScalars[i], gate.intArg(i));
            }
            assertSame(slot(ScratchSlots.K), gate.bufferArg(8));
            assertSame(slot(ScratchSlots.ATTN_OUT), gate.bufferArg(9));
            assertSame(slot(ScratchSlots.V), gate.bufferArg(11));
            assertSame(slot(ScratchSlots.Q), gate.bufferArg(12));
        }
    }

    // ------------------------------------------------------------------ LFM2 causal

    @Test
    public void lfm2CausalShortConvTrace() {
        open();
        put(TensorNames.weight(TensorNames.SHORTCONV_INPROJ), N_EMBD, 3 * N_EMBD);
        put(TensorNames.weight(TensorNames.SHORTCONV_CONV), L_CACHE, N_EMBD);
        put(TensorNames.weight(TensorNames.SHORTCONV_OUTPROJ), N_EMBD, N_EMBD);
        RecurrentSpec r = RecurrentSpec.builder(MixerKind.SHORTCONV).nEmbd(N_EMBD).lCache(L_CACHE).build();
        LayerSpec l = layer(MixerKind.SHORTCONV, r).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.LFM2, l, true), l);
        GpuBuffer convState = memory.recurrent(0).tensor(LayerMemoryPlan.STATE_CONV);
        GpuBuffer stateRows = memory.stepInputs().stateRows();

        for (int t : new int[]{1, 4}) {
            List<Dispatch> ds = t == 1 ? decode() : prefill(t);
            assertEquals(Arrays.asList(PROJ, ActKernels.GLU, SsmKernels.SSM_CONV, ActKernels.GLU, PROJ), names(ds));
            Dispatch conv = only(ds, SsmKernels.SSM_CONV);
            assertEquals(SsmKernels.CONV_ARG_COUNT, conv.argCount());
            int[] scalars = {N_EMBD, L_CACHE, t, 0, N_EMBD, 0, N_EMBD, 0, 0, (L_CACHE - 1) * N_EMBD};
            for (int i = 0; i < scalars.length; i++) {
                assertEquals("conv arg " + i, scalars[i], conv.intArg(i));
            }
            assertSame(slot(ScratchSlots.K), conv.bufferArg(10));
            assertSame(placeholders.readOnlyFloats(), conv.bufferArg(12));
            assertSame(convState, conv.bufferArg(13));
            assertSame(stateRows, conv.bufferArg(14));
            assertSame(slot(ScratchSlots.V), conv.bufferArg(15));
            assertArrayEquals(new long[]{SsmKernels.elementwiseGlobal(N_EMBD, EL)}, conv.global());
        }
    }

    // ------------------------------------------------------------------ LFM2 centered

    @Test
    public void lfm2CenteredShortConvTrace() {
        open();
        put(TensorNames.weight(TensorNames.SHORTCONV_INPROJ), N_EMBD, 3 * N_EMBD);
        put(TensorNames.weight(TensorNames.SHORTCONV_CONV), L_CACHE, N_EMBD);
        put(TensorNames.weight(TensorNames.SHORTCONV_OUTPROJ), N_EMBD, N_EMBD);
        RecurrentSpec r = RecurrentSpec.builder(MixerKind.SHORTCONV_CENTERED).nEmbd(N_EMBD).lCache(L_CACHE).build();
        LayerSpec l = layer(MixerKind.SHORTCONV_CENTERED, r).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.LFM2, l, false), l);

        List<Dispatch> ds = prefill(5);
        assertEquals(Arrays.asList(PROJ, ActKernels.GLU, SsmKernels.SHORTCONV_CENTERED, ActKernels.GLU, PROJ),
                names(ds));
        Dispatch conv = only(ds, SsmKernels.SHORTCONV_CENTERED);
        assertEquals(SsmKernels.CENTERED_ARG_COUNT, conv.argCount());
        int[] scalars = {N_EMBD, L_CACHE, 5, N_EMBD, 0, N_EMBD};
        for (int i = 0; i < scalars.length; i++) {
            assertEquals("centered arg " + i, scalars[i], conv.intArg(i));
        }
        assertSame(slot(ScratchSlots.K), conv.bufferArg(6));
        assertSame(slot(ScratchSlots.V), conv.bufferArg(8));
        assertArrayEquals(new long[]{SsmKernels.elementwiseGlobal(5L * N_EMBD, EL)}, conv.global());

        assertEquals(1, only(decode(), SsmKernels.SHORTCONV_CENTERED).intArg(SsmKernels.ARG_CENTERED_N_TOKENS));
    }

    // ------------------------------------------------------------------ falcon-h1 parallel attention

    private static LayerSpec parallelLayer() {
        AttentionSpec attn = AttentionSpec.builder().nHead(2).nHeadKv(2).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(0.125f).kvOwnerLayer(0).build();
        return layer(MixerKind.MAMBA2, mamba2Spec()).attention(attn).parallelAttention(true).build();
    }

    @Test
    public void parallelAttentionLayerRunsOnlyTheMamba2Branch() {
        open();
        mamba2Weights();
        LayerSpec l = parallelLayer();
        LayerSpec plain = layer(MixerKind.MAMBA2, mamba2Spec()).build();
        mixer = create(SsmBlocks.gpuMixers(), l);
        mixer.bind(context(Arch.MAMBA2, plain, true), l);

        List<String> expected = Arrays.asList(PROJ, SsmKernels.SSM_CONV, SsmKernels.SSM_SCAN,
                SsmKernels.MAMBA_GATE_OUT, NormKernels.GROUP_RMS, PROJ);
        assertEquals(expected, names(decode()));
        assertEquals(expected, names(prefill(3)));
    }
}

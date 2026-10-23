package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
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
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Op;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
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
import static org.junit.Assert.assertThrows;

/**
 * Dispatch traces of {@link GpuRwkv6Mixer} on the host-only fake backend with strict LLM argument rules: the rwkv6
 * WKV6 path (decode and prefill) and the rwkv6qwen2 QRWKV path (GQA head repeat, projection biases, {@code k -= k*w},
 * GLA). Projections and the {@code LLM_Copy2D_F32} staging of {@link GpuOps} are left out of the block-kernel
 * sequences and checked separately where the mixer issues them itself.
 */
public class GpuRwkv6MixerTest {

    private static final int C = 64;
    private static final int S = 32;
    private static final int H = C / S;
    private static final int E = 8;
    private static final int D = 16;
    private static final int UBATCH = 8;
    private static final int N_CTX = 64;

    private static final Set<String> BLOCK_KERNELS = blockKernels();

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private DeviceModelMemory memory;
    private GpuRwkv6Mixer mixer;
    private GpuForwardContext context;
    private final Map<String, GpuWeight> tensors = new HashMap<>();

    private static Set<String> blockKernels() {
        Set<String> s = new HashSet<>(RwkvKernels.ALL);
        s.addAll(Arrays.asList(RwkvBlocks.BATCH_BIAS_ADD, GpuOps.UNARY, GpuOps.GLU, GpuOps.ROW_NORM, GpuOps.AXPBY));
        return Collections.unmodifiableSet(s);
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

    private static RecurrentSpec rwkv6Spec() {
        return RecurrentSpec.builder(MixerKind.RWKV6).nEmbd(C).wkvHeadSize(S).tokenShiftCount(2)
                .timeMixExtraDim(E).timeDecayExtraDim(D).build();
    }

    private static RecurrentSpec qrwkvSpec() {
        return RecurrentSpec.builder(MixerKind.RWKV6_QWEN2).nEmbd(C).wkvHeadSize(S).tokenShiftCount(1).qrwkv(true)
                .timeMixExtraDim(E).timeDecayExtraDim(D).build();
    }

    private static LayerSpec layer(MixerKind kind, RecurrentSpec rs) {
        return LayerSpec.builder(0, kind, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.LAYER, 1e-5f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, C).build())
                .recurrent(rs)
                .build();
    }

    private static ModelSpec model(Arch arch, LayerSpec layer) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nCtxTrain(N_CTX)
                .wkvHeadSize(S).build();
        return ModelSpec.builder(hp, TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, C).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }

    private void open(Arch arch, LayerSpec layer, boolean withMemory) {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * C * 2)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * C)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        LlmKernels kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        GpuForwardContext.Builder b = GpuForwardContext.builder(kernels, ops, tensors::get, model(arch, layer));
        if (withMemory) {
            MemoryPlan plan = MemoryPlan.builder(MemoryKind.RECURRENT, N_CTX, UBATCH)
                    .layer(LayerMemoryPlan.builder(0, LayerMemoryPlan.Kind.RECURRENT)
                            .states(LayerMemoryPlan.recurrentState(layer.recurrent())).build())
                    .build();
            memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
            b.memory(memory);
        }
        mixer = new GpuRwkv6Mixer();
        context = b.build();
    }

    private GpuWeight put(LayerSpec layer, String suffix, long... ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        String name = layer.tensorName(suffix);
        GpuWeight w = GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true));
        tensors.put(name, w);
        return w;
    }

    private GpuWeight weight(LayerSpec layer, String base, long... ne) {
        return put(layer, TensorNames.weight(base), ne);
    }

    /** Tensors shared by both paths; {@code kv} is the key / value projection width. */
    private void commonTensors(LayerSpec layer, int kv) {
        weight(layer, TensorNames.TIME_MIX_W1, C, 5L * E);
        weight(layer, TensorNames.TIME_MIX_W2, E, C, 5);
        weight(layer, TensorNames.TIME_MIX_LERP_X, C);
        weight(layer, TensorNames.TIME_MIX_RECEPTANCE, C, C);
        weight(layer, TensorNames.TIME_MIX_KEY, C, kv);
        weight(layer, TensorNames.TIME_MIX_VALUE, C, kv);
        weight(layer, TensorNames.TIME_MIX_GATE, C, C);
        weight(layer, TensorNames.TIME_MIX_OUTPUT, C, C);
        weight(layer, TensorNames.TIME_MIX_DECAY_W1, C, D);
        weight(layer, TensorNames.TIME_MIX_DECAY_W2, D, C);
        weight(layer, TensorNames.TIME_MIX_DECAY, C);
    }

    /** rwkv6 with the five unfused lerps (packed at bind time), time_mix_first and the group norm. */
    private LayerSpec rwkv6() {
        LayerSpec layer = layer(MixerKind.RWKV6, rwkv6Spec());
        open(Arch.RWKV6, layer, true);
        commonTensors(layer, C);
        for (String lerp : Arrays.asList(TensorNames.TIME_MIX_LERP_W, TensorNames.TIME_MIX_LERP_K,
                TensorNames.TIME_MIX_LERP_V, TensorNames.TIME_MIX_LERP_R, TensorNames.TIME_MIX_LERP_G)) {
            weight(layer, lerp, C);
        }
        weight(layer, TensorNames.TIME_MIX_FIRST, S, H);
        weight(layer, TensorNames.TIME_MIX_LN, C);
        put(layer, TensorNames.bias(TensorNames.TIME_MIX_LN), C);
        return layer;
    }

    /** rwkv6qwen2: one KV head of two, fused lerps, projection biases, no time_mix_first. */
    private LayerSpec qrwkv() {
        LayerSpec layer = layer(MixerKind.RWKV6_QWEN2, qrwkvSpec());
        open(Arch.RWKV6QWEN2, layer, true);
        commonTensors(layer, S);
        weight(layer, TensorNames.TIME_MIX_LERP_FUSED, C, 1, 1, 5);
        put(layer, TensorNames.bias(TensorNames.TIME_MIX_RECEPTANCE), C);
        put(layer, TensorNames.bias(TensorNames.TIME_MIX_KEY), S);
        put(layer, TensorNames.bias(TensorNames.TIME_MIX_VALUE), S);
        return layer;
    }

    private List<Dispatch> run(Runnable r) {
        backend.clearLog();
        r.run();
        return backend.dispatches();
    }

    private static List<Dispatch> blockDispatches(List<Dispatch> all) {
        List<Dispatch> out = new ArrayList<>();
        for (Dispatch d : all) {
            if (BLOCK_KERNELS.contains(d.kernelName())) {
                out.add(d);
            }
        }
        return out;
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

    private GpuBuffer temp(String name) {
        return scratch.temp(name, 1).buffer();
    }

    private GpuBuffer stateTensor(String name) {
        return memory.recurrent(0).tensor(name);
    }

    // ------------------------------------------------------------------ rwkv6

    @Test
    public void bindCopiesW2RowBlocksAndPacksUnfusedLerps() {
        LayerSpec layer = rwkv6();
        backend.clearLog();
        mixer.bind(context, layer);
        GpuWeight w2 = tensors.get(layer.tensorName(TensorNames.weight(TensorNames.TIME_MIX_W2)));
        List<Op> copies = new ArrayList<>();
        for (Op op : backend.ops()) {
            if (op.kind() == OpKind.COPY) {
                copies.add(op);
            }
        }
        assertEquals(10, copies.size());
        long blockBytes = (long) C * E * Float.BYTES;
        for (int m = 0; m < 5; m++) {
            Op op = copies.get(m);
            assertSame(w2.shard(0), op.source());
            assertEquals(m * blockBytes, op.sourceOffset());
            assertEquals(0L, op.offset());
            assertEquals(blockBytes, op.bytes());
        }
        GpuBuffer packed = copies.get(5).buffer();
        String[] lerps = {TensorNames.TIME_MIX_LERP_W, TensorNames.TIME_MIX_LERP_K, TensorNames.TIME_MIX_LERP_V,
                TensorNames.TIME_MIX_LERP_R, TensorNames.TIME_MIX_LERP_G};
        for (int m = 0; m < 5; m++) {
            Op op = copies.get(5 + m);
            assertSame(tensors.get(layer.tensorName(TensorNames.weight(lerps[m]))).shard(0), op.source());
            assertSame(packed, op.buffer());
            assertEquals((long) m * C * Float.BYTES, op.offset());
            assertEquals((long) C * Float.BYTES, op.bytes());
        }
    }

    @Test
    public void rwkv6DecodeRunsWkv6GroupNormAndSiluGate() {
        LayerSpec layer = rwkv6();
        mixer.bind(context, layer);
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED).slice(C, C);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        List<Dispatch> ds = blockDispatches(run(() -> mixer.decode(7, in, out)));

        assertEquals(Arrays.asList(RwkvKernels.TOKEN_SHIFT_LERP, GpuOps.UNARY, RwkvKernels.TOKEN_SHIFT_LERP,
                GpuOps.UNARY, RwkvKernels.DECAY6, RwkvKernels.WKV6, GpuOps.ROW_NORM, GpuOps.GLU), names(ds));
        GpuBuffer shift = stateTensor(LayerMemoryPlan.STATE_SHIFT);
        GpuBuffer planes = temp("rwkv6.planes");

        Dispatch lerpX = ds.get(0);
        assertArrayEquals(new int[]{1, C, 0, RwkvKernels.SHIFT_MODE_LERP_X, 0, C}, ints(lerpX, 0, 6));
        assertSame(in.buffer(), lerpX.bufferArg(6));
        assertSame(shift, lerpX.bufferArg(7));
        assertSame(temp("rwkv6.xy"), lerpX.bufferArg(10));
        assertArrayEquals(new long[]{RwkvKernels.tokenShiftGlobalSize(1, C)}, lerpX.global());
        assertArrayEquals(new long[]{RwkvKernels.ELEMENTWISE_LOCAL}, lerpX.local());

        Dispatch lora = ds.get(2);
        assertArrayEquals(new int[]{1, C, 5, RwkvKernels.SHIFT_MODE_LORA, 0, C}, ints(lora, 0, 6));
        assertSame(planes, lora.bufferArg(10));

        Dispatch decay = ds.get(4);
        assertArrayEquals(new int[]{C, C}, ints(decay, 0, 2));
        assertSame(temp("rwkv6.w_raw"), decay.bufferArg(2));
        assertSame(tensors.get(layer.tensorName(TensorNames.weight(TensorNames.TIME_MIX_DECAY))).shard(0),
                decay.bufferArg(3));
        assertSame(temp("rwkv6.w"), decay.bufferArg(4));

        Dispatch wkv = ds.get(5);
        assertArrayEquals(new int[]{1, C, H}, ints(wkv, 0, 3));
        assertSame(temp("rwkv6.k"), wkv.bufferArg(3));
        assertSame(temp("rwkv6.v"), wkv.bufferArg(4));
        assertSame(temp("rwkv6.r"), wkv.bufferArg(5));
        assertSame(tensors.get(layer.tensorName(TensorNames.weight(TensorNames.TIME_MIX_FIRST))).shard(0),
                wkv.bufferArg(6));
        assertSame(temp("rwkv6.w"), wkv.bufferArg(7));
        assertSame(stateTensor(LayerMemoryPlan.STATE_SCAN), wkv.bufferArg(8));
        assertSame(memory.stepInputs().stateRows(), wkv.bufferArg(9));
        assertSame(temp("rwkv6.xy"), wkv.bufferArg(10));
        assertArrayEquals(new long[]{RwkvKernels.wkvGlobalSize(H, S)}, wkv.global());
        assertArrayEquals(new long[]{RwkvKernels.wkvLocalSize(S)}, wkv.local());

        Dispatch norm = ds.get(6);
        assertEquals(H, norm.intArg(0));
        assertEquals(S, norm.intArg(1));

        Dispatch gate = ds.get(7);
        assertEquals(1, gate.intArg(1));
        assertEquals(GluOp.SWIGLU.code(), gate.intArg(2));
        assertSame(temp("rwkv6.g"), gate.bufferArg(12));
        assertSame(temp("rwkv6.xy"), gate.bufferArg(13));
        assertSame(temp("rwkv6.xy"), gate.bufferArg(14));
    }

    @Test
    public void rwkv6PrefillDeinterleavesLoraRowsAndScalesTheGrids() {
        LayerSpec layer = rwkv6();
        mixer.bind(context, layer);
        int t = 4;
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        List<Dispatch> all = run(() -> mixer.prefill(0, t, in, out));

        GpuBuffer h = temp("rwkv6.h");
        GpuBuffer hd = temp("rwkv6.hd");
        int m = 0;
        for (Dispatch d : all) {
            if (GpuOps.COPY2D.equals(d.kernelName()) && d.bufferArg(d.argCount() - 2) == h) {
                assertSame(hd, d.bufferArg(d.argCount() - 1));
                assertArrayEquals(new int[]{t, E, 5 * E, m * E, E, m * t * E}, ints(d, 0, 6));
                m++;
            }
        }
        assertEquals(5, m);

        List<Dispatch> ds = blockDispatches(all);
        assertEquals(Arrays.asList(RwkvKernels.TOKEN_SHIFT_LERP, GpuOps.UNARY, RwkvKernels.TOKEN_SHIFT_LERP,
                GpuOps.UNARY, RwkvKernels.DECAY6, RwkvKernels.WKV6, GpuOps.ROW_NORM, GpuOps.GLU), names(ds));
        assertEquals(t, ds.get(0).intArg(0));
        assertEquals(0, ds.get(0).intArg(5));
        assertArrayEquals(new long[]{RwkvKernels.tokenShiftGlobalSize(t, C)}, ds.get(2).global());
        assertEquals(t * 5 * E, ds.get(1).intArg(0));
        assertEquals(t * D, ds.get(3).intArg(0));
        assertEquals(t * C, ds.get(4).intArg(0));
        assertArrayEquals(new long[]{RwkvKernels.elementwiseGlobalSize((long) t * C)}, ds.get(4).global());
        assertEquals(t, ds.get(5).intArg(0));
        assertEquals(t * H, ds.get(6).intArg(0));
        assertEquals(t, ds.get(7).intArg(1));
    }

    // ------------------------------------------------------------------ rwkv6qwen2

    @Test
    public void qrwkvRepeatsKvHeadsSubtractsDecayedKeysAndRunsGla() {
        LayerSpec layer = qrwkv();
        mixer.bind(context, layer);
        int t = 2;
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        List<Dispatch> all = run(() -> mixer.prefill(0, t, in, out));

        GpuBuffer kRaw = temp("rwkv6.k_raw");
        GpuBuffer k = temp("rwkv6.k");
        int repeats = 0;
        for (Dispatch d : all) {
            if (GpuOps.COPY2D.equals(d.kernelName()) && d.bufferArg(d.argCount() - 2) == kRaw) {
                assertSame(k, d.bufferArg(d.argCount() - 1));
                assertArrayEquals(new int[]{t, S, S, 0, H * S, repeats * S}, ints(d, 0, 6));
                repeats++;
            }
        }
        assertEquals(H, repeats);

        List<Dispatch> ds = blockDispatches(all);
        assertEquals(Arrays.asList(RwkvKernels.TOKEN_SHIFT_LERP, GpuOps.UNARY, RwkvKernels.TOKEN_SHIFT_LERP,
                RwkvBlocks.BATCH_BIAS_ADD, RwkvBlocks.BATCH_BIAS_ADD, RwkvBlocks.BATCH_BIAS_ADD, GpuOps.UNARY,
                RwkvKernels.DECAY6, GpuOps.GLU, GpuOps.AXPBY, RwkvKernels.GLA, GpuOps.GLU), names(ds));
        assertSame(tensors.get(layer.tensorName(TensorNames.weight(TensorNames.TIME_MIX_LERP_FUSED))).shard(0),
                ds.get(2).bufferArg(9));
        assertArrayEquals(new int[]{C, t}, ints(ds.get(3), 0, 2));
        assertArrayEquals(new int[]{S, t}, ints(ds.get(4), 0, 2));
        assertSame(kRaw, ds.get(4).bufferArg(3));
        assertEquals(GluOp.MUL.code(), ds.get(8).intArg(2));

        Dispatch gla = ds.get(10);
        assertArrayEquals(new int[]{t, C, H, RwkvKernels.GLA_G_PER_CHANNEL}, ints(gla, 0, 4));
        assertEquals(RwkvKernels.glaScale(S), gla.floatArg(4), 0.0f);
        assertSame(k, gla.bufferArg(5));
        assertSame(temp("rwkv6.v"), gla.bufferArg(6));
        assertSame(temp("rwkv6.r"), gla.bufferArg(7));
        assertSame(temp("rwkv6.w"), gla.bufferArg(8));
        assertSame(stateTensor(LayerMemoryPlan.STATE_SCAN), gla.bufferArg(9));

        assertEquals(GluOp.SIGMUL.code(), ds.get(11).intArg(2));
    }

    // ------------------------------------------------------------------ contract

    @Test
    public void bindNeedsRecurrentMemoryAndStepsNeedABinding() {
        LayerSpec layer = layer(MixerKind.RWKV6, rwkv6Spec());
        open(Arch.RWKV6, layer, false);
        commonTensors(layer, C);
        weight(layer, TensorNames.TIME_MIX_LERP_FUSED, C, 1, 1, 5);
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        assertThrows(IllegalStateException.class, () -> mixer.decode(0, in, out));
        assertThrows(IllegalStateException.class, () -> mixer.bind(context, layer));
    }

    @Test
    public void stepRejectsTokenCountsOutsideTheUbatch() {
        LayerSpec layer = rwkv6();
        mixer.bind(context, layer);
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        assertThrows(IllegalArgumentException.class, () -> mixer.prefill(0, 0, in, out));
        assertThrows(IllegalArgumentException.class, () -> mixer.prefill(0, UBATCH + 1, in, out));
    }
}

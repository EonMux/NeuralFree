package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.Int8MatvecMode;
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
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the deepseek2 MLA mixer ({@link GpuMlaMixer}) with a latent cache on the host-only fake backend
 * with strict LLM argument rules: the decode and prefill sequences (low-rank Q, RoPE of the rotary dims, latent RMS
 * norm, per-head absorption by {@code {T}_MatVecId}, the {@code [q_abs | q_pe]} assembly, the latent row store, flash
 * attention with one buffer bound as K and V, the per-head {@code attn_v_b} expansion and {@code attn_output}), the
 * unrotated Q8_0 latent cache and the geometry's plan-only rotation, the {@code attn_k_norm} RMS norm of the shared
 * rotary key (dots3note), the per-head sinks of every flash kernel (hy_v4 without an indexer), and the decompressed
 * cache over absorbed weights (the per-token transposed {@code attn_k_b} read and {@code attn_v_b} expansion).
 */
public class GpuMlaMixerTest {

    private static final int N_EMBD = 128;
    private static final int N_HEAD = 2;
    private static final int Q_LORA = 64;
    private static final int NOPE = 32;
    private static final int HEAD_V = 32;
    private static final int N_CTX = 64;
    private static final int UBATCH = 8;
    private static final float KQ_SCALE = 0.125f;

    private FakeComputeBackend backend;
    private DeviceModelMemory memory;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuMlaMixer mixer;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private int kvLora;
    private int rope;
    private int width;
    /** The layer's attention facet, null for none. */
    private AttentionSpec attention;

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

    private void open(KvDType kvType, int kvLoraRank, int nRot) {
        open(kvType, kvLoraRank, nRot, true);
    }

    /** Opens the fixture with an absorbed model over a latent ({@code latent}) or decompressed cache. */
    private void open(KvDType kvType, int kvLoraRank, int nRot, boolean latent) {
        open(kvType, kvLoraRank, nRot, FakeComputeBackend.defaultCapabilities(), latent);
    }

    private void open(KvDType kvType, int kvLoraRank, int nRot, DeviceCapabilities caps,
                      String... unavailableKernels) {
        open(kvType, kvLoraRank, nRot, caps, true, unavailableKernels);
    }

    private void open(KvDType kvType, int kvLoraRank, int nRot, DeviceCapabilities caps, boolean latent,
                      String... unavailableKernels) {
        kvLora = kvLoraRank;
        rope = nRot;
        width = kvLora + rope;
        backend = FakeComputeBackend.builder().strictLlmArgs(true).capabilities(caps)
                .unavailableKernels(unavailableKernels).build();
        ModelSpec spec = deepseek2(latentMla(latent));
        LayerSpec layer = spec.layers().get(0);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(kvType).typeV(kvType).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        putTensors();
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        mixer = (GpuMlaMixer) MlaBlocks.gpuMixers().get(0).create(layer);
        mixer.bind(ctx, layer);
    }

    private MlaSpec latentMla(boolean latent) {
        return MlaSpec.builder().nHead(N_HEAD).qLoraRank(Q_LORA).kvLoraRank(kvLora).headKNope(NOPE).headKRope(rope)
                .headV(HEAD_V).absorbed(true).latentCache(latent).ropeApplied(true).kqScale(KQ_SCALE).build();
    }

    private ModelSpec deepseek2(MlaSpec mla) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK2)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f).headKFull(width).headVFull(kvLora);
        hb.nHead(0, N_HEAD).nHeadKv(0, 1);
        QkNormKind qk = attention != null ? attention.qkNorm() : QkNormKind.NONE;
        LayerSpec l = LayerSpec.builder(0, MixerKind.MLA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(qk).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .mla(mla)
                .attention(attention)
                .rope(RopeSpec.builder(RopeType.NORM).nDims(rope).freqBase(10000f).build())
                .build();
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.DEEPSEEK2.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(l)
                .build();
    }

    private void putTensors() {
        put(TensorNames.ATTN_Q_A, N_EMBD, Q_LORA);
        put(TensorNames.ATTN_Q_A_NORM, Q_LORA);
        put(TensorNames.ATTN_Q_B, Q_LORA, N_HEAD * (NOPE + rope));
        put(TensorNames.ATTN_KV_A_MQA, N_EMBD, width);
        put(TensorNames.ATTN_KV_A_NORM, kvLora);
        put(TensorNames.ATTN_K_B, NOPE, kvLora, N_HEAD);
        put(TensorNames.ATTN_V_B, kvLora, HEAD_V, N_HEAD);
        put(TensorNames.ATTN_OUT, N_HEAD * HEAD_V, N_EMBD);
        if (attention != null && attention.qkNorm() == QkNormKind.RMS_HEAD) {
            put(TensorNames.ATTN_K_NORM, rope);
        }
        if (attention != null && attention.sinks()) {
            put(TensorNames.ATTN_SINKS, N_HEAD);
        }
    }

    /** The MLA attention facet of the fixture layer with the given k_pe norm and sinks. */
    private static AttentionSpec facet(int nRot, boolean kPeNorm, boolean sinks) {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(NOPE + nRot).headDimV(HEAD_V)
                .nRot(nRot).qkNorm(kPeNorm ? QkNormKind.RMS_HEAD : QkNormKind.NONE).sinks(sinks).kqScale(KQ_SCALE)
                .kvOwnerLayer(0).build();
    }

    private void put(String base, long... ne) {
        put(base, GGMLType.F32, ne);
    }

    /** The 2-D {@code [cols, rows]} tensor {@code base} of layer 0 on {@code type}. */
    private void put(String base, GGMLType type, long... ne) {
        String name = TensorNames.layerName(0, TensorNames.weight(base));
        long bytes;
        if (type == GGMLType.F32) {
            long n = 1;
            for (long d : ne) {
                n *= d;
            }
            bytes = n * Float.BYTES;
        } else {
            bytes = (long) (ne[0] / type.getBlockSize()) * type.getTypeSize() * ne[1];
        }
        tensors.put(name, GpuWeight.of(name, type, ne, backend.allocateBuffer(bytes, true)));
    }

    private GpuBuffer weight(String base) {
        return tensors.get(TensorNames.layerName(0, TensorNames.weight(base))).shard(0);
    }

    private ScratchSlot temp(String name, long floats) {
        return scratch.temp(name, floats);
    }

    private DeviceStepInputs steps() {
        return (DeviceStepInputs) memory.stepInputs();
    }

    private List<Dispatch> step(int pos0, int n) {
        steps().upload(pos0, n);
        backend.clearLog();
        ScratchSlot in = scratch.slot(ScratchSlots.NORMED);
        ScratchSlot out = scratch.slot(ScratchSlots.BRANCH);
        if (n == 1) {
            mixer.decode(pos0, in, out);
        } else {
            mixer.prefill(pos0, n, in, out);
        }
        return backend.dispatches();
    }

    /** The cache writes of {@code n} rows: {@link GpuMlaMixer#prefillKV} over the same step inputs. */
    private List<Dispatch> kvOnlyStep(int pos0, int n) {
        steps().upload(pos0, n);
        backend.clearLog();
        mixer.prefillKV(pos0, n, scratch.slot(ScratchSlots.NORMED));
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

    private static int firstIndex(List<Dispatch> ds, String kernel, int from) {
        for (int i = from; i < ds.size(); i++) {
            if (ds.get(i).kernelName().equals(kernel)) {
                return i;
            }
        }
        throw new AssertionError(kernel + " is not in the trace after " + from);
    }

    /** The MLA kernels of one step, in order: everything but the dense projections. */
    private static List<String> mlaKernels(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            String k = d.kernelName();
            if (k.equals(NormKernels.ROW_NORM) || k.equals(RopeKernels.ROPE_EXT) || k.contains("_MatVecId")
                    || k.equals(GpuOps.COPY2D) || k.startsWith("LLM_KV_Store_") || k.startsWith("LLM_Flash")
                    || k.equals(HadamardSpec.KERNEL)) {
                out.add(k);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ decode

    @Test
    public void decodeRunsTheAbsorbedLatentSequence() {
        open(KvDType.F16, 64, 32);
        int pos = 0;
        List<Dispatch> ds = step(pos, 1);
        String id = MoeKernels.matVecId(GGMLType.F32);
        List<String> expected = new ArrayList<>();
        Collections.addAll(expected, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT,
                NormKernels.ROW_NORM, id, GpuOps.COPY2D, GpuOps.COPY2D, RopeKernels.kvStore(KvDType.F16),
                AttnKernels.flashDecode(KvDType.F16), id);
        assertEquals(expected, mlaKernels(ds));

        GpuKvLayer kv = memory.kv(0);
        ScratchSlot q = temp(GpuMlaMixer.TEMP_Q, (long) UBATCH * N_HEAD * (NOPE + rope));
        ScratchSlot kva = temp(GpuMlaMixer.TEMP_KV_A, (long) UBATCH * width);
        ScratchSlot lat = temp(GpuMlaMixer.TEMP_LATENT, (long) UBATCH * N_HEAD * kvLora);
        ScratchSlot qlat = temp(GpuMlaMixer.TEMP_Q_LATENT, (long) UBATCH * N_HEAD * width);
        ScratchSlot attn = temp(GpuMlaMixer.TEMP_ATTN, (long) UBATCH * N_HEAD * HEAD_V);

        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        Dispatch kvNorm = norms.get(1);
        assertEquals(1, kvNorm.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(kvLora, kvNorm.intArg(1));
        assertEquals(width, kvNorm.intArg(2));
        assertSame(kva.buffer(), kvNorm.bufferArg(9));
        assertSame(kva.buffer(), kvNorm.bufferArg(12));
        assertSame(weight(TensorNames.ATTN_KV_A_NORM), kvNorm.bufferArg(10));

        List<Dispatch> ropes = named(ds, RopeKernels.ROPE_EXT);
        assertEquals(1, ropes.get(0).intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(1, ropes.get(0).intArg(RopeKernels.ARG_POS_STRIDE));
        assertSame(q.buffer(), ropes.get(0).bufferArg(ropes.get(0).argCount() - 1));
        assertSame(kva.buffer(), ropes.get(1).bufferArg(ropes.get(1).argCount() - 1));

        List<Dispatch> ids = named(ds, id);
        Dispatch absorb = ids.get(0);
        assertEquals(kvLora, absorb.intArg(0));
        assertEquals(NOPE, absorb.intArg(1));
        assertEquals(N_HEAD, absorb.intArg(MoeKernels.ARG_ID_N_SEL));
        assertEquals(NOPE + rope, absorb.intArg(3));
        assertSame(weight(TensorNames.ATTN_K_B), absorb.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(q.buffer(), absorb.bufferArg(MoeKernels.ARG_ID_INPUT));
        assertSame(lat.buffer(), absorb.bufferArg(MoeKernels.ARG_ID_OUTPUT));
        Dispatch expand = ids.get(1);
        assertEquals(HEAD_V, expand.intArg(0));
        assertEquals(kvLora, expand.intArg(1));
        assertEquals(N_HEAD, expand.intArg(MoeKernels.ARG_ID_N_SEL));
        assertEquals(kvLora, expand.intArg(3));
        assertSame(weight(TensorNames.ATTN_V_B), expand.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(lat.buffer(), expand.bufferArg(MoeKernels.ARG_ID_INPUT));
        assertSame(attn.buffer(), expand.bufferArg(MoeKernels.ARG_ID_OUTPUT));
        assertSame(absorb.bufferArg(MoeKernels.ARG_ID_SLOTS), expand.bufferArg(MoeKernels.ARG_ID_SLOTS));

        Dispatch store = named(ds, RopeKernels.kvStore(KvDType.F16)).get(0);
        assertEquals(1, store.intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(width, store.intArg(1));
        assertSame(kva.buffer(), store.bufferArg(5));
        assertSame(steps().cellIdx(kv.group()), store.bufferArg(6));
        assertSame(kv.k(), store.bufferArg(7));

        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertEquals(N_HEAD, fd.intArg(AttnKernels.ARG_FD_N_HEAD));
        assertEquals(1, fd.intArg(AttnKernels.ARG_FD_N_HEAD_KV));
        assertEquals(width, fd.intArg(AttnKernels.ARG_FD_D_K));
        assertEquals(kvLora, fd.intArg(AttnKernels.ARG_FD_D_V));
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_KV_START));
        assertEquals(pos + 1, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(pos, fd.intArg(AttnKernels.ARG_FD_Q_POS));
        assertEquals(width, fd.intArg(AttnKernels.ARG_FD_K_ROW_STRIDE));
        assertEquals(width, fd.intArg(AttnKernels.ARG_FD_V_ROW_STRIDE));
        assertEquals(KQ_SCALE, fd.floatArg(AttnKernels.ARG_FD_SCALE), 0.0f);
        assertEquals("a latent V reads the K rows",
                AttnKernels.FLAG_CAUSAL | AttnKernels.FLAG_DIRECT | AttnKernels.FLAG_V_ALIAS,
                fd.intArg(AttnKernels.ARG_FD_FLAGS));
        assertSame(qlat.buffer(), fd.bufferArg(AttnKernels.ARG_FD_BUF_Q));
        assertSame(kv.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(kv.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertSame(lat.buffer(), fd.bufferArg(AttnKernels.ARG_FD_BUF_OUT));

        Dispatch last = ds.get(ds.size() - 1);
        assertTrue(firstIndex(ds, AttnKernels.flashDecode(KvDType.F16), 0) < ds.size() - 1);
        assertSame(scratch.slot(ScratchSlots.BRANCH).buffer(), last.bufferArg(last.argCount() - 1));
    }

    @Test
    public void decodeAtLaterPositionWidensTheKeyRange() {
        open(KvDType.F16, 64, 32);
        step(0, 3);
        List<Dispatch> ds = step(3, 1);
        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertEquals(0, fd.intArg(AttnKernels.ARG_FD_KV_START));
        assertEquals(4, fd.intArg(AttnKernels.ARG_FD_KV_END));
        assertEquals(3, fd.intArg(AttnKernels.ARG_FD_Q_POS));
    }

    @Test
    public void decodeOnTheInt8LaneQuantizesTheSharedInputRowOnce() {
        openLane(KvDType.F16, 64, 32);
        List<Dispatch> ds = step(0, 1);
        List<String> names = kernelNames(ds);
        // The shared stream row backs the query-A and latent projections: one quantize serves both
        // single-token matvecs; the query-B projection reads the LoRA row off the lane (an F32 weight).
        assertEquals(names.toString(), 1, Collections.frequency(names, GpuOps.QUANTIZE_Q8_1));
        assertEquals(names.toString(), 2, Collections.frequency(names, "Q4_0_MatVec_Q8"));
        int shared = names.indexOf(GpuOps.QUANTIZE_Q8_1);
        int queryA = firstIndex(ds, "Q4_0_MatVec_Q8", 0);
        int latent = firstIndex(ds, "Q4_0_MatVec_Q8", queryA + 1);
        assertTrue(names.toString(), shared < queryA && shared < latent);
        // A two-row step takes the multi-column integer-dot lane: each stream projection quantizes its
        // two rows and runs the Q4_0 MatVecN kernel; the single-row share does not apply.
        List<String> prefill = kernelNames(step(1, 2));
        assertEquals(prefill.toString(), 2, Collections.frequency(prefill, GpuOps.QUANTIZE_Q8_1));
        assertEquals(prefill.toString(), 2, Collections.frequency(prefill, "Q4_0_MatVecN_Q8"));
        assertEquals(prefill.toString(), 0, Collections.frequency(prefill, "Q4_0_BatchMatMul_2D"));
    }

    /** Opens the fixture with the stream projections on Q4_0 and the int8 matvec lane forced on. */
    private void openLane(KvDType kvType, int kvLoraRank, int nRot) {
        kvLora = kvLoraRank;
        rope = nRot;
        width = kvLora + rope;
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        ModelSpec spec = deepseek2(latentMla(true));
        LayerSpec layer = spec.layers().get(0);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(kvType).typeV(kvType).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        memory = DeviceModelMemory.allocate(backend, backend.computeQueue(), plan, MropeTextPositions.PPP0, null);
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        kernels = LlmKernels.open(backend);
        ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders, Int8MatvecMode.ON);
        putTensors();
        put(TensorNames.ATTN_Q_A, GGMLType.Q4_0, N_EMBD, Q_LORA);
        put(TensorNames.ATTN_KV_A_MQA, GGMLType.Q4_0, N_EMBD, width);
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, tensors::get, spec).memory(memory).build();
        mixer = (GpuMlaMixer) MlaBlocks.gpuMixers().get(0).create(layer);
        mixer.bind(ctx, layer);
    }

    private static List<String> kernelNames(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    // ------------------------------------------------------------------ prefill

    @Test
    public void prefillAttendsAllRowsOverTheLatentCache() {
        open(KvDType.F16, 64, 32);
        int n = 5;
        List<Dispatch> ds = step(0, n);
        GpuKvLayer kv = memory.kv(0);
        ScratchSlot qlat = temp(GpuMlaMixer.TEMP_Q_LATENT, (long) UBATCH * N_HEAD * width);
        ScratchSlot lat = temp(GpuMlaMixer.TEMP_LATENT, (long) UBATCH * N_HEAD * kvLora);
        ScratchSlot q = temp(GpuMlaMixer.TEMP_Q, (long) UBATCH * N_HEAD * (NOPE + rope));
        ScratchSlot attn = temp(GpuMlaMixer.TEMP_ATTN, (long) UBATCH * N_HEAD * HEAD_V);

        assertTrue(named(ds, MoeKernels.matVecId(GGMLType.F32)).isEmpty());
        Dispatch bucket = named(ds, MoeKernels.BUCKETIZE).get(0);
        assertEquals(Arrays.asList(n, N_HEAD, N_HEAD), Arrays.asList(bucket.intArg(0), bucket.intArg(1),
                bucket.intArg(2)));
        Dispatch align = named(ds, MoeKernels.ALIGN_BLOCKS).get(0);
        int tiles = MoeKernels.groupedTiles(UBATCH * N_HEAD, N_HEAD);
        assertEquals(n, align.intArg(MoeKernels.ARG_N_TOKENS));
        assertEquals(tiles, align.intArg(3));
        List<Dispatch> ids = named(ds, MoeKernels.groupedGemm(GGMLType.F32));
        assertEquals(2, ids.size());
        Dispatch absorb = ids.get(0);
        Dispatch expandGg = ids.get(1);
        assertEquals(Arrays.asList(kvLora, NOPE, tiles, 1, NOPE + rope, q.offset(), lat.offset()),
                ggInts(absorb));
        assertEquals(Arrays.asList(HEAD_V, kvLora, tiles, 1, kvLora, lat.offset(), attn.offset()), ggInts(expandGg));
        for (Dispatch d : ids) {
            assertSame(align.bufferArg(8), d.bufferArg(MoeKernels.ARG_GG_GEMM_TOKEN));
            assertSame(align.bufferArg(9), d.bufferArg(MoeKernels.ARG_GG_GEMM_SLOT));
            assertEquals(0, d.intArg(MoeKernels.ARG_GG_EXPERT_BASE));
            assertEquals(N_HEAD, d.intArg(MoeKernels.ARG_GG_EXPERT_COUNT));
        }
        assertSame(q.buffer(), absorb.bufferArg(MoeKernels.ARG_GG_INPUT));
        assertSame(lat.buffer(), absorb.bufferArg(MoeKernels.ARG_GG_OUTPUT));
        assertSame(lat.buffer(), expandGg.bufferArg(MoeKernels.ARG_GG_INPUT));
        assertSame(attn.buffer(), expandGg.bufferArg(MoeKernels.ARG_GG_OUTPUT));
        assertTrue(ds.indexOf(bucket) < ds.indexOf(align) && ds.indexOf(align) < ds.indexOf(absorb));
        assertEquals(n, named(ds, RopeKernels.kvStore(KvDType.F16)).get(0).intArg(RopeKernels.ARG_N_TOKENS));
        List<Dispatch> copies = named(ds, GpuOps.COPY2D);
        assertEquals(2, copies.size());
        for (Dispatch c : copies) {
            assertEquals(n * N_HEAD, c.intArg(0));
        }
        assertEquals(kvLora, copies.get(0).intArg(1));
        assertEquals(rope, copies.get(1).intArg(1));

        FlashAttnTiling t = FlashAttnTiling.prefill(N_HEAD, 1, width, kvLora, kv.layout().rows(),
                kernels.faSmemFloats(), Math.max(1, kernels.caps().getComputeUnits()));
        if (t.plan() == FlashAttnTiling.Plan.TILED) {
            Dispatch fa = named(ds, AttnKernels.flashAttn(KvDType.F16)).get(0);
            assertEquals(n, fa.intArg(AttnKernels.ARG_FA_N_TOKENS));
            assertEquals(1, fa.intArg(AttnKernels.ARG_FA_N_HEAD_KV));
            assertEquals(width, fa.intArg(AttnKernels.ARG_FA_D_K));
            assertEquals(kvLora, fa.intArg(AttnKernels.ARG_FA_D_V));
            assertEquals(0, fa.intArg(AttnKernels.ARG_FA_KV_START));
            assertEquals(n, fa.intArg(AttnKernels.ARG_FA_KV_END));
            assertEquals(N_HEAD * width, fa.intArg(AttnKernels.ARG_FA_Q_ROW_STRIDE));
            assertEquals(N_HEAD * kvLora, fa.intArg(AttnKernels.ARG_FA_OUT_ROW_STRIDE));
            assertSame(qlat.buffer(), fa.bufferArg(AttnKernels.ARG_FA_BUF_Q));
            assertSame(kv.k(), fa.bufferArg(AttnKernels.ARG_FA_BUF_K));
            assertSame(kv.k(), fa.bufferArg(AttnKernels.ARG_FA_BUF_V));
            assertSame(steps().qPos(), fa.bufferArg(AttnKernels.ARG_FA_BUF_Q_POS));
            assertSame(lat.buffer(), fa.bufferArg(AttnKernels.ARG_FA_BUF_OUT));
            int combines = named(ds, AttnKernels.FLASH_COMBINE).size();
            assertEquals((fa.intArg(AttnKernels.ARG_FA_FLAGS) & AttnKernels.FLAG_DIRECT) != 0 ? 0 : 1,
                    combines);
        } else {
            List<Dispatch> rows = named(ds, AttnKernels.flashDecode(KvDType.F16));
            assertEquals(n, rows.size());
            for (int i = 0; i < n; i++) {
                Dispatch d = rows.get(i);
                assertEquals(i, d.intArg(AttnKernels.ARG_FD_Q_POS));
                assertEquals(i + 1, d.intArg(AttnKernels.ARG_FD_KV_END));
                assertEquals(qlat.offset() + i * N_HEAD * width, d.intArg(AttnKernels.ARG_FD_Q_OFF));
                assertEquals(lat.offset() + i * N_HEAD * kvLora, d.intArg(AttnKernels.ARG_FD_OUT_OFF));
            }
        }
        int store = firstIndex(ds, RopeKernels.kvStore(KvDType.F16), 0);
        int expand = ds.indexOf(ids.get(1));
        assertTrue(store < expand);

        List<Dispatch> decode = step(n, 1);
        List<Dispatch> perHead = named(decode, MoeKernels.matVecId(GGMLType.F32));
        assertEquals("one token keeps the per-head products", 2, perHead.size());
        assertTrue(named(decode, MoeKernels.BUCKETIZE).isEmpty());
        assertSame("the head sort reads the per-head slot table", perHead.get(0).bufferArg(MoeKernels.ARG_ID_SLOTS),
                bucket.bufferArg(3));
    }

    /**
     * The latent cache writes alone: the KV projection, its norm and the K RoPE, then the latent store. The query's
     * products, its RoPE, the absorption, the attention and the output projection do not run.
     */
    @Test
    public void kvOnlyPrefillStopsAtTheLatentStore() {
        open(KvDType.F16, 64, 32);
        List<Dispatch> whole = step(0, 5);
        List<Dispatch> kvOnly = kvOnlyStep(0, 5);

        assertEquals(Arrays.asList(RopeKernels.ROPE_EXT, NormKernels.ROW_NORM,
                RopeKernels.kvStore(KvDType.F16)), mlaKernels(kvOnly));
        assertTrue("the cache write runs fewer kernels: " + kvOnly.size() + " vs " + whole.size(),
                kvOnly.size() < whole.size());
        assertEquals(0, named(kvOnly, MoeKernels.matVecId(GGMLType.F32)).size());
        assertEquals(0, named(kvOnly, GpuOps.COPY2D).size());
        assertEquals(0, named(kvOnly, AttnKernels.flashAttn(KvDType.F16)).size());
        assertEquals(0, named(kvOnly, AttnKernels.flashDecode(KvDType.F16)).size());
        assertEquals(1, named(kvOnly, RopeKernels.ROPE_EXT).size());
        assertEquals(1, named(kvOnly, NormKernels.ROW_NORM).size());
        Dispatch store = named(kvOnly, RopeKernels.kvStore(KvDType.F16)).get(0);
        assertEquals(5, store.intArg(RopeKernels.ARG_N_TOKENS));
        assertSame(memory.kv(0).k(), store.bufferArg(7));
    }

    /** The scalar arguments of a grouped GEMM before has_scale: out, in, tiles, x_div, x_stride, in_off, out_off. */
    private static List<Integer> ggInts(Dispatch d) {
        List<Integer> v = new ArrayList<>();
        for (int i = 0; i < MoeKernels.ARG_GG_HAS_SCALE; i++) {
            v.add(d.intArg(i));
        }
        return v;
    }

    /** Without {@code LLM_MoE_AlignBlocks} the per-head products stay per row and split under the grid limit. */
    @Test
    public void perHeadProductsSplitOnTokenBoundariesUnderTheGridLimit() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("FakeDevice")
                .totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30)
                .computeUnits(32)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(1024)
                .supportsAsyncTransfer(true)
                .driverVersion("fake-1.0")
                .supportsFloat16(true)
                .supportsSubgroups(true)
                .maxLocalMemoryBytes(48L * 1024)
                .maxWorkGroupCount(new long[]{Integer.MAX_VALUE, 5, 65535})
                .build();
        open(KvDType.F16, 64, 32, caps, MoeKernels.ALIGN_BLOCKS);
        int n = 5;
        List<Dispatch> ds = step(0, n);
        ScratchSlot q = temp(GpuMlaMixer.TEMP_Q, (long) UBATCH * N_HEAD * (NOPE + rope));
        ScratchSlot lat = temp(GpuMlaMixer.TEMP_LATENT, (long) UBATCH * N_HEAD * kvLora);
        ScratchSlot attn = temp(GpuMlaMixer.TEMP_ATTN, (long) UBATCH * N_HEAD * HEAD_V);
        List<Dispatch> ids = named(ds, MoeKernels.matVecId(GGMLType.F32));
        assertEquals(6, ids.size());
        int[] sel = {4, 4, 2};
        for (int c = 0; c < 3; c++) {
            int row0 = 2 * c * N_HEAD;
            Dispatch absorb = ids.get(c);
            assertEquals(sel[c], absorb.intArg(MoeKernels.ARG_ID_N_SEL));
            assertEquals(q.offset() + row0 * (NOPE + rope), absorb.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(lat.offset() + row0 * kvLora, absorb.intArg(MoeKernels.ARG_ID_OUT_OFF));
            Dispatch expand = ids.get(3 + c);
            assertEquals(sel[c], expand.intArg(MoeKernels.ARG_ID_N_SEL));
            assertEquals(lat.offset() + row0 * kvLora, expand.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(attn.offset() + row0 * HEAD_V, expand.intArg(MoeKernels.ARG_ID_OUT_OFF));
        }
    }

    // ------------------------------------------------------------------ rotation

    @Test
    public void q8LatentCacheIsStoredUnrotated() {
        open(KvDType.Q8_0, 128, 64);
        assertEquals(KvDType.Q8_0, memory.kv(0).layout().dtypeK());
        assertSame(RotationPlan.NONE, memory.plan().layer(0).rotation());
        List<Dispatch> ds = step(0, 1);
        assertTrue(named(ds, HadamardSpec.KERNEL).isEmpty());
        assertEquals(1, named(ds, RopeKernels.kvStore(KvDType.Q8_0)).size());
        assertEquals(1, named(ds, AttnKernels.flashDecode(KvDType.Q8_0)).size());
    }

    @Test
    public void geometryTakesTheRotationOnlyFromThePlan() {
        kvLora = 128;
        rope = 64;
        width = kvLora + rope;
        ModelSpec spec = deepseek2(latentMla(true));
        LayerSpec layer = spec.layers().get(0);
        MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
        com.eonmux.neuralfree.gguf.memory.KvLayout layout = plan.layer(0).kv();

        assertEquals(0, MlaBlocks.Geometry.of(layer, layout, plan.layer(0).rotation()).rotK);
        RotationPlan k64 = RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, width, 0, width, 0, true, false);
        MlaBlocks.Geometry g = MlaBlocks.Geometry.of(layer, layout, k64);
        assertEquals(64, g.rotK);
        assertEquals(0, g.rotV);
        assertEquals(64, g.shiftRotK());

        RotationPlan withV = RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, width, kvLora, width, kvLora, true, false);
        assertThrows(IllegalStateException.class, () -> MlaBlocks.Geometry.of(layer, layout, withV));
        RotationPlan wide = RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, 256, 0, 256, 0, true, false);
        assertThrows(IllegalStateException.class, () -> MlaBlocks.Geometry.of(layer, layout, wide));
    }

    // ------------------------------------------------------------------ k_pe norm and sinks

    @Test
    public void kPeNormRunsOnTheSharedKeyBeforeItsRope() {
        rope = 32;
        attention = facet(rope, true, false);
        open(KvDType.F16, 64, 32);
        List<Dispatch> ds = step(0, 3);
        ScratchSlot kva = temp(GpuMlaMixer.TEMP_KV_A, (long) UBATCH * width);
        List<Dispatch> norms = named(ds, NormKernels.ROW_NORM);
        assertEquals(3, norms.size());
        Dispatch kPe = norms.get(1);
        assertEquals(3, kPe.intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(rope, kPe.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(width, kPe.intArg(NormKernels.ARG_ROW_NORM_ROW_STRIDE));
        assertEquals(kva.offset() + kvLora, kPe.intArg(NormKernels.ARG_ROW_NORM_X_OFF));
        assertEquals(kva.offset() + kvLora, kPe.intArg(NormKernels.ARG_ROW_NORM_Y_OFF));
        assertSame(kva.buffer(), kPe.bufferArg(NormKernels.ARG_ROW_NORM_X));
        assertSame(weight(TensorNames.ATTN_K_NORM), kPe.bufferArg(NormKernels.ARG_ROW_NORM_W));
        List<Dispatch> ropes = named(ds, RopeKernels.ROPE_EXT);
        assertTrue(ds.indexOf(kPe) < ds.indexOf(ropes.get(1)));
        assertEquals(kvLora, norms.get(2).intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
    }

    @Test
    public void layersWithoutAKeyNormRunNone() {
        attention = facet(32, false, false);
        open(KvDType.F16, 64, 32);
        List<Dispatch> norms = named(step(0, 1), NormKernels.ROW_NORM);
        assertEquals(2, norms.size());
        for (Dispatch d : norms) {
            assertFalse(d.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN) == rope);
        }
    }

    @Test
    public void sinksReachEveryFlashKernel() {
        attention = facet(32, false, true);
        open(KvDType.F16, 64, 32);
        GpuBuffer sinks = weight(TensorNames.ATTN_SINKS);
        Dispatch fd = named(step(0, 1), AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertTrue((fd.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_SINKS) != 0);
        assertSame(sinks, fd.bufferArg(AttnKernels.ARG_FD_BUF_SINKS));

        List<Dispatch> ds = step(1, 5);
        List<Dispatch> fa = named(ds, AttnKernels.flashAttn(KvDType.F16));
        for (Dispatch d : fa) {
            assertTrue((d.intArg(AttnKernels.ARG_FA_FLAGS) & AttnKernels.FLAG_SINKS) != 0);
            assertSame(sinks, d.bufferArg(AttnKernels.ARG_FA_BUF_SINKS));
        }
        for (Dispatch d : named(ds, AttnKernels.flashDecode(KvDType.F16))) {
            assertTrue((d.intArg(AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_SINKS) != 0);
            assertSame(sinks, d.bufferArg(AttnKernels.ARG_FD_BUF_SINKS));
        }
        for (Dispatch d : named(ds, AttnKernels.FLASH_COMBINE)) {
            assertEquals(1, d.intArg(AttnKernels.ARG_FC_HAS_SINKS));
            assertSame(sinks, d.bufferArg(AttnKernels.ARG_FC_BUF_SINKS));
        }
        assertTrue(fa.size() + named(ds, AttnKernels.flashDecode(KvDType.F16)).size() > 0);
    }

    // ------------------------------------------------- decompressed cache over absorbed weights

    /**
     * The decompressed cache over absorbed weights, decode: the per-token transposed {@code attn_k_b} read into
     * k_nope and the {@code attn_v_b} expansion into V replace the {@code attn_kv_b} product, then the k_nope copy
     * and the rope broadcast per head build the per-head cache for flash attention.
     */
    @Test
    public void decodeRunsTheAbsorbedDecompressedSequence() {
        open(KvDType.F16, 64, 32, false);
        List<Dispatch> ds = step(0, 1);
        String id = MoeKernels.matVecId(GGMLType.F32);
        String idT = MoeKernels.matVecIdT(GGMLType.F32);
        List<String> expected = new ArrayList<>();
        Collections.addAll(expected, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT,
                NormKernels.ROW_NORM, idT, id, GpuOps.COPY2D, GpuOps.COPY2D, GpuOps.COPY2D,
                RopeKernels.kvStore(KvDType.F16), RopeKernels.kvStore(KvDType.F16),
                AttnKernels.flashDecode(KvDType.F16));
        assertEquals(expected, mlaKernels(ds));

        GpuKvLayer kv = memory.kv(0);
        ScratchSlot kva = temp(GpuMlaMixer.TEMP_KV_A, (long) UBATCH * width);
        ScratchSlot kvb = temp(GpuMlaMixer.TEMP_KV_B, (long) UBATCH * N_HEAD * NOPE);
        ScratchSlot k = temp(GpuMlaMixer.TEMP_K, (long) UBATCH * N_HEAD * (NOPE + rope));
        ScratchSlot v = temp(GpuMlaMixer.TEMP_V, (long) UBATCH * N_HEAD * HEAD_V);
        ScratchSlot attn = temp(GpuMlaMixer.TEMP_ATTN, (long) UBATCH * N_HEAD * HEAD_V);

        Dispatch kNope = named(ds, idT).get(0);
        assertEquals(NOPE, kNope.intArg(0));
        assertEquals(kvLora, kNope.intArg(1));
        assertEquals(N_HEAD, kNope.intArg(MoeKernels.ARG_ID_N_SEL));
        assertEquals(0, kNope.intArg(3));
        assertSame(weight(TensorNames.ATTN_K_B), kNope.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(kva.buffer(), kNope.bufferArg(MoeKernels.ARG_ID_INPUT));
        assertSame(kvb.buffer(), kNope.bufferArg(MoeKernels.ARG_ID_OUTPUT));
        Dispatch expand = named(ds, id).get(0);
        assertEquals(HEAD_V, expand.intArg(0));
        assertEquals(kvLora, expand.intArg(1));
        assertEquals(N_HEAD, expand.intArg(MoeKernels.ARG_ID_N_SEL));
        assertEquals(0, expand.intArg(3));
        assertSame(weight(TensorNames.ATTN_V_B), expand.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(kva.buffer(), expand.bufferArg(MoeKernels.ARG_ID_INPUT));
        assertSame(v.buffer(), expand.bufferArg(MoeKernels.ARG_ID_OUTPUT));
        assertSame(kNope.bufferArg(MoeKernels.ARG_ID_SLOTS), expand.bufferArg(MoeKernels.ARG_ID_SLOTS));

        List<Dispatch> copies = named(ds, GpuOps.COPY2D);
        Dispatch kNopeCopy = copies.get(0);
        assertEquals(N_HEAD, kNopeCopy.intArg(0));
        assertEquals(NOPE, kNopeCopy.intArg(1));
        assertEquals(NOPE, kNopeCopy.intArg(2));
        assertEquals(NOPE + rope, kNopeCopy.intArg(4));
        assertSame(kvb.buffer(), kNopeCopy.bufferArg(6));
        assertSame(k.buffer(), kNopeCopy.bufferArg(7));
        for (int h = 0; h < N_HEAD; h++) {
            Dispatch ropeCopy = copies.get(1 + h);
            assertEquals(1, ropeCopy.intArg(0));
            assertEquals(rope, ropeCopy.intArg(1));
            assertEquals(width, ropeCopy.intArg(2));
            assertEquals(kva.offset() + kvLora, ropeCopy.intArg(3));
            assertEquals(N_HEAD * (NOPE + rope), ropeCopy.intArg(4));
            assertEquals(h * (NOPE + rope) + NOPE, ropeCopy.intArg(5));
            assertSame(kva.buffer(), ropeCopy.bufferArg(6));
            assertSame(k.buffer(), ropeCopy.bufferArg(7));
        }

        Dispatch fd = named(ds, AttnKernels.flashDecode(KvDType.F16)).get(0);
        assertEquals(NOPE + rope, fd.intArg(AttnKernels.ARG_FD_D_K));
        assertEquals(HEAD_V, fd.intArg(AttnKernels.ARG_FD_D_V));
        assertSame(kv.k(), fd.bufferArg(AttnKernels.ARG_FD_BUF_K));
        assertSame(kv.v(), fd.bufferArg(AttnKernels.ARG_FD_BUF_V));
        assertSame(attn.buffer(), fd.bufferArg(AttnKernels.ARG_FD_BUF_OUT));
    }

    /**
     * The same cache, prefill: one transposed read and one expansion per token with that token's latent row as the
     * broadcast input and its per-head k_nope and V as the outputs, then the row copies and both cache stores.
     */
    @Test
    public void prefillRunsTheAbsorbedDecompressedProductsPerToken() {
        open(KvDType.F16, 64, 32, false);
        int n = 3;
        List<Dispatch> ds = step(0, n);
        String id = MoeKernels.matVecId(GGMLType.F32);
        String idT = MoeKernels.matVecIdT(GGMLType.F32);
        List<Dispatch> decomp = named(ds, idT);
        List<Dispatch> expand = named(ds, id);
        assertEquals(n, decomp.size());
        assertEquals(n, expand.size());
        ScratchSlot kva = temp(GpuMlaMixer.TEMP_KV_A, (long) UBATCH * width);
        ScratchSlot kvb = temp(GpuMlaMixer.TEMP_KV_B, (long) UBATCH * N_HEAD * NOPE);
        ScratchSlot v = temp(GpuMlaMixer.TEMP_V, (long) UBATCH * N_HEAD * HEAD_V);
        for (int t = 0; t < n; t++) {
            Dispatch d = decomp.get(t);
            assertEquals(NOPE, d.intArg(0));
            assertEquals(kvLora, d.intArg(1));
            assertEquals(N_HEAD, d.intArg(MoeKernels.ARG_ID_N_SEL));
            assertEquals(0, d.intArg(3));
            assertEquals(t * width, d.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(t * N_HEAD * NOPE, d.intArg(MoeKernels.ARG_ID_OUT_OFF));
            assertSame(kva.buffer(), d.bufferArg(MoeKernels.ARG_ID_INPUT));
            assertSame(kvb.buffer(), d.bufferArg(MoeKernels.ARG_ID_OUTPUT));
            Dispatch e = expand.get(t);
            assertEquals(t * width, e.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(t * N_HEAD * HEAD_V, e.intArg(MoeKernels.ARG_ID_OUT_OFF));
            assertSame(v.buffer(), e.bufferArg(MoeKernels.ARG_ID_OUTPUT));
            assertTrue(ds.indexOf(d) < ds.indexOf(e));
        }

        List<Dispatch> copies = named(ds, GpuOps.COPY2D);
        assertEquals(1 + N_HEAD, copies.size());
        assertEquals(n * N_HEAD, copies.get(0).intArg(0));
        assertEquals(NOPE, copies.get(0).intArg(1));
        for (int h = 1; h <= N_HEAD; h++) {
            assertEquals(n, copies.get(h).intArg(0));
            assertEquals(rope, copies.get(h).intArg(1));
        }
        assertEquals(2, named(ds, RopeKernels.kvStore(KvDType.F16)).size());
        GpuKvLayer kv = memory.kv(0);
        List<Dispatch> stores = named(ds, RopeKernels.kvStore(KvDType.F16));
        assertSame(kv.k(), stores.get(0).bufferArg(7));
        assertSame(kv.v(), stores.get(1).bufferArg(7));
        assertTrue(named(ds, AttnKernels.flashAttn(KvDType.F16)).size()
                + named(ds, AttnKernels.flashDecode(KvDType.F16)).size() > 0);
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void keyNormAfterRopeIsRefused() {
        kvLora = 64;
        rope = 32;
        width = kvLora + rope;
        attention = facet(rope, true, false).toBuilder().qkNormAfterRope(true).build();
        LayerSpec layer = deepseek2(latentMla(true)).layers().get(0);
        assertThrows(UnsupportedOperationException.class, () -> MlaBlocks.gpuMixers().get(0).create(layer));
    }

    @Test
    public void factoriesExposeOneMlaMixerAndNoFfnOrExtras() {
        assertEquals(1, MlaBlocks.gpuMixers().size());
        assertEquals(1, MlaBlocks.cpuMixers().size());
        assertEquals(MixerKind.MLA, MlaBlocks.gpuMixers().get(0).kind());
        assertEquals(MixerKind.MLA, MlaBlocks.cpuMixers().get(0).kind());
        assertTrue(MlaBlocks.gpuFfns().isEmpty());
        assertTrue(MlaBlocks.cpuFfns().isEmpty());
        assertTrue(MlaBlocks.gpuExtras().isEmpty());
        assertTrue(MlaBlocks.cpuExtras().isEmpty());
    }
}

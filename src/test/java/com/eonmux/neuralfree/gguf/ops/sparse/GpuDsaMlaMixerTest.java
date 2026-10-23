package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuCompressedKv;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuIndexerCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMsaCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseBlocks.DsaLayout;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of {@link GpuDsaMlaMixer} on the host-only fake backend with strict LLM argument rules, for small
 * deepseek32, glm-dsa, dots3note and hy_v4 layers: the decode and prefill sequences of a full layer with the
 * lightning indexer, the per-step {@code n_kv} / {@code k} re-sets while the cache holds fewer than top-k rows, the
 * top-k reuse of a shared glm-dsa layer, the dots3note per-head gate and {@code k_pe} norm, the hy_v4 sinks and
 * element-wise gate, and the binding errors.
 */
public class GpuDsaMlaMixerTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 2;
    private static final int Q_LORA = 32;
    private static final int KV_LORA = 64;
    private static final int NOPE = 16;
    private static final int ROPE = 16;
    private static final int HEAD_K = NOPE + ROPE;
    private static final int HEAD_V = 16;
    private static final int D_K = KV_LORA + ROPE;
    private static final int IDX_HEADS = 2;
    private static final int IDX_DIM = 64;
    private static final int IDX_ROPE = 16;
    private static final int TOP_K = 4;
    private static final int ROWS = 16;
    private static final int UBATCH = 8;
    private static final float KQ_SCALE = 0.125f;
    private static final String MATVEC = "Float_MatVec";
    private static final String ID_MATVEC = MoeKernels.matVecId(GGMLType.F32);
    /** {@code weight} of {@code {T}_MatVecId} in signature order. */
    private static final int ID_ARG_WEIGHT = MoeKernels.ARG_ID_WEIGHT;
    /** {@code n_offs} of {@code LLM_RoPE_Ext} in signature order. */
    private static final int ROPE_ARG_N_OFFS = 5;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int local;
    private FakeMemory memory;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private final List<GpuMixer> mixers = new ArrayList<>();

    private void open() {
        open(FakeComputeBackend.defaultCapabilities());
    }

    private void open(DeviceCapabilities caps) {
        backend = FakeComputeBackend.builder().capabilities(caps).strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        local = ops.local();
        memory = new FakeMemory();
    }

    @After
    public void tearDown() {
        for (GpuMixer m : mixers) {
            m.close();
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

    private GpuWeight put(int il, String suffix, long... ne) {
        String name = TensorNames.layerName(il, suffix);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        GpuWeight w = GpuWeight.of(name, GGMLType.F32, ne, backend.allocateBuffer(n * Float.BYTES, true));
        tensors.put(name, w);
        return w;
    }

    /** The MLA tensors of a layer, plus the indexer tensors when {@code full}. */
    private void mlaTensors(int il, boolean full) {
        put(il, DsaLayout.WQ_A, N_EMBD, Q_LORA);
        put(il, DsaLayout.Q_A_NORM, Q_LORA);
        put(il, DsaLayout.WQ_B, Q_LORA, (long) N_HEAD * HEAD_K);
        put(il, DsaLayout.WKV_A, N_EMBD, D_K);
        put(il, DsaLayout.KV_A_NORM, KV_LORA);
        put(il, DsaLayout.WK_B, NOPE, KV_LORA, N_HEAD);
        put(il, DsaLayout.WV_B, KV_LORA, HEAD_V, N_HEAD);
        put(il, DsaLayout.WO, (long) N_HEAD * HEAD_V, N_EMBD);
        if (full) {
            put(il, DsaLayout.IDX_Q_B, Q_LORA, (long) IDX_HEADS * IDX_DIM);
            put(il, DsaLayout.IDX_PROJ, N_EMBD, IDX_HEADS);
            put(il, DsaLayout.IDX_K, N_EMBD, IDX_DIM);
            put(il, DsaLayout.IDX_K_NORM, IDX_DIM);
            put(il, DsaLayout.IDX_K_NORM_B, IDX_DIM);
        }
    }

    private static LayerSpec layer(int il, int source, RopeType idxRope, boolean ropeTail, boolean hadamard) {
        return layer(il, source, idxRope, ropeTail, hadamard, null);
    }

    /** As {@link #layer(int, int, RopeType, boolean, boolean)} with an attention facet (null for none). */
    private static LayerSpec layer(int il, int source, RopeType idxRope, boolean ropeTail, boolean hadamard,
                                   AttentionSpec attention) {
        DsaSpec dsa = DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX_DIM).indexerRopeDim(IDX_ROPE)
                .ropeType(idxRope).ropeTail(ropeTail).topK(TOP_K).full(source == il).sourceLayer(source)
                .hadamard(hadamard).defaultWeightScale().build();
        MlaSpec mla = MlaSpec.builder().nHead(N_HEAD).qLoraRank(Q_LORA).kvLoraRank(KV_LORA).headKNope(NOPE)
                .headKRope(ROPE).headV(HEAD_V).absorbed(true).kqScale(KQ_SCALE).build();
        QkNormKind qk = attention != null ? attention.qkNorm() : QkNormKind.NONE;
        return LayerSpec.builder(il, MixerKind.DSA_MLA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).qkNorm(qk).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .mla(mla)
                .dsa(dsa)
                .attention(attention)
                .rope(RopeSpec.builder(RopeType.NORM).nDims(ROPE).freqBase(10000f).build())
                .build();
    }

    private static LayerSpec deepseek32Layer(int il) {
        return layer(il, il, RopeType.NEOX, false, true);
    }

    /**
     * A full dots3note DSA layer: its attention facet records the {@code attn_k_norm} RMS norm of {@code k_pe} before
     * RoPE, as the spec resolver does for every dots3note layer (dots3note.cpp:90-91, 330-332).
     */
    private static LayerSpec dots3noteLayer(int il) {
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_K).headDimV(HEAD_V)
                .nRot(ROPE).qkNorm(QkNormKind.RMS_HEAD).kqScale(KQ_SCALE).kvOwnerLayer(il).build();
        return layer(il, il, RopeType.NEOX, false, true, a);
    }

    /** A latent cache of group 0 and, for a full layer, its F16 indexer cache with the forced rotation. */
    private void caches(int il, boolean full, boolean rotate) {
        caches(il, full, rotate, KvDType.F16);
    }

    /** As {@link #caches(int, boolean, boolean)} with an indexer cache of {@code indexerType}. */
    private void caches(int il, boolean full, boolean rotate, KvDType indexerType) {
        memory.kv.put(il, new FakeKv(il));
        if (full) {
            RotationPlan rot = rotate
                    ? RotationPlan.of(indexerType, indexerType, IDX_DIM, 0, IDX_DIM, 0, false, true)
                    : RotationPlan.NONE;
            memory.indexer.put(il, new FakeIndexer(il, rot, indexerType));
        }
    }

    private GpuForwardContext context(Arch arch, List<LayerSpec> layers) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(layers.size(), 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f).normEps(1e-6f).build();
        ModelSpec.Builder b = ModelSpec.builder(hp, TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(layers.size()))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build());
        for (LayerSpec l : layers) {
            b.layer(l);
        }
        GpuWeights weights = new GpuWeights() {
            @Override
            public GpuWeight find(String name) {
                return tensors.get(name);
            }
        };
        return GpuForwardContext.builder(ops.kernels(), ops, weights, b.build()).memory(memory).build();
    }

    private List<GpuMixer> bindAll(Arch arch, LayerSpec... layers) {
        GpuForwardContext ctx = context(arch, Arrays.asList(layers));
        List<GpuMixer> out = new ArrayList<>();
        for (LayerSpec l : layers) {
            GpuMixer m = SparseBlocks.gpuMixers().get(0).create(l);
            mixers.add(m);
            m.bind(ctx, l);
            out.add(m);
        }
        return out;
    }

    private List<Dispatch> decode(GpuMixer m, int pos) {
        memory.steps.upload(pos, 1);
        backend.clearLog();
        m.decode(pos, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private List<Dispatch> prefill(GpuMixer m, int pos0, int n) {
        memory.steps.upload(pos0, n);
        backend.clearLog();
        m.prefill(pos0, n, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH));
        return backend.dispatches();
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static Dispatch only(List<Dispatch> ds, String kernel) {
        Dispatch found = null;
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                assertEquals("one " + kernel, null, found);
                found = d;
            }
        }
        if (found == null) {
            throw new AssertionError("no " + kernel + " in " + names(ds));
        }
        return found;
    }

    private static List<Dispatch> all(List<Dispatch> ds, String kernel) {
        List<Dispatch> out = new ArrayList<>();
        for (Dispatch d : ds) {
            if (d.kernelName().equals(kernel)) {
                out.add(d);
            }
        }
        return out;
    }

    private GpuWeight weight(int il, String suffix) {
        return tensors.get(TensorNames.layerName(il, suffix));
    }

    private int sparseLocal() {
        return Math.min(SparseKernels.SPARSE_MLA_LOCAL, local);
    }

    // ------------------------------------------------------------------ deepseek32

    @Test
    public void deepseek32DecodeRunsIndexerThenSparseMla() {
        open();
        mlaTensors(0, true);
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.DEEPSEEK32, deepseek32Layer(0)).get(0);
        List<Dispatch> ds = decode(m, 5);

        String hadamard = HadamardSpec.KERNEL;
        assertEquals(Arrays.asList(
                MATVEC, NormKernels.ROW_NORM,
                MATVEC, RopeKernels.ROPE_EXT, hadamard, MATVEC, GpuOps.AXPBY,
                MATVEC, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT, hadamard, RopeKernels.kvStore(KvDType.F16),
                SparseKernels.indexerScore(KvDType.F16), SparseKernels.TOPK_ROWS,
                MATVEC, MATVEC, RopeKernels.ROPE_EXT, RopeKernels.ROPE_EXT, NormKernels.ROW_NORM,
                RopeKernels.kvStore(KvDType.F32),
                ID_MATVEC, GpuOps.COPY2D, GpuOps.COPY2D, SparseKernels.sparseDecode(KvDType.F32), ID_MATVEC,
                MATVEC), names(ds));

        FakeIndexer idx = memory.indexer.get(0);
        Dispatch score = only(ds, SparseKernels.indexerScore(KvDType.F16));
        assertEquals(1, score.intArg(SparseKernels.ARG_SCORE_N_TOKENS));
        assertEquals(6, score.intArg(SparseKernels.ARG_SCORE_N_KV));
        assertEquals(IDX_HEADS, score.intArg(2));
        assertEquals(IDX_DIM, score.intArg(3));
        assertEquals(1, score.intArg(4));
        assertSame(idx.keys, score.bufferArg(9));
        assertSame(memory.steps.qPos, score.bufferArg(10));
        assertSame(memory.steps.kvPos, score.bufferArg(11));

        Dispatch topK = only(ds, SparseKernels.TOPK_ROWS);
        assertArrayEquals(new int[]{1, 6, TOP_K}, new int[]{topK.intArg(0), topK.intArg(1), topK.intArg(2)});
        GpuBuffer selection = topK.bufferArg(4);

        Dispatch sparse = only(ds, SparseKernels.sparseDecode(KvDType.F32));
        assertEquals(N_HEAD, sparse.intArg(0));
        assertEquals(KV_LORA, sparse.intArg(1));
        assertEquals(ROPE, sparse.intArg(2));
        assertEquals(TOP_K, sparse.intArg(SparseKernels.ARG_DECODE_N_SEL));
        assertEquals(KQ_SCALE, sparse.floatArg(4), 0f);
        assertEquals(0, sparse.intArg(6));
        assertEquals(0, sparse.intArg(SparseKernels.ARG_DECODE_Q_OFF));
        assertSame(memory.kv.get(0).k, sparse.bufferArg(9));
        assertSame(selection, sparse.bufferArg(10));
        assertSame(placeholders.readOnlyFloats(), sparse.bufferArg(11));
        assertArrayEquals(SparseKernels.sparseDecodeGlobal(N_HEAD, sparseLocal()), sparse.global());

        Dispatch store = only(ds, RopeKernels.kvStore(KvDType.F32));
        assertEquals(1, store.intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(D_K, store.intArg(1));
        assertSame(memory.steps.cellIdx, store.arg(6));
        assertSame(memory.kv.get(0).k, store.arg(7));

        List<Dispatch> id = all(ds, ID_MATVEC);
        assertEquals(KV_LORA, id.get(0).intArg(0));
        assertEquals(NOPE, id.get(0).intArg(1));
        assertEquals(N_HEAD, id.get(0).intArg(MoeKernels.ARG_ID_N_SEL));
        assertSame(weight(0, DsaLayout.WK_B).shard(0), id.get(0).bufferArg(ID_ARG_WEIGHT));
        assertEquals(HEAD_V, id.get(1).intArg(0));
        assertEquals(KV_LORA, id.get(1).intArg(1));
        assertSame(weight(0, DsaLayout.WV_B).shard(0), id.get(1).bufferArg(ID_ARG_WEIGHT));

        Dispatch wo = ds.get(ds.size() - 1);
        assertSame(weight(0, DsaLayout.WO).shard(0), wo.bufferArg(2));
        assertSame(scratch.slot(ScratchSlots.BRANCH).buffer(), wo.bufferArg(4));
    }

    // The indexer cache has the context's type_k (llama-kv-cache-dsa.cpp:51): its key store and score kernels are
    // the variants of the indexer plan's type; the forced Hadamard rotation of q/k is unchanged.

    @Test
    public void deepseek32Q8IndexerStoresAndScoresQ8Rows() {
        checkIndexerKernelsOfType(KvDType.Q8_0);
    }

    @Test
    public void deepseek32F32IndexerStoresAndScoresF32Rows() {
        checkIndexerKernelsOfType(KvDType.F32);
    }

    private void checkIndexerKernelsOfType(KvDType type) {
        open();
        mlaTensors(0, true);
        caches(0, true, true, type);
        GpuMixer m = bindAll(Arch.DEEPSEEK32, deepseek32Layer(0)).get(0);
        List<Dispatch> ds = decode(m, 5);

        FakeIndexer idx = memory.indexer.get(0);
        Dispatch store = all(ds, RopeKernels.kvStore(type)).get(0);
        assertEquals(1, store.intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(IDX_DIM, store.intArg(1));
        assertEquals(IDX_DIM, store.intArg(4));
        assertSame(idx.keys, store.bufferArg(7));
        Dispatch score = only(ds, SparseKernels.indexerScore(type));
        assertEquals(IDX_DIM, score.intArg(3));
        assertSame(idx.keys, score.bufferArg(9));
        assertEquals(2, all(ds, HadamardSpec.KERNEL).size());
        for (KvDType other : KvDType.values()) {
            if (other != type) {
                assertFalse(names(ds).contains(SparseKernels.indexerScore(other)));
            }
        }
    }

    @Test
    public void deepseek32DecodeReSetsSelectionWidthWhileCacheIsShort() {
        open();
        mlaTensors(0, true);
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.DEEPSEEK32, deepseek32Layer(0)).get(0);

        List<Dispatch> first = decode(m, 0);
        assertEquals(1, only(first, SparseKernels.TOPK_ROWS).intArg(SparseKernels.ARG_TOPK_N_COLS));
        assertEquals(1, only(first, SparseKernels.TOPK_ROWS).intArg(SparseKernels.ARG_TOPK_K));
        assertEquals(1, only(first, SparseKernels.sparseDecode(KvDType.F32)).intArg(SparseKernels.ARG_DECODE_N_SEL));

        List<Dispatch> late = decode(m, ROWS + 3);
        assertEquals(ROWS, only(late, SparseKernels.indexerScore(KvDType.F16)).intArg(SparseKernels.ARG_SCORE_N_KV));
        assertEquals(TOP_K, only(late, SparseKernels.sparseDecode(KvDType.F32)).intArg(SparseKernels.ARG_DECODE_N_SEL));
    }

    @Test
    public void deepseek32PrefillReSetsTokenCountsAndUsesPrefillKernel() {
        open();
        mlaTensors(0, true);
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.DEEPSEEK32, deepseek32Layer(0)).get(0);
        int n = UBATCH;
        List<Dispatch> ds = prefill(m, 0, n);

        Dispatch score = only(ds, SparseKernels.indexerScore(KvDType.F16));
        assertEquals(n, score.intArg(SparseKernels.ARG_SCORE_N_TOKENS));
        assertEquals(n, score.intArg(SparseKernels.ARG_SCORE_N_KV));
        Dispatch topK = only(ds, SparseKernels.TOPK_ROWS);
        assertEquals(n, topK.intArg(SparseKernels.ARG_TOPK_N_ROWS));
        assertEquals(TOP_K, topK.intArg(SparseKernels.ARG_TOPK_K));

        Dispatch sparse = only(ds, SparseKernels.sparsePrefill(KvDType.F32));
        assertEquals(n, sparse.intArg(SparseKernels.ARG_PREFILL_N_TOKENS));
        assertEquals(TOP_K, sparse.intArg(SparseKernels.ARG_PREFILL_N_SEL));
        assertEquals(N_HEAD * D_K, sparse.intArg(8));
        assertSame(topK.bufferArg(4), sparse.bufferArg(11));
        assertArrayEquals(SparseKernels.sparsePrefillGlobal(n, N_HEAD, sparseLocal()), sparse.global());
        assertFalse(names(ds).contains(SparseKernels.sparseDecode(KvDType.F32)));

        for (Dispatch id : all(ds, ID_MATVEC)) {
            assertEquals(n * N_HEAD, id.intArg(MoeKernels.ARG_ID_N_SEL));
        }
        for (Dispatch rope : all(ds, RopeKernels.ROPE_EXT)) {
            assertEquals(n, rope.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(n, rope.intArg(RopeKernels.ARG_POS_STRIDE));
        }
        assertEquals(n, only(ds, RopeKernels.kvStore(KvDType.F16)).intArg(RopeKernels.ARG_N_TOKENS));
    }

    @Test
    public void prefillSplitsPerHeadProductsAtTheGridLimit() {
        open(new DeviceCapabilities.Builder()
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
                .build());
        mlaTensors(0, true);
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.DEEPSEEK32, deepseek32Layer(0)).get(0);

        List<Dispatch> ids = all(prefill(m, 0, 5), ID_MATVEC);

        int tokensPerCall = 5 / N_HEAD;
        int[] sel = {tokensPerCall * N_HEAD, tokensPerCall * N_HEAD, N_HEAD};
        assertEquals(2 * sel.length, ids.size());
        for (int c = 0; c < sel.length; c++) {
            int pair0 = c * tokensPerCall * N_HEAD;
            Dispatch absorb = ids.get(c);
            assertEquals(sel[c], absorb.intArg(MoeKernels.ARG_ID_N_SEL));
            assertEquals(pair0 * HEAD_K, absorb.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(pair0 * KV_LORA, absorb.intArg(MoeKernels.ARG_ID_OUT_OFF));
            assertEquals(sel[c], absorb.global()[1]);
            Dispatch expand = ids.get(sel.length + c);
            assertEquals(sel[c], expand.intArg(MoeKernels.ARG_ID_N_SEL));
            assertEquals(pair0 * KV_LORA, expand.intArg(MoeKernels.ARG_ID_IN_OFF));
            assertEquals(pair0 * HEAD_V, expand.intArg(MoeKernels.ARG_ID_OUT_OFF));
        }
    }

    // ------------------------------------------------------------------ glm-dsa

    @Test
    public void glmDsaSharedLayerReusesTheSourceSelection() {
        open();
        mlaTensors(0, true);
        mlaTensors(1, false);
        caches(0, true, true);
        caches(1, false, false);
        List<GpuMixer> ms = bindAll(Arch.GLM_DSA, layer(0, 0, RopeType.NORM, false, true),
                layer(1, 0, RopeType.NORM, false, true));

        List<Dispatch> full = decode(ms.get(0), 9);
        GpuBuffer selection = only(full, SparseKernels.TOPK_ROWS).bufferArg(4);
        List<Dispatch> shared = decode(ms.get(1), 9);

        for (KvDType type : KvDType.values()) {
            assertFalse(names(shared).contains(SparseKernels.indexerScore(type)));
        }
        assertFalse(names(shared).contains(SparseKernels.TOPK_ROWS));
        assertFalse(names(shared).contains(RopeKernels.kvStore(KvDType.F16)));
        Dispatch sparse = only(shared, SparseKernels.sparseDecode(KvDType.F32));
        assertSame(selection, sparse.bufferArg(10));
        assertEquals(TOP_K, sparse.intArg(SparseKernels.ARG_DECODE_N_SEL));
        assertSame(memory.kv.get(1).k, sparse.bufferArg(9));
    }

    @Test
    public void glmDsaSharedLayerRejectsSourceInAnotherGroup() {
        open();
        mlaTensors(0, true);
        mlaTensors(1, false);
        caches(0, true, true);
        memory.kv.put(1, new FakeKv(1, 1));
        LayerSpec l0 = layer(0, 0, RopeType.NORM, false, true);
        LayerSpec l1 = layer(1, 0, RopeType.NORM, false, true);
        GpuForwardContext ctx = context(Arch.GLM_DSA, Arrays.asList(l0, l1));
        GpuMixer m = SparseBlocks.gpuMixers().get(0).create(l1);
        mixers.add(m);
        assertThrows(IllegalStateException.class, () -> m.bind(ctx, l1));
    }

    @Test
    public void glmDsaSharedLayerRejectsSourceBehindAnotherFullLayer() {
        open();
        mlaTensors(0, true);
        mlaTensors(1, true);
        mlaTensors(2, false);
        caches(0, true, true);
        caches(1, true, true);
        caches(2, false, false);
        LayerSpec l0 = layer(0, 0, RopeType.NORM, false, true);
        LayerSpec l1 = layer(1, 1, RopeType.NORM, false, true);
        LayerSpec l2 = layer(2, 0, RopeType.NORM, false, true);
        GpuForwardContext ctx = context(Arch.GLM_DSA, Arrays.asList(l0, l1, l2));
        GpuMixer m = SparseBlocks.gpuMixers().get(0).create(l2);
        mixers.add(m);
        assertThrows(IllegalStateException.class, () -> m.bind(ctx, l2));
    }

    @Test
    public void glmDsaIndexerKeyNormWithoutWeightUsesBiasOnly() {
        open();
        mlaTensors(0, true);
        tensors.remove(TensorNames.layerName(0, DsaLayout.IDX_K_NORM));
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.GLM_DSA, layer(0, 0, RopeType.NORM, false, true)).get(0);

        List<Dispatch> ds = decode(m, 3);

        Dispatch keyNorm = null;
        for (Dispatch d : all(ds, NormKernels.ROW_NORM)) {
            if (d.intArg(NormKernels.ARG_ROW_NORM_MODE) == NormKind.LAYER.rowNormMode()) {
                assertEquals("one layer norm", null, keyNorm);
                keyNorm = d;
            }
        }
        if (keyNorm == null) {
            throw new AssertionError("no indexer key layer norm in " + names(ds));
        }
        assertEquals(NormKernels.FLAG_HAS_B, keyNorm.intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        assertSame(weight(0, DsaLayout.IDX_K_NORM_B).shard(0), keyNorm.bufferArg(NormKernels.ARG_ROW_NORM_B));
    }

    // ------------------------------------------------------------------ dots3note

    @Test
    public void dots3noteNormalizesKPeAndBroadcastsTheHeadGate() {
        open();
        mlaTensors(0, true);
        put(0, DsaLayout.K_PE_NORM, ROPE);
        GpuWeight gate = put(0, DsaLayout.GATE, N_EMBD, N_HEAD);
        caches(0, true, true);
        GpuMixer m = bindAll(Arch.DOTS3NOTE, dots3noteLayer(0)).get(0);
        List<Dispatch> ds = decode(m, 3);

        List<String> tail = names(ds).subList(ds.size() - 9, ds.size());
        assertEquals(Arrays.asList(ID_MATVEC, MATVEC, GpuOps.COPY2D, GpuOps.COPY2D, GpuOps.COPY2D, GpuOps.COPY2D,
                GpuOps.COPY2D, ActKernels.GLU, MATVEC), tail);
        assertSame(gate.shard(0), ds.get(ds.size() - 8).bufferArg(2));
        Dispatch glu = only(ds, ActKernels.GLU);
        assertEquals(GluOp.SIGMUL.code(), glu.intArg(ActKernels.ARG_GLU_OP));
        assertEquals(N_HEAD, glu.intArg(ActKernels.ARG_GLU_N_ROWS));
        assertEquals(HEAD_V, glu.intArg(ActKernels.ARG_GLU_N_COLS));

        List<Dispatch> norms = all(ds, NormKernels.ROW_NORM);
        assertEquals(4, norms.size());
        Dispatch kPe = norms.get(2);
        assertEquals(ROPE, kPe.intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(D_K, kPe.intArg(NormKernels.ARG_ROW_NORM_ROW_STRIDE));
        assertEquals(KV_LORA, kPe.intArg(NormKernels.ARG_ROW_NORM_X_OFF));
        assertSame(weight(0, DsaLayout.K_PE_NORM).shard(0), kPe.bufferArg(NormKernels.ARG_ROW_NORM_W));
    }

    // ------------------------------------------------------------------ hy_v4

    @Test
    public void hyV4AppliesSinksAndElementwiseGateWithoutRotation() {
        open();
        mlaTensors(0, true);
        GpuWeight sinks = put(0, DsaLayout.SINKS, N_HEAD);
        put(0, DsaLayout.GATE, N_EMBD, (long) N_HEAD * HEAD_V);
        caches(0, true, false);
        GpuMixer m = bindAll(Arch.HY_V4, layer(0, 0, RopeType.NORM, true, false)).get(0);
        List<Dispatch> ds = decode(m, 2);

        assertFalse(names(ds).contains(HadamardSpec.KERNEL));
        Dispatch sparse = only(ds, SparseKernels.sparseDecode(KvDType.F32));
        assertEquals(1, sparse.intArg(6));
        assertSame(sinks.shard(0), sparse.bufferArg(11));
        Dispatch glu = only(ds, ActKernels.GLU);
        assertEquals(1, glu.intArg(ActKernels.ARG_GLU_N_ROWS));
        assertEquals(N_HEAD * HEAD_V, glu.intArg(ActKernels.ARG_GLU_N_COLS));
        List<Dispatch> ropes = all(ds, RopeKernels.ROPE_EXT);
        assertEquals(IDX_DIM - IDX_ROPE, ropes.get(0).intArg(ROPE_ARG_N_OFFS));
    }

    // ------------------------------------------------------------------ binding errors

    @Test
    public void layoutRefusesNextnBlocks() {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM_DSA)).layerCounts(2, 1)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f).normEps(1e-6f).build();
        LayerSpec nextn = layer(1, 1, RopeType.NEOX, false, true);
        ModelSpec spec = ModelSpec.builder(hp, TensorCatalog.of(Arch.GLM_DSA.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(deepseek32Layer(0))
                .nextnLayer(nextn)
                .build();
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> DsaLayout.of(nextn, spec, suffix -> new long[0]));
        assertTrue(e.getMessage(), e.getMessage().contains("NextN"));
    }

    @Test
    public void factoryRejectsPooledIndexer() {
        open();
        DsaSpec dsa = DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX_DIM).indexerRopeDim(IDX_ROPE)
                .ropeType(RopeType.NEOX).topK(TOP_K).sourceLayer(0).kpool(4, 2, false, false)
                .defaultWeightScale().build();
        LayerSpec base = deepseek32Layer(0);
        LayerSpec pooled = LayerSpec.builder(0, MixerKind.DSA_MLA, FfnKind.NONE)
                .norm(base.norm()).residual(base.residual()).mla(base.mla()).dsa(dsa).rope(base.rope()).build();
        assertThrows(UnsupportedOperationException.class, () -> SparseBlocks.gpuMixers().get(0).create(pooled));
        assertThrows(UnsupportedOperationException.class, () -> SparseBlocks.cpuMixers().get(0).create(pooled));
    }

    @Test
    public void bindRejectsMissingIndexerCacheAndWrongShapes() {
        open();
        mlaTensors(0, true);
        memory.kv.put(0, new FakeKv(0));
        LayerSpec l = deepseek32Layer(0);
        GpuForwardContext ctx = context(Arch.DEEPSEEK32, Collections.singletonList(l));
        GpuMixer m = SparseBlocks.gpuMixers().get(0).create(l);
        mixers.add(m);
        assertThrows(IllegalStateException.class, () -> m.bind(ctx, l));

        put(0, DsaLayout.WQ_B, Q_LORA, (long) N_HEAD * HEAD_K + 1);
        caches(0, true, true);
        GpuMixer m2 = SparseBlocks.gpuMixers().get(0).create(l);
        mixers.add(m2);
        assertThrows(IllegalStateException.class, () -> m2.bind(ctx, l));
    }

    @Test
    public void familyHasNoFfnsOrExtras() {
        assertEquals(0, SparseBlocks.gpuFfns().size());
        assertEquals(0, SparseBlocks.cpuFfns().size());
        assertEquals(0, SparseBlocks.gpuExtras().size());
        assertEquals(0, SparseBlocks.cpuExtras().size());
        assertEquals(MixerKind.DSA_MLA, SparseBlocks.gpuMixers().get(0).kind());
        assertEquals(MixerKind.DSA_MLA, SparseBlocks.cpuMixers().get(0).kind());
    }

    // ------------------------------------------------------------------ memory fakes

    private final class FakeKv implements GpuKvLayer {
        final int il;
        final int group;
        final KvLayout layout = KvLayout.latent(ROWS, false, D_K, KV_LORA, KvDType.F32, KvWindow.FULL_CAUSAL);
        final GpuBuffer k = backend.allocateBuffer((long) ROWS * D_K * Float.BYTES);

        FakeKv(int il) {
            this(il, 0);
        }

        FakeKv(int il, int group) {
            this.il = il;
            this.group = group;
        }

        @Override public int layer() { return il; }
        @Override public KvLayout layout() { return layout; }
        @Override public int group() { return group; }
        @Override public RotationPlan rotation() { return RotationPlan.NONE; }
        @Override public GpuBuffer k() { return k; }
        @Override public GpuBuffer v() { return k; }
    }

    private final class FakeIndexer implements GpuIndexerCache {
        final int il;
        final LayerMemoryPlan.IndexerPlan plan;
        final GpuBuffer keys;

        FakeIndexer(int il, RotationPlan rotation, KvDType type) {
            this.il = il;
            this.plan = LayerMemoryPlan.IndexerPlan.of(ROWS, IDX_DIM, type, rotation);
            this.keys = backend.allocateBuffer(plan.bytes());
        }

        @Override public int layer() { return il; }
        @Override public LayerMemoryPlan.IndexerPlan plan() { return plan; }
        @Override public int group() { return 0; }
        @Override public GpuBuffer keys() { return keys; }
    }

    private final class FakeSteps implements GpuStepInputs {
        final GpuBuffer kvPos = backend.allocateBuffer((long) ROWS * Integer.BYTES);
        final GpuBuffer cellIdx = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        final GpuBuffer qPos = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        final GpuBuffer ropePos = backend.allocateBuffer(4L * UBATCH * Integer.BYTES);
        final GpuBuffer stateRows = backend.allocateBuffer(Integer.BYTES);
        int pos0;
        int nTokens;

        @Override public int groups() { return 2; }
        @Override public GpuBuffer kvPos(int group) { return kvPos; }
        @Override public GpuBuffer cellIdx(int group) { return cellIdx; }
        @Override public GpuBuffer qPos() { return qPos; }
        @Override public GpuBuffer ropePos() { return ropePos; }
        @Override public GpuBuffer stateRows() { return stateRows; }
        @Override public int pos0() { return pos0; }
        @Override public int nTokens() { return nTokens; }

        @Override
        public void upload(int pos0, int nTokens) {
            this.pos0 = pos0;
            this.nTokens = nTokens;
        }
    }

    private final class FakeMemory implements GpuMemory {
        final Map<Integer, FakeKv> kv = new HashMap<>();
        final Map<Integer, FakeIndexer> indexer = new HashMap<>();
        final FakeSteps steps = new FakeSteps();

        @Override public GpuKvLayer kv(int il) { return kv.get(il); }
        @Override public GpuKvLayer crossKv(int il) { return null; }
        @Override public GpuRecurrentState recurrent(int il) { return null; }
        @Override public GpuIndexerCache indexer(int il) { return indexer.get(il); }
        @Override public GpuMsaCache msa(int il) { return null; }
        @Override public List<GpuCompressedKv> compressed(int il) { return Collections.emptyList(); }
        @Override public GpuStepInputs stepInputs() { return steps; }
        @Override public MemoryPlan plan() { throw new UnsupportedOperationException("plan"); }
        @Override public SequenceState sequence() { throw new UnsupportedOperationException("sequence"); }
        @Override public void reset() { }
        @Override public boolean canRollback(int nTokens) { return false; }
        @Override public void rollback(int nTokens) { throw new UnsupportedOperationException("rollback"); }
        @Override public void snapshot() { throw new UnsupportedOperationException("snapshot"); }
        @Override public void restore() { throw new UnsupportedOperationException("restore"); }
        @Override public void close() { }
    }
}

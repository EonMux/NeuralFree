package com.eonmux.neuralfree.gguf.ops.sparse;

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
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMsaCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of the MiniMax-M3 sparse layer ({@link GpuMsaMixer}, models/minimax-m3.cpp:306-535) against the
 * host-only fake backend with strict LLM argument rules: one decode token and a prefill ubatch with separate Q/K/V,
 * the fused {@code attn_qkv} with its bias, and the family's routing of dense lead layers. Each trace checks the
 * kernel sequence and the arguments of the sparse kernels with buffer identities.
 */
public class GpuMsaMixerTest {

    private static final int N_EMBD = 64;
    private static final int UBATCH = 8;
    private static final int N_HEAD = 4;
    private static final int N_KV = 2;
    private static final int HD = 16;
    private static final int Q_DIM = N_HEAD * HD;
    private static final int K_DIM = N_KV * HD;
    private static final int IDX = 16;
    private static final int N_ROT = 8;
    private static final int BLK = 4;
    private static final int LOCAL_BLOCKS = 1;
    private static final int TOP_K = 2;
    private static final int ROWS = 64;
    private static final float KQ = 0.25f;

    private static final String NORM = GpuOps.ROW_NORM;
    private static final String ROPE = RopeKernels.ROPE_EXT;
    private static final String STORE = RopeKernels.kvStore(KvDType.F16);
    private static final String SCORE = MsaKernels.MSA_BLOCK_SCORE;
    private static final String TOPK = SparseKernels.TOPK_ROWS;
    private static final String GATHER = MsaKernels.MSA_GATHER_ATTN;
    private static final String COPY = GpuOps.COPY2D;
    private static final String BIAS = MsaBlocks.BATCH_BIAS_ADD;
    private static final String MV = "Float_MatVec";
    private static final Set<String> NON_PROJECTION = new HashSet<>(
            Arrays.asList(NORM, ROPE, STORE, SCORE, TOPK, GATHER, COPY, BIAS));

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private LlmKernels kernels;
    private GpuMixer mixer;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private GpuBuffer kCache;
    private GpuBuffer vCache;
    private GpuBuffer indexKeys;
    private GpuBuffer kvPos;
    private GpuBuffer cellIdx;
    private GpuBuffer qPos;
    private GpuBuffer ropePos;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        GpuScratch.Builder sb = GpuScratch.builder(backend, UBATCH);
        for (String s : Arrays.asList(ScratchSlots.NORMED, ScratchSlots.BRANCH)) {
            sb.slot(s, (long) UBATCH * N_EMBD);
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

    private void indexerAndNorms() {
        tensor("blk.0.attn_output.weight", Q_DIM, N_EMBD);
        tensor("blk.0.attn_q_norm.weight", HD);
        tensor("blk.0.attn_k_norm.weight", HD);
        tensor("blk.0.indexer.q_proj.weight", N_EMBD, N_KV * IDX);
        tensor("blk.0.indexer.k_proj.weight", N_EMBD, IDX);
        tensor("blk.0.indexer.q_norm.weight", IDX);
        tensor("blk.0.indexer.k_norm.weight", IDX);
    }

    private void separateQkv() {
        tensor("blk.0.attn_q.weight", N_EMBD, Q_DIM);
        tensor("blk.0.attn_k.weight", N_EMBD, K_DIM);
        tensor("blk.0.attn_v.weight", N_EMBD, K_DIM);
        indexerAndNorms();
    }

    private void fusedQkvWithBias() {
        tensor("blk.0.attn_qkv.weight", N_EMBD, Q_DIM + 2 * K_DIM);
        tensor("blk.0.attn_qkv.bias", Q_DIM + 2 * K_DIM);
        indexerAndNorms();
    }

    private static LayerSpec layer(boolean sparse) {
        AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_KV).headDimK(HD).headDimV(HD).nRot(N_ROT)
                .causal(true).qkNorm(QkNormKind.RMS_HEAD).kqScale(KQ).kvOwnerLayer(0).build();
        MsaSpec msa = MsaSpec.builder().idxHeads(N_KV).idxHeadDim(IDX).idxRopeDim(N_ROT).blockSize(BLK)
                .localBlocks(LOCAL_BLOCKS).topKBlocks(TOP_K).sparse(sparse).kqScale(KQ).build();
        return LayerSpec.builder(0, MixerKind.MSA, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .attention(a)
                .msa(msa)
                .rope(RopeSpec.builder(RopeType.NEOX).nDims(N_ROT).build())
                .build();
    }

    private static ModelSpec model(LayerSpec layer) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.MINIMAX_M3)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(1e-6f);
        return ModelSpec.builder(hb.build(), TensorCatalog.of(Arch.MINIMAX_M3.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }

    /** A one-sequence device memory: layer 0 owns an F16 K/V cache and an F16 MSA index-key cache in group 0. */
    private GpuMemory memory() {
        KvLayout layout = KvLayout.of(ROWS, false, K_DIM, K_DIM, KvDType.F16, KvDType.F16, KvWindow.FULL_CAUSAL);
        LayerMemoryPlan.MsaPlan plan = LayerMemoryPlan.MsaPlan.of(ROWS, IDX, KvDType.F16);
        kCache = backend.allocateBuffer((long) ROWS * K_DIM * 2);
        vCache = backend.allocateBuffer((long) ROWS * K_DIM * 2);
        indexKeys = backend.allocateBuffer((long) ROWS * IDX * 2);
        kvPos = backend.allocateBuffer((long) ROWS * Integer.BYTES);
        cellIdx = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        qPos = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        ropePos = backend.allocateBuffer(4L * UBATCH * Integer.BYTES);
        GpuKvLayer kv = proxy(GpuKvLayer.class, (name, args) -> {
            switch (name) {
                case "layer": return 0;
                case "layout": return layout;
                case "group": return 0;
                case "rotation": return RotationPlan.NONE;
                case "k": return kCache;
                case "v": return vCache;
                default: throw new UnsupportedOperationException(name);
            }
        });
        GpuMsaCache keys = proxy(GpuMsaCache.class, (name, args) -> {
            switch (name) {
                case "layer": return 0;
                case "plan": return plan;
                case "group": return 0;
                case "indexKeys": return indexKeys;
                default: throw new UnsupportedOperationException(name);
            }
        });
        GpuStepInputs steps = proxy(GpuStepInputs.class, (name, args) -> {
            switch (name) {
                case "kvPos": return kvPos;
                case "cellIdx": return cellIdx;
                case "qPos": return qPos;
                case "ropePos": return ropePos;
                default: throw new UnsupportedOperationException(name);
            }
        });
        return proxy(GpuMemory.class, (name, args) -> {
            switch (name) {
                case "kv": return kv;
                case "msa": return keys;
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

    private GpuMixer bind(LayerSpec layer) {
        GpuWeights ws = tensors::get;
        GpuForwardContext ctx = GpuForwardContext.builder(kernels, ops, ws, model(layer)).memory(memory()).build();
        mixer = MsaBlocks.gpuMixers().get(0).create(layer);
        mixer.bind(ctx, layer);
        return mixer;
    }

    private ScratchSlot slot(String name) {
        return scratch.slot(name);
    }

    /** The current buffer of a mixer temporary (a request no larger than the bound one returns it). */
    private GpuBuffer temp(String name) {
        return scratch.temp(name, 1).buffer();
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

    private static List<Dispatch> nonProjection(List<Dispatch> ds) {
        List<Dispatch> out = new ArrayList<>();
        for (Dispatch d : ds) {
            if (NON_PROJECTION.contains(d.kernelName())) {
                out.add(d);
            }
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

    // ------------------------------------------------------------------ checks shared by the traces

    private void checkStores(List<Dispatch> sparse, int n) {
        Dispatch ik = sparse.get(8);
        assertArrayEquals(new int[]{n, IDX, IDX, 0, IDX}, ints(ik, 0, 5));
        assertSame(temp(GpuMsaMixer.TEMP_IK), ik.bufferArg(5));
        assertSame(cellIdx, ik.bufferArg(6));
        assertSame(indexKeys, ik.bufferArg(7));
        Dispatch k = sparse.get(9);
        assertArrayEquals(new int[]{n, K_DIM, K_DIM, 0, K_DIM}, ints(k, 0, 5));
        assertSame(temp(GpuMsaMixer.TEMP_K), k.bufferArg(5));
        assertSame(kCache, k.bufferArg(7));
        Dispatch v = sparse.get(10);
        assertSame(temp(GpuMsaMixer.TEMP_V), v.bufferArg(5));
        assertSame(vCache, v.bufferArg(7));
    }

    private void checkSelection(List<Dispatch> sparse, int n, int nPos) {
        int nblk = MsaKernels.blockCount(nPos, BLK);
        Dispatch score = sparse.get(11);
        assertEquals(SCORE, score.kernelName());
        assertArrayEquals(new int[]{n, nPos, N_KV, IDX, BLK, LOCAL_BLOCKS, ROWS, KvDType.F16.code(), 0},
                ints(score, 0, 9));
        assertSame(temp(GpuMsaMixer.TEMP_IQ), score.bufferArg(9));
        assertSame(indexKeys, score.bufferArg(10));
        assertSame(qPos, score.bufferArg(11));
        assertSame(kvPos, score.bufferArg(12));
        assertSame(temp(GpuMsaMixer.TEMP_SCORES), score.bufferArg(13));
        int local = MsaKernels.blockScoreLocal(BLK);
        assertArrayEquals(MsaKernels.blockScoreGlobal(n, nPos, BLK, local), score.global());

        Dispatch topk = sparse.get(12);
        assertEquals(TOPK, topk.kernelName());
        assertArrayEquals(new int[]{n * N_KV, nblk, TOP_K}, ints(topk, 0, 3));
        assertSame(temp(GpuMsaMixer.TEMP_SCORES), topk.bufferArg(3));
        assertSame(temp(GpuMsaMixer.TEMP_BLOCKS), topk.bufferArg(4));

        Dispatch gather = sparse.get(13);
        assertEquals(GATHER, gather.kernelName());
        assertArrayEquals(new int[]{n, N_HEAD, N_KV, HD, BLK, TOP_K}, ints(gather, 0, 6));
        assertEquals(KQ, gather.floatArg(6), 0.0f);
        assertArrayEquals(new int[]{N_KV, 0, ROWS, K_DIM, K_DIM, KvDType.F16.code(), KvDType.F16.code(), Q_DIM, 0,
                Q_DIM, 0}, ints(gather, 7, 18));
        assertSame(temp(GpuMsaMixer.TEMP_Q), gather.bufferArg(18));
        assertSame(kCache, gather.bufferArg(19));
        assertSame(vCache, gather.bufferArg(20));
        assertSame(qPos, gather.bufferArg(21));
        assertSame(kvPos, gather.bufferArg(22));
        assertSame(temp(GpuMsaMixer.TEMP_BLOCKS), gather.bufferArg(23));
        assertSame(temp(GpuMsaMixer.TEMP_ATT), gather.bufferArg(24));
        assertArrayEquals(MsaKernels.gatherGlobal(n, N_HEAD, MsaKernels.GATHER_LOCAL), gather.global());
    }

    // ------------------------------------------------------------------ traces

    @Test
    public void decodeTrace() {
        open();
        separateQkv();
        LayerSpec layer = layer(true);
        bind(layer);
        List<Dispatch> ds = run(() -> mixer.decode(7, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        assertEquals(Arrays.asList(MV, MV, MV, NORM, NORM, ROPE, ROPE, MV, MV, NORM, NORM, ROPE, ROPE, STORE, STORE,
                STORE, SCORE, TOPK, GATHER, MV), names(ds));
        assertSame(buf("blk.0.attn_q.weight"), ds.get(0).bufferArg(2));
        assertSame(temp(GpuMsaMixer.TEMP_Q), ds.get(0).bufferArg(4));
        assertSame(buf("blk.0.indexer.q_proj.weight"), ds.get(7).bufferArg(2));
        assertSame(buf("blk.0.attn_output.weight"), ds.get(19).bufferArg(2));
        assertSame(slot(ScratchSlots.BRANCH).buffer(), ds.get(19).bufferArg(4));

        Dispatch ropeQ = ds.get(5);
        assertArrayEquals(new int[]{1, N_HEAD, HD, Q_DIM, N_ROT, 0}, ints(ropeQ, 0, 6));
        assertSame(ropePos, ropeQ.bufferArg(20));
        assertSame(temp(GpuMsaMixer.TEMP_Q), ropeQ.bufferArg(22));
        Dispatch ropeIq = ds.get(11);
        assertArrayEquals(new int[]{1, N_KV, IDX, N_KV * IDX, N_ROT, 0}, ints(ropeIq, 0, 6));
        assertSame(temp(GpuMsaMixer.TEMP_IQ), ropeIq.bufferArg(22));

        List<Dispatch> sparse = nonProjection(ds);
        checkStores(sparse, 1);
        checkSelection(sparse, 1, 8);
    }

    @Test
    public void prefillTrace() {
        open();
        separateQkv();
        LayerSpec layer = layer(true);
        bind(layer);
        List<Dispatch> ds = run(() -> mixer.prefill(5, 6, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        List<Dispatch> sparse = nonProjection(ds);
        assertEquals(Arrays.asList(NORM, NORM, ROPE, ROPE, NORM, NORM, ROPE, ROPE, STORE, STORE, STORE, SCORE, TOPK,
                GATHER), names(sparse));
        Dispatch ropeK = sparse.get(3);
        assertEquals(6, ropeK.intArg(RopeKernels.ARG_N_TOKENS));
        assertEquals(6, ropeK.intArg(RopeKernels.ARG_POS_STRIDE));
        checkStores(sparse, 6);
        checkSelection(sparse, 6, 11);
        assertTrue(names(ds).size() > sparse.size());
    }

    @Test
    public void fusedQkvBiasTrace() {
        open();
        fusedQkvWithBias();
        LayerSpec layer = layer(true);
        bind(layer);
        List<Dispatch> ds = run(() -> mixer.decode(0, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));

        assertEquals(Arrays.asList(MV, BIAS, COPY, COPY, COPY, NORM, NORM, ROPE, ROPE), names(ds).subList(0, 9));
        Dispatch bias = ds.get(1);
        assertArrayEquals(new int[]{Q_DIM + 2 * K_DIM, 1}, ints(bias, 0, 2));
        assertSame(buf("blk.0.attn_qkv.bias"), bias.bufferArg(2));
        assertSame(temp(GpuMsaMixer.TEMP_QKV), bias.bufferArg(3));
        List<Dispatch> sparse = nonProjection(ds);
        checkSelection(sparse.subList(4, sparse.size()), 1, 1);
    }

    @Test
    public void denseLayerNeedsTheAttentionFactory() {
        LayerSpec dense = layer(false);
        GpuMixerFactory f = MsaBlocks.gpuMixers().get(0);
        assertEquals(MixerKind.MSA, f.kind());
        assertEquals(MsaBlocks.Variant.DENSE, MsaBlocks.variant(dense));
        assertThrows(IllegalStateException.class, () -> f.create(dense));
        assertEquals(MsaBlocks.Variant.MSA, MsaBlocks.variant(layer(true)));
        assertTrue(f.create(layer(true)) instanceof GpuMsaMixer);
    }

    @Test
    public void stepBeyondTheCacheIsRejected() {
        open();
        separateQkv();
        bind(layer(true));
        assertThrows(IllegalArgumentException.class,
                () -> mixer.prefill(ROWS - 2, 4, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
        assertThrows(IllegalArgumentException.class,
                () -> mixer.prefill(0, UBATCH + 1, slot(ScratchSlots.NORMED), slot(ScratchSlots.BRANCH)));
    }
}

package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SwaType;
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
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
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
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks.Dsv4Layout;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.sparse.SparseKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
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
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dispatch traces of {@link GpuDsv4Mixer} on the host-only fake backend with strict LLM argument rules, for small
 * deepseek4 layers: the decode sequence of a CSA layer (both compressors, the lightning indexer over the compressed
 * indexer rows, the raw store and the compressed attention), the steps before the first compressed row, an HCA
 * prefill that completes a block, a window-only layer, the shared Hadamard block of quantized caches, the grouped
 * output projection, and the binding and step errors.
 */
public class GpuDsv4MixerTest {

    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD = 64;
    private static final int N_ROT = 16;
    private static final int Q_LORA = 32;
    private static final int N_GROUPS = 2;
    private static final int O_LORA = 16;
    private static final int O_GROUP_DIM = N_HEAD / N_GROUPS * HEAD;
    private static final int IDX_HEADS = 2;
    private static final int IDX_DIM = 64;
    private static final int TOP_K = 4;
    private static final int WINDOW = 16;
    private static final int RAW_ROWS = 32;
    private static final int N_CTX = 256;
    private static final int UBATCH = 8;
    private static final float BASE = 10000f;
    private static final float COMPRESS_BASE = 160000f;
    private static final float FREQ_SCALE = 0.25f;
    private static final float EXT = 1.0f;
    private static final int N_CTX_ORIG = 4096;
    private static final String MATVEC = "Float_MatVec";
    private static final String ID_MATVEC = MoeKernels.matVecId(GGMLType.F32);

    /** Argument positions of LLM_DSV4_CompressedAttn in signature order. */
    private static final int ATTN_RAW_TYPE = 7;
    private static final int ATTN_N_SWA = 8;
    private static final int ATTN_COMP_MODE = 9;
    private static final int ATTN_COMP_TYPE = 11;
    private static final int ATTN_RATIO = 12;
    private static final int ATTN_HAS_SINKS = 14;
    private static final int ATTN_ROT = 16;
    private static final int ATTN_THETA = 19;
    private static final int ATTN_FREQ_SCALE = 20;
    private static final int ATTN_EXT = 21;
    private static final int ATTN_FACTOR = 22;
    private static final int ATTN_Q = 25;
    private static final int ATTN_RAW_K = 26;
    private static final int ATTN_RAW_POS = 27;
    private static final int ATTN_COMP_K = 28;
    private static final int ATTN_SEL = 29;
    private static final int ATTN_SINKS = 30;
    private static final int ATTN_Q_POS = 31;
    private static final int ATTN_OUT = 32;
    /** Argument positions of LLM_DSV4_Compress / LLM_DSV4_LidUpdate. */
    private static final int COMPRESS_RATIO = 2;
    private static final int COMPRESS_WIDTH = 3;
    private static final int COMPRESS_ROT = 13;
    private static final int COMPRESS_CACHE = 23;
    private static final int LID_WIDTH = 2;
    private static final int LID_ROT = 12;
    private static final int LID_CACHE = 21;
    /** Argument positions of LLM_DSV4_StateUpdate and LLM_DSV4_CacheStore. */
    private static final int STATE_RATIO = 2;
    private static final int STATE_WIDTH = 3;
    private static final int STORE_ROT = 3;
    private static final int STORE_CELL_IDX = 7;
    private static final int STORE_CACHE = 8;
    /** {@code n_offs} and {@code theta_scale} of LLM_RoPE_Ext. */
    private static final int ROPE_N_OFFS = 5;
    private static final int ROPE_THETA = 9;
    /** {@code kv_pos} of LLM_DSA_IndexerScore. */
    private static final int SCORE_KV_POS = 11;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private int dsv4Local;
    private FakeMemory memory;
    private final Map<String, GpuWeight> tensors = new HashMap<>();
    private final List<GpuMixer> mixers = new ArrayList<>();

    private void open() {
        backend = FakeComputeBackend.builder().capabilities(FakeComputeBackend.defaultCapabilities())
                .strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH)
                .slot(ScratchSlots.NORMED, (long) UBATCH * N_EMBD)
                .slot(ScratchSlots.BRANCH, (long) UBATCH * N_EMBD)
                .build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
        dsv4Local = Math.min(Dsv4Kernels.LOCAL, ops.local());
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

    /** The tensors of a layer of compress ratio {@code ratio}, with {@code attn_sinks} when {@code sinks}. */
    private void layerTensors(int il, int ratio, boolean sinks) {
        put(il, Dsv4Layout.WQ_A, N_EMBD, Q_LORA);
        put(il, Dsv4Layout.Q_A_NORM, Q_LORA);
        put(il, Dsv4Layout.WQ_B, Q_LORA, (long) N_HEAD * HEAD);
        put(il, Dsv4Layout.WKV, N_EMBD, HEAD);
        put(il, Dsv4Layout.KV_NORM, HEAD);
        put(il, Dsv4Layout.WO_A, O_GROUP_DIM, O_LORA, N_GROUPS);
        put(il, Dsv4Layout.WO_B, (long) N_GROUPS * O_LORA, N_EMBD);
        if (sinks) {
            put(il, Dsv4Layout.SINKS, N_HEAD);
        }
        if (ratio != 0) {
            int sw = Dsv4Kernels.stateWidth(ratio, HEAD);
            put(il, Dsv4Layout.COMP_WKV, N_EMBD, sw);
            put(il, Dsv4Layout.COMP_WGATE, N_EMBD, sw);
            put(il, Dsv4Layout.COMP_APE, sw, ratio);
            put(il, Dsv4Layout.COMP_NORM, HEAD);
        }
        if (ratio == Dsv4Spec.RATIO_CSA) {
            int sw = Dsv4Kernels.stateWidth(Dsv4Spec.RATIO_CSA, IDX_DIM);
            put(il, Dsv4Layout.IDX_Q_B, Q_LORA, (long) IDX_HEADS * IDX_DIM);
            put(il, Dsv4Layout.IDX_PROJ, N_EMBD, IDX_HEADS);
            put(il, Dsv4Layout.IDX_COMP_WKV, N_EMBD, sw);
            put(il, Dsv4Layout.IDX_COMP_WGATE, N_EMBD, sw);
            put(il, Dsv4Layout.IDX_COMP_APE, sw, Dsv4Spec.RATIO_CSA);
            put(il, Dsv4Layout.IDX_COMP_NORM, IDX_DIM);
        }
    }

    private static LayerSpec layer(int il, int ratio, boolean sinks, int qLora) {
        Dsv4Spec.Builder v4 = Dsv4Spec.builder().nHead(N_HEAD).compressRatio(ratio).window(WINDOW).headDim(HEAD)
                .ropeDim(N_ROT).qLoraRank(qLora).oGroupCount(N_GROUPS).oLoraRank(O_LORA)
                .compressRopeBase(COMPRESS_BASE).sinks(sinks).kqScale(1.0f / (float) Math.sqrt(HEAD));
        if (ratio == Dsv4Spec.RATIO_CSA) {
            v4.indexer(DsaSpec.builder().indexerHeads(IDX_HEADS).indexerHeadDim(IDX_DIM).indexerRopeDim(N_ROT)
                    .ropeType(RopeType.NORM).ropeTail(true).topK(TOP_K).full(true).sourceLayer(il).hadamard(true)
                    .defaultWeightScale().build());
        }
        float[] corr = RopeMath.corrDims(N_ROT, N_CTX_ORIG, BASE, 32f, 1f);
        RopeSpec rope = RopeSpec.builder(RopeType.NORM).nDims(N_ROT).nOffs(HEAD - N_ROT).freqBase(BASE)
                .freqScale(FREQ_SCALE).yarn(EXT, 1.2f, 32f, 1f).corrDims(corr[0], corr[1]).nCtxOrig(N_CTX_ORIG)
                .build();
        return LayerSpec.builder(il, MixerKind.DSV4, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .dsv4(v4.build())
                .rope(rope)
                .build();
    }

    private static LayerSpec layer(int il, int ratio) {
        return layer(il, ratio, false, Q_LORA);
    }

    /**
     * The raw window cache of a layer and the compressed caches its ratio needs, whose compressor tensors keep
     * {@code snapshots} rollback snapshot planes.
     */
    private void caches(int il, int ratio, KvDType type, RotationPlan rotation, int snapshots) {
        memory.kv.put(il, new FakeKv(il, type, rotation));
        List<GpuCompressedKv> comp = new ArrayList<>();
        if (ratio == Dsv4Spec.RATIO_CSA) {
            comp.add(new FakeCompressed(il, CompressedKvPlan.Kind.CSA, type, snapshots));
            comp.add(new FakeCompressed(il, CompressedKvPlan.Kind.INDEXER, type, snapshots));
        } else if (ratio == Dsv4Spec.RATIO_HCA) {
            comp.add(new FakeCompressed(il, CompressedKvPlan.Kind.HCA, type, snapshots));
        }
        memory.compressed.put(il, comp);
    }

    private void caches(int il, int ratio, KvDType type, RotationPlan rotation) {
        caches(il, ratio, type, rotation, 0);
    }

    private void caches(int il, int ratio) {
        caches(il, ratio, KvDType.F32, RotationPlan.NONE);
    }

    private GpuForwardContext context(List<LayerSpec> layers) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK4)).layerCounts(layers.size(), 0)
                .nCtxTrain(N_CTX_ORIG).nEmbd(N_EMBD).normRmsEps(1e-6f).normEps(1e-6f).build();
        ModelSpec.Builder b = ModelSpec.builder(hp, TensorCatalog.of(Arch.DEEPSEEK4.ggufName(), Collections.emptyList()))
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

    private GpuMixer bind(LayerSpec l) {
        GpuForwardContext ctx = context(Collections.singletonList(l));
        GpuMixer m = Dsv4Blocks.gpuMixers().get(0).create(l);
        mixers.add(m);
        m.bind(ctx, l);
        return m;
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
        List<Dispatch> found = all(ds, kernel);
        if (found.size() != 1) {
            throw new AssertionError(found.size() + " dispatches of " + kernel + " in " + names(ds));
        }
        return found.get(0);
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

    private FakeCompressed compressed(int il, CompressedKvPlan.Kind kind) {
        for (GpuCompressedKv c : memory.compressed.get(il)) {
            if (c.plan().kind() == kind) {
                return (FakeCompressed) c;
            }
        }
        throw new AssertionError("no " + kind + " cache for layer " + il);
    }

    private int headTile() {
        return Dsv4Kernels.attnHeadTile(HEAD, N_HEAD, dsv4Local, ops.kernels().faSmemFloats());
    }

    private static float theta(float base) {
        return (float) Math.pow(base, -2.0 / N_ROT);
    }

    // ------------------------------------------------------------------ CSA

    @Test
    public void csaDecodeRunsCompressorsIndexerAndCompressedAttention() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, true);
        caches(0, Dsv4Spec.RATIO_CSA);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA, true, Q_LORA));
        List<Dispatch> ds = decode(m, 7);

        assertEquals(Arrays.asList(
                MATVEC, NormKernels.ROW_NORM, MATVEC, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT,
                MATVEC, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT,
                MATVEC, MATVEC, Dsv4Kernels.COMPRESS, Dsv4Kernels.STATE_UPDATE,
                MATVEC, MATVEC, Dsv4Kernels.lidUpdate(KvDType.F32), Dsv4Kernels.STATE_UPDATE,
                MATVEC, RopeKernels.ROPE_EXT, HadamardSpec.KERNEL, MATVEC, GpuOps.AXPBY,
                SparseKernels.indexerScore(KvDType.F32), SparseKernels.TOPK_ROWS,
                Dsv4Kernels.CACHE_STORE, Dsv4Kernels.COMPRESSED_ATTN, ID_MATVEC, MATVEC), names(ds));

        FakeCompressed csa = compressed(0, CompressedKvPlan.Kind.CSA);
        FakeCompressed lid = compressed(0, CompressedKvPlan.Kind.INDEXER);
        Dispatch compress = only(ds, Dsv4Kernels.COMPRESS);
        assertEquals(7, compress.intArg(Dsv4Kernels.ARG_COMPRESS_POS0));
        assertEquals(1, compress.intArg(Dsv4Kernels.ARG_COMPRESS_N_TOKENS));
        assertEquals(Dsv4Spec.RATIO_CSA, compress.intArg(COMPRESS_RATIO));
        assertEquals(HEAD, compress.intArg(COMPRESS_WIDTH));
        assertEquals(0, compress.intArg(COMPRESS_ROT));
        assertEquals(theta(COMPRESS_BASE), compress.floatArg(7), 0f);
        assertEquals(FREQ_SCALE, compress.floatArg(8), 0f);
        assertEquals(Dsv4Kernels.attnFactor(FREQ_SCALE, EXT), compress.floatArg(10), 0f);
        assertSame(csa.cache, compress.bufferArg(COMPRESS_CACHE));
        assertArrayEquals(new long[]{dsv4Local}, compress.global());

        Dispatch lidUpdate = only(ds, Dsv4Kernels.lidUpdate(KvDType.F32));
        assertEquals(7, lidUpdate.intArg(Dsv4Kernels.ARG_LID_POS0));
        assertEquals(IDX_DIM, lidUpdate.intArg(LID_WIDTH));
        assertEquals(Dsv4Layout.lidRotationBlock(IDX_DIM), lidUpdate.intArg(LID_ROT));
        assertSame(lid.cache, lidUpdate.bufferArg(LID_CACHE));

        List<Dispatch> states = all(ds, Dsv4Kernels.STATE_UPDATE);
        assertEquals(Dsv4Spec.RATIO_CSA, states.get(0).intArg(STATE_RATIO));
        assertEquals(HEAD, states.get(0).intArg(STATE_WIDTH));
        assertSame(csa.stateKv, states.get(0).bufferArg(9));
        assertEquals(IDX_DIM, states.get(1).intArg(STATE_WIDTH));
        assertSame(lid.stateScore, states.get(1).bufferArg(10));

        Dispatch score = only(ds, SparseKernels.indexerScore(KvDType.F32));
        assertEquals(1, score.intArg(SparseKernels.ARG_SCORE_N_TOKENS));
        assertEquals(2, score.intArg(SparseKernels.ARG_SCORE_N_KV));
        assertSame(lid.cache, score.bufferArg(9));
        FakeGpuBuffer lidPos = score.bufferArg(SCORE_KV_POS);
        assertArrayEquals(new int[]{3, 7, 11, 15}, lidPos.getInts(0, 4));
        Dispatch topK = only(ds, SparseKernels.TOPK_ROWS);
        assertEquals(2, topK.intArg(SparseKernels.ARG_TOPK_K));

        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        FakeKv raw = memory.kv.get(0);
        assertEquals(1, attn.intArg(Dsv4Kernels.ARG_ATTN_N_TOKENS));
        assertEquals(8, attn.intArg(Dsv4Kernels.ARG_ATTN_N_RAW));
        assertEquals(2, attn.intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
        assertEquals(2, attn.intArg(Dsv4Kernels.ARG_ATTN_SEL_STRIDE));
        assertEquals(WINDOW, attn.intArg(ATTN_N_SWA));
        assertEquals(Dsv4Kernels.CompMode.SELECTED.code(), attn.intArg(ATTN_COMP_MODE));
        assertEquals(Dsv4Spec.RATIO_CSA, attn.intArg(ATTN_RATIO));
        assertEquals(1, attn.intArg(ATTN_HAS_SINKS));
        assertEquals(theta(COMPRESS_BASE), attn.floatArg(ATTN_THETA), 0f);
        assertEquals(FREQ_SCALE, attn.floatArg(ATTN_FREQ_SCALE), 0f);
        assertEquals(EXT, attn.floatArg(ATTN_EXT), 0f);
        assertEquals(Dsv4Kernels.attnFactor(FREQ_SCALE, EXT), attn.floatArg(ATTN_FACTOR), 0f);
        assertSame(raw.k, attn.bufferArg(ATTN_RAW_K));
        assertSame(memory.steps.kvPos, attn.bufferArg(ATTN_RAW_POS));
        assertSame(csa.cache, attn.bufferArg(ATTN_COMP_K));
        assertSame(topK.bufferArg(4), attn.bufferArg(ATTN_SEL));
        assertSame(weight(0, Dsv4Layout.SINKS).shard(0), attn.bufferArg(ATTN_SINKS));
        assertSame(memory.steps.qPos, attn.bufferArg(ATTN_Q_POS));
        assertArrayEquals(Dsv4Kernels.attnGlobal(1, N_HEAD, headTile(), dsv4Local), attn.global());

        Dispatch store = only(ds, Dsv4Kernels.CACHE_STORE);
        assertEquals(1, store.intArg(Dsv4Kernels.ARG_STORE_N_TOKENS));
        assertSame(memory.steps.cellIdx, store.bufferArg(STORE_CELL_IDX));
        assertSame(raw.k, store.bufferArg(STORE_CACHE));

        Dispatch wo = ds.get(ds.size() - 1);
        assertSame(weight(0, Dsv4Layout.WO_B).shard(0), wo.bufferArg(2));
        assertSame(scratch.slot(ScratchSlots.BRANCH).buffer(), wo.bufferArg(4));
    }

    @Test
    public void csaDecodeBeforeTheFirstCompressedRowSkipsCompressionAndSelection() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        caches(0, Dsv4Spec.RATIO_CSA);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        List<Dispatch> ds = decode(m, 2);

        assertFalse(names(ds).contains(Dsv4Kernels.COMPRESS));
        assertNoIndexerKernels(ds);
        assertFalse(names(ds).contains(HadamardSpec.KERNEL));
        assertFalse(names(ds).contains(SparseKernels.TOPK_ROWS));
        assertEquals(2, all(ds, Dsv4Kernels.STATE_UPDATE).size());
        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        assertEquals(0, attn.intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
        assertEquals(0, attn.intArg(Dsv4Kernels.ARG_ATTN_SEL_STRIDE));
        assertEquals(3, attn.intArg(Dsv4Kernels.ARG_ATTN_N_RAW));
        assertEquals(0, attn.intArg(ATTN_HAS_SINKS));
        assertSame(placeholders.readOnlyFloats(), attn.bufferArg(ATTN_SINKS));

        List<Dispatch> next = decode(m, 3);
        assertEquals(3, only(next, Dsv4Kernels.COMPRESS).intArg(Dsv4Kernels.ARG_COMPRESS_POS0));
        assertEquals(1, only(next, SparseKernels.indexerScore(KvDType.F32)).intArg(SparseKernels.ARG_SCORE_N_KV));
        assertEquals(1, only(next, Dsv4Kernels.COMPRESSED_ATTN).intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
    }

    @Test
    public void csaPrefillSelectsUpToTopKRowsPerToken() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        caches(0, Dsv4Spec.RATIO_CSA);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        List<Dispatch> ds = prefill(m, 40, UBATCH);

        Dispatch compress = only(ds, Dsv4Kernels.COMPRESS);
        assertEquals(40, compress.intArg(Dsv4Kernels.ARG_COMPRESS_POS0));
        assertEquals(UBATCH, compress.intArg(Dsv4Kernels.ARG_COMPRESS_N_TOKENS));
        assertArrayEquals(Dsv4Kernels.compressGlobal(40, UBATCH, Dsv4Spec.RATIO_CSA, dsv4Local), compress.global());
        Dispatch score = only(ds, SparseKernels.indexerScore(KvDType.F32));
        assertEquals(UBATCH, score.intArg(SparseKernels.ARG_SCORE_N_TOKENS));
        assertEquals(12, score.intArg(SparseKernels.ARG_SCORE_N_KV));
        assertEquals(TOP_K, only(ds, SparseKernels.TOPK_ROWS).intArg(SparseKernels.ARG_TOPK_K));
        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        assertEquals(UBATCH, attn.intArg(Dsv4Kernels.ARG_ATTN_N_TOKENS));
        assertEquals(TOP_K, attn.intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
        assertEquals(TOP_K, attn.intArg(Dsv4Kernels.ARG_ATTN_SEL_STRIDE));
        assertEquals(RAW_ROWS, attn.intArg(Dsv4Kernels.ARG_ATTN_N_RAW));
        assertArrayEquals(Dsv4Kernels.attnGlobal(UBATCH, N_HEAD, headTile(), dsv4Local), attn.global());
        for (Dispatch rope : all(ds, RopeKernels.ROPE_EXT)) {
            assertEquals(UBATCH, rope.intArg(RopeKernels.ARG_N_TOKENS));
            assertEquals(UBATCH, rope.intArg(RopeKernels.ARG_POS_STRIDE));
        }
    }

    /**
     * With rollback snapshots ({@code n_rs_seq}) each compressor writes its snapshot planes after the compress /
     * LID-update dispatch has read the live plane and before the state update overwrites it, as llama.cpp orders
     * compress, snapshot and persist (deepseek4.cpp:982-1170, llama-kv-cache-dsv4.cpp:684-713).
     */
    @Test
    public void csaPrefillWritesRollbackSnapshotsBetweenCompressAndStateUpdate() {
        final int snapshots = 3;
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        caches(0, Dsv4Spec.RATIO_CSA, KvDType.F32, RotationPlan.NONE, snapshots);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        List<Dispatch> ds = prefill(m, 40, UBATCH);

        List<String> order = new ArrayList<>();
        for (String n : names(ds)) {
            if (n.equals(Dsv4Kernels.COMPRESS) || n.equals(Dsv4Kernels.lidUpdate(KvDType.F32))
                    || n.equals(Dsv4Kernels.STATE_SNAPSHOT) || n.equals(Dsv4Kernels.STATE_UPDATE)) {
                order.add(n);
            }
        }
        assertEquals(Arrays.asList(
                Dsv4Kernels.COMPRESS, Dsv4Kernels.STATE_SNAPSHOT, Dsv4Kernels.STATE_UPDATE,
                Dsv4Kernels.lidUpdate(KvDType.F32), Dsv4Kernels.STATE_SNAPSHOT, Dsv4Kernels.STATE_UPDATE), order);

        FakeCompressed csa = compressed(0, CompressedKvPlan.Kind.CSA);
        FakeCompressed lid = compressed(0, CompressedKvPlan.Kind.INDEXER);
        List<Dispatch> snaps = all(ds, Dsv4Kernels.STATE_SNAPSHOT);
        FakeCompressed[] owners = {csa, lid};
        int[] widths = {HEAD, IDX_DIM};
        for (int i = 0; i < 2; i++) {
            Dispatch d = snaps.get(i);
            assertEquals(Dsv4Kernels.SNAPSHOT_ARG_COUNT, d.args().size());
            assertEquals(40, d.intArg(Dsv4Kernels.ARG_SNAPSHOT_POS0));
            assertEquals(UBATCH, d.intArg(Dsv4Kernels.ARG_SNAPSHOT_N_TOKENS));
            assertEquals(Dsv4Spec.RATIO_CSA, d.intArg(2));
            assertEquals(widths[i], d.intArg(3));
            assertEquals(snapshots, d.intArg(Dsv4Kernels.ARG_SNAPSHOT_N_SNAP));
            assertSame(owners[i].stateKv, d.bufferArg(10));
            assertSame(owners[i].stateScore, d.bufferArg(11));
            assertArrayEquals(Dsv4Kernels.stateSnapshotGlobal(Dsv4Spec.RATIO_CSA, widths[i], snapshots, dsv4Local),
                    d.global());
        }

        List<Dispatch> next = decode(m, 48);
        assertEquals(48, all(next, Dsv4Kernels.STATE_SNAPSHOT).get(0).intArg(Dsv4Kernels.ARG_SNAPSHOT_POS0));
        assertEquals(1, all(next, Dsv4Kernels.STATE_SNAPSHOT).get(1).intArg(Dsv4Kernels.ARG_SNAPSHOT_N_TOKENS));
    }

    @Test
    public void compressorTensorsWithoutTheSnapshotPlanesAreRefusedAtBind() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_HCA, false);
        memory.kv.put(0, new FakeKv(0, KvDType.F32, RotationPlan.NONE));
        FakeCompressed full = new FakeCompressed(0, CompressedKvPlan.Kind.HCA, KvDType.F32, 2);
        GpuBuffer onePlane = backend.allocateBuffer(full.plan.stateTensorBytes() / full.plan.planes());
        GpuCompressedKv shortPlanes = new GpuCompressedKv() {
            @Override public int layer() { return 0; }
            @Override public CompressedKvPlan plan() { return full.plan; }
            @Override public GpuBuffer cache() { return full.cache; }
            @Override public GpuBuffer stateKv() { return onePlane; }
            @Override public GpuBuffer stateScore() { return full.stateScore; }
        };
        memory.compressed.put(0, Collections.singletonList(shortPlanes));
        LayerSpec l = layer(0, Dsv4Spec.RATIO_HCA);
        GpuForwardContext ctx = context(Collections.singletonList(l));
        GpuMixer m = Dsv4Blocks.gpuMixers().get(0).create(l);
        mixers.add(m);
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> m.bind(ctx, l));
        assertTrue(e.getMessage(), e.getMessage().contains("3 planes"));
    }

    // ------------------------------------------------------------------ HCA

    @Test
    public void hcaPrefillCompressesTheCompletedBlockAndAttendsDensely() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_HCA, false);
        caches(0, Dsv4Spec.RATIO_HCA);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_HCA));

        List<Dispatch> early = prefill(m, 0, UBATCH);
        assertFalse(names(early).contains(Dsv4Kernels.COMPRESS));
        assertEquals(0, only(early, Dsv4Kernels.COMPRESSED_ATTN).intArg(Dsv4Kernels.ARG_ATTN_N_COMP));

        List<Dispatch> ds = prefill(m, 124, UBATCH);
        FakeCompressed hca = compressed(0, CompressedKvPlan.Kind.HCA);
        Dispatch compress = only(ds, Dsv4Kernels.COMPRESS);
        assertEquals(124, compress.intArg(Dsv4Kernels.ARG_COMPRESS_POS0));
        assertEquals(Dsv4Spec.RATIO_HCA, compress.intArg(COMPRESS_RATIO));
        assertArrayEquals(new long[]{dsv4Local}, compress.global());
        Dispatch state = only(ds, Dsv4Kernels.STATE_UPDATE);
        assertEquals(Dsv4Spec.RATIO_HCA, state.intArg(STATE_RATIO));
        assertArrayEquals(Dsv4Kernels.stateUpdateGlobal(Dsv4Spec.RATIO_HCA, HEAD, dsv4Local), state.global());
        assertNoIndexerKernels(ds);

        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        assertEquals(Dsv4Kernels.CompMode.DENSE.code(), attn.intArg(ATTN_COMP_MODE));
        assertEquals(1, attn.intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
        assertEquals(0, attn.intArg(Dsv4Kernels.ARG_ATTN_SEL_STRIDE));
        assertSame(hca.cache, attn.bufferArg(ATTN_COMP_K));
        assertSame(placeholders.readOnlyInts(), attn.bufferArg(ATTN_SEL));
    }

    // ------------------------------------------------------------------ window-only

    @Test
    public void windowOnlyLayerUsesTheModelRopeAndNoCompressedKeys() {
        open();
        layerTensors(0, 0, false);
        caches(0, 0);
        GpuMixer m = bind(layer(0, 0));
        List<Dispatch> ds = decode(m, 5);

        assertEquals(Arrays.asList(
                MATVEC, NormKernels.ROW_NORM, MATVEC, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT,
                MATVEC, NormKernels.ROW_NORM, RopeKernels.ROPE_EXT,
                Dsv4Kernels.CACHE_STORE, Dsv4Kernels.COMPRESSED_ATTN, ID_MATVEC, MATVEC), names(ds));
        List<Dispatch> norms = all(ds, NormKernels.ROW_NORM);
        assertEquals(N_HEAD, norms.get(1).intArg(NormKernels.ARG_ROW_NORM_N_ROWS));
        assertEquals(HEAD, norms.get(1).intArg(NormKernels.ARG_ROW_NORM_ROW_LEN));
        assertEquals(0, norms.get(1).intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        assertEquals(NormKernels.FLAG_HAS_W, norms.get(0).intArg(NormKernels.ARG_ROW_NORM_FLAGS));
        for (Dispatch rope : all(ds, RopeKernels.ROPE_EXT)) {
            assertEquals(HEAD - N_ROT, rope.intArg(ROPE_N_OFFS));
            assertEquals(theta(BASE), rope.floatArg(ROPE_THETA), 0f);
            assertEquals(1f, rope.floatArg(10), 0f);
            assertEquals(0f, rope.floatArg(11), 0f);
            assertEquals(1f, rope.floatArg(12), 0f);
        }

        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        assertEquals(Dsv4Kernels.CompMode.NONE.code(), attn.intArg(ATTN_COMP_MODE));
        assertEquals(0, attn.intArg(Dsv4Kernels.ARG_ATTN_N_COMP));
        assertEquals(theta(BASE), attn.floatArg(ATTN_THETA), 0f);
        assertEquals(1f, attn.floatArg(ATTN_FREQ_SCALE), 0f);
        assertEquals(0f, attn.floatArg(ATTN_EXT), 0f);
        assertEquals(1f, attn.floatArg(ATTN_FACTOR), 0f);
        assertSame(placeholders.readOnlyFloats(), attn.bufferArg(ATTN_COMP_K));
        assertSame(placeholders.readOnlyInts(), attn.bufferArg(ATTN_SEL));
    }

    @Test
    public void groupedOutputRunsOneSlotPerTokenAndGroup() {
        open();
        layerTensors(0, 0, false);
        caches(0, 0);
        GpuMixer m = bind(layer(0, 0));
        List<Dispatch> ds = prefill(m, 0, UBATCH);

        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        Dispatch id = only(ds, ID_MATVEC);
        assertEquals(O_LORA, id.intArg(0));
        assertEquals(O_GROUP_DIM, id.intArg(1));
        assertEquals(UBATCH * N_GROUPS, id.intArg(MoeKernels.ARG_ID_N_SEL));
        assertEquals(O_GROUP_DIM, id.intArg(MoeKernels.ARG_ID_X_STRIDE));
        assertEquals(0, id.intArg(MoeKernels.ARG_ID_IN_OFF));
        assertSame(weight(0, Dsv4Layout.WO_A).shard(0), id.bufferArg(MoeKernels.ARG_ID_WEIGHT));
        assertSame(attn.bufferArg(ATTN_OUT), id.bufferArg(MoeKernels.ARG_ID_INPUT));
        int[] slots = id.bufferArg(MoeKernels.ARG_ID_SLOTS).getInts(0, UBATCH * N_GROUPS);
        for (int s = 0; s < slots.length; s++) {
            assertEquals(s % N_GROUPS, slots[s]);
        }
        Dispatch wo = ds.get(ds.size() - 1);
        assertTrue(wo.bufferArgs().contains(id.bufferArg(MoeKernels.ARG_ID_OUTPUT)));
        assertTrue(wo.bufferArgs().contains(weight(0, Dsv4Layout.WO_B).shard(0)));
        assertTrue(wo.bufferArgs().contains(scratch.slot(ScratchSlots.BRANCH).buffer()));
    }

    // ------------------------------------------------------------------ quantized caches

    @Test
    public void quantizedCachesShareOneHadamardBlock() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        RotationPlan rot = RotationPlan.of(KvDType.Q8_0, KvDType.Q8_0, HEAD, 0, HEAD, 0, true, false);
        caches(0, Dsv4Spec.RATIO_CSA, KvDType.Q8_0, rot);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        List<Dispatch> ds = decode(m, 11);

        assertEquals(rot.nRotK(), only(ds, Dsv4Kernels.CACHE_STORE).intArg(STORE_ROT));
        assertEquals(rot.nRotK(), only(ds, Dsv4Kernels.COMPRESS).intArg(COMPRESS_ROT));
        assertEquals(KvDType.Q8_0.code(), only(ds, Dsv4Kernels.COMPRESS).intArg(14));
        Dispatch attn = only(ds, Dsv4Kernels.COMPRESSED_ATTN);
        assertEquals(rot.nRotK(), attn.intArg(ATTN_ROT));
        assertEquals(KvDType.Q8_0.code(), attn.intArg(ATTN_RAW_TYPE));
        assertEquals(KvDType.Q8_0.code(), attn.intArg(ATTN_COMP_TYPE));
        Dispatch lidUpdate = only(ds, Dsv4Kernels.lidUpdate(KvDType.Q8_0));
        assertEquals(Dsv4Layout.lidRotationBlock(IDX_DIM), lidUpdate.intArg(LID_ROT));
        FakeCompressed lid = compressed(0, CompressedKvPlan.Kind.INDEXER);
        assertSame(lid.cache, lidUpdate.bufferArg(LID_CACHE));
        assertSame(lid.cache, only(ds, SparseKernels.indexerScore(KvDType.Q8_0)).bufferArg(9));
    }

    /**
     * The indexer cache's type is planned for its own row width, so it may differ from the raw window's: the LID
     * update and the indexer score follow the indexer cache.
     */
    @Test
    public void indexerKernelsFollowTheIndexerCacheType() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        memory.kv.put(0, new FakeKv(0, KvDType.F32, RotationPlan.NONE));
        memory.compressed.put(0, Arrays.<GpuCompressedKv>asList(
                new FakeCompressed(0, CompressedKvPlan.Kind.CSA, KvDType.F32),
                new FakeCompressed(0, CompressedKvPlan.Kind.INDEXER, KvDType.F16)));
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        List<Dispatch> ds = decode(m, 7);

        FakeCompressed lid = compressed(0, CompressedKvPlan.Kind.INDEXER);
        assertSame(lid.cache, only(ds, Dsv4Kernels.lidUpdate(KvDType.F16)).bufferArg(LID_CACHE));
        assertSame(lid.cache, only(ds, SparseKernels.indexerScore(KvDType.F16)).bufferArg(9));
        assertFalse(names(ds).contains(Dsv4Kernels.lidUpdate(KvDType.F32)));
        assertFalse(names(ds).contains(SparseKernels.indexerScore(KvDType.F32)));
    }

    /** No LID update or indexer score variant ran. */
    private static void assertNoIndexerKernels(List<Dispatch> ds) {
        List<String> ran = names(ds);
        for (KvDType type : KvDType.values()) {
            assertFalse(ran.contains(Dsv4Kernels.lidUpdate(type)));
            assertFalse(ran.contains(SparseKernels.indexerScore(type)));
        }
    }

    // ------------------------------------------------------------------ errors

    @Test
    public void bindRejectsACompressedCacheOfAnotherTypeThanTheRawWindow() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_HCA, false);
        memory.kv.put(0, new FakeKv(0, KvDType.F32, RotationPlan.NONE));
        memory.compressed.put(0, Collections.<GpuCompressedKv>singletonList(
                new FakeCompressed(0, CompressedKvPlan.Kind.HCA, KvDType.F16)));
        LayerSpec l = layer(0, Dsv4Spec.RATIO_HCA);
        GpuForwardContext ctx = context(Collections.singletonList(l));
        GpuMixer m = Dsv4Blocks.gpuMixers().get(0).create(l);
        mixers.add(m);
        assertThrows(IllegalStateException.class, () -> m.bind(ctx, l));
    }

    @Test
    public void bindRejectsMissingCachesAndWrongShapes() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        memory.kv.put(0, new FakeKv(0, KvDType.F32, RotationPlan.NONE));
        memory.compressed.put(0, Collections.<GpuCompressedKv>emptyList());
        LayerSpec l = layer(0, Dsv4Spec.RATIO_CSA);
        GpuForwardContext ctx = context(Collections.singletonList(l));
        GpuMixer m = Dsv4Blocks.gpuMixers().get(0).create(l);
        mixers.add(m);
        assertThrows(IllegalStateException.class, () -> m.bind(ctx, l));

        caches(0, Dsv4Spec.RATIO_CSA);
        put(0, Dsv4Layout.WO_A, O_GROUP_DIM, O_LORA + 1, N_GROUPS);
        GpuMixer m2 = Dsv4Blocks.gpuMixers().get(0).create(l);
        mixers.add(m2);
        assertThrows(IllegalStateException.class, () -> m2.bind(ctx, l));
    }

    @Test
    public void stepRejectsPositionsBeyondTheCompressedCache() {
        open();
        layerTensors(0, Dsv4Spec.RATIO_CSA, false);
        caches(0, Dsv4Spec.RATIO_CSA);
        GpuMixer m = bind(layer(0, Dsv4Spec.RATIO_CSA));
        int rows = compressed(0, CompressedKvPlan.Kind.CSA).plan.rows();
        int pos = rows * Dsv4Spec.RATIO_CSA + Dsv4Spec.RATIO_CSA - 1;
        memory.steps.upload(pos, 1);
        assertThrows(IllegalStateException.class,
                () -> m.decode(pos, scratch.slot(ScratchSlots.NORMED), scratch.slot(ScratchSlots.BRANCH)));
    }

    @Test
    public void factoryRejectsLayersWithoutTheQLoraPathAndMixersRefuseShifts() {
        open();
        LayerSpec noLora = layer(0, 0, false, 0);
        assertThrows(UnsupportedOperationException.class, () -> Dsv4Blocks.gpuMixers().get(0).create(noLora));
        assertThrows(UnsupportedOperationException.class, () -> Dsv4Blocks.cpuMixers().get(0).create(noLora));

        layerTensors(0, 0, false);
        caches(0, 0);
        GpuMixer m = bind(layer(0, 0));
        assertNotNull(m.shiftRefusal());
        assertThrows(UnsupportedOperationException.class, () -> m.shiftContext(0, 4));
    }

    @Test
    public void familyHasNoFfnsOrExtras() {
        assertEquals(0, Dsv4Blocks.gpuFfns().size());
        assertEquals(0, Dsv4Blocks.cpuFfns().size());
        assertEquals(0, Dsv4Blocks.gpuExtras().size());
        assertEquals(0, Dsv4Blocks.cpuExtras().size());
        assertEquals(MixerKind.DSV4, Dsv4Blocks.gpuMixers().get(0).kind());
        assertEquals(MixerKind.DSV4, Dsv4Blocks.cpuMixers().get(0).kind());
    }

    // ------------------------------------------------------------------ memory fakes

    private final class FakeKv implements GpuKvLayer {
        final int il;
        final KvLayout layout;
        final RotationPlan rotation;
        final GpuBuffer k;

        FakeKv(int il, KvDType type, RotationPlan rotation) {
            this.il = il;
            this.layout = KvLayout.latent(RAW_ROWS, false, HEAD, HEAD, type,
                    KvWindow.of(SwaType.STANDARD, WINDOW, true));
            this.rotation = rotation;
            this.k = backend.allocateBuffer(RAW_ROWS * type.rowBytes(HEAD));
        }

        @Override public int layer() { return il; }
        @Override public KvLayout layout() { return layout; }
        @Override public int group() { return 0; }
        @Override public RotationPlan rotation() { return rotation; }
        @Override public GpuBuffer k() { return k; }
        @Override public GpuBuffer v() { return k; }
    }

    private final class FakeCompressed implements GpuCompressedKv {
        final int il;
        final CompressedKvPlan plan;
        final GpuBuffer cache;
        final GpuBuffer stateKv;
        final GpuBuffer stateScore;

        FakeCompressed(int il, CompressedKvPlan.Kind kind, KvDType type) {
            this(il, kind, type, 0);
        }

        FakeCompressed(int il, CompressedKvPlan.Kind kind, KvDType type, int snapshots) {
            this.il = il;
            this.plan = CompressedKvPlan.of(kind, N_CTX, HEAD, IDX_DIM, type, snapshots);
            this.cache = backend.allocateBuffer(plan.cacheBytes());
            this.stateKv = backend.allocateBuffer(plan.stateTensorBytes());
            this.stateScore = backend.allocateBuffer(plan.stateTensorBytes());
        }

        @Override public int layer() { return il; }
        @Override public CompressedKvPlan plan() { return plan; }
        @Override public GpuBuffer cache() { return cache; }
        @Override public GpuBuffer stateKv() { return stateKv; }
        @Override public GpuBuffer stateScore() { return stateScore; }
    }

    private final class FakeSteps implements GpuStepInputs {
        final GpuBuffer kvPos = backend.allocateBuffer((long) RAW_ROWS * Integer.BYTES);
        final GpuBuffer cellIdx = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        final GpuBuffer qPos = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        final GpuBuffer ropePos = backend.allocateBuffer(4L * UBATCH * Integer.BYTES);
        final GpuBuffer stateRows = backend.allocateBuffer(Integer.BYTES);
        int pos0;
        int nTokens;

        @Override public int groups() { return 1; }
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
        final Map<Integer, List<GpuCompressedKv>> compressed = new HashMap<>();
        final FakeSteps steps = new FakeSteps();

        @Override public GpuKvLayer kv(int il) { return kv.get(il); }
        @Override public GpuKvLayer crossKv(int il) { return null; }
        @Override public GpuRecurrentState recurrent(int il) { return null; }
        @Override public GpuIndexerCache indexer(int il) { return null; }
        @Override public GpuMsaCache msa(int il) { return null; }

        @Override
        public List<GpuCompressedKv> compressed(int il) {
            List<GpuCompressedKv> c = compressed.get(il);
            return c != null ? c : Collections.<GpuCompressedKv>emptyList();
        }

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

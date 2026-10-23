package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuCompressedKv;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Blocks.Dsv4Layout;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.CompMode;
import com.eonmux.neuralfree.gguf.ops.dsv4.Dsv4Kernels.RopeParams;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.sparse.GpuDsaIndexer;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * DeepSeek-V4 attention mixer on the GPU ({@link com.eonmux.neuralfree.gguf.arch.MixerKind#DSV4}), llama.cpp
 * src/models/deepseek4.cpp:886-1229. Per step of {@code n} tokens at {@code pos0}, on the normalized layer input
 * {@code x}:
 * <ol>
 *   <li>{@code qr = RMS(W_q_a . x) * attn_q_a_norm}; {@code q = W_q_b . qr} as {@code [n][n_head][head_dim]}, a
 *       weightless RMS norm per head, RoPE on the last {@code n_rot} dims; {@code kv = RMS(W_kv . x) * attn_kv_norm},
 *       RoPE likewise ({@code LLM_RowNorm}, {@code LLM_RoPE_Ext}; deepseek4.cpp:918-940);</li>
 *   <li>CSA / HCA layers: {@code cur_kv = W_comp_kv . x}, {@code cur_score = W_comp_gate . x}, then
 *       {@code LLM_DSV4_Compress} writes the compressed rows the step completes, {@code LLM_DSV4_StateSnapshot}
 *       writes the rollback snapshot planes when the memory keeps them ({@code n_rs_seq}), and
 *       {@code LLM_DSV4_StateUpdate} persists the step's projections in the live compressor plane
 *       (deepseek4.cpp:953-1158, llama-kv-cache-dsv4.cpp:684-713);</li>
 *   <li>CSA layers: the indexer compressor likewise ({@code LLM_DSV4_LidUpdate_<KV>} into the indexer cache), then
 *       the lightning indexer ({@link GpuDsaIndexer}: queries from {@code qr} with the compress-base RoPE and the
 *       indexer cache's Hadamard rotation, scores over the visible indexer rows, top-k), deepseek4.cpp:596-660;</li>
 *   <li>{@code LLM_DSV4_CacheStore} writes the step's kv rows to the raw window cache at {@code cell_idx},
 *       Hadamard-rotated with the cache's block when K is quantized;</li>
 *   <li>{@code LLM_DSV4_CompressedAttn}: MQA with sinks over the raw window plus the selected (CSA) or all visible
 *       (HCA) compressed rows, rotating q, un-rotating the output and undoing the RoPE on its tail
 *       (deepseek4.cpp:714-860, 1185-1213);</li>
 *   <li>the grouped output: {@code oa[t][g] = W_o_a[g] . out[t][g]} over {@code n_groups} slices of
 *       {@code n_head / n_groups} heads ({@code {T}_MatVecId} with one slot per (token, group)), then
 *       {@code out = W_o_b . oa} (deepseek4.cpp:1215-1224).</li>
 * </ol>
 *
 * <p>Counts per step: {@code n_raw = min(raw rows, pos0 + n)} (rows past it hold no position of this single sequence
 * yet; the kernel masks the rest by {@code kv_pos}); HCA attends {@code n_comp = (p_last + 1) / 128} rows; CSA scores
 * {@code n_lid = (p_last + 1) / 4} indexer rows and attends {@code k = min(n_lid, top_k)} selected rows per token
 * ({@code sel_stride = k}), or none while no indexer row is visible. Each compressed row {@code r} of the indexer cache
 * holds position {@code (r + 1) * 4 - 1} in the mixer's {@value #TEMP_LID_POS} table, so the indexer's causal mask is
 * the compressed visibility {@code r < (p + 1) / 4}. The raw, CSA and HCA caches share one Hadamard block (the
 * layer's {@code RotationPlan}); the indexer cache uses its own forced block.</p>
 *
 * <p>Working buffers are {@code GpuScratch} temporaries reserved at bind; per step only token counts, positions,
 * row counts and global sizes are re-set. One sequence per dispatch: the kernels address the caches by row with no
 * stream offset. Not thread-safe.</p>
 */
public final class GpuDsv4Mixer implements GpuMixer {

    public static final String TEMP_QR = "dsv4.qr";
    public static final String TEMP_Q = "dsv4.q";
    public static final String TEMP_KV = "dsv4.kv";
    public static final String TEMP_CUR_KV = "dsv4.cur.kv";
    public static final String TEMP_CUR_SCORE = "dsv4.cur.score";
    public static final String TEMP_ATTN = "dsv4.attn";
    public static final String TEMP_OA = "dsv4.oa";
    public static final String TEMP_TOPK = "dsv4.topk";
    public static final String TEMP_GROUP_SLOTS = "dsv4.groups.";
    public static final String TEMP_LID_POS = "dsv4.lid.kvpos";

    private final List<PreparedDispatch> prepared = new ArrayList<>();

    private GpuForwardContext ctx;
    private GpuOps ops;
    private Dsv4Layout layout;
    private GpuStepInputs steps;
    private int ubatch;
    private int local;
    private int dsv4Local;
    private int idLocal;
    private int idChunkTokens;
    private int gTile;
    private int rawRows;

    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkv;
    private GpuWeight woB;
    /**
     * LoRA deltas of the projections llama.cpp runs through {@code build_lora_mm} (deepseek4.cpp:918-1039, 1223);
     * the grouped {@code attn_output_a} is a plain {@code ggml_mul_mat} there (:1218) and takes none.
     */
    private GpuProjectionDeltas.Site qaDeltas;
    private GpuProjectionDeltas.Site qbDeltas;
    private GpuProjectionDeltas.Site kvDeltas;
    private GpuProjectionDeltas.Site outDeltas;

    private ScratchSlot qr;
    private ScratchSlot q;
    private ScratchSlot kv;
    private ScratchSlot attn;
    private ScratchSlot oa;
    private ScratchSlot curKv;
    private ScratchSlot curScore;

    private PreparedDispatch qANorm;
    private PreparedDispatch qHeadNorm;
    private PreparedDispatch kvNorm;
    private PreparedDispatch qRope;
    private PreparedDispatch kvRope;
    private PreparedDispatch rawStore;
    private PreparedDispatch attention;
    private PreparedDispatch woA;

    private Compressor compressor;
    private Compressor lidCompressor;
    private GpuDsaIndexer indexer;
    private boolean closed;

    /** Use {@link Dsv4Blocks#gpuMixers()}. */
    GpuDsv4Mixer() {
    }

    /**
     * Binds the mixer: validates the layer's tensors against its spec ({@link Dsv4Layout}) and its caches against
     * the kernels, reserves the working buffers, binds the indexer of a CSA layer and prepares every dispatch.
     *
     * @throws IllegalStateException         when already bound or closed, the context has no memory, or the
     *                                       tensors or caches contradict the spec
     * @throws UnsupportedOperationException when a tensor type has no kernel or a grouped tensor cannot be bound
     */
    @Override
    public void bind(GpuForwardContext context, LayerSpec layer) {
        if (closed || ctx != null) {
            throw new IllegalStateException("DeepSeek-V4 mixer is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(layer, "layer");
        ctx = context;
        try {
            bindLayer(layer);
        } catch (RuntimeException e) {
            releaseAll(e);
            ctx = null;
            throw e;
        }
    }

    private void bindLayer(LayerSpec layer) {
        ops = ctx.ops();
        ubatch = ctx.scratch().ubatch();
        local = ops.local();
        NormKernels.checkReduceLocal(local);
        dsv4Local = Math.min(Dsv4Kernels.LOCAL, local);
        Dsv4Kernels.checkLocal(dsv4Local);
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        layout = Dsv4Layout.of(layer, ctx.spec(), suffix -> shape(weights.layerOrNull(layer, suffix)));
        Dsv4Layout l = layout;
        if (!ctx.hasMemory()) {
            throw new IllegalStateException(l.where + ": the DeepSeek-V4 mixer needs the model's KV memory");
        }
        GpuMemory memory = ctx.memory();
        steps = memory.stepInputs();
        GpuKvLayer raw = memory.kv(l.il);
        if (raw == null) {
            throw new IllegalStateException(l.where + ": no raw window cache for the DeepSeek-V4 layer");
        }
        l.checkRawLayout(raw.layout());
        rawRows = raw.layout().rows();
        KvDType rawType = raw.layout().dtypeK();
        int rot = raw.rotation().rotateK() ? raw.rotation().nRotK() : 0;
        Dsv4Kernels.checkCacheStore(l.headDim, rawType, rot, dsv4Local);

        GpuWeight woAW = bindProjections(layer, weights);
        int qWidth = l.nHead * l.headDim;
        qr = temp(TEMP_QR, l.qLora);
        q = temp(TEMP_Q, qWidth);
        kv = temp(TEMP_KV, l.headDim);
        attn = temp(TEMP_ATTN, qWidth);
        oa = temp(TEMP_OA, (long) l.nGroups * l.oLora);
        ScratchSlot slots = groupSlots();
        prepareProjections(layer, weights, raw, rawType, rot);
        prepareOutputGroups(woAW, slots);
        bindCompressedAttention(layer, weights, memory, raw, rawType, rot);
    }

    /**
     * The dense projections with their LoRA delta sites; returns the grouped {@code wo_a}, checked against the
     * id-matvec work-group size.
     */
    private GpuWeight bindProjections(LayerSpec layer, GpuWeights weights) {
        wqA = projection(weights.layer(layer, Dsv4Layout.WQ_A));
        wqB = projection(weights.layer(layer, Dsv4Layout.WQ_B));
        wkv = projection(weights.layer(layer, Dsv4Layout.WKV));
        woB = projection(weights.layer(layer, Dsv4Layout.WO_B));
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        qaDeltas = deltas.serve(layer, Dsv4Layout.WQ_A, wqA);
        qbDeltas = deltas.serve(layer, Dsv4Layout.WQ_B, wqB);
        kvDeltas = deltas.serve(layer, Dsv4Layout.WKV, wkv);
        outDeltas = deltas.serve(layer, Dsv4Layout.WO_B, woB);
        GpuWeight woAW = perGroup(weights.layer(layer, Dsv4Layout.WO_A));
        idLocal = Math.min(local, MoeKernels.MAX_LOCAL);
        MoeKernels.checkIdLocal(woAW.type(), idLocal);
        idChunkTokens = idChunkTokens(ctx.kernels().caps().getMaxWorkGroupCount()[1]);
        return woAW;
    }

    /** The q / per-head / kv norms, the q and kv RoPE and the raw-window store. */
    private void prepareProjections(LayerSpec layer, GpuWeights weights, GpuKvLayer raw, KvDType rawType, int rot) {
        Dsv4Layout l = layout;
        int qWidth = l.nHead * l.headDim;
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        qANorm = prepareNorm(vector(weights.layer(layer, Dsv4Layout.Q_A_NORM), l.qLora).shard(0), qr.buffer(),
                ubatch, l.qLora);
        qHeadNorm = prepareNorm(null, q.buffer(), ubatch * l.nHead, l.headDim);
        kvNorm = prepareNorm(vector(weights.layer(layer, Dsv4Layout.KV_NORM), l.headDim).shard(0), kv.buffer(),
                ubatch, l.headDim);
        RopeSpec r = l.attnRope;
        GpuBuffer ropePos = steps.ropePos();
        qRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(ubatch, l.nHead), new long[]{local},
                RopeKernels.ropeExtArgs(r, ubatch, l.nHead, l.headDim, qWidth, l.nope, ubatch, 0, false, ropePos, ph,
                        q.buffer()));
        kvRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(ubatch, 1), new long[]{local},
                RopeKernels.ropeExtArgs(r, ubatch, 1, l.headDim, l.headDim, l.nope, ubatch, 0, false, ropePos, ph,
                        kv.buffer()));
        rawStore = prepare(Dsv4Kernels.CACHE_STORE, Dsv4Kernels.cacheStoreGlobal(ubatch, dsv4Local),
                Dsv4Kernels.local(dsv4Local), new Object[]{ubatch, l.headDim, Dsv4Kernels.kvTypeCode(rawType), rot, 0,
                        l.headDim, kv.buffer(), steps.cellIdx(raw.group()), raw.k()});
    }

    /** The grouped low-rank output projection {@code wo_a} over the attention output. */
    private void prepareOutputGroups(GpuWeight woAW, ScratchSlot slots) {
        Dsv4Layout l = layout;
        woA = prepare(MoeKernels.matVecId(woAW.type()),
                MoeKernels.idMatVecGlobal(l.oLora, ubatch * l.nGroups, idLocal), new long[]{idLocal, 1},
                MoeKernels.matVecIdArgs(l.oLora, l.oGroupDim, ubatch * l.nGroups, l.oGroupDim, 0, 0, false, 0,
                        l.nGroups, slots.buffer(), woAW.shard(0), attn.buffer(),
                        ctx.placeholders().readOnlyFloats(), oa.buffer()));
    }

    /**
     * The KV compressor and compressed cache of a compressing layer, the lightning indexer and its own compressor
     * of a selecting layer, then the attention over the raw window and whatever the layer compresses.
     */
    private void bindCompressedAttention(LayerSpec layer, GpuWeights weights, GpuMemory memory, GpuKvLayer raw,
                                         KvDType rawType, int rot) {
        Dsv4Layout l = layout;
        GpuBuffer compK = ctx.placeholders().readOnlyFloats();
        GpuBuffer sel = ctx.placeholders().readOnlyInts();
        KvDType compType = null;
        if (l.mode != CompMode.NONE) {
            int curWidth = Math.max(l.stateWidth, l.idxStateWidth);
            curKv = temp(TEMP_CUR_KV, curWidth);
            curScore = temp(TEMP_CUR_SCORE, curWidth);
            GpuCompressedKv comp = compressedCache(memory, l.compressedKind(), rawType);
            compressor = bindCompressor(layer, weights, comp, rot, false);
            compK = comp.cache();
            compType = rawType;
            if (l.mode == CompMode.SELECTED) {
                GpuCompressedKv lid = compressedCache(memory, CompressedKvPlan.Kind.INDEXER, rawType);
                int lidRot = Dsv4Layout.lidRotationBlock(l.idxDim);
                lidCompressor = bindCompressor(layer, weights, lid, lidRot, true);
                ScratchSlot topK = temp(TEMP_TOPK, l.topK);
                bindIndexer(layer, weights, lid, lidRot, topK);
                sel = topK.buffer();
            }
        }
        bindAttention(layer, weights, raw, rawType, rot, compK, compType, sel);
    }

    /** {@code LLM_DSV4_CompressedAttn} over the raw window and the compressed cache, with its head tile. */
    private void bindAttention(LayerSpec layer, GpuWeights weights, GpuKvLayer raw, KvDType rawType, int rot,
                               GpuBuffer compK, KvDType compType, GpuBuffer sel) {
        Dsv4Layout l = layout;
        int smem = ctx.kernels().faSmemFloats();
        gTile = Dsv4Kernels.attnHeadTile(l.headDim, l.nHead, dsv4Local, smem);
        Dsv4Kernels.checkAttn(l.nHead, l.headDim, gTile, dsv4Local, smem, rawType, compType, rot, l.attnParams);
        GpuBuffer sinks = l.sinks ? vector(weights.layer(layer, Dsv4Layout.SINKS), l.nHead).shard(0)
                : ctx.placeholders().readOnlyFloats();
        RopeParams p = l.attnParams;
        int compCode = Dsv4Kernels.kvTypeCode(compType != null ? compType : rawType);
        attention = prepare(Dsv4Kernels.COMPRESSED_ATTN, Dsv4Kernels.attnGlobal(ubatch, l.nHead, gTile, dsv4Local),
                Dsv4Kernels.local(dsv4Local), new Object[]{ubatch, l.nHead, l.headDim, gTile, 0, 0, rawRows,
                        Dsv4Kernels.kvTypeCode(rawType), l.window, l.mode.code(), 0, compCode, l.ratio, 0,
                        l.sinks ? 1 : 0, l.kqScale, rot, p.nRot(), p.mode(), p.thetaScale(), p.freqScale(),
                        p.extFactor(), p.attnFactor(), p.corrLo(), p.corrHi(), q.buffer(), raw.k(),
                        steps.kvPos(raw.group()), compK, sel, sinks, steps.qPos(), attn.buffer()});
    }

    /** A compressed cache of the layer, checked against the layout. */
    private GpuCompressedKv compressedCache(GpuMemory memory, CompressedKvPlan.Kind kind, KvDType rawType) {
        GpuCompressedKv c = memory.compressed(layout.il, kind);
        if (c == null) {
            throw new IllegalStateException(layout.where + ": no " + kind + " compressed cache for the DeepSeek-V4"
                    + " layer");
        }
        layout.checkCompressedPlan(c.plan(), rawType);
        return c;
    }

    /**
     * The CSA / HCA compressor ({@code attn_compressor_*}, {@code LLM_DSV4_Compress}) or the indexer compressor
     * ({@code indexer_compressor_*}, {@code LLM_DSV4_LidUpdate}) of a cache, with its plane update.
     */
    private Compressor bindCompressor(LayerSpec layer, GpuWeights weights, GpuCompressedKv cache, int rotBlock,
                                      boolean lid) {
        Dsv4Layout l = layout;
        CompressedKvPlan plan = cache.plan();
        int ratio = plan.ratio();
        int width = plan.width();
        int sw = plan.stateWidth();
        String kvName = lid ? Dsv4Layout.IDX_COMP_WKV : Dsv4Layout.COMP_WKV;
        String gateName = lid ? Dsv4Layout.IDX_COMP_WGATE : Dsv4Layout.COMP_WGATE;
        GpuWeight wKv = projection(weights.layer(layer, kvName));
        GpuWeight wGate = projection(weights.layer(layer, gateName));
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        GpuProjectionDeltas.Site[] sites = {deltas.serve(layer, kvName, wKv), deltas.serve(layer, gateName, wGate)};
        GpuBuffer ape = vector(weights.layer(layer, lid ? Dsv4Layout.IDX_COMP_APE : Dsv4Layout.COMP_APE),
                (long) ratio * sw).shard(0);
        GpuBuffer norm = vector(weights.layer(layer, lid ? Dsv4Layout.IDX_COMP_NORM : Dsv4Layout.COMP_NORM), width)
                .shard(0);
        RopeParams p = l.compressParams;
        Dsv4Kernels.checkCompress(ratio, width, plan.dtype(), rotBlock, p, dsv4Local);
        long tensorBytes = plan.stateTensorBytes();
        if (cache.stateKv().sizeInBytes() < tensorBytes || cache.stateScore().sizeInBytes() < tensorBytes) {
            throw new IllegalStateException(l.where + ": the " + plan.kind() + " compressor tensors must hold "
                    + plan.planes() + " planes (" + tensorBytes + " bytes), got " + cache.stateKv().sizeInBytes() + "/"
                    + cache.stateScore().sizeInBytes());
        }
        PreparedDispatch compress;
        if (lid) {
            Dsv4Kernels.checkLidCache(plan.dtype(), rotBlock, width);
            compress = prepare(Dsv4Kernels.lidUpdate(plan.dtype()), Dsv4Kernels.local(dsv4Local),
                    Dsv4Kernels.local(dsv4Local),
                    new Object[]{0, ubatch, width, p.nRot(), p.mode(), l.rmsEps, p.thetaScale(), p.freqScale(),
                            p.extFactor(), p.attnFactor(), p.corrLo(), p.corrHi(), rotBlock, 0, 0, cache.stateKv(),
                            cache.stateScore(), curKv.buffer(), curScore.buffer(), ape, norm, cache.cache()});
        } else {
            compress = prepare(Dsv4Kernels.COMPRESS, Dsv4Kernels.local(dsv4Local), Dsv4Kernels.local(dsv4Local),
                    new Object[]{0, ubatch, ratio, width, p.nRot(), p.mode(), l.rmsEps, p.thetaScale(),
                            p.freqScale(), p.extFactor(), p.attnFactor(), p.corrLo(), p.corrHi(), rotBlock,
                            Dsv4Kernels.kvTypeCode(plan.dtype()), 0, 0, cache.stateKv(), cache.stateScore(),
                            curKv.buffer(), curScore.buffer(), ape, norm, cache.cache()});
        }
        int snapshots = cache.snapshots();
        PreparedDispatch snapshot = snapshots == 0 ? null : prepare(Dsv4Kernels.STATE_SNAPSHOT,
                Dsv4Kernels.stateSnapshotGlobal(ratio, width, snapshots, dsv4Local), Dsv4Kernels.local(dsv4Local),
                new Object[]{0, ubatch, ratio, width, snapshots, 0, 0, curKv.buffer(), curScore.buffer(), ape,
                        cache.stateKv(), cache.stateScore()});
        PreparedDispatch state = prepare(Dsv4Kernels.STATE_UPDATE,
                Dsv4Kernels.stateUpdateGlobal(ratio, width, dsv4Local), Dsv4Kernels.local(dsv4Local),
                new Object[]{0, ubatch, ratio, width, 0, 0, curKv.buffer(), curScore.buffer(), ape, cache.stateKv(),
                        cache.stateScore()});
        return new Compressor(plan.kind(), ratio, plan.rows(), wKv, wGate, sites, compress, snapshot, state, lid);
    }

    /**
     * The lightning indexer of a CSA layer: queries from {@code qr} and {@code x}, keys from the indexer cache the
     * indexer compressor fills, row positions from the constant {@value #TEMP_LID_POS} table.
     */
    private void bindIndexer(LayerSpec layer, GpuWeights weights, GpuCompressedKv lid, int lidRot, ScratchSlot topK) {
        Dsv4Layout l = layout;
        int lidRows = lid.plan().rows();
        indexer = new GpuDsaIndexer(ctx, l.where, l.indexerConfig(lidRot));
        indexer.bindQueries(weights.layer(layer, Dsv4Layout.IDX_Q_B), weights.layer(layer, Dsv4Layout.IDX_PROJ),
                steps.ropePos());
        indexer.bindSelection(lid.cache(), lid.plan().dtype(), lidRows, lidPositions(lidRows), steps.qPos(), topK);
        indexer.bindDeltas(layer);
    }

    // ------------------------------------------------------------------ steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(pos0, nTokens, in, out);
    }

    /**
     * DeepSeek-V4 compressed caches cannot be shifted: llama.cpp disables it (llama-kv-cache-dsv4.cpp:1452-1456),
     * because each compressed row summarizes a fixed block of positions.
     */
    @Override
    public String shiftRefusal() {
        return "context shift is not supported for DeepSeek-V4 compressed attention (llama-kv-cache-dsv4.cpp:1452)";
    }

    private void forward(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Dsv4Layout l = layout;
        checkStep(pos0, n, in, out);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        int last = pos0 + n - 1;

        ops.matmul(wqA, x, xOff, qr.buffer(), 0, n);
        qaDeltas.applyRows(n, x, xOff, qr.buffer(), 0);
        rowNorm(qANorm, n);
        ops.matmul(wqB, qr.buffer(), 0, q.buffer(), 0, n);
        qbDeltas.applyRows(n, qr.buffer(), 0, q.buffer(), 0);
        rowNorm(qHeadNorm, n * l.nHead);
        rope(qRope, n, l.nHead);
        ops.matmul(wkv, x, xOff, kv.buffer(), 0, n);
        kvDeltas.applyRows(n, x, xOff, kv.buffer(), 0);
        rowNorm(kvNorm, n);
        rope(kvRope, n, 1);

        int nComp = 0;
        int selStride = 0;
        if (compressor != null) {
            compressor.run(pos0, n, x, xOff);
        }
        if (l.mode == CompMode.SELECTED) {
            lidCompressor.run(pos0, n, x, xOff);
            int nLid = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_CSA);
            if (nLid > 0) {
                indexer.queries(n, qr.buffer(), 0, x, xOff);
                nComp = indexer.select(n, nLid);
                selStride = nComp;
            }
        } else if (l.mode == CompMode.DENSE) {
            nComp = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_HCA);
        }

        rawStore.setInt(Dsv4Kernels.ARG_STORE_N_TOKENS, n);
        rawStore.setGlobal(Dsv4Kernels.cacheStoreGlobal(n, dsv4Local));
        rawStore.dispatch();

        attention.setInt(Dsv4Kernels.ARG_ATTN_N_TOKENS, n);
        attention.setInt(Dsv4Kernels.ARG_ATTN_N_RAW, Math.min(rawRows, pos0 + n));
        attention.setInt(Dsv4Kernels.ARG_ATTN_N_COMP, nComp);
        attention.setInt(Dsv4Kernels.ARG_ATTN_SEL_STRIDE, selStride);
        attention.setGlobal(Dsv4Kernels.attnGlobal(n, l.nHead, gTile, dsv4Local));
        attention.dispatch();

        groupedOutput(n);
        ops.matmul(woB, oa.buffer(), 0, out.buffer(), out.offset(), n);
        outDeltas.applyRows(n, oa.buffer(), 0, out.buffer(), out.offset());
    }

    /**
     * {@code oa[t][g] = W_o_a[g] . attn[t][g]} over the {@code n * n_groups} (token, group) pairs, in calls of at
     * most {@link #idChunkTokens} tokens so the pair dimension of the grid stays within the device's work-group
     * count. Every call starts on a token boundary, so pair {@code s} of a call reads group {@code s % n_groups} of
     * the constant slot table.
     */
    private void groupedOutput(int n) {
        Dsv4Layout l = layout;
        for (int t0 = 0; t0 < n; t0 += idChunkTokens) {
            int sel = Math.min(idChunkTokens, n - t0) * l.nGroups;
            int pair0 = t0 * l.nGroups;
            woA.setInt(MoeKernels.ARG_ID_N_SEL, sel);
            woA.setInt(MoeKernels.ARG_ID_IN_OFF, pair0 * l.oGroupDim);
            woA.setInt(MoeKernels.ARG_ID_OUT_OFF, pair0 * l.oLora);
            woA.setGlobal(MoeKernels.idMatVecGlobal(l.oLora, sel, idLocal));
            woA.dispatch();
        }
    }

    private void rowNorm(PreparedDispatch d, int rows) {
        d.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, rows);
        d.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(rows), local));
        d.dispatch();
    }

    private void rope(PreparedDispatch d, int n, int heads) {
        d.setInt(RopeKernels.ARG_N_TOKENS, n);
        d.setInt(RopeKernels.ARG_POS_STRIDE, n);
        d.setGlobal(ropeGlobal(n, heads));
        d.dispatch();
    }

    private void checkStep(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        Dsv4Layout l = layout;
        if (n < 1 || n > ubatch || pos0 < 0) {
            throw new IllegalArgumentException(l.where + ": DeepSeek-V4 step of " + n + " tokens at " + pos0
                    + " is outside [1, " + ubatch + "] tokens at a non-negative position");
        }
        if (steps.pos0() != pos0 || steps.nTokens() != n) {
            throw new IllegalStateException(l.where + ": the step inputs hold " + steps.nTokens() + " tokens at "
                    + steps.pos0() + ", the DeepSeek-V4 step is " + n + " tokens at " + pos0);
        }
        if (compressor != null) {
            l.checkCapacity(pos0, n, compressor.ratio, compressor.rows, compressor.kind.toString());
        }
        if (lidCompressor != null) {
            l.checkCapacity(pos0, n, lidCompressor.ratio, lidCompressor.rows, lidCompressor.kind.toString());
        }
        long width = (long) n * l.nEmbd;
        Objects.requireNonNull(in, "in").require(width);
        Objects.requireNonNull(out, "out").require(width);
    }

    /** One compressor of the layer with its prepared kernels. */
    private final class Compressor {
        final CompressedKvPlan.Kind kind;
        final int ratio;
        final int rows;
        final GpuWeight wKv;
        final GpuWeight wGate;
        /** LoRA deltas of {@code wKv} and {@code wGate} (deepseek4.cpp:952-970, 1036-1039). */
        final GpuProjectionDeltas.Site kvDeltas;
        final GpuProjectionDeltas.Site gateDeltas;
        final PreparedDispatch compress;
        /** {@code LLM_DSV4_StateSnapshot}, null when the memory keeps no rollback snapshots. */
        final PreparedDispatch snapshot;
        final PreparedDispatch state;
        final int argPos0;
        final int argTokens;

        Compressor(CompressedKvPlan.Kind kind, int ratio, int rows, GpuWeight wKv, GpuWeight wGate,
                   GpuProjectionDeltas.Site[] deltas, PreparedDispatch compress, PreparedDispatch snapshot,
                   PreparedDispatch state, boolean lid) {
            this.kind = kind;
            this.ratio = ratio;
            this.rows = rows;
            this.wKv = wKv;
            this.wGate = wGate;
            this.kvDeltas = deltas[0];
            this.gateDeltas = deltas[1];
            this.compress = compress;
            this.snapshot = snapshot;
            this.state = state;
            this.argPos0 = lid ? Dsv4Kernels.ARG_LID_POS0 : Dsv4Kernels.ARG_COMPRESS_POS0;
            this.argTokens = lid ? Dsv4Kernels.ARG_LID_N_TOKENS : Dsv4Kernels.ARG_COMPRESS_N_TOKENS;
        }

        /**
         * Projects the step's kv and score rows, writes the compressed rows the step completes (no dispatch when it
         * completes none), writes the rollback snapshots from the live plane the compress kernel has read, then
         * persists the projections in that plane.
         */
        void run(int pos0, int n, GpuBuffer x, int xOff) {
            ops.matmul(wKv, x, xOff, curKv.buffer(), 0, n);
            kvDeltas.applyRows(n, x, xOff, curKv.buffer(), 0);
            ops.matmul(wGate, x, xOff, curScore.buffer(), 0, n);
            gateDeltas.applyRows(n, x, xOff, curScore.buffer(), 0);
            if (Dsv4Kernels.blockCount(pos0, n, ratio) > 0) {
                compress.setInt(argPos0, pos0);
                compress.setInt(argTokens, n);
                compress.setGlobal(Dsv4Kernels.compressGlobal(pos0, n, ratio, dsv4Local));
                compress.dispatch();
            }
            if (snapshot != null) {
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_POS0, pos0);
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_N_TOKENS, n);
                snapshot.dispatch();
            }
            state.setInt(Dsv4Kernels.ARG_STATE_POS0, pos0);
            state.setInt(Dsv4Kernels.ARG_STATE_N_TOKENS, n);
            state.dispatch();
        }
    }

    // ------------------------------------------------------------------ bind helpers

    /**
     * {@code LLM_RowNorm} RMS over {@code rows} rows of {@code rowLen} floats in place, weighted when {@code w} is
     * set.
     */
    private PreparedDispatch prepareNorm(GpuBuffer w, GpuBuffer x, int rows, int rowLen) {
        return prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(rows), local),
                new long[]{local}, NormKernels.rowNormArgs(NormKind.RMS, layout.rmsEps, rows, rowLen, rowLen, x, 0, x,
                        0, w, null, 1, ctx.placeholders().readOnlyFloats()));
    }

    /**
     * {@value #TEMP_GROUP_SLOTS}{@code <n_groups>}: {@code slots[s] = s % n_groups} for {@code s < ubatch * n_groups},
     * the {@code wo_a} group each (token, group) pair reads. Written once here; every holder of the name writes the
     * same values.
     */
    private ScratchSlot groupSlots() {
        int groups = layout.nGroups;
        int n = ubatch * groups;
        ScratchSlot s = temp(TEMP_GROUP_SLOTS + groups, groups);
        ByteBuffer bytes = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            bytes.putInt(i % groups);
        }
        closeEvent(ctx.queue().enqueueWriteBuffer(s.buffer(), 0, (long) n * Integer.BYTES, bytes.array()));
        return s;
    }

    /**
     * {@value #TEMP_LID_POS}: {@code kv_pos[r] = (r + 1) * 4 - 1} for the {@code rows} indexer rows, the position
     * each compressed row stands for. Written once here; every holder of the name writes the same values.
     */
    private GpuBuffer lidPositions(int rows) {
        ScratchSlot s = ctx.scratch().temp(TEMP_LID_POS, rows);
        if (s.offset() != 0) {
            throw new IllegalStateException(layout.where + ": DeepSeek-V4 buffer " + s + " must start at element 0");
        }
        ByteBuffer bytes = ByteBuffer.allocate(rows * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < rows; r++) {
            bytes.putInt(Dsv4Layout.compressedRowPosition(r, Dsv4Spec.RATIO_CSA));
        }
        closeEvent(ctx.queue().enqueueWriteBuffer(s.buffer(), 0, (long) rows * Integer.BYTES, bytes.array()));
        return s.buffer();
    }

    /**
     * Whole tokens per {@code {T}_MatVecId} call whose {@code n_groups} pairs each fit a grid of at most
     * {@code maxGroupsY} work-groups in its second dimension.
     *
     * @throws IllegalStateException when even one token's groups exceed the limit
     */
    private int idChunkTokens(long maxGroupsY) {
        long tokens = Math.min(Integer.MAX_VALUE, maxGroupsY) / layout.nGroups;
        if (tokens < 1) {
            throw new IllegalStateException(layout.where + ": " + layout.nGroups + " output groups exceed the device's "
                    + maxGroupsY + " work-groups in the second grid dimension");
        }
        return (int) Math.min(tokens, ubatch);
    }

    private GpuWeight projection(GpuWeight w) {
        if (!GpuOps.supportsWeightType(w.type())) {
            throw new UnsupportedOperationException(layout.where + ": no GPU matvec kernels for " + w.name()
                    + " of type " + w.type());
        }
        return w;
    }

    /**
     * The grouped {@code attn_output_a} for the id-matvec: one buffer, a type with id-matvec kernels, and no PRISM
     * transform or output scale, which llama.cpp's plain {@code ggml_mul_mat} of it does not apply either
     * (deepseek4.cpp:1218).
     */
    private GpuWeight perGroup(GpuWeight w) {
        if (!MoeKernels.supports(w.type())) {
            throw new UnsupportedOperationException(layout.where + ": no id-matvec kernels for " + w.name()
                    + " of type " + w.type());
        }
        if (!w.isSingleShard()) {
            throw new UnsupportedOperationException(layout.where + ": grouped tensor " + w.name() + " is split into "
                    + w.shardCount() + " row shards; the id-matvec reads every group from one buffer");
        }
        if (w.hadamard() != null || w.hasScale()) {
            throw new UnsupportedOperationException(layout.where + ": grouped tensor " + w.name()
                    + " carries a PRISM transform or output scale, which the grouped output product does not apply");
        }
        return w;
    }

    private GpuWeight vector(GpuWeight v, long n) {
        if (v.type() != GGMLType.F32 || !v.isSingleShard() || v.cols() * v.rows() != n) {
            throw new IllegalStateException(layout.where + ": " + v.name() + " must be a single-buffer F32 tensor of "
                    + n + " values, got " + v);
        }
        return v;
    }

    /** A temporary of {@code ubatch} rows of {@code width} floats, starting at element 0 of its buffer. */
    private ScratchSlot temp(String name, long width) {
        ScratchSlot s = ctx.scratch().temp(name, (long) ubatch * width);
        if (s.offset() != 0) {
            throw new IllegalStateException(layout.where + ": DeepSeek-V4 buffer " + s + " must start at element 0");
        }
        return s;
    }

    private long[] ropeGlobal(int n, int heads) {
        RopeSpec r = layout.attnRope;
        long items = RopeKernels.ropeExtWorkItems(r.type(), n, heads, r.nDims());
        return new long[]{RopeKernels.roundUp(Math.max(1L, items), local)};
    }

    private PreparedDispatch prepare(String kernel, long[] global, long[] localSize, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, localSize, args);
        prepared.add(d);
        return d;
    }

    private static long[] shape(GpuWeight w) {
        return w == null ? null : new long[]{w.cols(), w.rows()};
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("DeepSeek-V4 mixer is closed");
        }
        if (ctx == null) {
            throw new IllegalStateException("DeepSeek-V4 mixer is not bound");
        }
    }

    // ------------------------------------------------------------------ close

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        releaseAll(null);
    }

    private void releaseAll(RuntimeException cause) {
        RuntimeException failure = null;
        if (indexer != null) {
            try {
                indexer.close();
            } catch (RuntimeException e) {
                failure = e;
            }
            indexer = null;
        }
        failure = Closeables.closeAll(prepared, failure);
        prepared.clear();
        compressor = null;
        lidCompressor = null;
        if (failure != null) {
            if (cause != null) {
                cause.addSuppressed(failure);
            } else {
                throw failure;
            }
        }
    }

    @Override
    public String toString() {
        return "GpuDsv4Mixer{" + (layout != null ? layout.toString() : "unbound") + '}';
    }
}

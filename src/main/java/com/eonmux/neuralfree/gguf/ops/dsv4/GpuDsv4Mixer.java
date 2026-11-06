package com.eonmux.neuralfree.gguf.ops.dsv4;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
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
import com.eonmux.neuralfree.gguf.ops.sparse.SparseKernels;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
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
 * <p>A paged multi-sequence step ({@link #batch}) runs the token-wise projections over every row of the step and
 * stores the raw rows at the step's cells, then compresses, selects and attends segment by segment as llama.cpp runs
 * one DSV4 stream per sequence (llama-kv-cache-dsv4.cpp:51-60, 200-208), the twin of {@code CpuDsv4Mixer.batch}:
 * each segment's compressors fold its rows into its own sequence's compressed stream and planes
 * ({@code GpuMemory.compressed(il, kind, seq)}, rebound per launch), its indexer selects among its own stream's
 * compressed rows ({@link GpuDsaIndexer#selectRows}), and its rows attend every raw cell through the sequence's
 * masked {@code kv_pos} ({@code GpuStepInputs.sequenceKvPos}) at the segment's positions. The rows of a segment must
 * sit at consecutive positions (the compressors fold consecutive blocks) and the attention is causal, so a
 * non-causal step is refused by name.</p>
 *
 * <p>Working buffers are {@code GpuScratch} temporaries reserved at bind; per step only token counts, positions,
 * offsets, the stream buffers, row counts and global sizes are re-set. Not thread-safe.</p>
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
    public static final String TEMP_COMP_POS = "dsv4.comp.kvpos";
    public static final String TEMP_COMP_LATENT = "dsv4.comp.latent";
    public static final String TEMP_COMP_INDEX = "dsv4.comp.index";
    public static final String TEMP_COMP_ROWS = "dsv4.comp.rows";
    public static final String TEMP_BLOCK_CELLS = "dsv4.block.cells";
    public static final String TEMP_BLOCK_POS = "dsv4.block.pos";
    public static final String TEMP_ZERO_APE = "dsv4.zero.ape";
    public static final String TEMP_CAND = "dsv4.cand.blocks";

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
    private int rawGroup;
    /** The raw window group's {@code kv_pos}, which the contiguous step's attention reads. */
    private GpuBuffer rawKvPos;
    /** The {@value #TEMP_LID_POS} table of a CSA layer's indexer rows. */
    private GpuBuffer lidKvPos;
    /** The top-k selection buffer of a CSA layer. */
    private GpuBuffer selection;

    private GpuWeight wqA;
    private GpuWeight wqB;
    private GpuWeight wkv;
    private GpuWeight woB;
    /** The grouped {@code attn_output_a} the id-matvec kernel multiplies, one matrix per output group. */
    private GpuWeight woAW;
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

    // ---- DeepSeek-V4.1: shared compressed rows, the two-level candidate selection and the index key ----
    private boolean v41;
    private int compReadLayer;
    private int compStride;
    private int compRows;
    /** The compressed stream of each sequence of the layer's KV source, indexed by sequence. */
    private GpuCompressedKv[] readStreams;
    /** Positions held by the compressed rows of the read stream ({@code (r + 1) * ratio - 1}). */
    private GpuBuffer compParamPos;
    /** The top-k selection of the layer's indexer source, shared by every consumer of that source. */
    private ScratchSlot v41TopK;
    /** Candidate blocks of the layer's candidate source, shared by every consumer of that source. */
    private ScratchSlot candBlocks;
    private boolean picksCandidate;
    private boolean usesCandidate;
    private int candTopKBlocks;
    private int candBlockSize;

    private GpuWeight idxK;
    private GpuProjectionDeltas.Site idxKDeltas;
    private GpuBuffer idxKNormW;
    private ScratchSlot compLatent;
    private ScratchSlot compIndex;
    private ScratchSlot compRowsBuf;
    private ScratchSlot blockCells;
    /** Block positions and of the completed blocks of one fold, the rope positions of the latent and index key. */
    private ScratchSlot blockPositions;
    private GpuBuffer zeroApe;
    private HadamardSpec latentRotation;
    private int rawRot;
    private PreparedDispatch latentV41;
    private PreparedDispatch indexNorm;
    private PreparedDispatch indexRope;
    private PreparedDispatch latentRope;
    private PreparedDispatch compStore;

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
        v41 = l.v41 != null;
        if (v41) {
            // a window-only V4.1 layer reads no compressed rows; a compressed layer's rows hold latent + index key
            compStride = l.ratio == 0 ? l.headDim : l.compWidth;
        }
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
        rawGroup = raw.group();
        rawKvPos = steps.kvPos(rawGroup);
        steps.reserveSequenceAddressing(rawGroup);
        KvDType rawType = raw.layout().dtypeK();
        int rot = raw.rotation().rotateK() ? raw.rotation().nRotK() : 0;
        Dsv4Kernels.checkCacheStore(l.headDim, rawType, rot, dsv4Local);

        woAW = bindProjections(layer, weights);
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
        int compStride = l.headDim;
        if (l.mode != CompMode.NONE) {
            int curWidth = Math.max(l.stateWidth, l.idxStateWidth);
            curKv = temp(TEMP_CUR_KV, curWidth);
            curScore = temp(TEMP_CUR_SCORE, curWidth);
            if (v41) {
                bindV41(layer, weights, raw, rawType, rot);
                return;
            }
            GpuCompressedKv[] comp = compressedStreams(memory, l.compressedKind(), rawType);
            compressor = bindCompressor(layer, weights, comp, rot, false);
            compK = comp[0].cache();
            compType = rawType;
            if (l.mode == CompMode.SELECTED) {
                GpuCompressedKv[] lid = compressedStreams(memory, CompressedKvPlan.Kind.INDEXER, rawType);
                int lidRot = Dsv4Layout.lidRotationBlock(l.idxDim);
                lidCompressor = bindCompressor(layer, weights, lid, lidRot, true);
                ScratchSlot topK = temp(TEMP_TOPK, l.topK);
                bindIndexer(layer, weights, lid[0], lidRot, topK);
                sel = topK.buffer();
                selection = sel;
            }
        }
        bindAttention(layer, weights, raw, rawType, rot, compK, compType, compStride, sel);
    }

    /** {@code LLM_DSV4_CompressedAttn} over the raw window and the compressed cache, with its head tile. */
    private void bindAttention(LayerSpec layer, GpuWeights weights, GpuKvLayer raw, KvDType rawType, int rot,
                               GpuBuffer compK, KvDType compType, int compStride, GpuBuffer sel) {
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
                        Dsv4Kernels.kvTypeCode(rawType), l.window, l.attnMode.code(), 0, compCode, l.ratio, 0,
                        l.sinks ? 1 : 0, l.kqScale, rot, p.nRot(), p.mode(), p.thetaScale(), p.freqScale(),
                        p.extFactor(), p.attnFactor(), p.corrLo(), p.corrHi(), 0, compStride, q.buffer(), raw.k(),
                        steps.kvPos(raw.group()), compK, sel, sinks, steps.qPos(), attn.buffer()});
    }

    /**
     * The compressed cache of one kind of every sequence the memory holds ({@code GpuStepInputs.pagedMaxSequences},
     * one stream each), checked against the layout.
     */
    private GpuCompressedKv[] compressedStreams(GpuMemory memory, CompressedKvPlan.Kind kind, KvDType rawType) {
        GpuCompressedKv[] streams = new GpuCompressedKv[steps.pagedMaxSequences()];
        for (int seq = 0; seq < streams.length; seq++) {
            GpuCompressedKv c = memory.compressed(layout.il, kind, seq);
            if (c == null) {
                throw new IllegalStateException(layout.where + ": no " + kind + " compressed cache of sequence " + seq
                        + " for the DeepSeek-V4 layer");
            }
            layout.checkCompressedPlan(c.plan(), rawType);
            streams[seq] = c;
        }
        return streams;
    }

    /**
     * The CSA / HCA compressor ({@code attn_compressor_*}, {@code LLM_DSV4_Compress}) or the indexer compressor
     * ({@code indexer_compressor_*}, {@code LLM_DSV4_LidUpdate}) of a cache, with its plane update.
     */
    private Compressor bindCompressor(LayerSpec layer, GpuWeights weights, GpuCompressedKv[] streams, int rotBlock,
                                      boolean lid) {
        Dsv4Layout l = layout;
        GpuCompressedKv cache = streams[0];
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
        for (GpuCompressedKv c : streams) {
            if (c.stateKv().sizeInBytes() < tensorBytes || c.stateScore().sizeInBytes() < tensorBytes
                    || c.snapshots() != cache.snapshots()) {
                throw new IllegalStateException(l.where + ": the " + plan.kind() + " compressor tensors must hold "
                        + plan.planes() + " planes (" + tensorBytes + " bytes), got " + c.stateKv().sizeInBytes() + "/"
                        + c.stateScore().sizeInBytes());
            }
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
        return new Compressor(plan.kind(), ratio, plan.rows(), sw, streams, wKv, wGate, sites, compress, snapshot,
                state, lid);
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
                steps.ropePos(), null);
        lidKvPos = lidPositions(lidRows);
        indexer.bindSelection(lid.cache(), lid.plan().dtype(), lidRows, lidKvPos, steps.qPos(), topK);
        indexer.bindDeltas(layer);
        if (steps.pageShift() >= 0) {
            indexer.enableSegments();
        }
    }

    /**
     * The DeepSeek-V4.1 compressed attention of a layer (models/deepseek41.cpp:456-631): the compressed caches of its
     * KV source layer, its own compressor and index-key tensors when it is that source, its indexer when it computes
     * the top-k, and the shared candidate-block plane of the two-level selection. Consumers read the source layer's
     * streams and its top-k plane; every compressed layer attends its selected rows.
     */
    private void bindV41(LayerSpec layer, GpuWeights weights, GpuKvLayer raw, KvDType rawType, int rot) {
        Dsv4Layout l = layout;
        compReadLayer = l.v41.kvSourceLayer();
        CompressedKvPlan.Kind kind = l.compressedKind();
        int streams = steps.pagedMaxSequences();
        readStreams = new GpuCompressedKv[streams];
        CompressedKvPlan plan = null;
        for (int seq = 0; seq < streams; seq++) {
            GpuCompressedKv c = ctx.memory().compressed(compReadLayer, kind, seq);
            if (c == null) {
                throw new IllegalStateException(l.where + ": no " + kind + " compressed cache of KV source layer "
                        + compReadLayer + " for sequence " + seq);
            }
            l.checkCompressedPlan(c.plan(), rawType);
            if (plan == null) {
                plan = c.plan();
            } else if (c.plan().rows() != plan.rows() || c.plan().width() != plan.width()
                    || c.plan().dtype() != plan.dtype()) {
                throw new IllegalStateException(l.where + ": the " + kind + " streams of KV source layer "
                        + compReadLayer + " disagree: " + plan + " and " + c.plan());
            }
            if (c.cache().sizeInBytes() < c.plan().cacheBytes()) {
                throw new IllegalStateException(l.where + ": the " + kind + " cache of KV source layer "
                        + compReadLayer + " sequence " + seq + " is smaller than its " + c.plan().cacheBytes()
                        + "-byte plan");
            }
            readStreams[seq] = c;
        }
        compRows = plan.rows();
        compStride = plan.width();
        if (compStride != l.compWidth) {
            throw new IllegalStateException(l.where + ": the " + kind + " rows of KV source layer " + compReadLayer
                    + " are " + compStride + " values wide, the layer stores " + l.compWidth);
        }
        compParamPos = compressedPositions(compReadLayer, compRows, l.ratio);
        v41TopK = temp(TEMP_TOPK + ".src." + l.v41.topkSourceLayer(), l.topK);
        int cand = l.v41.candidateSourceLayer();
        if (cand >= 0) {
            candTopKBlocks = l.v41.candidateTopKBlocks();
            candBlockSize = l.v41.candidateBlockSize();
            picksCandidate = l.computesTopk && l.v41.picksCandidateBlocks();
            usesCandidate = l.computesTopk && l.v41.usesCandidateBlocks();
            if (picksCandidate || usesCandidate) {
                candBlocks = temp(TEMP_CAND + ".src." + cand, candTopKBlocks);
            }
        }
        if (l.computesTopk) {
            indexer = new GpuDsaIndexer(ctx, l.where, l.indexerConfig(0));
            indexer.bindQueries(weights.layer(layer, Dsv4Layout.IDX_Q_B),
                    weights.layer(layer, Dsv4Layout.IDX_PROJ), steps.ropePos(), null);
            indexer.bindSelection(readStreams[0].cache(), plan.dtype(), compRows, compStride, l.headDim,
                    compParamPos, steps.qPos(), v41TopK);
            indexer.bindDeltas(layer);
            indexer.enableSegments();
            if (picksCandidate || usesCandidate) {
                indexer.bindBlocks();
            }
        }
        if (l.ownsCompressed) {
            bindV41Compressor(layer, weights, plan);
        }
        bindAttention(layer, weights, raw, rawType, rot, readStreams[0].cache(), plan.dtype(), compStride,
                v41TopK.buffer());
    }

    /**
     * The DeepSeek-V4.1 compressor of a KV source layer ({@code LLM_DSV4_CompressV41}): pooled, normed latents
     * without rope or rotation, the index key of each latent ({@code indexer.attn_k}, its norm and rope), the latent
     * rope and cache rotation, then the combined {@code [latent | index key]} row through
     * {@code LLM_DSV4_CacheStore}. The state update and rollback snapshots are the ones of a non-overlapped
     * ratio-1/2 compressor.
     */
    private void bindV41Compressor(LayerSpec layer, GpuWeights weights, CompressedKvPlan plan) {
        Dsv4Layout l = layout;
        int ratio = plan.ratio();
        int sw = plan.stateWidth();
        GpuWeight wKv = projection(weights.layer(layer, Dsv4Layout.COMP_WKV));
        GpuWeight wGate = ratio > 1 ? projection(weights.layer(layer, Dsv4Layout.COMP_WGATE)) : null;
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        GpuProjectionDeltas.Site[] sites = {deltas.serve(layer, Dsv4Layout.COMP_WKV, wKv),
                wGate != null ? deltas.serve(layer, Dsv4Layout.COMP_WGATE, wGate) : null};
        GpuBuffer norm = vector(weights.layer(layer, Dsv4Layout.COMP_NORM), l.headDim).shard(0);
        Dsv4Kernels.checkWidth(l.headDim, plan.dtype());
        Dsv4Kernels.checkRope(l.compressParams, l.headDim);
        // the combined [latent | index key] row is what the cache store writes, in the cache's element type
        Dsv4Kernels.checkCacheStore(l.compWidth, plan.dtype(), 0, dsv4Local);
        long tensorBytes = plan.stateTensorBytes();
        for (GpuCompressedKv c : readStreams) {
            if (c.stateKv().sizeInBytes() < tensorBytes || c.stateScore().sizeInBytes() < tensorBytes) {
                throw new IllegalStateException(l.where + ": the " + plan.kind() + " compressor tensors must hold "
                        + plan.planes() + " planes (" + tensorBytes + " bytes), got " + c.stateKv().sizeInBytes()
                        + "/" + c.stateScore().sizeInBytes());
            }
        }
        zeroApe = zeroFloats(TEMP_ZERO_APE, (long) ratio * l.headDim);
        compLatent = temp(TEMP_COMP_LATENT, l.headDim);
        compIndex = temp(TEMP_COMP_INDEX, l.idxDim);
        compRowsBuf = temp(TEMP_COMP_ROWS, l.compWidth);
        blockCells = tempInt(TEMP_BLOCK_CELLS, ubatch);
        blockPositions = tempInt(TEMP_BLOCK_POS, ubatch);
        latentRotation = rawRot > 0 ? HadamardSpec.rotation(rawRot) : null;
        latentV41 = prepare(Dsv4Kernels.COMPRESS_V41, Dsv4Kernels.local(dsv4Local),
                Dsv4Kernels.local(dsv4Local),
                new Object[]{0, ubatch, ratio, l.headDim, l.rmsEps, 0, 0, readStreams[0].stateKv(),
                        readStreams[0].stateScore(), curKv.buffer(), curScore.buffer(), zeroApe, norm,
                        compLatent.buffer()});
        idxK = projection(weights.layer(layer, Dsv4Layout.IDX_K));
        idxKNormW = vector(weights.layer(layer, Dsv4Layout.IDX_K_NORM), l.idxDim).shard(0);
        idxKDeltas = deltas.serve(layer, Dsv4Layout.IDX_K, idxK);
        indexNorm = prepare(NormKernels.ROW_NORM, NormKernels.reduceGlobal(NormKernels.rowNormGroups(ubatch), local),
                new long[]{local}, NormKernels.rowNormArgs(NormKind.RMS, l.rmsEps, ubatch, l.idxDim, l.idxDim,
                        compIndex.buffer(), 0, compIndex.buffer(), 0, idxKNormW, null, 1,
                        ctx.placeholders().readOnlyFloats()));
        indexRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(l.indexerRope(), ubatch, 1), new long[]{local},
                RopeKernels.ropeExtArgs(l.indexerRope(), ubatch, 1, l.idxDim, l.idxDim, l.idxDim - l.nRot, ubatch, 0,
                        false, blockPositions.buffer(), ctx.placeholders().readOnlyFloats(), compIndex.buffer()));
        latentRope = prepare(RopeKernels.ROPE_EXT, ropeGlobal(l.compressRope(), ubatch, 1), new long[]{local},
                RopeKernels.ropeExtArgs(l.compressRope(), ubatch, 1, l.headDim, l.headDim, l.headDim - l.nRot, ubatch,
                        0, false, blockPositions.buffer(), ctx.placeholders().readOnlyFloats(), compLatent.buffer()));
        compStore = prepare(Dsv4Kernels.CACHE_STORE, Dsv4Kernels.cacheStoreGlobal(ubatch, dsv4Local),
                Dsv4Kernels.local(dsv4Local), new Object[]{ubatch, l.compWidth,
                        Dsv4Kernels.kvTypeCode(plan.dtype()), 0, 0, l.compWidth, compRowsBuf.buffer(),
                        blockCells.buffer(), readStreams[0].cache()});
        int snapshots = plan.snapshots();
        PreparedDispatch snapshot = snapshots == 0 ? null : prepare(Dsv4Kernels.STATE_SNAPSHOT,
                Dsv4Kernels.stateSnapshotGlobal(ratio, l.headDim, snapshots, dsv4Local), Dsv4Kernels.local(dsv4Local),
                new Object[]{0, ubatch, ratio, l.headDim, snapshots, 0, 0, curKv.buffer(), curScore.buffer(), zeroApe,
                        readStreams[0].stateKv(), readStreams[0].stateScore()});
        PreparedDispatch state = prepare(Dsv4Kernels.STATE_UPDATE,
                Dsv4Kernels.stateUpdateGlobal(ratio, l.headDim, dsv4Local), Dsv4Kernels.local(dsv4Local),
                new Object[]{0, ubatch, ratio, l.headDim, 0, 0, curKv.buffer(), curScore.buffer(), zeroApe,
                        readStreams[0].stateKv(), readStreams[0].stateScore()});
        compressor = new Compressor(plan.kind(), ratio, compRows, sw, readStreams, wKv, wGate, sites, latentV41,
                snapshot, state, false, true);
    }

    /**
     * The {@value #TEMP_COMP_POS}{@code .src.<il>} table: {@code kv_pos[r] = (r + 1) * ratio - 1}, the position each
     * compressed row of KV source layer {@code il} stands for. Written once here; the source and every consumer of the
     * same source write the same values.
     */
    private GpuBuffer compressedPositions(int il, int rows, int ratio) {
        ScratchSlot s = tempInt(TEMP_COMP_POS + ".src." + il, rows);
        ByteBuffer bytes = ByteBuffer.allocate(rows * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < rows; r++) {
            bytes.putInt(Dsv4Layout.compressedRowPosition(r, ratio));
        }
        closeEvent(ctx.queue().enqueueWriteBuffer(s.buffer(), 0, (long) rows * Integer.BYTES, bytes.array()));
        return s.buffer();
    }

    /** @throws IllegalStateException when the temporary does not start at element 0 of its buffer. */
    private ScratchSlot tempInt(String name, int n) {
        ScratchSlot s = ctx.scratch().temp(name, n);
        if (s.offset() != 0) {
            throw new IllegalStateException(layout.where + ": DeepSeek-V4 buffer " + s + " must start at element 0");
        }
        return s;
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
     * The cache writes of a step: the raw KV projection with its norm and RoPE, the compressors' folds and the raw
     * store. The query projections, the indexer's selection, the attention and the output projection do not run.
     */
    @Override
    public void prefillKV(int pos0, int n, ScratchSlot in) {
        ensureBound();
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
        Objects.requireNonNull(in, "in").require((long) n * l.nEmbd);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        // The raw-KV and compressor projections of a single-token step all read this row, so its Q8_1
        // blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(xOff, l.nEmbd)
                ? ops.shareQ8Activations(x, xOff, l.nEmbd) : null;
        try {
            projectKV(n, x, xOff);
            if (compressor != null) {
                compressor.run(pos0, n, x, xOff);
            }
            if (lidCompressor != null) {
                lidCompressor.run(pos0, n, x, xOff);
            }
            storeRaw(n);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    @Override
    public String kvRefusal() {
        ensureBound();
        return null;
    }

    /** The raw KV row alone: its projection with its delta, RMS norm and RoPE. */
    private void projectKV(int n, GpuBuffer x, int xOff) {
        ops.matmul(wkv, x, xOff, kv.buffer(), 0, n);
        kvDeltas.applyRows(n, x, xOff, kv.buffer(), 0);
        rowNorm(kvNorm, n);
        rope(kvRope, n, 1);
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
        // The query, raw-KV, compressor and indexer projections of a single-token step all read this row,
        // so its Q8_1 blocks are quantized once.
        GpuOps.SharedQ8 share = n == 1 && GpuOps.shareableQ8Activations(xOff, l.nEmbd)
                ? ops.shareQ8Activations(x, xOff, l.nEmbd) : null;
        try {
            project(n, x, xOff);

            int nComp = 0;
            int selStride = 0;
            int selOff = 0;
            if (compressor != null) {
                compressor.run(pos0, n, x, xOff);
            }
            if (v41) {
                if (l.ratio != 0) {
                    int nVis = Dsv4Kernels.visibleRows(last, l.ratio);
                    if (nVis > 0) {
                        if (l.computesTopk) {
                            indexer.queries(n, qr.buffer(), 0, x, xOff);
                        }
                        nComp = selectV41(0, 0, n, nVis, 0);
                        selStride = nComp;
                    }
                }
            } else if (l.mode == CompMode.SELECTED) {
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

            storeRaw(n);
            attend(0, 0, n, rawKvPos, Math.min(rawRows, pos0 + n), steps.qPos(), nComp, selStride, selOff);
            output(n, out);
        } finally {
            if (share != null) {
                share.close();
            }
        }
    }

    /**
     * One step over the rows of {@code s}: the contiguous step when no paged step is current, else the paged step
     * through the memory's current {@code PagedStep} (see the class description).
     *
     * @throws UnsupportedOperationException for a non-causal step or a segment whose rows are not at consecutive
     *                                       positions, or a non-contiguous step without a paged step
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        ensureBound();
        Objects.requireNonNull(s, "s");
        if (steps.currentPagedStep() == null) {
            GpuMixer.super.batch(s, in, out);
            return;
        }
        forwardPaged(s, in, out);
    }

    @Override
    public String batchRefusal() {
        return "the DeepSeek-V4 mixer mixes several sequences per step only over paged memory (a currentPagedStep)";
    }

    /**
     * The paged step: the token-wise projections and the raw store over the step's rows, the compressors folding each
     * segment into its sequence's streams, then each segment's selection and attention.
     */
    private void forwardPaged(StepLayout s, ScratchSlot in, ScratchSlot out) {
        Dsv4Layout l = layout;
        int n = s.nTokens();
        if (steps.stepNonCausal()) {
            throw new UnsupportedOperationException(l.where + ": DeepSeek-V4 attention and its compressors are"
                    + " causal; a non-causal step is not supported");
        }
        if (n < 1 || n > ubatch || steps.nTokens() != n) {
            throw new IllegalArgumentException(l.where + ": paged DeepSeek-V4 step of " + n + " tokens, the step"
                    + " inputs hold " + steps.nTokens() + " within [1, " + ubatch + "]");
        }
        for (int seg = 0; seg < s.nSeg(); seg++) {
            checkSegment(s, seg);
        }
        long width = (long) n * l.nEmbd;
        Objects.requireNonNull(in, "in").require(width);
        Objects.requireNonNull(out, "out").require(width);
        GpuBuffer x = in.buffer();
        int xOff = in.offset();
        project(n, x, xOff);
        foldSegments(compressor, s, x, xOff);
        foldSegments(lidCompressor, s, x, xOff);
        storeRaw(n);
        boolean selecting = false;
        if (v41 ? l.computesTopk : lidCompressor != null) {
            int ratio = v41 ? l.ratio : Dsv4Spec.RATIO_CSA;
            for (int seg = 0; seg < s.nSeg(); seg++) {
                selecting |= Dsv4Kernels.visibleRows(s.pos0(seg) + s.len(seg) - 1, ratio) > 0;
            }
            if (selecting) {
                indexer.queries(n, qr.buffer(), 0, x, xOff);
            }
        }
        for (int seg = 0; seg < s.nSeg(); seg++) {
            attendSegment(s.seq(seg), s.tokStart(seg), s.len(seg), s.pos0(seg));
        }
        output(n, out);
    }

    /** Projects every row of the step through a compressor, then folds each segment into its sequence's stream. */
    private void foldSegments(Compressor c, StepLayout s, GpuBuffer x, int xOff) {
        if (c == null) {
            return;
        }
        c.project(s.nTokens(), x, xOff);
        for (int seg = 0; seg < s.nSeg(); seg++) {
            c.fold(s.seq(seg), s.pos0(seg), s.len(seg), s.tokStart(seg));
        }
    }

    /**
     * One segment's attention: a CSA layer's selection among the sequence's own indexer rows, then the rows
     * {@code [t0, t0 + len)} at positions {@code pos0 ..} over every raw cell through the sequence's masked
     * {@code kv_pos} and the sequence's compressed stream.
     */
    private void attendSegment(int seq, int t0, int len, int pos0) {
        Dsv4Layout l = layout;
        int last = pos0 + len - 1;
        int nComp = 0;
        int selStride = 0;
        int selOff = 0;
        if (v41) {
            int nVis = l.ratio == 0 ? 0 : Dsv4Kernels.visibleRows(last, l.ratio);
            if (nVis > 0) {
                int k = SparseKernels.selectedCount(nVis, l.topK);
                selOff = t0 * k;
                selectV41(seq, t0, len, nVis, selOff);
                nComp = k;
                selStride = k;
            }
        } else if (l.mode == CompMode.SELECTED) {
            int nLid = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_CSA);
            if (nLid > 0) {
                nComp = indexer.selectRows(t0, len, nLid, lidCompressor.streams[seq].cache(), lidKvPos,
                        indexer.config().mask().causal(), selection, 0);
                selStride = nComp;
            }
        } else if (l.mode == CompMode.DENSE) {
            nComp = Dsv4Kernels.visibleRows(last, Dsv4Spec.RATIO_HCA);
        }
        steps.uploadSegmentPositions(pos0, len);
        attend(seq, t0, len, steps.sequenceKvPos(rawGroup, seq), rawRows, steps.segmentPositions(), nComp,
                selStride, selOff);
    }

    /**
     * The DeepSeek-V4.1 selection of the step's rows {@code [t0, t0 + n)} among the {@code nVis} visible compressed
     * rows of sequence {@code seq}, written at {@code selOff} of the source's shared top-k plane
     * (models/deepseek41.cpp:520-744). Only an indexer source computes it: inside the candidate blocks of an earlier
     * source when it consumes them, then (when it is their source) its own candidate blocks for later consumers.
     * Consumers reuse the selection their source wrote in this step.
     *
     * @return {@code k = min(nVis, topK)}, the row stride of the selection
     */
    private int selectV41(int seq, int t0, int n, int nVis, int selOff) {
        Dsv4Layout l = layout;
        int k = SparseKernels.selectedCount(nVis, l.topK);
        if (!l.computesTopk) {
            return k;
        }
        GpuBuffer keys = readStreams[seq].cache();
        boolean blocks = usesCandidate || picksCandidate;
        int nBlk = blocks ? SparseKernels.blockCount(nVis, candBlockSize) : 0;
        int stride = blocks ? Math.min(candTopKBlocks, nBlk) : 0;
        if (usesCandidate && candBlocks != null) {
            indexer.selectInsideBlocks(t0, n, nVis, candBlockSize, candBlocks.buffer(), t0 * stride, stride, v41TopK,
                    selOff, keys, compParamPos, true);
        } else {
            indexer.selectRows(t0, n, nVis, keys, compParamPos, true, v41TopK.buffer(), selOff);
        }
        if (picksCandidate && candBlocks != null) {
            indexer.selectBlocks(t0, n, nVis, l.ratio, candBlockSize, candTopKBlocks, candBlocks.buffer(),
                    t0 * stride, keys, compParamPos, true);
        }
        return k;
    }

    /** The compressed cache the attention reads for sequence {@code seq}. */
    private GpuBuffer compressedCache(int seq) {
        if (v41) {
            return readStreams != null ? readStreams[seq].cache() : ctx.placeholders().readOnlyFloats();
        }
        return compressor != null ? compressor.streams[seq].cache() : ctx.placeholders().readOnlyFloats();
    }

    /**
     * The q, per-head and kv norms and projections with their RoPE over the step's {@code n} rows
     * (deepseek4.cpp:918-940).
     */
    private void project(int n, GpuBuffer x, int xOff) {
        Dsv4Layout l = layout;
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
    }

    /** {@code LLM_DSV4_CacheStore} of the step's {@code n} kv rows at their {@code cell_idx}. */
    private void storeRaw(int n) {
        rawStore.setInt(Dsv4Kernels.ARG_STORE_N_TOKENS, n);
        rawStore.setGlobal(Dsv4Kernels.cacheStoreGlobal(n, dsv4Local));
        rawStore.dispatch();
    }

    /**
     * {@code LLM_DSV4_CompressedAttn} of rows {@code [t0, t0 + len)} at positions {@code qPos[0, len)} over raw rows
     * {@code [0, nRaw)} at {@code rawPos} and sequence {@code seq}'s compressed stream.
     */
    private void attend(int seq, int t0, int len, GpuBuffer rawPos, int nRaw, GpuBuffer qPos, int nComp,
                        int selStride, int selOff) {
        Dsv4Layout l = layout;
        int qOff = t0 * l.nHead * l.headDim;
        attention.setInt(Dsv4Kernels.ARG_ATTN_N_TOKENS, len);
        attention.setInt(Dsv4Kernels.ARG_ATTN_Q_OFF, qOff);
        attention.setInt(Dsv4Kernels.ARG_ATTN_OUT_OFF, qOff);
        attention.setInt(Dsv4Kernels.ARG_ATTN_N_RAW, nRaw);
        attention.setInt(Dsv4Kernels.ARG_ATTN_N_COMP, nComp);
        attention.setInt(Dsv4Kernels.ARG_ATTN_SEL_STRIDE, selStride);
        attention.setInt(Dsv4Kernels.ARG_ATTN_SEL_OFF, selOff);
        attention.setInt(Dsv4Kernels.ARG_ATTN_COMP_STRIDE, v41 ? compStride : l.headDim);
        attention.setBuffer(Dsv4Kernels.ARG_ATTN_RAW_POS, rawPos);
        attention.setBuffer(Dsv4Kernels.ARG_ATTN_Q_POS, qPos);
        attention.setBuffer(Dsv4Kernels.ARG_ATTN_COMP_K, compressedCache(seq));
        attention.setGlobal(Dsv4Kernels.attnGlobal(len, l.nHead, gTile, dsv4Local));
        attention.dispatch();
    }

    /** The grouped {@code attn_output_a} and {@code attn_output_b} over the step's {@code n} attention rows. */
    private void output(int n, ScratchSlot out) {
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
        GpuMatmulTap.heads(ops.queue(), woAW, l.nGroups, attn.buffer(), 0, l.oGroupDim, l.oGroupDim, n);
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

    /**
     * Checks one segment of a paged step: its sequence has a compressed stream, its rows sit at consecutive positions
     * from its {@code pos0}, and its compressed rows fit the streams.
     */
    private void checkSegment(StepLayout s, int seg) {
        Dsv4Layout l = layout;
        int seq = s.seq(seg);
        int streams = compressor != null ? compressor.streams.length
                : lidCompressor != null ? lidCompressor.streams.length
                : readStreams != null ? readStreams.length : 0;
        if (streams > 0 && seq >= streams) {
            throw new IllegalStateException(l.where + ": sequence " + seq + " has no compressed stream (the memory"
                    + " keeps " + streams + ")");
        }
        int t0 = s.tokStart(seg);
        for (int t = 0; t < s.len(seg); t++) {
            int p = steps.rowPosition(t0 + t);
            if (p != s.pos0(seg) + t) {
                throw new UnsupportedOperationException(l.where + ": row " + (t0 + t) + " of sequence " + seq
                        + " is at position " + p + ", not " + (s.pos0(seg) + t) + "; the DeepSeek-V4 compressors"
                        + " fold consecutive positions (a media chunk sharing positions is refused)");
            }
        }
        if (compressor != null) {
            l.checkCapacity(s.pos0(seg), s.len(seg), compressor.ratio, compressor.rows, compressor.kind.toString());
        }
        if (lidCompressor != null) {
            l.checkCapacity(s.pos0(seg), s.len(seg), lidCompressor.ratio, lidCompressor.rows,
                    lidCompressor.kind.toString());
        }
    }

    /**
     * One compressor of the layer with its prepared kernels and the stream of every sequence, which a fold rebinds.
     */
    private final class Compressor {
        final CompressedKvPlan.Kind kind;
        final int ratio;
        final int rows;
        /** Floats per projected row of {@code cur_kv} / {@code cur_score} (the plan's state width). */
        final int stateWidth;
        /** The compressed stream of each sequence, indexed by sequence. */
        final GpuCompressedKv[] streams;
        final GpuWeight wKv;
        final GpuWeight wGate;
        /** LoRA deltas of {@code wKv} and {@code wGate} (deepseek4.cpp:952-970, 1036-1039). */
        final GpuProjectionDeltas.Site kvDeltas;
        final GpuProjectionDeltas.Site gateDeltas;
        final PreparedDispatch compress;
        /** {@code LLM_DSV4_StateSnapshot}, null when the memory keeps no rollback snapshots. */
        final PreparedDispatch snapshot;
        final PreparedDispatch state;
        /** Indexer compressor of the DeepSeek-V4.0 CSA layer. */
        final boolean lid;
        final int argPos0;
        final int argTokens;
        final int argCurKvOff;
        final int argCurScoreOff;
        final int argStateKv;
        final int argStateScore;
        final int argCache;
        /** DeepSeek-V4.1: pooled latents, index keys and combined-row stores instead of the V4 compressed rows. */
        final boolean v41;

        Compressor(CompressedKvPlan.Kind kind, int ratio, int rows, int stateWidth, GpuCompressedKv[] streams,
                   GpuWeight wKv, GpuWeight wGate, GpuProjectionDeltas.Site[] deltas, PreparedDispatch compress,
                   PreparedDispatch snapshot, PreparedDispatch state, boolean lid) {
            this(kind, ratio, rows, stateWidth, streams, wKv, wGate, deltas, compress, snapshot, state, lid, false);
        }

        Compressor(CompressedKvPlan.Kind kind, int ratio, int rows, int stateWidth, GpuCompressedKv[] streams,
                   GpuWeight wKv, GpuWeight wGate, GpuProjectionDeltas.Site[] deltas, PreparedDispatch compress,
                   PreparedDispatch snapshot, PreparedDispatch state, boolean lid, boolean v41) {
            this.kind = kind;
            this.ratio = ratio;
            this.rows = rows;
            this.stateWidth = stateWidth;
            this.streams = streams;
            this.wKv = wKv;
            this.wGate = wGate;
            this.kvDeltas = deltas[0];
            this.gateDeltas = deltas[1];
            this.compress = compress;
            this.snapshot = snapshot;
            this.state = state;
            this.lid = lid;
            this.v41 = v41;
            this.argPos0 = v41 ? Dsv4Kernels.ARG_V41_POS0 : lid ? Dsv4Kernels.ARG_LID_POS0
                    : Dsv4Kernels.ARG_COMPRESS_POS0;
            this.argTokens = v41 ? Dsv4Kernels.ARG_V41_N_TOKENS : lid ? Dsv4Kernels.ARG_LID_N_TOKENS
                    : Dsv4Kernels.ARG_COMPRESS_N_TOKENS;
            this.argCurKvOff = v41 ? Dsv4Kernels.ARG_V41_CUR_KV_OFF : lid ? Dsv4Kernels.ARG_LID_CUR_KV_OFF
                    : Dsv4Kernels.ARG_COMPRESS_CUR_KV_OFF;
            this.argCurScoreOff = v41 ? Dsv4Kernels.ARG_V41_CUR_SCORE_OFF : lid ? Dsv4Kernels.ARG_LID_CUR_SCORE_OFF
                    : Dsv4Kernels.ARG_COMPRESS_CUR_SCORE_OFF;
            this.argStateKv = v41 ? Dsv4Kernels.ARG_V41_STATE_KV : lid ? Dsv4Kernels.ARG_LID_STATE_KV
                    : Dsv4Kernels.ARG_COMPRESS_STATE_KV;
            this.argStateScore = v41 ? Dsv4Kernels.ARG_V41_STATE_SCORE : lid ? Dsv4Kernels.ARG_LID_STATE_SCORE
                    : Dsv4Kernels.ARG_COMPRESS_STATE_SCORE;
            this.argCache = v41 ? -1 : lid ? Dsv4Kernels.ARG_LID_CACHE : Dsv4Kernels.ARG_COMPRESS_CACHE;
        }

        /** The single-sequence step: {@link #project} then {@link #fold} into stream 0. */
        void run(int pos0, int n, GpuBuffer x, int xOff) {
            project(n, x, xOff);
            fold(0, pos0, n, 0);
        }

        /** Projects the kv and score rows of the step's {@code n} rows into {@code cur_kv} / {@code cur_score}. */
        void project(int n, GpuBuffer x, int xOff) {
            ops.matmul(wKv, x, xOff, curKv.buffer(), 0, n);
            kvDeltas.applyRows(n, x, xOff, curKv.buffer(), 0);
            if (wGate != null) {
                ops.matmul(wGate, x, xOff, curScore.buffer(), 0, n);
                gateDeltas.applyRows(n, x, xOff, curScore.buffer(), 0);
            } else {
                // an ungated ratio-1 compressor never gates its single-position window; the kernel still reads a
                // finite score for it
                ops.fill(curScore.buffer(), 0, (long) n * stateWidth * Float.BYTES, 0);
            }
        }

        /**
         * Folds the projected rows {@code [t0, t0 + n)} at positions {@code pos0 ..} into sequence {@code seq}'s
         * stream: writes the compressed rows they complete (no dispatch when they complete none), writes the rollback
         * snapshots from the live plane the compress kernel has read, then persists the rows in that plane.
         */
        void fold(int seq, int pos0, int n, int t0) {
            if (v41) {
                foldV41(seq, pos0, n, t0);
                return;
            }
            GpuCompressedKv c = streams[seq];
            int off = t0 * stateWidth;
            if (Dsv4Kernels.blockCount(pos0, n, ratio) > 0) {
                compress.setInt(argPos0, pos0);
                compress.setInt(argTokens, n);
                compress.setInt(argCurKvOff, off);
                compress.setInt(argCurScoreOff, off);
                compress.setBuffer(argStateKv, c.stateKv());
                compress.setBuffer(argStateScore, c.stateScore());
                compress.setBuffer(argCache, c.cache());
                compress.setGlobal(Dsv4Kernels.compressGlobal(pos0, n, ratio, dsv4Local));
                compress.dispatch();
            }
            persist(seq, pos0, n, off);
        }

        /** The DeepSeek-V4.1 fold: the latents and index keys of the completed blocks, then the combined rows. */
        private void foldV41(int seq, int pos0, int n, int t0) {
            GpuCompressedKv c = streams[seq];
            int off = t0 * stateWidth;
            int nBlocks = Dsv4Kernels.blockCount(pos0, n, ratio);
            if (nBlocks > 0) {
                compress.setInt(argPos0, pos0);
                compress.setInt(argTokens, n);
                compress.setInt(argCurKvOff, off);
                compress.setInt(argCurScoreOff, off);
                compress.setBuffer(argStateKv, c.stateKv());
                compress.setBuffer(argStateScore, c.stateScore());
                compress.setGlobal(Dsv4Kernels.compressGlobal(pos0, n, ratio, dsv4Local));
                compress.dispatch();
                deriveV41(seq, pos0, nBlocks);
            }
            persist(seq, pos0, n, off);
        }

        /** The rollback snapshots and the plane update of the folded rows. */
        private void persist(int seq, int pos0, int n, int off) {
            GpuCompressedKv c = streams[seq];
            if (snapshot != null) {
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_POS0, pos0);
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_N_TOKENS, n);
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_CUR_KV_OFF, off);
                snapshot.setInt(Dsv4Kernels.ARG_SNAPSHOT_CUR_SCORE_OFF, off);
                snapshot.setBuffer(Dsv4Kernels.ARG_SNAPSHOT_STATE_KV, c.stateKv());
                snapshot.setBuffer(Dsv4Kernels.ARG_SNAPSHOT_STATE_SCORE, c.stateScore());
                snapshot.dispatch();
            }
            state.setInt(Dsv4Kernels.ARG_STATE_POS0, pos0);
            state.setInt(Dsv4Kernels.ARG_STATE_N_TOKENS, n);
            state.setInt(Dsv4Kernels.ARG_STATE_CUR_KV_OFF, off);
            state.setInt(Dsv4Kernels.ARG_STATE_CUR_SCORE_OFF, off);
            state.setBuffer(Dsv4Kernels.ARG_STATE_STATE_KV, c.stateKv());
            state.setBuffer(Dsv4Kernels.ARG_STATE_STATE_SCORE, c.stateScore());
            state.dispatch();
        }
    }

    /**
     * Derives the DeepSeek-V4.1 compressed rows of sequence {@code seq} from the pooled latents
     * ({@code LLM_DSV4_CompressV41}) of the step's completed blocks: the index key of each latent
     * ({@code indexer.attn_k}, RMS norm and the index-key rope), the latent rope and the cache rotation, then the
     * combined {@code [latent | index key]} rows stored at the block indices.
     */
    private void deriveV41(int seq, int pos0, int nBlocks) {
        Dsv4Layout l = layout;
        int b0 = Dsv4Kernels.firstBlock(pos0, l.ratio);
        writeBlockInts(blockPositions.buffer(), b0, nBlocks, l.ratio);
        writeBlockInts(blockCells.buffer(), b0, nBlocks, 1);
        ops.matmul(idxK, compLatent.buffer(), 0, compIndex.buffer(), 0, nBlocks);
        idxKDeltas.applyRows(nBlocks, compLatent.buffer(), 0, compIndex.buffer(), 0);
        indexNorm.setInt(NormKernels.ARG_ROW_NORM_N_ROWS, nBlocks);
        indexNorm.setGlobal(NormKernels.reduceGlobal(NormKernels.rowNormGroups(nBlocks), local));
        indexNorm.dispatch();
        runRope(indexRope, l.indexerRope(), nBlocks);
        runRope(latentRope, l.compressRope(), nBlocks);
        if (latentRotation != null) {
            // the raw window is stored rotated for a quantized cache; the compressed latents join that space
            ops.hadamard(latentRotation, compLatent.buffer(), 0, compLatent.buffer(), 0, nBlocks, l.headDim);
        }
        ops.copy2d(compLatent.buffer(), 0, l.headDim, compRowsBuf.buffer(), 0, compStride, nBlocks, l.headDim);
        ops.copy2d(compIndex.buffer(), 0, l.idxDim, compRowsBuf.buffer(), l.headDim, compStride, nBlocks, l.idxDim);
        compStore.setInt(Dsv4Kernels.ARG_STORE_N_TOKENS, nBlocks);
        compStore.setBuffer(Dsv4Kernels.ARG_STORE_CACHE, readStreams[seq].cache());
        compStore.setGlobal(Dsv4Kernels.cacheStoreGlobal(nBlocks, dsv4Local));
        compStore.dispatch();
    }

    /** {@code LLM_RoPE_Ext} over {@code rows} single-head rows of a spec's width at the block positions. */
    private void runRope(PreparedDispatch d, RopeSpec spec, int rows) {
        d.setInt(RopeKernels.ARG_N_TOKENS, rows);
        d.setInt(RopeKernels.ARG_POS_STRIDE, rows);
        d.setGlobal(ropeGlobal(spec, rows, 1));
        d.dispatch();
    }

    /** Writes {@code nBlocks} ints {@code (b0 + g) * mul} (the raw block index when {@code mul == 1}). */
    private void writeBlockInts(GpuBuffer dst, int b0, int nBlocks, int mul) {
        ByteBuffer bytes = ByteBuffer.allocate(nBlocks * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int g = 0; g < nBlocks; g++) {
            bytes.putInt((b0 + g) * mul);
        }
        closeEvent(ctx.queue().enqueueWriteBuffer(dst, 0, (long) nBlocks * Integer.BYTES, bytes.array()));
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
     * transform, input transform or output scale, which llama.cpp's plain {@code ggml_mul_mat} of it does not apply
     * either (deepseek4.cpp:1218).
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
        if (w.hadamard() != null || w.hasScale() || (w.inputTransform() != null && !w.inputTransform().isIdentity())) {
            throw new UnsupportedOperationException(layout.where + ": grouped tensor " + w.name()
                    + " carries an input transform or output scale, which the grouped output product does not apply");
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
        return ropeGlobal(layout.attnRope, n, heads);
    }

    private long[] ropeGlobal(RopeSpec r, int n, int heads) {
        long items = RopeKernels.ropeExtWorkItems(r.type(), n, heads, r.nDims());
        return new long[]{RopeKernels.roundUp(Math.max(1L, items), local)};
    }

    /**
     * A zero-filled temporary of {@code n} floats per row starting at element 0 of its buffer, written once at bind,
     * so every row the kernel's block position selects reads zeros.
     */
    private GpuBuffer zeroFloats(String name, long n) {
        ScratchSlot s = temp(name, n);
        ops.fill(s.buffer(), 0, s.floats() * Float.BYTES, 0);
        return s.buffer();
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

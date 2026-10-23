package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuIndexerCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU QSA attention of one k-pool indexer layer ({@code dsa().kpool() > 0}) for one sequence: the pooled lightning
 * indexer picks {@code K = floor(top_k / kpool)} complete pools per token and attention runs over their cells plus,
 * with {@code kpoolSelectTail}, the token's incomplete pool ({@code LLM_MSA_GatherAttn} with one selection per token).
 * Pool {@code i} holds positions {@code [i*kpool, (i+1)*kpool)}; its pooled key lives in the indexer row of its last
 * member (src/llama-memory-hybrid-idx.cpp:846-873). With {@code T} tokens at {@code pos0}, {@code n_new} =
 * {@code (pos0+T)/kpool - pos0/kpool} pools complete in the step and {@code n_pool = (pos0+T)/kpool} are scored.
 *
 * <p><b>glm5-next</b> (nope MLA with the latent cache, models/glm5-next.cpp:1046-1108; selection :892-1044):
 * <pre>
 * attn_q_a(x) -&gt; RMS(attn_q_a_norm) = qr;  attn_q_b(qr) = q[T][H][nope]
 * attn_kv_a_mqa(x) -&gt; RMS(attn_kv_a_norm) = kv[T][L]
 * {T}_MatVecId(attn_k_b, slot h) : q_abs[T][H][L] = wk_b[h] q[t][h]                           (:1073-1076)
 * full indexer layer:
 *   indexer.attn_q_b(qr) = iq[T][Hi][I]; LayerNorm(indexer.attn_k(x), indexer.k_norm w+b) = ik  (:904-911)
 *   indexer_compressor_gate(x) = ig; [ik | ig] -&gt; LLM_KV_Store_&lt;KV&gt; into rows
 *   [key | gate | pooled]                                                                       (:913-921)
 *   LLM_QSA_KpoolUpdate_&lt;KV&gt; GATED (ape) POOL then STORE for the n_new completed pools        (:923-946)
 *   indexer.proj(x) = w[T][Hi]; LLM_QSA_KpoolScore_&lt;KV&gt;(has_w, w_scale);
 *   LLM_TopK_Rows -&gt; selection                                                                 (:950-976)
 * shared indexer layer: the selection of the preceding full layer                              (:1077-1083)
 * LLM_HadamardRotate q_abs, kv (n_rot_k) when the latent cache rotates; LLM_KV_Store(kv)
 * LLM_MSA_GatherAttn(q_abs, latent K = V, kpool blocks, mla kq_scale) = o[T][H][L]               (:1097-1100)
 * LLM_HadamardRotate o (n_rot_k) back; {T}_MatVecId(attn_v_b, slot h) = o_v[T][H][Dv]; attn_output (:1104)
 * </pre>
 * llama.cpp stores the latent unrotated; with a rotating cache plan the scores are unchanged ({@code H} is
 * orthonormal) and the output is rotated back because V is the K row.
 *
 * <p><b>qwen4exp</b> (gated GQA, models/qwen4exp.cpp:741-1026):
 * <pre>
 * indexer.k_proj(x) = k_raw -&gt; LLM_KV_Store_&lt;KV&gt; into the raw-key slot of rows
 *   [raw | pooled]                                                                               (:762-767)
 * LLM_QSA_KpoolUpdate_&lt;KV&gt; MEAN POOL; RMS(indexer.k_norm); LLM_RoPE_Ext at each pool's first-member
 *   position [p, y, x, p] (llama-memory-hybrid-idx.cpp:866-873); STORE                           (:771-792)
 * indexer.q_proj(x) -&gt; RMS(indexer.q_norm) -&gt; LLM_RoPE_Ext = iq; LLM_QSA_KpoolScore (constant
 *   weight idx^-0.5); LLM_TopK_Rows -&gt; selection                                                 (:797-812)
 * attn_q(x) = qg[T][H][q | gate]; LLM_Copy2D_F32 q; RMS(attn_q_norm) per head; attn_k, attn_v; RMS(attn_k_norm)
 * LLM_RoPE_Ext (multi-section) on q and k at the step positions                                 (:956-989)
 * LLM_HadamardRotate q, k (n_rot_k), v (n_rot_v); LLM_KV_Store k, v; LLM_MSA_GatherAttn          (:894-931)
 * LLM_HadamardRotate o (n_rot_v) back; LLM_GLU SIGMUL(gate, o); attn_output                     (:934-1023)
 * </pre>
 * The pooled slot of a fresh cache row is not zeroed on the device: it is read only at the rep row of a complete
 * pool, which the STORE stage of the step completing the pool writes first.
 *
 * <p>All launches except the projections, norms, copies, fills and rotations ({@link GpuOps} calls) are
 * {@link PreparedDispatch}es bound once; a step re-sets only counts and grids, and uploads the {@code 4 * n_new}
 * pool positions of qwen4exp. Skipped work: no pooling when {@code n_new == 0}; no scoring when the step sees no
 * complete pool ({@code n_pool == 0}, the selection is filled with -1) or selects none ({@code K == 0}).
 * Not thread-safe.
 */
public final class GpuQsaMixer implements GpuMixer {

    static final String TEMP_QR = "qsa.qr";
    static final String TEMP_Q = "qsa.q";
    static final String TEMP_K = "qsa.k";
    static final String TEMP_V = "qsa.v";
    static final String TEMP_QG = "qsa.qg";
    static final String TEMP_QABS = "qsa.qabs";
    static final String TEMP_ATT = "qsa.att";
    static final String TEMP_VOUT = "qsa.vout";
    static final String TEMP_IQ = "qsa.iq";
    static final String TEMP_IK = "qsa.ik";
    static final String TEMP_IG = "qsa.ig";
    static final String TEMP_PACKED = "qsa.packed";
    static final String TEMP_W = "qsa.w";
    static final String TEMP_POOLED = "qsa.pooled";
    static final String TEMP_POOL_POS = "qsa.pool_pos";
    static final String TEMP_SCORES = "qsa.scores";

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();

    private GpuOps ops;
    private int local;
    private int ubatch;
    private float rmsEps;
    private float normEps;
    private boolean mla;
    private boolean full;
    private int nHead;
    private int posLimit;
    private int rotK;
    private int rotV;
    private KvLayout kvLayout;

    // selection
    private int kpool;
    private int selWidth;
    private int idx;
    private int nIdxH;
    private int mode;
    private ScratchSlot sel;

    // indexer (full layers)
    private GpuWeight wIq;
    private GpuWeight wIk;
    private GpuWeight wIg;
    private GpuWeight wProj;
    /** LoRA delta sites of the bound projections (null for absent ones); see {@link #bindDeltas}. */
    private GpuProjectionDeltas.Site qaDeltas;
    private GpuProjectionDeltas.Site qbDeltas;
    private GpuProjectionDeltas.Site kvaDeltas;
    private GpuProjectionDeltas.Site qDeltas;
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site oDeltas;
    private GpuProjectionDeltas.Site iqDeltas;
    private GpuProjectionDeltas.Site ikDeltas;
    private GpuProjectionDeltas.Site igDeltas;
    private GpuProjectionDeltas.Site projDeltas;
    private GpuWeight ikNormW;
    private GpuWeight ikNormB;
    private GpuWeight iqNorm;
    private LayerMemoryPlan.IndexerPlan idxPlan;
    private ScratchSlot iq;
    private ScratchSlot ik;
    private ScratchSlot ig;
    private ScratchSlot packed;
    private ScratchSlot w;
    private ScratchSlot pooled;
    private ScratchSlot poolPos;
    private byte[] poolPosBytes;
    private ByteBuffer poolPosView;
    private RopeSpec idxRope;
    private PreparedDispatch storeIdx;
    private PreparedDispatch kpoolUpdate;
    private PreparedDispatch kpoolScore;
    private PreparedDispatch topKRows;
    private PreparedDispatch ropeIq;
    private PreparedDispatch ropePool;

    // glm5-next latent MLA
    private int qLora;
    private int kvLora;
    private int qkNope;
    private int headV;
    private int idLocal;
    private int idChunkTokens;
    private GpuWeight wQa;
    private GpuWeight qaNorm;
    private GpuWeight wQb;
    private GpuWeight wKva;
    private GpuWeight kvaNorm;
    private GpuWeight wO;
    private ScratchSlot qr;
    private ScratchSlot qn;
    private ScratchSlot kvl;
    private ScratchSlot qabs;
    private ScratchSlot att;
    private ScratchSlot vout;
    private PreparedDispatch absorbK;
    private PreparedDispatch absorbV;

    // qwen4exp gated GQA
    private int nKv;
    private int hd;
    private int qDim;
    private int kDim;
    private RopeSpec rope;
    private GpuWeight wQ;
    private GpuWeight wK;
    private GpuWeight wV;
    private GpuWeight qNorm;
    private GpuWeight kNorm;
    private ScratchSlot qg;
    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private PreparedDispatch ropeQ;
    private PreparedDispatch ropeK;
    private PreparedDispatch storeV;

    private PreparedDispatch storeK;
    private PreparedDispatch gather;

    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a QSA layer or uses a feature the mixer does not
     *                                  implement
     * @throws IllegalStateException    when the context has no memory, a cache or tensor has the wrong shape, a
     *                                  shared layer's source is not a full indexer layer, or the mixer is already
     *                                  bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("QSA mixer is already bound or closed");
        }
        MsaBlocks.checkVariant(layer, MsaBlocks.Variant.QSA);
        try {
            bindChecked(ctx, layer);
            bound = true;
        } catch (RuntimeException e) {
            try {
                MsaBlocks.release(prepared, owned);
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private void bindChecked(GpuForwardContext ctx, LayerSpec layer) {
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("QSA layer " + MsaBlocks.describe(layer) + " needs a KV cache");
        }
        ops = ctx.ops();
        local = ops.local();
        ubatch = ctx.scratch().ubatch();
        rmsEps = ctx.spec().hparams().normRmsEps();
        normEps = ctx.spec().hparams().normEps();
        DsaSpec dsa = layer.dsa();
        mla = layer.mla() != null;
        full = dsa.full();
        kpool = dsa.kpool();
        selWidth = MsaBlocks.selectionWidth(dsa);
        idx = dsa.indexerHeadDim();
        nIdxH = dsa.indexerHeads();
        mode = MsaKernels.kpoolMode(dsa.kpoolRow());
        if (mla != (mode == MsaKernels.KPOOL_MODE_GATED)) {
            throw new IllegalArgumentException("layer " + MsaBlocks.describe(layer) + ": the gated k-pool belongs to"
                    + " the MLA layers (glm5-next) and the mean k-pool to the GQA layers (qwen4exp)");
        }
        MsaBlocks.checkQsaSource(ctx.spec(), layer);

        GpuMemory memory = ctx.memory();
        GpuKvLayer kv = memory.kv(layer.il());
        if (kv == null) {
            throw new IllegalStateException("QSA layer " + MsaBlocks.describe(layer) + " has no KV cache");
        }
        kvLayout = kv.layout();
        MsaBlocks.checkNoRing(layer, kvLayout, "KV cache");
        RotationPlan rot = kv.rotation();
        rotK = rot.nRotK();
        rotV = rot.nRotV();
        posLimit = kvLayout.rows();

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        GpuScratch scratch = ctx.scratch();
        sel = MsaBlocks.atZero(scratch.temp(MsaBlocks.QSA_SELECTION, MsaBlocks.selectionInts(ctx.spec(), ubatch)));
        if (mla) {
            bindMla(ctx, ws, layer, kv);
        } else {
            bindGqa(ctx, ws, layer, kv);
        }
        if (full) {
            bindIndexer(ctx, ws, layer, memory.indexer(layer.il()));
        }
        bindDeltas(ctx.projectionDeltas(), layer);
    }

    /**
     * Serves the LoRA delta site of every bound dense projection; each delta is added right after its product and
     * before any bias ({@code build_lora_mm}, llama-graph.cpp:1629-1667).
     */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer) {
        qaDeltas = serveDelta(deltas, layer, TensorNames.ATTN_Q_A, wQa);
        qbDeltas = serveDelta(deltas, layer, TensorNames.ATTN_Q_B, wQb);
        kvaDeltas = serveDelta(deltas, layer, TensorNames.ATTN_KV_A_MQA, wKva);
        qDeltas = serveDelta(deltas, layer, TensorNames.ATTN_Q, wQ);
        kDeltas = serveDelta(deltas, layer, TensorNames.ATTN_K, wK);
        vDeltas = serveDelta(deltas, layer, TensorNames.ATTN_V, wV);
        oDeltas = serveDelta(deltas, layer, TensorNames.ATTN_OUT, wO);
        iqDeltas = serveDelta(deltas, layer, mode == MsaKernels.KPOOL_MODE_GATED ? TensorNames.INDEXER_ATTN_Q_B : TensorNames.INDEXER_Q_PROJ, wIq);
        ikDeltas = serveDelta(deltas, layer, mode == MsaKernels.KPOOL_MODE_GATED ? TensorNames.INDEXER_ATTN_K : TensorNames.INDEXER_K_PROJ, wIk);
        igDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_KPOOL_GATE, wIg);
        projDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_PROJ, wProj);
    }

    private GpuProjectionDeltas.Site serveDelta(GpuProjectionDeltas deltas, LayerSpec layer, String base, GpuWeight w) {
        return w == null ? null : deltas.serve(layer, TensorNames.weight(base), w);
    }

    // ------------------------------------------------------------------ glm5-next

    private void bindMla(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, GpuKvLayer kv) {
        MlaSpec m = layer.mla();
        MsaBlocks.checkQsaMla(layer, m, kvLayout, rotV);
        nHead = m.nHead();
        qLora = m.qLoraRank();
        kvLora = m.kvLoraRank();
        qkNope = m.headKNope();
        headV = m.headV();
        MsaBlocks.checkRotation(rotK, kvLora);
        int nEmbd = MsaBlocks.mixerWidth(layer);
        wQa = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_Q_A, nEmbd, qLora);
        qaNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_Q_A_NORM), qLora, true, owned);
        wQb = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_Q_B, qLora, (long) nHead * qkNope);
        wKva = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_KV_A_MQA, nEmbd, kvLora);
        kvaNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_KV_A_NORM), kvLora, true,
                owned);
        GpuWeight wKb = MsaBlocks.gpuHeadStack(ws, layer, TensorNames.ATTN_K_B, qkNope, kvLora, nHead);
        GpuWeight wVb = MsaBlocks.gpuHeadStack(ws, layer, TensorNames.ATTN_V_B, kvLora, headV, nHead);
        wO = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, (long) nHead * headV, nEmbd);

        GpuScratch scratch = ctx.scratch();
        long ub = ubatch;
        qr = MsaBlocks.atZero(scratch.temp(TEMP_QR, ub * qLora));
        qn = MsaBlocks.atZero(scratch.temp(TEMP_Q, ub * nHead * qkNope));
        kvl = MsaBlocks.atZero(scratch.temp(TEMP_K, ub * kvLora));
        qabs = MsaBlocks.atZero(scratch.temp(TEMP_QABS, ub * nHead * kvLora));
        att = MsaBlocks.atZero(scratch.temp(TEMP_ATT, ub * nHead * kvLora));
        vout = MsaBlocks.atZero(scratch.temp(TEMP_VOUT, ub * nHead * headV));

        idLocal = idLocal(wKb, wVb);
        idChunkTokens = idChunkTokens(layer, ctx.kernels().caps().getMaxWorkGroupCount()[1]);
        GpuBuffer slots = headSlots(ctx);
        GpuBuffer ph = ctx.placeholders().readOnlyFloats();
        absorbK = keep(ops.prepare(MoeKernels.matVecId(wKb.type()), MoeKernels.idMatVecGlobal(kvLora, 1, idLocal),
                new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(kvLora, qkNope, 1, qkNope, 0, 0, false, 0, nHead,
                        slots, wKb.shard(0), qn.buffer(), ph, qabs.buffer())));
        absorbV = keep(ops.prepare(MoeKernels.matVecId(wVb.type()), MoeKernels.idMatVecGlobal(headV, 1, idLocal),
                new long[]{idLocal, 1}, MoeKernels.matVecIdArgs(headV, kvLora, 1, kvLora, 0, 0, false, 0, nHead,
                        slots, wVb.shard(0), att.buffer(), ph, vout.buffer())));

        GpuStepInputs step = ctx.memory().stepInputs();
        storeK = keep(MsaBlocks.prepareStore(ctx, kvLayout.dtypeK(), kvLora, kvLora, kvLayout.widthK(),
                kvl.buffer(), step.cellIdx(kv.group()), kv.k()));
        gather = keep(MsaBlocks.prepareGather(ctx, nHead, 1, kvLora, kpool, selWidth, m.kqScale(), 1,
                layer.dsa().kpoolSelectTail(), kvLayout, qabs.buffer(), nHead * kvLora, kv.k(), kv.k(), step.qPos(),
                step.kvPos(kv.group()), sel.buffer(), att.buffer(), nHead * kvLora));
    }

    /** Local size of the id-matvec kernels: the row-reduce size, capped at {@link MoeKernels#MAX_LOCAL}. */
    private int idLocal(GpuWeight... weights) {
        int l = Math.min(MoeKernels.MAX_LOCAL, local);
        for (GpuWeight t : weights) {
            MoeKernels.checkIdLocal(t.type(), l);
        }
        return l;
    }

    /**
     * Whole tokens per {@code {T}_MatVecId} call whose {@code n_head} slots fit a grid of at most {@code maxGroupsY}
     * work-groups in its second dimension (CUDA and Vulkan: 65535).
     *
     * @throws IllegalStateException when even one token's heads exceed the limit
     */
    private int idChunkTokens(LayerSpec layer, long maxGroupsY) {
        long tokens = Math.min(Integer.MAX_VALUE, maxGroupsY) / nHead;
        if (tokens < 1) {
            throw new IllegalStateException("layer " + MsaBlocks.describe(layer) + ": " + nHead + " MLA heads exceed"
                    + " the device's " + maxGroupsY + " work-groups in the second grid dimension");
        }
        return (int) Math.min(tokens, ubatch);
    }

    /** Device ints {@code s % nHead} for the {@code ubatch * nHead} (token, head) pairs of the per-head id-matvecs. */
    private GpuBuffer headSlots(GpuForwardContext ctx) {
        int n = Math.multiplyExact(ubatch, nHead);
        byte[] bytes = new byte[Math.multiplyExact(n, Integer.BYTES)];
        ByteBuffer view = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int s = 0; s < n; s++) {
            view.putInt(s * Integer.BYTES, s % nHead);
        }
        GpuBuffer b = ctx.backend().allocateBuffer(bytes.length, true);
        owned.add(b);
        GpuEvent e = ctx.queue().enqueueWriteBuffer(b, 0, bytes.length, bytes);
        if (e != null) {
            e.close();
        }
        return b;
    }

    // ------------------------------------------------------------------ qwen4exp

    private void bindGqa(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, GpuKvLayer kv) {
        AttentionSpec a = layer.attention();
        MsaBlocks.checkAttention(layer, a, AttnGateKind.Q_INTERLEAVED);
        if (a.qkNorm() != QkNormKind.RMS_HEAD || a.fusedQkv() || a.headDimV() != a.headDimK()) {
            throw new IllegalArgumentException("layer " + MsaBlocks.describe(layer) + ": QSA GQA needs separate"
                    + " attn_q/k/v, per-head RMS Q/K norms and equal K/V head sizes (qwen4exp.cpp:944-967)");
        }
        nHead = a.nHead();
        nKv = a.nHeadKv();
        hd = a.headDimK();
        qDim = a.qDim();
        kDim = a.kDim();
        if (kvLayout.vAliasesK() || kvLayout.widthK() != kDim || kvLayout.widthV() != kDim) {
            throw new IllegalStateException("layer " + MsaBlocks.describe(layer) + ": KV rows of " + kvLayout.widthK()
                    + " / " + kvLayout.widthV() + " values do not hold " + nKv + " heads of " + hd);
        }
        MsaBlocks.checkRotation(rotK, hd);
        MsaBlocks.checkRotation(rotV, hd);
        rope = MsaBlocks.rope(layer, hd, 0, "QSA attention");
        int nEmbd = MsaBlocks.mixerWidth(layer);
        wQ = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, 2L * qDim);
        wK = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, kDim);
        wV = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, kDim);
        wO = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, qDim, nEmbd);
        qNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), hd, true, owned);
        kNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), hd, true, owned);

        GpuScratch scratch = ctx.scratch();
        long ub = ubatch;
        qg = MsaBlocks.atZero(scratch.temp(TEMP_QG, ub * 2 * qDim));
        q = MsaBlocks.atZero(scratch.temp(TEMP_Q, ub * qDim));
        k = MsaBlocks.atZero(scratch.temp(TEMP_K, ub * kDim));
        v = MsaBlocks.atZero(scratch.temp(TEMP_V, ub * kDim));
        att = MsaBlocks.atZero(scratch.temp(TEMP_ATT, ub * qDim));

        GpuStepInputs step = ctx.memory().stepInputs();
        GpuBuffer ff = MsaBlocks.gpuFreqFactors(ctx, ws, rope, owned);
        ropeQ = keep(MsaBlocks.prepareRope(ctx, rope, ff, nHead, hd, qDim, step.ropePos(), q.buffer()));
        ropeK = keep(MsaBlocks.prepareRope(ctx, rope, ff, nKv, hd, kDim, step.ropePos(), k.buffer()));
        GpuBuffer cell = step.cellIdx(kv.group());
        storeK = keep(MsaBlocks.prepareStore(ctx, kvLayout.dtypeK(), kDim, kDim, kvLayout.widthK(), k.buffer(), cell,
                kv.k()));
        storeV = keep(MsaBlocks.prepareStore(ctx, kvLayout.dtypeV(), kDim, kDim, kvLayout.widthV(), v.buffer(), cell,
                kv.v()));
        gather = keep(MsaBlocks.prepareGather(ctx, nHead, nKv, hd, kpool, selWidth, a.kqScale(), 1,
                layer.dsa().kpoolSelectTail(), kvLayout, q.buffer(), qDim, kv.k(), kv.v(), step.qPos(),
                step.kvPos(kv.group()), sel.buffer(), att.buffer(), qDim));
    }

    // ------------------------------------------------------------------ indexer of a full layer

    private void bindIndexer(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, GpuIndexerCache cache) {
        DsaSpec dsa = layer.dsa();
        if (cache == null) {
            throw new IllegalStateException("full QSA layer " + MsaBlocks.describe(layer) + " has no indexer cache");
        }
        idxPlan = cache.plan();
        MsaBlocks.checkQsaIndexer(layer, dsa, idxPlan, kvLayout);
        int width = idxPlan.width();
        int gateOff = MsaKernels.kpoolGateOffset(dsa.kpoolRow(), idx);
        int pooledOff = MsaKernels.kpoolPooledOffset(dsa.kpoolRow(), idx);
        MsaKernels.checkKpool(kpool, idx, width, gateOff, pooledOff, idxPlan.rows(), mode);
        MsaKernels.checkKpoolCache(idxPlan.dtype(), idx, width, gateOff, pooledOff, mode);
        int nEmbd = MsaBlocks.mixerWidth(layer);
        GpuScratch scratch = ctx.scratch();
        GpuStepInputs step = ctx.memory().stepInputs();
        long ub = ubatch;
        int maxNew = MsaBlocks.maxNewPools(ubatch, kpool);
        iq = MsaBlocks.atZero(scratch.temp(TEMP_IQ, ub * nIdxH * idx));
        ik = MsaBlocks.atZero(scratch.temp(TEMP_IK, ub * idx));
        pooled = MsaBlocks.atZero(scratch.temp(TEMP_POOLED, (long) maxNew * idx));
        GpuBuffer ape;
        GpuBuffer weights;
        int storeElems;
        GpuBuffer storeSrc;
        if (mode == MsaKernels.KPOOL_MODE_GATED) {
            wIq = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_Q_B, qLora, (long) nIdxH * idx);
            wIk = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_K, nEmbd, idx);
            ikNormW = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true,
                    owned);
            ikNormB = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.bias(TensorNames.INDEXER_K_NORM), idx, true,
                    owned);
            wIg = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_KPOOL_GATE, nEmbd, idx);
            wProj = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_PROJ, nEmbd, nIdxH);
            ape = MsaBlocks.gpuF32(ctx, ws.layer(layer, TensorNames.weight(TensorNames.INDEXER_KPOOL_APE)),
                    (long) idx * kpool, owned);
            ig = MsaBlocks.atZero(scratch.temp(TEMP_IG, ub * idx));
            packed = MsaBlocks.atZero(scratch.temp(TEMP_PACKED, ub * 2 * idx));
            w = MsaBlocks.atZero(scratch.temp(TEMP_W, ub * nIdxH));
            weights = w.buffer();
            storeElems = 2 * idx;
            storeSrc = packed.buffer();
        } else {
            wIq = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_Q_PROJ, nEmbd, (long) nIdxH * idx);
            wIk = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_K_PROJ, nEmbd, idx);
            iqNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM), idx, true,
                    owned);
            ikNormW = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true,
                    owned);
            ape = ctx.placeholders().readOnlyFloats();
            weights = ctx.placeholders().readOnlyFloats();
            storeElems = idx;
            storeSrc = ik.buffer();
            bindIndexerRope(ctx, ws, layer, dsa, step, maxNew);
        }
        storeIdx = keep(MsaBlocks.prepareStore(ctx, idxPlan.dtype(), storeElems, storeElems, width, storeSrc,
                step.cellIdx(cache.group()), cache.keys()));
        kpoolUpdate = keep(ops.prepare(MsaKernels.kpoolUpdate(idxPlan.dtype()),
                MsaKernels.kpoolUpdateGlobal(1, idx, MsaKernels.KPOOL_UPDATE_LOCAL),
                MsaKernels.local(MsaKernels.KPOOL_UPDATE_LOCAL), 1, 0, MsaKernels.KPOOL_STAGE_POOL, mode, kpool, idx,
                width, gateOff, pooledOff, idxPlan.rows(), ape, cache.keys(), pooled.buffer()));
        if (selWidth > 0) {
            float wScale = dsa.weightScale();
            MsaKernels.checkKpoolScore(nIdxH, idx, kpool, width, pooledOff, idxPlan.rows(), wScale,
                    MsaKernels.KPOOL_SCORE_LOCAL);
            int maxPools = Math.max(1, MsaKernels.completePools(posLimit, kpool));
            ScratchSlot scores = MsaBlocks.atZero(scratch.temp(TEMP_SCORES, ub * maxPools));
            kpoolScore = keep(ops.prepare(MsaKernels.kpoolScore(idxPlan.dtype()),
                    MsaKernels.kpoolScoreGlobal(1, 1, MsaKernels.KPOOL_SCORE_LOCAL),
                    MsaKernels.local(MsaKernels.KPOOL_SCORE_LOCAL), 1, 1, nIdxH, idx, kpool, width, pooledOff,
                    idxPlan.rows(), mode == MsaKernels.KPOOL_MODE_GATED ? 1 : 0, wScale, 0, 0, iq.buffer(), weights,
                    cache.keys(), step.qPos(), step.kvPos(cache.group()), scores.buffer()));
            topKRows = keep(MsaBlocks.prepareTopK(ctx, selWidth, scores.buffer(), sel.buffer()));
        }
    }

    /** The qwen4exp indexer rope: the query heads at the step positions and the pooled keys at their pool positions. */
    private void bindIndexerRope(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer, DsaSpec dsa,
                                 GpuStepInputs step, int maxNew) {
        idxRope = MsaBlocks.rope(layer, idx, dsa.indexerRopeDim(), "QSA indexer");
        GpuBuffer ff = MsaBlocks.gpuFreqFactors(ctx, ws, idxRope, owned);
        int sections = MropeTextPositions.SECTIONS;
        poolPos = MsaBlocks.atZero(ctx.scratch().temp(TEMP_POOL_POS, (long) sections * maxNew));
        poolPosBytes = new byte[sections * maxNew * Integer.BYTES];
        poolPosView = ByteBuffer.wrap(poolPosBytes).order(ByteOrder.LITTLE_ENDIAN);
        ropeIq = keep(MsaBlocks.prepareRope(ctx, idxRope, ff, nIdxH, idx, nIdxH * idx, step.ropePos(), iq.buffer()));
        ropePool = keep(MsaBlocks.prepareRope(ctx, idxRope, ff, 1, idx, idx, poolPos.buffer(), pooled.buffer()));
    }

    private PreparedDispatch keep(PreparedDispatch d) {
        if (d != null) {
            prepared.add(d);
        }
        return d;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        run(pos0, nTokens, in, out);
    }

    private void run(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        if (!bound || closed) {
            throw new IllegalStateException("QSA mixer is not bound or already closed");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("QSA step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
        if (pos0 < 0 || pos0 + n > posLimit) {
            throw new IllegalArgumentException("QSA step at positions [" + pos0 + ", " + (pos0 + n) + ") exceeds the "
                    + posLimit + " cached positions");
        }
        if (mla) {
            runMla(pos0, n, in, out);
        } else {
            runGqa(pos0, n, in, out);
        }
    }

    private void runMla(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        ops.matmul(wQa, in.buffer(), in.offset(), qr.buffer(), 0, n);
        qaDeltas.applyRows(n, in.buffer(), in.offset(), qr.buffer(), 0);
        ops.norm(NormKind.RMS, qaNorm, null, rmsEps, qr.buffer(), 0, qr.buffer(), 0, n, qLora);
        ops.matmul(wQb, qr.buffer(), 0, qn.buffer(), 0, n);
        qbDeltas.applyRows(n, qr.buffer(), 0, qn.buffer(), 0);
        ops.matmul(wKva, in.buffer(), in.offset(), kvl.buffer(), 0, n);
        kvaDeltas.applyRows(n, in.buffer(), in.offset(), kvl.buffer(), 0);
        ops.norm(NormKind.RMS, kvaNorm, null, rmsEps, kvl.buffer(), 0, kvl.buffer(), 0, n, kvLora);
        runId(absorbK, kvLora, qkNope, n);
        if (full) {
            ops.matmul(wIq, qr.buffer(), 0, iq.buffer(), 0, n);
            iqDeltas.applyRows(n, qr.buffer(), 0, iq.buffer(), 0);
            ops.matmul(wIk, in.buffer(), in.offset(), ik.buffer(), 0, n);
            ikDeltas.applyRows(n, in.buffer(), in.offset(), ik.buffer(), 0);
            ops.norm(NormKind.LAYER, ikNormW, ikNormB, normEps, ik.buffer(), 0, ik.buffer(), 0, n, idx);
            ops.matmul(wIg, in.buffer(), in.offset(), ig.buffer(), 0, n);
            igDeltas.applyRows(n, in.buffer(), in.offset(), ig.buffer(), 0);
            ops.copy2d(ik.buffer(), 0, idx, packed.buffer(), 0, 2 * idx, n, idx);
            ops.copy2d(ig.buffer(), 0, idx, packed.buffer(), idx, 2 * idx, n, idx);
            MsaBlocks.runStore(storeIdx, idxPlan.dtype(), n, 2 * idx, local);
            pool(pos0, n);
            ops.matmul(wProj, in.buffer(), in.offset(), w.buffer(), 0, n);
            projDeltas.applyRows(n, in.buffer(), in.offset(), w.buffer(), 0);
            score(pos0, n);
        }
        MsaBlocks.rotate(ops, rotK, qabs.buffer(), n, nHead * kvLora);
        MsaBlocks.rotate(ops, rotK, kvl.buffer(), n, kvLora);
        MsaBlocks.runStore(storeK, kvLayout.dtypeK(), n, kvLora, local);
        MsaBlocks.runGather(gather, n, nHead);
        MsaBlocks.rotate(ops, rotK, att.buffer(), n, nHead * kvLora);
        runId(absorbV, headV, kvLora, n);
        ops.matmul(wO, vout.buffer(), 0, out.buffer(), out.offset(), n);
        oDeltas.applyRows(n, vout.buffer(), 0, out.buffer(), out.offset());
    }

    private void runGqa(int pos0, int n, ScratchSlot in, ScratchSlot out) {
        if (full) {
            ops.matmul(wIk, in.buffer(), in.offset(), ik.buffer(), 0, n);
            ikDeltas.applyRows(n, in.buffer(), in.offset(), ik.buffer(), 0);
            MsaBlocks.runStore(storeIdx, idxPlan.dtype(), n, idx, local);
            pool(pos0, n);
            ops.matmul(wIq, in.buffer(), in.offset(), iq.buffer(), 0, n);
            iqDeltas.applyRows(n, in.buffer(), in.offset(), iq.buffer(), 0);
            ops.norm(NormKind.RMS, iqNorm, null, rmsEps, iq.buffer(), 0, iq.buffer(), 0, n * nIdxH, idx);
            MsaBlocks.runRope(ropeIq, idxRope, n, nIdxH, n, local);
            score(pos0, n);
        }
        ops.matmul(wQ, in.buffer(), in.offset(), qg.buffer(), 0, n);
        qDeltas.applyRows(n, in.buffer(), in.offset(), qg.buffer(), 0);
        ops.copy2d(qg.buffer(), 0, 2 * hd, q.buffer(), 0, hd, n * nHead, hd);
        ops.norm(NormKind.RMS, qNorm, null, rmsEps, q.buffer(), 0, q.buffer(), 0, n * nHead, hd);
        ops.matmul(wK, in.buffer(), in.offset(), k.buffer(), 0, n);
        kDeltas.applyRows(n, in.buffer(), in.offset(), k.buffer(), 0);
        ops.matmul(wV, in.buffer(), in.offset(), v.buffer(), 0, n);
        vDeltas.applyRows(n, in.buffer(), in.offset(), v.buffer(), 0);
        ops.norm(NormKind.RMS, kNorm, null, rmsEps, k.buffer(), 0, k.buffer(), 0, n * nKv, hd);
        MsaBlocks.runRope(ropeQ, rope, n, nHead, n, local);
        MsaBlocks.runRope(ropeK, rope, n, nKv, n, local);
        MsaBlocks.rotate(ops, rotK, q.buffer(), n, qDim);
        MsaBlocks.rotate(ops, rotK, k.buffer(), n, kDim);
        MsaBlocks.rotate(ops, rotV, v.buffer(), n, kDim);
        MsaBlocks.runStore(storeK, kvLayout.dtypeK(), n, kDim, local);
        MsaBlocks.runStore(storeV, kvLayout.dtypeV(), n, kDim, local);
        MsaBlocks.runGather(gather, n, nHead);
        MsaBlocks.rotate(ops, rotV, att.buffer(), n, qDim);
        ops.glu(GluOp.SIGMUL, ActParams.NONE, qg.buffer(), hd, 2 * hd, att.buffer(), 0, hd, att.buffer(), 0, hd,
                n * nHead, hd);
        ops.matmul(wO, att.buffer(), 0, out.buffer(), out.offset(), n);
        oDeltas.applyRows(n, att.buffer(), 0, out.buffer(), out.offset());
    }

    /** Re-pools the pools the step completes (none when {@code n_new == 0}). */
    private void pool(int pos0, int n) {
        int nNew = MsaKernels.newPoolCount(pos0, n, kpool);
        if (nNew == 0) {
            return;
        }
        int first = MsaKernels.firstNewPool(pos0, kpool);
        kpoolUpdate.setInt(MsaKernels.ARG_KU_N_NEW, nNew);
        kpoolUpdate.setInt(MsaKernels.ARG_KU_FIRST_POOL, first);
        kpoolUpdate.setInt(MsaKernels.ARG_KU_STAGE, MsaKernels.KPOOL_STAGE_POOL);
        kpoolUpdate.setGlobal(MsaKernels.kpoolUpdateGlobal(nNew, idx, MsaKernels.KPOOL_UPDATE_LOCAL));
        kpoolUpdate.dispatch();
        if (mode == MsaKernels.KPOOL_MODE_MEAN) {
            ops.norm(NormKind.RMS, ikNormW, null, rmsEps, pooled.buffer(), 0, pooled.buffer(), 0, nNew, idx);
            uploadPoolPositions(first, nNew);
            MsaBlocks.runRope(ropePool, idxRope, nNew, 1, nNew, local);
        }
        kpoolUpdate.setInt(MsaKernels.ARG_KU_STAGE, MsaKernels.KPOOL_STAGE_STORE);
        kpoolUpdate.dispatch();
    }

    /**
     * Section-major RoPE positions of the new pools, {@code pos[s*nNew + j]}: the first member's position
     * {@code p = (first + j) * kpool} in sections 0 and 3 and its text-pattern height and width positions in sections
     * 1 and 2 (src/llama-memory-hybrid-idx.cpp:866-873, cell ext from src/llama-kv-cache.cpp:1155-1161). The queue
     * copies the host bytes when the write is enqueued.
     */
    private void uploadPoolPositions(int first, int nNew) {
        int sections = MropeTextPositions.SECTIONS;
        for (int j = 0; j < nNew; j++) {
            int p = (first + j) * kpool;
            for (int s = 0; s < sections; s++) {
                int value = s == 0 || s == sections - 1 ? p : idxRope.textPositions().position(s, p);
                poolPosView.putInt((s * nNew + j) * Integer.BYTES, value);
            }
        }
        GpuEvent e = ops.queue().enqueueWriteBuffer(poolPos.buffer(), 0, (long) sections * nNew * Integer.BYTES,
                poolPosBytes);
        if (e != null) {
            e.close();
        }
    }

    /** Scores the complete pools and selects {@code K} per token; -1 everywhere when no pool is complete yet. */
    private void score(int pos0, int n) {
        if (selWidth == 0) {
            return;
        }
        int nPool = MsaKernels.completePools(pos0 + n, kpool);
        if (nPool == 0) {
            ops.fill(sel.buffer(), 0, (long) n * selWidth * Integer.BYTES, -1);
            return;
        }
        kpoolScore.setInt(MsaKernels.ARG_KS_N_TOKENS, n);
        kpoolScore.setInt(MsaKernels.ARG_KS_N_POOL, nPool);
        kpoolScore.setGlobal(MsaKernels.kpoolScoreGlobal(n, nPool, MsaKernels.KPOOL_SCORE_LOCAL));
        kpoolScore.dispatch();
        MsaBlocks.runTopK(topKRows, n, nPool);
    }

    /**
     * A per-head {@code {T}_MatVecId} over the {@code n * n_head} (token, head) rows of the step, in calls of at most
     * {@link #idChunkTokens} tokens. Every call starts on a token boundary, so its slot {@code s} is head
     * {@code s % n_head} of the constant slot table.
     */
    private void runId(PreparedDispatch d, int outDim, int inDim, int n) {
        for (int t0 = 0; t0 < n; t0 += idChunkTokens) {
            int nSel = Math.min(idChunkTokens, n - t0) * nHead;
            int row0 = t0 * nHead;
            d.setInt(MoeKernels.ARG_ID_N_SEL, nSel);
            d.setInt(MoeKernels.ARG_ID_IN_OFF, row0 * inDim);
            d.setInt(MoeKernels.ARG_ID_OUT_OFF, row0 * outDim);
            d.setGlobal(MoeKernels.idMatVecGlobal(outDim, nSel, idLocal));
            d.dispatch();
        }
    }

    // ================================================================== close

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        MsaBlocks.release(prepared, owned);
    }

    @Override
    public String toString() {
        if (!bound) {
            return "GpuQsaMixer{unbound}";
        }
        return "GpuQsaMixer{" + (mla ? "MLA latent " + kvLora : "GQA " + nHead + "/" + nKv + "x" + hd) + ", kpool "
                + kpool + ", select " + selWidth + (full ? ", full indexer" : ", shared indexer") + '}';
    }
}

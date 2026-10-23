package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
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
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMsaCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU MiniMax-M3 block-sparse attention of one sparse layer ({@code msa().sparse()}), following
 * models/minimax-m3.cpp:306-535 for one sequence. With {@code T} tokens at positions {@code pos0 ..}, {@code H} query
 * heads, {@code G} KV heads (one index head per GQA group, {@code idxHeads == G}), head size {@code hd}, index size
 * {@code I}, block size {@code B} and {@code n_pos = pos0 + T}:
 *
 * <pre>
 * attn_qkv(x) -&gt; msa.qkv[T][Q|K|V] (+ bias), LLM_Copy2D_F32 -&gt; msa.q, msa.k, msa.v   (build_qkv, fused)
 *   or attn_q/k/v(x) -&gt; msa.q[T][H*hd], msa.k[T][G*hd], msa.v[T][G*hd] (+ biases)   (build_qkv, separate)
 * LLM_RowNorm RMS(attn_q_norm) per q head, RMS(attn_k_norm) per k head               (:315-318)
 * LLM_RoPE_Ext(n_rot) on q and k at the step positions                              (:320-328)
 * indexer.q_proj(x) -&gt; msa.iq[T][G][I], indexer.k_proj(x) -&gt; msa.ik[T][I]         (:344-347)
 * LLM_RowNorm RMS(indexer.q_norm) per iq head, RMS(indexer.k_norm); LLM_RoPE_Ext      (:348-352)
 * LLM_KV_Store(ik) -&gt; MSA index-key cache at its group's cell_idx                   (:354-357)
 * LLM_HadamardRotate q, k (n_rot_k) and v (n_rot_v) when the cache plan rotates       (:361-367)
 * LLM_KV_Store(k), LLM_KV_Store(v) -&gt; layer cache at cell_idx                       (:369-373)
 * LLM_MSA_BlockScore(iq, index keys) -&gt; msa.scores[T*G][ceil(n_pos/B)]              (:388-406, :473-488)
 * LLM_TopK_Rows(top_k_blocks) -&gt; msa.blocks[T*G][top_k]                             (:405, :488)
 * LLM_MSA_GatherAttn(q, K/V cache, blocks, kq_scale) -&gt; msa.att[T][H*hd]            (:408-447, :490-522)
 * LLM_HadamardRotate att (n_rot_v) when V is rotated                                 (:529-531)
 * attn_output(att) -&gt; out, with the weight's wo_s scale                              (:533-534)
 * </pre>
 *
 * The rope, store, score, top-k and gather launches are {@link PreparedDispatch}es bound once; a step re-sets only
 * the token count, the position count and the grid. The projections, norms, copies and rotations are
 * {@link GpuOps} calls. No host synchronization; the temporaries carry data only within one call. Not thread-safe.
 */
public final class GpuMsaMixer implements GpuMixer {

    static final String TEMP_QKV = "msa.qkv";
    static final String TEMP_Q = "msa.q";
    static final String TEMP_K = "msa.k";
    static final String TEMP_V = "msa.v";
    static final String TEMP_IQ = "msa.iq";
    static final String TEMP_IK = "msa.ik";
    static final String TEMP_ATT = "msa.att";
    static final String TEMP_SCORES = "msa.scores";
    static final String TEMP_BLOCKS = "msa.blocks";

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();

    private GpuOps ops;
    private int local;
    private int ubatch;
    private int nHead;
    private int nKv;
    private int hd;
    private int qDim;
    private int kDim;
    private int idx;
    private int blk;
    private int posLimit;
    private int rotK;
    private int rotV;
    private float rmsEps;
    private RopeSpec rope;
    private KvLayout kvLayout;
    private LayerMemoryPlan.MsaPlan msaPlan;

    private GpuWeight wQkv;
    private GpuWeight wQ;
    private GpuWeight wK;
    private GpuWeight wV;
    private GpuWeight wO;
    private GpuWeight qNorm;
    private GpuWeight kNorm;
    private GpuWeight wIq;
    private GpuWeight wIk;
    /** LoRA delta sites of the bound projections (null for absent ones); see {@link #bindDeltas}. */
    private GpuProjectionDeltas.Site qkvDeltas;
    private GpuProjectionDeltas.Site qDeltas;
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site oDeltas;
    private GpuProjectionDeltas.Site iqDeltas;
    private GpuProjectionDeltas.Site ikDeltas;
    private GpuWeight iqNorm;
    private GpuWeight ikNorm;

    private ScratchSlot qkv;
    private ScratchSlot q;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot iq;
    private ScratchSlot ik;
    private ScratchSlot att;

    private PreparedDispatch biasQkv;
    private PreparedDispatch biasQ;
    private PreparedDispatch biasK;
    private PreparedDispatch biasV;
    private PreparedDispatch ropeQ;
    private PreparedDispatch ropeK;
    private PreparedDispatch ropeIq;
    private PreparedDispatch ropeIk;
    private PreparedDispatch storeIk;
    private PreparedDispatch storeK;
    private PreparedDispatch storeV;
    private PreparedDispatch blockScore;
    private PreparedDispatch topKRows;
    private PreparedDispatch gather;
    private int blockScoreLocal;

    private boolean bound;
    private boolean closed;

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException when the layer is not a sparse MSA layer or uses an attention feature the
     *                                  mixer does not implement
     * @throws IllegalStateException    when the context has no memory, a cache or tensor has the wrong shape, or the
     *                                  mixer is already bound or closed
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("MSA mixer is already bound or closed");
        }
        MsaBlocks.checkVariant(layer, MsaBlocks.Variant.MSA);
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
        AttentionSpec a = layer.attention();
        MsaSpec msa = layer.msa();
        checkShapes(layer, a, msa);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("MSA layer " + MsaBlocks.describe(layer) + " needs a KV cache");
        }
        ops = ctx.ops();
        local = ops.local();
        ubatch = ctx.scratch().ubatch();
        rmsEps = ctx.spec().hparams().normRmsEps();
        rope = MsaBlocks.rope(layer, hd, 0, "MSA attention");
        MsaBlocks.rope(layer, idx, msa.idxRopeDim(), "MSA indexer");

        GpuMemory memory = ctx.memory();
        GpuKvLayer kv = memory.kv(layer.il());
        GpuMsaCache keys = memory.msa(layer.il());
        if (kv == null || keys == null) {
            throw new IllegalStateException("MSA layer " + MsaBlocks.describe(layer) + " needs a KV cache and an"
                    + " index-key cache");
        }
        checkCaches(layer, kv, keys);

        GpuWeights ws = ctx.weightSource().weightsFor(layer.weightLayer());
        bindWeights(ctx, ws, layer);
        bindDeltas(ctx.projectionDeltas(), layer);
        bindScratch(ctx.scratch());
        bindBiases(ctx, ws, layer);
        bindDispatches(ctx, ws, kv, keys, msa);
    }

    private void checkShapes(LayerSpec layer, AttentionSpec a, MsaSpec msa) {
        MsaBlocks.checkAttention(layer, a, AttnGateKind.NONE);
        String where = "layer " + MsaBlocks.describe(layer) + ": ";
        if (a.qkNorm() != QkNormKind.RMS_HEAD) {
            throw new IllegalArgumentException(where + "MSA applies per-head RMS Q/K norms (minimax-m3.cpp:315-318),"
                    + " the spec has " + a.qkNorm());
        }
        nHead = a.nHead();
        nKv = a.nHeadKv();
        hd = a.headDimK();
        if (a.headDimV() != hd) {
            throw new IllegalArgumentException(where + "MSA needs equal K and V head sizes (minimax-m3.cpp:212), got "
                    + hd + " / " + a.headDimV());
        }
        if (msa.idxHeads() != nKv) {
            throw new IllegalArgumentException(where + "MSA needs one index head per KV head (minimax-m3.cpp:382), got "
                    + msa.idxHeads() + " for " + nKv);
        }
        qDim = a.qDim();
        kDim = a.kDim();
        idx = msa.idxHeadDim();
        blk = msa.blockSize();
    }

    private void checkCaches(LayerSpec layer, GpuKvLayer kv, GpuMsaCache keys) {
        String where = "layer " + MsaBlocks.describe(layer) + ": ";
        kvLayout = kv.layout();
        msaPlan = keys.plan();
        MsaBlocks.checkNoRing(layer, kvLayout, "KV cache");
        if (kvLayout.vAliasesK() || kvLayout.widthK() != kDim || kvLayout.widthV() != kDim) {
            throw new IllegalStateException(where + "KV rows of " + kvLayout.widthK() + " / " + kvLayout.widthV()
                    + " values do not hold " + nKv + " heads of " + hd);
        }
        if (msaPlan.width() != idx) {
            throw new IllegalStateException(where + "index-key rows of " + msaPlan.width()
                    + " values, the indexer writes " + idx);
        }
        posLimit = Math.min(kvLayout.rows(), msaPlan.rows());
        RotationPlan rot = kv.rotation();
        rotK = rot.nRotK();
        rotV = rot.nRotV();
        MsaBlocks.checkRotation(rotK, hd);
        MsaBlocks.checkRotation(rotV, hd);
    }

    private void bindWeights(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer) {
        int nEmbd = MsaBlocks.mixerWidth(layer);
        wQkv = MsaBlocks.gpuMatrixOrNull(ws, layer, TensorNames.ATTN_QKV, nEmbd, qDim + 2L * kDim);
        if (wQkv == null) {
            wQ = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, qDim);
            wK = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, kDim);
            wV = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, kDim);
        }
        wO = MsaBlocks.gpuMatrix(ws, layer, TensorNames.ATTN_OUT, qDim, nEmbd);
        qNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), hd, true, owned);
        kNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), hd, true, owned);
        wIq = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_Q_PROJ, nEmbd, (long) nKv * idx);
        wIk = MsaBlocks.gpuMatrix(ws, layer, TensorNames.INDEXER_K_PROJ, nEmbd, idx);
        iqNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM), idx, true, owned);
        ikNorm = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true, owned);
    }

    /**
     * Serves the LoRA delta site of every bound dense projection; each delta is added right after its product and
     * before any bias ({@code build_lora_mm}, llama-graph.cpp:1629-1667).
     */
    private void bindDeltas(GpuProjectionDeltas deltas, LayerSpec layer) {
        qkvDeltas = serveDelta(deltas, layer, TensorNames.ATTN_QKV, wQkv);
        qDeltas = serveDelta(deltas, layer, TensorNames.ATTN_Q, wQ);
        kDeltas = serveDelta(deltas, layer, TensorNames.ATTN_K, wK);
        vDeltas = serveDelta(deltas, layer, TensorNames.ATTN_V, wV);
        oDeltas = serveDelta(deltas, layer, TensorNames.ATTN_OUT, wO);
        iqDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_Q_PROJ, wIq);
        ikDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_K_PROJ, wIk);
    }

    private GpuProjectionDeltas.Site serveDelta(GpuProjectionDeltas deltas, LayerSpec layer, String base, GpuWeight w) {
        return w == null ? null : deltas.serve(layer, TensorNames.weight(base), w);
    }

    private void bindScratch(GpuScratch scratch) {
        long ub = ubatch;
        if (wQkv != null) {
            qkv = MsaBlocks.atZero(scratch.temp(TEMP_QKV, ub * (qDim + 2L * kDim)));
        }
        q = MsaBlocks.atZero(scratch.temp(TEMP_Q, ub * qDim));
        k = MsaBlocks.atZero(scratch.temp(TEMP_K, ub * kDim));
        v = MsaBlocks.atZero(scratch.temp(TEMP_V, ub * kDim));
        iq = MsaBlocks.atZero(scratch.temp(TEMP_IQ, ub * nKv * idx));
        ik = MsaBlocks.atZero(scratch.temp(TEMP_IK, ub * idx));
        att = MsaBlocks.atZero(scratch.temp(TEMP_ATT, ub * qDim));
    }

    /**
     * Bias dispatches of {@code build_qkv} (llama-graph.cpp:1823-1834, 1856-1889): the fused bias on the fused rows;
     * otherwise the separate biases on the compact Q/K/V rows, which a fused weight takes only when all three exist.
     */
    private void bindBiases(GpuForwardContext ctx, GpuWeights ws, LayerSpec layer) {
        if (wQkv != null) {
            int qkvW = qDim + 2 * kDim;
            GpuWeight bqkv = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.bias(TensorNames.ATTN_QKV), qkvW, false,
                    owned);
            if (bqkv != null) {
                biasQkv = keep(MsaBlocks.prepareBiasAdd(ctx, bqkv, qkvW, qkv.buffer()));
                return;
            }
        }
        GpuWeight bq = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.bias(TensorNames.ATTN_Q), qDim, false, owned);
        GpuWeight bk = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.bias(TensorNames.ATTN_K), kDim, false, owned);
        GpuWeight bv = MsaBlocks.gpuVector(ctx, ws, layer, TensorNames.bias(TensorNames.ATTN_V), kDim, false, owned);
        if (wQkv != null && (bq == null || bk == null || bv == null)) {
            return;
        }
        biasQ = keep(MsaBlocks.prepareBiasAdd(ctx, bq, qDim, q.buffer()));
        biasK = keep(MsaBlocks.prepareBiasAdd(ctx, bk, kDim, k.buffer()));
        biasV = keep(MsaBlocks.prepareBiasAdd(ctx, bv, kDim, v.buffer()));
    }

    private void bindDispatches(GpuForwardContext ctx, GpuWeights ws, GpuKvLayer kv, GpuMsaCache keys, MsaSpec msa) {
        GpuScratch scratch = ctx.scratch();
        GpuStepInputs step = ctx.memory().stepInputs();
        GpuBuffer ff = MsaBlocks.gpuFreqFactors(ctx, ws, rope, owned);
        GpuBuffer pos = step.ropePos();
        ropeQ = keep(MsaBlocks.prepareRope(ctx, rope, ff, nHead, hd, qDim, pos, q.buffer()));
        ropeK = keep(MsaBlocks.prepareRope(ctx, rope, ff, nKv, hd, kDim, pos, k.buffer()));
        ropeIq = keep(MsaBlocks.prepareRope(ctx, rope, ff, nKv, idx, nKv * idx, pos, iq.buffer()));
        ropeIk = keep(MsaBlocks.prepareRope(ctx, rope, ff, 1, idx, idx, pos, ik.buffer()));

        storeIk = keep(MsaBlocks.prepareStore(ctx, msaPlan.dtype(), idx, idx, msaPlan.width(), ik.buffer(),
                step.cellIdx(keys.group()), keys.indexKeys()));
        GpuBuffer cell = step.cellIdx(kv.group());
        storeK = keep(MsaBlocks.prepareStore(ctx, kvLayout.dtypeK(), kDim, kDim, kvLayout.widthK(), k.buffer(), cell,
                kv.k()));
        storeV = keep(MsaBlocks.prepareStore(ctx, kvLayout.dtypeV(), kDim, kDim, kvLayout.widthV(), v.buffer(), cell,
                kv.v()));

        int topK = msa.topKBlocks();
        long ub = ubatch;
        blockScoreLocal = MsaKernels.blockScoreLocal(blk);
        MsaKernels.checkBlockScore(nKv, idx, blk, msa.localBlocks(), msaPlan.rows(), msaPlan.dtype(), blockScoreLocal);
        ScratchSlot scores = MsaBlocks.atZero(scratch.temp(TEMP_SCORES,
                ub * nKv * MsaKernels.blockCount(posLimit, blk)));
        ScratchSlot blocks = MsaBlocks.atZero(scratch.temp(TEMP_BLOCKS, ub * nKv * topK));
        blockScore = keep(ops.prepare(MsaKernels.MSA_BLOCK_SCORE,
                MsaKernels.blockScoreGlobal(1, 1, blk, blockScoreLocal), MsaKernels.local(blockScoreLocal),
                1, 1, nKv, idx, blk, msa.localBlocks(), msaPlan.rows(), MsaKernels.kvTypeCode(msaPlan.dtype()), 0,
                iq.buffer(), keys.indexKeys(), step.qPos(), step.kvPos(keys.group()), scores.buffer()));
        topKRows = keep(MsaBlocks.prepareTopK(ctx, topK, scores.buffer(), blocks.buffer()));
        gather = keep(MsaBlocks.prepareGather(ctx, nHead, nKv, hd, blk, topK, msa.kqScale(), nKv, false, kvLayout,
                q.buffer(), qDim, kv.k(), kv.v(), step.qPos(), step.kvPos(kv.group()), blocks.buffer(), att.buffer(),
                qDim));
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
            throw new IllegalStateException("MSA mixer is not bound or already closed");
        }
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("MSA step of " + n + " tokens is outside [1, " + ubatch + "]");
        }
        int nPos = pos0 + n;
        if (pos0 < 0 || nPos > posLimit) {
            throw new IllegalArgumentException("MSA step at positions [" + pos0 + ", " + nPos + ") exceeds the "
                    + posLimit + " cached positions");
        }
        project(n, in);
        ops.norm(NormKind.RMS, qNorm, null, rmsEps, q.buffer(), 0, q.buffer(), 0, n * nHead, hd);
        ops.norm(NormKind.RMS, kNorm, null, rmsEps, k.buffer(), 0, k.buffer(), 0, n * nKv, hd);
        MsaBlocks.runRope(ropeQ, rope, n, nHead, n, local);
        MsaBlocks.runRope(ropeK, rope, n, nKv, n, local);

        ops.matmul(wIq, in.buffer(), in.offset(), iq.buffer(), 0, n);
        iqDeltas.applyRows(n, in.buffer(), in.offset(), iq.buffer(), 0);
        ops.matmul(wIk, in.buffer(), in.offset(), ik.buffer(), 0, n);
        ikDeltas.applyRows(n, in.buffer(), in.offset(), ik.buffer(), 0);
        ops.norm(NormKind.RMS, iqNorm, null, rmsEps, iq.buffer(), 0, iq.buffer(), 0, n * nKv, idx);
        ops.norm(NormKind.RMS, ikNorm, null, rmsEps, ik.buffer(), 0, ik.buffer(), 0, n, idx);
        MsaBlocks.runRope(ropeIq, rope, n, nKv, n, local);
        MsaBlocks.runRope(ropeIk, rope, n, 1, n, local);
        MsaBlocks.runStore(storeIk, msaPlan.dtype(), n, idx, local);

        MsaBlocks.rotate(ops, rotK, q.buffer(), n, qDim);
        MsaBlocks.rotate(ops, rotK, k.buffer(), n, kDim);
        MsaBlocks.rotate(ops, rotV, v.buffer(), n, kDim);
        MsaBlocks.runStore(storeK, kvLayout.dtypeK(), n, kDim, local);
        MsaBlocks.runStore(storeV, kvLayout.dtypeV(), n, kDim, local);

        int nblk = MsaKernels.blockCount(nPos, blk);
        blockScore.setInt(MsaKernels.ARG_BS_N_TOKENS, n);
        blockScore.setInt(MsaKernels.ARG_BS_N_POS, nPos);
        blockScore.setGlobal(MsaKernels.blockScoreGlobal(n, nPos, blk, blockScoreLocal));
        blockScore.dispatch();
        MsaBlocks.runTopK(topKRows, n * nKv, nblk);
        MsaBlocks.runGather(gather, n, nHead);

        MsaBlocks.rotate(ops, rotV, att.buffer(), n, qDim);
        ops.matmul(wO, att.buffer(), 0, out.buffer(), out.offset(), n);
        oDeltas.applyRows(n, att.buffer(), 0, out.buffer(), out.offset());
    }

    /** Q, K and V of {@code build_qkv} into the compact {@code msa.q/k/v} rows, biases included. */
    private void project(int n, ScratchSlot in) {
        if (wQkv != null) {
            int qkvW = qDim + 2 * kDim;
            ops.matmul(wQkv, in.buffer(), in.offset(), qkv.buffer(), 0, n);
            qkvDeltas.applyRows(n, in.buffer(), in.offset(), qkv.buffer(), 0);
            MsaBlocks.runBiasAdd(biasQkv, n);
            ops.copy2d(qkv.buffer(), 0, qkvW, q.buffer(), 0, qDim, n, qDim);
            ops.copy2d(qkv.buffer(), qDim, qkvW, k.buffer(), 0, kDim, n, kDim);
            ops.copy2d(qkv.buffer(), qDim + kDim, qkvW, v.buffer(), 0, kDim, n, kDim);
        } else {
            ops.matmul(wQ, in.buffer(), in.offset(), q.buffer(), 0, n);
            qDeltas.applyRows(n, in.buffer(), in.offset(), q.buffer(), 0);
            ops.matmul(wK, in.buffer(), in.offset(), k.buffer(), 0, n);
            kDeltas.applyRows(n, in.buffer(), in.offset(), k.buffer(), 0);
            ops.matmul(wV, in.buffer(), in.offset(), v.buffer(), 0, n);
            vDeltas.applyRows(n, in.buffer(), in.offset(), v.buffer(), 0);
        }
        MsaBlocks.runBiasAdd(biasQ, n);
        MsaBlocks.runBiasAdd(biasK, n);
        MsaBlocks.runBiasAdd(biasV, n);
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
        return "GpuMsaMixer{" + (bound ? nHead + "q/" + nKv + "kv heads of " + hd + ", index " + idx + ", block " + blk
                + (rotK > 0 || rotV > 0 ? ", rotated" : "") : "unbound") + '}';
    }
}

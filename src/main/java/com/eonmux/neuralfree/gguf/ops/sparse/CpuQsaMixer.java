package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Objects;

/**
 * CPU twin of {@link GpuQsaMixer}: the same QSA k-pool layer (glm5-next latent MLA, models/glm5-next.cpp:892-1108;
 * qwen4exp gated GQA, models/qwen4exp.cpp:741-1026) with the same order of operations. Call sequence of a step of
 * {@code T} tokens at {@code pos0}, with {@code n_new} pools completed and {@code n_pool} complete:
 * <pre>
 * glm5-next:
 *   matmul attn_q_a; norm RMS attn_q_a_norm; matmul attn_q_b; matmul attn_kv_a_mqa; norm RMS attn_kv_a_norm
 *   matvecExpert attn_k_b per (token, head) -&gt; q_abs
 *   full: matmul indexer.attn_q_b(qr), indexer.attn_k; norm LAYER indexer.k_norm; matmul indexer_compressor_gate;
 *         writeK [key | gate | 0] at rowOf(pos); CpuQsa.kpoolPool GATED, CpuQsa.kpoolStore (n_new &gt; 0);
 *         matmul indexer.proj; CpuQsa.kpoolScore, CpuTopK.topKRows (n_pool &gt; 0, else selection = -1)
 *   cpuRotate q_abs, kv (n_rot_k); writeK(kv); CpuMsa.gatherAttn (latent K = V); cpuRotate out (n_rot_k)
 *   matvecExpert attn_v_b per (token, head); matmul attn_output
 * qwen4exp:
 *   matmul indexer.k_proj; writeK [raw | 0] at rowOf(pos); CpuQsa.kpoolPool MEAN; norm RMS indexer.k_norm;
 *   CpuRope.apply at the pool positions; CpuQsa.kpoolStore (n_new &gt; 0)
 *   matmul indexer.q_proj; norm RMS indexer.q_norm; CpuRope.apply; CpuQsa.kpoolScore, CpuTopK.topKRows
 *   matmul attn_q (q | gate per head), copy q; norm RMS attn_q_norm; matmul attn_k, attn_v; norm RMS attn_k_norm;
 *   CpuRope.apply q, k; cpuRotate q, k (n_rot_k), v (n_rot_v); writeK(k), writeV(v); CpuMsa.gatherAttn;
 *   cpuRotate out (n_rot_v); gluRows SIGMUL(gate, out); matmul attn_output
 * </pre>
 * The cached indexer rows are written whole with a zero pooled slot, as llama.cpp packs them; the STORE stage then
 * fills the pooled slot of each completed pool's last member. A shared-indexer layer reads the selection the
 * preceding full layer left in the scratch temporary {@link MsaBlocks#QSA_SELECTION}. Not thread-safe.
 */
public final class CpuQsaMixer implements CpuMixer {

    private CpuMatVec mv;
    private CpuMemory memory;
    private int ubatch;
    private float rmsEps;
    private float normEps;
    private boolean mla;
    private boolean full;
    private boolean tail;
    private int nHead;
    private int posLimit;
    private int rotK;
    private int rotV;
    private float kqScale;
    private KvAccessor kv;
    private KvLayout kvLayout;
    private int[] qPos;
    private int[] kvPos;
    private int[] sel;
    private float[] att;

    // selection
    private int kpool;
    private int selWidth;
    private int idx;
    private int nIdxH;
    private int mode;
    private int width;
    private int gateOff;
    private int pooledOff;
    private float wScale;
    private KvAccessor idxCache;
    private KvLayout idxLayout;
    private CpuTensorRef wIq;
    private CpuTensorRef wIk;
    private CpuTensorRef wIg;
    private CpuTensorRef wProj;
    /** LoRA delta sites of the bound projections (null for absent ones); see {@link #bindDeltas}. */
    private CpuProjectionDeltas.Site qaDeltas;
    private CpuProjectionDeltas.Site qbDeltas;
    private CpuProjectionDeltas.Site kvaDeltas;
    private CpuProjectionDeltas.Site qDeltas;
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site oDeltas;
    private CpuProjectionDeltas.Site iqDeltas;
    private CpuProjectionDeltas.Site ikDeltas;
    private CpuProjectionDeltas.Site igDeltas;
    private CpuProjectionDeltas.Site projDeltas;
    private float[] ikNormW;
    private float[] ikNormB;
    private float[] iqNorm;
    private float[] ape;
    private float[] iq;
    private float[] ik;
    private float[] ig;
    private float[] rows;
    private float[] w;
    private float[] pooled;
    private float[] scores;
    private int[] idxPos;
    private int[] poolPos;
    private RopeSpec idxRope;
    private RopeFreqTable idxTable;

    // glm5-next latent MLA
    private int qLora;
    private int kvLora;
    private int qkNope;
    private int headV;
    private CpuTensorRef wQa;
    private CpuTensorRef wQb;
    private CpuTensorRef wKva;
    private CpuTensorRef wKb;
    private CpuTensorRef wVb;
    private float[] qaNorm;
    private float[] kvaNorm;
    private float[] qr;
    private float[] qn;
    private float[] kvl;
    private float[] qabs;
    private float[] vout;

    // qwen4exp gated GQA
    private int nKv;
    private int hd;
    private int qDim;
    private int kDim;
    private RopeSpec rope;
    private RopeFreqTable ropeTable;
    private int[] ropePos;
    private float[] qNorm;
    private float[] kNorm;
    private float[] qg;
    private float[] q;
    private float[] k;
    private float[] v;

    private CpuTensorRef wQ;
    private CpuTensorRef wK;
    private CpuTensorRef wV;
    private CpuTensorRef wO;

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
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("QSA mixer is already bound or closed");
        }
        MsaBlocks.checkVariant(layer, MsaBlocks.Variant.QSA);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("QSA layer " + MsaBlocks.describe(layer) + " needs a KV cache");
        }
        mv = ctx.matVec();
        memory = ctx.memory();
        CpuScratch scratch = ctx.scratch();
        ubatch = scratch.ubatch();
        rmsEps = ctx.spec().hparams().normRmsEps();
        normEps = ctx.spec().hparams().normEps();
        DsaSpec dsa = layer.dsa();
        mla = layer.mla() != null;
        full = dsa.full();
        tail = dsa.kpoolSelectTail();
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

        kv = memory.kv(layer.il());
        if (kv == null) {
            throw new IllegalStateException("QSA layer " + MsaBlocks.describe(layer) + " has no KV cache");
        }
        kvLayout = kv.layout();
        MsaBlocks.checkNoRing(layer, kvLayout, "KV cache");
        LayerMemoryPlan plan = memory.plan().layer(layer.il());
        RotationPlan rot = plan.rotation();
        rotK = rot.nRotK();
        rotV = rot.nRotV();
        posLimit = kvLayout.rows();

        CpuWeights ws = ctx.weights();
        int ub = ubatch;
        sel = scratch.intTemp(MsaBlocks.QSA_SELECTION, MsaBlocks.selectionInts(ctx.spec(), ub));
        qPos = scratch.intTemp("qsa.q_pos", ub);
        kvPos = scratch.intTemp("qsa.kv_pos", kvLayout.rows());
        if (mla) {
            bindMla(ws, layer, scratch);
        } else {
            bindGqa(ws, layer, scratch);
        }
        if (full) {
            bindIndexer(ws, layer, dsa, plan.indexer(), scratch);
        }
        bindDeltas(ctx.projectionDeltas(), layer);
        bound = true;
    }

    /**
     * Serves the LoRA delta site of every bound dense projection; each delta is added right after its product and
     * before any bias ({@code build_lora_mm}, llama-graph.cpp:1629-1667).
     */
    private void bindDeltas(CpuProjectionDeltas deltas, LayerSpec layer) {
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

    private CpuProjectionDeltas.Site serveDelta(CpuProjectionDeltas deltas, LayerSpec layer, String base, CpuTensorRef w) {
        return w == null ? null : deltas.serve(layer, TensorNames.weight(base), w, mv);
    }

    private void bindMla(CpuWeights ws, LayerSpec layer, CpuScratch scratch) {
        MlaSpec m = layer.mla();
        MsaBlocks.checkQsaMla(layer, m, kvLayout, rotV);
        nHead = m.nHead();
        qLora = m.qLoraRank();
        kvLora = m.kvLoraRank();
        qkNope = m.headKNope();
        headV = m.headV();
        kqScale = m.kqScale();
        if (kv.nHeadKv() != 1 || kv.headDimK() != kvLora || kv.headDimV() != kvLora) {
            throw new IllegalStateException("layer " + MsaBlocks.describe(layer) + ": the latent cache reads "
                    + kv.nHeadKv() + " heads of " + kv.headDimK() + " / " + kv.headDimV() + ", QSA MLA needs one of "
                    + kvLora);
        }
        MsaBlocks.checkRotation(rotK, kvLora);
        int nEmbd = MsaBlocks.mixerWidth(layer);
        wQa = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q_A, nEmbd, qLora);
        qaNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_Q_A_NORM), qLora, true);
        wQb = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q_B, qLora, (long) nHead * qkNope);
        wKva = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_KV_A_MQA, nEmbd, kvLora);
        kvaNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_KV_A_NORM), kvLora, true);
        wKb = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_K_B, qkNope, (long) kvLora * nHead);
        wVb = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_V_B, kvLora, (long) headV * nHead);
        wO = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, (long) nHead * headV, nEmbd);

        int ub = ubatch;
        qr = scratch.temp(GpuQsaMixer.TEMP_QR, Math.multiplyExact(ub, qLora));
        qn = scratch.temp(GpuQsaMixer.TEMP_Q, Math.multiplyExact(ub, nHead * qkNope));
        kvl = scratch.temp(GpuQsaMixer.TEMP_K, Math.multiplyExact(ub, kvLora));
        qabs = scratch.temp(GpuQsaMixer.TEMP_QABS, Math.multiplyExact(ub, nHead * kvLora));
        att = scratch.temp(GpuQsaMixer.TEMP_ATT, Math.multiplyExact(ub, nHead * kvLora));
        vout = scratch.temp(GpuQsaMixer.TEMP_VOUT, Math.multiplyExact(ub, nHead * headV));
    }

    private void bindGqa(CpuWeights ws, LayerSpec layer, CpuScratch scratch) {
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
        kqScale = a.kqScale();
        if (kvLayout.vAliasesK() || kv.nHeadKv() != nKv || kv.headDimK() != hd || kv.headDimV() != hd) {
            throw new IllegalStateException("layer " + MsaBlocks.describe(layer) + ": the KV cache holds "
                    + kv.nHeadKv() + " heads of " + kv.headDimK() + " / " + kv.headDimV() + ", QSA needs " + nKv
                    + " of " + hd);
        }
        MsaBlocks.checkRotation(rotK, hd);
        MsaBlocks.checkRotation(rotV, hd);
        rope = MsaBlocks.rope(layer, hd, 0, "QSA attention");
        ropeTable = RopeFreqTable.of(rope, MsaBlocks.cpuFreqFactors(ws, rope));
        int nEmbd = MsaBlocks.mixerWidth(layer);
        wQ = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, 2L * qDim);
        wK = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, kDim);
        wV = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, kDim);
        wO = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, qDim, nEmbd);
        qNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), hd, true);
        kNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), hd, true);

        int ub = ubatch;
        qg = scratch.temp(GpuQsaMixer.TEMP_QG, Math.multiplyExact(ub, 2 * qDim));
        q = scratch.temp(GpuQsaMixer.TEMP_Q, Math.multiplyExact(ub, qDim));
        k = scratch.temp(GpuQsaMixer.TEMP_K, Math.multiplyExact(ub, kDim));
        v = scratch.temp(GpuQsaMixer.TEMP_V, Math.multiplyExact(ub, kDim));
        att = scratch.temp(GpuQsaMixer.TEMP_ATT, Math.multiplyExact(ub, qDim));
        ropePos = scratch.intTemp("qsa.rope_pos", Math.multiplyExact(MropeTextPositions.SECTIONS, ub));
    }

    private void bindIndexer(CpuWeights ws, LayerSpec layer, DsaSpec dsa, LayerMemoryPlan.IndexerPlan plan,
                             CpuScratch scratch) {
        idxCache = memory.indexer(layer.il());
        if (idxCache == null || plan == null) {
            throw new IllegalStateException("full QSA layer " + MsaBlocks.describe(layer) + " has no indexer cache");
        }
        MsaBlocks.checkQsaIndexer(layer, dsa, plan, kvLayout);
        idxLayout = idxCache.layout();
        MsaBlocks.checkNoRing(layer, idxLayout, "indexer cache");
        width = plan.width();
        gateOff = MsaKernels.kpoolGateOffset(dsa.kpoolRow(), idx);
        pooledOff = MsaKernels.kpoolPooledOffset(dsa.kpoolRow(), idx);
        MsaKernels.checkKpool(kpool, idx, width, gateOff, pooledOff, idxLayout.rows(), mode);
        MsaKernels.checkKpoolCache(idxLayout.dtypeK(), idx, width, gateOff, pooledOff, mode);
        wScale = dsa.weightScale();
        int nEmbd = MsaBlocks.mixerWidth(layer);
        int ub = ubatch;
        int maxNew = MsaBlocks.maxNewPools(ub, kpool);
        iq = scratch.temp(GpuQsaMixer.TEMP_IQ, Math.multiplyExact(ub, nIdxH * idx));
        ik = scratch.temp(GpuQsaMixer.TEMP_IK, Math.multiplyExact(ub, idx));
        rows = scratch.temp(GpuQsaMixer.TEMP_PACKED, Math.multiplyExact(ub, width));
        pooled = scratch.temp(GpuQsaMixer.TEMP_POOLED, Math.multiplyExact(maxNew, idx));
        scores = scratch.temp(GpuQsaMixer.TEMP_SCORES,
                Math.multiplyExact(ub, Math.max(1, MsaKernels.completePools(posLimit, kpool))));
        idxPos = scratch.intTemp("qsa.idx_pos", idxLayout.rows());
        if (mode == MsaKernels.KPOOL_MODE_GATED) {
            wIq = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_Q_B, qLora, (long) nIdxH * idx);
            wIk = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_ATTN_K, nEmbd, idx);
            ikNormW = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true);
            ikNormB = MsaBlocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.INDEXER_K_NORM), idx, true);
            wIg = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_KPOOL_GATE, nEmbd, idx);
            wProj = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_PROJ, nEmbd, nIdxH);
            ape = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_KPOOL_APE), idx * kpool, true);
            ig = scratch.temp(GpuQsaMixer.TEMP_IG, Math.multiplyExact(ub, idx));
            w = scratch.temp(GpuQsaMixer.TEMP_W, Math.multiplyExact(ub, nIdxH));
        } else {
            wIq = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_Q_PROJ, nEmbd, (long) nIdxH * idx);
            wIk = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_K_PROJ, nEmbd, idx);
            iqNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM), idx, true);
            ikNormW = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true);
            idxRope = MsaBlocks.rope(layer, idx, dsa.indexerRopeDim(), "QSA indexer");
            idxTable = RopeFreqTable.of(idxRope, MsaBlocks.cpuFreqFactors(ws, idxRope));
            poolPos = scratch.intTemp(GpuQsaMixer.TEMP_POOL_POS, Math.multiplyExact(MropeTextPositions.SECTIONS,
                    maxNew));
        }
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(pos0, n, in, out);
    }

    private void run(int pos0, int n, float[] in, float[] out) {
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
        MsaBlocks.fillQueryPositions(pos0, n, qPos);
        if (mla) {
            runMla(pos0, n, in, out);
        } else {
            runGqa(pos0, n, in, out);
        }
    }

    private void runMla(int pos0, int n, float[] in, float[] out) {
        mv.matmul(wQa, in, 0, qr, 0, n);
        qaDeltas.applyRows(n, in, 0, qr, 0);
        CpuOps.norm(NormKind.RMS, rmsEps, qr, 0, qr, 0, n, qLora, qaNorm, null);
        mv.matmul(wQb, qr, 0, qn, 0, n);
        qbDeltas.applyRows(n, qr, 0, qn, 0);
        mv.matmul(wKva, in, 0, kvl, 0, n);
        kvaDeltas.applyRows(n, in, 0, kvl, 0);
        CpuOps.norm(NormKind.RMS, rmsEps, kvl, 0, kvl, 0, n, kvLora, kvaNorm, null);
        for (int s = 0; s < n * nHead; s++) {
            mv.matvecExpert(wKb, s % nHead, nHead, qn, s * qkNope, qabs, s * kvLora);
        }
        if (full) {
            mv.matmul(wIq, qr, 0, iq, 0, n);
            iqDeltas.applyRows(n, qr, 0, iq, 0);
            mv.matmul(wIk, in, 0, ik, 0, n);
            ikDeltas.applyRows(n, in, 0, ik, 0);
            CpuOps.norm(NormKind.LAYER, normEps, ik, 0, ik, 0, n, idx, ikNormW, ikNormB);
            mv.matmul(wIg, in, 0, ig, 0, n);
            igDeltas.applyRows(n, in, 0, ig, 0);
            Arrays.fill(rows, 0, n * width, 0f);
            for (int t = 0; t < n; t++) {
                System.arraycopy(ik, t * idx, rows, t * width, idx);
                System.arraycopy(ig, t * idx, rows, t * width + gateOff, idx);
            }
            storeIndexerRows(pos0, n);
            pool(pos0, n);
            mv.matmul(wProj, in, 0, w, 0, n);
            projDeltas.applyRows(n, in, 0, w, 0);
            score(pos0, n, w);
        }
        MsaBlocks.cpuRotate(rotK, qabs, n, nHead * kvLora);
        MsaBlocks.cpuRotate(rotK, kvl, n, kvLora);
        for (int t = 0; t < n; t++) {
            kv.writeK(kvLayout.rowOf(pos0 + t), kvl, t * kvLora);
        }
        MsaBlocks.fillKvPos(memory, kv.group(), kvLayout, pos0 + n, kvPos);
        CpuMsa.gatherAttn(n, nHead, 1, kvLora, kpool, selWidth, kqScale, 1, tail, kvLayout.rows(), qabs, 0,
                nHead * kvLora, kv, kv, qPos, kvPos, sel, att, 0, nHead * kvLora);
        MsaBlocks.cpuRotate(rotK, att, n, nHead * kvLora);
        for (int s = 0; s < n * nHead; s++) {
            mv.matvecExpert(wVb, s % nHead, nHead, att, s * kvLora, vout, s * headV);
        }
        mv.matmul(wO, vout, 0, out, 0, n);
        oDeltas.applyRows(n, vout, 0, out, 0);
    }

    private void runGqa(int pos0, int n, float[] in, float[] out) {
        if (full) {
            mv.matmul(wIk, in, 0, ik, 0, n);
            ikDeltas.applyRows(n, in, 0, ik, 0);
            Arrays.fill(rows, 0, n * width, 0f);
            for (int t = 0; t < n; t++) {
                System.arraycopy(ik, t * idx, rows, t * width, idx);
            }
            storeIndexerRows(pos0, n);
            pool(pos0, n);
            mv.matmul(wIq, in, 0, iq, 0, n);
            iqDeltas.applyRows(n, in, 0, iq, 0);
            CpuOps.norm(NormKind.RMS, rmsEps, iq, 0, iq, 0, n * nIdxH, idx, iqNorm, null);
            RopePositions.fill(idxRope.type(), idxRope.textPositions(), pos0, n, ropePos);
            CpuRope.apply(idxTable, iq, 0, n, nIdxH, idx, nIdxH * idx, idxRope.nOffs(), ropePos, 0, n);
            score(pos0, n, null);
        }
        mv.matmul(wQ, in, 0, qg, 0, n);
        qDeltas.applyRows(n, in, 0, qg, 0);
        for (int s = 0; s < n * nHead; s++) {
            System.arraycopy(qg, s * 2 * hd, q, s * hd, hd);
        }
        CpuOps.norm(NormKind.RMS, rmsEps, q, 0, q, 0, n * nHead, hd, qNorm, null);
        mv.matmul(wK, in, 0, k, 0, n);
        kDeltas.applyRows(n, in, 0, k, 0);
        mv.matmul(wV, in, 0, v, 0, n);
        vDeltas.applyRows(n, in, 0, v, 0);
        CpuOps.norm(NormKind.RMS, rmsEps, k, 0, k, 0, n * nKv, hd, kNorm, null);
        RopePositions.fill(rope.type(), rope.textPositions(), pos0, n, ropePos);
        CpuRope.apply(ropeTable, q, 0, n, nHead, hd, qDim, rope.nOffs(), ropePos, 0, n);
        CpuRope.apply(ropeTable, k, 0, n, nKv, hd, kDim, rope.nOffs(), ropePos, 0, n);
        MsaBlocks.cpuRotate(rotK, q, n, qDim);
        MsaBlocks.cpuRotate(rotK, k, n, kDim);
        MsaBlocks.cpuRotate(rotV, v, n, kDim);
        for (int t = 0; t < n; t++) {
            int row = kvLayout.rowOf(pos0 + t);
            kv.writeK(row, k, t * kDim);
            kv.writeV(row, v, t * kDim);
        }
        MsaBlocks.fillKvPos(memory, kv.group(), kvLayout, pos0 + n, kvPos);
        CpuMsa.gatherAttn(n, nHead, nKv, hd, kpool, selWidth, kqScale, 1, tail, kvLayout.rows(), q, 0, qDim, kv, kv,
                qPos, kvPos, sel, att, 0, qDim);
        MsaBlocks.cpuRotate(rotV, att, n, qDim);
        CpuOps.gluRows(GluOp.SIGMUL, ActParams.NONE, n * nHead, hd, qg, hd, 2 * hd, att, 0, hd, att, 0, hd);
        mv.matmul(wO, att, 0, out, 0, n);
        oDeltas.applyRows(n, att, 0, out, 0);
    }

    private void storeIndexerRows(int pos0, int n) {
        for (int t = 0; t < n; t++) {
            idxCache.writeK(idxLayout.rowOf(pos0 + t), rows, t * width);
        }
    }

    /** Pools, normalizes and ropes (mean mode) and stores the pools the step completes. */
    private void pool(int pos0, int n) {
        int nNew = MsaKernels.newPoolCount(pos0, n, kpool);
        if (nNew == 0) {
            return;
        }
        int first = MsaKernels.firstNewPool(pos0, kpool);
        CpuQsa.kpoolPool(nNew, first, mode, kpool, idx, gateOff, idxLayout.rows(), ape, idxCache, pooled);
        if (mode == MsaKernels.KPOOL_MODE_MEAN) {
            CpuOps.norm(NormKind.RMS, rmsEps, pooled, 0, pooled, 0, nNew, idx, ikNormW, null);
            fillPoolPositions(first, nNew);
            CpuRope.apply(idxTable, pooled, 0, nNew, 1, idx, idx, idxRope.nOffs(), poolPos, 0, nNew);
        }
        CpuQsa.kpoolStore(nNew, first, kpool, idx, pooledOff, idxLayout.rows(), pooled, idxCache);
    }

    /** {@link GpuQsaMixer}'s pool positions: {@code [p, y(p), x(p), p]} of each new pool's first member. */
    private void fillPoolPositions(int first, int nNew) {
        int sections = MropeTextPositions.SECTIONS;
        for (int j = 0; j < nNew; j++) {
            int p = (first + j) * kpool;
            for (int s = 0; s < sections; s++) {
                poolPos[s * nNew + j] = s == 0 || s == sections - 1 ? p : idxRope.textPositions().position(s, p);
            }
        }
    }

    /** Scores the complete pools and selects {@code K} per token; -1 everywhere when no pool is complete yet. */
    private void score(int pos0, int n, float[] weights) {
        if (selWidth == 0) {
            return;
        }
        int nPool = MsaKernels.completePools(pos0 + n, kpool);
        if (nPool == 0) {
            Arrays.fill(sel, 0, n * selWidth, -1);
            return;
        }
        MsaBlocks.fillKvPos(memory, idxCache.group(), idxLayout, pos0 + n, idxPos);
        CpuQsa.kpoolScore(n, nPool, nIdxH, idx, kpool, pooledOff, idxLayout.rows(), wScale, iq, 0, weights, 0,
                idxCache, qPos, idxPos, scores);
        CpuTopK.topKRows(scores, 0, n, nPool, selWidth, sel, 0);
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        if (!bound) {
            return "CpuQsaMixer{unbound}";
        }
        return "CpuQsaMixer{" + (mla ? "MLA latent " + kvLora : "GQA " + nHead + "/" + nKv + "x" + hd) + ", kpool "
                + kpool + ", select " + selWidth + (full ? ", full indexer" : ", shared indexer") + '}';
    }
}

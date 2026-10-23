package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
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
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuMsaMixer}: the same MiniMax-M3 sparse layer (models/minimax-m3.cpp:306-535) with the same order
 * of operations. Call sequence of a step of {@code T} tokens at {@code pos0}:
 * <pre>
 * matmul attn_qkv (+ addRowBias) and row split, or matmul attn_q/k/v (+ addRowBias)
 * norm RMS attn_q_norm over T*H heads, attn_k_norm over T*G heads; CpuRope.apply q, k
 * matmul indexer.q_proj, indexer.k_proj; norm RMS indexer.q_norm (T*G rows), indexer.k_norm (T rows)
 * CpuRope.apply iq, ik; writeK(ik) into the MSA index-key cache at rowOf(pos)
 * cpuRotate q, k (n_rot_k), v (n_rot_v); writeK(k), writeV(v) at rowOf(pos)
 * CpuMsa.blockScore -&gt; CpuTopK.topKRows(top_k_blocks) -&gt; CpuMsa.gatherAttn(sel_heads = G)
 * cpuRotate att (n_rot_v); matmul attn_output
 * </pre>
 * The executor marks the step's rows ({@link CpuMemory#beginStep}) before the layers run. Working arrays come from
 * the scratch; nothing is allocated per step. Not thread-safe.
 */
public final class CpuMsaMixer implements CpuMixer {

    private CpuMatVec mv;
    private CpuMemory memory;
    private int ubatch;
    private int nHead;
    private int nKv;
    private int hd;
    private int qDim;
    private int kDim;
    private int idx;
    private int blk;
    private int localBlocks;
    private int topK;
    private float kqScale;
    private int posLimit;
    private int rotK;
    private int rotV;
    private float rmsEps;
    private RopeSpec rope;
    private RopeFreqTable ropeTable;
    private KvAccessor kv;
    private KvAccessor keys;
    private KvLayout kvLayout;
    private KvLayout keyLayout;

    private CpuTensorRef wQkv;
    private CpuTensorRef wQ;
    private CpuTensorRef wK;
    private CpuTensorRef wV;
    private CpuTensorRef wO;
    private CpuTensorRef wIq;
    private CpuTensorRef wIk;
    /** LoRA delta sites of the bound projections (null for absent ones); see {@link #bindDeltas}. */
    private CpuProjectionDeltas.Site qkvDeltas;
    private CpuProjectionDeltas.Site qDeltas;
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site oDeltas;
    private CpuProjectionDeltas.Site iqDeltas;
    private CpuProjectionDeltas.Site ikDeltas;
    private float[] bQkv;
    private float[] bQ;
    private float[] bK;
    private float[] bV;
    private float[] qNorm;
    private float[] kNorm;
    private float[] iqNorm;
    private float[] ikNorm;

    private float[] qkv;
    private float[] q;
    private float[] k;
    private float[] v;
    private float[] iq;
    private float[] ik;
    private float[] att;
    private float[] scores;
    private int[] blocks;
    private int[] qPos;
    private int[] ropePos;
    private int[] kvPos;
    private int[] keyPos;

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
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (bound || closed) {
            throw new IllegalStateException("MSA mixer is already bound or closed");
        }
        MsaBlocks.checkVariant(layer, MsaBlocks.Variant.MSA);
        AttentionSpec a = layer.attention();
        MsaSpec msa = layer.msa();
        checkShapes(layer, a, msa);
        if (!ctx.hasMemory()) {
            throw new IllegalStateException("MSA layer " + MsaBlocks.describe(layer) + " needs a KV cache");
        }
        mv = ctx.matVec();
        memory = ctx.memory();
        ubatch = ctx.scratch().ubatch();
        rmsEps = ctx.spec().hparams().normRmsEps();
        rope = MsaBlocks.rope(layer, hd, 0, "MSA attention");
        MsaBlocks.rope(layer, idx, msa.idxRopeDim(), "MSA indexer");
        bindCaches(layer);
        CpuWeights ws = ctx.weights();
        ropeTable = RopeFreqTable.of(rope, MsaBlocks.cpuFreqFactors(ws, rope));
        bindWeights(ws, layer);
        bindDeltas(ctx.projectionDeltas(), layer);
        bindScratch(ctx.scratch());
        bound = true;
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
        if (a.headDimV() != hd || msa.idxHeads() != nKv) {
            throw new IllegalArgumentException(where + "MSA needs equal K and V head sizes and one index head per KV"
                    + " head (minimax-m3.cpp:212, :382), got " + hd + " / " + a.headDimV() + " and " + msa.idxHeads()
                    + " index heads for " + nKv);
        }
        qDim = a.qDim();
        kDim = a.kDim();
        idx = msa.idxHeadDim();
        blk = msa.blockSize();
        localBlocks = msa.localBlocks();
        topK = msa.topKBlocks();
        kqScale = msa.kqScale();
    }

    private void bindCaches(LayerSpec layer) {
        String where = "layer " + MsaBlocks.describe(layer) + ": ";
        kv = memory.kv(layer.il());
        keys = memory.msa(layer.il());
        if (kv == null || keys == null) {
            throw new IllegalStateException("MSA layer " + MsaBlocks.describe(layer) + " needs a KV cache and an"
                    + " index-key cache");
        }
        kvLayout = kv.layout();
        keyLayout = keys.layout();
        MsaBlocks.checkNoRing(layer, kvLayout, "KV cache");
        MsaBlocks.checkNoRing(layer, keyLayout, "index-key cache");
        if (kvLayout.vAliasesK() || kv.nHeadKv() != nKv || kv.headDimK() != hd || kv.headDimV() != hd) {
            throw new IllegalStateException(where + "the KV cache holds " + kv.nHeadKv() + " heads of "
                    + kv.headDimK() + " / " + kv.headDimV() + ", MSA needs " + nKv + " of " + hd);
        }
        if (keys.nHeadKv() != 1 || keys.headDimK() != idx) {
            throw new IllegalStateException(where + "index-key rows of " + keys.headDimK()
                    + " values, the indexer writes " + idx);
        }
        posLimit = Math.min(kvLayout.rows(), keyLayout.rows());
        RotationPlan rot = memory.plan().layer(layer.il()).rotation();
        rotK = rot.nRotK();
        rotV = rot.nRotV();
        MsaBlocks.checkRotation(rotK, hd);
        MsaBlocks.checkRotation(rotV, hd);
    }

    private void bindWeights(CpuWeights ws, LayerSpec layer) {
        int nEmbd = MsaBlocks.mixerWidth(layer);
        int qkvW = qDim + 2 * kDim;
        wQkv = MsaBlocks.cpuMatrixOrNull(ws, layer, TensorNames.ATTN_QKV, nEmbd, qkvW);
        if (wQkv == null) {
            wQ = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_Q, nEmbd, qDim);
            wK = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_K, nEmbd, kDim);
            wV = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_V, nEmbd, kDim);
        }
        wO = MsaBlocks.cpuMatrix(ws, layer, TensorNames.ATTN_OUT, qDim, nEmbd);
        wIq = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_Q_PROJ, nEmbd, (long) nKv * idx);
        wIk = MsaBlocks.cpuMatrix(ws, layer, TensorNames.INDEXER_K_PROJ, nEmbd, idx);
        qNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_Q_NORM), hd, true);
        kNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.ATTN_K_NORM), hd, true);
        iqNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_Q_NORM), idx, true);
        ikNorm = MsaBlocks.cpuVector(ws, layer, TensorNames.weight(TensorNames.INDEXER_K_NORM), idx, true);
        // build_qkv biases (llama-graph.cpp:1823-1834, 1856-1889)
        if (wQkv != null) {
            bQkv = MsaBlocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.ATTN_QKV), qkvW, false);
        }
        if (bQkv == null) {
            bQ = MsaBlocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.ATTN_Q), qDim, false);
            bK = MsaBlocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.ATTN_K), kDim, false);
            bV = MsaBlocks.cpuVector(ws, layer, TensorNames.bias(TensorNames.ATTN_V), kDim, false);
            if (wQkv != null && (bQ == null || bK == null || bV == null)) {
                bQ = null;
                bK = null;
                bV = null;
            }
        }
    }

    /**
     * Serves the LoRA delta site of every bound dense projection; each delta is added right after its product and
     * before any bias ({@code build_lora_mm}, llama-graph.cpp:1629-1667).
     */
    private void bindDeltas(CpuProjectionDeltas deltas, LayerSpec layer) {
        qkvDeltas = serveDelta(deltas, layer, TensorNames.ATTN_QKV, wQkv);
        qDeltas = serveDelta(deltas, layer, TensorNames.ATTN_Q, wQ);
        kDeltas = serveDelta(deltas, layer, TensorNames.ATTN_K, wK);
        vDeltas = serveDelta(deltas, layer, TensorNames.ATTN_V, wV);
        oDeltas = serveDelta(deltas, layer, TensorNames.ATTN_OUT, wO);
        iqDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_Q_PROJ, wIq);
        ikDeltas = serveDelta(deltas, layer, TensorNames.INDEXER_K_PROJ, wIk);
    }

    private CpuProjectionDeltas.Site serveDelta(CpuProjectionDeltas deltas, LayerSpec layer, String base, CpuTensorRef w) {
        return w == null ? null : deltas.serve(layer, TensorNames.weight(base), w, mv);
    }

    private void bindScratch(CpuScratch scratch) {
        int ub = ubatch;
        if (wQkv != null) {
            qkv = scratch.temp(GpuMsaMixer.TEMP_QKV, Math.multiplyExact(ub, qDim + 2 * kDim));
        }
        q = scratch.temp(GpuMsaMixer.TEMP_Q, Math.multiplyExact(ub, qDim));
        k = scratch.temp(GpuMsaMixer.TEMP_K, Math.multiplyExact(ub, kDim));
        v = scratch.temp(GpuMsaMixer.TEMP_V, Math.multiplyExact(ub, kDim));
        iq = scratch.temp(GpuMsaMixer.TEMP_IQ, Math.multiplyExact(ub, nKv * idx));
        ik = scratch.temp(GpuMsaMixer.TEMP_IK, Math.multiplyExact(ub, idx));
        att = scratch.temp(GpuMsaMixer.TEMP_ATT, Math.multiplyExact(ub, qDim));
        scores = scratch.temp(GpuMsaMixer.TEMP_SCORES,
                Math.multiplyExact(ub * nKv, MsaKernels.blockCount(posLimit, blk)));
        blocks = scratch.intTemp(GpuMsaMixer.TEMP_BLOCKS, Math.multiplyExact(ub * nKv, topK));
        qPos = scratch.intTemp("msa.q_pos", ub);
        ropePos = scratch.intTemp("msa.rope_pos", Math.multiplyExact(4, ub));
        kvPos = scratch.intTemp("msa.kv_pos", kvLayout.rows());
        keyPos = scratch.intTemp("msa.key_pos", keyLayout.rows());
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
        MsaBlocks.fillQueryPositions(pos0, n, qPos);
        RopePositions.fill(rope.type(), rope.textPositions(), pos0, n, ropePos);

        project(n, in);
        CpuOps.norm(NormKind.RMS, rmsEps, q, 0, q, 0, n * nHead, hd, qNorm, null);
        CpuOps.norm(NormKind.RMS, rmsEps, k, 0, k, 0, n * nKv, hd, kNorm, null);
        CpuRope.apply(ropeTable, q, 0, n, nHead, hd, qDim, rope.nOffs(), ropePos, 0, n);
        CpuRope.apply(ropeTable, k, 0, n, nKv, hd, kDim, rope.nOffs(), ropePos, 0, n);

        mv.matmul(wIq, in, 0, iq, 0, n);
        iqDeltas.applyRows(n, in, 0, iq, 0);
        mv.matmul(wIk, in, 0, ik, 0, n);
        ikDeltas.applyRows(n, in, 0, ik, 0);
        CpuOps.norm(NormKind.RMS, rmsEps, iq, 0, iq, 0, n * nKv, idx, iqNorm, null);
        CpuOps.norm(NormKind.RMS, rmsEps, ik, 0, ik, 0, n, idx, ikNorm, null);
        CpuRope.apply(ropeTable, iq, 0, n, nKv, idx, nKv * idx, rope.nOffs(), ropePos, 0, n);
        CpuRope.apply(ropeTable, ik, 0, n, 1, idx, idx, rope.nOffs(), ropePos, 0, n);
        for (int t = 0; t < n; t++) {
            keys.writeK(keyLayout.rowOf(pos0 + t), ik, t * idx);
        }

        MsaBlocks.cpuRotate(rotK, q, n, qDim);
        MsaBlocks.cpuRotate(rotK, k, n, kDim);
        MsaBlocks.cpuRotate(rotV, v, n, kDim);
        for (int t = 0; t < n; t++) {
            int row = kvLayout.rowOf(pos0 + t);
            kv.writeK(row, k, t * kDim);
            kv.writeV(row, v, t * kDim);
        }

        MsaBlocks.fillKvPos(memory, keys.group(), keyLayout, nPos, keyPos);
        MsaBlocks.fillKvPos(memory, kv.group(), kvLayout, nPos, kvPos);
        int nblk = MsaKernels.blockCount(nPos, blk);
        CpuMsa.blockScore(n, nPos, nKv, idx, blk, localBlocks, keyLayout.rows(), iq, 0, keys, qPos, keyPos, scores);
        CpuTopK.topKRows(scores, 0, n * nKv, nblk, topK, blocks, 0);
        CpuMsa.gatherAttn(n, nHead, nKv, hd, blk, topK, kqScale, nKv, false, kvLayout.rows(), q, 0, qDim, kv, kv,
                qPos, kvPos, blocks, att, 0, qDim);

        MsaBlocks.cpuRotate(rotV, att, n, qDim);
        mv.matmul(wO, att, 0, out, 0, n);
        oDeltas.applyRows(n, att, 0, out, 0);
    }

    /** Q, K and V of {@code build_qkv} into the compact rows, biases included. */
    private void project(int n, float[] in) {
        if (wQkv != null) {
            int qkvW = qDim + 2 * kDim;
            mv.matmul(wQkv, in, 0, qkv, 0, n);
            qkvDeltas.applyRows(n, in, 0, qkv, 0);
            if (bQkv != null) {
                CpuOps.addRowBias(qkv, 0, n, qkvW, bQkv, 0);
            }
            for (int t = 0; t < n; t++) {
                System.arraycopy(qkv, t * qkvW, q, t * qDim, qDim);
                System.arraycopy(qkv, t * qkvW + qDim, k, t * kDim, kDim);
                System.arraycopy(qkv, t * qkvW + qDim + kDim, v, t * kDim, kDim);
            }
        } else {
            mv.matmul(wQ, in, 0, q, 0, n);
            qDeltas.applyRows(n, in, 0, q, 0);
            mv.matmul(wK, in, 0, k, 0, n);
            kDeltas.applyRows(n, in, 0, k, 0);
            mv.matmul(wV, in, 0, v, 0, n);
            vDeltas.applyRows(n, in, 0, v, 0);
        }
        addBias(q, n, qDim, bQ);
        addBias(k, n, kDim, bK);
        addBias(v, n, kDim, bV);
    }

    private static void addBias(float[] x, int n, int width, float[] bias) {
        if (bias != null) {
            CpuOps.addRowBias(x, 0, n, width, bias, 0);
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuMsaMixer{" + (bound ? nHead + "q/" + nKv + "kv heads of " + hd + ", index " + idx + ", block " + blk
                + (rotK > 0 || rotV > 0 ? ", rotated" : "") : "unbound") + '}';
    }
}

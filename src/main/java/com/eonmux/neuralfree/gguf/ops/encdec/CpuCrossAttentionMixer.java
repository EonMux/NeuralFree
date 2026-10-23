package com.eonmux.neuralfree.gguf.ops.encdec;

import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.function.IntUnaryOperator;

/**
 * T5 decoder cross-attention of a {@code CROSS_ATTENTION_T5} layer on the CPU; the twin of
 * {@link GpuCrossAttentionMixer} with the same step order and semantics (block contract in {@link EncDecBlocks}).
 *
 * <p>Per step: {@code wq_cross} into the {@code Q} slot ({@link CpuMatVec}); the K rotation of Q when the cross cache
 * is quantized and rotated; {@link CpuAttentionCore} over the cross rows {@code [0, nEnc)} (bidirectional, no bias)
 * into {@code ATTN_OUT}; the inverse V rotation; {@code wo_cross} into {@code out}. {@link #encode} fills the cross
 * cache from the encoder output once per prompt. Working arrays are scratch slots and a temporary reserved at
 * {@link #bind}.</p>
 */
public final class CpuCrossAttentionMixer implements CpuMixer {

    private static final IntUnaryOperator IDENTITY_ROWS = IntUnaryOperator.identity();

    private LayerSpec layer;
    private CpuMatVec matVec;
    private int ubatch;
    private int width;
    private AttentionSpec cross;

    private CpuTensorRef wqCross;
    private CpuTensorRef wkCross;
    private CpuTensorRef wvCross;
    private CpuTensorRef woCross;
    /** LoRA deltas of the cross-attention projections ({@code build_lora_mm}, t5.cpp:164-182). */
    private CpuProjectionDeltas.Site qDeltas;
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site oDeltas;

    private float[] q;
    private float[] k;
    private float[] v;
    private float[] attnOut;
    private int[] qPos;

    private KvAccessor crossKv;
    private HadamardSpec rotK;
    private HadamardSpec rotV;
    private CpuAttentionCore.Params params;

    private int nEnc;

    /** Unbound mixer; {@link #bind} attaches it to its layer. */
    public CpuCrossAttentionMixer() {
    }

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (this.layer != null) {
            throw new IllegalStateException("T5 cross-attention mixer is already bound to layer "
                    + EncDecBlocks.describe(this.layer));
        }
        EncDecBlocks.checkLayer(layer);
        this.layer = layer;
        this.matVec = ctx.matVec();
        this.cross = layer.crossAttention();
        this.width = layer.residual().streamWidth();
        bindWeights(ctx.weights());
        bindDeltas(ctx.projectionDeltas());
        bindScratch(ctx.scratch());
        bindMemory(ctx.memory());
        params = CpuAttentionCore.Params.of(cross, null, null);
    }

    private void bindWeights(CpuWeights ws) {
        wqCross = matrix(ws, EncDecBlocks.W_CROSS_Q, width, cross.qDim());
        wkCross = matrix(ws, EncDecBlocks.W_CROSS_K, width, cross.kDim());
        wvCross = matrix(ws, EncDecBlocks.W_CROSS_V, width, cross.vDim());
        woCross = matrix(ws, EncDecBlocks.W_CROSS_O, cross.outDim(), width);
    }

    private void bindDeltas(CpuProjectionDeltas deltas) {
        qDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_Q, wqCross, matVec);
        kDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_K, wkCross, matVec);
        vDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_V, wvCross, matVec);
        oDeltas = deltas.serve(layer, EncDecBlocks.W_CROSS_O, woCross, matVec);
    }

    private CpuTensorRef matrix(CpuWeights ws, String suffix, long cols, long rows) {
        CpuTensorRef w = ws.layer(layer, suffix);
        EncDecBlocks.checkMatrix(layer, layer.tensorName(suffix), w.cols(), w.rows(), cols, rows);
        return w;
    }

    private void bindScratch(CpuScratch scratch) {
        ubatch = scratch.ubatch();
        q = slot(scratch, CpuScratch.Q, cross.qDim());
        k = slot(scratch, CpuScratch.K, cross.kDim());
        v = slot(scratch, CpuScratch.V, cross.vDim());
        attnOut = slot(scratch, CpuScratch.ATTN_OUT, cross.outDim());
        qPos = scratch.intTemp(EncDecBlocks.CPU_TEMP_QPOS, ubatch);
    }

    private float[] slot(CpuScratch scratch, String name, int rowWidth) {
        float[] s = scratch.slot(name);
        if (s.length < (long) ubatch * rowWidth) {
            throw new IllegalStateException("scratch slot " + name + " holds " + s.length + " floats, T5 decoder layer "
                    + EncDecBlocks.describe(layer) + " needs " + ((long) ubatch * rowWidth));
        }
        return s;
    }

    private void bindMemory(CpuMemory memory) {
        int il = layer.il();
        crossKv = memory.crossKv(il);
        if (crossKv == null) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer)
                    + " needs a cross-attention KV cache, the memory has none");
        }
        if (crossKv.nHeadKv() != cross.nHeadKv() || crossKv.headDimK() != cross.headDimK()
                || crossKv.headDimV() != cross.headDimV()) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": cross cache of "
                    + crossKv.nHeadKv() + " heads of " + crossKv.headDimK() + "/" + crossKv.headDimV()
                    + " does not hold K/V of " + cross.nHeadKv() + " heads of " + cross.headDimK() + "/"
                    + cross.headDimV());
        }
        RotationPlan rotation = memory.plan().layer(il).rotation();
        rotK = rotation.rotateK() ? HadamardSpec.rotation(rotation.nRotK()) : null;
        rotV = rotation.rotateV() ? HadamardSpec.rotation(rotation.nRotV()) : null;
    }

    // ================================================================== encoder output

    /**
     * Computes this layer's cross K/V from the encoder output and stores them in rows {@code [0, nEnc)} of the cross
     * cache (llama.cpp t5.cpp:170-174). Called once per prompt, after the encoder ran and before the first decoder
     * step; a later call replaces the previous encoder output.
     *
     * @param enc  encoder output rows {@code enc[encOff + r*width ..]} (after {@code enc.output_norm})
     * @param nEnc encoder tokens, at most {@code ubatch} (llama-context.cpp:1628) and the cross cache rows
     * @throws IllegalStateException    before {@link #bind}
     * @throws IllegalArgumentException when {@code nEnc} is out of range, {@code enc} is too short or is the K or V
     *                                  scratch slot
     */
    public void encode(float[] enc, int encOff, int nEnc) {
        ensureBound();
        int rows = crossKv.layout().rows();
        if (nEnc < 1 || nEnc > ubatch || nEnc > rows) {
            throw new IllegalArgumentException("T5 decoder layer " + EncDecBlocks.describe(layer) + ": encoder output of "
                    + nEnc + " tokens must be in [1, " + Math.min(ubatch, rows) + "] (ubatch " + ubatch
                    + ", cross cache rows " + rows + ")");
        }
        if (enc == null || encOff < 0 || encOff + (long) nEnc * width > enc.length) {
            throw new IllegalArgumentException("encoder output needs " + nEnc + " rows of " + width + " floats from "
                    + encOff + ", got " + (enc == null ? "null" : enc.length + " floats"));
        }
        if (enc == k || enc == v) {
            throw new IllegalArgumentException("encoder output must not live in the K or V scratch slot");
        }
        int kDim = cross.kDim();
        int vDim = cross.vDim();
        matVec.matmul(wkCross, enc, encOff, k, 0, nEnc);
        kDeltas.applyRows(nEnc, enc, encOff, k, 0);
        matVec.matmul(wvCross, enc, encOff, v, 0, nEnc);
        vDeltas.applyRows(nEnc, enc, encOff, v, 0);
        rotate(rotK, k, nEnc, kDim);
        rotate(rotV, v, nEnc, vDim);
        for (int r = 0; r < nEnc; r++) {
            crossKv.writeK(r, k, r * kDim);
            crossKv.writeV(r, v, r * vDim);
        }
        this.nEnc = nEnc;
    }

    /** Encoder tokens of the current cross cache; 0 before the first {@link #encode}. */
    public int encodedTokens() {
        return nEnc;
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        prefill(pos, 1, in, out);
    }

    /** {@code out = wo_cross @ attn(wq_cross @ in, cross cache)} (t5.cpp:164-182). */
    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        ensureBound();
        if (nEnc == 0) {
            throw new IllegalStateException("T5 decoder layer " + EncDecBlocks.describe(layer)
                    + " runs before encode() stored the encoder output");
        }
        if (pos0 < 0 || n < 1 || n > ubatch) {
            throw new IllegalArgumentException("T5 decoder step needs pos0 >= 0 and nTokens in [1, " + ubatch + "], got "
                    + pos0 + "/" + n);
        }
        long rows = (long) n * width;
        if (in.length < rows || out.length < rows) {
            throw new IllegalArgumentException("T5 decoder step of " + n + " tokens needs " + rows + " input and output"
                    + " floats, got " + in.length + " and " + out.length);
        }
        if (in == out || in == q || out == q || in == attnOut || out == attnOut) {
            throw new IllegalArgumentException("T5 cross-attention input and output must be distinct from each other"
                    + " and from the Q and ATTN_OUT slots");
        }
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        int qDim = cross.qDim();
        int outDim = cross.outDim();
        matVec.matmul(wqCross, in, 0, q, 0, n);
        qDeltas.applyRows(n, in, 0, q, 0);
        rotate(rotK, q, n, qDim);
        CpuAttentionCore.attendRows(params, n, q, 0, qDim, qPos, 0, crossKv, IDENTITY_ROWS, 0, nEnc,
                attnOut, 0, outDim);
        rotate(rotV, attnOut, n, outDim);
        matVec.matmul(woCross, attnOut, 0, out, 0, n);
        oDeltas.applyRows(n, attnOut, 0, out, 0);
    }

    private static void rotate(HadamardSpec h, float[] a, int n, int rowWidth) {
        if (h != null) {
            CpuHadamard.applyInPlace(h, a, 0, n, rowWidth);
        }
    }

    /**
     * Context shift: the cross cache holds the encoder output, which has no decoder positions, so nothing moves (the
     * decoder self-attention cache is shifted by its own mixer and the memory).
     *
     * @throws IllegalArgumentException for a negative range
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureBound();
        if (nKeep < 0 || nDiscard < 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard >= 0, got " + nKeep + "/"
                    + nDiscard);
        }
    }

    /** The cross cache holds no decoder positions, so the shift never refuses here. */
    @Override
    public String shiftRefusal() {
        ensureBound();
        return null;
    }

    private void ensureBound() {
        if (layer == null) {
            throw new IllegalStateException("T5 cross-attention mixer is not bound");
        }
    }

    @Override
    public String toString() {
        return "CpuCrossAttentionMixer{" + (layer == null ? "unbound" : EncDecBlocks.describe(layer)) + ", nEnc="
                + nEnc + '}';
    }
}

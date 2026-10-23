package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.resolve.RopeResolver;

/**
 * The DFlash decoder stack (dflash.cpp:806-1106): {@code n_layer} pre-norm blocks of QK-normed GQA attention
 * over the draft cache and a gated FFN, with the optional DFlash2 convolutions, attention sinks, post norms and
 * layer output scale.
 *
 * <p>Injection (dflash.cpp:856-906), per layer: {@code K = wk . g} with per-head {@code rms * attn_k_norm},
 * {@code V = wv . g}, or without {@code wv} the weightless per-head RMS of the raw {@code wk . g}; K is rotated at
 * the row position. Noise block (dflash.cpp:947-1063): {@code x = embd(tokens) * embedding_scale}; per layer
 * {@code n = rms(x) * attn_norm}, optional input conv, Q/K/V as above plus {@code rms * attn_q_norm} on Q, RoPE on
 * Q and K, attention ({@code kq_scale}, sinks, window, the block's causal flag) and {@code wo}, then
 * {@code * attn_value_scale}, output conv, {@code post_attention_norm}, residual, {@code rms * ffn_norm}, FFN conv
 * pair around {@code down(act(gate) * up)}, {@code post_ffw_norm}, residual, {@code * layer_output_scale}; finally
 * {@code t_embd = rms(x) * output_norm} and the output head. The KV rows are rounded to the cache type; a quantized
 * cache rotates Q and K (after RoPE) and V before the store and the attention output back, as llama.cpp's
 * {@code k_rot} / {@code v_rot} (dflash.cpp:884-903, llama-graph.cpp:3498-3560). Not thread-safe.
 */
final class DflashBackbone implements DflashStack {

    private static final class Layer {
        float[] attnNorm;
        DflashMatrix wq;
        DflashMatrix wk;
        DflashMatrix wv;
        DflashMatrix wo;
        float[] qNorm;
        float[] kNorm;
        float[] postAttnNorm;
        float[] postFfnNorm;
        float[] outScale;
        float[] ffnNorm;
        DflashMatrix gate;
        DflashMatrix up;
        DflashMatrix down;
        float[] attnConvBase;
        DflashMatrix attnConvProj;
        float[] ffnConvBase;
        DflashMatrix ffnConvProj;
        RopeSpec rope;
        RopeFreqTable ropeTable;
        CpuAttentionCore.Params attn;
        int nHead;
        int nHeadKv;
    }

    private final DflashSpec spec;
    private final DflashKvCache cache;
    private final DflashMatrix tokEmbd;
    private final DflashMatrix output;
    private final float[] outputNorm;
    private final Layer[] layers;
    private final DflashConv conv;
    private final int nEmbd;
    private final int headDim;
    private final HadamardSpec rotK;
    private final HadamardSpec rotV;

    /**
     * @param tokEmbd token embeddings (the draft's, or the target's when the draft has none)
     * @param output  output head (the draft's, or the target's when the draft has none)
     * @throws IllegalArgumentException when a required tensor is missing or mis-shaped
     */
    DflashBackbone(DflashSpec spec, ModelHparams hp, CpuWeights draft, DflashMatrix tokEmbd, DflashMatrix output,
                   DflashKvCache cache, int nCtxSeq) {
        this.spec = spec;
        this.cache = cache;
        this.tokEmbd = tokEmbd;
        this.output = output;
        this.nEmbd = spec.nEmbd();
        this.headDim = spec.headDim();
        this.rotK = cache.rotationK();
        this.rotV = cache.rotationV();
        this.outputNorm = vector(draft.floats(TensorNames.weight(TensorNames.OUTPUT_NORM)), nEmbd, "output_norm");
        this.conv = spec.conv() ? new DflashConv(nEmbd, spec.convKernel(), spec.convGroup()) : null;
        RopeResolver ropes = new RopeResolver(hp, draft.catalog(), nCtxSeq);
        this.layers = new Layer[spec.nLayer()];
        for (int il = 0; il < layers.length; il++) {
            layers[il] = loadLayer(il, draft, ropes);
        }
    }

    private Layer loadLayer(int il, CpuWeights w, RopeResolver ropes) {
        Layer l = new Layer();
        l.nHead = spec.nHead(il);
        l.nHeadKv = spec.nHeadKv(il);
        final int qWidth = l.nHead * headDim;
        final int kvWidth = l.nHeadKv * headDim;
        l.attnNorm = layerVector(w, il, TensorNames.ATTN_NORM, nEmbd, true);
        l.wq = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_Q)).expect(qWidth, nEmbd);
        l.wk = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_K)).expect(kvWidth, nEmbd);
        l.wv = DflashMatrix.layerOrNull(w, il, TensorNames.weight(TensorNames.ATTN_V));
        if (l.wv != null) {
            l.wv.expect(kvWidth, nEmbd);
        }
        l.wo = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.ATTN_OUT)).expect(nEmbd, qWidth);
        l.qNorm = layerVector(w, il, TensorNames.ATTN_Q_NORM, headDim, true);
        l.kNorm = layerVector(w, il, TensorNames.ATTN_K_NORM, headDim, true);
        l.postAttnNorm = layerVector(w, il, TensorNames.ATTN_POST_NORM, nEmbd, false);
        l.postFfnNorm = layerVector(w, il, TensorNames.FFN_POST_NORM, nEmbd, false);
        l.outScale = layerVector(w, il, TensorNames.LAYER_OUT_SCALE, 1, false);
        l.ffnNorm = layerVector(w, il, TensorNames.FFN_NORM, nEmbd, true);
        l.gate = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_GATE));
        l.up = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_UP)).expect(l.gate.rows(), nEmbd);
        l.down = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.FFN_DOWN)).expect(nEmbd, l.gate.rows());
        if (conv != null) {
            l.attnConvBase = vector(w.layerFloats(il, TensorNames.DFLASH_ATTN_CONV_BASE), conv.baseLength(),
                    "blk." + il + "." + TensorNames.DFLASH_ATTN_CONV_BASE);
            l.attnConvProj = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.DFLASH_ATTN_CONV_PROJ))
                    .expect(conv.dynamicWidth(), nEmbd);
            l.ffnConvBase = vector(w.layerFloats(il, TensorNames.DFLASH_FFN_CONV_BASE), conv.baseLength(),
                    "blk." + il + "." + TensorNames.DFLASH_FFN_CONV_BASE);
            l.ffnConvProj = DflashMatrix.layer(w, il, TensorNames.weight(TensorNames.DFLASH_FFN_CONV_PROJ))
                    .expect(conv.dynamicWidth(), nEmbd);
        }
        l.rope = ropes.forLayer(il);
        if (l.rope != null) {
            // ggml_rope_multi takes no frequency factors; ggml_rope_ext takes layers[0].rope_freqs (dflash.cpp:832-840)
            float[] factors = RopeMath.multiSection(l.rope.type()) ? null : w.floats(l.rope.freqFactors());
            l.ropeTable = RopeFreqTable.of(l.rope, factors);
        }
        float[] sinks = layerVector(w, il, TensorNames.ATTN_SINKS, l.nHead, false);
        CpuAttentionCore.Builder b = CpuAttentionCore.Params.builder(l.nHead)
                .scale(spec.kqScale())
                .window(cache.window(il));
        if (sinks != null) {
            b.sinks(sinks);
        }
        l.attn = b.build();
        return l;
    }

    @Override
    public void inject(float[] inpG, int nRows, int pos0) {
        for (int il = 0; il < layers.length; il++) {
            final Layer l = layers[il];
            final int kvWidth = l.nHeadKv * headDim;
            float[] k = new float[nRows * kvWidth];
            l.wk.matmul(inpG, 0, k, 0, nRows);
            float[] v = values(l, inpG, k, nRows);
            NormMath.rowNorm(NormKind.RMS, spec.rmsEps(), nRows * l.nHeadKv, headDim, headDim, 1, k, 0, l.kNorm, null,
                    k, 0);
            rope(l, k, l.nHeadKv, nRows, pos0);
            rotate(rotK, k, nRows, kvWidth);
            rotate(rotV, v, nRows, kvWidth);
            for (int t = 0; t < nRows; t++) {
                cache.write(il, pos0 + t, k, t * kvWidth, v, t * kvWidth);
            }
        }
    }

    @Override
    public Block forward(int[] tokens, int pos0) {
        final int n = tokens.length;
        final float eps = spec.rmsEps();
        float[] x = new float[n * nEmbd];
        for (int t = 0; t < n; t++) {
            tokEmbd.row(tokens[t], x, t * nEmbd);
        }
        if (spec.embeddingScale() != 0.0f) {
            scale(x, spec.embeddingScale());
        }
        int[] qPos = new int[n];
        for (int t = 0; t < n; t++) {
            qPos[t] = pos0 + t;
        }
        for (int il = 0; il < layers.length; il++) {
            x = layer(il, x, n, pos0, qPos, eps);
        }
        float[] tEmbd = new float[n * nEmbd];
        CpuOps.rmsNorm(x, 0, tEmbd, 0, n, nEmbd, outputNorm, eps);
        final int vocab = output.rows();
        float[] logits = new float[n * vocab];
        output.matmul(tEmbd, 0, logits, 0, n);
        return new Block(n, logits, vocab, tEmbd, nEmbd);
    }

    private float[] layer(int il, float[] x, int n, int pos0, int[] qPos, float eps) {
        final Layer l = layers[il];
        final int qWidth = l.nHead * headDim;
        final int kvWidth = l.nHeadKv * headDim;

        float[] h = new float[n * nEmbd];
        CpuOps.rmsNorm(x, 0, h, 0, n, nEmbd, l.attnNorm, eps);
        float[] attnDyn = null;
        if (conv != null) {
            attnDyn = new float[n * conv.dynamicWidth()];
            l.attnConvProj.matmul(h, 0, attnDyn, 0, n);
            h = convolve(h, attnDyn, l.attnConvBase, 0, n);
        }

        float[] q = new float[n * qWidth];
        float[] k = new float[n * kvWidth];
        l.wq.matmul(h, 0, q, 0, n);
        l.wk.matmul(h, 0, k, 0, n);
        float[] v = values(l, h, k, n);
        NormMath.rowNorm(NormKind.RMS, eps, n * l.nHead, headDim, headDim, 1, q, 0, l.qNorm, null, q, 0);
        NormMath.rowNorm(NormKind.RMS, eps, n * l.nHeadKv, headDim, headDim, 1, k, 0, l.kNorm, null, k, 0);
        rope(l, q, l.nHead, n, pos0);
        rope(l, k, l.nHeadKv, n, pos0);
        rotate(rotK, q, n, qWidth);
        rotate(rotK, k, n, kvWidth);
        rotate(rotV, v, n, kvWidth);
        for (int t = 0; t < n; t++) {
            cache.write(il, pos0 + t, k, t * kvWidth, v, t * kvWidth);
        }

        float[] attn = new float[n * qWidth];
        CpuAttentionCore.attendRows(l.attn, n, q, 0, qWidth, qPos, 0, cache.layer(il), cache.rowPositions(il), 0,
                cache.rowEnd(il), attn, 0, qWidth);
        rotate(rotV, attn, n, qWidth);
        float[] cur = new float[n * nEmbd];
        l.wo.matmul(attn, 0, cur, 0, n);
        if (spec.attnValueScale() != 0.0f) {
            scale(cur, spec.attnValueScale());
        }
        if (attnDyn != null) {
            cur = convolve(cur, attnDyn, l.attnConvBase, 1, n);
        }
        if (l.postAttnNorm != null) {
            CpuOps.rmsNorm(cur, 0, cur, 0, n, nEmbd, l.postAttnNorm, eps);
        }
        float[] ffnInp = cur;
        add(ffnInp, x);

        float[] f = new float[n * nEmbd];
        CpuOps.rmsNorm(ffnInp, 0, f, 0, n, nEmbd, l.ffnNorm, eps);
        float[] ffnDyn = null;
        if (conv != null) {
            ffnDyn = new float[n * conv.dynamicWidth()];
            l.ffnConvProj.matmul(f, 0, ffnDyn, 0, n);
            f = convolve(f, ffnDyn, l.ffnConvBase, 0, n);
        }
        float[] out = ffn(l, f, n);
        if (ffnDyn != null) {
            out = convolve(out, ffnDyn, l.ffnConvBase, 1, n);
        }
        if (l.postFfnNorm != null) {
            CpuOps.rmsNorm(out, 0, out, 0, n, nEmbd, l.postFfnNorm, eps);
        }
        add(out, ffnInp);
        if (l.outScale != null) {
            scale(out, l.outScale[0]);
        }
        return out;
    }

    /** {@code down(act(gate . f) * (up . f))} with SiLU or GELU (llama-graph.cpp build_ffn, LLM_FFN_PAR). */
    private float[] ffn(Layer l, float[] f, int n) {
        final int nFf = l.gate.rows();
        float[] g = new float[n * nFf];
        float[] u = new float[n * nFf];
        l.gate.matmul(f, 0, g, 0, n);
        l.up.matmul(f, 0, u, 0, n);
        final boolean gelu = spec.gelu();
        for (int i = 0; i < g.length; i++) {
            g[i] = (gelu ? ActMath.geluTanh(g[i]) : ActMath.silu(g[i])) * u[i];
        }
        float[] out = new float[n * nEmbd];
        l.down.matmul(g, 0, out, 0, n);
        return out;
    }

    /** V rows: {@code wv . h}, or the weightless per-head RMS of the raw K projection when V is shared. */
    private float[] values(Layer l, float[] h, float[] kRaw, int n) {
        final int kvWidth = l.nHeadKv * headDim;
        float[] v = new float[n * kvWidth];
        if (l.wv != null) {
            l.wv.matmul(h, 0, v, 0, n);
            return v;
        }
        NormMath.rowNorm(NormKind.RMS, spec.rmsEps(), n * l.nHeadKv, headDim, headDim, 1, kRaw, 0, null, null, v, 0);
        return v;
    }

    private void rope(Layer l, float[] x, int nHeads, int n, int pos0) {
        if (l.ropeTable == null) {
            return;
        }
        int[] pos = RopePositions.text(l.rope, pos0, n);
        CpuRope.apply(l.ropeTable, x, 0, n, nHeads, headDim, nHeads * headDim, l.rope.nOffs(), pos, 0, n);
    }

    /** In-place block Hadamard of {@code n} rows of {@code width}; no-op without a rotation. */
    private static void rotate(HadamardSpec rot, float[] x, int n, int width) {
        if (rot != null) {
            CpuHadamard.applyInPlace(rot, x, 0, n, width);
        }
    }

    private float[] convolve(float[] in, float[] dyn, float[] base, int side, int n) {
        float[] out = new float[n * nEmbd];
        conv.apply(in, 0, dyn, 0, base, side, n, out, 0);
        return out;
    }

    private static void add(float[] acc, float[] x) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += x[i];
        }
    }

    private static void scale(float[] x, float s) {
        for (int i = 0; i < x.length; i++) {
            x[i] *= s;
        }
    }

    private static float[] layerVector(CpuWeights w, int il, String base, int len, boolean required) {
        float[] v = w.layerFloatsOrNull(il, TensorNames.weight(base));
        if (v == null) {
            if (required) {
                throw new IllegalArgumentException("dflash drafter: model has no tensor 'blk." + il + "."
                        + TensorNames.weight(base) + "'");
            }
            return null;
        }
        return vector(v, len, "blk." + il + "." + base);
    }

    static float[] vector(float[] v, int len, String name) {
        if (v.length != len) {
            throw new IllegalArgumentException("dflash drafter: tensor '" + name + "' holds " + v.length
                    + " values, expected " + len);
        }
        return v;
    }
}

package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * The modern-bert decision head (models/modern-bert.cpp:237-310): the last {@code n_layer_decision} blocks of the
 * model, run on the final hidden rows (after the output norm) once per question type {@code it} in 0..2:
 * <pre>
 *   x = hidden + token_types[it]
 *   per head block: a = wo attn(wqkv LN(x; attn_norm) + b) + wo_b      (no positions, bidirectional,
 *                   f = a + x                                           scale 1/sqrt(d_head))
 *                   x = down(relu(up LN(f; ffn_norm) + up_b)) + down_b + f
 *   score[it] = cls_out gelu_erf(cls LN(x; cls_norm) + cls_b) + cls_out_b
 * </pre>
 * with every LayerNorm biased at {@code eps = attention.layernorm_epsilon}; each token gets the scores of the three
 * types ({@code n_embd_out = 3}), read by the caller at the option markers. The body excludes the head blocks.
 *
 * <p>The same blocks serve lfm2 decision models ({@link Lfm2DecisionHead}), which take the head counts per layer
 * instead of the model-wide ones (lfm2.cpp:424-482). Immutable; {@link #score} allocates its result.</p>
 */
public final class ModernBertDecisionHead {

    /** Question types, one scorer output each (modern-bert.cpp:3-4). */
    public static final int N_DECISION_TYPES = 3;

    private final CpuMatVec mv;
    private final int nEmbd;
    private final float eps;
    private final Block[] blocks;
    private final float[] typeEmbd;
    private final float[] clsNorm;
    private final float[] clsNormB;
    private final CpuTensorRef cls;
    private final float[] clsB;
    private final CpuTensorRef clsOut;
    private final float[] clsOutB;

    ModernBertDecisionHead(ModelSpec spec, CpuWeights w, IntUnaryOperator nHead, IntUnaryOperator nHeadKv) {
        ModelHparams hp = spec.hparams();
        this.mv = w.matVec();
        this.nEmbd = hp.nEmbd();
        this.eps = hp.normEps();
        int nDecision = hp.nLayerDecision();
        int nLayer = hp.nLayer();
        if (nDecision <= 0 || nDecision >= nLayer) {
            throw new IllegalArgumentException(spec.arch() + ": " + nDecision + " decision blocks of " + nLayer
                    + " layers is not a decision head");
        }
        int headDim = hp.headVFull();
        this.blocks = new Block[nDecision];
        for (int i = 0; i < nDecision; i++) {
            int il = nLayer - nDecision + i;
            blocks[i] = new Block(w, il, nEmbd, nHead.applyAsInt(il), nHeadKv.applyAsInt(il), headDim);
        }
        this.typeEmbd = HeadsBlocks.vector(w, TensorNames.weight(TensorNames.TOKEN_TYPES),
                N_DECISION_TYPES * nEmbd);
        this.clsNorm = HeadsBlocks.vector(w, TensorNames.weight(TensorNames.CLS_NORM), nEmbd);
        this.clsNormB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.CLS_NORM), nEmbd);
        this.cls = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.CLS), nEmbd, nEmbd);
        this.clsB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.CLS), nEmbd);
        this.clsOut = HeadsBlocks.matrix(w, TensorNames.weight(TensorNames.CLS_OUT), nEmbd, 1);
        this.clsOutB = HeadsBlocks.vector(w, TensorNames.bias(TensorNames.CLS_OUT), 1);
    }

    /** The decision head of a modern-bert model (model-wide head counts) over the tensors of {@code weights}. */
    public static ModernBertDecisionHead of(ModelSpec spec, CpuWeights weights) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(weights, "weights");
        ModelHparams hp = spec.hparams();
        int nHead = hp.nHead(0);
        int nHeadKv = hp.nHeadKv(0);
        return new ModernBertDecisionHead(spec, weights, il -> nHead, il -> nHeadKv);
    }

    /** Index of the first head block; the body runs the layers before it. */
    public static int firstHeadLayer(ModelHparams hp) {
        return hp.nLayer() - hp.nLayerDecision();
    }

    /**
     * Scores of every token: {@code [t][it]} for question type {@code it}.
     *
     * @param hiddenRows the body's final hidden rows of every token (after the output norm), all text tokens
     */
    public float[][] score(float[][] hiddenRows) {
        float[] hidden = HeadsBlocks.pack(hiddenRows, nEmbd);
        int n = hiddenRows.length;
        float[][] scores = new float[n][N_DECISION_TYPES];
        for (int it = 0; it < N_DECISION_TYPES; it++) {
            float[] x = hidden.clone();
            for (int t = 0; t < n; t++) {
                HeadsBlocks.addInPlace(x, t * nEmbd, typeEmbd, it * nEmbd, nEmbd);
            }
            for (Block b : blocks) {
                x = b.forward(this, x, n);
            }
            float[] cur = HeadsBlocks.layerNormed(x, n, nEmbd, clsNorm, clsNormB, eps);
            cur = HeadsBlocks.linear(mv, cls, clsB, cur, n);
            HeadsBlocks.activate(ActOp.GELU_ERF, cur, cur.length);
            float[] s = HeadsBlocks.linear(mv, clsOut, clsOutB, cur, n);
            for (int t = 0; t < n; t++) {
                scores[t][it] = s[t];
            }
        }
        return scores;
    }

    /** One pre-norm head block with biases (modern-bert.cpp:111-129, 255-293). */
    private static final class Block {
        final int nHead;
        final int nHeadKv;
        final int headDim;
        final float[] attnNorm;
        final float[] attnNormB;
        final CpuTensorRef wqkv;
        final float[] wqkvB;
        final CpuTensorRef wo;
        final float[] woB;
        final float[] ffnNorm;
        final float[] ffnNormB;
        final CpuTensorRef up;
        final float[] upB;
        final CpuTensorRef down;
        final float[] downB;

        Block(CpuWeights w, int il, int nEmbd, int nHead, int nHeadKv, int headDim) {
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.headDim = headDim;
            if (nHead <= 0 || nHeadKv <= 0 || nHead % nHeadKv != 0 || headDim <= 0) {
                throw new IllegalArgumentException("decision block " + il + " has " + nHead + " heads, " + nHeadKv
                        + " KV heads of " + headDim);
            }
            this.attnNorm = layerVector(w, il, TensorNames.weight(TensorNames.ATTN_NORM), nEmbd);
            this.attnNormB = layerVector(w, il, TensorNames.bias(TensorNames.ATTN_NORM), nEmbd);
            int qkvRows = (nHead + 2 * nHeadKv) * headDim;
            this.wqkv = layerMatrix(w, il, TensorNames.ATTN_QKV, nEmbd, qkvRows);
            this.wqkvB = layerVector(w, il, TensorNames.bias(TensorNames.ATTN_QKV), qkvRows);
            this.wo = layerMatrix(w, il, TensorNames.ATTN_OUT, nHead * headDim, nEmbd);
            this.woB = layerVector(w, il, TensorNames.bias(TensorNames.ATTN_OUT), nEmbd);
            this.ffnNorm = layerVector(w, il, TensorNames.weight(TensorNames.FFN_NORM), nEmbd);
            this.ffnNormB = layerVector(w, il, TensorNames.bias(TensorNames.FFN_NORM), nEmbd);
            this.up = w.layer(il, TensorNames.weight(TensorNames.FFN_UP));
            if (up.cols() != nEmbd) {
                throw new IllegalArgumentException("decision block " + il + " ffn_up takes " + up.cols()
                        + " inputs, n_embd is " + nEmbd);
            }
            int ff = HeadsBlocks.toInt(up.rows(), up);
            this.upB = layerVector(w, il, TensorNames.bias(TensorNames.FFN_UP), ff);
            this.down = layerMatrix(w, il, TensorNames.FFN_DOWN, ff, nEmbd);
            this.downB = layerVector(w, il, TensorNames.bias(TensorNames.FFN_DOWN), nEmbd);
        }

        float[] forward(ModernBertDecisionHead h, float[] x, int n) {
            int nEmbd = h.nEmbd;
            float[] cur = HeadsBlocks.layerNormed(x, n, nEmbd, attnNorm, attnNormB, h.eps);
            float[] qkv = HeadsBlocks.linear(h.mv, wqkv, wqkvB, cur, n);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int stride = qDim + 2 * kvDim;
            float[] ctx = HeadsBlocks.attendAll(qkv, stride, n, qkv, qDim, qDim + kvDim, stride, n, nHead, nHeadKv,
                    headDim, 1.0f / (float) Math.sqrt(headDim));
            float[] ffnInp = HeadsBlocks.linear(h.mv, wo, woB, ctx, n);
            HeadsBlocks.addInPlace(ffnInp, x, ffnInp.length);
            cur = HeadsBlocks.layerNormed(ffnInp, n, nEmbd, ffnNorm, ffnNormB, h.eps);
            cur = HeadsBlocks.linear(h.mv, up, upB, cur, n);
            HeadsBlocks.activate(ActOp.RELU, cur, cur.length);
            float[] out = HeadsBlocks.linear(h.mv, down, downB, cur, n);
            HeadsBlocks.addInPlace(out, ffnInp, out.length);
            return out;
        }

        private static float[] layerVector(CpuWeights w, int il, String suffix, int n) {
            float[] v = w.layerFloats(il, suffix);
            if (v.length != n) {
                throw new IllegalArgumentException("decision block " + il + " " + suffix + " has " + v.length
                        + " floats, expected " + n);
            }
            return v;
        }

        private static CpuTensorRef layerMatrix(CpuWeights w, int il, String base, long cols, long rows) {
            CpuTensorRef m = w.layer(il, TensorNames.weight(base));
            if (m.cols() != cols || m.rows() != rows) {
                throw new IllegalArgumentException("decision block " + il + " " + base + " is " + m.cols() + " x "
                        + m.rows() + ", expected " + cols + " x " + rows);
            }
            return m;
        }
    }
}

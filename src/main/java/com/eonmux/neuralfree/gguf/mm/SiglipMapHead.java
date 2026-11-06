package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * SigLIP's multihead attention pooling head, vLLM's {@code SiglipMultiheadAttentionPoolingHead}
 * (vllm/model_executor/models/siglip.py) over the post-normed patch rows: the learned probe ({@code v.head.probe})
 * attends to every row through {@code torch.nn.MultiheadAttention} (the packed {@code v.head.attention.in_weight}
 * / {@code in_bias} Q/K/V projection, {@code n_head} heads, the {@code v.head.attention.out} projection), then
 * {@code x = attn + mlp(layernorm(attn))} with the head's layer norm ({@code v.head.ln}) and MLP (the two-tower
 * converter's swapped names: {@code v.head.ffn_down} is fc1, {@code v.head.ffn_up} is fc2) under the tower's
 * activation. The output is the one pooled row. Immutable weights; not thread-safe across calls.
 */
final class SiglipMapHead {

    static final String PROBE = "v.head.probe";
    static final String IN_WEIGHT = "v.head.attention.in_weight";
    static final String IN_BIAS = "v.head.attention.in_bias";
    static final String OUT = "v.head.attention.out";
    static final String LN = "v.head.ln";
    static final String FC1 = "v.head.ffn_down";
    static final String FC2 = "v.head.ffn_up";

    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final float[] probe;
    private final CpuTensorRef inWeight;
    private final float[] inBias;
    private final CpuTensorRef out;
    private final float[] outBias;
    private final float[] lnW;
    private final float[] lnB;
    private final CpuTensorRef fc1;
    private final float[] fc1B;
    private final CpuTensorRef fc2;
    private final float[] fc2B;

    private SiglipMapHead(ClipHparams hp, CpuMatVec matVec, Function<String, float[]> floats,
                          Function<String, CpuTensorRef> refs) {
        this.hp = hp;
        this.matVec = matVec;
        this.probe = floats.apply(PROBE);
        this.inWeight = refs.apply(IN_WEIGHT);
        this.inBias = floats.apply(IN_BIAS);
        this.out = refs.apply(OUT + ".weight");
        this.outBias = floats.apply(OUT + ".bias");
        this.lnW = floats.apply(LN + ".weight");
        this.lnB = floats.apply(LN + ".bias");
        this.fc1 = refs.apply(FC1 + ".weight");
        this.fc1B = floats.apply(FC1 + ".bias");
        this.fc2 = refs.apply(FC2 + ".weight");
        this.fc2B = floats.apply(FC2 + ".bias");
    }

    /** True when {@code tc} carries the head ({@value #PROBE}). */
    static boolean present(TensorCatalog tc) {
        return tc.get(PROBE) != null;
    }

    /**
     * Resolves and shape-checks the head tensors of {@code tc} for the vision tower {@code hp}, then binds them.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a head tensor is absent
     * @throws IllegalArgumentException when a shape differs from the head's
     */
    static SiglipMapHead bind(ClipHparams hp, TensorCatalog tc, CpuMatVec matVec, Function<String, float[]> floats,
                              Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(hp, "hparams");
        int d = hp.nEmbd();
        int ff = hp.nFf();
        TensorRef probe = tc.require(PROBE);
        if (probe.numElements() != d) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + PROBE + " has shape "
                    + probe + ", the loader expects " + d + " values");
        }
        shape(tc, tc.require(IN_WEIGHT), d, 3 * d);
        checkVector(tc, tc.require(IN_BIAS), 3 * d);
        shape(tc, tc.require(OUT + ".weight"), d, d);
        checkVector(tc, tc.require(OUT + ".bias"), d);
        checkVector(tc, tc.require(LN + ".weight"), d);
        checkVector(tc, tc.require(LN + ".bias"), d);
        shape(tc, tc.require(FC1 + ".weight"), d, ff);
        checkVector(tc, tc.require(FC1 + ".bias"), ff);
        shape(tc, tc.require(FC2 + ".weight"), ff, d);
        checkVector(tc, tc.require(FC2 + ".bias"), d);
        return new SiglipMapHead(hp, matVec, floats, refs);
    }

    /** The pooled row of the {@code n} post-normed rows {@code rows} ({@code [n][n_embd]}). */
    float[] pool(float[] rows, int n) {
        int d = hp.nEmbd();
        float[] qkvProbe = new float[3 * d];
        matVec.matmul(inWeight, probe, 0, qkvProbe, 0, 1);
        float[] kv = new float[n * 3 * d];
        matVec.matmul(inWeight, rows, 0, kv, 0, n);
        int nHead = hp.nHead();
        int hd = d / nHead;
        float scale = (float) (1.0 / Math.sqrt(hd));
        float[] attn = new float[d];
        float[] scores = new float[n];
        for (int h = 0; h < nHead; h++) {
            for (int t = 0; t < n; t++) {
                double dot = 0;
                for (int i = 0; i < hd; i++) {
                    dot += (qkvProbe[h * hd + i] + inBias[h * hd + i])
                            * (kv[t * 3 * d + d + h * hd + i] + inBias[d + h * hd + i]);
                }
                scores[t] = (float) (dot * scale);
            }
            CpuOps.softmax(scores, 0, n);
            for (int t = 0; t < n; t++) {
                for (int i = 0; i < hd; i++) {
                    attn[h * hd + i] += scores[t] * (kv[t * 3 * d + 2 * d + h * hd + i] + inBias[2 * d + h * hd + i]);
                }
            }
        }
        float[] x = new float[d];
        matVec.matmul(out, attn, 0, x, 0, 1);
        VisionRows.addBias(x, outBias);
        float[] normed = new float[d];
        CpuOps.layerNorm(x, 0, normed, 0, 1, d, lnW, lnB, hp.eps());
        float[] hidden = new float[hp.nFf()];
        matVec.matmul(fc1, normed, 0, hidden, 0, 1);
        VisionRows.addBias(hidden, fc1B);
        CpuOps.unary(hp.ffnOp().unary(), ActParams.NONE, hidden, 0, hidden, 0, hidden.length);
        float[] mlp = new float[d];
        matVec.matmul(fc2, hidden, 0, mlp, 0, 1);
        VisionRows.addBias(mlp, fc2B);
        for (int i = 0; i < d; i++) {
            x[i] += mlp[i];
        }
        return x;
    }
}

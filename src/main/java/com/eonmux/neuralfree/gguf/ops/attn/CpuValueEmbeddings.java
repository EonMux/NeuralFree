package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * limite's value-side stages of {@link CpuAttentionMixer}: the value embeddings ({@code value_embd} rows added to V,
 * limite.cpp:87-89, 149-155) and the exclusive self attention (XSA, limite.cpp:177-191), each token's query head
 * losing its projection on the token's own L2-normalized value. Not thread-safe; driven by the owning mixer's step
 * thread.
 */
final class CpuValueEmbeddings {

    private static final String T_VE = "attn.ve";
    private static final String T_VE_GATE = "attn.ve_gate";
    /** Elements per parallel task of the element-wise loops. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private final CpuAttentionMixer mixer;
    private CpuTensorRef valueEmbd;
    private CpuTensorRef veGateW;
    private float[] xsaAlpha;

    CpuValueEmbeddings(CpuAttentionMixer mixer, CpuWeights w) {
        this.mixer = mixer;
        LayerSpec layer = mixer.layer();
        AttentionSpec spec = mixer.spec();
        valueEmbd = w.get(TensorNames.weight(TensorNames.VALUE_EMBD));
        if (valueEmbd == null) {
            throw new MissingTensorException(mixer.ctx().spec().arch().ggufName(),
                    layer.il(), TensorNames.weight(TensorNames.VALUE_EMBD));
        }
        if (valueEmbd.cols() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + TensorNames.weight(TensorNames.VALUE_EMBD)
                    + " holds rows of " + valueEmbd.cols() + " values, V is " + spec.vDim() + " wide");
        }
        if (spec.valueEmbeddingGate()) {
            veGateW = w.layer(layer, TensorNames.weight(TensorNames.ATTN_VE_GATE));
            if (veGateW.cols() != mixer.nEmbd() || veGateW.rows() != spec.nHeadKv()) {
                throw new IllegalArgumentException("layer " + layer.il() + ": "
                        + TensorNames.weight(TensorNames.ATTN_VE_GATE) + " must be [" + mixer.nEmbd()
                        + ", " + spec.nHeadKv() + "]");
            }
        }
        if (spec.xsaEps() > 0f) {
            xsaAlpha = w.layerFloatsOrNull(layer, TensorNames.weight(TensorNames.ATTN_XSA_ALPHA));
            if (xsaAlpha == null || xsaAlpha.length != spec.nHead()) {
                throw new IllegalArgumentException("layer " + layer.il() + ": "
                        + TensorNames.weight(TensorNames.ATTN_XSA_ALPHA) + " must hold " + spec.nHead() + " values, got "
                        + (xsaAlpha == null ? 0 : xsaAlpha.length));
            }
        }
    }

    /**
     * limite value embeddings (limite.cpp:87-89, 149-155): each token's {@code value_embd} row, gated per KV head
     * by {@code sigmoid(attn_ve_gate @ x)} when the layer carries the gate, adds to V.
     */
    void apply(int n, float[] in, float[] v) {
        CpuScratch scratch = mixer.scratch();
        AttentionSpec spec = mixer.spec();
        int[] ids = scratch.ints(CpuScratch.TOKEN_IDS);
        int vDim = spec.vDim();
        int nHeadKv = spec.nHeadKv();
        int dV = spec.headDimV();
        float[] ve = scratch.temp(T_VE, mixer.ubatch() * vDim);
        float[] gate = veGateW != null ? scratch.temp(T_VE_GATE, mixer.ubatch() * nHeadKv) : null;
        if (gate != null) {
            mixer.matVec().matmul(veGateW, in, 0, mixer.nEmbd(), gate, 0, nHeadKv, n);
        }
        CpuParallel.parallelFor(n, Math.max(1, ELEMENT_GRAIN / Math.max(1, vDim)), (from, to) -> {
            for (int t = from; t < to; t++) {
                valueEmbd.decodeRow(ids[t], ve, t * vDim);
                float g = 1f;
                for (int h = 0; h < nHeadKv; h++) {
                    if (gate != null) {
                        g = ActMath.sigmoid(gate[t * nHeadKv + h]);
                    }
                    for (int d = 0; d < dV; d++) {
                        int i = t * vDim + h * dV + d;
                        v[i] += ve[i] * g;
                    }
                }
            }
        });
    }

    /**
     * limite exclusive self attention (limite.cpp:177-191): each query head loses its projection on the token's
     * own L2-normalized value, scaled by the head's {@code attn_xsa_alpha}.
     */
    void applyXsa(int n, float[] v, float[] attn) {
        AttentionSpec spec = mixer.spec();
        int dV = spec.headDimV();
        int vDim = spec.vDim();
        int outDim = spec.outDim();
        int nG = spec.gqaRatio();
        float eps = spec.xsaEps();
        CpuParallel.parallelFor(n, Math.max(1, ELEMENT_GRAIN / Math.max(1, outDim)), (from, to) -> {
            for (int t = from; t < to; t++) {
                for (int kvh = 0; kvh < spec.nHeadKv(); kvh++) {
                    int vBase = t * vDim + kvh * dV;
                    float sum = 0f;
                    for (int d = 0; d < dV; d++) {
                        sum += v[vBase + d] * v[vBase + d];
                    }
                    float norm = Math.max((float) Math.sqrt(sum), eps);
                    for (int g = 0; g < nG; g++) {
                        int aBase = t * outDim + (kvh * nG + g) * dV;
                        float proj = 0f;
                        for (int d = 0; d < dV; d++) {
                            proj += attn[aBase + d] * (v[vBase + d] / norm);
                        }
                        proj *= xsaAlpha[kvh * nG + g];
                        for (int d = 0; d < dV; d++) {
                            attn[aBase + d] -= (v[vBase + d] / norm) * proj;
                        }
                    }
                }
            }
        });
    }

    /** Whether the layer runs exclusive self attention (the spec sets {@code xsaEps}). */
    boolean hasXsa() {
        return xsaAlpha != null;
    }

    /** Whether the layer carries value embeddings (the spec's {@code valueEmbeddings} flag). */
    boolean hasValues() {
        return valueEmbd != null;
    }
}

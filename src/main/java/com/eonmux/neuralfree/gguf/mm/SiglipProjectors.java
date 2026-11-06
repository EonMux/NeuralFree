package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The SigLIP-encoder projectors of models/siglip.cpp beyond Gemma3: Idefics3 (the
 * {@code build_patch_merge_permute} shuffle and the linear {@code mm.model.fc}), Janus-Pro (the
 * {@code mm.0} / {@code mm.1} MLP with the encoder's activation), LFM2-VL and d1-omni (the merge permute, the
 * optional layer norm at eps 1e-5, the {@code mm.1} / {@code mm.2} MLP with GELU, exact GELU for d1-omni) and
 * Cohere2 vision (the merge permute, {@code mm.1}, the swapped SwiGLU {@code x * silu(gate)} over the
 * {@code [x, gate]} halves, {@code mm.2}). Written once against {@link VisionOps}. Package-private executor
 * support.
 */
final class SiglipProjectors {

    /** The LFM2 projector layer-norm epsilon, PyTorch's {@code nn.LayerNorm} default. */
    static final float LFM2_EPS = 1e-5f;

    private SiglipProjectors() {
    }

    /** Idefics3: {@code build_patch_merge_permute} by the scale factor, then {@code mm.model.fc}. */
    static float[][] idefics3(VisionOps ops, Encoded enc) {
        enc.requireGrid("the idefics3 projector");
        int s = ops.hp().nMerge();
        VisionOps.Block cur = ops.mergePermute(ops.upload(enc.rows, enc.width), ops.hp().nEmbd(), enc.pw, enc.ph,
                s);
        VisionOps.Block out = ops.linear("mm.model.fc.weight", cur, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** Janus-Pro: {@code mm.0} + the encoder activation + {@code mm.1} over every encoder row. */
    static float[][] janus(VisionOps ops, Encoded enc) {
        enc.requireGrid("the janus projector");
        VisionOps.Block h = ops.linear("mm.0.weight", ops.upload(enc.rows, enc.width), "mm.0.bias");
        ops.activate(h, ops.hp().ffnOp().unary());
        VisionOps.Block out = ops.linear("mm.1.weight", h, "mm.1.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** LFM2-VL / d1-omni: merge permute, the optional layer norm, {@code mm.1} + GELU + {@code mm.2}. */
    static float[][] lfm2(VisionOps ops, Encoded enc) {
        enc.requireGrid("the lfm2 projector");
        int s = ops.hp().nMerge();
        int width = ops.hp().nEmbd() * s * s;
        VisionOps.Block cur = ops.mergePermute(ops.upload(enc.rows, enc.width), ops.hp().nEmbd(), enc.pw, enc.ph,
                s);
        int n = cur.n();
        if (ops.hasVector("mm.input_norm.weight")) {
            ops.norm(cur, NormKind.LAYER, "mm.input_norm.weight", null, LFM2_EPS, n, width);
        }
        float[] normB = ops.hostVectorOrNull("mm.input_norm.bias");
        if (normB != null) {
            ops.addHost(cur, tiled(normB, n * width));
        }
        ActOp act = ops.hp().projectorType() == ProjectorType.D1OMNI_V ? ActOp.GELU_ERF : ActOp.GELU_TANH;
        VisionOps.Block h = ops.linear("mm.1.weight", cur, "mm.1.bias");
        ops.activate(h, act);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** {@code bias} tiled to {@code total} values (a row bias over the whole block). */
    private static float[] tiled(float[] bias, int total) {
        if (bias.length == 0 || total % bias.length != 0) {
            throw new IllegalArgumentException("a bias of " + bias.length + " values does not tile to " + total);
        }
        if (total == bias.length) {
            return bias;
        }
        float[] out = new float[total];
        for (int at = 0; at < total; at += bias.length) {
            System.arraycopy(bias, 0, out, at, bias.length);
        }
        return out;
    }

    /** Cohere2 vision: merge permute, {@code mm.1}, {@code x * silu(gate)}, {@code mm.2}. */
    static float[][] cohere2v(VisionOps ops, Encoded enc) {
        enc.requireGrid("the cohere2v projector");
        int s = ops.hp().nMerge();
        int width = ops.hp().nEmbd() * s * s;
        VisionOps.Block cur = ops.mergePermute(ops.upload(enc.rows, enc.width), ops.hp().nEmbd(), enc.pw, enc.ph,
                s);
        int n = cur.n();
        VisionOps.Block h = ops.linear("mm.1.weight", cur, "mm.1.bias");
        int half = h.width() / 2;
        float[] values = ops.download(h);
        float[] gated = new float[n * half];
        float[] gate = new float[half];
        for (int t = 0; t < n; t++) {
            System.arraycopy(values, (2 * t + 1) * half, gate, 0, half);
            VisionRows.activate(ActOp.SILU, gate);
            for (int i = 0; i < half; i++) {
                gated[t * half + i] = values[2 * t * half + i] * gate[i];
            }
        }
        VisionOps.Block out = ops.linear("mm.2.weight", ops.upload(gated, half), "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The LFM3-VL graph (models/lfm3vl.cpp {@code clip_graph_lfm3vl}): the patchifier (one conv kernel, or both
 * temporal kernels summed over the still image; no bias) in raster order and the RMS embedding norm; then per block
 * the RMS norm 1 and either the conv mixer ({@code conv2d_in_proj} to three {@code n_embd} slices, the first times
 * the third, the optional depthwise 3x3 {@code conv2d_spatial} over the patch grid, times the middle slice,
 * {@code conv2d_out_proj}) or the attention (separate Q/K/V, {@code n_head_kv} KV heads, the optional per-head
 * RMS Q/K norms, the 2D m-rope over raster positions at base 1e6, the output projection and its optional bias),
 * the residual, the RMS norm 2, the encoder's FFN and the residual; no post-norm. The projector: the layer norm
 * {@code mm.input_norm} (eps 1e-6), the 2x2 merge-group reorder, {@code mm.0} + GELU-erf + {@code mm.2}. The block
 * tensors are projector tensors ({@link #resolve}), read by name. Written once against {@link VisionOps}.
 * Package-private executor support.
 */
final class Lfm3vlGraph {

    /** {@code vision_rope_theta} of the reference loader. */
    static final float ROPE_FREQ_BASE = 1_000_000f;
    /** {@code projector_layer_norm_eps} of the projector input norm. */
    static final float PROJECTOR_NORM_EPS = 1e-6f;
    /** The merge window side the projector's reshapes hard-code. */
    static final int MERGE = 2;
    /** Kernel side of the spatial depthwise convolution. */
    static final int SPATIAL_KERNEL = 3;

    private Lfm3vlGraph() {
    }

    /** Tensor {@code name} of block {@code il}. */
    static String blockTensor(int il, String name) {
        return "v.blk." + il + "." + name;
    }

    /**
     * The embedding norm and every block's tensors into {@code c}, shape-checked: the RMS norms, a complete conv
     * mixer or a complete attention (never both, clip.cpp {@code load_tensors}), and the FFN.
     *
     * @throws IllegalArgumentException when a block is neither a complete conv nor a complete attention block, or
     *                                  a shape differs from the loader's (named)
     */
    static void resolve(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        int e = hp.nEmbd();
        c.vector("v.embedding_norm.weight", e);
        for (int il = 0; il < hp.nLayer(); il++) {
            c.vector(blockTensor(il, "ln1.weight"), e).optionalVector(blockTensor(il, "ln1.bias"), e)
                    .vector(blockTensor(il, "ln2.weight"), e).optionalVector(blockTensor(il, "ln2.bias"), e);
            boolean conv = tc.has(blockTensor(il, "conv2d_in_proj.weight"));
            boolean attn = tc.has(blockTensor(il, "attn_q.weight")) && tc.has(blockTensor(il, "attn_k.weight"))
                    && tc.has(blockTensor(il, "attn_v.weight")) && tc.has(blockTensor(il, "attn_out.weight"));
            if (conv == attn) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': lfm3vl block " + il
                        + " is neither a complete conv nor a complete attention block");
            }
            if (conv) {
                convTensors(c, tc, il, e);
            } else {
                attentionTensors(c, hp, il);
            }
            ffnTensors(c, tc, il, e);
        }
    }

    private static void convTensors(ClipProjectorTensors.Collector c, TensorCatalog tc, int il, int e) {
        c.matrix(blockTensor(il, "conv2d_in_proj.weight"), e, 3L * e)
                .matrix(blockTensor(il, "conv2d_out_proj.weight"), e, e);
        if (tc.has(blockTensor(il, "conv2d_spatial.weight"))) {
            c.depthwise(blockTensor(il, "conv2d_spatial.weight"), SPATIAL_KERNEL, e);
        }
    }

    private static void attentionTensors(ClipProjectorTensors.Collector c, ClipHparams hp, int il) {
        int e = hp.nEmbd();
        int dh = hp.dHead();
        int kv = hp.nHeadKv() * dh;
        c.matrix(blockTensor(il, "attn_q.weight"), e, e).matrix(blockTensor(il, "attn_k.weight"), e, kv)
                .matrix(blockTensor(il, "attn_v.weight"), e, kv).matrix(blockTensor(il, "attn_out.weight"), e, e)
                .optionalVector(blockTensor(il, "attn_out.bias"), e)
                .optionalVector(blockTensor(il, "attn_q_norm.weight"), dh)
                .optionalVector(blockTensor(il, "attn_k_norm.weight"), dh);
    }

    private static void ffnTensors(ClipProjectorTensors.Collector c, TensorCatalog tc, int il, int e) {
        String up = blockTensor(il, "ffn_up.weight");
        int f = c.outDim(up, e);
        c.matrix(up, e, f).optionalVector(blockTensor(il, "ffn_up.bias"), f)
                .matrix(blockTensor(il, "ffn_down.weight"), f, e).optionalVector(blockTensor(il, "ffn_down.bias"), e);
        if (tc.has(blockTensor(il, "ffn_gate.weight"))) {
            c.matrix(blockTensor(il, "ffn_gate.weight"), e, f).optionalVector(blockTensor(il, "ffn_gate.bias"), f);
        }
    }

    /**
     * The embedding rows of {@code image}: {@code (pw / 2) * (ph / 2)} rows of the projector output width.
     *
     * @throws IllegalArgumentException when the patch grid is not even per side (named)
     */
    static float[][] encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        if (pw % MERGE != 0 || ph % MERGE != 0) {
            throw new IllegalArgumentException("the lfm3vl graph merges 2x2 patches, the patch grid " + pw + "x" + ph
                    + " must be even per side");
        }
        int e = hp.nEmbd();
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        ops.norm(x, NormKind.RMS, "v.embedding_norm.weight", null, hp.eps(), n, e);
        int[] raster = new int[n];
        for (int t = 0; t < n; t++) {
            raster[t] = t;
        }
        VisionOps.RopePlan rope = VisionOps.RopePlan.qwen(QwenVisionGrid.ropePositions(raster, pw), ROPE_FREQ_BASE);
        for (int il = 0; il < hp.nLayer(); il++) {
            int mark = ops.mark();
            block(ops, il, x, pw, ph, rope);
            ops.release(mark);
        }
        ops.norm(x, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", PROJECTOR_NORM_EPS, n, e);
        x = ops.gatherRows(x, Qwen3VlGraph.mergeOrder(pw, ph, MERGE), e);
        VisionOps.Block h = ops.linear("mm.0.weight", ops.view(x, n / (MERGE * MERGE), MERGE * MERGE * e),
                "mm.0.bias");
        ops.activate(h, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** One block over the residual stream {@code x} in place: the mixer and the FFN, each added to the stream. */
    private static void block(VisionOps ops, int il, VisionOps.Block x, int pw, int ph, VisionOps.RopePlan rope) {
        int e = x.width();
        int n = x.n();
        float eps = ops.hp().eps();
        VisionOps.Block cur = ops.columns(x, 0, e);
        ops.norm(cur, NormKind.RMS, blockTensor(il, "ln1.weight"), blockTensor(il, "ln1.bias"), eps, n, e);
        VisionOps.Block mixed = ops.hasMatrix(blockTensor(il, "conv2d_in_proj.weight"))
                ? conv(ops, il, cur, pw, ph) : attention(ops, il, cur, rope);
        ops.addBlock(x, mixed);
        VisionOps.Block normed = ops.columns(x, 0, e);
        ops.norm(normed, NormKind.RMS, blockTensor(il, "ln2.weight"), blockTensor(il, "ln2.bias"), eps, n, e);
        ops.addBlock(x, ffn(ops, il, normed));
    }

    /** The gated conv mixer: {@code out_proj(middle * spatial(first * third))}. */
    private static VisionOps.Block conv(VisionOps ops, int il, VisionOps.Block cur, int pw, int ph) {
        int e = cur.width();
        int n = cur.n();
        VisionOps.Block proj = ops.linear(blockTensor(il, "conv2d_in_proj.weight"), cur, null);
        VisionOps.Block first = ops.columns(proj, 0, e);
        VisionOps.Block middle = ops.columns(proj, e, e);
        VisionOps.Block gated = ops.columns(proj, 2 * e, e);
        ops.glu(first, gated, GluOp.MUL, n, e);
        String spatial = blockTensor(il, "conv2d_spatial.weight");
        if (ops.hasVector(spatial)) {
            TensorRef kernel = ops.model().projectorVectors().get(spatial);
            gated = ops.depthwise3x3(gated, pw, ph, kernel, 1);
        }
        ops.glu(middle, gated, GluOp.MUL, n, e);
        return ops.linear(blockTensor(il, "conv2d_out_proj.weight"), gated, null);
    }

    /** GQA attention with the optional per-head Q/K norms and the 2D m-rope. */
    private static VisionOps.Block attention(VisionOps ops, int il, VisionOps.Block cur, VisionOps.RopePlan rope) {
        ClipHparams hp = ops.hp();
        int n = cur.n();
        int dh = ops.model().dHead();
        int nHead = hp.nHead();
        int nHeadKv = hp.nHeadKv();
        VisionOps.Block q = ops.linear(blockTensor(il, "attn_q.weight"), cur, null);
        VisionOps.Block k = ops.linear(blockTensor(il, "attn_k.weight"), cur, null);
        VisionOps.Block v = ops.linear(blockTensor(il, "attn_v.weight"), cur, null);
        String qNorm = blockTensor(il, "attn_q_norm.weight");
        if (ops.hasVector(qNorm)) {
            ops.norm(q, NormKind.RMS, qNorm, null, hp.eps(), n * nHead, dh);
        }
        String kNorm = blockTensor(il, "attn_k_norm.weight");
        if (ops.hasVector(kNorm)) {
            ops.norm(k, NormKind.RMS, kNorm, null, hp.eps(), n * nHeadKv, dh);
        }
        ops.rope(q, nHead, rope);
        ops.rope(k, nHeadKv, rope);
        VisionOps.Block a = ops.attend(q, 0, n, k, v, 0, n, nHeadKv * dh, nHead, nHead / nHeadKv, dh,
                ops.model().attnScale());
        return ops.linear(blockTensor(il, "attn_out.weight"), a, blockTensor(il, "attn_out.bias"));
    }

    /** {@code down(act(up(x)) [* gate]) } with the optional biases, the encoder's FFN op. */
    private static VisionOps.Block ffn(VisionOps ops, int il, VisionOps.Block x) {
        ClipHparams hp = ops.hp();
        VisionOps.Block up = ops.linear(blockTensor(il, "ffn_up.weight"), x, blockTensor(il, "ffn_up.bias"));
        String gateName = blockTensor(il, "ffn_gate.weight");
        if (ops.hasMatrix(gateName)) {
            VisionOps.Block gate = ops.linear(gateName, x, blockTensor(il, "ffn_gate.bias"));
            ops.glu(gate, up, hp.ffnGlu(), hp.ffnGluParams(), up.n(), up.width());
        } else {
            ops.activate(up, hp.ffnOp().unary());
        }
        return ops.linear(blockTensor(il, "ffn_down.weight"), up, blockTensor(il, "ffn_down.bias"));
    }
}

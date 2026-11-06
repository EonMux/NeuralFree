package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The tensor names of the DeepSeek-OCR SAM ViT-B backbone (clip-impl.h {@code TN_SAM_*}) and their shape-checked
 * collection into the projector tensors (clip.cpp {@code load_tensors}), so the CPU executor resolves them with the
 * projector's matrices and vectors. Static members only.
 */
final class SamTensors {

    /** The patch kernel {@code [p, p, 3, n_embd]}, its stride the kernel side. */
    static final String PATCH = "v.sam.patch_embd.weight";
    /** The patch bias. */
    static final String PATCH_BIAS = "v.sam.patch_embd.bias";
    /** The absolute position grid {@code [n_embd, side, side]}. */
    static final String POS = "v.sam.pos_embd.weight";
    /** The 1x1 neck conv {@code [1, 1, n_embd, c_neck]}, no bias. */
    static final String NECK_0 = "v.sam.neck.0.weight";
    /** The channel layer norm after the 1x1 neck conv. */
    static final String NECK_1 = "v.sam.neck.1";
    /** The 3x3 neck conv {@code [3, 3, c_neck, c_neck]}, padding 1, no bias. */
    static final String NECK_2 = "v.sam.neck.2.weight";
    /** The channel layer norm after the 3x3 neck conv. */
    static final String NECK_3 = "v.sam.neck.3";
    /** The first stride-2 3x3 conv {@code [3, 3, c_neck, c_net2]}, padding 1, no bias. */
    static final String NET_2 = "v.sam.net_2.weight";
    /** The second stride-2 3x3 conv {@code [3, 3, c_net2, c_out]}, padding 1, no bias. */
    static final String NET_3 = "v.sam.net_3.weight";

    /** The neck and downsampler kernel side. */
    static final int CONV_KERNEL = 3;

    private SamTensors() {
    }

    /** The name of tensor {@code suffix} of SAM block {@code il} ({@code v.sam.blk.N.suffix}). */
    static String block(int il, String suffix) {
        return "v.sam.blk." + il + "." + suffix;
    }

    /**
     * Collects every SAM tensor of {@code hp}'s backbone into {@code c}, shape-checked against the SAM width, the
     * head width and the kernel chain.
     *
     * @return the SAM output channels (the {@code net_3} width)
     * @throws IllegalArgumentException when a shape differs from the reference loader's (named)
     */
    static int collect(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        VisionFamilyHparams.Sam sam = hp.visionFamily().sam();
        if (sam == null) {
            throw new IllegalStateException("projector type " + hp.projectorType() + " has no SAM backbone keys");
        }
        long e = sam.nEmbd();
        long p = tc.require(PATCH).ne(0);
        c.kernel(PATCH, p, p, 3, e).vector(PATCH_BIAS, e);
        long side = tc.require(POS).ne(1);
        c.vector(POS, e, side, side);
        for (int il = 0; il < sam.nLayer(); il++) {
            block(c, tc, il, e, e / sam.nHead());
        }
        long neck = tc.require(NECK_0).ne(3);
        c.kernel(NECK_0, 1, 1, e, neck).vector(NECK_1 + ".weight", neck).vector(NECK_1 + ".bias", neck);
        int k = CONV_KERNEL;
        c.kernel(NECK_2, k, k, neck, neck).vector(NECK_3 + ".weight", neck).vector(NECK_3 + ".bias", neck);
        long net2 = tc.require(NET_2).ne(3);
        c.kernel(NET_2, k, k, neck, net2);
        long out = tc.require(NET_3).ne(3);
        c.kernel(NET_3, k, k, net2, out);
        if (out <= 0 || out > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + NET_3
                    + " has no usable output width");
        }
        return (int) out;
    }

    /** One SAM block: fused biased QKV, output projection, two layer norms, the rel-pos tables and the MLP. */
    private static void block(ClipProjectorTensors.Collector c, TensorCatalog tc, int il, long e, long dHead) {
        c.matrix(block(il, "attn.qkv.weight"), e, 3 * e).vector(block(il, "attn.qkv.bias"), 3 * e);
        c.matrix(block(il, "attn.out.weight"), e, e).vector(block(il, "attn.out.bias"), e);
        for (String ln : new String[]{"pre_ln", "post_ln"}) {
            c.vector(block(il, ln + ".weight"), e).vector(block(il, ln + ".bias"), e);
        }
        for (String rel : new String[]{"attn.pos_h.weight", "attn.pos_w.weight"}) {
            TensorRef r = tc.require(block(il, rel));
            c.vector(r.name(), dHead, r.ne(1));
        }
        int ff = c.outDim(block(il, "mlp.lin1.weight"), e);
        c.matrix(block(il, "mlp.lin1.weight"), e, ff).vector(block(il, "mlp.lin1.bias"), ff);
        c.matrix(block(il, "mlp.lin2.weight"), ff, e).vector(block(il, "mlp.lin2.bias"), e);
    }
}

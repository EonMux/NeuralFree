package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The HunyuanVL graph (models/hunyuanvl.cpp {@code clip_graph_hunyuanvl}): the patch rows plus the learned position
 * grid resampled on the host by the reference's bilinear rule (scale {@code (out + 0.1) / n_grid} per axis, pixel
 * centres, clamped neighbours), the layer-norm ViT, then the perceiver projector: an RMS norm with
 * {@code mm.pre_norm}, the {@code n_merge x n_merge} stride-{@code n_merge} conv {@code mm.0} with bias, GELU, the
 * pointwise conv {@code mm.2} with bias, the {@code v.image_newline} column after every row of merged tokens,
 * {@code mm.model.fc} with bias, the {@code mm.image_begin} / {@code mm.image_end} rows around the tokens, and a
 * closing RMS norm with {@code mm.post_norm}. The conv drops a trailing patch row or column that does not fill a
 * window, as the unpadded conv does. Written once against {@link VisionOps}. Package-private executor support.
 */
final class HunyuanVlGraph {

    /** The offset the reference adds to the target size when it computes the resampling scale. */
    static final float POS_SCALE_OFFSET = 0.1f;

    private HunyuanVlGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        ops.addHost(x, positions(ops.hostWeight(ops.model().posEmbd()), nEmbd, pw, ph));
        x = ops.tower(x, pw * ph, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        return new Encoded(ops.download(x), pw * ph, pw, ph);
    }

    /** The reference's host resampling of the square position grid to {@code pw x ph}, raster order. */
    static float[] positions(float[] posEmbd, int nEmbd, int pw, int ph) {
        int side = SiglipResizedGraph.side(posEmbd, nEmbd);
        float sx = (pw + POS_SCALE_OFFSET) / side;
        float sy = (ph + POS_SCALE_OFFSET) / side;
        return SiglipResizedGraph.bilinear(posEmbd, nEmbd, side, pw, ph, sx, sy, 0.5f);
    }

    /** The perceiver projector: {@code (pw / s + 1) * (ph / s) + 2} rows of the output width. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the hunyuanvl projector");
        ClipHparams hp = ops.hp();
        int nEmbd = hp.nEmbd();
        int s = hp.nMerge();
        int ow = enc.pw / s;
        int oh = enc.ph / s;
        if (ow == 0 || oh == 0) {
            throw new IllegalArgumentException("the hunyuanvl perceiver merges " + s + "x" + s + " patches, the"
                    + " patch grid " + enc.pw + "x" + enc.ph + " holds no full window");
        }
        VisionOps.Block normed = ops.upload(enc.rows.clone(), enc.width);
        ops.norm(normed, NormKind.RMS, "mm.pre_norm.weight", null, hp.eps(), enc.nTokens, nEmbd);
        VisionOps.Block cols = ops.unfoldWindows(normed, nEmbd, enc.pw, ow, oh, s);
        int c1 = ops.hostVector("mm.0.bias").length;
        VisionOps.Block h = ops.linearKernel("mm.0.weight", c1, nEmbd * s * s, cols, "mm.0.bias");
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block c2 = ops.linearKernel("mm.2.weight", ops.hostVector("mm.2.bias").length, c1, h, "mm.2.bias");
        int width = c2.width();
        VisionOps.Block tokens = insertNewlines(ops, c2, width, ow, oh);
        int nTok = (ow + 1) * oh;
        VisionOps.Block proj = ops.linear("mm.model.fc.weight", tokens, "mm.model.fc.bias");
        int d = ops.model().outputDim();
        float[] begin = ops.hostVector("mm.image_begin");
        float[] end = ops.hostVector("mm.image_end");
        float[] values = ops.download(proj);
        float[] out = new float[(nTok + 2) * d];
        System.arraycopy(begin, 0, out, 0, d);
        System.arraycopy(values, 0, out, d, nTok * d);
        System.arraycopy(end, 0, out, (nTok + 1) * d, d);
        VisionOps.Block framed = ops.upload(out, d);
        ops.norm(framed, NormKind.RMS, "mm.post_norm.weight", null, hp.eps(), nTok + 2, d);
        return VisionRows.tokenRows(ops.download(framed), d);
    }

    /** {@code [(ow + 1) * oh][width]} rows: each row of {@code ow} tokens followed by the newline embedding. */
    private static VisionOps.Block insertNewlines(VisionOps ops, VisionOps.Block tokens, int width, int ow,
                                                  int oh) {
        float[] newline = ops.hostVector("v.image_newline");
        if (newline.length != width) {
            throw new IllegalArgumentException("v.image_newline holds " + newline.length + " values, the perceiver"
                    + " width is " + width);
        }
        float[] values = ops.download(tokens);
        float[] out = new float[(ow + 1) * oh * width];
        for (int y = 0; y < oh; y++) {
            System.arraycopy(values, y * ow * width, out, y * (ow + 1) * width, ow * width);
            System.arraycopy(newline, 0, out, (y * (ow + 1) + ow) * width, width);
        }
        return ops.upload(out, width);
    }
}

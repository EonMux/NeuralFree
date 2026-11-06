package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The InternVL graph (models/internvl.cpp {@code clip_graph_internvl}): InternViT over one
 * {@code clip.vision.image_size} tile with the class token first and learned position rows, layer norm (RMS norm
 * for the 6B encoder: {@code n_embd} 3200 and 45 blocks), then the class token dropped, the pixel shuffle by
 * {@code clip.vision.projector.scale_factor}, a layer norm at eps 1e-5 and the GELU MLP. Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class InternVlGraph {

    /** Projector layer-norm epsilon, PyTorch's {@code nn.LayerNorm} default (modeling_internvl_chat.py). */
    private static final float PROJECTOR_EPS = 1e-5f;

    private InternVlGraph() {
    }

    /** The patch rows of {@code image} after the encoder, class token removed, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        ClipVit.requireImageSizeGrid(ops.model(), grid[0], grid[1]);
        int nEmbd = hp.nEmbd();
        int n = grid[0] * grid[1];
        VisionOps.Block x = ops.patchRows(image, grid[0], grid[1]);
        x = ops.insertRow(x, 0, ops.hostWeight(ops.model().classEmbd()));
        ops.addHost(x, ops.hostWeight(ops.model().posEmbd()));
        x = ops.tower(x, n + 1, normKind(hp), VisionOps.TowerOptions.NONE);
        float[] rows = ops.download(x);
        float[] out = new float[n * nEmbd];
        System.arraycopy(rows, nEmbd, out, 0, n * nEmbd);
        return new Encoded(out, n, grid[0], grid[1]);
    }

    /** The 6B InternViT (InternVL 2.5/3 26B, 38B, 78B) normalizes with RMS norm, the 300M ViT with layer norm. */
    static NormKind normKind(ClipHparams hp) {
        return hp.nEmbd() == 3200 && hp.nLayer() == 45 ? NormKind.RMS : NormKind.LAYER;
    }

    /** Pixel shuffle, layer norm, {@code mm.model.mlp.1} + GELU + {@code mm.model.mlp.3}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the internvl projector");
        int s = ops.hp().nMerge();
        int width = ops.hp().nEmbd() * s * s;
        VisionOps.Block cur = ops.shuffleInternvl(ops.upload(enc.rows, enc.width), ops.hp().nEmbd(), enc.pw,
                enc.ph, s);
        int n = cur.n();
        ops.norm(cur, NormKind.LAYER, "mm.model.mlp.0.weight", "mm.model.mlp.0.bias", PROJECTOR_EPS, n, width);
        VisionOps.Block h = ops.linear("mm.model.mlp.1.weight", cur, "mm.model.mlp.1.bias");
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.model.mlp.3.weight", h, "mm.model.mlp.3.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The Llama 4 vision graph (models/llama4.cpp {@code clip_graph_llama4}): the unfold-linear patch embedding over
 * one square {@code clip.vision.image_size} tile, the class token appended last, learned position rows, and a
 * layer-norm ViT whose attention also rotates Q and K with a 2D RoPE (column positions on the first half of each
 * head, row positions on the second, both 1-based with the class token at 0, base 10000). The class token is
 * dropped, the patches pixel-shuffled by {@code clip.vision.projector.scale_factor}, then two GELU layers and the
 * {@code mm.model.fc} projection. Written once against {@link VisionOps}. Package-private executor support.
 */
final class Llama4Graph {

    /** RoPE base of the vision attention (clip.cpp load_hparams). */
    static final double ROPE_THETA = 10000.0;

    private Llama4Graph() {
    }

    /** The patch rows of {@code image} after the encoder, class token removed, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        ClipVit.requireImageSizeGrid(ops.model(), grid[0], grid[1]);
        int pw = grid[0];
        int nEmbd = hp.nEmbd();
        int dh = ops.model().dHead();
        PixtralGraph.requireRopeHeadDim(dh, "llama4");
        int n = pw * grid[1];
        VisionOps.Block x = ops.insertRow(ops.patchRows(image, pw, grid[1]), n,
                ops.hostWeight(ops.model().classEmbd()));
        ops.addHost(x, ops.hostWeight(ops.model().posEmbd()));
        int[] posH = new int[n + 1];
        int[] posW = new int[n + 1];
        for (int i = 0; i < n; i++) {
            posH[i] = i / pw + 1;
            posW[i] = i % pw + 1;
        }
        x = ops.tower(x, n + 1, NormKind.LAYER,
                VisionOps.TowerOptions.NONE.withRope(VisionOps.RopePlan.twoHalves(posW, posH, ROPE_THETA, false)));
        float[] rows = ops.download(x);
        float[] out = new float[n * nEmbd];
        System.arraycopy(rows, 0, out, 0, n * nEmbd);
        return new Encoded(out, n, pw, grid[1]);
    }

    /** Pixel shuffle, {@code mm.model.mlp.1} + GELU, {@code mm.model.mlp.2} + GELU, {@code mm.model.fc}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the llama4 projector");
        if (enc.pw != enc.ph) {
            throw new IllegalArgumentException("the llama4 projector needs a square patch grid, got " + enc.pw
                    + "x" + enc.ph);
        }
        int s = ops.hp().nMerge();
        VisionOps.Block cur = ops.shuffleLlama4(ops.upload(enc.rows, enc.width), ops.hp().nEmbd(), enc.pw, enc.ph,
                s);
        int n = cur.n();
        VisionOps.Block h = ops.linear("mm.model.mlp.1.weight", cur, null);
        ops.activate(h, ActOp.GELU_TANH);
        h = ops.linear("mm.model.mlp.2.weight", h, null);
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.model.fc.weight", h, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

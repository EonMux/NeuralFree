package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;

/**
 * The SenseNova-U1 graph (models/sensenova-u1.cpp {@code clip_graph_sensenova_u1}): the biased patch conv, the
 * tanh GELU of {@code ggml_gelu}, then the 2D RoPE of {@code build_rope_2d} over one head as wide as the embedding
 * (column positions on the first half, row positions on the second, no frequency interleave, base 10000); the
 * projector is the biased 2x2 stride-2 conv {@code mm.0} over the rotated patch grid. The embedder has no
 * attention blocks. Written once against {@link VisionOps}. Package-private executor support.
 */
final class SenseNovaGraph {

    /** The per-side merge of the {@code mm.0} conv (the graph asserts {@code n_merge == 2}). */
    static final int MERGE = 2;

    /** RoPE base of the patch rows (clip.cpp load_hparams). */
    static final double ROPE_THETA = 10000.0;

    private SenseNovaGraph() {
    }

    /** The rotated patch rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        PixelShuffle.requireMultiple(pw, ph, MERGE, "sensenova_u1");
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        ops.activate(x, ActOp.GELU_TANH);
        int[] posX = new int[n];
        int[] posY = new int[n];
        for (int i = 0; i < n; i++) {
            posX[i] = i % pw;
            posY[i] = i / pw;
        }
        ops.rope(x, ops.hp().nHead(), VisionOps.RopePlan.twoHalves(posX, posY, ROPE_THETA, false));
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** The 2x2 stride-2 conv {@code mm.0} plus its bias: one row per merged window, raster order. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the sensenova_u1 projector");
        int nEmbd = ops.hp().nEmbd();
        int out = ops.model().outputDim();
        VisionOps.Block cols = ops.unfold(ops.upload(enc.rows, enc.width), nEmbd, enc.pw, enc.ph, MERGE);
        VisionOps.Block y = ops.linearKernel("mm.0.weight", out, nEmbd * MERGE * MERGE, cols, "mm.0.bias");
        return VisionRows.tokenRows(ops.download(y), out);
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The Kimi-VL graph (models/kimivl.cpp {@code clip_graph_kimivl}) and the Kimi K2.5 graph (models/kimik25.cpp
 * {@code clip_graph_kimik25}): the patch rows plus the learned positions resized to the patch grid, the
 * layer-norm ViT whose attention rotates Q and K with the 2D RoPE of {@code build_rope_2d} (column positions on
 * the first half of each head, row positions on the second, no frequency interleave, base 10000), then
 * {@code build_patch_merge_permute} by {@code clip.vision.projector.scale_factor} (zero padding to a multiple), a
 * layer norm over every {@code n_embd} chunk of the merged rows, and the GELU MLP {@code mm.1} / {@code mm.2}.
 * Kimi-VL stores its positions as a square grid resized with the antialiased bilinear filter and norms the
 * projector input at eps 1e-5; Kimi K2.5 stores a {@code [n_embd, W, H]} grid resized with the bicubic filter and
 * norms at the encoder's {@code layer_norm_epsilon}. Written once against {@link VisionOps}. Package-private
 * executor support.
 */
final class KimiVlGraph {

    /** RoPE base of the vision attention (clip.cpp load_hparams). */
    static final double ROPE_THETA = 10000.0;

    /** Kimi-VL projector layer-norm epsilon, PyTorch's {@code nn.LayerNorm} default. */
    static final float PROJECTOR_EPS = 1e-5f;

    private KimiVlGraph() {
    }

    /** The Kimi-VL encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        return encode(ops, image, "kimivl");
    }

    /**
     * The encoder rows of {@code image}, raster order, for the families sharing the tower ({@code family} names
     * the one whose head width is refused).
     */
    static Encoded encode(VisionOps ops, ClipImage image, String family) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        PixtralGraph.requireRopeHeadDim(ops.model().dHead(), "kimivl");
        return tower(ops, PaddleOcrGraph.resizedPatchRows(ops, image, grid[0], grid[1]), grid[0], grid[1]);
    }

    /**
     * The Kimi K2.5 encoder rows of {@code image}, raster order: the {@code [n_embd, W, H]} position grid is added
     * as stored when the patch grid is {@code W x H}, bicubic-resized to it otherwise
     * ({@code resize_position_embeddings_3d}).
     */
    static Encoded encodeK25(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        PixtralGraph.requireRopeHeadDim(ops.model().dHead(), "kimik25");
        int nEmbd = ops.hp().nEmbd();
        TensorRef posEmbd = ops.model().posEmbd();
        int srcW = (int) posEmbd.ne(1);
        int srcH = (int) posEmbd.ne(2);
        float[] pos = ops.hostWeight(posEmbd);
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        ops.addHost(x, srcW == pw && srcH == ph ? pos
                : Glm4vGraph.resizeBicubic(pos, nEmbd, srcW, srcH, pw, ph));
        return tower(ops, x, pw, ph);
    }

    private static Encoded tower(VisionOps ops, VisionOps.Block x, int pw, int ph) {
        int n = pw * ph;
        int[] posH = new int[n];
        int[] posW = new int[n];
        for (int i = 0; i < n; i++) {
            posH[i] = i / pw;
            posW[i] = i % pw;
        }
        VisionOps.Block out = ops.tower(x, n, NormKind.LAYER, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.twoHalves(posW, posH, ROPE_THETA, false)));
        return new Encoded(ops.download(out), n, pw, ph);
    }

    /** The Kimi-VL projector: the merge permute, the chunk-wise layer norm at eps 1e-5, then the GELU MLP. */
    static float[][] project(VisionOps ops, Encoded enc) {
        return project(ops, enc, PROJECTOR_EPS, "the kimivl projector");
    }

    /** The Kimi K2.5 projector: {@link #project} with the layer norm at the encoder's epsilon. */
    static float[][] projectK25(VisionOps ops, Encoded enc) {
        return project(ops, enc, ops.hp().eps(), "the kimik25 projector");
    }

    private static float[][] project(VisionOps ops, Encoded enc, float eps, String what) {
        enc.requireGrid(what);
        int nEmbd = ops.hp().nEmbd();
        int s = ops.hp().nMerge();
        VisionOps.Block cur = ops.mergePermute(ops.upload(enc.rows, enc.width), nEmbd, enc.pw, enc.ph, s);
        ops.norm(cur, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", eps,
                cur.n() * cur.width() / nEmbd, nEmbd);
        VisionOps.Block h = ops.linear("mm.1.weight", cur, "mm.1.bias");
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

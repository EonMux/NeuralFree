package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The PaddleOCR-VL graph (models/paddleocr.cpp {@code clip_graph_paddleocr}): the SigLIP patch rows in raster
 * order plus the learned position grid resized to the patch grid (antialiased bilinear), the layer-norm ViT with the
 * qwen vision m-rope over the raster (row, column) positions, then the mlp_AR projector: a layer norm at eps 1e-5,
 * {@code build_patch_merge_permute} by the fixed merge size 2 and the {@code mm.1} / {@code mm.2} MLP with the
 * encoder's activation. The patch grid must be a multiple of the merge size per side. Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class PaddleOcrGraph {

    /** Projector layer-norm epsilon of the reference graph. */
    static final float PROJECTOR_EPS = 1e-5f;

    private PaddleOcrGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        PixelShuffle.requireMultiple(pw, ph, ops.hp().nMerge(), "paddleocr");
        int n = pw * ph;
        VisionOps.Block x = resizedPatchRows(ops, image, pw, ph);
        x = ops.tower(x, n, NormKind.LAYER, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.qwen(rasterPositions(pw, ph))));
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** The raster patch rows plus the learned positions resized to the grid ({@code resize_position_embeddings}). */
    static VisionOps.Block resizedPatchRows(VisionOps ops, ClipImage image, int pw, int ph) {
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        TensorRef posEmbd = ops.model().posEmbd();
        float[] pos = ops.hostWeight(posEmbd);
        int side = SiglipResizedGraph.side(pos, nEmbd);
        float[] resized = side == pw && side == ph ? pos : SiglipResizedGraph.resize(pos, nEmbd, side, pw, ph);
        ops.addHost(x, resized);
        return x;
    }

    /** The qwen m-rope positions {@code [y | x | y | x]} of a raster grid ({@link QwenVisionGrid#ropePositions}). */
    static int[] rasterPositions(int pw, int ph) {
        int[] patches = new int[pw * ph];
        for (int t = 0; t < patches.length; t++) {
            patches[t] = t;
        }
        return QwenVisionGrid.ropePositions(patches, pw);
    }

    /** Layer norm, the merge permute, then {@code mm.1} + the encoder activation + {@code mm.2}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the paddleocr projector");
        int nEmbd = ops.hp().nEmbd();
        int s = ops.hp().nMerge();
        VisionOps.Block normed = ops.upload(enc.rows.clone(), enc.width);
        ops.norm(normed, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", PROJECTOR_EPS,
                enc.nTokens, nEmbd);
        VisionOps.Block cur = ops.mergePermute(normed, nEmbd, enc.pw, enc.ph, s);
        int n = cur.n();
        VisionOps.Block h = ops.linear("mm.1.weight", cur, "mm.1.bias");
        ops.activate(h, ops.hp().ffnOp().unary());
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

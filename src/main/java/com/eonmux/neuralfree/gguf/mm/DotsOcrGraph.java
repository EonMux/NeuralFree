package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The dots.ocr / dots3-note vision graph (models/dotsocr.cpp {@code clip_graph_dotsocr}): the raster patch rows
 * without learned positions, the RMS-norm ViT with the qwen vision m-rope over the raster (row, column) positions,
 * whose blocks are dense or routed-expert ({@code build_moe_ffn}: sigmoid router, top {@code expert_used_count}
 * experts, renormalized weights), the {@code mm.post_norm} RMS norm after the last block, then the merger: a layer
 * norm at eps 1e-6, {@code build_patch_merge_permute} by the merge size and the exact-GELU MLP {@code mm.0} /
 * {@code mm.2}. The patch grid must be a multiple of the merge size per side. Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class DotsOcrGraph {

    /** Merger layer-norm epsilon of the reference graph. */
    static final float PROJECTOR_EPS = 1e-6f;

    private DotsOcrGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        PixelShuffle.requireMultiple(pw, ph, ops.hp().nMerge(), "dots");
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        x = ops.tower(x, n, NormKind.RMS, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.qwen(PaddleOcrGraph.rasterPositions(pw, ph))));
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** Layer norm, the merge permute, then {@code mm.0} + exact GELU + {@code mm.2}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the dots merger");
        int nEmbd = ops.hp().nEmbd();
        int s = ops.hp().nMerge();
        VisionOps.Block normed = ops.upload(enc.rows.clone(), enc.width);
        ops.norm(normed, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", PROJECTOR_EPS,
                enc.nTokens, nEmbd);
        VisionOps.Block cur = ops.mergePermute(normed, nEmbd, enc.pw, enc.ph, s);
        int n = cur.n();
        VisionOps.Block h = ops.linear("mm.0.weight", cur, "mm.0.bias");
        ops.activate(h, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

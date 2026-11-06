package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The Youtu-VL graph (models/youtuvl.cpp {@code clip_graph_youtuvl}, a Siglip2 NaFlex tower): the patches in 2x2
 * merge-group order, each unfolded as {@code (ky, kx, c)} and projected by the linear {@code v.patch_embd}, a
 * layer-norm ViT with the 2D m-rope of the qwen encoders whose layers attend within the {@link QwenVisionGrid}
 * windows except the {@code clip.vision.wa_layer_indexes} layers (no windows when the list is empty), the post-norm
 * and the window order restored, then the patch merger: RMS norm ({@code mm.input_norm}), every 2x2 group as one
 * row, and the GELU MLP {@code mm.0} / {@code mm.2}. Written once against {@link VisionOps}. Package-private
 * executor support.
 */
final class YoutuVlGraph {

    /** The merge the graph's input reorder is written for. */
    static final int MERGE = 2;

    /** Epsilon of the merger's RMS norm. */
    static final float PROJECTOR_EPS = 1e-6f;

    private YoutuVlGraph() {
    }

    /** The encoder rows of {@code image} in merge-group order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        if (hp.nMerge() != MERGE) {
            throw new IllegalArgumentException("the youtuvl graph merges 2x2 patch groups; the file sets"
                    + " clip.vision.spatial_merge_size " + hp.nMerge());
        }
        int[] patches = ClipVit.grid(hp, image);
        int pw = patches[0];
        int ph = patches[1];
        if (pw % MERGE != 0 || ph % MERGE != 0) {
            throw new IllegalArgumentException("the youtuvl encoder needs a patch grid that is a multiple of 2 per"
                    + " side, got " + pw + "x" + ph);
        }
        int nEmbd = hp.nEmbd();
        VisionFamilyHparams fam = hp.visionFamily();
        QwenVisionGrid grid = QwenVisionGrid.of(hp, image.width(), image.height(),
                fam.fullAttnLayers().length > 0);
        int n = grid.nPatches();
        int block = nEmbd * grid.mergeArea();
        VisionOps.Block x = ops.patchEmbed(ClipPatches.mergeGroupsPixelMajor(image, hp.patchSize(), MERGE,
                grid.mergedWidth(), grid.mergedHeight()), n);
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.qwen(grid.positions()));
        if (grid.reorder()) {
            // every window a contiguous range of patch rows; the listed layers attend over the whole image
            x = ops.gatherRows(x, grid.invIdx(), block);
            opts = opts.withWindows(il -> fam.isFullAttnLayer(il) ? null : grid);
        }
        // the post-norm normalizes every row alone, so it commutes with the window restore
        x = ops.tower(x, n, NormKind.LAYER, opts);
        if (grid.reorder()) {
            x = ops.gatherRows(x, grid.idx(), block);
        }
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** RMS norm, every 2x2 group as one row, then {@code mm.0} + GELU + {@code mm.2}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block normed = ops.upload(enc.rows.clone(), enc.width);
        ops.norm(normed, NormKind.RMS, "mm.input_norm.weight", null, PROJECTOR_EPS, enc.nTokens, nEmbd);
        int n = enc.nTokens / (MERGE * MERGE);
        VisionOps.Block h = ops.linear("mm.0.weight", ops.view(normed, n, nEmbd * MERGE * MERGE), "mm.0.bias");
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

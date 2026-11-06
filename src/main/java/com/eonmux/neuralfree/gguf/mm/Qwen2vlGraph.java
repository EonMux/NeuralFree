package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The Qwen2-VL / Qwen2.5-VL graph (models/qwen2vl.cpp / qwen2.5vl.cpp), and the EXAONE 4.5 graph
 * (models/exaone4_5.cpp, the 2.5 graph with GQA fused Q/K/V and no patch bias): the patches of every 2x2 merge
 * group as one row, regrouped window by window when the 2.5 window attention is on, the tower with every norm in
 * the encoder's kind (RMS for 2.5 and EXAONE 4.5, layer for 2) and the qwen vision m-rope (each windowed layer
 * attends within its window's patch rows), and the merger {@code mm.0} + GELU + {@code mm.2} over the merged rows;
 * the window order is restored after the merger. Written once against {@link VisionOps}. Package-private executor support.
 */
final class Qwen2vlGraph {

    /** The merge exaone4_5.cpp and its position fill hard-code. */
    static final int EXAONE_MERGE = 2;

    private Qwen2vlGraph() {
    }

    /** The embedding rows of {@code image}, raster order over the merged tokens. */
    static float[][] encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        if (hp.projectorType() == ProjectorType.EXAONE4_5 && hp.nMerge() != EXAONE_MERGE) {
            throw new IllegalArgumentException("the exaone4_5 graph merges 2x2 patches, the file sets"
                    + " clip.vision.spatial_merge_size " + hp.nMerge());
        }
        QwenVisionGrid grid = QwenVisionGrid.of(hp, image.width(), image.height());
        int nEmbd = hp.nEmbd();
        int nPatches = grid.nPatches();
        VisionOps.Block x = ops.patchEmbed(ClipPatches.mergeGroups(image, hp.patchSize(), hp.nMerge(),
                grid.mergedWidth(), grid.mergedHeight()), nPatches);
        if (grid.reorder()) {
            // the graph reorders merged tokens, each a block of n_merge^2 contiguous patch rows
            x = ops.gatherRows(x, grid.invIdx(), nEmbd * grid.mergeArea());
        }
        // qwen2vl.cpp: every norm, the pre-norm included, is RMS for 2.5 (and EXAONE 4.5) and layer norm for 2
        boolean rms = hp.projectorType() == ProjectorType.QWEN25VL || hp.projectorType() == ProjectorType.EXAONE4_5;
        NormKind norm = rms ? NormKind.RMS : NormKind.LAYER;
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.qwen(grid.positions()));
        if (grid.reorder()) {
            // window attention in every layer but each n_wa_pattern-th, which attends over the whole image
            opts = opts.withWindows(il -> grid.windowed(il, hp.nWaPattern()) ? grid : null);
        }
        x = ops.tower(x, nPatches, norm, opts);
        VisionOps.Block out = merger(ops, ops.view(x, grid.nTokens(), nEmbd * grid.mergeArea()), grid.nTokens());
        int d = ops.model().outputDim();
        if (grid.reorder()) {
            out = ops.gatherRows(out, grid.idx(), d);
        }
        return VisionRows.tokenRows(ops.download(out), d);
    }

    /** Pixel shuffle of the merge-grouped rows (each group already contiguous), then the mm.0 / mm.2 MLP. */
    static VisionOps.Block merger(VisionOps ops, VisionOps.Block x, int nTokens) {
        VisionOps.Block h = ops.linear("mm.0.weight", x, "mm.0.bias");
        ops.activate(h, ActOp.GELU_TANH);
        return ops.linear("mm.2.weight", h, "mm.2.bias");
    }
}

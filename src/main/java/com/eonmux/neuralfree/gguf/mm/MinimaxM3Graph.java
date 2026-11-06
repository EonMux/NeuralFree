package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.Arrays;

/**
 * The MiniMax-M3 graph (models/minimax-m3.cpp {@code clip_graph_minimax_m3}): both patch kernels over the still
 * image, summed, in merge-group token order ({@code n_merge x n_merge} windows, each window's rows contiguous); the
 * layer-norm ViT with exact-GELU FFNs, Q and K rotated by two NEOX ropes (base 10000) over the
 * {@code [t | h | w | pad]} split of every head, the row position on the {@code h} third and the column position
 * on the {@code w} third, the time third and the padding unrotated; then the projector: {@code mm.1} + GELU-erf +
 * {@code mm.2} per patch, every window's rows concatenated, and {@code mm.merger.fc1} + GELU-erf +
 * {@code mm.merger.fc2}. Written once against {@link VisionOps}. Package-private executor support.
 */
final class MinimaxM3Graph {

    /** {@code vision_config.rope_theta} of the reference loader. */
    static final float ROPE_FREQ_BASE = 10000f;

    private MinimaxM3Graph() {
    }

    /**
     * The embedding rows of {@code image}: {@code (pw / m) * (ph / m)} rows of the projector output width.
     *
     * @throws IllegalArgumentException when the patch grid is not a multiple of the merge size per side, or the
     *                                  head is too narrow for the three rotated thirds (named)
     */
    static float[][] encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        int m = hp.nMerge();
        if (pw % m != 0 || ph % m != 0) {
            throw new IllegalArgumentException("the minimax_m3 encoder merges " + m + "x" + m + " patches, the patch"
                    + " grid " + pw + "x" + ph + " must be a multiple of " + m + " per side");
        }
        int nEmbd = hp.nEmbd();
        int n = pw * ph;
        int[] order = Qwen3VlGraph.mergeOrder(pw, ph, m);
        VisionOps.Block x = ops.gatherRows(ops.patchRows(image, pw, ph), order, nEmbd);
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE.withRope(rope(ops.model().dHead(), order, pw));
        x = ops.tower(x, n, NormKind.LAYER, opts);
        VisionOps.Block h = ops.linear("mm.1.weight", x, "mm.1.bias");
        ops.activate(h, ActOp.GELU_ERF);
        VisionOps.Block patches = ops.linear("mm.2.weight", h, "mm.2.bias");
        int groups = n / (m * m);
        VisionOps.Block merged = ops.view(patches, groups, patches.width() * m * m);
        VisionOps.Block f = ops.linear("mm.merger.fc1.weight", merged, "mm.merger.fc1.bias");
        ops.activate(f, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.merger.fc2.weight", f, "mm.merger.fc2.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /**
     * The two NEOX segments of {@code apply_rope}: {@code axd = 2 * floor(floor(2 * floor(d_head / 2) / 3) / 2)}
     * dims each, the row positions on {@code [axd, 2 axd)} and the column positions on {@code [2 axd, 3 axd)}.
     */
    static VisionOps.RopePlan rope(int dHead, int[] order, int pw) {
        int axd = 2 * ((2 * (dHead / 2) / 3) / 2);
        if (axd <= 0 || 3 * axd > dHead) {
            throw new IllegalArgumentException("the minimax_m3 rope splits a head into three rotated thirds, head"
                    + " width " + dHead + " leaves none");
        }
        int[] row = new int[order.length];
        int[] col = new int[order.length];
        for (int t = 0; t < order.length; t++) {
            row[t] = order[t] / pw;
            col[t] = order[t] % pw;
        }
        return VisionOps.RopePlan.neox(Arrays.asList(new VisionOps.NeoxSegment(axd, axd, row),
                new VisionOps.NeoxSegment(2 * axd, axd, col)), ROPE_FREQ_BASE);
    }
}

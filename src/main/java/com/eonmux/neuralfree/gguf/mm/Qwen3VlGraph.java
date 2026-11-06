package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.ArrayList;
import java.util.List;

/**
 * The Qwen3-VL graph (models/qwen3vl.cpp {@code clip_graph_qwen3vl}): both patch kernels over the still image plus
 * the patch bias, in merge-group token order (2x2 windows, each window's four rows contiguous); the learned
 * position grid resized to the patch grid by the align-corners bilinear kernel and reordered the same way; the
 * layer-norm ViT with the qwen vision m-rope; after every block that carries a deepstack merger, that block's
 * output merged 2x2 and run through the deepstack layer norm and GELU MLP; the post-norm; then the GELU merger over
 * the 2x2-merged rows. Every output row is the merger row followed by the deepstack rows in layer order
 * ({@code ggml_concat} along the features). The Ling3-VL graph (models/ling3vl.cpp {@code clip_graph_ling3vl})
 * runs the same tower (the patch bias optional) and, instead of the deepstack mergers, layer-norms every patch row
 * with {@code mm.input_norm} before the merger. Written once against {@link VisionOps}. Package-private executor
 * support.
 */
final class Qwen3VlGraph {

    /** The merge window side the graph's reshapes hard-code. */
    static final int MERGE = 2;

    private Qwen3VlGraph() {
    }

    /** The embedding rows of {@code image}: {@code (pw / 2) * (ph / 2)} rows of the projector output width. */
    static float[][] encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        if (pw % MERGE != 0 || ph % MERGE != 0) {
            throw new IllegalArgumentException("the qwen3vl encoder merges 2x2 patches, the patch grid " + pw + "x"
                    + ph + " must be even per side");
        }
        int nEmbd = hp.nEmbd();
        int n = pw * ph;
        int[] order = mergeOrder(pw, ph, MERGE);
        VisionOps.Block x = ops.gatherRows(ops.patchRows(image, pw, ph), order, nEmbd);
        float[] pos = SiglipResizedGraph.alignCorners(ops.hostWeight(ops.model().posEmbd()), nEmbd,
                SiglipResizedGraph.side(ops.hostWeight(ops.model().posEmbd()), nEmbd), pw, ph);
        ops.addHost(x, VisionRows.gatherRows(pos, order, nEmbd));
        int nMerged = n / (MERGE * MERGE);
        List<VisionOps.Block> deepstack = new ArrayList<>();
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.qwen(QwenVisionGrid.ropePositions(order, pw)))
                .withTap((done, out, nTokens) -> {
                    if (done > 0 && hasDeepstack(ops, done - 1)) {
                        deepstack.add(deepstack(ops, done - 1, out, nMerged));
                    }
                });
        x = ops.tower(x, n, NormKind.LAYER, opts);
        if (ProjectorEntry.PROJECTOR_LING3VL.equals(ops.model().projectorGraph())) {
            // ling3vl.cpp: the merger norm per patch row before the 2x2 merge, no deepstack
            ops.norm(x, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", hp.eps(), n, nEmbd);
        }
        VisionOps.Block merged = ops.linear("mm.0.weight", ops.view(x, nMerged, MERGE * MERGE * nEmbd), "mm.0.bias");
        ops.activate(merged, ActOp.GELU_TANH);
        merged = ops.linear("mm.2.weight", merged, "mm.2.bias");
        return concat(ops, merged, deepstack, nMerged);
    }

    /**
     * The raster patch index of every token in merge-group order ({@code ty, tx, dy, dx}, the qwen m-rope position
     * fill of clip.cpp and the permute chain of the graph).
     */
    static int[] mergeOrder(int pw, int ph, int s) {
        int[] order = new int[pw * ph];
        int t = 0;
        for (int ty = 0; ty < ph / s; ty++) {
            for (int tx = 0; tx < pw / s; tx++) {
                for (int dy = 0; dy < s; dy++) {
                    for (int dx = 0; dx < s; dx++) {
                        order[t++] = (ty * s + dy) * pw + tx * s + dx;
                    }
                }
            }
        }
        return order;
    }

    private static boolean hasDeepstack(VisionOps ops, int il) {
        return ops.hasMatrix("v.deepstack." + il + ".fc1.weight");
    }

    /** Block {@code il}'s rows viewed as {@code nMerged} rows of {@code 4 * n_embd}: layer norm, then the GELU MLP. */
    private static VisionOps.Block deepstack(VisionOps ops, int il, VisionOps.Block rows, int nMerged) {
        String p = "v.deepstack." + il + ".";
        int width = MERGE * MERGE * ops.hp().nEmbd();
        VisionOps.Block normed = ops.upload(ops.download(rows).clone(), width);
        ops.norm(normed, NormKind.LAYER, p + "norm.weight", p + "norm.bias", ops.hp().eps(), nMerged, width);
        VisionOps.Block h = ops.linear(p + "fc1.weight", normed, p + "fc1.bias");
        ops.activate(h, ActOp.GELU_TANH);
        return ops.linear(p + "fc2.weight", h, p + "fc2.bias");
    }

    /** Row {@code t}: the merger row, then each deepstack row, every block {@code width / (1 + deepstack)} wide. */
    private static float[][] concat(VisionOps ops, VisionOps.Block merged, List<VisionOps.Block> deepstack,
                                    int nMerged) {
        int width = ops.model().outputDim();
        int part = width / (1 + deepstack.size());
        float[] merger = ops.download(merged);
        float[][] out = new float[nMerged][width];
        for (int t = 0; t < nMerged; t++) {
            System.arraycopy(merger, t * part, out[t], 0, part);
        }
        for (int i = 0; i < deepstack.size(); i++) {
            float[] values = ops.download(deepstack.get(i));
            for (int t = 0; t < nMerged; t++) {
                System.arraycopy(values, t * part, out[t], (i + 1) * part, part);
            }
        }
        return out;
    }
}

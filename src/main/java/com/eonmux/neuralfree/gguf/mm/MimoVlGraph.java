package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The MiMo-VL graph (models/mimovl.cpp {@code clip_graph_mimovl}): the two patch convolutions summed over the
 * patches in 2x2 merge-group order, then the RMS-norm blocks in runs of one attention mode
 * ({@code clip.vision.wa_pattern_mode}): a full-attention block attends over every row; a windowed block attends to
 * the rows at most {@code clip.vision.window_size} away in the current order, with its per-head attention sinks. A
 * column-window run regroups the merge units column-major on entry and restores the row-major order after it; Q
 * and K rotate with the qwen 2D m-rope at the positions of the current order. Then the layer post-norm, every 2x2
 * group as one row, and the bias-free merger {@code mm.0} + GELU + {@code mm.2}. Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class MimoVlGraph {

    /** The merge the patch order is written for (the graph's patch reshape fixes it at 2). */
    static final int MERGE = 2;

    /** Epsilon of the post-norm, fixed by the graph. */
    static final float POST_NORM_EPS = 1e-6f;

    private static final int UNIT = MERGE * MERGE;

    private MimoVlGraph() {
    }

    /** The encoder rows of {@code image}, the patches in merge-group order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        if (pw % MERGE != 0 || ph % MERGE != 0) {
            throw new IllegalArgumentException("the mimovl encoder needs a patch grid that is a multiple of 2 per"
                    + " side, got " + pw + "x" + ph);
        }
        int mw = pw / MERGE;
        int mh = ph / MERGE;
        int n = pw * ph;
        int nEmbd = hp.nEmbd();
        int block = nEmbd * UNIT;
        int[] rowPatches = mergeGroupPatches(mw, mh);
        int[] toCol = columnUnits(mw, mh);
        int[] toRow = inverse(toCol);
        int[] rowPos = QwenVisionGrid.ropePositions(rowPatches, pw);
        int[] colPos = QwenVisionGrid.ropePositions(permuteUnits(rowPatches, toCol), pw);
        VisionFamilyHparams fam = hp.visionFamily();
        VisionRows.KeyRanges band = band(fam.windowRadius(), n);

        VisionOps.Block x = ops.patchEmbed(ClipPatches.mergeGroups(image, hp.patchSize(), MERGE, mw, mh), n);
        int prev = VisionFamilyHparams.MODE_FULL;
        int from = 0;
        while (from < hp.nLayer()) {
            int mode = fam.attnMode(from);
            int to = from + 1;
            while (to < hp.nLayer() && fam.attnMode(to) == mode) {
                to++;
            }
            if (mode == VisionFamilyHparams.MODE_COL && prev != VisionFamilyHparams.MODE_COL) {
                x = ops.gatherRows(x, toCol, block);
            } else if (mode != VisionFamilyHparams.MODE_COL && prev == VisionFamilyHparams.MODE_COL) {
                x = ops.gatherRows(x, toRow, block);
            }
            VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE.withBlocks(from, to)
                    .withRope(VisionOps.RopePlan.qwen(mode == VisionFamilyHparams.MODE_COL ? colPos : rowPos));
            if (mode != VisionFamilyHparams.MODE_FULL) {
                opts = opts.withWindows(il -> band).withSinks();
            }
            x = ops.tower(x, n, NormKind.RMS, opts);
            prev = mode;
            from = to;
        }
        if (prev == VisionFamilyHparams.MODE_COL) {
            x = ops.gatherRows(x, toRow, block);
        }
        ClipVisionModel m = ops.model();
        ops.normByRef(x, NormKind.LAYER, m.postLnWeight(), m.postLnBias(), POST_NORM_EPS, n, nEmbd);
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** Every 2x2 group of rows as one row, then the bias-free {@code mm.0} + GELU + {@code mm.2}. */
    static float[][] project(VisionOps ops, Encoded enc) {
        int n = enc.nTokens / UNIT;
        VisionOps.Block x = ops.view(ops.upload(enc.rows, enc.width), n, enc.width * UNIT);
        VisionOps.Block h = ops.linear("mm.0.weight", x, null);
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.2.weight", h, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /** The raster patch of every row in merge-group order {@code (ty, tx, dy, dx)} over {@code mw x mh} groups. */
    static int[] mergeGroupPatches(int mw, int mh) {
        int pw = mw * MERGE;
        int[] patches = new int[mw * mh * UNIT];
        int t = 0;
        for (int ty = 0; ty < mh; ty++) {
            for (int tx = 0; tx < mw; tx++) {
                for (int dy = 0; dy < MERGE; dy++) {
                    for (int dx = 0; dx < MERGE; dx++) {
                        patches[t++] = (ty * MERGE + dy) * pw + tx * MERGE + dx;
                    }
                }
            }
        }
        return patches;
    }

    /**
     * The column-major merge-unit order ({@code idx_col} of clip.cpp set_input): unit {@code c * mh + r} of the
     * column order is unit {@code r * mw + c} of the row order.
     */
    static int[] columnUnits(int mw, int mh) {
        int[] idx = new int[mw * mh];
        for (int r = 0; r < mh; r++) {
            for (int c = 0; c < mw; c++) {
                idx[c * mh + r] = r * mw + c;
            }
        }
        return idx;
    }

    /** The rows of {@code values}, one per patch, regrouped unit by unit: unit {@code u} takes unit {@code idx[u]}. */
    static int[] permuteUnits(int[] values, int[] idx) {
        int[] out = new int[values.length];
        for (int u = 0; u < idx.length; u++) {
            System.arraycopy(values, idx[u] * UNIT, out, u * UNIT, UNIT);
        }
        return out;
    }

    private static int[] inverse(int[] idx) {
        int[] inv = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            inv[idx[i]] = i;
        }
        return inv;
    }

    /** The banded window: row {@code q} attends to the rows {@code k} with {@code |q - k| <= radius}. */
    static VisionRows.KeyRanges band(int radius, int n) {
        return new VisionRows.KeyRanges() {
            @Override
            public int begin(int row) {
                return Math.max(0, row - radius);
            }

            @Override
            public int end(int row) {
                return (int) Math.min(n, (long) row + radius + 1);
            }
        };
    }
}

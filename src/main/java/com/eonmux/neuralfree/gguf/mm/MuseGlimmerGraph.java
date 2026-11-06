package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The Muse Glimmer graph (models/muse-glimmer.cpp {@code clip_graph_muse_glimmer}): the patch conv plus the
 * optional patch bias and the position grid resized by the half-pixel bilinear kernel; the rows regrouped window
 * by window ({@code side x side} windows of the patch grid, {@code side} the position grid's side, row-major, the
 * edge windows partial); the layer-norm ViT with exact-GELU FFNs and {@code build_rope_2d} (the 1-based column on
 * the first head half, the row on the second, base 10000), every block attending within its window except every
 * fourth block and the last, which attend over the whole image; the raster order restored; then the projector:
 * every {@code n_merge x n_merge} window unfolded channel-outer and {@code mm.0} + GELU-erf + {@code mm.1} +
 * GELU-erf + {@code mm.2}, all bias-free. Written once against {@link VisionOps}. Package-private executor support.
 */
final class MuseGlimmerGraph {

    /** {@code rope_theta} of the reference loader. */
    static final double ROPE_FREQ_BASE = 10000.0;
    /** {@code muse_glimmer_sparse_factor}: every fourth block attends globally. */
    static final int SPARSE_FACTOR = 4;

    private MuseGlimmerGraph() {
    }

    /** The window grouping of a patch grid: the row order, its inverse and every row's key range. */
    static final class Windows implements VisionRows.KeyRanges {
        final int[] perm;
        final int[] inv;
        private final int[] begin;
        private final int[] end;

        private Windows(int[] perm, int[] inv, int[] begin, int[] end) {
            this.perm = perm;
            this.inv = inv;
            this.begin = begin;
            this.end = end;
        }

        @Override
        public int begin(int row) {
            return begin[row];
        }

        @Override
        public int end(int row) {
            return end[row];
        }

        /** clip.cpp {@code set_input}: {@code win x win} windows in row-major order, each its patches row-major. */
        static Windows of(int pw, int ph, int win) {
            int n = pw * ph;
            int[] perm = new int[n];
            int[] inv = new int[n];
            int[] begin = new int[n];
            int[] end = new int[n];
            int t = 0;
            for (int wy = 0; wy < (ph + win - 1) / win; wy++) {
                for (int wx = 0; wx < (pw + win - 1) / win; wx++) {
                    int start = t;
                    for (int y = wy * win; y < Math.min(ph, (wy + 1) * win); y++) {
                        for (int x = wx * win; x < Math.min(pw, (wx + 1) * win); x++) {
                            perm[t] = y * pw + x;
                            inv[y * pw + x] = t;
                            t++;
                        }
                    }
                    for (int r = start; r < t; r++) {
                        begin[r] = start;
                        end[r] = t;
                    }
                }
            }
            return new Windows(perm, inv, begin, end);
        }
    }

    /**
     * The encoder rows of {@code image}: {@code pw * ph} rows in raster order.
     *
     * @throws IllegalArgumentException when the patch grid is not a multiple of the merge size per side (named)
     */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        int ds = hp.nMerge();
        if (pw % ds != 0 || ph % ds != 0) {
            throw new IllegalArgumentException("the muse-glimmer projector shuffles " + ds + "x" + ds + " patches,"
                    + " the patch grid " + pw + "x" + ph + " must be a multiple of " + ds + " per side");
        }
        int nEmbd = hp.nEmbd();
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        float[] pos = ops.hostWeight(ops.model().posEmbd());
        int side = SiglipResizedGraph.side(pos, nEmbd);
        ops.addHost(x, SiglipResizedGraph.halfPixel(pos, nEmbd, side, pw, ph));
        Windows windows = Windows.of(pw, ph, side);
        x = ops.gatherRows(x, windows.perm, nEmbd);
        int[] col = new int[n];
        int[] row = new int[n];
        for (int t = 0; t < n; t++) {
            col[t] = windows.perm[t] % pw + 1;
            row[t] = windows.perm[t] / pw + 1;
        }
        int nLayer = hp.nLayer();
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.twoHalves(col, row, ROPE_FREQ_BASE, false))
                .withWindows(il -> global(il, nLayer) ? null : windows);
        x = ops.tower(x, n, NormKind.LAYER, opts);
        return new Encoded(ops.download(ops.gatherRows(x, windows.inv, nEmbd)), n, pw, ph);
    }

    /** True for the blocks that attend over the whole image: every fourth and the last. */
    static boolean global(int il, int nLayer) {
        return il == nLayer - 1 || (il + 1) % SPARSE_FACTOR == 0;
    }

    /** The projector: {@code (pw / n_merge) * (ph / n_merge)} rows of the output width. */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the muse-glimmer projector");
        int ds = ops.hp().nMerge();
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.unfoldWindows(ops.upload(enc.rows, enc.width), nEmbd, enc.pw, enc.pw / ds,
                enc.ph / ds, ds);
        VisionOps.Block h = ops.linear("mm.0.weight", x, null);
        ops.activate(h, ActOp.GELU_ERF);
        h = ops.linear("mm.1.weight", h, null);
        ops.activate(h, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.2.weight", h, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

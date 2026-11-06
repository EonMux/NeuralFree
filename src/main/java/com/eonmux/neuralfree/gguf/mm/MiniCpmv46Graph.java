package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The MiniCPM-V 4.6 graph of the minicpmv4_6, minicpmv4_7 and minicpm_ocr_1_0 projector types (models/minicpmv.cpp
 * {@code clip_graph_minicpmv4_6}). The encoder: the patch convolution with its bias, the learned position rows
 * gathered from the 70x70 bucket table, the optional pre-norm, then the layer-norm tower. With a 4x merge
 * ({@code n_merge == 4}) the ViT merger runs after block {@code insert_layer_id}: a layer norm, attention inside
 * every 2x2 patch window (the tokens reordered window-major, each group of four attending to itself) with the
 * residual, then the 2x2 downsample (the four neighbors concatenated, layer-normed and run through
 * {@code ds_ffn_up} + tanh GELU + {@code ds_ffn_down}, plus their mean); the remaining blocks run on the halved grid.
 * With a 2x merge every block runs on the full grid. The optional post-norm closes the tower. The projector is the
 * final 2x2 merger (the four neighbors concatenated, {@code mm.input_norm}, {@code mm.up} + erf GELU +
 * {@code mm.down}); the separator rows of the image's suffix follow ({@code build_suffix}). Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class MiniCpmv46Graph {

    /** The separator rows the suffixes read ({@link ClipSuffix}). */
    static final String SEPARATORS = "v.tok_embd_sep";
    static final String MERGER_LN1 = "v.vit_merger.ln1";
    static final String MERGER_ATTN = "v.vit_merger.attn_";
    static final String MERGER_DS_LN = "v.vit_merger.ds_ln";
    static final String MERGER_DS_UP = "v.vit_merger.ds_ffn_up";
    static final String MERGER_DS_DOWN = "v.vit_merger.ds_ffn_down";
    /** The ViT merger's attention projections, {@code attn_<p>.weight} and the optional {@code .bias}. */
    static final String[] MERGER_PROJECTIONS = {"q", "k", "v", "out"};
    /** Every ViT merger tensor (clip.cpp {@code load_tensors}), required with a 4x merge. */
    static final String[] MERGER_TENSORS = mergerTensors();

    /** Tokens per ViT merger attention window (a 2x2 patch square). */
    private static final int WINDOW = 4;
    /** The {@code (row, column)} offsets of the four 2x2 neighbors, in concatenation order. */
    private static final int[][] NEIGHBORS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    private MiniCpmv46Graph() {
    }

    private static String[] mergerTensors() {
        List<String> names = new ArrayList<>();
        for (String norm : new String[]{MERGER_LN1, MERGER_DS_LN}) {
            names.add(norm + ".weight");
            names.add(norm + ".bias");
        }
        for (String p : MERGER_PROJECTIONS) {
            names.add(MERGER_ATTN + p + ".weight");
            names.add(MERGER_ATTN + p + ".bias");
        }
        for (String ffn : new String[]{MERGER_DS_UP, MERGER_DS_DOWN}) {
            names.add(ffn + ".weight");
            names.add(ffn + ".bias");
        }
        return names.toArray(new String[0]);
    }

    // ------------------------------------------------------------------ encoder

    /**
     * The encoder rows of {@code image}: the halved grid after the ViT merger with a 4x merge, else the patch grid,
     * raster order.
     *
     * @throws IllegalArgumentException when the patch grid's sides are not multiples of the merge
     */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipHparams hp = ops.hp();
        int[] grid = ClipVit.grid(hp, image);
        int pw = grid[0];
        int ph = grid[1];
        int merge = hp.nMerge();
        if (pw % merge != 0 || ph % merge != 0) {
            throw new IllegalArgumentException("the " + hp.projectorType() + " mergers need a patch grid whose"
                    + " sides are multiples of " + merge + " (clip.vision.projector.scale_factor), got " + pw + "x"
                    + ph);
        }
        int n = pw * ph;
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        MiniCpmvGraph.addBucketPositions(ops, x, pw, ph);
        if (merge != 4) {
            x = ops.tower(x, n, NormKind.LAYER, VisionOps.TowerOptions.NONE);
            return new Encoded(ops.download(x), n, pw, ph);
        }
        ClipVisionModel m = ops.model();
        int nEmbd = hp.nEmbd();
        if (m.preLnWeight() != null) {
            ops.normByRef(x, NormKind.LAYER, m.preLnWeight(), m.preLnBias(), hp.eps(), n, nEmbd);
        }
        int insert = hp.visionFamily().insertLayerId();
        x = ops.tower(x, n, NormKind.LAYER, VisionOps.TowerOptions.NONE.withBlocks(0, insert + 1));
        x = vitMerger(ops, x, pw, ph);
        int hw = pw / 2;
        int hh = ph / 2;
        x = ops.tower(x, hw * hh, NormKind.LAYER, VisionOps.TowerOptions.NONE.withBlocks(insert + 1, hp.nLayer()));
        if (m.postLnWeight() != null) {
            ops.normByRef(x, NormKind.LAYER, m.postLnWeight(), m.postLnBias(), hp.eps(), hw * hh, nEmbd);
        }
        return new Encoded(ops.download(x), hw * hh, hw, hh);
    }

    /** The ViT merger over the {@code pw x ph} stream: window attention with its residual, then the downsample. */
    private static VisionOps.Block vitMerger(VisionOps ops, VisionOps.Block x, int pw, int ph) {
        ClipHparams hp = ops.hp();
        int nEmbd = hp.nEmbd();
        int n = pw * ph;
        int[] order = windowOrder(pw, ph);
        VisionOps.Block cur = ops.gatherRows(x, order, nEmbd);
        ops.norm(cur, NormKind.LAYER, MERGER_LN1 + ".weight", MERGER_LN1 + ".bias", hp.eps(), n, nEmbd);
        VisionOps.Block q = ops.linear(MERGER_ATTN + "q.weight", cur, MERGER_ATTN + "q.bias");
        VisionOps.Block k = ops.linear(MERGER_ATTN + "k.weight", cur, MERGER_ATTN + "k.bias");
        VisionOps.Block v = ops.linear(MERGER_ATTN + "v.weight", cur, MERGER_ATTN + "v.bias");
        int dh = ops.model().dHead();
        float scale = (float) (1.0 / Math.sqrt(dh));
        VisionOps.Block attn = ops.attendWindows(q, k, v, n, nEmbd, hp.nHead(), 1, dh, scale, WINDOWS);
        attn = ops.linear(MERGER_ATTN + "out.weight", attn, MERGER_ATTN + "out.bias");
        VisionOps.Block stream = ops.gatherRows(attn, inverse(order), nEmbd);
        ops.addBlock(stream, x);
        return downsample(ops, stream, pw, ph);
    }

    /**
     * The ViT merger's 2x2 downsample to the {@code (pw / 2) x (ph / 2)} grid: the four neighbors concatenated,
     * layer-normed, {@code ds_ffn_up} + tanh GELU + {@code ds_ffn_down}, plus the neighbors' mean.
     */
    private static VisionOps.Block downsample(VisionOps ops, VisionOps.Block x, int pw, int ph) {
        ClipHparams hp = ops.hp();
        int nEmbd = hp.nEmbd();
        int hw = pw / 2;
        int hh = ph / 2;
        int nds = hw * hh;
        List<VisionOps.Block> parts = neighbors(ops, x, hw, hh, pw, nEmbd);
        VisionOps.Block mean = ops.gatherRows(x, neighborRows(0, 0, hh, hw, pw), nEmbd);
        for (int i = 1; i < parts.size(); i++) {
            ops.addBlock(mean, parts.get(i));
        }
        float[] quarter = new float[nEmbd];
        Arrays.fill(quarter, 0.25f);
        ops.mulRowScale(mean, quarter);
        VisionOps.Block cat = ops.concatChannels(parts, nds, nEmbd);
        ops.norm(cat, NormKind.LAYER, MERGER_DS_LN + ".weight", MERGER_DS_LN + ".bias", hp.eps(), nds,
                NEIGHBORS.length * nEmbd);
        VisionOps.Block up = ops.linear(MERGER_DS_UP + ".weight", cat, MERGER_DS_UP + ".bias");
        ops.activate(up, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear(MERGER_DS_DOWN + ".weight", up, MERGER_DS_DOWN + ".bias");
        ops.addBlock(out, mean);
        return out;
    }

    // ------------------------------------------------------------------ projector

    /**
     * The final merger over the encoder grid: the four 2x2 neighbors concatenated, {@code mm.input_norm},
     * {@code mm.up} + erf GELU + {@code mm.down}; {@code (pw / 2) x (ph / 2)} rows, raster order.
     */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the " + ops.hp().projectorType() + " merger");
        int width = enc.width;
        int hw = enc.pw / 2;
        int hh = enc.ph / 2;
        int n = hw * hh;
        VisionOps.Block x = ops.upload(enc.rows, width);
        VisionOps.Block cat = ops.concatChannels(neighbors(ops, x, hw, hh, enc.pw, width), n, width);
        ops.norm(cat, NormKind.LAYER, "mm.input_norm.weight", "mm.input_norm.bias", ops.hp().eps(), n,
                NEIGHBORS.length * width);
        VisionOps.Block up = ops.linear("mm.up.weight", cat, "mm.up.bias");
        ops.activate(up, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.down.weight", up, "mm.down.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /**
     * The {@link #SEPARATORS} rows of {@code suffix}, in order: what {@code build_suffix} appends after the image
     * tokens.
     *
     * @throws IllegalArgumentException when the encoder is not a MiniCPM-V 4.6 merger graph, the file carries no
     *                                  separator rows, or a suffix row is beyond them (named)
     */
    static float[][] suffixRows(VisionOps ops, ClipSuffix suffix) {
        if (!ProjectorEntry.PROJECTOR_MINICPMV46_MERGER.equals(ops.model().projectorGraph())) {
            throw new IllegalArgumentException("separator rows (" + suffix + ") follow the image tokens of the"
                    + " minicpmv4_6, minicpmv4_7 and minicpm_ocr_1_0 mergers only; this encoder is "
                    + ops.hp().projectorType());
        }
        float[] table = ops.hostVectorOrNull(SEPARATORS);
        if (table == null) {
            throw new IllegalArgumentException("the image asks for the separator rows " + suffix + ", but the file"
                    + " carries no " + SEPARATORS);
        }
        int width = ops.model().outputDim();
        int nSep = table.length / width;
        float[][] out = new float[suffix.size()][];
        int at = 0;
        for (int idx : suffix.rows()) {
            if (idx >= nSep) {
                throw new IllegalArgumentException("the separator suffix " + suffix + " reads row " + idx + " of "
                        + SEPARATORS + ", which holds " + nSep + " rows");
            }
            out[at++] = Arrays.copyOfRange(table, idx * width, (idx + 1) * width);
        }
        return out;
    }

    // ------------------------------------------------------------------ index tables

    /** The four 2x2 neighbors of every cell of the {@code hw x hh} grid over a {@code stride}-wide stream. */
    private static List<VisionOps.Block> neighbors(VisionOps ops, VisionOps.Block x, int hw, int hh, int stride,
                                                   int width) {
        List<VisionOps.Block> parts = new ArrayList<>(NEIGHBORS.length);
        for (int[] off : NEIGHBORS) {
            parts.add(ops.gatherRows(x, neighborRows(off[0], off[1], hh, hw, stride), width));
        }
        return parts;
    }

    /** {@code make_ds_idx}: row {@code (2i + offR) * stride + 2j + offC} for every cell {@code (i, j)}. */
    static int[] neighborRows(int offR, int offC, int dsH, int dsW, int stride) {
        int[] idx = new int[dsH * dsW];
        for (int i = 0; i < dsH; i++) {
            for (int j = 0; j < dsW; j++) {
                idx[i * dsW + j] = (2 * i + offR) * stride + 2 * j + offC;
            }
        }
        return idx;
    }

    /** The window-major order of the {@code pw x ph} patches: the four patches of each 2x2 window in turn. */
    static int[] windowOrder(int pw, int ph) {
        int[] idx = new int[pw * ph];
        int k = 0;
        for (int wi = 0; wi < ph / 2; wi++) {
            for (int wj = 0; wj < pw / 2; wj++) {
                for (int[] off : NEIGHBORS) {
                    idx[k++] = (2 * wi + off[0]) * pw + 2 * wj + off[1];
                }
            }
        }
        return idx;
    }

    private static int[] inverse(int[] order) {
        int[] inv = new int[order.length];
        for (int i = 0; i < order.length; i++) {
            inv[order[i]] = i;
        }
        return inv;
    }

    /** Every window-major row attends to the four rows of its window. */
    private static final VisionRows.KeyRanges WINDOWS = new VisionRows.KeyRanges() {
        @Override
        public int begin(int row) {
            return row - row % WINDOW;
        }

        @Override
        public int end(int row) {
            return row - row % WINDOW + WINDOW;
        }
    };
}

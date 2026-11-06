package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The Pixtral graph (models/pixtral.cpp {@code clip_graph_pixtral}): an RMS-norm ViT over any patch grid whose
 * attention rotates Q and K with a 2D RoPE (row positions on the first half of each head, column positions on the
 * second, interleaved frequencies, base 10000), then the Mistral Small 3.1 patch merger when the file carries it
 * (RMS norm, {@code s x s} unfold, {@code mm.patch_merger}), the GELU MLP, and for Pixtral one {@code [IMG_BREAK]}
 * embedding after every token row but the last (LightOnOCR runs the same graph without the break rows). Written
 * once against {@link VisionOps}. Package-private executor support.
 */
final class PixtralGraph {

    /** RoPE base of the vision attention (clip.cpp load_hparams). */
    static final double ROPE_THETA = 10000.0;

    private PixtralGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int n = pw * grid[1];
        int dh = ops.model().dHead();
        requireRopeHeadDim(dh, "pixtral");
        VisionOps.Block x = ops.patchRows(image, pw, grid[1]);
        int[] posH = new int[n];
        int[] posW = new int[n];
        for (int i = 0; i < n; i++) {
            posH[i] = i / pw;
            posW[i] = i % pw;
        }
        x = ops.tower(x, n, NormKind.RMS, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.twoHalves(posH, posW, ROPE_THETA, true)));
        return new Encoded(ops.download(x), n, pw, grid[1]);
    }

    /** The two-halves rope rotates adjacent pairs within each half head: the head width must divide by 4. */
    static void requireRopeHeadDim(int dh, String family) {
        if (dh % 4 != 0) {
            throw new IllegalArgumentException("the " + family + " 2D RoPE needs a head width divisible by 4, got "
                    + dh);
        }
    }

    /**
     * Optional patch merger, {@code mm.1} + GELU + {@code mm.2}, then the {@code [IMG_BREAK]} rows when the
     * projector resolved them (Pixtral; LightOnOCR has none).
     */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the pixtral projector");
        int s = ops.hp().nMerge();
        if (enc.pw % s != 0 || enc.ph % s != 0) {
            throw new IllegalArgumentException("the pixtral patch merger needs a patch grid that is a multiple of"
                    + " clip.vision.spatial_merge_size " + s + " per side, got " + enc.pw + "x" + enc.ph);
        }
        VisionOps.Block cur = ops.upload(enc.rows, enc.width);
        int n = enc.nTokens;
        if (ops.hasMatrix("mm.patch_merger.weight")) {
            cur = merge(ops, cur, enc, s);
            n = (enc.pw / s) * (enc.ph / s);
        }
        VisionOps.Block h = ops.linear("mm.1.weight", cur, "mm.1.bias");
        ops.activate(h, ActOp.GELU_TANH);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        float[] imgBreak = ops.hostVectorOrNull("v.token_embd.img_break");
        if (imgBreak == null) {
            return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
        }
        return withImageBreaks(ops.download(out), ops.model().outputDim(), enc.pw / s, enc.ph / s, imgBreak);
    }

    /**
     * The Mistral3 patch merger: RMS norm with {@code mm.input_norm}, then every {@code s x s} window unfolded
     * channel-major ({@code ggml_im2col}: slot {@code (c * s + ky) * s + kx}) and projected back to {@code n_embd}.
     */
    private static VisionOps.Block merge(VisionOps ops, VisionOps.Block cur, Encoded enc, int s) {
        int nEmbd = ops.hp().nEmbd();
        ops.norm(cur, NormKind.RMS, "mm.input_norm.weight", null, ops.hp().eps(), enc.nTokens, nEmbd);
        VisionOps.Block cols = ops.unfold(cur, nEmbd, enc.pw, enc.ph, s);
        return ops.linear("mm.patch_merger.weight", cols, null);
    }

    /** {@code [px * py + py - 1][d]} rows: each row of {@code px} tokens, then the break embedding, but the last. */
    static float[][] withImageBreaks(float[] tokens, int d, int px, int py, float[] imgBreak) {
        float[][] out = new float[px * py + py - 1][];
        int at = 0;
        for (int y = 0; y < py; y++) {
            for (int x = 0; x < px; x++) {
                float[] row = new float[d];
                System.arraycopy(tokens, (y * px + x) * d, row, 0, d);
                out[at++] = row;
            }
            if (y + 1 < py) {
                out[at++] = imgBreak.clone();
            }
        }
        return out;
    }
}

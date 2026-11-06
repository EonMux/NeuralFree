package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The DeepSeek-V4 and V4.1 graph (models/deepseek4v.cpp {@code clip_graph_deepseek4v}): the RMS-normalized ViT
 * without learned positions whose attention rotates Q and K with the vision m-rope over the patch row and column
 * (sections {@code d_head / 4} each over {@code d_head / 2} dimensions, base 10000, the qwen vision rope), then the
 * aligner: the patch grid zero-padded to a multiple of {@code n_merge} and unfolded {@code n_merge x n_merge},
 * the GELU-erf MLP {@code mm.1} / {@code mm.2}, and the token block that reorders the aligner rows and the
 * sentinel rows ({@link DeepseekV4Layout}; V4 reads the image's lead pads, {@link ClipImage#leadPad()}). Written
 * once against {@link VisionOps}. Package-private executor support.
 */
final class DeepseekV4Graph {

    private DeepseekV4Graph() {
    }

    /** The token block of {@code image}. */
    static float[][] encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        int n = pw * ph;
        int nEmbd = ops.hp().nEmbd();
        int m = ops.hp().nMerge();
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        x = ops.tower(x, n, NormKind.RMS, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.qwen(ropePositions(pw, ph))));
        VisionOps.Block cols = ops.unfoldPadded(x, nEmbd, pw, ph, m);
        VisionOps.Block h = ops.linear("mm.1.weight", cols, "mm.1.bias");
        ops.activate(h, ActOp.GELU_ERF);
        VisionOps.Block out = ops.linear("mm.2.weight", h, "mm.2.bias");
        float[][] rows = VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
        int w = DeepseekV4Layout.gridSide(pw, m);
        int gh = DeepseekV4Layout.gridSide(ph, m);
        boolean v4 = ProjectorEntry.PROJECTOR_DEEPSEEK4V.equals(ops.model().projectorGraph());
        int[] idx = v4 ? DeepseekV4Layout.index4v(w, gh, image.leadPad()) : DeepseekV4Layout.index41v(w, gh);
        float[][] sentinels = {ops.hostVector("v.token_embd.img_start"), ops.hostVector("v.token_embd.img_end"),
                ops.hostVector("v.image_newline"), v4 ? ops.hostVector("v.token_embd.img_pad") : null};
        return assemble(rows, sentinels, idx);
    }

    /** The m-rope positions {@code [row | col | row | col]} of the raster patches. */
    static int[] ropePositions(int pw, int ph) {
        int n = pw * ph;
        int[] pos = new int[QwenVisionGrid.POSITION_SECTIONS * n];
        for (int i = 0; i < n; i++) {
            pos[i] = i / pw;
            pos[n + i] = i % pw;
            pos[2 * n + i] = i / pw;
            pos[3 * n + i] = i % pw;
        }
        return pos;
    }

    /** Output row {@code t} is grid row {@code idx[t]}, or the sentinel {@code idx[t] - rows.length}. */
    private static float[][] assemble(float[][] rows, float[][] sentinels, int[] idx) {
        float[][] out = new float[idx.length][];
        for (int t = 0; t < idx.length; t++) {
            int i = idx[t];
            float[] src = i < rows.length ? rows[i] : sentinels[i - rows.length];
            if (src == null) {
                throw new IllegalStateException("the token block reads sentinel " + (i - rows.length)
                        + ", which the projector does not carry");
            }
            out[t] = src.clone();
        }
        return out;
    }
}

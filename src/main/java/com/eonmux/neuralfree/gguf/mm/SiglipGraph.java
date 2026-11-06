package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The square SigLIP graph (models/siglip.cpp): the patches of the exact {@code image_size} grid with the optional
 * class row and the learned position rows, and the layer-norm ViT with its optional pre- and post-norms. Also the
 * Gemma3 projector: the average pool over the square patch grid, the RMS norm and the input projection. Written
 * once against {@link VisionOps}. Package-private executor support.
 */
final class SiglipGraph {

    private SiglipGraph() {
    }

    /** The encoder rows of the tile, the square grid in raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        ClipVisionModel model = ops.model();
        int[] grid = ClipVit.grid(ops.hp(), image);
        ClipVit.requireImageSizeGrid(model, grid[0], grid[1]);
        int nPos = model.nPos();
        VisionOps.Block x = ops.patchRows(image, grid[0], grid[1]);
        float[] classEmbd = ops.hostWeight(model.classEmbd());
        if (classEmbd != null) {
            x = ops.insertRow(x, model.nPatches(), classEmbd);
        }
        ops.addHost(x, ops.hostWeight(model.posEmbd()));
        x = ops.tower(x, nPos, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        return new Encoded(ops.download(x), nPos, grid[0], grid[1]);
    }

    /** Average pool over the square patch grid, RMS norm, then the input projection. */
    static float[][] gemma3(VisionOps ops, Encoded enc) {
        enc.requireGrid("the gemma3 projector");
        int side = enc.pw;
        int k = ops.hp().nMerge();
        int nEmbd = ops.hp().nEmbd();
        if (enc.pw != enc.ph || side % k != 0) {
            throw new IllegalArgumentException("the gemma3 pool needs a square patch grid whose side is a multiple"
                    + " of the pool size " + k + ", got " + enc.pw + "x" + enc.ph);
        }
        int outSide = side / k;
        int nTokens = outSide * outSide;
        float[] pooled = new float[nTokens * nEmbd];
        float inv = 1f / (k * k);
        float[] rows = enc.rows;
        for (int oy = 0; oy < outSide; oy++) {
            for (int ox = 0; ox < outSide; ox++) {
                int dst = (oy * outSide + ox) * nEmbd;
                for (int dy = 0; dy < k; dy++) {
                    for (int dx = 0; dx < k; dx++) {
                        int src = ((oy * k + dy) * side + (ox * k + dx)) * nEmbd;
                        for (int c = 0; c < nEmbd; c++) {
                            pooled[dst + c] += rows[src + c];
                        }
                    }
                }
                for (int c = 0; c < nEmbd; c++) {
                    pooled[dst + c] *= inv;
                }
            }
        }
        VisionOps.Block x = ops.upload(pooled, nEmbd);
        ops.norm(x, NormKind.RMS, "mm.soft_emb_norm.weight", null, ops.hp().eps(), nTokens, nEmbd);
        VisionOps.Block out = ops.linear("mm.input_projection.weight", x, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}

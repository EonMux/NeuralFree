package com.eonmux.neuralfree.gguf.mm;

/**
 * The Step3-VL graph (models/step3vl.cpp {@code clip_graph_step3vl}): the Kimi-VL tower ({@link KimiVlGraph}: the
 * learned position grid resized antialiased, layer norm, the 2D RoPE of {@code build_rope_2d} with column
 * positions on the first head half and row positions on the second, base 10000); the projector runs the two 3x3
 * stride-2 convs {@code mm.0} and {@code mm.1} with zero padding 1 and their optional biases over the patch grid,
 * then the bias-free {@code mm.model.fc}. Written once against {@link VisionOps}. Package-private executor support.
 */
final class Step3vlGraph {

    /** Kernel side of the two downsamplers. */
    static final int KERNEL = 3;

    /** Stride of each downsampler. */
    static final int STRIDE = 2;

    /** Zero padding of each downsampler. */
    static final int PAD = 1;

    /** Patches merged per side by the two stride-2 downsamplers ({@code n_merge}). */
    static final int MERGE = STRIDE * STRIDE;

    private Step3vlGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        return KimiVlGraph.encode(ops, image, "step3vl");
    }

    /**
     * The two downsamplers and {@code mm.model.fc}: one row per merged cell, raster order.
     *
     * @throws IllegalArgumentException when the patch grid is not a multiple of {@link #MERGE} per side, where the
     *                                  convs would not give the {@code clip_n_output_tokens} grid
     */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the step3vl projector");
        PixelShuffle.requireMultiple(enc.pw, enc.ph, MERGE, "step3vl");
        VisionOps.Block cur = ops.upload(enc.rows, enc.width);
        int w = enc.pw;
        int h = enc.ph;
        int c = ops.hp().nEmbd();
        for (String conv : new String[]{"mm.0", "mm.1"}) {
            VisionOps.Block cols = ops.im2col(cur, c, w, h, KERNEL, STRIDE, PAD);
            int out = Math.toIntExact(ops.model().projectorMatrices().get(conv + ".weight").ne(3));
            cur = ops.linearKernel(conv + ".weight", out, c * KERNEL * KERNEL, cols, conv + ".bias");
            w = PixelShuffle.convOut(w, KERNEL, STRIDE, PAD);
            h = PixelShuffle.convOut(h, KERNEL, STRIDE, PAD);
            c = out;
        }
        VisionOps.Block y = ops.linear("mm.model.fc.weight", cur, null);
        return VisionRows.tokenRows(ops.download(y), ops.model().outputDim());
    }
}

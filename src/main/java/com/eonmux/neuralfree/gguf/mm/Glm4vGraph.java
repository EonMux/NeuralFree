package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;

/**
 * The GLM-4V graph (models/glm4v.cpp {@code clip_graph_glm4v}, GLM-4.1V / GLM-4.5V / GLM-OCR, and GLM-5V, whose
 * SiLU-gated FFNs in the tower and the projector clamp at {@code clip.vision.swiglu_clamp}): the two patch
 * convolutions of the temporal Conv3D summed, in 2x2 merge-group order ({@code (ty, tx, dy, dx)}), the patch
 * bias, the optional RMS embedding norm, the learned position rows resized to the grid with ggml's bicubic
 * {@code ggml_interpolate} and laid out in the same order, then an RMS-norm ViT with the 2D m-rope of the qwen
 * encoders. The projector merges every 2x2 group with the {@code mm.patch_merger} conv, then {@code mm.model.fc},
 * a layer norm, GELU-erf and the gated FFN {@code mm.up} / {@code mm.gate} / {@code mm.down}. Written once
 * against {@link VisionOps}. Package-private executor support.
 */
final class Glm4vGraph {

    /** The merge the graph's input reorder is written for. */
    static final int MERGE = 2;

    /** Epsilon of the projector's layer norm (nn.LayerNorm default). */
    static final float PROJECTOR_EPS = 1e-5f;

    private Glm4vGraph() {
    }

    /** The encoder rows of {@code image} in merge-group order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        requireMerge(ops.hp());
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        if (pw % MERGE != 0 || ph % MERGE != 0) {
            throw new IllegalArgumentException("the glm4v encoder needs a patch grid that is a multiple of 2 per"
                    + " side, got " + pw + "x" + ph);
        }
        int nEmbd = ops.hp().nEmbd();
        int n = pw * ph;
        int[] order = Qwen3VlGraph.mergeOrder(pw, ph, MERGE);
        // both conv kernels over the same patches, summed, then the bias (the patch embedding)
        VisionOps.Block x = ops.patchEmbed(ClipPatches.mergeGroups(image, ops.hp().patchSize(), MERGE, pw / MERGE,
                ph / MERGE), n);
        if (ops.model().normEmbdWeight() != null) {
            ops.normByRef(x, NormKind.RMS, ops.model().normEmbdWeight(), ops.model().normEmbdBias(),
                    ops.hp().eps(), n, nEmbd);
        }
        if (ops.model().posEmbd() != null) {
            float[] pos = ops.hostWeight(ops.model().posEmbd());
            int side = SiglipResizedGraph.side(pos, nEmbd);
            float[] raster = side == pw && side == ph ? pos : resizeBicubic(pos, nEmbd, side, pw, ph);
            ops.addHost(x, VisionRows.gatherRows(raster, order, nEmbd));
        }
        x = ops.tower(x, n, NormKind.RMS, VisionOps.TowerOptions.NONE.withRope(
                VisionOps.RopePlan.qwen(QwenVisionGrid.ropePositions(order, pw))));
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** Patch merger conv, {@code mm.model.fc}, layer norm, GELU-erf, then the gated FFN. */
    static float[][] project(VisionOps ops, Encoded enc) {
        int nEmbd = ops.hp().nEmbd();
        int group = MERGE * MERGE;
        int nOut = enc.nTokens / group;
        // conv slot (c * 2 + dy) * 2 + dx reads patch dy * 2 + dx of the group
        float[] cols = new float[nOut * nEmbd * group];
        for (int g = 0; g < nOut; g++) {
            for (int r = 0; r < group; r++) {
                for (int c = 0; c < nEmbd; c++) {
                    cols[(g * nEmbd + c) * group + r] = enc.rows[(g * group + r) * nEmbd + c];
                }
            }
        }
        VisionOps.Block cur = ops.linearKernel("mm.patch_merger.weight", ops.hostVector("mm.patch_merger.bias").length,
                nEmbd * group, ops.upload(cols, nEmbd * group), "mm.patch_merger.bias");
        cur = ops.linear("mm.model.fc.weight", cur, null);
        int width = cur.width();
        ops.norm(cur, NormKind.LAYER, "mm.post_norm.weight", "mm.post_norm.bias", PROJECTOR_EPS, nOut, width);
        ops.activate(cur, ActOp.GELU_ERF);
        VisionOps.Block gate = ops.linear("mm.gate.weight", cur, "mm.gate.bias");
        VisionOps.Block up = ops.linear("mm.up.weight", cur, "mm.up.bias");
        ops.glu(gate, up, ops.hp().ffnGlu(), ops.hp().ffnGluParams(), nOut, up.width());
        VisionOps.Block out = ops.linear("mm.down.weight", up, "mm.down.bias");
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    /**
     * The {@code [side * side][n_embd]} position grid resized to {@code [height * width][n_embd]}, raster order:
     * the {@code ggml_interpolate} bicubic kernel ({@code GGML_SCALE_MODE_BICUBIC}, half-pixel centers, the
     * {@code a = -0.75} convolution of PyTorch, edges clamped), computed in float as the kernel does.
     */
    static float[] resizeBicubic(float[] pos, int nEmbd, int side, int width, int height) {
        return resizeBicubic(pos, nEmbd, side, side, width, height);
    }

    /**
     * {@link #resizeBicubic(float[], int, int, int, int)} over a {@code srcWidth x srcHeight} grid of
     * {@code [srcHeight * srcWidth][n_embd]} rows.
     */
    static float[] resizeBicubic(float[] pos, int nEmbd, int srcWidth, int srcHeight, int width, int height) {
        float sf0 = (float) width / srcWidth;
        float sf1 = (float) height / srcHeight;
        float[] out = new float[width * height * nEmbd];
        float[] wx = new float[4];
        float[] wy = new float[4];
        for (int i1 = 0; i1 < height; i1++) {
            float y = (i1 + 0.5f) / sf1 - 0.5f;
            int y0 = (int) Math.floor(y);
            cubicWeights(y - y0, wy);
            for (int i0 = 0; i0 < width; i0++) {
                float x = (i0 + 0.5f) / sf0 - 0.5f;
                int x0 = (int) Math.floor(x);
                cubicWeights(x - x0, wx);
                int dst = (i1 * width + i0) * nEmbd;
                for (int c = 0; c < nEmbd; c++) {
                    float val = 0f;
                    for (int j = 0; j < 4; j++) {
                        int sy = clamp(y0 + j - 1, srcHeight);
                        float row = 0f;
                        for (int i = 0; i < 4; i++) {
                            row += pos[(sy * srcWidth + clamp(x0 + i - 1, srcWidth)) * nEmbd + c] * wx[i];
                        }
                        val += row * wy[j];
                    }
                    out[dst + c] = val;
                }
            }
        }
        return out;
    }

    /** The four bicubic taps at fractional offset {@code t} ({@code a = -0.75}). */
    private static void cubicWeights(float t, float[] w) {
        float a = -0.75f;
        w[0] = far(t + 1, a);
        w[1] = near(t, a);
        w[2] = near(1 - t, a);
        w[3] = far(2 - t, a);
    }

    private static float near(float x, float a) {
        return ((a + 2) * x - (a + 3)) * x * x + 1;
    }

    private static float far(float x, float a) {
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    }

    private static int clamp(int v, int n) {
        return Math.max(0, Math.min(v, n - 1));
    }

    private static void requireMerge(ClipHparams hp) {
        if (hp.nMerge() != MERGE) {
            throw new IllegalArgumentException("the glm4v graph merges 2x2 patch groups; the file sets"
                    + " clip.vision.spatial_merge_size " + hp.nMerge());
        }
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;

import java.util.Arrays;

/**
 * The SigLIP encoder over any patch grid (models/siglip.cpp for Phi-4 multimodal, LFM2 and d1-omni): the learned
 * position rows, stored as a square grid, are resized to the image's patch grid with an antialiased bilinear
 * filter (clip.cpp {@code resize_position_embeddings}, {@code ggml_interpolate} with
 * {@code GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}), then the layer-norm ViT runs. Package-private
 * executor support.
 */
final class SiglipResizedGraph {

    private SiglipResizedGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        float[] pos = ops.hostWeight(ops.model().posEmbd());
        int side = side(pos, nEmbd);
        float[] resized = side == pw && side == ph ? pos : resize(pos, nEmbd, side, pw, ph);
        ops.addHost(x, resized);
        x = ops.tower(x, pw * ph, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        return new Encoded(ops.download(x), pw * ph, pw, ph);
    }

    /**
     * The {@code [side * side][n_embd]} position grid resized to {@code [height * width][n_embd]}, raster order:
     * the {@code ggml_interpolate} antialiased bilinear kernel (a triangle filter whose support widens with the
     * downscale factor, PyTorch's {@code interpolate(antialias=True)}), computed in float as the kernel does.
     */
    static float[] resize(float[] pos, int nEmbd, int side, int width, int height) {
        float sf0 = (float) width / side;
        float sf1 = (float) height / side;
        float support0 = Math.max(1.0f, 1.0f / sf0);
        float support1 = Math.max(1.0f, 1.0f / sf1);
        float[] out = new float[width * height * nEmbd];
        float[] acc = new float[nEmbd];
        for (int i1 = 0; i1 < height; i1++) {
            float y = (i1 + 0.5f) / sf1;
            int yMin = Math.max((int) (y - support1 + 0.5f), 0);
            int yMax = Math.min((int) (y + support1 + 0.5f), side);
            for (int i0 = 0; i0 < width; i0++) {
                float x = (i0 + 0.5f) / sf0;
                int xMin = Math.max((int) (x - support0 + 0.5f), 0);
                int xMax = Math.min((int) (x + support0 + 0.5f), side);
                Arrays.fill(acc, 0f);
                float total = 0f;
                for (int sy = yMin; sy < yMax; sy++) {
                    float wy = triangle((sy - y + 0.5f) / support1);
                    for (int sx = xMin; sx < xMax; sx++) {
                        float wgt = triangle((sx - x + 0.5f) / support0) * wy;
                        if (wgt <= 0f) {
                            continue;
                        }
                        int src = (sy * side + sx) * nEmbd;
                        for (int c = 0; c < nEmbd; c++) {
                            acc[c] += pos[src + c] * wgt;
                        }
                        total += wgt;
                    }
                }
                int dst = (i1 * width + i0) * nEmbd;
                for (int c = 0; c < nEmbd; c++) {
                    out[dst + c] = total > 0f ? acc[c] / total : acc[c];
                }
            }
        }
        return out;
    }

    /**
     * The {@code [side * side][n_embd]} position grid resized to {@code [height * width][n_embd]}, raster order,
     * by the clamped bilinear kernel of {@code ggml_interpolate} ({@code GGML_SCALE_MODE_BILINEAR}): output
     * {@code (i0, i1)} samples source {@code ((i0 + offset) / sf0 - offset, (i1 + offset) / sf1 - offset)}, the
     * two neighbours per axis clamped to the grid and the fractions clamped to {@code [0, 1]}.
     */
    static float[] bilinear(float[] pos, int nEmbd, int side, int width, int height, float sf0, float sf1,
                            float offset) {
        float[] out = new float[width * height * nEmbd];
        for (int i1 = 0; i1 < height; i1++) {
            float y = (i1 + offset) / sf1 - offset;
            int y0 = clamp((int) Math.floor(y), side);
            int y1 = clamp((int) Math.floor(y) + 1, side);
            float dy = Math.max(0f, Math.min(y - y0, 1f));
            for (int i0 = 0; i0 < width; i0++) {
                float x = (i0 + offset) / sf0 - offset;
                int x0 = clamp((int) Math.floor(x), side);
                int x1 = clamp((int) Math.floor(x) + 1, side);
                float dx = Math.max(0f, Math.min(x - x0, 1f));
                int a = (y0 * side + x0) * nEmbd;
                int b = (y0 * side + x1) * nEmbd;
                int c = (y1 * side + x0) * nEmbd;
                int d = (y1 * side + x1) * nEmbd;
                int dst = (i1 * width + i0) * nEmbd;
                for (int e = 0; e < nEmbd; e++) {
                    out[dst + e] = pos[a + e] * (1 - dx) * (1 - dy) + pos[b + e] * dx * (1 - dy)
                            + pos[c + e] * (1 - dx) * dy + pos[d + e] * dx * dy;
                }
            }
        }
        return out;
    }

    /**
     * {@link #bilinear} with {@code GGML_SCALE_FLAG_ALIGN_CORNERS} (Qwen3-VL): no pixel offset, and the scale
     * {@code (out - 1) / (in - 1)} per axis when both sides exceed one; the grid itself when the sizes match.
     */
    static float[] alignCorners(float[] pos, int nEmbd, int side, int width, int height) {
        if (side == width && side == height) {
            return pos;
        }
        float sf0 = width > 1 && side > 1 ? (float) (width - 1) / (side - 1) : (float) width / side;
        float sf1 = height > 1 && side > 1 ? (float) (height - 1) / (side - 1) : (float) height / side;
        return bilinear(pos, nEmbd, side, width, height, sf0, sf1, 0f);
    }

    /**
     * clip.cpp {@code resize_position_embeddings(GGML_SCALE_MODE_BILINEAR)}: {@link #bilinear} at the scale
     * {@code out / in} per axis with the half-pixel offset; the grid itself when the sizes match.
     */
    static float[] halfPixel(float[] pos, int nEmbd, int side, int width, int height) {
        if (side == width && side == height) {
            return pos;
        }
        return bilinear(pos, nEmbd, side, width, height, (float) width / side, (float) height / side, 0.5f);
    }

    /** The side of the square position grid of {@code posEmbd}, {@code n_embd} values per row. */
    static int side(float[] posEmbd, int nEmbd) {
        return (int) Math.round(Math.sqrt(posEmbd.length / (double) nEmbd));
    }

    private static int clamp(int i, int n) {
        return Math.max(0, Math.min(i, n - 1));
    }

    private static float triangle(float x) {
        return Math.max(1.0f - Math.abs(x), 0.0f);
    }
}

package com.eonmux.neuralfree.gguf.mm;

/**
 * The patch-embedding input of the clip graphs as a matrix: one row per patch holding the conv input in weight
 * order (channel, kernel row, kernel column), so {@code ggml_conv_2d} with stride = kernel = patch size becomes one
 * matmul by the conv weight read as a {@code [n_embd][3 * ps * ps]} matrix. Row {@code t}, slot
 * {@code (c * ps + ky) * ps + kx} reads pixel {@code (x0 + kx, y0 + ky)} of channel {@code c}; the pixel-major
 * layout of {@link #mergeGroupsPixelMajor} feeds a linear patch embedding instead.
 */
final class ClipPatches {

    private ClipPatches() {
    }

    /** Values per patch row: {@code 3 * ps * ps}. */
    static int rowWidth(int ps) {
        return 3 * ps * ps;
    }

    /** Values per patch row of a temporal pair: the two frames' patches side by side. */
    static int pairRowWidth(int ps) {
        return 2 * 3 * ps * ps;
    }

    /** The {@code pw x ph} patches in raster order ({@code t = py * pw + px}). */
    static float[] raster(ClipImage image, int ps, int pw, int ph) {
        float[] second = image.secondFrame();
        int width = second == null ? rowWidth(ps) : pairRowWidth(ps);
        float[] out = new float[Math.multiplyExact(pw * ph, width)];
        float[] pixels = image.pixels();
        int t = 0;
        for (int py = 0; py < ph; py++) {
            for (int px = 0; px < pw; px++) {
                gather(pixels, image.width(), px * ps, py * ps, ps, out, t * width);
                if (second != null) {
                    gather(second, image.width(), px * ps, py * ps, ps, out, t * width + rowWidth(ps));
                }
                t++;
            }
        }
        return out;
    }

    /**
     * The patches in merge-group order {@code (ty, tx, dy, dx)} over an {@code mw x mh} grid of
     * {@code merge x merge} groups: the token order of the qwen encoders (the fill loop of clip.cpp). A temporal
     * pair doubles every row: the first frame's patch, then the second frame's patch at the same grid cell.
     */
    static float[] mergeGroups(ClipImage image, int ps, int merge, int mw, int mh) {
        return mergeGroups(image, ps, merge, mw, mh, false);
    }

    /**
     * {@link #mergeGroups(ClipImage, int, int, int, int)} with each patch laid out pixel-major, slot
     * {@code (ky * ps + kx) * 3 + c}: the input of the Siglip2 NaFlex linear patch embedding (Youtu-VL), whose
     * {@code nn.Linear} reads the patch as the graph's permute chain leaves it.
     */
    static float[] mergeGroupsPixelMajor(ClipImage image, int ps, int merge, int mw, int mh) {
        return mergeGroups(image, ps, merge, mw, mh, true);
    }

    private static float[] mergeGroups(ClipImage image, int ps, int merge, int mw, int mh, boolean pixelMajor) {
        float[] second = image.secondFrame();
        if (second != null && pixelMajor) {
            throw new IllegalArgumentException("a temporal pair patchifies channel-major; the pixel-major patch"
                    + " layout takes single frames");
        }
        int width = second == null ? rowWidth(ps) : pairRowWidth(ps);
        float[] out = new float[Math.multiplyExact(mw * mh * merge * merge, width)];
        float[] pixels = image.pixels();
        int t = 0;
        for (int ty = 0; ty < mh; ty++) {
            for (int tx = 0; tx < mw; tx++) {
                for (int dy = 0; dy < merge; dy++) {
                    for (int dx = 0; dx < merge; dx++) {
                        int x0 = (tx * merge + dx) * ps;
                        int y0 = (ty * merge + dy) * ps;
                        if (pixelMajor) {
                            gatherPixelMajor(pixels, image.width(), x0, y0, ps, out, t * width);
                        } else {
                            gather(pixels, image.width(), x0, y0, ps, out, t * width);
                            if (second != null) {
                                gather(second, image.width(), x0, y0, ps, out, t * width + rowWidth(ps));
                            }
                        }
                        t++;
                    }
                }
            }
        }
        return out;
    }

    private static void gatherPixelMajor(float[] pixels, int imageWidth, int x0, int y0, int ps, float[] dst,
                                         int dstOff) {
        for (int ky = 0; ky < ps; ky++) {
            int src = ((y0 + ky) * imageWidth + x0) * 3;
            System.arraycopy(pixels, src, dst, dstOff + ky * ps * 3, ps * 3);
        }
    }

    private static void gather(float[] pixels, int imageWidth, int x0, int y0, int ps, float[] dst, int dstOff) {
        int at = dstOff;
        for (int c = 0; c < 3; c++) {
            for (int ky = 0; ky < ps; ky++) {
                int rowBase = (y0 + ky) * imageWidth + x0;
                for (int kx = 0; kx < ps; kx++) {
                    dst[at++] = pixels[(rowBase + kx) * 3 + c];
                }
            }
        }
    }
}

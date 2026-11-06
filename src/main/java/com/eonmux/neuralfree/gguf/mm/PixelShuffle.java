package com.eonmux.neuralfree.gguf.mm;

import java.util.function.UnaryOperator;

/**
 * The pixel-shuffle (space-to-depth) merges of the vision projectors, each transcribed from its ggml
 * reshape / permute chain: every output token concatenates {@code s * s} encoder rows of an {@code s x s} patch
 * window, in the order the family's reference graph lays them out. Input rows are raster order
 * ({@code y * pw + x}), {@code n_embd} wide. Package-private executor support.
 */
final class PixelShuffle {

    /** The largest row id {@link #chunkSources} carries exactly in a float. */
    private static final int MAX_FLOAT_ID = (1 << 24) - 1;

    private PixelShuffle() {
    }

    /**
     * clip.cpp {@code build_patch_merge_permute} (Idefics3, also the square tiles of Cohere2V): the grid is
     * zero-padded to multiples of {@code s}, then unshuffled along the height and the width.
     *
     * @return {@code [(pw'/s) * (ph'/s)][n_embd * s * s]} rows, {@code pw'} and {@code ph'} the padded sides
     */
    static float[] mergePermute(float[] rows, int nEmbd, int pw, int ph, int s) {
        int wp = align(pw, s);
        int hp = align(ph, s);
        float[] cur = rows;
        if (wp != pw || hp != ph) {
            cur = new float[nEmbd * wp * hp];
            for (int y = 0; y < ph; y++) {
                System.arraycopy(rows, y * pw * nEmbd, cur, y * wp * nEmbd, pw * nEmbd);
            }
        }
        // [n_embd*s, wp/s, hp] -> [n_embd*s, hp, wp/s] -> as [n_embd*s*s, hp/s, wp/s] -> [n_embd*s*s, wp/s, hp/s]
        cur = VisionRows.swap12(cur, nEmbd * s, wp / s, hp);
        return VisionRows.swap12(cur, nEmbd * s * s, hp / s, wp / s);
    }

    /**
     * The InternVL shuffle (models/internvl.cpp): the rows viewed as {@code [n_embd*s, ph/s, pw]}, dims 1 and 2
     * swapped, viewed as {@code [n_embd*s*s, ph/s, pw/s]} and swapped again.
     */
    static float[] internvl(float[] rows, int nEmbd, int pw, int ph, int s) {
        requireMultiple(pw, ph, s, "internvl");
        float[] cur = VisionRows.swap12(rows, nEmbd * s, ph / s, pw);
        return VisionRows.swap12(cur, nEmbd * s * s, ph / s, pw / s);
    }

    /**
     * The Llama 4 shuffle (models/llama4.cpp, Llama4VisionPixelShuffleMLP): the rows viewed as
     * {@code [n_embd*s, pw/s, ph]}, dims 1 and 2 swapped, then flattened as {@code [n_embd*s*s]} rows.
     */
    static float[] llama4(float[] rows, int nEmbd, int pw, int ph, int s) {
        requireMultiple(pw, ph, s, "llama4");
        return VisionRows.swap12(rows, nEmbd * s, pw / s, ph);
    }

    /**
     * Every {@code s x s} window unfolded channel-major ({@code ggml_im2col} of a stride-{@code s} conv: slot
     * {@code (c * s + ky) * s + kx}), the input of a conv kernel read as a {@code [n_out, n_embd * s * s]} matrix.
     *
     * @return {@code [(pw/s) * (ph/s)][n_embd * s * s]} rows, raster order over the windows
     * @throws IllegalArgumentException when a side of the grid is not a multiple of {@code s}
     */
    static float[] unfold(float[] rows, int nEmbd, int pw, int ph, int s) {
        requireMultiple(pw, ph, s, "conv");
        return unfoldWindows(rows, nEmbd, pw, pw / s, ph / s, s);
    }

    /**
     * The output side of a {@code ggml_conv_2d} over {@code in} values with kernel {@code k}, stride
     * {@code stride} and zero padding {@code pad}: {@code (in + 2 pad - k) / stride + 1}.
     *
     * @throws IllegalArgumentException when the kernel does not fit the padded side
     */
    static int convOut(int in, int k, int stride, int pad) {
        if (k <= 0 || stride <= 0 || pad < 0 || in <= 0 || in + 2 * pad < k) {
            throw new IllegalArgumentException("a " + k + "x" + k + " conv at stride " + stride + " and padding " + pad
                    + " does not fit a side of " + in);
        }
        return (in + 2 * pad - k) / stride + 1;
    }

    /**
     * The im2col of {@code ggml_conv_2d}: every {@code k x k} window of a {@code pw x ph} grid of {@code n_embd}
     * rows at stride {@code stride} with zero padding {@code pad}, unfolded channel-major (value
     * {@code (c * k + ky) * k + kx}, the layout of a {@code [k, k, c_in, c_out]} kernel read as a matrix).
     *
     * @return {@code [ow * oh][n_embd * k * k]} rows, raster order over the output grid ({@link #convOut})
     */
    static float[] im2col(float[] rows, int nEmbd, int pw, int ph, int k, int stride, int pad) {
        return windows(rows, nEmbd, pw, ph, k, stride, pad, pad, convOut(pw, k, stride, pad),
                convOut(ph, k, stride, pad));
    }

    /**
     * {@link #im2col} whose windows start {@code padTop} / {@code padLeft} cells before the grid (a
     * SAME-padded conv's asymmetric padding), laid out over the caller's {@code ow x oh} output grid.
     */
    static float[] im2colPadded(float[] rows, int nEmbd, int pw, int ph, int k, int stride, int padTop, int padLeft,
                                int ow, int oh) {
        return windows(rows, nEmbd, pw, ph, k, stride, padTop, padLeft, ow, oh);
    }

    /**
     * Every {@code s x s} window of a {@code pw x ph} grid zero-padded at the right and bottom to a multiple of
     * {@code s} ({@code ggml_pad}, then the stride-{@code s} im2col), unfolded channel-major as {@link #unfold}.
     *
     * @return {@code [ceil(pw / s) * ceil(ph / s)][n_embd * s * s]} rows, raster order over the windows
     */
    static float[] unfoldPadded(float[] rows, int nEmbd, int pw, int ph, int s) {
        if (s <= 0) {
            throw new IllegalArgumentException("the unfold window must be positive, got " + s);
        }
        return windows(rows, nEmbd, pw, ph, s, s, 0, 0, (pw + s - 1) / s, (ph + s - 1) / s);
    }

    /**
     * The {@code ow x oh} windows of side {@code k} at stride {@code stride} whose first window starts
     * {@code padTop} / {@code padLeft} cells before the grid, unfolded channel-major; cells outside the grid
     * read zero.
     */
    private static float[] windows(float[] rows, int nEmbd, int pw, int ph, int k, int stride, int padTop,
                                   int padLeft, int ow, int oh) {
        if ((long) pw * ph * nEmbd != rows.length) {
            throw new IllegalArgumentException("the im2col reads " + pw + "x" + ph + " rows of " + nEmbd
                    + ", the block holds " + rows.length + " values");
        }
        int width = nEmbd * k * k;
        float[] cols = new float[Math.multiplyExact(ow * oh, width)];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int dst = (oy * ow + ox) * width;
                for (int ky = 0; ky < k; ky++) {
                    int y = oy * stride - padTop + ky;
                    for (int kx = 0; kx < k; kx++) {
                        int x = ox * stride - padLeft + kx;
                        if (y < 0 || y >= ph || x < 0 || x >= pw) {
                            continue;
                        }
                        int src = (y * pw + x) * nEmbd;
                        for (int c = 0; c < nEmbd; c++) {
                            cols[dst + (c * k + ky) * k + kx] = rows[src + c];
                        }
                    }
                }
            }
        }
        return cols;
    }

    /**
     * The first {@code ow x oh} windows of a {@code pw}-wide grid unfolded as {@link #unfold} does; the patches
     * past the last full window are dropped, as an unpadded stride-{@code s} conv drops them.
     *
     * @return {@code [ow * oh][n_embd * s * s]} rows, raster order over the windows
     */
    static float[] unfoldWindows(float[] rows, int nEmbd, int pw, int ow, int oh, int s) {
        int k = nEmbd * s * s;
        float[] cols = new float[ow * oh * k];
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int dst = (oy * ow + ox) * k;
                for (int c = 0; c < nEmbd; c++) {
                    for (int ky = 0; ky < s; ky++) {
                        for (int kx = 0; kx < s; kx++) {
                            int src = ((oy * s + ky) * pw + ox * s + kx) * nEmbd + c;
                            cols[dst + (c * s + ky) * s + kx] = rows[src];
                        }
                    }
                }
            }
        }
        return cols;
    }

    /**
     * The source row of every one-row chunk of a reorder over {@code nRows} rows, {@code -1} for a zero pad: the
     * reorder run over the row ids at {@code n_embd = 1}. Every reorder of this class moves whole {@code n_embd}
     * chunks the same way at any width, so a device executor gathers by this table and matches the transcription
     * here (the channel-major unfolds also need {@link #channelMajor}).
     *
     * @param reorder the reorder at {@code n_embd = 1}, from {@code nRows} values to its output values
     * @throws IllegalArgumentException when {@code nRows} exceeds the exactly representable float ids
     */
    static int[] chunkSources(int nRows, UnaryOperator<float[]> reorder) {
        if (nRows <= 0 || nRows > MAX_FLOAT_ID) {
            throw new IllegalArgumentException("a reorder table covers 1 to " + MAX_FLOAT_ID + " rows, got " + nRows);
        }
        float[] ids = new float[nRows];
        for (int t = 0; t < nRows; t++) {
            ids[t] = t + 1;
        }
        float[] moved = reorder.apply(ids);
        int[] src = new int[moved.length];
        for (int i = 0; i < moved.length; i++) {
            src[i] = (int) moved[i] - 1;
        }
        return src;
    }

    /**
     * The column permutation from the patch-major layout of an unfolded {@code s x s} window (slot
     * {@code (ky * s + kx) * n_embd + c}, the window's rows side by side) to the channel-major layout of
     * {@link #unfoldWindows} (slot {@code (c * s + ky) * s + kx}): entry {@code k} is the patch-major column that
     * channel-major column {@code k} reads.
     */
    static int[] channelMajor(int nEmbd, int s) {
        int[] perm = new int[nEmbd * s * s];
        for (int c = 0; c < nEmbd; c++) {
            for (int ky = 0; ky < s; ky++) {
                for (int kx = 0; kx < s; kx++) {
                    perm[(c * s + ky) * s + kx] = (ky * s + kx) * nEmbd + c;
                }
            }
        }
        return perm;
    }

    /** Refuses a patch grid whose sides are not multiples of the merge size {@code s}, naming the family. */
    static void requireMultiple(int pw, int ph, int s, String family) {
        if (pw % s != 0 || ph % s != 0) {
            throw new IllegalArgumentException("the " + family + " pixel shuffle needs a patch grid that is a"
                    + " multiple of the scale factor " + s + " per side, got " + pw + "x" + ph);
        }
    }

    private static int align(int v, int s) {
        return (v + s - 1) / s * s;
    }
}

package com.eonmux.neuralfree.gguf.mm;

/**
 * The token blocks of the DeepSeek-V4 vision encoders over an {@code w x h} aligner grid: which aligner row or
 * sentinel row each output row reads (clip.cpp {@code set_input} {@code layout_idx}, clip-impl.h
 * {@code dsv4_get_block_layout} and {@code dsv41_n_block_tokens}). The index of a sentinel is past the
 * {@code w * h} grid rows: {@link #START}, {@link #END}, {@link #NEWLINE} and {@link #PAD} added to {@code w * h}.
 * Static methods only.
 *
 * <p>DeepSeek-V4 (deepseek4v): {@code [PAD] * lead [START] <rows> [PAD] * padLast [END]}, where every grid row ends
 * with a newline, an odd row count takes a full row of pads, and each pair of adjacent rows interleaves
 * column-wise ("N-layout"); {@code padLast} keeps the block even. DeepSeek-V4.1 (deepseek41v): the reading order
 * {@code [START] (row [NEWLINE]) * h [END]}.</p>
 */
final class DeepseekV4Layout {

    /** Offset past the grid rows of the image-start sentinel row. */
    static final int START = 0;
    /** Offset past the grid rows of the image-end sentinel row. */
    static final int END = 1;
    /** Offset past the grid rows of the newline sentinel row. */
    static final int NEWLINE = 2;
    /** Offset past the grid rows of the pad sentinel row (deepseek4v only). */
    static final int PAD = 3;

    /** The token alignment of the DeepSeek-V4 text model ({@code COMPRESS_PAD_TO}); lead pads reach it. */
    static final int ALIGN = 4;

    /** The most lead pads a block takes ({@code ALIGN - 1}). */
    static final int MAX_LEAD_PAD = ALIGN - 1;

    private DeepseekV4Layout() {
    }

    /** The aligner grid side over {@code patches} patches merged by {@code merge}: {@code ceil(patches / merge)}. */
    static int gridSide(int patches, int merge) {
        return (patches + merge - 1) / merge;
    }

    /** The lead pads before an image after {@code tokensBefore} prompt tokens ({@code mtmd_tokenize}). */
    static int leadPad(int tokensBefore) {
        if (tokensBefore < 0) {
            throw new IllegalArgumentException("the token count before the image must not be negative, got "
                    + tokensBefore);
        }
        return ALIGN - 1 - tokensBefore % ALIGN;
    }

    /** The deepseek4v block length over a {@code w x h} grid with {@code lead} lead pads. */
    static int tokens4v(int w, int h, int lead) {
        requireGrid(w, h);
        int rows = h + h % 2;
        int rowLen = w + 1;
        int padLast = (rows / 2 * rowLen) % 2 * 2;
        return Math.addExact(lead + 1 + padLast + 1, Math.multiplyExact(rows, rowLen));
    }

    /** The deepseek41v block length over a {@code w x h} grid. */
    static int tokens41v(int w, int h) {
        requireGrid(w, h);
        return Math.addExact(Math.multiplyExact(h, w + 1), 2);
    }

    /** The deepseek4v row indices over a {@code w x h} grid with {@code lead} lead pads. */
    static int[] index4v(int w, int h, int lead) {
        if (lead < 0) {
            throw new IllegalArgumentException("the lead pad must not be negative, got " + lead);
        }
        int n = w * h;
        int[] idx = new int[tokens4v(w, h, lead)];
        int at = 0;
        for (int i = 0; i < lead; i++) {
            idx[at++] = n + PAD;
        }
        idx[at++] = n + START;
        int rows = h + h % 2;
        int rowLen = w + 1;
        for (int t = 0; t < rows * rowLen; t++) {
            int g = t / (2 * rowLen);
            int rem = t % (2 * rowLen);
            int c = rem / 2;
            int r = 2 * g + rem % 2;
            if (r >= h) {
                idx[at++] = n + PAD;
            } else if (c == w) {
                idx[at++] = n + NEWLINE;
            } else {
                idx[at++] = r * w + c;
            }
        }
        while (at < idx.length - 1) {
            idx[at++] = n + PAD;
        }
        idx[at] = n + END;
        return idx;
    }

    /** The deepseek41v row indices over a {@code w x h} grid. */
    static int[] index41v(int w, int h) {
        int n = w * h;
        int[] idx = new int[tokens41v(w, h)];
        int at = 0;
        idx[at++] = n + START;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                idx[at++] = r * w + c;
            }
            idx[at++] = n + NEWLINE;
        }
        idx[at] = n + END;
        return idx;
    }

    private static void requireGrid(int w, int h) {
        if (w <= 0 || h <= 0) {
            throw new IllegalArgumentException("the aligner grid needs positive sides, got " + w + "x" + h);
        }
    }
}

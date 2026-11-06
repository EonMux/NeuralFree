package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Token geometry of one image on the Qwen2-VL / Qwen2.5-VL and Youtu-VL encoders, filled exactly as the graph
 * inputs of clip.cpp {@code set_input} for these projectors:
 * <ul>
 *   <li>patch rows in merge-group order {@code (ty, tx, dy, dx)};</li>
 *   <li>the window plan (Qwen2.5-VL: {@code clip.vision.n_wa_pattern > 0}; Youtu-VL: a non-empty
 *       {@code clip.vision.wa_layer_indexes}): merged tokens grouped window by window
 *       ({@code window_idx} / {@code inv_window_idx}, windows of {@code clip.vision.window_size} pixels, 112 when
 *       the key is absent), each window a contiguous range of patch rows; the block-diagonal attention mask of the
 *       windowed layers is the key range {@code [windowBegin(w), windowEnd(w))} of the window {@code w = windowOfRow(r)} of each
 *       patch row {@code r};</li>
 *   <li>the 2D m-rope positions {@code [y | x | y | x]} of every patch row in encoder order, remapped with the
 *       window order when the tokens are regrouped.</li>
 * </ul>
 * {@link #rope} is the vision rope of {@code ggml_rope_multi(..., d_head/2, {d/4, d/4, d/4, d/4},
 * GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1)}. As {@link VisionRows.KeyRanges} the grid gives each patch
 * row the key range of its window. Immutable; array accessors return the stored arrays, which callers must not
 * modify.
 */
final class QwenVisionGrid implements VisionRows.KeyRanges {

    /** Window side in pixels when {@code clip.vision.window_size} is absent (clip.cpp). */
    static final int DEFAULT_WINDOW_PIXELS = 112;
    /** Rope frequency base of the vision rope. */
    static final float ROPE_FREQ_BASE = 10000f;
    /** {@code n_ctx_orig} of the vision rope call. */
    static final int ROPE_N_CTX_ORIG = 32768;
    /** Position sections per patch row ({@code positions} holds {@code 4 * nPatches} ints). */
    static final int POSITION_SECTIONS = 4;

    private final int mw;
    private final int mh;
    private final int mpow;
    private final boolean reorder;
    private final int[] idx;
    private final int[] invIdx;
    private final int[] positions;
    private final int[] windowBounds;
    private final int[] rowWindow;

    private QwenVisionGrid(int mw, int mh, int mpow, boolean reorder, int[] idx, int[] invIdx, int[] positions,
                           int[] windowBounds, int[] rowWindow) {
        this.mw = mw;
        this.mh = mh;
        this.mpow = mpow;
        this.reorder = reorder;
        this.idx = idx;
        this.invIdx = invIdx;
        this.positions = positions;
        this.windowBounds = windowBounds;
        this.rowWindow = rowWindow;
    }

    /**
     * The grid of a {@code width x height} image.
     *
     * @throws IllegalArgumentException when the image is not a whole number of patches, the patch grid is not a
     *                                  multiple of the merge size per side, or the window is below one merged token
     */
    static QwenVisionGrid of(ClipHparams hp, int width, int height) {
        return of(hp, width, height, hp.nWaPattern() > 0);
    }

    /**
     * The grid of a {@code width x height} image, with the window plan when {@code windowAttention} (Youtu-VL:
     * a non-empty {@code clip.vision.wa_layer_indexes}; the qwen2.5 encoders: {@code n_wa_pattern > 0}).
     *
     * @throws IllegalArgumentException as {@link #of(ClipHparams, int, int)}
     */
    static QwenVisionGrid of(ClipHparams hp, int width, int height, boolean windowAttention) {
        int ps = hp.patchSize();
        int merge = hp.nMerge();
        if (width % ps != 0 || height % ps != 0) {
            throw new IllegalArgumentException("image size " + width + "x" + height + " is not a multiple of the"
                    + " patch size " + ps);
        }
        int pw = width / ps;
        int ph = height / ps;
        if (pw % merge != 0 || ph % merge != 0 || pw == 0 || ph == 0) {
            throw new IllegalArgumentException("patch grid " + pw + "x" + ph + " must be a positive multiple of the"
                    + " merge size " + merge + " per side");
        }
        int mw = pw / merge;
        int mh = ph / merge;
        int mpow = merge * merge;
        int nTokens = mw * mh;
        int[] idx = new int[nTokens];
        int[] invIdx = new int[nTokens];
        boolean reorder = windowAttention;
        int[] windowBounds;
        if (reorder) {
            windowBounds = windowOrder(mw, mh, mpow, windowTokens(hp), idx, invIdx);
        } else {
            for (int i = 0; i < nTokens; i++) {
                idx[i] = i;
                invIdx[i] = i;
            }
            windowBounds = new int[]{0, nTokens * mpow};
        }
        return new QwenVisionGrid(mw, mh, mpow, reorder, idx, invIdx, positions(mw, mh, merge, idx),
                windowBounds, rowWindows(windowBounds));
    }

    /**
     * Fills {@code idx} / {@code invIdx} with the window order of an {@code mw x mh} merged grid split into
     * {@code window x window} tiles (row-major tiles, row-major tokens within a tile) and returns the patch-row
     * bounds of the windows, {@code windowCount + 1} values from 0.
     */
    private static int[] windowOrder(int mw, int mh, int mpow, int window, int[] idx, int[] invIdx) {
        List<Integer> bounds = new ArrayList<>();
        bounds.add(0);
        int dst = 0;
        for (int y = 0; y < mh; y += window) {
            for (int x = 0; x < mw; x += window) {
                int winH = Math.min(window, mh - y);
                int winW = Math.min(window, mw - x);
                for (int dy = 0; dy < winH; dy++) {
                    for (int dx = 0; dx < winW; dx++) {
                        int src = (y + dy) * mw + (x + dx);
                        idx[src] = dst;
                        invIdx[dst] = src;
                        dst++;
                    }
                }
                bounds.add(dst * mpow);
            }
        }
        int[] out = new int[bounds.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = bounds.get(i);
        }
        return out;
    }

    /** Merged tokens per window side: {@code window_size / patch_size / n_merge}. */
    private static int windowTokens(ClipHparams hp) {
        int pixels = hp.attnWindowSize() > 0 ? hp.attnWindowSize() : DEFAULT_WINDOW_PIXELS;
        int window = pixels / hp.patchSize() / hp.nMerge();
        if (window < 1) {
            throw new IllegalArgumentException("clip.vision.window_size " + pixels + " is below one merged token ("
                    + hp.patchSize() * hp.nMerge() + " pixels)");
        }
        return window;
    }

    /** {@code positions[remap + k * n]}: y, x, y, x of every patch row, at its row in encoder order. */
    private static int[] positions(int mw, int mh, int merge, int[] idx) {
        int mpow = merge * merge;
        int n = mw * mh * mpow;
        int[] pos = new int[POSITION_SECTIONS * n];
        int ptr = 0;
        for (int ty = 0; ty < mh; ty++) {
            for (int tx = 0; tx < mw; tx++) {
                for (int dy = 0; dy < merge; dy++) {
                    for (int dx = 0; dx < merge; dx++) {
                        int remap = idx[ptr / mpow] * mpow + ptr % mpow;
                        int y = ty * merge + dy;
                        int x = tx * merge + dx;
                        pos[remap] = y;
                        pos[n + remap] = x;
                        pos[2 * n + remap] = y;
                        pos[3 * n + remap] = x;
                        ptr++;
                    }
                }
            }
        }
        return pos;
    }

    /**
     * The {@code 4 * n} rope positions {@code [y | x | y | x]} of {@code n} tokens, token {@code t} holding raster
     * patch {@code patches[t]} of a grid {@code pw} patches wide (the raster and merge-group fills of clip.cpp
     * {@code set_input} for the qwen3vl, paddleocr and dots graphs).
     */
    static int[] ropePositions(int[] patches, int pw) {
        int n = patches.length;
        int[] pos = new int[POSITION_SECTIONS * n];
        for (int t = 0; t < n; t++) {
            int y = patches[t] / pw;
            int x = patches[t] % pw;
            pos[t] = y;
            pos[n + t] = x;
            pos[2 * n + t] = y;
            pos[3 * n + t] = x;
        }
        return pos;
    }

    private static int[] rowWindows(int[] bounds) {
        int[] rows = new int[bounds[bounds.length - 1]];
        for (int w = 0; w + 1 < bounds.length; w++) {
            Arrays.fill(rows, bounds[w], bounds[w + 1], w);
        }
        return rows;
    }

    /**
     * The vision rope of a head width: {@code n_dims = d_head / 2} over the {@code [y, x]} sections of
     * {@code d_head / 4} pairs each (the further sections of the clip call are never reached by a pair).
     *
     * @throws IllegalArgumentException when {@code d_head} is not a positive multiple of 4
     */
    static RopeSpec rope(int dHead) {
        return rope(dHead, ROPE_FREQ_BASE);
    }

    /** {@link #rope(int)} at frequency base {@code base}. */
    static RopeSpec rope(int dHead, float base) {
        if (dHead <= 0 || dHead % 4 != 0) {
            throw new IllegalArgumentException("the qwen vision rope needs a head width that is a positive multiple"
                    + " of 4, got " + dHead);
        }
        int quarter = dHead / 4;
        return RopeSpec.builder(RopeType.VISION)
                .nDims(dHead / 2)
                .freqBase(base)
                .nCtxOrig(ROPE_N_CTX_ORIG)
                .sections(new int[]{quarter, quarter, 0, 0})
                .build();
    }

    /** Merged tokens per row of the output grid. */
    int mergedWidth() {
        return mw;
    }

    /** Merged tokens per column of the output grid. */
    int mergedHeight() {
        return mh;
    }

    /** Output tokens: {@code mw * mh}. */
    int nTokens() {
        return mw * mh;
    }

    /** Patch rows the encoder runs over: {@code nTokens * n_merge^2}. */
    int nPatches() {
        return mw * mh * mpow;
    }

    /** Patch rows per merged token. */
    int mergeArea() {
        return mpow;
    }

    /** True when the tokens are regrouped window by window (Qwen2.5-VL window attention). */
    boolean reorder() {
        return reorder;
    }

    /** {@code window_idx}: the window-ordered slot of each merged token in raster order. */
    int[] idx() {
        return idx;
    }

    /** {@code inv_window_idx}: the raster merged token held by each window-ordered slot. */
    int[] invIdx() {
        return invIdx;
    }

    /** The {@code 4 * nPatches} rope positions in encoder row order. */
    int[] positions() {
        return positions;
    }

    /** Windows of the block-diagonal mask; one window spanning every row when there is no window plan. */
    int windowCount() {
        return windowBounds.length - 1;
    }

    /** First patch row of window {@code w}. */
    int windowBegin(int w) {
        return windowBounds[w];
    }

    /** One past the last patch row of window {@code w}. */
    int windowEnd(int w) {
        return windowBounds[w + 1];
    }

    /** The window holding patch row {@code row}. */
    int windowOfRow(int row) {
        return rowWindow[row];
    }

    /** First key row of the window holding patch row {@code row}. */
    @Override
    public int begin(int row) {
        return windowBounds[rowWindow[row]];
    }

    /** One past the last key row of the window holding patch row {@code row}. */
    @Override
    public int end(int row) {
        return windowBounds[rowWindow[row] + 1];
    }

    /** True when windowed layer {@code il} attends within the windows, false for a full-attention layer. */
    boolean windowed(int il, int nWaPattern) {
        return reorder && (il + 1) % nWaPattern != 0;
    }
}

package com.eonmux.neuralfree.gguf.mm;

/**
 * The token layout of a llava-next "anyres" image (Granite Vision 4.1): the overview tile's {@code side x side}
 * tokens, then the {@code gridX x gridY} tile grid read as one {@code gridX * side x gridY * side} token canvas,
 * cropped to the area the aspect-preserving resize filled ({@code clip_anyres_unpad}, tools/mtmd/clip-impl.h), with
 * one newline row after every canvas row. Static methods only.
 */
public final class AnyresGrid {

    private AnyresGrid() {
    }

    /**
     * The token area of a {@code curW x curH} canvas left after removing the padding the aspect-preserving resize of
     * an {@code origW x origH} image added ({@code clip_anyres_unpad}).
     *
     * @return {@code {offX, offY, width, height}}
     */
    public static int[] unpad(int curW, int curH, int origW, int origH) {
        if ((float) origW / origH > (float) curW / curH) {
            int newH = (int) Math.floor((double) origH * curW / origW + 1e-7);
            int offY = (curH - newH) / 2;
            return new int[]{0, offY, curW, curH - 2 * offY};
        }
        int newW = (int) Math.floor((double) origW * curH / origH + 1e-7);
        int offX = (curW - newW) / 2;
        return new int[]{offX, 0, curW - 2 * offX, curH};
    }

    /**
     * The token count of an anyres image whose tiles each yield {@code side x side} tokens
     * ({@code clip_n_output_tokens} of a tiled entry): {@code side^2 + h * (w + 1)}, {@code w x h} the unpadded
     * canvas.
     *
     * @throws IllegalArgumentException when a size is not positive or the count does not fit an int
     */
    public static int tokens(int side, int gridX, int gridY, int origW, int origH) {
        if (side <= 0 || gridX <= 0 || gridY <= 0 || origW <= 0 || origH <= 0) {
            throw new IllegalArgumentException("anyres tile side " + side + ", grid " + gridX + "x" + gridY
                    + " and original size " + origW + "x" + origH + " must be positive");
        }
        int[] crop = unpad(Math.multiplyExact(gridX, side), Math.multiplyExact(gridY, side), origW, origH);
        long total = (long) side * side + (long) crop[3] * (crop[2] + 1);
        if (total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the anyres image needs " + total + " tokens, more than "
                    + Integer.MAX_VALUE);
        }
        return (int) total;
    }
}

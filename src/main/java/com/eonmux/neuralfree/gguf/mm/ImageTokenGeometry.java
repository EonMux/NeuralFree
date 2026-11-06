package com.eonmux.neuralfree.gguf.mm;

import java.util.Objects;

/**
 * The projector geometry an {@link ImageTokenFormula} reads to count the image tokens of one tile: the vision patch
 * size, the per-side patch merge ({@code n_merge}), the trained image size, the resampler query count and the
 * Granite Vision QFormer window and query sides (llama.cpp {@code clip_hparams}, tools/mtmd/clip-model.h).
 * Immutable.
 */
public final class ImageTokenGeometry {

    private final int patchSize;
    private final int mergeSize;
    private final int imageSize;
    private final int queryNum;
    private final int windowSide;
    private final int querySide;

    private ImageTokenGeometry(int patchSize, int mergeSize, int imageSize, int queryNum, int windowSide,
                               int querySide) {
        this.patchSize = patchSize;
        this.mergeSize = mergeSize;
        this.imageSize = imageSize;
        this.queryNum = queryNum;
        this.windowSide = windowSide;
        this.querySide = querySide;
    }

    /**
     * The geometry of a projector.
     *
     * @param patchSize the vision patch size; positive
     * @param mergeSize the per-side patch merge; positive
     * @param imageSize the trained image size ({@code clip.vision.image_size}); 0 when the file sets none (the
     *                  SenseNova-U1 converter writes 0)
     * @param queryNum  the resampler query count; {@code 0} when the projector has no resampler
     * @throws IllegalArgumentException naming the value that is out of range
     */
    public static ImageTokenGeometry of(int patchSize, int mergeSize, int imageSize, int queryNum) {
        requirePositive("patch size", patchSize);
        requirePositive("merge", mergeSize);
        if (imageSize < 0) {
            throw new IllegalArgumentException("the image size must not be negative, got " + imageSize);
        }
        if (queryNum < 0) {
            throw new IllegalArgumentException("the resampler query count must not be negative, got " + queryNum);
        }
        return new ImageTokenGeometry(patchSize, mergeSize, imageSize, queryNum, 0, 0);
    }

    /**
     * This geometry with the Granite Vision QFormer sides: every {@code windowSide x windowSide} patch window
     * becomes {@code querySide x querySide} tokens ({@code downsample_window_side}, {@code downsample_query_side}).
     *
     * @throws IllegalArgumentException when a side is not positive
     */
    public ImageTokenGeometry withWindows(int windowSide, int querySide) {
        requirePositive("window side", windowSide);
        requirePositive("query side", querySide);
        return new ImageTokenGeometry(patchSize, mergeSize, imageSize, queryNum, windowSide, querySide);
    }

    /** The vision patch size; positive. */
    public int patchSize() {
        return patchSize;
    }

    /** The per-side patch merge; positive. */
    public int mergeSize() {
        return mergeSize;
    }

    /** The trained image size; 0 when the file sets none. */
    public int imageSize() {
        return imageSize;
    }

    /** The resampler query count; {@code 0} when unset. */
    public int queryNum() {
        return queryNum;
    }

    /** The QFormer window side in patches; {@code 0} when unset. */
    public int windowSide() {
        return windowSide;
    }

    /** The QFormer query side in tokens per window side; {@code 0} when unset. */
    public int querySide() {
        return querySide;
    }

    /** {@code patchSize * mergeSize}: the pixel side of one merged output token. */
    public int mergedPatchSize() {
        return patchSize * mergeSize;
    }

    private static void requirePositive(String what, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("the " + what + " must be positive, got " + value);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ImageTokenGeometry)) {
            return false;
        }
        ImageTokenGeometry other = (ImageTokenGeometry) o;
        return patchSize == other.patchSize && mergeSize == other.mergeSize && imageSize == other.imageSize
                && queryNum == other.queryNum && windowSide == other.windowSide && querySide == other.querySide;
    }

    @Override
    public int hashCode() {
        return Objects.hash(patchSize, mergeSize, imageSize, queryNum, windowSide, querySide);
    }

    @Override
    public String toString() {
        return "ImageTokenGeometry{patch=" + patchSize + ", merge=" + mergeSize + ", imageSize=" + imageSize
                + ", queryNum=" + queryNum + ", window=" + windowSide + ", query=" + querySide + "}";
    }
}

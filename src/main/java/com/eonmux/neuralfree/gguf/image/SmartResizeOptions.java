package com.eonmux.neuralfree.gguf.image;

/**
 * The inputs of the aspect-ratio-preserving "smart resize" that computes a dynamic-resolution target size
 * ({@code img_tool::calc_size_preserved_opt}, tools/mtmd/mtmd-image.cpp; {@code smart_resize} of the transformers
 * image processors). The result sides are multiples of {@code alignSize}, the area is pulled into
 * {@code [minPixels, maxPixels]} and, when {@code longestEdge} is set, scaled down to it first.
 *
 * <p>Immutable. A limit of {@code 0} disables that limit, as llama.cpp's options do.</p>
 */
public final class SmartResizeOptions {

    private final int alignSize;
    private final long minPixels;
    private final long maxPixels;
    private final int longestEdge;
    private final boolean roundHalfEven;

    private SmartResizeOptions(Builder b) {
        this.alignSize = b.alignSize;
        this.minPixels = b.minPixels;
        this.maxPixels = b.maxPixels;
        this.longestEdge = b.longestEdge;
        this.roundHalfEven = b.roundHalfEven;
    }

    /**
     * The side multiple; always positive.
     *
     * @throws IllegalArgumentException when {@code alignSize} is not positive
     */
    public static SmartResizeOptions alignTo(int alignSize) {
        return builder(alignSize).build();
    }

    /** A new builder with the given side multiple and all limits disabled. */
    public static Builder builder(int alignSize) {
        return new Builder(alignSize);
    }

    /** The side multiple; positive. */
    public int alignSize() { return alignSize; }

    /** The smallest accepted pixel area, {@code 0} when disabled. */
    public long minPixels() { return minPixels; }

    /** The largest accepted pixel area, {@code 0} when disabled. */
    public long maxPixels() { return maxPixels; }

    /**
     * The longest side the image scales down to before the pixel limits apply, so
     * {@code minPixels} can push an edge back above it; {@code 0} when disabled.
     */
    public int longestEdge() { return longestEdge; }

    /** Whether sides round half to even (Python {@code round}), as the transformers smart resize does. */
    public boolean roundHalfEven() { return roundHalfEven; }

    /** Builder of an immutable {@link SmartResizeOptions}. */
    public static final class Builder {

        private final int alignSize;
        private long minPixels = 0;
        private long maxPixels = 0;
        private int longestEdge = 0;
        private boolean roundHalfEven = false;

        private Builder(int alignSize) {
            if (alignSize <= 0) {
                throw new IllegalArgumentException("alignSize must be positive, got " + alignSize);
            }
            this.alignSize = alignSize;
        }

        /** Sets the smallest accepted pixel area; negative values are refused. */
        public Builder minPixels(long minPixels) {
            if (minPixels < 0) throw new IllegalArgumentException("minPixels must be non-negative, got " + minPixels);
            this.minPixels = minPixels;
            return this;
        }

        /** Sets the largest accepted pixel area; negative values are refused. */
        public Builder maxPixels(long maxPixels) {
            if (maxPixels < 0) throw new IllegalArgumentException("maxPixels must be non-negative, got " + maxPixels);
            this.maxPixels = maxPixels;
            return this;
        }

        /** Sets the longest-side downscale; negative values are refused. */
        public Builder longestEdge(int longestEdge) {
            if (longestEdge < 0) throw new IllegalArgumentException("longestEdge must be non-negative, got " + longestEdge);
            this.longestEdge = longestEdge;
            return this;
        }

        /** Sets whether sides round half to even instead of half away from zero. */
        public Builder roundHalfEven(boolean roundHalfEven) {
            this.roundHalfEven = roundHalfEven;
            return this;
        }

        public SmartResizeOptions build() {
            return new SmartResizeOptions(this);
        }
    }
}

package com.eonmux.neuralfree.gguf.image;

import java.util.Objects;

/**
 * Image size in pixels, the {@code clip_image_size} of llama.cpp (tools/mtmd/clip-impl.h). Width and height are
 * non-negative; {@code 0} marks the "no value" size the slice planners use for an empty refined size or an empty
 * grid, not a real image.
 *
 * <p>Immutable.</p>
 */
public final class ImageSize {

    /** The size with both sides {@code 0}. */
    public static final ImageSize EMPTY = new ImageSize(0, 0);

    private final int width;
    private final int height;

    /**
     * Creates a size.
     *
     * @throws IllegalArgumentException when a side is negative
     */
    public ImageSize(int width, int height) {
        if (width < 0) throw new IllegalArgumentException("width must be non-negative, got " + width);
        if (height < 0) throw new IllegalArgumentException("height must be non-negative, got " + height);
        this.width = width;
        this.height = height;
    }

    public int width() { return width; }

    public int height() { return height; }

    /** {@code width * height}. */
    public int area() { return width * height; }

    /** Whether both sides are {@code 0}. */
    public boolean isEmpty() { return width == 0 && height == 0; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ImageSize)) return false;
        ImageSize that = (ImageSize) o;
        return width == that.width && height == that.height;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height);
    }

    @Override
    public String toString() {
        return width + "x" + height;
    }
}

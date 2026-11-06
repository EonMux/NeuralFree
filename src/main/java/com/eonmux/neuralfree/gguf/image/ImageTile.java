package com.eonmux.neuralfree.gguf.image;

/**
 * One prepared tile a vision encoder consumes: its size and its pixels as the normalized floats of llama.cpp's
 * {@code clip_image_f32} (interleaved R, G, B, row-major, each channel {@code (v / 255 - mean) / std}). Most
 * preprocessors produce 8-bit {@link RasterImage} tiles; those that resample after normalizing produce
 * {@link FloatImage} tiles, which hold their normalized samples. Implementations are immutable.
 */
public interface ImageTile {

    /** The width in pixels; positive. */
    int width();

    /** The height in pixels; positive. */
    int height();

    /** The {@link ImageSize} of this tile. */
    default ImageSize size() {
        return new ImageSize(width(), height());
    }

    /**
     * The pixels normalized per channel, {@code (v / 255 - mean) / std}.
     *
     * @throws IllegalArgumentException when mean or std has not exactly 3 entries or a std entry is zero, or, for a
     *                                  tile normalized during preprocessing, when they are not the pair it was
     *                                  normalized with
     */
    float[] normalized(float[] mean, float[] std);
}

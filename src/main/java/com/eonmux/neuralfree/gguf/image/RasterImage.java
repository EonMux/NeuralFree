package com.eonmux.neuralfree.gguf.image;

import java.util.Arrays;
import java.util.Objects;

/**
 * A decoded RGB image, the {@code clip_image_u8} of llama.cpp (tools/mtmd/clip-impl.h): the pixels are one
 * {@code byte} per channel, interleaved R, G, B in row-major order. Produced by {@link ImageDecoder} and by the
 * resize, crop and slice operations of {@link ImageOps}; the preprocessors consume it and produce normalized
 * floats through {@code PreparedImages}.
 *
 * <p>Immutable: the constructor copies the pixel array in, and no operation writes it back.</p>
 */
public final class RasterImage implements ImageTile {

    private final int width;
    private final int height;
    private final byte[] rgb;

    /**
     * Wraps a copy of {@code rgb} ({@code 3} bytes per pixel, row-major, R G B).
     *
     * @throws IllegalArgumentException for negative or zero-when-nonempty dimensions or a wrong array length
     */
    public RasterImage(int width, int height, byte[] rgb) {
        if (width <= 0) throw new IllegalArgumentException("width must be positive, got " + width);
        if (height <= 0) throw new IllegalArgumentException("height must be positive, got " + height);
        Objects.requireNonNull(rgb, "rgb");
        if (rgb.length != (long) width * height * 3) {
            throw new IllegalArgumentException("rgb length is " + rgb.length + " but " + width + "x" + height
                    + " needs " + ((long) width * height * 3));
        }
        this.width = width;
        this.height = height;
        this.rgb = rgb.clone();
    }

    private RasterImage(int width, int height, byte[] rgb, boolean unused) {
        this.width = width;
        this.height = height;
        this.rgb = rgb;
    }

    /** A solid image of the given color, the pad canvas builder of {@link ImageOps}. */
    public static RasterImage filled(int width, int height, int r, int g, int b) {
        if (width <= 0) throw new IllegalArgumentException("width must be positive, got " + width);
        if (height <= 0) throw new IllegalArgumentException("height must be positive, got " + height);
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            rgb[i * 3] = (byte) r;
            rgb[i * 3 + 1] = (byte) g;
            rgb[i * 3 + 2] = (byte) b;
        }
        return new RasterImage(width, height, rgb, false);
    }

    @Override
    public int width() { return width; }

    @Override
    public int height() { return height; }

    /** The {@link ImageSize} of this image. */
    @Override
    public ImageSize size() { return new ImageSize(width, height); }

    /** The red channel at {@code (x, y)}, in {@code [0, 255]}. */
    public int red(int x, int y) { return rgb[(y * width + x) * 3] & 0xFF; }

    /** The green channel at {@code (x, y)}, in {@code [0, 255]}. */
    public int green(int x, int y) { return rgb[(y * width + x) * 3 + 1] & 0xFF; }

    /** The blue channel at {@code (x, y)}, in {@code [0, 255]}. */
    public int blue(int x, int y) { return rgb[(y * width + x) * 3 + 2] & 0xFF; }

    /** The pixel at {@code (x, y)} as {@code 0xRRGGBB}. */
    public int pixel(int x, int y) {
        int i = (y * width + x) * 3;
        return (rgb[i] & 0xFF) << 16 | (rgb[i + 1] & 0xFF) << 8 | (rgb[i + 2] & 0xFF);
    }

    /** The interleaved R, G, B bytes as a copy, so the caller cannot mutate this image. */
    public byte[] rgb() { return rgb.clone(); }

    /** The interleaved R, G, B bytes without a copy; package-private, read-only use. */
    byte[] rgbArray() { return rgb; }

    /**
     * The pixels as normalized floats in the same interleaved R, G, B layout: each channel becomes
     * {@code (value / 255 - mean) / std}, the {@code clip_image_f32} conversion and normalize of llama.cpp
     * (tools/mtmd/mtmd-image.cpp).
     *
     * @throws IllegalArgumentException when mean or std has not exactly 3 entries or a std entry is zero
     */
    @Override
    public float[] normalized(float[] mean, float[] std) {
        requireMeanStd(mean, std);
        float[] out = new float[rgb.length];
        for (int i = 0; i < rgb.length; i++) {
            float v = (rgb[i] & 0xFF) / 255.0f;
            out[i] = (v - mean[i % 3]) / std[i % 3];
        }
        return out;
    }

    static void requireMeanStd(float[] mean, float[] std) {
        if (mean == null || mean.length != 3) {
            throw new IllegalArgumentException("image mean must hold 3 values, got "
                    + (mean == null ? "null" : mean.length));
        }
        if (std == null || std.length != 3) {
            throw new IllegalArgumentException("image std must hold 3 values, got "
                    + (std == null ? "null" : std.length));
        }
        for (int i = 0; i < 3; i++) {
            if (std[i] == 0.0f) {
                throw new IllegalArgumentException("image std[" + i + "] is zero");
            }
        }
    }

    /** Builds images from caller-owned pixels without a copy, inside the package. */
    static RasterImage wrap(int width, int height, byte[] rgb) {
        return new RasterImage(width, height, rgb, false);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RasterImage)) return false;
        RasterImage that = (RasterImage) o;
        return width == that.width && height == that.height && Arrays.equals(rgb, that.rgb);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(width, height);
        result = 31 * result + Arrays.hashCode(rgb);
        return result;
    }

    @Override
    public String toString() {
        return "RasterImage(" + width + "x" + height + ")";
    }
}

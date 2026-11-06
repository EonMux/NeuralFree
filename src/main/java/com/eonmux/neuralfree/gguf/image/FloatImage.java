package com.eonmux.neuralfree.gguf.image;

import java.util.Arrays;
import java.util.Objects;

/**
 * An RGB image of normalized float samples (interleaved R, G, B, row-major), the {@code clip_image_f32} of a
 * preprocessor that resamples after normalizing, such as the Step3-VL bilinear resize of tools/mtmd/mtmd-image.cpp,
 * which interpolates the normalized corner samples. The image remembers the per-channel mean and standard deviation
 * it was normalized with, and only hands its samples out for those.
 *
 * <p>Immutable: the constructor copies the arrays in.</p>
 */
public final class FloatImage implements ImageTile {

    private final int width;
    private final int height;
    private final float[] samples;
    private final float[] mean;
    private final float[] std;

    /**
     * Wraps copies of the normalized samples ({@code 3} floats per pixel, row-major, R G B) and of the mean and
     * standard deviation they were normalized with.
     *
     * @throws IllegalArgumentException for a non-positive side, a wrong array length, or a mean or standard deviation
     *                                  that is not 3 values with a non-zero deviation
     */
    public FloatImage(int width, int height, float[] samples, float[] mean, float[] std) {
        this(Objects.requireNonNull(samples, "samples").clone(), copy(mean), copy(std), width, height);
    }

    private FloatImage(float[] samples, float[] mean, float[] std, int width, int height) {
        if (width <= 0) throw new IllegalArgumentException("width must be positive, got " + width);
        if (height <= 0) throw new IllegalArgumentException("height must be positive, got " + height);
        if (samples.length != (long) width * height * 3) {
            throw new IllegalArgumentException("samples length is " + samples.length + " but " + width + "x"
                    + height + " needs " + ((long) width * height * 3));
        }
        RasterImage.requireMeanStd(mean, std);
        this.width = width;
        this.height = height;
        this.samples = samples;
        this.mean = mean;
        this.std = std;
    }

    /** Builds an image from caller-owned arrays without a copy, inside the package. */
    static FloatImage wrap(int width, int height, float[] samples, float[] mean, float[] std) {
        return new FloatImage(samples, mean, std, width, height);
    }

    @Override
    public int width() { return width; }

    @Override
    public int height() { return height; }

    /** The normalized sample of channel {@code c} (0 red, 1 green, 2 blue) at {@code (x, y)}. */
    public float sample(int x, int y, int c) {
        return samples[(y * width + x) * 3 + c];
    }

    /**
     * A copy of the samples.
     *
     * @throws IllegalArgumentException when the mean or standard deviation is not the pair the image was normalized
     *                                  with: its samples cannot be renormalized without the rounding the reference
     *                                  avoided
     */
    @Override
    public float[] normalized(float[] mean, float[] std) {
        RasterImage.requireMeanStd(mean, std);
        if (!Arrays.equals(this.mean, mean) || !Arrays.equals(this.std, std)) {
            throw new IllegalArgumentException("this image was normalized with mean " + Arrays.toString(this.mean)
                    + " and std " + Arrays.toString(this.std) + ", not mean " + Arrays.toString(mean) + " and std "
                    + Arrays.toString(std));
        }
        return samples.clone();
    }

    private static float[] copy(float[] values) {
        return values == null ? null : values.clone();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FloatImage)) return false;
        FloatImage that = (FloatImage) o;
        return width == that.width && height == that.height && Arrays.equals(samples, that.samples)
                && Arrays.equals(mean, that.mean) && Arrays.equals(std, that.std);
    }

    @Override
    public int hashCode() {
        int h = 31 * Objects.hash(width, height) + Arrays.hashCode(samples);
        return 31 * (31 * h + Arrays.hashCode(mean)) + Arrays.hashCode(std);
    }

    @Override
    public String toString() {
        return "FloatImage(" + width + "x" + height + ")";
    }
}

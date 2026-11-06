package com.eonmux.neuralfree.gguf.image;

import java.util.Objects;

/**
 * How {@link ImageDecoder} turns decoded pixels into the RGB raster the image preprocessors read: vLLM's image
 * loader options ({@code ImageMediaIO}: {@code image_mode} and {@code rgba_background_color}, set through
 * {@code --media-io-kwargs}). In {@link Mode#RGB} mode a transparent image is composited over the background
 * colour (vLLM's {@code rgba_to_rgb}); in {@link Mode#L} mode the pixels become their luma (PIL's
 * {@code convert("L")}, which ignores the alpha channel) on all three channels, as an image processor's
 * {@code convert_rgb} widens a grayscale image. Immutable.
 */
public final class ImageLoadOptions {

    /** The colour mode an image is loaded in (vLLM's {@code image_mode}). */
    public enum Mode {
        /** Red, green and blue, transparency composited over the background colour (vLLM's default). */
        RGB,
        /** 8-bit luma, ITU-R 601-2 weights, read back as three equal channels. */
        L
    }

    /** vLLM's default background colour of transparent pixels: white. */
    public static final int WHITE = 0xFFFFFF;

    private static final ImageLoadOptions DEFAULTS = new ImageLoadOptions(Mode.RGB, WHITE);

    private final Mode mode;
    private final int background;

    private ImageLoadOptions(Mode mode, int background) {
        this.mode = mode;
        this.background = background;
    }

    /** RGB mode over a white background, vLLM's defaults. */
    public static ImageLoadOptions defaults() {
        return DEFAULTS;
    }

    /**
     * Options of {@code mode} over the {@code background} colour {@code 0xRRGGBB}.
     *
     * @throws IllegalArgumentException when the background holds bits beyond 24
     */
    public static ImageLoadOptions of(Mode mode, int background) {
        Objects.requireNonNull(mode, "mode");
        if ((background & ~0xFFFFFF) != 0) {
            throw new IllegalArgumentException("the background colour is 0xRRGGBB, got 0x"
                    + Integer.toHexString(background));
        }
        return new ImageLoadOptions(mode, background);
    }

    /** The colour mode. */
    public Mode mode() {
        return mode;
    }

    /** The background colour transparent pixels composite over in RGB mode, {@code 0xRRGGBB}. */
    public int background() {
        return background;
    }

    /** PIL's {@code L} value of an RGB pixel: {@code (19595 R + 38470 G + 7471 B + 0x8000) >> 16}. */
    static int luma(int r, int g, int b) {
        return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
    }

    /** One channel of {@code c} at alpha {@code a} over the background channel {@code bg}, rounded. */
    static int over(int c, int a, int bg) {
        return (c * a + bg * (255 - a) + 127) / 255;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ImageLoadOptions)) {
            return false;
        }
        ImageLoadOptions other = (ImageLoadOptions) o;
        return mode == other.mode && background == other.background;
    }

    @Override
    public int hashCode() {
        return mode.hashCode() * 31 + background;
    }

    @Override
    public String toString() {
        return "ImageLoadOptions{" + mode + ", background=#" + String.format("%06x", background) + '}';
    }
}

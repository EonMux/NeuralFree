package com.eonmux.neuralfree.gguf.mm;

import java.util.Arrays;
import java.util.Objects;

/**
 * A preprocessed image for the vision encoder (the counterpart of llama.cpp {@code clip_image_f32}):
 * already resized and normalized by the caller, so the encoder sees only the forward graphs. Pixels are
 * interleaved RGB floats, value {@code c} of the pixel at {@code (x, y)} stored at index
 * {@code (y * width + x) * 3 + c}. Flags of {@code clip_image_f32} travel with it: the leading pad rows of a
 * DeepSeek-V4 token block ({@code lead_pad}), whether a DeepSeek-OCR global view ends with the view separator
 * row ({@code add_viewsep}) and the separator rows asked after its tokens ({@link #suffix()},
 * {@code clip_image_f32.suffix_type}). A frame of a temporal pair carries the video's next frame as its
 * {@link #secondFrame() second frame}: the two frames share one patch grid, and the qwen-style encoders with two
 * Conv3d patch kernels run one kernel over each frame and add ({@code build_inp_temporal_merge} with a batch of
 * two frames). Immutable; the pixel arrays are copied at construction.
 */
public final class ClipImage {

    private final int width;
    private final int height;
    private final float[] pixels;
    private final float[] secondFrame;
    private final int leadPad;
    private final boolean viewSeparator;
    private final ClipSuffix suffix;

    private ClipImage(int width, int height, float[] pixels, float[] secondFrame, int leadPad,
                      boolean viewSeparator, ClipSuffix suffix) {
        this.width = width;
        this.height = height;
        this.pixels = pixels;
        this.secondFrame = secondFrame;
        this.leadPad = leadPad;
        this.viewSeparator = viewSeparator;
        this.suffix = suffix;
    }

    /**
     * An image over {@code pixels}.
     *
     * @throws IllegalArgumentException for a non-positive size, or a pixel array that does not hold exactly
     *                                  {@code width * height * 3} values
     */
    public static ClipImage of(int width, int height, float[] pixels) {
        Objects.requireNonNull(pixels, "pixels");
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("image size must be positive, got " + width + "x" + height);
        }
        long need = (long) width * height * 3;
        if (pixels.length != need) {
            throw new IllegalArgumentException("an image of " + width + "x" + height + " holds " + need
                    + " values, got " + pixels.length);
        }
        return new ClipImage(width, height, pixels.clone(), null, 0, false, ClipSuffix.NONE);
    }

    /**
     * This image with the second frame of a temporal pair over {@code secondFrame}: the two frames share the
     * size and the patch grid, and the encoders with two patch kernels run one over each. Only the encoders with
     * a second kernel (the qwen-style temporal merge) accept a pair; the other graphs refuse it.
     *
     * @throws IllegalArgumentException when the second frame does not hold exactly {@code width * height * 3}
     *                                  values
     */
    public ClipImage withSecondFrame(float[] secondFrame) {
        Objects.requireNonNull(secondFrame, "secondFrame");
        long need = (long) width * height * 3;
        if (secondFrame.length != need) {
            throw new IllegalArgumentException("a second frame of " + width + "x" + height + " holds " + need
                    + " values, got " + secondFrame.length);
        }
        return new ClipImage(width, height, pixels, secondFrame.clone(), leadPad, viewSeparator, suffix);
    }

    /**
     * This image with {@code leadPad} leading pad rows in its token block ({@code clip_image_f32::lead_pad},
     * read by the DeepSeek-V4 encoder only).
     *
     * @throws IllegalArgumentException for a negative count
     */
    public ClipImage withLeadPad(int leadPad) {
        if (leadPad < 0) {
            throw new IllegalArgumentException("the lead pad must not be negative, got " + leadPad);
        }
        return new ClipImage(width, height, pixels, secondFrame, leadPad, viewSeparator, suffix);
    }

    /**
     * This image marked as a global view whose rows end with the view separator
     * ({@code clip_image_f32::add_viewsep}, read by the DeepSeek-OCR encoders only).
     */
    public ClipImage withViewSeparator(boolean viewSeparator) {
        return new ClipImage(width, height, pixels, secondFrame, leadPad, viewSeparator, suffix);
    }

    /**
     * This image with the separator rows {@code suffix} after its tokens. Only the MiniCPM-V 4.6 merger graphs
     * append them; the other encoders refuse an image with a suffix.
     */
    public ClipImage withSuffix(ClipSuffix suffix) {
        return new ClipImage(width, height, pixels, secondFrame, leadPad, viewSeparator,
                Objects.requireNonNull(suffix, "suffix"));
    }

    /** The leading pad rows of the DeepSeek-V4 token block; 0 by default. */
    public int leadPad() {
        return leadPad;
    }

    /** Whether the DeepSeek-OCR rows end with the view separator; false by default. */
    public boolean viewSeparator() {
        return viewSeparator;
    }

    /** Width in pixels. */
    public int width() {
        return width;
    }

    /** Height in pixels. */
    public int height() {
        return height;
    }

    /** The interleaved RGB values, {@code (y * width + x) * 3 + c} (a copy). */
    public float[] pixels() {
        return pixels.clone();
    }

    /** The second frame of a temporal pair (a copy), or null for a single-frame image. */
    public float[] secondFrame() {
        return secondFrame == null ? null : secondFrame.clone();
    }

    /** Whether the image carries the second frame of a temporal pair. */
    public boolean isPair() {
        return secondFrame != null;
    }

    /** The separator rows after the image's tokens; {@link ClipSuffix#NONE} unless set. */
    public ClipSuffix suffix() {
        return suffix;
    }

    /** The channel {@code c} value of pixel {@code (x, y)}. */
    public float pixel(int x, int y, int c) {
        if (x < 0 || x >= width || y < 0 || y >= height || c < 0 || c > 2) {
            throw new IllegalArgumentException("pixel (" + x + ", " + y + ", " + c + ") is outside the image");
        }
        return pixels[(y * width + x) * 3 + c];
    }

    @Override
    public String toString() {
        return "ClipImage{" + width + "x" + height + ", " + Arrays.hashCode(pixels)
                + (isPair() ? ", pair" : "") + (leadPad > 0 ? ", leadPad=" + leadPad : "")
                + (viewSeparator ? ", viewSeparator" : "")
                + (suffix == ClipSuffix.NONE ? "" : ", suffix=" + suffix) + "}";
    }
}

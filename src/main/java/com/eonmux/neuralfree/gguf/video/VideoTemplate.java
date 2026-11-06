package com.eonmux.neuralfree.gguf.video;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Locale;

/**
 * How the frames and timestamps of one video are laid out in the prompt, the {@code mtmd_video_template} of
 * llama.cpp (tools/mtmd/mtmd.h): a {@link #DEFAULT} video opens with the {@code "Video:"} prefix and stamps
 * {@code [XmY.YYs]} text every interval; a {@link #GROUP} video stamps {@code <X.X seconds>} before every temporal
 * group's frames, the midpoint time of the group (the LFM3-VL template and the HF Qwen3-VL processors).
 */
public enum VideoTemplate {

    /** The {@code "Video:"} prefix, then an {@code [XmY.YYs]} timestamp every interval (mtmd's default). */
    DEFAULT,

    /** An {@code <X.X seconds>} timestamp before every temporal group (mtmd's LFM3-VL template). */
    GROUP;

    /** The text before the first frame of a {@link #DEFAULT} video ({@code prompt_start} of mtmd-helper). */
    public String videoPrefix() {
        return "Video:";
    }

    /**
     * The {@code [XmY.YYs]} stamp of a {@link #DEFAULT} frame at {@code seconds} into the video, the two decimals
     * rounded half to even as C's {@code snprintf} and the HF processors format.
     *
     * @throws IllegalArgumentException when called on the {@link #GROUP} template or the seconds are negative
     */
    public String frameTimestamp(double seconds) {
        requireDefault(seconds);
        BigDecimal rounded = new BigDecimal(seconds).setScale(2, RoundingMode.HALF_EVEN);
        int minutes = rounded.intValue() / 60;
        double rest = rounded.subtract(BigDecimal.valueOf(minutes * 60)).doubleValue();
        return String.format(Locale.ROOT, "[%dm%.2fs]", minutes, rest);
    }

    /**
     * The {@code <X.X seconds>} stamp before a {@link #GROUP} temporal group at its midpoint {@code seconds}, the
     * one decimal rounded half to even over the seconds' binary value, as C's {@code snprintf} and the HF
     * formatters print it.
     *
     * @throws IllegalArgumentException when called on the {@link #DEFAULT} template or the seconds are negative
     */
    public String groupTimestamp(double seconds) {
        requireGroup(seconds);
        return "<" + new BigDecimal(seconds).setScale(1, RoundingMode.HALF_EVEN).toPlainString()
                + " seconds>";
    }

    private void requireDefault(double seconds) {
        if (this != DEFAULT) {
            throw new IllegalArgumentException("the " + name() + " video template has no per-frame [XmY.YYs]"
                    + " timestamps");
        }
        if (seconds < 0) {
            throw new IllegalArgumentException("a frame timestamp needs a non-negative time, got " + seconds);
        }
    }

    private void requireGroup(double seconds) {
        if (this != GROUP) {
            throw new IllegalArgumentException("the " + name() + " video template has no per-group <X.X seconds>"
                    + " timestamps");
        }
        if (seconds < 0) {
            throw new IllegalArgumentException("a group timestamp needs a non-negative time, got " + seconds);
        }
    }
}

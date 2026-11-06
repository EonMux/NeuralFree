package com.eonmux.neuralfree.gguf.video;

import java.util.Map;
import java.util.Objects;

/**
 * The stream metadata of one probed video, the fields {@code ffprobe} reports and llama.cpp's
 * {@code mtmd_helper_bitmap} probe reads (tools/mtmd/mtmd-helper.cpp): width, height, frame rate, frame count,
 * duration and the stream's start time (the offset the LFM3-Omni soundtrack is measured from). A probe may report
 * no frame rate (0), the variable or unknown rate of a container without one; the projector family's
 * {@link VideoConfig} then assumes the rate its reference does or refuses the source, and the plan it builds holds
 * the resolved rate. A frame count may be derived from the duration and a known rate. Immutable.
 */
public final class VideoMeta {

    private final int width;
    private final int height;
    private final double fps;
    private final int nFrames;
    private final double duration;
    private final double startTime;

    private VideoMeta(int width, int height, double fps, int nFrames, double duration, double startTime) {
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.nFrames = nFrames;
        this.duration = duration;
        this.startTime = startTime;
    }

    /**
     * The metadata of a video of {@code width} x {@code height} pixels at {@code fps} frames per second with
     * {@code nFrames} frames over {@code duration} seconds starting at 0, the values ffprobe reports; a rate of 0
     * is the unknown rate of a stream that reports none.
     *
     * @throws IllegalArgumentException when a size is not positive, the frame rate is negative or not finite, the
     *                                  duration is negative or not finite, or the frame count is not positive
     */
    public static VideoMeta of(int width, int height, double fps, int nFrames, double duration) {
        return of(width, height, fps, nFrames, duration, 0);
    }

    /**
     * {@link #of(int, int, double, int, double)} with the stream's {@code startTime} seconds offset.
     *
     * @throws IllegalArgumentException when a size is not positive, the frame rate is negative or not finite, the
     *                                  duration is negative or not finite, the frame count is not positive, or the
     *                                  start time is not finite
     */
    public static VideoMeta of(int width, int height, double fps, int nFrames, double duration, double startTime) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("a video needs a positive size, got " + width + "x" + height);
        }
        if (!(fps >= 0) || Double.isNaN(fps) || Double.isInfinite(fps)) {
            throw new IllegalArgumentException("a video needs a non-negative frame rate, got " + fps);
        }
        if (duration < 0 || Double.isNaN(duration) || Double.isInfinite(duration)) {
            throw new IllegalArgumentException("a video needs a non-negative duration, got " + duration);
        }
        if (nFrames <= 0) {
            throw new IllegalArgumentException("a video needs a positive frame count, got " + nFrames);
        }
        if (Double.isNaN(startTime) || Double.isInfinite(startTime)) {
            throw new IllegalArgumentException("a video needs a finite start time, got " + startTime);
        }
        return new VideoMeta(width, height, fps, nFrames, duration, startTime);
    }

    /**
     * The metadata of the probe fields {@code ffprobe} printed ({@code width}, {@code height},
     * {@code r_frame_rate}, {@code nb_frames}, {@code duration}, {@code start_time}); a missing or {@code N/A}
     * frame count comes from the duration and a known rate, a missing duration from the frame count and a known
     * rate, a missing or {@code N/A} rate is 0 (unknown) and a missing start time is 0.
     *
     * @throws IllegalArgumentException when no width or height is reported, no frame count can be counted (neither
     *                                  {@code nb_frames} nor a duration beside a known rate), or the values do not
     *                                  parse
     */
    public static VideoMeta ofProbe(Map<String, String> fields) {
        Objects.requireNonNull(fields, "fields");
        int width = intOf(fields, "width");
        int height = intOf(fields, "height");
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("the video stream reports a " + width + "x" + height
                    + " frame size; both sides must be positive");
        }
        double fps = rationalOf(fields, "r_frame_rate");
        if (!(fps > 0) || Double.isInfinite(fps)) {
            fps = 0; // no usable rate: the container is variable, unknown or silent about it
        }
        int nFrames = intOr(fields, "nb_frames", -1);
        double duration = doubleOr(fields, "duration", -1);
        if (!(duration >= 0) || Double.isInfinite(duration)) {
            duration = -1; // missing or unusable
        }
        double startTime = doubleOr(fields, "start_time", 0);
        if (nFrames <= 0 && duration > 0 && fps > 0) {
            nFrames = (int) Math.round(duration * fps);
        }
        if (duration < 0 && nFrames > 0 && fps > 0) {
            duration = nFrames / fps;
        }
        if (nFrames <= 0) {
            throw new IllegalArgumentException(fps > 0
                    ? "the video stream reports neither a frame count (nb_frames) nor a duration, so the frames to"
                            + " sample cannot be counted"
                    : "the video stream reports neither a frame count (nb_frames) nor a frame rate (r_frame_rate),"
                            + " so the frames to sample cannot be counted");
        }
        if (duration < 0) {
            duration = 0; // unknown: the frame count and the rate still stand
        }
        return new VideoMeta(width, height, fps, nFrames, duration, startTime);
    }

    private static int intOf(Map<String, String> fields, String key) {
        return intOr(fields, key, -1);
    }

    private static int intOr(Map<String, String> fields, String key, int fallback) {
        String raw = fields.get(key);
        if (raw == null || raw.isBlank() || "N/A".equals(raw)) {
            return fallback;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("the video stream field " + key + " is '" + raw + "', not a whole"
                    + " number");
        }
    }

    private static double doubleOr(Map<String, String> fields, String key, double fallback) {
        String raw = fields.get(key);
        if (raw == null || raw.isBlank() || "N/A".equals(raw)) {
            return fallback;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("the video stream field " + key + " is '" + raw + "', not a number");
        }
    }

    /** {@code num/den} of an ffprobe rational such as {@code 30000/1001}, or a plain number; 0 when neither. */
    private static double rationalOf(Map<String, String> fields, String key) {
        String raw = fields.get(key);
        if (raw == null || raw.isBlank() || "N/A".equals(raw)) {
            return 0;
        }
        int slash = raw.indexOf('/');
        if (slash > 0) {
            try {
                double den = Double.parseDouble(raw.substring(slash + 1).trim());
                if (den > 0) {
                    return Double.parseDouble(raw.substring(0, slash).trim()) / den;
                }
                return 0;
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("the video stream field " + key + " is '" + raw + "', not a"
                        + " num/den rational");
            }
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("the video stream field " + key + " is '" + raw + "', not a number");
        }
    }

    /** Frame width in pixels. */
    public int width() {
        return width;
    }

    /** Frame height in pixels. */
    public int height() {
        return height;
    }

    /** Frames per second of the stream; 0 when the probe reports none (a variable or unknown rate). */
    public double fps() {
        return fps;
    }

    /** Whether the stream reports a frame rate (a probe may report none). */
    public boolean hasFrameRate() {
        return fps > 0;
    }

    /**
     * This metadata at {@code fps}, the rate a projector family assumes for a stream that reports none; the
     * duration, frame count and start time stay.
     *
     * @throws IllegalArgumentException when the rate is not positive or not finite
     */
    public VideoMeta withFps(double fps) {
        if (!(fps > 0) || Double.isNaN(fps) || Double.isInfinite(fps)) {
            throw new IllegalArgumentException("a video needs a positive frame rate, got " + fps);
        }
        return new VideoMeta(width, height, fps, nFrames, duration, startTime);
    }

    /** The frame count of the stream. */
    public int nFrames() {
        return nFrames;
    }

    /** The duration in seconds. */
    public double duration() {
        return duration;
    }

    /** The stream's start time in seconds (ffprobe {@code start_time}); 0 when it reports none. */
    public double startTime() {
        return startTime;
    }

    /**
     * The time in seconds of source frame {@code index}.
     *
     * @throws IllegalArgumentException when the index is outside the stream
     * @throws IllegalStateException    when the stream reports no frame rate, so its frame times are unknown
     */
    public double timeOf(int index) {
        if (index < 0 || index >= nFrames) {
            throw new IllegalArgumentException("frame index " + index + " is outside the " + nFrames + " frames of"
                    + " the video");
        }
        if (fps <= 0) {
            throw new IllegalStateException("the video stream reports no frame rate, so frame " + index + " has no"
                    + " known time");
        }
        return index / fps;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof VideoMeta)) {
            return false;
        }
        VideoMeta m = (VideoMeta) o;
        return width == m.width && height == m.height && nFrames == m.nFrames && fps == m.fps
                && duration == m.duration && startTime == m.startTime;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height, fps, nFrames, duration, startTime);
    }

    @Override
    public String toString() {
        return "VideoMeta{" + width + "x" + height + ", fps=" + (hasFrameRate() ? String.valueOf(fps) : "unknown")
                + ", frames=" + nFrames + ", duration=" + duration + "s, start=" + startTime + "s}";
    }
}

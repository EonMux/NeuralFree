package com.eonmux.neuralfree.gguf.video;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.image.RasterImage;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One decoded video: its probed metadata, sampling plan, the frames the plan picked, and the soundtrack slices of
 * the temporal groups when the projector family hears one (the LFM3-Omni template). The frames sit in plan order,
 * one per sampled source frame, each at the size the decode produced (the preprocessor's recommended frame size
 * when it names one, the probed size otherwise). Immutable.
 */
public final class DecodedVideo {

    private final VideoMeta meta;
    private final VideoSamplePlan plan;
    private final List<RasterImage> frames;
    private final List<AudioClip> soundtrack;

    /**
     * A decoded video over {@code frames}, the plan's sampled frames, without a soundtrack.
     *
     * @throws IllegalArgumentException when the frames do not match the plan's count
     */
    public DecodedVideo(VideoMeta meta, VideoSamplePlan plan, List<RasterImage> frames) {
        this(meta, plan, frames, Collections.emptyList());
    }

    /**
     * A decoded video over {@code frames} and the soundtrack slice of every temporal group, in group order; the
     * slices are empty when the video has no soundtrack.
     *
     * @throws IllegalArgumentException when the frames do not match the plan's count, a frame does not match the
     *                                  probed stream's size, or the soundtrack does not hold one clip per group
     */
    public DecodedVideo(VideoMeta meta, VideoSamplePlan plan, List<RasterImage> frames, List<AudioClip> soundtrack) {
        this.meta = Objects.requireNonNull(meta, "meta");
        this.plan = Objects.requireNonNull(plan, "plan");
        this.frames = List.copyOf(Objects.requireNonNull(frames, "frames"));
        this.soundtrack = List.copyOf(Objects.requireNonNull(soundtrack, "soundtrack"));
        if (frames.size() != plan.nFrames()) {
            throw new IllegalArgumentException(frames.size() + " decoded frames are not the " + plan.nFrames()
                    + " the plan samples");
        }
        for (int i = 0; i < frames.size(); i++) {
            RasterImage frame = frames.get(i);
            if (frame.width() <= 0 || frame.height() <= 0) {
                throw new IllegalArgumentException("a decoded frame of " + frame.width() + "x" + frame.height()
                        + " is empty");
            }
            if (frame.width() != frames.get(0).width() || frame.height() != frames.get(0).height()) {
                throw new IllegalArgumentException("decoded frame " + i + " of " + frame.width() + "x"
                        + frame.height() + " does not match the " + frames.get(0).width() + "x"
                        + frames.get(0).height() + " of the first frame");
            }
        }
        if (!soundtrack.isEmpty() && soundtrack.size() != plan.groups()) {
            throw new IllegalArgumentException("a soundtrack of " + soundtrack.size() + " slices is not the "
                    + plan.groups() + " temporal groups the plan forms");
        }
    }

    /** The probed stream metadata. */
    public VideoMeta meta() {
        return meta;
    }

    /** The plan the frames follow. */
    public VideoSamplePlan plan() {
        return plan;
    }

    /** The sampled frames in plan order. */
    public List<RasterImage> frames() {
        return frames;
    }

    /**
     * The soundtrack slice of every temporal group, in group order, empty when the video has no soundtrack; a
     * group whose span carries no samples has an empty clip.
     */
    public List<AudioClip> soundtrack() {
        return soundtrack;
    }

    @Override
    public String toString() {
        return "DecodedVideo{" + frames.size() + " frames of " + meta.width() + "x" + meta.height()
                + (soundtrack.isEmpty() ? "" : ", " + soundtrack.size() + " soundtrack slices") + "}";
    }
}

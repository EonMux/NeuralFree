package com.eonmux.neuralfree.gguf.video;

import java.util.Arrays;
import java.util.Objects;

/**
 * One video's sampling plan: the source frame indices to decode, the sampling rate the frames play at, the
 * temporal groups they merge into and the timestamp of every group. The indices are
 * {@code round(linspace(0, n - 1, count))}, ties to even, the sampling of the HF video processors
 * ({@code np.linspace(...).round()}) and llama.cpp's {@code nearbyint} picks (tools/mtmd/mtmd-helper.cpp), or the
 * Qwen2-VL loader's float32 {@code torch.arange(0, n, n / count)} indices (vLLM's
 * {@code Qwen2VLVideoBackend}); a group's timestamp is the midpoint time of its first and last frame
 * ({@code _calculate_timestamps} of the HF Qwen3-VL processor, mtmd's LFM3 template). The source holds the frame
 * rate its projector family resolved, so every source time is known. Immutable.
 */
public final class VideoSamplePlan {

    private final VideoMeta source;
    private final int[] indices;
    private final double targetFps;
    private final VideoTemplate template;
    private final int temporalMerge;
    private final double[] groupTimestamps;

    private VideoSamplePlan(VideoMeta source, int[] indices, double targetFps, VideoTemplate template,
                            int temporalMerge) {
        this.source = Objects.requireNonNull(source, "source");
        if (!source.hasFrameRate()) {
            throw new IllegalArgumentException("a sampling plan needs a source frame rate; the stream reports none"
                    + " and no projector family assumed one");
        }
        this.indices = Objects.requireNonNull(indices, "indices");
        this.targetFps = targetFps;
        this.template = Objects.requireNonNull(template, "template");
        this.temporalMerge = temporalMerge;
        this.groupTimestamps = groupTimestamps(indices, source, template, temporalMerge);
    }

    /**
     * The plan of sampled {@code indices} over {@code source}, the sampled frames playing at {@code targetFps}.
     *
     * @throws IllegalArgumentException for a merge size that is not positive, no indices, or a source without a
     *                                  frame rate
     */
    public static VideoSamplePlan of(VideoMeta source, int[] indices, double targetFps, VideoTemplate template,
                                     int temporalMerge) {
        if (temporalMerge <= 0) {
            throw new IllegalArgumentException("the temporal merge must be positive, got " + temporalMerge);
        }
        if (indices.length == 0) {
            throw new IllegalArgumentException("a video samples at least one frame");
        }
        return new VideoSamplePlan(source, indices, targetFps, template, temporalMerge);
    }

    /**
     * The plan over {@code source}: every frame ({@code indices = 0 .. n - 1}), playing at the source rate.
     *
     * @throws IllegalArgumentException for a merge size that is not positive or a source without a frame rate
     */
    public static VideoSamplePlan ofAll(VideoMeta source, VideoTemplate template, int temporalMerge) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(template, "template");
        if (temporalMerge <= 0) {
            throw new IllegalArgumentException("the temporal merge must be positive, got " + temporalMerge);
        }
        int[] indices = new int[source.nFrames()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        return new VideoSamplePlan(source, indices, source.fps(), template, temporalMerge);
    }

    /** The group midpoints of {@code indices}, the time of each group's first and last frame averaged. */
    private static double[] groupTimestamps(int[] indices, VideoMeta source, VideoTemplate template,
                                            int temporalMerge) {
        if (template != VideoTemplate.GROUP || temporalMerge <= 1) {
            return new double[0];
        }
        double[] out = new double[groupCount(indices.length, temporalMerge)];
        for (int g = 0; g < out.length; g++) {
            int first = g * temporalMerge;
            int last = Math.min(first + temporalMerge - 1, indices.length - 1);
            out[g] = (source.timeOf(indices[first]) + source.timeOf(indices[last])) / 2;
        }
        return out;
    }

    /** The number of merge groups {@code n} frames form, the odd last frame its own group. */
    public static int groupCount(int n, int temporalMerge) {
        return (n + temporalMerge - 1) / temporalMerge;
    }

    /** The probed video the indices name. */
    public VideoMeta source() {
        return source;
    }

    /** The source frame indices to decode, ascending (a copy). */
    public int[] indices() {
        return indices.clone();
    }

    /** The frames the plan samples. */
    public int nFrames() {
        return indices.length;
    }

    /** The frames-per-second rate the sampled frames play at (the timestamps of the default template follow it). */
    public double targetFps() {
        return targetFps;
    }

    /** The prompt layout of the frames. */
    public VideoTemplate template() {
        return template;
    }

    /** The frames one encoded chunk merges (the temporal patch size of the reference processors). */
    public int temporalMerge() {
        return temporalMerge;
    }

    /** The encoded chunks the frames form, the odd last frame its own chunk. */
    public int groups() {
        return groupCount(indices.length, temporalMerge);
    }

    /**
     * The timestamp of group {@code group}: the midpoint time of its first and last frame.
     *
     * @throws IndexOutOfBoundsException when the group is outside the plan
     * @throws IllegalArgumentException  when the template stamps no group timestamps
     */
    public double groupTimestamp(int group) {
        Objects.checkIndex(group, groupTimestamps.length);
        return groupTimestamps[group];
    }

    /**
     * The count of {@code [XmY.YYs]} stamps the first {@code frames} output frames carry at an
     * {@code intervalMs} interval: a stamp when the frame's elapsed time has crossed the next multiple of the
     * interval (mtmd-helper's counter). The {@link VideoTemplate#GROUP} template and a non-positive interval
     * stamp nothing.
     */
    public int frameTimestamps(int frames, int intervalMs) {
        if (template != VideoTemplate.DEFAULT || intervalMs <= 0) {
            return 0;
        }
        int stamps = 0;
        double next = intervalMs;
        for (int f = 0; f < frames; f++) {
            if (f / targetFps * 1000.0 >= next) {
                stamps++;
                next += intervalMs;
            }
        }
        return stamps;
    }

    /** The time in seconds the default template stamps output frame {@code frame} at. */
    public double frameTime(int frame) {
        Objects.checkIndex(frame, indices.length);
        return frame / targetFps;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof VideoSamplePlan)) {
            return false;
        }
        VideoSamplePlan p = (VideoSamplePlan) o;
        return source.equals(p.source) && template == p.template && temporalMerge == p.temporalMerge
                && Double.compare(targetFps, p.targetFps) == 0 && Arrays.equals(indices, p.indices)
                && Arrays.equals(groupTimestamps, p.groupTimestamps);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(source, template, temporalMerge, targetFps);
        h = 31 * h + Arrays.hashCode(indices);
        return 31 * h + Arrays.hashCode(groupTimestamps);
    }

    @Override
    public String toString() {
        return "VideoSamplePlan{" + nFrames() + " of " + source.nFrames() + " frames at " + targetFps + " fps, "
                + template + ", merge " + temporalMerge + "}";
    }
}

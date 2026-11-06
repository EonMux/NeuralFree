package com.eonmux.neuralfree.gguf.video;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;

import java.util.EnumSet;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * The video input a projector family takes, the counterpart of llama.cpp's
 * {@code mtmd_video_config} / {@code mtmd_get_video_config} (tools/mtmd/mtmd.cpp) and vLLM's per-processor video
 * backends (vllm/multimodal/video.py): the prompt template of the frames, the frames one encoded chunk merges
 * (the temporal patch size, 2 for the Qwen-VL, Ling3-VL, LFM3-VL and Nemotron-Omni families), the recommended
 * sampling rate with its frame clamps, how the frame count and source indices are taken, and the source rate the
 * family assumes when the container reports none. The LFM3-VL and Qwen-VL families carry a recommendation, as the
 * references do; the other families take the caller's rate. The LFM3-Omni variant hears its video's soundtrack.
 * The Qwen2-VL family follows vLLM's {@code Qwen2VLVideoBackend}: its count is floored to the temporal patch size
 * and its indices come from {@code torch.arange}, not {@code round(linspace)}. Qwen3-VL assumes a 24 fps source
 * when the stream reports no rate; the Qwen2-VL family, whose vLLM loader raises on a zero rate, and every other
 * family, which follows llama.cpp's helper, refuse such a source. The plan for one video resolves through
 * {@link #plan}. Immutable.
 */
public final class VideoConfig {

    /**
     * The projector families whose frames merge two at a time ({@code clip_model_n_temporal_merge}): the Qwen-VL,
     * Ling3-VL, LFM3-VL and Nemotron-Omni encoders carry two temporal patch kernels.
     */
    private static final Set<ProjectorType> TEMPORAL_MERGE = EnumSet.of(ProjectorType.QWEN2VL,
            ProjectorType.QWEN25VL, ProjectorType.QWEN3VL, ProjectorType.LING3VL, ProjectorType.LFM3VL,
            ProjectorType.NEMOTRON_OMNI);

    /** The source rate vLLM's Qwen3-VL loader assumes when the container reports none. */
    private static final double QWEN3_SOURCE_FPS = 24.0;
    /** The highest sampling rate vLLM's Qwen2-VL loader honours (its {@code _MAX_FPS}). */
    private static final double QWEN2_MAX_FPS = 30.0;

    /** The LFM3-VL sampling recommendation: 2 fps with a {@code [4, 768]} frame clamp. */
    private static final double LFM3_FPS = 2.0;
    private static final int LFM3_MIN_FRAMES = 4;
    private static final int LFM3_MAX_FRAMES = 768;
    /** The LFM3-Omni variant, which also hears the soundtrack, samples at most 384 frames. */
    private static final int LFM3_OMNI_MAX_FRAMES = 384;
    /**
     * The Qwen-VL video loaders' recommendation (vllm/multimodal/video.py): 2 fps with a {@code [4, 768]} frame
     * clamp. Qwen3-VL stamps the midpoint of every temporal group, as its processor does; Qwen2-VL and Qwen2.5-VL
     * keep llama.cpp's default {@code "Video:"} template.
     */
    private static final double QWEN_FPS = 2.0;
    private static final int QWEN_MIN_FRAMES = 4;
    private static final int QWEN_MAX_FRAMES = 768;

    /** How a family's plan takes its frame count and source indices. */
    private enum Sampling {
        /** {@code round(linspace(0, n - 1, count))} over the clamped count, the HF processors' indices. */
        LINSPACE,
        /** vLLM's Qwen2-VL loader: the count floored to the temporal patch size, {@code torch.arange} indices. */
        QWEN2VL
    }

    private final VideoTemplate template;
    private final int temporalMerge;
    private final double fps;
    private final int minFrames;
    private final int maxFrames;
    private final boolean soundtrack;
    private final Sampling sampling;
    private final double assumedSourceFps;

    private VideoConfig(VideoTemplate template, int temporalMerge, double fps, int minFrames, int maxFrames,
                        boolean soundtrack, Sampling sampling, double assumedSourceFps) {
        this.template = template;
        this.temporalMerge = temporalMerge;
        this.fps = fps;
        this.minFrames = minFrames;
        this.maxFrames = maxFrames;
        this.soundtrack = soundtrack;
        this.sampling = sampling;
        this.assumedSourceFps = assumedSourceFps;
    }

    /** The input of a projector without a video recommendation: the default template, one frame per chunk. */
    public static VideoConfig defaults() {
        return new VideoConfig(VideoTemplate.DEFAULT, 1, 0, 1, Integer.MAX_VALUE, false, Sampling.LINSPACE, 0);
    }

    /**
     * The input the projector family takes: the LFM3-VL recommendation (group timestamps, 2 fps, a
     * {@code [4, 384]} frame clamp for the omni variant that hears the soundtrack and {@code [4, 768]} otherwise),
     * the Qwen-VL recommendation (2 fps and a {@code [4, 768]} clamp; Qwen3-VL stamps its temporal groups and
     * assumes a 24 fps source when the stream reports none, the Qwen2 family keeps the default template and
     * samples like vLLM's Qwen2-VL loader) and temporal merge 2 for the Qwen-VL, LFM3-VL, Ling3-VL and
     * Nemotron-Omni families. Every other projector keeps the default template, one frame per chunk and the
     * caller's sampling.
     *
     * @param soundtrack whether the video carries an LFM3-Audio soundtrack (llama.cpp's
     *                   {@code mtmd_context::video_has_soundtrack}: LFM3-VL vision beside LFM3-A audio)
     * @throws NullPointerException when {@code type} is null
     */
    public static VideoConfig forProjector(ProjectorType type, boolean soundtrack) {
        Objects.requireNonNull(type, "type");
        if (type == ProjectorType.LFM3VL) {
            return new VideoConfig(VideoTemplate.GROUP, 2, LFM3_FPS, LFM3_MIN_FRAMES,
                    soundtrack ? LFM3_OMNI_MAX_FRAMES : LFM3_MAX_FRAMES, soundtrack, Sampling.LINSPACE, 0);
        }
        if (type == ProjectorType.QWEN3VL) {
            return new VideoConfig(VideoTemplate.GROUP, 2, QWEN_FPS, QWEN_MIN_FRAMES, QWEN_MAX_FRAMES, false,
                    Sampling.LINSPACE, QWEN3_SOURCE_FPS);
        }
        if (type == ProjectorType.QWEN2VL || type == ProjectorType.QWEN25VL) {
            return new VideoConfig(VideoTemplate.DEFAULT, 2, QWEN_FPS, QWEN_MIN_FRAMES, QWEN_MAX_FRAMES, false,
                    Sampling.QWEN2VL, 0);
        }
        if (TEMPORAL_MERGE.contains(type)) {
            return new VideoConfig(VideoTemplate.DEFAULT, 2, 0, 1, Integer.MAX_VALUE, false, Sampling.LINSPACE, 0);
        }
        return defaults();
    }

    /** The prompt layout of the frames. */
    public VideoTemplate template() {
        return template;
    }

    /** The frames one encoded chunk merges (the temporal patch size of the reference processors). */
    public int temporalMerge() {
        return temporalMerge;
    }

    /** The recommended sampling rate, or 0 when the family names none. */
    public double fps() {
        return fps;
    }

    /** The smallest frame count the family samples, or 1 when it names none. */
    public int minFrames() {
        return minFrames;
    }

    /** The largest frame count the family samples. */
    public int maxFrames() {
        return maxFrames;
    }

    /** Whether the video's soundtrack is decoded and encoded with the frames (the LFM3-Omni template). */
    public boolean soundtrack() {
        return soundtrack;
    }

    /**
     * The source rate the family assumes when the stream reports none (the Qwen3-VL loader's 24 fps), or 0 when
     * the family refuses such a source.
     */
    public double assumedSourceFps() {
        return assumedSourceFps;
    }

    /**
     * {@code source} at the frame rate the family's reference assumes when the stream reports none: the Qwen3-VL
     * loader's 24 fps (vLLM's {@code Qwen3VLVideoBackend._prepare_source}). Every other family refuses such a
     * source, as its reference does: vLLM's Qwen2-VL loader raises on a zero rate, and llama.cpp's helper rejects
     * a probe without one.
     *
     * @throws MediaInputException naming {@code field} when the stream reports no rate and the family assumes none
     */
    public VideoMeta resolveSource(VideoMeta source, String field) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(field, "field");
        if (source.hasFrameRate()) {
            return source;
        }
        if (assumedSourceFps > 0) {
            return source.withFps(assumedSourceFps);
        }
        if (sampling == Sampling.QWEN2VL) {
            throw new MediaInputException(field + ": the Qwen2-VL video loader needs a known source frame rate, but"
                    + " the container reports none (r_frame_rate 0 or N/A, a variable or unknown rate)");
        }
        throw new MediaInputException(field + ": the video stream reports no frame rate (r_frame_rate 0 or N/A);"
                + " NeuralFree samples video frames by source index, which needs one");
    }

    /**
     * The sampling plan of {@code source} under this config, following the reference of each family: a family
     * with its own recommendation (the Qwen-VL and LFM3-VL loaders) samples at the request's rate or its own and
     * ignores the request's frame count, as vLLM's fps-driven loaders do; a family without one takes the smaller
     * of the request's frame count and the request's fps-derived count, as vLLM's generic video backend does. The
     * count is clamped to the family's frame range and the source's frames, and the indices are
     * {@code round(linspace(0, n - 1, count))}, except for the Qwen2-VL family, whose count is floored to the
     * temporal patch size and whose indices follow {@code torch.arange} (see {@link #resolveSource} for a source
     * without a rate). A video longer than {@code maxDurationSeconds} is refused (vLLM's {@code max_duration}).
     *
     * @param fpsRequest         the rate the caller asked for, or a negative value (vLLM's {@code -1}) for the
     *                           model's own
     * @param numFramesRequest   the frame count the caller asked for, or a negative value for the model's own
     * @param maxDurationSeconds the longest video the caller accepts, or 0 for no limit
     * @param fallbackFps        the rate to sample at when neither the request nor the family names one; 0 for the
     *                           source rate
     * @throws MediaInputException     when the video runs longer than the accepted duration, reports no frame rate
     *                                 and the family assumes none, or is shorter than the Qwen2-VL family's
     *                                 temporal patch
     * @throws IllegalArgumentException when a request rate of 0 or a request count of 0 is passed
     */
    public VideoSamplePlan plan(VideoMeta source, Double fpsRequest, Integer numFramesRequest,
                                int maxDurationSeconds, String field) {
        return plan(source, fpsRequest, numFramesRequest, maxDurationSeconds, 0, field);
    }

    /**
     * {@link #plan(VideoMeta, Double, Integer, int, String)} with {@code fallbackFps} as the rate a video samples at
     * when neither the request nor the family names one; 0 or less takes the source rate. This is llama.cpp's
     * {@code --video-fps} (its default 4 fps): the family's recommendation wins over it.
     */
    public VideoSamplePlan plan(VideoMeta source, Double fpsRequest, Integer numFramesRequest,
                                int maxDurationSeconds, double fallbackFps, String field) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(field, "field");
        if (maxDurationSeconds > 0 && source.duration() > maxDurationSeconds) {
            throw new MediaInputException(field + ": the video runs " + String.format(Locale.ROOT,
                    "%.1f", source.duration()) + " s, over the " + maxDurationSeconds
                    + " s max_duration the video loader accepts");
        }
        if (fpsRequest != null && fpsRequest == 0) {
            throw new IllegalArgumentException("the video sampling rate must not be 0; a negative value takes the"
                    + " model's own");
        }
        if (numFramesRequest != null && numFramesRequest == 0) {
            throw new IllegalArgumentException("the video frame count must not be 0; a negative value takes the"
                    + " model's own");
        }
        VideoMeta resolved = resolveSource(source, field);
        Double rateRequest = fpsRequest != null && fpsRequest > 0 ? fpsRequest : null;
        Integer countRequest = numFramesRequest != null && numFramesRequest > 0 ? numFramesRequest : null;
        if (sampling == Sampling.QWEN2VL) {
            double rate = rateRequest != null ? rateRequest : fps;
            int count = qwen2VlCount(resolved, rate, field);
            return planOf(resolved, qwen2VlIndices(resolved.nFrames(), count), count);
        }
        int count;
        if (fps > 0) {
            // vLLM's fps-driven family loaders (Qwen3-VL, LFM3-VL) ignore num_frames
            double rate = rateRequest != null ? rateRequest : fps;
            count = (int) (resolved.nFrames() / resolved.fps() * rate);
        } else {
            // vLLM's generic backend: the smaller of the requested count and the fps-derived count
            int sampled = resolved.nFrames();
            if (countRequest != null) {
                sampled = Math.min(countRequest, sampled);
            }
            double rate = rateRequest != null ? rateRequest : fallbackFps;
            if (rate > 0) {
                sampled = Math.min(sampled, (int) Math.floor(resolved.duration() * rate));
            }
            count = sampled;
        }
        count = Math.max(Math.min(minFrames, resolved.nFrames()), Math.min(count, resolved.nFrames()));
        count = Math.min(count, maxFrames);
        count = Math.max(1, count);
        return planOf(resolved, linspace(resolved.nFrames(), count), count);
    }

    /** The plan of {@code indices} over the resolved {@code source}, playing {@code count} frames over its span. */
    private VideoSamplePlan planOf(VideoMeta source, int[] indices, int count) {
        double rate = source.duration() > 0 ? count / source.duration() : source.fps();
        return VideoSamplePlan.of(source, indices, rate, template, temporalMerge);
    }

    /**
     * The Qwen2-VL loader's frame count (vLLM's {@code Qwen2VLVideoBackend.compute_frames_index_to_sample}):
     * {@code floor(n / temporal_patch_size) * temporal_patch_size} over the rate-capped count, clamped to the
     * family's frame range floored to the patch size and to the source's frames.
     *
     * @throws MediaInputException naming {@code field} when the video is shorter than the floored frame range
     */
    private int qwen2VlCount(VideoMeta source, double rate, String field) {
        int total = source.nFrames();
        int flooredMax = Math.floorDiv(Math.min(maxFrames, total), temporalMerge) * temporalMerge;
        double n = (double) total / source.fps() * Math.min(rate, QWEN2_MAX_FPS);
        n = Math.min(Math.max(n, minFrames), flooredMax);
        int count = (int) (Math.floor(n / temporalMerge) * temporalMerge);
        if (count <= 0) {
            throw new MediaInputException(field + ": the Qwen2-VL loader samples whole " + temporalMerge
                    + "-frame temporal patches, but the video has " + total + " frame(s); at least " + temporalMerge
                    + " frames are needed");
        }
        return count;
    }

    /**
     * The {@code torch.arange(0, total, total / count).int()} indices of the Qwen2-VL loader, each clamped to the
     * last source frame. {@code torch.arange} evaluates {@code i * step} in float32, so a product that sits on a
     * frame boundary truncates differently from the same arithmetic in double, and the float32 step can emit one
     * index past the count, the tail vLLM clamps.
     */
    static int[] qwen2VlIndices(int total, int count) {
        double step = (double) total / count;
        int length = (int) Math.ceil(total / step);
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) {
            float value = (float) (i * step);
            indices[i] = Math.min((int) value, total - 1);
        }
        return indices;
    }

    /** {@code round(linspace(0, total - 1, count))}, ties to even, the HF processors' indices. */
    private static int[] linspace(int total, int count) {
        int[] out = new int[count];
        if (count == 1) {
            out[0] = 0;
            return out;
        }
        double step = (double) (total - 1) / (count - 1);
        for (int i = 0; i < count; i++) {
            out[i] = (int) Math.rint(i * step);
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof VideoConfig)) {
            return false;
        }
        VideoConfig c = (VideoConfig) o;
        return template == c.template && temporalMerge == c.temporalMerge && minFrames == c.minFrames
                && maxFrames == c.maxFrames && soundtrack == c.soundtrack && sampling == c.sampling
                && Double.compare(fps, c.fps) == 0 && Double.compare(assumedSourceFps, c.assumedSourceFps) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(template, temporalMerge, fps, minFrames, maxFrames, soundtrack, sampling,
                assumedSourceFps);
    }

    @Override
    public String toString() {
        return "VideoConfig{" + template + ", merge " + temporalMerge + ", fps=" + (fps > 0 ? fps : "model's own")
                + ", frames in [" + minFrames + ", " + (maxFrames == Integer.MAX_VALUE ? "unbounded" : maxFrames)
                + "]" + (sampling == Sampling.QWEN2VL ? ", qwen2-vl sampling" : "")
                + (assumedSourceFps > 0 ? ", assumes " + assumedSourceFps + " fps when unknown" : "")
                + (soundtrack ? ", soundtrack" : "") + "}";
    }
}

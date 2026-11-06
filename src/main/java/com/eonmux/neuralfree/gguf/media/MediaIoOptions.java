package com.eonmux.neuralfree.gguf.media;

import com.eonmux.neuralfree.gguf.image.ImageLoadOptions;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * The per-modality media loader options, vLLM's {@code --media-io-kwargs} (vllm/config/multimodal.py
 * {@code media_io_kwargs}): a JSON object keyed by modality whose values are the loader's keyword arguments. The
 * options the media loaders here support are the image loader's ({@code ImageMediaIO}) and the video loader's
 * ({@code VideoMediaIO}):
 *
 * <ul>
 *   <li>{@code image_mode}: {@code "RGB"} (the default) or {@code "L"} (grayscale), see
 *       {@link ImageLoadOptions.Mode};</li>
 *   <li>{@code rgba_background_color}: the colour transparent pixels composite over, three integers in
 *       {@code [0, 255]} (default white);</li>
 *   <li>{@code num_frames} (video): the frame count to sample, {@code -1} for the model's own (the default);</li>
 *   <li>{@code fps} (video): the frames-per-second rate to sample at, {@code -1} for the model's own (the
 *       default);</li>
 *   <li>{@code max_duration} (video): the longest video in seconds, 300 by default; longer videos refuse.</li>
 * </ul>
 *
 * <p>The audio loader takes no options; any other key or modality is refused by name. A modality's configured
 * options are hash factors of its items' keys ({@link MediaKey}), as vLLM hashes {@code media_io_kwargs} into
 * {@code mm_hash}: a non-empty option set changes the key even of a uuid. Immutable.</p>
 */
public final class MediaIoOptions {

    /** The flag the options come from, named in refusals. */
    public static final String FLAG = "--media-io-kwargs";

    private static final String IMAGE_MODE = "image_mode";
    private static final String BACKGROUND = "rgba_background_color";
    private static final String NUM_FRAMES = "num_frames";
    private static final String FPS = "fps";
    private static final String MAX_DURATION = "max_duration";
    private static final MediaIoOptions DEFAULTS = new MediaIoOptions(ImageLoadOptions.defaults(), "",
            Video.defaults(), "");

    private final ImageLoadOptions image;
    /** The image options as configured, in canonical form; empty when none are. */
    private final String imageFactors;
    private final Video video;
    /** The video options as configured, in canonical form; empty when none are. */
    private final String videoFactors;

    private MediaIoOptions(ImageLoadOptions image, String imageFactors, Video video, String videoFactors) {
        this.image = image;
        this.imageFactors = imageFactors;
        this.video = video;
        this.videoFactors = videoFactors;
    }

    /** No options: every loader at vLLM's defaults, no hash factors. */
    public static MediaIoOptions defaults() {
        return DEFAULTS;
    }

    /**
     * The options of a JSON object such as {@code {"image": {"image_mode": "L"}}}; null or blank is
     * {@link #defaults()}.
     *
     * @throws IllegalArgumentException for text that is not a JSON object, a modality other than image or audio,
     *                                  a modality value that is not an object, an option the modality's loader
     *                                  does not take, or an option value out of its range; the message names it
     */
    public static MediaIoOptions parse(String json) {
        if (json == null || json.trim().isEmpty()) {
            return defaults();
        }
        Object parsed;
        try {
            parsed = Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (JsonException e) {
            throw new IllegalArgumentException(FLAG + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException(FLAG + " must be a JSON object keyed by modality, such as"
                    + " {\"image\": {\"image_mode\": \"RGB\"}}");
        }
        ImageLoadOptions image = ImageLoadOptions.defaults();
        String imageFactors = "";
        Video video = Video.defaults();
        String videoFactors = "";
        for (Map.Entry<?, ?> e : ((Map<?, ?>) parsed).entrySet()) {
            String modality = String.valueOf(e.getKey());
            Map<?, ?> options = options(modality, e.getValue());
            if (MediaLimits.Modality.IMAGE.key().equals(modality)) {
                image = imageOptions(options);
                imageFactors = canonical(modality, options);
            } else if (MediaLimits.Modality.AUDIO.key().equals(modality)) {
                if (!options.isEmpty()) {
                    throw new IllegalArgumentException(FLAG + " audio option '" + options.keySet().iterator().next()
                            + "' is not supported: the audio loader takes no options (WAV, AIFF and AU decode"
                            + " in-process, other formats through ffmpeg)");
                }
            } else if (MediaLimits.Modality.VIDEO.key().equals(modality)) {
                video = videoOptions(options);
                videoFactors = canonical(modality, options);
            } else {
                throw new IllegalArgumentException(FLAG + " names the modality '" + modality + "'; the loaders"
                        + " that take options are image, audio and video");
            }
        }
        return new MediaIoOptions(image, imageFactors, video, videoFactors);
    }

    private static Map<?, ?> options(String modality, Object value) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException(FLAG + " " + modality + " must be a JSON object of loader options,"
                    + " got " + value);
        }
        return (Map<?, ?>) value;
    }

    /** The video loader's options: the sampling overrides and the longest accepted video. */
    private static Video videoOptions(Map<?, ?> options) {
        Video.Builder video = Video.builder();
        for (Map.Entry<?, ?> e : options.entrySet()) {
            String key = String.valueOf(e.getKey());
            if (NUM_FRAMES.equals(key)) {
                video.numFrames(whole(FLAG + " video " + NUM_FRAMES, e.getValue()));
            } else if (FPS.equals(key)) {
                video.fps(number(FLAG + " video " + FPS, e.getValue()));
            } else if (MAX_DURATION.equals(key)) {
                video.maxDurationSeconds(whole(FLAG + " video " + MAX_DURATION, e.getValue()));
            } else {
                throw new IllegalArgumentException(FLAG + " video option '" + key + "' is not supported: the video"
                        + " loader takes " + NUM_FRAMES + ", " + FPS + " and " + MAX_DURATION);
            }
        }
        return video.build();
    }

    /** A whole number option value. */
    private static int whole(String what, Object value) {
        if (!(value instanceof Number) || ((Number) value).doubleValue() != Math.rint(((Number) value).doubleValue())
                || ((Number) value).longValue() < Integer.MIN_VALUE || ((Number) value).longValue() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " must be a whole number, got " + value);
        }
        return ((Number) value).intValue();
    }

    /** A finite number option value. */
    private static double number(String what, Object value) {
        if (!(value instanceof Number) || Double.isNaN(((Number) value).doubleValue())
                || Double.isInfinite(((Number) value).doubleValue())) {
            throw new IllegalArgumentException(what + " must be a finite number, got " + value);
        }
        return ((Number) value).doubleValue();
    }

    private static ImageLoadOptions imageOptions(Map<?, ?> options) {
        ImageLoadOptions.Mode mode = ImageLoadOptions.Mode.RGB;
        int background = ImageLoadOptions.WHITE;
        for (Map.Entry<?, ?> e : options.entrySet()) {
            String key = String.valueOf(e.getKey());
            if (IMAGE_MODE.equals(key)) {
                mode = mode(e.getValue());
            } else if (BACKGROUND.equals(key)) {
                background = background(e.getValue());
            } else {
                throw new IllegalArgumentException(FLAG + " image option '" + key + "' is not supported: the image"
                        + " loader takes " + IMAGE_MODE + " and " + BACKGROUND);
            }
        }
        return ImageLoadOptions.of(mode, background);
    }

    private static ImageLoadOptions.Mode mode(Object value) {
        if (value instanceof String) {
            for (ImageLoadOptions.Mode m : ImageLoadOptions.Mode.values()) {
                if (m.name().equals(value)) {
                    return m;
                }
            }
        }
        throw new IllegalArgumentException(FLAG + " image " + IMAGE_MODE + " " + quoted(value) + " is not"
                + " supported: the vision encoders read RGB images, loaded as \"RGB\" or as grayscale \"L\"");
    }

    /** vLLM's {@code rgba_background_color} check: a list of three integers in {@code [0, 255]}. */
    private static int background(Object value) {
        String refusal = FLAG + " image " + BACKGROUND + " must be a list of 3 integers in the range [0, 255],"
                + " got " + value;
        if (!(value instanceof List) || ((List<?>) value).size() != 3) {
            throw new IllegalArgumentException(refusal);
        }
        int rgb = 0;
        for (Object c : (List<?>) value) {
            if (!(c instanceof Number) || ((Number) c).doubleValue() != Math.rint(((Number) c).doubleValue())
                    || ((Number) c).doubleValue() < 0 || ((Number) c).doubleValue() > 255) {
                throw new IllegalArgumentException(refusal);
            }
            rgb = (rgb << 8) | ((Number) c).intValue();
        }
        return rgb;
    }

    /**
     * The canonical form of one modality's options, the hash factor of its items: {@code modality} and the
     * options with sorted keys; empty for no options (vLLM drops an empty factor).
     */
    private static String canonical(String modality, Map<?, ?> options) {
        if (options.isEmpty()) {
            return "";
        }
        Map<String, Object> sorted = new TreeMap<>();
        for (Map.Entry<?, ?> e : options.entrySet()) {
            sorted.put(String.valueOf(e.getKey()), canonicalValue(e.getValue()));
        }
        return "media_io_kwargs=" + modality + sorted;
    }

    private static Object canonicalValue(Object value) {
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            return d == Math.rint(d) ? Long.toString((long) d) : Double.toString(d);
        }
        if (value instanceof List) {
            StringBuilder out = new StringBuilder("[");
            for (Object v : (List<?>) value) {
                out.append(out.length() > 1 ? "," : "").append(canonicalValue(v));
            }
            return out.append(']').toString();
        }
        return quoted(value);
    }

    private static String quoted(Object value) {
        return value instanceof String ? "\"" + value + "\"" : String.valueOf(value);
    }

    /** The image loader's options. */
    public ImageLoadOptions image() {
        return image;
    }

    /** The video loader's options. */
    public Video video() {
        return video;
    }

    /**
     * The hash factors of {@code modality}'s items in canonical form, empty when the modality has no configured
     * options.
     */
    public String hashFactors(MediaLimits.Modality modality) {
        Objects.requireNonNull(modality, "modality");
        return modality == MediaLimits.Modality.IMAGE ? imageFactors
                : modality == MediaLimits.Modality.VIDEO ? videoFactors : "";
    }

    /** Whether no option is configured. */
    public boolean isDefault() {
        return imageFactors.isEmpty() && videoFactors.isEmpty();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof MediaIoOptions && imageFactors.equals(((MediaIoOptions) o).imageFactors)
                && image.equals(((MediaIoOptions) o).image) && video.equals(((MediaIoOptions) o).video);
    }

    @Override
    public int hashCode() {
        return imageFactors.hashCode() * 31 + video.hashCode();
    }

    @Override
    public String toString() {
        if (imageFactors.isEmpty() && videoFactors.isEmpty()) {
            return "MediaIoOptions{defaults}";
        }
        String joined = (imageFactors.isEmpty() ? "" : imageFactors)
                + (!imageFactors.isEmpty() && !videoFactors.isEmpty() ? ", " : "")
                + (videoFactors.isEmpty() ? "" : videoFactors);
        return "MediaIoOptions{" + joined + '}';
    }

    /**
     * The video loader's options, vLLM's {@code VideoMediaIO} keyword arguments: the frame count and rate to
     * sample at (each unset by default, the model's own recommendation winning) and the longest video in seconds.
     * Immutable; built with {@link #builder()}.
     */
    public static final class Video {

        /** vLLM's unset marker: the model's own recommendation decides. */
        public static final int UNSET = -1;
        /** vLLM's default longest video. */
        public static final int DEFAULT_MAX_DURATION_SECONDS = 300;

        private final int numFrames;
        private final double fps;
        private final int maxDurationSeconds;

        private Video(int numFrames, double fps, int maxDurationSeconds) {
            this.numFrames = numFrames;
            this.fps = fps;
            this.maxDurationSeconds = maxDurationSeconds;
        }

        /** The defaults: the model's own sampling, 300 s of video at most. */
        public static Video defaults() {
            return builder().build();
        }

        /** A builder at the defaults. */
        public static Builder builder() {
            return new Builder();
        }

        /** The frame count to sample, or {@link #UNSET} for the model's own. */
        public int numFrames() {
            return numFrames;
        }

        /** The rate to sample at, or {@link #UNSET} for the model's own. */
        public double fps() {
            return fps;
        }

        /** The longest video in seconds; 0 takes no limit. */
        public int maxDurationSeconds() {
            return maxDurationSeconds;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Video && numFrames == ((Video) o).numFrames && fps == ((Video) o).fps
                    && maxDurationSeconds == ((Video) o).maxDurationSeconds;
        }

        @Override
        public int hashCode() {
            return Objects.hash(numFrames, fps, maxDurationSeconds);
        }

        @Override
        public String toString() {
            return "Video{numFrames=" + numFrames + ", fps=" + fps + ", maxDuration=" + maxDurationSeconds + "s}";
        }

        /** Builder of {@link Video}. */
        public static final class Builder {
            private int numFrames = UNSET;
            private double fps = UNSET;
            private int maxDurationSeconds = DEFAULT_MAX_DURATION_SECONDS;

            private Builder() {
            }

            /**
             * The frame count to sample, {@link #UNSET} for the model's own.
             *
             * @throws IllegalArgumentException when the count is not a whole number of at least {@code -1}
             */
            public Builder numFrames(int numFrames) {
                if (numFrames < UNSET || (numFrames == 0)) {
                    throw new IllegalArgumentException(FLAG + " video " + NUM_FRAMES + " must be a whole number of"
                            + " at least " + UNSET + " other than 0, got " + numFrames);
                }
                this.numFrames = numFrames;
                return this;
            }

            /**
             * The rate to sample at, {@link #UNSET} for the model's own.
             *
             * @throws IllegalArgumentException when the rate is not a positive number or {@code -1}
             */
            public Builder fps(double fps) {
                if (fps < UNSET || fps == 0 || Double.isNaN(fps) || Double.isInfinite(fps)) {
                    throw new IllegalArgumentException(FLAG + " video " + FPS + " must be a positive number or "
                            + UNSET + ", got " + fps);
                }
                this.fps = fps;
                return this;
            }

            /**
             * The longest video in seconds; 0 takes no limit.
             *
             * @throws IllegalArgumentException when the duration is negative
             */
            public Builder maxDurationSeconds(int seconds) {
                if (seconds < 0) {
                    throw new IllegalArgumentException(FLAG + " video " + MAX_DURATION + " must not be negative,"
                            + " got " + seconds);
                }
                this.maxDurationSeconds = seconds;
                return this;
            }

            /** The options. */
            public Video build() {
                return new Video(numFrames, fps, maxDurationSeconds);
            }
        }
    }
}

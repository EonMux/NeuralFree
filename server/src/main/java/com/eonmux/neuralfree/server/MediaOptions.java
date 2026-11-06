package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConfig;
import com.eonmux.neuralfree.gguf.mm.MediaContext;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.mm.MmprojSources;
import com.eonmux.neuralfree.gguf.video.VideoOptions;
import com.eonmux.neuralfree.gguf.video.VideoPruner;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The multimodal input settings of the server: the projector that encodes images, videos and audio — the explicit
 * file of llama.cpp's {@code --mmproj}, its URL download ({@code --mmproj-url} into the hub cache) or the automatic
 * sibling search ({@code --mmproj-auto}, the {@code *mmproj*.gguf} beside the model file) — where media URLs may be
 * loaded from ({@link MediaConnector.Options}: vLLM's
 * {@code --allowed-media-domains} and {@code --allowed-local-media-path}, llama.cpp's {@code --media-path}, the
 * fetch timeout and size limit), the per-prompt item limits (vLLM's {@code --limit-mm-per-prompt}) and the
 * per-image token budget (llama.cpp's {@code --image-min-tokens} / {@code --image-max-tokens}), where the vision
 * encoder runs ({@link MmprojOptions}: llama.cpp's {@code --mmproj-offload} and {@code --mmproj-device}), the
 * byte budget of each layer of the projector's media cache, its encodings and its processed media (vLLM's
 * {@code --mm-processor-cache-gb}), the media loader options (vLLM's {@code --media-io-kwargs}) and the video
 * input settings ({@link VideoOptions}: llama.cpp's {@code --video-ffmpeg-dir}, {@code --video-fps} and
 * {@code --video-timestamp-interval}, vLLM's {@code --video-pruning-rate} and {@code --video-pruning-method}), and
 * where the encoder output of an item is shared with the other servers of the deployment
 * ({@link EncoderCacheConfig}: vLLM's {@code --ec-transfer-config}). Immutable.
 *
 * @param mmproj     the projector GGUF, or null when the server takes no media input
 * @param connector  where and how media URLs load
 * @param limits     the per-prompt item limits
 * @param budget     the per-image token budget
 * @param encoder    where the vision encoder runs
 * @param cacheBytes the byte budget of each media cache layer; 0 caches nothing
 * @param io         the media loader options, hash factors of every item's key
 * @param video      the video input settings
 * @param encoderCache where the encoder output of an item is shared with other servers (vLLM's
 *                    {@code --ec-transfer-config}), or null to keep it on this server
 */
record MediaOptions(String mmproj, MediaConnector.Options connector, MediaLimits limits, ImageTokenBudget budget,
                    MmprojOptions encoder, long cacheBytes, MediaIoOptions io, VideoOptions video,
                    EncoderCacheConfig encoderCache) {

    /** The projector GGUF path. */
    static final String MMPROJ_PROPERTY = "neuralfree.server.mmproj";
    /** The projector URL ({@code --mmproj-url}), downloaded once into the hub cache at startup. */
    static final String MMPROJ_URL_PROPERTY = "neuralfree.server.mmproj.url";
    /** The {@code --mmproj-auto} switch: {@code false} turns the sibling search beside the model off. */
    static final String MMPROJ_AUTO_PROPERTY = "neuralfree.server.mmproj.auto";
    /** The model file the automatic sibling search sits beside. */
    private static final String MODEL_PROPERTY = "neuralfree.server.model";
    /** The comma-separated hosts http(s) media URLs may name; unset allows any host. */
    static final String DOMAINS_PROPERTY = "neuralfree.server.media.domains";
    /** The directory file media URLs may read from; unset refuses file URLs. */
    static final String LOCAL_PATH_PROPERTY = "neuralfree.server.media.path";
    /** The media fetch timeout in seconds. */
    static final String TIMEOUT_PROPERTY = "neuralfree.server.media.timeout";
    /** The largest media payload in bytes. */
    static final String MAX_BYTES_PROPERTY = "neuralfree.server.media.max.bytes";
    /** Whether media URL redirects are followed (each target re-checked). */
    static final String REDIRECTS_PROPERTY = "neuralfree.server.media.redirects";
    /** The per-prompt item limits, a JSON object such as {@code {"image": 2, "audio": 1}}. */
    static final String LIMITS_PROPERTY = "neuralfree.server.media.limits";
    /** The smallest image token count of dynamic-resolution projectors. */
    static final String IMAGE_MIN_TOKENS_PROPERTY = "neuralfree.server.image.min.tokens";
    /** The largest image token count of dynamic-resolution projectors. */
    static final String IMAGE_MAX_TOKENS_PROPERTY = "neuralfree.server.image.max.tokens";
    /** The media cache layers' budget in GiB (vLLM's {@code mm_processor_cache_gb}); 0 turns the cache off. */
    static final String CACHE_GB_PROPERTY = "neuralfree.server.media.cache.gb";
    /** The media loader options, a JSON object keyed by modality (vLLM's {@code media_io_kwargs}). */
    static final String IO_KWARGS_PROPERTY = "neuralfree.server.media.io.kwargs";
    /** The directory of the ffmpeg and ffprobe binaries (llama.cpp's {@code --video-ffmpeg-dir}); unset uses the PATH. */
    static final String VIDEO_FFMPEG_DIR_PROPERTY = "neuralfree.server.video.ffmpeg.dir";
    /** The frames-per-second rate a video samples at without a model or request rate (llama.cpp's {@code --video-fps}). */
    static final String VIDEO_FPS_PROPERTY = "neuralfree.server.video.fps";
    /** The interval in milliseconds of the default template's timestamps ({@code --video-timestamp-interval}). */
    static final String VIDEO_TIMESTAMP_PROPERTY = "neuralfree.server.video.timestamp.interval";
    /** The fraction of every video's embedding rows pruning drops (vLLM's {@code --video-pruning-rate}). */
    static final String VIDEO_PRUNING_RATE_PROPERTY = "neuralfree.server.video.pruning.rate";
    /** The pruning algorithm of a positive rate: {@code evs} or {@code vidcom2} ({@code --video-pruning-method}). */
    static final String VIDEO_PRUNING_METHOD_PROPERTY = "neuralfree.server.video.pruning.method";

    MediaOptions {
        Objects.requireNonNull(connector, "connector");
        Objects.requireNonNull(limits, "limits");
        Objects.requireNonNull(budget, "budget");
        Objects.requireNonNull(encoder, "encoder");
        Objects.requireNonNull(io, "io");
        Objects.requireNonNull(video, "video");
        if (cacheBytes < 0) {
            throw new IllegalArgumentException("the media cache budget must not be negative, got " + cacheBytes);
        }
        if (mmproj != null && mmproj.isBlank()) {
            mmproj = null;
        }
    }

    /**
     * No projector: media parts are refused; URLs would load under the connector defaults, a projector would
     * offload its vision encoder ({@link MmprojOptions#defaults()}), cache under {@link #defaultCacheBytes()},
     * load media under the default loader options, decode video under the default video options and keep its
     * encoder output on this server.
     */
    static MediaOptions none() {
        return new MediaOptions(null, MediaConnector.Options.defaults(), MediaLimits.defaults(),
                ImageTokenBudget.none(), MmprojOptions.defaults(), defaultCacheBytes(), MediaIoOptions.defaults(),
                VideoOptions.defaults(), null);
    }

    /**
     * The budget of each cache layer when none is configured: vLLM's 4 GiB ({@link MediaContext#DEFAULT_CACHE_BYTES}),
     * at most an eighth of the JVM's heap limit, so the two layers, which live on the heap, take at most a quarter.
     */
    static long defaultCacheBytes() {
        return Math.min(MediaContext.DEFAULT_CACHE_BYTES, Runtime.getRuntime().maxMemory() / 8);
    }

    /** {@link #none()} with the projector at {@code mmproj}. */
    static MediaOptions of(String mmproj) {
        return none().withMmproj(mmproj);
    }

    /** A copy with the projector at {@code mmproj}. */
    MediaOptions withMmproj(String mmproj) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy loading media under {@code options}. */
    MediaOptions withConnector(MediaConnector.Options options) {
        return new MediaOptions(mmproj, options, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy with the per-prompt {@code limits}. */
    MediaOptions withLimits(MediaLimits limits) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy running the vision encoder where {@code encoder} chooses. */
    MediaOptions withEncoder(MmprojOptions encoder) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy caching at most {@code cacheBytes} in each media cache layer. */
    MediaOptions withCacheBytes(long cacheBytes) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy loading media under the loader options {@code io}. */
    MediaOptions withIo(MediaIoOptions io) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy decoding video under {@code video}. */
    MediaOptions withVideo(VideoOptions video) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /** A copy sharing encoder output where {@code encoderCache} says, or keeping it local for null. */
    MediaOptions withEncoderCache(EncoderCacheConfig encoderCache) {
        return new MediaOptions(mmproj, connector, limits, budget, encoder, cacheBytes, io, video, encoderCache);
    }

    /**
     * The options of the {@code neuralfree.server.mmproj*}, {@code neuralfree.server.media.*},
     * {@code neuralfree.server.image.*} and {@code neuralfree.server.video.*} system properties, with the encoder
     * device of {@link MmprojOptions#fromSystemProperties()} ({@code neuralfree.mmproj.offload},
     * {@code neuralfree.mmproj.device}).
     *
     * @throws IllegalArgumentException when a number is not a whole positive number, the cache budget is not a
     *                                  non-negative number of GiB, the local path is not a directory, a domain
     *                                  entry is blank, the limits or the loader options are not a valid object,
     *                                  the ffmpeg directory is not a directory, an encoder property is
     *                                  malformed, or the projector URL does not download
     */
    static MediaOptions fromSystemProperties() {
        MediaConnector.Options.Builder connector = MediaConnector.Options.builder();
        String domains = System.getProperty(DOMAINS_PROPERTY);
        if (domains != null && !domains.isBlank()) {
            connector.allowedDomains(list(domains));
        }
        String local = System.getProperty(LOCAL_PATH_PROPERTY);
        if (local != null && !local.isBlank()) {
            connector.localRoot(Path.of(local.trim()));
        }
        long timeout = positive(TIMEOUT_PROPERTY, MediaConnector.Options.DEFAULT_TIMEOUT.toSeconds());
        connector.timeout(Duration.ofSeconds(timeout));
        connector.maxBytes(positive(MAX_BYTES_PROPERTY, MediaConnector.Options.DEFAULT_MAX_BYTES));
        connector.allowRedirects(Boolean.parseBoolean(System.getProperty(REDIRECTS_PROPERTY, "true").trim()));
        ImageTokenBudget.Builder budget = ImageTokenBudget.builder();
        budget.minTokens((int) Math.min(Integer.MAX_VALUE, whole(IMAGE_MIN_TOKENS_PROPERTY, 0)));
        budget.maxTokens((int) Math.min(Integer.MAX_VALUE, whole(IMAGE_MAX_TOKENS_PROPERTY, 0)));
        return new MediaOptions(projector(), connector.build(),
                MediaLimits.parse(System.getProperty(LIMITS_PROPERTY)), budget.build(),
                MmprojOptions.fromSystemProperties(), cacheBytes(System.getProperty(CACHE_GB_PROPERTY)),
                MediaIoOptions.parse(System.getProperty(IO_KWARGS_PROPERTY)), videoOptions(),
                EncoderCacheConfig.fromProperties(System.getProperties()));
    }

    /**
     * The projector of the {@code neuralfree.server.mmproj*} properties, llama.cpp's sources in order: the
     * explicit {@code neuralfree.server.mmproj} path, else the {@code neuralfree.server.mmproj.url} download
     * into the hub cache, else — while {@code neuralfree.server.mmproj.auto} is not {@code false} — the
     * {@code *mmproj*.gguf} sibling beside the {@code neuralfree.server.model} file. Null without any, and the
     * server refuses media input by name.
     *
     * @throws IllegalArgumentException when a URL download fails
     */
    private static String projector() {
        String explicit = System.getProperty(MMPROJ_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return explicit.trim();
        }
        if (explicit != null) {
            // a blank projector path turns media input off, over the URL and the sibling search alike
            return null;
        }
        String url = System.getProperty(MMPROJ_URL_PROPERTY);
        if (url != null && !url.isBlank()) {
            try {
                return MmprojSources.downloadFromUrl(url.trim(), HubCache.defaultRoot()).toString();
            } catch (IOException e) {
                throw new IllegalArgumentException(MMPROJ_URL_PROPERTY + " cannot download '" + url.trim()
                        + "': " + e.getMessage(), e);
            }
        }
        if ("false".equalsIgnoreCase(System.getProperty(MMPROJ_AUTO_PROPERTY, "").trim())) {
            return null;
        }
        String model = System.getProperty(MODEL_PROPERTY);
        if (model == null || model.isBlank()) {
            return null;
        }
        Path sibling = MmprojSources.siblingBeside(Path.of(model.trim()));
        return sibling == null ? null : sibling.toString();
    }

    /** The video options of the {@code neuralfree.server.video.*} properties. */
    private static VideoOptions videoOptions() {
        VideoOptions.Builder video = VideoOptions.builder();
        String ffmpegDir = System.getProperty(VIDEO_FFMPEG_DIR_PROPERTY);
        if (ffmpegDir != null && !ffmpegDir.isBlank()) {
            video.ffmpegDir(Path.of(ffmpegDir.trim()));
        }
        String fps = System.getProperty(VIDEO_FPS_PROPERTY);
        if (fps != null && !fps.isBlank()) {
            try {
                video.fpsTarget(Double.parseDouble(fps.trim()));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(VIDEO_FPS_PROPERTY + " must be a number, got '" + fps + "'", e);
            }
        }
        video.timestampIntervalMs((int) Math.min(Integer.MAX_VALUE,
                whole(VIDEO_TIMESTAMP_PROPERTY, VideoOptions.DEFAULT_TIMESTAMP_INTERVAL_MS)));
        video.pruningRate(pruningRate(System.getProperty(VIDEO_PRUNING_RATE_PROPERTY)));
        video.pruningMethod(VideoPruner.Method.of(System.getProperty(VIDEO_PRUNING_METHOD_PROPERTY,
                VideoOptions.DEFAULT_PRUNING_METHOD.key()).trim()));
        return video.build();
    }

    /**
     * The pruning rate of a property value; blank is 0 (no pruning).
     *
     * @throws IllegalArgumentException when the value is not a number in {@code [0, 1)}
     */
    private static double pruningRate(String raw) {
        if (raw == null || raw.isBlank()) {
            return 0;
        }
        double v;
        try {
            v = Double.parseDouble(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(VIDEO_PRUNING_RATE_PROPERTY + " must be a number in [0, 1), got '"
                    + raw + "'", e);
        }
        if (!(v >= 0) || v >= 1 || Double.isNaN(v) || Double.isInfinite(v)) {
            throw new IllegalArgumentException(VIDEO_PRUNING_RATE_PROPERTY + " must be a number in [0, 1), got '"
                    + raw + "'");
        }
        return v;
    }

    /**
     * The cache budget in bytes of a GiB count such as {@code 4} or {@code 0.5} (vLLM's
     * {@code --mm-processor-cache-gb}); null or blank gives {@link #defaultCacheBytes()}.
     *
     * @throws IllegalArgumentException when the value is not a finite non-negative number
     */
    static long cacheBytes(String gib) {
        if (gib == null || gib.isBlank()) {
            return defaultCacheBytes();
        }
        double v;
        try {
            v = Double.parseDouble(gib.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(CACHE_GB_PROPERTY + " must be a number of GiB, got '" + gib + "'", e);
        }
        if (!(v >= 0) || Double.isInfinite(v)) {
            throw new IllegalArgumentException(CACHE_GB_PROPERTY + " must be a non-negative number of GiB, got '"
                    + gib + "'");
        }
        return (long) Math.min(Long.MAX_VALUE, v * (1L << 30));
    }

    private static List<String> list(String raw) {
        List<String> out = new ArrayList<>();
        for (String entry : raw.split("[,\\s]+")) {
            if (!entry.isEmpty()) {
                out.add(entry);
            }
        }
        return out;
    }

    private static long positive(String property, long fallback) {
        long v = whole(property, fallback);
        if (v <= 0) {
            throw new IllegalArgumentException(property + " must be positive, got " + v);
        }
        return v;
    }

    private static long whole(String property, long fallback) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            long v = Long.parseLong(raw.trim());
            if (v < 0) {
                throw new IllegalArgumentException(property + " must not be negative, got " + v);
            }
            return v;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + " must be a whole number, got '" + raw + "'", e);
        }
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioDecoder;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.image.FloatImage;
import com.eonmux.neuralfree.gguf.image.ImageDecodeException;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessor;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.image.PreparedImages;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.video.DecodedVideo;
import com.eonmux.neuralfree.gguf.video.VideoConfig;
import com.eonmux.neuralfree.gguf.video.VideoDecoder;
import com.eonmux.neuralfree.gguf.video.VideoMeta;
import com.eonmux.neuralfree.gguf.video.VideoOptions;
import com.eonmux.neuralfree.gguf.video.VideoPruner;
import com.eonmux.neuralfree.gguf.video.VideoSamplePlan;
import com.eonmux.neuralfree.gguf.video.VideoTemplate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * The multimodal prompt context of one projector file beside a text model, the counterpart of llama.cpp's
 * {@code mtmd_context} and its tokenizer ({@code mtmd_tokenizer::add_media}, tools/mtmd/mtmd.cpp): it decodes image
 * and audio bytes, preprocesses an image into the family's tiles ({@link ImagePreprocessors}), encodes every tile with
 * the vision encoder on the device {@link MmprojOptions} choose ({@link VisionEncoder}: CPU or GPU) and every mel
 * chunk with the CPU audio encoder ({@link CpuAudioEncoder}), and appends the result to a
 * {@link MediaPrompt.Builder} as text chunks of the family's marker text ({@link MediaMarkers}) and media chunks of
 * embedding rows. A video is probed and sampled by the video input of its projector family
 * ({@link com.eonmux.neuralfree.gguf.video.VideoConfig}), decoded to frames through ffmpeg
 * ({@link com.eonmux.neuralfree.gguf.video.VideoDecoder}), merged two frames to a chunk by a projector with a
 * temporal patch kernel, stamped with the template's timestamps and pruned by the configured video token pruning
 * ({@link com.eonmux.neuralfree.gguf.video.VideoPruner}). An LFM3-Omni video also decodes its soundtrack and
 * appends each temporal group's audio rows after the group's frames ({@link VideoSoundtrack}).
 *
 * <p>Positions follow the text model: on a multi-section RoPE model (M-RoPE) an unsliced image takes the 2D grid
 * positions of its merged token grid (SenseNova-U1: the zero-based grid at one temporal position), a MiniCPM-V 4.7
 * image the canvas positions of its one chunk ({@link MediaRopePositions#canvas}), and every other media chunk one
 * position per row in every section; on other models every row takes one position. Image rows of the families
 * {@link MediaMarkers#nonCausalImage()} names decode without the causal mask. A DeepSeek-V4 image block opens with
 * the lead pads that align it to the text model's 4-token blocks after the prompt before it
 * ({@link MediaMarkers#imageLeadPadAlign()}); the image encodes once and each place takes its own pad count. A
 * Granite Vision anyres image ({@link PreparedImages#isAnyres()}) is one chunk of its assembled overview and tile
 * grid, on one token row.</p>
 *
 * <p>Opening checks what {@code mtmd_context} checks: the projector's output width must equal the text model's
 * embedding width (and the vision and audio widths each other). A MiniCPM-V 4.7 projector needs an M-RoPE text
 * model, the only one its canvas positions describe.</p>
 *
 * <p>Every item is named by a {@link MediaKey} (vLLM's {@code mm_hash}): the client's {@code uuid} when the part
 * carries one, else the SHA-256 digest of its bytes, with the modality's loader options ({@link MediaIoOptions},
 * vLLM's {@code --media-io-kwargs}) as hash factors, and — on the encoded layer, the audio layer and the prompt
 * chunks — the request's adapters folded in as another factor when the request selects any
 * ({@link #keyOf(MediaLimits.Modality, String, byte[], LoraSlotKey)}, vLLM keying its encoder and prefix caches by
 * the adapter name and the media hash). Media are cached in two least-recently-used layers, each
 * bounded by the byte budget of vLLM's {@code --mm-processor-cache-gb}: the encoded layer by the adapter-scoped
 * key, the processed layer by the plain item (the decode runs no adapters). The encoded
 * layer holds an item's marker text and embedding rows, so a repeat skips decoding, preprocessing and encoding; it
 * is looked up first. The processed layer holds the decoded and preprocessed image tiles and the decoded audio
 * samples (vLLM's processor cache), so an item whose encoding was evicted or reset is encoded again without
 * decoding. A uuid either layer holds needs no media data, so a later request may repeat the uuid alone; a token
 * count followed by the generation of the same request encodes once.</p>
 *
 * <p>A request that selects adapters encodes its media inside a LoRA session on the vision encoder
 * ({@link VisionEncoder#openLoraSession}): the session slot registers the request's adapters' vision pairs in place
 * of the startup ones for exactly that encode, under the context monitor, so a concurrent request's tiles never run
 * through another request's adapters, and a request that names no adapter encodes under the startup adapters as
 * always. One context serves one projector for every replica of the model.</p>
 *
 * <p>A context opened with a shared encoder cache ({@link EncoderCacheConnector}, vLLM's encoder cache connector)
 * also exchanges item encodings with the other servers of its deployment: a consumer loads an item's encoding from
 * the connector before it reads the item's data, decodes or encodes it, and a producer saves the encoding of every
 * item it encodes. The sharing key is the item's {@link MediaKey}, and an entry loaded from the connector is
 * checked against this projector's row width before it is used.</p>
 *
 * <p>Each media chunk carries the item's key and its place among the item's chunks ({@link MediaChunk#item()},
 * {@link MediaChunk#piece()}), and the identity {@code key#piece} ({@link MediaChunk#id()}) an engine's encoder
 * cache keys the shared encoder output by. The image decode follows the loader options, so the same options decode
 * an item the same way and name it by the same key.</p>
 *
 * <p>The projector weights {@link #release() release} while the text model sleeps and {@link #reopen() reopen}
 * from the same file on the wake; the markers, widths and capabilities stay, and an encode in between refuses with
 * {@link ReleasedException}. Thread-safe: encodes are serialized (the encoders are single-threaded objects), a
 * request's whole encode under its adapters holding the context monitor. Close
 * it to release the encoders and the projector weights for good.</p>
 */
public final class MediaContext implements AutoCloseable {

    /** The default encoded-media cache budget in bytes: vLLM's {@code mm_processor_cache_gb} default of 4 GiB. */
    public static final long DEFAULT_CACHE_BYTES = 4L << 30;

    private static final String IMAGE = MediaLimits.Modality.IMAGE.key();
    private static final String AUDIO = MediaLimits.Modality.AUDIO.key();

    /** Tokenizes marker and prompt text: special tokens parsed, none added. */
    public interface TextEncoder {
        /** The tokens of {@code text}. */
        int[] encode(String text);
    }

    /** An encode refused because the projector weights are released (the engine sleeps). */
    public static final class ReleasedException extends IllegalStateException {
        private static final long serialVersionUID = 1L;

        ReleasedException() {
            super("the multimodal projector is released while the engine sleeps; wake the engine to encode media");
        }
    }

    private final String path;
    private final MmprojOptions encoderOptions;
    private final MmprojModel.Caps caps;
    private final boolean hasVision;
    private final boolean hasAudio;
    /** Why image / audio input is refused, for a context without that encoder. */
    private final String visionRefusal;
    private final String audioRefusal;
    private final String audioProjector;
    private final ImageProcessorParams imageParams;
    private final ImagePreprocessor preprocessor;
    private final MediaMarkers markers;
    private final boolean mrope;
    private final int width;
    /** The media loader options (vLLM's {@code --media-io-kwargs}) item keys and image decodes follow. */
    private final MediaIoOptions io;
    /** The video input the projector family takes and the settings the video decode and pruning follow. */
    private final VideoConfig videoConfig;
    private final VideoOptions videoOptions;
    private final VideoDecoder videoDecoder;
    private final EncoderOutputCache<MediaKey, List<MediaPiece>> cache;
    private final EncoderOutputCache<MediaKey, ProcessedMedia> processed;
    /** The appender of the family's encoded pieces to a prompt. */
    private final MediaPromptAppender appender;
    /** The shared encoder cache exchange (vLLM's encoder cache connector), inert without a connector. */
    private final SharedEncoderCache sharedCache;
    /** The adapters the vision encoder applies, held for a reopen. */
    private final List<LoraAttachment.Binding> loras;
    /** The loaded projector; null while released. Guarded by {@code this}. */
    private MmprojEncoders encoders;
    private volatile String visionDevice;
    private volatile boolean released;
    private boolean closed;

    private MediaContext(String path, MmprojOptions encoderOptions, MmprojEncoders encoders,
                         ImageProcessorParams imageParams, MediaMarkers markers, boolean mrope, int width,
                         long cacheBytes, MediaIoOptions io, VideoConfig videoConfig, VideoOptions videoOptions,
                         EncoderCacheConnector shared, List<LoraAttachment.Binding> loras) {
        this.path = path;
        this.loras = List.copyOf(loras);
        this.encoderOptions = encoderOptions;
        this.encoders = encoders;
        this.caps = encoders.mmproj.caps();
        this.hasVision = encoders.vision != null;
        this.hasAudio = encoders.audio != null;
        this.visionRefusal = refusal("image", "vision", encoders.mmproj.unrunnableReason(ProjectorModality.VISION));
        this.audioRefusal = refusal("audio", "audio", encoders.mmproj.unrunnableReason(ProjectorModality.AUDIO));
        this.visionDevice = hasVision ? encoders.vision.device() : null;
        this.audioProjector = hasAudio ? String.valueOf(encoders.audio.model().hparams().projectorType()) : null;
        this.imageParams = imageParams;
        this.preprocessor = imageParams == null ? null : ImagePreprocessors.forParams(imageParams);
        this.markers = markers;
        this.mrope = mrope;
        this.width = width;
        this.io = io;
        this.videoConfig = videoConfig;
        this.videoOptions = videoOptions;
        this.videoDecoder = new VideoDecoder(videoOptions.ffmpegDir());
        this.cache = new EncoderOutputCache<>(cacheBytes, MediaPiece::bytesOf);
        this.processed = new EncoderOutputCache<>(cacheBytes, ProcessedMedia::bytes);
        this.appender = new MediaPromptAppender(markers, mrope);
        this.sharedCache = new SharedEncoderCache(shared, cache, width);
    }

    /**
     * Opens the projector at {@code path} for a text model of embedding width {@code textEmbd}.
     *
     * @param path       the mmproj GGUF
     * @param textEmbd   the text model's input embedding width
     * @param mrope      whether the text model reads several RoPE positions per token (M-RoPE / IMRoPE)
     * @param budget     the per-image token budget ({@code --image-min-tokens} / {@code --image-max-tokens})
     * @param hasToken   whether the text vocabulary holds a token text (families whose markers depend on it)
     * @param encoder    where the vision encoder runs ({@link MmprojModel#openVisionEncoder(MmprojOptions)})
     * @param cacheBytes the byte budget of each media cache layer (encoded and processed); 0 caches nothing
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException when the file holds no usable encoder, its family has no image
     *                                  preprocessing in this build, its width differs from the text model's, its
     *                                  canvas images need an M-RoPE text model {@code mrope} denies, or the cache
     *                                  budget is negative
     */
    public static MediaContext open(String path, int textEmbd, boolean mrope, ImageTokenBudget budget,
                                    Predicate<String> hasToken, MmprojOptions encoder, long cacheBytes)
            throws IOException {
        return open(path, textEmbd, mrope, budget, hasToken, encoder, cacheBytes, MediaIoOptions.defaults());
    }

    /**
     * {@link #open(String, int, boolean, ImageTokenBudget, Predicate, MmprojOptions, long)} with the media loader
     * options {@code io} (vLLM's {@code --media-io-kwargs}), which the image decode follows and every item key
     * hashes in.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException as the other overload refuses
     */
    public static MediaContext open(String path, int textEmbd, boolean mrope, ImageTokenBudget budget,
                                    Predicate<String> hasToken, MmprojOptions encoder, long cacheBytes,
                                    MediaIoOptions io) throws IOException {
        return open(path, textEmbd, mrope, budget, hasToken, encoder, cacheBytes, io, VideoOptions.defaults());
    }

    /**
     * {@link #open(String, int, boolean, ImageTokenBudget, Predicate, MmprojOptions, long, MediaIoOptions)} with
     * the video input settings {@code video}: the ffmpeg directory, the fallback sampling rate, the timestamp
     * interval and the token pruning rate and method the video decodes and encodes follow.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException as the other overload refuses
     */
    public static MediaContext open(String path, int textEmbd, boolean mrope, ImageTokenBudget budget,
                                    Predicate<String> hasToken, MmprojOptions encoder, long cacheBytes,
                                    MediaIoOptions io, VideoOptions video) throws IOException {
        return open(path, textEmbd, mrope, budget, hasToken, encoder, cacheBytes, io, video, List.of());
    }

    /**
     * {@link #open(String, int, boolean, ImageTokenBudget, Predicate, MmprojOptions, long, MediaIoOptions,
     * VideoOptions)} with the LoRA adapters {@code loras} whose vision-encoder and connector pairs the encoder
     * applies to every row it encodes (the startup adapters of the text model). The bindings are held for a later
     * {@link #reopen()}, which attaches them again.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException as the other overload refuses, and when a binding's vision pair targets a
     *                                  tensor the encoder does not serve
     */
    public static MediaContext open(String path, int textEmbd, boolean mrope, ImageTokenBudget budget,
                                    Predicate<String> hasToken, MmprojOptions encoder, long cacheBytes,
                                    MediaIoOptions io, VideoOptions video, List<LoraAttachment.Binding> loras)
            throws IOException {
        return open(path, textEmbd, mrope, budget, hasToken, encoder, cacheBytes, io, video, null, loras);
    }

    /**
     * {@link #open(String, int, boolean, ImageTokenBudget, Predicate, MmprojOptions, long, MediaIoOptions,
     * VideoOptions, List)} with the shared encoder cache {@code shared} (vLLM's encoder cache connector): a
     * {@link EncoderCacheConnector.Role#CONSUMER} looks an item up there before it decodes or encodes it, and a
     * {@link EncoderCacheConnector.Role#PRODUCER} saves every item it encodes, keyed by the item's {@link MediaKey}.
     * The context closes the connector with itself; null keeps every item local.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException as the other overload refuses
     */
    public static MediaContext open(String path, int textEmbd, boolean mrope, ImageTokenBudget budget,
                                    Predicate<String> hasToken, MmprojOptions encoder, long cacheBytes,
                                    MediaIoOptions io, VideoOptions video, EncoderCacheConnector shared,
                                    List<LoraAttachment.Binding> loras) throws IOException {
        Objects.requireNonNull(budget, "budget");
        Objects.requireNonNull(hasToken, "hasToken");
        Objects.requireNonNull(encoder, "encoder");
        Objects.requireNonNull(io, "io");
        VideoOptions videoOptions = Objects.requireNonNull(video, "video");
        if (cacheBytes < 0) {
            throw new IllegalArgumentException("the media cache budget must not be negative, got " + cacheBytes);
        }
        MmprojEncoders loaded = MmprojEncoders.open(path, encoder, textEmbd, loras);
        try {
            MmprojModel mm = loaded.mmproj;
            ImageProcessorParams params = null;
            MediaMarkers markers = MediaMarkers.none();
            VideoConfig videoConfig = VideoConfig.defaults();
            if (mm.supportsVision()) {
                params = ImageProcessorParams.fromGguf(mm.metadata(), budget);
                ImagePreprocessors.forParams(params);
                ClipHparams hp = mm.visionHparams();
                markers = MediaMarkers.vision(hp.projectorType(), hp.minicpmvVersion(),
                        loaded.vision.model().outputDim(), hasToken);
                if (markers.canvasImages() && !mrope) {
                    throw new IllegalArgumentException("the " + hp.projectorType() + " projector of " + path
                            + " lays every image out on an M-RoPE canvas (mtmd MTMD_POS_TYPE_CANVAS); the text"
                            + " model has no multi-section RoPE");
                }
                // mtmd_get_video_config: the LFM3-Omni projector (LFM3-VL vision beside an LFM3-Audio tower in
                // one file) hears its video's soundtrack, which clamps its frames tighter
                boolean soundtrack = mm.supportsAudio()
                        && mm.audioHparams().projectorType() == ProjectorType.LFM3A;
                videoConfig = VideoConfig.forProjector(hp.projectorType(), soundtrack);
            }
            if (mm.supportsAudio()) {
                markers = markers.withAudio(mm.audioHparams().projectorType());
            }
            return new MediaContext(path, encoder, loaded, params, markers, mrope, textEmbd, cacheBytes, io,
                    videoConfig, videoOptions, shared, loras);
        } catch (RuntimeException e) {
            loaded.close();
            throw e;
        }
    }

    /** The refusal of {@code input} for a context without its encoder: the registry's reason when one exists. */
    private static String refusal(String input, String encoder, String unrunnableReason) {
        return input + " input is not supported: " + (unrunnableReason == null
                ? "the projector has no " + encoder + " encoder"
                : "the projector's " + encoder + " encoder does not run (" + unrunnableReason + ")");
    }

    /** Whether image input is accepted (the file has a vision encoder NeuralFree runs). */
    public boolean supportsVision() {
        return hasVision;
    }

    /** Whether audio input is accepted (the file has an audio encoder NeuralFree runs). */
    public boolean supportsAudio() {
        return hasAudio;
    }

    /**
     * Whether video input is accepted: a vision encoder, whose frames the video path encodes. The ffmpeg binary
     * the frames decode through is checked when a video decodes, and refused by name when it is missing.
     */
    public boolean supportsVideo() {
        return hasVision;
    }

    /** The video input the projector family takes. */
    public VideoConfig videoConfig() {
        return videoConfig;
    }

    /** The video input settings the decode and pruning follow. */
    public VideoOptions videoOptions() {
        return videoOptions;
    }

    /**
     * Where the vision encoder runs ({@link VisionEncoder#device()}), or null without a vision encoder; while
     * released, where it ran last.
     */
    public String visionDevice() {
        return visionDevice;
    }

    /** The marker texts the prompts are built with. */
    public MediaMarkers markers() {
        return markers;
    }

    /** The embedding width of every media row. */
    public int width() {
        return width;
    }

    /** The projector file's capabilities. */
    public MmprojModel.Caps caps() {
        return caps;
    }

    // ------------------------------------------------------------------ appending

    /** The media loader options (vLLM's {@code --media-io-kwargs}) item keys and image decodes follow. */
    public MediaIoOptions ioOptions() {
        return io;
    }

    /**
     * The key of an item a client named {@code uuid}, or of {@code bytes} without a uuid, under this context's
     * loader options: the identity the processed layer holds, and the identity the encoded-media cache and the
     * prefix cache share for a request that names no adapter.
     */
    public MediaKey keyOf(MediaLimits.Modality modality, String uuid, byte[] bytes) {
        return keyOf(modality, uuid, bytes, null);
    }

    /**
     * The key of an item under this context's loader options, scoped to the request's adapters {@code lora} when it
     * selects any: vLLM keys its encoder cache and prefix cache by the adapter name and the media hash
     * ({@code _get_mm_identifier}), so the same image under two adapters encodes and caches twice and a prompt's
     * media chunks never collide across adapters. The processed layer keeps the plain key: the decode runs no
     * adapters.
     */
    public MediaKey keyOf(MediaLimits.Modality modality, String uuid, byte[] bytes, LoraSlotKey lora) {
        String factors = io.hashFactors(modality);
        if (lora != null) {
            String adapterFactor = "lora=" + lora.identity();
            factors = factors.isEmpty() ? adapterFactor : factors + "," + adapterFactor;
        }
        return uuid != null ? MediaKey.ofUuid(modality, uuid, factors) : MediaKey.ofContent(modality, bytes, factors);
    }

    /**
     * Appends one image: its marker text and its embedding chunks.
     *
     * @param prompt     the prompt being built
     * @param bytes      the encoded image (PNG, JPEG, BMP, GIF, WebP)
     * @param imageIndex the image's 0-based index among the prompt's images ({@code <image_id>} of MiniCPM-V 4.6)
     * @param text       the tokenizer of the marker text
     * @throws MediaInputException when the context has no vision encoder or the image cannot be decoded or
     *                             preprocessed
     */
    public void appendImage(MediaPrompt.Builder prompt, byte[] bytes, int imageIndex, TextEncoder text) {
        Objects.requireNonNull(bytes, "bytes");
        appendImage(prompt, (String) null, () -> bytes, imageIndex, text);
    }

    /**
     * Appends one image a client may name with a {@code uuid} (vLLM's media part {@code uuid}): the cached encoding
     * under that uuid when there is one, else the encoding of the cached prepared tiles — {@code data} is then not
     * read — else the encoding of {@code data}, cached in both layers under the uuid. Without a uuid the image is
     * cached under the digest of its bytes, hashed with the loader options.
     *
     * @param prompt     the prompt being built
     * @param uuid       the client's cache key of the image, or null
     * @param data       the encoded image's bytes, read at most once; null when the client sent only the uuid
     * @param imageIndex the image's 0-based index among the prompt's images ({@code <image_id>} of MiniCPM-V 4.6)
     * @param text       the tokenizer of the marker text
     * @throws MediaInputException when the context has no vision encoder, neither layer holds the uuid and no data
     *                             came with it, or the image cannot be loaded, decoded or preprocessed
     * @throws ReleasedException   when the image must be encoded while the projector is released
     */
    public void appendImage(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, int imageIndex,
                            TextEncoder text) {
        appendImage(prompt, uuid, data, imageIndex, text, null);
    }

    /**
     * {@link #appendImage(MediaPrompt.Builder, String, Supplier, int, TextEncoder)} under the request's adapters
     * {@code lora}: the image is cached and its chunks keyed by the adapter identity, and the encode runs in a LoRA
     * session whose slot holds the adapters' vision pairs in place of the startup ones ({@code lora} of the bare
     * key runs none).
     */
    public void appendImage(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, int imageIndex,
                            TextEncoder text, LoraSlotKey lora) {
        if (!hasVision) {
            throw new MediaInputException(visionRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.IMAGE, uuid, data, lora);
        appender.append(prompt, source.key(), encodedImage(source), imageIndex, text, markers.nonCausalImage());
    }

    /**
     * Appends one image under the item key {@code key} (vLLM's {@code mm_hash}): the cached encoding under that
     * key when there is one, else the encoding of {@code data}, cached in both layers under it. The key is taken
     * as given, so a client that received it from another server names the same item and reuses its cache entry.
     *
     * @param prompt     the prompt being built
     * @param key        the item's key
     * @param data       the encoded image's bytes, read at most once; null when the cache holds the key
     * @param imageIndex the image's 0-based index among the prompt's images ({@code <image_id>} of MiniCPM-V 4.6)
     * @param text       the tokenizer of the marker text
     * @throws MediaInputException when the context has no vision encoder, neither layer holds the key and no data
     *                             came with it, or the image cannot be loaded, decoded or preprocessed
     * @throws ReleasedException   when the image must be encoded while the projector is released
     */
    public void appendImage(MediaPrompt.Builder prompt, MediaKey key, Supplier<byte[]> data, int imageIndex,
                            TextEncoder text) {
        if (!hasVision) {
            throw new MediaInputException(visionRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.IMAGE, key, data);
        appender.append(prompt, key, encodedImage(source), imageIndex, text, markers.nonCausalImage());
    }

    /**
     * Appends one audio clip: its marker text and one embedding chunk per encoder window.
     *
     * @param prompt the prompt being built
     * @param bytes  the encoded audio (WAV, AIFF, AU; MP3, FLAC, OGG, M4A through ffmpeg)
     * @param name   a name for the clip in error messages, such as its format
     * @param text   the tokenizer of the marker text
     * @throws MediaInputException when the context has no audio encoder or the audio cannot be decoded
     */
    public void appendAudio(MediaPrompt.Builder prompt, byte[] bytes, String name, TextEncoder text) {
        Objects.requireNonNull(bytes, "bytes");
        appendAudio(prompt, (String) null, () -> bytes, name, text);
    }

    /**
     * Appends one audio clip a client may name with a {@code uuid}: the contract of
     * {@link #appendImage(MediaPrompt.Builder, String, Supplier, int, TextEncoder)} for audio.
     *
     * @param prompt the prompt being built
     * @param uuid   the client's cache key of the clip, or null
     * @param data   the encoded audio's bytes, read at most once; null when the client sent only the uuid
     * @param name   a name for the clip in error messages, such as its format
     * @param text   the tokenizer of the marker text
     * @throws MediaInputException when the context has no audio encoder, neither layer holds the uuid and no data
     *                             came with it, or the audio cannot be loaded or decoded
     * @throws ReleasedException   when the clip must be encoded while the projector is released
     */
    public void appendAudio(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, String name,
                            TextEncoder text) {
        appendAudio(prompt, uuid, data, name, text, null);
    }

    /**
     * {@link #appendAudio(MediaPrompt.Builder, String, Supplier, String, TextEncoder)} under the request's adapters
     * {@code lora}: the clip's cached encoding and prompt chunks are keyed by the adapter identity (vLLM keys the
     * encoder cache by the adapter for every modality); the audio encode itself runs no adapters.
     */
    public void appendAudio(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, String name,
                            TextEncoder text, LoraSlotKey lora) {
        if (!hasAudio) {
            throw new MediaInputException(audioRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.AUDIO, uuid, data, lora);
        appender.append(prompt, source.key(), encodedAudio(source, name), 0, text, false);
    }

    /**
     * Appends one audio clip under the item key {@code key}: the contract of
     * {@link #appendImage(MediaPrompt.Builder, MediaKey, Supplier, int, TextEncoder)} for audio.
     *
     * @throws MediaInputException when the context has no audio encoder, neither layer holds the key and no data
     *                             came with it, or the audio cannot be loaded or decoded
     * @throws ReleasedException   when the clip must be encoded while the projector is released
     */
    public void appendAudio(MediaPrompt.Builder prompt, MediaKey key, Supplier<byte[]> data, String name,
                            TextEncoder text) {
        if (!hasAudio) {
            throw new MediaInputException(audioRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.AUDIO, key, data);
        appender.append(prompt, key, encodedAudio(source, name), 0, text, false);
    }

    // ------------------------------------------------------------------ video

    /**
     * Appends one video: the template's frames, sampled by the video loader options and the projector family's
     * recommendation, decoded through ffmpeg, merged two frames to a chunk by a projector with a temporal patch
     * kernel, stamped with the template's timestamps and pruned by the configured video token pruning.
     *
     * @param prompt the prompt being built
     * @param bytes  the encoded video (any container ffmpeg reads)
     * @param field  the request field the video came from, named in every refusal
     * @param text   the tokenizer of the marker text
     * @throws MediaInputException when the context has no vision encoder, the video cannot be probed or decoded,
     *                             the ffmpeg binary is missing, or a frame does not fit the vision encoder
     */
    public void appendVideo(MediaPrompt.Builder prompt, byte[] bytes, String field, TextEncoder text) {
        Objects.requireNonNull(bytes, "bytes");
        appendVideo(prompt, (String) null, () -> bytes, field, text);
    }

    /**
     * Appends one video a client may name with a {@code uuid}: the contract of
     * {@link #appendImage(MediaPrompt.Builder, String, Supplier, int, TextEncoder)} for videos.
     *
     * @throws MediaInputException when the context has no vision encoder, neither layer holds the uuid and no data
     *                             came with it, or the video cannot be probed or decoded
     * @throws ReleasedException   when the video must be encoded while the projector is released
     */
    public void appendVideo(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, String field,
                            TextEncoder text) {
        appendVideo(prompt, uuid, data, field, text, null);
    }

    /**
     * {@link #appendVideo(MediaPrompt.Builder, String, Supplier, String, TextEncoder)} under the request's adapters
     * {@code lora}: the video is cached and its chunks keyed by the adapter identity, and its frames encode in a
     * LoRA session whose slot holds the adapters' vision pairs in place of the startup ones.
     */
    public void appendVideo(MediaPrompt.Builder prompt, String uuid, Supplier<byte[]> data, String field,
                            TextEncoder text, LoraSlotKey lora) {
        if (!hasVision) {
            throw new MediaInputException(visionRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.VIDEO, uuid, data, lora);
        appender.append(prompt, source.key(), encodedVideo(source, field), 0, text, markers.nonCausalImage());
    }

    /**
     * Appends one video under the item key {@code key}: the contract of
     * {@link #appendImage(MediaPrompt.Builder, MediaKey, Supplier, int, TextEncoder)} for videos.
     *
     * @throws MediaInputException when the context has no vision encoder, neither layer holds the key and no data
     *                             came with it, or the video cannot be probed or decoded
     * @throws ReleasedException   when the video must be encoded while the projector is released
     */
    public void appendVideo(MediaPrompt.Builder prompt, MediaKey key, Supplier<byte[]> data, String field,
                            TextEncoder text) {
        if (!hasVision) {
            throw new MediaInputException(visionRefusal);
        }
        MediaItem source = new MediaItem(this, MediaLimits.Modality.VIDEO, key, data);
        appender.append(prompt, key, encodedVideo(source, field), 0, text, markers.nonCausalImage());
    }

    /** The encoding of a video: the encoded layer's or the shared encoder cache's, else its sampled frames'. */
    private List<MediaPiece> encodedVideo(MediaItem source, String field) {
        List<MediaPiece> hit = sharedCache.shared(source);
        if (hit != null) {
            return hit;
        }
        return encodeUnderLora(source, () -> {
            List<MediaPiece> again = cache.peek(source.key());
            if (again != null) {
                return again;
            }
            DecodedVideo decoded = sampledVideo(source, field);
            return sharedCache.store(source, layoutVideo(decoded, field));
        });
    }

    /** The sampled frames of a video: the processed layer's, else probed, planned and decoded from its bytes. */
    private DecodedVideo sampledVideo(MediaItem source, String field) {
        ProcessedMedia hit = processed.get(source.baseKey());
        if (hit != null) {
            return hit.video;
        }
        VideoMeta meta = videoDecoder.probe(source.bytes(), field);
        MediaIoOptions.Video video = io.video();
        VideoSamplePlan plan = videoConfig.plan(meta, video.fps(), video.numFrames(), video.maxDurationSeconds(),
                videoOptions.fpsTarget(), field);
        ImageSize frameSize = preprocessor == null ? null
                : preprocessor.videoFrameSize(plan.nFrames(), meta.width(), meta.height());
        DecodedVideo decoded = videoDecoder.decode(source.bytes(), plan, frameSize, soundtrackRate(), field);
        ProcessedMedia out = ProcessedMedia.video(decoded);
        processed.put(source.baseKey(), out);
        return decoded;
    }

    /**
     * The rate the soundtrack decodes at: the audio tower's rate for an LFM3-Omni video, 0 for a video whose
     * family hears none.
     */
    private int soundtrackRate() {
        if (!videoConfig.soundtrack() || !hasAudio) {
            return 0;
        }
        return loaded().audio.model().hparams().audio().sampleRate();
    }

    /**
     * The marker text and row pieces of a sampled video, mtmd's lazy video layout: the default template opens
     * with the {@code "Video:"} prefix and stamps {@code [XmY.YYs]} text after the frames whose time has crossed
     * the interval; the group template stamps {@code <X.X seconds>} before every group. Every group is framed by
     * the image markers and its rows pruned by the configured video token pruning. The soundtrack of an LFM3-Omni
     * video encodes as one clip whose rows follow each group's frames ({@code add_soundtrack_piece}).
     */
    private List<MediaPiece> layoutVideo(DecodedVideo decoded, String field) {
        VideoSamplePlan plan = decoded.plan();
        VideoTemplate template = plan.template();
        List<RasterImage> frames = decoded.frames();
        int merge = plan.temporalMerge();
        List<List<MediaPiece>> groups = new ArrayList<>(plan.groups());
        for (int g = 0; g < plan.groups(); g++) {
            int first = g * merge;
            int count = Math.min(merge, frames.size() - first);
            List<MediaPiece> group = new ArrayList<>(3);
            if (count == 2) {
                group.add(MediaPiece.text(markers.imageBegin()));
                group.add(framePairChunk(frames.get(first), frames.get(first + 1), field));
                group.add(MediaPiece.text(markers.imageEnd()));
            } else {
                // one frame on its own: the family's whole image layout, markers included
                group.addAll(frameChunks(frames.get(first), field));
            }
            groups.add(group);
        }
        pruneVideo(groups, field);
        addSoundtrack(groups, decoded, field);
        List<MediaPiece> out = new ArrayList<>();
        if (template == VideoTemplate.DEFAULT) {
            out.add(MediaPiece.text(template.videoPrefix()));
            int interval = videoOptions.timestampIntervalMs();
            double nextStampMs = interval;
            for (int g = 0; g < groups.size(); g++) {
                addGroup(out, groups.get(g));
                int lastFrame = Math.min((g + 1) * merge, plan.nFrames()) - 1;
                if (interval > 0 && lastFrame / plan.targetFps() * 1000.0 >= nextStampMs) {
                    out.add(MediaPiece.text(template.frameTimestamp(plan.frameTime(lastFrame))));
                    nextStampMs += interval;
                }
            }
        } else {
            for (int g = 0; g < groups.size(); g++) {
                out.add(MediaPiece.text(template.groupTimestamp(plan.groupTimestamp(g))));
                addGroup(out, groups.get(g));
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** A group's pieces, the chunk pruned to no rows dropped (its stamp text stays, as vLLM's does). */
    private static void addGroup(List<MediaPiece> out, List<MediaPiece> group) {
        for (MediaPiece p : group) {
            if (p.kind == MediaPiece.Kind.GRID && p.rows.length == 0) {
                continue;
            }
            out.add(p);
        }
    }

    /**
     * Appends each temporal group's part of the soundtrack after its frames, mtmd's {@code add_soundtrack_piece}:
     * the group slices concatenate into one clip, which the audio tower encodes once; a group takes the rows from
     * its first sample's position to the next group's, the last group the rest of the track. The pieces carry no
     * audio markers, as the reference keeps them inside the video's span.
     */
    private void addSoundtrack(List<List<MediaPiece>> groups, DecodedVideo decoded, String field) {
        List<AudioClip> slices = decoded.soundtrack();
        if (slices.isEmpty()) {
            return;
        }
        float[][][] parts;
        synchronized (this) {
            parts = VideoSoundtrack.rows(slices, loaded().audio, field);
        }
        for (int g = 0; g < groups.size(); g++) {
            if (parts[g].length > 0) {
                groups.get(g).add(MediaPiece.rows(parts[g]));
            }
        }
    }

    /**
     * The chunks of one sampled frame: the family's image layout (a sliced frame keeps its tiles, a fixed-size
     * frame its one chunk).
     */
    private List<MediaPiece> frameChunks(RasterImage frame, String field) {
        PreparedImages prepared = prepareFrame(frame, field);
        return layoutImage(prepared);
    }

    /**
     * The chunk of two merged frames: both preprocessed (identical sizes), the pair encoded as one patch grid,
     * the first frame run through the patch kernel the still images use and the second through the temporal
     * kernel ({@code build_inp_temporal_merge} with a batch of two frames).
     */
    private MediaPiece framePairChunk(RasterImage first, RasterImage second, String field) {
        PreparedImages a = prepareFrame(first, field);
        PreparedImages b = prepareFrame(second, field);
        ImageTile tileA = singlePairTile(a, "first", field);
        ImageTile tileB = singlePairTile(b, "second", field);
        if (tileA.width() != tileB.width() || tileA.height() != tileB.height()) {
            throw new MediaInputException(field + ": the frames of a temporal pair preprocess to different sizes, "
                    + tileA.width() + "x" + tileA.height() + " against " + tileB.width() + "x" + tileB.height());
        }
        ClipImage pair = clipImage(tileA).withSecondFrame(clipImage(tileB).pixels());
        float[][] rows = encodeTile(pair, tileA);
        ImageSize grid = imageParams.tokenGridFor(tileA.size());
        if (grid.width() * grid.height() == rows.length) {
            return MediaPiece.grid(rows, grid.width(), grid.height());
        }
        return MediaPiece.grid(rows, rows.length, 1);
    }

    /** The one tile a frame of a temporal pair preprocesses to; a sliced frame refuses the pair. */
    private ImageTile singlePairTile(PreparedImages prepared, String which, String field) {
        List<ImageTile> tiles = prepared.encodedTiles();
        if (tiles.size() != 1 || prepared.hasOverview() || prepared.isAnyres()) {
            throw new MediaInputException(field + ": the " + which + " frame of a temporal pair preprocesses to "
                    + (prepared.hasOverview() ? "an overview with " : "") + tiles.size() + " tile(s); a projector"
                    + " with a temporal patch kernel encodes frames of one tile each, so video input is refused");
        }
        return tiles.get(0);
    }

    /** A frame preprocessed for the projector, named in every refusal. */
    private PreparedImages prepareFrame(RasterImage frame, String field) {
        try {
            return preprocessor.prepareVideoFrame(frame);
        } catch (IllegalArgumentException e) {
            throw new MediaInputException(field + ": a video frame cannot be preprocessed for the projector: "
                    + e.getMessage(), e);
        }
    }

    /**
     * The video token pruning over every group's chunk: a configured rate keeps {@code 1 - rate} of the video's
     * embedding rows (vLLM's EVS, which always keeps the first frame's rows, or VidCom2, which keeps at least one
     * row per frame). The groups must be single uniform chunks, the layout the temporal-patch and fixed-size
     * families produce; the others refuse pruning by name.
     */
    private void pruneVideo(List<List<MediaPiece>> groups, String field) {
        double rate = videoOptions.pruningRate();
        if (rate <= 0 || groups.isEmpty()) {
            return;
        }
        VideoPruner.Method method = videoOptions.pruningMethod();
        List<MediaPiece> chunks = new ArrayList<>(groups.size());
        for (List<MediaPiece> group : groups) {
            MediaPiece chunk = singleGridChunk(group);
            if (chunk == null) {
                throw new MediaInputException(field + ": the video token pruning runs over single-chunk frames,"
                        + " and the projector's frame layout splits a frame over " + group.size() + " pieces");
            }
            chunks.add(chunk);
        }
        int rowsPerFrame = chunks.get(0).rows.length;
        for (MediaPiece chunk : chunks) {
            if (chunk.rows.length != rowsPerFrame) {
                throw new MediaInputException(field + ": the video token pruning needs uniform frames, and the"
                        + " frames encode to " + rowsPerFrame + " and " + chunk.rows.length + " rows");
            }
        }
        float[][] rows = new float[chunks.size() * rowsPerFrame][];
        for (int g = 0; g < chunks.size(); g++) {
            System.arraycopy(chunks.get(g).rows, 0, rows, g * rowsPerFrame, rowsPerFrame);
        }
        boolean[] mask = method == VideoPruner.Method.EVS
                ? VideoPruner.evsMask(rows, chunks.size(), rowsPerFrame, rate)
                : VideoPruner.vidcom2Mask(rows, chunks.size(), rowsPerFrame, rate);
        for (int g = 0; g < groups.size(); g++) {
            replaceChunk(groups.get(g), chunks.get(g).withMask(mask, g * rowsPerFrame, rowsPerFrame));
        }
    }

    /** The one grid chunk of a group, or null when the group is not a single chunk. */
    private static MediaPiece singleGridChunk(List<MediaPiece> group) {
        MediaPiece chunk = null;
        for (MediaPiece p : group) {
            if (p.kind == MediaPiece.Kind.GRID) {
                if (chunk != null) {
                    return null;
                }
                chunk = p;
            } else if (p.kind != MediaPiece.Kind.TEXT) {
                return null;
            }
        }
        return chunk;
    }

    /** Swaps the grid chunk of a group for the pruned one. */
    private static void replaceChunk(List<MediaPiece> group, MediaPiece pruned) {
        for (int i = 0; i < group.size(); i++) {
            if (group.get(i).kind == MediaPiece.Kind.GRID) {
                group.set(i, pruned);
                return;
            }
        }
    }


    /**
     * Whether the item a client named {@code uuid} for {@code modality} is cached, encoded or processed, or, on a
     * consumer of the shared encoder cache, stored there: a part repeating that uuid then needs no data. The answer
     * may be stale by the time the part is appended (another request may evict the entry, or another server may
     * overwrite it), so a part that carries data keeps it.
     */
    public boolean hasCached(MediaLimits.Modality modality, String uuid) {
        return hasCached(modality, uuid, null);
    }

    /**
     * {@link #hasCached(MediaLimits.Modality, String)} for the item a request naming the adapters {@code lora}
     * repeats: the encoded layer holds the adapter-scoped key, the processed layer the plain item.
     */
    public boolean hasCached(MediaLimits.Modality modality, String uuid, LoraSlotKey lora) {
        Objects.requireNonNull(modality, "modality");
        if (uuid == null) {
            return false;
        }
        MediaKey key = keyOf(modality, uuid, null, lora);
        if (cache.contains(key) || processed.contains(keyOf(modality, uuid, null))) {
            return true;
        }
        return sharedCache.holds(key);
    }

    // ------------------------------------------------------------------ encoding

    /** The encoding of an image: the encoded layer's or the shared encoder cache's, else its prepared tiles'. */
    private List<MediaPiece> encodedImage(MediaItem source) {
        List<MediaPiece> hit = sharedCache.shared(source);
        if (hit != null) {
            return hit;
        }
        return encodeUnderLora(source, () -> {
            List<MediaPiece> again = cache.peek(source.key());
            if (again != null) {
                return again;
            }
            ProcessedMedia prepared = preparedImage(source);
            return sharedCache.store(source, layoutImage(prepared.image));
        });
    }

    /**
     * Runs {@code encode} with the item's request's adapters applied: a request that selects any holds the context
     * monitor for the whole encode, opens a LoRA session on the vision encoder whose slot registers the adapters'
     * vision pairs in place of the startup ones (a bare key registers none), and closes the session after, so no
     * other encode's tiles run through the session and no other request waits mid-image. A request without adapters
     * encodes as before, holding the monitor per tile.
     */
    private List<MediaPiece> encodeUnderLora(MediaItem source, Supplier<List<MediaPiece>> encode) {
        if (source.lora() == null || !hasVision) {
            // no adapters to apply, or an audio-only projector whose encodes run none
            return encode.get();
        }
        synchronized (this) {
            VisionEncoder vision = loaded().vision;
            int slot = vision.openLoraSession(bindingsOf(source.lora()));
            try {
                return encode.get();
            } finally {
                vision.closeLoraSession(slot);
            }
        }
    }

    /** The bindings of the request's adapter key: each part's adapter at its scale, in the key's order. */
    private static List<LoraAttachment.Binding> bindingsOf(LoraSlotKey lora) {
        List<LoraAttachment.Binding> out = new ArrayList<>(lora.parts().size());
        for (LoraSlotKey.Part part : lora.parts()) {
            out.add(new LoraAttachment.Binding(part.handle().adapter(), part.scale()));
        }
        return out;
    }

    /** The prepared tiles of an image: the processed layer's, else decoded and preprocessed from its bytes. */
    private ProcessedMedia preparedImage(MediaItem source) {
        ProcessedMedia hit = processed.get(source.baseKey());
        if (hit != null) {
            return hit;
        }
        RasterImage image;
        try {
            image = ImageDecoder.decode(source.bytes(), io.image());
        } catch (ImageDecodeException e) {
            throw new MediaInputException("the image cannot be decoded: " + e.getMessage(), e);
        }
        PreparedImages prepared;
        try {
            prepared = preprocessor.prepare(image);
        } catch (IllegalArgumentException e) {
            throw new MediaInputException("the image cannot be preprocessed for the projector: " + e.getMessage(), e);
        }
        ProcessedMedia out = ProcessedMedia.image(prepared);
        processed.put(source.baseKey(), out);
        return out;
    }

    /** The marker text and row pieces of a prepared image, in mtmd's order (mtmd_tokenizer::add_media). */
    private List<MediaPiece> layoutImage(PreparedImages prepared) {
        List<MediaPiece> out = new ArrayList<>();
        out.add(MediaPiece.text(markers.imageBegin()));
        boolean tiled = (prepared.gridX() > 0 && prepared.gridY() > 0) || prepared.hasOverview();
        if (prepared.isAnyres()) {
            out.add(anyresChunk(prepared));
        } else if (tiled && markers.canvasImages()) {
            layoutCanvas(prepared, out);
        } else if (tiled) {
            layoutTiles(prepared, out);
        } else {
            out.add(singleChunk(prepared));
        }
        out.add(MediaPiece.text(markers.imageEnd()));
        return Collections.unmodifiableList(out);
    }

    /**
     * A llava-uhd image: the overview and the slice grid with the family's slice template
     * ({@link MediaMarkers#tiledLayout}).
     */
    private void layoutTiles(PreparedImages prepared, List<MediaPiece> out) {
        List<ImageTile> tiles = slices(prepared);
        int cols = tiles.isEmpty() ? 0 : prepared.gridX();
        int rows = tiles.isEmpty() ? 0 : prepared.gridY();
        ImageTile overview = prepared.overview();
        List<ClipImage> images = new ArrayList<>(tiles.size() + 1);
        images.add(clipImage(overview).withViewSeparator(markers.overviewViewSeparator()));
        for (ImageTile tile : tiles) {
            images.add(clipImage(tile));
        }
        List<float[][]> encoded = encodeTiles(images);
        for (MediaMarkers.LayoutPart part : markers.tiledLayout(cols, rows)) {
            switch (part.kind()) {
                case TEXT:
                    out.add(MediaPiece.text(part.text()));
                    break;
                case IMAGE_ID:
                    out.add(MediaPiece.imageId());
                    break;
                case OVERVIEW:
                    out.add(MediaPiece.rows(encoded.get(0)));
                    break;
                default:
                    out.add(MediaPiece.rows(encoded.get(part.slice() + 1)));
                    break;
            }
        }
    }

    /** The slices of a llava-uhd image, checked against its grid; the image must have its overview. */
    private static List<ImageTile> slices(PreparedImages prepared) {
        if (!prepared.hasOverview()) {
            throw new MediaInputException("the image preprocessing produced slices without an overview image");
        }
        List<ImageTile> tiles = prepared.tiles();
        int cols = tiles.isEmpty() ? 0 : prepared.gridX();
        int rows = tiles.isEmpty() ? 0 : prepared.gridY();
        if (tiles.size() != cols * rows) {
            throw new IllegalStateException("the image preprocessing produced " + tiles.size()
                    + " slices for a " + prepared.gridX() + "x" + prepared.gridY() + " grid");
        }
        return tiles;
    }

    /**
     * A MiniCPM-V 4.7 image ({@code add_canvas_chunk}): the {@link MediaMarkers#canvasLayout} text, then one chunk
     * of the overview's rows and every slice's rows, row by row, each followed by its separator rows: the
     * overview's {@code </image>} (and {@code <slice>} before the slices), a slice's {@code </slice><slice>},
     * {@code </slice>\n<slice>} at the end of a row and {@code </slice>} after the last.
     */
    private void layoutCanvas(PreparedImages prepared, List<MediaPiece> out) {
        List<ImageTile> slices = slices(prepared);
        int cols = slices.isEmpty() ? 0 : prepared.gridX();
        int rows = slices.isEmpty() ? 0 : prepared.gridY();
        List<ImageTile> entries = new ArrayList<>(1 + slices.size());
        List<ClipSuffix> suffixes = new ArrayList<>(1 + slices.size());
        entries.add(prepared.overview());
        suffixes.add(slices.isEmpty() ? ClipSuffix.MINICPMV_OV : ClipSuffix.MINICPMV_OV_SLICE);
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                entries.add(slices.get(y * cols + x));
                suffixes.add(y == rows - 1 && x == cols - 1 ? ClipSuffix.MINICPMV_LAST
                        : x == cols - 1 ? ClipSuffix.MINICPMV_ROW_END : ClipSuffix.MINICPMV_SLICE);
            }
        }
        for (MediaMarkers.LayoutPart part : markers.canvasLayout()) {
            switch (part.kind()) {
                case TEXT:
                    out.add(MediaPiece.text(part.text()));
                    break;
                case IMAGE_ID:
                    out.add(MediaPiece.imageId());
                    break;
                default:
                    out.add(canvasChunk(entries, suffixes, cols, rows));
                    break;
            }
        }
    }

    /**
     * An anyres image as one chunk ({@link VisionEncoder#encodeAnyres}): the overview's rows, then the tile grid
     * cropped to the original aspect ratio with a newline row per token row, laid out as one token row.
     */
    private MediaPiece anyresChunk(PreparedImages prepared) {
        List<ClipImage> tiles = new ArrayList<>(prepared.tiles().size());
        for (ImageTile tile : prepared.tiles()) {
            tiles.add(clipImage(tile));
        }
        ImageSize original = prepared.anyresOriginal();
        float[][] rows;
        synchronized (this) {
            try {
                rows = loaded().vision.encodeAnyres(clipImage(prepared.overview()), tiles, prepared.gridX(),
                        prepared.gridY(), original.width(), original.height());
            } catch (IllegalArgumentException e) {
                throw new MediaInputException("the " + prepared.gridX() + "x" + prepared.gridY() + " anyres grid of"
                        + " the " + original.width() + "x" + original.height() + " image does not fit the vision"
                        + " encoder: " + e.getMessage(), e);
            }
        }
        return MediaPiece.grid(rows, rows.length, 1);
    }

    /**
     * The canvas chunk of {@code entries}: every entry encoded with its suffix, its token grid scaled from the last
     * entry's ({@code canvas_tile_grid}).
     */
    private MediaPiece canvasChunk(List<ImageTile> entries, List<ClipSuffix> suffixes, int cols, int rows) {
        ImageSize refSize = entries.get(entries.size() - 1).size();
        ImageSize ref = imageParams.tokenGridFor(refSize);
        List<ClipImage> images = new ArrayList<>(entries.size());
        for (int k = 0; k < entries.size(); k++) {
            images.add(clipImage(entries.get(k)).withSuffix(suffixes.get(k)));
        }
        List<float[][]> encoded = encodeTiles(images);
        List<MediaRopePositions.CanvasTile> tiles = new ArrayList<>(entries.size());
        int total = 0;
        for (int k = 0; k < entries.size(); k++) {
            ImageTile e = entries.get(k);
            ClipSuffix suffix = suffixes.get(k);
            int gw = (int) ((long) e.width() * ref.width() / refSize.width());
            int gh = (int) ((long) e.height() * ref.height() / refSize.height());
            float[][] tileRows = encoded.get(k);
            if (gw <= 0 || gh <= 0 || tileRows.length != gw * gh + suffix.size()) {
                throw new IllegalStateException("the canvas tile " + e.width() + "x" + e.height() + " encoded to "
                        + tileRows.length + " rows, its " + gw + "x" + gh + " token grid and " + suffix
                        + " separators need " + (gw * gh + suffix.size()));
            }
            tiles.add(new MediaRopePositions.CanvasTile(gw, gh, suffix.size()));
            total += tileRows.length;
        }
        float[][] all = new float[total][];
        int at = 0;
        for (float[][] e : encoded) {
            System.arraycopy(e, 0, all, at, e.length);
            at += e.length;
        }
        return MediaPiece.canvas(all, cols, rows, tiles);
    }

    /**
     * An unsliced image as one chunk: the rows of every tile in order, laid out on the merged token grid of the
     * first tile for M-RoPE ({@code clip_n_output_tokens_x/y}) when the rows fill that grid.
     */
    private MediaPiece singleChunk(PreparedImages prepared) {
        List<ImageTile> tiles = prepared.encodedTiles();
        if (tiles.isEmpty()) {
            throw new MediaInputException("the image preprocessing produced no image tiles");
        }
        int align = markers.imageLeadPadAlign();
        if (align > 0) {
            return leadPaddedChunk(tiles, align - 1);
        }
        List<ClipImage> images = new ArrayList<>(tiles.size());
        for (ImageTile tile : tiles) {
            images.add(clipImage(tile));
        }
        List<float[][]> encoded = encodeTiles(images);
        int total = 0;
        for (float[][] rows : encoded) {
            total += rows.length;
        }
        float[][] rows = new float[total][];
        int at = 0;
        for (float[][] e : encoded) {
            System.arraycopy(e, 0, rows, at, e.length);
            at += e.length;
        }
        ImageSize grid = imageParams.tokenGridFor(tiles.get(0).size());
        if (tiles.size() == 1 && grid.width() * grid.height() == total) {
            return MediaPiece.grid(rows, grid.width(), grid.height());
        }
        return MediaPiece.grid(rows, total, 1);
    }

    /**
     * A single-tile image whose block opens with lead pads: encoded once with the most pads, {@code maxLead}, and
     * trimmed at append time to the pads its place in the prompt takes (a block with fewer pads is the same block
     * without its first pad rows).
     */
    private MediaPiece leadPaddedChunk(List<ImageTile> tiles, int maxLead) {
        if (tiles.size() != 1) {
            throw new IllegalStateException("a lead-padded image block encodes one tile, the preprocessing produced "
                    + tiles.size());
        }
        ImageTile tile = tiles.get(0);
        return MediaPiece.leadPadded(encodeTile(clipImage(tile).withLeadPad(maxLead), tile), maxLead);
    }

    private float[][] encodeTile(ImageTile tile) {
        return encodeTile(tile, ClipSuffix.NONE);
    }

    private ClipImage clipImage(ImageTile tile) {
        return ClipImage.of(tile.width(), tile.height(), tile.normalized(imageParams.mean(), imageParams.std()));
    }

    /** The rows of {@code tile} followed by the separator rows of {@code suffix}. */
    private float[][] encodeTile(ImageTile tile, ClipSuffix suffix) {
        return encodeTile(clipImage(tile).withSuffix(suffix), tile);
    }

    private float[][] encodeTile(ClipImage image, ImageTile tile) {
        synchronized (this) {
            try {
                return loaded().vision.encode(image);
            } catch (IllegalArgumentException e) {
                throw new MediaInputException("the image tile " + tile.width() + "x" + tile.height()
                        + " does not fit the vision encoder: " + e.getMessage(), e);
            }
        }
    }

    /**
     * The rows of every image of one item as one encode batch ({@link VisionEncoder#encodeBatch}), in input order:
     * a data-parallel encoder spreads the batch over its ranks, every other encoder runs the images in turn.
     */
    private List<float[][]> encodeTiles(List<ClipImage> images) {
        synchronized (this) {
            try {
                return loaded().vision.encodeBatch(images);
            } catch (IllegalArgumentException e) {
                throw new MediaInputException("the image tiles do not fit the vision encoder: " + e.getMessage(), e);
            }
        }
    }

    /** The encoding of an audio clip: the encoded layer's or the shared encoder cache's, else its decoded samples'. */
    private List<MediaPiece> encodedAudio(MediaItem source, String name) {
        List<MediaPiece> hit = sharedCache.shared(source);
        if (hit != null) {
            return hit;
        }
        return encodeUnderLora(source, () -> {
            List<MediaPiece> again = cache.peek(source.key());
            if (again != null) {
                return again;
            }
            ProcessedMedia samples = decodedAudio(source, name);
            List<float[][]> chunks;
            synchronized (this) {
                chunks = loaded().audio.embed(samples.audio);
            }
            if (chunks.isEmpty()) {
                throw new MediaInputException("the audio produced no encoder windows");
            }
            List<MediaPiece> out = new ArrayList<>();
            out.add(MediaPiece.text(markers.audioBegin()));
            for (float[][] rows : chunks) {
                out.add(MediaPiece.rows(rows));
            }
            out.add(MediaPiece.text(markers.audioEnd()));
            return sharedCache.store(source, Collections.unmodifiableList(out));
        });
    }

    /** The decoded samples of an audio clip: the processed layer's, else decoded from its bytes. */
    private ProcessedMedia decodedAudio(MediaItem source, String name) {
        ProcessedMedia hit = processed.get(source.baseKey());
        if (hit != null) {
            return hit;
        }
        AudioClip clip;
        try {
            clip = AudioDecoder.decode(source.bytes(), name);
        } catch (IOException | IllegalArgumentException e) {
            throw new MediaInputException("the audio cannot be decoded: " + e.getMessage(), e);
        }
        ProcessedMedia out = ProcessedMedia.audio(clip);
        processed.put(source.baseKey(), out);
        return out;
    }

    // ------------------------------------------------------------------ cache

    /** The bytes the cached encodings hold. */
    public long cachedBytes() {
        return cache.bytes();
    }

    /** The number of cached encodings. */
    public int cachedItems() {
        return cache.size();
    }

    /** The byte budget of each cache layer. */
    public long cacheBudget() {
        return cache.maxBytes();
    }

    /** The encoded-layer lookups that found their item, since the context opened (vLLM's {@code mm_cache_hits}). */
    public long cacheHits() {
        return cache.hits();
    }

    /** The encoded-layer lookups made (one per media item appended), since the context opened. */
    public long cacheQueries() {
        return cache.queries();
    }

    /** The bytes the processed layer's tiles and samples hold. */
    public long processedBytes() {
        return processed.bytes();
    }

    /** The number of processed items cached. */
    public int processedItems() {
        return processed.size();
    }

    /** The processed-layer lookups (made on encoded-layer misses) that found their item, since the context opened. */
    public long processedHits() {
        return processed.hits();
    }

    /** The processed-layer lookups made, since the context opened. */
    public long processedQueries() {
        return processed.queries();
    }

    /** The items this consumer loaded from the shared encoder cache, since the context opened. */
    public long sharedLoads() {
        return sharedCache.loads();
    }

    /** The items this producer saved to the shared encoder cache, since the context opened. */
    public long sharedSaves() {
        return sharedCache.saves();
    }

    /**
     * Drops both cache layers (vLLM's {@code reset_mm_cache}): the next request decodes and encodes its media
     * anew, and a uuid sent alone needs its data again.
     */
    public void clearCache() {
        cache.clear();
        processed.clear();
    }

    /**
     * Drops the cached encodings only (vLLM's {@code reset_encoder_cache}): the next request of an item encodes
     * it again from its processed tiles or samples while the processed layer holds them.
     */
    public void clearEncodings() {
        cache.clear();
    }

    // ------------------------------------------------------------------ release and reopen

    /** Whether the projector weights are released (between {@link #release()} and {@link #reopen()}). */
    public boolean released() {
        return released;
    }

    /**
     * Releases the projector weights and encoders while the text model sleeps (vLLM's sleep level 2 discards the
     * weights): the markers, widths and capabilities stay, cached media stay until {@link #clearCache()}, and
     * an encode refuses with {@link ReleasedException} until {@link #reopen()}. Waits for an encode in flight;
     * releasing a released or closed context does nothing.
     *
     * @throws IOException when the projector file does not close
     */
    public synchronized void release() throws IOException {
        if (closed || encoders == null) {
            return;
        }
        MmprojEncoders current = encoders;
        encoders = null;
        released = true;
        current.close();
    }

    /**
     * Re-opens the released projector from its file on the encoder device it opened on, re-checking the width
     * against the text model; reopening a loaded context does nothing.
     *
     * @throws IOException              when the file cannot be read (the context stays released)
     * @throws IllegalStateException    when the context is closed
     * @throws IllegalArgumentException when the file no longer fits the text model (the context stays released)
     */
    public synchronized void reopen() throws IOException {
        ensureOpen();
        if (encoders != null) {
            return;
        }
        MmprojEncoders loaded = MmprojEncoders.open(path, encoderOptions, width, loras);
        if ((loaded.vision != null) != hasVision || (loaded.audio != null) != hasAudio) {
            loaded.close();
            throw new IllegalArgumentException("the projector " + path + " changed its encoders since it opened");
        }
        encoders = loaded;
        visionDevice = hasVision ? loaded.vision.device() : null;
        released = false;
    }

    /** The loaded encoders; the caller holds {@code this}. */
    private MmprojEncoders loaded() {
        ensureOpen();
        if (encoders == null) {
            throw new ReleasedException();
        }
        return encoders;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the media context is closed");
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        cache.clear();
        processed.clear();
        MmprojEncoders current = encoders;
        encoders = null;
        IOException failure = null;
        if (current != null) {
            try {
                current.close();
            } catch (IOException e) {
                failure = e;
            }
        }
        try {
            sharedCache.close();
        } catch (IOException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "MediaContext{vision=" + (imageParams == null ? "none"
                : imageParams.projectorName() + " on " + visionDevice())
                + ", audio=" + (audioProjector == null ? "none" : audioProjector)
                + ", mrope=" + mrope + ", width=" + width + (released ? ", released" : "") + '}';
    }
}

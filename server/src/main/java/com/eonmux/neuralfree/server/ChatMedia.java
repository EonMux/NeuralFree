package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioDecoder;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.media.MediaLimits.Modality;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConnector;
import com.eonmux.neuralfree.gguf.mm.MediaContext;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.video.VideoDecoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The media inputs of the serving routes, llama.cpp's {@code oaicompat_content_load_media} and vLLM's chat content
 * parser over the projector's {@link MediaContext}: it reads the media content parts of a chat request
 * ({@code image_url}, {@code input_audio}, {@code audio_url}, {@code video_url}, {@code input_video}) or the
 * {@code multimodal_data} of a raw prompt, checks
 * them against the per-prompt limits, loads their bytes ({@link MediaConnector}) and builds the media prompt of a
 * rendered text whose {@link ChatMessages#MEDIA_MARKER} and {@link ChatMessages#PROMPT_EMBEDS_MARKER} markers stand
 * where the parts sat. Without a projector every media part is refused with llama.cpp's hint to provide one.
 *
 * <p>A media part may carry vLLM's {@code uuid}: the client's cache key of the item. When the projector's media
 * cache holds that uuid, encoded or processed, the part's URL or data is not loaded and the cache serves it, so a
 * repeated item may be sent as the uuid alone ({@code "image_url": null}); a uuid the cache does not hold must come
 * with its data.
 * The projector itself {@link #release() releases} while the engine sleeps and {@link #reopen() reopens} on the
 * wake. Thread-safe; close it to release the projector.</p>
 */
final class ChatMedia implements AutoCloseable {

    /** llama.cpp's refusal of a media part the projector cannot take (server-common.cpp). */
    private static final String HINT = " - hint: if this is unexpected, you may need to provide the mmproj";
    private static final Pattern MARKERS = Pattern.compile(Pattern.quote(ChatMessages.MEDIA_MARKER) + "|"
            + Pattern.quote(ChatMessages.PROMPT_EMBEDS_MARKER));

    /**
     * One media item: its modality, the client's cache key, its bytes and the request field it came from.
     *
     * @param modality the item's modality
     * @param uuid     the client's {@code uuid}, or null
     * @param data     the item's bytes, loaded on first read (the URL of a part whose uuid was cached when the
     *                 request was read loads only if the cache dropped it since); null for a uuid sent alone
     * @param field    the request field, for error messages
     */
    record Item(Modality modality, String uuid, Supplier<byte[]> data, String field) {

        /** An item of loaded {@code bytes} without a uuid. */
        static Item of(Modality modality, byte[] bytes, String field) {
            return new Item(modality, null, () -> bytes, field);
        }

        /** The item's bytes, loading them now when deferred; null for a uuid sent alone. */
        byte[] bytes() {
            return data == null ? null : data.get();
        }
    }

    /** Tokenizes prompt text segments: special tokens parsed, none added. */
    interface TextEncoder extends MediaContext.TextEncoder { }

    /**
     * One item's place in a built media prompt: the item's key (vLLM's {@code mm_hash}), the media item the
     * projector encodes, the precomputed rows of an embedding part ({@code null} for a media item) and the
     * half-open prompt-row span the item contributes, marker text included.
     */
    record Placement(MediaKey key, Item item, float[][] rows, int offset, int length) { }

    /** A built media prompt and the placements of the items it spliced in, in prompt order. */
    record Rendered(MediaPrompt prompt, List<Placement> placements) { }

    /**
     * One item to splice into an already rendered token sequence: its key, the media the projector encodes (or the
     * precomputed rows of an embedding part), and the row span it occupies in the token ids.
     */
    record Placed(MediaKey key, Item item, float[][] rows, int offset, int length) { }

    private final MediaContext context;
    private final MediaConnector connector;
    private final MediaLimits limits;

    private ChatMedia(MediaContext context, MediaConnector connector, MediaLimits limits) {
        this.context = context;
        this.connector = connector;
        this.limits = limits;
    }

    /**
     * The media inputs of {@code options}: the projector opens for a text model of embedding width
     * {@code textEmbd} when one is configured, its vision encoder on the device {@link MediaOptions#encoder()}
     * chooses.
     *
     * @throws IOException when the projector cannot be read
     */
    static ChatMedia open(MediaOptions options, int textEmbd, boolean mrope, Predicate<String> hasToken)
            throws IOException {
        return open(options, textEmbd, mrope, hasToken, List.of());
    }

    /**
     * {@link #open(MediaOptions, int, boolean, Predicate)} with the adapters {@code loras} whose vision-encoder
     * and connector pairs the projector applies (the startup LoRA adapters of the text model).
     *
     * @throws IOException when the projector cannot be read, or the encoder cache directory cannot be used
     */
    static ChatMedia open(MediaOptions options, int textEmbd, boolean mrope, Predicate<String> hasToken,
                          List<LoraAttachment.Binding> loras) throws IOException {
        EncoderCacheConnector connector = options.encoderCache() == null ? null : options.encoderCache().open();
        MediaContext context;
        try {
            context = options.mmproj() == null ? null
                    : MediaContext.open(options.mmproj(), textEmbd, mrope, options.budget(), hasToken,
                            options.encoder(), options.cacheBytes(), options.io(), options.video(), connector,
                            loras);
        } catch (IOException | RuntimeException e) {
            if (connector != null) {
                connector.close();
            }
            throw e;
        }
        if (context == null && connector != null) {
            connector.close();
        }
        return new ChatMedia(context, new MediaConnector(options.connector()), options.limits());
    }

    /** Media inputs without a projector: every media part refuses. */
    static ChatMedia disabled() {
        return new ChatMedia(null, new MediaConnector(MediaConnector.Options.defaults()), MediaLimits.defaults());
    }

    /** Whether a projector is loaded. */
    boolean enabled() {
        return context != null;
    }

    /** Whether image input is accepted. */
    boolean vision() {
        return context != null && context.supportsVision();
    }

    /** Whether audio input is accepted. */
    boolean audio() {
        return context != null && context.supportsAudio();
    }

    /** Whether video input is accepted: a vision encoder, whose frames the video path encodes. */
    boolean video() {
        return context != null && context.supportsVideo();
    }

    // ------------------------------------------------------------------ cache and projector lifecycle

    /**
     * Drops the projector's media cache, encodings and processed media (vLLM's {@code reset_mm_cache}); nothing to
     * do without a projector.
     */
    void clearCache() {
        if (context != null) {
            context.clearCache();
        }
    }

    /**
     * Drops the projector's cached encodings only, keeping the processed media (part of vLLM's
     * {@code reset_encoder_cache}); nothing to do without a projector.
     */
    void clearEncodings() {
        if (context != null) {
            context.clearEncodings();
        }
    }

    /** The media items looked up in the projector's cache, 0 without a projector. */
    long cacheQueries() {
        return context == null ? 0 : context.cacheQueries();
    }

    /**
     * The media items the projector's cache served without decoding them (an encoding, or processed tiles or
     * samples to encode), 0 without a projector.
     */
    long cacheHits() {
        return context == null ? 0 : context.cacheHits() + context.processedHits();
    }

    /** The bytes the projector's cache holds, encodings and processed media, 0 without a projector. */
    long cacheUsageBytes() {
        return context == null ? 0 : context.cachedBytes() + context.processedBytes();
    }

    /** The cached encodings' bytes, 0 without a projector. */
    long cachedBytes() {
        return context == null ? 0 : context.cachedBytes();
    }

    /** The items a consumer loaded from the shared encoder cache, 0 without a projector. */
    long sharedLoads() {
        return context == null ? 0 : context.sharedLoads();
    }

    /** The items a producer saved to the shared encoder cache, 0 without a projector. */
    long sharedSaves() {
        return context == null ? 0 : context.sharedSaves();
    }

    /** Whether the projector weights are released (the engine sleeps with its weights released). */
    boolean released() {
        return context != null && context.released();
    }

    /** Releases the projector weights while every model sleeps released (a sleep of level 2). */
    void release() throws IOException {
        if (context != null) {
            context.release();
        }
    }

    /**
     * Re-opens the released projector from its file (the wake); a loaded projector stays as it is.
     *
     * @throws IOException              when the file cannot be read
     * @throws IllegalArgumentException when the file no longer fits the model
     */
    void reopen() throws IOException {
        if (context != null) {
            context.reopen();
        }
    }

    /**
     * The bytes of one image reference of a pooling input (a CLIP dual encoder's image side), loaded under the
     * connector's rules; no projector is involved.
     *
     * @param reference an http(s), data or file URL, or raw base64
     * @param field     the request field the reference came from, named in every refusal
     * @throws MediaInputException when the reference does not load
     */
    byte[] imageBytes(String reference, String field) {
        return connector.load(reference, field).bytes();
    }

    // ------------------------------------------------------------------ chat parts

    /**
     * The media items of the chat's content parts, in message and part order, loaded: the order the rendered
     * prompt's media markers take. A part whose {@code uuid} the media cache holds defers its load.
     *
     * @throws MediaInputException when a part is malformed, its modality is not accepted, the parts exceed the
     *                             per-prompt limits, a URL does not load, or a uuid sent alone is not cached
     */
    List<Item> items(ChatRequest chat) {
        return items(chat, () -> null);
    }

    /**
     * {@link #items(ChatRequest)} for a request naming adapters {@code mediaLoras}: a part's uuid is looked up in
     * the cache under the adapters' identity (vLLM keys the cache entries by the request's lora name and the media
     * hash), so a uuid the request's own adapters encoded needs no data while the same uuid under other adapters
     * does.
     *
     * @throws MediaInputException when a part is malformed, its modality is not accepted, the parts exceed the
     *                             per-prompt limits, a URL does not load, or a uuid sent alone is not cached
     */
    List<Item> items(ChatRequest chat, Supplier<LoraSlotKey> mediaLoras) {
        List<Map<?, ?>> parts = new ArrayList<>();
        List<String> where = new ArrayList<>();
        int images = 0;
        int audios = 0;
        int videos = 0;
        for (int m = 0; m < chat.messages().size(); m++) {
            if (!(chat.messages().get(m).get(ChatMessages.CONTENT) instanceof List<?> content)) {
                continue;
            }
            for (int p = 0; p < content.size(); p++) {
                if (!(content.get(p) instanceof Map<?, ?> part)) {
                    continue; // the template input refuses non-object parts by name
                }
                Modality modality = modalityOf(part.get("type"));
                if (modality == null) {
                    continue;
                }
                requireAccepted(modality);
                if (modality == Modality.IMAGE) {
                    images++;
                } else if (modality == Modality.AUDIO) {
                    audios++;
                } else {
                    videos++;
                }
                parts.add(part);
                where.add("messages[" + m + "].content[" + p + "]." + part.get("type"));
            }
        }
        limits.check(Modality.IMAGE, images);
        limits.check(Modality.AUDIO, audios);
        limits.check(Modality.VIDEO, videos);
        // the adapters' identity resolves on the replica's pump thread, so a request without media parts never
        // reaches it: its text path needs no adapter identity
        LoraSlotKey lora = parts.isEmpty() ? null : mediaLoras.get();
        List<Item> out = new ArrayList<>(parts.size());
        for (int i = 0; i < parts.size(); i++) {
            out.add(load(parts.get(i), where.get(i), lora));
        }
        return out;
    }

    /** The modality of a media part type, or null for every other part. */
    private static Modality modalityOf(Object type) {
        if (ChatMessages.PART_IMAGE.equals(type)) {
            return Modality.IMAGE;
        }
        if (ChatMessages.PART_AUDIO.equals(type) || ChatMessages.PART_AUDIO_URL.equals(type)) {
            return Modality.AUDIO;
        }
        if (ChatMessages.PART_VIDEO_URL.equals(type) || ChatMessages.PART_INPUT_VIDEO.equals(type)) {
            return Modality.VIDEO;
        }
        return null;
    }

    private void requireAccepted(Modality modality) {
        if (modality == Modality.IMAGE && !vision()) {
            throw new MediaInputException("image input is not supported" + HINT);
        }
        if (modality == Modality.AUDIO && !audio()) {
            throw new MediaInputException("audio input is not supported" + HINT);
        }
        if (modality == Modality.VIDEO && !video()) {
            throw new MediaInputException("video input is not supported" + HINT);
        }
    }

    /**
     * One media part as an item: without a uuid its data loads now; with a uuid the cache holds under the request's
     * adapters {@code lora}, the load defers to a cache miss; with a uuid the cache does not hold, the data must be
     * there and loads now.
     */
    private Item load(Map<?, ?> part, String field, LoraSlotKey lora) {
        Modality modality = modalityOf(part.get("type"));
        String uuid = uuidOf(part, field);
        Supplier<byte[]> source = source(part, field, uuid != null);
        if (uuid != null && context.hasCached(modality, uuid, lora)) {
            return new Item(modality, uuid, source == null ? null : new Deferred(source), field);
        }
        if (source == null) {
            throw new MediaInputException(field + ": the " + modality.key() + " with uuid '" + uuid + "' is not in"
                    + " the media cache and the part carries no data; send it with its data");
        }
        byte[] bytes = source.get();
        return new Item(modality, uuid, () -> bytes, field);
    }

    /**
     * The loader of a part's bytes: its image, audio or video URL, or its inline audio or video data; null when a
     * part with a uuid carries none (vLLM accepts a null, empty or absent media value beside a uuid).
     */
    private Supplier<byte[]> source(Map<?, ?> part, String field, boolean hasUuid) {
        Object type = part.get("type");
        if (ChatMessages.PART_IMAGE.equals(type) || ChatMessages.PART_AUDIO_URL.equals(type)
                || ChatMessages.PART_VIDEO_URL.equals(type)) {
            Object value = part.get(type);
            Object url = value instanceof Map<?, ?> m ? m.get("url") : value;
            if (hasUuid && (url == null || "".equals(url))) {
                return null;
            }
            String target = url(value, field);
            return () -> connector.load(target, field + ".url").bytes();
        }
        Object inner = part.get(ChatMessages.PART_AUDIO.equals(type) ? "input_audio" : "input_video");
        if (inner == null && hasUuid) {
            return null;
        }
        if (!(inner instanceof Map<?, ?> media)) {
            throw new MediaInputException(field + " requires an '" + innerName(type) + "' object with 'data' or"
                    + " 'url'");
        }
        if (media.get("data") instanceof String data && !data.isEmpty()) {
            return () -> connector.inline(data, field + ".data").bytes();
        }
        if (media.get("url") instanceof String url) {
            return () -> connector.load(url, field + ".url").bytes();
        }
        if (hasUuid) {
            return null;
        }
        throw new MediaInputException(field + " requires 'data' (base64 " + innerName(type) + ") or 'url'");
    }

    /** The inner field name of an inline media part: {@code input_audio} or {@code input_video}. */
    private static String innerName(Object type) {
        return ChatMessages.PART_AUDIO.equals(type) ? "input_audio" : "input_video";
    }

    /**
     * A part's {@code uuid} (vLLM's media cache key), or null: a string, or a number taken as its text.
     *
     * @throws MediaInputException when the uuid is blank or neither a string nor a number
     */
    private static String uuidOf(Map<?, ?> part, String field) {
        Object uuid = part.get("uuid");
        if (uuid == null) {
            return null;
        }
        String text;
        if (uuid instanceof String s) {
            text = s;
        } else if (uuid instanceof Number n) {
            text = n.toString();
        } else {
            throw new MediaInputException(field + ".uuid must be a string");
        }
        if (text.isBlank()) {
            throw new MediaInputException(field + ".uuid must not be blank");
        }
        return text;
    }

    /** The URL of an {@code image_url} / {@code audio_url} value: an object with {@code url}, or the URL itself. */
    private static String url(Object value, String field) {
        if (value instanceof String s) {
            return s;
        }
        if (value instanceof Map<?, ?> m && m.get("url") instanceof String s) {
            return s;
        }
        throw new MediaInputException(field + " requires a 'url'");
    }

    /** A source loaded once, on first read: the deferred data of a part whose uuid was cached. */
    private static final class Deferred implements Supplier<byte[]> {
        private final Supplier<byte[]> source;
        private byte[] bytes;

        Deferred(Supplier<byte[]> source) {
            this.source = source;
        }

        @Override
        public synchronized byte[] get() {
            if (bytes == null) {
                bytes = source.get();
            }
            return bytes;
        }
    }

    // ------------------------------------------------------------------ raw prompts

    /**
     * The media items of a raw prompt's {@code multimodal_data} (llama.cpp's {@code /completion} prompt object):
     * base64 entries, each sniffed as video, audio or image the way {@code mtmd_helper_bitmap_init_from_buf}
     * does, the video containers before the audio ones (an MP4 video also carries the {@code ftyp} the audio
     * sniff reads as M4A).
     *
     * @throws MediaInputException when an entry is not base64, its modality is not accepted, or the entries exceed
     *                             the per-prompt limits
     */
    List<Item> items(List<?> multimodalData, String field) {
        List<Item> out = new ArrayList<>(multimodalData.size());
        int images = 0;
        int audios = 0;
        int videos = 0;
        for (int i = 0; i < multimodalData.size(); i++) {
            String at = field + "[" + i + "]";
            if (!(multimodalData.get(i) instanceof String data)) {
                throw new MediaInputException(at + " must be a base64 string");
            }
            byte[] bytes = connector.inline(data, at).bytes();
            Modality modality = VideoDecoder.isVideo(bytes) ? Modality.VIDEO
                    : AudioDecoder.isAudio(bytes) ? Modality.AUDIO : Modality.IMAGE;
            requireAccepted(modality);
            if (modality == Modality.IMAGE) {
                images++;
            } else if (modality == Modality.AUDIO) {
                audios++;
            } else {
                videos++;
            }
            out.add(Item.of(modality, bytes, at));
        }
        limits.check(Modality.IMAGE, images);
        limits.check(Modality.AUDIO, audios);
        limits.check(Modality.VIDEO, videos);
        return out;
    }

    // ------------------------------------------------------------------ decision requests

    /**
     * The media items of a decision request's {@code files} or {@code images} array (llama.cpp's {@code
     * decision_load_image}): every entry a data URL, sniffed as video, audio or image the way {@code
     * mtmd_helper_bitmap_init_from_buf} does.
     *
     * @throws MediaInputException when an entry is not a data URL, its modality is not accepted, or it does not load
     */
    List<Item> decisionItems(List<?> urls, String field) {
        List<Item> out = new ArrayList<>(urls.size());
        for (int i = 0; i < urls.size(); i++) {
            String at = field + "[" + i + "]";
            if (!(urls.get(i) instanceof String url) || !url.startsWith("data:")) {
                throw new MediaInputException("media must be data URLs (data:image/..., data:audio/... or"
                        + " data:video/...;base64,...)");
            }
            byte[] bytes = connector.load(url, at).bytes();
            Modality modality = VideoDecoder.isVideo(bytes) ? Modality.VIDEO
                    : AudioDecoder.isAudio(bytes) ? Modality.AUDIO : Modality.IMAGE;
            out.add(Item.of(modality, bytes, at));
        }
        return out;
    }

    /**
     * One audio clip of a decision request (llama.cpp's {@code decision_load_audio}): base64 data, as a data URL
     * or bare.
     *
     * @throws MediaInputException when the data is a URL, is not accepted, or does not decode
     */
    Item decisionAudio(String data, String field) {
        if (data.startsWith("http") || data.startsWith("file://")) {
            throw new MediaInputException("audio must be base64 data");
        }
        byte[] bytes = data.startsWith("data:") ? connector.load(data, field).bytes()
                : connector.inline(data, field).bytes();
        return Item.of(Modality.AUDIO, bytes, field);
    }

    /** Refuses a decision item whose modality the projector does not take (llama.cpp's mtmd hint). */
    void checkDecisionItem(Item item) {
        requireAccepted(item.modality());
    }

    // ------------------------------------------------------------------ prompts

    /**
     * The media prompt of {@code rendered}: the text between markers tokenized, each
     * {@link ChatMessages#MEDIA_MARKER} replaced by the next media item's marker text and embeddings, each
     * {@link ChatMessages#PROMPT_EMBEDS_MARKER} by the next precomputed row group (one position per row) as one
     * media item, keyed by the digest of its rows ({@link com.eonmux.neuralfree.gguf.media.MediaKey#ofRows}), as
     * vLLM hashes an {@code image_embeds} tensor into one {@code mm_hash}.
     *
     * @param bos the BOS token to start the prompt with, or -1 for none
     * @throws MediaInputException        when the markers and the items or row groups do not pair up, or an item
     *                                    does not load or decode
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    MediaPrompt prompt(String rendered, List<Item> media, List<float[][]> embeds, int bos, TextEncoder text) {
        return promptWithPlacements(rendered, media, embeds, bos, text, () -> null).prompt();
    }

    /**
     * {@link #prompt(String, List, List, int, TextEncoder)} for a request naming adapters {@code mediaLoras}: each
     * media item encodes in a LoRA session over the projector (its vision pairs in place of the startup ones) and
     * its prompt chunks key by the adapters' identity.
     *
     * @throws MediaInputException        when the markers and the items or row groups do not pair up, or an item
     *                                    does not load or decode
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    MediaPrompt prompt(String rendered, List<Item> media, List<float[][]> embeds, int bos, TextEncoder text,
                       Supplier<LoraSlotKey> mediaLoras) {
        return promptWithPlacements(rendered, media, embeds, bos, text, mediaLoras).prompt();
    }

    /**
     * {@link #prompt} with the row span and key of every spliced item: a media item's span covers its marker text
     * and encoded rows, an embedding part's span its rows. A caller that sends the prompt to another server uses
     * the spans as its {@code mm_placeholders} and the keys as its {@code mm_hashes}.
     *
     * @throws MediaInputException        when the markers and the items or row groups do not pair up, or an item
     *                                    does not load or decode
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    Rendered promptWithPlacements(String rendered, List<Item> media, List<float[][]> embeds, int bos,
                                  TextEncoder text) {
        return promptWithPlacements(rendered, media, embeds, bos, text, () -> null);
    }

    /**
     * {@link #promptWithPlacements(String, List, List, int, TextEncoder)} for a request naming adapters
     * {@code mediaLoras}: each media item encodes in a LoRA session over the projector (its vision pairs in place
     * of the startup ones) and its chunks and placements key by the adapters' identity.
     *
     * @throws MediaInputException        when the markers and the items or row groups do not pair up, or an item
     *                                    does not load or decode
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    Rendered promptWithPlacements(String rendered, List<Item> media, List<float[][]> embeds, int bos,
                                  TextEncoder text, Supplier<LoraSlotKey> mediaLoras) {
        MediaPrompt.Builder prompt = MediaPrompt.builder();
        if (bos >= 0) {
            prompt.appendText(new int[] {bos});
        }
        Iterator<Item> items = media.iterator();
        Iterator<float[][]> rows = embeds.iterator();
        List<Placement> placements = new ArrayList<>(media.size() + embeds.size());
        // only media items carry the adapters' identity into their cache keys and encodes; an embeds-only prompt
        // splices rows the caller encoded and resolves nothing
        LoraSlotKey lora = media.isEmpty() ? null : mediaLoras.get();
        int images = 0;
        int markers = 0;
        int rowMarkers = 0;
        Matcher m = MARKERS.matcher(rendered);
        int from = 0;
        while (m.find()) {
            appendSegment(prompt, rendered.substring(from, m.start()), text);
            from = m.end();
            if (m.group().equals(ChatMessages.MEDIA_MARKER)) {
                markers++;
                if (!items.hasNext()) {
                    continue;
                }
                Item item = items.next();
                int at = prompt.tokens();
                append(prompt, item, images, text, lora);
                placements.add(new Placement(keyOf(item, lora), item, null, at, prompt.tokens() - at));
                if (item.modality() == Modality.IMAGE) {
                    images++;
                }
            } else {
                rowMarkers++;
                if (rows.hasNext()) {
                    float[][] group = rows.next();
                    int at = prompt.tokens();
                    prompt.addMedia(MediaRopePositions.normal(at, group.length), group, false);
                    placements.add(new Placement(MediaKey.ofRows(group), null, group, at, group.length));
                }
            }
        }
        appendSegment(prompt, rendered.substring(from), text);
        if (markers != media.size()) {
            throw new MediaInputException("the prompt holds " + markers + " media marker(s) for " + media.size()
                    + " media item(s); every image, video and audio input needs its own marker");
        }
        if (rowMarkers != embeds.size()) {
            throw new MediaInputException("the chat template rendered " + rowMarkers + " embedding placeholders for "
                    + embeds.size() + " embedding parts; the rows cannot be spliced");
        }
        if (prompt.isEmpty()) {
            throw new MediaInputException("the prompt holds no tokens and no media");
        }
        return new Rendered(prompt.build(), placements);
    }

    /**
     * The media prompt of an expanded token sequence: the text tokens outside the placed items' spans, with each
     * item spliced back in where its span sits, encoded through the media cache under the item's key. A generate
     * server reproduces the render server's prompt from its token ids and feature placeholders this way.
     *
     * @param tokenIds the expanded prompt token ids; an item's span holds one placeholder token per media row
     * @param placed   the items in prompt order, each with the row span the render server reported
     * @param text     the tokenizer of the projector's marker text
     * @throws MediaInputException when the spans overlap or leave the token ids, or the projector reproduces a
     *                             different row count for an item
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    MediaPrompt promptAt(int[] tokenIds, List<Placed> placed, TextEncoder text) {
        if (isDisabled()) {
            throw new MediaInputException("the request carries multimodal features, but this server has no"
                    + " projector; start it with --mmproj");
        }
        MediaPrompt.Builder prompt = MediaPrompt.builder();
        int cursor = 0;
        int images = 0;
        for (Placed item : placed) {
            if (item.offset() < cursor) {
                throw new MediaInputException("multimodal placeholders overlap: an item starts at row "
                        + item.offset() + " while the previous one ends at " + cursor);
            }
            if ((long) item.offset() + item.length() > tokenIds.length) {
                throw new MediaInputException("multimodal placeholder at row " + item.offset() + " of "
                        + item.length() + " row(s) leaves the " + tokenIds.length + "-token prompt");
            }
            if (item.offset() > cursor) {
                prompt.appendText(java.util.Arrays.copyOfRange(tokenIds, cursor, item.offset()));
            }
            int before = prompt.tokens();
            if (item.rows() != null) {
                prompt.addMedia(MediaRopePositions.normal(before, item.rows().length), item.rows(), false);
            } else {
                append(prompt, item.key(), item.item(), images, text);
                if (item.item().modality() == Modality.IMAGE) {
                    images++;
                }
            }
            int rows = prompt.tokens() - before;
            if (rows != item.length()) {
                throw new MediaInputException("the media item " + item.key() + " encodes to " + rows + " row(s) on"
                        + " this server, the render server reported " + item.length());
            }
            cursor = item.offset() + rows;
        }
        if (cursor < tokenIds.length) {
            prompt.appendText(java.util.Arrays.copyOfRange(tokenIds, cursor, tokenIds.length));
        }
        if (prompt.isEmpty()) {
            throw new MediaInputException("the prompt holds no tokens and no media");
        }
        return prompt.build();
    }

    /** Appends one item's marker text and encoding through the item's key. */
    private void append(MediaPrompt.Builder prompt, Item item, int imageIndex, TextEncoder text) {
        append(prompt, keyOf(item), item, imageIndex, text);
    }

    /**
     * Appends one item's marker text and encoding under the request's adapters {@code lora}: the cached encoding of
     * its uuid or bytes under that identity, else a fresh encode, which a released projector refuses as the
     * sleeping engine does.
     */
    private void append(MediaPrompt.Builder prompt, Item item, int imageIndex, TextEncoder text, LoraSlotKey lora) {
        if (lora == null) {
            append(prompt, keyOf(item), item, imageIndex, text);
            return;
        }
        try {
            if (item.modality() == Modality.IMAGE) {
                context.appendImage(prompt, item.uuid(), item.data(), imageIndex, text, lora);
            } else if (item.modality() == Modality.AUDIO) {
                context.appendAudio(prompt, item.uuid(), item.data(), item.field(), text, lora);
            } else {
                context.appendVideo(prompt, item.uuid(), item.data(), item.field(), text, lora);
            }
        } catch (MediaContext.ReleasedException e) {
            throw new EngineUnavailableException(e.getMessage());
        }
    }

    /** Appends one item's marker text and encoding through {@code key}. */
    private void append(MediaPrompt.Builder prompt, MediaKey key, Item item, int imageIndex, TextEncoder text) {
        try {
            if (item.modality() == Modality.IMAGE) {
                context.appendImage(prompt, key, item.data(), imageIndex, text);
            } else if (item.modality() == Modality.AUDIO) {
                context.appendAudio(prompt, key, item.data(), item.field(), text);
            } else {
                context.appendVideo(prompt, key, item.data(), item.field(), text);
            }
        } catch (MediaContext.ReleasedException e) {
            throw new EngineUnavailableException(e.getMessage());
        }
    }

    /**
     * The media prompt of a collapsed token sequence: each {@code -1} token stands for the next item of
     * {@code items}, whose marker text and encoded rows are spliced in its place (vLLM's {@code content_parts}
     * expansion). Every other token is text. The answer carries the row span and key of every spliced item, for
     * the generate answer's {@code mm_placeholders}.
     *
     * @throws MediaInputException when a token is neither a non-negative id nor the {@code -1} placeholder, the
     *                             placeholders and items do not pair up, or an item does not encode
     * @throws EngineUnavailableException when an item must be encoded while the projector is released
     */
    Rendered partsWithPlacements(int[] tokenIds, List<Item> items, TextEncoder text) {
        if (isDisabled()) {
            throw new MediaInputException("the request carries content_parts, but this server has no projector;"
                    + " start it with --mmproj");
        }
        MediaPrompt.Builder prompt = MediaPrompt.builder();
        Iterator<Item> next = items.iterator();
        List<Placement> placements = new ArrayList<>(items.size());
        int cursor = 0;
        int images = 0;
        int at = 0;
        for (int i = 0; i <= tokenIds.length; i++) {
            if (i < tokenIds.length && tokenIds[i] >= 0) {
                continue;
            }
            if (i > cursor) {
                int[] segment = java.util.Arrays.copyOfRange(tokenIds, cursor, i);
                for (int token : segment) {
                    if (token < 0) {
                        throw new MediaInputException("token_ids holds " + token + "; content_parts expects -1 at"
                                + " the media placeholders and non-negative ids elsewhere");
                    }
                }
                prompt.appendText(segment);
            }
            if (i == tokenIds.length) {
                break;
            }
            if (!next.hasNext()) {
                throw new MediaInputException("token_ids holds more media placeholders than content_parts has"
                        + " items");
            }
            Item item = next.next();
            int from = prompt.tokens();
            append(prompt, item, images, text);
            placements.add(new Placement(keyOf(item), item, null, from, prompt.tokens() - from));
            if (item.modality() == Modality.IMAGE) {
                images++;
            }
            cursor = i + 1;
            at++;
        }
        if (next.hasNext()) {
            throw new MediaInputException("content_parts holds " + items.size() + " items for " + at
                    + " media placeholder(s) in token_ids");
        }
        if (prompt.isEmpty()) {
            throw new MediaInputException("the prompt holds no tokens and no media");
        }
        return new Rendered(prompt.build(), placements);
    }

    /**
     * One content part ({@code {"type": "image_url", "url": ...}}) as a loaded item: the chat part shapes
     * {@code image_url}, {@code audio_url} and {@code video_url} with a {@code url} and optional {@code uuid}, or
     * the inline {@code input_audio} and {@code input_video} shapes.
     *
     * @throws MediaInputException when the part is malformed, its modality is not accepted or its URL does not load
     */
    Item contentPart(Map<?, ?> part, String field) {
        Object type = part.get("type");
        Modality modality = modalityOf(type);
        if (modality == null) {
            throw new MediaInputException(field + ".type must be image_url, audio_url or video_url, got '"
                    + type + "'");
        }
        Map<String, Object> chat = new java.util.LinkedHashMap<>();
        chat.put("type", type);
        Object uuid = part.get("uuid");
        if (uuid != null) {
            chat.put("uuid", uuid);
        }
        if (ChatMessages.PART_IMAGE.equals(type) || ChatMessages.PART_AUDIO_URL.equals(type)
                || ChatMessages.PART_VIDEO_URL.equals(type)) {
            Object url = part.get("url");
            if (url == null && uuid == null) {
                throw new MediaInputException(field + " requires a 'url'");
            }
            Map<String, Object> value = new java.util.LinkedHashMap<>();
            if (url != null) {
                value.put("url", url);
            }
            chat.put(String.valueOf(type), value);
        } else {
            String inner = ChatMessages.PART_AUDIO.equals(type) ? "input_audio" : "input_video";
            Object value = part.get(inner);
            if (value instanceof Map<?, ?> map) {
                chat.put(inner, map);
            } else if (value == null && uuid != null) {
                chat.put(inner, new java.util.LinkedHashMap<>());
            } else {
                throw new MediaInputException(field + " requires an '" + inner + "' object with 'data' or 'url'");
            }
        }
        return load(chat, field, null);
    }

    /**
     * The item's media-cache key: its uuid's, or the digest of its content under the loader options. An item whose
     * bytes are not held (a uuid the cache serves) is keyed by the uuid alone.
     */
    private MediaKey keyOf(Item item) {
        return keyOf(item, null);
    }

    /** The item's media-cache key under the request's adapters {@code lora}: its identity folded in as a factor. */
    private MediaKey keyOf(Item item, LoraSlotKey lora) {
        return context.keyOf(item.modality(), item.uuid(), item.uuid() == null ? item.bytes() : null, lora);
    }

    /** Whether the context has no projector, so no item can be encoded. */
    private boolean isDisabled() {
        return context == null;
    }

    /** The tokens of one text segment between markers; an empty segment adds none (mtmd's {@code split_text}). */
    private static void appendSegment(MediaPrompt.Builder prompt, String segment, TextEncoder text) {
        if (!segment.isEmpty()) {
            prompt.appendText(text.encode(segment));
        }
    }

    @Override
    public void close() throws IOException {
        if (context != null) {
            context.close();
        }
    }
}

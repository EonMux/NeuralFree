package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.MediaContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The media turns of a locally loaded multimodal model, the {@code mtmd-cli} equivalent (llama.cpp
 * tools/mtmd/mtmd-cli.cpp): the projector opens as a {@link MediaContext} over the model's embedding width, RoPE
 * layout and vocabulary, and every turn renders through the model's chat template — each media file a content
 * part the template leaves a {@link com.eonmux.neuralfree.gguf.chat.ChatMessages#MEDIA_MARKER} for — then the
 * segments between the markers tokenize and the files' encodings splice in as one {@link MediaPrompt}, which the
 * {@link BatchEngine} generates (its paged memory the demo enables before the model opens, its encoder cache
 * sharing the encoding of a repeated file). As a {@link ChatBackend} it serves the console chat of a
 * projector-configured run, the {@code /image}, {@code /audio} and {@code /video} commands included; the
 * {@code -p} turn of {@link LLMDemo} calls {@link #turn} directly. Immutable.
 */
final class LocalMedia implements ChatBackend, AutoCloseable {

    /** The media content part types a staged message may carry ({@link MediaParts}). */
    private static final Pattern MEDIA_TYPES =
            Pattern.compile("image_url|input_audio|audio_url|video_url|input_video");

    private static final String DATA_URI_MARK = ";base64,";
    /** The marker the rendered prompt holds where a media part sat. */
    private static final Pattern MARKER =
            Pattern.compile(Pattern.quote(com.eonmux.neuralfree.gguf.chat.ChatMessages.MEDIA_MARKER));

    private final EngineModel model;
    private final BatchEngine engine;
    private final MediaContext context;
    private final LLMDemoOptions opts;
    private final GGUFTokenizer tokenizer;
    /** The request of every turn: the demo's with the model's sampling defaults applied. */
    private final GenerationRequest base;
    /** The projector file, named in the banner and the refusals. */
    private final Path projector;
    private final PrintStream out;
    /** The BOS token a media prompt starts with when the vocabulary adds one, else -1. */
    private final int bos;

    private LocalMedia(EngineModel model, BatchEngine engine, MediaContext context, LLMDemoOptions opts,
                       GenerationRequest base, Path projector, PrintStream out) {
        this.model = model;
        this.engine = engine;
        this.context = context;
        this.opts = opts;
        this.base = base;
        this.projector = projector;
        this.out = out;
        this.tokenizer = model.tokenizer();
        this.bos = tokenizer.vocab().addBos() ? tokenizer.bosId() : -1;
    }

    /**
     * The media path of a locally loaded model: the projector at {@code projector} opens for the model's embedding
     * width, RoPE layout and vocabulary, and the batch engine runs its turns over the model's (paged) memory.
     *
     * @param stats unused here; the media backend prints its turn statistics itself
     * @throws IOException              when the projector cannot be read
     * @throws IllegalArgumentException when the projector does not fit the model
     * @throws UnsupportedOperationException when the model or its memory cannot run batched generation
     */
    static LocalMedia open(EngineModel model, EngineConfig cfg, LLMDemoOptions opts, Path projector) throws IOException {
        boolean mrope = model.spec().traits().ropeType().usesSections();
        GGUFTokenizer tokenizer = model.tokenizer();
        MediaContext context = MediaContext.open(projector.toString(), model.embeddingLength(), mrope,
                opts.imageBudget(), token -> tokenizer.getTokenId(token) >= 0, opts.encoderOptions(),
                MediaContext.DEFAULT_CACHE_BYTES);
        try {
            BatchEngine engine = BatchEngine.forModel(model, cfg);
            GenerationRequest base = opts.generation.withModelDefaults(SamplingDefaults.forModel(model, cfg));
            return new LocalMedia(model, engine, context, opts, base, projector, System.out);
        } catch (RuntimeException e) {
            context.close();
            throw e;
        }
    }

    /** Runs the console chat of {@code media} on the standard streams and the process interrupt state. */
    static void run(LocalMedia media, LLMDemoOptions opts, Consumer<String> stats) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            new ChatSession(media, opts, in, System.out, transcript, Interrupts.system()).run();
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
    }

    @Override
    public void printBanner(PrintStream banner) {
        SamplerConfig sampler = base.sampler();
        banner.println("model      : " + model.architecture());
        banner.println("projector  : " + projector);
        banner.println("modalities : " + String.join(", ", modalities()));
        banner.println(String.format(Locale.ROOT,
                "settings   : temperature=%.2f, top_p=%.2f, top_k=%d, min_p=%.3f, typical_p=%.2f, max_tokens=%d,"
                        + " repeat_penalty=%.2f, presence_penalty=%.2f, frequency_penalty=%.2f, repeat_last_n=%d,"
                        + " seed=%d%s%s",
                sampler.temperature(), sampler.topP(), sampler.topK(), sampler.minP(), sampler.typicalP(),
                base.maxNewTokens(), sampler.repeatPenalty(), sampler.presencePenalty(),
                sampler.frequencyPenalty(), sampler.penaltyLastN(), sampler.seed(),
                base.ignoreEos() ? ", ignore_eos" : "",
                base.stopStrings().isEmpty() ? "" : ", stop=" + base.stopStrings()));
    }

    /** The modalities the projector serves, image first. */
    private List<String> modalities() {
        List<String> out = new ArrayList<>(3);
        if (context.supportsVision()) {
            out.add("image");
        }
        if (context.supportsAudio()) {
            out.add("audio");
        }
        if (context.supportsVideo()) {
            out.add("video");
        }
        return out;
    }

    @Override
    public String mediaRefusal(String kind) {
        switch (kind) {
            case "audio":
                return context.supportsAudio() ? null
                        : "the projector of " + projector + " has no audio encoder";
            case "video":
            case "image":
                return context.supportsVision() ? null
                        : "the projector of " + projector + " has no vision encoder";
            default:
                return "'" + kind + "' is not a media kind; the projector takes image, audio and video";
        }
    }

    @Override
    public ChatAnswer answer(List<Map<String, Object>> messages, BooleanSupplier cancelled) throws IOException {
        String rendered;
        try {
            rendered = tokenizer.formatChat(LocalChat.chatRequest(messages, opts.thinking, opts.templateKwargs));
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            return ChatAnswer.failed("", "", "the chat template cannot render the conversation: " + e.getMessage());
        }
        List<MediaItem> items = itemsOf(messages);
        ChannelPrinter printer = new ChannelPrinter(out, opts.showAllChannels, opts.debugPrompt);
        GenerationRequest request = LocalChat.templated(base, opts.thinking, opts.templateKwargs).toBuilder()
                .cancellation(cancelled).build();
        GenerationResult result;
        try {
            result = generate(prompt(rendered, items), request, printer, rendered);
        } catch (MediaInputException | ContextFullException e) {
            return ChatAnswer.failed("", "", e.getMessage());
        }
        out.println();
        if (result.stopReason() == StopReason.CANCELLED) {
            out.println("[interrupted]");
        }
        return ChatAnswer.of(printer.content(), printer.reasoning());
    }

    @Override
    public void reset() {
        engine.clearPrefixCache();
    }

    @Override
    public void close() throws IOException {
        engine.close();
        context.close();
    }

    /**
     * One media turn of the {@code -p} run: {@code renderedPrompt} splices the {@code --image}/{@code --audio}/
     * {@code --video} files at their markers, and the batch engine generates over it.
     *
     * @throws IOException naming the file when one cannot be read, or the batch run fails
     */
    GenerationResult turn(String renderedPrompt, GenerationRequest request, ReasoningFilter.Sink sink)
            throws IOException {
        List<MediaItem> items = new ArrayList<>();
        for (LLMDemoOptions.MediaFile file : opts.media) {
            items.add(itemOf(file.kind, Paths.get(file.path)));
        }
        return generate(prompt(renderedPrompt, items), request, sink, renderedPrompt);
    }

    // ------------------------------------------------------------------ prompts

    /**
     * The media prompt of {@code rendered}: the text between the markers tokenized (no added special tokens, the
     * marker-spliced special tokens parsed), each marker replaced by the next item's marker text and encoding,
     * the prompt starting with the BOS when the vocabulary adds one.
     *
     * @throws MediaInputException when the markers and the items do not pair up, a file cannot be decoded, or the
     *                             prompt holds no tokens and no media
     */
    private MediaPrompt prompt(String rendered, List<MediaItem> items) {
        MediaPrompt.Builder prompt = MediaPrompt.builder();
        if (bos >= 0) {
            prompt.appendText(new int[] {bos});
        }
        MediaContext.TextEncoder text = segment -> tokenizer.encode(segment, false, true);
        Iterator<MediaItem> next = items.iterator();
        int images = 0;
        int markers = 0;
        Matcher m = MARKER.matcher(rendered);
        int from = 0;
        while (m.find()) {
            appendSegment(prompt, rendered.substring(from, m.start()), text);
            from = m.end();
            markers++;
            if (!next.hasNext()) {
                continue;
            }
            MediaItem item = next.next();
            append(prompt, item, images, text);
            if (item.kind.equals("image")) {
                images++;
            }
        }
        appendSegment(prompt, rendered.substring(from), text);
        if (markers != items.size()) {
            throw new MediaInputException("the prompt holds " + markers + " media marker(s) for " + items.size()
                    + " media item(s); every file needs its own slot in the rendered prompt");
        }
        if (prompt.isEmpty()) {
            throw new MediaInputException("the prompt holds no tokens and no media");
        }
        return prompt.build();
    }

    /** Appends one item's marker text and encoding, the projector refusing a modality it does not serve. */
    private void append(MediaPrompt.Builder prompt, MediaItem item, int imageIndex, MediaContext.TextEncoder text) {
        byte[] bytes = item.data.get();
        if (item.kind.equals("image")) {
            context.appendImage(prompt, bytes, imageIndex, text);
        } else if (item.kind.equals("audio")) {
            context.appendAudio(prompt, bytes, item.field, text);
        } else {
            context.appendVideo(prompt, bytes, item.field, text);
        }
    }

    /** The tokens of one text segment between markers; an empty segment adds none (mtmd's {@code split_text}). */
    private static void appendSegment(MediaPrompt.Builder prompt, String segment, MediaContext.TextEncoder text) {
        if (!segment.isEmpty()) {
            prompt.appendText(text.encode(segment));
        }
    }

    /** Submits one media prompt, drains the engine and answers with the sequence's result. */
    private GenerationResult generate(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                      String rendered) throws IOException {
        BatchSequence handle = engine.submit(prompt, request, sink, rendered, null);
        engine.drain();
        if (handle.state() == BatchSequence.State.FAILED) {
            throw new IOException("the media turn failed: " + handle.error());
        }
        return handle.result();
    }

    // ------------------------------------------------------------------ media items

    /**
     * The media items of the conversation's content parts, in message and part order — the order the rendered
     * prompt's markers take. A part's bytes ride the data URI or the inline data {@link MediaParts} staged.
     *
     * @throws MediaInputException when a media part carries no data, or its data is not base64
     */
    private static List<MediaItem> itemsOf(List<Map<String, Object>> messages) {
        List<MediaItem> items = new ArrayList<>();
        for (int m = 0; m < messages.size(); m++) {
            Object content = messages.get(m).get("content");
            if (!(content instanceof List)) {
                continue;
            }
            List<?> parts = (List<?>) content;
            for (int p = 0; p < parts.size(); p++) {
                if (!(parts.get(p) instanceof Map)) {
                    continue;
                }
                MediaItem item = partItem((Map<?, ?>) parts.get(p), "messages[" + m + "].content[" + p + "]");
                if (item != null) {
                    items.add(item);
                }
            }
        }
        return items;
    }

    /** The media item of one content part, or null for a text part. */
    private static MediaItem partItem(Map<?, ?> part, String field) {
        Object type = part.get("type");
        if (!(type instanceof String) || !MEDIA_TYPES.matcher((String) type).matches()) {
            return null;
        }
        String kind = kindOf((String) type);
        String data = dataOf(part, (String) type);
        if (data == null || data.isEmpty()) {
            throw new MediaInputException(field + ": the " + kind + " part carries no base64 data for the local"
                    + " media turn");
        }
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(data);
        } catch (IllegalArgumentException e) {
            throw new MediaInputException(field + ": the " + kind + " part data is not base64: " + e.getMessage());
        }
        return new MediaItem(kind, () -> bytes, field + "." + type);
    }

    /** The modality of a media part type ({@link MediaParts} builds {@code image_url}, {@code input_audio},
     *  {@code input_video}; the {@code *_url} spellings take the same kinds). */
    private static String kindOf(String type) {
        if (type.equals("input_audio") || type.equals("audio_url")) {
            return "audio";
        }
        if (type.equals("input_video") || type.equals("video_url")) {
            return "video";
        }
        return "image";
    }

    /** The base64 of a part: the inline {@code data} field, or the payload of a {@code data:} URL. */
    private static String dataOf(Map<?, ?> part, String type) {
        Object value = part.get(type);
        if (value instanceof Map) {
            Object data = ((Map<?, ?>) value).get("data");
            if (data instanceof String) {
                return (String) data;
            }
            Object url = ((Map<?, ?>) value).get("url");
            return url instanceof String ? payload((String) url) : null;
        }
        return value instanceof String ? payload((String) value) : null;
    }

    /** The base64 payload of a {@code data:} URI, or null when the URL carries none. */
    private static String payload(String url) {
        int at = url.indexOf(DATA_URI_MARK);
        return at >= 0 ? url.substring(at + DATA_URI_MARK.length()) : null;
    }

    /** One file of {@code --image}/{@code --audio}/{@code --video}, read for its turn. */
    private static MediaItem itemOf(String kind, Path path) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("--" + kind + ": no such media file '" + path + "'");
        }
        byte[] bytes = Files.readAllBytes(path);
        return new MediaItem(kind, () -> bytes, "--" + kind + " " + path);
    }

    /** One media input of a turn: its modality kind, its bytes and where it came from, for the refusals. */
    private static final class MediaItem {

        final String kind;
        final Supplier<byte[]> data;
        final String field;

        MediaItem(String kind, Supplier<byte[]> data, String field) {
            this.kind = kind;
            this.data = data;
            this.field = field;
        }
    }
}

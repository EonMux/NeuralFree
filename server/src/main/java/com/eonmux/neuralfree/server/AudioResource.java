package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.jboss.resteasy.reactive.server.multipart.FormValue;
import org.jboss.resteasy.reactive.server.multipart.MultipartFormDataInput;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.Base64;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/**
 * OpenAI's speech-to-text routes, {@code POST /v1/audio/transcriptions} and {@code POST /v1/audio/translations}
 * (with llama.cpp's unversioned {@code /audio/transcriptions} alias), over the existing audio-enabled chat
 * inference path: the uploaded clip becomes one {@code audio_url} content part beside an instruction text part,
 * and the model's answer text is the transcript (llama.cpp's {@code convert_transcriptions_to_chatcmpl}).
 *
 * <p>A multipart request carries the audio in {@code file} and the OpenAI form fields. The batch runner's
 * {@code file_url} form ({@link #batch}) carries the same options as JSON. Supported options are {@code model},
 * {@code prompt}, {@code language}, {@code to_language} (translations only), {@code response_format}
 * ({@code json}, the default, or {@code text}), {@code stream}, the sampling fields ({@code temperature},
 * {@code max_tokens}, {@code max_completion_tokens}, {@code seed}, {@code top_p}, {@code top_k}, {@code min_p},
 * {@code repetition_penalty}, {@code frequency_penalty}, {@code presence_penalty}); every other field and response
 * format is refused by name. Streaming answers vLLM's {@code transcription.chunk} / {@code translation.chunk}
 * server-sent events, decoded from the chat stream's deltas.
 *
 * <p>Translations follow vLLM's {@code TranslationRequest}: {@code language} names the spoken language,
 * {@code to_language} the target (English when absent). Transcriptions follow vLLM's
 * {@code TranscriptionRequest}: {@code language} is a hint, and {@code to_language} is refused because translating
 * is the other route. Without a projector that accepts audio the chat path refuses the part by name.</p>
 */
@Path("/")
public class AudioResource {

    /**
     * The generation cap when a request names none: SGLang's speech-to-text floor, large enough for a transcript
     * of a short clip and far below the contexts this server serves.
     */
    static final int DEFAULT_MAX_TOKENS = 448;

    /** The upload cap and streaming buffer cap: vLLM's {@code VLLM_MAX_AUDIO_CLIP_FILESIZE_MB} of 25 MB. */
    static final long MAX_AUDIO_BYTES = MediaConnector.Options.DEFAULT_MAX_BYTES;

    /** The request fields both routes take (vLLM's speech-to-text request models). */
    private static final Set<String> OPTIONS = Set.of("model", "prompt", "language", "to_language",
            "response_format", "stream", "temperature", "max_tokens", "max_completion_tokens", "seed", "top_p",
            "top_k", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty");

    /** The sampling fields the chat body carries unchanged. */
    private static final List<String> SAMPLING = List.of("temperature", "max_tokens", "max_completion_tokens",
            "seed", "top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty", "presence_penalty");
    private static final String ENGLISH = "English";

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The current HTTP request, so the delegated chat stream cancels on a disconnect; null in manual wiring. */
    @Inject
    HttpServerRequest http;

    /** The resumable stream sessions the delegated chat stream records into when it names a conversation id. */
    @Inject
    StreamSessions streams;

    public AudioResource() {
    }

    /** Test and manual wiring: builds the resource over an explicit service. */
    AudioResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    AudioResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    @POST
    @Path("/v1/audio/transcriptions")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces({MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN, "text/event-stream"})
    public Response transcriptions(MultipartFormDataInput form) {
        return multipart(form, false);
    }

    /** llama.cpp's unversioned alias of the transcription route. */
    @POST
    @Path("/audio/transcriptions")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces({MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN, "text/event-stream"})
    public Response transcriptionsAlias(MultipartFormDataInput form) {
        return transcriptions(form);
    }

    @POST
    @Path("/v1/audio/translations")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces({MediaType.APPLICATION_JSON, MediaType.TEXT_PLAIN, "text/event-stream"})
    public Response translations(MultipartFormDataInput form) {
        return multipart(form, true);
    }

    // ------------------------------------------------------------------ request forms

    private Response multipart(MultipartFormDataInput form, boolean translation) {
        byte[] file = null;
        Map<String, Object> options = new LinkedHashMap<>();
        for (Map.Entry<String, Collection<FormValue>> entry : form.getValues().entrySet()) {
            String name = entry.getKey();
            if (entry.getValue().size() != 1) {
                throw ServerJson.badRequest('"' + name + "\" must occur exactly once");
            }
            FormValue value = entry.getValue().iterator().next();
            if (name.equals("file")) {
                if (!value.isFileItem()) {
                    throw ServerJson.badRequest("\"file\" must be an uploaded file");
                }
                file = readFile(value);
            } else {
                if (!OPTIONS.contains(name)) {
                    throw ServerJson.badRequest("unsupported field \"" + name + "\"");
                }
                if (value.isFileItem()) {
                    throw ServerJson.badRequest('"' + name + "\" must be a form field");
                }
                options.put(name, formValue(name, value.getValue()));
            }
        }
        return upload(file, options, translation);
    }

    private static byte[] readFile(FormValue value) {
        try (InputStream in = value.getFileItem().getInputStream()) {
            byte[] bytes = in.readNBytes((int) MAX_AUDIO_BYTES + 1);
            if (bytes.length > MAX_AUDIO_BYTES) {
                throw ServerJson.badRequest("\"file\" exceeds " + (MAX_AUDIO_BYTES / (1024 * 1024)) + " MiB");
            }
            return bytes;
        } catch (IOException e) {
            throw ServerJson.badRequest("cannot read \"file\": " + e.getMessage());
        }
    }

    private static Object formValue(String name, String value) {
        try {
            return switch (name) {
                case "max_tokens", "max_completion_tokens", "top_k", "seed" -> Long.parseLong(value);
                case "temperature", "top_p", "min_p", "repetition_penalty", "frequency_penalty",
                        "presence_penalty" -> {
                    double number = Double.parseDouble(value);
                    if (!Double.isFinite(number)) {
                        throw new IllegalArgumentException();
                    }
                    yield number;
                }
                case "stream" -> {
                    if (!value.equals("true") && !value.equals("false")) {
                        throw new IllegalArgumentException();
                    }
                    yield Boolean.valueOf(value);
                }
                default -> value;
            };
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest("invalid value for \"" + name + "\"");
        }
    }

    /** Runs an uploaded clip through the same mapping as a batch file URL. */
    Response upload(byte[] file, Map<String, Object> options, boolean translation) {
        if (file == null || file.length == 0) {
            throw ServerJson.badRequest("\"file\" must be a non-empty audio file");
        }
        if (file.length > MAX_AUDIO_BYTES) {
            throw ServerJson.badRequest("\"file\" exceeds " + (MAX_AUDIO_BYTES / (1024 * 1024)) + " MiB");
        }
        return recognize(dataUrl(file), options, translation, "file");
    }

    /** Runs a vLLM batch record using its {@code file_url} and the normal media connector. */
    Response batch(JsonView body, boolean translation) {
        if (body.has("file")) {
            throw ServerJson.badRequest("\"file\" is not supported in batch; use \"file_url\"");
        }
        Map<String, Object> options = new LinkedHashMap<>();
        for (String key : body.keys()) {
            if (!key.equals("file_url")) {
                options.put(key, body.raw(key));
            }
        }
        if (!"json".equals(String.valueOf(options.getOrDefault("response_format", "json")))) {
            throw ServerJson.badRequest("\"response_format\" must be \"json\" in a batch record");
        }
        if (Boolean.TRUE.equals(options.get("stream")) || "true".equals(options.get("stream"))) {
            throw ServerJson.badRequest("\"stream\" must be false or omitted: a batch request must not stream");
        }
        try {
            String url = body.str("file_url");
            if (url.isBlank()) {
                throw ServerJson.badRequest("\"file_url\" must not be empty");
            }
            return recognize(url, options, translation, "file_url");
        } catch (JsonException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The data URL of an uploaded clip; the media connector sniffs the container from its bytes. */
    private static String dataUrl(byte[] file) {
        return "data:application/octet-stream;base64," + Base64.getEncoder().encodeToString(file);
    }

    /** A PCM16 little-endian 16 kHz mono stream as a WAV data URL (the realtime endpoint's audio format). */
    static String pcmDataUrl(byte[] pcm) {
        float[] samples = new float[pcm.length / 2];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (short) ((pcm[2 * i] & 0xFF) | (pcm[2 * i + 1] << 8)) / 32768.0f;
        }
        return dataUrl(WavWriter.bytes(samples, 16000));
    }

    // ------------------------------------------------------------------ the chat mapping

    /** Runs the prepared chat request and shapes the transcript as the request's format asks. */
    private Response recognize(String reference, Map<String, Object> options, boolean translation, String fileField) {
        Prepared prepared = prepare(reference, options, translation, fileField);
        if (prepared.stream()) {
            return streamResponse(prepared);
        }
        String text = runText(prepared);
        if (prepared.textFormat()) {
            return Response.ok(text).type("text/plain; charset=utf-8").build();
        }
        return Response.ok(ServerJson.write(Map.of("text", text))).type(MediaType.APPLICATION_JSON).build();
    }

    /** The transcript of a streamed chat request; {@code sink} receives every non-empty delta. */
    String transcribePcm(byte[] pcm, Map<String, Object> options, boolean translation, SseChatText.Delta sink)
            throws IOException {
        Prepared prepared = prepare(pcmDataUrl(pcm), options, translation, "audio");
        return streamText(prepared, sink);
    }

    private Prepared prepare(String reference, Map<String, Object> options, boolean translation, String fileField) {
        for (String key : options.keySet()) {
            if (!OPTIONS.contains(key)) {
                throw ServerJson.badRequest("unsupported field \"" + key + "\"");
            }
        }
        try {
            JsonView request = JsonView.of(options, "body");
            String format = request.optStr("response_format", "json");
            if (!format.equals("json") && !format.equals("text")) {
                throw ServerJson.badRequest("\"response_format\" must be \"json\" or \"text\", got \""
                        + format + "\"");
            }
            boolean stream = request.optBool("stream", false);
            if (options.get("model") != null && !(options.get("model") instanceof String)) {
                throw ServerJson.badRequest("\"model\" must be a string");
            }
            String language = request.optStr("language", "");
            String toLanguage = request.optStr("to_language", "");
            if (!translation && !toLanguage.isEmpty()) {
                throw ServerJson.badRequest("\"to_language\" is only supported by /v1/audio/translations");
            }
            if (translation && toLanguage.isEmpty()) {
                toLanguage = "en";
            }
            String prompt = instruction(request.optStr("prompt", ""), language, toLanguage, translation);
            Map<String, Object> chat = chat(reference, options, prompt);
            return new Prepared(chat, modelName(options), stream, format.equals("text"), translation, fileField);
        } catch (JsonException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
    }

    /** The user instruction the audio part accompanies, in llama.cpp's transcription-prompt shape. */
    private static String instruction(String prompt, String language, String toLanguage, boolean translation) {
        if (translation) {
            String target = toLanguage.equals("en") || toLanguage.equals("eng") ? ENGLISH : toLanguage;
            String out = "Translate the audio to " + target;
            if (!language.isEmpty()) {
                out += " (source language: " + language + ")";
            }
            return prompt.isEmpty() ? out : out + ". Context: " + prompt;
        }
        String out = prompt.isEmpty() ? "Transcribe audio to text" : prompt;
        if (!language.isEmpty()) {
            out += " (language: " + language + ")";
        }
        return out;
    }

    /** The chat body: the supported sampling fields, the instruction and the clip. */
    private static Map<String, Object> chat(String reference, Map<String, Object> options, String prompt) {
        Map<String, Object> chat = new LinkedHashMap<>();
        if (options.get("model") instanceof String name && !name.isBlank()) {
            chat.put("model", name);
        }
        for (String field : SAMPLING) {
            if (options.containsKey(field)) {
                chat.put(field, jsonNumber(options.get(field)));
            }
        }
        if (!chat.containsKey("temperature")) {
            chat.put("temperature", 0.0);
        }
        if (!chat.containsKey("max_completion_tokens") && !chat.containsKey("max_tokens")) {
            chat.put("max_tokens", (long) DEFAULT_MAX_TOKENS);
        }
        chat.put("messages", List.of(Map.of("role", "user", "content", List.of(
                Map.of("type", "text", "text", prompt),
                Map.of("type", "audio_url", "audio_url", Map.of("url", reference))))));
        return chat;
    }

    /**
     * The JSON form of a sampling value: the chat mapping reads a parsed document, whose integers are
     * {@code long}s and whose decimals are {@code double}s, so the narrower Java widths a caller builds the
     * options map with are widened here.
     */
    private static Object jsonNumber(Object value) {
        if (value instanceof Integer || value instanceof Short || value instanceof Byte) {
            return ((Number) value).longValue();
        }
        if (value instanceof Float) {
            return ((Number) value).doubleValue();
        }
        return value;
    }

    /** The model name a streamed chunk carries: the request's, else the served name. */
    private String modelName(Map<String, Object> options) {
        Object model = options.get("model");
        if (model instanceof String name && !name.isBlank()) {
            return name;
        }
        return service.served().primary();
    }

    /**
     * Rewrites the chat path's media field in a refusal to the form field the client sent, so an unusable upload
     * or URL is named as {@code file} or {@code file_url}.
     */
    private static WebApplicationException mediaRefusal(WebApplicationException e, String fileField) {
        String message = String.valueOf(OpenAiResource.errorEnvelope(e).get("message"));
        String mediaField = "messages[0].content[1].audio_url";
        if (e.getResponse().getStatus() == 400 && message.contains(mediaField)) {
            return ServerJson.badRequest(message.replace(mediaField + ".url", fileField)
                    .replace(mediaField, fileField));
        }
        return e;
    }

    // ------------------------------------------------------------------ the transcript

    private String runText(Prepared prepared) {
        Response response = chatCompletion(prepared.chat(), prepared.fileField());
        JsonView body = ServerJson.parse(String.valueOf(response.getEntity()));
        return content(body);
    }

    /** The first choice's answer text, with reasoning content dropped. */
    private static String content(JsonView body) {
        List<Object> choices = body.list("choices");
        if (choices.isEmpty()) {
            throw ServerJson.error(500, "the model answered no choice", "server_error");
        }
        JsonView choice = JsonView.of(choices.get(0), "choices[0]");
        return choice.obj("message").optStr("content", "");
    }

    private Response streamResponse(Prepared prepared) {
        Map<String, Object> chat = chatBody(prepared.chat(), true);
        Response response = chatCompletion(chat, prepared.fileField());
        Object entity = response.getEntity();
        if (!(entity instanceof StreamingOutput inner)) {
            throw ServerJson.error(500, "the chat stream did not open", "server_error");
        }
        String id = (prepared.translation() ? "trsl-" : "trsc-") + UUID.randomUUID();
        long created = System.currentTimeMillis() / 1000;
        String object = prepared.translation() ? "translation.chunk" : "transcription.chunk";
        StreamingOutput out = body -> {
            SseChatText parser = new SseChatText(delta -> {
                try {
                    ServerJson.event(body, null, chunk(id, object, created, prepared.model(), delta, null));
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
            try {
                inner.write(parser);
                parser.finish();
            } catch (UncheckedIOException e) {
                throw e.getCause();
            }
            String finish = parser.finishReason() == null ? "stop" : parser.finishReason();
            ServerJson.event(body, null, chunk(id, object, created, prepared.model(), "", finish));
            ServerJson.data(body, "[DONE]");
        };
        return Response.ok(out).type("text/event-stream").build();
    }

    private String streamText(Prepared prepared, SseChatText.Delta sink) throws IOException {
        Response response = chatCompletion(chatBody(prepared.chat(), true), prepared.fileField());
        Object entity = response.getEntity();
        if (!(entity instanceof StreamingOutput inner)) {
            throw new IOException("the chat stream did not open");
        }
        SseChatText parser = new SseChatText(sink);
        try {
            inner.write(parser);
            parser.finish();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
        return parser.text();
    }

    private static Map<String, Object> chunk(String id, String object, long created, String model, String delta,
                                             String finish) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("delta", Map.of("content", delta));
        choice.put("finish_reason", finish);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", object);
        chunk.put("created", created);
        chunk.put("model", model);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    /** The streamed chat body: the request's stream flag is forced on. */
    private static Map<String, Object> chatBody(Map<String, Object> chat, boolean stream) {
        Map<String, Object> out = new LinkedHashMap<>(chat);
        out.put("stream", stream);
        return out;
    }

    private Response chatCompletion(Map<String, Object> chat, String fileField) {
        try {
            return new OpenAiResource(service, router, http, streams)
                    .chatCompletion(JsonView.of(chat, "body"));
        } catch (WebApplicationException e) {
            throw mediaRefusal(e, fileField);
        }
    }

    /** A prepared speech request: its chat body, streamed model name and answer shaping. */
    private record Prepared(Map<String, Object> chat, String model, boolean stream, boolean textFormat,
                            boolean translation, String fileField) {
    }
}

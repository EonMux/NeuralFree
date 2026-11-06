package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioDecoder;
import com.eonmux.neuralfree.gguf.audio.UnsupportedAudioFormatException;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.media.MediaConnector;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.http.HttpServerRequest;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.jboss.resteasy.reactive.server.multipart.FormValue;
import org.jboss.resteasy.reactive.server.multipart.MultipartFormDataInput;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The audio generation endpoints of llama.cpp's {@code post_tts} (server.cpp:188, server-context.cpp:6435):
 * {@code POST /v1/audio/speech} and its llama.cpp-native alias {@code POST /tts}. The body takes the text input
 * ({@code "input"}, the {@code "prompt"} alias accepted), the speech fields llama.cpp accepts ({@code "lang"},
 * {@code "voice"}, {@code "seed"}, {@code "top_k"}, {@code "top_p"}, {@code "max_tokens"} / {@code "n_predict"}),
 * the {@code response_format} ({@code "wav"}, the default, or {@code "pcm"}) and, for a wavtokenizer-dec model, the
 * audio codes to synthesize.
 *
 * <p>{@code "stream": true} delivers the waveform as the decoders produce it: llama.cpp's streamed TTS writes the
 * WAV header with the unknown-length sentinel once and appends 16-bit PCM runs as each decoder window is decoded
 * (deltas of the code2wav and mimi windows, one hop per frame for the spectral decoder); {@code "pcm"} streams raw
 * little-endian samples with no header. OpenAI's {@code "stream_format"} selects the envelope: {@code "audio"} (the
 * default) the raw stream, {@code "sse"} the {@code speech.audio.delta} events carrying base64 audio and the final
 * {@code speech.audio.done}. A client that disconnects cancels the generation.
 *
 * <p>Voice cloning takes a reference waveform as the multipart file field {@code speaker_ref} or, in a JSON body, as
 * the base64 {@code speaker_ref_b64} string (llama.cpp's {@code post_tts} fields). The bytes go through the server's
 * {@link AudioDecoder}, and a format it cannot decode, a reference over {@link #MAX_SPEAKER_REF_BYTES}, and one
 * longer than {@link #MAX_SPEAKER_SECONDS} are 400s naming {@code speaker_ref}. A generator file without a speaker
 * encoder answers 501 naming the field instead of ignoring it.
 *
 * <p>Text reaches the loaded model's {@link EngineService#synthesizeSpeech(String, SpeechOptions) speech
 * generator}, so the route holds no generator-specific code: a qwen3tts backbone with its code predictor and
 * code2wav generator file, a pockettts backbone with its flow-matching and mimi decoder generator file, or an
 * lfm3a backbone with its MaskGIT depthformer and liquid-codec generator file (llama.cpp's {@code --mmproj},
 * {@code neuralfree.server.mmproj}), speaks the text. A model without a speech-generator backbone, or a
 * speech-generator backbone without its generator file, answers 501 naming the missing piece. The codec-code path
 * of a wavtokenizer-dec model ({@code "codes"} as an integer array, or integer text in {@code "input"})
 * synthesizes the waveform from the model's spectral frames through its declared speech geometry
 * ({@code audio.sample_rate}, {@code audio.n_fft}, {@code audio.hop_length}) and the shared streaming iSTFT.
 *
 * <p>In router mode the body's {@code model} selects the entry the speech serves on, as every other POST route
 * routes; a missing name is a 400 and an unknown one a 404 ({@link ModelRouter#target}).</p>
 */
@Path("/")
public class TtsResource {

    /** The largest accepted speaker reference, the media connector's upload limit. */
    static final long MAX_SPEAKER_REF_BYTES = MediaConnector.Options.DEFAULT_MAX_BYTES;
    /** The longest accepted reference waveform, in seconds; llama.cpp's pockettts encoder keeps 30. */
    static final double MAX_SPEAKER_SECONDS = 120.0;

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** The current HTTP request, so a streamed response cancels its generation when the client goes away. */
    @Inject
    HttpServerRequest http;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public TtsResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public TtsResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public TtsResource() {
    }

    /** The OpenAI-compatible speech endpoint. */
    @POST
    @Path("/v1/audio/speech")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces({MediaType.APPLICATION_OCTET_STREAM, "audio/wav", "text/event-stream"})
    public Response speech(String bodyText) {
        return respond(ServerJson.parse(bodyText));
    }

    /** The llama.cpp-native alias. */
    @POST
    @Path("/tts")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces({MediaType.APPLICATION_OCTET_STREAM, "audio/wav", "text/event-stream"})
    public Response tts(String bodyText) {
        return respond(ServerJson.parse(bodyText));
    }

    /** The OpenAI-compatible endpoint with an uploaded voice reference ({@code speaker_ref}). */
    @POST
    @Path("/v1/audio/speech")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces({MediaType.APPLICATION_OCTET_STREAM, "audio/wav", "text/event-stream"})
    public Response speechMultipart(MultipartFormDataInput form) {
        return multipart(form);
    }

    /** The llama.cpp-native alias with an uploaded voice reference. */
    @POST
    @Path("/tts")
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Produces({MediaType.APPLICATION_OCTET_STREAM, "audio/wav", "text/event-stream"})
    public Response ttsMultipart(MultipartFormDataInput form) {
        return multipart(form);
    }

    /**
     * A multipart request: the {@code speaker_ref} file plus the JSON body's fields as form fields. The file takes
     * precedence over a {@code speaker_ref_b64} field, as llama.cpp's {@code post_tts} does.
     */
    private Response multipart(MultipartFormDataInput form) {
        byte[] speakerRef = null;
        Map<String, Object> fields = new LinkedHashMap<>();
        for (Map.Entry<String, Collection<FormValue>> entry : form.getValues().entrySet()) {
            String name = entry.getKey();
            if (entry.getValue().size() != 1) {
                throw ServerJson.badRequest('"' + name + "\" must occur exactly once");
            }
            FormValue value = entry.getValue().iterator().next();
            if (name.equals("speaker_ref")) {
                if (!value.isFileItem()) {
                    throw ServerJson.badRequest("\"speaker_ref\" must be an uploaded audio file");
                }
                speakerRef = readSpeakerRef(value);
            } else {
                if (value.isFileItem()) {
                    throw ServerJson.badRequest('"' + name + "\" must be a form field");
                }
                fields.put(name, formValue(name, value.getValue()));
            }
        }
        return respond(JsonView.of(fields, "multipart form", ""), speakerRef);
    }

    private static byte[] readSpeakerRef(FormValue value) {
        try (InputStream in = value.getFileItem().getInputStream()) {
            byte[] bytes = in.readNBytes((int) MAX_SPEAKER_REF_BYTES + 1);
            if (bytes.length > MAX_SPEAKER_REF_BYTES) {
                throw ServerJson.badRequest("\"speaker_ref\" exceeds "
                        + (MAX_SPEAKER_REF_BYTES / (1024 * 1024)) + " MiB");
            }
            return bytes;
        } catch (IOException e) {
            throw ServerJson.badRequest("cannot read \"speaker_ref\": " + e.getMessage());
        }
    }

    /** A form field with the JSON type the body parser would give it. */
    private static Object formValue(String name, String value) {
        try {
            switch (name) {
                case "stream":
                    if (!value.equals("true") && !value.equals("false")) {
                        throw new IllegalArgumentException();
                    }
                    return Boolean.valueOf(value);
                case "seed":
                case "top_k":
                case "max_tokens":
                case "n_predict":
                    return Long.parseLong(value);
                case "top_p":
                    return Double.parseDouble(value);
                default:
                    return value;
            }
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest("invalid value for \"" + name + "\"");
        }
    }

    private Response respond(JsonView body) {
        return respond(body, null);
    }

    private Response respond(JsonView body, byte[] uploadedSpeakerRef) {
        EngineService target = ModelRouter.target(router, service, body, null);
        ServerJson.requireLoaded(target);
        if (body.has("speaker_ref")) {
            throw ServerJson.badRequest("\"speaker_ref\" must be an uploaded audio file in a multipart request,"
                    + " not a JSON field; send its bytes as \"speaker_ref_b64\" instead");
        }
        boolean stream = body.optBool("stream", false);
        boolean sse = streamFormat(body, stream);
        boolean pcm = responseFormat(body);
        AudioClip reference = speakerReference(body, uploadedSpeakerRef);
        int[] codes = codes(target, body);
        if (codes != null) {
            if (reference != null) {
                throw ServerJson.badRequest("\"speaker_ref\" needs a speech-generator backbone; this model"
                        + " synthesizes from audio codes");
            }
            return codesResponse(target, codes, stream, pcm, sse);
        }
        String text = text(body);
        if (text == null || text.trim().isEmpty()) {
            throw ServerJson.badRequest("\"input\" must be a non-empty string");
        }
        SpeechOptions options;
        try {
            options = options(body).withSpeakerReference(reference);
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        }
        if (!stream) {
            return audio(responseSpeech(target, text, options), pcm);
        }
        int sampleRate;
        try {
            sampleRate = target.speechSampleRate();
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, e.getMessage(), "not_supported_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "speech generation failed: " + e.getMessage(), "server_error");
        }
        return streamResponse(sampleRate, pcm, sse,
                sink -> target.synthesizeSpeech(text, options, sink));
    }

    /** The waveform of the integer-code path, streamed or one-shot; the model must declare a decoder first. */
    private Response codesResponse(EngineService target, int[] codes, boolean stream, boolean pcm, boolean sse) {
        int sampleRate;
        try {
            sampleRate = target.speechCodeSampleRate();
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, "this server does not support audio generation: " + e.getMessage(),
                    "not_supported_error");
        }
        try {
            if (!stream) {
                return audio(target.synthesizeSpeech(codes), pcm);
            }
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, "this server does not support audio generation: " + e.getMessage(),
                    "not_supported_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest("\"codes\": " + e.getMessage());
        }
        return streamResponse(sampleRate, pcm, sse, sink -> target.synthesizeSpeech(codes, sink));
    }

    /** The non-streamed body of one synthesized clip: canonical WAV, or raw 16-bit little-endian PCM. */
    private static Response audio(AudioClip audio, boolean pcm) {
        if (pcm) {
            byte[] chunk = WavWriter.pcmChunk(audio.samples(), 0, audio.length());
            return Response.ok((StreamingOutput) (OutputStream out) -> out.write(chunk))
                    .type(MediaType.APPLICATION_OCTET_STREAM).build();
        }
        byte[] wav;
        try {
            wav = WavWriter.bytes(audio.samples(), audio.sampleRate());
        } catch (IllegalArgumentException e) {
            throw ServerJson.error(400, e.getMessage(), "invalid_request_error");
        }
        return Response.ok(wav).type("audio/wav").build();
    }

    /**
     * A streamed response over {@code source}: the WAV header with the unknown-length sentinel followed by the
     * decoded PCM runs, raw PCM without a header, or OpenAI's speech server-sent events. The rate comes from the
     * generator, validated before the stream starts so an unsupported model is refused with its own status.
     */
    private Response streamResponse(int sampleRate, boolean pcm, boolean sse, TtsStream.Source source) {
        AtomicBoolean cancelled = new AtomicBoolean();
        ClientLink.onClose(http, () -> cancelled.set(true));
        TtsStream stream = new TtsStream(source, cancelled::get, sampleRate, !pcm, sse);
        // the first chunk (or the source's refusal) is known before the response starts, as llama.cpp's streamed
        // route waits for its first result; an input the generator refuses is then answered with its own status
        try {
            stream.awaitFirst();
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, e.getMessage(), "not_supported_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "speech generation failed: " + e.getMessage(), "server_error");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(503, "the speech generation was interrupted", "unavailable_error");
        }
        StreamingOutput out = stream::write;
        String type = sse ? "text/event-stream" : pcm ? MediaType.APPLICATION_OCTET_STREAM : "audio/wav";
        return Response.ok(out).type(type).build();
    }

    /** The text-path waveform of the one-shot form, with its refusals mapped to HTTP. */
    private static AudioClip responseSpeech(EngineService target, String text, SpeechOptions options) {
        try {
            return target.synthesizeSpeech(text, options);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, e.getMessage(), "not_supported_error");
        } catch (IllegalArgumentException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "speech generation failed: " + e.getMessage(), "server_error");
        }
    }

    /**
     * The {@code stream_format} field: "audio" (the default) for the raw byte stream, "sse" for OpenAI's speech
     * events. "sse" names a streaming envelope, so it requires {@code "stream": true}; anything else is refused by
     * name.
     */
    private static boolean streamFormat(JsonView body, boolean stream) {
        String format = body.optStr("stream_format", "audio");
        if (!"audio".equals(format) && !"sse".equals(format)) {
            throw ServerJson.badRequest("\"stream_format\" must be \"audio\" or \"sse\", got \"" + format + "\"");
        }
        if (!stream && "sse".equals(format)) {
            throw ServerJson.badRequest("\"stream_format\": \"sse\" needs \"stream\": true");
        }
        return "sse".equals(format);
    }

    /** The {@code response_format} field: "wav" (the default) or "pcm"; anything else is refused by name. */
    private static boolean responseFormat(JsonView body) {
        String format = body.optStr("response_format", "wav");
        if ("wav".equals(format)) {
            return false;
        }
        if ("pcm".equals(format)) {
            return true;
        }
        throw ServerJson.badRequest("\"response_format\" must be \"wav\" or \"pcm\", got \"" + format + "\"");
    }

    /**
     * The speaker reference of the request: the uploaded multipart file, else the base64 {@code speaker_ref_b64}
     * field, decoded through the server's audio decoder; null when the request carries neither.
     */
    private static AudioClip speakerReference(JsonView body, byte[] uploaded) {
        byte[] data = uploaded;
        if (data == null && body.has("speaker_ref_b64")) {
            String encoded = body.optStr("speaker_ref_b64", null);
            if (encoded == null || encoded.isEmpty()) {
                throw ServerJson.badRequest("\"speaker_ref_b64\" must be a non-empty base64 string");
            }
            long maxEncoded = 4L * ((MAX_SPEAKER_REF_BYTES + 2) / 3);
            if (encoded.length() > maxEncoded) {
                throw ServerJson.badRequest("\"speaker_ref\" exceeds " + (MAX_SPEAKER_REF_BYTES / (1024 * 1024))
                        + " MiB");
            }
            try {
                data = Base64.getDecoder().decode(encoded);
            } catch (IllegalArgumentException e) {
                throw ServerJson.badRequest("\"speaker_ref_b64\" is not valid base64");
            }
        }
        if (data == null) {
            return null;
        }
        if (data.length == 0) {
            throw ServerJson.badRequest("\"speaker_ref\" holds no bytes");
        }
        if (data.length > MAX_SPEAKER_REF_BYTES) {
            throw ServerJson.badRequest("\"speaker_ref\" exceeds " + (MAX_SPEAKER_REF_BYTES / (1024 * 1024))
                    + " MiB");
        }
        AudioClip clip;
        try {
            clip = AudioDecoder.decode(data, "speaker_ref");
        } catch (UnsupportedAudioFormatException e) {
            throw ServerJson.badRequest("\"speaker_ref\": " + e.getMessage());
        } catch (IOException e) {
            throw ServerJson.badRequest("\"speaker_ref\" could not be decoded: " + e.getMessage());
        }
        if (clip.durationSeconds() > MAX_SPEAKER_SECONDS) {
            throw ServerJson.badRequest("\"speaker_ref\" is " + Math.round(clip.durationSeconds())
                    + " s long; the limit is " + (int) MAX_SPEAKER_SECONDS + " s");
        }
        return clip;
    }

    /**
     * The speech options of the text path, with llama.cpp's defaults: language English, no voice, seed 0, top-k 50,
     * top-p 1, 512 frames and the generator's stream window.
     */
    private static SpeechOptions options(JsonView body) {
        SpeechOptions options = SpeechOptions.defaults()
                .withLanguage(body.optStr("lang", ""))
                .withVoice(body.optStr("voice", ""))
                .withSeed(body.optI64("seed", SpeechOptions.DEFAULT_SEED));
        if (body.has("top_k")) {
            options = options.withTopK(body.optI32("top_k", SpeechOptions.DEFAULT_TOP_K));
        }
        if (body.has("top_p")) {
            options = options.withTopP((float) body.optF64("top_p", SpeechOptions.DEFAULT_TOP_P));
        }
        int frames = body.has("max_tokens")
                ? body.optI32("max_tokens", SpeechOptions.DEFAULT_MAX_FRAMES)
                : body.optI32("n_predict", SpeechOptions.DEFAULT_MAX_FRAMES);
        if (frames > 0) {
            options = options.withMaxFrames(frames);
        }
        return options;
    }

    /** The text field, {@code "input"} or its {@code "prompt"} alias; null when absent. */
    private static String text(JsonView body) {
        String text = body.optStr("input", null);
        return text != null ? text : body.optStr("prompt", null);
    }

    /**
     * The codes to synthesize: the {@code "codes"} integer array when present; else, for a server without a
     * text-to-speech generator, the integer text of {@code "input"} (or the {@code "prompt"} alias) parsed like the
     * demo's {@code --audio-codes}; null when the request carries no codes — non-numeric text included, which is a
     * speech request for the generator.
     *
     * <p>A qwen3tts, pockettts or lfm3a backbone reads {@code "input"} as text even when it is all digits, so the
     * text path speaks it instead of the wavtokenizer decoder reading it as codes.
     */
    private int[] codes(EngineService target, JsonView body) {
        if (body.has("codes")) {
            Object raw = body.raw("codes");
            if (!(raw instanceof List<?> list)) {
                throw ServerJson.badRequest("\"codes\" must be an array of integers");
            }
            int[] codes = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object entry = list.get(i);
                if (!(entry instanceof Number n)) {
                    throw ServerJson.badRequest("\"codes[" + i + "]\" must be an integer, got "
                            + (entry == null ? "null" : entry.getClass().getSimpleName()));
                }
                codes[i] = n.intValue();
            }
            return codes;
        }
        if (target.supportsTextSpeech()) {
            return null;
        }
        String text = text(body);
        if (text == null || text.isBlank()) {
            return null;
        }
        return parseCodes(text);
    }

    /** Integer codes from comma or whitespace separated text; null when a token is not an integer. */
    private static int[] parseCodes(String text) {
        String[] parts = text.trim().split("[,\\s]+");
        List<Integer> values = new ArrayList<>(parts.length);
        for (String part : parts) {
            if (part.isEmpty()) {
                continue;
            }
            try {
                values.add(Integer.parseInt(part));
            } catch (NumberFormatException e) {
                return null;
            }
        }
        int[] codes = new int[values.size()];
        for (int i = 0; i < codes.length; i++) {
            codes[i] = values.get(i);
        }
        return codes;
    }
}

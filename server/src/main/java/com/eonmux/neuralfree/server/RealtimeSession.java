package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

/**
 * One {@code /v1/realtime} transcription session: vLLM's {@code RealtimeConnection} over the chat-based speech
 * mapping. The client sends {@code session.update} (optionally naming the model), accumulates base64 PCM16
 * little-endian mono at 16 kHz with {@code input_audio_buffer.append}, and {@code input_audio_buffer.commit} runs
 * the clip: the answer streams back as {@code transcription.delta} events and closes with
 * {@code transcription.done}. {@code input_audio_buffer.clear} (llama.cpp) drops the buffer, and a commit with
 * {@code final: true} (vLLM) transcribes the clip and ends the connection's audio input. Errors are
 * {@code {"type": "error", "error", "code"}} events that leave the session open, as vLLM's do.
 *
 * <p>The events are produced synchronously on the calling thread, so a connection processes one commit at a time;
 * a session is bound to one WebSocket connection.</p>
 */
final class RealtimeSession {

    /** Delivers one event's JSON text to the client. */
    @FunctionalInterface
    interface EventSink {
        void send(String json);
    }

    private static final int MAX_AUDIO_BYTES = (int) AudioResource.MAX_AUDIO_BYTES - WavWriter.HEADER_BYTES;

    private final EngineService service;
    private final ModelRouter router;
    private final AudioResource audio;
    private final EventSink sink;
    private final String id = "sess-" + UUID.randomUUID();
    private final long created = System.currentTimeMillis() / 1000;
    private final ByteArrayOutputStream pcm = new ByteArrayOutputStream();
    private String model;
    private boolean closed;
    /** Whether a {@code final} commit ended the audio input of the connection. */
    private boolean finished;

    RealtimeSession(EngineService service, ModelRouter router, EventSink sink) {
        this(service, router, new AudioResource(service, router), sink);
    }

    RealtimeSession(EngineService service, ModelRouter router, AudioResource audio, EventSink sink) {
        this.service = service;
        this.router = router;
        this.audio = audio;
        this.sink = sink;
    }

    /** Sends {@code session.created}, the first event of a connection. */
    void open() {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "session.created");
        event.put("id", id);
        event.put("created", created);
        send(event);
    }

    /** Handles one client event; a malformed event answers an error and leaves the session open. */
    void receive(String body) {
        if (closed) {
            return;
        }
        JsonView event;
        try {
            event = ServerJson.parse(body);
        } catch (RuntimeException e) {
            error(messageOf(e), "invalid_json");
            return;
        }
        try {
            dispatch(event);
        } catch (WebApplicationException e) {
            error(messageOf(e), "processing_error");
        } catch (RuntimeException e) {
            error(String.valueOf(e.getMessage()), "processing_error");
        }
    }

    /** Ends the session: drops the buffered audio and ignores later events. */
    void close() {
        closed = true;
        pcm.reset();
    }

    private void dispatch(JsonView event) {
        String type = event.optStr("type", "");
        switch (type) {
            case "session.update", "transcription_session.update" -> update(event);
            case "input_audio_buffer.append" -> {
                if (finished) {
                    error("input_audio_buffer.append: the audio input is finished", "input_audio_buffer_finished");
                } else {
                    append(event);
                }
            }
            case "input_audio_buffer.commit" -> {
                if (finished) {
                    error("input_audio_buffer.commit: the audio input is finished", "input_audio_buffer_finished");
                } else {
                    commit(event);
                }
            }
            case "input_audio_buffer.clear" -> {
                pcm.reset();
                send(Map.of("type", "input_audio_buffer.cleared"));
            }
            default -> error("Unknown event type: " + type, "unknown_event");
        }
    }

    /** vLLM's {@code session.update}: validates the named model; an omitted one serves the single model. */
    private void update(JsonView event) {
        String requested = event.optStr("model", "");
        if (!requested.isBlank()) {
            try {
                ModelRouter.target(router, service, JsonView.of(Map.of("model", requested), "session.update"),
                        null);
            } catch (WebApplicationException e) {
                error(messageOf(e), "model_not_found");
                return;
            }
            model = requested;
        } else if (router != null && router.active()) {
            error("Missing required field: model", "invalid_event");
        }
    }

    /** vLLM's {@code input_audio_buffer.append}: base64 PCM16 little-endian mono at 16 kHz. */
    private void append(JsonView event) {
        byte[] chunk;
        try {
            chunk = Base64.getDecoder().decode(event.str("audio"));
        } catch (RuntimeException e) {
            error("Invalid audio data: \"audio\" must be base64 PCM16", "invalid_audio");
            return;
        }
        if (chunk.length == 0) {
            error("Invalid audio data: the chunk is empty", "invalid_audio");
            return;
        }
        if (chunk.length % 2 != 0) {
            error("Invalid audio data: PCM16 needs whole samples", "invalid_audio");
            return;
        }
        if ((long) pcm.size() + chunk.length > MAX_AUDIO_BYTES) {
            error("Maximum file size exceeded", "invalid_audio");
            return;
        }
        pcm.writeBytes(chunk);
    }

    /**
     * vLLM's {@code input_audio_buffer.commit}: transcribes the buffered clip and streams its text back.
     * {@code final} marks the last audio of the connection: the buffered clip is transcribed, and later audio
     * events are refused.
     */
    private void commit(JsonView event) {
        boolean last = event.optBool("final", false);
        if (pcm.size() == 0) {
            if (!last) {
                error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty");
                return;
            }
            finished = true;
            done("");
            return;
        }
        byte[] clip = pcm.toByteArray();
        Map<String, Object> options = new LinkedHashMap<>();
        if (model != null) {
            options.put("model", model);
        }
        StringBuilder text = new StringBuilder();
        try {
            audio.transcribePcm(clip, options, false, delta -> {
                text.append(delta);
                send(Map.of("type", "transcription.delta", "delta", delta));
            });
        } catch (WebApplicationException e) {
            error(messageOf(e), "processing_error");
            return;
        } catch (IOException e) {
            error(String.valueOf(e.getMessage()), "processing_error");
            return;
        }
        pcm.reset();
        done(text.toString());
        if (last) {
            finished = true;
        }
    }

    /** The closing {@code transcription.done} event of a committed clip. */
    private void done(String text) {
        Map<String, Object> done = new LinkedHashMap<>();
        done.put("type", "transcription.done");
        done.put("text", text);
        send(done);
    }

    private void send(Map<String, ?> event) {
        sink.send(ServerJson.write(event));
    }

    private void error(String message, String code) {
        Map<String, Object> event = new LinkedHashMap<>();
        event.put("type", "error");
        event.put("error", message == null ? "the request failed" : message);
        event.put("code", code);
        send(event);
    }

    private static String messageOf(RuntimeException e) {
        if (e instanceof WebApplicationException failure) {
            Object message = OpenAiResource.errorEnvelope(failure).get("message");
            return message == null ? failure.getMessage() : String.valueOf(message);
        }
        return e.getMessage();
    }
}

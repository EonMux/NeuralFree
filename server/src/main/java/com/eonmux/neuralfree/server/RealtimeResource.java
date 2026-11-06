package com.eonmux.neuralfree.server;

import io.quarkus.websockets.next.OnClose;
import io.quarkus.websockets.next.OnOpen;
import io.quarkus.websockets.next.OnTextMessage;
import io.quarkus.websockets.next.WebSocket;
import io.quarkus.websockets.next.WebSocketConnection;
import io.smallrye.common.annotation.Blocking;
import jakarta.inject.Inject;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The OpenAI realtime transcription WebSocket, {@code ws://host/v1/realtime}, as vLLM's
 * {@code speech_to_text/realtime} serves it: the connection opens with {@code session.created}, the client
 * accumulates PCM16 audio with {@code input_audio_buffer.append}, and {@code input_audio_buffer.commit} streams
 * {@code transcription.delta} events followed by {@code transcription.done}. The event handling lives in
 * {@link RealtimeSession}; this endpoint owns one session per connection and routes its messages.
 *
 * <p>The handler runs blocking, on a worker thread, because a commit runs the model synchronously; the endpoint's
 * default serial inbound processing keeps one connection's events ordered. The API key check of the HTTP routes
 * applies to the upgrade through {@link ApiKeyUpgradeCheck}.</p>
 */
@WebSocket(path = "/v1/realtime")
public class RealtimeResource {

    @Inject
    EngineService service;

    @Inject
    ModelRouter router;

    private final Map<String, RealtimeSession> sessions = new ConcurrentHashMap<>();

    @OnOpen
    @Blocking
    void open(WebSocketConnection connection) {
        RealtimeSession session = new RealtimeSession(service, router,
                event -> connection.sendTextAndAwait(event));
        sessions.put(connection.id(), session);
        session.open();
    }

    @OnTextMessage
    @Blocking
    void message(WebSocketConnection connection, String text) {
        RealtimeSession session = sessions.get(connection.id());
        if (session != null) {
            session.receive(text);
        }
    }

    @OnClose
    void close(WebSocketConnection connection) {
        RealtimeSession session = sessions.remove(connection.id());
        if (session != null) {
            session.close();
        }
    }
}

package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import io.quarkus.test.common.http.TestHTTPResource;
import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@QuarkusTest
@QuarkusTestResource(value = AudioHttpTest.Launch.class, restrictToAnnotatedClass = true)
class AudioHttpTest {
    private static final Duration TIMEOUT = Duration.ofSeconds(120);

    @TestHTTPResource URI base;

    /** The values the launch properties had before this class set them, for the after-class restore. */
    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    /**
     * Quarkus keeps the class's test application (and its system properties) alive until the next application
     * starts, so the properties are put back here, before any later test class reads them.
     */
    @AfterAll
    static void restoreLaunchProperties() {
        PREVIOUS.forEach((name, value) -> {
            if (value == null) {
                System.clearProperty(name);
            } else {
                System.setProperty(name, value);
            }
        });
    }

    @Test
    void multipartUploadsReachBothRoutes() throws Exception {
        for (String route : new String[] {"transcriptions", "translations"}) {
            HttpResponse<String> response = send(route, "");
            assertEquals(200, response.statusCode(), response.body());
            assertNotNull(ServerTestSupport.view(response.body()).str("text"));
        }
    }

    @Test
    void unversionedTranscriptionAliasReachesTheSameRoute() throws Exception {
        HttpResponse<String> response = sendTo("/audio/transcriptions", "");
        assertEquals(200, response.statusCode(), response.body());
        assertNotNull(ServerTestSupport.view(response.body()).str("text"));
    }

    @Test
    void multipartUnknownFieldsAreRejectedByName() throws Exception {
        HttpResponse<String> response = send("transcriptions", "--test-boundary\r\nContent-Disposition: form-data; name=\"timestamp_granularities\"\r\n\r\nword\r\n");
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(response.body().contains("timestamp_granularities"));
    }

    @Test
    void realtimeWebsocketTranscribesCommittedAudio() throws Exception {
        BlockingQueue<String> events = new LinkedBlockingQueue<>();
        WebSocket socket = HttpClient.newHttpClient().newWebSocketBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .buildAsync(websocketUri(), new TextListener(events))
                .get(10, TimeUnit.SECONDS);
        try {
            assertEquals("session.created", await(events, "session.created").str("type"));
            send(socket, "{\"type\":\"session.update\",\"model\":\"synthetic\"}");
            String audio = Base64.getEncoder().encodeToString(new byte[16000]);
            send(socket, "{\"type\":\"input_audio_buffer.append\",\"audio\":\"" + audio + "\"}");
            send(socket, "{\"type\":\"input_audio_buffer.commit\"}");
            JsonView done = await(events, "transcription.done");
            assertNotNull(done.str("text"));
        } finally {
            close(socket);
        }
    }

    @Test
    void realtimeWebsocketRefusesAnEmptyCommit() throws Exception {
        BlockingQueue<String> events = new LinkedBlockingQueue<>();
        WebSocket socket = HttpClient.newHttpClient().newWebSocketBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .buildAsync(websocketUri(), new TextListener(events))
                .get(10, TimeUnit.SECONDS);
        try {
            await(events, "session.created");
            send(socket, "{\"type\":\"input_audio_buffer.commit\"}");
            JsonView error = await(events, "error");
            assertEquals("input_audio_buffer_commit_empty", error.str("code"));
        } finally {
            close(socket);
        }
    }

    // ------------------------------------------------------------------ helpers

    private URI websocketUri() {
        String root = base.toString().replaceFirst("/$", "");
        return URI.create(root.replaceFirst("^http", "ws") + "/v1/realtime");
    }

    private static void send(WebSocket socket, String text) throws Exception {
        socket.sendText(text, true).get(10, TimeUnit.SECONDS);
    }

    /** Closes the socket; a peer that already vanished is not a test failure. */
    private static void close(WebSocket socket) {
        try {
            socket.sendClose(WebSocket.NORMAL_CLOSURE, "done").get(10, TimeUnit.SECONDS);
        } catch (Exception e) {
            socket.abort();
        }
    }

    /** The next event of {@code type}; earlier events of other types are skipped. */
    private static JsonView await(BlockingQueue<String> events, String type) throws InterruptedException {
        long deadline = System.nanoTime() + TIMEOUT.toNanos();
        while (System.nanoTime() < deadline) {
            String message = events.poll(1, TimeUnit.SECONDS);
            if (message == null) {
                continue;
            }
            JsonView event = ServerTestSupport.view(message);
            if (type.equals(event.optStr("type", ""))) {
                return event;
            }
        }
        throw new AssertionError("no " + type + " event within " + TIMEOUT);
    }

    /** Collects complete text frames. */
    private static final class TextListener implements WebSocket.Listener {
        private final BlockingQueue<String> events;
        private final StringBuilder partial = new StringBuilder();

        TextListener(BlockingQueue<String> events) {
            this.events = events;
        }

        @Override
        public CompletionStage<?> onText(WebSocket socket, CharSequence data, boolean last) {
            partial.append(data);
            if (last) {
                events.add(partial.toString());
                partial.setLength(0);
            }
            socket.request(1);
            return null;
        }
    }

    private HttpResponse<String> send(String route, String extra) throws Exception {
        return sendTo("/v1/audio/" + route, extra);
    }

    private HttpResponse<String> sendTo(String path, String extra) throws Exception {
        ByteBuffer wav = ByteBuffer.allocate(44 + 32000).order(ByteOrder.LITTLE_ENDIAN);
        wav.put(new byte[] {'R','I','F','F'}).putInt(32036).put(new byte[] {'W','A','V','E'});
        wav.put(new byte[] {'f','m','t',' '}).putInt(16).putShort((short) 1).putShort((short) 1);
        wav.putInt(16000).putInt(32000).putShort((short) 2).putShort((short) 16);
        wav.put(new byte[] {'d','a','t','a'}).putInt(32000);
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        body.write(("--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
        body.write(wav.array());
        body.write(("\r\n--test-boundary\r\nContent-Disposition: form-data; name=\"max_tokens\"\r\n\r\n2\r\n"
                + "--test-boundary\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\n0\r\n"
                + "--test-boundary\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n7\r\n"
                + "--test-boundary\r\nContent-Disposition: form-data; name=\"top_p\"\r\n\r\n1.0\r\n"
                + extra + "--test-boundary--\r\n").getBytes(StandardCharsets.US_ASCII));
        HttpRequest request = HttpRequest.newBuilder(base.resolve(path))
                .header("Content-Type", "multipart/form-data; boundary=test-boundary")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body.toByteArray())).build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString());
    }

    public static class Launch implements QuarkusTestResourceLifecycleManager {
        private final Map<String, String> backups = new LinkedHashMap<>();
        @Override public Map<String, String> start() {
            try {
                Path model = SyntheticModels.llama().writeTempFile();
                Path projector = MmprojTestFiles.qwen2aAudio(SyntheticModels.Config.llama().nEmbd());
                Map.of("neuralfree.server.model", model.toString(), "neuralfree.server.mmproj", projector.toString(),
                        "neuralfree.backend", "cpu", "neuralfree.ctx", "2048", "neuralfree.parallel", "1",
                        "neuralfree.kv.paged", "true")
                        .forEach((key, value) -> {
                            backups.put(key, System.setProperty(key, value));
                            PREVIOUS.put(key, backups.get(key));
                        });
                return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        @Override public void stop() {
            backups.forEach((key, value) -> { if (value == null) System.clearProperty(key); else System.setProperty(key, value); });
        }
    }
}

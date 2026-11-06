package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture;
import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import io.quarkus.test.common.http.TestHTTPResource;
import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The speech routes over real HTTP: a synthetic qwen3tts backbone and its generator file serve {@code /tts} and
 * {@code /v1/audio/speech}, and the streamed response carries the same waveform as the one-shot one, both as raw
 * bytes and as OpenAI's speech server-sent events.
 */
@QuarkusTest
@QuarkusTestResource(value = TtsHttpTest.Launch.class, restrictToAnnotatedClass = true)
class TtsHttpTest {

    private static final Duration TIMEOUT = Duration.ofSeconds(120);
    private static final String SPEECH_BODY =
            "{\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 4";

    /** The launch properties this class replaced, restored before later test classes read them. */
    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    @TestHTTPResource
    URI base;

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
    void streamedSpeechCarriesTheOneShotWaveform() throws Exception {
        HttpResponse<byte[]> oneShot = post("/v1/audio/speech", SPEECH_BODY + "}");
        assertEquals(200, oneShot.statusCode(), new String(oneShot.body(), StandardCharsets.UTF_8));
        assertEquals("audio/wav", oneShot.headers().firstValue("content-type").orElse(""));

        HttpResponse<byte[]> streamed = post("/v1/audio/speech", SPEECH_BODY + ", \"stream\": true}");
        assertEquals(200, streamed.statusCode(), new String(streamed.body(), StandardCharsets.UTF_8));
        assertEquals("audio/wav", streamed.headers().firstValue("content-type").orElse(""));
        byte[] body = streamed.body();
        assertEquals(-1, WavWriter.dataSamples(body), "the streamed header carries the unknown length");
        assertArrayEqualsPayload(oneShot.body(), body);

        HttpResponse<byte[]> pcm = post("/tts", SPEECH_BODY
                + ", \"stream\": true, \"response_format\": \"pcm\"}");
        assertEquals(200, pcm.statusCode());
        byte[] canonical = oneShot.body();
        byte[] payload = Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length);
        org.junit.jupiter.api.Assertions.assertArrayEquals(payload, pcm.body());
    }

    @Test
    void sseStreamCarriesBase64DeltasAndTheDoneEvent() throws Exception {
        HttpResponse<String> response = postText("/v1/audio/speech",
                SPEECH_BODY + ", \"stream\": true, \"stream_format\": \"sse\"}");
        assertEquals(200, response.statusCode(), response.body());
        assertEquals("text/event-stream", response.headers().firstValue("content-type").orElse(""));
        byte[] oneShot = post("/v1/audio/speech", SPEECH_BODY + "}").body();

        java.io.ByteArrayOutputStream audio = new java.io.ByteArrayOutputStream();
        boolean done = false;
        for (String frame : response.body().split("\n\n")) {
            String event = null;
            String data = null;
            for (String line : frame.split("\n")) {
                if (line.startsWith("event: ")) {
                    event = line.substring("event: ".length());
                } else if (line.startsWith("data: ")) {
                    data = line.substring("data: ".length());
                }
            }
            if (data == null) {
                continue;
            }
            if ("speech.audio.delta".equals(event)) {
                com.eonmux.neuralfree.utils.json.JsonView delta = ServerTestSupport.view(data);
                assertEquals("speech.audio.delta", delta.str("type"));
                audio.write(Base64.getDecoder().decode(delta.str("audio")));
            } else if ("speech.audio.done".equals(event)) {
                assertEquals("speech.audio.done", ServerTestSupport.view(data).str("type"));
                done = true;
            }
        }
        assertTrue(done, "the stream ends with the done event");
        assertArrayEqualsPayload(oneShot, audio.toByteArray());
    }

    @Test
    void aRefusedInputAnswersBeforeTheStreamStarts() throws Exception {
        HttpResponse<byte[]> response = post("/v1/audio/speech",
                SPEECH_BODY + ", \"stream\": true, \"voice\": \"alice\"}");
        assertEquals(501, response.statusCode());
        assertTrue(new String(response.body(), StandardCharsets.UTF_8).contains("voice"));
    }

    /** The PCM payload of a canonical or streamed WAV matches, byte for byte. */
    private static void assertArrayEqualsPayload(byte[] canonical, byte[] streamed) {
        byte[] expected = Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length);
        byte[] actual = Arrays.copyOfRange(streamed, WavWriter.HEADER_BYTES, streamed.length);
        org.junit.jupiter.api.Assertions.assertArrayEquals(expected, actual);
    }

    private HttpResponse<byte[]> post(String path, String body) throws Exception {
        return HttpClient.newHttpClient().send(request(path, body), HttpResponse.BodyHandlers.ofByteArray());
    }

    private HttpResponse<String> postText(String path, String body) throws Exception {
        return HttpClient.newHttpClient().send(request(path, body), HttpResponse.BodyHandlers.ofString());
    }

    private HttpRequest request(String path, String body) {
        return HttpRequest.newBuilder(base.resolve(path))
                .timeout(TIMEOUT)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
    }

    public static class Launch implements QuarkusTestResourceLifecycleManager {
        private final Map<String, String> backups = new LinkedHashMap<>();

        @Override
        public Map<String, String> start() {
            try {
                Path model = Qwen3TtsBackboneFixture.write();
                Path generator = new Qwen3TtsClipFixture(2024L, Qwen3TtsBackboneFixture.EMB).write();
                Map.of("neuralfree.server.model", model.toString(),
                        "neuralfree.server.mmproj", generator.toString(),
                        "neuralfree.backend", "cpu",
                        "neuralfree.ctx", "2048",
                        "neuralfree.parallel", "1",
                        "neuralfree.kv.paged", "false",
                        "neuralfree.warmup", "false")
                        .forEach((key, value) -> {
                            backups.put(key, System.setProperty(key, value));
                            PREVIOUS.put(key, backups.get(key));
                        });
                return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void stop() {
            backups.forEach((key, value) -> {
                if (value == null) {
                    System.clearProperty(key);
                } else {
                    System.setProperty(key, value);
                }
            });
        }
    }
}

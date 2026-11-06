package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@code /v1/realtime}'s event protocol over the real engine: {@code session.created}, appended PCM16 audio
 * committed into {@code transcription.delta}/{@code transcription.done}, and the error events of unusable input.
 */
class RealtimeSessionTest {
    private static EngineService service;

    private final List<JsonView> events = new ArrayList<>();
    private RealtimeSession session;

    @BeforeAll
    static void open() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = ServerTestSupport.config().memory(MemoryConfig.builder()
                .nCtx(2048).ubatch(64).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.qwen2aAudio(SyntheticModels.Config.llama().nEmbd()).toString()));
    }

    @AfterAll
    static void close() { service.close(); }

    @BeforeEach
    void connect() {
        events.clear();
        session = new RealtimeSession(service, null, json -> events.add(ServerTestSupport.view(json)));
        session.open();
    }

    @Test
    void opensWithSessionCreatedAndStreamsTheCommittedClip() {
        assertEquals("session.created", events.get(0).str("type"));
        assertTrue(events.get(0).str("id").startsWith("sess-"));

        long queries = service.mediaCacheQueries();
        session.receive("{\"type\":\"session.update\",\"model\":\"synthetic\"}");
        session.receive(append(pcm(8000)));
        session.receive("{\"type\":\"input_audio_buffer.commit\"}");

        assertTrue(service.mediaCacheQueries() > queries, "the clip reached the projector");
        StringBuilder deltas = new StringBuilder();
        for (JsonView event : events.subList(1, events.size() - 1)) {
            assertEquals("transcription.delta", event.str("type"));
            deltas.append(event.str("delta"));
        }
        JsonView done = events.get(events.size() - 1);
        assertEquals("transcription.done", done.str("type"));
        assertEquals(deltas.toString(), done.str("text"));
    }

    @Test
    void refusesEmptyCommitUnknownEventAndMalformedAudio() {
        session.receive("{\"type\":\"input_audio_buffer.commit\"}");
        assertError("input_audio_buffer_commit_empty");

        session.receive("{\"type\":\"nonsense\"}");
        assertError("unknown_event");

        session.receive("{\"type\":\"input_audio_buffer.append\",\"audio\":\"%%%\"}");
        assertError("invalid_audio");

        String odd = Base64.getEncoder().encodeToString(new byte[] {1, 2, 3});
        session.receive("{\"type\":\"input_audio_buffer.append\",\"audio\":\"" + odd + "\"}");
        assertError("invalid_audio");

        session.receive("not json");
        assertError("invalid_json");
    }

    @Test
    void finalCommitTranscribesTheClipAndEndsTheAudioInput() {
        session.receive(append(pcm(4000)));
        session.receive("{\"type\":\"input_audio_buffer.commit\",\"final\":true}");
        assertEquals("transcription.done", events.get(events.size() - 1).str("type"));

        session.receive(append(pcm(100)));
        assertError("input_audio_buffer_finished");
        session.receive("{\"type\":\"input_audio_buffer.commit\"}");
        assertError("input_audio_buffer_finished");
    }

    @Test
    void finalCommitOnAnEmptyBufferClosesWithEmptyText() {
        session.receive("{\"type\":\"input_audio_buffer.commit\",\"final\":true}");
        JsonView done = events.get(events.size() - 1);
        assertEquals("transcription.done", done.str("type"));
        assertEquals("", done.str("text"));
    }

    @Test
    void clearDropsTheBufferedClip() {
        session.receive(append(pcm(4000)));
        session.receive("{\"type\":\"input_audio_buffer.clear\"}");
        assertEquals("input_audio_buffer.cleared", events.get(events.size() - 1).str("type"));
        session.receive("{\"type\":\"input_audio_buffer.commit\"}");
        assertError("input_audio_buffer_commit_empty");
    }

    private void assertError(String code) {
        JsonView last = events.get(events.size() - 1);
        assertEquals("error", last.str("type"), last.toString());
        assertEquals(code, last.str("code"));
        assertNotNull(last.str("error"));
    }

    private static String append(byte[] pcm) {
        return "{\"type\":\"input_audio_buffer.append\",\"audio\":\""
                + Base64.getEncoder().encodeToString(pcm) + "\"}";
    }

    /** {@code millis} of silent PCM16 little-endian mono at 16 kHz. */
    private static byte[] pcm(int millis) {
        return new byte[millis * 16 * 2];
    }
}

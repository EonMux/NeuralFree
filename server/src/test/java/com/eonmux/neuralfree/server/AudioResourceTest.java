package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AudioResourceTest {
    private static EngineService service;
    private static AudioResource audio;
    private static byte[] wav;

    @BeforeAll
    static void open() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig config = ServerTestSupport.config().memory(MemoryConfig.builder()
                .nCtx(2048).ubatch(64).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(MmprojTestFiles.qwen2aAudio(SyntheticModels.Config.llama().nEmbd()).toString()));
        audio = new AudioResource(service);
        ByteBuffer b = ByteBuffer.allocate(44 + 32000).order(ByteOrder.LITTLE_ENDIAN);
        b.put(new byte[] {'R','I','F','F'}).putInt(32036).put(new byte[] {'W','A','V','E'});
        b.put(new byte[] {'f','m','t',' '}).putInt(16).putShort((short) 1).putShort((short) 1);
        b.putInt(16000).putInt(32000).putShort((short) 2).putShort((short) 16);
        b.put(new byte[] {'d','a','t','a'}).putInt(32000);
        wav = b.array();
    }

    @AfterAll
    static void close() { service.close(); }

    @Test
    void transcribesAndTranslatesUploadedAudioThroughRealEngine() {
        Map<String, Object> options = Map.of("max_tokens", 2, "temperature", 0);
        for (boolean translation : List.of(false, true)) {
            long queries = service.mediaCacheQueries();
            JsonView result = ServerTestSupport.view(audio.upload(wav, options, translation).getEntity());
            assertNotNull(result.str("text"));
            assertTrue(service.mediaCacheQueries() > queries, "the audio reached the projector");
        }
    }

    @Test
    void translationTakesTheSourceAndTargetLanguageFields() {
        Map<String, Object> options = Map.of("language", "fr", "to_language", "en", "max_tokens", 2,
                "temperature", 0);
        JsonView result = ServerTestSupport.view(audio.upload(wav, options, true).getEntity());
        assertNotNull(result.str("text"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(wav, Map.of("to_language", "en"), false))
                .contains("to_language"), "transcriptions do not translate");
    }

    @Test
    void samplingFieldsReachTheChatRequest() {
        Map<String, Object> options = Map.of("seed", 7L, "top_k", 5L, "top_p", 0.9, "min_p", 0.0,
                "repetition_penalty", 1.0, "max_tokens", 2, "temperature", 0);
        assertNotNull(ServerTestSupport.view(audio.upload(wav, options, false).getEntity()).str("text"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(wav, Map.of("seed", "abc"), false))
                .contains("seed"), "a non-numeric sampling value refuses naming its field");
    }

    @Test
    void textResponseFormatAnswersThePlainTranscript() {
        Response response = audio.upload(wav, Map.of("response_format", "text", "max_tokens", 2, "temperature", 0),
                false);
        assertTrue(String.valueOf(response.getMediaType()).startsWith("text/plain"), response.getMediaType().toString());
        assertNotNull(response.getEntity());
    }

    @Test
    void streamingAnswersTranscriptionChunksWhoseDeltasJoinToTheTranscript() throws Exception {
        Map<String, Object> options = Map.of("stream", true, "max_tokens", 2, "temperature", 0);
        Response response = audio.upload(wav, options, false);
        assertEquals("text/event-stream", String.valueOf(response.getMediaType()));
        String stream = ServerTestSupport.streamText(response);
        assertTrue(stream.endsWith("data: [DONE]\n\n"), stream);
        List<JsonView> chunks = ServerTestSupport.dataFrames(stream);
        assertFalse(chunks.isEmpty());
        StringBuilder deltas = new StringBuilder();
        for (JsonView chunk : chunks) {
            assertEquals("transcription.chunk", chunk.str("object"));
            JsonView choice = JsonView.of(((List<?>) chunk.raw("choices")).get(0), "chunk.choices[0]");
            deltas.append(choice.obj("delta").optStr("content", ""));
        }
        JsonView last = chunks.get(chunks.size() - 1);
        JsonView lastChoice = JsonView.of(((List<?>) last.raw("choices")).get(0), "chunk.choices[0]");
        assertNotNull(lastChoice.raw("finish_reason"), "the last chunk closes the stream");
        String plain = ServerTestSupport.view(audio.upload(wav,
                Map.of("max_tokens", 2, "temperature", 0), false).getEntity()).str("text");
        assertEquals(plain, deltas.toString());
    }

    @Test
    void streamingTranslationsAnswerTranslationChunks() throws Exception {
        Map<String, Object> options = Map.of("stream", true, "to_language", "fr", "max_tokens", 2,
                "temperature", 0);
        Response response = audio.upload(wav, options, true);
        assertEquals("text/event-stream", String.valueOf(response.getMediaType()));
        String stream = ServerTestSupport.streamText(response);
        assertTrue(stream.endsWith("data: [DONE]\n\n"), stream);
        List<JsonView> chunks = ServerTestSupport.dataFrames(stream);
        assertFalse(chunks.isEmpty());
        for (JsonView chunk : chunks) {
            assertEquals("translation.chunk", chunk.str("object"));
            assertTrue(chunk.str("id").startsWith("trsl-"), chunk.str("id"));
        }
    }

    @Test
    void refusesUnsupportedFieldsAndFormatsByName() {
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(wav, Map.of("hotwords", "nf"), false))
                .contains("hotwords"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(wav, Map.of("response_format", "srt"), false))
                .contains("response_format"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(wav,
                Map.of("timestamp_granularities", List.of("word")), false)).contains("timestamp_granularities"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(new byte[0], Map.of(), false)).contains("file"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.upload(new byte[] {1, 2, 3}, Map.of("max_tokens", 1), false))
                .contains("file"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.batch(
                ServerJson.parse("{\"file_url\":\"data:audio/wav;base64,***\"}"), false)).contains("file_url"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.batch(ServerJson.parse("{}"), false))
                .contains("file_url"));
        assertTrue(ServerTestSupport.errorOf(() -> audio.batch(ServerJson.parse("{\"file\":\"x\"}"), false))
                .contains("file"));
    }

    @Test
    void batchUsesFileUrlAndSameMapping(@TempDir Path dir) throws Exception {
        String url = "data:audio/wav;base64," + Base64.getEncoder().encodeToString(wav);
        String body = ServerJson.write(Map.of("file_url", url, "max_tokens", 2, "temperature", 0));
        String trans = "{\"custom_id\":\"a\",\"method\":\"POST\",\"url\":\"/v1/audio/transcriptions\",\"body\":" + body + "}";
        String transl = "{\"custom_id\":\"b\",\"method\":\"POST\",\"url\":\"/v1/audio/translations\",\"body\":" + body + "}";
        Path input = dir.resolve("input.jsonl"), output = dir.resolve("output.jsonl");
        Files.writeString(input, trans + "\n" + transl + "\n");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, 4);
        assertEquals(2, summary.succeeded());
        List<String> lines = Files.readAllLines(output);
        for (int i = 0; i < 2; i++) {
            JsonView record = ServerTestSupport.view(lines.get(i));
            assertNull(record.raw("error"));
            JsonView result = record.obj("response").obj("body");
            String direct = ServerTestSupport.view(audio.upload(wav, Map.of("max_tokens", 2, "temperature", 0), i == 1).getEntity()).str("text");
            assertEquals(direct, result.str("text"));
        }
    }

    @Test
    void batchRefusesStreamingAndNonJsonFormatsByField(@TempDir Path dir) throws Exception {
        String url = "data:audio/wav;base64," + Base64.getEncoder().encodeToString(wav);
        Path input = dir.resolve("input.jsonl"), output = dir.resolve("output.jsonl");
        Files.writeString(input,
                "{\"custom_id\":\"s\",\"method\":\"POST\",\"url\":\"/v1/audio/transcriptions\",\"body\":"
                        + "{\"file_url\":\"" + url + "\",\"stream\":true}}\n"
                        + "{\"custom_id\":\"t\",\"method\":\"POST\",\"url\":\"/v1/audio/transcriptions\",\"body\":"
                        + "{\"file_url\":\"" + url + "\",\"response_format\":\"text\"}}\n");
        BatchRunner.Summary summary = BatchRunner.run(service, input, output, 4);
        assertEquals(2, summary.failed());
        List<String> lines = Files.readAllLines(output);
        assertTrue(ServerTestSupport.view(lines.get(0)).obj("error").str("message").contains("stream"));
        assertTrue(ServerTestSupport.view(lines.get(1)).obj("error").str("message").contains("response_format"));
    }
}

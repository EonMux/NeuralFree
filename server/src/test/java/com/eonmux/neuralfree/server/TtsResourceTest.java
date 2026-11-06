package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.jboss.resteasy.reactive.server.multipart.FileItem;
import org.jboss.resteasy.reactive.server.multipart.FormValue;
import org.jboss.resteasy.reactive.server.multipart.MultipartFormDataInput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The audio generation endpoints over the real service: a loaded wavtokenizer-dec synthesizes the codes to
 * canonical WAV or raw PCM bytes, a speech-generator backbone speaks text one-shot or as a stream (WAV, raw PCM or
 * OpenAI's speech events), a {@code speaker_ref} upload or {@code speaker_ref_b64} field clones a voice, the
 * refusals name their field ({@code response_format}, {@code codes}, {@code input}, {@code speaker_ref}), a model
 * without a speech decoder answers 501 like llama.cpp's {@code has_cap_tts} refusal, and an unloaded service 503.
 */
class TtsResourceTest {

    private static EngineService service;
    private static EngineService noSpeech;
    private static TtsResource resource;
    private static TtsResource silent;
    private static Path file;
    private static Path llamaFile;

    @BeforeAll
    static void open() throws IOException {
        file = SyntheticModels.wavtokenizer(8000, 8, 2).writeTempFile();
        llamaFile = SyntheticModels.llama().writeTempFile();
        EngineConfig config = speechConfig();
        service = new EngineService(file.toString(), config);
        noSpeech = new EngineService(llamaFile.toString(), ServerTestSupport.pagedConfig(64, 2));
        resource = new TtsResource(service);
        silent = new TtsResource(noSpeech);
    }

    /** The engine config of the tests: the tests' backend, no warmup decode (no token forward pass). */
    private static EngineConfig speechConfig() {
        return EngineConfig.builder().backend(ServerTestSupport.backend())
                .sourceOptions(ModelSourceOptions.builder().warmup(false).build()).build();
    }

    @AfterAll
    static void close() throws IOException {
        service.close();
        noSpeech.close();
        Files.deleteIfExists(file);
        Files.deleteIfExists(llamaFile);
    }

    @Test
    void codesAnswerCanonicalWav() throws IOException {
        int[] codes = {1, 2, 3, 4};
        Response response = resource.speech("{\"codes\": [" + codes[0] + ", " + codes[1] + ", " + codes[2] + ", "
                + codes[3] + "]}");
        assertEquals(200, response.getStatus());
        byte[] wav = (byte[]) response.getEntity();
        try (EngineModel direct = LLMModelFactory.open(file.toString(), speechConfig())) {
            AudioClip clip = direct.synthesizeSpeech(codes);
            assertArrayEquals(WavWriter.bytes(clip.samples(), clip.sampleRate()), wav);
        }
        assertEquals((codes.length - 1) * 2, WavWriter.dataSamples(wav));
    }

    @Test
    void theTtsAliasAnswersTheSameBytes() {
        Response a = resource.speech("{\"codes\": [5, 6]}");
        Response b = resource.tts("{\"codes\": [5, 6]}");
        assertEquals(200, a.getStatus());
        assertEquals(200, b.getStatus());
        assertArrayEquals((byte[]) a.getEntity(), (byte[]) b.getEntity());
    }

    @Test
    void textCodesInInputDecodeTheSameWay() {
        Response a = resource.speech("{\"input\": \"7, 8, 9\"}");
        Response b = resource.speech("{\"codes\": [7, 8, 9]}");
        assertEquals(200, a.getStatus());
        assertEquals(200, b.getStatus());
        assertArrayEquals((byte[]) a.getEntity(), (byte[]) b.getEntity());
    }

    @Test
    void pcmAnswersRawLittleEndianSamples() throws IOException {
        Response wav = resource.speech("{\"codes\": [2, 3]}");
        Response pcm = resource.speech("{\"codes\": [2, 3], \"response_format\": \"pcm\"}");
        assertEquals(200, pcm.getStatus());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ((StreamingOutput) pcm.getEntity()).write(out);
        byte[] samples = out.toByteArray();
        assertEquals(2, samples.length / 2);
        byte[] wavBytes = (byte[]) wav.getEntity();
        for (int i = 0; i < samples.length; i++) {
            assertEquals(wavBytes[44 + i], samples[i]);
        }
    }

    @Test
    void refusalsNameTheirField() {
        assertEquals(400, statusOf(() -> resource.speech("{\"codes\": [1], \"response_format\": \"mp3\"}")));
        assertTrue(errorOf(() -> resource.speech("{\"codes\": [1], \"response_format\": \"mp3\"}"))
                .contains("response_format"));
        assertEquals(400, statusOf(() -> resource.speech("{\"codes\": [\"a\"]}")));
        assertTrue(errorOf(() -> resource.speech("{\"codes\": [\"a\"]}")).contains("codes[0]"));
        assertEquals(400, statusOf(() -> resource.speech("{\"codes\": \"1, 2\"}")));
        assertTrue(errorOf(() -> resource.speech("{\"codes\": \"1, 2\"}")).contains("codes"));
        assertEquals(400, statusOf(() -> resource.speech(
                "{\"codes\": [1], \"stream\": true, \"stream_format\": \"webm\"}")));
        assertTrue(errorOf(() -> resource.speech(
                "{\"codes\": [1], \"stream\": true, \"stream_format\": \"webm\"}")).contains("stream_format"));
        assertEquals(400, statusOf(() -> resource.speech("{\"codes\": [1], \"stream_format\": \"sse\"}")));
        assertTrue(errorOf(() -> resource.speech("{\"codes\": [1], \"stream_format\": \"sse\"}"))
                .contains("stream_format"));
    }

    @Test
    void streamingWavWritesTheSentinelHeaderThenTheOneShotPcm() throws IOException {
        Response plain = resource.speech("{\"codes\": [1, 2, 3, 4]}");
        Response streamed = resource.speech("{\"codes\": [1, 2, 3, 4], \"stream\": true}");
        assertEquals(200, streamed.getStatus());
        assertEquals("audio/wav", streamed.getMediaType().toString());
        byte[] canonical = (byte[]) plain.getEntity();
        byte[] stream = body(streamed);
        assertEquals(-1, WavWriter.dataSamples(stream),
                "the streaming header carries the unknown-length sentinel");
        assertArrayEquals(Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length),
                Arrays.copyOfRange(stream, WavWriter.HEADER_BYTES, stream.length));
    }

    @Test
    void streamingPcmWritesRawSamplesWithoutAHeader() throws IOException {
        Response plain = resource.speech("{\"codes\": [1, 2, 3]}");
        Response streamed = resource.speech("{\"codes\": [1, 2, 3], \"stream\": true,"
                + " \"response_format\": \"pcm\"}");
        byte[] canonical = (byte[]) plain.getEntity();
        assertArrayEquals(Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length),
                body(streamed));
    }

    @Test
    void streamingTextMatchesTheOneShotWaveform() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(99L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            String body = "\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 4";
            Response oneShot = r.speech("{" + body + "}");
            Response streamed = r.speech("{" + body + ", \"stream\": true}");
            assertEquals(200, streamed.getStatus());
            byte[] canonical = (byte[]) oneShot.getEntity();
            byte[] stream = body(streamed);
            assertEquals(-1, WavWriter.dataSamples(stream));
            assertArrayEquals(Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length),
                    Arrays.copyOfRange(stream, WavWriter.HEADER_BYTES, stream.length));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void aStreamedInputRefusalAnswersBeforeTheStreamStarts() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(100L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            assertEquals(501, statusOf(() -> r.speech(
                    "{\"input\": \"Hello\", \"voice\": \"alice\", \"stream\": true}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"Hello\", \"voice\": \"alice\", \"stream\": true}"))
                    .contains("voice"));
            assertEquals(501, statusOf(() -> r.speech(
                    "{\"input\": \"Hello\", \"lang\": \"klingon\", \"stream\": true}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"Hello\", \"lang\": \"klingon\", \"stream\": true}"))
                    .contains("klingon"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void sseStreamCarriesBase64AudioDeltasAndADoneEvent() throws IOException {
        Response plain = resource.speech("{\"codes\": [1, 2, 3]}");
        Response sse = resource.speech("{\"codes\": [1, 2, 3], \"stream\": true, \"stream_format\": \"sse\"}");
        assertEquals(200, sse.getStatus());
        assertEquals("text/event-stream", sse.getMediaType().toString());
        String text = new String(body(sse), StandardCharsets.UTF_8);
        ByteArrayOutputStream pcm = new ByteArrayOutputStream();
        boolean done = false;
        for (String frame : text.split("\n\n")) {
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
            com.eonmux.neuralfree.utils.json.JsonView body = ServerTestSupport.view(data);
            if ("speech.audio.delta".equals(event)) {
                assertEquals("speech.audio.delta", body.str("type"));
                pcm.write(Base64.getDecoder().decode(body.str("audio")));
            } else if ("speech.audio.done".equals(event)) {
                assertEquals("speech.audio.done", body.str("type"));
                done = true;
            }
        }
        assertTrue(done, "the stream ends with the done event");
        byte[] canonical = (byte[]) plain.getEntity();
        byte[] decoded = pcm.toByteArray();
        assertArrayEquals(Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length),
                Arrays.copyOfRange(decoded, WavWriter.HEADER_BYTES, decoded.length));
    }

    private static byte[] body(Response response) throws IOException {
        Object entity = response.getEntity();
        if (entity instanceof byte[] bytes) {
            return bytes;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ((StreamingOutput) entity).write(out);
        return out.toByteArray();
    }

    /** A one-second 24 kHz sine as WAV bytes: a reference the server's audio decoder accepts. */
    private static byte[] referenceWav() {
        float[] samples = new float[24000];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
        }
        return WavWriter.bytes(samples, 24000);
    }

    private static String base64(byte[] bytes) {
        return Base64.getEncoder().encodeToString(bytes);
    }

    /** A plain multipart form field. */
    private static FormValue field(String value) {
        return new FormValue() {
            @Override
            public String getValue() {
                return value;
            }

            @Override
            public String getCharset() {
                return null;
            }

            @Override
            public FileItem getFileItem() {
                return null;
            }

            @Override
            public boolean isFileItem() {
                return false;
            }

            @Override
            public String getFileName() {
                return null;
            }

            @Override
            public jakarta.ws.rs.core.MultivaluedMap<String, String> getHeaders() {
                return null;
            }
        };
    }

    /** An uploaded multipart file item. */
    private static FormValue upload(byte[] bytes) {
        FileItem item = new FileItem() {
            @Override
            public boolean isInMemory() {
                return true;
            }

            @Override
            public Path getFile() {
                return null;
            }

            @Override
            public long getFileSize() {
                return bytes.length;
            }

            @Override
            public InputStream getInputStream() {
                return new ByteArrayInputStream(bytes);
            }

            @Override
            public void delete() {
            }

            @Override
            public void write(Path target) throws IOException {
                Files.write(target, bytes);
            }
        };
        return new FormValue() {
            @Override
            public String getValue() {
                return null;
            }

            @Override
            public String getCharset() {
                return null;
            }

            @Override
            public FileItem getFileItem() {
                return item;
            }

            @Override
            public boolean isFileItem() {
                return true;
            }

            @Override
            public String getFileName() {
                return "reference.wav";
            }

            @Override
            public jakarta.ws.rs.core.MultivaluedMap<String, String> getHeaders() {
                return null;
            }
        };
    }

    private static MultipartFormDataInput form(Map<String, FormValue> values) {
        Map<String, Collection<FormValue>> map = new LinkedHashMap<>();
        for (Map.Entry<String, FormValue> entry : values.entrySet()) {
            List<FormValue> one = new ArrayList<>(1);
            one.add(entry.getValue());
            map.put(entry.getKey(), one);
        }
        return () -> map;
    }

    @Test
    void speakerRefB64ClonesThroughTheGeneratorFile() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1235L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB, true).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            String body = "{\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 1,"
                    + " \"speaker_ref_b64\": \"" + base64(referenceWav()) + "\"}";
            Response response = r.speech(body);
            assertEquals(200, response.getStatus());
            assertEquals(com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture.HOP,
                    WavWriter.dataSamples((byte[]) response.getEntity()));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void aSpeakerReferenceStreamsTheClonedVoice() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1236L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB, true).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            String body = "\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 4,"
                    + " \"speaker_ref_b64\": \"" + base64(referenceWav()) + "\"";
            Response oneShot = r.speech("{" + body + "}");
            Response streamed = r.speech("{" + body + ", \"stream\": true}");
            assertEquals(200, streamed.getStatus());
            assertEquals("audio/wav", streamed.getMediaType().toString());
            byte[] canonical = (byte[]) oneShot.getEntity();
            byte[] stream = body(streamed);
            assertEquals(-1, WavWriter.dataSamples(stream));
            assertArrayEquals(Arrays.copyOfRange(canonical, WavWriter.HEADER_BYTES, canonical.length),
                    Arrays.copyOfRange(stream, WavWriter.HEADER_BYTES, stream.length));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void multipartSpeakerRefClonesThroughTheGeneratorFile() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1237L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB, true).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            Map<String, FormValue> values = new LinkedHashMap<>();
            values.put("input", field("Hello"));
            values.put("top_k", field("1"));
            values.put("top_p", field("1"));
            values.put("max_tokens", field("1"));
            values.put("speaker_ref", upload(referenceWav()));
            // the uploaded file wins over the base64 field, as llama.cpp's post_tts does
            values.put("speaker_ref_b64", field("not base64 at all"));
            Response response = r.speechMultipart(form(values));
            assertEquals(200, response.getStatus());
            assertEquals(com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture.HOP,
                    WavWriter.dataSamples((byte[]) response.getEntity()));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void speakerRefRefusalsNameTheField() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1238L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB, true).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            assertEquals(400, statusOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"!!!\"}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"!!!\"}"))
                    .contains("speaker_ref_b64"));
            assertEquals(400, statusOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"\"}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"\"}"))
                    .contains("speaker_ref_b64"));
            String notAudio = base64(new byte[]{1, 2, 3, 4, 5, 6, 7, 8});
            assertEquals(400, statusOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \""
                    + notAudio + "\"}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"" + notAudio + "\"}"))
                    .contains("speaker_ref"));
            String tooLong = base64(WavWriter.bytes(new float[121 * 8000], 8000));
            assertEquals(400, statusOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \""
                    + tooLong + "\"}")));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"hi\", \"speaker_ref_b64\": \"" + tooLong + "\"}"))
                    .contains("121"));

            Map<String, FormValue> notAFile = new LinkedHashMap<>();
            notAFile.put("input", field("hi"));
            notAFile.put("speaker_ref", field("abc"));
            assertEquals(400, statusOf(() -> r.speechMultipart(form(notAFile))));
            assertTrue(errorOf(() -> r.speechMultipart(form(notAFile))).contains("uploaded audio file"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void aSpeakerReferenceNeedsAGeneratorWithAnEncoder() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1239L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            String body = "{\"input\": \"hi\", \"speaker_ref_b64\": \"" + base64(referenceWav()) + "\"}";
            assertEquals(501, statusOf(() -> r.speech(body)));
            assertTrue(errorOf(() -> r.speech(body)).contains("speaker_ref"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void aSpeakerReferenceIsRefusedOnTheCodesPath() {
        String body = "{\"codes\": [1, 2], \"speaker_ref_b64\": \"" + base64(referenceWav()) + "\"}";
        assertEquals(400, statusOf(() -> resource.speech(body)));
        assertTrue(errorOf(() -> resource.speech(body)).contains("speaker_ref"));
        assertTrue(errorOf(() -> resource.speech(body)).contains("audio codes"));
    }

    @Test
    void textWithoutAGeneratorFileNamesTheMissingBackbone() {
        assertEquals(501, statusOf(() -> resource.speech("{\"input\": \"speak this\"}")));
        String message = errorOf(() -> resource.speech("{\"input\": \"speak this\"}"));
        assertTrue(message.contains("no speech-generator backbone"));
        assertTrue(message.contains("qwen3tts"));
    }

    @Test
    void textSynthesizesAudioThroughTheGeneratorFile() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture(1234L,
                com.eonmux.neuralfree.gguf.speech.Qwen3TtsBackboneFixture.EMB).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            String body = "{\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 1}";
            Response response = r.speech(body);
            assertEquals(200, response.getStatus());
            byte[] wav = (byte[]) response.getEntity();
            assertEquals(com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture.HOP, WavWriter.dataSamples(wav));
            Response pcm = r.speech("{\"input\": \"Hello\", \"top_k\": 1, \"top_p\": 1, \"max_tokens\": 1,"
                    + " \"response_format\": \"pcm\"}");
            assertEquals(200, pcm.getStatus());
            ByteArrayOutputStream raw = new ByteArrayOutputStream();
            ((StreamingOutput) pcm.getEntity()).write(raw);
            assertEquals(com.eonmux.neuralfree.gguf.speech.Qwen3TtsClipFixture.HOP, raw.size() / 2);
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void pocketTtsTextSynthesizesAudioThroughItsGeneratorFile() throws IOException {
        Path backbone = com.eonmux.neuralfree.gguf.speech.PocketTtsBackboneFixture.write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.PocketTtsFixtures(4321L, "english")
                .writeWithEosBias(-100f);
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            Response response = r.speech("{\"input\": \"Hello\", \"max_tokens\": 2}");
            assertEquals(200, response.getStatus());
            byte[] wav = (byte[]) response.getEntity();
            assertEquals(2 * com.eonmux.neuralfree.gguf.speech.PocketTtsFixtures.HOP,
                    WavWriter.dataSamples(wav));
            Response voice = assertThrows(jakarta.ws.rs.WebApplicationException.class,
                    () -> r.speech("{\"input\": \"Hello\", \"voice\": \"alice\"}")).getResponse();
            assertEquals(501, voice.getStatus());
            assertTrue(errorOf(() -> r.speech("{\"input\": \"Hello\", \"voice\": \"alice\"}"))
                    .contains("voice"));
            assertTrue(errorOf(() -> r.speech("{\"input\": \"Hello\", \"lang\": \"de\"}")).contains("lang"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void lfm3aTextSpeaksTheTtsTemplateRender() throws IOException {
        Path backbone = new com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture(true, true).writer()
                .str("tokenizer.chat_template.tts", "{% for m in messages %}{{ m['content'] }}{% endfor %}"
                        + "<|audio_start|>").writeTempFile();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Lfm3aGenClipFixture(
                com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture.EMB, false).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            Response response = r.speech("{\"input\": \"Hi\", \"max_tokens\": 2}");
            assertEquals(200, response.getStatus());
            byte[] wav = (byte[]) response.getEntity();
            int first = com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_FRAME_STACK
                    * com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_HOP
                    - (com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_N_FFT
                    - com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_HOP) / 2;
            int frame = com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_FRAME_STACK
                    * com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_HOP;
            int tail = (com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_N_FFT
                    - com.eonmux.neuralfree.gguf.speech.Lfm3aCodec.DEFAULT_HOP) / 2;
            assertEquals(first + frame + tail, WavWriter.dataSamples(wav),
                    "the rendered prompt speaks two frames: the short first chunk, the full one, the tail");
            assertEquals(200, statusOf(() -> r.speech("{\"input\": \"Hi\", \"voice\": \"alloy\"}")),
                    "a voice is a speech-template variable, as llama.cpp's /tts passes it, not a preset to refuse");
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void lfm3aTextWithoutTheSpeechTemplateNamesIt() throws IOException {
        Path backbone = new com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture(true, true).write();
        Path generator = new com.eonmux.neuralfree.gguf.speech.Lfm3aGenClipFixture(
                com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture.EMB, false).write();
        EngineService speech = new EngineService(backbone.toString(), speechConfig(), ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
        try {
            TtsResource r = new TtsResource(speech);
            assertEquals(501, statusOf(() -> r.speech("{\"input\": \"Hi\"}")));
            String message = errorOf(() -> r.speech("{\"input\": \"Hi\"}"));
            assertTrue(message.contains("tokenizer.chat_template.tts"), message);
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void speakerReferencesAndSpeechOptionsAreRefusedByName() {
        assertEquals(400, statusOf(() -> resource.speech("{\"input\": \"hi\", \"speaker_ref\": \"abc\"}")));
        assertTrue(errorOf(() -> resource.speech("{\"input\": \"hi\", \"speaker_ref\": \"abc\"}"))
                .contains("speaker_ref"));
        assertEquals(400, statusOf(() -> resource.speech("{\"input\": \"hi\", \"top_p\": 1.5}")));
        assertTrue(errorOf(() -> resource.speech("{\"input\": \"hi\", \"top_p\": 1.5}")).contains("top_p"));
        assertEquals(400, statusOf(() -> resource.speech("{\"input\": \"hi\", \"top_k\": -1}")));
        assertTrue(errorOf(() -> resource.speech("{\"input\": \"hi\", \"top_k\": -1}")).contains("top_k"));
        assertEquals(400, statusOf(() -> resource.speech("{\"input\": \" \"}")));
        assertTrue(errorOf(() -> resource.speech("{\"input\": \" \"}")).contains("input"));
    }

    @Test
    void aModelWithoutASpeechDecoderAnswersNotSupported() {
        assertEquals(501, statusOf(() -> silent.speech("{\"codes\": [1, 2]}")));
        String message = errorOf(() -> silent.speech("{\"codes\": [1, 2]}"));
        assertTrue(message.contains("does not support audio generation"));
    }

    @Test
    void anUnloadedServiceAnswersUnavailable() throws IOException {
        EngineService unloaded = new EngineService(null, speechConfig());
        try {
            TtsResource none = new TtsResource(unloaded);
            assertEquals(503, statusOf(() -> none.speech("{\"codes\": [1]}")));
        } finally {
            unloaded.close();
        }
    }
}

package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aBackboneFixture;
import com.eonmux.neuralfree.gguf.speech.Lfm3aClipFixture;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The speech route of an lfm3a backbone over the real service: the request's text, language and voice render the
 * model's tts template and the generator file's MaskGIT depthformer and liquid-codec decoder answer with the
 * fixture's one-frame waveform; digit text reaches the generator as text, and the refusals name their field.
 */
class Lfm3aTtsResourceTest {

    /** The service over the synthetic lfm3a backbone and its generator file. */
    private static EngineService openSpeech(Path backbone, Path generator) throws IOException {
        // the fixture's 64-token context is smaller than a rendered tts prompt, so the test widens it
        EngineConfig config = EngineConfig.builder().backend(backend())
                .memory(MemoryConfig.builder().nCtx(256).ubatch(256).build())
                .sourceOptions(ModelSourceOptions.builder().warmup(false).build()).build();
        return new EngineService(backbone.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(generator.toString()));
    }

    @Test
    void textSynthesizesAudioThroughTheGeneratorFile() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path generator = Lfm3aClipFixture.deterministicEos().writeTempFile();
        EngineService speech = openSpeech(backbone, generator);
        try {
            TtsResource resource = new TtsResource(speech);
            String body = "{\"input\": \"Hello\", \"lang\": \"de\", \"voice\": \"male\", \"seed\": 11,"
                    + " \"max_tokens\": 8}";
            Response response = resource.speech(body);
            assertEquals(200, response.getStatus());
            byte[] wav = (byte[]) response.getEntity();
            // the fixture's one decoded frame plus the flushed ISTFT tail, 16-bit mono at 24 kHz
            int samples = WavWriter.dataSamples(wav);
            assertTrue(samples > 0,
                    "the waveform carries the decoded frame and the tail, got " + samples + " samples");
            assertEquals(24000, wav[24] & 0xFF | (wav[25] & 0xFF) << 8 | (wav[26] & 0xFF) << 16
                    | (wav[27] & 0xFF) << 24);
            Response pcm = resource.speech("{\"input\": \"Hello\", \"lang\": \"de\", \"voice\": \"male\","
                    + " \"seed\": 11, \"max_tokens\": 8, \"response_format\": \"pcm\"}");
            assertEquals(200, pcm.getStatus());
            ByteArrayOutputStream raw = new ByteArrayOutputStream();
            ((StreamingOutput) pcm.getEntity()).write(raw);
            assertEquals(samples, raw.size() / 2);
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void digitTextReachesTheGeneratorInsteadOfTheCodeDecoder() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path generator = Lfm3aClipFixture.deterministicEos().writeTempFile();
        EngineService speech = openSpeech(backbone, generator);
        try {
            TtsResource resource = new TtsResource(speech);
            assertEquals(200, resource.speech("{\"input\": \"12 44 7\", \"max_tokens\": 2}").getStatus());
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void refusalsNameTheirField() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path generator = Lfm3aClipFixture.deterministicEos().writeTempFile();
        EngineService speech = openSpeech(backbone, generator);
        try {
            TtsResource resource = new TtsResource(speech);
            assertEquals(400, statusOf(() -> resource.speech("{\"input\": \" \"}")));
            assertTrue(errorOf(() -> resource.speech("{\"input\": \" \"}")).contains("input"));
            assertEquals(400, statusOf(() -> resource.speech("{\"input\": \"hi\", \"speaker_ref\": \"a\"}")));
            assertTrue(errorOf(() -> resource.speech("{\"input\": \"hi\", \"speaker_ref\": \"a\"}"))
                    .contains("speaker_ref"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
            Files.deleteIfExists(generator);
        }
    }

    @Test
    void streamingMatchesTheOneShotWaveform() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path generator = Lfm3aClipFixture.deterministicEos().writeTempFile();
        EngineService speech = openSpeech(backbone, generator);
        try {
            TtsResource resource = new TtsResource(speech);
            String body = "\"input\": \"Hello\", \"lang\": \"de\", \"voice\": \"male\", \"seed\": 11,"
                    + " \"max_tokens\": 8";
            Response oneShot = resource.speech("{" + body + "}");
            Response streamed = resource.speech("{" + body + ", \"stream\": true}");
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

    private static byte[] body(Response response) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ((StreamingOutput) response.getEntity()).write(out);
        return out.toByteArray();
    }

    @Test
    void aBackboneWithoutItsGeneratorFileNamesTheMissingPiece() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        EngineConfig config = EngineConfig.builder().backend(backend())
                .sourceOptions(ModelSourceOptions.builder().warmup(false).build()).build();
        EngineService speech = new EngineService(backbone.toString(), config);
        try {
            TtsResource resource = new TtsResource(speech);
            assertEquals(501, statusOf(() -> resource.speech("{\"input\": \"hi\"}")));
            String message = errorOf(() -> resource.speech("{\"input\": \"hi\"}"));
            assertTrue(message.contains("neuralfree.server.mmproj"));
            assertTrue(message.contains("lfm3a"));
        } finally {
            speech.close();
            Files.deleteIfExists(backbone);
        }
    }
}

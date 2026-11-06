package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lfm3a pipeline end to end: the synthetic lfm3 backbone, the synthetic generator file and the speech seam.
 * The fixture's constant path samples code 0 in every codebook of the first frame and draws the end code in the
 * second, so the whole run is one decoded frame followed by the stop, whose audio equals the codec session's
 * independent decode.
 */
public class Lfm3aSpeechGeneratorTest {

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(256).ubatch(256)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static Lfm3aSpeechGenerator openGenerator() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        Path clip = Lfm3aClipFixture.deterministicEos().writeTempFile();
        GGUFModel model = new GGUFModel(backbone.toString(), config());
        Lfm3aGenClip generatorClip = Lfm3aGenClip.open(clip.toString());
        return Lfm3aSpeechGenerator.open(model, generatorClip);
    }

    @Test
    public void oneFrameOfCodesThenTheEndCodeStops() throws IOException {
        try (Lfm3aSpeechGenerator generator = openGenerator()) {
            assertEquals("lfm3a", generator.name());

            int[][] codes = generator.generateCodes("Hello world", SpeechOptions.defaults());
            assertEquals(1, codes.length);
            assertArrayEquals(new int[Lfm3aClipFixture.CODEBOOKS], codes[0]);

            // the audio is the codec session's decode of those codes plus the trimmed flush tail
            AudioClip expectedBody;
            AudioClip expectedTail;
            try (Lfm3aGenClip clip = Lfm3aGenClip.open(
                    Lfm3aClipFixture.deterministicEos().writeTempFile().toString());
                 Lfm3aCodecSession session = clip.codec().newSession()) {
                expectedBody = session.decodeClip(codes[0]);
                expectedTail = session.flushClip();
            }
            AudioClip audio = generator.synthesize("Hello world", SpeechOptions.defaults());
            assertEquals(24000, audio.sampleRate());
            assertEquals(expectedBody.length() + expectedTail.length(), audio.length());
            float[] expected = new float[expectedBody.length() + expectedTail.length()];
            System.arraycopy(expectedBody.samples(), 0, expected, 0, expectedBody.length());
            System.arraycopy(expectedTail.samples(), 0, expected, expectedBody.length(), expectedTail.length());
            assertArrayEquals(expected, audio.samples(), 0f);
        }
    }

    @Test
    public void theStreamedChunksConcatenateToTheOneShotAudio() throws IOException {
        try (Lfm3aSpeechGenerator generator = openGenerator()) {
            List<AudioClip> chunks = new ArrayList<>();
            generator.synthesize("Hi", SpeechOptions.defaults(), chunks::add);
            AudioClip whole = generator.synthesize("Hi", SpeechOptions.defaults());
            int streamed = 0;
            for (AudioClip chunk : chunks) {
                assertEquals(24000, chunk.sampleRate());
                streamed += chunk.length();
            }
            assertEquals(whole.length(), streamed);
            assertTrue(chunks.size() >= 2);
        }
    }

    @Test
    public void theSameSeedReproducesTheSameCodes() throws IOException {
        try (Lfm3aSpeechGenerator generator = openGenerator()) {
            SpeechOptions options = SpeechOptions.defaults().withSeed(11).withMaxFrames(8);
            int[][] a = generator.generateCodes("Hello", options);
            int[][] b = generator.generateCodes("Hello", options);
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                assertArrayEquals(a[i], b[i]);
            }
        }
    }

    @Test
    public void maxFramesBoundsTheRunWhenTheEndCodeNeverWins() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Lfm3aGenClip clip = Lfm3aGenClip.open(
                     Lfm3aClipFixture.deterministicNoEos().writeTempFile().toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            int[][] codes = generator.generateCodes("Hi", SpeechOptions.defaults().withMaxFrames(3));
            assertEquals(3, codes.length);
            for (int[] frame : codes) {
                assertEquals(Lfm3aClipFixture.CODEBOOKS, frame.length);
                for (int code : frame) {
                    assertTrue(code >= 0 && code < Lfm3aClipFixture.VOCAB - 1);
                }
            }
        }
    }

    @Test
    public void theTtsTemplateRendersLanguageVoiceAndTheControlTail() {
        Map<String, Object> content = new LinkedHashMap<>();
        content.put("role", "user");
        content.put("content", "Hi there.");
        List<Map<String, Object>> messages = List.of(content);
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("bos_token", "<s>");
        context.put("messages", messages);
        context.put("language", "de");
        context.put("voice", "male");
        String rendered = MiniJinja.render(Lfm3aBackboneFixture.TTS_TEMPLATE, context);
        assertEquals("<s><|im_start|>system\nPerform TTS in German using a male voice.<|im_end|>\n"
                + "<|im_start|>user\nHi there.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>", rendered);

        // the defaults: unknown language keeps the raw code, an off-list voice falls back to female
        context.put("language", "zz");
        context.put("voice", "robot");
        rendered = MiniJinja.render(Lfm3aBackboneFixture.TTS_TEMPLATE, context);
        assertEquals("<s><|im_start|>system\nPerform TTS in zz using a female voice.<|im_end|>\n"
                + "<|im_start|>user\nHi there.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>", rendered);
    }

    @Test
    public void refusesBlankText() throws IOException {
        try (Lfm3aSpeechGenerator generator = openGenerator()) {
            assertThrows(IllegalArgumentException.class, () -> generator.synthesize("  ",
                    SpeechOptions.defaults()));
        }
    }

    @Test
    public void refusesAModelWithoutTheTtsTemplate() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsWriter(false).writeTempFile();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Lfm3aGenClip clip = Lfm3aGenClip.open(
                     Lfm3aClipFixture.deterministicEos().writeTempFile().toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> generator.generateCodes("Hi", SpeechOptions.defaults()));
            assertTrue(e.getMessage().contains("tokenizer.chat_template.tts"));
        }
    }

    @Test
    public void refusesAMixedPromptByPieceName() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsWriter(true)
                .str("tokenizer.chat_template.tts",
                        "{{- bos_token -}}{{- '<|im_start|>assistant\\n<|mixed_start|>' -}}")
                .writeTempFile();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Lfm3aGenClip clip = Lfm3aGenClip.open(
                     Lfm3aClipFixture.deterministicEos().writeTempFile().toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> generator.generateCodes("Hi", SpeechOptions.defaults()));
            assertTrue(e.getMessage().contains("<|mixed_start|>"));
        }
    }

    @Test
    public void refusesATemplateThatOmitsTheAudioPiece() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsWriter(true)
                .str("tokenizer.chat_template.tts",
                        "{{- bos_token -}}{{- '<|im_start|>assistant\\nplain text' -}}")
                .writeTempFile();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Lfm3aGenClip clip = Lfm3aGenClip.open(
                     Lfm3aClipFixture.deterministicEos().writeTempFile().toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> generator.generateCodes("Hi", SpeechOptions.defaults()));
            assertTrue(e.getMessage().contains("<|audio_start|>"));
        }
    }

    @Test
    public void refusesUseAfterClose() throws IOException {
        Lfm3aSpeechGenerator generator = openGenerator();
        generator.close();
        assertThrows(IllegalStateException.class,
                () -> generator.synthesize("Hi", SpeechOptions.defaults()));
        generator.close();
    }

    @Test
    public void theNoEosRunIsDeterministicAcrossSeeds() throws IOException {
        Path backbone = Lfm3aBackboneFixture.ttsBackbone();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Lfm3aGenClip clip = Lfm3aGenClip.open(
                     Lfm3aClipFixture.deterministicNoEos().writeTempFile().toString());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, clip)) {
            // the model's zero temperature makes the draws argmaxes, so both seeds take the same codes
            int[][] a = generator.generateCodes("Hi",
                    SpeechOptions.defaults().withSeed(1).withMaxFrames(2));
            int[][] b = generator.generateCodes("Hi",
                    SpeechOptions.defaults().withSeed(2).withMaxFrames(2));
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                assertArrayEquals(a[i], b[i]);
            }
        }
    }
}

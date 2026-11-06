package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The pockettts pipeline end to end: a synthetic backbone, the synthetic generator file and the speech seam. */
public class PocketTtsSpeechGeneratorTest {

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(128)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static SpeechOptions frames(int maxFrames) {
        return SpeechOptions.defaults().withMaxFrames(maxFrames);
    }

    private static PocketTtsClip generatorFile(long seed, String variant, float eosBias) throws IOException {
        return PocketTtsClip.open(new PocketTtsFixtures(seed, variant).writeWithEosBias(eosBias).toString());
    }

    @Test
    public void generatesFramesAndTheirAudio() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(1L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            assertEquals(PocketTtsSpeechGenerator.NAME, generator.name());
            assertEquals(2, generator.generateLatents("Hi", frames(2)).length);
            AudioClip audio = generator.synthesize("Hi", frames(2));
            assertEquals(2 * PocketTtsFixtures.HOP, audio.length());
            assertEquals(24000, audio.sampleRate());
            for (float sample : audio.samples()) {
                assertTrue(sample >= -1f && sample <= 1f);
            }
        }
    }

    @Test
    public void streamingSinkReceivesTheSameSamples() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(2L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            List<AudioClip> chunks = new ArrayList<>();
            generator.synthesize("Hi", frames(2), chunks::add);
            assertEquals("the two-frame window is flushed once", 1, chunks.size());
            assertEquals(2 * PocketTtsFixtures.HOP, chunks.get(0).length());

            chunks.clear();
            generator.synthesize("Hi", frames(2).withChunkFrames(1), chunks::add);
            assertEquals("a one-frame chunk splits the window's samples", 2, chunks.size());
            assertEquals(PocketTtsFixtures.HOP, chunks.get(0).length());
            assertEquals(PocketTtsFixtures.HOP, chunks.get(1).length());
            AudioClip whole = generator.synthesize("Hi", frames(2));
            for (int i = 0; i < PocketTtsFixtures.HOP; i++) {
                assertEquals(whole.samples()[i], chunks.get(0).samples()[i], 0f);
                assertEquals(whole.samples()[PocketTtsFixtures.HOP + i], chunks.get(1).samples()[i], 0f);
            }
        }
    }

    @Test
    public void endOfSpeechFallsBackToThePromptLengthGuess() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(3L, "english", 100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            // english pins no tail, so a prompt of at most six tokens keeps five frames
            assertEquals(5, generator.generateLatents("Hi", frames(10)).length);
            assertEquals(5 * PocketTtsFixtures.HOP, generator.synthesize("Hi", frames(10)).length());
            // "Hello world." exceeds six tokens and keeps three
            assertEquals(3, generator.generateLatents("Hello world.", frames(10)).length);
        }
    }

    @Test
    public void endOfSpeechKeepsThePackTail() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(4L, "french_24l", 100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            SpeechOptions french = frames(20).withLanguage("fr");
            assertEquals("french_24l keeps eight frames after the probe fires", 8,
                    generator.generateLatents("Bonjour", french).length);
            assertEquals(8 * PocketTtsFixtures.HOP, generator.synthesize("Bonjour", french).length());
        }
    }

    @Test
    public void maxFramesCapsEveryChunk() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(5L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            assertEquals(1, generator.generateLatents("Hi", frames(1)).length);
            assertEquals(3, generator.generateLatents("Hi", frames(3)).length);
        }
    }

    @Test
    public void refusesBlankTextAVoiceAndAnotherLanguage() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(6L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> generator.synthesize("  ", frames(1))).getMessage().contains("blank"));
            UnsupportedOperationException voice = assertThrows(UnsupportedOperationException.class,
                    () -> generator.synthesize("hi", frames(1).withVoice("alice")));
            assertTrue(voice.getMessage().contains("alice"));
            assertTrue(voice.getMessage().contains("speaker"));
            UnsupportedOperationException language = assertThrows(UnsupportedOperationException.class,
                    () -> generator.synthesize("hi", frames(1).withLanguage("de")));
            assertTrue(language.getMessage().contains("lang"));
            assertTrue(language.getMessage().contains("english"));
        }
    }

    /** A short reference waveform: two 1920-sample model frames. */
    private static AudioClip reference() {
        float[] samples = new float[2 * PocketTtsFixtures.HOP];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
        }
        return new AudioClip(samples, 24000);
    }

    @Test
    public void aSpeakerReferenceConditionsTheSpeech() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = PocketTtsClip.open(new PocketTtsFixtures(7L, "english", true)
                     .writeWithEosBias(-100f).toString());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            SpeechOptions cloning = frames(2).withSpeakerReference(reference());
            assertEquals(2, generator.generateLatents("Hi", cloning).length);
            AudioClip audio = generator.synthesize("Hi", cloning);
            assertEquals(2 * PocketTtsFixtures.HOP, audio.length());
            assertEquals(2 * PocketTtsFixtures.HOP, generator.synthesize("Hi", frames(2)).length());
        }
    }

    @Test
    public void aSpeakerReferenceNeedsTheEncoder() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(8L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> generator.synthesize("hi", frames(1).withSpeakerReference(reference())));
            assertTrue(e.getMessage(), e.getMessage().contains("speaker_ref"));
        }
    }

    @Test
    public void theVoicePromptStartsWithTheBosBeforeVoiceRow() throws IOException {
        Path backbone = PocketTtsBackboneFixture.writeWithBosBeforeVoice();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = PocketTtsClip.open(new PocketTtsFixtures(9L, "english", true)
                     .writeWithEosBias(-100f).toString());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            int[] chunk = {100, 101};
            float[][] voice = new float[2][PocketTtsFixtures.EMB];
            for (float[] row : voice) {
                java.util.Arrays.fill(row, 3f);
            }
            float[][] rows = generator.promptRows(chunk, voice);
            assertEquals("bos row, two voice rows, the chunk and audio_bos", 1 + 2 + chunk.length + 1,
                    rows.length);
            assertArrayEquals("the voice prompt starts with <|bos_before_voice|>",
                    model.embeddingRow(PocketTtsBackboneFixture.BOS_BEFORE_VOICE), rows[0], 0f);
            assertArrayEquals(voice[0], rows[1], 0f);
            assertArrayEquals(voice[1], rows[2], 0f);
            assertArrayEquals(model.embeddingRow(PocketTtsBackboneFixture.AUDIO_BOS),
                    rows[rows.length - 1], 0f);
            // without a reference the chunk starts at once, with no voice rows at all
            assertEquals(chunk.length + 1, generator.promptRows(chunk, null).length);
        }
    }

    @Test
    public void preparesTextAndBudgetsAsTheReference() {
        assertEquals("Hello, world.", PocketTtsSpeechGenerator.prepareText("  hello; world  ", false));
        assertEquals("Line one and more.", PocketTtsSpeechGenerator.prepareText("line\none and more", false));
        assertEquals("Already there.", PocketTtsSpeechGenerator.prepareText("Already there.", false));
        assertEquals("        Hi.", PocketTtsSpeechGenerator.prepareText("hi", true));
        assertEquals("Four words here now ok.", PocketTtsSpeechGenerator.prepareText("four words here now ok", true));
        assertEquals("", PocketTtsSpeechGenerator.prepareText("   \n  ", false));
        assertEquals(3, PocketTtsSpeechGenerator.countWords(" one  two three "));
        assertEquals(25, PocketTtsSpeechGenerator.chunkBudget(0));
        assertEquals(38, PocketTtsSpeechGenerator.chunkBudget(3));
        assertEquals(75, PocketTtsSpeechGenerator.chunkBudget(12));
    }

    @Test
    public void splitsLongPromptsIntoChunksOfAtMostFiftyTokens() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             PocketTtsClip clip = generatorFile(8L, "english", -100f);
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, clip)) {
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < 60; i++) {
                text.append("word,");
            }
            int[] ids = model.tokenizer().encode(PocketTtsSpeechGenerator.prepareText(text.toString(), false),
                    false, false);
            assertTrue(ids.length > 2 * PocketTtsSpeechGenerator.MAX_CHUNK_TOKENS);
            List<int[]> chunks = generator.splitChunks(ids);
            assertTrue("the prompt is split", chunks.size() > 1);
            int at = 0;
            for (int[] chunk : chunks) {
                assertTrue("chunk of " + chunk.length + " tokens", chunk.length <= 50);
                for (int id : chunk) {
                    assertEquals(ids[at++], id);
                }
            }
            assertEquals("the chunks keep every token in order", ids.length, at);
            assertEquals(PocketTtsFixtures.HOP, generator.synthesize(text.toString(), frames(1)).length());
        }
    }

    @Test
    public void closedGeneratorRefuses() throws IOException {
        Path backbone = PocketTtsBackboneFixture.write();
        PocketTtsClip clip = generatorFile(7L, "english", -100f);
        PocketTtsSpeechGenerator generator;
        GGUFModel model = new GGUFModel(backbone.toString(), config());
        try {
            generator = PocketTtsSpeechGenerator.open(model, clip);
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
        generator.close();
        assertThrows(IllegalStateException.class, () -> generator.generateLatents("hi", frames(1)));
        assertThrows(IllegalStateException.class, () -> generator.synthesize("hi", frames(1)));
        model.close();
    }
}

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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The qwen3tts pipeline end to end: a synthetic backbone, the synthetic generator file and the speech seam. */
public class Qwen3TtsSpeechGeneratorTest {

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(64)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static SpeechOptions oneFrame() {
        return SpeechOptions.defaults().withTopK(1).withTopP(1f).withMaxFrames(1);
    }

    @Test
    public void generatesOneFrameAndItsAudio() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(41L, Qwen3TtsBackboneFixture.EMB);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            assertEquals(Qwen3TtsSpeechGenerator.NAME, generator.name());
            assertEquals(24000, generator.sampleRate());

            int[][] codes = generator.generateCodes("Hello world", oneFrame());
            assertEquals(1, codes.length);
            assertEquals(Qwen3TtsClipFixture.PAGES + 1, codes[0].length);
            assertEquals("top-k 1 samples codec 0 from the one positive head row", 0, codes[0][0]);

            AudioClip audio = generator.synthesize("Hello world", oneFrame());
            assertEquals(Qwen3TtsClipFixture.HOP, audio.length());
            assertEquals(24000, audio.sampleRate());
            for (float sample : audio.samples()) {
                assertTrue(sample >= -1f && sample <= 1f);
            }
        }
    }

    @Test
    public void streamingSinkReceivesTheSameSamples() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(42L, Qwen3TtsBackboneFixture.EMB);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            List<AudioClip> chunks = new ArrayList<>();
            generator.synthesize("Hi", oneFrame(), chunks::add);
            int samples = 0;
            for (AudioClip chunk : chunks) {
                assertEquals(24000, chunk.sampleRate());
                samples += chunk.length();
            }
            assertEquals("the streamed chunks concatenate", generator.synthesize("Hi", oneFrame()).length(), samples);

            SpeechOptions twoFrames = oneFrame().withMaxFrames(2);
            chunks.clear();
            generator.synthesize("Hi", twoFrames, chunks::add);
            assertEquals("the generator's window holds both frames", 1, chunks.size());
            chunks.clear();
            generator.synthesize("Hi", twoFrames.withChunkFrames(1), chunks::add);
            assertEquals("a one-frame chunk splits the window's samples", 2, chunks.size());
            assertEquals(2 * Qwen3TtsClipFixture.HOP, chunks.get(0).length() + chunks.get(1).length());
            AudioClip whole = generator.synthesize("Hi", twoFrames);
            assertEquals(2 * Qwen3TtsClipFixture.HOP, whole.length());
            for (int i = 0; i < Qwen3TtsClipFixture.HOP; i++) {
                assertEquals(whole.samples()[i], chunks.get(0).samples()[i], 0f);
                assertEquals(whole.samples()[Qwen3TtsClipFixture.HOP + i], chunks.get(1).samples()[i], 0f);
            }
        }
    }

    @Test
    public void windowsDecodeWhileTheBackboneGenerates() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(45L, Qwen3TtsBackboneFixture.EMB);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            // four frames over a three-frame decoder window: the first window decodes before the fourth frame
            SpeechOptions four = oneFrame().withMaxFrames(Qwen3TtsClipFixture.WINDOW + 1);
            int[][] codes = generator.generateCodes("Hi", four);
            assertEquals(Qwen3TtsClipFixture.WINDOW + 1, codes.length);

            List<AudioClip> chunks = new ArrayList<>();
            generator.synthesize("Hi", four, chunks::add);
            assertEquals("one clip per decoder window, the last one short", 2, chunks.size());
            assertEquals(Qwen3TtsClipFixture.WINDOW * Qwen3TtsClipFixture.HOP, chunks.get(0).length());
            assertEquals(Qwen3TtsClipFixture.HOP, chunks.get(1).length());

            AudioClip whole = generator.synthesize("Hi", four);
            assertEquals((codes.length) * Qwen3TtsClipFixture.HOP, whole.length());
            int at = 0;
            for (AudioClip chunk : chunks) {
                for (float sample : chunk.samples()) {
                    assertEquals(whole.samples()[at++], sample, 0f);
                }
            }
            assertEquals(whole.length(), at);

            chunks.clear();
            generator.synthesize("Hi", four.withChunkFrames(1), chunks::add);
            assertEquals("a per-frame chunk subdivides each window without changing its decode batch",
                    codes.length, chunks.size());
            for (AudioClip chunk : chunks) {
                assertEquals(Qwen3TtsClipFixture.HOP, chunk.length());
            }
        }
    }

    @Test
    public void refusesBlankTextAndAVoiceByName() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(43L, Qwen3TtsBackboneFixture.EMB);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> generator.synthesize("  ", oneFrame())).getMessage().contains("blank"));
            UnsupportedOperationException voice = assertThrows(UnsupportedOperationException.class,
                    () -> generator.synthesize("hi", oneFrame().withVoice("alice")));
            assertTrue(voice.getMessage().contains("alice"));
            assertTrue(voice.getMessage().contains("speaker"));
        }
    }

    /** A short reference waveform at the encoder's rate. */
    private static AudioClip reference() {
        float[] samples = new float[4096];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0));
        }
        return new AudioClip(samples, 24000);
    }

    @Test
    public void aSpeakerReferenceConditionsTheSpeech() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(45L, Qwen3TtsBackboneFixture.EMB, true);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            SpeechOptions cloning = oneFrame().withSpeakerReference(reference());
            int[][] codes = generator.generateCodes("Hello", cloning);
            assertEquals(1, codes.length);
            assertEquals(Qwen3TtsClipFixture.PAGES + 1, codes[0].length);
            AudioClip audio = generator.synthesize("Hello", cloning);
            assertEquals(Qwen3TtsClipFixture.HOP, audio.length());
            // the same text without a reference runs the prompt without speaker rows
            assertEquals(Qwen3TtsClipFixture.HOP, generator.synthesize("Hello", oneFrame()).length());
        }
    }

    @Test
    public void aSpeakerReferenceNeedsTheEncoder() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(46L, Qwen3TtsBackboneFixture.EMB);
        try (GGUFModel model = new GGUFModel(backbone.toString(), config());
             Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, clip)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> generator.synthesize("hi", oneFrame().withSpeakerReference(reference())));
            assertTrue(e.getMessage(), e.getMessage().contains("speaker_ref"));
        }
    }

    @Test
    public void closedGeneratorRefuses() throws IOException {
        Path backbone = Qwen3TtsBackboneFixture.write();
        Qwen3TtsClipFixture fixture = new Qwen3TtsClipFixture(44L, Qwen3TtsBackboneFixture.EMB);
        Qwen3TtsClip clip = Qwen3TtsClip.open(fixture.write().toString());
        Qwen3TtsSpeechGenerator generator;
        GGUFModel model = new GGUFModel(backbone.toString(), config());
        try {
            generator = Qwen3TtsSpeechGenerator.open(model, clip);
        } catch (IOException | RuntimeException e) {
            model.close();
            throw e;
        }
        generator.close();
        assertThrows(IllegalStateException.class, () -> generator.generateCodes("hi", oneFrame()));
        model.close();
    }
}

package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator.Mode;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator.OmniSink;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator.OmniTurn;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The omni pipeline end to end over the synthetic backbone and generator: the interleaved loop of
 * mtmd-helper-gen.cpp's {@code lfm3a_gen_audio_pipeline} driven by head rows chosen so every trace is
 * hand-computable. The all-ones embeddings make every text step's logits the head's column sums —
 * {@code audio_start} outranks the fillers, so the continuing TEXT logit bias is what keeps a turn in text, and
 * {@code text_end}'s raised row hands a mixed turn over to audio. Frames decode the constant bias spectrum, so a
 * turn's audio length is the chunk arithmetic alone: the first frame drains the iSTFT's head padding and the
 * flush keeps the tail minus its own {@code (n_fft - hop) / 2} of end padding.
 */
public class Lfm3aOmniGeneratorTest {

    /** The pipeline's control token ids in the synthetic vocabulary. */
    private static final int AUDIO_START = Lfm3aBackboneFixture.AUDIO_START;
    private static final int MIXED_START = Lfm3aBackboneFixture.MIXED_START;
    /** The byte token of 'A' (the plain-text prompt's single token). */
    private static final int BYTE_A = 3 + 'A';

    private static final int STEPS_CAP = SpeechOptions.DEFAULT_MAX_FRAMES;
    private static final int TEXT_BLOCK = Lfm3aSpeechGenerator.N_TEXT_BLOCK;
    private static final int AUDIO_BLOCK = Lfm3aSpeechGenerator.N_AUDIO_BLOCK;
    private static final int CHUNK_FIRST = Lfm3aCodec.DEFAULT_FRAME_STACK * Lfm3aCodec.DEFAULT_HOP
            - (Lfm3aCodec.DEFAULT_N_FFT - Lfm3aCodec.DEFAULT_HOP) / 2;
    private static final int CHUNK_REST = Lfm3aCodec.DEFAULT_FRAME_STACK * Lfm3aCodec.DEFAULT_HOP;
    /** The flush keeps the iSTFT tail minus its own (n_fft - hop) / 2 of end padding. */
    private static final int FLUSH = (Lfm3aCodec.DEFAULT_N_FFT - Lfm3aCodec.DEFAULT_HOP) / 2;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(1024).ubatch(64)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** One opened pipeline: the backbone model, the generator file and the bound generator. */
    private static final class Rig implements AutoCloseable {
        final GGUFModel model;
        final Lfm3aGenClip clip;
        final Lfm3aSpeechGenerator generator;

        Rig(boolean textEndWins, boolean omniTokens, int stateWidth, boolean endCodeDraws) throws IOException {
            Path backbone = new Lfm3aBackboneFixture(textEndWins, omniTokens).write();
            Path clipFile = new Lfm3aGenClipFixture(stateWidth, endCodeDraws).write();
            this.model = new GGUFModel(backbone.toString(), config());
            try {
                this.clip = Lfm3aGenClip.open(clipFile.toString());
                this.generator = Lfm3aSpeechGenerator.open(model, clip);
            } catch (IOException | RuntimeException e) {
                model.close();
                throw e;
            }
        }

        @Override
        public void close() throws IOException {
            generator.close();
            model.close();
        }
    }

    /** A sink that records the pieces and the chunks as they happen. */
    private static final class Recording implements OmniSink {
        final List<String> pieces = new ArrayList<>();
        final List<AudioClip> chunks = new ArrayList<>();

        @Override
        public void text(String piece) {
            pieces.add(piece);
        }

        @Override
        public void audio(AudioClip chunk) {
            chunks.add(chunk);
        }

        int samples() {
            int n = 0;
            for (AudioClip chunk : chunks) {
                n += chunk.length();
            }
            return n;
        }
    }

    @Test
    public void freshTtsSpeaksTwoFramesOfSilence() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            assertEquals(Lfm3aSpeechGenerator.NAME, rig.generator.name());
            AudioClip audio = rig.generator.synthesize("Say hi", SpeechOptions.defaults().withMaxFrames(2));
            assertEquals(24000, audio.sampleRate());
            assertEquals("frame one drains the head padding, frame two is full, the flush holds the tail",
                    CHUNK_FIRST + CHUNK_REST + FLUSH, audio.length());
            float peak = 0f;
            for (float sample : audio.samples()) {
                peak = Math.max(peak, Math.abs(sample));
            }
            assertTrue("the constant bias spectrum stays inaudible, peak " + peak, peak < 1e-9f);
        }
    }

    @Test
    public void freshEndFrameStopsWithoutAWaveform() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, true)) {
            AudioClip audio = rig.generator.synthesize("Say hi", SpeechOptions.defaults());
            assertEquals("a fresh end frame decodes nothing and the empty session flushes nothing", 0,
                    audio.length());
        }
    }

    @Test
    public void freshPromptMustOpenWithAudioStart() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> rig.generator.generate(new int[]{BYTE_A}, Mode.AUDIO, false,
                            SpeechOptions.defaults(), null));
            assertTrue(e.getMessage(), e.getMessage().contains("tts"));
        }
    }

    @Test
    public void continuingTextBansTheSwitchTokens() throws IOException {
        // audio_start outranks the fillers, so the logit bias is the only thing keeping the turn in text
        try (Rig rig = new Rig(false, true, Lfm3aBackboneFixture.EMB, false)) {
            Recording sink = new Recording();
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A}, Mode.TEXT, true,
                    SpeechOptions.defaults(), sink);
            assertEquals("no audio without a mode switch", 0, sink.chunks.size());
            assertEquals(STEPS_CAP, turn.text().length() / "<|filler_0|>".length());
            assertEquals("<|filler_0|>".repeat(STEPS_CAP), turn.text());
            assertEquals(STEPS_CAP, sink.pieces.size());
        }
    }

    @Test
    public void mixedTextEndHandsTheTurnToAudio() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A, MIXED_START}, Mode.MIXED, true,
                    SpeechOptions.defaults().withMaxFrames(2), null);
            assertEquals("text_end is control and never reaches the text", "", turn.text());
            assertEquals("one frame of audio plus the flush", CHUNK_FIRST + FLUSH, turn.audio().length());
        }
    }

    @Test
    public void anEndFrameInsideMixedReturnsToTextAndStops() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, true)) {
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A, MIXED_START}, Mode.MIXED, true,
                    SpeechOptions.defaults(), null);
            assertEquals("", turn.text());
            assertEquals("the end frame flushes the untouched session to nothing", 0, turn.audio().length());
        }
    }

    @Test
    public void interleavedBlocksRunToTheStepCap() throws IOException {
        try (Rig rig = new Rig(false, true, Lfm3aBackboneFixture.EMB, false)) {
            Recording sink = new Recording();
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A, MIXED_START}, Mode.MIXED, true,
                    SpeechOptions.defaults(), sink);
            assertEquals("every text slot samples audio_start, a control: no text arrives", 0,
                    turn.text().length());
            int cycles = STEPS_CAP / (TEXT_BLOCK + AUDIO_BLOCK);
            int tail = STEPS_CAP % (TEXT_BLOCK + AUDIO_BLOCK);
            int frames = cycles * AUDIO_BLOCK + Math.max(0, tail - TEXT_BLOCK);
            assertEquals(cycles + " full blocks and the tail's " + (tail - TEXT_BLOCK) + " frames", frames,
                    sink.chunks.size() - 1);
            assertEquals("the flush adds the last chunk", 1 + frames, sink.chunks.size());
            assertEquals(CHUNK_FIRST + (frames - 1) * CHUNK_REST + FLUSH, turn.audio().length());
            assertEquals(turn.audio().length(), sink.samples());
            for (AudioClip chunk : sink.chunks) {
                assertEquals(24000, chunk.sampleRate());
            }
        }
    }

    @Test
    public void streamingSinkSeesTheTurnAsItHappens() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            Recording sink = new Recording();
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A, AUDIO_START}, Mode.AUDIO, true,
                    SpeechOptions.defaults().withMaxFrames(2), sink);
            assertEquals(String.join("", sink.pieces), turn.text());
            assertEquals(sink.samples(), turn.audio().length());
            assertTrue(sink.chunks.size() >= 3);
            assertEquals("two speech frames, no flush step", 2, turn.steps());
            assertEquals("the two prompt tokens and the continued turn's switch token", 3, turn.promptTokens());
        }
    }

    @Test
    public void aRenderedPromptMayNameAVoice() throws IOException {
        // the prompt is rendered by the caller, so a voice reaches the model through the template, not the
        // generator: generate() must not refuse it (the raw-text synthesize() path does)
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            OmniTurn turn = rig.generator.generate(new int[]{BYTE_A, AUDIO_START}, Mode.AUDIO, true,
                    SpeechOptions.defaults().withVoice("female").withMaxFrames(1), null);
            assertEquals(CHUNK_FIRST + FLUSH, turn.audio().length());
        }
    }

    @Test
    public void refusesBlankTextAndAVoiceByName() throws IOException {
        try (Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false)) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> rig.generator.synthesize("  ", SpeechOptions.defaults())).getMessage().contains("blank"));
            UnsupportedOperationException voice = assertThrows(UnsupportedOperationException.class,
                    () -> rig.generator.synthesize("hi", SpeechOptions.defaults().withVoice("alloy")));
            assertTrue(voice.getMessage(), voice.getMessage().contains("alloy"));
            assertTrue(voice.getMessage(), voice.getMessage().contains("voice"));
        }
    }

    @Test
    public void closedGeneratorRefuses() throws IOException {
        Rig rig = new Rig(true, true, Lfm3aBackboneFixture.EMB, false);
        try {
            rig.generator.close();
            assertThrows(IllegalStateException.class,
                    () -> rig.generator.synthesize("hi", SpeechOptions.defaults()));
        } finally {
            rig.model.close();
        }
    }

    @Test
    public void refusesAnOmniUnreadyVocabulary() {
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> new Rig(true, false, Lfm3aBackboneFixture.EMB, false));
        assertTrue(e.getMessage(), e.getMessage().contains("audio_start"));
    }

    @Test
    public void refusesAMismatchedProjection() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new Rig(true, true, 8, false));
        assertTrue(e.getMessage(), e.getMessage().contains("projection"));
    }

    @Test
    public void refusesABackboneOfAnotherArchitecture() throws IOException {
        Path backbone = SyntheticModels.lfm2().writeTempFile();
        Path clipFile = new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, false).write();
        GGUFModel model = new GGUFModel(backbone.toString(), config());
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(clipFile.toString())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> Lfm3aSpeechGenerator.open(model, clip));
            assertTrue(e.getMessage(), e.getMessage().contains("lfm3"));
        } finally {
            model.close();
        }
    }
}

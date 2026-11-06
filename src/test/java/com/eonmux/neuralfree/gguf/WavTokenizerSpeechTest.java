package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end speech synthesis over the synthetic wavtokenizer-dec engine model: the codes map one-to-one to
 * spectral frames, the declared geometry keys drive the iSTFT, and a model without the keys refuses by name.
 */
public class WavTokenizerSpeechTest {

    private static final int RATE = 8000;
    private static final int N_FFT = 8;
    private static final int HOP = 2;

    private Path withKeys;
    private Path withoutKeys;
    private EngineModel model;

    @Before
    public void open() throws IOException {
        withKeys = SyntheticModels.wavtokenizer(RATE, N_FFT, HOP).writeTempFile();
        withoutKeys = SyntheticModels.wavtokenizer(0, 0, 0).writeTempFile();
        // no warmup decode: the decoder has no token forward pass
        EngineConfig config = EngineConfig.builder()
                .sourceOptions(com.eonmux.neuralfree.gguf.ModelSourceOptions.builder().warmup(false).build())
                .build();
        model = LLMModelFactory.open(withKeys.toString(), config);
    }

    @After
    public void close() throws IOException {
        if (model != null) {
            model.close();
        }
        Files.deleteIfExists(withKeys);
        Files.deleteIfExists(withoutKeys);
    }

    @Test
    public void codesSynthesizeToTheCenteredPcm() {
        SpeechFormat format = model.speechFormat();
        assertNotNull(format);
        assertEquals(RATE, format.sampleRate());
        assertEquals(N_FFT, format.nFft());
        assertEquals(HOP, format.hopLength());
        int[] codes = {1, 2, 3, 4};
        AudioClip clip = model.synthesizeSpeech(codes);
        assertEquals((codes.length - 1) * HOP, clip.length());
        assertEquals(RATE, clip.sampleRate());
        for (float v : clip.samples()) {
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    public void synthesizeMatchesTheStagedDecode() throws IOException {
        int[] codes = {5, 6, 7};
        AudioClip clip = model.synthesizeSpeech(codes);
        AudioClip staged = model.speechFormat().decode(model.decodeAudioCodes(codes));
        assertEquals(staged.length(), clip.length());
        assertEquals(staged.sampleRate(), clip.sampleRate());
        assertArrayEquals(staged.samples(), clip.samples(), 1e-9f);
    }

    @Test
    public void streamingDecodeConcatenatesToTheOneShotClip() {
        int[] codes = {5, 6, 7, 8, 9};
        AudioClip whole = model.synthesizeSpeech(codes);
        java.util.List<AudioClip> chunks = new java.util.ArrayList<>();
        model.synthesizeSpeech(codes, chunks::add);
        java.util.List<Float> samples = new java.util.ArrayList<>();
        for (AudioClip chunk : chunks) {
            assertEquals(RATE, chunk.sampleRate());
            for (float v : chunk.samples()) {
                samples.add(v);
            }
        }
        assertEquals(whole.length(), samples.size());
        for (int i = 0; i < whole.length(); i++) {
            assertEquals(whole.samples()[i], samples.get(i), 0f);
        }
        assertEquals(model.requireSpeechFormat().sampleRate(), RATE);
    }

    @Test
    public void theWavRoundTripsTheClip() {
        AudioClip clip = model.synthesizeSpeech(new int[] {9, 10, 11, 12, 13});
        byte[] wav = WavWriter.bytes(clip.samples(), clip.sampleRate());
        assertEquals(clip.length(), WavWriter.dataSamples(wav));
    }

    @Test
    public void refusalsNameTheMissingPiece() throws IOException {
        EngineConfig config = EngineConfig.builder()
                .sourceOptions(com.eonmux.neuralfree.gguf.ModelSourceOptions.builder().warmup(false).build())
                .build();
        try (EngineModel plain = LLMModelFactory.open(withoutKeys.toString(), config)) {
            assertNull(plain.speechFormat());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> plain.synthesizeSpeech(new int[] {1, 2}));
            assertTrue(e.getMessage().contains("audio.sample_rate"));
            assertTrue(e.getMessage().contains("audio.n_fft"));
            assertTrue(e.getMessage().contains("audio.hop_length"));
            IllegalArgumentException badCode = assertThrows(IllegalArgumentException.class,
                    () -> model.synthesizeSpeech(new int[] {-1}));
            assertTrue(badCode.getMessage().contains("codebook"));
        }
    }
}

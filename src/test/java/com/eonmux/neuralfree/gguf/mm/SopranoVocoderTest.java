package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.ops.audio.VocosDecoder;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The soprano decoder over a synthetic generator GGUF: the file keys gate the open, the Vocos head runs the
 * graph of soprano.cpp, and {@code decode} applies the PCM trimming of {@code decode_spectrum} at the loader's
 * audio constants (32 kHz, one frame per 512 samples).
 */
public class SopranoVocoderTest {

    private static final int INP = SopranoFixtures.INP;

    private Path file;

    @Before
    public void open() throws IOException {
        file = SopranoFixtures.write();
    }

    @After
    public void close() throws IOException {
        Files.deleteIfExists(file);
    }

    @Test
    public void theFileOpensWithTheLoaderGeometry() throws IOException {
        SopranoVocoder vocoder = SopranoVocoder.open(file.toString());
        try {
            assertEquals(INP, vocoder.decoder().nInp());
            assertEquals(512, vocoder.decoder().nFft());
            assertEquals(SopranoFixtures.BLOCKS, vocoder.decoder().blocks());
        } finally {
            vocoder.close();
        }
    }

    @Test
    public void decodeAppliesDecodeSpectrumTrimming() throws IOException {
        SopranoVocoder vocoder = SopranoVocoder.open(file.toString());
        try {
            int frames = 3;
            float[][] feats = new float[frames][INP];
            AudioClip clip = vocoder.decode(feats);
            int up = VocosDecoder.UPSCALE * (frames - 1) + 1;
            assertEquals((up - 1) * VocosDecoder.HOP_LENGTH, clip.length());
            assertEquals(VocosDecoder.SAMPLE_RATE, clip.sampleRate());
            for (float v : clip.samples()) {
                assertTrue(Float.isFinite(v));
            }
            byte[] wav = WavWriter.bytes(clip.samples(), clip.sampleRate());
            assertEquals(clip.length(), WavWriter.dataSamples(wav));
        } finally {
            vocoder.close();
        }
    }

    @Test
    public void theSpectrumCarriesThePolarHead() throws IOException {
        SopranoVocoder vocoder = SopranoVocoder.open(file.toString());
        try {
            float[][] spectra = vocoder.spectrum(new float[2][INP]);
            int up = VocosDecoder.UPSCALE + 1;
            assertEquals(up, spectra.length);
            assertEquals(2 * vocoder.decoder().nBins(), spectra[0].length);
        } finally {
            vocoder.close();
        }
    }

    @Test
    public void refusalsNameTheFileAndTheBatch() throws IOException {
        SopranoVocoder vocoder = SopranoVocoder.open(file.toString());
        try {
            IllegalArgumentException small = assertThrows(IllegalArgumentException.class,
                    () -> vocoder.decode(new float[1][INP]));
            assertTrue(small.getMessage().contains("at least 2 frames"));
            IllegalArgumentException wide = assertThrows(IllegalArgumentException.class,
                    () -> vocoder.decode(new float[3][INP + 1]));
            assertTrue(wide.getMessage().contains("the model takes " + INP));
        } finally {
            vocoder.close();
        }
        assertThrows(IllegalStateException.class, () -> vocoder.decode(new float[3][INP]));
    }

    @Test
    public void aNonSopranoClipFileIsRefusedByName() throws IOException {
        Path missing = SopranoFixtures.soprano(null).writeTempFile();
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> SopranoVocoder.open(missing.toString()));
            assertTrue(e.getMessage().contains("clip.gen.audio.projector_type"));
        } finally {
            Files.deleteIfExists(missing);
        }
        Path wrong = SopranoFixtures.soprano("lfm3a_gen").writeTempFile();
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> SopranoVocoder.open(wrong.toString()));
            assertTrue(e.getMessage().contains("lfm3a_gen"));
        } finally {
            Files.deleteIfExists(wrong);
        }
    }
}

package com.eonmux.neuralfree.gguf.audio;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/** The speech geometry keys of a wavtokenizer-dec file and the centered-iSTFT decode contract. */
public class SpeechFormatTest {

    @Test
    public void theKeysAreAllOrNone() {
        assertNull(SpeechFormat.ofGguf(kv()));
        SpeechFormat f = SpeechFormat.ofGguf(kv("audio.sample_rate", 24000, "audio.n_fft", 8, "audio.hop_length", 2));
        assertEquals(24000, f.sampleRate());
        assertEquals(8, f.nFft());
        assertEquals(2, f.hopLength());
        assertEquals(5, f.nBins());
        assertEquals(10, f.featureWidth());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> SpeechFormat.ofGguf(kv("audio.sample_rate", 24000, "audio.n_fft", 8)));
        assertEquals(true, e.getMessage().contains("audio.hop_length"));
    }

    @Test
    public void decodeAppliesTheCenteredTrim() {
        SpeechFormat f = new SpeechFormat(24000, 8, 2);
        int frames = 6;
        float[][] spectra = constantSpectra(frames, f.featureWidth(), 0.0f);
        // frames - 1 hops of audio, the n_fft/2 head dropped
        AudioClip clip = f.decode(spectra);
        assertEquals((frames - 1) * f.hopLength(), clip.length());
        assertEquals(24000, clip.sampleRate());
        // silence in, silence out
        for (float v : clip.samples()) {
            assertEquals(0.0f, v, 1e-6f);
        }
        assertEquals(0, f.decode(constantSpectra(1, f.featureWidth(), 0f)).length());
    }

    @Test
    public void refusalsNameTheContract() {
        SpeechFormat f = new SpeechFormat(24000, 8, 2);
        assertThrows(IllegalArgumentException.class, () -> f.decode(new float[0][]));
        assertThrows(IllegalArgumentException.class,
                () -> f.decode(constantSpectra(2, f.featureWidth() - 1, 0f)));
        assertThrows(IllegalArgumentException.class, () -> new SpeechFormat(0, 8, 2));
        assertThrows(IllegalArgumentException.class, () -> new SpeechFormat(24000, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> new SpeechFormat(24000, 8, 9));
    }

    @Test
    public void streamingDecodeConcatenatesToTheOneShotClip() {
        int[][] geometries = {{8, 2}, {8, 6}, {16, 4}, {6, 3}, {12, 12}};
        for (int[] geometry : geometries) {
            SpeechFormat f = new SpeechFormat(24000, geometry[0], geometry[1]);
            for (int frames = 1; frames <= 9; frames++) {
                float[][] spectra = randomSpectra(frames, f.featureWidth(), 31L * frames + geometry[0]);
                AudioClip whole = f.decode(spectra);
                java.util.List<AudioClip> chunks = new java.util.ArrayList<>();
                f.decode(spectra, chunks::add);
                java.util.List<Float> streamed = new java.util.ArrayList<>();
                for (AudioClip chunk : chunks) {
                    assertEquals(24000, chunk.sampleRate());
                    for (float v : chunk.samples()) {
                        streamed.add(v);
                    }
                }
                assertEquals("geometry " + geometry[0] + "/" + geometry[1] + ", " + frames + " frames",
                        whole.length(), streamed.size());
                for (int i = 0; i < whole.length(); i++) {
                    assertEquals(whole.samples()[i], streamed.get(i), 0f);
                }
            }
        }
    }

    @Test
    public void theStreamDropsTheSameHeadAndTailAsTheOneShotDecode() {
        SpeechFormat f = new SpeechFormat(24000, 8, 2);
        SpeechFormat.Stream stream = f.stream();
        float[][] spectra = randomSpectra(5, f.featureWidth(), 7L);
        java.util.List<Float> streamed = new java.util.ArrayList<>();
        for (float[] row : spectra) {
            for (float v : stream.push(row)) {
                streamed.add(v);
            }
        }
        for (float v : stream.finish()) {
            streamed.add(v);
        }
        AudioClip whole = f.decode(spectra);
        assertEquals(whole.length(), streamed.size());
        for (int i = 0; i < whole.length(); i++) {
            assertEquals(whole.samples()[i], streamed.get(i), 0f);
        }
        assertThrows(IllegalStateException.class, () -> stream.push(spectra[0]));
        assertEquals(0, stream.finish().length);
        assertThrows(IllegalArgumentException.class, () -> f.stream().push(new float[f.featureWidth() - 1]));
    }

    private static float[][] randomSpectra(int frames, int width, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        float[][] out = new float[frames][width];
        for (float[] row : out) {
            for (int i = 0; i < row.length; i++) {
                row[i] = (rng.nextFloat() - 0.5f) * 2f;
            }
        }
        return out;
    }

    private static float[][] constantSpectra(int frames, int width, float v) {
        float[][] out = new float[frames][width];
        for (float[] row : out) {
            java.util.Arrays.fill(row, v);
        }
        return out;
    }

    private static GgufKv kv(Object... pairs) {
        Map<String, Object> map = new HashMap<>();
        for (int i = 0; i + 1 < pairs.length; i += 2) {
            map.put((String) pairs[i], ((Number) pairs[i + 1]).longValue());
        }
        return GgufKv.ofMap(map);
    }
}

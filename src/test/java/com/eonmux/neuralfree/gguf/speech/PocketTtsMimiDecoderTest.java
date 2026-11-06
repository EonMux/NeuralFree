package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link PocketTtsMimiSession}: the resolved geometry and state slots, the GEN_WAV graph and the streaming SEANet
 * decoder against an independent double-precision transcription of llama.cpp's pockettts graphs, the equivalence of
 * one window and several streamed windows, and the refusals.
 */
public class PocketTtsMimiDecoderTest {

    private static final float DELTA = 1e-4f;

    private static PocketTtsClip open(long seed) throws IOException {
        return PocketTtsClip.open(new PocketTtsFixtures(seed).write().toString());
    }

    private static float[][] frames(int n, long seed) {
        Random rng = new Random(seed);
        float[][] out = new float[n][PocketTtsFixtures.LATENT];
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < out[t].length; i++) {
                out[t][i] = (rng.nextFloat() * 2f - 1f) * 0.8f;
            }
        }
        return out;
    }

    private static void assertSamples(double[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("sample " + i, expected[i], actual[i], DELTA);
        }
    }

    @Test
    public void geometryFollowsTheReferenceConstants() throws IOException {
        try (PocketTtsClip clip = open(1L)) {
            PocketTtsMimiDecoder decoder = clip.decoder();
            assertEquals(PocketTtsFixtures.LATENT, decoder.latent());
            assertEquals(PocketTtsFixtures.WIDTH, decoder.width());
            assertEquals(PocketTtsFixtures.TFM_LAYERS, decoder.nLayer());
            assertEquals(PocketTtsFixtures.TFM_HEADS, decoder.nHead());
            assertEquals(PocketTtsFixtures.TFM_HEADS, decoder.nHeadKv());
            assertEquals(250, decoder.context());
            assertEquals(PocketTtsFixtures.HOP, decoder.hop());
            assertEquals(24000, decoder.sampleRate());
            assertEquals(8, decoder.windowFrames());
            assertEquals(Arrays.asList("tfm_pos", "tfm_k_0", "tfm_v_0", "up", "dec_in", "dec_up_0", "dec_res_0",
                    "dec_up_1", "dec_res_1", "dec_up_2", "dec_res_2", "dec_out"), decoder.stateSlots());
        }
    }

    @Test
    public void oneWindowMatchesTheReferenceGraph() throws IOException {
        try (PocketTtsClip clip = open(2L)) {
            PocketTtsFixtures fixture = new PocketTtsFixtures(2L);
            PocketTtsReference reference = new PocketTtsReference(fixture);
            float[][] window = frames(PocketTtsMimiDecoder.WINDOW_FRAMES, 21L);
            double[] expected = reference.decode(window);
            PocketTtsMimiSession session = clip.decoder().newSession();
            assertSamples(expected, session.decode(window));
            assertEquals(PocketTtsFixtures.HOP * PocketTtsMimiDecoder.WINDOW_FRAMES, expected.length);
            assertEquals(PocketTtsMimiDecoder.DOWNSAMPLE * PocketTtsMimiDecoder.WINDOW_FRAMES, session.position());
        }
    }

    @Test
    public void partialWindowsMatchTheReferenceGraph() throws IOException {
        try (PocketTtsClip clip = open(3L)) {
            for (int n : new int[]{1, 3, 5}) {
                PocketTtsReference reference = new PocketTtsReference(new PocketTtsFixtures(3L));
                float[][] window = frames(n, 100L + n);
                assertSamples(reference.decode(window), clip.decoder().newSession().decode(window));
            }
        }
    }

    /**
     * A window streamed in two calls produces the same samples as one call, and both match the reference's own
     * streamed state: the convolution contexts, the transposed-convolution tails, the attention window and the
     * continuing RoPE positions all bridge the calls.
     */
    @Test
    public void streamedWindowsConcatenateToTheWholeWindow() throws IOException {
        try (PocketTtsClip clip = open(4L)) {
            float[][] all = frames(PocketTtsMimiDecoder.WINDOW_FRAMES, 41L);
            float[][] first = Arrays.copyOfRange(all, 0, 4);
            float[][] second = Arrays.copyOfRange(all, 4, 8);
            PocketTtsReference reference = new PocketTtsReference(new PocketTtsFixtures(4L));
            double[] whole = reference.decode(all);

            float[] once = clip.decoder().newSession().decode(all);
            assertSamples(whole, once);

            PocketTtsMimiSession streamed = clip.decoder().newSession();
            float[] a = streamed.decode(first);
            assertEquals(4 * PocketTtsFixtures.HOP, a.length);
            float[] b = streamed.decode(second);
            assertEquals(4 * PocketTtsFixtures.HOP, b.length);
            float[] joined = new float[a.length + b.length];
            System.arraycopy(a, 0, joined, 0, a.length);
            System.arraycopy(b, 0, joined, a.length, b.length);
            assertSamples(whole, joined);
            assertEquals(8 * PocketTtsMimiDecoder.DOWNSAMPLE, streamed.position());
        }
    }

    @Test
    public void sessionsAreIndependent() throws IOException {
        try (PocketTtsClip clip = open(5L)) {
            float[][] window = frames(4, 51L);
            float[] a = clip.decoder().newSession().decode(window);
            float[] b = clip.decoder().newSession().decode(window);
            assertEquals(a.length, b.length);
            for (int i = 0; i < a.length; i++) {
                assertEquals(a[i], b[i], 0f);
            }
        }
    }

    @Test
    public void decodeClipCarriesTheSampleRate() throws IOException {
        try (PocketTtsClip clip = open(6L)) {
            AudioClip audio = clip.decoder().newSession().decodeClip(frames(2, 61L));
            assertEquals(24000, audio.sampleRate());
            assertEquals(2 * PocketTtsFixtures.HOP, audio.length());
            for (float sample : audio.samples()) {
                org.junit.Assert.assertTrue(sample >= -1f && sample <= 1f);
            }
        }
    }

    @Test
    public void refusesBadBatchesAndClosedSessions() throws IOException {
        try (PocketTtsClip clip = open(7L)) {
            PocketTtsMimiSession session = clip.decoder().newSession();
            assertThrows(IllegalArgumentException.class, () -> session.decode(new float[0][]));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(new float[PocketTtsMimiDecoder.WINDOW_FRAMES + 1][]));
            assertThrows(IllegalArgumentException.class, () -> session.decode(new float[][]{{1f, 2f}}));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(new float[][]{new float[PocketTtsFixtures.LATENT + 1]}));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(new float[][]{{1f, Float.NaN, 0f}}));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(new float[][]{{1f, Float.POSITIVE_INFINITY, 0f}}));
            session.close();
            assertThrows(IllegalStateException.class, () -> session.decode(frames(1, 71L)));
        }
    }
}

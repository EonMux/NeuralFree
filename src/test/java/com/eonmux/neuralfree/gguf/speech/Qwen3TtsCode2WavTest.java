package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Qwen3TtsCode2Wav}: the resolved geometry, the constant-path output of the quantizer and decoder graphs, the
 * persisted window state, the padded partial batch and the input refusals.
 */
public class Qwen3TtsCode2WavTest {

    private static Qwen3TtsClip open() throws IOException {
        return Qwen3TtsClip.open(Qwen3TtsFixtures.writer().writeTempFile().toString());
    }

    private static int[][] codes(int frames) {
        int[][] out = new int[frames][];
        for (int i = 0; i < frames; i++) {
            out[i] = new int[]{0, 0, 0};
        }
        return out;
    }

    @Test
    public void geometryFollowsTheGeneratorFile() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            Qwen3TtsCode2Wav wav = clip.code2Wav();
            assertEquals(Qwen3TtsFixtures.WINDOW, wav.window());
            assertEquals(2, wav.hop());
            assertEquals(24000, wav.sampleRate());
            assertEquals(Qwen3TtsFixtures.CODES, wav.codesPerFrame());
            assertEquals(Qwen3TtsFixtures.CODEBOOKS, wav.codebooks());
            assertEquals(Arrays.asList("tfm_pos", "tfm_k_0", "tfm_v_0", "pre_conv", "up0_dwconv", "dac_entry",
                    "dac0_tail", "dac0_res0", "dac_post_conv"), wav.stateSlots());
        }
    }

    @Test
    public void constantPathDecodesTheGoldenSamples() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            Qwen3TtsCode2WavSession session = clip.code2Wav().newSession();
            float[] one = session.decode(codes(1));
            assertEquals(2, one.length);
            for (float v : one) {
                assertEquals(Qwen3TtsFixtures.GOLDEN, v, 1e-6f);
            }
            float[] full = session.decode(codes(Qwen3TtsFixtures.WINDOW));
            assertEquals(Qwen3TtsFixtures.WINDOW * 2, full.length);
            for (float v : full) {
                assertEquals(Qwen3TtsFixtures.GOLDEN, v, 1e-6f);
            }
            assertTrue(session.position() > 0f);
        }
    }

    @Test
    public void partialBatchStillEmitsItsFramesAndAdvancesTheWindow() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            Qwen3TtsCode2WavSession session = clip.code2Wav().newSession();
            assertEquals(0f, session.position(), 0f);
            assertEquals(2, session.decode(codes(1)).length);
            assertEquals(Qwen3TtsFixtures.WINDOW, session.position(), 0f);
            assertEquals(4, session.decode(codes(2)).length);
            assertEquals(2f * Qwen3TtsFixtures.WINDOW, session.position(), 0f);
        }
    }

    @Test
    public void sessionsAreIndependent() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            float[] a = clip.code2Wav().newSession().decode(codes(3));
            float[] b = clip.code2Wav().newSession().decode(codes(3));
            assertArrayEquals(a, b, 0f);
        }
    }

    @Test
    public void decodeClipCarriesTheSampleRate() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            AudioClip clipAudio = clip.code2Wav().newSession().decodeClip(codes(2));
            assertEquals(24000, clipAudio.sampleRate());
            assertEquals(4, clipAudio.length());
        }
    }

    @Test
    public void refusesBadBatchesAndCodes() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            Qwen3TtsCode2WavSession session = clip.code2Wav().newSession();
            assertThrows(IllegalArgumentException.class, () -> session.decode(new int[0][]));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(codes(Qwen3TtsFixtures.WINDOW + 1)));
            assertThrows(IllegalArgumentException.class, () -> session.decode(new int[][]{{0, 0}}));
            assertThrows(IllegalArgumentException.class, () -> session.decode(new int[][]{{0, 0, 0, 0}}));
            assertThrows(IllegalArgumentException.class, () -> session.decode(new int[][]{{-1, 0, 0}}));
            assertThrows(IllegalArgumentException.class,
                    () -> session.decode(new int[][]{{0, Qwen3TtsFixtures.VOCAB, 0}}));
        }
    }

    @Test
    public void closeRefusesFurtherDecodes() throws IOException {
        try (Qwen3TtsClip clip = open()) {
            Qwen3TtsCode2WavSession session = clip.code2Wav().newSession();
            session.close();
            assertThrows(IllegalStateException.class, () -> session.decode(codes(1)));
        }
    }
}

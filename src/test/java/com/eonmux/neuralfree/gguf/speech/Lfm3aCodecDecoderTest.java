package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The liquid-codec decoder against its synthetic graph: every quantizer and transformer weight is zero over unit
 * norms, so a decoded position is the constant zero vector and the polar spectrum is the {@code out_proj} bias
 * alone — log magnitudes of {@code -30} over the {@code n_bins} bins and zero phase — which the streaming iSTFT
 * turns into samples under {@code 1e-9}. The chunk arithmetic follows the fixed STFT: six positions per frame,
 * the first frame draining the {@code (n_fft - hop) / 2} head padding, the flush holding the same amount again
 * (the raw iSTFT tail of {@code n_fft - hop} less its own tail padding).
 */
public class Lfm3aCodecDecoderTest {

    private static final int CB = Lfm3aDepthformer.N_CODEBOOKS;
    private static final int POSITIONS = Lfm3aCodec.DEFAULT_FRAME_STACK;
    private static final int HOP = Lfm3aCodec.DEFAULT_HOP;
    private static final int HEAD_PAD = (Lfm3aCodec.DEFAULT_N_FFT - HOP) / 2;
    private static final int TAIL = Lfm3aCodec.DEFAULT_N_FFT - HOP;

    private static Path clipFile(boolean endCodeDraws) throws IOException {
        return new Lfm3aGenClipFixture(Lfm3aBackboneFixture.EMB, endCodeDraws).write();
    }

    private static int[] codes(int frames) {
        int[] codes = new int[CB * frames];
        return codes;
    }

    @Test
    public void decodesFrameSizedChunksWithTheReferencePadding() throws IOException {
        Path file = clipFile(false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aCodecSession wav = clip.codec().newSession();
            assertEquals(CB, clip.codec().nCodebooks());
            assertEquals(Lfm3aGenClipFixture.FSQ_VOCAB, clip.fsqVocab());

            float[] first = wav.decodeClip(codes(1)).samples();
            // the first call gives up (n_fft - hop) / 2 of its hop samples to the head padding
            assertEquals(POSITIONS * HOP - HEAD_PAD, first.length);
            float[] second = wav.decodeClip(codes(1)).samples();
            assertEquals(POSITIONS * HOP, second.length);
            for (float sample : first) {
                assertTrue(Math.abs(sample) < 1e-9f);
            }
            float[] tail = wav.flushClip().samples();
            assertEquals("the flush drains the remaining n_fft - hop and drops its own tail padding",
                    HEAD_PAD, tail.length);
        }
    }

    @Test
    public void chunkedFramesEqualTheOneShotCall() throws IOException {
        Path file = clipFile(false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            float[] whole = clip.codec().newSession().decodeClip(codes(2)).samples();

            Lfm3aCodecSession streaming = clip.codec().newSession();
            float[] head = streaming.decodeClip(codes(1)).samples();
            float[] rest = streaming.decodeClip(codes(1)).samples();
            assertEquals(whole.length, head.length + rest.length);
            float[] joined = new float[whole.length];
            System.arraycopy(head, 0, joined, 0, head.length);
            System.arraycopy(rest, 0, joined, head.length, rest.length);
            assertArrayEquals("the session's states carry across chunks", whole, joined, 0f);
        }
    }

    @Test
    public void flushResetsAndTheNextFrameStartsFresh() throws IOException {
        Path file = clipFile(false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aCodecSession wav = clip.codec().newSession();
            wav.decodeClip(codes(1));
            assertEquals(HEAD_PAD, wav.flushClip().length());
            assertEquals("a session that decoded no frame flushes to nothing", 0,
                    clip.codec().newSession().flushClip().length());

            Lfm3aCodecSession reset = clip.codec().newSession();
            assertEquals(POSITIONS * HOP - HEAD_PAD, reset.decodeClip(codes(1)).length());
            reset.reset();
            assertEquals("the reset restores the head padding budget", POSITIONS * HOP - HEAD_PAD,
                    reset.decodeClip(codes(1)).length());
            reset.reset();
            assertEquals(0, reset.flushClip().length());
        }
    }

    @Test
    public void theConstantBiasSpectrumIsInaudible() throws IOException {
        Path file = clipFile(false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            float[] pcm = clip.codec().newSession().decodeClip(codes(4)).samples();
            assertEquals(4 * POSITIONS * HOP - HEAD_PAD, pcm.length);
            // exp(-30) magnitudes over a zero phase: every sample sits far under any audible level
            float peak = 0f;
            for (float sample : pcm) {
                peak = Math.max(peak, Math.abs(sample));
            }
            assertTrue("peak " + peak, peak < 1e-9f);
        }
    }

    @Test
    public void refusesWrongCodeCountsAndCodes() throws IOException {
        Path file = clipFile(false);
        try (Lfm3aGenClip clip = Lfm3aGenClip.open(file.toString())) {
            Lfm3aCodecSession wav = clip.codec().newSession();
            assertThrows(IllegalArgumentException.class, () -> wav.decodeClip(new int[CB - 1]));
            assertThrows(IllegalArgumentException.class, () -> wav.decodeClip(new int[CB + 1]));
            int[] endCode = new int[CB];
            Arrays.fill(endCode, Lfm3aGenClipFixture.FSQ_VOCAB);
            assertThrows(IllegalArgumentException.class, () -> wav.decodeClip(endCode));
            int[] negative = new int[CB];
            negative[0] = -1;
            assertThrows(IllegalArgumentException.class, () -> wav.decodeClip(negative));
        }
    }

    @Test
    public void mirrorsTheStreamingIstftTheGraphFeeds() throws IOException {
        // an independent session over the shared iSTFT must reproduce the decoder's sample counts
        StreamingIstft istft = new StreamingIstft(Lfm3aCodec.DEFAULT_N_FFT, HOP);
        float[] frame = new float[2 * istft.nBins()];
        assertEquals("the first frame gives its whole hop to the head padding", 0,
                istft.processFrame(frame).length);
        assertEquals("the second frame drains the remaining head padding", HEAD_PAD - HOP,
                istft.processFrame(frame).length);
        assertEquals("full frames yield a hop each", HOP, istft.processFrame(frame).length);
        assertEquals(TAIL, istft.flush().length);
    }
}

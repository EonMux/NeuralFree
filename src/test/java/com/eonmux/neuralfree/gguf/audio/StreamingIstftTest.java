package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The streaming iSTFT against a direct inverse DFT, the overlap-add round trip through the forward FFT, and the
 * streaming contract (head trim, flush length, reset).
 */
public class StreamingIstftTest {

    @Test
    public void theInverseTransformMatchesTheDirectDft() {
        int n = 8;
        int hop = 2;
        StreamingIstft istft = new StreamingIstft(n, hop);
        Random rnd = new Random(7);
        float[][] frames = new float[3][2 * istft.nBins()];
        for (float[] frame : frames) {
            for (int i = 0; i < frame.length; i++) {
                frame[i] = rnd.nextFloat() * 2 - 1;
            }
        }
        istft.processFrame(frames[0].clone());
        istft.processFrame(frames[1].clone());
        float[] out = istft.processFrame(frames[2].clone());

        double[] hann = hann(n);
        int calls = 3;
        assertEquals(2, out.length);
        for (int i = 0; i < out.length; i++) {
            // buffer position i after three calls: the overlap-add of every frame's iFFT at the position the
            // shifts left it at, normalized by the same positions' window-square sum
            double acc = 0;
            double ws = 0;
            for (int k = 0; k < calls; k++) {
                int at = i + (calls - 1 - k) * hop;
                if (at < n) {
                    double x = directIfftReal(frames[k], n)[at];
                    acc += x * hann[at];
                    ws += hann[at] * hann[at];
                }
            }
            assertEquals("sample " + i, acc / ws, out[i], 1e-4);
        }
    }

    @Test
    public void aRoundTripRecoversTheSignal() {
        int nFft = 16;
        int hop = 4;
        int frames = 12;
        // the centered-STFT layout of the references: frames of (frames - 1) * hop samples padded by n_fft/2
        int n = (frames - 1) * hop;
        float[] signal = new float[n];
        Random rnd = new Random(11);
        for (int i = 0; i < n; i++) {
            signal[i] = (float) Math.sin(2.0 * Math.PI * 3.0 * i / nFft) * (0.5f + rnd.nextFloat() * 0.2f);
        }
        float[] padded = new float[nFft + n + nFft];
        System.arraycopy(signal, 0, padded, nFft, n);
        Fft fft = new Fft(nFft);
        double[] hann = hann(nFft);
        StreamingIstft istft = new StreamingIstft(nFft, hop);
        float[] pcm = new float[0];
        for (int f = 0; f < frames; f++) {
            float[] windowed = new float[nFft];
            for (int i = 0; i < nFft; i++) {
                windowed[i] = padded[f * hop + i] * (float) hann[i];
            }
            float[] spec = new float[2 * nFft];
            fft.forward(windowed, spec);
            float[] frame = new float[2 * (nFft / 2 + 1)];
            System.arraycopy(spec, 0, frame, 0, frame.length);
            pcm = concat(pcm, istft.processFrame(frame));
        }
        pcm = concat(pcm, istft.flush());
        // the stream holds frames*hop - (n_fft - hop)/2 + (n_fft - hop) samples; the shared iSTFT drops
        // (n_fft - hop)/2 of the stream head, so the signal starts n_fft - (n_fft - hop)/2 in
        assertEquals(frames * hop - (nFft - hop) / 2 + (nFft - hop), pcm.length);
        int offset = nFft - (nFft - hop) / 2;
        float[] recovered = Arrays.copyOfRange(pcm, offset, offset + n);
        for (int i = 8; i < n - 8; i++) {
            assertEquals("sample " + i, signal[i], recovered[i], 2e-3f);
        }
    }

    @Test
    public void theHeadTrimDrainsAcrossFrames() {
        // (n_fft - hop) / 2 = 5 <= hop: the first frame returns no samples, the second the remainder
        StreamingIstft istft = new StreamingIstft(16, 5);
        float[] frame = new float[2 * istft.nBins()];
        assertEquals(0, istft.processFrame(frame.clone()).length);
        assertEquals(5, istft.processFrame(frame.clone()).length);
        // from here the trim is drained: every frame returns a full hop
        assertEquals(5, istft.processFrame(frame.clone()).length);
    }

    @Test
    public void flushDrainsTheOverlapTail() {
        StreamingIstft istft = new StreamingIstft(16, 4);
        float[] frame = new float[2 * istft.nBins()];
        istft.processFrame(frame);
        float[] tail = istft.flush();
        // the n_fft - hop samples the buffers still hold (mtmd-audio.cpp:1610-1622)
        assertEquals(12, tail.length);
        // the reference does not clear on flush: the shifted buffers answer the same length again
        assertEquals(12, istft.flush().length);
    }

    @Test
    public void resetRestoresTheColdStart() {
        StreamingIstft istft = new StreamingIstft(16, 4);
        Random rnd = new Random(3);
        float[] frame = new float[2 * istft.nBins()];
        for (int i = 0; i < frame.length; i++) {
            frame[i] = rnd.nextFloat() - 0.5f;
        }
        float[] first = istft.processFrame(frame.clone());
        istft.reset();
        float[] second = istft.processFrame(frame.clone());
        assertArrayEquals(first, second, 1e-6f);
    }

    @Test
    public void refusalsNameTheContract() {
        assertThrows(IllegalArgumentException.class, () -> new StreamingIstft(0, 1));
        assertThrows(IllegalArgumentException.class, () -> new StreamingIstft(8, 0));
        assertThrows(IllegalArgumentException.class, () -> new StreamingIstft(8, 9));
        StreamingIstft istft = new StreamingIstft(8, 2);
        assertThrows(IllegalArgumentException.class, () -> istft.processFrame(new float[2 * istft.nBins() - 2]));
        assertThrows(NullPointerException.class, () -> istft.processFrame(null));
    }

    private static void assertThrows(Class<? extends Exception> type, Runnable call) {
        try {
            call.run();
        } catch (Exception e) {
            assertTrue("expected " + type.getSimpleName() + ", got " + e, type.isInstance(e));
            return;
        }
        throw new AssertionError("the call did not throw " + type.getSimpleName());
    }

    /** Real part of the direct inverse DFT of a mirrored bin spectrum, in double precision. */
    private static double[] directIfftReal(float[] frame, int n) {
        float[] full = new float[2 * n];
        int nBins = n / 2 + 1;
        for (int j = 0; j < nBins; j++) {
            full[2 * j] = frame[2 * j];
            full[2 * j + 1] = frame[2 * j + 1];
        }
        for (int j = 1; j < nBins - 1; j++) {
            full[2 * (n - j)] = frame[2 * j];
            full[2 * (n - j) + 1] = -frame[2 * j + 1];
        }
        double[] out = new double[n];
        for (int t = 0; t < n; t++) {
            double acc = 0;
            for (int k = 0; k < n; k++) {
                double angle = 2.0 * Math.PI * k * t / n;
                acc += full[2 * k] * Math.cos(angle) - full[2 * k + 1] * Math.sin(angle);
            }
            out[t] = acc / n;
        }
        return out;
    }

    /** The periodic Hann window (mtmd-audio.cpp:27-33). */
    private static double[] hann(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / n));
        }
        return w;
    }

    private static float[] concat(float[] a, float[] b) {
        float[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}

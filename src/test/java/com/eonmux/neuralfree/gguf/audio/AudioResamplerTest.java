package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertThrows;

/** The resampler against spectral expectations: rate passthrough, frame counts and tone preservation. */
public class AudioResamplerTest {

    private static AudioClip sine(double freqHz, int seconds, int rate) {
        float[] s = new float[seconds * rate];
        for (int i = 0; i < s.length; i++) {
            s[i] = (float) Math.sin(2.0 * Math.PI * freqHz * i / rate);
        }
        return new AudioClip(s, rate);
    }

    /** Amplitude of tone {@code freq} in the clip, by Goertzel. */
    private static double goertzel(AudioClip clip, double freq) {
        double k = 2.0 * Math.PI * freq / clip.sampleRate();
        double coeff = 2.0 * Math.cos(k);
        double s1 = 0.0;
        double s2 = 0.0;
        for (int i = 0; i < clip.length(); i++) {
            double s0 = clip.samples()[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        return Math.sqrt(s1 * s1 + s2 * s2 - coeff * s1 * s2) / (clip.length() / 2.0);
    }

    @Test
    public void sameRateReturnsTheClip() {
        AudioClip clip = sine(440, 1, 16000);
        assertSame(clip, AudioResampler.resample(clip, 16000));
    }

    @Test
    public void frameCountIsCeilOfTheRatio() {
        float[] s = new float[1001];
        AudioClip clip = new AudioClip(s, 8000);
        assertEquals((1001 * 16000 + 7999) / 8000, AudioResampler.resample(clip, 16000).length());
        assertEquals((1001 * 11025 + 7999) / 8000, AudioResampler.resample(clip, 11025).length());
    }

    @Test
    public void downsampleKeepsTheTone() {
        AudioClip clip = sine(440, 2, 16000);
        AudioClip out = AudioResampler.resample(clip, 8000);
        assertEquals(8000, out.sampleRate());
        double amp = goertzel(out, 440);
        double leak = goertzel(out, 700);
        assertEquals(1.0, amp, 0.05);
        assertTrue("spectral leak " + leak + " too high", leak < 0.05);
    }

    @Test
    public void upsampleKeepsTheTone() {
        AudioClip clip = sine(440, 2, 8000);
        AudioClip out = AudioResampler.resample(clip, 16000);
        assertEquals(1.0, goertzel(out, 440), 0.05);
        assertTrue(goertzel(out, 3000) < 0.05);
    }

    @Test
    public void downsampleAttenuatesAboveTheTargetNyquist() {
        // 6 kHz tone at 16 kHz aliases to 2 kHz after resampling to 8 kHz if the cutoff were missing
        AudioClip clip = sine(6000, 1, 16000);
        AudioClip out = AudioResampler.resample(clip, 8000);
        double atTarget = goertzel(out, 2000);
        assertTrue("alias amplitude " + atTarget + " not attenuated", atTarget < 0.2);
    }

    @Test
    public void emptyAndInvalidArgumentsAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> AudioResampler.resample(sine(440, 1, 16000), 0));
        assertThrows(IllegalArgumentException.class, () -> new AudioClip(new float[1], 0));
        assertThrows(IllegalArgumentException.class, () -> new AudioClip(new float[] { Float.NaN }, 16000));
        assertEquals(0, AudioResampler.resample(AudioClip.silent(0, 16000), 8000).length());
    }
}

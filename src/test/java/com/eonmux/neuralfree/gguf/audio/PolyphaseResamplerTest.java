package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The polyphase resampler against the direct windowed-sinc form it tabulates (a copy kept here as the reference):
 * bit-identical where the rate ratio makes every position exact, and within one float ulp of the output elsewhere,
 * on random signals at the rates audio inputs arrive in.
 */
public class PolyphaseResamplerTest {

    @Test
    public void exactRatiosMatchTheDirectFormBitForBit() {
        int[][] pairs = {{48000, 16000}, {32000, 16000}, {8000, 16000}, {16000, 32000}, {24000, 16000},
                {64000, 16000}};
        for (int[] pair : pairs) {
            AudioClip clip = noise(pair[0], 3 * pair[0] + 17, pair[0]);
            float[] want = DirectResampler.resample(clip, pair[1]).samples();
            float[] got = AudioResampler.resample(clip, pair[1]).samples();
            assertArrayEquals(pair[0] + " -> " + pair[1], want, got, 0f);
        }
    }

    @Test
    public void otherRatiosStayWithinAnUlpOfTheDirectForm() {
        int[][] pairs = {{44100, 16000}, {22050, 16000}, {11025, 16000}, {44100, 24000}, {16000, 44100},
                {48000, 44100}, {12345, 16000}, {16000, 48000}};
        for (int[] pair : pairs) {
            AudioClip clip = noise(pair[0], 2 * pair[0] + 5, pair[0] + 1);
            float[] want = DirectResampler.resample(clip, pair[1]).samples();
            float[] got = AudioResampler.resample(clip, pair[1]).samples();
            assertEquals(want.length, got.length);
            for (int i = 0; i < want.length; i++) {
                float tolerance = Math.max(Math.ulp(want[i]), 1e-7f);
                assertTrue(pair[0] + " -> " + pair[1] + " frame " + i + ": " + want[i] + " vs " + got[i],
                        Math.abs(want[i] - got[i]) <= tolerance);
            }
        }
    }

    @Test
    public void shortClipsAndManyPhasesEvaluateDirectly() {
        // 16001 phases exceed the table, and a 3-frame clip is shorter than its table
        AudioClip wide = noise(44100, 4410, 3);
        assertArrayEquals(DirectResampler.resample(wide, 16001).samples(),
                AudioResampler.resample(wide, 16001).samples(), 0f);
        AudioClip tiny = noise(44100, 3, 4);
        assertArrayEquals(DirectResampler.resample(tiny, 16000).samples(),
                AudioResampler.resample(tiny, 16000).samples(), 0f);
    }

    private static AudioClip noise(int rate, int frames, long seed) {
        Random random = new Random(seed);
        float[] s = new float[frames];
        for (int i = 0; i < frames; i++) {
            s[i] = random.nextFloat() * 2f - 1f;
        }
        return new AudioClip(s, rate);
    }

    /** The direct windowed-sinc form: the kernel evaluated at every tap of every output frame. */
    private static final class DirectResampler {
        static AudioClip resample(AudioClip clip, int targetRate) {
            int nIn = clip.length();
            long nOut = ((long) nIn * targetRate + clip.sampleRate() - 1) / clip.sampleRate();
            float[] in = clip.samples();
            float[] out = new float[(int) nOut];
            double step = (double) clip.sampleRate() / targetRate;
            double halfWidth = AudioResampler.TAPS * Math.max(1.0, step);
            double cutoff = 0.5 / Math.max(1.0, step);
            for (int t = 0; t < out.length; t++) {
                double pos = t * step;
                int center = (int) Math.floor(pos);
                double frac = pos - center;
                double sum = 0.0;
                int span = (int) Math.ceil(halfWidth);
                for (int k = -span; k <= span; k++) {
                    int src = center + k;
                    if (src < 0 || src >= nIn) {
                        continue;
                    }
                    sum += in[src] * kernel(k - frac, halfWidth, cutoff);
                }
                out[t] = (float) sum;
            }
            return new AudioClip(out, targetRate);
        }

        private static double kernel(double d, double w, double fc) {
            if (d == 0.0) {
                return 2.0 * fc;
            }
            double sinc = Math.sin(2.0 * Math.PI * fc * d) / (Math.PI * d);
            double x = d / w;
            double hann = 0.5 * (1.0 + Math.cos(Math.PI * x));
            return sinc * hann;
        }
    }
}

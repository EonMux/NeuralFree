package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The log-mel frame loop (frame ranges in parallel, each mel filter folded over its non-zero bins) against the
 * serial loop with the dense fold it replaces (a copy kept here as the reference): bit-identical spectrograms for
 * the Whisper, conformer (Qwen/parakeet style) and Gemma 3n-style framings, including frames whose power overflows.
 */
public class LogMelFrameParityTest {

    @Test
    public void whisperFramingMatchesTheSerialDenseLoop() {
        MelFrameParams params = MelFrameParams.builder(128, 400, 160, 16000)
                .padding(MelFrameParams.Padding.WHISPER).naturalLog(false).hannWindowSize(400).build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(128).nFft(400).sampleRate(16000)
                .slaneyAreaNorm(true).build();
        check(params, filters, 1);
    }

    @Test
    public void conformerFramingMatchesTheSerialDenseLoop() {
        MelFrameParams params = MelFrameParams.builder(80, 512, 160, 16000)
                .padding(MelFrameParams.Padding.CENTER).preemph(0.97f).naturalLog(true).floorAdd(true)
                .normPerFeature(true).stdEpsAfterSqrt(true).hannWindowSize(400).hannPeriodic(false).build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(80).nFft(512).sampleRate(16000)
                .slaneyAreaNorm(true).build();
        check(params, filters, 2);
    }

    @Test
    public void htkMagnitudeFramingMatchesTheSerialDenseLoop() {
        MelFrameParams params = MelFrameParams.builder(128, 512, 160, 16000)
                .padding(MelFrameParams.Padding.NONE).magnitude(true).naturalLog(true).melFloor(1e-5f)
                .hannWindowSize(320).hannLeftAligned(true).build();
        MelFilterbank filters = new MelFilterbank.Builder().nMel(128).nFft(512).sampleRate(16000)
                .useHtk(true).fmin(125).fmax(7600).build();
        check(params, filters, 3);
    }

    private static void check(MelFrameParams params, MelFilterbank filters, long seed) {
        Random random = new Random(seed);
        float[] padded = new float[16000 * 7 + 123];
        for (int i = 0; i < padded.length; i++) {
            padded[i] = random.nextFloat() * 2f - 1f;
        }
        // a burst whose power overflows to infinity: the dense fold turns it into NaN through the zero taps
        for (int i = 50_000; i < 50_400; i++) {
            padded[i] = 3e19f;
        }
        int nLen = (padded.length - params.frameSize()) / params.hopLength() + 1;
        assertTrue("the clip spans several parallel ranges", nLen > 4 * LogMel.FRAMES_PER_RANGE);
        float[] hann = LogMel.hannWindow(params);
        float[] want = new float[params.nMel() * nLen];
        float[] got = new float[params.nMel() * nLen];
        serialDenseLoop(padded, params, filters, hann, want, nLen);
        LogMel.frameLoop(padded, params, filters, hann, got, nLen);
        boolean sawNaN = false;
        for (int i = 0; i < want.length; i++) {
            sawNaN |= Float.isNaN(want[i]);
            assertEquals("value " + i, Float.floatToRawIntBits(want[i]), Float.floatToRawIntBits(got[i]));
        }
        assertTrue("the overflowing burst reached the fold", sawNaN);
    }

    /** The serial frame loop with the dense nMel x nBins fold. */
    private static void serialDenseLoop(float[] padded, MelFrameParams params, MelFilterbank filters,
                                        float[] hann, float[] data, int nLen) {
        int frame = params.frameSize();
        int hop = params.hopLength();
        int win = params.hannWindowSize();
        int winPad = params.hannLeftAligned() ? 0 : (frame - win) / 2;
        int nBins = params.nFftBins();
        int nMel = params.nMel();
        Fft fft = new Fft(frame);
        float[] fftOut = new float[2 * frame];
        float[] power = new float[nBins];
        float[] real = new float[frame];
        for (int t = 0; t < nLen; t++) {
            int offset = t * hop;
            for (int j = 0; j < frame; j++) {
                real[j] = 0.0f;
            }
            for (int j = 0; j < win; j++) {
                int src = offset + winPad + j;
                if (src >= 0 && src < padded.length) {
                    real[winPad + j] = hann[j] * padded[src];
                }
            }
            fft.forward(real, fftOut);
            for (int k = 0; k < nBins; k++) {
                float re = fftOut[2 * k];
                float im = fftOut[2 * k + 1];
                float p = re * re + im * im;
                power[k] = params.magnitude() ? (float) Math.sqrt(p) : p;
            }
            for (int m = 0; m < nMel; m++) {
                double sum = 0.0;
                for (int k = 0; k < nBins; k++) {
                    sum += power[k] * filters.weight(m, k);
                }
                sum = params.floorAdd() ? sum + params.melFloor() : Math.max(sum, params.melFloor());
                data[m * nLen + t] = params.naturalLog() ? (float) Math.log(sum) : (float) Math.log10(sum);
            }
        }
    }
}

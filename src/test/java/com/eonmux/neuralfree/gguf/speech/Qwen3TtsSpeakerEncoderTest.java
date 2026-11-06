package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The qwen3tts speaker encoder against a transcription of llama.cpp's graph math
 * (tools/mtmd/models/qwen3tts-spkenc.cpp) over the same synthetic generator file. The reference keeps the C++
 * {@code [C, T]} layout and its explicit reflection padding, so a transposed block, a wrong chunk order or a
 * missing ReLU changes the result.
 */
public class Qwen3TtsSpeakerEncoderTest {

    private static final int SAMPLES = 4096;

    private static AudioClip referenceClip() {
        Random rng = new Random(99L);
        float[] samples = new float[SAMPLES];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.3 * Math.sin(2.0 * Math.PI * 220.0 * i / 24000.0)
                    + 0.1 * (rng.nextDouble() * 2.0 - 1.0));
        }
        return new AudioClip(samples, Qwen3TtsSpeakerEncoder.SAMPLE_RATE);
    }

    private static Qwen3TtsSpeakerEncoder open(Qwen3TtsClipFixture f) throws IOException {
        Qwen3TtsClip clip = Qwen3TtsClip.open(f.write().toString());
        Qwen3TtsSpeakerEncoder encoder = clip.speakerEncoder();
        assertNotNull(encoder);
        // the clip and its mmap stay open for the test's lifetime; the JVM releases them on exit
        return encoder;
    }

    @Test
    public void theFixtureCarriesTheSpeakerEncoder() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(7L, 6, true);
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(f.write().toString())) {
            assertNotNull(clip.speakerEncoder());
            assertEquals(Qwen3TtsSpeakerEncoder.PROJECTOR_TYPE, clip.speakerEncoder().name());
            assertEquals(6, clip.speakerEncoder().width());
            assertEquals(24000, clip.speakerEncoder().sampleRate());
        }
        Qwen3TtsClipFixture plain = new Qwen3TtsClipFixture(7L, 6);
        try (Qwen3TtsClip clip = Qwen3TtsClip.open(plain.write().toString())) {
            assertNull(clip.speakerEncoder());
        }
    }

    @Test
    public void theEmbeddingMatchesTheReferenceGraph() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(11L, 6, true);
        Qwen3TtsSpeakerEncoder encoder = open(f);
        AudioClip clip = referenceClip();
        float[][] mel = encoder.logMel(clip);
        float[] got = encoder.encode(clip)[0];
        float[] want = reference(f, mel);
        assertEquals(f.talkerWidth(), got.length);
        for (int i = 0; i < want.length; i++) {
            assertEquals("embedding value " + i, want[i], got[i], 2e-4f);
        }
    }

    @Test
    public void theMelFrontEndReflectsAndFloorsLikeTheReference() throws IOException {
        Qwen3TtsClipFixture f = new Qwen3TtsClipFixture(3L, 6, true);
        Qwen3TtsSpeakerEncoder encoder = open(f);
        float[][] mel = encoder.logMel(referenceClip());
        int pad = (Qwen3TtsSpeakerEncoder.N_FFT - Qwen3TtsSpeakerEncoder.HOP) / 2;
        int expectedFrames = (SAMPLES + 2 * pad - Qwen3TtsSpeakerEncoder.N_FFT) / Qwen3TtsSpeakerEncoder.HOP + 1;
        assertEquals(expectedFrames, mel.length);
        assertEquals(Qwen3TtsSpeakerEncoder.N_MEL, mel[0].length);
        for (float[] frame : mel) {
            for (float value : frame) {
                assertTrue("a mel value is finite", Float.isFinite(value));
            }
        }
        // the encoder refuses a clip shorter than the reflections
        AudioClip tooShort = AudioClip.silent(pad, Qwen3TtsSpeakerEncoder.SAMPLE_RATE);
        try {
            encoder.encode(tooShort);
            throw new AssertionError("a clip shorter than the reflections was accepted");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("reflections"));
        }
    }

    // ------------------------------------------------------------------ reference graph (C++ layout)

    /** The reference graph over frame-major mel, transposed to the C++ {@code [C, T]} layout. */
    private static float[] reference(Qwen3TtsClipFixture f, float[][] mel) {
        float[][] x = transpose(mel);
        x = relu(convSame(x, f.get("a.conv1d.0.weight"), f.get("a.conv1d.0.bias"), 5, 128, 8, 1));
        float[][][] outs = new float[3][][];
        int[] dilations = {2, 3, 4};
        for (int il = 0; il < 3; il++) {
            x = block(x, f, il + 1, dilations[il]);
            outs[il] = x;
        }
        float[][] cat = concat(outs);
        float[][] mfa = relu(convSame(cat, f.get("a.conv_out.weight"), f.get("a.conv_out.bias"), 1, 24, 8, 1));
        float[] stats = asp(mfa, f);
        float[][] statsMatrix = new float[stats.length][1];
        for (int i = 0; i < stats.length; i++) {
            statsMatrix[i][0] = stats[i];
        }
        float[][] projected = convSame(statsMatrix, f.get("mm.a.fc.weight"), f.get("mm.a.fc.bias"), 1, 16, 6, 1);
        return projected[0].clone();
    }

    private static float[][] block(float[][] x, Qwen3TtsClipFixture f, int bid, int dilation) {
        String p = "a.blk." + bid + ".";
        float[][] h = relu(convSame(x, f.get(p + "conv_pw1.weight"), f.get(p + "conv_pw1.bias"), 1, 8, 8, 1));
        h = res2net(h, f, p, dilation);
        h = relu(convSame(h, f.get(p + "conv_pw2.weight"), f.get(p + "conv_pw2.bias"), 1, 8, 8, 1));
        h = squeezeExcite(h, f, p);
        float[][] out = new float[h.length][h[0].length];
        for (int c = 0; c < h.length; c++) {
            for (int t = 0; t < h[c].length; t++) {
                out[c][t] = h[c][t] + x[c][t];
            }
        }
        return out;
    }

    private static float[][] res2net(float[][] x, Qwen3TtsClipFixture f, String p, int dilation) {
        int channels = x.length;
        int frames = x[0].length;
        int chunk = channels / 8;
        float[][] out = new float[channels][frames];
        float[][] previous = null;
        for (int i = 0; i < 8; i++) {
            float[][] part = new float[chunk][frames];
            for (int c = 0; c < chunk; c++) {
                System.arraycopy(x[i * chunk + c], 0, part[c], 0, frames);
            }
            if (i == 0) {
                for (int c = 0; c < chunk; c++) {
                    System.arraycopy(part[c], 0, out[c], 0, frames);
                }
                continue;
            }
            if (i >= 2) {
                for (int c = 0; c < chunk; c++) {
                    for (int t = 0; t < frames; t++) {
                        part[c][t] += previous[c][t];
                    }
                }
            }
            float[][] y = relu(convSame(part, f.get(p + "res2." + (i - 1) + ".weight"),
                    f.get(p + "res2." + (i - 1) + ".bias"), 3, chunk, chunk, dilation));
            for (int c = 0; c < chunk; c++) {
                System.arraycopy(y[c], 0, out[i * chunk + c], 0, frames);
            }
            previous = y;
        }
        return out;
    }

    private static float[][] squeezeExcite(float[][] x, Qwen3TtsClipFixture f, String p) {
        int channels = x.length;
        int frames = x[0].length;
        float[][] mean = new float[channels][1];
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            for (int t = 0; t < frames; t++) {
                sum += x[c][t];
            }
            mean[c][0] = (float) (sum / frames);
        }
        float[][] gate = relu(convSame(mean, f.get(p + "se_conv1.weight"), f.get(p + "se_conv1.bias"), 1, 8, 2, 1));
        gate = convSame(gate, f.get(p + "se_conv2.weight"), f.get(p + "se_conv2.bias"), 1, 2, 8, 1);
        float[][] out = new float[channels][frames];
        for (int c = 0; c < channels; c++) {
            float g = (float) (1.0 / (1.0 + Math.exp(-gate[c][0])));
            for (int t = 0; t < frames; t++) {
                out[c][t] = x[c][t] * g;
            }
        }
        return out;
    }

    private static float[] asp(float[][] x, Qwen3TtsClipFixture f) {
        int channels = x.length;
        int frames = x[0].length;
        float[][] mean = new float[channels][1];
        float[][] std = new float[channels][1];
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            for (int t = 0; t < frames; t++) {
                sum += x[c][t];
            }
            mean[c][0] = (float) (sum / frames);
            double var = 0.0;
            for (int t = 0; t < frames; t++) {
                double d = x[c][t] - mean[c][0];
                var += d * d;
            }
            std[c][0] = (float) Math.sqrt(Math.max(var / frames, 1e-12));
        }
        float[][] cat = new float[3 * channels][frames];
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < frames; t++) {
                cat[c][t] = x[c][t];
                cat[channels + c][t] = mean[c][0];
                cat[2 * channels + c][t] = std[c][0];
            }
        }
        float[][] hidden = relu(convSame(cat, f.get("a.asp_tdnn.weight"), f.get("a.asp_tdnn.bias"), 1, 24, 4, 1));
        for (int c = 0; c < hidden.length; c++) {
            for (int t = 0; t < frames; t++) {
                hidden[c][t] = (float) Math.tanh(hidden[c][t]);
            }
        }
        float[][] attn = convSame(hidden, f.get("a.asp_attn.weight"), f.get("a.asp_attn.bias"), 1, 4, 8, 1);
        for (int c = 0; c < channels; c++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < frames; t++) {
                max = Math.max(max, attn[c][t]);
            }
            double sum = 0.0;
            for (int t = 0; t < frames; t++) {
                sum += Math.exp(attn[c][t] - max);
            }
            for (int t = 0; t < frames; t++) {
                attn[c][t] = (float) (Math.exp(attn[c][t] - max) / sum);
            }
        }
        double[] wMean = new double[channels];
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < frames; t++) {
                wMean[c] += (double) attn[c][t] * x[c][t];
            }
        }
        float[] out = new float[2 * channels];
        for (int c = 0; c < channels; c++) {
            double var = 0.0;
            for (int t = 0; t < frames; t++) {
                double d = x[c][t] - wMean[c];
                var += (double) attn[c][t] * d * d;
            }
            out[c] = (float) wMean[c];
            out[channels + c] = (float) Math.sqrt(Math.max(var, 1e-12));
        }
        return out;
    }

    /** {@code conv1d_same}: reflection padding on both ends, then the valid convolution. */
    private static float[][] convSame(float[][] x, float[] w, float[] b, int k, int in, int out, int dilation) {
        int frames = x[0].length;
        int pad = ((k - 1) * dilation) / 2;
        int span = (k - 1) * dilation + 1;
        float[][] padded = new float[in][frames + 2 * pad];
        for (int c = 0; c < in; c++) {
            for (int t = 0; t < frames; t++) {
                padded[c][pad + t] = x[c][t];
            }
            for (int i = 1; i <= pad; i++) {
                padded[c][pad - i] = x[c][i];
                padded[c][pad + frames - 1 + i] = x[c][frames - 1 - i];
            }
        }
        int outFrames = padded[0].length - span + 1;
        float[][] y = new float[out][outFrames];
        for (int oc = 0; oc < out; oc++) {
            for (int t = 0; t < outFrames; t++) {
                float acc = b[oc];
                for (int ic = 0; ic < in; ic++) {
                    int wBase = k * (ic + in * oc);
                    for (int j = 0; j < k; j++) {
                        acc += w[j + wBase] * padded[ic][t + j * dilation];
                    }
                }
                y[oc][t] = acc;
            }
        }
        return y;
    }

    private static float[][] relu(float[][] x) {
        for (float[] row : x) {
            for (int i = 0; i < row.length; i++) {
                row[i] = Math.max(0f, row[i]);
            }
        }
        return x;
    }

    private static float[][] transpose(float[][] x) {
        float[][] out = new float[x[0].length][x.length];
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < x[t].length; c++) {
                out[c][t] = x[t][c];
            }
        }
        return out;
    }

    private static float[][] concat(float[][][] parts) {
        int channels = 0;
        for (float[][] part : parts) {
            channels += part.length;
        }
        int frames = parts[0][0].length;
        float[][] out = new float[channels][frames];
        int at = 0;
        for (float[][] part : parts) {
            for (float[] row : part) {
                System.arraycopy(row, 0, out[at++], 0, frames);
            }
        }
        return out;
    }
}

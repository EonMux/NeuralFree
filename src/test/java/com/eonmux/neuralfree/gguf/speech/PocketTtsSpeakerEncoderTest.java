package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

/**
 * The pockettts speaker encoder against a transcription of llama.cpp's graph math
 * (tools/mtmd/models/pockettts-spkenc.cpp and pockettts-seanet.cpp) over the same synthetic generator file: the
 * causal SEANet encoder, the sliding-window transformer, the strided downsample and the speaker projection.
 */
public class PocketTtsSpeakerEncoderTest {

    /** Two model frames: two 1920-sample blocks. */
    private static final int SAMPLES = 3840;

    private static AudioClip referenceClip() {
        Random rng = new Random(1234L);
        float[] samples = new float[SAMPLES];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.25 * Math.sin(2.0 * Math.PI * 180.0 * i / 24000.0)
                    + 0.05 * (rng.nextDouble() * 2.0 - 1.0));
        }
        return new AudioClip(samples, PocketTtsSpeakerEncoder.SAMPLE_RATE);
    }

    @Test
    public void theFixtureCarriesTheSpeakerEncoder() throws IOException {
        PocketTtsFixtures f = new PocketTtsFixtures(5L, "english", true);
        try (PocketTtsClip clip = PocketTtsClip.open(f.write().toString())) {
            assertNotNull(clip.speakerEncoder());
            assertEquals(PocketTtsSpeakerEncoder.PROJECTOR_TYPE, clip.speakerEncoder().name());
            assertEquals(PocketTtsFixtures.EMB, clip.speakerEncoder().width());
        }
        PocketTtsFixtures plain = new PocketTtsFixtures(5L);
        try (PocketTtsClip clip = PocketTtsClip.open(plain.write().toString())) {
            assertNull(clip.speakerEncoder());
        }
    }

    @Test
    public void theRowsMatchTheReferenceGraph() throws IOException {
        PocketTtsFixtures f = new PocketTtsFixtures(17L, "english", true);
        try (PocketTtsClip clip = PocketTtsClip.open(f.write().toString())) {
            PocketTtsSpeakerEncoder encoder = clip.speakerEncoder();
            AudioClip audio = referenceClip();
            float[][] got = encoder.encode(audio);
            float[][] want = reference(f, audio.samples());
            assertEquals(want.length, got.length);
            assertEquals(PocketTtsFixtures.EMB, got[0].length);
            for (int t = 0; t < want.length; t++) {
                for (int c = 0; c < want[t].length; c++) {
                    assertEquals("row " + t + " value " + c, want[t][c], got[t][c], 2e-4f);
                }
            }
        }
    }

    @Test
    public void oneRowIsEmittedPerModelFrame() throws IOException {
        PocketTtsFixtures f = new PocketTtsFixtures(2L, "english", true);
        try (PocketTtsClip clip = PocketTtsClip.open(f.write().toString())) {
            PocketTtsSpeakerEncoder encoder = clip.speakerEncoder();
            assertEquals(1, encoder.encode(AudioClip.silent(1, 24000)).length);
            assertEquals(1, encoder.encode(AudioClip.silent(1920, 24000)).length);
            assertEquals(2, encoder.encode(AudioClip.silent(1921, 24000)).length);
            assertEquals(2, encoder.encode(AudioClip.silent(3840, 24000)).length);
            assertEquals(3, encoder.encode(AudioClip.silent(3841, 24000)).length);
            // a reference at another rate is resampled first
            assertEquals(1, encoder.encode(AudioClip.silent(960, 12000)).length);
        }
    }

    @Test
    public void longReferencesAreTruncatedToTheReferenceLimit() throws IOException {
        PocketTtsFixtures f = new PocketTtsFixtures(2L, "english", true);
        try (PocketTtsClip clip = PocketTtsClip.open(f.write().toString())) {
            int seconds = PocketTtsSpeakerEncoder.MAX_SECONDS + 1;
            float[][] rows = clip.speakerEncoder().encode(AudioClip.silent(seconds * 24000, 24000));
            assertEquals((int) (PocketTtsSpeakerEncoder.MAX_SECONDS * PocketTtsSpeakerEncoder.FRAME_RATE),
                    rows.length);
        }
    }

    // ------------------------------------------------------------------ reference graph

    static float[][] reference(PocketTtsFixtures f, float[] samples) {
        int nPadded = ((samples.length + 1919) / 1920) * 1920;
        float[] padded = new float[nPadded];
        System.arraycopy(samples, 0, padded, 0, samples.length);
        float[][] x = new float[nPadded][1];
        for (int i = 0; i < nPadded; i++) {
            x[i][0] = padded[i];
        }
        x = conv(x, f.get("a.seanet.conv_in.weight"), f.get("a.seanet.conv_in.bias"), 3, 1, 4, 1, 1, false);
        int[] strides = {4, 5, 6};
        for (int stage = 0; stage < 3; stage++) {
            String p = "a.seanet.blk." + stage + ".";
            float[][] h = elu(x);
            h = conv(h, f.get(p + "res_conv1.weight"), f.get(p + "res_conv1.bias"), 3, 4, 4, 1, 1, false);
            h = elu(h);
            h = conv(h, f.get(p + "res_conv2.weight"), f.get(p + "res_conv2.bias"), 1, 4, 4, 1, 1, false);
            for (int t = 0; t < x.length; t++) {
                for (int c = 0; c < 4; c++) {
                    h[t][c] += x[t][c];
                }
            }
            x = elu(h);
            x = conv(x, f.get(p + "scale_conv.weight"), f.get(p + "scale_conv.bias"), 2 * strides[stage], 4, 4,
                    strides[stage], 1, false);
        }
        x = elu(x);
        x = conv(x, f.get("a.seanet.conv_out.weight"), f.get("a.seanet.conv_out.bias"), 3, 4, 4, 1, 1, false);
        x = transformer(x, f);
        x = conv(x, f.get("a.downsample.conv.weight"), null, 2 * PocketTtsSpeakerEncoder.DOWNSAMPLE, 4,
                PocketTtsFixtures.LATENT, PocketTtsSpeakerEncoder.DOWNSAMPLE, 1, true);
        float[][] out = new float[x.length][PocketTtsFixtures.EMB];
        float[] proj = f.get("a.speaker_proj.weight");
        for (int t = 0; t < x.length; t++) {
            for (int o = 0; o < out[t].length; o++) {
                double sum = 0.0;
                for (int i = 0; i < PocketTtsFixtures.LATENT; i++) {
                    sum += proj[i + PocketTtsFixtures.LATENT * o] * x[t][i];
                }
                out[t][o] = (float) sum;
            }
        }
        return out;
    }

    static float[][] transformer(float[][] x, PocketTtsFixtures f) {
        int n = x.length;
        int width = 4;
        String p = "a.blk.0.";
        float[][] cur = new float[n][width];
        for (int t = 0; t < n; t++) {
            layerNorm(x[t], f.get(p + "ln1.weight"), f.get(p + "ln1.bias"), cur[t]);
        }
        float[] q = new float[n * width];
        float[] k = new float[n * width];
        float[] v = new float[n * width];
        matmul(f.get(p + "attn_q.weight"), cur, q);
        matmul(f.get(p + "attn_k.weight"), cur, k);
        matmul(f.get(p + "attn_v.weight"), cur, v);
        for (int t = 0; t < n; t++) {
            rope(q, t * width, t);
            rope(k, t * width, t);
        }
        float[] attn = new float[n * width];
        for (int t = 0; t < n; t++) {
            double max = Double.NEGATIVE_INFINITY;
            int first = Math.max(0, t - PocketTtsSpeakerEncoder.CONTEXT + 1);
            float[] scores = new float[n];
            for (int s = first; s <= t; s++) {
                double dot = 0.0;
                for (int d = 0; d < width; d++) {
                    dot += q[t * width + d] * k[s * width + d];
                }
                scores[s] = (float) (dot / Math.sqrt(width));
                max = Math.max(max, scores[s]);
            }
            double sum = 0.0;
            for (int s = first; s <= t; s++) {
                sum += Math.exp(scores[s] - max);
            }
            for (int d = 0; d < width; d++) {
                double acc = 0.0;
                for (int s = first; s <= t; s++) {
                    acc += Math.exp(scores[s] - max) / sum * v[s * width + d];
                }
                attn[t * width + d] = (float) acc;
            }
        }
        float[] projected = new float[n * width];
        matmul(f.get(p + "attn_out.weight"), reshape(attn, n, width), projected);
        float[][] out = new float[n][width];
        float[] ls1 = f.get(p + "ls1.weight");
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[t][c] = x[t][c] + projected[t * width + c] * ls1[c];
            }
        }
        float[][] normed = new float[n][width];
        for (int t = 0; t < n; t++) {
            layerNorm(out[t], f.get(p + "ln2.weight"), f.get(p + "ln2.bias"), normed[t]);
        }
        float[] up = new float[n * PocketTtsFixtures.TFM_FF];
        matmul(f.get(p + "ffn_up.weight"), normed, up);
        for (int i = 0; i < up.length; i++) {
            double inner = 0.7978845608028654 * (up[i] + 0.044715 * up[i] * up[i] * up[i]);
            up[i] = (float) (0.5 * up[i] * (1.0 + Math.tanh(inner)));
        }
        float[] down = new float[n * width];
        matmul(f.get(p + "ffn_down.weight"), reshape(up, n, PocketTtsFixtures.TFM_FF), down);
        float[] ls2 = f.get(p + "ls2.weight");
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < width; c++) {
                out[t][c] += down[t * width + c] * ls2[c];
            }
        }
        return out;
    }

    private static void matmul(float[] w, float[][] x, float[] y) {
        int in = x[0].length;
        int out = y.length / x.length;
        for (int t = 0; t < x.length; t++) {
            for (int o = 0; o < out; o++) {
                double sum = 0.0;
                for (int i = 0; i < in; i++) {
                    sum += w[i + in * o] * x[t][i];
                }
                y[t * out + o] = (float) sum;
            }
        }
    }

    private static void layerNorm(float[] x, float[] weight, float[] bias, float[] out) {
        double mean = 0.0;
        for (float value : x) {
            mean += value;
        }
        mean /= x.length;
        double var = 0.0;
        for (float value : x) {
            double d = value - mean;
            var += d * d;
        }
        var /= x.length;
        double scale = 1.0 / Math.sqrt(var + 1e-5);
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) ((x[i] - mean) * scale * weight[i] + bias[i]);
        }
    }

    private static void rope(float[] x, int off, int pos) {
        for (int i = 0; i < 2; i++) {
            double angle = pos * Math.pow(10000.0, -2.0 * i / 4);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float x0 = x[off + 2 * i];
            float x1 = x[off + 2 * i + 1];
            x[off + 2 * i] = x0 * cos - x1 * sin;
            x[off + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    /**
     * A causal convolution: {@code k - stride} frames of left context (zeros, or a repeat of the first frame) and
     * zero tail padding, emitting {@code ceil(T / stride)} rows.
     */
    private static float[][] conv(float[][] x, float[] w, float[] b, int k, int in, int out, int stride, int dilation,
                                  boolean replicateFirst) {
        int frames = x.length;
        int span = (k - 1) * dilation + 1;
        int left = Math.max(0, span - stride);
        int outFrames = (frames + stride - 1) / stride;
        float[][] y = new float[outFrames][out];
        for (int t = 0; t < outFrames; t++) {
            for (int oc = 0; oc < out; oc++) {
                float acc = b == null ? 0f : b[oc];
                for (int j = 0; j < k; j++) {
                    int src = t * stride + j * dilation - left;
                    for (int ic = 0; ic < in; ic++) {
                        float value;
                        if (src < 0) {
                            value = replicateFirst ? x[0][ic] : 0f;
                        } else if (src >= frames) {
                            value = 0f;
                        } else {
                            value = x[src][ic];
                        }
                        acc += w[j + k * (ic + in * oc)] * value;
                    }
                }
                y[t][oc] = acc;
            }
        }
        return y;
    }

    private static float[][] elu(float[][] x) {
        float[][] y = new float[x.length][];
        for (int t = 0; t < x.length; t++) {
            y[t] = new float[x[t].length];
            for (int c = 0; c < x[t].length; c++) {
                y[t][c] = x[t][c] > 0f ? x[t][c] : (float) Math.expm1(x[t][c]);
            }
        }
        return y;
    }

    private static float[][] reshape(float[] flat, int rows, int width) {
        float[][] out = new float[rows][width];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * width, out[i], 0, width);
        }
        return out;
    }
}

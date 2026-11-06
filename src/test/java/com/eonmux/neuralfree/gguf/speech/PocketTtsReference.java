package com.eonmux.neuralfree.gguf.speech;

/**
 * The reference math of the llama.cpp pockettts graphs in double precision: the flow network of
 * tools/mtmd/models/pockettts-gen.cpp {@code flow_forward} / {@code time_embed}, and the GEN_WAV graph plus
 * pockettts-seanet.cpp's causal decoder. The tests compare {@link PocketTtsFlow} and
 * {@link PocketTtsMimiSession} against this independent transcription of the reference.
 */
final class PocketTtsReference {

    private static final int LATENT = PocketTtsFixtures.LATENT;
    private static final int FLOW = PocketTtsFixtures.FLOW;
    private static final int WIDTH = PocketTtsFixtures.WIDTH;
    private static final int TFM_HEADS = PocketTtsFixtures.TFM_HEADS;
    private static final int DHEAD = PocketTtsFixtures.TFM_DHEAD;
    private static final int FF = PocketTtsFixtures.TFM_FF;
    private static final int LAYERS = PocketTtsFixtures.TFM_LAYERS;
    private static final int PREFIX = PocketTtsMimiDecoder.CONTEXT - 1;
    private static final int CONTEXT = PocketTtsMimiDecoder.CONTEXT;
    private static final int KV = TFM_HEADS * DHEAD;
    private static final double GELU_C1 = 0.7978845608028654;
    private static final double GELU_C2 = 0.044715;

    private final PocketTtsFixtures f;
    private int tfmPos;
    private final double[][] kCache = new double[LAYERS][PREFIX * KV];
    private final double[][] vCache = new double[LAYERS][PREFIX * KV];
    private final double[][] upState = new double[PocketTtsFixtures.UP_K - PocketTtsMimiDecoder.DOWNSAMPLE][WIDTH];
    private final double[][] decInState = new double[PocketTtsFixtures.CONV_K - 1][WIDTH];
    private final double[][][] decUpStates = new double[3][][];
    private final double[][][] decResStates = new double[3][][];
    private final double[][] decOutState = new double[PocketTtsFixtures.CONV_K - 1][WIDTH];

    PocketTtsReference(PocketTtsFixtures fixture) {
        this.f = fixture;
        for (int s = 0; s < 3; s++) {
            int stride = PocketTtsFixtures.STAGE_STRIDES[s];
            decUpStates[s] = new double[2 * stride - stride][WIDTH];
            decResStates[s] = new double[PocketTtsFixtures.RES_K - 1][WIDTH];
        }
    }

    // ================================================================== flow

    double eos(double[] cond) {
        float[] w = f.get("a.gen.out_eos.weight");
        float[] b = f.get("a.gen.out_eos.bias");
        double acc = b[0];
        for (int i = 0; i < cond.length; i++) {
            acc += (double) w[i] * cond[i];
        }
        return acc;
    }

    double[] project(double[] latent) {
        float[] w = f.get("a.gen.input_linear.weight");
        double[] out = new double[PocketTtsFixtures.EMB];
        for (int o = 0; o < out.length; o++) {
            for (int i = 0; i < LATENT; i++) {
                out[o] += (double) w[i + LATENT * o] * latent[i];
            }
        }
        return out;
    }

    /** One integration step from {@code noise} (llama.cpp GEN_CODE). */
    double[] flow(double[] cond, double[] noise) {
        double[] v = velocity(cond, noise);
        double[] x = new double[LATENT];
        for (int i = 0; i < LATENT; i++) {
            x[i] = noise[i] + v[i];
        }
        return x;
    }

    private double[] velocity(double[] cond, double[] x) {
        double[] cur = project("a.gen.flow.input_proj", x);
        double[] ts = add(timeEmbed(0, 0.0), timeEmbed(1, 1.0));
        for (int i = 0; i < FLOW; i++) {
            ts[i] /= 2.0;
        }
        double[] c = project("a.gen.flow.cond_embd", cond);
        double[] y = new double[FLOW];
        for (int i = 0; i < FLOW; i++) {
            y[i] = ts[i] + c[i];
        }
        for (int b = 0; b < PocketTtsFixtures.FLOW_BLOCKS; b++) {
            String p = "a.gen.flow.blk." + b + ".";
            double[] mod = project(p + "ada", silu(y));
            double[] h = layerNorm(cur, f.get(p + "norm.weight"), f.get(p + "norm.bias"), 1e-6);
            for (int i = 0; i < FLOW; i++) {
                h[i] = h[i] * (1.0 + mod[FLOW + i]) + mod[i];
            }
            h = silu(project(p + "up", h));
            h = project(p + "down", h);
            for (int i = 0; i < FLOW; i++) {
                cur[i] += mod[2 * FLOW + i] * h[i];
            }
        }
        double[] mod = project("a.gen.flow.final.ada", silu(y));
        cur = layerNorm(cur, null, null, 1e-6);
        for (int i = 0; i < FLOW; i++) {
            cur[i] = cur[i] * (1.0 + mod[FLOW + i]) + mod[i];
        }
        return project("a.gen.flow.final.proj", cur);
    }

    private double[] timeEmbed(int index, double t) {
        String p = "a.gen.flow.time." + index + ".";
        float[] freqs = f.get(p + "freqs");
        double[] emb = new double[2 * freqs.length];
        for (int i = 0; i < freqs.length; i++) {
            emb[i] = Math.cos((double) freqs[i] * t);
            emb[freqs.length + i] = Math.sin((double) freqs[i] * t);
        }
        double[] cur = silu(project(p + "up", emb));
        cur = project(p + "down", cur);
        double mean = 0.0;
        for (double v : cur) {
            mean += v;
        }
        mean /= cur.length;
        double var = 0.0;
        for (double v : cur) {
            double d = v - mean;
            var += d * d;
        }
        var /= cur.length;
        double denom = Math.sqrt(var * cur.length / (double) (cur.length - 1) + 1e-5);
        float[] norm = f.get(p + "norm");
        for (int i = 0; i < cur.length; i++) {
            cur[i] = cur[i] / denom * norm[i];
        }
        return cur;
    }

    private double[] project(String base, double[] x) {
        float[] w = f.get(base + ".weight");
        float[] b = f.get(base + ".bias");
        int in = (int) f.ne(base + ".weight")[0];
        int out = (int) f.ne(base + ".weight")[1];
        double[] y = new double[out];
        for (int o = 0; o < out; o++) {
            double acc = b[o];
            for (int i = 0; i < in; i++) {
                acc += (double) w[i + in * o] * x[i];
            }
            y[o] = acc;
        }
        return y;
    }

    // ================================================================== mimi decoder

    double[] decode(float[][] frames) {
        int n = frames.length;
        double[][] x = quantize(frames);
        x = depthwiseTranspose(x, f.get("a.gen.wav.upsample.weight"), PocketTtsFixtures.UP_K,
                PocketTtsMimiDecoder.DOWNSAMPLE, WIDTH, upState);
        x = transformer(x);
        x = seanet(x);
        double[] out = new double[n * PocketTtsFixtures.HOP];
        for (int t = 0; t < out.length; t++) {
            out[t] = Math.max(-1.0, Math.min(1.0, x[t][0]));
        }
        return out;
    }

    private double[][] quantize(float[][] frames) {
        float[] std = f.get("a.gen.emb_std");
        float[] mean = f.get("a.gen.emb_mean");
        float[] quant = f.get("a.gen.wav.quant_out.weight");
        double[][] out = new double[frames.length][WIDTH];
        for (int t = 0; t < frames.length; t++) {
            for (int o = 0; o < WIDTH; o++) {
                double acc = 0.0;
                for (int i = 0; i < LATENT; i++) {
                    double v = (double) frames[t][i] * std[i] + mean[i];
                    acc += (double) quant[i + LATENT * o] * v;
                }
                out[t][o] = acc;
            }
        }
        return out;
    }

    private double[][] transformer(double[][] input) {
        double[][] x = input;
        int base = tfmPos;
        for (int l = 0; l < LAYERS; l++) {
            x = layer(x, l, base);
        }
        tfmPos += input.length;
        return x;
    }

    private double[][] layer(double[][] x, int il, int base) {
        int nPos = x.length;
        String p = "a.gen.wav.tfm.blk." + il + ".";
        double[][] normed = new double[nPos][WIDTH];
        for (int t = 0; t < nPos; t++) {
            normed[t] = layerNorm(x[t], f.get(p + "ln1.weight"), f.get(p + "ln1.bias"), PocketTtsFixtures.EPS);
        }
        double[][] q = matVec(f.get(p + "attn_q.weight"), normed, WIDTH, WIDTH);
        double[][] k = matVec(f.get(p + "attn_k.weight"), normed, WIDTH, WIDTH);
        double[][] v = matVec(f.get(p + "attn_v.weight"), normed, WIDTH, WIDTH);
        for (int t = 0; t < nPos; t++) {
            rope(q[t], base + t);
            rope(k[t], base + t);
        }
        int total = PREFIX + nPos;
        double[][] keys = new double[total][KV];
        double[][] vals = new double[total][KV];
        for (int j = 0; j < PREFIX; j++) {
            System.arraycopy(kCache[il], j * KV, keys[j], 0, KV);
            System.arraycopy(vCache[il], j * KV, vals[j], 0, KV);
        }
        for (int t = 0; t < nPos; t++) {
            System.arraycopy(k[t], 0, keys[PREFIX + t], 0, KV);
            System.arraycopy(v[t], 0, vals[PREFIX + t], 0, KV);
        }
        double[][] attn = attention(q, keys, vals, base);
        double[][] projected = matVec(f.get(p + "attn_out.weight"), attn, WIDTH, WIDTH);
        float[] ls1 = f.get(p + "ls1.weight");
        double[][] out = new double[nPos][WIDTH];
        for (int t = 0; t < nPos; t++) {
            for (int c = 0; c < WIDTH; c++) {
                out[t][c] = x[t][c] + projected[t][c] * ls1[c];
            }
        }
        double[][] normed2 = new double[nPos][WIDTH];
        for (int t = 0; t < nPos; t++) {
            normed2[t] = layerNorm(out[t], f.get(p + "ln2.weight"), f.get(p + "ln2.bias"), PocketTtsFixtures.EPS);
        }
        double[][] up = matVec(f.get(p + "ffn_up.weight"), normed2, WIDTH, FF);
        for (int t = 0; t < nPos; t++) {
            for (int i = 0; i < FF; i++) {
                up[t][i] = gelu(up[t][i]);
            }
        }
        double[][] down = matVec(f.get(p + "ffn_down.weight"), up, FF, WIDTH);
        float[] ls2 = f.get(p + "ls2.weight");
        for (int t = 0; t < nPos; t++) {
            for (int c = 0; c < WIDTH; c++) {
                out[t][c] += down[t][c] * ls2[c];
            }
        }
        for (int j = 0; j < PREFIX; j++) {
            System.arraycopy(keys[nPos + j], 0, kCache[il], j * KV, KV);
            System.arraycopy(vals[nPos + j], 0, vCache[il], j * KV, KV);
        }
        return out;
    }

    private double[][] attention(double[][] q, double[][] keys, double[][] values, int base) {
        int nPos = q.length;
        int total = keys.length;
        double scale = 1.0 / Math.sqrt(DHEAD);
        double[][] out = new double[nPos][KV];
        double[] scores = new double[total];
        for (int t = 0; t < nPos; t++) {
            for (int h = 0; h < TFM_HEADS; h++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < total; j++) {
                    int keyPos = j - PREFIX + base;
                    int diff = PREFIX + t - j;
                    if (keyPos < 0 || diff < 0 || diff >= CONTEXT) {
                        scores[j] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double dot = 0.0;
                    for (int d = 0; d < DHEAD; d++) {
                        dot += q[t][h * DHEAD + d] * keys[j][h * DHEAD + d];
                    }
                    scores[j] = dot * scale;
                    max = Math.max(max, scores[j]);
                }
                double sum = 0.0;
                for (int j = 0; j < total; j++) {
                    if (scores[j] == Double.NEGATIVE_INFINITY) {
                        scores[j] = 0.0;
                    } else {
                        scores[j] = Math.exp(scores[j] - max);
                        sum += scores[j];
                    }
                }
                for (int d = 0; d < DHEAD; d++) {
                    double acc = 0.0;
                    for (int j = 0; j < total; j++) {
                        acc += scores[j] * values[j][h * DHEAD + d];
                    }
                    out[t][h * DHEAD + d] = acc / sum;
                }
            }
        }
        return out;
    }

    private double[][] seanet(double[][] input) {
        double[][] x = causalConv(input, f.get("a.gen.wav.seanet.conv_in.weight"),
                f.get("a.gen.wav.seanet.conv_in.bias"), PocketTtsFixtures.CONV_K, WIDTH, WIDTH, decInState);
        for (int s = 0; s < 3; s++) {
            String p = "a.gen.wav.seanet.blk." + s + ".";
            int stride = PocketTtsFixtures.STAGE_STRIDES[s];
            x = elu(x);
            x = transpose(x, f.get(p + "scale_conv.weight"), f.get(p + "scale_conv.bias"), 2 * stride, stride,
                    WIDTH, WIDTH, decUpStates[s]);
            double[][] h = elu(x);
            h = causalConv(h, f.get(p + "res_conv1.weight"), f.get(p + "res_conv1.bias"),
                    PocketTtsFixtures.RES_K, WIDTH, WIDTH, decResStates[s]);
            h = elu(h);
            h = causalConv(h, f.get(p + "res_conv2.weight"), f.get(p + "res_conv2.bias"), 1, WIDTH, WIDTH, null);
            for (int t = 0; t < x.length; t++) {
                for (int c = 0; c < WIDTH; c++) {
                    x[t][c] += h[t][c];
                }
            }
        }
        x = elu(x);
        return causalConv(x, f.get("a.gen.wav.seanet.conv_out.weight"),
                f.get("a.gen.wav.seanet.conv_out.bias"), PocketTtsFixtures.CONV_K, WIDTH, 1, decOutState);
    }

    private static double[][] causalConv(double[][] x, float[] w, float[] b, int k, int in, int out,
                                         double[][] state) {
        int t = x.length;
        int pad = k - 1;
        double[][] full = new double[pad + t][in];
        if (pad > 0) {
            for (int i = 0; i < pad; i++) {
                System.arraycopy(state[i], 0, full[i], 0, in);
            }
        }
        for (int i = 0; i < t; i++) {
            System.arraycopy(x[i], 0, full[pad + i], 0, in);
        }
        double[][] y = new double[t][out];
        for (int i = 0; i < t; i++) {
            for (int oc = 0; oc < out; oc++) {
                double acc = b[oc];
                for (int ic = 0; ic < in; ic++) {
                    for (int j = 0; j < k; j++) {
                        acc += (double) w[j + k * (ic + in * oc)] * full[i + j][ic];
                    }
                }
                y[i][oc] = acc;
            }
        }
        if (state != null) {
            for (int i = 0; i < pad; i++) {
                System.arraycopy(full[t + i], 0, state[i], 0, in);
            }
        }
        return y;
    }

    private static double[][] transpose(double[][] x, float[] w, float[] b, int k, int stride, int in, int out,
                                        double[][] state) {
        int t = x.length;
        int trim = k - stride;
        int fullLen = (t - 1) * stride + k;
        double[][] full = new double[fullLen][out];
        for (int i = 0; i < t; i++) {
            for (int ic = 0; ic < in; ic++) {
                for (int j = 0; j < k; j++) {
                    int row = i * stride + j;
                    for (int oc = 0; oc < out; oc++) {
                        full[row][oc] += (double) w[j + k * (oc + out * ic)] * x[i][ic];
                    }
                }
            }
        }
        return overlap(full, trim, t * stride, out, state, b);
    }

    private static double[][] depthwiseTranspose(double[][] x, float[] w, int k, int stride, int channels,
                                                 double[][] state) {
        int t = x.length;
        int trim = k - stride;
        int fullLen = (t - 1) * stride + k;
        double[][] full = new double[fullLen][channels];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < k; j++) {
                int row = i * stride + j;
                for (int c = 0; c < channels; c++) {
                    full[row][c] += (double) w[j + k * c] * x[i][c];
                }
            }
        }
        return overlap(full, trim, t * stride, channels, state, null);
    }

    private static double[][] overlap(double[][] full, int trim, int emit, int channels, double[][] state,
                                      float[] bias) {
        double[][] y = new double[emit][channels];
        for (int i = 0; i < trim; i++) {
            for (int c = 0; c < channels; c++) {
                y[i][c] = full[i][c] + state[i][c] + (bias == null ? 0.0 : bias[c]);
            }
        }
        for (int i = trim; i < emit; i++) {
            for (int c = 0; c < channels; c++) {
                y[i][c] = full[i][c] + (bias == null ? 0.0 : bias[c]);
            }
        }
        for (int i = 0; i < trim; i++) {
            System.arraycopy(full[emit + i], 0, state[i], 0, channels);
        }
        return y;
    }

    private static void rope(double[] x, int pos) {
        int half = DHEAD / 2;
        for (int i = 0; i < half; i++) {
            double angle = pos * Math.pow(PocketTtsMimiDecoder.ROPE_THETA, -2.0 * i / DHEAD);
            double cos = Math.cos(angle);
            double sin = Math.sin(angle);
            double x0 = x[2 * i];
            double x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos - x1 * sin;
            x[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    private static double[][] matVec(float[] w, double[][] x, int in, int out) {
        double[][] y = new double[x.length][out];
        for (int t = 0; t < x.length; t++) {
            for (int o = 0; o < out; o++) {
                double acc = 0.0;
                for (int i = 0; i < in; i++) {
                    acc += (double) w[i + in * o] * x[t][i];
                }
                y[t][o] = acc;
            }
        }
        return y;
    }

    private static double[] layerNorm(double[] x, float[] weight, float[] bias, double eps) {
        int n = x.length;
        double mean = 0.0;
        for (double v : x) {
            mean += v;
        }
        mean /= n;
        double var = 0.0;
        for (double v : x) {
            double d = v - mean;
            var += d * d;
        }
        var /= n;
        double scale = 1.0 / Math.sqrt(var + eps);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double v = (x[i] - mean) * scale;
            out[i] = weight == null ? v : v * weight[i] + bias[i];
        }
        return out;
    }

    private static double[][] elu(double[][] x) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double[] row = new double[x[t].length];
            for (int c = 0; c < row.length; c++) {
                row[c] = x[t][c] > 0.0 ? x[t][c] : Math.expm1(x[t][c]);
            }
            y[t] = row;
        }
        return y;
    }

    private static double[] silu(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] / (1.0 + Math.exp(-x[i]));
        }
        return y;
    }

    private static double[] add(double[] a, double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    private static double gelu(double x) {
        return 0.5 * x * (1.0 + Math.tanh(GELU_C1 * (x + GELU_C2 * x * x * x)));
    }
}

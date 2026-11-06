package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Shared support of the audio tower tests: {@code clip.audio.*} metadata, F32 GGUF writing, binding the encoder
 * over a fixture, and the double-precision ggml primitives the per-family reference transcriptions of llama.cpp
 * tools/mtmd/models are built from. Rows are token-major {@code double[n][width]}. Package-private test support.
 */
final class AudioTowerSupport {

    private AudioTowerSupport() {
    }

    /** The shared audio keys of one projector; the family keys are added by the caller. */
    static Map<String, Object> meta(String projector, int nMel, int hidden, int ff, int layers, int heads,
                                    int projectionDim, float eps) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_audio_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.audio.num_mel_bins", nMel);
        m.put("clip.audio.embedding_length", hidden);
        m.put("clip.audio.feed_forward_length", ff);
        m.put("clip.audio.block_count", layers);
        m.put("clip.audio.projection_dim", projectionDim);
        m.put("clip.audio.attention.head_count", heads);
        m.put("clip.audio.attention.layer_norm_epsilon", eps);
        return m;
    }

    static ClipHparams hparams(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.AUDIO);
    }

    static CpuAudioEncoder encoder(MmprojFixture f, Map<String, Object> meta) {
        return CpuAudioEncoder.bind(ClipAudioModel.from(hparams(meta), f.catalog()), CpuMatVec.PLAIN, f::get,
                f::ref);
    }

    /** Writes the metadata (ints, int arrays, floats, strings, booleans) and every tensor as F32. */
    static GgufTestWriter writeF32(GgufTestWriter w, Map<String, Object> meta, MmprojFixture f) {
        for (Map.Entry<String, Object> e : meta.entrySet()) {
            Object v = e.getValue();
            if (e.getKey().equals("general.architecture")) {
                w.architecture((String) v);
            } else if (v instanceof String) {
                w.str(e.getKey(), (String) v);
            } else if (v instanceof Boolean) {
                w.bool(e.getKey(), (Boolean) v);
            } else if (v instanceof Float) {
                w.f32(e.getKey(), (Float) v);
            } else if (v instanceof int[]) {
                w.i32Array(e.getKey(), (int[]) v);
            } else {
                w.u32(e.getKey(), ((Number) v).longValue());
            }
        }
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), f.get(e.getKey()));
        }
        return w;
    }

    /** A deterministic mel-major chunk of values in {@code [-1, 1]}. */
    static MelChunk chunk(int nMel, int nLen, long seed) {
        Random r = new Random(seed);
        float[] v = new float[nMel * nLen];
        for (int i = 0; i < v.length; i++) {
            v[i] = r.nextFloat() * 2f - 1f;
        }
        return MelChunk.of(nMel, nLen, v);
    }

    /** Asserts every value within {@code tol} relative to {@code max(1, |want|)}. */
    static void assertRows(double[][] want, float[][] got, double tol) {
        CpuAudioEncoderTest.assertRows(want, got, tol);
        assertEquals(want.length, got.length);
    }

    // ------------------------------------------------------------------ double-precision ggml primitives

    /** The mel-major chunk as token-major rows {@code [nLen][nMel]}. */
    static double[][] frames(MelChunk c) {
        double[][] x = new double[c.nLen()][c.nMel()];
        for (int m = 0; m < c.nMel(); m++) {
            for (int t = 0; t < c.nLen(); t++) {
                x[t][m] = c.data()[m * c.nLen() + t];
            }
        }
        return x;
    }

    /** {@code w x + b} per row; {@code w} is {@code [rows][x width]} in ggml order, {@code b} may be null. */
    static double[][] dense(double[][] x, float[] w, float[] b) {
        int cols = x[0].length;
        int rows = w.length / cols;
        double[][] y = new double[x.length][rows];
        for (int t = 0; t < x.length; t++) {
            for (int r = 0; r < rows; r++) {
                double s = b != null ? b[r] : 0;
                for (int c = 0; c < cols; c++) {
                    s += (double) w[r * cols + c] * x[t][c];
                }
                y[t][r] = s;
            }
        }
        return y;
    }

    /** {@code ggml_norm} then the optional weight and bias, per row. */
    static double[][] layerNorm(double[][] x, float[] w, float[] b, double eps) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double mean = 0;
            for (double v : x[t]) {
                mean += v;
            }
            mean /= x[t].length;
            double var = 0;
            for (double v : x[t]) {
                var += (v - mean) * (v - mean);
            }
            double inv = 1.0 / Math.sqrt(var / x[t].length + eps);
            y[t] = new double[x[t].length];
            for (int c = 0; c < x[t].length; c++) {
                y[t][c] = (x[t][c] - mean) * inv * (w != null ? w[c] : 1) + (b != null ? b[c] : 0);
            }
        }
        return y;
    }

    /** {@code ggml_rms_norm} then the optional weight and bias, per row. */
    static double[][] rmsNorm(double[][] x, float[] w, float[] b, double eps) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double ss = 0;
            for (double v : x[t]) {
                ss += v * v;
            }
            double inv = 1.0 / Math.sqrt(ss / x[t].length + eps);
            y[t] = new double[x[t].length];
            for (int c = 0; c < x[t].length; c++) {
                y[t][c] = x[t][c] * inv * (w != null ? w[c] : 1) + (b != null ? b[c] : 0);
            }
        }
        return y;
    }

    static double[][] add(double[][] a, double[][] b) {
        double[][] y = new double[a.length][];
        for (int t = 0; t < a.length; t++) {
            y[t] = new double[a[t].length];
            for (int c = 0; c < a[t].length; c++) {
                y[t][c] = a[t][c] + b[t][c];
            }
        }
        return y;
    }

    static double[][] scale(double[][] a, double s) {
        double[][] y = new double[a.length][];
        for (int t = 0; t < a.length; t++) {
            y[t] = new double[a[t].length];
            for (int c = 0; c < a[t].length; c++) {
                y[t][c] = a[t][c] * s;
            }
        }
        return y;
    }

    /** Copy of {@code x} with {@code f} applied to every value. */
    static double[][] map(double[][] x, java.util.function.DoubleUnaryOperator f) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            y[t] = new double[x[t].length];
            for (int c = 0; c < x[t].length; c++) {
                y[t][c] = f.applyAsDouble(x[t][c]);
            }
        }
        return y;
    }

    static double geluErf(double v) {
        return 0.5 * v * (1 + MmprojFixture.Reference.erf(v / Math.sqrt(2)));
    }

    static double silu(double v) {
        return v / (1 + Math.exp(-v));
    }

    static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    static double reluSqr(double v) {
        double r = Math.max(v, 0);
        return r * r;
    }

    /** {@code build_ffn} without a gate: up(+b), the activation, down(+b). */
    static double[][] ffn(double[][] x, float[] up, float[] upB, java.util.function.DoubleUnaryOperator act,
                          float[] down, float[] downB) {
        return dense(map(dense(x, up, upB), act), down, downB);
    }

    /**
     * Softmax attention over {@code [n][heads * dh]} rows at {@code scale}; {@code allowed} decides which key each
     * query row sees.
     */
    static double[][] attend(double[][] q, double[][] k, double[][] v, int heads, double scale,
                             java.util.function.BiPredicate<Integer, Integer> allowed) {
        int n = q.length;
        int nk = k.length;
        int dh = q[0].length / heads;
        double[][] out = new double[n][q[0].length];
        for (int h = 0; h < heads; h++) {
            for (int i = 0; i < n; i++) {
                double[] s = new double[nk];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < nk; j++) {
                    if (!allowed.test(i, j)) {
                        s[j] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double d = 0;
                    for (int c = 0; c < dh; c++) {
                        d += q[i][h * dh + c] * k[j][h * dh + c];
                    }
                    s[j] = d * scale;
                    max = Math.max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < nk; j++) {
                    s[j] = s[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(s[j] - max);
                    sum += s[j];
                }
                for (int j = 0; j < nk; j++) {
                    for (int c = 0; c < dh; c++) {
                        out[i][h * dh + c] += s[j] / sum * v[j][h * dh + c];
                    }
                }
            }
        }
        return out;
    }

    /**
     * NEOX rotary embedding of the first {@code nRot} dims of every {@code dh}-wide head: dim {@code i} pairs with
     * {@code i + nRot / 2} at angle {@code pos * theta^(-2i / nRot)}.
     */
    static double[][] ropeNeox(double[][] x, int heads, int nRot, int[] pos, double theta) {
        int dh = x[0].length / heads;
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            y[t] = x[t].clone();
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < nRot / 2; i++) {
                    double ang = pos[t] * Math.pow(theta, -2.0 * i / nRot);
                    double a = x[t][h * dh + i];
                    double b = x[t][h * dh + i + nRot / 2];
                    y[t][h * dh + i] = a * Math.cos(ang) - b * Math.sin(ang);
                    y[t][h * dh + i + nRot / 2] = a * Math.sin(ang) + b * Math.cos(ang);
                }
            }
        }
        return y;
    }

    /**
     * {@code ggml_conv_2d(kernel [kw, kh, cIn, cOut], img [w, h, cIn])} at stride {@code s} and padding {@code p}
     * over {@code img[c][y][x]}, plus the bias per output channel: {@code [cOut][oh][ow]}.
     */
    static double[][][] conv2d(double[][][] img, float[] kernel, float[] bias, int kw, int kh, int s, int p) {
        int cIn = img.length;
        int h = img[0].length;
        int w = img[0][0].length;
        int cOut = kernel.length / (kw * kh * cIn);
        int ow = (w + 2 * p - kw) / s + 1;
        int oh = (h + 2 * p - kh) / s + 1;
        double[][][] out = new double[cOut][oh][ow];
        for (int co = 0; co < cOut; co++) {
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    double acc = bias != null ? bias[co] : 0;
                    for (int ci = 0; ci < cIn; ci++) {
                        for (int ky = 0; ky < kh; ky++) {
                            int y = oy * s + ky - p;
                            for (int kx = 0; kx < kw; kx++) {
                                int x = ox * s + kx - p;
                                if (y >= 0 && y < h && x >= 0 && x < w) {
                                    acc += kernel[kx + kw * (ky + kh * (ci + cIn * co))] * img[ci][y][x];
                                }
                            }
                        }
                    }
                    out[co][oy][ox] = acc;
                }
            }
        }
        return out;
    }

    /** Depthwise {@code ggml_conv_2d_dw}: channel {@code c} with its own {@code [kw, kh]} plane, plus the bias. */
    static double[][][] depthwise2d(double[][][] img, float[] kernel, float[] bias, int kw, int kh, int s, int p) {
        int c = img.length;
        int h = img[0].length;
        int w = img[0][0].length;
        int ow = (w + 2 * p - kw) / s + 1;
        int oh = (h + 2 * p - kh) / s + 1;
        double[][][] out = new double[c][oh][ow];
        for (int ch = 0; ch < c; ch++) {
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    double acc = bias != null ? bias[ch] : 0;
                    for (int ky = 0; ky < kh; ky++) {
                        int y = oy * s + ky - p;
                        for (int kx = 0; kx < kw; kx++) {
                            int x = ox * s + kx - p;
                            if (y >= 0 && y < h && x >= 0 && x < w) {
                                acc += kernel[kx + kw * (ky + kh * ch)] * img[ch][y][x];
                            }
                        }
                    }
                    out[ch][oy][ox] = acc;
                }
            }
        }
        return out;
    }

    /**
     * Depthwise conv along time over {@code [n][c]} rows: {@code out[t][ch] = sum_k x[t + k - padLeft][ch] *
     * kernel[k + K * ch]}, zero outside, {@code n + padLeft + padRight - K + 1} rows.
     */
    static double[][] depthwise1d(double[][] x, float[] kernel, int kLen, int padLeft, int padRight) {
        int n = x.length;
        int c = x[0].length;
        int nOut = n + padLeft + padRight - kLen + 1;
        double[][] y = new double[nOut][c];
        for (int t = 0; t < nOut; t++) {
            for (int ch = 0; ch < c; ch++) {
                double acc = 0;
                for (int k = 0; k < kLen; k++) {
                    int src = t + k - padLeft;
                    if (src >= 0 && src < n) {
                        acc += x[src][ch] * kernel[k + kLen * ch];
                    }
                }
                y[t][ch] = acc;
            }
        }
        return y;
    }

    static float[][] toFloat(double[][] x) {
        return MmprojFixture.Reference.toFloat(x);
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.conv2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise1d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.layerNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.scale;

/**
 * A double-precision transcription of the NeMo conformer pieces of llama.cpp tools/mtmd/models/conformer.cpp and
 * parakeet.cpp, for the conformer-family tower tests: the subsampling stem, the relative-position attention with
 * its transformer-XL shift spelled out as the graph's pad / reshape / view, and the block. Also synthesizes the
 * block tensors of a fixture. Package-private test support.
 */
final class ConformerReference {

    private ConformerReference() {
    }

    /** Fills the stem tensors: conv channels {@code c0}, {@code c3}, {@code c6} and the output projection. */
    static void stem(MmprojFixture f, int nMel, int hidden, int c0, int c3, int c6) {
        f.weight("a.conv1d.0.weight", 3, 3, 1, c0);
        f.weight("a.conv1d.0.bias", 1, 1, c0);
        f.weight("a.conv1d.2.weight", 3, 3, 1, c0);
        f.weight("a.conv1d.2.bias", 1, 1, c0);
        f.weight("a.conv1d.3.weight", 1, 1, c0, c3);
        f.weight("a.conv1d.3.bias", 1, 1, c3);
        f.weight("a.conv1d.5.weight", 3, 3, 1, c3);
        f.weight("a.conv1d.5.bias", 1, 1, c3);
        f.weight("a.conv1d.6.weight", 1, 1, c3, c6);
        f.weight("a.conv1d.6.bias", 1, 1, c6);
        int w = nMel;
        for (int i = 0; i < 3; i++) {
            w = (w - 1) / 2 + 1;
        }
        f.weight("a.pre_encode.out.weight", (long) w * c6, hidden);
        f.weight("a.pre_encode.out.bias", hidden);
    }

    /**
     * Fills block {@code il}: with {@code lfm} the conformer.cpp biases (ln biases, first-FFN biases, q/k/v/out,
     * conv and second-FFN biases), otherwise the parakeet batch-norm statistics and biased block norms.
     */
    static void block(MmprojFixture f, int il, boolean lfm, int e, int heads, int ff, int posWidth, int kernel) {
        String p = "a.blk." + il + ".";
        f.norm(p + "ffn_norm.weight", p + "ffn_norm.bias", e);
        f.norm(p + "ln1.weight", p + "ln1.bias", e);
        f.norm(p + "norm_conv.weight", p + "norm_conv.bias", e);
        f.norm(p + "conv_norm.weight", p + "conv_norm.bias", e);
        f.norm(p + "ffn_norm_1.weight", p + "ffn_norm_1.bias", e);
        f.norm(p + "ln2.weight", p + "ln2.bias", e);
        for (String s : new String[]{"", "_1"}) {
            f.weight(p + "ffn_up" + s + ".weight", e, ff);
            f.weight(p + "ffn_down" + s + ".weight", ff, e);
        }
        for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
            f.weight(p + t + ".weight", e, e);
        }
        f.weight(p + "pos_bias_u", e / heads, heads);
        f.weight(p + "pos_bias_v", e / heads, heads);
        f.weight(p + "linear_pos.weight", posWidth, e);
        f.weight(p + "conv_pw1.weight", e, 2L * e);
        f.weight(p + "conv_dw.weight", kernel, e);
        f.weight(p + "conv_pw2.weight", e, e);
        if (lfm) {
            for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out", "conv_dw", "conv_pw2",
                    "ffn_down", "ffn_down_1"}) {
                f.weight(p + t + ".bias", e);
            }
            f.weight(p + "conv_pw1.bias", 2L * e);
            f.weight(p + "ffn_up.bias", ff);
            f.weight(p + "ffn_up_1.bias", ff);
        } else {
            f.put(p + "conv_norm_mean", -0.2f, 0.2f, e);
            f.put(p + "conv_norm_var", 0.5f, 1.5f, e);
        }
    }

    /** The stem over the transposed mel ({@code [x = mel, y = frame]}): {@code [frames / 8][hidden]} rows. */
    static double[][] stem(MmprojFixture f, MelChunk chunk) {
        double[][][] img = new double[1][chunk.nLen()][chunk.nMel()];
        for (int t = 0; t < chunk.nLen(); t++) {
            for (int m = 0; m < chunk.nMel(); m++) {
                img[0][t][m] = chunk.value(m, t);
            }
        }
        img = relu(conv2d(img, f.get("a.conv1d.0.weight"), f.get("a.conv1d.0.bias"), 3, 3, 2, 1));
        img = depthwise2d(img, f.get("a.conv1d.2.weight"), f.get("a.conv1d.2.bias"), 3, 3, 2, 1);
        img = relu(conv2d(img, f.get("a.conv1d.3.weight"), f.get("a.conv1d.3.bias"), 1, 1, 1, 0));
        img = depthwise2d(img, f.get("a.conv1d.5.weight"), f.get("a.conv1d.5.bias"), 3, 3, 2, 1);
        img = relu(conv2d(img, f.get("a.conv1d.6.weight"), f.get("a.conv1d.6.bias"), 1, 1, 1, 0));
        int c = img.length;
        int h = img[0].length;
        int w = img[0][0].length;
        double[][] rows = new double[h][w * c];
        for (int t = 0; t < h; t++) {
            for (int ch = 0; ch < c; ch++) {
                for (int x = 0; x < w; x++) {
                    rows[t][x + w * ch] = img[ch][t][x];
                }
            }
        }
        return dense(rows, f.get("a.pre_encode.out.weight"), f.get("a.pre_encode.out.bias"));
    }

    private static double[][][] relu(double[][][] img) {
        for (double[][] plane : img) {
            for (double[] row : plane) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = Math.max(row[i], 0);
                }
            }
        }
        return img;
    }

    /** Interleaved sin (even) / cos (odd) rows of distance {@code (rows - 1) / 2 - j} times {@code freq[i]}. */
    static double[][] posEmb(int rows, int width, double[] freq) {
        double[][] p = new double[rows][width];
        int center = (rows - 1) / 2;
        for (int j = 0; j < rows; j++) {
            for (int i = 0; i < width / 2; i++) {
                p[j][2 * i] = Math.sin((center - j) * freq[i]);
                p[j][2 * i + 1] = Math.cos((center - j) * freq[i]);
            }
        }
        return p;
    }

    /**
     * One block of conformer.cpp ({@code lfm}) or parakeet.cpp at norm epsilon {@code eps}: full attention when
     * {@code pos} holds {@code 2n - 1} rows, parakeet's local attention otherwise.
     */
    static double[][] block(MmprojFixture f, int il, boolean lfm, double[][] x, double[][] pos, int heads,
                            int kernel, double eps) {
        String p = "a.blk." + il + ".";
        double[][] h = layerNorm(x, f.get(p + "ffn_norm.weight"), f.get(p + "ffn_norm.bias"), eps);
        x = add(x, scale(ffn(h, f.get(p + "ffn_up.weight"), bias(f, p + "ffn_up.bias", lfm),
                AudioTowerSupport::silu, f.get(p + "ffn_down.weight"), bias(f, p + "ffn_down.bias", lfm)), 0.5));
        h = layerNorm(x, f.get(p + "ln1.weight"), bias(f, p + "ln1.bias", true), eps);
        x = add(x, attention(f, p, lfm, h, pos, heads));
        h = layerNorm(x, f.get(p + "norm_conv.weight"), f.get(p + "norm_conv.bias"), eps);
        x = add(x, convModule(f, p, lfm, h, kernel));
        h = layerNorm(x, f.get(p + "ffn_norm_1.weight"), f.get(p + "ffn_norm_1.bias"), eps);
        x = add(x, scale(ffn(h, f.get(p + "ffn_up_1.weight"), bias(f, p + "ffn_up_1.bias", lfm),
                AudioTowerSupport::silu, f.get(p + "ffn_down_1.weight"), bias(f, p + "ffn_down_1.bias", lfm)),
                0.5));
        return layerNorm(x, f.get(p + "ln2.weight"), bias(f, p + "ln2.bias", true), eps);
    }

    private static float[] bias(MmprojFixture f, String name, boolean read) {
        return read && f.has(name) ? f.get(name) : null;
    }

    private static double[][] attention(MmprojFixture f, String p, boolean lfm, double[][] h, double[][] posEmb,
                                        int heads) {
        int n = h.length;
        int e = h[0].length;
        int dh = e / heads;
        double[][] q = dense(h, f.get(p + "attn_q.weight"), bias(f, p + "attn_q.bias", lfm));
        double[][] k = dense(h, f.get(p + "attn_k.weight"), bias(f, p + "attn_k.bias", lfm));
        double[][] v = dense(h, f.get(p + "attn_v.weight"), bias(f, p + "attn_v.bias", lfm));
        double[][] pos = dense(posEmb, f.get(p + "linear_pos.weight"), null);
        float[] u = f.get(p + "pos_bias_u");
        float[] vb = f.get(p + "pos_bias_v");
        int posLen = pos.length;
        double[][] out = new double[n][e];
        for (int hd = 0; hd < heads; hd++) {
            double[] padded = posLen == 2 * n - 1 ? shiftInput(q, pos, vb, hd, dh) : null;
            for (int i = 0; i < n; i++) {
                double[] sc = padded != null ? fullRow(q, k, padded, posLen, u, i, hd, dh)
                        : localRow(q, k, pos, u, vb, i, hd, dh);
                double max = Double.NEGATIVE_INFINITY;
                for (double s : sc) {
                    max = Math.max(max, s);
                }
                double sum = 0;
                for (int t = 0; t < n; t++) {
                    sc[t] = Math.exp(sc[t] - max);
                    sum += sc[t];
                }
                for (int t = 0; t < n; t++) {
                    for (int d = 0; d < dh; d++) {
                        out[i][hd * dh + d] += sc[t] / sum * v[t][hd * dh + d];
                    }
                }
            }
        }
        return dense(out, f.get(p + "attn_out.weight"), bias(f, p + "attn_out.bias", lfm));
    }

    private static double dot(double[] a, float[] bias, double[] b, int off, int dh) {
        double s = 0;
        for (int d = 0; d < dh; d++) {
            s += (a[off + d] + bias[off + d]) * b[off + d];
        }
        return s;
    }

    /**
     * The rel-shift input of one head: matrix_bd {@code [pos_len][q_len]} of {@code (q + v) . p} with one zero
     * column padded in front of each query row.
     */
    private static double[] shiftInput(double[][] q, double[][] pos, float[] vb, int hd, int dh) {
        int n = q.length;
        int posLen = pos.length;
        double[] padded = new double[n * (posLen + 1)];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < posLen; j++) {
                padded[i * (posLen + 1) + 1 + j] = dot(q[i], vb, pos[j], hd * dh, dh);
            }
        }
        return padded;
    }

    /**
     * Full-attention scores of query {@code i}: the padded input reshaped to {@code [q_len, pos_len + 1]}, viewed
     * from offset {@code q_len} and made contiguous as {@code [pos_len, q_len]}, its first {@code q_len} columns
     * added to {@code (q + u) . k}.
     */
    private static double[] fullRow(double[][] q, double[][] k, double[] padded, int posLen, float[] u, int i,
                                    int hd, int dh) {
        int n = q.length;
        double[] sc = new double[n];
        for (int t = 0; t < n; t++) {
            sc[t] = (dot(q[i], u, k[t], hd * dh, dh) + padded[n + t + posLen * i]) / Math.sqrt(dh);
        }
        return sc;
    }

    /**
     * Local-attention scores of query {@code i} (parakeet.cpp): window entry {@code j} is key {@code i - W + j}
     * with position row {@code j} ({@code W = (pos_len - 1) / 2}); keys outside the sequence are masked.
     */
    private static double[] localRow(double[][] q, double[][] k, double[][] pos, float[] u, float[] vb, int i,
                                     int hd, int dh) {
        int n = q.length;
        int reach = (pos.length - 1) / 2;
        double[] sc = new double[n];
        java.util.Arrays.fill(sc, Double.NEGATIVE_INFINITY);
        for (int j = 0; j < pos.length; j++) {
            int key = i - reach + j;
            if (key >= 0 && key < n) {
                sc[key] = (dot(q[i], u, k[key], hd * dh, dh) + dot(q[i], vb, pos[j], hd * dh, dh)) / Math.sqrt(dh);
            }
        }
        return sc;
    }

    private static double[][] convModule(MmprojFixture f, String p, boolean lfm, double[][] h, int kernel) {
        int e = h[0].length;
        double[][] x = dense(h, f.get(p + "conv_pw1.weight"), bias(f, p + "conv_pw1.bias", lfm));
        double[][] g = new double[x.length][e];
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < e; c++) {
                g[t][c] = x[t][c] * AudioTowerSupport.sigmoid(x[t][e + c]);
            }
        }
        double[][] c = depthwise1d(g, f.get(p + "conv_dw.weight"), kernel, (kernel - 1) / 2, (kernel - 1) / 2);
        float[] dwB = bias(f, p + "conv_dw.bias", lfm);
        float[] w = f.get(p + "conv_norm.weight");
        float[] b = f.get(p + "conv_norm.bias");
        for (double[] row : c) {
            for (int ch = 0; ch < e; ch++) {
                double val = row[ch] + (dwB == null ? 0 : dwB[ch]);
                if (!lfm) {
                    val = (val - f.get(p + "conv_norm_mean")[ch]) / Math.sqrt(f.get(p + "conv_norm_var")[ch]);
                }
                row[ch] = AudioTowerSupport.silu(val * w[ch] + b[ch]);
            }
        }
        return dense(c, f.get(p + "conv_pw2.weight"), bias(f, p + "conv_pw2.bias", lfm));
    }
}

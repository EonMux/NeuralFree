package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.conv2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise1d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The gemma4a tower against a double-precision transcription of llama.cpp tools/mtmd/models/gemma4a.cpp
 * ({@code clip_graph_gemma4a}): the two-conv stem with its channel norm, a block with every optional operator
 * (attention pre/post norms, per-dim Q/K scales, the relative-position term, FFN post-norms, the conv module with
 * its post-norm, the block norm) and ClippableLinear bounds, a block without the conv module, the chunked
 * 12-row causal attention with its softcap, the output projection and the embedder; the row count and refusals.
 */
public class Gemma4aTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 16;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int TEXT = 6;
    static final int[] CH = {3, 4};

    static Map<String, Object> meta() {
        return AudioTowerSupport.meta("gemma4a", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, 1e-5f);
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(91);
        f.weight("a.conv1d.0.weight", 3, 3, 1, CH[0]);
        f.weight("a.conv1d.0.bias", 1, 1, CH[0]);
        f.put("a.conv1d.0.norm.weight", 0.8f, 1.2f, CH[0]);
        f.weight("a.conv1d.1.weight", 3, 3, CH[0], CH[1]);
        f.weight("a.input_projection.weight", 4L * CH[1], HIDDEN);
        f.weight("a.input_projection.bias", HIDDEN);
        int dh = HIDDEN / HEADS;
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            boolean full = il == 0;
            f.put(p + "ffn_norm.weight", 0.8f, 1.2f, HIDDEN);
            f.put(p + "ffn_norm_1.weight", 0.8f, 1.2f, HIDDEN);
            for (String s : new String[]{"", "_1"}) {
                f.weight(p + "ffn_up" + s + ".weight", HIDDEN, FF);
                f.weight(p + "ffn_down" + s + ".weight", FF, HIDDEN);
            }
            for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.weight(p + t + ".weight", HIDDEN, HIDDEN);
            }
            f.weight(p + "conv_pw1.weight", HIDDEN, 2L * HIDDEN);
            f.weight(p + "conv_dw.weight", Gemma4aTower.CONV_KERNEL, HIDDEN);
            f.weight(p + "conv_pw2.weight", HIDDEN, HIDDEN);
            // a bias the graph never adds: it must not change the output
            f.weight(p + "ffn_up.bias", FF);
            if (full) {
                f.weight(p + "attn_out.bias", HIDDEN);
                f.put(p + "attn_pre_norm.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "per_dim_scale.weight", 0.5f, 1.5f, dh);
                f.put(p + "per_dim_k_scale.weight", 0.5f, 1.5f, dh);
                f.weight(p + "attn_k_rel.weight", HIDDEN, HIDDEN);
                f.put(p + "attn_post_norm.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "ffn_post_norm.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "conv_norm.weight", 0.8f, 1.2f, HIDDEN);
                f.weight(p + "conv_dw.bias", HIDDEN);
                f.put(p + "norm_conv.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "ffn_post_norm_1.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "ln2.weight", 0.8f, 1.2f, HIDDEN);
                f.put(p + "ffn_up.input_max", 0.4f, 0.4f, 1);
                f.put(p + "ffn_up.input_min", -0.4f, -0.4f, 1);
                f.put(p + "ffn_up.output_max", 0.3f, 0.3f, 1);
                f.put(p + "attn_q.output_min", -0.2f, -0.2f, 1);
            } else {
                f.put(p + "ln1.weight", 0.8f, 1.2f, HIDDEN);
            }
        }
        f.weight("a.pre_encode.out.weight", HIDDEN, HIDDEN);
        f.weight("a.pre_encode.out.bias", HIDDEN);
        f.put("mm.a.soft_emb_norm.weight", 0.8f, 1.2f, HIDDEN);
        f.weight("mm.a.input_projection.weight", HIDDEN, TEXT);
        return f;
    }

    /** {@code w x} with the ClippableLinear bounds of {@code base} and the bias when {@code biased}. */
    private static double[][] lin(MmprojFixture f, String base, double[][] x, boolean biased) {
        double[] b = {bound(f, base, "input_min", -Double.MAX_VALUE), bound(f, base, "input_max", Double.MAX_VALUE),
                bound(f, base, "output_min", -Double.MAX_VALUE), bound(f, base, "output_max", Double.MAX_VALUE)};
        double[][] in = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            in[t] = x[t].clone();
            for (int i = 0; i < in[t].length; i++) {
                in[t][i] = Math.max(Math.min(in[t][i], b[1]), b[0]);
            }
        }
        double[][] y = AudioTowerSupport.dense(in, f.get(base + ".weight"), null);
        float[] bias = biased && f.has(base + ".bias") ? f.get(base + ".bias") : null;
        for (double[] row : y) {
            for (int i = 0; i < row.length; i++) {
                row[i] = Math.max(Math.min(row[i], b[3]), b[2]) + (bias == null ? 0 : bias[i]);
            }
        }
        return y;
    }

    private static double bound(MmprojFixture f, String base, String which, double dflt) {
        return f.has(base + "." + which) ? f.get(base + "." + which)[0] : dflt;
    }

    private static float[] opt(MmprojFixture f, String name) {
        return f.has(name) ? f.get(name) : null;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][] x = stem(f, chunk);
        for (int il = 0; il < LAYERS; il++) {
            x = block(f, "a.blk." + il + ".", x);
        }
        x = lin(f, "a.pre_encode.out", x, true);
        x = rmsNorm(x, f.get("mm.a.soft_emb_norm.weight"), null, 1e-6);
        return lin(f, "mm.a.input_projection", x, false);
    }

    private static double[][] stem(MmprojFixture f, MelChunk chunk) {
        double[][][] img = new double[1][chunk.nLen()][chunk.nMel()];
        for (int t = 0; t < chunk.nLen(); t++) {
            for (int m = 0; m < chunk.nMel(); m++) {
                img[0][t][m] = chunk.value(m, t);
            }
        }
        for (int i = 0; i < 2; i++) {
            img = conv2d(img, f.get("a.conv1d." + i + ".weight"), opt(f, "a.conv1d." + i + ".bias"), 3, 3, 2, 1);
            float[] norm = opt(f, "a.conv1d." + i + ".norm.weight");
            int c = img.length;
            for (int y = 0; y < img[0].length; y++) {
                for (int xx = 0; xx < img[0][0].length; xx++) {
                    double[][] px = new double[1][c];
                    for (int ch = 0; ch < c; ch++) {
                        px[0][ch] = img[ch][y][xx];
                    }
                    if (norm != null) {
                        px = AudioTowerSupport.layerNorm(px, norm, null, 1e-6);
                    }
                    for (int ch = 0; ch < c; ch++) {
                        img[ch][y][xx] = Math.max(px[0][ch], 0);
                    }
                }
            }
        }
        int c = img.length;
        int h = img[0].length;
        int w = img[0][0].length;
        double[][] rows = new double[h][w * c];
        for (int t = 0; t < h; t++) {
            for (int fr = 0; fr < w; fr++) {
                for (int ch = 0; ch < c; ch++) {
                    rows[t][ch + c * fr] = img[ch][t][fr];
                }
            }
        }
        return lin(f, "a.input_projection", rows, true);
    }

    private static double[][] block(MmprojFixture f, String p, double[][] x) {
        x = addScaled(x, ffn(f, p, "", x, "ffn_norm", "ffn_post_norm"), 0.5);
        x = addScaled(x, attention(f, p, x), 1);
        if (f.has(p + "conv_norm.weight")) {
            x = addScaled(x, convModule(f, p, x), 1);
        }
        x = addScaled(x, ffn(f, p, "_1", x, "ffn_norm_1", "ffn_post_norm_1"), 0.5);
        return f.has(p + "ln2.weight") ? rmsNorm(x, f.get(p + "ln2.weight"), null, 1e-6) : x;
    }

    private static double[][] addScaled(double[][] a, double[][] b, double s) {
        return AudioTowerSupport.add(a, AudioTowerSupport.scale(b, s));
    }

    private static double[][] ffn(MmprojFixture f, String p, String s, double[][] x, String norm, String post) {
        double[][] h = lin(f, p + "ffn_up" + s, rmsNorm(x, f.get(p + norm + ".weight"), null, 1e-6), false);
        h = lin(f, p + "ffn_down" + s, AudioTowerSupport.map(h, AudioTowerSupport::silu), false);
        return f.has(p + post + ".weight") ? rmsNorm(h, f.get(p + post + ".weight"), null, 1e-6) : h;
    }

    private static double[][] attention(MmprojFixture f, String p, double[][] x) {
        int n = x.length;
        int dh = HIDDEN / HEADS;
        float[] pre = f.has(p + "attn_pre_norm.weight") ? f.get(p + "attn_pre_norm.weight") : f.get(p + "ln1.weight");
        double[][] h = rmsNorm(x, pre, null, 1e-6);
        double[][] q = lin(f, p + "attn_q", h, false);
        double[][] k = lin(f, p + "attn_k", h, false);
        double[][] v = lin(f, p + "attn_v", h, false);
        double qs = (1 / Math.sqrt(dh)) / Math.log(2);
        double ks = Math.log(1 + Math.E) / Math.log(2);
        float[] pq = opt(f, p + "per_dim_scale.weight");
        float[] pk = opt(f, p + "per_dim_k_scale.weight");
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < HIDDEN; i++) {
                q[t][i] *= qs * (pq == null ? 1 : pq[i % dh]);
                k[t][i] *= ks * (pk == null ? 1 : pk[i % dh]);
            }
        }
        double[][] rel = f.has(p + "attn_k_rel.weight")
                ? AudioTowerSupport.dense(posEmb(), f.get(p + "attn_k_rel.weight"), null) : null;
        double[][] out = new double[n][HIDDEN];
        for (int hd = 0; hd < HEADS; hd++) {
            for (int i = 0; i < n; i++) {
                double[] sc = new double[n];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < n; j++) {
                    if (j > i || i - j >= Gemma4aTower.MAX_PAST) {
                        sc[j] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double s = 0;
                    for (int d = 0; d < dh; d++) {
                        s += q[i][hd * dh + d] * (k[j][hd * dh + d]
                                + (rel == null ? 0 : rel[Gemma4aTower.MAX_PAST - (i - j)][hd * dh + d]));
                    }
                    sc[j] = 50 * Math.tanh(s / 50);
                    max = Math.max(max, sc[j]);
                }
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    sc[j] = sc[j] == Double.NEGATIVE_INFINITY ? 0 : Math.exp(sc[j] - max);
                    sum += sc[j];
                }
                for (int j = 0; j < n; j++) {
                    for (int d = 0; d < dh; d++) {
                        out[i][hd * dh + d] += sc[j] / sum * v[j][hd * dh + d];
                    }
                }
            }
        }
        double[][] o = lin(f, p + "attn_out", out, true);
        return f.has(p + "attn_post_norm.weight") ? rmsNorm(o, f.get(p + "attn_post_norm.weight"), null, 1e-6) : o;
    }

    /** The 13 relative-position rows: position {@code 12 - p}, sines then cosines. */
    private static double[][] posEmb() {
        int half = HIDDEN / 2;
        double inc = Math.log(10000.0) / Math.max(half - 1, 1);
        double[][] pe = new double[Gemma4aTower.MAX_PAST + 1][HIDDEN];
        for (int r = 0; r < pe.length; r++) {
            for (int i = 0; i < half; i++) {
                double a = (Gemma4aTower.MAX_PAST - r) * Math.exp(-i * inc);
                pe[r][i] = Math.sin(a);
                pe[r][i + half] = Math.cos(a);
            }
        }
        return pe;
    }

    private static double[][] convModule(MmprojFixture f, String p, double[][] x) {
        double[][] u = lin(f, p + "conv_pw1", rmsNorm(x, f.get(p + "conv_norm.weight"), null, 1e-6), false);
        double[][] g = new double[x.length][HIDDEN];
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < HIDDEN; c++) {
                g[t][c] = u[t][c] * AudioTowerSupport.sigmoid(u[t][HIDDEN + c]);
            }
        }
        int kLen = Gemma4aTower.CONV_KERNEL;
        double[][] c = depthwise1d(g, f.get(p + "conv_dw.weight"), kLen, kLen - 1, 0);
        float[] b = opt(f, p + "conv_dw.bias");
        for (double[] row : c) {
            for (int ch = 0; ch < HIDDEN; ch++) {
                row[ch] += b == null ? 0 : b[ch];
            }
        }
        if (f.has(p + "norm_conv.weight")) {
            c = rmsNorm(c, f.get(p + "norm_conv.weight"), null, 1e-6);
        }
        return lin(f, p + "conv_pw2", AudioTowerSupport.map(c, AudioTowerSupport::silu), false);
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 61, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(16, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsTwoStrideTwoConvs() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 1; n <= 40; n++) {
                assertEquals(((n - 1) / 2) / 2 + 1, m.nTokens(n));
                assertEquals(enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length, m.nTokens(n));
            }
        }
    }

    @Test
    public void refusesAConvModuleKernelOtherThanFiveAndPartialAttention() {
        MmprojFixture f = fixture();
        f.weight("a.blk.0.conv_dw.weight", 3, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.0.conv_dw.weight"));
        MmprojFixture g = fixture();
        g.remove("a.blk.1.attn_k.weight");
        IllegalArgumentException a = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), g.catalog()));
        assertTrue(a.getMessage(), a.getMessage().contains("gemma4a layer 1"));
    }

    @Test
    public void refusesAMissingEmbedderProjectionAndAVectorBound() {
        MmprojFixture f = fixture();
        f.remove("mm.a.input_projection.weight");
        assertThrows(RuntimeException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        MmprojFixture g = fixture();
        g.put("a.blk.0.attn_v.input_max", 1f, 1f, 2);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), g.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.0.attn_v.input_max"));
    }
}

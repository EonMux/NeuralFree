package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.attend;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise1d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.layerNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.map;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.scale;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The granite_speech tower against a double-precision transcription of llama.cpp
 * tools/mtmd/models/granite-speech.cpp ({@code clip_graph_granite_speech}): the input projection, conformer blocks
 * with chunked Shaw attention over a partial last chunk, a conv module whose inner width differs from the
 * embedding width, the CTC branch with its own class count, feature taps of the projected input and a block, and
 * the QFormer over a partial last window; the row count and the refusals.
 */
public class GraniteSpeechTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 6;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int LAYERS = 3;
    static final int CONTEXT = 5;
    static final int MAX_POS = 6;
    static final int KERNEL = 3;
    static final int INNER = 10;
    static final int CLASSES = 7;
    static final int[] FEATURE_LAYERS = {0, 2};
    static final int WINDOW = 4;
    static final int DOWNSAMPLE = 2;
    static final int PROJ_HEADS = 2;
    static final int PROJ_FF = 9;
    static final int TEXT = 6;
    static final float EPS = 1e-5f;

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("granite_speech", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT,
                EPS);
        m.put("clip.audio.chunk_size", CONTEXT);
        m.put("clip.audio.conv_kernel_size", KERNEL);
        m.put("clip.audio.max_pos_emb", MAX_POS);
        m.put("clip.audio.projector.window_size", WINDOW);
        m.put("clip.audio.projector.downsample_rate", DOWNSAMPLE);
        m.put("clip.audio.projector.head_count", PROJ_HEADS);
        m.put("clip.audio.feature_layer", FEATURE_LAYERS);
        return m;
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(101);
        f.weight("a.input_projection.weight", N_MEL, HIDDEN);
        f.weight("a.input_projection.bias", HIDDEN);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            for (String norm : new String[]{"ffn_norm", "ffn_norm_1", "norm_conv", "ln1", "ln2"}) {
                f.norm(p + norm + ".weight", p + norm + ".bias", HIDDEN);
            }
            for (String up : new String[]{"ffn_up", "ffn_up_1"}) {
                f.weight(p + up + ".weight", HIDDEN, FF);
                f.weight(p + up + ".bias", FF);
            }
            for (String down : new String[]{"ffn_down", "ffn_down_1"}) {
                f.weight(p + down + ".weight", FF, HIDDEN);
                f.weight(p + down + ".bias", HIDDEN);
            }
            for (String proj : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.weight(p + proj + ".weight", HIDDEN, HIDDEN);
            }
            f.weight(p + "attn_out.bias", HIDDEN);
            f.weight(p + "attn_rel_pos_emb", HIDDEN / HEADS, 2 * MAX_POS + 1);
            f.weight(p + "conv_pw1.weight", HIDDEN, 2 * INNER);
            f.weight(p + "conv_pw1.bias", 2 * INNER);
            f.weight(p + "conv_dw.weight", KERNEL, INNER);
            f.norm(p + "conv_norm.weight", p + "conv_norm.bias", INNER);
            f.weight(p + "conv_pw2.weight", INNER, HIDDEN);
            f.weight(p + "conv_pw2.bias", HIDDEN);
        }
        f.weight("a.enc_ctc_out.weight", HIDDEN, CLASSES);
        f.weight("a.enc_ctc_out.bias", CLASSES);
        f.weight("a.enc_ctc_out_mid.weight", CLASSES, HIDDEN);
        f.weight("a.enc_ctc_out_mid.bias", HIDDEN);
        f.weight("a.proj_query", HIDDEN, WINDOW / DOWNSAMPLE, 1);
        f.norm("a.proj_norm.weight", "a.proj_norm.bias", HIDDEN);
        long encWidth = (long) HIDDEN * (FEATURE_LAYERS.length + 1);
        for (int il = 0; il < GraniteSpeechTower.PROJ_LAYERS; il++) {
            String p = "a.proj_blk." + il + ".";
            for (String t : new String[]{"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
                    "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out"}) {
                long in = t.equals("cross_attn_k") || t.equals("cross_attn_v") ? encWidth : HIDDEN;
                f.weight(p + t + ".weight", in, HIDDEN);
                f.weight(p + t + ".bias", HIDDEN);
            }
            for (String norm : new String[]{"self_attn_norm", "cross_attn_norm", "ffn_norm"}) {
                f.norm(p + norm + ".weight", p + norm + ".bias", HIDDEN);
            }
            f.weight(p + "ffn_up.weight", HIDDEN, PROJ_FF);
            f.weight(p + "ffn_up.bias", PROJ_FF);
            f.weight(p + "ffn_down.weight", PROJ_FF, HIDDEN);
            f.weight(p + "ffn_down.bias", HIDDEN);
        }
        f.weight("a.proj_linear.weight", HIDDEN, TEXT);
        f.weight("a.proj_linear.bias", TEXT);
        return f;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][] x = dense(AudioTowerSupport.frames(chunk), f.get("a.input_projection.weight"),
                f.get("a.input_projection.bias"));
        List<double[][]> taps = new ArrayList<>();
        if (tapped(0)) {
            taps.add(x);
        }
        for (int il = 0; il < LAYERS; il++) {
            x = block(f, "a.blk." + il + ".", x);
            if (tapped(il + 1)) {
                taps.add(x);
            }
            if (il + 1 == LAYERS / 2) {
                double[][] mid = softmax(dense(x, f.get("a.enc_ctc_out.weight"), f.get("a.enc_ctc_out.bias")));
                x = add(x, dense(mid, f.get("a.enc_ctc_out_mid.weight"), f.get("a.enc_ctc_out_mid.bias")));
            }
        }
        taps.add(x);
        int windows = (x.length + WINDOW - 1) / WINDOW;
        double[][] enc = new double[windows * WINDOW][HIDDEN * taps.size()];
        for (int t = 0; t < x.length; t++) {
            for (int s = 0; s < taps.size(); s++) {
                System.arraycopy(taps.get(s)[t], 0, enc[t], s * HIDDEN, HIDDEN);
            }
        }
        return dense(qformer(f, enc, windows), f.get("a.proj_linear.weight"), f.get("a.proj_linear.bias"));
    }

    private static boolean tapped(int layer) {
        for (int v : FEATURE_LAYERS) {
            if (v == layer) {
                return true;
            }
        }
        return false;
    }

    private static double[][] block(MmprojFixture f, String p, double[][] x) {
        x = add(x, scale(halfFfn(f, p, "ffn_norm", "ffn_up", "ffn_down", x), 0.5));
        x = add(x, attention(f, p, norm(f, p + "ln1", x)));
        x = add(x, conv(f, p, norm(f, p + "norm_conv", x)));
        x = add(x, scale(halfFfn(f, p, "ffn_norm_1", "ffn_up_1", "ffn_down_1", x), 0.5));
        return norm(f, p + "ln2", x);
    }

    private static double[][] norm(MmprojFixture f, String base, double[][] x) {
        return layerNorm(x, f.get(base + ".weight"), f.get(base + ".bias"), EPS);
    }

    private static double[][] halfFfn(MmprojFixture f, String p, String norm, String up, String down,
                                      double[][] x) {
        return ffn(norm(f, p + norm, x), f.get(p + up + ".weight"), f.get(p + up + ".bias"),
                AudioTowerSupport::silu, f.get(p + down + ".weight"), f.get(p + down + ".bias"));
    }

    /** Per chunk of {@code CONTEXT} rows and head: {@code (q.k + q.rel[i - j + MAX_POS]) / sqrt(dh)}. */
    private static double[][] attention(MmprojFixture f, String p, double[][] h) {
        double[][] q = dense(h, f.get(p + "attn_q.weight"), null);
        double[][] k = dense(h, f.get(p + "attn_k.weight"), null);
        double[][] v = dense(h, f.get(p + "attn_v.weight"), null);
        float[] rel = f.get(p + "attn_rel_pos_emb");
        int dh = HIDDEN / HEADS;
        double[][] out = new double[h.length][HIDDEN];
        for (int start = 0; start < h.length; start += CONTEXT) {
            int len = Math.min(CONTEXT, h.length - start);
            for (int head = 0; head < HEADS; head++) {
                for (int i = 0; i < len; i++) {
                    double[] s = new double[len];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < len; j++) {
                        double dot = 0;
                        for (int d = 0; d < dh; d++) {
                            double qv = q[start + i][head * dh + d];
                            dot += qv * k[start + j][head * dh + d] + qv * rel[(i - j + MAX_POS) * dh + d];
                        }
                        s[j] = dot / Math.sqrt(dh);
                        max = Math.max(max, s[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < len; j++) {
                        s[j] = Math.exp(s[j] - max);
                        sum += s[j];
                    }
                    for (int j = 0; j < len; j++) {
                        for (int d = 0; d < dh; d++) {
                            out[start + i][head * dh + d] += s[j] / sum * v[start + j][head * dh + d];
                        }
                    }
                }
            }
        }
        return dense(out, f.get(p + "attn_out.weight"), f.get(p + "attn_out.bias"));
    }

    private static double[][] conv(MmprojFixture f, String p, double[][] h) {
        double[][] x = dense(h, f.get(p + "conv_pw1.weight"), f.get(p + "conv_pw1.bias"));
        double[][] glu = new double[h.length][INNER];
        for (int t = 0; t < h.length; t++) {
            for (int c = 0; c < INNER; c++) {
                glu[t][c] = x[t][c] * AudioTowerSupport.sigmoid(x[t][INNER + c]);
            }
        }
        double[][] c = depthwise1d(glu, f.get(p + "conv_dw.weight"), KERNEL, KERNEL / 2, KERNEL / 2);
        float[] bnW = f.get(p + "conv_norm.weight");
        float[] bnB = f.get(p + "conv_norm.bias");
        for (double[] row : c) {
            for (int ch = 0; ch < INNER; ch++) {
                row[ch] = AudioTowerSupport.silu(row[ch] * bnW[ch] + bnB[ch]);
            }
        }
        return dense(c, f.get(p + "conv_pw2.weight"), f.get(p + "conv_pw2.bias"));
    }

    private static double[][] softmax(double[][] x) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double v : x[t]) {
                max = Math.max(max, v);
            }
            double sum = 0;
            y[t] = new double[x[t].length];
            for (int i = 0; i < x[t].length; i++) {
                y[t][i] = Math.exp(x[t][i] - max);
                sum += y[t][i];
            }
            for (int i = 0; i < x[t].length; i++) {
                y[t][i] /= sum;
            }
        }
        return y;
    }

    private static double[][] qformer(MmprojFixture f, double[][] enc, int windows) {
        int nq = WINDOW / DOWNSAMPLE;
        double[][] base = new double[nq][HIDDEN];
        float[] query = f.get("a.proj_query");
        for (int i = 0; i < nq; i++) {
            for (int c = 0; c < HIDDEN; c++) {
                base[i][c] = query[i * HIDDEN + c];
            }
        }
        base = layerNorm(base, f.get("a.proj_norm.weight"), f.get("a.proj_norm.bias"), GraniteSpeechTower.PROJ_EPS);
        double[][] x = new double[windows * nq][];
        for (int r = 0; r < x.length; r++) {
            x[r] = base[r % nq].clone();
        }
        double scale = 1 / Math.sqrt((double) HIDDEN / PROJ_HEADS);
        for (int il = 0; il < GraniteSpeechTower.PROJ_LAYERS; il++) {
            String p = "a.proj_blk." + il + ".";
            double[][] a = attend(lin(f, p + "self_attn_q", x), lin(f, p + "self_attn_k", x),
                    lin(f, p + "self_attn_v", x), PROJ_HEADS, scale, (i, j) -> i / nq == j / nq);
            x = postNorm(f, p + "self_attn_norm", x, lin(f, p + "self_attn_out", a));
            a = attend(lin(f, p + "cross_attn_q", x), lin(f, p + "cross_attn_k", enc), lin(f, p + "cross_attn_v", enc),
                    PROJ_HEADS, scale, (i, j) -> j / WINDOW == i / nq);
            x = postNorm(f, p + "cross_attn_norm", x, lin(f, p + "cross_attn_out", a));
            double[][] h = ffn(x, f.get(p + "ffn_up.weight"), f.get(p + "ffn_up.bias"),
                    GraniteSpeechTowerTest::geluTanh, f.get(p + "ffn_down.weight"), f.get(p + "ffn_down.bias"));
            x = postNorm(f, p + "ffn_norm", x, h);
        }
        return x;
    }

    private static double[][] lin(MmprojFixture f, String base, double[][] x) {
        return dense(x, f.get(base + ".weight"), f.get(base + ".bias"));
    }

    private static double[][] postNorm(MmprojFixture f, String base, double[][] x, double[][] y) {
        return layerNorm(add(x, y), f.get(base + ".weight"), f.get(base + ".bias"), GraniteSpeechTower.PROJ_EPS);
    }

    private static double geluTanh(double v) {
        return 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 23, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(12, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsTheWindowsTimesTheQueries() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 1; n <= 30; n++) {
                assertEquals((n + WINDOW - 1) / WINDOW * (WINDOW / DOWNSAMPLE), m.nTokens(n));
                assertEquals(m.nTokens(n), enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length);
            }
        }
    }

    @Test
    public void refusesAShortRelativePositionTable() {
        MmprojFixture f = fixture();
        f.weight("a.blk.1.attn_rel_pos_emb", HIDDEN / HEADS, MAX_POS + CONTEXT - 1);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("attn_rel_pos_emb"));
    }

    @Test
    public void refusesBadMetadataAndTaps() {
        Map<String, Object> even = meta();
        even.put("clip.audio.conv_kernel_size", 4);
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(even));
        Map<String, Object> repeated = meta();
        repeated.put("clip.audio.feature_layer", new int[]{2, 2});
        assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(repeated), fixture().catalog()));
        MmprojFixture wrongMid = fixture();
        wrongMid.weight("a.enc_ctc_out_mid.weight", HIDDEN, HIDDEN);
        assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), wrongMid.catalog()));
    }
}

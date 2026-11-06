package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.attend;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.conv2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise1d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.depthwise2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.layerNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.map;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ropeNeox;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The lfm3a tower against a double-precision transcription of llama.cpp tools/mtmd/models/lfm3a.cpp
 * ({@code clip_graph_lfm3a}) from a zero streaming state: the SiLU downsampler over the {@code [frames, mel]}
 * image, a short-conv block, a sliding-window GQA attention block with RMS head norms and NEOX RoPE at 1e6, gated
 * and ungated SiLU FFNs, the RMS trunk norm and the GELU-erf projector; the row count and the block refusals.
 */
public class Lfm3aTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 16;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int KV_HEADS = 1;
    static final int FF = 12;
    static final int TEXT = 6;
    static final int ADAPTER = 10;
    static final int WINDOW = 4;
    static final int TAPS = 3;
    static final float EPS = 1e-5f;
    /** Block kinds: short conv with a gated FFN, attention with a gated FFN, short conv with a plain FFN. */
    static final boolean[] CONV = {true, false, true};

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("lfm3a", N_MEL, HIDDEN, FF, CONV.length, HEADS, TEXT, EPS);
        m.put("clip.audio.attention.head_count_kv", KV_HEADS);
        m.put("clip.audio.window_size", WINDOW);
        return m;
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(81);
        int[] c = {4, 4, 3};
        f.weight("a.conv1d.0.weight", 3, 3, 1, c[0]);
        f.weight("a.conv1d.0.bias", 1, 1, c[0]);
        f.weight("a.conv1d.2.weight", 3, 3, 1, c[0]);
        f.weight("a.conv1d.2.bias", 1, 1, c[0]);
        f.weight("a.conv1d.3.weight", 1, 1, c[0], c[1]);
        f.weight("a.conv1d.3.bias", 1, 1, c[1]);
        f.weight("a.conv1d.5.weight", 3, 3, 1, c[1]);
        f.weight("a.conv1d.5.bias", 1, 1, c[1]);
        f.weight("a.conv1d.6.weight", 1, 1, c[1], c[2]);
        f.weight("a.conv1d.6.bias", 1, 1, c[2]);
        f.weight("a.pre_encode.out.weight", 2L * c[2], HIDDEN);
        f.weight("a.pre_encode.out.bias", HIDDEN);
        int dh = HIDDEN / HEADS;
        for (int il = 0; il < CONV.length; il++) {
            String p = "a.blk." + il + ".";
            f.put(p + "ln1.weight", 0.8f, 1.2f, HIDDEN);
            f.put(p + "ln2.weight", 0.8f, 1.2f, HIDDEN);
            f.weight(p + "ffn_up.weight", HIDDEN, FF);
            if (il < 2) {
                f.weight(p + "ffn_gate.weight", HIDDEN, FF);
            }
            f.weight(p + "ffn_down.weight", FF, HIDDEN);
            if (CONV[il]) {
                f.weight(p + "conv_pw1.weight", HIDDEN, 3L * HIDDEN);
                f.weight(p + "conv_dw.weight", TAPS, HIDDEN);
                f.weight(p + "conv_pw2.weight", HIDDEN, HIDDEN);
            } else {
                f.weight(p + "attn_q.weight", HIDDEN, HIDDEN);
                f.weight(p + "attn_k.weight", HIDDEN, (long) KV_HEADS * dh);
                f.weight(p + "attn_v.weight", HIDDEN, (long) KV_HEADS * dh);
                f.weight(p + "attn_out.weight", HIDDEN, HIDDEN);
                f.put(p + "attn_q_norm.weight", 0.8f, 1.2f, dh);
                f.put(p + "attn_k_norm.weight", 0.8f, 1.2f, dh);
            }
        }
        f.put("a.post_ln.weight", 0.8f, 1.2f, HIDDEN);
        f.norm("mm.a.mlp.0.weight", "mm.a.mlp.0.bias", HIDDEN);
        f.weight("mm.a.mlp.1.weight", HIDDEN, ADAPTER);
        f.weight("mm.a.mlp.1.bias", ADAPTER);
        f.weight("mm.a.mlp.3.weight", ADAPTER, TEXT);
        f.weight("mm.a.mlp.3.bias", TEXT);
        return f;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][] x = stem(f, chunk);
        for (int il = 0; il < CONV.length; il++) {
            String p = "a.blk." + il + ".";
            double[][] h = rmsNorm(x, f.get(p + "ln1.weight"), null, EPS);
            x = add(x, CONV[il] ? shortConv(f, p, h) : attention(f, p, h));
            h = rmsNorm(x, f.get(p + "ln2.weight"), null, EPS);
            double[][] up = dense(h, f.get(p + "ffn_up.weight"), null);
            if (f.has(p + "ffn_gate.weight")) {
                double[][] g = map(dense(h, f.get(p + "ffn_gate.weight"), null), AudioTowerSupport::silu);
                for (int t = 0; t < up.length; t++) {
                    for (int j = 0; j < up[t].length; j++) {
                        up[t][j] *= g[t][j];
                    }
                }
            } else {
                up = map(up, AudioTowerSupport::silu);
            }
            x = add(x, dense(up, f.get(p + "ffn_down.weight"), null));
        }
        x = rmsNorm(x, f.get("a.post_ln.weight"), null, EPS);
        x = layerNorm(x, f.get("mm.a.mlp.0.weight"), f.get("mm.a.mlp.0.bias"), 1e-5);
        return ffn(x, f.get("mm.a.mlp.1.weight"), f.get("mm.a.mlp.1.bias"), AudioTowerSupport::geluErf,
                f.get("mm.a.mlp.3.weight"), f.get("mm.a.mlp.3.bias"));
    }

    /** The downsampler over {@code img[c][mel][frame]}; the flatten puts the mel bin minor. */
    private static double[][] stem(MmprojFixture f, MelChunk chunk) {
        double[][][] img = new double[1][chunk.nMel()][chunk.nLen()];
        for (int m = 0; m < chunk.nMel(); m++) {
            for (int t = 0; t < chunk.nLen(); t++) {
                img[0][m][t] = chunk.value(m, t);
            }
        }
        img = silu(conv2d(img, f.get("a.conv1d.0.weight"), f.get("a.conv1d.0.bias"), 3, 3, 2, 1));
        img = depthwise2d(img, f.get("a.conv1d.2.weight"), f.get("a.conv1d.2.bias"), 3, 3, 2, 1);
        img = silu(conv2d(img, f.get("a.conv1d.3.weight"), f.get("a.conv1d.3.bias"), 1, 1, 1, 0));
        img = depthwise2d(img, f.get("a.conv1d.5.weight"), f.get("a.conv1d.5.bias"), 3, 3, 2, 1);
        img = silu(conv2d(img, f.get("a.conv1d.6.weight"), f.get("a.conv1d.6.bias"), 1, 1, 1, 0));
        int c = img.length;
        int fr = img[0].length;
        int n = img[0][0].length;
        double[][] rows = new double[n][fr * c];
        for (int t = 0; t < n; t++) {
            for (int ch = 0; ch < c; ch++) {
                for (int m = 0; m < fr; m++) {
                    rows[t][m + fr * ch] = img[ch][m][t];
                }
            }
        }
        return dense(rows, f.get("a.pre_encode.out.weight"), f.get("a.pre_encode.out.bias"));
    }

    private static double[][][] silu(double[][][] img) {
        for (double[][] plane : img) {
            for (double[] row : plane) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = AudioTowerSupport.silu(row[i]);
                }
            }
        }
        return img;
    }

    private static double[][] shortConv(MmprojFixture f, String p, double[][] h) {
        double[][] bcx = dense(h, f.get(p + "conv_pw1.weight"), null);
        double[][] bx = new double[h.length][HIDDEN];
        for (int t = 0; t < h.length; t++) {
            for (int c = 0; c < HIDDEN; c++) {
                bx[t][c] = bcx[t][c] * bcx[t][2 * HIDDEN + c];
            }
        }
        double[][] conv = depthwise1d(bx, f.get(p + "conv_dw.weight"), TAPS, TAPS - 1, 0);
        for (int t = 0; t < h.length; t++) {
            for (int c = 0; c < HIDDEN; c++) {
                conv[t][c] *= bcx[t][HIDDEN + c];
            }
        }
        return dense(conv, f.get(p + "conv_pw2.weight"), null);
    }

    private static double[][] attention(MmprojFixture f, String p, double[][] h) {
        int dh = HIDDEN / HEADS;
        int[] pos = AudioOps.iota(h.length);
        double[][] q = ropeNeox(headNorm(dense(h, f.get(p + "attn_q.weight"), null), f.get(p + "attn_q_norm.weight")),
                HEADS, dh, pos, 1e6);
        double[][] k = ropeNeox(headNorm(dense(h, f.get(p + "attn_k.weight"), null), f.get(p + "attn_k_norm.weight")),
                KV_HEADS, dh, pos, 1e6);
        double[][] v = dense(h, f.get(p + "attn_v.weight"), null);
        double[][] a = attend(q, repeatHeads(k), repeatHeads(v), HEADS, 1 / Math.sqrt(dh),
                (i, j) -> j <= i && j > i - WINDOW);
        return dense(a, f.get(p + "attn_out.weight"), null);
    }

    private static double[][] headNorm(double[][] x, float[] w) {
        int dh = w.length;
        double[][] y = new double[x.length][x[0].length];
        for (int t = 0; t < x.length; t++) {
            for (int h = 0; h < x[t].length / dh; h++) {
                double[][] one = {java.util.Arrays.copyOfRange(x[t], h * dh, (h + 1) * dh)};
                System.arraycopy(rmsNorm(one, w, null, EPS)[0], 0, y[t], h * dh, dh);
            }
        }
        return y;
    }

    /** The KV heads repeated for their query-head group. */
    private static double[][] repeatHeads(double[][] kv) {
        int dh = HIDDEN / HEADS;
        double[][] out = new double[kv.length][HIDDEN];
        for (int t = 0; t < kv.length; t++) {
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(kv[t], (h / (HEADS / KV_HEADS)) * dh, out[t], h * dh, dh);
            }
        }
        return out;
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 77, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(10, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsThreeStrideTwoConvs() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 1; n <= 40; n++) {
                assertEquals(((((n + 1) / 2) + 1) / 2 + 1) / 2, m.nTokens(n));
                assertEquals(enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length, m.nTokens(n));
            }
        }
    }

    @Test
    public void refusesABlockThatIsBothOrNeither() {
        MmprojFixture both = fixture();
        both.weight("a.blk.1.conv_pw1.weight", HIDDEN, 3L * HIDDEN);
        both.weight("a.blk.1.conv_dw.weight", TAPS, HIDDEN);
        both.weight("a.blk.1.conv_pw2.weight", HIDDEN, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), both.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("lfm3a layer 1"));
        MmprojFixture neither = fixture();
        neither.remove("a.blk.1.attn_k_norm.weight");
        assertThrows(RuntimeException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), neither.catalog()));
    }

    @Test
    public void refusesAStemTheStreamingStateCannotHold() {
        MmprojFixture f = fixture();
        f.weight("a.conv1d.3.weight", 1, 1, 4, 5);
        f.weight("a.conv1d.3.bias", 1, 1, 5);
        f.weight("a.conv1d.5.weight", 3, 3, 1, 5);
        f.weight("a.conv1d.5.bias", 1, 1, 5);
        f.weight("a.conv1d.6.weight", 1, 1, 5, 3);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.conv1d.3.weight"));
        Map<String, Object> m = meta();
        m.put("clip.audio.num_mel_bins", 18);
        IllegalArgumentException mel = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(m), fixture().catalog()));
        assertTrue(mel.getMessage(), mel.getMessage().contains("num_mel_bins"));
    }

    @Test
    public void refusesAMissingWindowAndAOneTapShortConv() {
        Map<String, Object> m = meta();
        m.remove("clip.audio.window_size");
        assertThrows(RuntimeException.class, () -> AudioTowerSupport.hparams(m));
        MmprojFixture f = fixture();
        f.weight("a.blk.0.conv_dw.weight", 1, HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.blk.0.conv_dw.weight"));
    }
}

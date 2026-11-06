package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.attend;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.conv2d;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ffn;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.layerNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.map;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.ropeNeox;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The dots3note audio tower against a double-precision transcription of llama.cpp tools/mtmd/models/dots3note.cpp
 * ({@code clip_graph_dots3note_a}): the GELU-erf conv2d stem, the {@code f + OH * c} flatten, conv_out, the RMS
 * build_vit blocks with half-head NEOX rotary positions, the layer-normed GELU-erf adapter, and the
 * {@code (n + 7) / 8} row count; plus its 60-second mtmd-audio.cpp chunking.
 */
public class Dots3NoteATowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 16;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int TEXT = 5;
    static final int ADAPTER = 10;
    static final float EPS = 1e-6f;
    static final int[] CH = {2, 3, 4};

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("dots3note_a", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, EPS);
        m.put("clip.use_silu", true);
        return m;
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(51);
        int cIn = 1;
        for (int i = 0; i < 3; i++) {
            f.weight("a.conv2d." + (i + 1) + ".weight", 3, 3, cIn, CH[i]);
            f.weight("a.conv2d." + (i + 1) + ".bias", CH[i]);
            cIn = CH[i];
        }
        f.weight("a.conv_out.weight", 2L * CH[2], HIDDEN);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", HIDDEN);
            f.norm(p + "ln2.weight", p + "ln2.bias", HIDDEN);
            for (String t : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.weight(p + t + ".weight", HIDDEN, HIDDEN);
                f.weight(p + t + ".bias", HIDDEN);
            }
            f.weight(p + "ffn_up.weight", HIDDEN, FF);
            f.weight(p + "ffn_up.bias", FF);
            if (il == 0) {
                f.weight(p + "ffn_gate.weight", HIDDEN, FF);
            }
            f.weight(p + "ffn_down.weight", FF, HIDDEN);
            f.weight(p + "ffn_down.bias", HIDDEN);
        }
        f.put("a.post_ln.weight", 0.8f, 1.2f, HIDDEN);
        f.norm("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", HIDDEN);
        f.weight("mm.a.mlp.1.weight", HIDDEN, ADAPTER);
        f.weight("mm.a.mlp.1.bias", ADAPTER);
        f.weight("mm.a.mlp.3.weight", ADAPTER, TEXT);
        f.weight("mm.a.mlp.3.bias", TEXT);
        return f;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][][] img = new double[1][N_MEL][chunk.nLen()];
        for (int m = 0; m < N_MEL; m++) {
            for (int t = 0; t < chunk.nLen(); t++) {
                img[0][m][t] = chunk.value(m, t);
            }
        }
        for (int i = 1; i <= 3; i++) {
            img = conv2d(img, f.get("a.conv2d." + i + ".weight"), f.get("a.conv2d." + i + ".bias"), 3, 3, 2, 1);
            for (double[][] plane : img) {
                for (double[] row : plane) {
                    for (int x = 0; x < row.length; x++) {
                        row[x] = AudioTowerSupport.geluErf(row[x]);
                    }
                }
            }
        }
        int c = img.length;
        int oh = img[0].length;
        int ow = img[0][0].length;
        double[][] x = new double[ow][oh * c];
        for (int t = 0; t < ow; t++) {
            for (int ch = 0; ch < c; ch++) {
                for (int fr = 0; fr < oh; fr++) {
                    x[t][fr + oh * ch] = img[ch][fr][t];
                }
            }
        }
        x = dense(x, f.get("a.conv_out.weight"), null);
        int[] pos = AudioOps.iota(ow);
        int dh = HIDDEN / HEADS;
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            double[][] h = rmsNorm(x, f.get(p + "ln1.weight"), f.get(p + "ln1.bias"), EPS);
            double[][] q = ropeNeox(dense(h, f.get(p + "attn_q.weight"), f.get(p + "attn_q.bias")), HEADS, dh / 2,
                    pos, 1e4);
            double[][] k = ropeNeox(dense(h, f.get(p + "attn_k.weight"), f.get(p + "attn_k.bias")), HEADS, dh / 2,
                    pos, 1e4);
            double[][] v = dense(h, f.get(p + "attn_v.weight"), f.get(p + "attn_v.bias"));
            double[][] a = attend(q, k, v, HEADS, 1 / Math.sqrt(dh), (i, j) -> true);
            x = add(x, dense(a, f.get(p + "attn_out.weight"), f.get(p + "attn_out.bias")));
            h = rmsNorm(x, f.get(p + "ln2.weight"), f.get(p + "ln2.bias"), EPS);
            double[][] up = dense(h, f.get(p + "ffn_up.weight"), f.get(p + "ffn_up.bias"));
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
            x = add(x, dense(up, f.get(p + "ffn_down.weight"), f.get(p + "ffn_down.bias")));
        }
        x = rmsNorm(x, f.get("a.post_ln.weight"), null, EPS);
        x = layerNorm(x, f.get("mm.a.norm_pre.weight"), f.get("mm.a.norm_pre.bias"), 1e-5);
        return ffn(x, f.get("mm.a.mlp.1.weight"), f.get("mm.a.mlp.1.bias"), AudioTowerSupport::geluErf,
                f.get("mm.a.mlp.3.weight"), f.get("mm.a.mlp.3.bias"));
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 21, 3);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(3, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsThreeStrideTwoConvs() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            assertEquals(1, m.nTokens(1));
            assertEquals(1, m.nTokens(8));
            assertEquals(2, m.nTokens(9));
            assertEquals(375, m.nTokens(3000));
            for (int n = 1; n <= 40; n++) {
                assertEquals(enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length, m.nTokens(n));
            }
        }
    }

    @Test
    public void framesSixtySecondChunksOfValidFrames() {
        // 61 s at 16 kHz: one full 60 s chunk of 6000 frames, then 1 s of 100 frames
        float[] samples = RawWaveTowersTest.samples(16000 * 61, 2);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            List<MelChunk> chunks = enc.chunks(new AudioClip(samples, 16000));
            assertEquals(2, chunks.size());
            assertEquals(6000, chunks.get(0).nLen());
            assertEquals(100, chunks.get(1).nLen());
            assertEquals(N_MEL, chunks.get(0).nMel());
            assertEquals(750 + 13, enc.nTokens(new AudioClip(samples, 16000)));
        }
    }

    @Test
    public void refusesAConvOutOfAnotherHeight() {
        MmprojFixture f = fixture();
        f.weight("a.conv_out.weight", 3L * CH[2], HIDDEN);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), f.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("a.conv_out.weight"));
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.attend;
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
 * The mimo_audio tower against a double-precision transcription of llama.cpp tools/mtmd/models/mimo-audio.cpp
 * ({@code clip_graph_mimo_audio}): the half-padded conv stem, ViT blocks with NEOX RoPE under the full and the
 * windowed causal masks, the skip connection of block 2's output, the post norm, the downsample, two residual
 * quantizers using fewer bins than their tables hold, the summed code embeddings, the local transformer over
 * zero-padded groups and the grouped GELU-erf projector; the row count and the refusals.
 */
public class MimoAudioTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 8;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int FF = 12;
    static final int[] WA_PATTERN = {0, -1, 0, -1};
    static final int WINDOW = 2;
    static final int[] BINS = {5, 3};
    static final int CODEBOOK_ROWS = 6;
    static final int CODE_VOCAB = 7;
    static final int LOCAL_LAYERS = 2;
    static final int LOCAL_FF = 10;
    static final int GROUP = 3;
    static final int ADAPTER = 9;
    static final int TEXT = 6;
    static final float EPS = 1e-5f;

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("mimo_audio", N_MEL, HIDDEN, FF, WA_PATTERN.length, HEADS,
                TEXT, EPS);
        m.put("clip.audio.rvq.num_quantizers", BINS.length);
        m.put("clip.audio.rvq.codebook_size", BINS);
        m.put("clip.audio.wa_pattern_mode", WA_PATTERN);
        m.put("clip.audio.window_size", WINDOW);
        m.put("clip.audio.local_block_count", LOCAL_LAYERS);
        m.put("clip.audio.local_group_size", GROUP);
        return m;
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(97);
        f.weight("a.conv1d.1.weight", 3, N_MEL, HIDDEN);
        f.weight("a.conv1d.1.bias", 1, HIDDEN);
        f.weight("a.conv1d.2.weight", 3, HIDDEN, HIDDEN);
        f.weight("a.conv1d.2.bias", 1, HIDDEN);
        for (int il = 0; il < WA_PATTERN.length; il++) {
            String p = "a.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", HIDDEN);
            f.norm(p + "ln2.weight", p + "ln2.bias", HIDDEN);
            for (String proj : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.weight(p + proj + ".weight", HIDDEN, HIDDEN);
                f.weight(p + proj + ".bias", HIDDEN);
            }
            f.weight(p + "ffn_up.weight", HIDDEN, FF);
            f.weight(p + "ffn_up.bias", FF);
            f.weight(p + "ffn_down.weight", FF, HIDDEN);
            f.weight(p + "ffn_down.bias", HIDDEN);
        }
        f.norm("a.post_ln.weight", "a.post_ln.bias", HIDDEN);
        f.weight("a.downsample.conv.weight", 2, HIDDEN, HIDDEN);
        f.norm("a.downsample.norm.weight", "a.downsample.norm.bias", HIDDEN);
        f.put("a.rvq.codebook.weight", -1f, 1f, HIDDEN, CODEBOOK_ROWS, BINS.length);
        f.weight("mm.a.code_embd.weight", HIDDEN, CODE_VOCAB, BINS.length);
        for (int il = 0; il < LOCAL_LAYERS; il++) {
            String p = "mm.a.local_blk." + il + ".";
            f.put(p + "ln1.weight", 0.8f, 1.2f, HIDDEN);
            f.put(p + "ln2.weight", 0.8f, 1.2f, HIDDEN);
            for (String proj : new String[]{"attn_q", "attn_k", "attn_v"}) {
                f.weight(p + proj + ".weight", HIDDEN, HIDDEN);
                f.weight(p + proj + ".bias", HIDDEN);
            }
            f.weight(p + "attn_out.weight", HIDDEN, HIDDEN);
            f.weight(p + "ffn_gate.weight", HIDDEN, LOCAL_FF);
            f.weight(p + "ffn_up.weight", HIDDEN, LOCAL_FF);
            f.weight(p + "ffn_down.weight", LOCAL_FF, HIDDEN);
        }
        f.put("mm.a.local_norm.weight", 0.8f, 1.2f, HIDDEN);
        f.weight("mm.a.mlp.1.weight", (long) HIDDEN * GROUP, ADAPTER);
        f.weight("mm.a.mlp.2.weight", ADAPTER, TEXT);
        return f;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][] x = conv(AudioTowerSupport.frames(chunk), f.get("a.conv1d.1.weight"), f.get("a.conv1d.1.bias"),
                3, 1, 1);
        x = conv(x, f.get("a.conv1d.2.weight"), f.get("a.conv1d.2.bias"), 3, 2, 1);
        double[][] skip = null;
        for (int il = 0; il < WA_PATTERN.length; il++) {
            x = block(f, il, x);
            if (il == MimoAudioTower.SKIP_LAYER) {
                skip = x;
            }
        }
        x = layerNorm(add(x, skip), f.get("a.post_ln.weight"), f.get("a.post_ln.bias"), EPS);
        x = conv(x, f.get("a.downsample.conv.weight"), null, 2, 2, 0);
        x = layerNorm(x, f.get("a.downsample.norm.weight"), f.get("a.downsample.norm.bias"), EPS);
        int groups = (x.length + GROUP - 1) / GROUP;
        double[][] local = new double[groups * GROUP][HIDDEN];
        System.arraycopy(quantize(f, x), 0, local, 0, x.length);
        for (int il = 0; il < LOCAL_LAYERS; il++) {
            local = localLayer(f, "mm.a.local_blk." + il + ".", local);
        }
        local = rmsNorm(local, f.get("mm.a.local_norm.weight"), null, EPS);
        double[][] grouped = new double[groups][HIDDEN * GROUP];
        for (int t = 0; t < local.length; t++) {
            System.arraycopy(local[t], 0, grouped[t / GROUP], (t % GROUP) * HIDDEN, HIDDEN);
        }
        return ffn(grouped, f.get("mm.a.mlp.1.weight"), null, AudioTowerSupport::geluErf,
                f.get("mm.a.mlp.2.weight"), null);
    }

    /** {@code ggml_conv_1d(w [k, cIn, cOut], x, stride, pad, 1)} over token-major rows, bias, GELU-erf. */
    private static double[][] conv(double[][] x, float[] w, float[] b, int k, int stride, int pad) {
        int cIn = x[0].length;
        int cOut = w.length / (k * cIn);
        int n = (x.length + 2 * pad - k) / stride + 1;
        double[][] y = new double[n][cOut];
        for (int t = 0; t < n; t++) {
            for (int co = 0; co < cOut; co++) {
                double s = b != null ? b[co] : 0;
                for (int ci = 0; ci < cIn; ci++) {
                    for (int j = 0; j < k; j++) {
                        int src = t * stride + j - pad;
                        if (src >= 0 && src < x.length) {
                            s += (double) w[j + k * (ci + cIn * co)] * x[src][ci];
                        }
                    }
                }
                y[t][co] = AudioTowerSupport.geluErf(s);
            }
        }
        return y;
    }

    private static double[][] block(MmprojFixture f, int il, double[][] x) {
        String p = "a.blk." + il + ".";
        double[][] h = layerNorm(x, f.get(p + "ln1.weight"), f.get(p + "ln1.bias"), EPS);
        int[] pos = AudioOps.iota(x.length);
        int dh = HIDDEN / HEADS;
        double[][] q = ropeNeox(dense(h, f.get(p + "attn_q.weight"), f.get(p + "attn_q.bias")), HEADS, dh, pos, 1e4);
        double[][] k = ropeNeox(dense(h, f.get(p + "attn_k.weight"), f.get(p + "attn_k.bias")), HEADS, dh, pos, 1e4);
        double[][] v = dense(h, f.get(p + "attn_v.weight"), f.get(p + "attn_v.bias"));
        boolean full = WA_PATTERN[il] == -1;
        double[][] a = attend(q, k, v, HEADS, 1 / Math.sqrt(dh), (i, j) -> j <= i && (full || i - j <= WINDOW));
        x = add(x, dense(a, f.get(p + "attn_out.weight"), f.get(p + "attn_out.bias")));
        h = layerNorm(x, f.get(p + "ln2.weight"), f.get(p + "ln2.bias"), EPS);
        return add(x, ffn(h, f.get(p + "ffn_up.weight"), f.get(p + "ffn_up.bias"), AudioTowerSupport::geluErf,
                f.get(p + "ffn_down.weight"), f.get(p + "ffn_down.bias")));
    }

    /** Per row and quantizer the first entry maximizing {@code 2 c.r - |c|^2}; the summed code embeddings. */
    private static double[][] quantize(MmprojFixture f, double[][] x) {
        float[] cb = f.get("a.rvq.codebook.weight");
        float[] ce = f.get("mm.a.code_embd.weight");
        double[][] out = new double[x.length][HIDDEN];
        for (int t = 0; t < x.length; t++) {
            double[] r = x[t].clone();
            for (int q = 0; q < BINS.length; q++) {
                int best = 0;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int b = 0; b < BINS[q]; b++) {
                    double dot = 0;
                    double norm = 0;
                    for (int c = 0; c < HIDDEN; c++) {
                        double v = cb[c + HIDDEN * (b + CODEBOOK_ROWS * q)];
                        dot += v * r[c];
                        norm += v * v;
                    }
                    if (2 * dot - norm > bestScore) {
                        best = b;
                        bestScore = 2 * dot - norm;
                    }
                }
                for (int c = 0; c < HIDDEN; c++) {
                    r[c] -= cb[c + HIDDEN * (best + CODEBOOK_ROWS * q)];
                    out[t][c] += ce[c + HIDDEN * (best + CODE_VOCAB * q)];
                }
            }
        }
        return out;
    }

    private static double[][] localLayer(MmprojFixture f, String p, double[][] x) {
        double[][] h = rmsNorm(x, f.get(p + "ln1.weight"), null, EPS);
        int[] pos = new int[x.length];
        for (int i = 0; i < pos.length; i++) {
            pos[i] = i % GROUP;
        }
        int dh = HIDDEN / HEADS;
        double theta = MimoAudioTower.LOCAL_ROPE_THETA;
        double[][] q = ropeNeox(dense(h, f.get(p + "attn_q.weight"), f.get(p + "attn_q.bias")), HEADS, dh, pos,
                theta);
        double[][] k = ropeNeox(dense(h, f.get(p + "attn_k.weight"), f.get(p + "attn_k.bias")), HEADS, dh, pos,
                theta);
        double[][] v = dense(h, f.get(p + "attn_v.weight"), f.get(p + "attn_v.bias"));
        double[][] a = attend(q, k, v, HEADS, 1 / Math.sqrt(dh), (i, j) -> i / GROUP == j / GROUP);
        x = add(x, dense(a, f.get(p + "attn_out.weight"), null));
        h = rmsNorm(x, f.get(p + "ln2.weight"), null, EPS);
        double[][] up = dense(h, f.get(p + "ffn_up.weight"), null);
        double[][] g = map(dense(h, f.get(p + "ffn_gate.weight"), null), AudioTowerSupport::silu);
        for (int t = 0; t < up.length; t++) {
            for (int j = 0; j < up[t].length; j++) {
                up[t][j] *= g[t][j];
            }
        }
        return add(x, dense(up, f.get(p + "ffn_down.weight"), null));
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 29, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(3, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsTheStemTheDownsampleAndTheGroups() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 3; n <= 40; n++) {
                int stem = (n - 1) / 2 + 1;
                int ds = (stem - 2) / 2 + 1;
                assertEquals((ds + GROUP - 1) / GROUP, m.nTokens(n));
                assertEquals(m.nTokens(n), enc.encode(AudioTowerSupport.chunk(N_MEL, n, n)).length);
            }
        }
    }

    @Test
    public void refusesAChunkTooShortForTheDownsample() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(AudioTowerSupport.chunk(N_MEL, 2, 1)));
            assertTrue(e.getMessage(), e.getMessage().contains("mimo_audio"));
        }
    }

    @Test
    public void refusesBadTables() {
        MmprojFixture smallCodebook = fixture();
        smallCodebook.put("a.rvq.codebook.weight", -1f, 1f, HIDDEN, 4, BINS.length);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), smallCodebook.catalog()));
        assertTrue(e.getMessage(), e.getMessage().contains("codebook_size"));
        MmprojFixture noLocalBias = fixture();
        noLocalBias.remove("mm.a.local_blk.1.attn_k.bias");
        assertThrows(RuntimeException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), noLocalBias.catalog()));
        Map<String, Object> shallow = meta();
        shallow.put("clip.audio.block_count", 2);
        shallow.put("clip.audio.wa_pattern_mode", new int[]{0, -1});
        assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(shallow), fixture().catalog()));
    }
}

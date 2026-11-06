package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.add;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.attend;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.dense;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.map;
import static com.eonmux.neuralfree.gguf.mm.AudioTowerSupport.rmsNorm;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The voxtral_rt tower against a double-precision transcription of llama.cpp tools/mtmd/models/voxtral-rt.cpp
 * ({@code clip_graph_voxtral_rt}) as the first call of a stream: the causal conv stem with the zero stem tail, a
 * layer with the gated FFN and the output and down biases and a layer without them, Q and V biases, interleaved
 * RoPE at 1e6, GQA with a head width independent of the embedding width, the sliding window, the RMS trunk norm,
 * the frame stack and the GELU-erf projector; the row count and the refusals.
 */
public class VoxtralRtTowerTest {

    private static final double TOL = 2e-4;
    static final int N_MEL = 8;
    static final int HIDDEN = 8;
    static final int HEADS = 2;
    static final int KV_HEADS = 1;
    static final int HEAD_DIM = 6;
    static final int FF = 12;
    static final int LAYERS = 2;
    static final int STACK = 2;
    static final int WINDOW = 3;
    static final int ADAPTER = 10;
    static final int TEXT = 6;
    static final float EPS = 1e-5f;

    static Map<String, Object> meta() {
        Map<String, Object> m = AudioTowerSupport.meta("voxtral_rt", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, EPS);
        m.put("clip.audio.attention.head_count_kv", KV_HEADS);
        m.put("clip.audio.attention.head_dim", HEAD_DIM);
        m.put("clip.audio.projector.stack_factor", STACK);
        m.put("clip.audio.window_size", WINDOW);
        return m;
    }

    static MmprojFixture fixture() {
        MmprojFixture f = new MmprojFixture(91);
        f.weight("a.conv1d.1.weight", 3, N_MEL, HIDDEN);
        f.weight("a.conv1d.1.bias", 1, HIDDEN);
        f.weight("a.conv1d.2.weight", 3, HIDDEN, HIDDEN);
        f.weight("a.conv1d.2.bias", 1, HIDDEN);
        int qWidth = HEADS * HEAD_DIM;
        int kvWidth = KV_HEADS * HEAD_DIM;
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            f.put(p + "ln1.weight", 0.8f, 1.2f, HIDDEN);
            f.put(p + "ln2.weight", 0.8f, 1.2f, HIDDEN);
            f.weight(p + "attn_q.weight", HIDDEN, qWidth);
            f.weight(p + "attn_q.bias", qWidth);
            f.weight(p + "attn_k.weight", HIDDEN, kvWidth);
            f.weight(p + "attn_v.weight", HIDDEN, kvWidth);
            f.weight(p + "attn_v.bias", kvWidth);
            f.weight(p + "attn_out.weight", qWidth, HIDDEN);
            f.weight(p + "ffn_up.weight", HIDDEN, FF);
            f.weight(p + "ffn_down.weight", FF, HIDDEN);
            if (il == 0) {
                f.weight(p + "attn_out.bias", HIDDEN);
                f.weight(p + "ffn_gate.weight", HIDDEN, FF);
                f.weight(p + "ffn_down.bias", HIDDEN);
            }
        }
        f.put("a.post_ln.weight", 0.8f, 1.2f, HIDDEN);
        f.weight("mm.a.mlp.1.weight", (long) HIDDEN * STACK, ADAPTER);
        f.weight("mm.a.mlp.2.weight", ADAPTER, TEXT);
        return f;
    }

    static double[][] reference(MmprojFixture f, MelChunk chunk) {
        double[][] c1 = conv(AudioTowerSupport.frames(chunk), f.get("a.conv1d.1.weight"), f.get("a.conv1d.1.bias"), 1);
        double[][] tailed = new double[c1.length + 1][];
        tailed[0] = new double[HIDDEN];
        System.arraycopy(c1, 0, tailed, 1, c1.length);
        double[][] x = conv(tailed, f.get("a.conv1d.2.weight"), f.get("a.conv1d.2.bias"), 2);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            x = add(x, attention(f, p, rmsNorm(x, f.get(p + "ln1.weight"), null, EPS)));
            double[][] h = rmsNorm(x, f.get(p + "ln2.weight"), null, EPS);
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
            x = add(x, dense(up, f.get(p + "ffn_down.weight"), bias(f, p + "ffn_down.bias")));
        }
        x = rmsNorm(x, f.get("a.post_ln.weight"), null, EPS);
        double[][] stacked = new double[x.length / STACK][HIDDEN * STACK];
        for (int t = 0; t < x.length; t++) {
            System.arraycopy(x[t], 0, stacked[t / STACK], (t % STACK) * HIDDEN, HIDDEN);
        }
        double[][] h = map(dense(stacked, f.get("mm.a.mlp.1.weight"), null), AudioTowerSupport::geluErf);
        return dense(h, f.get("mm.a.mlp.2.weight"), null);
    }

    private static float[] bias(MmprojFixture f, String name) {
        return f.has(name) ? f.get(name) : null;
    }

    /**
     * {@code ggml_conv_1d(w [3, cIn, cOut], x, stride, 0, 1)} over token-major rows, plus the bias and the GELU-erf.
     */
    private static double[][] conv(double[][] x, float[] w, float[] b, int stride) {
        int cIn = x[0].length;
        int cOut = b.length;
        int n = (x.length - 3) / stride + 1;
        double[][] y = new double[n][cOut];
        for (int t = 0; t < n; t++) {
            for (int co = 0; co < cOut; co++) {
                double s = b[co];
                for (int ci = 0; ci < cIn; ci++) {
                    for (int j = 0; j < 3; j++) {
                        s += (double) w[j + 3 * (ci + cIn * co)] * x[t * stride + j][ci];
                    }
                }
                y[t][co] = AudioTowerSupport.geluErf(s);
            }
        }
        return y;
    }

    private static double[][] attention(MmprojFixture f, String p, double[][] h) {
        int[] pos = AudioOps.iota(h.length);
        double[][] q = ropeNormal(dense(h, f.get(p + "attn_q.weight"), f.get(p + "attn_q.bias")), pos);
        double[][] k = ropeNormal(dense(h, f.get(p + "attn_k.weight"), null), pos);
        double[][] v = dense(h, f.get(p + "attn_v.weight"), f.get(p + "attn_v.bias"));
        double[][] a = attend(q, repeatHeads(k), repeatHeads(v), HEADS, 1 / Math.sqrt(HEAD_DIM),
                (i, j) -> j <= i && j > i - WINDOW);
        return dense(a, f.get(p + "attn_out.weight"), bias(f, p + "attn_out.bias"));
    }

    /** {@code GGML_ROPE_TYPE_NORMAL}: dims {@code 2i, 2i + 1} of every head rotate by {@code pos * 1e6^(-2i/dh)}. */
    private static double[][] ropeNormal(double[][] x, int[] pos) {
        double[][] y = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            y[t] = x[t].clone();
            for (int h = 0; h < x[t].length / HEAD_DIM; h++) {
                for (int i = 0; i < HEAD_DIM / 2; i++) {
                    double ang = pos[t] * Math.pow(1e6, -2.0 * i / HEAD_DIM);
                    int at = h * HEAD_DIM + 2 * i;
                    y[t][at] = x[t][at] * Math.cos(ang) - x[t][at + 1] * Math.sin(ang);
                    y[t][at + 1] = x[t][at] * Math.sin(ang) + x[t][at + 1] * Math.cos(ang);
                }
            }
        }
        return y;
    }

    /** The KV heads repeated for their query-head group. */
    private static double[][] repeatHeads(double[][] kv) {
        double[][] out = new double[kv.length][HEADS * HEAD_DIM];
        for (int t = 0; t < kv.length; t++) {
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(kv[t], (h / (HEADS / KV_HEADS)) * HEAD_DIM, out[t], h * HEAD_DIM, HEAD_DIM);
            }
        }
        return out;
    }

    @Test
    public void towerMatchesReference() {
        MmprojFixture f = fixture();
        MelChunk chunk = AudioTowerSupport.chunk(N_MEL, 2 + 2 * 12, 5);
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(f, meta())) {
            float[][] out = enc.encode(chunk);
            AudioTowerSupport.assertRows(reference(f, chunk), out, TOL);
            assertEquals(6, out.length);
            assertEquals(TEXT, out[0].length);
        }
    }

    @Test
    public void rowCountIsTheStackedStrideTwoFrames() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            ClipAudioModel m = enc.model();
            for (int n = 3; n <= 40; n++) {
                assertEquals((n - 2) / 2 / STACK, m.nTokens(n));
            }
            for (int tokens = 1; tokens <= 5; tokens++) {
                int frames = 2 + 2 * STACK * tokens;
                assertEquals(tokens, enc.encode(AudioTowerSupport.chunk(N_MEL, frames, tokens)).length);
                assertEquals(tokens, m.nTokens(frames));
            }
        }
    }

    @Test
    public void refusesAChunkOfPartialTokens() {
        try (CpuAudioEncoder enc = AudioTowerSupport.encoder(fixture(), meta())) {
            for (int frames : new int[]{3, 4, 2 + 2 * 3, 2 + 2 * STACK + 1 + 2}) {
                IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                        () -> enc.encode(AudioTowerSupport.chunk(N_MEL, frames, 1)));
                assertTrue(e.getMessage(), e.getMessage().contains("voxtral_rt"));
            }
        }
    }

    @Test
    public void refusesMissingBiasesAndMismatchedHeads() {
        MmprojFixture noQBias = fixture();
        noQBias.remove("a.blk.1.attn_q.bias");
        assertThrows(RuntimeException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), noQBias.catalog()));
        MmprojFixture fullKv = fixture();
        fullKv.weight("a.blk.0.attn_k.weight", HIDDEN, HEADS * HEAD_DIM);
        assertThrows(IllegalArgumentException.class,
                () -> ClipAudioModel.from(AudioTowerSupport.hparams(meta()), fullKv.catalog()));
    }

    @Test
    public void onlyVoxtralRtTakesAFreeHeadWidth() {
        assertEquals(HEAD_DIM, AudioTowerSupport.hparams(meta()).dHead());
        Map<String, Object> other = AudioTowerSupport.meta("lfm3a", N_MEL, HIDDEN, FF, LAYERS, HEADS, TEXT, EPS);
        other.put("clip.audio.attention.head_dim", HEAD_DIM);
        other.put("clip.audio.window_size", WINDOW);
        assertThrows(GgufFormatException.class, () -> AudioTowerSupport.hparams(other));
    }
}

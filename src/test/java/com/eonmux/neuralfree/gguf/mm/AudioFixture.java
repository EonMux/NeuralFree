package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.util.HashMap;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.layerNorm;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.linear;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.orNull;
import static com.eonmux.neuralfree.gguf.mm.MmprojFixture.Reference.rmsNorm;

/**
 * Synthetic audio mmproj fixtures (the whisper family with each of its projectors, and qwen3a) plus an
 * independent double-precision transcription of llama.cpp tools/mtmd/models/whisper-enc.cpp and qwen3a.cpp over
 * the shared {@code build_vit} tower. Package-private test support of the {@code mm} audio tests.
 */
final class AudioFixture {

    static final int N_MEL = 8;
    static final int HIDDEN = 16;
    static final int N_HEAD = 2;
    static final int LAYERS = 2;
    static final int FF = 32;
    static final int TEXT = 12;
    static final int MAX_POS = 64;
    static final float EPS = 1e-5f;

    static final int Q3_MEL = 16;
    static final int Q3_HIDDEN = 12;
    static final int Q3_HEADS = 3;
    static final int Q3_C1 = 8;
    static final int Q3_C = 8;

    private AudioFixture() {
    }

    // ================================================================== fixtures

    /** A whisper-family checkpoint with the projector {@code projector} and stack factor {@code stack}. */
    static MmprojFixture whisper(String projector, int stack, long seed) {
        return whisper(projector, stack, seed, MAX_POS);
    }

    /** {@link #whisper(String, int, long)} with a position table of {@code maxPos} rows. */
    static MmprojFixture whisper(String projector, int stack, long seed, int maxPos) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("a.conv1d.1.weight", 3, N_MEL, HIDDEN);
        f.put("a.conv1d.1.bias", -0.1f, 0.1f, 1, HIDDEN);
        f.weight("a.conv1d.2.weight", 3, HIDDEN, HIDDEN);
        f.put("a.conv1d.2.bias", -0.1f, 0.1f, 1, HIDDEN);
        f.weight("a.position_embd.weight", HIDDEN, maxPos);
        f.norm("a.post_ln.weight", "a.post_ln.bias", HIDDEN);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", HIDDEN);
            f.norm(p + "ln2.weight", p + "ln2.bias", HIDDEN);
            f.weight(p + "attn_q.weight", HIDDEN, HIDDEN);
            f.weight(p + "attn_q.bias", HIDDEN);
            f.weight(p + "attn_k.weight", HIDDEN, HIDDEN);
            f.weight(p + "attn_v.weight", HIDDEN, HIDDEN);
            f.weight(p + "attn_v.bias", HIDDEN);
            f.weight(p + "attn_out.weight", HIDDEN, HIDDEN);
            f.weight(p + "attn_out.bias", HIDDEN);
            f.weight(p + "ffn_up.weight", HIDDEN, FF);
            f.weight(p + "ffn_up.bias", FF);
            f.weight(p + "ffn_down.weight", FF, HIDDEN);
            f.weight(p + "ffn_down.bias", HIDDEN);
        }
        whisperProjector(f, projector, stack);
        return f;
    }

    private static void whisperProjector(MmprojFixture f, String projector, int stack) {
        int s = Math.max(1, stack);
        switch (projector) {
            case "qwen2a":
                f.weight("mm.a.fc.weight", HIDDEN, TEXT);
                f.weight("mm.a.fc.bias", TEXT);
                break;
            case "ultravox":
                f.put("mm.a.norm_pre.weight", 0.8f, 1.2f, HIDDEN * s);
                f.weight("mm.a.mlp.1.weight", HIDDEN * s, FF);
                f.put("mm.a.norm_mid.weight", 0.8f, 1.2f, FF / 2);
                f.weight("mm.a.mlp.2.weight", FF / 2, TEXT);
                break;
            case "voxtral":
                f.weight("mm.a.mlp.1.weight", HIDDEN * s, FF);
                f.weight("mm.a.mlp.2.weight", FF, TEXT);
                break;
            case "musicflamingo":
                f.weight("mm.a.mlp.1.weight", HIDDEN, FF);
                f.weight("mm.a.mlp.1.bias", FF);
                f.weight("mm.a.mlp.2.weight", FF, TEXT);
                f.weight("mm.a.mlp.2.bias", TEXT);
                break;
            case "meralion":
                f.norm("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", HIDDEN * s);
                f.weight("mm.a.mlp.0.weight", HIDDEN * s, FF);
                f.weight("mm.a.mlp.0.bias", FF);
                f.weight("mm.a.mlp.1.weight", FF, FF);
                f.weight("mm.a.mlp.1.bias", FF);
                f.weight("mm.a.mlp.2.weight", FF, FF);
                f.weight("mm.a.mlp.2.bias", FF);
                f.weight("mm.a.mlp.3.weight", FF, TEXT);
                f.weight("mm.a.mlp.3.bias", TEXT);
                break;
            case "glma":
                f.norm("mm.a.norm_pre.weight", "mm.a.norm_pre.bias", HIDDEN);
                f.weight("mm.a.mlp.1.weight", HIDDEN * s, FF);
                f.weight("mm.a.mlp.1.bias", FF);
                f.weight("mm.a.mlp.2.weight", FF, TEXT);
                f.weight("mm.a.mlp.2.bias", TEXT);
                f.weight("v.boi", TEXT);
                f.weight("v.eoi", TEXT);
                break;
            default:
                throw new IllegalArgumentException(projector);
        }
    }

    /** A qwen3a checkpoint: three conv2d blocks, conv_out, one block and the MLP projector. */
    static MmprojFixture qwen3a(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("a.conv2d.1.weight", 3, 3, 1, Q3_C1);
        f.weight("a.conv2d.1.bias", Q3_C1);
        f.weight("a.conv2d.2.weight", 3, 3, Q3_C1, Q3_C);
        f.weight("a.conv2d.2.bias", Q3_C);
        f.weight("a.conv2d.3.weight", 3, 3, Q3_C, Q3_C);
        f.weight("a.conv2d.3.bias", Q3_C);
        f.weight("a.conv_out.weight", 2 * Q3_C, Q3_HIDDEN);
        f.weight("a.position_embd.weight", Q3_HIDDEN, ClipAudioModel.QWEN3A_TOKENS_PER_CHUNK);
        String p = "a.blk.0.";
        f.norm(p + "ln1.weight", p + "ln1.bias", Q3_HIDDEN);
        f.norm(p + "ln2.weight", p + "ln2.bias", Q3_HIDDEN);
        for (String m : new String[]{"q", "k", "v"}) {
            f.weight(p + "attn_" + m + ".weight", Q3_HIDDEN, Q3_HIDDEN);
            f.weight(p + "attn_" + m + ".bias", Q3_HIDDEN);
        }
        f.weight(p + "attn_out.weight", Q3_HIDDEN, Q3_HIDDEN);
        f.weight(p + "attn_out.bias", Q3_HIDDEN);
        f.weight(p + "ffn_up.weight", Q3_HIDDEN, FF);
        f.weight(p + "ffn_up.bias", FF);
        f.weight(p + "ffn_down.weight", FF, Q3_HIDDEN);
        f.weight(p + "ffn_down.bias", Q3_HIDDEN);
        f.weight("mm.a.mlp.1.weight", Q3_HIDDEN, FF);
        f.weight("mm.a.mlp.1.bias", FF);
        f.weight("mm.a.mlp.2.weight", FF, TEXT);
        f.weight("mm.a.mlp.2.bias", TEXT);
        return f;
    }

    /** The {@code clip.*} metadata of a whisper-family checkpoint ({@code stack} 0 leaves the key out). */
    static Map<String, Object> whisperMeta(String projector, int stack) {
        return meta(projector, stack, N_MEL, HIDDEN, N_HEAD, LAYERS);
    }

    /** The {@code clip.*} metadata of the qwen3a fixture. */
    static Map<String, Object> qwen3aMeta() {
        return meta("qwen3a", 0, Q3_MEL, Q3_HIDDEN, Q3_HEADS, 1);
    }

    private static Map<String, Object> meta(String projector, int stack, int nMel, int hidden, int heads,
                                            int layers) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_audio_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.audio.num_mel_bins", nMel);
        m.put("clip.audio.embedding_length", hidden);
        m.put("clip.audio.feed_forward_length", FF);
        m.put("clip.audio.block_count", layers);
        m.put("clip.audio.projection_dim", TEXT);
        m.put("clip.audio.attention.head_count", heads);
        m.put("clip.audio.attention.layer_norm_epsilon", EPS);
        if (stack > 0) {
            m.put("clip.audio.projector.stack_factor", stack);
        }
        return m;
    }

    static ClipHparams hparams(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.AUDIO);
    }

    /** Writes the metadata and tensors of a fixture to {@code w}; matrices in {@code matrixType}. */
    static GgufTestWriter write(GgufTestWriter w, Map<String, Object> meta, MmprojFixture f, GGMLType matrixType) {
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
            } else if (v instanceof float[]) {
                w.f32Array(e.getKey(), (float[]) v);
            } else {
                w.u32(e.getKey(), ((Number) v).longValue());
            }
        }
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            String name = e.getKey();
            long[] ne = e.getValue();
            GGMLType type = storedType(name, ne, matrixType);
            if (type == GGMLType.F16) {
                // the reference reads what the file stores
                float[] v = f.get(name);
                for (int i = 0; i < v.length; i++) {
                    v[i] = HalfFloats.halfToFloat(GgufTestWriter.floatToHalfBits(v[i]) & 0xFFFF);
                }
            }
            w.tensorFromFloats(name, type, ne, f.get(name));
        }
        return w;
    }

    /**
     * The stored type of a fixture tensor: conv kernels F16 (the converter forces them), projection matrices
     * {@code matrixType} when their rows fit its blocks, everything else F32.
     */
    static GGMLType storedType(String name, long[] ne, GGMLType matrixType) {
        if (name.contains(".conv") && name.endsWith(".weight")) {
            return GGMLType.F16;
        }
        boolean matrix = ne.length == 2 && ne[1] > 1 && !name.contains("position_embd");
        if (!matrix) {
            return GGMLType.F32;
        }
        if (matrixType == GGMLType.Q8_0 && ne[0] % 32 != 0) {
            return GGMLType.F32;
        }
        return matrixType;
    }

    /** A deterministic mel chunk in [-1, 1], mel-major. */
    static float[] mel(int nMel, int nLen, long seed) {
        java.util.Random r = new java.util.Random(seed);
        float[] v = new float[nMel * nLen];
        for (int i = 0; i < v.length; i++) {
            v[i] = r.nextFloat() * 2f - 1f;
        }
        return v;
    }

    // ================================================================== reference

    /** The whisper-family graph in double precision over a mel-major chunk of {@code nLen} frames. */
    static double[][] whisperReference(MmprojFixture f, String projector, int stack, float[] mel, int nLen) {
        double[][] x = whisperFront(f, mel, nLen);
        boolean pool = projector.equals("qwen2a") || projector.equals("voxtral") || projector.equals("musicflamingo");
        x = tower(f, x, "a", N_HEAD, pool);
        boolean stacks = projector.equals("ultravox") || projector.equals("voxtral") || projector.equals("meralion");
        if (stacks) {
            x = stack(x, stack);
        }
        switch (projector) {
            case "qwen2a":
                return dense(x, f.get("mm.a.fc.weight"), f.get("mm.a.fc.bias"), TEXT);
            case "ultravox":
                return ultravox(f, x);
            case "voxtral":
            case "musicflamingo":
                return geluMlp(f, x);
            case "meralion":
                return meralion(f, x);
            case "glma":
                return glma(f, x, stack);
            default:
                throw new IllegalArgumentException(projector);
        }
    }

    /** conv1d(stride 1) + GELU-erf, conv1d(stride 2) + GELU-erf, transpose, + position rows. */
    private static double[][] whisperFront(MmprojFixture f, float[] mel, int nLen) {
        double[][] in = new double[N_MEL][nLen];
        for (int m = 0; m < N_MEL; m++) {
            for (int t = 0; t < nLen; t++) {
                in[m][t] = mel[m * nLen + t];
            }
        }
        double[][] h1 = conv1d(in, f.get("a.conv1d.1.weight"), f.get("a.conv1d.1.bias"), 1);
        double[][] h2 = conv1d(h1, f.get("a.conv1d.2.weight"), f.get("a.conv1d.2.bias"), 2);
        int nPos = h2[0].length;
        float[] pos = f.get("a.position_embd.weight");
        double[][] x = new double[nPos][HIDDEN];
        for (int t = 0; t < nPos; t++) {
            for (int e = 0; e < HIDDEN; e++) {
                x[t][e] = h2[e][t] + pos[t * HIDDEN + e];
            }
        }
        return x;
    }

    /** {@code ggml_conv_1d_ph} (kernel 3, half padding) over channel-major input, bias, GELU-erf. */
    private static double[][] conv1d(double[][] in, float[] w, float[] b, int stride) {
        int inCh = in.length;
        int len = in[0].length;
        int outCh = b.length;
        int outLen = (len + 2 - 3) / stride + 1;
        double[][] out = new double[outCh][outLen];
        for (int oc = 0; oc < outCh; oc++) {
            for (int t = 0; t < outLen; t++) {
                double acc = b[oc];
                for (int ic = 0; ic < inCh; ic++) {
                    for (int k = 0; k < 3; k++) {
                        int src = t * stride + k - 1;
                        if (src >= 0 && src < len) {
                            acc += in[ic][src] * w[(oc * inCh + ic) * 3 + k];
                        }
                    }
                }
                out[oc][t] = geluErf(acc);
            }
        }
        return out;
    }

    /** {@code build_vit} with layer norms, separate Q/K/V, GELU-erf FFN, optional pool and post-norm. */
    static double[][] tower(MmprojFixture f, double[][] x, String prefix, int nHead, boolean pool) {
        int width = x[0].length;
        for (int il = 0; f.has(prefix + ".blk." + il + ".ln1.weight"); il++) {
            String p = prefix + ".blk." + il + ".";
            double[][] cur = new double[x.length][width];
            layerNorm(cur, x, f.get(p + "ln1.weight"), orNull(f, p + "ln1.bias"), EPS);
            double[][] attn = attention(f, p, cur, nHead);
            add(x, dense(attn, f.get(p + "attn_out.weight"), orNull(f, p + "attn_out.bias"), width));
            layerNorm(cur, x, f.get(p + "ln2.weight"), orNull(f, p + "ln2.bias"), EPS);
            double[][] up = dense(cur, f.get(p + "ffn_up.weight"), orNull(f, p + "ffn_up.bias"), FF);
            for (double[] row : up) {
                for (int i = 0; i < row.length; i++) {
                    row[i] = geluErf(row[i]);
                }
            }
            add(x, dense(up, f.get(p + "ffn_down.weight"), orNull(f, p + "ffn_down.bias"), width));
        }
        if (pool) {
            double[][] pooled = new double[(x.length - 2) / 2 + 1][width];
            for (int t = 0; t < pooled.length; t++) {
                for (int e = 0; e < width; e++) {
                    pooled[t][e] = (x[2 * t][e] + x[2 * t + 1][e]) * 0.5;
                }
            }
            x = pooled;
        }
        if (f.has(prefix + ".post_ln.weight")) {
            layerNorm(x, f.get(prefix + ".post_ln.weight"), orNull(f, prefix + ".post_ln.bias"), EPS);
        }
        return x;
    }

    private static double[][] attention(MmprojFixture f, String p, double[][] x, int nHead) {
        int t = x.length;
        int e = x[0].length;
        int dh = e / nHead;
        double[][] q = dense(x, f.get(p + "attn_q.weight"), orNull(f, p + "attn_q.bias"), e);
        double[][] k = dense(x, f.get(p + "attn_k.weight"), orNull(f, p + "attn_k.bias"), e);
        double[][] v = dense(x, f.get(p + "attn_v.weight"), orNull(f, p + "attn_v.bias"), e);
        double[][] out = new double[t][e];
        double scale = 1.0 / Math.sqrt(dh);
        for (int h = 0; h < nHead; h++) {
            int off = h * dh;
            for (int a = 0; a < t; a++) {
                double[] s = new double[t];
                double max = Double.NEGATIVE_INFINITY;
                for (int b = 0; b < t; b++) {
                    for (int d = 0; d < dh; d++) {
                        s[b] += q[a][off + d] * k[b][off + d];
                    }
                    s[b] *= scale;
                    max = Math.max(max, s[b]);
                }
                double sum = 0;
                for (int b = 0; b < t; b++) {
                    s[b] = Math.exp(s[b] - max);
                    sum += s[b];
                }
                for (int b = 0; b < t; b++) {
                    for (int d = 0; d < dh; d++) {
                        out[a][off + d] += s[b] / sum * v[b][off + d];
                    }
                }
            }
        }
        return out;
    }

    /** {@code build_stack}: the flat rows zero-padded to a multiple of {@code factor * width}, re-viewed. */
    static double[][] stack(double[][] x, int factor) {
        int width = x[0].length;
        int stride = width * factor;
        int padded = (x.length * width + stride - 1) / stride * stride;
        double[] flat = new double[padded];
        for (int t = 0; t < x.length; t++) {
            System.arraycopy(x[t], 0, flat, t * width, width);
        }
        double[][] out = new double[padded / stride][stride];
        for (int r = 0; r < out.length; r++) {
            System.arraycopy(flat, r * stride, out[r], 0, stride);
        }
        return out;
    }

    private static double[][] ultravox(MmprojFixture f, double[][] x) {
        rmsNorm(x, f.get("mm.a.norm_pre.weight"), 1e-6f);
        double[][] h = dense(x, f.get("mm.a.mlp.1.weight"), null, FF);
        int half = FF / 2;
        double[][] g = new double[x.length][half];
        for (int r = 0; r < x.length; r++) {
            for (int i = 0; i < half; i++) {
                g[r][i] = silu(h[r][half + i]) * h[r][i];
            }
        }
        rmsNorm(g, f.get("mm.a.norm_mid.weight"), 1e-6f);
        return dense(g, f.get("mm.a.mlp.2.weight"), null, TEXT);
    }

    private static double[][] geluMlp(MmprojFixture f, double[][] x) {
        double[][] h = dense(x, f.get("mm.a.mlp.1.weight"), orNull(f, "mm.a.mlp.1.bias"), FF);
        for (double[] row : h) {
            for (int i = 0; i < row.length; i++) {
                row[i] = geluErf(row[i]);
            }
        }
        return dense(h, f.get("mm.a.mlp.2.weight"), orNull(f, "mm.a.mlp.2.bias"), TEXT);
    }

    private static double[][] meralion(MmprojFixture f, double[][] x) {
        layerNorm(x, f.get("mm.a.norm_pre.weight"), f.get("mm.a.norm_pre.bias"), EPS);
        double[][] h0 = dense(x, f.get("mm.a.mlp.0.weight"), f.get("mm.a.mlp.0.bias"), FF);
        for (double[] row : h0) {
            for (int i = 0; i < row.length; i++) {
                row[i] = silu(row[i]);
            }
        }
        double[][] gate = dense(h0, f.get("mm.a.mlp.1.weight"), f.get("mm.a.mlp.1.bias"), FF);
        double[][] pool = dense(h0, f.get("mm.a.mlp.2.weight"), f.get("mm.a.mlp.2.bias"), FF);
        for (int r = 0; r < gate.length; r++) {
            for (int i = 0; i < FF; i++) {
                gate[r][i] = silu(gate[r][i]) * pool[r][i];
            }
        }
        return dense(gate, f.get("mm.a.mlp.3.weight"), f.get("mm.a.mlp.3.bias"), TEXT);
    }

    private static double[][] glma(MmprojFixture f, double[][] x, int stack) {
        layerNorm(x, f.get("mm.a.norm_pre.weight"), f.get("mm.a.norm_pre.bias"), EPS);
        double[][] body = geluMlp(f, stack(x, stack));
        double[][] out = new double[body.length + 2][];
        out[0] = toDouble(f.get("v.boi"));
        System.arraycopy(body, 0, out, 1, body.length);
        out[body.length + 1] = toDouble(f.get("v.eoi"));
        return out;
    }

    /** The qwen3a graph in double precision over a mel-major chunk of {@code nLen} frames. */
    static double[][] qwen3aReference(MmprojFixture f, float[] mel, int nLen) {
        int batch = ClipAudioModel.QWEN3A_CHUNK_FRAMES;
        int per = ClipAudioModel.QWEN3A_TOKENS_PER_CHUNK;
        int nBatches = nLen / batch;
        double[][] x = new double[nBatches * per][];
        float[] pos = f.get("a.position_embd.weight");
        for (int b = 0; b < nBatches; b++) {
            double[][][] img = new double[1][Q3_MEL][batch];
            for (int m = 0; m < Q3_MEL; m++) {
                for (int t = 0; t < batch; t++) {
                    img[0][m][t] = mel[m * nLen + b * batch + t];
                }
            }
            img = conv2d(img, f.get("a.conv2d.1.weight"), f.get("a.conv2d.1.bias"));
            img = conv2d(img, f.get("a.conv2d.2.weight"), f.get("a.conv2d.2.bias"));
            img = conv2d(img, f.get("a.conv2d.3.weight"), f.get("a.conv2d.3.bias"));
            int oh = img[0].length;
            int ow = img[0][0].length;
            double[][] feats = new double[ow][oh * Q3_C];
            for (int t = 0; t < ow; t++) {
                for (int fr = 0; fr < oh; fr++) {
                    for (int c = 0; c < Q3_C; c++) {
                        feats[t][fr + oh * c] = img[c][fr][t];
                    }
                }
            }
            double[][] proj = dense(feats, f.get("a.conv_out.weight"), orNull(f, "a.conv_out.bias"), Q3_HIDDEN);
            for (int p = 0; p < per; p++) {
                for (int e = 0; e < Q3_HIDDEN; e++) {
                    proj[p][e] += pos[p * Q3_HIDDEN + e];
                }
                x[b * per + p] = proj[p];
            }
        }
        x = tower(f, x, "a", Q3_HEADS, false);
        return geluMlp(f, x);
    }

    /** conv2d 3x3, stride 2, padding 1 over {@code img[c][y=mel][x=time]}, bias, GELU-erf. */
    private static double[][][] conv2d(double[][][] img, float[] w, float[] b) {
        int cIn = img.length;
        int h = img[0].length;
        int wd = img[0][0].length;
        int oh = (h + 2 - 3) / 2 + 1;
        int ow = (wd + 2 - 3) / 2 + 1;
        double[][][] out = new double[b.length][oh][ow];
        for (int co = 0; co < b.length; co++) {
            for (int y = 0; y < oh; y++) {
                for (int x = 0; x < ow; x++) {
                    double acc = b[co];
                    for (int ci = 0; ci < cIn; ci++) {
                        for (int ky = 0; ky < 3; ky++) {
                            for (int kx = 0; kx < 3; kx++) {
                                int sy = 2 * y + ky - 1;
                                int sx = 2 * x + kx - 1;
                                if (sy >= 0 && sy < h && sx >= 0 && sx < wd) {
                                    acc += img[ci][sy][sx] * w[((co * cIn + ci) * 3 + ky) * 3 + kx];
                                }
                            }
                        }
                    }
                    out[co][y][x] = geluErf(acc);
                }
            }
        }
        return out;
    }

    // ================================================================== math

    static double[][] dense(double[][] x, float[] w, float[] b, int rows) {
        double[][] y = new double[x.length][rows];
        linear(x, w, b, y);
        return y;
    }

    private static void add(double[][] x, double[][] y) {
        for (int t = 0; t < x.length; t++) {
            for (int c = 0; c < x[t].length; c++) {
                x[t][c] += y[t][c];
            }
        }
    }

    static double geluErf(double v) {
        return MmprojFixture.Reference.unary(ClipFfnOp.GELU_ERF, v);
    }

    static double silu(double v) {
        return v / (1.0 + Math.exp(-v));
    }

    private static double[] toDouble(float[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i];
        }
        return out;
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * A synthetic Gemma 3n mmproj (the gemma3na audio tower, optionally beside the gemma3nv vision keys) laid out as
 * the converter writes it: the sub-sample conv kernels in F32, {@code a.pre_encode.out.weight} in F16, the other
 * matrices in F32, the depthwise kernels as {@code [k, 1, hidden]}. Small widths (128 mel bins, conv channels 8
 * and 4, hidden 16 over 2 heads, 2 blocks, text width 12); the reference rows of {@code gemma3na-reference.txt}
 * come from Hugging Face transformers over this file. Package-private test support.
 */
final class Gemma3nAudioFixture {

    static final int N_MEL = 128;
    static final int C0 = 8;
    static final int C1 = 4;
    static final int HIDDEN = 16;
    static final int HEADS = 2;
    static final int LAYERS = 2;
    static final int FF = 4 * HIDDEN;
    static final int KERNEL = 5;
    static final int VOCAB = 128;
    static final int TEXT = 12;
    static final int SAMPLE_RATE = 16000;

    private Gemma3nAudioFixture() {
    }

    /** The tower, conformer and embedder tensors of seed {@code seed}. */
    static MmprojFixture tensors(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        int f1 = Gemma3nAudioModel.convOut(Gemma3nAudioModel.convOut(N_MEL));
        f.weight("a.conv1d.0.weight", 3, 3, 1, C0);
        f.put("a.conv1d.0.norm.weight", 0.8f, 1.2f, C0);
        f.put("a.conv1d.1.weight", -0.3f, 0.3f, 3, 3, C0, C1);
        f.put("a.conv1d.1.norm.weight", 0.8f, 1.2f, C1);
        f.put("a.pre_encode.out.weight", -0.2f, 0.2f, (long) f1 * C1, HIDDEN);
        for (int il = 0; il < LAYERS; il++) {
            String p = "a.blk." + il + ".";
            for (String norm : new String[]{"ffn_norm", "ffn_post_norm", "ln1", "ln2", "conv_norm", "norm_conv",
                    "ffn_norm_1", "ffn_post_norm_1", "layer_pre_norm"}) {
                f.put(p + norm + ".weight", 0.8f, 1.2f, HIDDEN);
            }
            for (String suffix : new String[]{"", "_1"}) {
                f.weight(p + "ffn_up" + suffix + ".weight", HIDDEN, FF);
                f.put(p + "ffn_down" + suffix + ".weight", -0.25f, 0.25f, FF, HIDDEN);
            }
            for (String square : new String[]{"attn_q", "attn_k", "attn_v", "attn_out", "linear_pos", "conv_pw2"}) {
                f.weight(p + square + ".weight", HIDDEN, HIDDEN);
            }
            f.put(p + "per_dim_scale", -0.5f, 0.5f, HIDDEN / HEADS);
            f.weight(p + "conv_pw1.weight", HIDDEN, 2 * HIDDEN);
            f.weight(p + "conv_dw.weight", KERNEL, 1, HIDDEN);
        }
        f.weight("mm.a.embedding.weight", HIDDEN, VOCAB);
        f.put("mm.a.hard_emb_norm.weight", 0.8f, 1.2f, HIDDEN);
        f.put("mm.a.soft_emb_norm.weight", 0.8f, 1.2f, HIDDEN);
        f.weight("mm.a.input_projection.weight", HIDDEN, TEXT);
        return f;
    }

    /** The audio keys the converter writes ({@code clip.audio.projector_type gemma3na}, eps 1e-5). */
    static Map<String, Object> meta() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_audio_encoder", true);
        m.put("clip.audio.projector_type", "gemma3na");
        m.put("clip.audio.num_mel_bins", N_MEL);
        m.put("clip.audio.embedding_length", HIDDEN);
        m.put("clip.audio.feed_forward_length", FF);
        m.put("clip.audio.block_count", LAYERS);
        m.put("clip.audio.projection_dim", TEXT);
        m.put("clip.audio.attention.head_count", HEADS);
        m.put("clip.audio.attention.layer_norm_epsilon", 1e-5f);
        return m;
    }

    /** {@link #meta} plus the gemma3nv vision keys of the converter (a MobileNetV5 tower NeuralFree does not run). */
    static Map<String, Object> withVisionMeta(Map<String, Object> audio) {
        Map<String, Object> m = new HashMap<>(audio);
        m.put("clip.has_vision_encoder", true);
        m.put("clip.vision.projector_type", "gemma3nv");
        m.put("clip.vision.image_size", 768);
        m.put("clip.vision.patch_size", 3);
        m.put("clip.vision.embedding_length", 2048);
        m.put("clip.vision.feed_forward_length", 8192);
        m.put("clip.vision.block_count", 128);
        m.put("clip.vision.projection_dim", TEXT);
        m.put("clip.vision.attention.head_count", 8);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        m.put("clip.vision.image_mean", new float[]{0f, 0f, 0f});
        m.put("clip.vision.image_std", new float[]{1f, 1f, 1f});
        return m;
    }

    static ClipHparams hparams(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.AUDIO);
    }

    /**
     * Writes {@code f} under {@code meta} to a temporary GGUF: {@code a.pre_encode.out.weight} in F16 (its values
     * in {@code f} rounded to what the file stores), every other tensor in F32.
     */
    static Path write(Map<String, Object> meta, MmprojFixture f) throws IOException {
        GgufTestWriter w = new GgufTestWriter();
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
            GGMLType type = name.equals("a.pre_encode.out.weight") ? GGMLType.F16 : GGMLType.F32;
            float[] v = f.get(name);
            if (type == GGMLType.F16) {
                for (int i = 0; i < v.length; i++) {
                    v[i] = HalfFloats.halfToFloat(GgufTestWriter.floatToHalfBits(v[i]) & 0xFFFF);
                }
            }
            w.tensorFromFloats(name, type, e.getValue(), v);
        }
        return w.writeTempFile();
    }

    /**
     * {@code seconds} of deterministic 16 kHz audio: a gliding tone, a fixed harmonic pair and low noise, with a
     * silent stretch in the middle.
     */
    static float[] waveform(double seconds, long seed) {
        int n = (int) Math.round(seconds * SAMPLE_RATE);
        Random r = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            double t = (double) i / SAMPLE_RATE;
            double glide = Math.sin(2 * Math.PI * (220 * t + 40 * t * t));
            double pair = 0.5 * Math.sin(2 * Math.PI * 950 * t) + 0.25 * Math.sin(2 * Math.PI * 3100 * t);
            double gate = (i / (SAMPLE_RATE / 4)) % 5 == 3 ? 0.0 : 1.0;
            v[i] = (float) (0.3 * gate * (glide + pair) + 0.01 * (r.nextFloat() * 2 - 1));
        }
        return v;
    }
}

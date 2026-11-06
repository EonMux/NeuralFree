package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Synthetic mmproj GGUF files whose projector output width matches a given text model, shaped like the fixtures
 * of the {@code mm} tests: a llava CLIP with its MLP projector (8x8 image, 4x4 patches, 4 rows per image), a
 * SigLIP with the Gemma 3 pool projector (16x16 image, 4 rows), a Qwen2-VL merger with its mm.0/mm.2 MLP (8x8
 * image, 2x2 patches, merge 2, a 2x2 token grid), a whisper audio tower with the qwen2a projector, and a
 * Qwen2.5-Omni file holding both kinds of encoder under the legacy {@code qwen2.5o} type. Test support shared with
 * the server tests.
 */
public final class MmprojTestFiles {

    private static final int E = 8;
    private static final int FF = 12;
    /** The whisper position table: a 3000-frame mel window halves to 1500 positions in the conv front end. */
    private static final int WHISPER_POSITIONS = 1500;

    private MmprojTestFiles() {
    }

    /** A llava ({@code mlp}) projector writing rows of {@code width}. */
    public static Path llava(int width) throws IOException {
        MmprojFixture f = new MmprojFixture(11);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.class_embd", E, 1);
        f.weight("v.position_embd.weight", E, 5);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, 2, false);
        f.weight("mm.0.weight", E, width);
        f.weight("mm.0.bias", width);
        f.weight("mm.2.weight", width, width);
        f.weight("mm.2.bias", width);
        Map<String, Object> meta = meta("mlp", 8, 4, 2, width);
        meta.put("clip.use_gelu", true);
        return VisionFamilyFixtures.gguf(f, meta).writeTempFile();
    }

    /** A SigLIP projector with the Gemma 3 average-pool head writing rows of {@code width}. */
    public static Path gemma3(int width) throws IOException {
        MmprojFixture f = new MmprojFixture(12);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, 2, true);
        f.weight("mm.input_projection.weight", E, width);
        f.weight("mm.soft_emb_norm.weight", E);
        Map<String, Object> meta = meta("gemma3", 16, 4, 2, width);
        meta.put("clip.vision.projector.scale_factor", 2);
        return VisionFamilyFixtures.gguf(f, meta).writeTempFile();
    }

    /** A Qwen2-VL merger projector (the {@code mm.0} / {@code mm.2} MLP) writing rows of {@code width}. */
    public static Path qwen2vl(int width) throws IOException {
        MmprojFixture f = qwenVision(13, false, width);
        Map<String, Object> meta = meta("qwen2vl_merger", 8, 2, 3, width);
        meta.put("clip.use_silu", true);
        meta.put("clip.vision.spatial_merge_size", 2);
        return VisionFamilyFixtures.gguf(f, meta).writeTempFile();
    }

    /**
     * A whisper audio tower with the qwen2a projector writing rows of {@code width}: each 30 s window of 16 kHz
     * audio encodes to 750 rows.
     */
    public static Path qwen2aAudio(int width) throws IOException {
        MmprojFixture f = AudioFixture.whisper("qwen2a", 0, 14, WHISPER_POSITIONS);
        f.weight("mm.a.fc.weight", AudioFixture.HIDDEN, width);
        f.weight("mm.a.fc.bias", width);
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen2a", 0);
        meta.put("clip.audio.projection_dim", width);
        return AudioFixture.write(new GgufTestWriter(), meta, f, GGMLType.F32).writeTempFile();
    }

    /**
     * A Qwen2.5-Omni file as its converter writes it: the common {@code clip.projector_type qwen2.5o} over a
     * Qwen2.5-VL vision encoder (both temporal patch kernels, split Q/K/V, RMS norms, a full-attention block
     * after every windowed one, the mm.0 / mm.2 merger; 8x8 image, 2x2 patches, merge 2) and the whisper audio
     * tower of {@link #qwen2aAudio}, both writing rows of {@code width}.
     */
    public static Path qwen25Omni(int width) throws IOException {
        MmprojFixture audio = AudioFixture.whisper("qwen2a", 0, 15, WHISPER_POSITIONS);
        audio.weight("mm.a.fc.weight", AudioFixture.HIDDEN, width);
        audio.weight("mm.a.fc.bias", width);
        Map<String, Object> meta = AudioFixture.whisperMeta("qwen2.5o", 0);
        meta.put("clip.audio.projection_dim", width);
        meta.putAll(meta("qwen2.5o", 8, 2, 3, width));
        meta.put("clip.use_silu", true);
        meta.put("clip.vision.spatial_merge_size", 2);
        meta.put("clip.vision.n_wa_pattern", 2);
        GgufTestWriter w = AudioFixture.write(new GgufTestWriter(), meta, audio, GGMLType.F32);
        MmprojFixture vision = qwenVision(16, true, width);
        for (Map.Entry<String, long[]> e : vision.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), GGMLType.F32, e.getValue(), vision.get(e.getKey()));
        }
        return w.writeTempFile();
    }

    /**
     * A three-block Qwen2-VL vision encoder and merger of output {@code width}: both Conv3d patch kernels (the
     * temporal pair the converter writes for the qwen2vl encoder), fused QKV and layer-norm post norm, or with
     * {@code qwen25} the Qwen2.5-VL converter's layout (Q/K/V split, RMS post norm).
     */
    private static MmprojFixture qwenVision(long seed, boolean qwen25, int width) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        if (qwen25) {
            f.weight("v.post_ln.weight", E);
        } else {
            f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        }
        for (int il = 0; il < 3; il++) {
            String p = "v.blk." + il + ".";
            f.weight(p + "ln1.weight", E);
            f.weight(p + "ln2.weight", E);
            if (qwen25) {
                for (String qkv : new String[]{"attn_q", "attn_k", "attn_v"}) {
                    f.weight(p + qkv + ".weight", E, E);
                    f.weight(p + qkv + ".bias", E);
                }
            } else {
                f.weight(p + "attn_qkv.weight", E, 3 * E);
                f.weight(p + "attn_qkv.bias", 3 * E);
            }
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            gatedFfn(f, p);
        }
        f.weight("mm.0.weight", 4 * E, width);
        f.weight("mm.0.bias", width);
        f.weight("mm.2.weight", width, width);
        f.weight("mm.2.bias", width);
        return f;
    }

    private static Map<String, Object> meta(String projector, int image, int patch, int layers, int width) {
        Map<String, Object> m = new HashMap<>(VisionFamilyFixtures.meta(projector, image, patch, width));
        m.put("clip.vision.block_count", layers);
        m.put("clip.vision.attention.head_count_kv", 2);
        return m;
    }

    /** CLIP / SigLIP blocks: layer norms with biases, fused QKV, an ungated (or gated) FFN. */
    private static void layers(MmprojFixture f, int count, boolean gated) {
        for (int il = 0; il < count; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", E);
            f.norm(p + "ln2.weight", p + "ln2.bias", E);
            f.weight(p + "attn_qkv.weight", E, 3 * E);
            f.weight(p + "attn_qkv.bias", 3 * E);
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            if (gated) {
                gatedFfn(f, p);
            } else {
                f.weight(p + "ffn_up.weight", E, FF);
                f.weight(p + "ffn_up.bias", FF);
                f.weight(p + "ffn_down.weight", FF, E);
                f.weight(p + "ffn_down.bias", E);
            }
        }
    }

    private static void gatedFfn(MmprojFixture f, String p) {
        f.weight(p + "ffn_up.weight", E, FF);
        f.weight(p + "ffn_up.bias", FF);
        f.weight(p + "ffn_gate.weight", E, FF);
        f.weight(p + "ffn_gate.bias", FF);
        f.weight(p + "ffn_down.weight", FF, E);
        f.weight(p + "ffn_down.bias", E);
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * Synthetic mmproj tensors and metadata of the minicpmv, internvl, pixtral, idefics3, llama4 and phi4 families,
 * shaped as their converters write them (llama.cpp convert_hf_to_gguf.py, legacy-models/minicpmv-*), small
 * enough for a double-precision reference. Package-private test support of the {@code mm} tests.
 */
final class VisionFamilyFixtures {

    static final int E = 8;
    static final int FF = 12;
    static final int HEADS = 2;
    static final int LAYERS = 2;
    /** Resampler width: one multiple of the fixed 128-wide resampler head, two heads. */
    static final int RESAMPLER_D = 256;
    static final int QUERIES = 3;

    private VisionFamilyFixtures() {
    }

    /** The common clip metadata of one family. */
    static Map<String, Object> meta(String projector, int imageSize, int patch, int projectionDim) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_vision_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.vision.image_size", imageSize);
        m.put("clip.vision.patch_size", patch);
        m.put("clip.vision.embedding_length", E);
        m.put("clip.vision.feed_forward_length", FF);
        m.put("clip.vision.block_count", LAYERS);
        m.put("clip.vision.projection_dim", projectionDim);
        m.put("clip.vision.attention.head_count", HEADS);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        m.put("clip.vision.image_mean", new float[]{0.5f, 0.5f, 0.5f});
        m.put("clip.vision.image_std", new float[]{0.5f, 0.5f, 0.5f});
        return m;
    }

    static ClipHparams hparams(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.VISION);
    }

    // ------------------------------------------------------------------ encoder pieces

    /** {@link #LAYERS} blocks of width {@link #E}: fused or separate Q/K/V, the optional biases, gate and scales. */
    static void layers(MmprojFixture f, boolean fused, boolean biases, boolean normBias, boolean gated,
                       boolean layerScale) {
        for (int il = 0; il < LAYERS; il++) {
            String p = "v.blk." + il + ".";
            if (normBias) {
                f.norm(p + "ln1.weight", p + "ln1.bias", E);
                f.norm(p + "ln2.weight", p + "ln2.bias", E);
            } else {
                f.put(p + "ln1.weight", 0.8f, 1.2f, E);
                f.put(p + "ln2.weight", 0.8f, 1.2f, E);
            }
            if (fused) {
                f.weight(p + "attn_qkv.weight", E, 3 * E);
                if (biases) {
                    f.weight(p + "attn_qkv.bias", 3 * E);
                }
            } else {
                for (String n : new String[]{"attn_q", "attn_k", "attn_v"}) {
                    f.weight(p + n + ".weight", E, E);
                    if (biases) {
                        f.weight(p + n + ".bias", E);
                    }
                }
            }
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "ffn_up.weight", E, FF);
            f.weight(p + "ffn_down.weight", FF, E);
            if (gated) {
                f.weight(p + "ffn_gate.weight", E, FF);
            }
            if (biases) {
                f.weight(p + "attn_out.bias", E);
                f.weight(p + "ffn_up.bias", FF);
                f.weight(p + "ffn_down.bias", E);
                if (gated) {
                    f.weight(p + "ffn_gate.bias", FF);
                }
            }
            if (layerScale) {
                f.put(p + "ls1.weight", 0.2f, 1.5f, E);
                f.put(p + "ls2.weight", 0.2f, 1.5f, E);
            }
        }
    }

    // ------------------------------------------------------------------ families

    /** MiniCPM-V: SigLIP ViT (4x4 patches, 70x70 bucket rows), resampler of width 256 over 3 queries. */
    static MmprojFixture minicpmv(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, MiniCpmvGraph.POS_BUCKETS * MiniCpmvGraph.POS_BUCKETS);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        int d = RESAMPLER_D;
        f.weight("resampler.pos_embed_k", d, 2);
        f.put("resampler.query", -1f, 1f, d, QUERIES);
        f.put("resampler.kv.weight", -0.3f, 0.3f, E, d);
        for (String p : new String[]{"q", "k", "v", "out"}) {
            f.put("resampler.attn." + p + ".weight", -0.08f, 0.08f, d, d);
            f.put("resampler.attn." + p + ".bias", -0.1f, 0.1f, d);
        }
        for (String p : new String[]{"q", "kv", "post"}) {
            f.norm("resampler.ln_" + p + ".weight", "resampler.ln_" + p + ".bias", d);
        }
        f.put("resampler.proj.weight", -0.08f, 0.08f, d, d);
        return f;
    }

    static Map<String, Object> minicpmvMeta() {
        // the converter writes projection_dim 0: the width is the resampler's
        Map<String, Object> m = meta("resampler", 8, 4, 0);
        m.put("clip.use_gelu", true);
        m.put("clip.minicpmv_version", 4);
        m.put("clip.minicpmv_query_num", QUERIES);
        return m;
    }

    /** InternVL: InternViT (2x2 patches over an 8x8 tile, layer scale), scale factor 2, GELU MLP to 6. */
    static MmprojFixture internvl(long seed, String qkNorm) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.class_embd", E, 1);
        f.weight("v.position_embd.weight", E, 17);
        layers(f, "per_head".equals(qkNorm), true, true, false, true);
        if (qkNorm != null) {
            int width = "per_head".equals(qkNorm) ? E / HEADS : E;
            for (int il = 0; il < LAYERS; il++) {
                f.put("v.blk." + il + ".attn_q_norm.weight", 0.8f, 1.2f, width);
                f.put("v.blk." + il + ".attn_k_norm.weight", 0.8f, 1.2f, width);
            }
        }
        f.norm("mm.model.mlp.0.weight", "mm.model.mlp.0.bias", 4 * E);
        f.weight("mm.model.mlp.1.weight", 4 * E, 10);
        f.weight("mm.model.mlp.1.bias", 10);
        f.weight("mm.model.mlp.3.weight", 10, 6);
        f.weight("mm.model.mlp.3.bias", 6);
        return f;
    }

    static Map<String, Object> internvlMeta() {
        Map<String, Object> m = meta("internvl", 8, 2, 6);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** Pixtral: RMS ViT (2x2 patches), SiLU-gated FFN, optional 2x2 patch merger, GELU MLP to 6, IMG_BREAK. */
    static MmprojFixture pixtral(long seed, boolean merger) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.pre_ln.weight", 0.8f, 1.2f, E);
        layers(f, false, false, false, true, false);
        if (merger) {
            f.put("mm.input_norm.weight", 0.8f, 1.2f, E);
            f.weight("mm.patch_merger.weight", 4 * E, E);
            f.weight("mm.1.bias", 10);
            f.weight("mm.2.bias", 6);
        }
        f.weight("mm.1.weight", E, 10);
        f.weight("mm.2.weight", 10, 6);
        f.weight("v.token_embd.img_break", 6);
        return f;
    }

    static Map<String, Object> pixtralMeta(boolean merger) {
        Map<String, Object> m = meta("pixtral", 16, 2, 6);
        m.put("clip.use_silu", true);
        if (merger) {
            m.put("clip.vision.spatial_merge_size", 2);
        }
        return m;
    }

    /** Idefics3: SigLIP ViT (4x4 patches over a 16x16 tile), scale factor 2, linear projection to 6. */
    static MmprojFixture idefics3(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, true, true, true, false, false);
        f.weight("mm.model.fc.weight", 4 * E, 6);
        return f;
    }

    static Map<String, Object> idefics3Meta() {
        Map<String, Object> m = meta("idefics3", 16, 4, 6);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** Llama 4: unfold-linear patches (2x2), class token, pre/post norm, scale factor 2, two GELU layers. */
    static MmprojFixture llama4(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 3 * 2 * 2, E);
        f.weight("v.class_embd", E);
        f.weight("v.position_embd.weight", E, 17);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.weight("mm.model.mlp.1.weight", 4 * E, 10);
        f.weight("mm.model.mlp.2.weight", 10, 10);
        f.weight("mm.model.fc.weight", 10, 6);
        return f;
    }

    static Map<String, Object> llama4Meta() {
        Map<String, Object> m = meta("llama4", 8, 2, 6);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** Phi-4: SigLIP ViT (4x4 patches, a 4x4 position grid, no post norm), GELU MLP to 6. */
    static MmprojFixture phi4(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        layers(f, true, true, true, false, false);
        f.weight("mm.0.weight", E, 10);
        f.weight("mm.0.bias", 10);
        f.weight("mm.2.weight", 10, 6);
        f.weight("mm.2.bias", 6);
        return f;
    }

    static Map<String, Object> phi4Meta() {
        Map<String, Object> m = meta("phi4", 16, 4, 6);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.image_min_pixels", 64);
        m.put("clip.vision.image_max_pixels", 4096);
        return m;
    }

    // ------------------------------------------------------------------ binding and files

    static CpuVisionEncoder encoder(MmprojFixture f, ClipHparams hp) {
        return CpuVisionEncoder.bind(ClipVisionModel.from(hp, f.catalog()),
                com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec.PLAIN, f::get, f::ref);
    }

    /** The fixture as an F32 mmproj GGUF writer with the metadata of {@code meta}. */
    static GgufTestWriter gguf(MmprojFixture f, Map<String, Object> meta) {
        return gguf(f, meta, false);
    }

    /**
     * The fixture as an mmproj GGUF writer with the metadata of {@code meta}; with {@code f16Matrices} every tensor
     * of rank 2 or more is stored as F16, the vectors as F32.
     */
    static GgufTestWriter gguf(MmprojFixture f, Map<String, Object> meta, boolean f16Matrices) {
        return gguf(f, meta, (name, shape) -> f16Matrices && shape.length >= 2 ? GGMLType.F16 : GGMLType.F32);
    }

    /** The fixture as an mmproj GGUF writer with the metadata of {@code meta}, each tensor stored as {@code type}. */
    static GgufTestWriter gguf(MmprojFixture f, Map<String, Object> meta,
                               BiFunction<String, long[], GGMLType> type) {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        for (Map.Entry<String, Object> e : meta.entrySet()) {
            String k = e.getKey();
            Object v = e.getValue();
            if (k.equals("general.architecture")) {
                continue;
            }
            if (v instanceof Boolean) {
                w.bool(k, (Boolean) v);
            } else if (v instanceof String) {
                w.str(k, (String) v);
            } else if (v instanceof Float) {
                w.f32(k, (Float) v);
            } else if (v instanceof float[]) {
                w.f32Array(k, (float[]) v);
            } else if (v instanceof int[]) {
                w.i32Array(k, (int[]) v);
            } else if (k.equals("clip.minicpmv_version")) {
                w.i32(k, (Integer) v);
            } else {
                w.u32(k, (Integer) v);
            }
        }
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            w.tensorFromFloats(e.getKey(), type.apply(e.getKey(), e.getValue()), e.getValue(), f.get(e.getKey()));
        }
        return w;
    }
}

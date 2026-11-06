package com.eonmux.neuralfree.gguf.mm;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.layers;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.meta;

/**
 * Synthetic mmproj tensors and metadata of the glm5v, lfm3vl, ling3vl, minimax_m3, muse-glimmer, nemotron_v2_vl,
 * nemotron_omni and exaone4_5 families, shaped as llama.cpp clip.cpp {@code load_tensors} reads them and carrying
 * every key its {@code load_hparams} requires, so that the same file loads in both engines. {@link #cases()}
 * lists one image per family; the reference rows of those images, encoded by llama.cpp's clip, are the test
 * resource {@link #REFERENCE_RESOURCE}. Every projector ends at width {@link #OUT}. Package-private test support of
 * the {@code mm} tests.
 */
final class VisionFamiliesFFixtures {

    /** Output width of every projector. */
    static final int OUT = 6;
    /** Hidden width of the projector MLPs. */
    static final int HID = 10;
    /** The llama.cpp reference rows of {@link #cases()}, next to this class. */
    static final String REFERENCE_RESOURCE = "vision-families-f-reference.txt";

    /** MiniMax-M3 tower width: the head must hold three rotated thirds of at least one pair each. */
    static final int MINIMAX_E = 24;
    static final int MINIMAX_FF = 32;
    /** LFM3-VL blocks: a conv block with the spatial kernel, an attention block, a conv block without it. */
    static final int LFM3VL_LAYERS = 3;
    /** Nemotron register rows prepended to the patch rows. */
    static final int NEMOTRON_REGISTERS = 2;
    /** EXAONE 4.5 patch side: 112-pixel windows are then 4 merged tokens per side. */
    static final int EXAONE_PATCH = 14;
    /** The GLM-5V SwiGLU clamp, low enough that the fixture's gates and ups reach it. */
    static final float GLM5V_SWIGLU_CLAMP = 0.6f;

    private VisionFamiliesFFixtures() {
    }

    /** One family's fixture, metadata and reference image. */
    static final class Case {
        final String name;
        final ProjectorType type;
        final MmprojFixture fixture;
        final Map<String, Object> meta;
        final int width;
        final int height;
        final long pixelSeed;

        Case(String name, ProjectorType type, MmprojFixture fixture, Map<String, Object> meta, int width, int height,
             long pixelSeed) {
            this.name = name;
            this.type = type;
            this.fixture = fixture;
            this.meta = Collections.unmodifiableMap(meta);
            this.width = width;
            this.height = height;
            this.pixelSeed = pixelSeed;
        }

        /** The normalized pixels of the reference image. */
        float[] pixels() {
            return MmprojFixture.pixels(width, height, pixelSeed);
        }
    }

    /** One case per family, in the order of the reference resource. */
    static List<Case> cases() {
        return Arrays.asList(
                new Case("glm5v", ProjectorType.GLM5V, glm5v(901), glm5vMeta(), 8, 12, 911),
                new Case("lfm3vl", ProjectorType.LFM3VL, lfm3vl(902), lfm3vlMeta(), 8, 12, 912),
                new Case("ling3vl", ProjectorType.LING3VL, ling3vl(903), ling3vlMeta(), 8, 12, 913),
                new Case("minimax_m3", ProjectorType.MINIMAX_M3, minimaxM3(904), minimaxM3Meta(), 8, 12, 914),
                new Case("muse-glimmer", ProjectorType.MUSE_GLIMMER, museGlimmer(905), museGlimmerMeta(), 12, 8,
                        915),
                new Case("nemotron_v2_vl", ProjectorType.NEMOTRON_V2_VL, nemotronV2Vl(906), nemotronV2VlMeta(), 8, 8,
                        916),
                new Case("nemotron_omni", ProjectorType.NEMOTRON_OMNI, nemotronOmni(907), nemotronOmniMeta(), 12, 8,
                        917),
                new Case("exaone4_5", ProjectorType.EXAONE4_5, exaone45(908), exaone45Meta(), 168, 112, 918));
    }

    /** The case named {@code name}. */
    static Case byName(String name) {
        for (Case c : cases()) {
            if (c.name.equals(name)) {
                return c;
            }
        }
        throw new IllegalArgumentException("no vision-families-f case named " + name);
    }

    private static void mlp(MmprojFixture f, String up, String down, int in, boolean biases) {
        f.weight(up + ".weight", in, HID);
        f.weight(down + ".weight", HID, OUT);
        if (biases) {
            f.weight(up + ".bias", HID);
            f.weight(down + ".bias", OUT);
        }
    }

    private static void pixelLimits(Map<String, Object> m) {
        m.put("clip.vision.image_min_pixels", 64);
        m.put("clip.vision.image_max_pixels", 1 << 20);
    }

    // ------------------------------------------------------------------ glm5v

    /** GLM-5V: the GLM-4V fixture (patch convs, position grid, RMS SiLU-gated blocks, merger and gated MLP). */
    static MmprojFixture glm5v(long seed) {
        return VisionFamilyFixturesB.glm4v(seed, false);
    }

    static Map<String, Object> glm5vMeta() {
        Map<String, Object> m = VisionFamilyFixturesB.glm4vMeta();
        m.put("clip.projector_type", "glm5v");
        m.put("clip.vision.swiglu_clamp", GLM5V_SWIGLU_CLAMP);
        pixelLimits(m);
        return m;
    }

    // ------------------------------------------------------------------ lfm3vl

    /**
     * LFM3-VL: both patch kernels, the RMS embedding norm; block 0 a conv block with the spatial kernel and RMS
     * norm biases, block 1 a GQA attention block with Q/K norms and an output bias, block 2 a conv block without
     * the spatial kernel; the layer-normed {@code mm.0} / {@code mm.2} projector.
     */
    static MmprojFixture lfm3vl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        f.put("v.embedding_norm.weight", 0.8f, 1.2f, E);
        int kv = E / VisionFamilyFixtures.HEADS;
        for (int il = 0; il < LFM3VL_LAYERS; il++) {
            String p = "v.blk." + il + ".";
            if (il == 0) {
                f.norm(p + "ln1.weight", p + "ln1.bias", E);
                f.norm(p + "ln2.weight", p + "ln2.bias", E);
            } else {
                f.put(p + "ln1.weight", 0.8f, 1.2f, E);
                f.put(p + "ln2.weight", 0.8f, 1.2f, E);
            }
            if (il == 1) {
                f.weight(p + "attn_q.weight", E, E);
                f.weight(p + "attn_k.weight", E, kv);
                f.weight(p + "attn_v.weight", E, kv);
                f.weight(p + "attn_out.weight", E, E);
                f.weight(p + "attn_out.bias", E);
                f.put(p + "attn_q_norm.weight", 0.8f, 1.2f, kv);
                f.put(p + "attn_k_norm.weight", 0.8f, 1.2f, kv);
            } else {
                f.weight(p + "conv2d_in_proj.weight", E, 3 * E);
                f.weight(p + "conv2d_out_proj.weight", E, E);
                if (il == 0) {
                    f.weight(p + "conv2d_spatial.weight", 3, 3, 1, E);
                }
            }
            f.weight(p + "ffn_up.weight", E, VisionFamilyFixtures.FF);
            f.weight(p + "ffn_up.bias", VisionFamilyFixtures.FF);
            f.weight(p + "ffn_down.weight", VisionFamilyFixtures.FF, E);
            f.weight(p + "ffn_down.bias", E);
        }
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        mlp(f, "mm.0", "mm.2", 4 * E, true);
        return f;
    }

    static Map<String, Object> lfm3vlMeta() {
        Map<String, Object> m = meta("lfm3vl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.block_count", LFM3VL_LAYERS);
        m.put("clip.vision.attention.head_count_kv", 1);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.image_resize_algo", "bilinear");
        pixelLimits(m);
        return m;
    }

    // ------------------------------------------------------------------ ling3vl

    /** Ling-3-VL: the Qwen3-VL tower without deepstack, the layer-normed merger with a tanh-GELU MLP. */
    static MmprojFixture ling3vl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        layers(f, true, true, true, false, false);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        mlp(f, "mm.0", "mm.2", 4 * E, true);
        return f;
    }

    static Map<String, Object> ling3vlMeta() {
        Map<String, Object> m = meta("ling3vl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    // ------------------------------------------------------------------ minimax_m3

    /** MiniMax-M3: both patch kernels without bias, a pre- and post-norm tower, the per-patch and merge MLPs. */
    static MmprojFixture minimaxM3(long seed) {
        int e = MINIMAX_E;
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, e);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, e);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", e);
        f.norm("v.post_ln.weight", "v.post_ln.bias", e);
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", e);
            f.norm(p + "ln2.weight", p + "ln2.bias", e);
            for (String n : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.put(p + n + ".weight", -0.3f, 0.3f, e, e);
                f.weight(p + n + ".bias", e);
            }
            f.put(p + "ffn_up.weight", -0.3f, 0.3f, e, MINIMAX_FF);
            f.weight(p + "ffn_up.bias", MINIMAX_FF);
            f.put(p + "ffn_down.weight", -0.3f, 0.3f, MINIMAX_FF, e);
            f.weight(p + "ffn_down.bias", e);
        }
        f.put("mm.1.weight", -0.3f, 0.3f, e, HID);
        f.weight("mm.1.bias", HID);
        f.weight("mm.2.weight", HID, 8);
        f.weight("mm.2.bias", 8);
        f.weight("mm.merger.fc1.weight", 4 * 8, 12);
        f.weight("mm.merger.fc1.bias", 12);
        f.weight("mm.merger.fc2.weight", 12, OUT);
        f.weight("mm.merger.fc2.bias", OUT);
        return f;
    }

    static Map<String, Object> minimaxM3Meta() {
        Map<String, Object> m = meta("minimax_m3", 8, 2, OUT);
        m.put("clip.vision.embedding_length", MINIMAX_E);
        m.put("clip.vision.feed_forward_length", MINIMAX_FF);
        return m;
    }

    // ------------------------------------------------------------------ muse-glimmer

    /** Muse Glimmer: a 2x2 position grid (2x2-patch windows), a sparse then a global block, the 3-linear MLP. */
    static MmprojFixture museGlimmer(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 4);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.weight("mm.0.weight", 4 * E, HID);
        f.weight("mm.1.weight", HID, HID);
        f.weight("mm.2.weight", HID, OUT);
        return f;
    }

    static Map<String, Object> museGlimmerMeta() {
        return meta("muse-glimmer", 8, 2, OUT);
    }

    // ------------------------------------------------------------------ nemotron

    /** A RADIO tower over 2x2 patches: patch bias, register rows, fused biased Q/K/V, layer-norm blocks. */
    private static MmprojFixture radio(long seed, int posRows) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.class_embd", E, NEMOTRON_REGISTERS);
        f.weight("v.position_embd.weight", E, posRows);
        layers(f, true, true, true, false, false);
        f.put("mm.model.mlp.0.weight", 0.8f, 1.2f, 4 * E);
        f.weight("mm.model.mlp.1.weight", 4 * E, HID);
        f.weight("mm.model.mlp.3.weight", HID, OUT);
        return f;
    }

    /** Nemotron Nano V2 VL: position rows of the 8x8 image-size grid, added as stored. */
    static MmprojFixture nemotronV2Vl(long seed) {
        return radio(seed, 16);
    }

    static Map<String, Object> nemotronV2VlMeta() {
        Map<String, Object> m = meta("nemotron_v2_vl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** Nemotron Omni: a 3x3 position grid resized to the image, plus the 6-channel video patch kernel. */
    static MmprojFixture nemotronOmni(long seed) {
        MmprojFixture f = radio(seed, 9);
        f.weight("v.patch_embd.weight.1", 2, 2, 6, E);
        return f;
    }

    static Map<String, Object> nemotronOmniMeta() {
        Map<String, Object> m = meta("nemotron_omni", 8, 2, OUT);
        m.put("clip.vision.projector.scale_factor", 2);
        pixelLimits(m);
        return m;
    }

    // ------------------------------------------------------------------ exaone4_5

    /**
     * EXAONE 4.5: both 14x14 patch kernels without bias, RMS blocks with fused biased GQA Q/K/V (one KV head) and
     * a SiLU-gated FFN, the RMS post norm, the qwen2.5 merger.
     */
    static MmprojFixture exaone45(long seed) {
        int kv = E / VisionFamilyFixtures.HEADS;
        int ff = VisionFamilyFixtures.FF;
        MmprojFixture f = new MmprojFixture(seed);
        f.put("v.patch_embd.weight", -0.1f, 0.1f, EXAONE_PATCH, EXAONE_PATCH, 3, E);
        f.put("v.patch_embd.weight.1", -0.1f, 0.1f, EXAONE_PATCH, EXAONE_PATCH, 3, E);
        f.put("v.post_ln.weight", 0.8f, 1.2f, E);
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            f.put(p + "ln1.weight", 0.8f, 1.2f, E);
            f.put(p + "ln2.weight", 0.8f, 1.2f, E);
            f.weight(p + "attn_qkv.weight", E, E + 2 * kv);
            f.weight(p + "attn_qkv.bias", E + 2 * kv);
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            f.weight(p + "ffn_up.weight", E, ff);
            f.weight(p + "ffn_up.bias", ff);
            f.weight(p + "ffn_gate.weight", E, ff);
            f.weight(p + "ffn_gate.bias", ff);
            f.weight(p + "ffn_down.weight", ff, E);
            f.weight(p + "ffn_down.bias", E);
        }
        mlp(f, "mm.0", "mm.2", 4 * E, true);
        return f;
    }

    static Map<String, Object> exaone45Meta() {
        Map<String, Object> m = meta("exaone4_5", 112, EXAONE_PATCH, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.attention.head_count_kv", 1);
        m.put("clip.vision.n_wa_pattern", 2);
        pixelLimits(m);
        return m;
    }
}

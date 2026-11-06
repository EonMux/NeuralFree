package com.eonmux.neuralfree.gguf.mm;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.layers;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.meta;

/**
 * Synthetic mmproj tensors and metadata of the stacked-feature llava (Granite Vision 3), ldp, ldpv2, glm-edge,
 * cogvlm, glm4v, kimivl, lfm2, d1omni_v, dots_ocr, youtuvl and granite4_vision families, shaped as their
 * converters write them (llama.cpp convert_hf_to_gguf.py and the legacy llava / glm-edge converters), small enough
 * for a double-precision reference. Package-private test support of the {@code mm} tests.
 */
final class VisionFamilyFixturesB {

    /** Projector output width of every fixture. */
    static final int OUT = 6;
    /** Granite Vision 4.1: the QFormer head width is fixed at 64, so the tower is 64 wide. */
    static final int GRANITE_E = 64;
    static final int GRANITE_FF = 48;
    static final int GRANITE_HEADS = 4;
    static final int GRANITE_WINDOW = 4;
    static final int GRANITE_QUERY = 2;
    /** The granite4_vision tile side ({@code clip.vision.image_size}): 8x8 patches, 4x4 tokens. */
    static final int GRANITE_IMAGE = 16;

    private VisionFamilyFixturesB() {
    }

    // ------------------------------------------------------------------ llava graph families

    /** A llava tower over 4x4 patches of a 16x16 tile: patch bias, learned positions, layer-norm pre-norm. */
    private static MmprojFixture llavaTower(long seed, boolean classToken, boolean postNorm) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        if (classToken) {
            f.weight("v.class_embd", E, 1);
        }
        f.weight("v.position_embd.weight", E, classToken ? 17 : 16);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        if (postNorm) {
            f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        }
        layers(f, false, true, true, false, false);
        return f;
    }

    /** Granite Vision 3: feature layers 0, 1 and 2 stacked to 3 * n_embd, two-layer GELU MLP. */
    static MmprojFixture llavaStacked(long seed) {
        MmprojFixture f = llavaTower(seed, false, true);
        f.weight("mm.0.weight", 3 * E, OUT);
        f.weight("mm.0.bias", OUT);
        f.weight("mm.2.weight", OUT, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    static Map<String, Object> llavaStackedMeta() {
        Map<String, Object> m = meta("mlp", 16, 4, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.feature_layer", new int[]{0, 1, 2});
        return m;
    }

    /** MobileVLM LDP: class token, no post-norm, two mobile blocks over the 4x4 grid. */
    static MmprojFixture ldp(long seed) {
        MmprojFixture f = llavaTower(seed, true, false);
        f.weight("mm.model.mlp.1.weight", E, 10);
        f.weight("mm.model.mlp.1.bias", 10);
        f.weight("mm.model.mlp.3.weight", 10, OUT);
        f.weight("mm.model.mlp.3.bias", OUT);
        for (int blk = 1; blk <= 2; blk++) {
            String p = "mm.model.mb_block." + blk + ".block.";
            f.weight(p + "0.0.weight", 3, 3, 1, OUT);
            f.norm(p + "0.1.weight", p + "0.1.bias", OUT);
            f.weight(p + "1.fc1.weight", OUT, 3);
            f.weight(p + "1.fc1.bias", 3);
            f.weight(p + "1.fc2.weight", 3, OUT);
            f.weight(p + "1.fc2.bias", OUT);
            f.weight(p + "2.0.weight", OUT, OUT);
            f.norm(p + "2.1.weight", p + "2.1.bias", OUT);
        }
        return f;
    }

    /** MobileVLM v2 LDPv2: class token, GELU MLP, 2x2 pool and the depthwise position encoding. */
    static MmprojFixture ldpv2(long seed) {
        MmprojFixture f = llavaTower(seed, true, false);
        f.weight("mm.model.mlp.0.weight", E, 10);
        f.weight("mm.model.mlp.0.bias", 10);
        f.weight("mm.model.mlp.2.weight", 10, OUT);
        f.weight("mm.model.mlp.2.bias", OUT);
        f.weight("mm.model.peg.0.weight", 3, 3, OUT, 1);
        f.weight("mm.model.peg.0.bias", OUT);
        return f;
    }

    /** GLM-Edge: no class token, the whole tower with post-norm, the conv adapter and its GLU. */
    static MmprojFixture glmEdge(long seed) {
        MmprojFixture f = llavaTower(seed, false, true);
        f.weight("adapter.conv.weight", 2, 2, E, 10);
        f.weight("adapter.conv.bias", 10);
        f.weight("adapter.linear.linear.weight", 10, 12);
        f.norm("adapter.linear.norm1.weight", "adapter.linear.norm1.bias", 12);
        f.weight("adapter.linear.dense_h_to_4h.weight", 12, 14);
        f.weight("adapter.linear.gate.weight", 12, 14);
        f.weight("adapter.linear.dense_4h_to_h.weight", 14, OUT);
        f.weight("adapter.boi", OUT);
        f.weight("adapter.eoi", OUT);
        return f;
    }

    static Map<String, Object> llavaMeta(String projector) {
        Map<String, Object> m = meta(projector, 16, 4, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    // ------------------------------------------------------------------ cogvlm, glm4v

    /** CogVLM: class token last, fused biased Q/K/V, post-norm blocks, GLU projector with BOI / EOI. */
    static MmprojFixture cogvlm(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.class_embd", E, 1);
        f.weight("v.position_embd.weight", E, 5);
        layers(f, true, true, true, false, false);
        f.weight("mm.model.fc.weight", E, 10);
        f.norm("mm.post_fc_norm.weight", "mm.post_fc_norm.bias", 10);
        f.weight("mm.up.weight", 10, 14);
        f.weight("mm.gate.weight", 10, 14);
        f.weight("mm.down.weight", 14, OUT);
        f.weight("v.boi", OUT);
        f.weight("v.eoi", OUT);
        return f;
    }

    static Map<String, Object> cogvlmMeta() {
        Map<String, Object> m = meta("cogvlm", 8, 4, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    /**
     * GLM-4V over 2x2 patches: the two patch convs, patch bias, RMS embedding norm and a 3x3 position grid
     * (GLM-OCR: neither), RMS blocks with biased Q/K/V and a SiLU-gated FFN, the patch merger and gated MLP.
     */
    static MmprojFixture glm4v(long seed, boolean ocr) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        if (!ocr) {
            f.norm("v.norm_embd.weight", "v.norm_embd.bias", E);
            f.weight("v.position_embd.weight", E, 9);
        }
        f.put("v.post_ln.weight", 0.8f, 1.2f, E);
        layers(f, false, true, false, true, false);
        f.weight("mm.patch_merger.weight", 2, 2, E, 10);
        f.weight("mm.patch_merger.bias", 10);
        f.weight("mm.model.fc.weight", 10, 12);
        f.norm("mm.post_norm.weight", "mm.post_norm.bias", 12);
        f.weight("mm.up.weight", 12, 14);
        f.weight("mm.up.bias", 14);
        f.weight("mm.gate.weight", 12, 14);
        f.weight("mm.gate.bias", 14);
        f.weight("mm.down.weight", 14, OUT);
        f.weight("mm.down.bias", OUT);
        return f;
    }

    static Map<String, Object> glm4vMeta() {
        Map<String, Object> m = meta("glm4v", 8, 2, OUT);
        m.put("clip.use_silu", true);
        return m;
    }

    // ------------------------------------------------------------------ kimivl, lfm2, d1omni_v, dots_ocr

    /** Kimi-VL over 2x2 patches: a 3x3 position grid to resize, 2D RoPE, scale factor 2, normed GELU MLP. */
    static MmprojFixture kimivl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 9);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        f.weight("mm.1.weight", 4 * E, 10);
        f.weight("mm.1.bias", 10);
        f.weight("mm.2.weight", 10, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    static Map<String, Object> kimivlMeta() {
        Map<String, Object> m = meta("kimivl", 12, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /**
     * LFM2 (input norm weight and bias) or d1-omni (bias only) on the resized SigLIP over 2x2 patches, a 4x4
     * position grid, scale factor 2.
     */
    static MmprojFixture lfm2(long seed, boolean d1omni) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, true, true, true, false, false);
        if (d1omni) {
            f.put("mm.input_norm.bias", -0.1f, 0.1f, 4 * E);
        } else {
            f.norm("mm.input_norm.weight", "mm.input_norm.bias", 4 * E);
        }
        f.weight("mm.1.weight", 4 * E, 10);
        f.weight("mm.1.bias", 10);
        f.weight("mm.2.weight", 10, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    static Map<String, Object> lfm2Meta(String projector) {
        Map<String, Object> m = meta(projector, 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** dots.ocr over 2x2 patches: RMS blocks with a SiLU-gated FFN, mm.post_norm, scale factor 2. */
    static MmprojFixture dotsOcr(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        layers(f, false, true, false, true, false);
        f.put("mm.post_norm.weight", 0.8f, 1.2f, E);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        f.weight("mm.0.weight", 4 * E, 10);
        f.weight("mm.0.bias", 10);
        f.weight("mm.2.weight", 10, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    static Map<String, Object> dotsOcrMeta() {
        Map<String, Object> m = meta("dots_ocr", 8, 2, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    // ------------------------------------------------------------------ youtuvl, granite4_vision

    /** Youtu-VL over 2x2 patches: the linear NaFlex patch embedding, pre/post norm, 8-pixel windows. */
    static MmprojFixture youtuvl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 3 * 2 * 2, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.put("mm.input_norm.weight", 0.8f, 1.2f, E);
        f.weight("mm.0.weight", 4 * E, 10);
        f.weight("mm.0.bias", 10);
        f.weight("mm.2.weight", 10, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    /** Window attention in every layer but {@code fullAttn}; an empty list attends over the whole image. */
    static Map<String, Object> youtuvlMeta(int... fullAttn) {
        Map<String, Object> m = meta("youtuvl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.window_size", 8);
        m.put("clip.vision.wa_layer_indexes", fullAttn);
        return m;
    }

    /**
     * Granite Vision 4.1 over 2x2 patches of a 16x16 tile: a 64-wide SigLIP tower, then two WindowQFormers on
     * layers 0 and 1, the first with the 2x2 sampling offset 3, the second with the average pool.
     */
    static MmprojFixture granite4(long seed) {
        int e = GRANITE_E;
        MmprojFixture f = new MmprojFixture(seed);
        small(f, "v.patch_embd.weight", 2, 2, 3, e);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, e);
        small(f, "v.position_embd.weight", e, 64);
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", e);
            f.norm(p + "ln2.weight", p + "ln2.bias", e);
            for (String n : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                small(f, p + n + ".weight", e, e);
                f.put(p + n + ".bias", -0.1f, 0.1f, e);
            }
            small(f, p + "ffn_up.weight", e, GRANITE_FF);
            f.put(p + "ffn_up.bias", -0.1f, 0.1f, GRANITE_FF);
            small(f, p + "ffn_down.weight", GRANITE_FF, e);
            f.put(p + "ffn_down.bias", -0.1f, 0.1f, e);
        }
        f.weight("v.image_newline", OUT);
        for (int bid = 0; bid < 2; bid++) {
            String p = "v.proj_blk." + bid + ".";
            small(f, p + "img_pos", e, GRANITE_WINDOW * GRANITE_WINDOW);
            small(f, p + "query", e, GRANITE_QUERY * GRANITE_QUERY);
            for (String n : new String[]{"norm", "post_norm", "self_attn_norm", "cross_attn_norm", "ffn_norm"}) {
                f.norm(p + n + ".weight", p + n + ".bias", e);
            }
            for (String n : new String[]{"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
                    "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out"}) {
                small(f, p + n + ".weight", e, e);
                f.put(p + n + ".bias", -0.1f, 0.1f, e);
            }
            small(f, p + "ffn_up.weight", e, GRANITE_FF);
            f.put(p + "ffn_up.bias", -0.1f, 0.1f, GRANITE_FF);
            small(f, p + "ffn_down.weight", GRANITE_FF, e);
            f.put(p + "ffn_down.bias", -0.1f, 0.1f, e);
            small(f, p + "linear.weight", e, OUT);
            f.put(p + "linear.bias", -0.1f, 0.1f, OUT);
        }
        return f;
    }

    static Map<String, Object> granite4Meta() {
        Map<String, Object> m = meta("granite4_vision", GRANITE_IMAGE, 2, OUT);
        m.put("clip.vision.embedding_length", GRANITE_E);
        m.put("clip.vision.feed_forward_length", GRANITE_FF);
        m.put("clip.vision.attention.head_count", GRANITE_HEADS);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.feature_layer", new int[]{0, 1});
        m.put("clip.vision.projector.spatial_offsets", new int[]{3, -1});
        m.put("clip.vision.projector.query_side", GRANITE_QUERY);
        m.put("clip.vision.projector.window_side", GRANITE_WINDOW);
        return m;
    }

    /** Weights of a 64-wide tower, scaled down so the activations stay of order one. */
    private static void small(MmprojFixture f, String name, long... ne) {
        f.put(name, -0.15f, 0.15f, ne);
    }
}

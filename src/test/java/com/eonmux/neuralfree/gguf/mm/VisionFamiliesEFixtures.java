package com.eonmux.neuralfree.gguf.mm;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.layers;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.meta;

/**
 * Synthetic mmproj tensors and metadata of the sensenova_u1, step3vl, deepseek4v, deepseek41v, deepseekocr and
 * deepseekocr2 families, shaped as llama.cpp clip.cpp {@code load_tensors} reads them and written with the keys
 * their converters write. Every encoder runs 2x2 patches of an {@link VisionFamilyFixtures#E}-wide tower and every
 * projector ends at width {@link #OUT}. Package-private test support of the {@code mm} tests.
 */
final class VisionFamiliesEFixtures {

    /** Output width of every projector. */
    static final int OUT = 6;
    /** Output channels of the Step3-VL downsamplers. */
    static final int STEP_C1 = 10;
    static final int STEP_C2 = 12;
    /** Hidden width of the DeepSeek-V4 aligner MLP. */
    static final int DSV4_HID = 10;

    private VisionFamiliesEFixtures() {
    }

    // ------------------------------------------------------------------ sensenova_u1

    /** SenseNova-U1: the biased patch conv and the biased 2x2 conv {@code mm.0}; no blocks. */
    static MmprojFixture sensenova(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("mm.0.weight", 2, 2, E, OUT);
        f.put("mm.0.bias", -0.1f, 0.1f, OUT);
        return f;
    }

    /** The keys the SenseNova-U1 converter writes: no blocks, one head as wide as the embedding, image size 0. */
    static Map<String, Object> sensenovaMeta() {
        Map<String, Object> m = meta("sensenova_u1", 0, 2, OUT);
        m.put("clip.vision.block_count", 0);
        m.put("clip.vision.feed_forward_length", 0);
        m.put("clip.vision.attention.head_count", 1);
        m.put("clip.vision.attention.head_dim", E);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.image_min_pixels", 64);
        m.put("clip.vision.image_max_pixels", 4096);
        return m;
    }

    // ------------------------------------------------------------------ step3vl

    /**
     * Step3-VL: separate Q/K/V, a 4x4 position grid, pre and post norms, the two 3x3 downsamplers (with biases when
     * {@code biases}) and {@code mm.model.fc}.
     */
    static MmprojFixture step3vl(long seed, boolean biases) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.weight("mm.0.weight", 3, 3, E, STEP_C1);
        f.weight("mm.1.weight", 3, 3, STEP_C1, STEP_C2);
        if (biases) {
            f.put("mm.0.bias", -0.1f, 0.1f, STEP_C1);
            f.put("mm.1.bias", -0.1f, 0.1f, STEP_C2);
        }
        f.weight("mm.model.fc.weight", STEP_C2, OUT);
        return f;
    }

    static Map<String, Object> step3vlMeta() {
        Map<String, Object> m = meta("step3vl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 4);
        return m;
    }

    // ------------------------------------------------------------------ deepseek4v / deepseek41v

    /**
     * DeepSeek-V4 (with the pad sentinel when {@code v4}) or V4.1: fused QKV, RMS norms, the SwiGLU FFN, the post
     * norm, the 2x2 aligner MLP and the sentinel rows.
     */
    static MmprojFixture deepseekV4(long seed, boolean v4) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        layers(f, true, true, false, true, false);
        f.put("v.post_ln.weight", 0.8f, 1.2f, E);
        f.weight("mm.1.weight", 4 * E, DSV4_HID);
        f.weight("mm.1.bias", DSV4_HID);
        f.weight("mm.2.weight", DSV4_HID, OUT);
        f.weight("mm.2.bias", OUT);
        f.weight("v.token_embd.img_start", OUT);
        f.weight("v.token_embd.img_end", OUT);
        f.weight("v.image_newline", OUT);
        if (v4) {
            f.weight("v.token_embd.img_pad", OUT);
        }
        return f;
    }

    /** The keys the DeepSeek-V4 converters write: SwiGLU, the aligner merge, the min pixels. */
    static Map<String, Object> deepseekV4Meta(String projector) {
        Map<String, Object> m = meta(projector, 64, 2, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.image_min_pixels", 64);
        return m;
    }

    // ------------------------------------------------------------------ deepseekocr, deepseekocr2

    /** SAM width, MLP width, heads and blocks: blocks 0 and 1 attend in windows, block 2 over the whole grid. */
    static final int SAM_E = 8;
    static final int SAM_FF = 12;
    static final int SAM_HEADS = 2;
    static final int SAM_LAYERS = 3;
    /** The SAM attention window side; an 8-patch grid pads to 9. */
    static final int SAM_WINDOW = 3;
    /** The SAM position grid side, resized to every image's patch grid but a 64-pixel one. */
    static final int SAM_POS_SIDE = 4;
    /** The neck and first downsampler channels. */
    static final int SAM_NECK = 6;
    static final int SAM_NET2 = 7;
    /** The CLIP position grid side (a 2x2 grid plus one row). */
    static final int OCR_POS_SIDE = 2;

    /**
     * The SAM backbone: the 16x16 patch conv, a 4x4 position grid, three blocks (rel-pos tables of the window
     * length, but a 7-row column table at block 1 and 9-row tables at the global block 2, both resized), the neck
     * and the two downsamplers into {@link VisionFamilyFixtures#E} channels.
     */
    private static void sam(MmprojFixture f) {
        int d = SAM_E / SAM_HEADS;
        f.put("v.sam.patch_embd.weight", -0.05f, 0.05f, 16, 16, 3, SAM_E);
        f.put("v.sam.patch_embd.bias", -0.1f, 0.1f, SAM_E);
        f.weight("v.sam.pos_embd.weight", SAM_E, SAM_POS_SIDE, SAM_POS_SIDE);
        for (int il = 0; il < SAM_LAYERS; il++) {
            String p = "v.sam.blk." + il + ".";
            f.norm(p + "pre_ln.weight", p + "pre_ln.bias", SAM_E);
            f.norm(p + "post_ln.weight", p + "post_ln.bias", SAM_E);
            f.weight(p + "attn.qkv.weight", SAM_E, 3 * SAM_E);
            f.weight(p + "attn.qkv.bias", 3 * SAM_E);
            f.weight(p + "attn.out.weight", SAM_E, SAM_E);
            f.weight(p + "attn.out.bias", SAM_E);
            int local = 2 * SAM_WINDOW - 1;
            f.weight(p + "attn.pos_w.weight", d, il == 2 ? 9 : local);
            f.weight(p + "attn.pos_h.weight", d, il == 2 ? 9 : il == 1 ? 7 : local);
            f.weight(p + "mlp.lin1.weight", SAM_E, SAM_FF);
            f.weight(p + "mlp.lin1.bias", SAM_FF);
            f.weight(p + "mlp.lin2.weight", SAM_FF, SAM_E);
            f.weight(p + "mlp.lin2.bias", SAM_E);
        }
        f.weight("v.sam.neck.0.weight", 1, 1, SAM_E, SAM_NECK);
        f.norm("v.sam.neck.1.weight", "v.sam.neck.1.bias", SAM_NECK);
        f.weight("v.sam.neck.2.weight", 3, 3, SAM_NECK, SAM_NECK);
        f.norm("v.sam.neck.3.weight", "v.sam.neck.3.bias", SAM_NECK);
        f.weight("v.sam.net_2.weight", 3, 3, SAM_NECK, SAM_NET2);
        f.weight("v.sam.net_3.weight", 3, 3, SAM_NET2, E);
    }

    /**
     * DeepSeek-OCR: SAM, the CLIP tower (fused biased QKV, biased layer norms, the pre-norm, a class token and a
     * 2x2-plus-one position table), {@code mm.model.fc} over the CLIP and SAM rows, the newline and the separator.
     */
    static MmprojFixture deepseekOcr(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        sam(f);
        f.weight("v.class_embd", E);
        f.weight("v.position_embd.weight", E, OCR_POS_SIDE * OCR_POS_SIDE + 1);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        layers(f, true, true, true, false, false);
        f.weight("mm.model.fc.weight", 2 * E, OUT);
        f.weight("mm.model.fc.bias", OUT);
        f.weight("v.image_newline", OUT);
        f.weight("v.view_seperator", OUT, 1);
        return f;
    }

    /**
     * DeepSeek-OCR 2: SAM, the Qwen2 encoder (separate biased Q/K/V of one KV head for two query heads, RMS norms,
     * the SwiGLU FFN, the post-norm), the 768 and 1024 queries, {@code mm.model.fc} and the separator.
     */
    static MmprojFixture deepseekOcr2(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        sam(f);
        int kv = E / VisionFamilyFixtures.HEADS;
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            f.put(p + "ln1.weight", 0.8f, 1.2f, E);
            f.put(p + "ln2.weight", 0.8f, 1.2f, E);
            f.weight(p + "attn_q.weight", E, E);
            f.weight(p + "attn_q.bias", E);
            f.weight(p + "attn_k.weight", E, kv);
            f.weight(p + "attn_k.bias", kv);
            f.weight(p + "attn_v.weight", E, kv);
            f.weight(p + "attn_v.bias", kv);
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "ffn_up.weight", E, VisionFamilyFixtures.FF);
            f.weight(p + "ffn_gate.weight", E, VisionFamilyFixtures.FF);
            f.weight(p + "ffn_down.weight", VisionFamilyFixtures.FF, E);
        }
        f.put("v.post_ln.weight", 0.8f, 1.2f, E);
        f.weight("v.resample_query_768.weight", E, DeepseekOcrGraph.QUERIES_768);
        f.weight("v.resample_query_1024.weight", E, DeepseekOcrGraph.QUERIES_1024);
        f.weight("mm.model.fc.weight", E, OUT);
        f.weight("mm.model.fc.bias", OUT);
        f.weight("v.view_seperator", OUT, 1);
        return f;
    }

    /**
     * The keys the DeepSeek-OCR converters write: the CLIP-L patch and image sizes (clip.cpp sets 16 and 1024),
     * GELU, the SAM keys and window, and for OCR 2 one KV head.
     */
    static Map<String, Object> deepseekOcrMeta(String projector) {
        Map<String, Object> m = meta(projector, 224, 14, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.window_size", SAM_WINDOW);
        m.put("clip.vision.sam.block_count", SAM_LAYERS);
        m.put("clip.vision.sam.embedding_length", SAM_E);
        m.put("clip.vision.sam.head_count", SAM_HEADS);
        m.put("clip.vision.preproc_min_tiles", 2);
        m.put("clip.vision.preproc_max_tiles", "deepseekocr".equals(projector) ? 9 : 6);
        if ("deepseekocr2".equals(projector)) {
            m.put("clip.vision.attention.head_count_kv", 1);
        }
        return m;
    }
}

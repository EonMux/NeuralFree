package com.eonmux.neuralfree.gguf.mm;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.LAYERS;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.layers;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.meta;

/**
 * Synthetic mmproj tensors and metadata of the qwen3vl, paddleocr, kimivl, dots_ocr, dots3note_v, hunyuanvl,
 * janus_pro, lfm2, d1omni_v, cohere2v and lightonocr families, shaped as llama.cpp clip.cpp
 * {@code load_tensors} reads them, small enough for a double-precision reference. Every encoder runs 2x2
 * patches of an {@link VisionFamilyFixtures#E}-wide, two-head, two-block tower and every projector ends at
 * width {@link #OUT}. Package-private test support of the {@code mm} tests.
 */
final class VisionFamiliesCFixtures {

    /** Output width of every projector. */
    static final int OUT = 6;
    /** Hidden width of the projector MLPs. */
    static final int HID = 10;
    /** Routed experts of the dots3-note MoE block, and the experts each token uses. */
    static final int EXPERTS = 4;
    static final int EXPERTS_USED = 2;
    /** Expert FFN width of the dots3-note MoE block. */
    static final int EXPERT_FF = 6;
    /** Channels of the two HunyuanVL perceiver convs. */
    static final int HY_C1 = 10;
    static final int HY_C2 = 12;

    private VisionFamiliesCFixtures() {
    }

    private static void mlp(MmprojFixture f, String up, String down, int in) {
        f.weight(up + ".weight", in, HID);
        f.weight(up + ".bias", HID);
        f.weight(down + ".weight", HID, OUT);
        f.weight(down + ".bias", OUT);
    }

    // ------------------------------------------------------------------ qwen3vl

    /** Qwen3-VL: two patch kernels plus bias, a 4x4 position grid, post norm, deepstack on both blocks. */
    static MmprojFixture qwen3vl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, true, true, true, false, false);
        mlp(f, "mm.0", "mm.2", 4 * E);
        for (int il = 0; il < LAYERS; il++) {
            String p = "v.deepstack." + il + ".";
            f.norm(p + "norm.weight", p + "norm.bias", 4 * E);
            mlp(f, p + "fc1", p + "fc2", 4 * E);
        }
        return f;
    }

    static Map<String, Object> qwen3vlMeta() {
        Map<String, Object> m = meta("qwen3vl_merger", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    // ------------------------------------------------------------------ paddleocr / kimivl

    /** PaddleOCR-VL: separate Q/K/V, a 4x4 position grid, post norm, the mlp_AR projector. */
    static MmprojFixture paddleocr(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        mlp(f, "mm.1", "mm.2", 4 * E);
        return f;
    }

    static Map<String, Object> paddleocrMeta() {
        Map<String, Object> m = meta("paddleocr", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    /** Kimi-VL: separate Q/K/V, a 4x4 position grid, post norm, the chunk-normed GELU projector. */
    static MmprojFixture kimivl(long seed) {
        MmprojFixture f = paddleocr(seed);
        f.remove("v.post_ln.weight");
        f.remove("v.post_ln.bias");
        return f;
    }

    static Map<String, Object> kimivlMeta() {
        Map<String, Object> m = meta("kimivl", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    // ------------------------------------------------------------------ dots

    /** dots.ocr: RMS tower with SiLU-gated FFN, no learned positions, the mm.post_norm trunk norm, the merger. */
    static MmprojFixture dotsOcr(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.put("mm.post_norm.weight", 0.8f, 1.2f, E);
        layers(f, true, false, false, true, false);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        mlp(f, "mm.0", "mm.2", 4 * E);
        return f;
    }

    static Map<String, Object> dotsOcrMeta() {
        Map<String, Object> m = meta("dots_ocr", 8, 2, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** dots3-note vision: the dots.ocr fixture whose last block routes over {@link #EXPERTS} SiLU experts. */
    static MmprojFixture dots3note(long seed) {
        return dots3note(seed, EXPERTS);
    }

    /** The dots3-note fixture whose last block routes over {@code experts} SiLU experts. */
    static MmprojFixture dots3note(long seed, int experts) {
        MmprojFixture f = dotsOcr(seed);
        String p = "v.blk." + (LAYERS - 1) + ".";
        for (String n : new String[]{"ffn_up.weight", "ffn_gate.weight", "ffn_down.weight"}) {
            f.remove(p + n);
        }
        f.weight(p + "ffn_gate_inp.weight", E, experts);
        f.weight(p + "ffn_gate_exps.weight", E, EXPERT_FF, experts);
        f.weight(p + "ffn_up_exps.weight", E, EXPERT_FF, experts);
        f.weight(p + "ffn_down_exps.weight", EXPERT_FF, E, experts);
        f.put(p + "exp_probs_b.weight", -0.3f, 0.3f, experts);
        return f;
    }

    static Map<String, Object> dots3noteMeta() {
        Map<String, Object> m = meta("dots3note_v", 8, 2, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.expert_used_count", EXPERTS_USED);
        return m;
    }

    // ------------------------------------------------------------------ hunyuanvl

    /** HunyuanVL: separate Q/K/V, a 4x4 position grid, the conv perceiver to {@link #OUT}. */
    static MmprojFixture hunyuanvl(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        layers(f, false, true, true, false, false);
        f.put("mm.pre_norm.weight", 0.8f, 1.2f, E);
        f.weight("mm.0.weight", 2, 2, E, HY_C1);
        f.weight("mm.0.bias", HY_C1);
        f.weight("mm.2.weight", 1, 1, HY_C1, HY_C2);
        f.weight("mm.2.bias", HY_C2);
        f.weight("v.image_newline", HY_C2);
        f.weight("mm.model.fc.weight", HY_C2, OUT);
        f.weight("mm.model.fc.bias", OUT);
        f.weight("mm.image_begin", OUT);
        f.weight("mm.image_end", OUT);
        f.put("mm.post_norm.weight", 0.8f, 1.2f, OUT);
        return f;
    }

    static Map<String, Object> hunyuanvlMeta() {
        // hunyuanvl runs GELU whatever the file selects (clip.cpp load_hparams)
        Map<String, Object> m = meta("hunyuanvl", 8, 2, OUT);
        m.put("clip.use_silu", true);
        return m;
    }

    // ------------------------------------------------------------------ siglip projectors

    /** A SigLIP tower of 2x2 patches over the 8x8 image-size grid (a 4x4 position grid) with post norm. */
    private static MmprojFixture siglip(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, true, true, true, false, false);
        return f;
    }

    /** Janus-Pro: the SigLIP tower, the {@code mm.0} / {@code mm.1} MLP. */
    static MmprojFixture janus(long seed) {
        MmprojFixture f = siglip(seed);
        mlp(f, "mm.0", "mm.1", E);
        return f;
    }

    static Map<String, Object> janusMeta() {
        Map<String, Object> m = meta("janus_pro", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        return m;
    }

    /** LFM2-VL: the SigLIP tower over any grid, merge 2, optional input layer norm, GELU MLP. */
    static MmprojFixture lfm2(long seed, boolean inputNorm) {
        MmprojFixture f = siglip(seed);
        if (inputNorm) {
            f.norm("mm.input_norm.weight", "mm.input_norm.bias", 4 * E);
        }
        mlp(f, "mm.1", "mm.2", 4 * E);
        return f;
    }

    static Map<String, Object> lfm2Meta(String projector) {
        Map<String, Object> m = meta(projector, 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    /** Cohere2 vision: the SigLIP tower, merge 2, {@code mm.1} to {@code [x, gate]}, the swapped SwiGLU. */
    static MmprojFixture cohere2v(long seed) {
        MmprojFixture f = siglip(seed);
        f.weight("mm.1.weight", 4 * E, 2 * HID);
        f.weight("mm.1.bias", 2 * HID);
        f.weight("mm.2.weight", HID, OUT);
        f.weight("mm.2.bias", OUT);
        return f;
    }

    static Map<String, Object> cohere2vMeta() {
        Map<String, Object> m = meta("cohere2v", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.preproc_max_tiles", 4);
        return m;
    }

    // ------------------------------------------------------------------ lightonocr

    /** LightOnOCR: the Pixtral fixture with the 2x2 patch merger and no {@code [IMG_BREAK]} embedding. */
    static MmprojFixture lightonocr(long seed) {
        MmprojFixture f = VisionFamilyFixtures.pixtral(seed, true);
        f.remove("v.token_embd.img_break");
        return f;
    }

    static Map<String, Object> lightonocrMeta() {
        Map<String, Object> m = VisionFamilyFixtures.pixtralMeta(true);
        m.put("clip.projector_type", "lightonocr");
        return m;
    }
}

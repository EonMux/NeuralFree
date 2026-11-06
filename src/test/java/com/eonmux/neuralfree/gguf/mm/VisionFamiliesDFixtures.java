package com.eonmux.neuralfree.gguf.mm;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.layers;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.meta;

/**
 * Synthetic mmproj tensors and metadata of the kimik25, gemma4v, gemma4uv, MiniCPM-V 4.6, MiMo-VL and Gemma 3n
 * families,
 * shaped as llama.cpp clip.cpp {@code load_tensors} reads them. The encoders read 2x2 patches (the gemma4uv
 * embedder merges them to 4x4), the towers are two-head and two-block (mimovl: {@link #MIMO_HEADS} query heads
 * over {@link #MIMO_KV_HEADS} key/value heads, one block per {@link #MIMO_MODES} entry), {@link VisionFamilyFixtures#E}
 * wide ({@link #G} for gemma4v), and every projector ends at width {@link #OUT}; the rows are checked against
 * llama.cpp's own clip graphs ({@link LlamaCppParity}). Package-private test support of the {@code mm} tests.
 */
final class VisionFamiliesDFixtures {

    /** Output width of every projector. */
    static final int OUT = 6;
    /** Hidden width of the projector MLPs. */
    static final int HID = 10;
    /** Width of the gemma4v tower: two heads of 8, so each NEOX half rotates two frequencies. */
    static final int G = 16;
    /** Rows of each gemma4 position table. */
    static final int AXIS_ROWS = 6;
    /** Rows of the MiniCPM-V 70x70 bucket position table. */
    static final int BUCKETS = MiniCpmvGraph.POS_BUCKETS * MiniCpmvGraph.POS_BUCKETS;
    /** Rows of the MiniCPM-V 4.7 separator table: {@code </image>}, {@code <slice>}, {@code </slice>}, {@code \n}. */
    static final int SEPARATORS = 4;
    /** MiMo-VL query heads. */
    static final int MIMO_HEADS = 4;
    /** MiMo-VL key/value heads. */
    static final int MIMO_KV_HEADS = 2;
    /** MiMo-VL head width, independent of the tower width {@link VisionFamilyFixtures#E}. */
    static final int MIMO_HEAD_DIM = 8;
    /**
     * MiMo-VL per-block attention modes: a column window entered from the start, full attention, a row window,
     * and a column window that closes the tower.
     */
    static final int[] MIMO_MODES = {1, -1, 0, 1};

    private VisionFamiliesDFixtures() {
    }

    private static void mlp(MmprojFixture f, String up, String down, int in) {
        f.weight(up + ".weight", in, HID);
        f.weight(up + ".bias", HID);
        f.weight(down + ".weight", HID, OUT);
        f.weight(down + ".bias", OUT);
    }

    // ------------------------------------------------------------------ kimik25

    /**
     * Kimi K2.5: separate biased Q/K/V, a {@code [E, 3, 2]} position grid (3 columns, 2 rows) the graph resizes
     * bicubically, post norm, the chunk-normed GELU projector.
     */
    static MmprojFixture kimik25(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.position_embd.weight", E, 3, 2);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        layers(f, false, true, true, false, false);
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", E);
        mlp(f, "mm.1", "mm.2", 4 * E);
        return f;
    }

    /** The kimik25 metadata; the layer-norm epsilon is large so the projector norm's epsilon shows in the rows. */
    static Map<String, Object> kimik25Meta() {
        Map<String, Object> m = meta("kimik25", 8, 2, OUT);
        m.put("clip.use_gelu", true);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-2f);
        return m;
    }

    // ------------------------------------------------------------------ gemma4v

    /**
     * Gemma 4 vision: a bias-free patch kernel, {@code [G, 6, 2]} column and row tables, RMS blocks with per-head
     * Q/K norms, post norms, an output scale and a gated FFN, clamp bounds on some block matrices and on the input
     * projection (some bounds absent), and the standardization pair.
     */
    static MmprojFixture gemma4v(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, G);
        f.weight("v.position_embd.weight", G, AXIS_ROWS, 2);
        int dh = G / VisionFamilyFixtures.HEADS;
        int ff = 20;
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            for (String n : new String[]{"ln1", "ln2", "attn_post_norm", "ffn_post_norm"}) {
                f.put(p + n + ".weight", 0.8f, 1.2f, G);
            }
            f.put(p + "out_scale.weight", 0.5f, 1.5f, G);
            for (String n : new String[]{"attn_q", "attn_k", "attn_v", "attn_out"}) {
                f.weight(p + n + ".weight", G, G);
            }
            f.put(p + "attn_q_norm.weight", 0.8f, 1.2f, dh);
            f.put(p + "attn_k_norm.weight", 0.8f, 1.2f, dh);
            f.weight(p + "ffn_up.weight", G, ff);
            f.weight(p + "ffn_gate.weight", G, ff);
            f.weight(p + "ffn_down.weight", ff, G);
            f.put(p + "attn_q.input_min", -0.9f, -0.6f, 1);
            f.put(p + "attn_q.input_max", 0.6f, 0.9f, 1);
            f.put(p + "attn_out.output_max", 0.1f, 0.3f, 1);
            f.put(p + "ffn_down.input_min", -0.2f, -0.1f, 1);
            f.put(p + "ffn_down.output_min", -0.4f, -0.2f, 1);
            f.put(p + "ffn_down.output_max", 0.2f, 0.4f, 1);
        }
        f.put("v.std_bias", -0.2f, 0.2f, G);
        f.put("v.std_scale", 0.5f, 1.5f, G);
        f.weight("mm.input_projection.weight", G, OUT);
        f.put("mm.input_projection.input_min", -1.2f, -0.8f, 1);
        f.put("mm.input_projection.input_max", 0.8f, 1.2f, 1);
        f.put("mm.input_projection.output_max", 0.3f, 0.6f, 1);
        return f;
    }

    /** The gemma4v metadata; {@code pool} is the projector scale factor, the 3x3 default when null. */
    static Map<String, Object> gemma4vMeta(Integer pool) {
        Map<String, Object> m = meta("gemma4v", 12, 2, OUT);
        m.put("clip.vision.embedding_length", G);
        m.put("clip.vision.feed_forward_length", 20);
        m.put("clip.use_gelu", true);
        if (pool != null) {
            m.put("clip.vision.projector.scale_factor", pool);
        }
        return m;
    }

    // ------------------------------------------------------------------ gemma4uv

    /**
     * Gemma 4 unified: the {@code [48, E]} projection of 4x4 patch vectors with its bias, the three patch norms,
     * {@code [E, 6, 2]} column and row tables and the input projection.
     */
    static MmprojFixture gemma4uv(long seed) {
        MmprojFixture f = new MmprojFixture(seed);
        int in = 3 * 4 * 4;
        f.weight("v.patch_embd.weight", in, E);
        f.weight("v.patch_embd.bias", E);
        f.norm("v.patch_norm.1.weight", "v.patch_norm.1.bias", in);
        f.norm("v.patch_norm.2.weight", "v.patch_norm.2.bias", E);
        f.norm("v.patch_norm.3.weight", "v.patch_norm.3.bias", E);
        f.weight("v.position_embd.weight", E, AXIS_ROWS, 2);
        f.weight("mm.input_projection.weight", E, OUT);
        return f;
    }

    /** The gemma4uv metadata: no blocks, 2x2 patches merged 2x2 by the projection. */
    static Map<String, Object> gemma4uvMeta() {
        Map<String, Object> m = meta("gemma4uv", 16, 2, OUT);
        m.put("clip.vision.feed_forward_length", 0);
        m.put("clip.vision.block_count", 0);
        m.put("clip.vision.attention.head_count", 0);
        m.put("clip.vision.projector.scale_factor", 2);
        return m;
    }

    // ------------------------------------------------------------------ minicpmv4_6

    /**
     * MiniCPM-V 4.6: the biased patch kernel, the {@code [E, 4900]} bucket table, layer-normed blocks with separate
     * biased Q/K/V and layer scales, the final merger with its input-norm bias, and the separator rows. With
     * {@code vitMerger} the ViT merger (biases on Q and V only, a bias-free {@code ds_ffn_down}) and the pre- and
     * post-norms; with {@code swappedFfn} every block stores its FFN up and down tensors under each other's names,
     * as the legacy conversions clip.cpp swaps back.
     */
    static MmprojFixture minicpmv46(long seed, boolean vitMerger, boolean swappedFfn) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.bias", E);
        f.weight("v.position_embd.weight", E, BUCKETS);
        layers(f, false, true, true, false, true);
        if (swappedFfn) {
            for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
                String p = "v.blk." + il + ".";
                swap(f, p + "ffn_up.weight", p + "ffn_down.weight");
                swap(f, p + "ffn_up.bias", p + "ffn_down.bias");
            }
        }
        if (vitMerger) {
            f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
            f.norm("v.post_ln.weight", "v.post_ln.bias", E);
            f.norm("v.vit_merger.ln1.weight", "v.vit_merger.ln1.bias", E);
            for (String p : MiniCpmv46Graph.MERGER_PROJECTIONS) {
                f.weight(MiniCpmv46Graph.MERGER_ATTN + p + ".weight", E, E);
            }
            f.weight("v.vit_merger.attn_q.bias", E);
            f.weight("v.vit_merger.attn_v.bias", E);
            f.norm("v.vit_merger.ds_ln.weight", "v.vit_merger.ds_ln.bias", 4 * E);
            f.weight("v.vit_merger.ds_ffn_up.weight", 4 * E, HID);
            f.weight("v.vit_merger.ds_ffn_up.bias", HID);
            f.weight("v.vit_merger.ds_ffn_down.weight", HID, E);
        }
        f.norm("mm.input_norm.weight", "mm.input_norm.bias", 4 * E);
        mlp(f, "mm.up", "mm.down", 4 * E);
        f.weight(MiniCpmv46Graph.SEPARATORS, OUT, SEPARATORS);
        return f;
    }

    private static void swap(MmprojFixture f, String a, String b) {
        long[] shape = f.shapes.get(a);
        float[] values = f.values.get(a);
        f.shapes.put(a, f.shapes.get(b));
        f.values.put(a, f.values.get(b));
        f.shapes.put(b, shape);
        f.values.put(b, values);
    }

    /**
     * The metadata of projector type {@code type}: {@code merge} the scale factor (the 4x default when null) and
     * {@code insertLayer} the ViT merger's block (the default block 0 when null).
     */
    static Map<String, Object> minicpmv46Meta(String type, Integer merge, Integer insertLayer) {
        Map<String, Object> m = meta(type, 16, 2, OUT);
        m.put("clip.use_gelu", true);
        if (merge != null) {
            m.put("clip.vision.projector.scale_factor", merge);
        }
        if (insertLayer != null) {
            m.put("clip.vision.wa_layer_indexes", new int[]{insertLayer});
        }
        return m;
    }

    // ------------------------------------------------------------------ mimovl

    /**
     * MiMo-VL: the two bias-free patch kernels, RMS blocks (biased norms in block 0 only) with the fused biased
     * grouped-query Q/K/V of {@link #MIMO_HEAD_DIM}-wide heads, a biased output projection and SwiGLU FFN, the
     * per-head sinks of every windowed block, the post-norm, and the bias-free {@code mm.0} / {@code mm.2}.
     */
    static MmprojFixture mimovl(long seed) {
        return mimovl(seed, MIMO_HEADS, MIMO_KV_HEADS, MIMO_HEAD_DIM, MIMO_MODES);
    }

    /** {@link #mimovl(long)} with {@code heads} query heads over {@code kvHeads}, {@code headDim} wide, per mode. */
    static MmprojFixture mimovl(long seed, int heads, int kvHeads, int headDim, int[] modes) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 2, 2, 3, E);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, E);
        int q = heads * headDim;
        int qkv = q + 2 * kvHeads * headDim;
        int ff = VisionFamilyFixtures.FF;
        for (int il = 0; il < modes.length; il++) {
            String p = "v.blk." + il + ".";
            if (il == 0) {
                f.norm(p + "ln1.weight", p + "ln1.bias", E);
                f.norm(p + "ln2.weight", p + "ln2.bias", E);
            } else {
                f.put(p + "ln1.weight", 0.8f, 1.2f, E);
                f.put(p + "ln2.weight", 0.8f, 1.2f, E);
            }
            f.weight(p + "attn_qkv.weight", E, qkv);
            f.weight(p + "attn_qkv.bias", qkv);
            f.weight(p + "attn_out.weight", q, E);
            f.weight(p + "attn_out.bias", E);
            f.weight(p + "ffn_gate.weight", E, ff);
            f.weight(p + "ffn_gate.bias", ff);
            f.weight(p + "ffn_up.weight", E, ff);
            f.weight(p + "ffn_up.bias", ff);
            f.weight(p + "ffn_down.weight", ff, E);
            f.weight(p + "ffn_down.bias", E);
            if (modes[il] != -1) {
                f.put(p + "attn_sinks", -1f, 1.5f, heads);
            }
        }
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        f.weight("mm.0.weight", 4 * E, HID);
        f.weight("mm.2.weight", HID, OUT);
        return f;
    }

    /** The mimovl metadata: SiLU, the grouped heads, {@link #MIMO_MODES} and the window radius {@code window}. */
    static Map<String, Object> mimovlMeta(int window) {
        return mimovlMeta(window, MIMO_HEADS, MIMO_KV_HEADS, MIMO_MODES);
    }

    /** {@link #mimovlMeta(int)} with {@code heads} query heads over {@code kvHeads}, one block per mode. */
    static Map<String, Object> mimovlMeta(int window, int heads, int kvHeads, int[] modes) {
        Map<String, Object> m = meta("mimovl", 16, 2, OUT);
        m.put("clip.use_silu", true);
        m.put("clip.vision.block_count", modes.length);
        m.put("clip.vision.attention.head_count", heads);
        m.put("clip.vision.attention.head_count_kv", kvHeads);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.window_size", window);
        m.put("clip.vision.wa_pattern_mode", modes.clone());
        m.put("clip.vision.image_min_pixels", 64);
        m.put("clip.vision.image_max_pixels", 4096);
        return m;
    }

    // ------------------------------------------------------------------ gemma3nv

    /** Channels of the Gemma 3n features after the fusion adapter, the embedder's input width. */
    static final int MNV5_EMBD = 6;

    /**
     * Gemma 3n MobileNetV5: the biased, normed stem conv to 4 channels; stage 0 two edge residuals (the first with
     * both norms, the second bias-free of norms and residual); stage 1 two inverted residuals (a 5x5 depthwise mid,
     * layer scales, the second without a depthwise start); stage 2 an inverted residual and a multi-query attention
     * block of two 4-wide query heads with downsampled, normed keys and values; with {@code fourStages} stage 3 an
     * inverted residual and a plain attention block, else three stages; the fusion adapter with both pointwise
     * convs and all norms down to {@link #MNV5_EMBD} channels; the soft embedding norm and the input projection to
     * {@link #OUT}.
     */
    static MmprojFixture gemma3nv(long seed, boolean fourStages) {
        MmprojFixture f = new MmprojFixture(seed);
        f.weight(MobileNetV5Layout.STEM_CONV, 3, 3, 3, 4);
        f.weight(MobileNetV5Layout.STEM_BIAS, 1, 1, 4, 1);
        bn(f, MobileNetV5Layout.STEM_NORM, 4);
        String b = "v.blk.0.0.";
        f.weight(b + MobileNetV5Layout.CONV_EXP, 3, 3, 4, 8);
        bn(f, b + MobileNetV5Layout.BN1, 8);
        f.weight(b + MobileNetV5Layout.CONV_PWL, 1, 1, 8, 4);
        bn(f, b + MobileNetV5Layout.BN2, 4);
        b = "v.blk.0.1.";
        f.weight(b + MobileNetV5Layout.CONV_EXP, 3, 3, 4, 8);
        f.weight(b + MobileNetV5Layout.CONV_PWL, 1, 1, 8, 4);
        inverted(f, "v.blk.1.0.", 4, 3, 8, 5, 6);
        inverted(f, "v.blk.1.1.", 6, 0, 12, 3, 6);
        inverted(f, "v.blk.2.0.", 6, 3, 8, 3, 8);
        attention(f, "v.blk.2.1.", true);
        int fused = 8;
        if (fourStages) {
            inverted(f, "v.blk.3.0.", 8, 0, 8, 3, 8);
            attention(f, "v.blk.3.1.", false);
            fused += 8;
        }
        f.weight(MobileNetV5Layout.MSFA_EXPAND, 1, 1, fused, 12);
        bn(f, MobileNetV5Layout.MSFA_EXPAND_NORM, 12);
        f.weight(MobileNetV5Layout.MSFA_PROJECT, 1, 1, 12, MNV5_EMBD);
        bn(f, MobileNetV5Layout.MSFA_PROJECT_NORM, MNV5_EMBD);
        bn(f, MobileNetV5Layout.MSFA_NORM, MNV5_EMBD);
        bn(f, MobileNetV5Layout.SOFT_EMB_NORM, MNV5_EMBD);
        f.weight(MobileNetV5Layout.INPUT_PROJECTION, MNV5_EMBD, OUT);
        return f;
    }

    private static void bn(MmprojFixture f, String name, int c) {
        f.put(name, 0.8f, 1.2f, c);
    }

    /**
     * An inverted residual from {@code in} channels: a depthwise start of side {@code start} (none when 0), the
     * expansion to {@code mid}, the depthwise mid of side {@code k}, the projection to {@code out}, every stage
     * normed, and the layer scale.
     */
    private static void inverted(MmprojFixture f, String b, int in, int start, int mid, int k, int out) {
        if (start > 0) {
            f.weight(b + MobileNetV5Layout.DW_START, start, start, 1, in);
            bn(f, b + MobileNetV5Layout.DW_START_NORM, in);
        }
        f.weight(b + MobileNetV5Layout.PW_EXP, 1, 1, in, mid);
        bn(f, b + MobileNetV5Layout.PW_EXP_NORM, mid);
        f.weight(b + MobileNetV5Layout.DW_MID, k, k, 1, mid);
        bn(f, b + MobileNetV5Layout.DW_MID_NORM, mid);
        f.weight(b + MobileNetV5Layout.PW_PROJ, 1, 1, mid, out);
        bn(f, b + MobileNetV5Layout.PW_PROJ_NORM, out);
        f.put(b + MobileNetV5Layout.LAYER_SCALE, 0.2f, 1.5f, 1, 1, out, 1);
    }

    /**
     * An 8-channel attention block of two 4-wide query heads over one 4-wide key/value head; with {@code down}
     * the block norm, the downsampling key and value convs with their norms and the layer scale.
     */
    private static void attention(MmprojFixture f, String b, boolean down) {
        f.weight(b + MobileNetV5Layout.QUERY, 1, 1, 8, 8);
        f.weight(b + MobileNetV5Layout.KEY, 1, 1, 8, 4);
        f.weight(b + MobileNetV5Layout.VALUE, 1, 1, 8, 4);
        f.weight(b + MobileNetV5Layout.OUTPUT, 1, 1, 8, 8);
        if (down) {
            bn(f, b + MobileNetV5Layout.ATTN_NORM, 8);
            f.weight(b + MobileNetV5Layout.KEY_DOWN, 3, 3, 1, 8);
            bn(f, b + MobileNetV5Layout.KEY_NORM, 8);
            f.weight(b + MobileNetV5Layout.VALUE_DOWN, 3, 3, 1, 8);
            bn(f, b + MobileNetV5Layout.VALUE_NORM, 8);
            f.put(b + MobileNetV5Layout.LAYER_SCALE, 0.2f, 1.5f, 1, 1, 8, 1);
        }
    }

    /** The gemma3nv metadata: {@code imageSize / 2} output tokens, the features {@link #MNV5_EMBD} channels wide. */
    static Map<String, Object> gemma3nvMeta(int imageSize) {
        return gemma3nvMeta(imageSize, 2);
    }

    /** The gemma3nv metadata of {@code imageSize / patchSize} output tokens. */
    static Map<String, Object> gemma3nvMeta(int imageSize, int patchSize) {
        Map<String, Object> m = meta("gemma3nv", imageSize, patchSize, OUT);
        m.put("clip.vision.embedding_length", MNV5_EMBD);
        m.put("clip.vision.image_mean", new float[]{0f, 0f, 0f});
        m.put("clip.vision.image_std", new float[]{1f, 1f, 1f});
        return m;
    }

    /** The gemma3nv conv kernels and input projection, the tensors a converted file stores as F16. */
    static boolean gemma3nvHalfTensor(String name, long[] shape) {
        boolean channelVector = shape.length == 4 && shape[0] == 1 && shape[1] == 1 && shape[3] == 1;
        return shape.length >= 2 && !channelVector;
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU vision encoder against an independent double-precision transcription of the llama.cpp clip graphs
 * (tools/mtmd/models/{llava,siglip,qwen2vl}.cpp): the llava CLIP ViT with class token and MLP projector, the
 * SigLIP ViT with the Gemma3 pool projector, the Qwen2-VL 2D m-rope encoder with its pixel-shuffle merger and
 * the Qwen2.5-VL window attention (a window plan that regroups the tokens, with the positions remapped).
 */
public class CpuVisionEncoderTest {

    private static final float TOL = 2e-4f;

    // ================================================================== fixtures

    private static Map<String, Object> clipMeta(String projector, int image, int patch, int embd, int ff,
                                                int layers, int proj, int heads, int kvHeads, boolean gelu,
                                                boolean silu, Integer waPattern, Integer window) {
        return clipMeta(projector, image, patch, embd, ff, layers, proj, heads, kvHeads, gelu, silu, waPattern,
                window, null);
    }

    static Map<String, Object> clipMeta(String projector, int image, int patch, int embd, int ff,
                                        int layers, int proj, int heads, int kvHeads, boolean gelu,
                                        boolean silu, Integer waPattern, Integer window, Integer merge) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.has_vision_encoder", true);
        m.put("clip.projector_type", projector);
        m.put("clip.vision.image_size", image);
        m.put("clip.vision.patch_size", patch);
        m.put("clip.vision.embedding_length", embd);
        m.put("clip.vision.feed_forward_length", ff);
        m.put("clip.vision.block_count", layers);
        m.put("clip.vision.projection_dim", proj);
        m.put("clip.vision.attention.head_count", heads);
        m.put("clip.vision.attention.head_count_kv", kvHeads);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        m.put("clip.vision.image_mean", new float[]{0.48f, 0.45f, 0.4f});
        m.put("clip.vision.image_std", new float[]{0.26f, 0.26f, 0.27f});
        if (gelu) {
            m.put("clip.use_gelu", true);
        }
        if (silu) {
            m.put("clip.use_silu", true);
        }
        if (waPattern != null) {
            m.put("clip.vision.n_wa_pattern", waPattern);
        }
        if (window != null) {
            m.put("clip.vision.window_size", window);
        }
        if (merge != null) {
            m.put(mergeKey(projector), merge);
        }
        return m;
    }

    /** The key the family's converter writes the merge under (clip.cpp load_hparams). */
    static String mergeKey(String projector) {
        return projector.startsWith("qwen") || projector.equals("pixtral")
                ? "clip.vision.spatial_merge_size" : "clip.vision.projector.scale_factor";
    }

    static ClipHparams hparams(Map<String, Object> meta) {
        return ClipHparams.from(GgufKv.ofMap(meta), ProjectorModality.VISION);
    }

    /** llava-style CLIP: class token, pre/post norm, gelu FFN, MLP projector over an 8x8 image, 4x4 patches. */
    static MmprojFixture llavaFixture(long seed, boolean fusedQkv, boolean twoLayerMlp) {
        int E = 8;
        int PS = 4;
        int FF = 12;
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", PS, PS, 3, E);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, E);
        f.weight("v.class_embd", E, 1);
        f.weight("v.position_embd.weight", E, 5);
        f.norm("v.pre_ln.weight", "v.pre_ln.bias", E);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        for (int il = 0; il < 2; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", E);
            f.norm(p + "ln2.weight", p + "ln2.bias", E);
            if (fusedQkv) {
                f.weight(p + "attn_qkv.weight", E, 3 * E);
                f.weight(p + "attn_qkv.bias", 3 * E);
            } else {
                f.weight(p + "attn_q.weight", E, E);
                f.weight(p + "attn_q.bias", E);
                f.weight(p + "attn_k.weight", E, E);
                f.weight(p + "attn_k.bias", E);
                f.weight(p + "attn_v.weight", E, E);
                f.weight(p + "attn_v.bias", E);
            }
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            f.weight(p + "ffn_up.weight", E, FF);
            f.weight(p + "ffn_up.bias", FF);
            f.weight(p + "ffn_down.weight", FF, E);
            f.weight(p + "ffn_down.bias", E);
        }
        f.weight("mm.0.weight", E, 6);
        f.weight("mm.0.bias", 6);
        if (twoLayerMlp) {
            f.weight("mm.2.weight", 6, 6);
            f.weight("mm.2.bias", 6);
        }
        return f;
    }

    /** siglip-style encoder without class token or pre-norm, gemma3 pool projector over a 16x16 image. */
    static MmprojFixture siglipFixture(long seed) {
        int E = 8;
        int PS = 4;
        int FF = 12;
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", PS, PS, 3, E);
        f.weight("v.position_embd.weight", E, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        for (int il = 0; il < 2; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", E);
            f.norm(p + "ln2.weight", p + "ln2.bias", E);
            f.weight(p + "attn_qkv.weight", E, 3 * E);
            f.weight(p + "attn_qkv.bias", 3 * E);
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            f.weight(p + "ffn_up.weight", E, FF);
            f.weight(p + "ffn_up.bias", FF);
            f.weight(p + "ffn_gate.weight", E, FF);
            f.weight(p + "ffn_gate.bias", FF);
            f.weight(p + "ffn_down.weight", FF, E);
            f.weight(p + "ffn_down.bias", E);
        }
        f.weight("mm.input_projection.weight", E, 6);
        f.weight("mm.soft_emb_norm.weight", E);
        return f;
    }

    /** qwen2vl or qwen2.5vl: 8x8 image, 2x2 patches, merge 2, silu-gated FFN, merger projector. */
    static MmprojFixture qwenFixture(long seed, boolean v25) {
        int E = 8;
        int PS = 2;
        int FF = 12;
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", PS, PS, 3, E);
        if (v25) {
            f.weight("v.post_ln.weight", E);
        } else {
            f.norm("v.post_ln.weight", "v.post_ln.bias", E);
        }
        for (int il = 0; il < 3; il++) {
            String p = "v.blk." + il + ".";
            f.weight(p + "ln1.weight", E);
            f.weight(p + "ln2.weight", E);
            f.weight(p + "attn_qkv.weight", E, 3 * E);
            f.weight(p + "attn_qkv.bias", 3 * E);
            f.weight(p + "attn_out.weight", E, E);
            f.weight(p + "attn_out.bias", E);
            f.weight(p + "ffn_up.weight", E, FF);
            f.weight(p + "ffn_up.bias", FF);
            f.weight(p + "ffn_gate.weight", E, FF);
            f.weight(p + "ffn_gate.bias", FF);
            f.weight(p + "ffn_down.weight", FF, E);
            f.weight(p + "ffn_down.bias", E);
        }
        f.weight("mm.0.weight", 4 * E, 6);
        f.weight("mm.0.bias", 6);
        f.weight("mm.2.weight", 6, 6);
        f.weight("mm.2.bias", 6);
        return f;
    }

    private static CpuVisionEncoder encoder(MmprojFixture f, ClipHparams hp) {
        return CpuVisionEncoder.bind(ClipVisionModel.from(hp, f.catalog()), CpuMatVec.PLAIN,
                f::get, f::ref);
    }

    private static void assertRows(float[][] want, float[][] got) {
        assertEquals(want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(want[t].length, got[t].length);
            assertArrayEquals("row " + t, want[t], got[t], TOL);
        }
    }

    private static float[][] flatRows(List<float[]> rows) {
        return rows.toArray(new float[0][]);
    }

    // ================================================================== llava CLIP

    @Test
    public void llavaClipWithSeparateQkvMatchesTheGraph() {
        MmprojFixture f = llavaFixture(11, false, false);
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 100);
        float[][] got = enc.encode(ClipImage.of(8, 8, pixels));
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 8, 8), got);
        // the llava graph reads n_patches rows from row 1 on: patches 1..3 and the class token
        assertEquals(4, got.length);
        assertEquals(6, got[0].length);
    }

    @Test
    public void llavaClipWithFusedQkvAndTwoLayerMlpMatchesTheGraph() {
        MmprojFixture f = llavaFixture(12, true, true);
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 101);
        float[][] got = enc.encode(ClipImage.of(8, 8, pixels));
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 8, 8), got);
    }

    @Test
    public void llavaClipLegacyFfnSwapIsApplied() {
        MmprojFixture f = llavaFixture(13, false, false);
        // the old conversions store the pair swapped: the name ffn_up holds the [n_ff, n_embd] down matrix and
        // ffn_down holds the [n_embd, n_ff] up matrix (clip.cpp is_ffn_swapped)
        for (int il = 0; il < 2; il++) {
            String p = "v.blk." + il + ".";
            swap(f, p + "ffn_up.weight", p + "ffn_down.weight");
            swap(f, p + "ffn_up.bias", p + "ffn_down.bias");
        }
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 102);
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 8, 8), enc.encode(ClipImage.of(8, 8, pixels)));
    }

    static void exportSwap(MmprojFixture f, String a, String b) { swap(f, a, b); }

    static MmprojFixture exportFixture(long seed, boolean fusedQkv, boolean twoLayerMlp) {
        return llavaFixture(seed, fusedQkv, twoLayerMlp);
    }

    /**
     * The qwen fixture with the tensors of a converted Qwen2-VL file: the second Conv3d frame
     * {@code v.patch_embd.weight.1} and a merger whose hidden width ({@code 4 * n_embd}) differs from its output.
     */
    static MmprojFixture qwenTemporalFixture(long seed, boolean v25) {
        MmprojFixture f = qwenFixture(seed, v25);
        f.weight("v.patch_embd.weight.1", 2, 2, 3, 8);
        f.weight("mm.0.weight", 4 * 8, 4 * 8);
        f.weight("mm.0.bias", 4 * 8);
        f.weight("mm.2.weight", 4 * 8, 6);
        return f;
    }

    static MmprojFixture exportQwenFixture(long seed, boolean v25) {
        return qwenFixture(seed, v25);
    }

    private static void swap(MmprojFixture f, String a, String b) {
        float[] va = f.values.get(a);
        long[] sa = f.shapes.get(a);
        float[] vb = f.values.get(b);
        long[] sb = f.shapes.get(b);
        f.values.put(a, vb);
        f.shapes.put(a, sb);
        f.values.put(b, va);
        f.shapes.put(b, sa);
    }

    @Test
    public void repeatCallsAreDeterministicAndCloseRefuses() {
        MmprojFixture f = llavaFixture(14, true, false);
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        ClipImage img = ClipImage.of(8, 8, MmprojFixture.pixels(8, 8, 103));
        float[][] first = enc.encode(img);
        float[][] again = enc.encode(img);
        assertRows(first, again);
        enc.close();
        assertThrows(IllegalStateException.class, () -> enc.encode(img));
    }

    @Test
    public void wrongGridIsRefusedByName() {
        MmprojFixture f = llavaFixture(15, false, false);
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> enc.encode(ClipImage.of(12, 12, new float[12 * 12 * 3])));
        assertTrue(e.getMessage().contains("patch grid"));
        assertThrows(IllegalArgumentException.class,
                () -> enc.encode(ClipImage.of(6, 6, new float[6 * 6 * 3])));
    }

    // ================================================================== siglip + gemma3

    @Test
    public void siglipWithGemma3PoolMatchesTheGraph() {
        MmprojFixture f = siglipFixture(21);
        // neither use_gelu nor use_silu: the FFN runs gelu_quick
        ClipHparams hp = hparams(clipMeta("gemma3", 16, 4, 8, 12, 2, 6, 2, 2, false, false, null, null, 2));
        assertEquals(ClipFfnOp.GELU_QUICK, hp.ffnOp());
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(16, 16, 200);
        float[][] got = enc.encode(ClipImage.of(16, 16, pixels));
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 16, 16), got);
        assertEquals(4, got.length);
    }

    @Test
    public void siglipWithGatedGeluQuickMatchesTheGraph() {
        MmprojFixture f = siglipFixture(22);
        ClipHparams hp = hparams(clipMeta("gemma3", 16, 4, 8, 12, 2, 6, 2, 2, false, false, null, null, 2));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(16, 16, 201);
        assertRows(MmprojFixture.Reference.clip(f, hp, pixels, 16, 16), enc.encode(ClipImage.of(16, 16, pixels)));
    }

    // ================================================================== qwen2vl

    @Test
    public void qwen2vlMatchesTheGraph() {
        MmprojFixture f = qwenFixture(31, false);
        ClipHparams hp = hparams(clipMeta("qwen2vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 300);
        float[][] got = enc.encode(ClipImage.of(8, 8, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 8, 8), got);
        assertEquals(4, got.length);
    }

    @Test
    public void qwen2vlSecondFrameKernelAndWideMergerMatchTheGraph() {
        MmprojFixture f = qwenTemporalFixture(37, false);
        ClipHparams hp = hparams(clipMeta("qwen2vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        assertNotNull(enc.model().patchEmbd1());
        float[] pixels = MmprojFixture.pixels(8, 12, 301);
        float[][] got = enc.encode(ClipImage.of(8, 12, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 8, 12), got);
        assertEquals(6, got.length);
        assertEquals(6, got[0].length);
        f.remove("v.patch_embd.weight.1");
        float[][] single = encoder(f, hp).encode(ClipImage.of(8, 12, pixels));
        assertFalse("the second frame kernel contributes", Math.abs(single[0][0] - got[0][0]) < 1e-6f);
    }

    @Test
    public void qwen25vlWindowAttentionMatchesTheGraph() {
        MmprojFixture f = qwenFixture(32, true);
        // window_size 4 = one merged token per window: every mask row sees only its own 4 patches
        ClipHparams hp = hparams(clipMeta("qwen2.5vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, 2, 4));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 301);
        float[][] got = enc.encode(ClipImage.of(8, 8, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 8, 8), got);
        assertEquals(4, got.length);
    }

    @Test
    public void qwen25vlWindowReorderMovesThePositionsWithTheRows() {
        MmprojFixture f = qwenFixture(35, true);
        // a 16x8 image of 2-pixel patches is 4x2 merged tokens; 8-pixel windows hold 2x2 merged tokens, so the
        // window order {0, 1, 4, 5, 2, 3, 6, 7} differs from the raster order
        ClipHparams hp = hparams(clipMeta("qwen2.5vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, 2, 8));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(16, 8, 303);
        float[][] got = enc.encode(ClipImage.of(16, 8, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 16, 8), got);
        assertEquals(8, got.length);
    }

    @Test
    public void qwen25vlPreNormIsRms() {
        MmprojFixture f = qwenFixture(34, true);
        // qwen2vl.cpp normalizes v.pre_ln with the encoder's norm type, RMS for 2.5
        f.weight("v.pre_ln.weight", 8);
        ClipHparams hp = hparams(clipMeta("qwen2.5vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, 2, 4));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(8, 8, 302);
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 8, 8), enc.encode(ClipImage.of(8, 8, pixels)));
    }

    @Test
    public void qwen2vlRectangleMatchesTheGraph() {
        MmprojFixture f = qwenFixture(34, false);
        ClipHparams hp = hparams(clipMeta("qwen2vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        // 8 wide, 12 high: a 2x3 grid of merged tokens, so row and column positions differ
        float[] pixels = MmprojFixture.pixels(8, 12, 302);
        float[][] got = enc.encode(ClipImage.of(8, 12, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 8, 12), got);
        assertEquals(6, got.length);
    }

    @Test
    public void qwen25vlWindowsThatRegroupTokensMatchTheGraph() {
        MmprojFixture f = qwenFixture(35, true);
        // window_size 8 = 2x2 merged tokens per window over a 3x3 merged grid: windows of 4, 2, 2 and 1 tokens
        // regroup the tokens and remap the positions
        ClipHparams hp = hparams(clipMeta("qwen2.5vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, 2, 8));
        QwenVisionGrid grid = QwenVisionGrid.of(hp, 12, 12);
        assertTrue(grid.reorder());
        assertEquals(4, grid.windowCount());
        // window (0, 0) holds tokens 0, 1, 3, 4; token 2 opens the second window
        assertEquals(4, grid.idx()[2]);
        assertEquals(2, grid.invIdx()[4]);
        assertEquals(16, grid.windowEnd(0));
        assertEquals(0, grid.windowOfRow(15));
        assertEquals(1, grid.windowOfRow(16));
        CpuVisionEncoder enc = encoder(f, hp);
        float[] pixels = MmprojFixture.pixels(12, 12, 303);
        float[][] got = enc.encode(ClipImage.of(12, 12, pixels));
        assertRows(MmprojFixture.Reference.qwen(f, hp, pixels, 12, 12), got);
        assertEquals(9, got.length);
    }

    @Test
    public void qwen25vlDefaultWindowIs112Pixels() {
        Map<String, Object> meta = clipMeta("qwen2.5vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, 2, null);
        // 112 / patch 2 / merge 2 = 28 merged tokens per window side: one window over a 3x3 grid
        QwenVisionGrid grid = QwenVisionGrid.of(hparams(meta), 12, 12);
        assertEquals(1, grid.windowCount());
        assertEquals(36, grid.windowEnd(0));
        MmprojFixture f = qwenFixture(36, true);
        float[] pixels = MmprojFixture.pixels(12, 12, 304);
        assertRows(MmprojFixture.Reference.qwen(f, hparams(meta), pixels, 12, 12),
                encoder(f, hparams(meta)).encode(ClipImage.of(12, 12, pixels)));
    }

    @Test
    public void qwenRectangleGridFailsWithoutMergeAlignment() {
        MmprojFixture f = qwenFixture(33, false);
        ClipHparams hp = hparams(clipMeta("qwen2vl_merger", 8, 2, 8, 12, 3, 6, 2, 2, false, true, null, null));
        CpuVisionEncoder enc = encoder(f, hp);
        // 6x8 image: 3x4 patch grid, merge 2 cannot tile a side of 3
        assertThrows(IllegalArgumentException.class,
                () -> enc.encode(ClipImage.of(6, 8, new float[6 * 8 * 3])));
    }

    // ================================================================== geometry refusals

    @Test
    public void shapeMismatchesAreRejected() {
        MmprojFixture f = llavaFixture(41, false, false);
        f.shapes.put("v.position_embd.weight", new long[]{8, 7});
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hp, f.catalog()));
        assertTrue(e.getMessage().contains("v.position_embd.weight"));

        MmprojFixture g = llavaFixture(42, false, false);
        g.shapes.put("mm.0.weight", new long[]{8, 9});
        ClipHparams hp2 = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hp2, g.catalog())).getMessage().contains("mm.0.weight"));
    }

    @Test
    public void missingTensorsAreRejectedByTable() {
        MmprojFixture f = llavaFixture(43, false, false);
        f.remove("v.post_ln.weight");
        ClipHparams hp = hparams(clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null));
        assertThrows(com.eonmux.neuralfree.gguf.spec.MissingTensorException.class,
                () -> ClipVisionModel.from(hp, f.catalog()));
    }

    @Test
    public void hparamsAndImageLayoutRoundTrip() {
        Map<String, Object> meta = clipMeta("mlp", 8, 4, 8, 12, 2, 6, 2, 2, true, false, null, null);
        ClipHparams hp = hparams(meta);
        assertEquals(ProjectorType.MLP, hp.projectorType());
        assertEquals(4, hp.dHead());
        assertEquals(2, hp.nHeadKv());
        assertEquals(3, hp.imageMean().length);
        assertEquals(0.48f, hp.imageMean()[0], 0f);
        assertThrows(IllegalArgumentException.class, () -> ClipImage.of(0, 4, new float[12]));
        assertThrows(IllegalArgumentException.class, () -> ClipImage.of(2, 2, new float[2 * 2 * 3 + 1]));
        ClipImage img = ClipImage.of(2, 2, new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        assertEquals(7f, img.pixel(0, 1, 0), 0f);
        assertEquals(12f, img.pixel(1, 1, 2), 0f);
        List<float[]> rows = new ArrayList<>();
        rows.add(new float[]{1});
        assertEquals(1, flatRows(rows).length);
    }
}

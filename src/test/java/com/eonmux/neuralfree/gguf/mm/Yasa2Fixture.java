package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;

import java.util.Map;

/**
 * Synthetic yasa2 (ConvNeXt-V2) mmproj files. {@link #full()}: a 64-pixel tile of 4-pixel patches, a 16x16 stem
 * grid, a first stage of two blocks at 8 channels, a downsampled second stage of one block at 12 channels (8x8,
 * pool kernel 1), every optional tensor present and a position table that matches the final grid.
 * {@link #sparse()}: a 128-pixel tile, two downsamples to a 16x16 grid at 16 channels (pool kernel 2), no stem
 * norm, no GRN affine, no projector biases, and a 64-row position table the 256-token grid skips. Package-private
 * test support; the llama.cpp reference rows of both live in {@code yasa2-reference.txt}.
 */
final class Yasa2Fixture {

    static final int PATCH = 4;
    static final int OUT = 16;

    final String name;
    final MmprojFixture tensors;
    final Map<String, Object> meta;
    final int imageSize;

    private Yasa2Fixture(String name, MmprojFixture tensors, Map<String, Object> meta, int imageSize) {
        this.name = name;
        this.tensors = tensors;
        this.meta = meta;
        this.imageSize = imageSize;
    }

    /** Every optional tensor present; the final 8x8 grid pools with kernel 1. */
    static Yasa2Fixture full() {
        MmprojFixture f = new MmprojFixture(91);
        f.weight("v.patch_embd.weight", PATCH, PATCH, 3, 8);
        f.put("v.patch_embd.bias", -0.1f, 0.1f, 8);
        f.norm("v.patch_ln.weight", "v.patch_ln.bias", 8);
        block(f, "v.stage.0.blk.0.", 8, true);
        block(f, "v.stage.0.blk.1.", 8, true);
        f.norm("v.stage.1.down.ln.weight", "v.stage.1.down.ln.bias", 8);
        f.weight("v.stage.1.down.conv.weight", 2, 2, 8, 12);
        f.put("v.stage.1.down.conv.bias", -0.1f, 0.1f, 12);
        block(f, "v.stage.1.blk.0.", 12, true);
        f.norm("v.backbone_ln.weight", "v.backbone_ln.bias", 12);
        f.put("v.vision_pos_embed", -0.3f, 0.3f, 12, 64);
        f.weight("mm.0.weight", 12, 20);
        f.put("mm.0.bias", -0.1f, 0.1f, 20);
        f.weight("mm.2.weight", 20, OUT);
        f.put("mm.2.bias", -0.1f, 0.1f, OUT);
        return new Yasa2Fixture("full", f, meta(64, 12), 64);
    }

    /** Optional tensors left out; the final 16x16 grid pools with kernel 2 and skips its position table. */
    static Yasa2Fixture sparse() {
        MmprojFixture f = new MmprojFixture(92);
        f.weight("v.patch_embd.weight", PATCH, PATCH, 3, 4);
        block(f, "v.stage.0.blk.0.", 4, false);
        f.weight("v.stage.1.down.conv.weight", 2, 2, 4, 8);
        block(f, "v.stage.1.blk.0.", 8, false);
        f.norm("v.stage.2.down.ln.weight", "v.stage.2.down.ln.bias", 8);
        f.weight("v.stage.2.down.conv.weight", 2, 2, 8, 16);
        f.put("v.stage.2.down.conv.bias", -0.1f, 0.1f, 16);
        block(f, "v.stage.2.blk.0.", 16, false);
        f.put("v.vision_pos_embed", -0.3f, 0.3f, 16, 64);
        f.weight("mm.0.weight", 16, 24);
        f.weight("mm.2.weight", 24, OUT);
        return new Yasa2Fixture("sparse", f, meta(128, 16), 128);
    }

    /** One ConvNeXt block of {@code ch} channels and a {@code 4 * ch} hidden width. */
    private static void block(MmprojFixture f, String p, int ch, boolean affine) {
        f.weight(p + "dw.weight", Yasa2Graph.DW_KERNEL, Yasa2Graph.DW_KERNEL, 1, ch);
        f.put(p + "dw.bias", -0.1f, 0.1f, ch);
        f.norm(p + "ln.weight", p + "ln.bias", ch);
        f.weight(p + "pw1.weight", ch, 4 * ch);
        f.put(p + "pw1.bias", -0.1f, 0.1f, 4 * ch);
        if (affine) {
            f.put(p + "grn.weight", -0.5f, 0.5f, 4 * ch);
            f.put(p + "grn.bias", -0.1f, 0.1f, 4 * ch);
        }
        f.weight(p + "pw2.weight", 4 * ch, ch);
        f.put(p + "pw2.bias", -0.1f, 0.1f, ch);
    }

    /** The {@code clip.*} keys of a yasa2 file: no transformer blocks, the projector width from {@code mm.2}. */
    private static Map<String, Object> meta(int imageSize, int nEmbd) {
        Map<String, Object> m = VisionFamilyFixtures.meta("yasa2", imageSize, PATCH, OUT);
        m.put("clip.vision.block_count", 0);
        m.put("clip.vision.feed_forward_length", 0);
        m.put("clip.vision.embedding_length", nEmbd);
        m.put("clip.vision.attention.head_count", 1);
        m.put("clip.use_gelu", true);
        return m;
    }

    /**
     * The GGUF writer of the file. The depthwise kernels are stored as F16, as the converter writes them and as
     * {@code ggml_conv_2d_dw} requires on the CPU backend; everything else is F32.
     */
    GgufTestWriter gguf() {
        GgufTestWriter w = VisionFamilyFixtures.gguf(tensors, meta);
        for (String name : w.tensorNames()) {
            if (name.endsWith(".dw.weight")) {
                w.tensorFromFloats(name, GGMLType.F16, w.tensorShape(name), tensors.get(name));
            }
        }
        return w;
    }

    /** The deterministic normalized test image of this fixture. */
    float[] pixels() {
        return MmprojFixture.pixels(imageSize, imageSize, 93);
    }

    ClipHparams hparams() {
        return VisionFamilyFixtures.hparams(meta);
    }
}

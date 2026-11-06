package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesTest.assertRows;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.encoder;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.hparams;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixturesB.OUT;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU encoder of the stacked-feature llava, ldp, ldpv2, glm-edge, cogvlm, glm4v, kimivl, lfm2, d1omni_v,
 * dots_ocr, youtuvl and granite4_vision families against the double-precision transcription of their llama.cpp
 * graphs ({@link VisionFamilyReferenceB}), plus the hparams each family reads and the inputs each refuses by name.
 */
public class VisionFamiliesBTest {

    private static float[] px(int width, int height, long seed) {
        return MmprojFixture.pixels(width, height, seed);
    }

    private static <T extends Throwable> String message(Class<T> type, org.junit.function.ThrowingRunnable r) {
        return assertThrows(type, r).getMessage();
    }

    // ================================================================== llava graph

    @Test
    public void llavaStacksTheFeatureLayersAlongTheChannels() {
        MmprojFixture f = VisionFamilyFixturesB.llavaStacked(101);
        ClipHparams hp = hparams(VisionFamilyFixturesB.llavaStackedMeta());
        assertEquals(3 * E, LlavaGraph.featureWidth(hp));
        assertEquals(2, LlavaGraph.maxFeatureLayer(hp));
        float[] p = px(16, 16, 1001);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 16, p));
        assertRows(MmprojFixture.Reference.clip(f, hp, p, 16, 16), got);
        assertEquals(16, got.length);
        assertEquals(OUT, got[0].length);
    }

    @Test
    public void llavaStackRefusesAProjectorOfTheUnstackedWidth() {
        MmprojFixture f = VisionFamilyFixturesB.llavaStacked(102);
        f.weight("mm.0.weight", E, OUT);
        String msg = message(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.llavaStackedMeta()), f.catalog()));
        assertTrue(msg, msg.contains("mm.0.weight"));
    }

    @Test
    public void ldpMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.ldp(103);
        ClipHparams hp = hparams(VisionFamilyFixturesB.llavaMeta("ldp"));
        float[] p = px(16, 16, 1002);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 16, p));
        assertRows(VisionFamilyReferenceB.ldp(f, hp, p, 16), got);
        // the stride-2 block halves the 4x4 grid
        assertEquals(4, got.length);
    }

    @Test
    public void ldpRefusesAKernelThatIsNotThreeByThree() {
        MmprojFixture f = VisionFamilyFixturesB.ldp(104);
        f.weight("mm.model.mb_block.1.block.0.0.weight", 9, OUT);
        String msg = message(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.llavaMeta("ldp")), f.catalog()));
        assertTrue(msg, msg.contains("mm.model.mb_block.1.block.0.0.weight"));
        MmprojFixture g = VisionFamilyFixturesB.ldp(105);
        g.remove("mm.model.mb_block.2.block.1.fc2.bias");
        assertThrows(MissingTensorException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.llavaMeta("ldp")), g.catalog()));
    }

    @Test
    public void ldpv2MatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.ldpv2(106);
        ClipHparams hp = hparams(VisionFamilyFixturesB.llavaMeta("ldpv2"));
        float[] p = px(16, 16, 1003);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 16, p));
        assertRows(VisionFamilyReferenceB.ldpv2(f, hp, p, 16), got);
        assertEquals(4, got.length);
        assertThrows(IllegalArgumentException.class,
                () -> encoder(f, hp).encode(ClipImage.of(8, 8, px(8, 8, 1))));
    }

    @Test
    public void glmEdgeRunsTheWholeTowerAndFramesTheAdapterOutput() {
        MmprojFixture f = VisionFamilyFixturesB.glmEdge(107);
        ClipHparams hp = hparams(VisionFamilyFixturesB.llavaMeta("adapter"));
        assertEquals(VisionFamilyFixtures.LAYERS, LlavaGraph.maxFeatureLayer(hp));
        float[] p = px(16, 16, 1004);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 16, p));
        assertRows(VisionFamilyReferenceB.glmEdge(f, hp, p, 16), got);
        assertEquals(1 + 4 + 1, got.length);
        assertArrayEquals(f.get("adapter.boi"), got[0], 0f);
        assertArrayEquals(f.get("adapter.eoi"), got[5], 0f);
    }

    // ================================================================== cogvlm, glm4v

    @Test
    public void cogvlmPostNormBlocksMatchTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.cogvlm(108);
        ClipHparams hp = hparams(VisionFamilyFixturesB.cogvlmMeta());
        float[] p = px(8, 8, 1005);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 8, p));
        assertRows(VisionFamilyReferenceB.cogvlm(f, hp, p, 8), got);
        assertEquals(1 + 4 + 1, got.length);
        assertArrayEquals(f.get("v.boi"), got[0], 0f);
    }

    @Test
    public void cogvlmRefusesAFileWithoutItsClassTokenOrFusedQkvBias() {
        MmprojFixture f = VisionFamilyFixturesB.cogvlm(109);
        f.remove("v.class_embd");
        ClipHparams hp = hparams(VisionFamilyFixturesB.cogvlmMeta());
        assertTrue(message(MissingTensorException.class, () -> ClipVisionModel.from(hp, f.catalog()))
                .contains("v.class_embd"));
        MmprojFixture g = VisionFamilyFixturesB.cogvlm(110);
        g.remove("v.blk.1.attn_qkv.bias");
        assertTrue(message(MissingTensorException.class, () -> ClipVisionModel.from(hp, g.catalog()))
                .contains("v.blk.1.attn_qkv.bias"));
    }

    @Test
    public void glm4vMatchesTheGraphOnARectangleWithResizedPositions() {
        MmprojFixture f = VisionFamilyFixturesB.glm4v(111, false);
        ClipHparams hp = hparams(VisionFamilyFixturesB.glm4vMeta());
        assertEquals(2, hp.nMerge());
        float[] p = px(8, 16, 1006);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 16, p));
        assertRows(VisionFamilyReferenceB.glm4v(f, hp, p, 8, 16), got);
        assertEquals(4 * 8 / 4, got.length);
    }

    @Test
    public void glmOcrWithoutPositionsOrEmbeddingNormMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.glm4v(112, true);
        ClipHparams hp = hparams(VisionFamilyFixturesB.glm4vMeta());
        float[] p = px(8, 8, 1007);
        assertRows(VisionFamilyReferenceB.glm4v(f, hp, p, 8, 8), encoder(f, hp).encode(ClipImage.of(8, 8, p)));
    }

    @Test
    public void glm4vBicubicResizeIsTheIdentityAtTheStoredSide() {
        float[] grid = px(3, 3, 1008);
        assertArrayEquals(grid, Glm4vGraph.resizeBicubic(grid, 3, 3, 3, 3), 1e-6f);
        double[][] want = VisionFamilyReferenceB.bicubic(grid, 3, 3, 5, 2);
        float[] got = Glm4vGraph.resizeBicubic(grid, 3, 3, 5, 2);
        for (int t = 0; t < want.length; t++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(want[t][c], got[t * 3 + c], 1e-5);
            }
        }
    }

    @Test
    public void glm4vRefusesOddGroupGridsAClassTokenAndAMissingTemporalKernel() {
        MmprojFixture f = VisionFamilyFixturesB.glm4v(113, false);
        ClipHparams hp = hparams(VisionFamilyFixturesB.glm4vMeta());
        String msg = message(IllegalArgumentException.class,
                () -> encoder(f, hp).encode(ClipImage.of(6, 8, px(6, 8, 1))));
        assertTrue(msg, msg.contains("multiple of 2"));
        MmprojFixture g = VisionFamilyFixturesB.glm4v(114, false);
        g.weight("v.class_embd", E, 1);
        assertTrue(message(IllegalArgumentException.class, () -> ClipVisionModel.from(hp, g.catalog()))
                .contains("v.class_embd"));
        MmprojFixture h = VisionFamilyFixturesB.glm4v(115, false);
        h.remove("v.patch_embd.weight.1");
        assertTrue(message(MissingTensorException.class, () -> ClipVisionModel.from(hp, h.catalog()))
                .contains("v.patch_embd.weight.1"));
    }

    // ================================================================== kimivl, lfm2, d1omni_v, dots_ocr

    @Test
    public void kimivlMatchesTheGraphWithResizedPositionsAndTwoHalvesRope() {
        MmprojFixture f = VisionFamilyFixturesB.kimivl(116);
        ClipHparams hp = hparams(VisionFamilyFixturesB.kimivlMeta());
        float[] p = px(8, 12, 1009);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 12, p));
        assertRows(VisionFamilyReferenceB.kimivl(f, hp, p, 8, 12), got);
        assertEquals(2 * 3, got.length);
    }

    @Test
    public void lfm2AndD1omniMatchTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.lfm2(117, false);
        ClipHparams hp = hparams(VisionFamilyFixturesB.lfm2Meta("lfm2"));
        float[] p = px(8, 12, 1010);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 12, p));
        assertRows(VisionFamilyReferenceB.lfm2(f, hp, p, 8, 12, false), got);
        assertEquals(2 * 3, got.length);

        // d1-omni: the input bias without its norm, and the erf GELU
        MmprojFixture g = VisionFamilyFixturesB.lfm2(118, true);
        ClipHparams hg = hparams(VisionFamilyFixturesB.lfm2Meta("d1omni_v"));
        float[] q = px(8, 8, 1011);
        assertRows(VisionFamilyReferenceB.lfm2(g, hg, q, 8, 8, true), encoder(g, hg).encode(ClipImage.of(8, 8, q)));
    }

    @Test
    public void dotsOcrMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.dotsOcr(119);
        ClipHparams hp = hparams(VisionFamilyFixturesB.dotsOcrMeta());
        float[] p = px(12, 8, 1012);
        float[][] got = encoder(f, hp).encode(ClipImage.of(12, 8, p));
        assertRows(VisionFamilyReferenceB.dotsOcr(f, hp, p, 12, 8), got);
        assertEquals(3 * 2, got.length);
    }

    @Test
    public void dotsOcrRequiresItsScaleFactorAndTrunkNorm() {
        Map<String, Object> m = VisionFamilyFixturesB.dotsOcrMeta();
        m.remove("clip.vision.projector.scale_factor");
        assertTrue(message(GgufFormatException.class, () -> hparams(m)).contains("scale_factor"));
        MmprojFixture f = VisionFamilyFixturesB.dotsOcr(120);
        f.remove("mm.post_norm.weight");
        assertTrue(message(MissingTensorException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.dotsOcrMeta()), f.catalog()))
                .contains("mm.post_norm.weight"));
    }

    // ================================================================== youtuvl

    @Test
    public void youtuvlWindowAttentionMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.youtuvl(121);
        ClipHparams hp = hparams(VisionFamilyFixturesB.youtuvlMeta(1));
        assertArrayEquals(new int[]{1}, hp.visionFamily().fullAttnLayers());
        // 16x8 of 2-pixel patches: 4x2 merged tokens in 2x2-token windows, a reorder that is not the identity
        float[] p = px(16, 8, 1013);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 8, p));
        assertRows(VisionFamilyReferenceB.youtuvl(f, hp, p, 16, 8, new int[]{1}), got);
        assertEquals(8, got.length);
    }

    @Test
    public void youtuvlWithoutWindowLayersAttendsOverTheWholeImage() {
        MmprojFixture f = VisionFamilyFixturesB.youtuvl(122);
        ClipHparams hp = hparams(VisionFamilyFixturesB.youtuvlMeta());
        float[] p = px(16, 8, 1014);
        assertRows(VisionFamilyReferenceB.youtuvl(f, hp, p, 16, 8, new int[0]),
                encoder(f, hp).encode(ClipImage.of(16, 8, p)));
    }

    @Test
    public void youtuvlRefusesMissingWindowKeysAndAGatedFfn() {
        Map<String, Object> m = VisionFamilyFixturesB.youtuvlMeta(1);
        m.remove("clip.vision.wa_layer_indexes");
        assertTrue(message(GgufFormatException.class, () -> hparams(m)).contains("wa_layer_indexes"));
        Map<String, Object> range = VisionFamilyFixturesB.youtuvlMeta(VisionFamilyFixtures.LAYERS);
        assertTrue(message(GgufFormatException.class, () -> hparams(range)).contains("out of range"));
        MmprojFixture f = VisionFamilyFixturesB.youtuvl(123);
        f.weight("v.blk.0.ffn_gate.weight", E, VisionFamilyFixtures.FF);
        assertTrue(message(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.youtuvlMeta(1)), f.catalog()))
                .contains("v.blk.0.ffn_gate.weight"));
    }

    // ================================================================== granite4_vision

    @Test
    public void granite4TileMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixturesB.granite4(124);
        ClipHparams hp = hparams(VisionFamilyFixturesB.granite4Meta());
        assertArrayEquals(new int[]{3, -1}, hp.visionFamily().spatialOffsets());
        CpuVisionEncoder enc = encoder(f, hp);
        assertEquals(2 * OUT, enc.model().outputDim());
        float[] p = px(16, 16, 1015);
        float[][] got = enc.encode(ClipImage.of(16, 16, p));
        float[][] want = MmprojFixture.Reference.toFloat(VisionFamilyReferenceB.granite4Tile(f, hp, p, 16));
        assertRows(want, got);
        // (8 / window 4) * query 2 = 4 tokens per side
        assertEquals(16, got.length);
    }

    @Test
    public void granite4AnyresAssemblesTheOverviewAndTheUnpaddedGrid() {
        MmprojFixture f = VisionFamilyFixturesB.granite4(125);
        ClipHparams hp = hparams(VisionFamilyFixturesB.granite4Meta());
        float[][] tiles = {px(16, 16, 1016), px(16, 16, 1017), px(16, 16, 1018)};
        List<ClipImage> grid = Arrays.asList(ClipImage.of(16, 16, tiles[1]), ClipImage.of(16, 16, tiles[2]));
        float[][] got = encoder(f, hp).encodeAnyres(ClipImage.of(16, 16, tiles[0]), grid, 2, 1, 30, 10);
        assertRows(VisionFamilyReferenceB.granite4Anyres(f, hp, tiles, 16, 2, 1, 30, 10), got);
        // 16 overview tokens, then the 8x4 grid cropped to 8x2 by the 3:1 aspect, a newline after each row
        assertEquals(16 + 2 * (8 + 1), got.length);
        assertArrayEquals(new int[]{0, 1, 8, 2}, AnyresGrid.unpad(8, 4, 30, 10));
        assertArrayEquals(new int[]{2, 0, 4, 8}, AnyresGrid.unpad(8, 8, 10, 20));
    }

    @Test
    public void granite4RefusesMismatchedGridsAndOtherEncoders() {
        MmprojFixture f = VisionFamilyFixturesB.granite4(126);
        CpuVisionEncoder enc = encoder(f, hparams(VisionFamilyFixturesB.granite4Meta()));
        ClipImage tile = ClipImage.of(16, 16, px(16, 16, 1));
        assertTrue(message(IllegalArgumentException.class,
                () -> enc.encodeAnyres(tile, Collections.singletonList(tile), 2, 1, 30, 10)).contains("2 tiles"));
        assertThrows(IllegalArgumentException.class,
                () -> enc.encodeAnyres(tile, Collections.singletonList(tile), 1, 1, 0, 10));
        CpuVisionEncoder other = encoder(VisionFamilyFixturesB.cogvlm(127),
                hparams(VisionFamilyFixturesB.cogvlmMeta()));
        ClipImage small = ClipImage.of(8, 8, px(8, 8, 2));
        assertTrue(message(IllegalArgumentException.class,
                () -> other.encodeAnyres(small, Collections.singletonList(small), 1, 1, 8, 8))
                .contains("granite4_vision"));
    }

    @Test
    public void granite4RefusesInconsistentProjectorKeysAndGeometry() {
        Map<String, Object> m = VisionFamilyFixturesB.granite4Meta();
        m.put("clip.vision.projector.spatial_offsets", new int[]{3});
        assertTrue(message(GgufFormatException.class, () -> hparams(m)).contains("spatial_offsets"));
        Map<String, Object> q = VisionFamilyFixturesB.granite4Meta();
        q.remove("clip.vision.projector.query_side");
        assertTrue(message(GgufFormatException.class, () -> hparams(q)).contains("query_side"));
        Map<String, Object> w = VisionFamilyFixturesB.granite4Meta();
        w.put("clip.vision.projector.window_side", 3);
        MmprojFixture f = VisionFamilyFixturesB.granite4(128);
        assertTrue(message(IllegalArgumentException.class, () -> ClipVisionModel.from(hparams(w), f.catalog()))
                .contains("window_side"));
        MmprojFixture g = VisionFamilyFixturesB.granite4(129);
        g.remove("v.proj_blk.1.cross_attn_v.bias");
        assertTrue(message(MissingTensorException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixturesB.granite4Meta()), g.catalog()))
                .contains("v.proj_blk.1.cross_attn_v.bias"));
    }
}

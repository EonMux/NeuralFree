package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesCFixtures.EXPERTS;
import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesCFixtures.EXPERTS_USED;
import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesCFixtures.OUT;
import static com.eonmux.neuralfree.gguf.mm.VisionFamiliesTest.assertRows;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.LAYERS;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.encoder;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.hparams;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU encoder of the qwen3vl, paddleocr, kimivl, dots_ocr, dots3note_v, hunyuanvl, janus_pro, lfm2, d1omni_v,
 * cohere2v and lightonocr families against the double-precision transcription of their llama.cpp graphs
 * ({@link VisionFamiliesCReference}, {@link VisionFamilyReference}), plus the hparams each family reads and the
 * inputs each refuses by name.
 */
public class VisionFamiliesCTest {

    /** The reference graph of one family. */
    private interface Reference {
        float[][] encode(MmprojFixture f, ClipHparams hp, float[] px, int width, int height);
    }

    /** Encodes a {@code width x height} image and checks it against {@code ref}; returns the rows. */
    private static float[][] check(MmprojFixture f, Map<String, Object> meta, Reference ref, int width, int height,
                                   int tokens, int outWidth) {
        ClipHparams hp = hparams(meta);
        CpuVisionEncoder enc = encoder(f, hp);
        float[] px = MmprojFixture.pixels(width, height, 1000L + width * 31L + height);
        float[][] got = enc.encode(ClipImage.of(width, height, px));
        assertEquals(tokens, got.length);
        assertEquals(outWidth, enc.model().outputDim());
        assertRows(ref.encode(f, hp, px, width, height), got);
        return got;
    }

    private static IllegalArgumentException refuse(MmprojFixture f, Map<String, Object> meta, int width,
                                                   int height) {
        CpuVisionEncoder enc = encoder(f, hparams(meta));
        return assertThrows(IllegalArgumentException.class, () -> enc.encode(ClipImage.of(width, height,
                MmprojFixture.pixels(width, height, 3))));
    }

    // ================================================================== qwen3vl

    @Test
    public void qwen3vlMatchesTheGraphWithDeepstackOnResizedPositions() {
        // 6x4 patches: the 4x4 position grid is resized (align corners), 3x2 merged tokens, two deepstack blocks
        check(VisionFamiliesCFixtures.qwen3vl(61), VisionFamiliesCFixtures.qwen3vlMeta(),
                VisionFamiliesCReference::qwen3vl, 12, 8, 6, OUT * (1 + LAYERS));
        // the 4x4 grid itself
        check(VisionFamiliesCFixtures.qwen3vl(62), VisionFamiliesCFixtures.qwen3vlMeta(),
                VisionFamiliesCReference::qwen3vl, 8, 8, 4, OUT * (1 + LAYERS));
    }

    @Test
    public void qwen3vlWithOneDeepstackBlockWidensOnce() {
        MmprojFixture f = VisionFamiliesCFixtures.qwen3vl(63);
        for (String n : new String[]{"norm.weight", "norm.bias", "fc1.weight", "fc1.bias", "fc2.weight",
                "fc2.bias"}) {
            f.remove("v.deepstack.0." + n);
        }
        check(f, VisionFamiliesCFixtures.qwen3vlMeta(), VisionFamiliesCReference::qwen3vl, 8, 12, 6, 2 * OUT);
    }

    @Test
    public void qwen3vlReadsTheMergeAndRefusesOddGrids() {
        Map<String, Object> meta = VisionFamiliesCFixtures.qwen3vlMeta();
        assertEquals(2, hparams(meta).nMerge());
        IllegalArgumentException e = refuse(VisionFamiliesCFixtures.qwen3vl(64), meta, 6, 8);
        assertTrue(e.getMessage(), e.getMessage().contains("even"));
        meta.put("clip.vision.spatial_merge_size", 4);
        IllegalArgumentException merge = assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(meta), VisionFamiliesCFixtures.qwen3vl(64).catalog()));
        assertTrue(merge.getMessage(), merge.getMessage().contains("spatial_merge_size"));
    }

    // ================================================================== paddleocr / kimivl

    @Test
    public void paddleocrMatchesTheGraph() {
        check(VisionFamiliesCFixtures.paddleocr(65), VisionFamiliesCFixtures.paddleocrMeta(),
                VisionFamiliesCReference::paddleocr, 12, 8, 6, OUT);
        check(VisionFamiliesCFixtures.paddleocr(66), VisionFamiliesCFixtures.paddleocrMeta(),
                VisionFamiliesCReference::paddleocr, 8, 8, 4, OUT);
    }

    @Test
    public void paddleocrFixesTheMergeAndRefusesUnalignedGrids() {
        Map<String, Object> meta = VisionFamiliesCFixtures.paddleocrMeta();
        meta.put("clip.vision.spatial_merge_size", 3);
        assertEquals(2, hparams(meta).nMerge());
        IllegalArgumentException e = refuse(VisionFamiliesCFixtures.paddleocr(67), meta, 6, 8);
        assertTrue(e.getMessage(), e.getMessage().contains("paddleocr"));
    }

    @Test
    public void kimivlMatchesTheGraphWithPaddedMerge() {
        // 3x2 patches pad to 4x2 before the merge: the padded chunks normalize to the norm bias
        check(VisionFamiliesCFixtures.kimivl(68), VisionFamiliesCFixtures.kimivlMeta(),
                VisionFamiliesCReference::kimivl, 6, 4, 2, OUT);
        check(VisionFamiliesCFixtures.kimivl(69), VisionFamiliesCFixtures.kimivlMeta(),
                VisionFamiliesCReference::kimivl, 8, 12, 6, OUT);
    }

    // ================================================================== dots

    @Test
    public void dotsOcrMatchesTheGraph() {
        check(VisionFamiliesCFixtures.dotsOcr(70), VisionFamiliesCFixtures.dotsOcrMeta(),
                VisionFamiliesCReference::dots, 12, 8, 6, OUT);
    }

    @Test
    public void dots3noteRoutesItsMoeBlockAsTheGraph() {
        MmprojFixture f = VisionFamiliesCFixtures.dots3note(71);
        ClipHparams hp = hparams(VisionFamiliesCFixtures.dots3noteMeta());
        assertEquals(EXPERTS_USED, hp.nExpertUsed());
        ClipVisionModel model = ClipVisionModel.from(hp, f.catalog());
        assertFalse(model.layers().get(0).moe());
        assertTrue(model.layers().get(LAYERS - 1).moe());
        assertEquals(EXPERTS, model.layers().get(LAYERS - 1).nExpert());
        check(f, VisionFamiliesCFixtures.dots3noteMeta(), VisionFamiliesCReference::dots, 8, 8, 4, OUT);
        check(VisionFamiliesCFixtures.dots3note(72), VisionFamiliesCFixtures.dots3noteMeta(),
                VisionFamiliesCReference::dots, 8, 12, 6, OUT);
    }

    @Test
    public void dotsRefusesWhatTheMoeGraphCannotRun() {
        Map<String, Object> noUsed = VisionFamiliesCFixtures.dots3noteMeta();
        noUsed.remove("clip.vision.expert_used_count");
        RuntimeException e = assertThrows(RuntimeException.class, () -> hparams(noUsed));
        assertTrue(e.getMessage(), e.getMessage().contains("expert_used_count"));

        Map<String, Object> gelu = VisionFamiliesCFixtures.dots3noteMeta();
        gelu.remove("clip.use_silu");
        gelu.put("clip.use_gelu", true);
        IllegalArgumentException op = assertThrows(IllegalArgumentException.class,
                () -> encoder(VisionFamiliesCFixtures.dots3note(73), hparams(gelu)));
        assertTrue(op.getMessage(), op.getMessage().contains("SiLU"));

        IllegalArgumentException grid = refuse(VisionFamiliesCFixtures.dotsOcr(74),
                VisionFamiliesCFixtures.dotsOcrMeta(), 6, 8);
        assertTrue(grid.getMessage(), grid.getMessage().contains("dots"));
    }

    @Test
    public void topKPicksTheLargestBiasedProbabilitiesHighestFirst() {
        float[] probs = {0.9f, 0.2f, 0.5f, 0.6f};
        assertArrayEquals(new int[]{0, 3}, CpuVitTower.topK(probs, 0, 4, null, 2));
        assertArrayEquals(new int[]{1, 0}, CpuVitTower.topK(probs, 0, 4, new float[]{0f, 1f, 0f, 0f}, 2));
    }

    // ================================================================== hunyuanvl

    @Test
    public void hunyuanvlMatchesTheGraph() {
        // 6x4 patches: 3x2 merged tokens, a newline per row, begin and end: (3 + 1) * 2 + 2
        check(VisionFamiliesCFixtures.hunyuanvl(75), VisionFamiliesCFixtures.hunyuanvlMeta(),
                VisionFamiliesCReference::hunyuanvl, 12, 8, 10, OUT);
        // 7x4 patches: the unpadded conv drops the last column
        check(VisionFamiliesCFixtures.hunyuanvl(76), VisionFamiliesCFixtures.hunyuanvlMeta(),
                VisionFamiliesCReference::hunyuanvl, 14, 8, 10, OUT);
        assertEquals(ClipFfnOp.GELU, hparams(VisionFamiliesCFixtures.hunyuanvlMeta()).ffnOp());
    }

    // ================================================================== siglip projectors

    @Test
    public void janusMatchesTheGraphAndKeepsTheImageSizeGrid() {
        check(VisionFamiliesCFixtures.janus(77), VisionFamiliesCFixtures.janusMeta(),
                VisionFamiliesCReference::janus, 8, 8, 16, OUT);
        IllegalArgumentException e = refuse(VisionFamiliesCFixtures.janus(78), VisionFamiliesCFixtures.janusMeta(),
                12, 8);
        assertTrue(e.getMessage(), e.getMessage().contains("image_size"));
    }

    @Test
    public void lfm2AndD1omniMatchTheGraph() {
        check(VisionFamiliesCFixtures.lfm2(79, true), VisionFamiliesCFixtures.lfm2Meta("lfm2"),
                VisionFamiliesCReference::lfm2, 12, 8, 6, OUT);
        // d1-omni without the input norm, exact GELU, a 5x4 grid padded to 6x4
        check(VisionFamiliesCFixtures.lfm2(80, false), VisionFamiliesCFixtures.lfm2Meta("d1omni_v"),
                VisionFamiliesCReference::lfm2, 10, 8, 6, OUT);
    }

    @Test
    public void cohere2vMatchesTheGraph() {
        check(VisionFamiliesCFixtures.cohere2v(81), VisionFamiliesCFixtures.cohere2vMeta(),
                VisionFamiliesCReference::cohere2v, 8, 8, 4, OUT);
    }

    @Test
    public void lightonocrRunsThePixtralGraphWithoutBreakRows() {
        check(VisionFamiliesCFixtures.lightonocr(82), VisionFamiliesCFixtures.lightonocrMeta(),
                VisionFamilyReference::pixtral, 8, 12, 6, OUT);
    }

    // ================================================================== shared pieces

    @Test
    public void qwenVisionRopeRestartsTheFrequenciesForTheColumnSection() {
        int dh = 8;
        float[] x = new float[dh];
        for (int i = 0; i < dh; i++) {
            x[i] = 0.25f + i;
        }
        float[] got = x.clone();
        // the token holds patch (row 3, column 5) of a grid 7 patches wide
        int[] pos = QwenVisionGrid.ropePositions(new int[]{3 * 7 + 5}, 7);
        assertArrayEquals(new int[]{3, 5, 3, 5}, pos);
        CpuRope.apply(RopeFreqTable.of(QwenVisionGrid.rope(dh), null), got, 0, 1, 1, dh, dh, 0, pos, 0, 1);
        // pairs (j, j + 4): j = 0, 1 rotate by row 3, j = 2, 3 by column 5, each at 10000^(-4k/8), k = 0, 1
        double[] angles = {3, 3 * Math.pow(10000, -0.5), 5, 5 * Math.pow(10000, -0.5)};
        for (int j = 0; j < 4; j++) {
            double c = Math.cos(angles[j]);
            double s = Math.sin(angles[j]);
            assertEquals(x[j] * c - x[j + 4] * s, got[j], 1e-5);
            assertEquals(x[j] * s + x[j + 4] * c, got[j + 4], 1e-5);
        }
    }

    @Test
    public void alignCornersResizeSamplesTheGridCorners() {
        float[] pos = new float[4 * 4];
        for (int i = 0; i < pos.length; i++) {
            pos[i] = i;
        }
        float[] got = SiglipResizedGraph.alignCorners(pos, 1, 4, 7, 3);
        double[][] want = VisionFamiliesCReference.alignCorners(pos, 1, 4, 7, 3);
        for (int t = 0; t < want.length; t++) {
            assertEquals(want[t][0], got[t], 1e-5);
        }
        assertEquals(0f, got[0], 0f);
        assertEquals(15f, got[got.length - 1], 1e-5f);
    }
}

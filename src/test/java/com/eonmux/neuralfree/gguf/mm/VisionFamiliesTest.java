package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import org.junit.Test;

import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.E;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.QUERIES;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.RESAMPLER_D;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.encoder;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.hparams;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The CPU encoder of the minicpmv, internvl, pixtral, idefics3, llama4 and phi4 families against the
 * double-precision transcription of their llama.cpp graphs ({@link VisionFamilyReference}), plus the hparams
 * each family reads and the inputs each refuses by name.
 */
public class VisionFamiliesTest {

    private static final float TOL = 2e-4f;

    static void assertRows(float[][] want, float[][] got) {
        assertEquals("token count", want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertEquals(want[t].length, got[t].length);
            assertArrayEquals("row " + t, want[t], got[t], TOL);
        }
    }

    private static float[][] encode(CpuVisionEncoder enc, int width, int height, long seed) {
        return enc.encode(ClipImage.of(width, height, MmprojFixture.pixels(width, height, seed)));
    }

    // ================================================================== minicpmv

    @Test
    public void minicpmvResamplerMatchesTheGraphOnARectangularSlice() {
        MmprojFixture f = VisionFamilyFixtures.minicpmv(51);
        ClipHparams hp = hparams(VisionFamilyFixtures.minicpmvMeta());
        CpuVisionEncoder enc = encoder(f, hp);
        // a 12x8 slice: 3x2 patches, the bucket positions and sin-cos keys see a non-square grid
        float[] px = MmprojFixture.pixels(12, 8, 500);
        float[][] got = enc.encode(ClipImage.of(12, 8, px));
        assertRows(VisionFamilyReference.minicpmv(f, hp, px, 12, 8), got);
        assertEquals(QUERIES, got.length);
        assertEquals(RESAMPLER_D, got[0].length);
        assertEquals(RESAMPLER_D, enc.model().outputDim());
    }

    @Test
    public void minicpmvEverySliceSizeEncodesToTheQueryCount() {
        MmprojFixture f = VisionFamilyFixtures.minicpmv(52);
        ClipHparams hp = hparams(VisionFamilyFixtures.minicpmvMeta());
        CpuVisionEncoder enc = encoder(f, hp);
        float[] px = MmprojFixture.pixels(8, 16, 501);
        assertRows(VisionFamilyReference.minicpmv(f, hp, px, 8, 16), enc.encode(ClipImage.of(8, 16, px)));
        assertEquals(QUERIES, encode(enc, 4, 4, 502).length);
    }

    @Test
    public void minicpmvLegacyFfnSwapIsApplied() {
        MmprojFixture f = VisionFamilyFixtures.minicpmv(53);
        for (int il = 0; il < VisionFamilyFixtures.LAYERS; il++) {
            String p = "v.blk." + il + ".";
            CpuVisionEncoderTest.exportSwap(f, p + "ffn_up.weight", p + "ffn_down.weight");
            CpuVisionEncoderTest.exportSwap(f, p + "ffn_up.bias", p + "ffn_down.bias");
        }
        ClipHparams hp = hparams(VisionFamilyFixtures.minicpmvMeta());
        float[] px = MmprojFixture.pixels(8, 8, 503);
        assertRows(VisionFamilyReference.minicpmv(f, hp, px, 8, 8), encoder(f, hp).encode(ClipImage.of(8, 8, px)));
    }

    @Test
    public void minicpmvQueryCountFollowsTheVersionWhenTheKeyIsAbsent() {
        Map<String, Object> m = VisionFamilyFixtures.minicpmvMeta();
        m.remove("clip.minicpmv_query_num");
        assertEquals(64, hparams(m).minicpmvQueryNum());
        assertEquals(4, hparams(m).minicpmvVersion());
        for (int v : new int[]{3, 5, 6, 100045}) {
            m.put("clip.minicpmv_version", v);
            assertEquals("version " + v, 64, hparams(m).minicpmvQueryNum());
        }
        m.put("clip.minicpmv_version", 2);
        assertEquals(96, hparams(m).minicpmvQueryNum());
        m.remove("clip.minicpmv_version");
        // an absent version is MiniCPM-V 2.5 (version 2): 96 queries
        assertEquals(2, hparams(m).minicpmvVersion());
        assertEquals(96, hparams(m).minicpmvQueryNum());
        assertEquals(0, hparams(m).projectionDim());
    }

    @Test
    public void minicpmvRefusesClassTokensAndQueryMismatchesByName() {
        MmprojFixture f = VisionFamilyFixtures.minicpmv(54);
        f.weight("v.class_embd", E, 1);
        ClipHparams hp = hparams(VisionFamilyFixtures.minicpmvMeta());
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ClipVisionModel.from(hp, f.catalog()))
                .getMessage().contains("v.class_embd"));

        MmprojFixture g = VisionFamilyFixtures.minicpmv(55);
        Map<String, Object> m = VisionFamilyFixtures.minicpmvMeta();
        m.put("clip.minicpmv_query_num", QUERIES + 1);
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ClipVisionModel.from(hparams(m),
                g.catalog())).getMessage().contains("resampler.query"));

        MmprojFixture h = VisionFamilyFixtures.minicpmv(56);
        h.remove("resampler.pos_embed_k");
        assertThrows(MissingTensorException.class, () -> ClipVisionModel.from(hp, h.catalog()));
        // the projection_dim 0 of the converter is only accepted where the width comes from a tensor
        Map<String, Object> mlp = VisionFamilyFixtures.meta("mlp", 8, 4, 0);
        assertThrows(com.eonmux.neuralfree.gguf.hparams.GgufFormatException.class, () -> hparams(mlp));
    }

    // ================================================================== internvl

    @Test
    public void internvlMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixtures.internvl(61, null);
        ClipHparams hp = hparams(VisionFamilyFixtures.internvlMeta());
        assertEquals(2, hp.nMerge());
        float[] px = MmprojFixture.pixels(8, 8, 600);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 8, px));
        assertRows(VisionFamilyReference.internvl(f, hp, px, 8, 8), got);
        assertEquals(4, got.length);
        assertEquals(6, got[0].length);
    }

    @Test
    public void internvlQkNormsPerRowAndPerHeadMatchTheGraph() {
        for (String mode : new String[]{"per_row", "per_head"}) {
            MmprojFixture f = VisionFamilyFixtures.internvl(62, mode);
            ClipHparams hp = hparams(VisionFamilyFixtures.internvlMeta());
            float[] px = MmprojFixture.pixels(8, 8, 601);
            assertRows(VisionFamilyReference.internvl(f, hp, px, 8, 8),
                    encoder(f, hp).encode(ClipImage.of(8, 8, px)));
        }
    }

    @Test
    public void internvlNormTypeAndTileSizeFollowTheReference() {
        Map<String, Object> m = VisionFamilyFixtures.internvlMeta();
        assertEquals(NormKind.LAYER, InternVlGraph.normKind(hparams(m)));
        m.put("clip.vision.embedding_length", 3200);
        m.put("clip.vision.attention.head_count", 25);
        m.put("clip.vision.block_count", 45);
        assertEquals(NormKind.RMS, InternVlGraph.normKind(hparams(m)));

        MmprojFixture f = VisionFamilyFixtures.internvl(63, null);
        CpuVisionEncoder enc = encoder(f, hparams(VisionFamilyFixtures.internvlMeta()));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> encode(enc, 12, 12, 602)).getMessage()
                .contains("patch grid"));
        MmprojFixture g = VisionFamilyFixtures.internvl(64, null);
        g.remove("v.class_embd");
        assertThrows(MissingTensorException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixtures.internvlMeta()), g.catalog()));
    }

    // ================================================================== pixtral

    @Test
    public void pixtralWithPatchMergerMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixtures.pixtral(71, true);
        ClipHparams hp = hparams(VisionFamilyFixtures.pixtralMeta(true));
        // 8x12 image: 4x6 patches, merged 2x3 tokens, one [IMG_BREAK] after each of the first two rows
        float[] px = MmprojFixture.pixels(8, 12, 700);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 12, px));
        assertRows(VisionFamilyReference.pixtral(f, hp, px, 8, 12), got);
        assertEquals(2 * 3 + 2, got.length);
        assertArrayEquals(f.get("v.token_embd.img_break"), got[2], 0f);
        assertArrayEquals(f.get("v.token_embd.img_break"), got[5], 0f);
    }

    @Test
    public void pixtralWithoutMergerMatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixtures.pixtral(72, false);
        ClipHparams hp = hparams(VisionFamilyFixtures.pixtralMeta(false));
        float[] px = MmprojFixture.pixels(10, 6, 701);
        float[][] got = encoder(f, hp).encode(ClipImage.of(10, 6, px));
        assertRows(VisionFamilyReference.pixtral(f, hp, px, 10, 6), got);
        assertEquals(5 * 3 + 2, got.length);
    }

    @Test
    public void pixtralRefusesMisalignedGridsAndAMergeWithoutMerger() {
        MmprojFixture f = VisionFamilyFixtures.pixtral(73, true);
        CpuVisionEncoder enc = encoder(f, hparams(VisionFamilyFixtures.pixtralMeta(true)));
        // 6x8: 3x4 patches, a side of 3 cannot merge by 2
        assertTrue(assertThrows(IllegalArgumentException.class, () -> encode(enc, 6, 8, 702)).getMessage()
                .contains("spatial_merge_size"));
        MmprojFixture g = VisionFamilyFixtures.pixtral(74, false);
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixtures.pixtralMeta(true)), g.catalog()))
                .getMessage().contains("mm.patch_merger.weight"));
        MmprojFixture h = VisionFamilyFixtures.pixtral(75, true);
        h.remove("mm.input_norm.weight");
        assertThrows(MissingTensorException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixtures.pixtralMeta(true)), h.catalog()));
    }

    // ================================================================== idefics3

    @Test
    public void idefics3MatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixtures.idefics3(81);
        ClipHparams hp = hparams(VisionFamilyFixtures.idefics3Meta());
        float[] px = MmprojFixture.pixels(16, 16, 800);
        float[][] got = encoder(f, hp).encode(ClipImage.of(16, 16, px));
        assertRows(VisionFamilyReference.idefics3(f, hp, px, 16, 16), got);
        assertEquals(4, got.length);
        assertEquals(6, got[0].length);
    }

    @Test
    public void idefics3NeedsAScaleFactor() {
        MmprojFixture f = VisionFamilyFixtures.idefics3(82);
        Map<String, Object> m = VisionFamilyFixtures.idefics3Meta();
        m.remove("clip.vision.projector.scale_factor");
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ClipVisionModel.from(hparams(m),
                f.catalog())).getMessage().contains("scale_factor"));
    }

    // ================================================================== llama4

    @Test
    public void llama4MatchesTheGraph() {
        MmprojFixture f = VisionFamilyFixtures.llama4(91);
        ClipHparams hp = hparams(VisionFamilyFixtures.llama4Meta());
        float[] px = MmprojFixture.pixels(8, 8, 900);
        float[][] got = encoder(f, hp).encode(ClipImage.of(8, 8, px));
        assertRows(VisionFamilyReference.llama4(f, hp, px, 8, 8), got);
        assertEquals(4, got.length);
        assertEquals(6, got[0].length);
    }

    @Test
    public void llama4RefusesTheConvPatchShape() {
        MmprojFixture f = VisionFamilyFixtures.llama4(92);
        f.shapes.put("v.patch_embd.weight", new long[]{2, 2, 3, E});
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixtures.llama4Meta()), f.catalog()))
                .getMessage().contains("v.patch_embd.weight"));
    }

    // ================================================================== phi4

    @Test
    public void phi4ResizesItsPositionGridToTheImage() {
        MmprojFixture f = VisionFamilyFixtures.phi4(101);
        ClipHparams hp = hparams(VisionFamilyFixtures.phi4Meta());
        assertEquals(1, hp.nMerge());
        // 12x24: 3x6 patches, the 4x4 position grid shrinks on x and grows on y
        float[] px = MmprojFixture.pixels(12, 24, 1000);
        float[][] got = encoder(f, hp).encode(ClipImage.of(12, 24, px));
        assertRows(VisionFamilyReference.phi4(f, hp, px, 12, 24), got);
        assertEquals(18, got.length);
        // a quarter-size grid: the antialias support spans two source rows per output row
        float[] small = MmprojFixture.pixels(8, 8, 1001);
        assertRows(VisionFamilyReference.phi4(f, hp, small, 8, 8),
                encoder(f, hp).encode(ClipImage.of(8, 8, small)));
    }

    @Test
    public void phi4AtTheTrainedGridUsesThePositionRowsAsStored() {
        MmprojFixture f = VisionFamilyFixtures.phi4(102);
        ClipHparams hp = hparams(VisionFamilyFixtures.phi4Meta());
        float[] px = MmprojFixture.pixels(16, 16, 1002);
        assertRows(VisionFamilyReference.phi4(f, hp, px, 16, 16), encoder(f, hp).encode(ClipImage.of(16, 16, px)));
        float[] pos = f.get("v.position_embd.weight");
        float[] same = SiglipResizedGraph.resize(pos, E, 4, 4, 4);
        assertArrayEquals(pos, same, 1e-6f);
    }

    @Test
    public void phi4RefusesANonSquarePositionTable() {
        MmprojFixture f = VisionFamilyFixtures.phi4(103);
        f.put("v.position_embd.weight", -0.5f, 0.5f, E, 15);
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ClipVisionModel.from(hparams(VisionFamilyFixtures.phi4Meta()), f.catalog()))
                .getMessage().contains("square"));
    }

    // ================================================================== merge keys

    @Test
    public void eachFamilyReadsTheMergeUnderItsConverterKey() {
        Map<String, Object> gemma = VisionFamilyFixtures.meta("gemma3", 64, 4, 6);
        assertEquals("absent scale factor", 4, hparams(gemma).nMerge());
        gemma.put("clip.vision.spatial_merge_size", 2);
        assertEquals("spatial_merge_size is not the gemma3 key", 4, hparams(gemma).nMerge());
        gemma.put("clip.vision.projector.scale_factor", 2);
        assertEquals(2, hparams(gemma).nMerge());

        Map<String, Object> pixtral = VisionFamilyFixtures.pixtralMeta(false);
        pixtral.put("clip.vision.projector.scale_factor", 2);
        assertEquals(1, hparams(pixtral).nMerge());

        Map<String, Object> phi4 = VisionFamilyFixtures.phi4Meta();
        phi4.put("clip.vision.spatial_merge_size", 2);
        phi4.put("clip.vision.projector.scale_factor", 2);
        assertEquals("phi4 never merges", 1, hparams(phi4).nMerge());
    }

    // ================================================================== shared pieces

    @Test
    public void pixelShufflesMatchTheirTorchLayouts() {
        int e = 3;
        int pw = 4;
        int ph = 4;
        float[] rows = new float[pw * ph * e];
        double[][] x = new double[pw * ph][e];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
            x[i / e][i % e] = i;
        }
        float[][] rowMajor = MmprojFixture.Reference.toFloat(VisionFamilyReference.spaceToDepth(x, pw, ph, 2, false));
        float[][] colMajor = MmprojFixture.Reference.toFloat(VisionFamilyReference.spaceToDepth(x, pw, ph, 2, true));
        assertRows(rowMajor, VisionRows.tokenRows(PixelShuffle.mergePermute(rows, e, pw, ph, 2), 4 * e));
        assertRows(rowMajor, VisionRows.tokenRows(PixelShuffle.internvl(rows, e, pw, ph, 2), 4 * e));
        assertRows(colMajor, VisionRows.tokenRows(PixelShuffle.llama4(rows, e, pw, ph, 2), 4 * e));
        // a 3x3 grid pads to 4x4 with zero rows before the idefics3 shuffle
        float[] odd = new float[9 * e];
        java.util.Arrays.fill(odd, 1f);
        float[][] padded = VisionRows.tokenRows(PixelShuffle.mergePermute(odd, e, 3, 3, 2), 4 * e);
        assertEquals(4, padded.length);
        assertEquals(0f, padded[3][4 * e - 1], 0f);
        assertEquals(1f, padded[0][0], 0f);
    }

    @Test
    public void ropeHalvesRotateAdjacentPairs() {
        int dh = 8;
        float[] x = new float[dh];
        double[][] ref = new double[1][dh];
        for (int i = 0; i < dh; i++) {
            x[i] = i + 1;
            ref[0][i] = i + 1;
        }
        VisionRope.twoHalves(x, new int[]{3}, new int[]{5}, 1, 1, dh, 10000.0, true);
        VisionFamilyReference.rope2d(ref, new int[]{3}, new int[]{5}, dh, 10000.0, true);
        assertArrayEquals(MmprojFixture.Reference.toFloat(ref)[0], x, 1e-5f);
        // pair (0, 1) at frequency 1: angle 3
        assertEquals(1 * Math.cos(3) - 2 * Math.sin(3), x[0], 1e-5);
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.image.Idefics3Preprocessor;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.image.PreparedImages;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.encoder;
import static com.eonmux.neuralfree.gguf.mm.VisionFamilyFixtures.hparams;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The image token count of the preprocessing ({@link ImageProcessorParams#tokensFor}, llama.cpp
 * {@code clip_n_output_tokens}) against the rows the vision encoder of the same file emits, per projector family,
 * and the whole path from a decoded image through the preprocessor to the encoder for Pixtral and Idefics3.
 */
public class VisionTokenCountTest {

    /** Encodes one {@code width x height} tile and checks the count against the encoder rows and the reference. */
    private static void assertCount(String family, MmprojFixture f, Map<String, Object> meta, int width, int height,
                                    int reference) {
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(meta), ImageTokenBudget.none());
        int rows;
        try (CpuVisionEncoder enc = encoder(f, hparams(meta))) {
            rows = enc.encode(ClipImage.of(width, height, MmprojFixture.pixels(width, height, 9100))).length;
        }
        assertEquals(family + " encoder rows", reference, rows);
        assertEquals(family + " counted tokens", rows, params.tokensFor(new ImageSize(width, height)));
    }

    /** The meta with the pixel limits the dynamic-resolution converters always write. */
    private static Map<String, Object> withPixelLimits(Map<String, Object> meta) {
        meta.put("clip.vision.image_min_pixels", 16);
        meta.put("clip.vision.image_max_pixels", 4096);
        return meta;
    }

    @Test
    public void pixtralCountsTheMergedGridAndTheBreakRows() {
        // 12x8, patch 2: 6x4 patches, 6 * 4 tokens + 3 [IMG_BREAK] rows
        assertCount("pixtral", VisionFamilyFixtures.pixtral(9101, false), VisionFamilyFixtures.pixtralMeta(false),
                12, 8, 6 * 4 + 3);
        // with the 2x2 patch merger: 3x2 tokens + 1 break row
        assertCount("pixtral merger", VisionFamilyFixtures.pixtral(9102, true), VisionFamilyFixtures.pixtralMeta(true),
                12, 8, 3 * 2 + 1);
        // a tall tile: 2x6 merged tokens + 5 break rows
        assertCount("pixtral merger tall", VisionFamilyFixtures.pixtral(9103, true),
                VisionFamilyFixtures.pixtralMeta(true), 8, 24, 2 * 6 + 5);
    }

    @Test
    public void scaleFactorFamiliesCountPatchesOverTheMergeSquare() {
        assertCount("idefics3", VisionFamilyFixtures.idefics3(9111), VisionFamilyFixtures.idefics3Meta(), 16, 16, 4);
        assertCount("internvl", VisionFamilyFixtures.internvl(9112, null), VisionFamilyFixtures.internvlMeta(),
                8, 8, 4);
        assertCount("llama4", VisionFamilyFixtures.llama4(9113), VisionFamilyFixtures.llama4Meta(), 8, 8, 4);
        assertCount("cohere2v", VisionFamiliesCFixtures.cohere2v(9114), VisionFamiliesCFixtures.cohere2vMeta(),
                8, 8, 4);
    }

    @Test
    public void unmergedAndResampledFamiliesCountTheirRows() {
        assertCount("phi4", VisionFamilyFixtures.phi4(9121), VisionFamilyFixtures.phi4Meta(), 12, 24, 3 * 6);
        assertCount("resampler", VisionFamilyFixtures.minicpmv(9122), VisionFamilyFixtures.minicpmvMeta(), 8, 16,
                VisionFamilyFixtures.QUERIES);
    }

    @Test
    public void kimiCountsTheAlignedMergedGrid() {
        // 6x4, patch 2: 3x2 patches; the merge pads to 4x2, ceil(3/2) * ceil(2/2) = 2 tokens (plain patches: 6)
        assertCount("kimivl", VisionFamiliesCFixtures.kimivl(9131), VisionFamiliesCFixtures.kimivlMeta(), 6, 4, 2);
        assertCount("kimivl even", VisionFamiliesCFixtures.kimivl(9132), VisionFamiliesCFixtures.kimivlMeta(),
                8, 12, 2 * 3);
    }

    @Test
    public void paddleAndDotsCountTheAlignedPatchesOverTheMergeSquare() {
        assertCount("paddleocr", VisionFamiliesCFixtures.paddleocr(9141),
                withPixelLimits(VisionFamiliesCFixtures.paddleocrMeta()), 8, 12, 2 * 3);
        assertCount("dots_ocr", VisionFamiliesCFixtures.dotsOcr(9142),
                withPixelLimits(VisionFamiliesCFixtures.dotsOcrMeta()), 8, 12, 2 * 3);
    }

    @Test
    public void hunyuanCountsTheNewlineAndTheBeginEndRows() {
        // 10x8, patch 2: 5x4 patches -> 2x2 merged (the 5th column dropped), a newline per row, begin and end
        assertCount("hunyuanvl", VisionFamiliesCFixtures.hunyuanvl(9151), VisionFamiliesCFixtures.hunyuanvlMeta(),
                10, 8, 3 * 2 + 2);
        assertCount("hunyuanvl even", VisionFamiliesCFixtures.hunyuanvl(9152),
                VisionFamiliesCFixtures.hunyuanvlMeta(), 8, 12, 3 * 3 + 2);
    }

    @Test
    public void qwen3CountsTheMergedConvGrid() {
        assertCount("qwen3vl", VisionFamiliesCFixtures.qwen3vl(9161), VisionFamiliesCFixtures.qwen3vlMeta(),
                8, 12, 2 * 3);
    }

    /** Every tile of {@code prepared} through the encoder: the rows of the whole image. */
    private static int encodedRows(MmprojFixture f, Map<String, Object> meta, PreparedImages prepared) {
        int rows = 0;
        List<ImageTile> tiles = prepared.encodedTiles();
        List<float[]> pixels = prepared.normalized();
        try (CpuVisionEncoder enc = encoder(f, hparams(meta))) {
            for (int i = 0; i < tiles.size(); i++) {
                ImageTile tile = tiles.get(i);
                int n = enc.encode(ClipImage.of(tile.width(), tile.height(), pixels.get(i))).length;
                assertEquals("tile " + i, prepared.tokensFor(i), n);
                rows += n;
            }
        }
        return rows;
    }

    @Test
    public void pixtralPreprocessingFeedsTheEncoderTheCountedRows() {
        Map<String, Object> meta = VisionFamilyFixtures.pixtralMeta(true);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(meta), ImageTokenBudget.none());
        PreparedImages prepared = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(13, 9, 200, 100, 50));
        assertEquals(prepared.totalTokens(), encodedRows(VisionFamilyFixtures.pixtral(9171, true), meta, prepared));

        meta.put("clip.vision.preproc_image_size", 16);
        params = ImageProcessorParams.fromGguf(GgufKv.ofMap(meta), ImageTokenBudget.none());
        prepared = ImagePreprocessors.forParams(params).prepare(RasterImage.filled(40, 22, 10, 20, 30));
        assertEquals(prepared.totalTokens(), encodedRows(VisionFamilyFixtures.pixtral(9172, true), meta, prepared));
    }

    /** The prepared image's rows through the encoder of {@code f}; returns the prepared image. */
    private static PreparedImages assertPreparedRows(String family, MmprojFixture f, Map<String, Object> meta,
                                                     RasterImage image) {
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(meta), ImageTokenBudget.none());
        PreparedImages prepared = ImagePreprocessors.forParams(params).prepare(image);
        assertEquals(family, prepared.totalTokens(), encodedRows(f, meta, prepared));
        return prepared;
    }

    @Test
    public void lfm2TilesAndThumbnailFeedTheEncoderTheCountedRows() {
        // 8-pixel tiles of 2-pixel patches: a 60x36 image (pixel limit 256 * 16) tiles, the thumbnail last
        Map<String, Object> meta = VisionFamiliesCFixtures.lfm2Meta("lfm2");
        meta.put("clip.vision.preproc_tile_size", 8);
        PreparedImages tiled = assertPreparedRows("lfm2", VisionFamiliesCFixtures.lfm2(9191, true), meta,
                RasterImage.filled(120, 72, 10, 120, 240));
        assertTrue(tiled.hasOverview());
        assertTrue(tiled.tiles().size() >= 2);
        PreparedImages single = assertPreparedRows("lfm2 single", VisionFamiliesCFixtures.lfm2(9192, true), meta,
                RasterImage.filled(30, 18, 10, 120, 240));
        assertFalse(single.hasOverview());
        assertEquals(1, single.tiles().size());

        Map<String, Object> d1 = VisionFamilyFixturesB.lfm2Meta("d1omni_v");
        d1.put("clip.vision.preproc_tile_size", 8);
        assertPreparedRows("d1omni_v", VisionFamilyFixturesB.lfm2(9193, true), d1,
                RasterImage.filled(120, 72, 10, 120, 240));
    }

    @Test
    public void theNewlyPreprocessedFamiliesFeedTheEncoderTheCountedRows() {
        RasterImage image = RasterImage.filled(30, 20, 200, 90, 40);
        assertPreparedRows("janus_pro", VisionFamiliesCFixtures.janus(9194), VisionFamiliesCFixtures.janusMeta(),
                image);
        assertPreparedRows("glm_edge", VisionFamilyFixturesB.glmEdge(9195), VisionFamilyFixturesB.llavaMeta("adapter"),
                image);
        assertPreparedRows("cogvlm", VisionFamilyFixturesB.cogvlm(9196), VisionFamilyFixturesB.cogvlmMeta(), image);
        assertPreparedRows("youtuvl", VisionFamilyFixturesB.youtuvl(9197), VisionFamilyFixturesB.youtuvlMeta(),
                image);
        assertPreparedRows("lightonocr", VisionFamiliesCFixtures.lightonocr(9198),
                VisionFamiliesCFixtures.lightonocrMeta(), RasterImage.filled(40, 22, 10, 20, 30));
        assertPreparedRows("dots3note_v", VisionFamiliesCFixtures.dots3note(9199),
                withPixelLimits(VisionFamiliesCFixtures.dots3noteMeta()), image);
        PreparedImages llama4 = assertPreparedRows("llama4", VisionFamilyFixtures.llama4(9200),
                VisionFamilyFixtures.llama4Meta(), RasterImage.filled(20, 14, 60, 70, 80));
        assertTrue(llama4.tiles().size() > 1);
        assertFalse(llama4.overviewFirst());
    }

    @Test
    public void idefics3PreprocessingFeedsTheEncoderTheCountedRows() {
        // image size 16, longest edge 32: a 40x24 image resizes to 32x20, tiles 2x2, plus the overview
        Map<String, Object> meta = VisionFamilyFixtures.idefics3Meta();
        meta.put("clip.vision.preproc_image_size", 32);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(meta), ImageTokenBudget.none());
        PreparedImages prepared = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(40, 24, 90, 60, 30));
        assertTrue(ImagePreprocessors.forParams(params) instanceof Idefics3Preprocessor);
        assertEquals(5, prepared.encodedTiles().size());
        assertEquals(5 * 4, prepared.totalTokens());
        assertEquals(prepared.totalTokens(), encodedRows(VisionFamilyFixtures.idefics3(9181), meta, prepared));
    }
}

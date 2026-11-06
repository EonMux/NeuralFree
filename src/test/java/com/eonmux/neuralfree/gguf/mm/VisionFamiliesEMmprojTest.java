package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The step3vl, sensenova_u1, deepseek4v, deepseek41v, deepseekocr and deepseekocr2 encoders and the granite4_vision
 * anyres assembly through a real mmproj
 * GGUF file, against the rows llama.cpp's own clip computes for the same file and pixels
 * ({@link ClipReferenceRows}); the kimivl case checks the reference pipeline on a family whose graph is verified
 * independently.
 */
public class VisionFamiliesEMmprojTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path write(GgufTestWriter w) throws IOException {
        Path file = tmp.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    /** Encodes {@code image} on the CPU encoder of the fixture's file and checks it against the reference rows. */
    private float[][] encodeCase(String name, MmprojFixture f, Map<String, Object> meta, ClipImage image,
                                 String flags) throws IOException {
        Path file = write(VisionFamilyFixtures.gguf(f, meta));
        boolean exported = ClipReferenceRows.export(name, file, image, flags);
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            float[][] got = enc.encode(image);
            if (!exported) {
                ClipReferenceRows.assertMatches(name, got);
            }
            return got;
        }
    }

    private static ClipImage image(int width, int height, long seed) {
        return ClipImage.of(width, height, MmprojFixture.pixels(width, height, seed));
    }

    @Test
    public void kimivlMatchesTheReferencePipeline() throws IOException {
        encodeCase("kimivl_square", VisionFamiliesCFixtures.kimivl(93), VisionFamiliesCFixtures.kimivlMeta(),
                image(8, 8, 80), "");
        float[][] rows = encodeCase("kimivl", VisionFamiliesCFixtures.kimivl(93), VisionFamiliesCFixtures.kimivlMeta(),
                image(10, 8, 79), "");
        assertEquals(6, rows.length);
    }

    @Test
    public void sensenovaMatchesTheReference() throws IOException {
        encodeCase("sensenova_small", VisionFamiliesEFixtures.sensenova(201), VisionFamiliesEFixtures.sensenovaMeta(),
                image(8, 4, 81), "");
        float[][] rows = encodeCase("sensenova", VisionFamiliesEFixtures.sensenova(201),
                VisionFamiliesEFixtures.sensenovaMeta(), image(12, 8, 82), "");
        assertEquals(3 * 2, rows.length);
        assertEquals(VisionFamiliesEFixtures.OUT, rows[0].length);
    }

    @Test
    public void step3vlMatchesTheReference() throws IOException {
        float[][] rows = encodeCase("step3vl", VisionFamiliesEFixtures.step3vl(211, true),
                VisionFamiliesEFixtures.step3vlMeta(), image(16, 16, 83), "");
        assertEquals(4, rows.length);
        encodeCase("step3vl_native", VisionFamiliesEFixtures.step3vl(212, true), VisionFamiliesEFixtures.step3vlMeta(),
                image(8, 8, 84), "");
        rows = encodeCase("step3vl_nobias", VisionFamiliesEFixtures.step3vl(213, false),
                VisionFamiliesEFixtures.step3vlMeta(), image(8, 16, 85), "");
        assertEquals(2, rows.length);
    }

    /** {@code image} carrying {@code lead} lead pads. */
    private static ClipImage lead(ClipImage image, int lead) {
        return image.withLeadPad(lead);
    }

    @Test
    public void deepseek4vMatchesTheReferenceForEveryLeadPad() throws IOException {
        // 5x3 patches pad to the 3x2 aligner grid: 2 rows of 3 tokens and a newline, no pad row
        float[][] rows = encodeCase("deepseek4v", VisionFamiliesEFixtures.deepseekV4(221, true),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"), image(10, 6, 86), "");
        assertEquals(DeepseekV4Layout.tokens4v(3, 2, 0), rows.length);
        rows = encodeCase("deepseek4v_lead3", VisionFamiliesEFixtures.deepseekV4(221, true),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"), lead(image(10, 6, 86), 3), "lead=3");
        assertEquals(DeepseekV4Layout.tokens4v(3, 2, 3), rows.length);
        // 4x6 patches: a 2x3 grid, an odd row count padded by a row of pads
        encodeCase("deepseek4v_oddrows", VisionFamiliesEFixtures.deepseekV4(222, true),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek4v"), lead(image(8, 12, 87), 1), "lead=1");
    }

    @Test
    public void deepseek41vMatchesTheReference() throws IOException {
        float[][] rows = encodeCase("deepseek41v", VisionFamiliesEFixtures.deepseekV4(223, false),
                VisionFamiliesEFixtures.deepseekV4Meta("deepseek41v"), image(10, 6, 88), "");
        assertEquals(DeepseekV4Layout.tokens41v(3, 2), rows.length);
    }

    /** {@code image} as the global view whose rows end with the view separator. */
    private static ClipImage global(ClipImage image) {
        return image.withViewSeparator(true);
    }

    @Test
    public void deepseekOcrMatchesTheReference() throws IOException {
        // 128 px: an 8x8 SAM grid (padded windows, resized positions and tables), 2x2 tower rows, no CLIP resize
        float[][] rows = encodeCase("deepseekocr_global", VisionFamiliesEFixtures.deepseekOcr(231),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"), global(image(128, 128, 89)), "viewsep");
        assertEquals(2 * (2 + 1) + 1, rows.length);
        // 192 px: 3x3 tower rows, the CLIP position table resized
        rows = encodeCase("deepseekocr_global3", VisionFamiliesEFixtures.deepseekOcr(231),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"), global(image(192, 192, 90)), "viewsep");
        assertEquals(3 * (3 + 1) + 1, rows.length);
        // a row of two 64 px tiles: the SAM position grid at its own size, one tower row per tile
        rows = encodeCase("deepseekocr_row", VisionFamiliesEFixtures.deepseekOcr(232),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"), image(64, 128, 91), "");
        assertEquals(2 + 1, rows.length);
        // a row of two 128 px tiles: two token rows interleaving the tiles
        rows = encodeCase("deepseekocr_row2", VisionFamiliesEFixtures.deepseekOcr(232),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr"), image(128, 256, 92), "");
        assertEquals((2 * 2 + 1) * 2, rows.length);
    }

    @Test
    public void deepseekOcr2MatchesTheReference() throws IOException {
        float[][] rows = encodeCase("deepseekocr2_tile", VisionFamiliesEFixtures.deepseekOcr2(241),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2"), image(768, 768, 93), "");
        assertEquals(DeepseekOcrGraph.QUERIES_768, rows.length);
        rows = encodeCase("deepseekocr2_global", VisionFamiliesEFixtures.deepseekOcr2(241),
                VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2"), global(image(1024, 1024, 94)), "viewsep");
        assertEquals(DeepseekOcrGraph.QUERIES_1024 + 1, rows.length);
    }

    /** clip.cpp load_hparams requires the KV head count of the grouped-query DeepSeek-OCR 2 encoder. */
    @Test
    public void deepseekOcr2WithoutKvHeadCountIsRefusedByName() throws IOException {
        Map<String, Object> meta = new HashMap<>(VisionFamiliesEFixtures.deepseekOcrMeta("deepseekocr2"));
        meta.remove("clip.vision.attention.head_count_kv");
        Path file = write(VisionFamilyFixtures.gguf(VisionFamiliesEFixtures.deepseekOcr2(241), meta));
        Exception e = assertThrows(Exception.class, () -> {
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertNotNull(enc);
            }
        });
        assertTrue(e.getMessage(), e.getMessage().contains("clip.vision.attention.head_count_kv"));
    }

    /**
     * One anyres image of the granite4_vision fixture: the CPU encoder's {@code encodeAnyres} of the overview and
     * the {@code gx x gy} tiles against the rows llama.cpp's clip computes for the same tiles stacked top to bottom
     * (the overview first, the tiles in raster order) with the entry's anyres grid and original size.
     */
    private float[][] graniteAnyresCase(String name, int gx, int gy, int origW, int origH, long seed)
            throws IOException {
        int side = VisionFamilyFixturesB.GRANITE_IMAGE;
        int values = side * side * 3;
        List<ClipImage> tiles = new ArrayList<>();
        float[] stacked = new float[values * (1 + gx * gy)];
        for (int t = 0; t <= gx * gy; t++) {
            ClipImage tile = image(side, side, seed + t);
            System.arraycopy(tile.pixels(), 0, stacked, t * values, values);
            tiles.add(tile);
        }
        Path file = write(VisionFamilyFixtures.gguf(VisionFamilyFixturesB.granite4(251),
                VisionFamilyFixturesB.granite4Meta()));
        String flags = "anyres=" + gx + "," + gy + "," + origW + "," + origH;
        boolean exported = ClipReferenceRows.export(name, file,
                ClipImage.of(side, side * (1 + gx * gy), stacked), flags);
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
            float[][] got = enc.encodeAnyres(tiles.get(0), tiles.subList(1, tiles.size()), gx, gy, origW, origH);
            if (!exported) {
                ClipReferenceRows.assertMatches(name, got);
            }
            return got;
        }
    }

    @Test
    public void granite4AnyresMatchesTheReference() throws IOException {
        // 4x4 tokens per tile; a wide 2x1 grid crops rows, a tall 1x2 grid crops columns, a square 2x2 crops none
        float[][] rows = graniteAnyresCase("granite4_anyres_wide", 2, 1, 30, 10, 95);
        assertEquals(16 + 2 * 9, rows.length);
        assertEquals(rows.length, AnyresGrid.tokens(4, 2, 1, 30, 10));
        rows = graniteAnyresCase("granite4_anyres_tall", 1, 2, 10, 20, 96);
        assertEquals(16 + 8 * 5, rows.length);
        assertEquals(rows.length, AnyresGrid.tokens(4, 1, 2, 10, 20));
        rows = graniteAnyresCase("granite4_anyres_square", 2, 2, 40, 40, 97);
        assertEquals(16 + 8 * 9, rows.length);
    }
}

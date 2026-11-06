package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;

import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Idefics3 / SmolVLM preprocessing against llama.cpp {@code mtmd_image_preprocessor_idefics3}
 * (tools/mtmd/mtmd-image.cpp) and the idefics3 arm of {@code load_hparams} (clip.cpp): the longest-edge resize,
 * the tile grid, the tile order and the overview, and the smart-resize fallback.
 */
public class Idefics3PreprocessorTest {

    /** SmolVLM-like geometry: 512 tiles of 16-pixel patches, scale factor 4 (64 tokens per tile). */
    private static Map<String, Object> metadata(Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.projector_type", "idefics3");
        m.put("clip.vision.patch_size", 16);
        m.put("clip.vision.image_size", 512);
        m.put("clip.vision.projector.scale_factor", 4);
        m.put("clip.vision.image_mean", new float[] {0.5f, 0.5f, 0.5f});
        m.put("clip.vision.image_std", new float[] {0.5f, 0.5f, 0.5f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams params(ImageTokenBudget budget, Object... extra) {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(extra)), budget);
    }

    private static Idefics3Preprocessor preprocessor(ImageTokenBudget budget, Object... extra) {
        return (Idefics3Preprocessor) ImagePreprocessors.forParams(params(budget, extra));
    }

    /** The reference longest-edge resize, transcribed with its C++ integer and double arithmetic. */
    private static int[] referenceResize(int width, int height, int longestEdge) {
        double aspectRatio = (double) width / height;
        int w;
        int h;
        if (width >= height) {
            w = longestEdge;
            h = (int) (longestEdge / aspectRatio);
            h += h % 2;
        } else {
            h = longestEdge;
            w = (int) (longestEdge * aspectRatio);
            w += w % 2;
        }
        return new int[] {w, h};
    }

    @Test
    public void theFamilyDefaultsFollowLoadHparams() {
        ImageProcessorParams p = params(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 1536);
        assertSame(ProjectorType.IDEFICS3.visionPreprocessing(), p.preprocessing());
        assertEquals(ResizeAlgo.LANCZOS, p.resizeAlgo());
        assertEquals(1536, p.longestEdge());
        assertEquals(4, p.mergeSize());
        assertEquals(0, p.minPixels());
        assertEquals(0, p.maxPixels());
        assertFalse(p.overviewFirst());
        // without the key, the old-file fallback applies
        assertEquals(0, params(ImageTokenBudget.none()).longestEdge());
        // a token budget sets the pixel limits over the (16 * 4)^2 merged patch area
        ImageProcessorParams budgeted = params(ImageTokenBudget.builder().minTokens(64).maxTokens(256).build(),
                "clip.vision.preproc_image_size", 1536);
        assertEquals(64L * 64 * 64, budgeted.minPixels());
        assertEquals(256L * 64 * 64, budgeted.maxPixels());
    }

    @Test
    public void theLongestEdgeResizeMatchesTheReference() {
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 1536);
        assertTrue(pre.usesLongestEdge());
        // 1000x600: 1536 x (int)(1536 / 1.6667) = 921, rounded up to even 922; tiles 3 x 2
        assertEquals(new ImageSize(1536, 922), pre.longestEdgeSize(new ImageSize(1000, 600)));
        assertEquals(new ImageSize(3, 2), pre.tileGrid(new ImageSize(1536, 922)));
        // portrait: 600x1000 -> (int)(1536 * 0.6) = 921 -> 922 x 1536
        assertEquals(new ImageSize(922, 1536), pre.longestEdgeSize(new ImageSize(600, 1000)));
        // small images upscale to the longest edge too
        assertEquals(new ImageSize(1536, 1536), pre.longestEdgeSize(new ImageSize(100, 100)));
        for (int[] size : new int[][] {{1000, 600}, {600, 1000}, {1920, 1080}, {333, 777}, {4000, 3}, {1537, 1536},
                {7, 5}, {512, 512}, {2048, 511}}) {
            int[] ref = referenceResize(size[0], size[1], 1536);
            assertEquals(size[0] + "x" + size[1], new ImageSize(ref[0], ref[1]),
                    pre.longestEdgeSize(new ImageSize(size[0], size[1])));
        }
    }

    @Test
    public void tilesComeRowMajorBeforeTheOverview() {
        // longest edge 1024 over 512 tiles: a 2048x1024 image resizes to 1024x512, one row of two tiles
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 1024);
        RasterImage image = halves(2048, 1024);
        PreparedImages out = pre.prepare(image);
        assertEquals(2, out.gridX());
        assertEquals(1, out.gridY());
        assertEquals(2, out.tiles().size());
        assertTrue(out.hasOverview());
        assertFalse(out.overviewFirst());
        assertEquals(new ImageSize(512, 512), out.overview().size());
        List<ImageTile> encoded = out.encodedTiles();
        assertSame(out.overview(), encoded.get(encoded.size() - 1));
        // the left tile is the red half, the right tile the blue half
        assertArrayEquals(new int[] {255, 0, 0}, pixel((RasterImage) out.tiles().get(0), 256, 256));
        assertArrayEquals(new int[] {0, 0, 255}, pixel((RasterImage) out.tiles().get(1), 256, 256));
        // 64 tokens per 512 tile: 2 tiles and the overview
        assertEquals(3 * 64, out.totalTokens());
    }

    @Test
    public void aPartialTileStretchesUpToWholeTiles() {
        // 1000x600 at longest edge 1536: 1536x922 stretches to 1536x1024, six tiles and the overview
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 1536);
        PreparedImages out = pre.prepare(RasterImage.filled(1000, 600, 40, 80, 120));
        assertEquals(3, out.gridX());
        assertEquals(2, out.gridY());
        assertEquals(6, out.tiles().size());
        for (ImageTile tile : out.tiles()) {
            assertEquals(new ImageSize(512, 512), tile.size());
        }
        assertEquals(7 * 64, out.totalTokens());
    }

    @Test
    public void oneTileImagesStillCarryTheOverview() {
        // 300x200 at longest edge 512: 512x342 fits one tile, encoded with the overview after it
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 512);
        PreparedImages out = pre.prepare(RasterImage.filled(300, 200, 1, 2, 3));
        assertEquals(1, out.gridX());
        assertEquals(1, out.gridY());
        assertEquals(2, out.encodedTiles().size());
        assertEquals(2 * 64, out.totalTokens());
    }

    @Test
    public void aTokenBudgetSwitchesToTheSmartResizeSlicing() {
        // max 256 tokens = 1024^2 pixels: 1000x600 under longest edge 1536 is 1536x1024 aligned to 512, over the
        // cap; beta = sqrt(600000 / 1048576), the sides floor to 1024x512 -> tiles 2 x 1
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.builder().maxTokens(256).build(),
                "clip.vision.preproc_image_size", 1536);
        assertFalse(pre.usesLongestEdge());
        PreparedImages out = pre.prepare(RasterImage.filled(1000, 600, 40, 80, 120));
        assertEquals(2, out.gridX());
        assertEquals(1, out.gridY());
        assertEquals(2, out.tiles().size());
        assertEquals(new ImageSize(512, 512), out.overview().size());
        assertFalse(out.overviewFirst());
        assertEquals(3 * 64, out.totalTokens());
    }

    @Test
    public void oldFilesWithoutTheLongestEdgeRoundToWholeTiles() {
        // no preproc_image_size: 1000x600 rounds to 1024x512 (2 x 1 tiles); 1800x1300 to 2048x1536 (4 x 3)
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none());
        assertFalse(pre.usesLongestEdge());
        PreparedImages out = pre.prepare(RasterImage.filled(1000, 600, 0, 0, 0));
        assertEquals(2, out.gridX());
        assertEquals(1, out.gridY());
        out = pre.prepare(RasterImage.filled(1800, 1300, 0, 0, 0));
        assertEquals(4, out.gridX());
        assertEquals(3, out.gridY());
        assertEquals(12, out.tiles().size());
        assertEquals(13 * 64, out.totalTokens());
    }

    @Test
    public void elongatedImagesThatLoseASideAreRefused() {
        Idefics3Preprocessor pre = preprocessor(ImageTokenBudget.none(), "clip.vision.preproc_image_size", 1024);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> pre.longestEdgeSize(new ImageSize(10000, 4)));
        assertTrue(e.getMessage(), e.getMessage().contains("10000x4"));
        IllegalArgumentException none = assertThrows(IllegalArgumentException.class,
                () -> preprocessor(ImageTokenBudget.none()).longestEdgeSize(new ImageSize(100, 100)));
        assertTrue(none.getMessage(), none.getMessage().contains("clip.vision.preproc_image_size"));
    }

    /** A red left half and a blue right half. */
    private static RasterImage halves(int width, int height) {
        byte[] rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = (y * width + x) * 3;
                if (x < width / 2) {
                    rgb[i] = (byte) 255;
                } else {
                    rgb[i + 2] = (byte) 255;
                }
            }
        }
        return RasterImage.wrap(width, height, rgb);
    }

    private static int[] pixel(RasterImage image, int x, int y) {
        byte[] rgb = image.rgbArray();
        int i = (y * image.width() + x) * 3;
        return new int[] {rgb[i] & 0xff, rgb[i + 1] & 0xff, rgb[i + 2] & 0xff};
    }
}

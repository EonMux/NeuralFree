package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

/**
 * The Cohere2 vision tiling against llama.cpp's {@code mtmd_image_preprocessor_cohere2v}: the grid is the
 * whole-tile factorization of the tile count with the least downscale (least upscale when every grid shrinks),
 * a single tile skips the slices, and the tiles encode before the overview.
 */
public class Cohere2vPreprocessorTest {

    private static ImageProcessorParams cohere2v() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "cohere2v");
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.preproc_max_tiles", 4);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
    }

    @Test
    public void wideImagePicksTheLeastDownscaledGrid() {
        // 1000x300 fits 3x1 tiles at scale 1008/1000 = 1.008 (downscale), the least of the fitting grids
        PreparedImages out = ImagePreprocessors.forParams(cohere2v())
                .prepare(RasterImage.filled(1000, 300, 0, 0, 0));
        assertEquals(3, out.tiles().size());
        assertEquals(3, out.gridX());
        assertEquals(1, out.gridY());
        // tiles first, overview last
        assertEquals(4, out.encodedTiles().size());
        assertNotNull(out.overview());
        assertEquals(out.overview(), out.encodedTiles().get(3));
        // cohere2v tokens divide by merge^2: 576 patches / 4 per tile
        assertEquals(4 * 144, out.totalTokens());
    }

    @Test
    public void smallImageBecomesOneTileWithoutSlices() {
        // every grid of >1 tile must downscale, so the 1x1 grid (least upscale, scale 1) wins
        PreparedImages out = ImagePreprocessors.forParams(cohere2v())
                .prepare(RasterImage.filled(300, 300, 0, 0, 0));
        assertEquals(1, out.encodedTiles().size());
        assertEquals(0, out.tiles().size());
        assertEquals(0, out.gridX());
        assertEquals(144, out.totalTokens());
    }

    @Test
    public void tallImageTakesATallGrid() {
        // 300x1000: fitting grids downscale; 1x3 tiles (1008 height) gives scale 1.008, the least downscale
        PreparedImages out = ImagePreprocessors.forParams(cohere2v())
                .prepare(RasterImage.filled(300, 1000, 0, 0, 0));
        assertEquals(3, out.tiles().size());
        assertEquals(1, out.gridX());
        assertEquals(3, out.gridY());
    }

    @Test
    public void missingTileCapIsRefusedByName() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "cohere2v");
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            fail("expected a missing tile cap refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("preproc_max_tiles"));
        }
    }

    @Test
    public void outOfRangeTileCapIsRefused() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "cohere2v");
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.preproc_max_tiles", 300);
        m.put("clip.vision.projector.scale_factor", 2);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            fail("expected a tile cap range refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("[1, 256]"));
        }
    }
}

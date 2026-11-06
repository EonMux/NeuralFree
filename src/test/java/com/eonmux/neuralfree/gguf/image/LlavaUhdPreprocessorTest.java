package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The llava-uhd slicing against llama.cpp's {@code mtmd_image_preprocessor_llava_uhd}: the pinpoints planner
 * (LLaVA-1.6) picks the candidate with the least wasted area, the dynamic grid planner (MiniCPM-V) follows the
 * aspect ratio, and the slices sit row-major over the refined image.
 */
public class LlavaUhdPreprocessorTest {

    private static Map<String, Object> metadata(String projector, Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", projector);
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams llava16() {
        // the classic LLaVA-1.6 pinpoints over a 336 grid
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata("mlp",
                "clip.vision.image_grid_pinpoints",
                new int[] {336, 336, 672, 336, 336, 672, 672, 672, 1008, 336, 336, 1008, 1008, 672, 672, 1008,
                        1008, 1008})), ImageTokenBudget.none());
    }

    @Test
    public void smallImagesProduceOnlyTheOverview() {
        PreparedImages out = ImagePreprocessors.forParams(llava16())
                .prepare(RasterImage.filled(300, 300, 10, 20, 30));
        assertNotNull(out.overview());
        assertEquals(336, out.overview().width());
        assertTrue(out.tiles().isEmpty());
        assertEquals(0, out.gridX());
        assertEquals(1 * 576, out.totalTokens());
    }

    @Test
    public void pinpointsPickTheLeastWastedCandidate() {
        // 800x600: the effective resolution caps at the original 480000 pixels, so every big candidate ties
        // and the tiebreak keeps (1008, 672), whose wasted area 197376 is the smallest of the tied ones
        PreparedImages out = ImagePreprocessors.forParams(llava16())
                .prepare(RasterImage.filled(800, 600, 0, 0, 255));
        // overview 336x336 + a 3x2 grid of 336x336 slices from the 1008x672 refined image
        assertEquals(6, out.tiles().size());
        assertEquals(3, out.gridX());
        assertEquals(2, out.gridY());
        assertEquals(336, out.overview().width());
        for (ImageTile tile : out.tiles()) {
            assertEquals(336, tile.width());
            assertEquals(336, tile.height());
        }
        assertEquals(7 * 576, out.totalTokens());
    }

    @Test
    public void slicesCarryTheContentOfTheirRectangle() {
        // left half red, right half blue: slice 0 (left column) is red, slice 2 (right column) is blue
        RasterImage wide = new RasterImage(672, 336, new byte[672 * 336 * 3]);
        byte[] rgb = wide.rgb();
        for (int y = 0; y < 336; y++) {
            for (int x = 0; x < 672; x++) {
                int i = (y * 672 + x) * 3;
                rgb[i] = (byte) (x < 336 ? 255 : 0);
                rgb[i + 2] = (byte) (x < 336 ? 0 : 255);
            }
        }
        RasterImage src = RasterImage.wrap(672, 336, rgb);
        PreparedImages out = ImagePreprocessors.forParams(llava16()).prepare(src);
        assertEquals(2, out.tiles().size());
        assertEquals(0xFF0000, ((RasterImage) out.tiles().get(0)).pixel(160, 160));
        assertEquals(0x0000FF, ((RasterImage) out.tiles().get(1)).pixel(160, 160));
    }

    @Test
    public void dynamicGridFollowsTheAspectRatio() {
        // MiniCPM-V: image_size 448, no pinpoints; 900x600 (ratio 2.69 -> multiple 3, best grid 2x1)
        Map<String, Object> m = metadata("resampler", "clip.minicpmv_version", 4,
                "clip.vision.image_size", 448);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        PreparedImages out = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(900, 600, 5, 6, 7));
        assertEquals(2, out.tiles().size());
        assertEquals(2, out.gridX());
        assertEquals(1, out.gridY());
        // refined 784x518 split into 392x518 cells; each resampler tile answers 64 queries
        assertEquals(392, out.tiles().get(0).width());
        assertEquals(518, out.tiles().get(0).height());
        assertEquals(3 * 64, out.totalTokens());
    }

    @Test
    public void minicpmvSmallImagesStayOverviewOnly() {
        Map<String, Object> m = metadata("resampler", "clip.minicpmv_version", 4,
                "clip.vision.image_size", 448);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        PreparedImages out = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(400, 300, 5, 6, 7));
        // area 120000 <= 448^2: no slices, one aspect-aligned overview (518x392)
        assertEquals(1, out.encodedTiles().size());
        assertTrue(out.tiles().isEmpty());
        assertNotNull(out.overview());
        assertEquals(518, out.overview().width());
        assertEquals(392, out.overview().height());
        assertEquals(64, out.totalTokens());
    }

    @Test
    public void internvlEncodesSlicesBeforeTheOverview() {
        Map<String, Object> m = metadata("internvl",
                "clip.vision.preproc_min_tiles", 1,
                "clip.vision.preproc_max_tiles", 2);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals(false, params.overviewFirst());
        PreparedImages out = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(300, 300, 0, 0, 0));
        assertEquals(1, out.encodedTiles().size());
        assertEquals(0, out.tiles().size());
        assertNotNull(out.overview());
        // the overview is the encoded tail
        assertEquals(out.overview(), out.encodedTiles().get(0));
    }

    @Test
    public void internvlTilesLargeImagesFirst() {
        Map<String, Object> m = metadata("internvl",
                "clip.vision.preproc_min_tiles", 1,
                "clip.vision.preproc_max_tiles", 2);
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        // the 672x336 candidate carries the most effective resolution for 1000x300
        PreparedImages out = ImagePreprocessors.forParams(params)
                .prepare(RasterImage.filled(1000, 300, 0, 0, 0));
        assertEquals(2, out.tiles().size());
        assertEquals(2, out.gridX());
        assertEquals(1, out.gridY());
        assertEquals(3, out.encodedTiles().size());
        // slice first, overview last
        assertEquals(out.overview(), out.encodedTiles().get(2));
        assertEquals(3 * 576, out.totalTokens());
    }
}

package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The fixed-size preprocessor against llama.cpp's {@code mtmd_image_preprocessor_fixed_size} with the LLaVA-1.5
 * family defaults: one square tile per image, aspect-preserving with the CLIP-average pad color, whose patch
 * grid is the whole token count.
 */
public class FixedSizePreprocessorTest {

    private static Map<String, Object> metadata(Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "mlp");
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams llava15() {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata()), ImageTokenBudget.none());
    }

    @Test
    public void wideImagePadsTopAndBottom() {
        ImagePreprocessor pre = ImagePreprocessors.forParams(llava15());
        // 672x112 fits 336x56 at scale 0.5, centered at y=140
        PreparedImages out = pre.prepare(RasterImage.filled(672, 112, 10, 20, 30));
        assertEquals(1, out.encodedTiles().size());
        assertEquals(336, out.encodedTiles().get(0).width());
        assertEquals(336, out.encodedTiles().get(0).height());
        assertNull(out.overview());
        assertEquals(0, out.gridX());
        assertEquals(0, out.gridY());
        RasterImage tile = (RasterImage) out.encodedTiles().get(0);
        assertEquals(0x0A141E, tile.pixel(167, 167));
        assertEquals(0x7A7468, tile.pixel(0, 0));
        assertEquals(0x7A7468, tile.pixel(167, 139));
        assertEquals(0x0A141E, tile.pixel(167, 140));
        assertEquals(0x0A141E, tile.pixel(167, 195));
        assertEquals(0x7A7468, tile.pixel(167, 196));
    }

    @Test
    public void tokenCountIsThePatchGrid() {
        PreparedImages out = ImagePreprocessors.forParams(llava15())
                .prepare(RasterImage.filled(336, 336, 0, 0, 0));
        assertEquals(24 * 24, out.totalTokens());
        assertEquals(576, out.tokensFor(0));
    }

    @Test
    public void normalizedFloatsCarryMeanStdPerChannel() {
        PreparedImages out = ImagePreprocessors.forParams(llava15())
                .prepare(RasterImage.filled(2, 2, 255, 0, 0));
        // the fixed-size tile is the 336x336 CLIP canvas, all pixels solid red
        assertEquals(1, out.normalized().size());
        float[] floats = out.normalized().get(0);
        assertEquals(336 * 336 * 3, floats.length);
        assertEquals((1.0f - 0.481f) / 0.269f, floats[0], 1e-5f);
        assertEquals((0.0f - 0.457f) / 0.261f, floats[1], 1e-5f);
        assertEquals((0.0f - 0.408f) / 0.276f, floats[2], 1e-5f);
        // the next pixel repeats the channel order
        assertEquals(floats[0], floats[3], 1e-6f);
    }

    @Test
    public void gemma3DividesTheGridByTheMergeFactor() {
        Map<String, Object> m = metadata();
        m.put("clip.vision.projector_type", "gemma3");
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        PreparedImages out = ImagePreprocessors.forParams(params).prepare(RasterImage.filled(500, 500, 0, 0, 0));
        // 336x336 tile: (336/14)^2 patches / 4^2 = 36 tokens
        assertEquals(336, out.encodedTiles().get(0).width());
        assertEquals(36, out.totalTokens());
        assertEquals(new ImageSize(36, 1), out.tokenGrid());
    }

    @Test
    public void sameSizeInputSkipsTheResample() {
        RasterImage src = RasterImage.filled(336, 336, 1, 2, 3);
        PreparedImages out = ImagePreprocessors.forParams(llava15()).prepare(src);
        assertSame(src, out.encodedTiles().get(0));
        assertTrue(out.overviewFirst());
    }
}

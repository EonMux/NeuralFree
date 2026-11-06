package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The per-tile image token counts of llama.cpp clip_n_output_tokens and clip_n_output_tokens_x/y. */
public class ImageTokenFormulaTest {

    private static ImageTokenGeometry geometry(int patch, int merge) {
        return ImageTokenGeometry.of(patch, merge, 336, 0);
    }

    /**
     * A transcription of the reference {@code clip_n_output_tokens} switch arms (tools/mtmd/clip.cpp), with the
     * reference's integer arithmetic, for the formulas below.
     */
    private static int reference(ImageTokenFormula formula, int nx, int ny, ImageTokenGeometry g) {
        int patch = g.patchSize();
        int nMerge = g.mergeSize();
        int nPatches = (nx / patch) * (ny / patch);
        switch (formula) {
            case PATCHES:
                return nPatches;
            case PATCHES_OVER_MERGE:
                return nPatches / (nMerge * nMerge);
            case MERGED_CONV:
                return (nx / (patch * nMerge)) * (ny / (patch * nMerge));
            case LDP:
                return nPatches / 4;
            case LDP_BOI_EOI:
                return nPatches / 4 + 2;
            case MERGED_ROWS_WITH_BREAKS: {
                int x = nx / patch / nMerge;
                int y = ny / patch / nMerge;
                return y * x + y - 1;
            }
            case ALIGNED_MERGED_GRID: {
                int out = patch * nMerge;
                return ((nx + out - 1) / out) * ((ny + out - 1) / out);
            }
            case ALIGNED_PATCHES_OVER_MERGE: {
                int stride = nMerge * nMerge;
                return ((nPatches + stride - 1) / stride * stride) / stride;
            }
            case IMAGE_SIZE_OVER_PATCH:
                return g.imageSize() / patch;
            case MERGED_CONV_WITH_NEWLINES: {
                int ow = (nx / patch) / nMerge;
                int oh = (ny / patch) / nMerge;
                return (ow + 1) * oh + 2;
            }
            default:
                throw new AssertionError(formula);
        }
    }

    @Test
    public void eachFormulaCountsTheReferenceTokens() {
        // 448x336 tile, patch 14 -> 32x24 patches
        ImageTokenGeometry g = geometry(14, 2);
        assertEquals(32 * 24, ImageTokenFormula.PATCHES.tokens(448, 336, g));
        assertEquals(32 * 24 / 4, ImageTokenFormula.PATCHES_OVER_MERGE.tokens(448, 336, g));
        assertEquals(16 * 12, ImageTokenFormula.MERGED_CONV.tokens(448, 336, g));
        assertEquals(32 * 24 / 4, ImageTokenFormula.LDP.tokens(448, 336, geometry(14, 1)));
        assertEquals(32 * 24 / 4 + 2, ImageTokenFormula.LDP_BOI_EOI.tokens(448, 336, geometry(14, 1)));
        assertEquals(96, ImageTokenFormula.RESAMPLER.tokens(448, 336, ImageTokenGeometry.of(14, 1, 448, 96)));
        assertEquals(64, ImageTokenFormula.POOLED_8X8.tokens(448, 336, geometry(14, 1)));
        // pixtral: 16x12 merged tokens and 11 [IMG_BREAK] rows
        assertEquals(16 * 12 + 12 - 1, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.tokens(448, 336, g));
        assertEquals(16 * 12, ImageTokenFormula.ALIGNED_MERGED_GRID.tokens(448, 336, g));
        assertEquals(32 * 24 / 4, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE.tokens(448, 336, g));
        // hunyuanvl: 16x12 merged grid, a newline per row, begin and end rows
        assertEquals(17 * 12 + 2, ImageTokenFormula.MERGED_CONV_WITH_NEWLINES.tokens(448, 336, g));
    }

    @Test
    public void everyFormulaMatchesTheReferenceTranscriptionOverAlignedTiles() {
        ImageTokenFormula[] formulas = {ImageTokenFormula.PATCHES, ImageTokenFormula.PATCHES_OVER_MERGE,
                ImageTokenFormula.MERGED_CONV, ImageTokenFormula.LDP, ImageTokenFormula.LDP_BOI_EOI,
                ImageTokenFormula.MERGED_ROWS_WITH_BREAKS, ImageTokenFormula.ALIGNED_MERGED_GRID,
                ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE, ImageTokenFormula.IMAGE_SIZE_OVER_PATCH,
                ImageTokenFormula.MERGED_CONV_WITH_NEWLINES};
        for (int merge = 1; merge <= 4; merge++) {
            ImageTokenGeometry g = ImageTokenGeometry.of(16, merge, 512, 0);
            for (int gx = 1; gx <= 5; gx++) {
                for (int gy = 1; gy <= 5; gy++) {
                    int w = gx * 16 * merge;
                    int h = gy * 16 * merge;
                    for (ImageTokenFormula f : formulas) {
                        assertEquals(f + " " + w + "x" + h + " merge " + merge, reference(f, w, h, g),
                                f.tokens(w, h, g));
                    }
                }
            }
        }
    }

    @Test
    public void pixtralCountsTheBreakRowsOfTheMergedGrid() {
        // Mistral Small 3.1: patch 14, merge 2; a 1540x812 tile is 110x58 patches -> 55x29 tokens + 28 breaks
        assertEquals(55 * 29 + 28, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.tokens(1540, 812, geometry(14, 2)));
        // Pixtral 12B: patch 16, no merge; 1024x512 is 64x32 tokens + 31 breaks
        assertEquals(64 * 32 + 31, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.tokens(1024, 512, geometry(16, 1)));
        // a single token row has no break
        assertEquals(64, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.tokens(1024, 16, geometry(16, 1)));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.tokens(1540, 826, geometry(14, 2))).getMessage()
                .contains("height 826 is not a multiple of 28"));
    }

    @Test
    public void kimiCountsTheAlignedMergedGrid() {
        // 434x294 with patch 14, merge 2: 31x21 patches -> ceil(434/28) x ceil(294/28) = 16 x 11
        assertEquals(16 * 11, ImageTokenFormula.ALIGNED_MERGED_GRID.tokens(434, 294, geometry(14, 2)));
        // plain patches would be 31 * 21 = 651
        assertEquals(31 * 21, ImageTokenFormula.PATCHES.tokens(434, 294, geometry(14, 2)));
    }

    @Test
    public void paddleAndDotsCountTheAlignedPatchesOverTheMergeSquare() {
        // 42x56 with patch 14, merge 2: 3x4 = 12 patches -> 12 / 4 = 3 tokens, where the merged grid is 1x2
        assertEquals(3, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE.tokens(42, 56, geometry(14, 2)));
        // 42x42: 9 patches align up to 12 -> 3 tokens
        assertEquals(3, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE.tokens(42, 42, geometry(14, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.MERGED_CONV.tokens(42, 56, geometry(14, 2)));
    }

    @Test
    public void gemma3nCountsImageSizeOverPatchForAnyTile() {
        ImageTokenGeometry g = ImageTokenGeometry.of(3, 1, 768, 0);
        assertEquals(256, ImageTokenFormula.IMAGE_SIZE_OVER_PATCH.tokens(768, 768, g));
        assertEquals(256, ImageTokenFormula.IMAGE_SIZE_OVER_PATCH.tokens(300, 200, g));
    }

    @Test
    public void hunyuanFloorsTheMergedGridAndAddsItsExtraRows() {
        // 70x56 with patch 14, merge 2: 5x4 patches -> 2x2 merged (the 5th column dropped) -> 3 * 2 + 2
        assertEquals(3 * 2 + 2, ImageTokenFormula.MERGED_CONV_WITH_NEWLINES.tokens(70, 56, geometry(14, 2)));
    }

    @Test
    public void onlyTheMergedGridHasTokenRows() {
        ImageTokenGeometry g = geometry(14, 2);
        assertEquals(16, ImageTokenGrid.MERGED.columns(448, 336, g, 192));
        assertEquals(12, ImageTokenGrid.MERGED.rows(448, 336, g));
        assertEquals(192, ImageTokenGrid.ROW.columns(448, 336, g, 192));
        assertEquals(1, ImageTokenGrid.ROW.rows(448, 336, g));
        assertEquals(ImageTokenGrid.MERGED, ImageTokenFormula.MERGED_CONV.defaultGrid());
        assertEquals(ImageTokenGrid.MERGED, ImageTokenFormula.MERGED_CONV_WITH_NEWLINES.defaultGrid());
        assertEquals(ImageTokenGrid.ROW, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE.defaultGrid());
        assertEquals(ImageTokenGrid.ROW, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS.defaultGrid());
    }

    @Test
    public void misalignedTilesAndMissingGeometryAreRefused() {
        ImageTokenGeometry g = geometry(14, 2);
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.MERGED_CONV.tokens(450, 336, g)).getMessage().contains("multiple"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.PATCHES.tokens(448, 340, g)).getMessage().contains("height"));
        // 3x3 patches cannot merge 2x2
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.PATCHES_OVER_MERGE.tokens(42, 42, g)).getMessage().contains("merge^2"));
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> ImageTokenFormula.RESAMPLER.tokens(448, 336, g)).getMessage().contains("query"));
        assertThrows(IllegalArgumentException.class, () -> ImageTokenFormula.PATCHES.tokens(0, 336, g));
        assertThrows(IllegalArgumentException.class, () -> ImageTokenGrid.MERGED.rows(448, 0, g));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ImageTokenGeometry.of(0, 1, 336, 0))
                .getMessage().contains("patch size"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ImageTokenGeometry.of(14, 0, 336, 0))
                .getMessage().contains("merge"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ImageTokenGeometry.of(14, 1, -1, 0))
                .getMessage().contains("image size"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> ImageTokenGeometry.of(14, 1, 336, -1))
                .getMessage().contains("query"));
    }

    @Test
    public void theResamplerQueryCountFollowsTheKeyThenTheVersion() {
        Map<String, Object> m = new HashMap<>();
        assertEquals(96, ImageTokenFormula.resamplerQueryNum(GgufKv.ofMap(m)));
        for (int version : new int[] {3, 4, 5, 6, 100045}) {
            m.put("clip.minicpmv_version", version);
            assertEquals("version " + version, 64, ImageTokenFormula.resamplerQueryNum(GgufKv.ofMap(m)));
        }
        m.put("clip.minicpmv_version", 2);
        assertEquals(96, ImageTokenFormula.resamplerQueryNum(GgufKv.ofMap(m)));
        m.put("clip.minicpmv_query_num", 80);
        assertEquals(80, ImageTokenFormula.resamplerQueryNum(GgufKv.ofMap(m)));
    }
}

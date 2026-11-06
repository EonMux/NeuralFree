package com.eonmux.neuralfree.gguf.mm;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link MediaMarkers}: the marker texts of llama.cpp's {@code mtmd_context::init_vision} / {@code init_audio} per
 * projector family: plain begin / end pairs, the MiniCPM-V slice templates, the vocabulary-dependent Kimi K2.5
 * pair, the non-causal image rows, the families without markers, the slice-grid layouts per family, and the audio
 * pairs added beside them.
 */
public class MediaMarkersTest {

    private static MediaMarkers vision(ProjectorType projector) {
        return MediaMarkers.vision(projector, 0, 0, t -> false);
    }

    @Test
    public void familiesWithAMarkerPairWrapTheImage() {
        assertPair(vision(ProjectorType.QWEN2VL), "<|vision_start|>", "<|vision_end|>");
        assertPair(vision(ProjectorType.QWEN25VL), "<|vision_start|>", "<|vision_end|>");
        assertPair(vision(ProjectorType.GEMMA3), "<start_of_image>", "<end_of_image>");
        assertPair(vision(ProjectorType.LLAMA4), "<|image_start|>", "<|image_end|>");
        assertPair(vision(ProjectorType.INTERNVL), "<img>", "</img>");
        assertPair(vision(ProjectorType.PIXTRAL), "", "[IMG_END]");
        assertPair(vision(ProjectorType.GLM4V), "<|begin_of_image|>", "<|end_of_image|>");
    }

    @Test
    public void llavaAndPhi4AddNoMarkerText() {
        for (ProjectorType p : new ProjectorType[] {ProjectorType.MLP, ProjectorType.MLP_NORM,
                ProjectorType.PHI4}) {
            MediaMarkers m = vision(p);
            assertPair(m, "", "");
            assertEquals("", m.overviewStart());
            assertEquals("", m.sliceStart());
            assertFalse(m.nonCausalImage());
        }
    }

    @Test
    public void onlyGemma3ImageRowsDecodeWithoutTheCausalMask() {
        assertTrue(vision(ProjectorType.GEMMA3).nonCausalImage());
        assertFalse(vision(ProjectorType.GEMMA3NV).nonCausalImage());
        assertFalse(vision(ProjectorType.QWEN2VL).nonCausalImage());
    }

    @Test
    public void miniCpmvVersionsPickTheirSliceTemplate() {
        MediaMarkers v25 = MediaMarkers.vision(ProjectorType.MINICPMV, 2, 0, t -> false);
        assertEquals("<image>", v25.overviewStart());
        assertEquals("<slice>", v25.slicesStart());
        assertEquals("<image>", v25.sliceStart());
        assertEquals("\n", v25.rowEnd());
        assertTrue(v25.overviewFirst());

        MediaMarkers v26 = MediaMarkers.vision(ProjectorType.MINICPMV, 3, 0, t -> false);
        assertEquals("<image>", v26.overviewStart());
        assertEquals("", v26.slicesStart());
        assertEquals("<slice>", v26.sliceStart());
        assertEquals("</slice>", v26.sliceEnd());
        assertFalse(v26.imageId());
        assertTrue(MediaMarkers.vision(ProjectorType.MINICPMV4_6, 0, 0, t -> false).imageId());

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MediaMarkers.vision(ProjectorType.MINICPMV, 7, 0, t -> false));
        assertTrue(e.getMessage(), e.getMessage().contains("minicpmv version: 7"));
    }

    @Test
    public void cohere2vEndsSlicesAndTheOverviewWithALineBreak() {
        MediaMarkers m = vision(ProjectorType.COHERE2V);
        assertPair(m, "<|START_OF_IMG|>", "<|END_OF_IMG|>");
        assertEquals("<|IMG_LINE_BREAK|>", m.sliceEnd());
        assertEquals("<|IMG_LINE_BREAK|>", m.overviewEnd());
    }

    @Test
    public void kimiK25MarkersFollowTheVocabulary() {
        assertPair(MediaMarkers.vision(ProjectorType.KIMIK25, 0, 0, "<|begin_of_image|>"::equals),
                "<|begin_of_image|>", "<|end_of_image|>");
        assertPair(MediaMarkers.vision(ProjectorType.KIMIK25, 0, 0, t -> false), "<|media_begin|>", "<|media_end|>");
    }

    @Test
    public void audioMarkersJoinTheImageMarkers() {
        MediaMarkers omni = vision(ProjectorType.QWEN25VL).withAudio(ProjectorType.QWEN25O);
        assertPair(omni, "<|vision_start|>", "<|vision_end|>");
        assertEquals("<|audio_bos|>", omni.audioBegin());
        assertEquals("<|audio_eos|>", omni.audioEnd());
        MediaMarkers qwen3 = MediaMarkers.none().withAudio(ProjectorType.QWEN3A);
        assertEquals("<|audio_start|>", qwen3.audioBegin());
        assertEquals("<|audio_end|>", qwen3.audioEnd());
        assertEquals("[BEGIN_AUDIO]", MediaMarkers.none().withAudio(ProjectorType.VOXTRAL).audioBegin());
        MediaMarkers gemma3n = MediaMarkers.none().withAudio(ProjectorType.GEMMA3NA);
        assertEquals("<start_of_audio>", gemma3n.audioBegin());
        assertEquals("<end_of_audio>", gemma3n.audioEnd());
        MediaMarkers ultravox = MediaMarkers.none().withAudio(ProjectorType.ULTRAVOX);
        assertEquals("", ultravox.audioBegin());
        assertEquals("", ultravox.audioEnd());
    }

    @Test
    public void theNewlyPreprocessedFamiliesTakeTheirFrames() {
        assertPair(vision(ProjectorType.GEMMA4V), "<|image>", "<image|>");
        assertPair(vision(ProjectorType.GEMMA4UV), "<|image>", "<image|>");
        assertPair(vision(ProjectorType.LIGHTONOCR), "<|im_start|>", "<|im_end|>");
        assertPair(vision(ProjectorType.MINIMAX_M3), "]<]start of image[>[", "]<]end of image[>[");
        assertPair(vision(ProjectorType.MUSE_GLIMMER), "<|image_start|>", "<|image_end|>");
        assertPair(vision(ProjectorType.LFM3VL), "<|vision_start|>", "<|vision_end|>");
        assertPair(vision(ProjectorType.YOUTUVL), "<|vision_start|>", "<|vision_end|>");
        assertPair(vision(ProjectorType.GLM5V), "<|begin_of_image|>", "<|end_of_image|>");
        assertPair(vision(ProjectorType.DOTS3NOTE_V), "<|img|>", "<|endofimg|>");
        assertPair(vision(ProjectorType.DEEPSEEKOCR), "", "\n");
        assertPair(vision(ProjectorType.DEEPSEEKOCR2), "", "\n");
        assertPair(vision(ProjectorType.LFM2), "<|image_start|>", "<|image_end|>");
        assertPair(vision(ProjectorType.STEP3VL), "", "");
        // d1-omni shares lfm2's tiles and thumbnail without any separator text
        for (ProjectorType p : new ProjectorType[] {ProjectorType.D1OMNI_V, ProjectorType.GLM_EDGE,
                ProjectorType.COGVLM, ProjectorType.JANUS_PRO}) {
            MediaMarkers m = vision(p);
            assertPair(m, "", "");
            assertEquals(p.typeName(), "[slice(0), slice(1), slice(2), slice(3), overview]",
                    m.tiledLayout(2, 2).toString());
        }
    }

    @Test
    public void gemma4ImageRowsAreCausalOnlyForTheSmallProjectors() {
        assertFalse(MediaMarkers.vision(ProjectorType.GEMMA4V, 0, 1536, t -> false).nonCausalImage());
        assertFalse(MediaMarkers.vision(ProjectorType.GEMMA4V, 0, 2560, t -> false).nonCausalImage());
        assertTrue(MediaMarkers.vision(ProjectorType.GEMMA4V, 0, 5376, t -> false).nonCausalImage());
        assertTrue(MediaMarkers.vision(ProjectorType.GEMMA4UV, 0, 1536, t -> false).nonCausalImage());
        assertTrue(vision(ProjectorType.SENSENOVA_U1).nonCausalImage());
    }

    // the 2x2 slice-grid walks of mtmd_tokenizer::add_media with each family's init_vision markers

    @Test
    public void idefics3NamesEveryTileAndEndsWithTheGlobalImage() {
        assertLayout(vision(ProjectorType.IDEFICS3),
                "[text(<fake_token_around_image><row_1_col_1>), slice(0), "
                        + "text(<fake_token_around_image><row_1_col_2>), slice(1), text(\\n), "
                        + "text(<fake_token_around_image><row_2_col_1>), slice(2), "
                        + "text(<fake_token_around_image><row_2_col_2>), slice(3), "
                        + "text(\\n\\n<fake_token_around_image><global-img>), overview, "
                        + "text(<fake_token_around_image>)]");
        assertEquals("<fake_token_around_image><row_3_col_12>", vision(ProjectorType.IDEFICS3).sliceStart(3, 12));
    }

    @Test
    public void miniCpmv25FramesSlicesLikeTheOverviewInsideOneSliceBlock() {
        assertLayout(MediaMarkers.vision(ProjectorType.MINICPMV, 2, 0, t -> false),
                "[text(<image>), overview, text(</image>), text(<slice>), "
                        + "text(<image>), slice(0), text(</image>), text(<image>), slice(1), text(</image>), "
                        + "text(\\n), "
                        + "text(<image>), slice(2), text(</image>), text(<image>), slice(3), text(</image>), "
                        + "text(</slice>)]");
    }

    @Test
    public void miniCpmv26FramesEverySlice() {
        String grid = "text(<image>), overview, text(</image>), "
                + "text(<slice>), slice(0), text(</slice>), text(<slice>), slice(1), text(</slice>), text(\\n), "
                + "text(<slice>), slice(2), text(</slice>), text(<slice>), slice(3), text(</slice>)]";
        assertLayout(MediaMarkers.vision(ProjectorType.MINICPMV, 3, 0, t -> false), "[" + grid);
        for (ProjectorType p : new ProjectorType[] {ProjectorType.MINICPMV4_6, ProjectorType.MINICPMV4_7,
                ProjectorType.MINICPM_OCR_1_0}) {
            assertLayout(vision(p), "[image_id, " + grid);
            assertEquals("[image_id, text(<image>), overview, text(</image>)]", vision(p).tiledLayout(0, 0)
                    .toString());
        }
    }

    @Test
    public void miniCpmv47LaysItsImagesOutAsOneCanvasChunk() {
        MediaMarkers m = vision(ProjectorType.MINICPMV4_7);
        assertTrue(m.canvasImages());
        assertEquals("[image_id, text(<image>), canvas]", m.canvasLayout().toString());
        for (ProjectorType p : new ProjectorType[] {ProjectorType.MINICPMV4_6, ProjectorType.MINICPM_OCR_1_0,
                ProjectorType.MINICPMV}) {
            assertFalse(vision(p).canvasImages());
            assertThrows(IllegalStateException.class, () -> vision(p).canvasLayout());
        }
    }

    @Test
    public void llama4SeparatesTilesAndEndsEveryRow() {
        MediaMarkers m = vision(ProjectorType.LLAMA4);
        assertLayout(m, "[slice(0), text(<|tile_x_separator|>), slice(1), text(<|tile_y_separator|>), "
                + "slice(2), text(<|tile_x_separator|>), slice(3), text(<|tile_y_separator|>), "
                + "text(<|image|>), overview]");
        assertEquals("[text(<|image|>), overview]", m.tiledLayout(0, 0).toString());
    }

    @Test
    public void step3vlFramesPatchesAndTheOverviewLast() {
        assertLayout(vision(ProjectorType.STEP3VL),
                "[text(<patch_start>), slice(0), text(<patch_end>), text(<patch_start>), slice(1), "
                        + "text(<patch_end>), text(<patch_newline>), "
                        + "text(<patch_start>), slice(2), text(<patch_end>), text(<patch_start>), slice(3), "
                        + "text(<patch_end>), text(<im_start>), overview, text(<im_end>)]");
    }

    @Test
    public void lfm2NamesEveryTileThenTheThumbnail() {
        assertLayout(vision(ProjectorType.LFM2),
                "[text(<|img_row_1_col_1|>), slice(0), text(<|img_row_1_col_2|>), slice(1), "
                        + "text(<|img_row_2_col_1|>), slice(2), text(<|img_row_2_col_2|>), slice(3), "
                        + "text(<|img_thumbnail|>), overview]");
    }

    @Test
    public void cohere2vBreaksTheLineAfterEveryTile() {
        assertLayout(vision(ProjectorType.COHERE2V),
                "[slice(0), text(<|IMG_LINE_BREAK|>), slice(1), text(<|IMG_LINE_BREAK|>), "
                        + "slice(2), text(<|IMG_LINE_BREAK|>), slice(3), text(<|IMG_LINE_BREAK|>), "
                        + "overview, text(<|IMG_LINE_BREAK|>)]");
    }

    @Test
    public void deepseekOcrPlacesTheGlobalViewAfterItsTileRows() {
        assertEquals("[slice(0), slice(1), overview]", vision(ProjectorType.DEEPSEEKOCR).tiledLayout(1, 2)
                .toString());
        assertEquals("[overview]", vision(ProjectorType.DEEPSEEKOCR2).tiledLayout(0, 0).toString());
    }

    @Test
    public void aGridMustHaveBothSidesOrNeither() {
        MediaMarkers m = vision(ProjectorType.LFM2);
        assertThrows(IllegalArgumentException.class, () -> m.tiledLayout(2, 0));
        assertThrows(IllegalArgumentException.class, () -> m.tiledLayout(0, 1));
        assertThrows(IllegalArgumentException.class, () -> m.tiledLayout(-1, -1));
    }

    private static void assertLayout(MediaMarkers m, String expected) {
        assertEquals(expected, m.tiledLayout(2, 2).toString());
    }

    private static void assertPair(MediaMarkers m, String begin, String end) {
        assertEquals(begin, m.imageBegin());
        assertEquals(end, m.imageEnd());
    }
}

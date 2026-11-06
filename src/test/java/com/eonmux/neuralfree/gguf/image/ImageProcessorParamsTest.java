package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.mm.ClipHparams;
import com.eonmux.neuralfree.gguf.mm.ImageTokenFormula;
import com.eonmux.neuralfree.gguf.mm.PatchMergeRule;
import com.eonmux.neuralfree.gguf.mm.ProjectorEntry;
import com.eonmux.neuralfree.gguf.mm.ProjectorModality;
import com.eonmux.neuralfree.gguf.mm.ProjectorRegistry;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;
import com.eonmux.neuralfree.gguf.mm.UnsupportedProjectorException;
import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;
import com.eonmux.neuralfree.gguf.plugin.ProjectorProvider;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The metadata load of {@code clip_model_loader::load_hparams} over {@code clip.vision.*} keys: family
 * defaults, GGUF keys over them, and the token budget last ({@code custom_image_min_tokens} precedence).
 */
public class ImageProcessorParamsTest {

    /** A minimal valid metadata map for the projector, with the given extra keys. */
    private static Map<String, Object> metadata(ProjectorType projector, Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", projector.typeName());
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    @Test
    public void qwenFamilyDefaultsMatchLlamaCpp() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.QWEN2VL)), ImageTokenBudget.none());
        assertEquals("qwen2vl_merger", p.projectorName());
        assertEquals(ProjectorType.QWEN2VL.visionPreprocessing(), p.preprocessing());
        assertEquals(14, p.patchSize());
        assertEquals(2, p.mergeSize());
        assertEquals(ResizeAlgo.BICUBIC, p.resizeAlgo());
        assertEquals(PadStyle.NONE, p.padStyle());
        assertEquals(true, p.roundHalfEven());
        // set_limit_image_tokens(8, 4096) through the (14 * 2)^2 merged patch area
        assertEquals(8L * 784L, p.minPixels());
        assertEquals(4096L * 784L, p.maxPixels());
    }

    @Test
    public void ggufPixelKeysOverrideTheFamilyDefaults() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.QWEN2VL,
                "clip.vision.image_min_pixels", 100000,
                "clip.vision.image_max_pixels", 2000000)), ImageTokenBudget.none());
        assertEquals(100000L, p.minPixels());
        assertEquals(2000000L, p.maxPixels());
    }

    @Test
    public void tokenBudgetWinsOverGgufPixelKeys() {
        // --image-min-tokens/--image-max-tokens replace the pixel limits, per merged patch area
        ImageTokenBudget budget = ImageTokenBudget.builder().minTokens(1024).maxTokens(2048).build();
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.QWEN2VL,
                "clip.vision.image_min_pixels", 100000,
                "clip.vision.image_max_pixels", 2000000)), budget);
        assertEquals(1024L * 784L, p.minPixels());
        assertEquals(2048L * 784L, p.maxPixels());
    }

    @Test
    public void spatialMergeKeyOverridesTheFamilyDefault() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.QWEN25VL,
                "clip.vision.spatial_merge_size", 1)), ImageTokenBudget.none());
        assertEquals(1, p.mergeSize());
    }

    @Test
    public void scaleFactorKeyAppliesWhenNoSpatialMergeIsCarried() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.MINICPMV4_7,
                "clip.vision.projector.scale_factor", 2)), ImageTokenBudget.none());
        assertEquals(2, p.mergeSize());
        assertEquals(PadStyle.NONE, p.overviewPadStyle());
        assertEquals(PadStyle.NONE, p.refinedPadStyle());
    }

    @Test
    public void llavaPinpointsEnableTheSliceLayout() {
        Map<String, Object> m = metadata(ProjectorType.MLP,
                "clip.vision.image_grid_pinpoints",
                new int[] {336, 336, 672, 336, 336, 672, 672, 672, 1008, 336, 336, 1008, 1008, 672, 672, 1008,
                        1008, 1008});
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals(9, p.gridPinpoints().size());
        assertEquals(new ImageSize(1008, 1008), p.gridPinpoints().get(8));
        // llava-1.6 defaults: overview stretches, refined pads with the CLIP average color
        assertEquals(PadStyle.NONE, p.overviewPadStyle());
        assertEquals(PadStyle.CEIL, p.refinedPadStyle());
        assertEquals(122, p.refinedPadColor()[0]);
        assertEquals(true, p.overviewFirst());
    }

    @Test
    public void llavaWithoutPinpointsStaysFixedSize() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.MLP)), ImageTokenBudget.none());
        assertEquals(true, p.gridPinpoints().isEmpty());
        assertEquals(PadStyle.CEIL, p.padStyle());
        assertEquals(122, p.padColor()[0]);
        assertEquals(0, p.minPixels());
        assertEquals(0, p.maxPixels());
    }

    @Test
    public void gemma3CarriesMergeFourAndBilinear() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.GEMMA3)), ImageTokenBudget.none());
        assertEquals(4, p.mergeSize());
        assertEquals(ResizeAlgo.BILINEAR, p.resizeAlgo());
    }

    @Test
    public void pixtralLongestEdgeDerivesTheTokenCap() {
        // preproc_image_size 1024 over merged patch 16 -> 64 tokens per side -> 4096 tokens -> 4096 * 256 pixels
        Map<String, Object> m = metadata(ProjectorType.PIXTRAL,
                "clip.vision.patch_size", 16,
                "clip.vision.preproc_image_size", 1024);
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals(1024, p.longestEdge());
        assertEquals(true, p.floorBeforeAlign());
        assertEquals(PadStyle.NONE, p.padStyle());
        assertEquals(0L, p.minPixels());
        assertEquals(4096L * 256L, p.maxPixels());
    }

    @Test
    public void pixtralWithoutLongestEdgeUsesTheEightTo1024Defaults() {
        Map<String, Object> m = metadata(ProjectorType.PIXTRAL, "clip.vision.patch_size", 16);
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals(0, p.longestEdge());
        assertEquals(8L * 256L, p.minPixels());
        assertEquals(1024L * 256L, p.maxPixels());
    }

    @Test
    public void internvlBuildsTheTileCandidates() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.INTERNVL)), ImageTokenBudget.none());
        assertEquals(false, p.overviewFirst());
        assertEquals(35, p.gridPinpoints().size());
        // ordered by grid width: a=1 first with heights 1..12
        assertEquals(new ImageSize(336, 336), p.gridPinpoints().get(0));
        assertEquals(new ImageSize(336, 336 * 12), p.gridPinpoints().get(11));
        assertEquals(new ImageSize(336 * 2, 336), p.gridPinpoints().get(12));
    }

    @Test
    public void missingRequiredKeysAreRefusedByName() {
        Map<String, Object> m = metadata(ProjectorType.MLP);
        m.remove("clip.vision.patch_size");
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            fail("expected a missing key refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("clip.vision.patch_size"));
        }
        Map<String, Object> m2 = metadata(ProjectorType.MLP);
        m2.put("clip.vision.image_mean", new float[] {0.5f, 0.5f});
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m2), ImageTokenBudget.none());
            fail("expected a short mean refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("clip.vision.image_mean"));
        }
    }

    @Test
    public void oddPinpointsAreRefusedByName() {
        Map<String, Object> m = metadata(ProjectorType.MLP,
                "clip.vision.image_grid_pinpoints", new int[] {336, 336, 672});
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            fail("expected an odd pinpoints refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("image_grid_pinpoints"));
        }
    }

    @Test
    public void granite4VisionNeedsItsPinpointsAndCountsItsQformerWindows() {
        Map<String, Object> m = metadata(ProjectorType.GRANITE4_VISION, "clip.vision.projector.window_side", 8,
                "clip.vision.projector.query_side", 4);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.vision.image_grid_pinpoints"));
        m.put("clip.vision.image_grid_pinpoints", new int[] {336, 336, 672, 336});
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals(VisionPreprocessing.Family.GRANITE4V, p.preprocessing().family());
        assertEquals(8, p.windowSide());
        assertEquals(4, p.querySide());
        assertEquals(8, p.tokenGeometry().windowSide());
        // 24 patches per side: 3 windows of 4 query tokens, 12 x 12
        assertEquals(144, p.tokensFor(new ImageSize(336, 336)));
        assertEquals(144 + 6 * 25, p.anyresTokens(new ImageSize(336, 336), 2, 1, new ImageSize(1000, 250)));
        assertEquals(PadStyle.NONE, p.overviewPadStyle());
        assertEquals(PadStyle.CEIL, p.refinedPadStyle());
        assertEquals(ResizeAlgo.BICUBIC, p.refinedResizeAlgo());
        assertEquals(p.windowSide(), p.toBuilder().build().windowSide());
        m.remove("clip.vision.projector.query_side");
        assertThrows(RuntimeException.class,
                () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
    }

    @Test
    public void unsupportedProjectorsAreRefusedByName() {
        for (String name : new String[] {"ultravox", "made_up_projector", "minicpmv4.6"}) {
            Map<String, Object> m = metadata(ProjectorType.MLP);
            m.put("clip.vision.projector_type", name);
            UnsupportedProjectorException e = assertThrows(UnsupportedProjectorException.class,
                    () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
            assertTrue(e.getMessage(), e.getMessage().contains("'" + name + "'"));
            assertEquals(name, e.projectorType());
        }
        Map<String, Object> m = metadata(ProjectorType.MLP);
        m.remove("clip.vision.projector_type");
        GgufFormatException missing = assertThrows(GgufFormatException.class,
                () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
        assertTrue(missing.getMessage().contains("clip.projector_type"));
    }

    @Test
    public void theCommonProjectorTypeKeyWinsLikeTheEncoder() {
        Map<String, Object> m = metadata(ProjectorType.MLP);
        m.remove("clip.vision.projector_type");
        m.put("clip.projector_type", "qwen2vl_merger");
        assertEquals("qwen2vl_merger", ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none())
                .projectorName());
        // the legacy mixed-modality name resolves to the vision merger, as ClipHparams resolves it
        m.put("clip.projector_type", "qwen2.5o");
        m.put("clip.vision.projector_type", "mlp");
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        assertEquals("qwen2.5vl_merger", p.projectorName());
        assertEquals(2, p.mergeSize());
    }

    @Test
    public void referenceSpellingsOfTheMiniCpmvMergersResolve() {
        for (String name : new String[] {"minicpmv4_6", "minicpmv4_7"}) {
            Map<String, Object> m = metadata(ProjectorType.MLP);
            m.put("clip.vision.projector_type", name);
            ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            assertEquals(4, p.mergeSize());
            assertEquals(ImageTokenFormula.MERGED_CONV, p.preprocessing().tokenFormula());
        }
    }

    @Test
    public void theMergeIsReadWithTheRuleTheEncoderUses() {
        // pixtral merges by spatial_merge_size (clip.cpp), never by the projector scale factor
        ImageProcessorParams pixtral = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.PIXTRAL,
                "clip.vision.spatial_merge_size", 2, "clip.vision.projector.scale_factor", 4)),
                ImageTokenBudget.none());
        assertEquals(2, pixtral.mergeSize());
        // llama4 reads the projector scale factor: (336 / 14)^2 patches / 2^2
        ImageProcessorParams llama4 = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.LLAMA4,
                "clip.vision.projector.scale_factor", 2)), ImageTokenBudget.none());
        assertEquals(2, llama4.mergeSize());
        assertEquals(24 * 24 / 4, llama4.tokensFor(new ImageSize(336, 336)));
        for (ProjectorType type : new ProjectorType[] {ProjectorType.MLP, ProjectorType.PIXTRAL,
                ProjectorType.LLAMA4, ProjectorType.GEMMA3, ProjectorType.QWEN2VL, ProjectorType.INTERNVL,
                ProjectorType.PHI4, ProjectorType.MINICPMV, ProjectorType.IDEFICS3}) {
            Map<String, Object> m = encoderMetadata(type);
            m.put("clip.vision.spatial_merge_size", 3);
            m.put("clip.vision.projector.scale_factor", 2);
            m.put("clip.vision.image_min_pixels", 1024);
            m.put("clip.vision.image_max_pixels", 4096 * 784);
            int encoderMerge = ClipHparams.from(GgufKv.ofMap(m), ProjectorModality.VISION).nMerge();
            assertEquals(type.typeName(), encoderMerge,
                    ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()).mergeSize());
        }
    }

    /** The metadata of a full vision encoder of the projector, over a 504-pixel (36-patch) grid. */
    private static Map<String, Object> encoderMetadata(ProjectorType type) {
        Map<String, Object> m = metadata(type, "clip.vision.image_size", 504);
        m.put("clip.has_vision_encoder", true);
        m.put("clip.vision.embedding_length", 64);
        m.put("clip.vision.feed_forward_length", 128);
        m.put("clip.vision.block_count", 2);
        m.put("clip.vision.projection_dim", 32);
        m.put("clip.vision.attention.head_count", 4);
        m.put("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        return m;
    }

    @Test
    public void aRequiredMergeKeyIsRefusedByName() {
        Map<String, Object> m = metadata(ProjectorType.DOTS_OCR, "clip.vision.image_min_pixels", 1024,
                "clip.vision.image_max_pixels", 4096 * 784);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
        assertTrue(e.getMessage().contains("clip.vision.projector.scale_factor"));
    }

    @Test
    public void aPluginTypeDeclaresItsPreprocessingAndTokenCount() {
        VisionPreprocessing declared = VisionPreprocessing.of(VisionPreprocessing.Family.QWEN_MERGER,
                ImageTokenFormula.PATCHES_OVER_MERGE, PatchMergeRule.spatialMergeSize(2));
        ProjectorEntry row = ProjectorEntry.keyed("future_vl", ProjectorType.UNKNOWN, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER, declared);
        ProjectorEntry inherited = ProjectorEntry.keyed("qwen3vl_merger_next", ProjectorType.QWEN3VL,
                ProjectorModality.VISION, ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        ProjectorEntry bare = ProjectorEntry.keyed("future_bare", ProjectorType.UNKNOWN, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        List<ProjectorProvider<ProjectorEntry>> providers =
                Collections.singletonList(() -> Arrays.asList(row, inherited, bare));
        // built-in names resolve through their ProjectorType, so the plugin rows alone stand in for the registry
        Map<String, ProjectorEntry> rows = ProjectorRegistry.mergeProviders(Collections.emptyMap(), providers);

        assertSame(declared, ProjectorRegistry.visionPreprocessing("future_vl", rows));
        assertSame(ProjectorType.QWEN3VL.visionPreprocessing(),
                ProjectorRegistry.visionPreprocessing("qwen3vl_merger_next", rows));
        UnsupportedProjectorException e = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.visionPreprocessing("future_bare", rows));
        assertTrue(e.getMessage().contains("'future_bare'"));

        Map<String, Object> m = metadata(ProjectorType.MLP, "clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.projector_type", "future_vl");
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), "future_vl",
                ProjectorRegistry.visionPreprocessing("future_vl", rows), ImageTokenBudget.none());
        assertEquals("future_vl", p.projectorName());
        // the declared formula: (784 / 14)^2 patches / 2^2, where the built-in merger counts (784 / 28)^2
        assertEquals(56 * 56 / 4, p.tokensFor(new ImageSize(784, 784)));
        assertEquals(PadStyle.NONE, p.padStyle());
        assertTrue(ImagePreprocessors.forParams(p) instanceof DynSizePreprocessor);
        assertTrue(p.toString().contains("future_vl"));
    }

    @Test
    public void qwenFamiliesWithoutPixelKeysStillRequireThemWhenTheConverterAlwaysWritesThem() {
        // mimovl reads the pixel keys as required: absent keys are refused by name
        Map<String, Object> m = metadata(ProjectorType.MIMOVL);
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            fail("expected a missing pixel keys refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("clip.vision.image_min_pixels"));
        }
    }

    @Test
    public void tokenCountsFollowTheProjectorFormulas() {
        ImageProcessorParams qwen = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.QWEN2VL)), ImageTokenBudget.none());
        // merged conv: (784 / (14 * 2))^2
        assertEquals(28 * 28, qwen.tokensFor(new ImageSize(784, 784)));
        assertEquals(new ImageSize(28, 28), qwen.tokenGridFor(new ImageSize(784, 784)));

        ImageProcessorParams gemma = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.GEMMA3)), ImageTokenBudget.none());
        // patches / merge^2: (896 / 14)^2 / 16
        assertEquals(64 * 64 / 16, gemma.tokensFor(new ImageSize(896, 896)));

        ImageProcessorParams llava = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.MLP)), ImageTokenBudget.none());
        assertEquals(24 * 24, llava.tokensFor(new ImageSize(336, 336)));

        Map<String, Object> ldpMeta = metadata(ProjectorType.LDP);
        ldpMeta.put("clip.vision.patch_size", 14);
        ImageProcessorParams ldp = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(ldpMeta), ImageTokenBudget.none());
        // patches / 4: the LDP projectors carry no BOI/EOI rows
        assertEquals(24 * 24 / 4, ldp.tokensFor(new ImageSize(336, 336)));
        ImageProcessorParams glmEdge = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.GLM_EDGE)), ImageTokenBudget.none());
        // the GLM-Edge adapter adds its BOI and EOI rows
        assertEquals(24 * 24 / 4 + 2, glmEdge.tokensFor(new ImageSize(336, 336)));

        Map<String, Object> resamplerMeta = metadata(ProjectorType.MINICPMV,
                "clip.minicpmv_version", 4);
        ImageProcessorParams resampler = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(resamplerMeta), ImageTokenBudget.none());
        // the legacy version fallback: MiniCPM-V 4 resamplers answer 64 queries per tile
        assertEquals(64, resampler.minicpmvQueryNum());
        assertEquals(64, resampler.tokensFor(new ImageSize(336, 336)));

        Map<String, Object> resamplerV2 = metadata(ProjectorType.MINICPMV);
        ImageProcessorParams resamplerLegacy = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(resamplerV2), ImageTokenBudget.none());
        // version defaults to 2 -> 96 queries (clip.cpp:1424-1438)
        assertEquals(96, resamplerLegacy.minicpmvQueryNum());

        ImageProcessorParams yasa = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.YASA2)), ImageTokenBudget.none());
        assertEquals(64, yasa.tokensFor(new ImageSize(896, 896)));
    }

    /** The params of {@code type} over the default metadata plus {@code extra}. */
    private static ImageProcessorParams params(ProjectorType type, Object... extra) {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(type, extra)), ImageTokenBudget.none());
    }

    @Test
    public void eachTypeCountsTheReferenceTokensOfAConcreteTile() {
        // clip_n_output_tokens per type (llama.cpp tools/mtmd/clip.cpp), patch 14
        ImageProcessorParams pixtral = params(ProjectorType.PIXTRAL, "clip.vision.spatial_merge_size", 2);
        // 1540x812: 110x58 patches, 55x29 merged tokens and 28 [IMG_BREAK] rows
        assertEquals(55 * 29 + 28, pixtral.tokensFor(new ImageSize(1540, 812)));
        assertEquals(new ImageSize(55 * 29 + 28, 1), pixtral.tokenGridFor(new ImageSize(1540, 812)));

        // gemma3n: no merge, image_size / patch_size tokens whatever the tile
        ImageProcessorParams gemma3n = params(ProjectorType.GEMMA3NV);
        assertEquals(1, gemma3n.mergeSize());
        assertEquals(336 / 14, gemma3n.tokensFor(new ImageSize(336, 336)));

        // kimi: the merged grid with each side aligned up, ceil(434 / 28) * ceil(294 / 28)
        for (ProjectorType kimi : new ProjectorType[] {ProjectorType.KIMIVL, ProjectorType.KIMIK25}) {
            ImageProcessorParams p = params(kimi, "clip.vision.projector.scale_factor", 2);
            assertEquals(kimi.typeName(), 16 * 11, p.tokensFor(new ImageSize(434, 294)));
        }

        // paddleocr / dots_ocr: 42x56 is 3x4 = 12 patches -> 12 / 2^2 = 3 tokens
        ImageProcessorParams paddle = params(ProjectorType.PADDLEOCR, "clip.vision.image_min_pixels", 1024,
                "clip.vision.image_max_pixels", 4096 * 784);
        assertEquals(3, paddle.tokensFor(new ImageSize(42, 56)));
        // 9 patches align up to 12 -> 3 tokens
        assertEquals(3, paddle.tokensFor(new ImageSize(42, 42)));
        // paddleocr lays its tokens on the merged grid, dots.ocr on one row
        assertEquals(new ImageSize(16, 12), paddle.tokenGridFor(new ImageSize(448, 336)));
        ImageProcessorParams dots = params(ProjectorType.DOTS_OCR, "clip.vision.projector.scale_factor", 2,
                "clip.vision.image_min_pixels", 1024, "clip.vision.image_max_pixels", 4096 * 784);
        assertEquals(3, dots.tokensFor(new ImageSize(42, 56)));
        assertEquals(new ImageSize(16 * 12, 1), dots.tokenGridFor(new ImageSize(448, 336)));

        // hunyuanvl: 16x12 merged grid, a newline per row, begin and end rows
        ImageProcessorParams hunyuan = params(ProjectorType.HUNYUANVL);
        assertEquals(17 * 12 + 2, hunyuan.tokensFor(new ImageSize(448, 336)));
        assertEquals(new ImageSize(16, 12), hunyuan.tokenGridFor(new ImageSize(448, 336)));

        // nemotron v2: the projector scale factor, not spatial_merge_size: 24^2 patches / 2^2
        ImageProcessorParams nemotronV2 = params(ProjectorType.NEMOTRON_V2_VL,
                "clip.vision.projector.scale_factor", 2, "clip.vision.spatial_merge_size", 3);
        assertEquals(2, nemotronV2.mergeSize());
        assertEquals(24 * 24 / 4, nemotronV2.tokensFor(new ImageSize(336, 336)));
        assertEquals(1, params(ProjectorType.NEMOTRON_V2_VL, "clip.vision.spatial_merge_size", 2).mergeSize());

        // nemotron omni: 32x24 patches / 2^2
        ImageProcessorParams nemotronOmni = params(ProjectorType.NEMOTRON_OMNI,
                "clip.vision.projector.scale_factor", 2, "clip.vision.image_min_pixels", 1024,
                "clip.vision.image_max_pixels", 4096 * 784);
        assertEquals(32 * 24 / 4, nemotronOmni.tokensFor(new ImageSize(448, 336)));

        // idefics3: 24^2 patches / 3^2
        assertEquals(24 * 24 / 9, params(ProjectorType.IDEFICS3, "clip.vision.projector.scale_factor", 3)
                .tokensFor(new ImageSize(336, 336)));
    }

    @Test
    public void requiredScaleFactorsAreRefusedByName() {
        for (ProjectorType type : new ProjectorType[] {ProjectorType.COHERE2V, ProjectorType.NEMOTRON_OMNI,
                ProjectorType.DOTS_OCR}) {
            Map<String, Object> m = metadata(type, "clip.vision.image_min_pixels", 1024,
                    "clip.vision.image_max_pixels", 4096 * 784, "clip.vision.preproc_max_tiles", 4);
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none()));
            assertTrue(type.typeName() + ": " + e.getMessage(),
                    e.getMessage().contains("'clip.vision.projector.scale_factor'"));
        }
    }

    @Test
    public void unmultipledTilesAreRefused() {
        ImageProcessorParams qwen = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.QWEN2VL)), ImageTokenBudget.none());
        try {
            qwen.tokensFor(new ImageSize(783, 784));
            fail("expected a geometry refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("multiple"));
        }
    }

    @Test
    public void builderCopiesArraysInAndOut() {
        ImageProcessorParams base = ImageProcessorParams.fromGguf(
                GgufKv.ofMap(metadata(ProjectorType.MLP)), ImageTokenBudget.none());
        int[] color = {1, 2, 3};
        ImageProcessorParams p = base.toBuilder().padColor(color).build();
        color[0] = 99;
        assertEquals(1, p.padColor()[0]);
        p.padColor()[1] = 99;
        assertEquals(2, p.padColor()[1]);
        assertEquals(true, base.toString().contains("mlp"));
    }

    @Test
    public void invertedPixelLimitsAreRefused() {
        try {
            ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(ProjectorType.QWEN2VL)),
                    ImageTokenBudget.none()).toBuilder().minPixels(5000).maxPixels(1000).build();
            fail("expected an inverted limits refusal");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("minPixels"));
        }
    }
}

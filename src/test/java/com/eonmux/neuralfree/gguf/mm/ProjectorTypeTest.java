package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The projector type table against the reference name list (llama.cpp tools/mtmd/clip-impl.h). */
public class ProjectorTypeTest {

    /** The PROJECTOR_TYPE_NAMES values of clip-impl.h, in the reference declaration order. */
    private static final String[] REFERENCE_NAMES = {
            "mlp", "mlp_norm", "ldp", "ldpv2", "resampler", "adapter", "qwen2vl_merger", "qwen3vl_merger",
            "ling3vl", "step3vl", "gemma3", "gemma3nv", "gemma3na", "gemma4v", "gemma4a", "gemma4uv",
            "gemma4ua", "phi4", "idefics3", "pixtral", "qwen2.5vl_merger", "ultravox", "internvl", "llama4",
            "qwen2a", "qwen3a", "glma", "qwen2.5o", "voxtral", "voxtral_rt", "meralion", "musicflamingo",
            "lfm2", "lfm3vl", "d1omni_v", "kimivl", "paddleocr", "lightonocr", "cogvlm", "janus_pro",
            "dots_ocr", "dots3note_v", "dots3note_a", "deepseekocr", "deepseekocr2", "deepseek4v",
            "deepseek41v", "lfm2a", "d1omni_a", "lfm3a", "glm4v", "glm5v", "youtuvl", "yasa2", "kimik25",
            "nemotron_v2_vl", "nemotron_omni", "hunyuanvl", "parakeet", "exaone4_5", "minicpmv4_6",
            "minicpmv4_7", "minicpm_ocr_1_0", "granite_speech", "mimovl", "minimax_m3", "granite4_vision",
            "mimo_audio", "qwen3tts_spkenc", "qwen3tts_gen", "pockettts_spkenc", "pockettts_gen", "soprano",
            "lfm3a_gen", "muse-glimmer", "sensenova_u1", "cohere2v", "muse-spark-audio",
    };

    @Test
    public void theNameTableMatchesTheReference() {
        int i = 0;
        for (ProjectorType type : ProjectorType.named()) {
            assertEquals("position " + i, REFERENCE_NAMES[i], type.typeName());
            i++;
        }
        assertEquals(REFERENCE_NAMES.length, i);
    }

    @Test
    public void fromNameResolvesExactlyAndRefusesNothing() {
        assertEquals(ProjectorType.MLP, ProjectorType.fromName("mlp"));
        assertEquals(ProjectorType.QWEN25VL, ProjectorType.fromName("qwen2.5vl_merger"));
        assertEquals(ProjectorType.MUSE_GLIMMER, ProjectorType.fromName("muse-glimmer"));
        assertSame(ProjectorType.UNKNOWN, ProjectorType.fromName("made_up_projector"));
        assertSame(ProjectorType.UNKNOWN, ProjectorType.fromName("MLP"));
        assertNull(ProjectorType.UNKNOWN.typeName());
        assertThrows(IllegalArgumentException.class, () -> ProjectorType.fromName(null));
    }

    @Test
    public void thePreprocessedTypesAndTheirTokenFormulas() {
        // each type's clip_n_output_tokens arm in llama.cpp tools/mtmd/clip.cpp
        Map<String, ImageTokenFormula> expected = new LinkedHashMap<>();
        for (String n : new String[] {"mlp", "mlp_norm", "phi4", "janus_pro"}) {
            expected.put(n, ImageTokenFormula.PATCHES);
        }
        expected.put("cogvlm", ImageTokenFormula.PATCHES_BOI_EOI);
        expected.put("ldp", ImageTokenFormula.LDP);
        expected.put("ldpv2", ImageTokenFormula.LDP);
        expected.put("adapter", ImageTokenFormula.LDP_BOI_EOI);
        expected.put("resampler", ImageTokenFormula.RESAMPLER);
        expected.put("yasa2", ImageTokenFormula.POOLED_8X8);
        for (String n : new String[] {"gemma3", "gemma4v", "gemma4uv", "idefics3", "internvl", "llama4",
                "nemotron_v2_vl", "nemotron_omni", "cohere2v"}) {
            expected.put(n, ImageTokenFormula.PATCHES_OVER_MERGE);
        }
        expected.put("gemma3nv", ImageTokenFormula.IMAGE_SIZE_OVER_PATCH);
        expected.put("pixtral", ImageTokenFormula.MERGED_ROWS_WITH_BREAKS);
        expected.put("kimivl", ImageTokenFormula.ALIGNED_MERGED_GRID);
        expected.put("kimik25", ImageTokenFormula.ALIGNED_MERGED_GRID);
        expected.put("lfm2", ImageTokenFormula.ALIGNED_MERGED_GRID);
        expected.put("d1omni_v", ImageTokenFormula.ALIGNED_MERGED_GRID);
        expected.put("paddleocr", ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE);
        expected.put("dots_ocr", ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE);
        expected.put("dots3note_v", ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE);
        expected.put("deepseekocr", ImageTokenFormula.DEEPSEEK_OCR);
        expected.put("deepseekocr2", ImageTokenFormula.DEEPSEEK_OCR2);
        expected.put("deepseek4v", ImageTokenFormula.DEEPSEEK4V_BLOCK);
        expected.put("deepseek41v", ImageTokenFormula.DEEPSEEK41V_BLOCK);
        expected.put("granite4_vision", ImageTokenFormula.GRANITE4_WINDOWS);
        expected.put("hunyuanvl", ImageTokenFormula.MERGED_CONV_WITH_NEWLINES);
        for (String n : new String[] {"qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger", "ling3vl", "mimovl",
                "glm4v", "glm5v", "exaone4_5", "minicpmv4_6", "minicpmv4_7", "minicpm_ocr_1_0", "minimax_m3",
                "youtuvl", "muse-glimmer", "lfm3vl", "step3vl", "lightonocr", "sensenova_u1"}) {
            expected.put(n, ImageTokenFormula.MERGED_CONV);
        }
        Map<String, ImageTokenFormula> got = new LinkedHashMap<>();
        for (ProjectorType t : ProjectorType.preprocessed()) {
            got.put(t.typeName(), t.visionPreprocessing().tokenFormula());
            assertSame(t.typeName(), t.patchMerge(), t.visionPreprocessing().merge());
        }
        assertEquals(new TreeMap<>(expected), new TreeMap<>(got));
        assertNull(ProjectorType.ULTRAVOX.visionPreprocessing());
        assertNull(ProjectorType.UNKNOWN.visionPreprocessing());
    }

    @Test
    public void theMropeTypesLayTheirTokensOnTheMergedGrid() {
        // the clip_n_output_tokens_x/y arms: everything else is one row of all tokens (the DeepSeek-OCR arm only
        // matters under M-RoPE, which those models do not decode with)
        Set<String> merged = new TreeSet<>(Arrays.asList("qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger",
                "ling3vl", "mimovl", "glm4v", "glm5v", "exaone4_5", "paddleocr", "hunyuanvl", "youtuvl",
                "muse-glimmer", "lfm3vl", "step3vl", "minicpmv4_6", "minicpmv4_7", "minicpm_ocr_1_0",
                "sensenova_u1"));
        Set<String> got = new TreeSet<>();
        for (ProjectorType t : ProjectorType.preprocessed()) {
            if (t.visionPreprocessing().tokenGrid() == ImageTokenGrid.MERGED) {
                got.add(t.typeName());
            }
        }
        assertEquals(merged, got);
    }

    @Test
    public void mergeRulesFollowTheConverterKeys() {
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.QWEN2VL.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(1), ProjectorType.PIXTRAL.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(4), ProjectorType.GEMMA3.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.LLAMA4.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.IDEFICS3.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(4), ProjectorType.MINICPMV4_6.patchMerge());
        assertEquals(PatchMergeRule.required("clip.vision.projector.scale_factor"),
                ProjectorType.DOTS_OCR.patchMerge());
        assertEquals(PatchMergeRule.fixed(2), ProjectorType.PADDLEOCR.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.QWEN3VL.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.HUNYUANVL.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(1), ProjectorType.LIGHTONOCR.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.KIMIVL.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.LFM2.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.D1OMNI_V.patchMerge());
        assertEquals(PatchMergeRule.required("clip.vision.projector.scale_factor"),
                ProjectorType.COHERE2V.patchMerge());
        assertEquals(PatchMergeRule.required("clip.vision.spatial_merge_size"),
                ProjectorType.DOTS3NOTE_V.patchMerge());
        assertEquals(PatchMergeRule.fixed(1), ProjectorType.JANUS_PRO.patchMerge());
        // clip.cpp load_hparams: step3vl's two stride-2 downsamplers, gemma4's 3x3 pooling and the MiniCPM-OCR
        // merger default the scale factor; lfm3vl requires the spatial merge, the others fix it at 2
        assertEquals(PatchMergeRule.projectorScaleFactor(4), ProjectorType.STEP3VL.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(3), ProjectorType.GEMMA4V.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(3), ProjectorType.GEMMA4UV.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(4), ProjectorType.MINICPM_OCR_1_0.patchMerge());
        assertEquals(PatchMergeRule.required("clip.vision.spatial_merge_size"), ProjectorType.LFM3VL.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.GLM5V.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.MINIMAX_M3.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.MUSE_GLIMMER.patchMerge());
        // clip.cpp load_hparams: gemma3n merges nothing unless the scale factor says so; nemotron v2 reads the
        // optional scale factor; nemotron omni requires it
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.GEMMA3NV.patchMerge());
        assertEquals(PatchMergeRule.projectorScaleFactor(1), ProjectorType.NEMOTRON_V2_VL.patchMerge());
        assertEquals(PatchMergeRule.required("clip.vision.projector.scale_factor"),
                ProjectorType.NEMOTRON_OMNI.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.GLM4V.patchMerge());
        assertEquals(PatchMergeRule.spatialMergeSize(2), ProjectorType.YOUTUVL.patchMerge());
        for (ProjectorType t : new ProjectorType[] {ProjectorType.MLP, ProjectorType.MLP_NORM, ProjectorType.PHI4,
                ProjectorType.MINICPMV, ProjectorType.ULTRAVOX, ProjectorType.LDP,
                ProjectorType.LDPV2, ProjectorType.GLM_EDGE, ProjectorType.COGVLM, ProjectorType.GRANITE4_VISION}) {
            assertEquals(t.typeName(), PatchMergeRule.fixed(1), t.patchMerge());
        }
    }

    @Test
    public void theDeclaredNamePrefersTheCommonKey() {
        Map<String, Object> m = new HashMap<>();
        m.put("clip.vision.projector_type", "gemma3");
        assertEquals("gemma3", ProjectorType.declaredName(GgufKv.ofMap(m), ProjectorModality.VISION));
        m.put("clip.projector_type", "pixtral");
        assertEquals("pixtral", ProjectorType.declaredName(GgufKv.ofMap(m), ProjectorModality.VISION));
        m.put("clip.projector_type", "qwen2.5o");
        assertEquals("qwen2.5vl_merger", ProjectorType.declaredName(GgufKv.ofMap(m), ProjectorModality.VISION));
        assertEquals("qwen2a", ProjectorType.declaredName(GgufKv.ofMap(m), ProjectorModality.AUDIO));
        m.put("clip.projector_type", "plugin_family");
        assertEquals("plugin_family", ProjectorType.declaredName(GgufKv.ofMap(m), ProjectorModality.VISION));
        assertTrue(assertThrows(GgufFormatException.class,
                () -> ProjectorType.declaredName(GgufKv.ofMap(new HashMap<>()), ProjectorModality.VISION))
                .getMessage().contains("clip.projector_type"));
    }

    @Test
    public void ffnOpKeysResolveAsClipCppDoes() {
        assertEquals(ClipFfnOp.GELU, ClipFfnOp.fromUseGeluSilu(true, false));
        assertEquals(ClipFfnOp.SILU, ClipFfnOp.fromUseGeluSilu(false, true));
        assertEquals(ClipFfnOp.GELU_QUICK, ClipFfnOp.fromUseGeluSilu(false, false));
        assertThrows(IllegalArgumentException.class, () -> ClipFfnOp.fromUseGeluSilu(true, true));
        assertEquals(ActOp.GELU_TANH, ClipFfnOp.GELU.unary());
        assertEquals(GluOp.GEGLU_TANH, ClipFfnOp.GELU.glu());
        assertEquals(GluOp.SWIGLU, ClipFfnOp.SILU.glu());
        assertEquals(ActOp.RELU_SQR, ClipFfnOp.RELU_SQR.unary());
        assertThrows(UnsupportedOperationException.class, ClipFfnOp.RELU_SQR::glu);
    }
}

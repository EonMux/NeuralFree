package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing.Family;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static com.eonmux.neuralfree.gguf.mm.PatchMergeRule.fixed;
import static com.eonmux.neuralfree.gguf.mm.PatchMergeRule.projectorScaleFactor;
import static com.eonmux.neuralfree.gguf.mm.PatchMergeRule.spatialMergeSize;

/**
 * The projector type of an mmproj file, from its {@code clip.projector_type} metadata (llama.cpp
 * {@code projector_type}, tools/mtmd/clip-impl.h). The name table is the exact
 * {@code PROJECTOR_TYPE_NAMES} map of the reference: a name matches case-sensitively, everything else is
 * {@link #UNKNOWN}.
 *
 * <p>Each type carries what the vision encoder and the image preprocessing share about it: the
 * {@link PatchMergeRule} its per-side patch merge is read with, and, for the types NeuralFree preprocesses, the
 * {@link VisionPreprocessing} (preprocessing family, image token formula and grid). The types NeuralFree neither
 * preprocesses nor encodes declare a fixed merge of 1.
 */
public enum ProjectorType {

    MLP("mlp", llava(ImageTokenFormula.PATCHES)),
    MLP_NORM("mlp_norm", llava(ImageTokenFormula.PATCHES)),
    LDP("ldp", llava(ImageTokenFormula.LDP)),
    LDPV2("ldpv2", llava(ImageTokenFormula.LDP)),
    MINICPMV("resampler", vision(Family.MINICPMV, ImageTokenFormula.RESAMPLER, fixed(1))),
    GLM_EDGE("adapter", vision(Family.GLM_EDGE, ImageTokenFormula.LDP_BOI_EOI, fixed(1))),
    QWEN2VL("qwen2vl_merger", qwenMerger()),
    QWEN3VL("qwen3vl_merger", qwenMerger()),
    LING3VL("ling3vl", qwenMerger()),
    STEP3VL("step3vl", VisionPreprocessing.of(Family.STEP3VL, ImageTokenFormula.MERGED_CONV, ImageTokenGrid.MERGED,
            projectorScaleFactor(4))),
    GEMMA3("gemma3", gemma3()),
    GEMMA3NV("gemma3nv", vision(Family.GEMMA3, ImageTokenFormula.IMAGE_SIZE_OVER_PATCH, projectorScaleFactor(1))),
    GEMMA3NA("gemma3na"),
    GEMMA4V("gemma4v", vision(Family.GEMMA4V, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(3))),
    GEMMA4A("gemma4a"),
    GEMMA4UV("gemma4uv", vision(Family.GEMMA4UV, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(3))),
    GEMMA4UA("gemma4ua"),
    PHI4("phi4", vision(Family.PHI4, ImageTokenFormula.PATCHES, fixed(1))),
    IDEFICS3("idefics3", vision(Family.IDEFICS3, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(1))),
    PIXTRAL("pixtral", vision(Family.PIXTRAL, ImageTokenFormula.MERGED_ROWS_WITH_BREAKS, spatialMergeSize(1))),
    QWEN25VL("qwen2.5vl_merger", qwenMerger()),
    ULTRAVOX("ultravox"),
    INTERNVL("internvl", vision(Family.INTERNVL, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(1))),
    LLAMA4("llama4", vision(Family.LLAMA4, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(1))),
    QWEN2A("qwen2a"),
    QWEN3A("qwen3a"),
    GLMA("glma"),
    QWEN25O("qwen2.5o"),
    VOXTRAL("voxtral"),
    VOXTRAL_RT("voxtral_rt"),
    MERALION("meralion"),
    MUSIC_FLAMINGO("musicflamingo"),
    LFM2("lfm2", lfm2()),
    LFM3VL("lfm3vl", VisionPreprocessing.of(Family.LFM3VL, ImageTokenFormula.MERGED_CONV, ImageTokenGrid.MERGED,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE))),
    D1OMNI_V("d1omni_v", lfm2()),
    KIMIVL("kimivl", kimi()),
    PADDLEOCR("paddleocr", VisionPreprocessing.of(Family.PADDLEOCR, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE,
            ImageTokenGrid.MERGED, fixed(2))),
    LIGHTONOCR("lightonocr", VisionPreprocessing.of(Family.LIGHTONOCR, ImageTokenFormula.MERGED_CONV,
            ImageTokenGrid.ROW, spatialMergeSize(1))),
    COGVLM("cogvlm", llava(ImageTokenFormula.PATCHES_BOI_EOI)),
    JANUS_PRO("janus_pro", vision(Family.JANUS_PRO, ImageTokenFormula.PATCHES, fixed(1))),
    DOTS_OCR("dots_ocr", vision(Family.DOTS_OCR, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR))),
    DOTS3NOTE_V("dots3note_v", vision(Family.DOTS3NOTE_V, ImageTokenFormula.ALIGNED_PATCHES_OVER_MERGE,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE))),
    DOTS3NOTE_A("dots3note_a"),
    DEEPSEEKOCR("deepseekocr", vision(Family.DEEPSEEK_OCR, ImageTokenFormula.DEEPSEEK_OCR, fixed(1))),
    DEEPSEEKOCR2("deepseekocr2", vision(Family.DEEPSEEK_OCR2, ImageTokenFormula.DEEPSEEK_OCR2, fixed(1))),
    DEEPSEEK4V("deepseek4v", vision(Family.DEEPSEEK4V, ImageTokenFormula.DEEPSEEK4V_BLOCK,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR))),
    DEEPSEEK41V("deepseek41v", vision(Family.DEEPSEEK41V, ImageTokenFormula.DEEPSEEK41V_BLOCK,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR))),
    LFM2A("lfm2a"),
    D1OMNI_A("d1omni_a"),
    LFM3A("lfm3a"),
    GLM4V("glm4v", vision(Family.GLM4V, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    GLM5V("glm5v", vision(Family.GLM5V, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    YOUTUVL("youtuvl", vision(Family.YOUTUVL, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    YASA2("yasa2", vision(Family.YASA2, ImageTokenFormula.POOLED_8X8, fixed(1))),
    KIMIK25("kimik25", kimi()),
    NEMOTRON_V2_VL("nemotron_v2_vl", vision(Family.NEMOTRON_V2_VL, ImageTokenFormula.PATCHES_OVER_MERGE,
            projectorScaleFactor(1))),
    NEMOTRON_OMNI("nemotron_omni", vision(Family.NEMOTRON_OMNI, ImageTokenFormula.PATCHES_OVER_MERGE,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR))),
    HUNYUANVL("hunyuanvl", vision(Family.HUNYUANVL, ImageTokenFormula.MERGED_CONV_WITH_NEWLINES,
            spatialMergeSize(2))),
    PARAKEET("parakeet"),
    EXAONE4_5("exaone4_5", vision(Family.EXAONE4_5, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    MINICPMV4_6("minicpmv4_6", minicpmvMerger()),
    MINICPMV4_7("minicpmv4_7", minicpmvMerger()),
    MINICPM_OCR_1_0("minicpm_ocr_1_0", vision(Family.MINICPM_OCR, ImageTokenFormula.MERGED_CONV,
            projectorScaleFactor(4))),
    GRANITE_SPEECH("granite_speech"),
    MIMOVL("mimovl", vision(Family.MIMOVL, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    MINIMAX_M3("minimax_m3", VisionPreprocessing.of(Family.MINIMAX_M3, ImageTokenFormula.MERGED_CONV,
            ImageTokenGrid.ROW, spatialMergeSize(2))),
    GRANITE4_VISION("granite4_vision", vision(Family.GRANITE4V, ImageTokenFormula.GRANITE4_WINDOWS, fixed(1))),
    MIMO_AUDIO("mimo_audio"),
    QWEN3TTS_SPKENC("qwen3tts_spkenc"),
    QWEN3TTS_GEN("qwen3tts_gen"),
    POCKETTTS_SPKENC("pockettts_spkenc"),
    POCKETTTS_GEN("pockettts_gen"),
    SOPRANO("soprano"),
    LFM3A_GEN("lfm3a_gen"),
    MUSE_GLIMMER("muse-glimmer", vision(Family.MUSE_GLIMMER, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    SENSENOVA_U1("sensenova_u1", vision(Family.SENSENOVA_U1, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2))),
    COHERE2V("cohere2v", vision(Family.COHERE2V, ImageTokenFormula.PATCHES_OVER_MERGE,
            PatchMergeRule.required(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR))),
    MUSE_SPARK_AUDIO("muse-spark-audio"),
    UNKNOWN(null);

    private static final Map<String, ProjectorType> BY_NAME;
    static {
        Map<String, ProjectorType> m = new HashMap<>();
        for (ProjectorType t : values()) {
            if (t.ggufName != null) {
                m.put(t.ggufName, t);
            }
        }
        BY_NAME = Collections.unmodifiableMap(m);
    }

    private final String ggufName;
    private final PatchMergeRule merge;
    private final VisionPreprocessing preprocessing;

    ProjectorType(String name) {
        this(name, fixed(1));
    }

    ProjectorType(String name, PatchMergeRule merge) {
        this.ggufName = name;
        this.merge = merge;
        this.preprocessing = null;
    }

    ProjectorType(String name, VisionPreprocessing preprocessing) {
        this.ggufName = name;
        this.merge = preprocessing.merge();
        this.preprocessing = preprocessing;
    }

    private static VisionPreprocessing vision(Family family, ImageTokenFormula formula, PatchMergeRule merge) {
        return VisionPreprocessing.of(family, formula, merge);
    }

    /** The LLaVA family never merges patches. */
    private static VisionPreprocessing llava(ImageTokenFormula formula) {
        return vision(Family.LLAVA, formula, fixed(1));
    }

    /** The Qwen mergers merge 2x2 by default, {@code clip.vision.spatial_merge_size} otherwise. */
    private static VisionPreprocessing qwenMerger() {
        return vision(Family.QWEN_MERGER, ImageTokenFormula.MERGED_CONV, spatialMergeSize(2));
    }

    /** Gemma 3 reduces every side by 4 unless {@code clip.vision.projector.scale_factor} says otherwise. */
    private static VisionPreprocessing gemma3() {
        return vision(Family.GEMMA3, ImageTokenFormula.PATCHES_OVER_MERGE, projectorScaleFactor(4));
    }

    /** LFM2-VL and d1-omni count the merged grid with each side aligned up, on one row. */
    private static VisionPreprocessing lfm2() {
        return VisionPreprocessing.of(Family.LFM2, ImageTokenFormula.ALIGNED_MERGED_GRID, ImageTokenGrid.ROW,
                projectorScaleFactor(1));
    }

    /** Kimi-VL and Kimi K2.5 count the merged grid with each side aligned up. */
    private static VisionPreprocessing kimi() {
        return vision(Family.KIMIVL, ImageTokenFormula.ALIGNED_MERGED_GRID, projectorScaleFactor(1));
    }

    /** The MiniCPM-V 4.6/4.7 ViT and final mergers: 4x per side by default. */
    private static VisionPreprocessing minicpmvMerger() {
        return vision(Family.MINICPMV_MERGER, ImageTokenFormula.MERGED_CONV, projectorScaleFactor(4));
    }

    /** The {@code clip.projector_type} value, or null for {@link #UNKNOWN}. */
    public String typeName() {
        return ggufName;
    }

    /** How the vision encoder and the image preprocessing read the per-side patch merge of this type. */
    public PatchMergeRule patchMerge() {
        return merge;
    }

    /** The image preprocessing of this type, or null when NeuralFree does not preprocess images for it. */
    public VisionPreprocessing visionPreprocessing() {
        return preprocessing;
    }

    /**
     * The type an encoder of {@code modality} runs as: the legacy mixed-modality {@link #QWEN25O} resolves to
     * {@link #QWEN25VL} for vision and {@link #QWEN2A} otherwise (clip.cpp load_hparams); every other type is
     * itself.
     */
    public ProjectorType forModality(ProjectorModality modality) {
        Objects.requireNonNull(modality, "modality");
        if (this == QWEN25O) {
            return modality == ProjectorModality.VISION ? QWEN25VL : QWEN2A;
        }
        return this;
    }

    /**
     * The projector type name a file declares for its encoder of {@code modality}: {@code clip.projector_type},
     * or {@code clip.<modality>.projector_type} when the common key is absent (mixed-modality files), with the
     * legacy {@code qwen2.5o} resolved per modality ({@link #forModality}). The name may be one the reference does
     * not know (a plugin row's key).
     *
     * @throws GgufFormatException when neither key is present
     */
    public static String declaredName(GgufKv kv, ProjectorModality modality) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(modality, "modality");
        String raw = kv.str(GgufKeys.CLIP_PROJECTOR_TYPE, null);
        if (raw == null) {
            raw = kv.str("clip." + modality.keyPrefix() + ".projector_type", null);
        }
        if (raw == null) {
            throw new GgufFormatException("required metadata key '" + GgufKeys.CLIP_PROJECTOR_TYPE + "' is missing");
        }
        ProjectorType type = fromName(raw);
        return type == UNKNOWN ? raw : type.forModality(modality).typeName();
    }

    /** The values whose {@link #typeName()} is not null, in declaration order (unmodifiable). */
    public static Iterable<ProjectorType> named() {
        return Collections.unmodifiableList(Arrays.asList(values()).subList(0, values().length - 1));
    }

    /** The named types that carry a {@link #visionPreprocessing()}, in declaration order (unmodifiable). */
    public static List<ProjectorType> preprocessed() {
        List<ProjectorType> out = new ArrayList<>();
        for (ProjectorType t : named()) {
            if (t.preprocessing != null) {
                out.add(t);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The type of a {@code clip.projector_type} value; unknown names resolve to {@link #UNKNOWN}, like
     * {@code clip_projector_type_from_string}.
     *
     * @throws IllegalArgumentException for a null name
     */
    public static ProjectorType fromName(String name) {
        if (name == null) {
            throw new IllegalArgumentException("projector type name is null");
        }
        ProjectorType t = BY_NAME.get(name);
        return t != null ? t : UNKNOWN;
    }

    @Override
    public String toString() {
        return ggufName != null ? ggufName : String.format(Locale.ROOT, "UNKNOWN(%s)", super.toString());
    }
}

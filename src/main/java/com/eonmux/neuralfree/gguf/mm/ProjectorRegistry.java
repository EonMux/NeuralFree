package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.plugin.Plugins;
import com.eonmux.neuralfree.gguf.plugin.ProjectorProvider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Registry of every projector family, keyed by the {@code clip.projector_type} name: the merge of the built-in
 * rows of this package and the rows operator-installed plugin jars contribute through
 * {@link Plugins#load(ProjectorProvider.class)}, built the {@code ArchRegistry} way. A value resolves to its
 * {@link ProjectorEntry} on first use; a value without a row, and a row that only records a rejection, refuse
 * the file with that reason.
 *
 * <p>Plugin rows carry their own keys, so a jar can register a family the reference does not name yet (for
 * example under the {@code architectures[0]} of a Hugging Face projector checkpoint); a key the built-in table
 * or an earlier provider already registers is rejected with both sources named: plugins never replace rows.
 *
 * <p>{@link #visionPreprocessing} resolves a name to the {@link VisionPreprocessing} its images are prepared and
 * counted with: the built-in type's own, or the one a plugin row declares for its key.
 *
 * <p>Built once on first use.
 */
public final class ProjectorRegistry {

    private ProjectorRegistry() {
    }

    private static final class Holder {
        static final Map<String, ProjectorEntry> ENTRIES = withProviderRows();

        private static Map<String, ProjectorEntry> withProviderRows() {
            return mergeProviders(builtIn(), providers());
        }
    }

    @SuppressWarnings("unchecked")
    private static List<ProjectorProvider<ProjectorEntry>> providers() {
        // the SPI is generic; the registry's row type is ProjectorEntry (see ProjectorProvider)
        return (List<ProjectorProvider<ProjectorEntry>>) (List<?>) Plugins.load(ProjectorProvider.class);
    }

    /**
     * The entry of a {@code clip.projector_type} value; unknown names are refused.
     *
     * @throws UnsupportedProjectorException when no table registers the name as runnable
     */
    public static ProjectorEntry get(String projectorTypeName) {
        ProjectorEntry e = Holder.ENTRIES.get(projectorTypeName);
        if (e == null) {
            throw new UnsupportedProjectorException(projectorTypeName, "projector type '" + projectorTypeName
                    + "' has no entry in the projector registry (" + Arch.CLIP.ggufName() + " file); a plugin jar"
                    + " can register it (ProjectorProvider)");
        }
        if (!e.runnable()) {
            throw new UnsupportedProjectorException(projectorTypeName,
                    "projector type '" + projectorTypeName + "' is registered as not runnable: " + e.rejection());
        }
        return e;
    }

    /**
     * The entry the encoder of {@code modality} runs as under a {@code clip.projector_type} value: the legacy
     * mixed-modality {@code qwen2.5o} resolves to {@code qwen2.5vl_merger} for vision and {@code qwen2a} for audio
     * ({@link ProjectorType#forModality}, clip.cpp {@code load_hparams}); every other name is {@link #get(String)}.
     *
     * @throws UnsupportedProjectorException when no table registers the resolved name as runnable, or registers it
     *                                       for another modality
     */
    public static ProjectorEntry get(String projectorTypeName, ProjectorModality modality) {
        Objects.requireNonNull(projectorTypeName, "projectorTypeName");
        Objects.requireNonNull(modality, "modality");
        ProjectorType type = ProjectorType.fromName(projectorTypeName);
        String resolved = type == ProjectorType.UNKNOWN ? projectorTypeName : type.forModality(modality).typeName();
        ProjectorEntry e = get(resolved);
        if (e.modality() != modality) {
            throw new UnsupportedProjectorException(projectorTypeName, "projector type '" + projectorTypeName
                    + "' is registered for the " + e.modality() + " encoder, not the " + modality + " one");
        }
        return e;
    }

    /**
     * The image preprocessing of a {@code clip.projector_type} value: the built-in type's own
     * {@link ProjectorType#visionPreprocessing()}, or the one the plugin row filed under the name declares. A
     * built-in type is preprocessed whether or not NeuralFree runs its encoder.
     *
     * @throws UnsupportedProjectorException naming the type when neither declares image preprocessing
     */
    public static VisionPreprocessing visionPreprocessing(String projectorTypeName) {
        return visionPreprocessing(projectorTypeName, Holder.ENTRIES);
    }

    /**
     * {@link #visionPreprocessing(String)} over the given rows. Public for the plugin tests, which resolve names
     * against the rows {@link #mergeProviders} builds.
     *
     * @throws UnsupportedProjectorException naming the type when neither the type nor a row declares image
     *                                       preprocessing
     */
    public static VisionPreprocessing visionPreprocessing(String projectorTypeName,
                                                          Map<String, ProjectorEntry> rows) {
        Objects.requireNonNull(projectorTypeName, "projectorTypeName");
        Objects.requireNonNull(rows, "rows");
        ProjectorType type = ProjectorType.fromName(projectorTypeName);
        if (type != ProjectorType.UNKNOWN) {
            if (type.visionPreprocessing() == null) {
                throw new UnsupportedProjectorException(projectorTypeName, "projector type '" + projectorTypeName
                        + "' has no image preprocessing in NeuralFree; the preprocessed vision families are "
                        + preprocessedNames());
            }
            return type.visionPreprocessing();
        }
        ProjectorEntry row = rows.get(projectorTypeName);
        if (row == null) {
            throw new UnsupportedProjectorException(projectorTypeName, "projector type '" + projectorTypeName
                    + "' is unknown and has no entry in the projector registry; a plugin jar can register it with"
                    + " its image preprocessing (ProjectorProvider)");
        }
        if (row.visionPreprocessing() == null) {
            throw new UnsupportedProjectorException(projectorTypeName, "projector type '" + projectorTypeName
                    + "' is registered without image preprocessing (" + row + ")");
        }
        return row.visionPreprocessing();
    }

    private static String preprocessedNames() {
        StringBuilder sb = new StringBuilder();
        for (ProjectorType t : ProjectorType.preprocessed()) {
            sb.append(sb.length() == 0 ? "" : ", ").append(t.typeName());
        }
        return sb.toString();
    }

    /** True when a table registers the name as runnable. */
    public static boolean has(String projectorTypeName) {
        ProjectorEntry e = Holder.ENTRIES.get(projectorTypeName);
        return e != null && e.runnable();
    }

    /** The rows of the registry in registration order (unmodifiable). */
    public static List<ProjectorEntry> entries() {
        return Collections.unmodifiableList(new ArrayList<>(Holder.ENTRIES.values()));
    }

    /** The built-in rows: the implemented families, plus a rejection row for every other known type. */
    static Map<String, ProjectorEntry> builtIn() {
        Map<String, ProjectorEntry> m = new LinkedHashMap<>();
        runnable(m, ProjectorType.MLP, ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        runnable(m, ProjectorType.MLP_NORM, ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP_NORM);
        runnable(m, ProjectorType.GEMMA3, ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_GEMMA3);
        runnable(m, ProjectorType.QWEN2VL, ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        runnable(m, ProjectorType.QWEN25VL, ProjectorEntry.ENCODER_QWEN2VL_VIT,
                ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        runnable(m, ProjectorType.MINICPMV, ProjectorEntry.ENCODER_MINICPMV_VIT,
                ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER);
        runnable(m, ProjectorType.INTERNVL, ProjectorEntry.ENCODER_INTERNVL_VIT, ProjectorEntry.PROJECTOR_INTERNVL);
        runnable(m, ProjectorType.PIXTRAL, ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.PROJECTOR_PIXTRAL);
        runnable(m, ProjectorType.IDEFICS3, ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_IDEFICS3);
        runnable(m, ProjectorType.LLAMA4, ProjectorEntry.ENCODER_LLAMA4_VIT, ProjectorEntry.PROJECTOR_LLAMA4);
        runnable(m, ProjectorType.PHI4, ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_PHI4);
        runnable(m, ProjectorType.QWEN3VL, ProjectorEntry.ENCODER_QWEN3VL_VIT, ProjectorEntry.PROJECTOR_QWEN3VL_MERGER);
        runnable(m, ProjectorType.PADDLEOCR, ProjectorEntry.ENCODER_PADDLEOCR_VIT, ProjectorEntry.PROJECTOR_PADDLEOCR);
        runnable(m, ProjectorType.KIMIVL, ProjectorEntry.ENCODER_KIMIVL_VIT, ProjectorEntry.PROJECTOR_KIMIVL);
        runnable(m, ProjectorType.KIMIK25, ProjectorEntry.ENCODER_KIMIK25_VIT, ProjectorEntry.PROJECTOR_KIMIK25);
        runnable(m, ProjectorType.DOTS_OCR, ProjectorEntry.ENCODER_DOTS_VIT, ProjectorEntry.PROJECTOR_DOTS);
        runnable(m, ProjectorType.DOTS3NOTE_V, ProjectorEntry.ENCODER_DOTS_VIT, ProjectorEntry.PROJECTOR_DOTS);
        runnable(m, ProjectorType.HUNYUANVL, ProjectorEntry.ENCODER_HUNYUANVL_VIT, ProjectorEntry.PROJECTOR_HUNYUANVL);
        runnable(m, ProjectorType.JANUS_PRO, ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_JANUS);
        runnable(m, ProjectorType.LFM2, ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_LFM2);
        runnable(m, ProjectorType.D1OMNI_V, ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_LFM2);
        runnable(m, ProjectorType.COHERE2V, ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_COHERE2V);
        runnable(m, ProjectorType.LIGHTONOCR, ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.PROJECTOR_LIGHTONOCR);
        runnable(m, ProjectorType.LDP, ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LDP);
        runnable(m, ProjectorType.LDPV2, ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LDPV2);
        runnable(m, ProjectorType.GLM_EDGE, ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_GLM_EDGE);
        runnable(m, ProjectorType.COGVLM, ProjectorEntry.ENCODER_COGVLM_VIT, ProjectorEntry.PROJECTOR_COGVLM);
        runnable(m, ProjectorType.GLM4V, ProjectorEntry.ENCODER_GLM4V_VIT, ProjectorEntry.PROJECTOR_GLM4V);
        runnable(m, ProjectorType.YOUTUVL, ProjectorEntry.ENCODER_YOUTUVL_VIT, ProjectorEntry.PROJECTOR_YOUTUVL);
        runnable(m, ProjectorType.GRANITE4_VISION, ProjectorEntry.ENCODER_GRANITE4_VIT,
                ProjectorEntry.PROJECTOR_GRANITE4);
        runnable(m, ProjectorType.YASA2, ProjectorEntry.ENCODER_YASA2_CONVNEXT, ProjectorEntry.PROJECTOR_YASA2);
        runnable(m, ProjectorType.GLM5V, ProjectorEntry.ENCODER_GLM4V_VIT, ProjectorEntry.PROJECTOR_GLM4V);
        runnable(m, ProjectorType.LFM3VL, ProjectorEntry.ENCODER_LFM3VL_VIT, ProjectorEntry.PROJECTOR_LFM3VL);
        runnable(m, ProjectorType.LING3VL, ProjectorEntry.ENCODER_QWEN3VL_VIT, ProjectorEntry.PROJECTOR_LING3VL);
        runnable(m, ProjectorType.MINIMAX_M3, ProjectorEntry.ENCODER_MINIMAX_M3_VIT,
                ProjectorEntry.PROJECTOR_MINIMAX_M3);
        runnable(m, ProjectorType.MUSE_GLIMMER, ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT,
                ProjectorEntry.PROJECTOR_MUSE_GLIMMER);
        runnable(m, ProjectorType.NEMOTRON_V2_VL, ProjectorEntry.ENCODER_NEMOTRON_VIT,
                ProjectorEntry.PROJECTOR_NEMOTRON);
        runnable(m, ProjectorType.NEMOTRON_OMNI, ProjectorEntry.ENCODER_NEMOTRON_VIT,
                ProjectorEntry.PROJECTOR_NEMOTRON);
        runnable(m, ProjectorType.EXAONE4_5, ProjectorEntry.ENCODER_QWEN2VL_VIT,
                ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        runnable(m, ProjectorType.SENSENOVA_U1, ProjectorEntry.ENCODER_SENSENOVA_PATCH,
                ProjectorEntry.PROJECTOR_SENSENOVA);
        runnable(m, ProjectorType.STEP3VL, ProjectorEntry.ENCODER_STEP3VL_VIT, ProjectorEntry.PROJECTOR_STEP3VL);
        runnable(m, ProjectorType.DEEPSEEK4V, ProjectorEntry.ENCODER_DEEPSEEK4V_VIT,
                ProjectorEntry.PROJECTOR_DEEPSEEK4V);
        runnable(m, ProjectorType.DEEPSEEK41V, ProjectorEntry.ENCODER_DEEPSEEK4V_VIT,
                ProjectorEntry.PROJECTOR_DEEPSEEK41V);
        runnable(m, ProjectorType.DEEPSEEKOCR, ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP,
                ProjectorEntry.PROJECTOR_DEEPSEEKOCR);
        runnable(m, ProjectorType.DEEPSEEKOCR2, ProjectorEntry.ENCODER_DEEPSEEKOCR2_SAM_QWEN2,
                ProjectorEntry.PROJECTOR_DEEPSEEKOCR2);
        runnable(m, ProjectorType.GEMMA4V, ProjectorEntry.ENCODER_GEMMA4V_VIT, ProjectorEntry.PROJECTOR_GEMMA4V);
        runnable(m, ProjectorType.GEMMA4UV, ProjectorEntry.ENCODER_GEMMA4UV, ProjectorEntry.PROJECTOR_GEMMA4UV);
        for (ProjectorType t : new ProjectorType[]{ProjectorType.MINICPMV4_6, ProjectorType.MINICPMV4_7,
                ProjectorType.MINICPM_OCR_1_0}) {
            runnable(m, t, ProjectorEntry.ENCODER_MINICPMV46_VIT, ProjectorEntry.PROJECTOR_MINICPMV46_MERGER);
        }
        runnable(m, ProjectorType.MIMOVL, ProjectorEntry.ENCODER_MIMOVL_VIT, ProjectorEntry.PROJECTOR_MIMOVL_MERGER);
        runnable(m, ProjectorType.GEMMA3NV, ProjectorEntry.ENCODER_MOBILENETV5, ProjectorEntry.PROJECTOR_GEMMA3NV);
        audio(m, ProjectorType.ULTRAVOX, ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_ULTRAVOX);
        audio(m, ProjectorType.QWEN2A, ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_QWEN2A);
        audio(m, ProjectorType.VOXTRAL, ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_VOXTRAL);
        audio(m, ProjectorType.MUSIC_FLAMINGO, ProjectorEntry.ENCODER_WHISPER,
                ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO);
        audio(m, ProjectorType.MERALION, ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_MERALION);
        audio(m, ProjectorType.GLMA, ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_GLMA);
        audio(m, ProjectorType.QWEN3A, ProjectorEntry.ENCODER_QWEN3A, ProjectorEntry.PROJECTOR_QWEN3A);
        audio(m, ProjectorType.LFM2A, ProjectorEntry.ENCODER_CONFORMER, ProjectorEntry.PROJECTOR_CONFORMER);
        audio(m, ProjectorType.D1OMNI_A, ProjectorEntry.ENCODER_CONFORMER, ProjectorEntry.PROJECTOR_CONFORMER);
        audio(m, ProjectorType.LFM3A, ProjectorEntry.ENCODER_LFM3A, ProjectorEntry.PROJECTOR_LFM3A);
        audio(m, ProjectorType.GEMMA4A, ProjectorEntry.ENCODER_GEMMA4A, ProjectorEntry.PROJECTOR_GEMMA4A);
        audio(m, ProjectorType.GEMMA4UA, ProjectorEntry.ENCODER_GEMMA4UA, ProjectorEntry.PROJECTOR_GEMMA4UA);
        audio(m, ProjectorType.PARAKEET, ProjectorEntry.ENCODER_PARAKEET, ProjectorEntry.PROJECTOR_PARAKEET);
        audio(m, ProjectorType.GRANITE_SPEECH, ProjectorEntry.ENCODER_GRANITE_SPEECH,
                ProjectorEntry.PROJECTOR_GRANITE_SPEECH);
        audio(m, ProjectorType.VOXTRAL_RT, ProjectorEntry.ENCODER_VOXTRAL_RT, ProjectorEntry.PROJECTOR_VOXTRAL_RT);
        audio(m, ProjectorType.MIMO_AUDIO, ProjectorEntry.ENCODER_MIMO_AUDIO, ProjectorEntry.PROJECTOR_MIMO_AUDIO);
        audio(m, ProjectorType.MUSE_SPARK_AUDIO, ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO,
                ProjectorEntry.PROJECTOR_MUSE_SPARK_AUDIO);
        audio(m, ProjectorType.DOTS3NOTE_A, ProjectorEntry.ENCODER_DOTS3NOTE_A,
                ProjectorEntry.PROJECTOR_DOTS3NOTE_A);
        audio(m, ProjectorType.GEMMA3NA, ProjectorEntry.ENCODER_GEMMA3N_USM, ProjectorEntry.PROJECTOR_GEMMA3NA);
        ProjectorEntry qwen3TtsGen = ProjectorEntry.of(ProjectorType.QWEN3TTS_GEN, ProjectorModality.GEN_AUDIO,
                ProjectorEntry.ENCODER_QWEN3TTS_GEN, ProjectorEntry.PROJECTOR_QWEN3TTS_GEN);
        m.put(qwen3TtsGen.key(), qwen3TtsGen);
        ProjectorEntry pocketTtsGen = ProjectorEntry.of(ProjectorType.POCKETTTS_GEN, ProjectorModality.GEN_AUDIO,
                ProjectorEntry.ENCODER_POCKETTTS_GEN, ProjectorEntry.PROJECTOR_POCKETTTS_GEN);
        m.put(pocketTtsGen.key(), pocketTtsGen);
        ProjectorEntry lfm3aGen = ProjectorEntry.of(ProjectorType.LFM3A_GEN, ProjectorModality.GEN_AUDIO,
                ProjectorEntry.ENCODER_LFM3A_GEN, ProjectorEntry.PROJECTOR_LFM3A_GEN);
        m.put(lfm3aGen.key(), lfm3aGen);
        m.put(ProjectorType.QWEN25O.typeName(), ProjectorEntry.rejected(ProjectorType.QWEN25O, "the legacy"
                + " mixed-modality type names no single encoder; each encoder of the file resolves it per modality"
                + " (get(name, modality)): vision runs as qwen2.5vl_merger, audio as qwen2a"));
        for (ProjectorType t : ProjectorType.named()) {
            m.computeIfAbsent(t.typeName(), name -> ProjectorEntry.rejected(t, "NeuralFree builds no forward"
                    + " graphs for this family; the implemented vision families are " + RUNNABLE_VISION_NAMES
                    + ", the implemented audio families " + RUNNABLE_AUDIO_NAMES));
        }
        return m;
    }

    /** The {@code clip.projector_type} values of the built-in runnable vision rows, for the refusal message. */
    private static final String RUNNABLE_VISION_NAMES = "mlp, mlp_norm, gemma3, qwen2vl_merger, qwen2.5vl_merger,"
            + " qwen3vl_merger, resampler (minicpmv), internvl, pixtral, lightonocr, idefics3, llama4, phi4, lfm2,"
            + " d1omni_v, janus_pro, cohere2v, paddleocr, kimivl, kimik25, dots_ocr, dots3note_v, hunyuanvl, ldp,"
            + " ldpv2, adapter (glm-edge), cogvlm, glm4v, glm5v, youtuvl, granite4_vision, yasa2, lfm3vl, ling3vl,"
            + " minimax_m3, muse-glimmer, nemotron_v2_vl, nemotron_omni, exaone4_5, sensenova_u1, step3vl,"
            + " deepseek4v, deepseek41v, deepseekocr, deepseekocr2, gemma4v, gemma4uv, minicpmv4_6, minicpmv4_7,"
            + " minicpm_ocr_1_0, mimovl and gemma3nv";

    /** The {@code clip.projector_type} values of the built-in runnable audio rows, for the refusal message. */
    private static final String RUNNABLE_AUDIO_NAMES = "ultravox, qwen2a, voxtral, musicflamingo, meralion, glma"
            + ", qwen3a, lfm2a, d1omni_a, lfm3a, gemma4a, gemma4ua, parakeet, granite_speech, voxtral_rt,"
            + " mimo_audio, muse-spark-audio, dots3note_a and gemma3na";

    /** The {@code clip.projector_type} values of the built-in runnable audio-generation rows. */
    private static final String RUNNABLE_GEN_AUDIO_NAMES = "qwen3tts_gen, pockettts_gen and lfm3a_gen";

    private static void runnable(Map<String, ProjectorEntry> m, ProjectorType type, String encoder,
                                 String projector) {
        ProjectorEntry e = ProjectorEntry.of(type, ProjectorModality.VISION, encoder, projector);
        m.put(e.key(), e);
    }

    private static void audio(Map<String, ProjectorEntry> m, ProjectorType type, String encoder, String projector) {
        ProjectorEntry e = ProjectorEntry.of(type, ProjectorModality.AUDIO, encoder, projector);
        m.put(e.key(), e);
    }

    /**
     * Adds the rows of plugin {@link ProjectorProvider}s on top of the built-in map, in provider-name order
     * ({@link Plugins#load} sorts them). A key the built-in map or an earlier provider registers is rejected
     * with both sources named: plugins never replace rows.
     *
     * <p>Public for the plugin tests, which rebuild what {@link Holder} builds without the
     * {@code META-INF/services} resources of the production class path.
     *
     * @throws IllegalStateException with both sources named
     */
    public static Map<String, ProjectorEntry> mergeProviders(Map<String, ProjectorEntry> builtIn,
                                                             List<ProjectorProvider<ProjectorEntry>> providers) {
        Objects.requireNonNull(builtIn, "builtIn");
        Objects.requireNonNull(providers, "providers");
        Map<String, ProjectorEntry> m = new LinkedHashMap<>(builtIn);
        Map<String, String> source = new LinkedHashMap<>();
        for (String key : builtIn.keySet()) {
            source.put(key, "the built-in projector registry");
        }
        for (ProjectorProvider<ProjectorEntry> p : providers) {
            addProviderRows(m, source, p);
        }
        return Collections.unmodifiableMap(m);
    }

    @SuppressWarnings("unchecked")
    private static void addProviderRows(Map<String, ProjectorEntry> m, Map<String, String> source,
                                        ProjectorProvider<?> p) {
        String name = p.getClass().getName();
        List<ProjectorEntry> rows = ((ProjectorProvider<ProjectorEntry>) p).projectorEntries();
        if (rows == null) {
            throw new IllegalStateException("plugin provider " + name + " returned null rows");
        }
        for (ProjectorEntry e : rows) {
            if (e == null) {
                throw new IllegalStateException("plugin provider " + name + " returned a null row");
            }
            String prev = source.get(e.key());
            if (prev != null) {
                throw new IllegalStateException("projector key '" + e.key() + "' is registered by " + prev
                        + " and by plugin provider " + name + "; plugins never replace registered rows");
            }
            source.put(e.key(), "plugin provider " + name);
            m.put(e.key(), e);
        }
    }
}

package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.plugin.ProjectorProvider;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The projector registry: built-in rows, refusals by name, and the plugin SPI merge in the ArchRegistry style. */
public class ProjectorRegistryTest {

    @Test
    public void implementedFamiliesAreRunnable() {
        ProjectorEntry mlp = ProjectorRegistry.get("mlp");
        assertTrue(mlp.runnable());
        assertEquals(ProjectorModality.VISION, mlp.modality());
        assertEquals(ProjectorEntry.ENCODER_CLIP_VIT, mlp.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_LLAVA_MLP, mlp.projectorGraph());

        assertEquals(ProjectorEntry.ENCODER_CLIP_VIT, ProjectorRegistry.get("mlp_norm").encoderGraph());
        assertEquals(ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorRegistry.get("gemma3").encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_GEMMA3, ProjectorRegistry.get("gemma3").projectorGraph());
        assertEquals(ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorRegistry.get("qwen2vl_merger").encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_QWEN2VL_MERGER, ProjectorRegistry.get("qwen2.5vl_merger")
                .projectorGraph());
    }

    @Test
    public void visionFamiliesOfTheClipGraphsAreRunnable() {
        assertRow("resampler", ProjectorEntry.ENCODER_MINICPMV_VIT, ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER);
        assertRow("internvl", ProjectorEntry.ENCODER_INTERNVL_VIT, ProjectorEntry.PROJECTOR_INTERNVL);
        assertRow("pixtral", ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.PROJECTOR_PIXTRAL);
        assertRow("idefics3", ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_IDEFICS3);
        assertRow("llama4", ProjectorEntry.ENCODER_LLAMA4_VIT, ProjectorEntry.PROJECTOR_LLAMA4);
        assertRow("phi4", ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_PHI4);
        assertRow("qwen3vl_merger", ProjectorEntry.ENCODER_QWEN3VL_VIT, ProjectorEntry.PROJECTOR_QWEN3VL_MERGER);
        assertRow("paddleocr", ProjectorEntry.ENCODER_PADDLEOCR_VIT, ProjectorEntry.PROJECTOR_PADDLEOCR);
        assertRow("kimivl", ProjectorEntry.ENCODER_KIMIVL_VIT, ProjectorEntry.PROJECTOR_KIMIVL);
        assertRow("dots_ocr", ProjectorEntry.ENCODER_DOTS_VIT, ProjectorEntry.PROJECTOR_DOTS);
        assertRow("dots3note_v", ProjectorEntry.ENCODER_DOTS_VIT, ProjectorEntry.PROJECTOR_DOTS);
        assertRow("hunyuanvl", ProjectorEntry.ENCODER_HUNYUANVL_VIT, ProjectorEntry.PROJECTOR_HUNYUANVL);
        assertRow("janus_pro", ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_JANUS);
        assertRow("lfm2", ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_LFM2);
        assertRow("d1omni_v", ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT, ProjectorEntry.PROJECTOR_LFM2);
        assertRow("cohere2v", ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_COHERE2V);
        assertRow("lightonocr", ProjectorEntry.ENCODER_PIXTRAL_VIT, ProjectorEntry.PROJECTOR_LIGHTONOCR);
        assertTrue(ProjectorRegistry.get("resampler").tensors().optional().contains("v.blk.%d.ls1.weight"));
    }

    @Test
    public void llavaGlmCogvlmYoutuGraniteAndYasa2FamiliesAreRunnable() {
        assertRow("ldp", ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LDP);
        assertRow("ldpv2", ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LDPV2);
        assertRow("adapter", ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_GLM_EDGE);
        assertRow("cogvlm", ProjectorEntry.ENCODER_COGVLM_VIT, ProjectorEntry.PROJECTOR_COGVLM);
        assertRow("glm4v", ProjectorEntry.ENCODER_GLM4V_VIT, ProjectorEntry.PROJECTOR_GLM4V);
        assertRow("youtuvl", ProjectorEntry.ENCODER_YOUTUVL_VIT, ProjectorEntry.PROJECTOR_YOUTUVL);
        assertRow("granite4_vision", ProjectorEntry.ENCODER_GRANITE4_VIT, ProjectorEntry.PROJECTOR_GRANITE4);
        assertRow("yasa2", ProjectorEntry.ENCODER_YASA2_CONVNEXT, ProjectorEntry.PROJECTOR_YASA2);
    }

    @Test
    public void theLegacyOmniTypeResolvesPerModality() {
        ProjectorEntry vision = ProjectorRegistry.get("qwen2.5o", ProjectorModality.VISION);
        assertSame(ProjectorRegistry.get("qwen2.5vl_merger"), vision);
        assertEquals(ProjectorEntry.ENCODER_QWEN2VL_VIT, vision.encoderGraph());
        ProjectorEntry audio = ProjectorRegistry.get("qwen2.5o", ProjectorModality.AUDIO);
        assertSame(ProjectorRegistry.get("qwen2a"), audio);
        assertEquals(ProjectorEntry.PROJECTOR_QWEN2A, audio.projectorGraph());
        // the bare name names no single encoder
        assertFalse(ProjectorRegistry.has("qwen2.5o"));
        UnsupportedProjectorException bare = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.get("qwen2.5o"));
        assertTrue(bare.getMessage(), bare.getMessage().contains("qwen2.5vl_merger"));
        assertTrue(bare.getMessage(), bare.getMessage().contains("qwen2a"));
        // every other name is itself, and a row of the other modality is refused by name
        assertSame(ProjectorRegistry.get("yasa2"), ProjectorRegistry.get("yasa2", ProjectorModality.VISION));
        UnsupportedProjectorException crossed = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.get("qwen2a", ProjectorModality.VISION));
        assertTrue(crossed.getMessage(), crossed.getMessage().contains("'qwen2a' is registered for the AUDIO encoder"));
    }

    @Test
    public void glm5vLfm3vlLingMinimaxMuseNemotronAndExaoneFamiliesAreRunnable() {
        assertRow("glm5v", ProjectorEntry.ENCODER_GLM4V_VIT, ProjectorEntry.PROJECTOR_GLM4V);
        assertRow("lfm3vl", ProjectorEntry.ENCODER_LFM3VL_VIT, ProjectorEntry.PROJECTOR_LFM3VL);
        assertRow("ling3vl", ProjectorEntry.ENCODER_QWEN3VL_VIT, ProjectorEntry.PROJECTOR_LING3VL);
        assertRow("minimax_m3", ProjectorEntry.ENCODER_MINIMAX_M3_VIT, ProjectorEntry.PROJECTOR_MINIMAX_M3);
        assertRow("muse-glimmer", ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT, ProjectorEntry.PROJECTOR_MUSE_GLIMMER);
        assertRow("nemotron_v2_vl", ProjectorEntry.ENCODER_NEMOTRON_VIT, ProjectorEntry.PROJECTOR_NEMOTRON);
        assertRow("nemotron_omni", ProjectorEntry.ENCODER_NEMOTRON_VIT, ProjectorEntry.PROJECTOR_NEMOTRON);
        assertRow("exaone4_5", ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        assertTrue(ProjectorRegistry.get("ling3vl").tensors().optional().contains("v.patch_embd.bias"));
    }

    @Test
    public void kimiGemma4MinicpmvMergerMimovlAndGemma3nFamiliesAreRunnable() {
        assertRow("kimik25", ProjectorEntry.ENCODER_KIMIK25_VIT, ProjectorEntry.PROJECTOR_KIMIK25);
        assertRow("gemma4v", ProjectorEntry.ENCODER_GEMMA4V_VIT, ProjectorEntry.PROJECTOR_GEMMA4V);
        assertRow("gemma4uv", ProjectorEntry.ENCODER_GEMMA4UV, ProjectorEntry.PROJECTOR_GEMMA4UV);
        for (String type : new String[]{"minicpmv4_6", "minicpmv4_7", "minicpm_ocr_1_0"}) {
            assertRow(type, ProjectorEntry.ENCODER_MINICPMV46_VIT, ProjectorEntry.PROJECTOR_MINICPMV46_MERGER);
        }
        assertRow("mimovl", ProjectorEntry.ENCODER_MIMOVL_VIT, ProjectorEntry.PROJECTOR_MIMOVL_MERGER);
        assertRow("gemma3nv", ProjectorEntry.ENCODER_MOBILENETV5, ProjectorEntry.PROJECTOR_GEMMA3NV);
    }

    private static void assertRow(String name, String encoder, String projector) {
        ProjectorEntry e = ProjectorRegistry.get(name);
        assertTrue(e.runnable());
        assertEquals(ProjectorModality.VISION, e.modality());
        assertEquals(encoder, e.encoderGraph());
        assertEquals(projector, e.projectorGraph());
        assertEquals(encoder, e.tensors().encoderGraph());
    }

    @Test
    public void whisperQwen3aAndGemma3nAudioFamiliesAreRunnable() {
        String[][] rows = {
                {"ultravox", ProjectorEntry.PROJECTOR_ULTRAVOX},
                {"qwen2a", ProjectorEntry.PROJECTOR_QWEN2A},
                {"voxtral", ProjectorEntry.PROJECTOR_VOXTRAL},
                {"musicflamingo", ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO},
                {"meralion", ProjectorEntry.PROJECTOR_MERALION},
                {"glma", ProjectorEntry.PROJECTOR_GLMA},
        };
        for (String[] row : rows) {
            ProjectorEntry e = ProjectorRegistry.get(row[0]);
            assertTrue(row[0], e.runnable());
            assertEquals(row[0], ProjectorModality.AUDIO, e.modality());
            assertEquals(row[0], ProjectorEntry.ENCODER_WHISPER, e.encoderGraph());
            assertEquals(row[0], row[1], e.projectorGraph());
            assertEquals(row[0], ProjectorEntry.ENCODER_WHISPER, e.tensors().encoderGraph());
        }
        ProjectorEntry qwen3a = ProjectorRegistry.get("qwen3a");
        assertEquals(ProjectorModality.AUDIO, qwen3a.modality());
        assertEquals(ProjectorEntry.ENCODER_QWEN3A, qwen3a.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_QWEN3A, qwen3a.projectorGraph());
        ProjectorEntry lfm2a = ProjectorRegistry.get("lfm2a");
        assertEquals(ProjectorModality.AUDIO, lfm2a.modality());
        assertEquals(ProjectorEntry.ENCODER_CONFORMER, lfm2a.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_CONFORMER, lfm2a.projectorGraph());
        ProjectorEntry granite = ProjectorRegistry.get("granite_speech");
        assertEquals(ProjectorEntry.ENCODER_GRANITE_SPEECH, granite.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_GRANITE_SPEECH, granite.projectorGraph());
        ProjectorEntry muse = ProjectorRegistry.get("muse-spark-audio");
        assertEquals(ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO, muse.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_MUSE_SPARK_AUDIO, muse.projectorGraph());
        ProjectorEntry gemma3na = ProjectorRegistry.get("gemma3na", ProjectorModality.AUDIO);
        assertEquals(ProjectorEntry.ENCODER_GEMMA3N_USM, gemma3na.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_GEMMA3NA, gemma3na.projectorGraph());
        assertEquals(ProjectorEntry.ENCODER_GEMMA3N_USM, gemma3na.tensors().encoderGraph());
    }

    @Test
    public void entriesPairAnEncoderWithItsOwnModality() {
        assertEquals(ProjectorModality.AUDIO, ProjectorEntry.modalityOf(ProjectorEntry.ENCODER_WHISPER));
        assertEquals(ProjectorModality.VISION, ProjectorEntry.modalityOf(ProjectorEntry.ENCODER_CLIP_VIT));
        assertThrows(IllegalArgumentException.class, () -> ProjectorEntry.keyed("odd_pairing",
                ProjectorType.QWEN2A, ProjectorModality.VISION, ProjectorEntry.ENCODER_WHISPER,
                ProjectorEntry.PROJECTOR_QWEN2A));
    }

    @Test
    public void everyKnownTypeHasARowAndUnknownIsRefused() {
        int runnable = 0;
        for (ProjectorType type : ProjectorType.named()) {
            if (ProjectorRegistry.has(type.typeName())) {
                runnable++;
                continue;
            }
            UnsupportedProjectorException e = assertThrows(UnsupportedProjectorException.class,
                    () -> ProjectorRegistry.get(type.typeName()));
            assertTrue(e.getMessage().contains(type.typeName()));
            assertFalse(e.getMessage().isBlank());
        }
        assertEquals(74, runnable);
        assertEquals(ProjectorType.values().length - 1, ProjectorRegistry.entries().size());
        assertThrows(UnsupportedProjectorException.class, () -> ProjectorRegistry.get("not_a_projector"));
        UnsupportedProjectorException known = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.get("qwen2.5o"));
        assertTrue(known.getMessage().contains("not runnable"));
    }

    @Test
    public void theGenAudioFamiliesAreRunnable() {
        ProjectorEntry qwen3tts = ProjectorRegistry.get("qwen3tts_gen", ProjectorModality.GEN_AUDIO);
        assertTrue(qwen3tts.runnable());
        assertEquals(ProjectorEntry.ENCODER_QWEN3TTS_GEN, qwen3tts.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_QWEN3TTS_GEN, qwen3tts.projectorGraph());
        ProjectorEntry pockettts = ProjectorRegistry.get("pockettts_gen", ProjectorModality.GEN_AUDIO);
        assertTrue(pockettts.runnable());
        assertEquals(ProjectorModality.GEN_AUDIO, pockettts.modality());
        assertEquals(ProjectorEntry.ENCODER_POCKETTTS_GEN, pockettts.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_POCKETTTS_GEN, pockettts.projectorGraph());
        assertEquals(ProjectorEntry.ENCODER_POCKETTTS_GEN, pockettts.tensors().encoderGraph());
        ProjectorEntry lfm3a = ProjectorRegistry.get("lfm3a_gen", ProjectorModality.GEN_AUDIO);
        assertTrue(lfm3a.runnable());
        assertEquals(ProjectorModality.GEN_AUDIO, lfm3a.modality());
        assertEquals(ProjectorEntry.ENCODER_LFM3A_GEN, lfm3a.encoderGraph());
        assertEquals(ProjectorEntry.PROJECTOR_LFM3A_GEN, lfm3a.projectorGraph());
        assertEquals(ProjectorEntry.ENCODER_LFM3A_GEN, lfm3a.tensors().encoderGraph());
        assertTrue(lfm3a.tensors().required().stream().anyMatch(r -> r.alternatives().stream()
                .anyMatch(names -> names.contains("a.gen.code.embd.weight"))));
        assertTrue(lfm3a.tensors().required().stream().anyMatch(r -> r.alternatives().stream()
                .anyMatch(names -> names.contains("a.gen.wav.quant.first.codebook.weight"))));
        assertTrue(lfm3a.tensors().optional().contains("a.gen.code.blk.%d.attn_post_norm.weight"));
        assertTrue(lfm3a.tensors().optional().contains("a.gen.code.blk.%d.ffn_post_norm.weight"));
        UnsupportedProjectorException wrongModality = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.get("lfm3a_gen", ProjectorModality.VISION));
        assertTrue(wrongModality.getMessage().contains("GEN_AUDIO"));
    }

    @Test
    public void pluginRowsExtendTheRegistryUnderNewKeys() {
        ProjectorEntry extra = ProjectorEntry.keyed("qwen3vl_merger_next", ProjectorType.QWEN3VL,
                ProjectorModality.VISION, ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        ProjectorEntry rejected = ProjectorEntry.keyedRejected("pixtral_plus", ProjectorType.PIXTRAL,
                "not built yet");
        List<ProjectorProvider<ProjectorEntry>> providers =
                Collections.singletonList(new StaticProvider(Arrays.asList(extra, rejected)));
        Map<String, ProjectorEntry> merged =
                ProjectorRegistry.mergeProviders(ProjectorRegistry.builtIn(), providers);
        assertSame(extra, merged.get("qwen3vl_merger_next"));
        assertSame(rejected, merged.get("pixtral_plus"));
        assertTrue(merged.get("mlp").runnable());
    }

    @Test
    public void pluginCollisionsNameBothSources() {
        ProjectorEntry row = ProjectorEntry.of(ProjectorType.MLP, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        List<ProjectorProvider<ProjectorEntry>> providers =
                Collections.singletonList(new StaticProvider(Collections.singletonList(row)));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ProjectorRegistry.mergeProviders(ProjectorRegistry.builtIn(), providers));
        assertTrue(e.getMessage().contains("mlp"));
        assertTrue(e.getMessage().contains("the built-in projector registry"));
        assertTrue(e.getMessage().contains(StaticProvider.class.getName()));
    }

    @Test
    public void providerRowsCollidingAmongThemselvesAreRejected() {
        ProjectorEntry row = ProjectorEntry.keyed("ling3vl_next", ProjectorType.LING3VL, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_QWEN2VL_VIT, ProjectorEntry.PROJECTOR_QWEN2VL_MERGER);
        List<ProjectorProvider<ProjectorEntry>> providers = Arrays.asList(
                new StaticProvider(Collections.singletonList(row)),
                new StaticProvider(Collections.singletonList(row)));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ProjectorRegistry.mergeProviders(ProjectorRegistry.builtIn(), providers));
        assertTrue(e.getMessage().contains("ling3vl_next"));
        assertTrue(e.getMessage().contains("plugins never replace"));
    }

    @Test
    public void unknownTypedRowsNeedAnExplicitKey() {
        assertThrows(IllegalArgumentException.class,
                () -> ProjectorEntry.of(ProjectorType.UNKNOWN, ProjectorModality.VISION,
                        ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP));
        ProjectorEntry keyed = ProjectorEntry.keyed("future", ProjectorType.UNKNOWN, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_CLIP_VIT, ProjectorEntry.PROJECTOR_LLAVA_MLP);
        assertEquals("future", keyed.key());
        assertEquals(ProjectorType.UNKNOWN, keyed.type());
    }

    @Test
    public void runnableVisionRowsCarryTheirTypesPreprocessing() {
        for (ProjectorEntry e : ProjectorRegistry.entries()) {
            if (e.runnable() && e.modality() == ProjectorModality.VISION) {
                assertSame(e.key(), e.type().visionPreprocessing(), e.visionPreprocessing());
            } else {
                assertNull(e.key(), e.visionPreprocessing());
            }
        }
        assertSame(ProjectorType.QWEN2VL.visionPreprocessing(),
                ProjectorRegistry.visionPreprocessing("qwen2vl_merger"));
        assertTrue(ProjectorRegistry.has("qwen3vl_merger"));
        assertSame(ProjectorType.QWEN3VL.visionPreprocessing(),
                ProjectorRegistry.visionPreprocessing("qwen3vl_merger"));
        assertTrue(ProjectorRegistry.has("glm4v"));
        assertSame(ProjectorType.GLM4V.visionPreprocessing(), ProjectorRegistry.visionPreprocessing("glm4v"));
        assertSame(ProjectorType.IDEFICS3.visionPreprocessing(), ProjectorRegistry.visionPreprocessing("idefics3"));
        for (ProjectorType t : new ProjectorType[] {ProjectorType.SENSENOVA_U1, ProjectorType.STEP3VL,
                ProjectorType.DEEPSEEK4V, ProjectorType.DEEPSEEK41V, ProjectorType.DEEPSEEKOCR,
                ProjectorType.DEEPSEEKOCR2, ProjectorType.GRANITE4_VISION, ProjectorType.GEMMA4V,
                ProjectorType.GEMMA4UV, ProjectorType.MINICPMV4_6, ProjectorType.MINICPMV4_7,
                ProjectorType.MINICPM_OCR_1_0, ProjectorType.MIMOVL, ProjectorType.GEMMA3NV}) {
            assertTrue(t.typeName(), ProjectorRegistry.get(t.typeName()).runnable());
            assertSame(t.visionPreprocessing(), ProjectorRegistry.visionPreprocessing(t.typeName()));
        }
        UnsupportedProjectorException audio = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.visionPreprocessing("ultravox"));
        assertTrue(audio.getMessage().contains("'ultravox'"));
        assertTrue(audio.getMessage().contains("granite4_vision"));
        UnsupportedProjectorException unknown = assertThrows(UnsupportedProjectorException.class,
                () -> ProjectorRegistry.visionPreprocessing("not_a_projector"));
        assertTrue(unknown.getMessage().contains("'not_a_projector'"));
        assertTrue(unknown.getMessage().contains("ProjectorProvider"));
    }

    @Test
    public void onlyVisionRowsDeclarePreprocessing() {
        VisionPreprocessing p = ProjectorType.GEMMA3.visionPreprocessing();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ProjectorEntry.keyed("audio_next", ProjectorType.UNKNOWN, ProjectorModality.AUDIO,
                        ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_ULTRAVOX, p));
        assertTrue(e.getMessage().contains("audio_next"));
        ProjectorEntry row = ProjectorEntry.keyed("gemma3_next", ProjectorType.UNKNOWN, ProjectorModality.VISION,
                ProjectorEntry.ENCODER_SIGLIP_VIT, ProjectorEntry.PROJECTOR_GEMMA3, p);
        assertSame(p, row.visionPreprocessing());
        assertNull(ProjectorEntry.keyed("ultravox_next", ProjectorType.ULTRAVOX, ProjectorModality.AUDIO,
                ProjectorEntry.ENCODER_WHISPER, ProjectorEntry.PROJECTOR_ULTRAVOX).visionPreprocessing());
    }

    private static final class StaticProvider implements ProjectorProvider<ProjectorEntry> {
        private final List<ProjectorEntry> rows;

        StaticProvider(List<ProjectorEntry> rows) {
            this.rows = rows;
        }

        @Override
        public List<ProjectorEntry> projectorEntries() {
            return rows;
        }
    }
}

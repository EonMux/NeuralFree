package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The projector architectures the remaining vision-tower classes register: every Hugging Face name {@code archs.tsv}
 * lists for one of them resolves, through {@link HfArchRegistry}, to a {@code clip} entry whose arch hook returns
 * {@link Arch#CLIP}. Loading the registry also merges the tables, so a name registered twice for one kind fails here.
 */
public class HfVisionTowerRegistryTest {

    /** The Hugging Face names of the converter classes this change ports. */
    private static final List<String> NAMES = Collections.unmodifiableList(Arrays.asList(
            // DeepseekOCRVisionModel, DeepseekOCR2VisionModel, UnlimitedOCRVisionModel
            "DeepseekOCRForCausalLM", "DeepseekOCR2ForCausalLM", "UnlimitedOCRForCausalLM",
            // DeepseekV4FlashVisionModel, DeepseekV41VisionModel
            "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM",
            // MiniCPMV4_6VisionModel, MiniCPMV4_7VisionModel, MiniCPMOCR1_0VisionModel
            "MiniCPMV4_6ForConditionalGeneration", "MiniCPMV4_7ForConditionalGeneration",
            "MiniCPMOcrForConditionalGeneration",
            // MiniCPMOCRSeqClsVisionModel
            "MiniCPMOCRForSequenceClassification",
            // LFM2VLModel, LFM3VLMmprojModel
            "Lfm2VlForConditionalGeneration", "Lfm3VlForConditionalGeneration",
            // Granite4VisionMmprojModel, NemotronNanoV2VLModel (with RADIOModel)
            "Granite4VisionForConditionalGeneration", "NemotronH_Nano_VL_V2", "RADIOModel",
            // BailingMoeV3VLVisionModel
            "BailingMoeV3VLForConditionalGeneration",
            // ClefVisionModel, H2OLightningVisionModel, Qwen4ExpVisionModel
            "ClefModel", "H2OLightningModel", "Qwen4ExpForConditionalGeneration",
            // CogVLMVisionModel, Cohere2VisionModel, DotsOCRVisionModel, Exaone4_5VisionModel
            "CogVLMForCausalLM", "Cohere2VisionForConditionalGeneration", "DotsOCRForCausalLM",
            "Exaone4_5_ForConditionalGeneration",
            // Glm5vModel, H2O, HunyuanVLVisionModel, JanusProVisionModel, KimiK25Model
            "Glm5vForConditionalGeneration", "HunYuanVLForConditionalGeneration", "JanusForConditionalGeneration",
            "KimiK25ForConditionalGeneration",
            // MiniMaxM3VisionModel, MuseGlimmerVisionModel, MuseSparkVisionModel, PaddleOCRVisionModel
            "MiniMaxM3SparseForConditionalGeneration", "MuseGlimmerForConditionalGeneration",
            "MuseSparkForConditionalGeneration", "PaddleOCRVisionModel",
            // Phi4VisionMmprojModel, PplxDeciderVisionModel, SenseNovaU1VisionModel
            "Phi4ForCausalLMV", "PplxDeciderModel", "NEOVisionModel",
            // Step3VLVisionModel, VllmD3VisionModel, YoutuVLVisionModel
            "StepVLForConditionalGeneration", "Step3p7ForConditionalGeneration", "VllmD3Model",
            "YoutuVLForConditionalGeneration",
            // Dots3NoteMmprojModel
            "Dots3NoteForCausalLM", "Dots3NoteForConditionalGeneration"));

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyPortedTowerResolvesToClip() throws IOException {
        Path dir = tmp.newFolder("registry").toPath();
        for (String name : NAMES) {
            HfConfig cfg = HfConfigLoader.fromJson(dir, "{\"architectures\":[\"" + name + "\"]}");
            HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.MMPROJ);
            assertNotNull(name, entry);
            assertFalse(name + " is rejected: " + entry.rejection(), entry.isRejected());
            assertEquals(name, HfArchEntry.Kind.MMPROJ, entry.kind());
            assertEquals(name, Arch.CLIP,
                    entry.arch().resolve(cfg, HfTensorIndex.of(new LinkedHashMap<>())));
            assertTrue(name + " is not listed as supported",
                    HfArchRegistry.supportedNames(HfArchEntry.Kind.MMPROJ).contains(name));
        }
    }

    @Test
    public void noProjectorRowOfThePortedClassesIsUnmapped() {
        List<String> unmapped = HfArchRegistry.unmapped(HfArchEntry.Kind.MMPROJ);
        for (String name : NAMES) {
            assertFalse(name + " has no projector entry", unmapped.contains(name));
        }
    }
}

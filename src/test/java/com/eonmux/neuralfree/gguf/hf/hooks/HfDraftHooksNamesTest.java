package com.eonmux.neuralfree.gguf.hf.hooks;

import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The draft names the target dispatch of {@link HfDraftHooks} recognizes: every draft family that serves a target
 * checkpoint of its own, including vLLM's rewritten and parallel-drafting spellings of the same module.
 */
public class HfDraftHooksNamesTest {

    /**
     * The family names the recursive target dispatch falls back on, as {@code target_cls is type(self)} does in the
     * reference converters. vLLM's rewritten spellings ({@code Qwen3OmniDSparkModel} for {@code Qwen3DSparkModel},
     * {@code DFlashMuseGlimmerAssistantModel} for {@code MuseGlimmerAssistantModel}) name the same module, so they
     * belong to the set beside their base names.
     */
    @Test
    public void theDraftNameSetHoldsEveryDraftFamily() {
        Set<String> expected = new HashSet<>();
        expected.add("DFlashDraftModel");
        expected.add("DFlash2DraftModel");
        expected.add("Qwen3DSparkModel");
        expected.add("Qwen3OmniDSparkModel");
        expected.add("DSparkDraftModel");
        expected.add("DSparkSpeculator");
        expected.add("Lfm2DSparkDraftModel");
        expected.add("LingDSparkModel");
        expected.add("DeepseekV4DSparkModel");
        expected.add("DeepseekV41DSparkModel");
        expected.add("MistralLarge4DSparkModel");
        expected.add("Gemma4DSparkModel");
        expected.add("LiLiCorrDraftModel");
        expected.add("MuseGlimmerAssistantModel");
        expected.add("DFlashMuseGlimmerAssistantModel");
        assertEquals(expected, HfDraftHooks.DRAFT_NAMES);
    }

    /** A converted target model's own architecture name must not be mistaken for a draft family. */
    @Test
    public void ordinaryTargetNamesAreNotDraftFamilies() {
        for (String name : new String[]{"Qwen3ForCausalLM", "LlamaForCausalLM", "MiMoForCausalLM",
                "DeepseekV3ForCausalLM"}) {
            assertTrue(name, !HfDraftHooks.DRAFT_NAMES.contains(name));
        }
    }
}

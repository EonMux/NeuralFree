package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The names vLLM registers for a draft checkpoint whose module is the one NeuralFree already converts: the alias
 * shares the base family's entry, so it converts with the same hooks and routes to the same GGUF architecture. The
 * vLLM draft classes that ship a different checkpoint layout each stay refused by name until their layout is
 * delivered.
 */
public class HfDraftAliasRoutingTest {

    @Test
    public void mimoMtpModelSharesTheMimoConversion() {
        HfArchEntry mimo = entry("MiMoForCausalLM");
        assertSame(mimo, entry("MiMoMTPModel"));
        assertEquals(Arch.QWEN2, mimo.arch().resolve(config("MiMoMTPModel"), HfTensorIndex.of(
                Collections.emptyMap())));
    }

    @Test
    public void qwen3OmniDSparkSharesTheQwen3DSparkConversion() {
        assertSame(entry("Qwen3DSparkModel"), entry("Qwen3OmniDSparkModel"));
    }

    @Test
    public void theDflashSpelledMuseGlimmerAssistantSharesItsConversion() {
        assertSame(entry("MuseGlimmerAssistantModel"), entry("DFlashMuseGlimmerAssistantModel"));
    }

    /**
     * The remaining vLLM draft classes whose checkpoints NeuralFree's drafter engines cannot describe yet: each is
     * still refused with a message naming it. The DeepSeek EAGLE-3 drafts are delivered over the EAGLE3 row with
     * MLA attention; the other layouts still await their drafter.
     */
    @Test
    public void undeliveredDraftLayoutsStayRefusedByName() {
        for (String name : new String[]{"DFlash2BeamDraftModel", "DSparkV41DraftModel", "DSparkMiniMaxDraftModel",
                "MuseSparkDSparkModel", "K3DSparkModel", "Qwen3XPressModel", "DFlashQwen3XPressModel",
                "DFlashLagunaForCausalLM"}) {
            GgufFormatException e = assertThrows(name, GgufFormatException.class,
                    () -> HfArchRegistry.resolve(config(name), HfArchEntry.Kind.TEXT));
            assertTrue(name, e.getMessage().contains(name));
        }
        // EagleDeepSeekMTPModel is registered as an EAGLE-1 drafter, which runs GQA attention only: a DeepSeek
        // checkpoint of it still loads and is refused where the draft is bound to its MLA target.
        assertTrue(HfArchRegistry.has("EagleDeepSeekMTPModel", HfArchEntry.Kind.TEXT));
    }

    /** Each vLLM concat-fc draft class routes to its own GGUF architecture row. */
    @Test
    public void concatFcDraftsUseTheirOwnArches() {
        assertEquals(Arch.EAGLE_MISTRAL_LARGE3, archOf("EagleMistralLarge3ForCausalLM"));
        assertEquals(Arch.EAGLE_LLAMA4, archOf("EagleLlama4ForCausalLM"));
        assertEquals(Arch.EAGLE_MINICPM, archOf("EagleMiniCPMForCausalLM"));
        assertEquals(Arch.EAGLE_COHERE, archOf("EagleCohereForCausalLM"));
    }

    private static Arch archOf(String hfName) {
        return entry(hfName).arch().resolve(config(hfName), HfTensorIndex.of(Collections.emptyMap()));
    }

    /** The DeepSeek EAGLE-3 drafts share the EAGLE3 row with the Qwen3 layout and sit beside the Llama one. */
    @Test
    public void deepseekEagle3DraftsUseTheEagle3Row() {
        HfArchEntry v2 = entry("Eagle3DeepseekV2ForCausalLM");
        assertSame(v2, entry("Eagle3DeepseekV3ForCausalLM"));
        assertEquals(Arch.EAGLE3, v2.arch().resolve(config("Eagle3DeepseekV2ForCausalLM"),
                HfTensorIndex.of(Collections.emptyMap())));
    }

    private static HfArchEntry entry(String hfName) {
        return HfArchRegistry.resolve(config(hfName), HfArchEntry.Kind.TEXT);
    }

    private static com.eonmux.neuralfree.gguf.hf.config.HfConfig config(String hfName) {
        return HfConfigLoader.fromJson(Path.of("."),
                "{\"architectures\": [\"" + hfName + "\"], \"num_hidden_layers\": 1}");
    }
}

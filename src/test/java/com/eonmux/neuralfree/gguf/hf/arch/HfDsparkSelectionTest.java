package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfArchRegistry#resolveDspark}: the converter's {@code --dspark} selects the draft entry from the target
 * checkpoint's architecture (convert_hf_to_gguf.py:272-287), never from the architectures the checkpoint names.
 */
public class HfDsparkSelectionTest {

    private static HfConfig config(String arch) {
        return HfConfigLoader.fromJson(Path.of("."), "{\"architectures\": [\"" + arch + "\"]}");
    }

    @Test
    public void everyTargetArchitectureSelectsItsDraft() {
        assertEquals("DeepseekV4DSparkModel",
                HfArchRegistry.resolveDspark(config("DeepseekV4ForCausalLM")).hfNames().get(0));
        assertEquals("DeepseekV41DSparkModel",
                HfArchRegistry.resolveDspark(config("DeepseekV41ForCausalLM")).hfNames().get(0));
        assertEquals("MistralLarge4DSparkModel",
                HfArchRegistry.resolveDspark(config("MistralLarge4ForConditionalGeneration")).hfNames().get(0));
        assertEquals("the drafts are the DFLASH converters",
                Arch.DFLASH, HfArchRegistry.resolveDspark(config("DeepseekV41ForCausalLM"))
                        .arch().resolve(config("DeepseekV41ForCausalLM"), null));
    }

    @Test
    public void anotherTargetIsRefusedWithTheConvertersMessage() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolveDspark(config("LlamaForCausalLM")));
        assertTrue(e.getMessage(), e.getMessage().contains("--dspark is only supported for"
                + " DeepseekV4ForCausalLM, DeepseekV41ForCausalLM and Mistral Large 4"));
    }
}

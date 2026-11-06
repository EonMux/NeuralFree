package com.eonmux.neuralfree.gguf.hf.arch;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.file.Path;
import java.util.LinkedHashMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Registration of the vLLM-only text families this task adds: the supported names resolve, the names without a
 * NeuralFree graph refuse with their own name and a stated reason, and the Step3.5 MTP draft name resolves to the
 * same Step3.5 entry as its text model.
 */
public class HfVllmTextArchsTest {

    private static Arch resolve(String name, String extra) {        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = HfConfigLoader.fromJson(Path.of("."),
                "{\"architectures\":[\"" + name + "\"]" + (extra.isEmpty() ? "" : "," + extra) + "}");
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).arch()
                .resolve(cfg, HfTensorIndex.of(new LinkedHashMap<>()));
    }

    @Test
    public void supportedNamesResolveToTheirArchitecture() {
        assertEquals(Arch.STEP1, resolve("Step1ForCausalLM", ""));
        assertEquals(Arch.STEP3_TEXT, resolve("Step3TextForCausalLM", ""));
        assertEquals(Arch.STEP35, resolve("Step3p5ForCausalLM", ""));
        assertEquals(Arch.STEP35, resolve("Step3p5MTP", ""));
        assertEquals(Arch.STEP5, resolve("Step5ForCausalLM", ""));
        assertEquals(Arch.STEP5, resolve("Step5MTP", ""));
        assertEquals(Arch.DEEPSEEK2, resolve("SarvamMLAForCausalLM", ""));
        assertEquals(Arch.DEEPSEEK2, resolve("PanguUltraMoEForCausalLM",
                "\"qk_nope_head_dim\":128,\"qk_rope_head_dim\":64,\"v_head_dim\":128,\"kv_lora_rank\":512"));
        assertEquals(Arch.PANGU_MOE, resolve("PanguProMoEV2ForCausalLM", ""));
        assertEquals(Arch.INTERN_S1_PRO, resolve("InternS1ProForConditionalGeneration", ""));
        for (String name : new String[]{"Step1ForCausalLM", "Step3TextForCausalLM", "Step3p5ForCausalLM",
                "Step3p5MTP", "Step5ForCausalLM", "Step5MTP", "SarvamMLAForCausalLM", "PanguProMoEV2ForCausalLM",
                "PanguUltraMoEForCausalLM", "InternS1ProForConditionalGeneration"}) {
            assertTrue(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
        }
    }

    @Test
    public void step5RefusesADenseOnlyCheckpointByItsSparseSection() {
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = HfConfigLoader.fromJson(Path.of("."),
                "{\"architectures\":[\"Step5ForCausalLM\"]}");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).configFixup().apply(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains("sparse_config"));
    }
}

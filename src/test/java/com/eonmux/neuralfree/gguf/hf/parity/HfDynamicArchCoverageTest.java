package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBertVariantHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDeepseekHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLinearHooksB;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistral3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfMistralNativeHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfSsmHybridHooksA;
import com.eonmux.neuralfree.gguf.hf.names.HfArchNames;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Every {@code dynamic_archs.tsv} row resolves to the row's {@code gguf_arch} for a
 * minimal configuration that meets the row's condition (dynamic_archs.tsv holds the class's
 * {@code self.model_arch} assignments, HfArchNames).
 *
 * <p>The resolver of the delivered runtime ports is the entry's {@code arch} hook per converter class; the check calls
 * the ported static directly ({@code Hf*Hooks.arch or -Arch}) with a minimal {@link HfConfig} built from
 * {@link HfConfigLoader#fromJson} for exactly the members the condition reads, and an empty tensor index (the
 * conditions of the delivered rows depend on the configuration only). One entry per {@code dynamic_archs.tsv} row:
 * the expected arch is the row's own {@link HfArchNames.DynamicArch#arch()} so the assertion tracks the table.</p>
 */
public class HfDynamicArchCoverageTest {

    @FunctionalInterface
    private interface ArchHook extends BiFunction<HfConfig, HfTensorIndex, Arch> {
    }

    @Test
    public void everyDynamicRowResolvesToItsArch() throws IOException {
        int checked = 0;
        for (HfArchNames.Row row : HfArchNames.rows(HfArchEntry.Kind.TEXT)) {
            checked += checkRows(row.dynamicArchs());
        }
        // MistralModel converts the Mistral-native params.json layout, which no HF architecture name selects.
        checked += checkRows(HfArchNames.dynamicArchs("MistralModel"));
        assertTrue("dynamic_archs.tsv has rows", checked > 0);
    }

    private int checkRows(List<HfArchNames.DynamicArch> rows) throws IOException {
        for (HfArchNames.DynamicArch d : rows) {
            ArchHook hook = hookOf(d.converterClass());
            Path dir = Files.createTempDirectory("hf-dynamic-arch");
            dir.toFile().deleteOnExit();
            Arch resolved = hook.apply(minimalConfig(d, dir), HfTensorIndex.of(new LinkedHashMap<>()));
            assertEquals(d.toString(), d.arch(), resolved);
        }
        return rows.size();
    }

    /** Rows never carry a null class in the delivered table; guard them anyway. */
    private ArchHook hookOf(String converterClass) {
        switch (converterClass) {
            case "NomicBertModel":
                return HfBertVariantHooks::nomicArch;
            case "XLMRobertaModel":
                return HfBertHooks::xlmRobertaArch;
            case "DeepseekV2Model":
                return HfDeepseekHooks::arch;
            case "GraniteHybridModel":
                return HfSsmHybridHooksA::graniteHybridArch;
            case "LlamaModel":
                return HfLlamaHooks::arch;
            case "MiniCPMV4_7TextModel":
                return HfLinearHooksB::miniCpmV47Arch;
            case "MistralModel":
                return HfMistralNativeHooks::arch;
            case "Mistral3Model":
                return HfMistral3Hooks::arch;
            case "NemotronHModel":
                return HfSsmHybridHooksA::nemotronHArch;
            case "NemotronHPuzzleModel":
                return HfSsmHybridHooksA::nemotronHPuzzleArch;
            default:
                throw new AssertionError("dynamic_archs.tsv class has no ported arch hook: " + converterClass);
        }
    }

    /**
     * A minimal {@code config.json} per row: every validated condition of the delivered file may only need one array
     * member or boolean, so each condition maps to the override fields it reads (the file's {@code condition} column
     * strings), and the row's class provides the rest of the required fields through the class defaults.
     */
    private static HfConfig minimalConfig(HfArchNames.DynamicArch d, Path dir) {
        Map<String, String> minimal = minimalConfigs();
        String condition = d.condition() == null ? "" : d.condition();
        for (Map.Entry<String, String> e : minimal.entrySet()) {
            if (condition.contains(e.getKey())) {
                return HfConfigLoader.fromJson(dir, e.getValue());
            }
        }
        return HfConfigLoader.fromJson(dir, "{\"architectures\":[\"LlamaForCausalLM\"]}");
    }

    /** The condition fragments and the minimal config that satisfies each, one per row of dynamic_archs.tsv. */
    private static Map<String, String> minimalConfigs() {
        Map<String, String> m = new LinkedHashMap<>();
        // NomicBertModel rows: self.is_moe of moe_every_n_layers (bert.py:327-328); the negated condition contains
        // the plain one, so it is matched first.
        m.put("not self.is_moe", "{\"architectures\":[\"NomicBertModel\"],\"moe_every_n_layers\":0}");
        m.put("self.is_moe","{\"architectures\":[\"NomicBertForCausalLM\"],\"moe_every_n_layers\":2}");
        // XLMRobertaModel row: a truthy lora_adaptations, the jina-embeddings-v3 task names (bert.py:478-481).
        m.put("lora_adaptations", "{\"architectures\":[\"XLMRobertaModel\"],"
                + "\"lora_adaptations\":[\"retrieval.query\",\"retrieval.passage\"]}");
        // DeepseekV2Model row: the OCR root architectures (deepseek.py:251).
        m.put("origin_hf_arch in ('DeepseekOCRForCausalLM'",
                "{\"architectures\":[\"DeepseekOCRForCausalLM\"],\"model_type\":\"deepseek_v2\"}");
        // GraniteHybridModel rows: _ssm_layers and has_experts of the layer patterns (granite.py:468). The
        // all-attention layer_types plus a block count make _ssm_layers empty (granite.py:452-455).
        m.put("not self._ssm_layers and has_experts",
                "{\"architectures\":[\"GraniteMoeHybridForCausalLM\"],\"num_hidden_layers\":2,"
                        + "\"layer_types\":[\"attention\",\"attention\"],\"num_experts_per_tok\":2}");
        m.put("not self._ssm_layers and not has_experts",
                "{\"architectures\":[\"GraniteHybridForCausalLM\"],\"num_hidden_layers\":2,"
                        + "\"layer_types\":[\"attention\",\"attention\"]}");
        // LlamaModel row: a draft-vocab single-layer draft (llama.py:52).
        m.put("'draft_vocab_size' in self.hparams and self.hparams['num_hidden_layers'] == 1",
                "{\"architectures\":[\"LlamaModel\"],\"draft_vocab_size\":16,\"num_hidden_layers\":1}");
        // MiniCPMV4_7TextModel rows: the text_config model_type (minicpm.py:224-226). The negated condition
        // contains the plain one, so it is matched first.
        m.put("not (text_config.get('model_type') == 'qwen3_5_moe_text')",
                "{\"architectures\":[\"MiniCPMV4_7ForCausalLM\"],\"text_config\":{\"model_type\":\"qwen3_5_text\"}}");
        m.put("text_config.get('model_type') == 'qwen3_5_moe_text'",
                "{\"architectures\":[\"MiniCPMV4_7ForCausalLM\"],\"text_config\":{\"model_type\":\"qwen3_5_moe_text\"}}");
        // MistralModel row: llama_4_scaling absent (mistral.py:36).
        m.put("'llama_4_scaling' not in self.hparams", "{\"architectures\":[\"MistralForCausalLM\"]}");
        // Mistral3Model rows: hf_arch / model_type switches (mistral3.py:23-47).
        m.put("and not (self.hf_arch in ('Ministral3Model', 'Mistral3Model'))",
                "{\"architectures\":[\"Mistral3ForCausalLM\"],\"model_type\":\"llama\"}");
        // The headless names are the Nemotron Embed branch, which the converter accepts only with is_causal False
        // (mistral3.py:66-69).
        m.put("and self.hf_arch in ('Ministral3Model', 'Mistral3Model')",
                "{\"architectures\":[\"Ministral3Model\"],\"model_type\":\"llama\",\"is_causal\":false}");
        m.put("self.hparams.get('model_type') == 'mistral4'",
                "{\"architectures\":[\"Mistral3ForCausalLM\"],\"model_type\":\"mistral4\"}");
        // NemotronHModel row: num_experts_per_tok in {**hparams, **text_config} (nemotron.py:449-451). The
        // NemotronHPuzzleModel row has no condition (nemotron.py:758) and takes the default config.
        m.put("has_moe_params", "{\"architectures\":[\"NemotronHForCausalLM\"],\"num_experts_per_tok\":2}");
        return m;
    }
}

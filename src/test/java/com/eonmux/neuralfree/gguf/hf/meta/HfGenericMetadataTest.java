package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

/**
 * {@link HfGenericMetadata} writes what {@code TextModel.set_gguf_parameters} writes (LC/conversion/base.py:1606-1720)
 * for synthetic configurations: the exact keys in converter order, their values in the GGUF Java value model and
 * their types, through every RoPE scaling branch, M-RoPE sections, causal flags and expert gating functions.
 */
public class HfGenericMetadataTest {

    private static final String LLAMA_BASE = "'architectures': ['LlamaForCausalLM'], 'hidden_size': 4096,"
            + " 'intermediate_size': 11008, 'num_attention_heads': 32, 'num_key_value_heads': 8,"
            + " 'num_hidden_layers': 2, 'max_position_embeddings': 4096, 'rms_norm_eps': 1e-05,"
            + " 'vocab_size': 32000";

    private static HfMetadata generic(Arch arch, String hfName, String singleQuotedJson) {
        HfArchEntry entry = HfArchEntry.of(List.of(hfName), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        HfLoadContext ctx = HfLoadContext.builder(
                HfConfigLoader.fromJson(Path.of("."), "{" + singleQuotedJson.replace('\'', '"') + "}"),
                entry, arch).build();
        HfMetadataBuilder kv = new HfMetadataBuilder(arch.ggufName());
        HfGenericMetadata.apply(ctx, kv);
        return kv.build();
    }

    private static HfMetadata llama(String extra) {
        return generic(Arch.LLAMA, "LlamaForCausalLM", LLAMA_BASE + (extra.isEmpty() ? "" : ", " + extra));
    }

    private static List<String> keys(HfMetadata m) {
        return new ArrayList<>(m.keys());
    }

    @Test
    public void llamaConfigWritesTheConverterKeySet() {
        HfMetadata m = llama("'rope_theta': 500000.0, 'head_dim': 128");
        assertEquals(Arrays.asList("llama.block_count", "llama.context_length", "llama.embedding_length",
                "llama.feed_forward_length", "llama.attention.head_count", "llama.attention.head_count_kv",
                "llama.rope.freq_base", "llama.attention.layer_norm_rms_epsilon", "llama.attention.key_length",
                "llama.attention.value_length"), keys(m));
        assertEquals(2, m.get("llama.block_count"));
        assertEquals(4096, m.get("llama.context_length"));
        assertEquals(4096, m.get("llama.embedding_length"));
        assertEquals(11008, m.get("llama.feed_forward_length"));
        assertEquals(32, m.get("llama.attention.head_count"));
        assertEquals(8, m.get("llama.attention.head_count_kv"));
        assertEquals(500000f, m.get("llama.rope.freq_base"));
        assertEquals(1e-5f, m.get("llama.attention.layer_norm_rms_epsilon"));
        assertEquals(128, m.get("llama.attention.key_length"));
        assertEquals(128, m.get("llama.attention.value_length"));
        for (String k : Arrays.asList("llama.block_count", "llama.context_length", "llama.embedding_length",
                "llama.feed_forward_length", "llama.attention.head_count", "llama.attention.head_count_kv",
                "llama.attention.key_length", "llama.attention.value_length")) {
            assertEquals(k, GGUFValueType.UINT32, m.type(k));
        }
        assertEquals(GGUFValueType.FLOAT32, m.type("llama.rope.freq_base"));
        assertEquals(GGUFValueType.FLOAT32, m.type("llama.attention.layer_norm_rms_epsilon"));
    }

    @Test
    public void perLayerListsBecomeInt32Arrays() {
        HfMetadata m = llama("'intermediate_size': [1024, 2048], 'num_key_value_heads': [8, 4]");
        assertEquals(GGUFValueType.ARRAY, m.type("llama.feed_forward_length"));
        assertEquals(GGUFValueType.INT32, m.elementType("llama.feed_forward_length"));
        assertEquals(Arrays.asList(1024, 2048), m.get("llama.feed_forward_length"));
        assertEquals(Arrays.asList(8, 4), m.get("llama.attention.head_count_kv"));
    }

    @Test
    public void qwen2YarnWritesTheYarnKeysInOrder() {
        HfMetadata m = generic(Arch.QWEN2, "Qwen2ForCausalLM", "'architectures': ['Qwen2ForCausalLM'],"
                + " 'hidden_size': 896, 'intermediate_size': 4864, 'num_attention_heads': 14,"
                + " 'num_key_value_heads': 2, 'num_hidden_layers': 24, 'max_position_embeddings': 32768,"
                + " 'rms_norm_eps': 1e-06, 'rope_theta': 1000000.0, 'rope_scaling': {'type': 'yarn',"
                + " 'factor': 4.0, 'original_max_position_embeddings': 32768, 'attention_factor': 0.5,"
                + " 'beta_fast': 32, 'beta_slow': 1, 'truncate': false}");
        assertEquals(Arrays.asList("qwen2.block_count", "qwen2.context_length", "qwen2.embedding_length",
                "qwen2.feed_forward_length", "qwen2.attention.head_count", "qwen2.attention.head_count_kv",
                "qwen2.rope.scaling.type", "qwen2.rope.scaling.factor", "qwen2.rope.scaling.original_context_length",
                "qwen2.rope.scaling.yarn_truncate", "qwen2.rope.scaling.yarn_attn_factor",
                "qwen2.rope.scaling.yarn_beta_fast", "qwen2.rope.scaling.yarn_beta_slow", "qwen2.rope.freq_base",
                "qwen2.attention.layer_norm_rms_epsilon"), keys(m));
        assertEquals(24, m.get("qwen2.block_count"));
        assertEquals("yarn", m.get("qwen2.rope.scaling.type"));
        assertEquals(GGUFValueType.STRING, m.type("qwen2.rope.scaling.type"));
        assertEquals(4f, m.get("qwen2.rope.scaling.factor"));
        assertEquals(32768, m.get("qwen2.rope.scaling.original_context_length"));
        assertEquals(GGUFValueType.UINT32, m.type("qwen2.rope.scaling.original_context_length"));
        assertEquals(false, m.get("qwen2.rope.scaling.yarn_truncate"));
        assertEquals(GGUFValueType.BOOL, m.type("qwen2.rope.scaling.yarn_truncate"));
        assertEquals(0.5f, m.get("qwen2.rope.scaling.yarn_attn_factor"));
        assertEquals(32f, m.get("qwen2.rope.scaling.yarn_beta_fast"));
        assertEquals(1f, m.get("qwen2.rope.scaling.yarn_beta_slow"));
        assertEquals(1e6f, m.get("qwen2.rope.freq_base"));
        assertEquals(1e-6f, m.get("qwen2.attention.layer_norm_rms_epsilon"));
    }

    @Test
    public void yarnWithoutOriginalContextIsRejected() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> llama("'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0}"));
        assertEquals("config.json: required key [rope_params.original_max_position_embeddings] is missing",
                e.getMessage());
    }

    @Test
    public void yarnTakesTheRootOriginalContextThroughTheMirror() {
        HfMetadata m = llama("'original_max_position_embeddings': 8192,"
                + " 'rope_scaling': {'rope_type': 'yarn', 'factor': 2.0}");
        assertEquals(8192, m.get("llama.rope.scaling.original_context_length"));
    }

    @Test
    public void linearScalingWritesTypeAndFactor() {
        HfMetadata m = llama("'rope_scaling': {'type': 'linear', 'factor': 2.0}");
        assertEquals("linear", m.get("llama.rope.scaling.type"));
        assertEquals(2f, m.get("llama.rope.scaling.factor"));
        assertFalse(m.has("llama.rope.scaling.original_context_length"));
    }

    @Test
    public void linearWithoutFactorAndDelegatedTypesWriteNoScaling() {
        for (String scaling : Arrays.asList("{'rope_type': 'linear'}", "{'rope_type': 'dynamic', 'factor': 2.0}",
                "{'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0,"
                        + " 'original_max_position_embeddings': 8192}",
                "{'rope_type': 'default'}")) {
            HfMetadata m = llama("'rope_scaling': " + scaling);
            assertFalse(scaling, m.has("llama.rope.scaling.type"));
            assertFalse(scaling, m.has("llama.rope.scaling.factor"));
        }
    }

    @Test
    public void longropeWritesTheTypeOnly() {
        for (String type : Arrays.asList("su", "longrope")) {
            HfMetadata m = llama("'rope_scaling': {'type': '" + type + "', 'long_factor': [1.0, 2.0],"
                    + " 'short_factor': [1.0, 1.0]}");
            assertEquals("longrope", m.get("llama.rope.scaling.type"));
            assertFalse(m.has("llama.rope.scaling.factor"));
        }
    }

    @Test
    public void mropeSectionsArePaddedToFour() {
        HfMetadata m = llama("'rope_scaling': {'type': 'mrope', 'mrope_section': [16, 24, 24]}");
        assertEquals(Arrays.asList(16, 24, 24, 0), m.get("llama.rope.dimension_sections"));
        assertEquals(GGUFValueType.ARRAY, m.type("llama.rope.dimension_sections"));
        assertEquals(GGUFValueType.INT32, m.elementType("llama.rope.dimension_sections"));
        assertFalse(m.has("llama.rope.scaling.type"));
        HfMetadata five = llama("'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 2, 3, 4, 5]}");
        assertEquals(Arrays.asList(1, 2, 3, 4), five.get("llama.rope.dimension_sections"));
    }

    @Test
    public void slidingThetaIsWrittenAfterTheGlobalTheta() {
        HfMetadata m = llama("'rope_theta': 1000000.0, 'rope_local_base_freq': 10000.0");
        List<String> k = keys(m);
        assertEquals(k.indexOf("llama.rope.freq_base") + 1, k.indexOf("llama.rope.freq_base_swa"));
        assertEquals(1e6f, m.get("llama.rope.freq_base"));
        assertEquals(1e4f, m.get("llama.rope.freq_base_swa"));
    }

    @Test
    public void nonCausalFlagsWriteCausalFalse() {
        for (String flag : Arrays.asList("'is_causal': false", "'use_bidirectional_attention': true")) {
            HfMetadata m = llama(flag);
            List<String> k = keys(m);
            assertEquals(flag, false, m.get("llama.attention.causal"));
            assertEquals(GGUFValueType.BOOL, m.type("llama.attention.causal"));
            assertEquals(k.indexOf("llama.attention.head_count_kv") + 1, k.indexOf("llama.attention.causal"));
        }
        for (String flag : Arrays.asList("'is_causal': true", "'use_bidirectional_attention': false")) {
            assertFalse(flag, llama(flag).has("llama.attention.causal"));
        }
    }

    @Test
    public void expertKeysAndGatingFunctions() {
        HfMetadata m = llama("'num_experts': 64, 'num_experts_per_tok': 8, 'n_group': 8, 'topk_group': 4,"
                + " 'scoring_func': 'sigmoid'");
        assertEquals(64, m.get("llama.expert_count"));
        assertEquals(8, m.get("llama.expert_used_count"));
        assertEquals(GGUFValueType.UINT32, m.type("llama.expert_used_count"));
        assertEquals(8, m.get("llama.expert_group_count"));
        assertEquals(4, m.get("llama.expert_group_used_count"));
        assertEquals(2, m.get("llama.expert_gating_func"));
        assertEquals(GGUFValueType.UINT32, m.type("llama.expert_gating_func"));
        List<String> k = keys(m);
        assertEquals(Arrays.asList("llama.expert_count", "llama.expert_used_count", "llama.expert_group_count",
                "llama.expert_group_used_count", "llama.expert_gating_func"),
                k.subList(k.indexOf("llama.expert_count"), k.size()));
        assertEquals(1, llama("'score_func': 'softmax'").get("llama.expert_gating_func"));
        assertEquals(4, llama("'router_score_function': 'sqrtsoftplus'").get("llama.expert_gating_func"));
        assertFalse(llama("'scoring_func': null").has("llama.expert_gating_func"));
    }

    @Test
    public void unsupportedScoreFunctionsAreRejected() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> llama("'scoring_func': 'relu'"));
        assertEquals("Unsupported expert score gating function value: relu", e.getMessage());
    }

    @Test
    public void wrongValueTypesNameTheAliases() {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> llama("'n_group': 5000000000"));
        assertEquals("config.json: [n_group]: metadata key 'llama.expert_group_count' value 5000000000 is outside"
                + " [0, 4294967295]", e.getMessage());
    }
}

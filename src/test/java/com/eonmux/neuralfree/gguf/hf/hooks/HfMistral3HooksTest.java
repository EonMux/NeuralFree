package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link HfMistral3Hooks} against {@code Mistral3Model} and its three inner implementations
 * (LC/conversion/mistral3.py:21-100): the {@code __init__} dispatch to MISTRAL4 and the non-causal Nemotron embedding
 * branch, the required YaRN log multiplier and attention temperature of Ministral 3 (mistral3.py:25-32), the
 * {@code realtime_delay_tokens} of VoxtralRealtime (mistral3.py:98-100) and the embedding branch's
 * {@code add_token(False)} vocabulary (mistral3.py:37-40).
 */
public class HfMistral3HooksTest {

    private static final String LLAMA_BASE = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-05, 'vocab_size': 8, 'rope_theta': 10000.0";
    /** The YaRN scaling of a Ministral 3 configuration, with the two required members. */
    private static final String YARN = "'rope_scaling': {'rope_type': 'yarn', 'factor': 8.0,"
            + " 'original_max_position_embeddings': 8, 'mscale_all_dim': 1.0, 'llama_4_scaling_beta': 0.5}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ the dispatch

    /** {@code model_type} mistral4 is MISTRAL4; the headless roots are the embedding branch; text is MISTRAL3. */
    @Test
    public void archFollowsTheImplementationChoice() {
        assertEquals(Arch.MISTRAL4, HfMistral3Hooks.arch(config("{'architectures': ['Mistral3Model'],"
                + LLAMA_BASE + ", 'model_type': 'mistral4'}"), null));
        assertEquals(Arch.MISTRAL3, HfMistral3Hooks.arch(config(
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + "}"), null));
        assertEquals(Arch.MISTRAL3, HfMistral3Hooks.arch(config(
                "{'architectures': ['Mistral3ForConditionalGeneration'], " + LLAMA_BASE + "}"), null));
    }

    @Test
    public void theEmbeddingBranchRefusesACausalCheckpoint() {
        for (String root : new String[] {HfMistral3Hooks.MINISTRAL3_HEADLESS, HfMistral3Hooks.MISTRAL3_HEADLESS}) {
            assertThrows(GgufFormatException.class,
                    () -> HfMistral3Hooks.arch(config("{'architectures': ['" + root + "'], " + LLAMA_BASE + "}"),
                            null));
            assertThrows(GgufFormatException.class, () -> HfMistral3Hooks.arch(config(
                    "{'architectures': ['" + root + "'], " + LLAMA_BASE + ", 'is_causal': true}"), null));
        }
        assertEquals(HfMistral3Hooks.MODEL_TYPE_MISTRAL4, "mistral4");
        assertEquals(HfMistral3Hooks.MODEL_TYPE_MINISTRAL3, "ministral3");
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void ministral3KvAddsTheYarnLogMulAndTheAttentionTemperature() throws IOException {
        HfLoadContext ctx = context(Arch.MISTRAL3, "Ministral3ForCausalLM",
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3',"
                        + YARN + "}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MISTRAL3.ggufName());
        HfMistral3Hooks.ministral3Kv(ctx, kv);
        HfMetadata m = kv.build();
        assertEquals(1.0f, num(m.get(String.format(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, Arch.MISTRAL3.ggufName()))),
                0f);
        assertEquals(0.5f, num(m.get(String.format(GgufKeys.ATTENTION_TEMPERATURE_SCALE,
                Arch.MISTRAL3.ggufName()))), 0f);
    }

    @Test
    public void nonMinistral3ConfigurationsGetNoExtraKeys() throws IOException {
        HfLoadContext ctx = context(Arch.MISTRAL3, "VoxtralRealtimeForConditionalGeneration",
                "{'architectures': ['VoxtralRealtimeForConditionalGeneration'], " + LLAMA_BASE + "}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MISTRAL3.ggufName());
        HfMistral3Hooks.ministral3Kv(ctx, kv);
        assertNull(kv.build().get(String.format(GgufKeys.ATTENTION_TEMPERATURE_SCALE, Arch.MISTRAL3.ggufName())));
    }

    @Test
    public void ministral3RefusesMissingOrWrongRopeTypes() {
        for (String bad : new String[] {
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3'}",
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3',"
                        + " 'rope_scaling': {'rope_type': 'linear', 'factor': 8.0}}",
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3',"
                        + " 'rope_scaling': {'rope_type': 'yarn', 'factor': 8.0, 'llama_4_scaling_beta': 0.5}}",
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3',"
                        + " 'rope_scaling': {'rope_type': 'yarn', 'factor': 8.0, 'mscale_all_dim': 1.0}}"}) {
            HfLoadContext ctx = context(Arch.MISTRAL3, "Ministral3ForCausalLM", bad);
            try {
                HfMistral3Hooks.ministral3Kv(ctx, new HfMetadataBuilder(Arch.MISTRAL3.ggufName()));
                fail("the configuration must be refused");
            } catch (GgufFormatException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("ministral3")
                        || expected.getMessage().contains("rope"));
            }
        }
    }

    @Test
    public void metaComposesTheLlamaMetadataAndTheMinistral3Keys() throws IOException {
        HfLoadContext ctx = context(Arch.MISTRAL3, "Ministral3ForCausalLM",
                "{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE + ", 'model_type': 'ministral3',"
                        + YARN + "}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MISTRAL3.ggufName());
        HfMistral3Hooks.meta(ctx, kv, null);
        HfMetadata m = kv.build();
        String g = Arch.MISTRAL3.ggufName();
        assertEquals(2, num(m.get(String.format(GgufKeys.ATTENTION_HEAD_COUNT, g))));
        assertEquals(1, num(m.get(String.format(GgufKeys.ATTENTION_HEAD_COUNT_KV, g))));
        assertEquals(1e-05f, num(m.get(String.format(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, g))), 1e-9f);
        assertEquals(1.0f, num(m.get(String.format(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, g))), 0f);
        assertNull(m.get(String.format(GgufKeys.POOLING_TYPE, g)));
    }

    // ------------------------------------------------------------------ VoxtralRealtimeModel

    @Test
    public void voxtralMetaWritesTheRealtimeDelayTokens() throws IOException {
        HfLoadContext ctx = context(Arch.MISTRAL3, "VoxtralRealtimeForConditionalGeneration",
                "{'architectures': ['VoxtralRealtimeForConditionalGeneration'], " + LLAMA_BASE
                        + ", 'default_num_delay_tokens': 8}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.MISTRAL3.ggufName());
        HfMistral3Hooks.voxtralMeta(ctx, kv, null);
        assertEquals(8, num(kv.build().get(String.format(GgufKeys.REALTIME_DELAY_TOKENS,
                Arch.MISTRAL3.ggufName()))));
    }

    // ------------------------------------------------------------------ the embedding branch vocabulary

    @Test
    public void theEmbeddingBranchVocabTurnsBothSpecialTokensOff() throws Exception {
        String config = "{'architectures': ['Mistral3Model'], " + LLAMA_BASE + ", 'is_causal': false}";
        Map<String, Object> embed = vocab(config, Arch.MISTRAL3);
        assertEquals(Boolean.FALSE, embed.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertEquals(Boolean.FALSE, embed.get(GgufKeys.TOKENIZER_ADD_EOS));
        Map<String, Object> text = vocab("{'architectures': ['Ministral3ForCausalLM'], " + LLAMA_BASE
                + ", 'model_type': 'ministral3', " + YARN + "}", Arch.MISTRAL3);
        assertFalse(text.containsKey(GgufKeys.TOKENIZER_ADD_BOS));
        assertFalse(text.containsKey(GgufKeys.TOKENIZER_ADD_EOS));
    }

    // ------------------------------------------------------------------ helpers

    /** The vocabulary of {@code configs} over a distributed GGUF tokenizer, as the assembler builds it. */
    private Map<String, Object> vocab(String configs, Arch arch) throws Exception {
        Path dir = tmp.newFolder().toPath();
        Path gguf = tmp.getRoot().toPath().resolve("tokenizer.gguf");
        new GgufTestWriter().architecture("llama").str(GgufKeys.TOKENIZER_MODEL, "llama")
                .strArray(GgufKeys.TOKENIZER_LIST, "<unk>", "<s>", "</s>").writeTo(gguf);
        Files.writeString(dir.resolve("config.json"), configs.replace('\'', '"'), StandardCharsets.UTF_8);
        HfConfig cfg = config(configs);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, arch)
                .tokenizerFiles(HfTokenizerFiles.locate(dir).withTokenizerDir(gguf))
                .build();
        return HfMistral3Hooks.vocab(ctx, (tj, encode) -> "llama");
    }

    private static HfConfig config(String singleQuotedJson) {
        return HfConfigLoader.fromJson(Path.of("."), singleQuotedJson.replace('\'', '"'));
    }

    private HfLoadContext context(Arch arch, String hfArch, String members) {
        HfConfig cfg = config(members);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        Arch resolved = entry.arch().resolve(cfg, HfTensorIndex.of(Map.of()));
        return HfLoadContext.builder(cfg, entry, resolved).build();
    }

    private static float num(Object v) {
        return ((Number) v).floatValue();
    }
}

package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The name, configuration and side-file hooks of the vLLM aliases: {@link HfVllmAliasHooks}, the
 * {@link HfMistralLarge3Hooks} renames, and the vLLM names of the Clef and Decision 3.0 checkpoints
 * ({@link HfEncoderHooks#clefFixup}, {@link HfDecisionHooks#vllmD3Fixup}).
 */
public class HfVllmAliasHooksTest {

    private static final UnaryOperator<String> IDENTITY = n -> n;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HfConfig config(String singleQuoted) throws IOException {
        return HfConfigLoader.fromJson(tmp.getRoot().toPath(), SyntheticHf.json(singleQuoted));
    }

    private void sideFile(String name, String singleQuoted) throws IOException {
        Files.write(tmp.getRoot().toPath().resolve(name),
                SyntheticHf.json(singleQuoted).getBytes(StandardCharsets.UTF_8));
    }

    @Test
    public void headlessNamesTakeTheModelPrefix() {
        assertEquals("model.layers.0.mlp.up_proj.weight",
                HfVllmAliasHooks.withModelPrefix("layers.0.mlp.up_proj.weight"));
        assertEquals("model.embed_tokens.weight", HfVllmAliasHooks.withModelPrefix("embed_tokens.weight"));
        assertEquals("model.norm.weight", HfVllmAliasHooks.withModelPrefix("model.norm.weight"));
        assertEquals("lm_head.weight", HfVllmAliasHooks.withModelPrefix("lm_head.weight"));

        HfNameFilter filter = HfVllmAliasHooks.headless(HfDenseHooksA.GEMMA_FILTER);
        assertEquals("model.layers.1.self_attn.q_proj.weight",
                filter.filter("layers.1.self_attn.q_proj.weight", IDENTITY));
        assertNull("the inner filter still drops the tied head", filter.filter("lm_head.weight", IDENTITY));
        assertThrows(NullPointerException.class, () -> HfVllmAliasHooks.headless(null));
    }

    @Test
    public void mimoMtpLayersBecomeTheAppendedNextnBlocks() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.getRoot().toPath(), Arch.QWEN2, "MiMoForCausalLM",
                "'num_nextn_predict_layers':1", new java.util.LinkedHashMap<>(), true);
        HfNameFilter filter = HfVllmAliasHooks.mimoFilter(ctx);
        assertEquals("model.layers.2.enorm.weight",
                filter.filter("model.mtp_layers.0.token_layernorm.weight", IDENTITY));
        assertEquals("model.layers.2.hnorm.weight",
                filter.filter("model.mtp_layers.0.hidden_layernorm.weight", IDENTITY));
        assertEquals("model.layers.2.eh_proj.weight",
                filter.filter("model.mtp_layers.0.input_proj.weight", IDENTITY));
        assertEquals("model.layers.2.shared_head.norm.weight",
                filter.filter("model.mtp_layers.0.final_layernorm.weight", IDENTITY));
        assertEquals("model.layers.2.self_attn.q_proj.weight",
                filter.filter("model.mtp_layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("model.layers.2.input_layernorm.weight",
                filter.filter("model.mtp_layers.0.input_layernorm.weight", IDENTITY));
        assertEquals("model.layers.0.input_layernorm.weight",
                filter.filter("model.layers.0.input_layernorm.weight", IDENTITY));
    }

    @Test
    public void mimoMtpLayersDropWithNoMtp() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.getRoot().toPath(), Arch.QWEN2, "MiMoForCausalLM",
                "'num_nextn_predict_layers':1", new java.util.LinkedHashMap<>(), false);
        HfNameFilter filter = HfVllmAliasHooks.mimoFilter(ctx);
        assertNull(filter.filter("model.mtp_layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                filter.filter("model.layers.0.self_attn.q_proj.weight", IDENTITY));
    }

    @Test
    public void mimoRefusesMtpTensorsTheLayerDoesNotDefine() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.getRoot().toPath(), Arch.QWEN2, "MiMoForCausalLM",
                "'num_nextn_predict_layers':1", new java.util.LinkedHashMap<>(), true);
        HfNameFilter filter = HfVllmAliasHooks.mimoFilter(ctx);
        GgufFormatException unknown = assertThrows(GgufFormatException.class,
                () -> filter.filter("model.mtp_layers.0.mtp_block.self_attn.q_proj.weight", IDENTITY));
        assertTrue(unknown.getMessage(), unknown.getMessage().contains("mtp_block.self_attn.q_proj.weight"));
        GgufFormatException beyond = assertThrows(GgufFormatException.class,
                () -> filter.filter("model.mtp_layers.1.self_attn.q_proj.weight", IDENTITY));
        assertTrue(beyond.getMessage(), beyond.getMessage().contains("num_nextn_predict_layers"));
    }

    @Test
    public void mimoRefusesMtpTensorsWithoutADeclaredCount() throws IOException {
        HfLoadContext ctx = HfQwenHooksTest.context(tmp.getRoot().toPath(), Arch.QWEN2, "MiMoForCausalLM",
                "", new java.util.LinkedHashMap<>(), true);
        HfNameFilter filter = HfVllmAliasHooks.mimoFilter(ctx);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> filter.filter("model.mtp_layers.0.self_attn.q_proj.weight", IDENTITY));
        assertTrue(e.getMessage(), e.getMessage().contains("num_nextn_predict_layers"));
    }

    @Test
    public void teleChat2NamesBecomeLlamaNames() {
        assertEquals("model.embed_tokens.weight",
                HfVllmAliasHooks.teleChat2Filter("transformer.word_embeddings.weight", IDENTITY));
        assertEquals("model.layers.3.self_attn.q_proj.weight",
                HfVllmAliasHooks.teleChat2Filter("transformer.h.3.self_attention.query.weight", IDENTITY));
        assertEquals("model.layers.3.self_attn.key_value.weight",
                HfVllmAliasHooks.teleChat2Filter("transformer.h.3.self_attention.key_value.weight", IDENTITY));
        assertEquals("model.layers.3.self_attn.o_proj.bias",
                HfVllmAliasHooks.teleChat2Filter("transformer.h.3.self_attention.dense.bias", IDENTITY));
        assertEquals("model.layers.3.mlp.down_proj.bias",
                HfVllmAliasHooks.teleChat2Filter("transformer.h.3.mlp.down_proj.bias", IDENTITY));
        assertEquals("model.norm.weight", HfVllmAliasHooks.teleChat2Filter("transformer.ln_f.weight", IDENTITY));
        assertEquals("lm_head.weight", HfVllmAliasHooks.teleChat2Filter("lm_head.weight", IDENTITY));
    }

    @Test
    public void teleChat2ConfigurationTakesTheLlamaKeys() throws IOException {
        HfConfig cfg = HfVllmAliasHooks.teleChat2Fixup(config("{'architectures':['TeleChat2ForCausalLM'],"
                + "'hidden_size':8,'n_layer':3,'n_head':2,'ffn_hidden_size':16,'layer_norm_epsilon':1e-5,"
                + "'hidden_act':'gelu'}"));
        assertEquals(3, cfg.requireInt("num_hidden_layers"));
        assertEquals(2, cfg.requireInt("num_attention_heads"));
        assertEquals(16, cfg.requireInt("intermediate_size"));
        assertEquals(1e-5, cfg.findFloat("rms_norm_eps").getAsDouble(), 0.0);
        assertTrue("the epsilon is an RMSNorm epsilon", !cfg.findFloat("layer_norm_epsilon").isPresent());
        assertEquals("silu", cfg.flat().get("hidden_act"));
    }

    @Test
    public void mistralLarge3NativeNamesBecomeDeepseekNames() {
        assertEquals("model.layers.4.input_layernorm.weight",
                HfMistralLarge3Hooks.rename("layers.4.attention_norm.weight"));
        assertEquals("model.layers.4.self_attn.q_a_proj.weight",
                HfMistralLarge3Hooks.rename("layers.4.attention.wq_a.weight"));
        assertEquals("model.layers.4.self_attn.kv_a_proj_with_mqa.input_scale",
                HfMistralLarge3Hooks.rename("layers.4.attention.wkv_a_with_mqa.qscale_act"));
        assertEquals("model.layers.4.self_attn.kv_b_proj.weight_scale",
                HfMistralLarge3Hooks.rename("layers.4.attention.wkv_b.qscale_weight"));
        assertEquals("model.layers.4.self_attn.o_proj.weight", HfMistralLarge3Hooks.rename("layers.4.attention.wo.weight"));
        assertEquals("model.layers.4.post_attention_layernorm.weight",
                HfMistralLarge3Hooks.rename("layers.4.ffn_norm.weight"));
        assertEquals("model.layers.4.mlp.gate.weight", HfMistralLarge3Hooks.rename("layers.4.gate.weight"));
        assertEquals("model.layers.4.mlp.shared_experts.down_proj.weight",
                HfMistralLarge3Hooks.rename("layers.4.shared_experts.w2.weight"));
        assertEquals("model.layers.4.mlp.experts.17.up_proj.weight",
                HfMistralLarge3Hooks.rename("layers.4.experts.17.w3.weight"));
        assertEquals("model.layers.4.mlp.down_proj.weight", HfMistralLarge3Hooks.rename("layers.4.feed_forward.w2.weight"));
        assertEquals("model.embed_tokens.weight", HfMistralLarge3Hooks.rename("tok_embeddings.weight"));
        assertEquals("model.norm.weight", HfMistralLarge3Hooks.rename("norm.weight"));
        assertEquals("lm_head.weight", HfMistralLarge3Hooks.rename("output.weight"));
        assertEquals("the Hugging Face names pass unchanged", "model.layers.4.self_attn.q_a_proj.weight",
                HfMistralLarge3Hooks.rename("model.layers.4.self_attn.q_a_proj.weight"));
        assertEquals("model.layers.4.self_attn.o_proj.input_scale",
                HfMistralLarge3Hooks.rename("model.layers.4.self_attn.o_proj.qscale_act"));
    }

    @Test
    public void clefFlashTakesTheInlineJointHeadConfiguration() throws IOException {
        HfConfig cfg = HfEncoderHooks.clefFixup(config("{'architectures':['ClefFlashForJointSchema'],"
                + "'joint_head_config':{'num_labels':3}}"));
        assertEquals(3L, ((Map<?, ?>) cfg.flat().get("decision")).get("num_labels"));
    }

    @Test
    public void clefFlashPrefersTheSideFile() throws IOException {
        sideFile("joint_head_config.json", "{'num_labels':5}");
        HfConfig cfg = HfEncoderHooks.clefFixup(config("{'architectures':['ClefFlashForJointSchema'],"
                + "'joint_head_config':{'num_labels':3}}"));
        assertEquals(5L, ((Map<?, ?>) cfg.flat().get("decision")).get("num_labels"));
    }

    @Test
    public void clefWithoutAJointHeadConfigurationIsRefused() throws IOException {
        HfConfig flash = config("{'architectures':['ClefFlashForJointSchema']}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfEncoderHooks.clefFixup(flash));
        assertTrue(e.getMessage(), e.getMessage().contains("no joint_head_config.json and no inline"));
        HfConfig clef = config("{'architectures':['ClefModel'],'joint_head_config':{'num_labels':3}}");
        e = assertThrows(GgufFormatException.class, () -> HfEncoderHooks.clefFixup(clef));
        assertTrue("ClefModel reads the side file only: " + e.getMessage(),
                e.getMessage().endsWith("has no joint_head_config.json"));
        HfConfig bad = config("{'architectures':['ClefFlashForJointSchema'],'joint_head_config':[1]}");
        assertThrows(GgufFormatException.class, () -> HfEncoderHooks.clefFixup(bad));
    }

    @Test
    public void d3TakesTheDecisionConfiguration() throws IOException {
        HfConfig inline = HfDecisionHooks.vllmD3Fixup(config("{'architectures':['D3ForSequenceClassification'],"
                + "'decision_config':{'decision_type':'choice'}}"));
        assertEquals("choice", ((Map<?, ?>) inline.flat().get("decision")).get("decision_type"));

        sideFile("decision_config.json", "{'decision_type':'score'}");
        HfConfig side = HfDecisionHooks.vllmD3Fixup(config("{'architectures':['D3ForSequenceClassification'],"
                + "'decision_config':{'decision_type':'choice'}}"));
        assertEquals("score", ((Map<?, ?>) side.flat().get("decision")).get("decision_type"));

        HfConfig detected = config("{'architectures':['VllmD3Model'],'decision':{'decision_type':'noul'}}");
        assertSame("a configuration with its decision block is unchanged", detected,
                HfDecisionHooks.vllmD3Fixup(detected));
    }

    @Test
    public void d3WithoutADecisionConfigurationIsRefused() throws IOException {
        HfConfig none = config("{'architectures':['D3ForSequenceClassification']}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfDecisionHooks.vllmD3Fixup(none));
        assertTrue(e.getMessage(), e.getMessage().contains("no decision_config.json and no inline"));
        HfConfig bad = config("{'architectures':['D3ForSequenceClassification'],'decision_config':'x'}");
        assertThrows(GgufFormatException.class, () -> HfDecisionHooks.vllmD3Fixup(bad));
    }

    @Test
    public void d3RefusesAnInvalidSideFile() throws IOException {
        Path file = tmp.getRoot().toPath().resolve("decision_config.json");
        Files.write(file, "[1, 2]".getBytes(StandardCharsets.UTF_8));
        HfConfig cfg = config("{'architectures':['D3ForSequenceClassification']}");
        assertThrows(GgufFormatException.class, () -> HfDecisionHooks.vllmD3Fixup(cfg));
    }
}

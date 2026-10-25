package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfConfigLoader}, {@link HfConfig} and {@link HfCheckpointDetectors}: text_config flattening, alias lookup,
 * overrides, sub-config renames, architecture detection, Mistral native translation, every checkpoint detector on
 * a synthetic directory, the content-name rule and the config caps.
 */
public class HfConfigLoaderTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** {@code json} with single quotes turned into double quotes. */
    static String json(String json) {
        return json.replace('\'', '"');
    }

    static void write(Path file, String content) throws IOException {
        Files.createDirectories(file.getParent());
        Files.write(file, json(content).getBytes(StandardCharsets.UTF_8));
    }

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    private static HfConfig parse(String config) {
        return HfConfigLoader.fromJson(Path.of("."), json(config));
    }

    // ------------------------------------------------------------------ flattening and lookups

    @Test
    public void textConfigWinsInTheFlattenedView() {
        HfConfig cfg = parse("{'architectures':['LlavaForConditionalGeneration'],'hidden_size':1,'model_type':'llava',"
                + "'text_config':{'hidden_size':2,'vocab_size':10}}");
        assertEquals(2L, cfg.flat().get("hidden_size"));
        assertEquals(1L, cfg.raw().raw("hidden_size"));
        assertEquals(10, cfg.findInt("vocab_size").getAsInt());
        assertEquals("llava", cfg.modelType());
        assertEquals(2L, cfg.text().raw("hidden_size"));
        assertThrows(UnsupportedOperationException.class, () -> cfg.flat().put("x", 1L));
    }

    @Test
    public void firstPresentAliasWinsAndNullEndsTheSearch() {
        HfConfig cfg = parse("{'architectures':['A'],'n_layers':3,'num_hidden_layers':5,'n_head':null,'num_heads':4}");
        assertEquals(3, cfg.blockCount());
        assertEquals(5, cfg.findInt("num_hidden_layers", "n_layers").getAsInt());
        assertFalse(cfg.find("n_head", "num_heads").isPresent());
        assertEquals(7, cfg.findInt(7, "n_head", "num_heads"));
        assertEquals(4, cfg.findInt("missing", "num_heads").getAsInt());
        assertEquals("x", cfg.findStringOr("x", "missing"));
        assertThrows(GgufFormatException.class, () -> cfg.requireInt("missing"));
    }

    @Test
    public void typedLookupsConvertAndCheckTypes() {
        HfConfig cfg = parse("{'architectures':['A'],'rms_norm_eps':1e-6,'hidden_size':4096.0,'tie':true,"
                + "'act':'silu','sizes':[1,2,3]}");
        assertEquals(1e-6, cfg.findFloat("rms_norm_eps").getAsDouble(), 0);
        assertEquals(4096, cfg.findInt("hidden_size").getAsInt());
        assertEquals(Boolean.TRUE, cfg.findBool("tie").get());
        assertEquals("silu", cfg.findString("act").get());
        assertEquals(Arrays.asList(1, 2, 3), cfg.findIntList("sizes").get());
        assertThrows(GgufFormatException.class, () -> cfg.findInt("act"));
        assertThrows(GgufFormatException.class, () -> cfg.findInt("rms_norm_eps"));
        assertThrows(GgufFormatException.class, () -> cfg.findBool("act"));
        assertThrows(GgufFormatException.class, () -> cfg.findIntList("hidden_size"));
    }

    @Test
    public void withFlatReplacesFlatValuesOnly() {
        HfConfig cfg = parse("{'architectures':['A'],'hidden_size':8,'rope_theta':100.0}");
        Map<String, Object> replacements = new LinkedHashMap<>();
        replacements.put("hidden_size", 16L);
        replacements.put("rope_theta", 500.0);
        HfConfig copy = cfg.withFlat(replacements);
        assertEquals(16, copy.findInt("hidden_size").getAsInt());
        assertEquals(8L, copy.raw().raw("hidden_size"));
        assertEquals(500.0, copy.rope().theta().getAsDouble(), 0);
        assertEquals(8, cfg.findInt("hidden_size").getAsInt());
    }

    @Test
    public void configCapsNameTheFileAndKey() {
        assertCap("{'architectures':['A'],'vocab_size':16777217}", "vocab_size");
        assertCap("{'architectures':['A'],'num_hidden_layers':4097}", "num_hidden_layers");
        assertCap("{'architectures':['A'],'num_experts':65537}", "num_experts");
        assertCap("{'architectures':['A'],'num_attention_heads':65537}", "num_attention_heads");
        assertCap("{'architectures':['A'],'hidden_size':1048577}", "hidden_size");
        assertCap("{'architectures':['A'],'text_config':{'intermediate_size':-1}}", "text_config.intermediate_size");
        HfConfig ok = parse("{'architectures':['A'],'vocab_size':16777216,'max_position_embeddings':100000000,"
                + "'sliding_window':100000000}");
        assertEquals(16777216, ok.findInt("vocab_size").getAsInt());
        assertEquals(100000000, ok.findInt("max_position_embeddings").getAsInt());
        assertEquals(100000000, ok.findInt("sliding_window").getAsInt());
    }

    private static void assertCap(String config, String key) {
        HfConfig cfg = parse(config);
        String bare = key.substring(key.lastIndexOf('.') + 1);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> cfg.findInt(bare));
        assertTrue(e.getMessage(), e.getMessage().contains("config.json") && e.getMessage().contains("'" + key + "'"));
    }

    // ------------------------------------------------------------------ architecture and renames

    @Test
    public void architectureFollowsTheConverter() {
        assertEquals("LlamaForCausalLM", parse("{'architectures':['LlamaForCausalLM']}").architecture());
        assertEquals("Qwen2ForCausalLM", parse("{'architectures':['Wrapper'],"
                + "'text_config':{'architectures':['Qwen2ForCausalLM']}}").architecture());
        assertEquals("NEOChatModel", parse("{'architectures':['NEOChatModel'],"
                + "'text_config':{'architectures':['Qwen3ForCausalLM']}}").architecture());
        assertEquals("MambaForCausalLM", parse("{'ssm_cfg':{}}").architecture());
        assertEquals("Mamba2ForCausalLM", parse("{'ssm_cfg':{'layer':'Mamba2'}}").architecture());
        assertThrows(GgufFormatException.class, () -> parse("{'hidden_size':1}"));
        assertThrows(GgufFormatException.class, () -> parse("{'architectures':'Llama'}"));
    }

    @Test
    public void subConfigsAreCopiedToTextConfigInOrder() {
        HfConfig llm = parse("{'architectures':['InternVL'],'llm_config':{'hidden_size':7}}");
        assertEquals(7L, llm.flat().get("hidden_size"));
        assertTrue(llm.raw().raw("llm_config") instanceof Map);
        HfConfig thinker = parse("{'architectures':['Qwen2_5OmniModel'],"
                + "'thinker_config':{'text_config':{'hidden_size':9},'vision_config':{'depth':2},"
                + "'audio_config':{'d_model':5}}}");
        assertEquals(9L, thinker.flat().get("hidden_size"));
        assertEquals(2L, thinker.vision().get().raw("depth"));
        assertEquals(5L, thinker.audio().get().raw("d_model"));
        HfConfig lfm = parse("{'architectures':['A'],'language_config':{'x':1},'lfm':{'x':2}}");
        assertEquals(2L, lfm.flat().get("x"));
        assertThrows(GgufFormatException.class, () -> parse("{'architectures':['A'],'llm_config':3}"));
        assertThrows(GgufFormatException.class, () -> parse("{'architectures':['A'],'thinker_config':{}}"));
    }

    // ------------------------------------------------------------------ loading

    @Test
    public void overridesDeepMergeAfterRenames() throws IOException {
        Path d = dir();
        write(d.resolve("config.json"), "{'architectures':['A'],'llm_config':{'hidden_size':4,'rope_theta':10.0},"
                + "'rope_scaling':{'rope_type':'linear','factor':2.0}}");
        write(d.resolve("generation_config.json"), "{'eos_token_id':[1,2]}");
        Map<String, Object> textOverride = new LinkedHashMap<>();
        textOverride.put("rope_theta", 99.0);
        Map<String, Object> scaling = new LinkedHashMap<>();
        scaling.put("factor", 4.0);
        Map<String, Object> overrides = new LinkedHashMap<>();
        overrides.put("text_config", textOverride);
        overrides.put("rope_scaling", scaling);
        overrides.put("architectures", List.of("B"));
        HfConfig cfg = HfConfigLoader.load(d, overrides);
        assertEquals("B", cfg.architecture());
        assertEquals(4L, cfg.flat().get("hidden_size"));
        assertEquals(99.0, cfg.rope().theta().getAsDouble(), 0);
        assertEquals("linear", cfg.rope().global().type().get());
        assertEquals(4.0, cfg.rope().global().factor().getAsDouble(), 0);
        assertEquals(List.of(1L, 2L), cfg.generation().get().raw("eos_token_id"));
        assertEquals(d, cfg.dir());
        assertFalse(cfg.mistralNative());

        Map<String, Object> llm = new LinkedHashMap<>();
        llm.put("hidden_size", 6L);
        Map<String, Object> sourceOverride = new LinkedHashMap<>();
        sourceOverride.put("llm_config", llm);
        HfConfig viaSource = HfConfigLoader.load(d, sourceOverride);
        assertEquals(6L, viaSource.flat().get("hidden_size"));
        assertEquals(10.0, viaSource.rope().theta().getAsDouble(), 0);
    }

    @Test
    public void malformedOrMissingFilesAreInvalidCheckpoints() throws IOException {
        Path d = dir();
        IOException missing = assertThrows(IOException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        assertTrue(missing.getMessage().startsWith("Invalid Hugging Face checkpoint"));
        write(d.resolve("config.json"), "{'architectures':['A'],");
        IOException bad = assertThrows(IOException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        assertTrue(bad.getMessage().startsWith("Invalid Hugging Face checkpoint"));
        write(d.resolve("config.json"), "[1]");
        assertThrows(IOException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        assertThrows(IOException.class, () -> HfConfigLoader.load(d.resolve("nope"), (Map<String, Object>) null));
        IOException unreadable = assertThrows(IOException.class,
                () -> HfConfigLoader.readObject(d, d.resolve("absent.json")));
        assertTrue(unreadable.getMessage(), unreadable.getMessage().startsWith("Invalid Hugging Face checkpoint '"
                + d + "': cannot read"));
    }

    @Test
    public void sideFilesFollowTheContentNameRule() throws IOException {
        Path d = dir();
        write(d.resolve("config.json"), "{'architectures':['A']}");
        HfConfig cfg = HfConfigLoader.load(d, (Map<String, Object>) null);
        assertEquals(d.resolve("sub/x.json"), cfg.sideFile("sub/x.json"));
        assertEquals(d.resolve("x.json"), cfg.sideFile("./x.json"));
        for (String bad : new String[]{"../x.json", "a/../../x", "/etc/passwd", "\\x", "C:x", "a\0b", "", "."}) {
            assertThrows(bad, IOException.class, () -> cfg.sideFile(bad));
        }
    }

    // ------------------------------------------------------------------ Mistral native

    @Test
    public void mistralNativeDenseParamsAreTranslated() throws IOException {
        Path d = dir();
        write(d.resolve("params.json"), "{'dim':4096,'n_layers':32,'n_heads':32,'n_kv_heads':8,'head_dim':128,"
                + "'hidden_dim':14336,'norm_eps':1e-5,'rope_theta':1000000.0,'vocab_size':32768,"
                + "'vision_encoder':{'hidden_size':1024}}");
        HfConfig cfg = HfConfigLoader.load(d, (Map<String, Object>) null);
        assertTrue(cfg.mistralNative());
        assertEquals(HfConfigLoader.MISTRAL_NATIVE, cfg.architecture());
        assertEquals(4096, cfg.findInt("hidden_size").getAsInt());
        assertEquals(32, cfg.findInt("num_hidden_layers").getAsInt());
        assertEquals(32, cfg.findInt("num_attention_heads").getAsInt());
        assertEquals(8, cfg.findInt("num_key_value_heads").getAsInt());
        assertEquals(14336, cfg.findInt("intermediate_size").getAsInt());
        assertEquals(1e-5, cfg.findFloat("rms_norm_eps").getAsDouble(), 0);
        assertEquals(128, cfg.findInt("head_dim").getAsInt());
        assertEquals(1e6, cfg.rope().theta().getAsDouble(), 0);
        assertEquals(1024L, cfg.vision().get().raw("hidden_size"));
        assertEquals("params.json", cfg.source());
    }

    @Test
    public void mistralNativeMoeParamsGetTheRemapsAndDefaults() throws IOException {
        Path d = dir();
        write(d.resolve("params.json"), "{'dim':7168,'n_layers':61,'n_heads':128,'n_kv_heads':128,"
                + "'hidden_dim':18432,'norm_eps':1e-6,'vocab_size':129280,'rope_theta':10000.0,"
                + "'yarn':{'factor':40.0,'beta':32.0,'alpha':1.0,'original_max_position_embeddings':4096,"
                + "'apply_scale':true},"
                + "'moe':{'num_experts':256,'num_experts_per_tok':8,'expert_hidden_dim':2048,'route_every_n':1,"
                + "'first_k_dense_replace':3,'routed_scale':2.5,'num_shared_experts':1,'num_expert_groups':8,"
                + "'num_expert_groups_per_tok':4}}");
        HfConfig cfg = HfConfigLoader.load(d, (Map<String, Object>) null);
        assertEquals(HfConfigLoader.MISTRAL_NATIVE_MOE, cfg.architecture());
        assertEquals(256, cfg.findInt("n_routed_experts").getAsInt());
        assertEquals(8, cfg.findInt("num_experts_per_tok").getAsInt());
        assertEquals(2048, cfg.findInt("moe_intermediate_size").getAsInt());
        assertEquals(1, cfg.findInt("moe_layer_freq").getAsInt());
        assertEquals(3, cfg.findInt("first_k_dense_replace").getAsInt());
        assertEquals(2.5, cfg.findFloat("routed_scaling_factor").getAsDouble(), 0);
        assertEquals(1, cfg.findInt("n_shared_experts").getAsInt());
        assertEquals(8, cfg.findInt("n_group").getAsInt());
        assertEquals(4, cfg.findInt("topk_group").getAsInt());
        assertEquals("softmax", cfg.findString("scoring_func").get());
        assertEquals(Boolean.TRUE, cfg.findBool("norm_topk_prob").get());
        assertTrue(cfg.flat().containsKey("topk_method"));
        assertNull(cfg.flat().get("topk_method"));
        assertEquals("transformer", cfg.modelType());
        assertEquals("silu", cfg.findString("hidden_act").get());
        assertEquals(Boolean.FALSE, cfg.findBool("tie_word_embeddings").get());
        assertEquals(128000, cfg.findInt("max_position_embeddings").getAsInt());
    }

    @Test
    public void mistralNativeBlocksAreValidated() throws IOException {
        Path d = dir();
        write(d.resolve("params.json"), "{'dim':8,'moe':{'num_experts':2}}");
        assertThrows(GgufFormatException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        write(d.resolve("params.json"), "{'dim':8,'yarn':{'factor':4.0}}");
        assertThrows(GgufFormatException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        write(d.resolve("params.json"), "{'dim':8,'quantization':'fp8'}");
        assertThrows(GgufFormatException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        write(d.resolve("config.json"), "{'architectures':['MistralForCausalLM']}");
        assertFalse(HfConfigLoader.load(d, (Map<String, Object>) null).mistralNative());
    }

    // ------------------------------------------------------------------ checkpoint detectors

    @Test
    public void detectorNamesKeepTheConverterOrder() {
        assertEquals(List.of("OpenJev", "ModernBertDecision", "Clef", "H2OLightning", "Lev", "Kev", "Nimble", "D1",
                "D1Omni", "PocketTTS", "PplxDecider", "VllmD3", "VllmDecision2"), HfCheckpointDetectors.names());
    }

    @Test
    public void plainDetectorsSetTheirArchitecture() throws IOException {
        Path openJev = dir();
        write(openJev.resolve("helper/shim.py"), "");
        write(openJev.resolve("config.json"), "{'architectures':['Qwen3ForCausalLM'],'hidden_size':8}");
        assertDetected(openJev, "OpenJev", "OpenJevModel");

        Path clef = dir();
        write(clef.resolve("joint_head_config.json"), "{'num_labels':3}");
        write(clef.resolve("config.json"), "{'architectures':['X']}");
        HfConfig c = assertDetected(clef, "Clef", "ClefModel");
        assertEquals(3L, ((Map<?, ?>) c.raw().raw("decision")).get("num_labels"));

        Path h2o = dir();
        write(h2o.resolve("h2o_lightning_shim.py"), "");
        write(h2o.resolve("serve_config.json"), "{}");
        write(h2o.resolve("config.json"), "{'architectures':['X']}");
        assertDetected(h2o, "H2OLightning", "H2OLightningModel");

        Path pplx = dir();
        write(pplx.resolve("decision_config.json"), "{'threshold':0.5}");
        write(pplx.resolve("readout.safetensors"), "");
        write(pplx.resolve("config.json"), "{'architectures':['X']}");
        assertDetected(pplx, "PplxDecider", "PplxDeciderModel");

        Path d3 = dir();
        write(d3.resolve("decision_config.json"), "{'format_id':'d3-code-readout-v1'}");
        write(d3.resolve("config.json"), "{'architectures':['X']}");
        HfConfig v = assertDetected(d3, "VllmD3", "VllmD3Model");
        assertEquals("d3-code-readout-v1", ((Map<?, ?>) v.raw().raw("decision")).get("format_id"));

        Path plain = dir();
        write(plain.resolve("config.json"), "{'architectures':['LlamaForCausalLM']}");
        assertEquals(Optional.empty(), HfCheckpointDetectors.match(plain));
    }

    @Test
    public void modernBertDecisionAppendsHeadLayers() throws IOException {
        Path d = dir();
        write(d.resolve("encoder/config.json"), "{'architectures':['ModernBertModel'],'num_hidden_layers':2,"
                + "'intermediate_size':8,'hidden_size':4}");
        write(d.resolve("rl_agent_config.json"), "{'head_layers':1}");
        HfConfig cfg = assertDetected(d, "ModernBertDecision", "ModernBertDecisionModel");
        assertEquals(3, cfg.blockCount());
        assertEquals(List.of(8L, 8L, 16L), cfg.flat().get("intermediate_size"));
        assertEquals("encoder/config.json", cfg.source());
    }

    @Test
    public void decisionLoraDetectorsReadALocalBaseModel() throws IOException {
        Path lev = dir();
        write(lev.resolve("lev_release.json"), "{}");
        write(lev.resolve("adapter_config.json"), "{'base_model_name_or_path':'base'}");
        write(lev.resolve("base/config.json"), "{'architectures':['Qwen3ForCausalLM'],'hidden_size':16,"
                + "'llm_config':{'hidden_size':1}}");
        HfConfig cfg = assertDetected(lev, "Lev", "LevModel");
        assertEquals(lev.resolve("base"), cfg.baseModelDir().get());
        assertEquals(16L, cfg.flat().get("hidden_size"));

        Path kev = dir();
        Path base = dir();
        write(base.resolve("config.json"), "{'architectures':['X'],'hidden_size':32}");
        write(kev.resolve("adapter_config.json"), "{'base_model_name_or_path':'org/remote-model'}");
        write(kev.resolve("head.pt"), "");
        write(kev.resolve("training_config.json"), "{'args':{'head_dim':4}}");
        assertEquals(Optional.of("Kev"), HfCheckpointDetectors.match(kev));
        IOException noBase = assertThrows(IOException.class,
                () -> HfConfigLoader.load(kev, (Map<String, Object>) null));
        assertTrue(noBase.getMessage(), noBase.getMessage().contains("--hf-base-dir"));
        HfConfig k = HfConfigLoader.load(kev, null, base);
        assertEquals("KevModel", k.architecture());
        assertEquals(base, k.baseModelDir().get());

        Path nimble = dir();
        write(nimble.resolve("adapter_config.json"), "{'base_model_name_or_path':'" + base + "'}");
        write(nimble.resolve("schema_config.json"), "{'task':'schema_candidate_classification_v2'}");
        assertDetected(nimble, "Nimble", "NimbleModel");
    }

    @Test
    public void d1DetectorsRewriteTheTextModel() throws IOException {
        Path d1 = dir();
        write(d1.resolve("config.json"), "{'architectures':['Lfm2VlForConditionalGeneration'],"
                + "'auto_map':{'AutoModel':'modeling_d1.D1Model'},'text_config':{'hidden_size':8}}");
        assertDetected(d1, "D1", "D1Model");

        Path omni = dir();
        write(omni.resolve("config.json"), "{'architectures':['D1OmniModel'],'model_type':'d1_omni',"
                + "'head_layers':1,'text_config':{'num_hidden_layers':2,'block_ffn_dim_multiplier':1.5,"
                + "'intermediate_size':12,'block_multiple_of':8,'hidden_size':4}}");
        HfConfig cfg = assertDetected(omni, "D1Omni", "D1OmniModel");
        assertEquals(3, cfg.blockCount());
        assertEquals(List.of(16L, 16L, 16L), cfg.flat().get("intermediate_size"));
        assertEquals(Boolean.FALSE, cfg.flat().get("block_auto_adjust_ff_dim"));
    }

    @Test
    public void pocketTtsIsDerivedFromTensorShapes() throws IOException {
        Path d = dir();
        writeSafetensors(d.resolve("model.safetensors"), "{'__metadata__':{'format':'pt'},"
                + tensor("flow_lm.bos_emb", "[32]") + ","
                + tensor("flow_lm.conditioner.embed.weight", "[100,128]") + ","
                + tensor("flow_lm.transformer.layers.0.norm1.weight", "[128]") + ","
                + tensor("flow_lm.transformer.layers.1.norm1.weight", "[128]") + ","
                + tensor("flow_lm.transformer.layers.0.linear1.weight", "[512,128]") + ","
                + tensor("mimi.encoder_transformer.transformer.layers.0.norm1.weight", "[64]") + ","
                + tensor("mimi.encoder_transformer.transformer.layers.0.linear1.weight", "[256,64]") + "}");
        HfConfig cfg = assertDetected(d, "PocketTTS", "PocketTTSModel");
        assertEquals(101, cfg.findInt("vocab_size").getAsInt());
        assertEquals(128, cfg.findInt("hidden_size").getAsInt());
        assertEquals(2, cfg.blockCount());
        assertEquals(512, cfg.findInt("intermediate_size").getAsInt());
        assertEquals(2, cfg.findInt("num_attention_heads").getAsInt());
        assertEquals(1L, cfg.audio().get().raw("num_hidden_layers"));
        assertEquals(64L, cfg.audio().get().raw("hidden_size"));
        assertEquals(10000.0, cfg.rope().theta().getAsDouble(), 0);

        Path other = dir();
        writeSafetensors(other.resolve("model.safetensors"), "{" + tensor("model.embed.weight", "[4,4]") + "}");
        assertEquals(Optional.empty(), HfCheckpointDetectors.match(other));

        Path corrupt = dir();
        writeSafetensors(corrupt.resolve("model.safetensors"), "{" + tensor("flow_lm.bos_emb", "[-1]") + "}");
        IOException e = assertThrows(IOException.class, () -> HfCheckpointDetectors.match(corrupt));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid safetensors file"));
    }

    @Test
    public void vllmDecision2ReadsTheBackboneAndScoreBias() throws IOException {
        Path d = dir();
        write(d.resolve("config.json"), "{'decision_format':'vllm-sr-decision',"
                + "'backbone':{'config':'backbone/config.json'},'model_config':'head.json'}");
        write(d.resolve("backbone/config.json"), "{'architectures':['Qwen3ForCausalLM'],'hidden_size':64}");
        write(d.resolve("head.json"), "{'num_labels':2}");
        write(d.resolve("score_bias.json"), "{'offsets':[0.5,-0.5]}");
        HfConfig cfg = assertDetected(d, "VllmDecision2", "VllmDecision2Model");
        Map<?, ?> decision = (Map<?, ?>) cfg.raw().raw("decision");
        assertEquals(2L, decision.get("num_labels"));
        assertEquals(List.of(0.5, -0.5), decision.get("score_bias"));
        assertEquals(64L, cfg.flat().get("hidden_size"));
        assertEquals("backbone/config.json", cfg.source());

        write(d.resolve("config.json"), "{'decision_format':'vllm-sr-decision',"
                + "'backbone':{'config':'../outside/config.json'},'model_config':'head.json'}");
        IOException e = assertThrows(IOException.class, () -> HfConfigLoader.load(d, (Map<String, Object>) null));
        assertTrue(e.getMessage(), e.getMessage().contains("'..'"));
    }

    private static HfConfig assertDetected(Path dir, String detector, String architecture) throws IOException {
        assertEquals(Optional.of(detector), HfCheckpointDetectors.match(dir));
        HfConfig cfg = HfConfigLoader.load(dir, (Map<String, Object>) null);
        assertEquals(architecture, cfg.architecture());
        return cfg;
    }

    private static String tensor(String name, String shape) {
        return "'" + name + "':{'dtype':'F32','shape':" + shape + ",'data_offsets':[0,0]}";
    }

    static void writeSafetensors(Path file, String header) throws IOException {
        byte[] h = json(header).getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(8 + h.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(h.length).put(h);
        Files.createDirectories(file.getParent());
        Files.write(file, buf.array());
    }
}

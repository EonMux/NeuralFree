package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfSpecialFixturesT5;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Hugging Face names vLLM registers over families NeuralFree converts open through the registered entries: each
 * alias checkpoint builds a spec claiming every tensor and has the tensors (names, dims, types and bytes) of the same
 * model under its canonical name, plus the metadata the alias adds (embedding pooling, muP scales, the TeleChat3
 * YaRN attention factor, the Mistral Large 3 query temperature). The plain aliases resolve to the entry of the name
 * they alias.
 */
public class HfVllmAliasesSourcesTest {

    private static final int DIM = 8;
    private static final int HEADS = 2;
    private static final int KV_HEADS = 1;
    private static final int HEAD_DIM = DIM / HEADS;
    private static final int KV_DIM = KV_HEADS * HEAD_DIM;
    private static final int FFN = 16;
    private static final int LAYERS = 2;
    private static final int VOCAB = 8;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ plain aliases

    @Test
    public void plainAliasesResolveToTheEntryOfTheirName() {
        assertSame(entry("Glm5NextForConditionalGeneration"), entry("Glm5NextForCausalLM"));
        assertSame(entry("MuseGlimmerForConditionalGeneration"), entry("MuseGlimmerForCausalLM"));
        assertSame(entry("ClefModel"), entry("ClefFlashForJointSchema"));
        assertSame(entry("VllmD3Model"), entry("D3ForSequenceClassification"));
        for (String name : new String[]{"MistralModel", "TeleChat3ForCausalLM", "TeleFLMForCausalLM",
                "TeleChat2ForCausalLM", "Gemma2Model", "MiMoForCausalLM", "DeepseekV3BidirectionalModel",
                "MistralLarge3ForCausalLM", "LlamaBidirectionalForSequenceClassification"}) {
            assertTrue(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
        }
    }

    @Test
    public void theRemainingVllmClassifierNamesNameTheirMissingBackbone() {
        for (String name : new String[]{"GteNewForSequenceClassification",
                "OpenAIPrivacyFilterForTokenClassification", "LlamaNemotronVLForSequenceClassification",
                "ClefOmniForJointSchema"}) {
            assertTrue(name, HfArchRegistry.isRejected(name, HfArchEntry.Kind.TEXT));
            assertFalse(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
            GgufFormatException e = assertThrows(GgufFormatException.class, () -> entry(name));
            assertTrue(e.getMessage(), e.getMessage().contains(name));
        }
    }

    @Test
    public void theConvertedVllmPoolingHeadsAreRealEntries() {
        for (String name : new String[]{"Qwen3ASRForcedAlignerForTokenClassification", "BgeM3EmbeddingModel"}) {
            assertFalse(name, HfArchRegistry.isRejected(name, HfArchEntry.Kind.TEXT));
            assertTrue(name, HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(name));
        }
    }

    private static HfArchEntry entry(String architecture) {
        return HfArchRegistry.resolve(HfConfigLoader.fromJson(Path.of("."),
                "{\"architectures\": [\"" + architecture + "\"]}"), HfArchEntry.Kind.TEXT);
    }

    // ------------------------------------------------------------------ Llama family

    @Test
    public void mistralModelIsHeadlessLlamaWithLastTokenPooling() throws IOException {
        Opened reference = open(llama(llamaConfig("LlamaForCausalLM", ""), n -> n, false), "llama");
        Opened alias = open(llama(llamaConfig("MistralModel", ""), HfVllmAliasesSourcesTest::headless, false),
                "mistral-model");
        assertEquals("llama", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertFalse(reference.metadata.containsKey("llama.pooling_type"));
        assertEquals(PoolingKind.LAST.ggufCode(), ((Number) alias.metadata.get("llama.pooling_type")).intValue());
    }

    @Test
    public void mistralModelTakesTheSentenceTransformersPooling() throws IOException {
        SyntheticHf.Builder b = llama(llamaConfig("MistralModel", ""), HfVllmAliasesSourcesTest::headless, false)
                .file("modules.json", SyntheticHf.json("[{'idx':0,'name':'0','path':'',"
                        + "'type':'sentence_transformers.models.Transformer'},{'idx':1,'name':'1',"
                        + "'path':'1_Pooling','type':'sentence_transformers.models.Pooling'}]"))
                .file("1_Pooling/config.json", SyntheticHf.json("{'pooling_mode_mean_tokens':true}"));
        Opened alias = open(b, "mistral-model-mean");
        assertEquals(PoolingKind.MEAN.ggufCode(), ((Number) alias.metadata.get("llama.pooling_type")).intValue());
    }

    @Test
    public void llamaBidirectionalSequenceClassificationScoresTheLastRow() throws IOException {
        SyntheticHf.Builder b = llama(llamaConfig("LlamaBidirectionalForSequenceClassification",
                ",'is_causal':false,'num_labels':2"), n -> n, false)
                .f32("score.weight", 2, DIM)
                .f32("score.bias", 2);
        Opened alias = open(b, "llama-bidirectional-seq");
        assertEquals("llama-embed", alias.metadata.get("general.architecture"));
        assertEquals(PoolingKind.LAST.ggufCode(),
                ((Number) alias.metadata.get("llama-embed.pooling_type")).intValue());
        assertEquals("sequence", alias.metadata.get("llama-embed.classifier.scope"));
        assertEquals("softmax", alias.metadata.get("llama-embed.classifier.pooler_activation"));
        assertTrue("the score projection is the classifier output", alias.tensors.containsKey("cls.output.weight"));
        assertTrue(alias.tensors.containsKey("cls.output.bias"));
        assertFalse("the sequence classifier stores no LM head", alias.tensors.containsKey("output.weight"));
    }

    @Test
    public void teleChat2IsLlamaWithItsKeyValueSplit() throws IOException {
        Opened reference = open(teleChat2Reference(), "telechat2-reference");
        Opened alias = open(teleChat2(), "telechat2");
        assertEquals("llama", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertTrue(alias.tensors.containsKey("blk.0.attn_output.bias"));
        assertTrue(alias.tensors.containsKey("blk.1.ffn_down.bias"));
        assertEquals(arch(reference.metadata, "llama."), arch(alias.metadata, "llama."));
    }

    @Test
    public void teleChat2RefusesAKeyValueOfTheWrongShape() throws IOException {
        SyntheticHf.Builder b = teleChat2();
        b.tensor("transformer.h.0.self_attention.key_value.weight", StDType.F32, new long[]{KV_DIM, DIM},
                SyntheticHf.f32(SyntheticHf.values(1, KV_DIM * DIM)));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(b, "telechat2-bad"));
        assertTrue(e.getMessage(), e.getMessage().contains("key_value"));
    }

    @Test
    public void teleFlmIsGraniteWithTheMupMultipliers() throws IOException {
        Opened reference = open(llama(llamaConfig("LlamaForCausalLM", ""), n -> n, false), "teleflm-reference");
        Opened alias = open(llama(llamaConfig("TeleFLMForCausalLM",
                ",'use_mup':true,'input_mult':10.0,'output_mult':2.0,'mup_scale_factor':4.0"), n -> n, false),
                "teleflm");
        assertEquals("granite", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertEquals(10.0, ((Number) alias.metadata.get("granite.embedding_scale")).doubleValue(), 0.0);
        assertEquals("logits * output_mult / mup_scale_factor = logits / 2", 2.0,
                ((Number) alias.metadata.get("granite.logit_scale")).doubleValue(), 0.0);

        Opened plain = open(llama(llamaConfig("TeleFLMForCausalLM", ",'use_mup':false"), n -> n, false),
                "teleflm-plain");
        assertNull(plain.metadata.get("granite.embedding_scale"));
        assertEquals(1.0, ((Number) plain.metadata.get("granite.logit_scale")).doubleValue(), 0.0);
    }

    @Test
    public void teleFlmRefusesWhatGraniteCannotCarry() {
        SyntheticHf.Builder noMult = llama(llamaConfig("TeleFLMForCausalLM",
                ",'use_mup':true,'output_mult':2.0,'mup_scale_factor':4.0"), n -> n, false);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(noMult, "teleflm-nomult"));
        assertTrue(e.getMessage(), e.getMessage().contains("input_mult"));
        SyntheticHf.Builder llama3 = llama(llamaConfig("TeleFLMForCausalLM", ",'rope_scaling':{'rope_type':"
                + "'llama3','factor':8.0,'low_freq_factor':1.0,'high_freq_factor':4.0,"
                + "'original_max_position_embeddings':64}"), n -> n, false);
        e = assertThrows(GgufFormatException.class, () -> open(llama3, "teleflm-llama3"));
        assertTrue(e.getMessage(), e.getMessage().contains("TeleFLMForCausalLM"));
    }

    @Test
    public void teleChat3WritesTheYarnAttentionFactorRatio() throws IOException {
        Opened alias = open(llama(llamaConfig("TeleChat3ForCausalLM", ",'rope_scaling':{'rope_type':"
                + "'telechat3-yarn','factor':4.0,'original_max_position_embeddings':32}"), n -> n, false),
                "telechat3");
        assertEquals("llama", alias.metadata.get("general.architecture"));
        assertEquals("yarn", alias.metadata.get("llama.rope.scaling.type"));
        double expected = (0.07 * Math.log(4.0) + 1.0) / (0.1 * Math.log(4.0) + 1.0);
        assertEquals(expected, ((Number) alias.metadata.get("llama.rope.scaling.attn_factor")).doubleValue(), 1e-6);

        Opened plain = open(llama(llamaConfig("TeleChat3ForCausalLM", ""), n -> n, false), "telechat3-plain");
        assertNull(plain.metadata.get("llama.rope.scaling.attn_factor"));
    }

    @Test
    public void mimoIsQwen2WithoutMtpLayers() throws IOException {
        Opened reference = open(llama(llamaConfig("Qwen2ForCausalLM", ""), n -> n, true), "qwen2");
        Opened alias = open(llama(llamaConfig("MiMoForCausalLM", ""), n -> n, true), "mimo");
        assertEquals("qwen2", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertNull(alias.metadata.get("qwen2.nextn_predict_layers"));
    }

    /** MiMo's MTP layers become the appended Qwen2 blocks and the NextN tensors of the draft head. */
    @Test
    public void mimoKeepsItsMtpLayersAsNextnBlocks() throws IOException {
        Opened alias = open(mimoWithMtp(), "mimo-mtp", true);
        assertEquals("qwen2", alias.metadata.get("general.architecture"));
        assertEquals(1, ((Number) alias.metadata.get("qwen2.nextn_predict_layers")).intValue());
        assertTrue(alias.tensors.containsKey("blk.2.attn_q.weight"));
        assertTrue(alias.tensors.containsKey("blk.2.attn_output.weight"));
        assertTrue(alias.tensors.containsKey("blk.2.ffn_down.weight"));
        assertTrue(alias.tensors.containsKey("blk.2.nextn.enorm.weight"));
        assertTrue(alias.tensors.containsKey("blk.2.nextn.hnorm.weight"));
        assertTrue(alias.tensors.containsKey("blk.2.nextn.shared_head_norm.weight"));
        assertTrue("the LM head is the target's", !alias.tensors.containsKey("blk.2.nextn.shared_head_head.weight"));
        assertTrue("the token table is the target's", !alias.tensors.containsKey("blk.2.nextn.embed_tokens.weight"));
        String eh = alias.tensors.get("blk.2.nextn.eh_proj.weight");
        assertTrue(eh, eh.startsWith("[16, 8] F32 "));
        assertArrayEquals("input_proj's hidden half second, MiMo's [hidden | embed] order",
                swappedInputProjection(MIMO_INPUT_PROJ_SEED), storedFloats(eh), 0f);
    }

    /** The MTP tensors are dropped under --no-mtp. */
    @Test
    public void mimoNoMtpDropsItsMtpLayers() throws IOException {
        Opened reference = open(llama(llamaConfig("Qwen2ForCausalLM", ""), n -> n, true), "qwen2");
        Opened alias = open(mimoWithMtp(), "mimo-mtp-dropped", false);
        assertEquals(reference.tensors, alias.tensors);
        assertNull(alias.metadata.get("qwen2.nextn_predict_layers"));
    }

    /** A full MiMo MTP layer: its own norms and input projection beside a Qwen2 decoder block. */
    private static SyntheticHf.Builder mimoWithMtp() {
        SyntheticHf.Builder b = llama(llamaConfig("MiMoForCausalLM", ",'num_nextn_predict_layers':1"), n -> n, true);
        String p = "model.mtp_layers.0.";
        put(b, p + "token_layernorm.weight", MIMO_INPUT_PROJ_SEED + 1, DIM);
        put(b, p + "hidden_layernorm.weight", MIMO_INPUT_PROJ_SEED + 2, DIM);
        put(b, p + "input_proj.weight", MIMO_INPUT_PROJ_SEED, DIM, 2 * DIM);
        put(b, p + "final_layernorm.weight", MIMO_INPUT_PROJ_SEED + 3, DIM);
        put(b, p + "input_layernorm.weight", MIMO_INPUT_PROJ_SEED + 4, DIM);
        put(b, p + "self_attn.q_proj.weight", MIMO_INPUT_PROJ_SEED + 5, DIM, DIM);
        put(b, p + "self_attn.k_proj.weight", MIMO_INPUT_PROJ_SEED + 6, KV_DIM, DIM);
        put(b, p + "self_attn.v_proj.weight", MIMO_INPUT_PROJ_SEED + 7, KV_DIM, DIM);
        put(b, p + "self_attn.o_proj.weight", MIMO_INPUT_PROJ_SEED + 8, DIM, DIM);
        put(b, p + "post_attention_layernorm.weight", MIMO_INPUT_PROJ_SEED + 9, DIM);
        put(b, p + "mlp.gate_proj.weight", MIMO_INPUT_PROJ_SEED + 10, FFN, DIM);
        put(b, p + "mlp.up_proj.weight", MIMO_INPUT_PROJ_SEED + 11, FFN, DIM);
        put(b, p + "mlp.down_proj.weight", MIMO_INPUT_PROJ_SEED + 12, DIM, FFN);
        return b;
    }

    private static final int MIMO_INPUT_PROJ_SEED = 40;

    /** {@link #mimoWithMtp()}'s input projection with its embed and hidden input halves swapped. */
    private static float[] swappedInputProjection(int seed) {
        float[] src = SyntheticHf.values(seed, DIM * 2 * DIM);
        float[] out = new float[src.length];
        for (int r = 0; r < DIM; r++) {
            for (int c = 0; c < 2 * DIM; c++) {
                out[r * 2 * DIM + c] = src[r * 2 * DIM + (c + DIM) % (2 * DIM)];
            }
        }
        return out;
    }

    /** The floats of an {@link Opened#tensors} value ({@code "[dims] TYPE base64"}). */
    private static float[] storedFloats(String value) {
        byte[] bytes = Base64.getDecoder().decode(value.substring(value.lastIndexOf(' ') + 1));
        ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bytes.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = data.getFloat();
        }
        return out;
    }

    // ------------------------------------------------------------------ Gemma2

    @Test
    public void gemma2ModelIsHeadlessGemma2WithLastTokenPooling() throws IOException {
        Opened reference = open(gemma2("Gemma2ForCausalLM", n -> n), "gemma2");
        Opened alias = open(gemma2("Gemma2Model", HfVllmAliasesSourcesTest::headless), "gemma2-model");
        assertEquals("gemma2", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertEquals(PoolingKind.LAST.ggufCode(), ((Number) alias.metadata.get("gemma2.pooling_type")).intValue());
    }

    // ------------------------------------------------------------------ DeepSeek family

    @Test
    public void deepseekV3BidirectionalIsHeadlessNonCausalDeepseek() throws IOException {
        Opened reference = open(deepseek("DeepseekV3ForCausalLM", "", n -> n), "deepseek");
        Opened alias = open(deepseek("DeepseekV3BidirectionalModel", ",'is_causal':false",
                HfVllmAliasesSourcesTest::headless), "deepseek-bidirectional");
        assertEquals("deepseek2", alias.metadata.get("general.architecture"));
        assertEquals(reference.tensors, alias.tensors);
        assertEquals(Boolean.FALSE, alias.metadata.get("deepseek2.attention.causal"));
        assertEquals(PoolingKind.LAST.ggufCode(),
                ((Number) alias.metadata.get("deepseek2.pooling_type")).intValue());
    }

    @Test
    public void mistralLarge3NativeNamesConvertLikeDeepseek() throws IOException {
        String scaling = ",'llama_4_scaling':{'beta':0.1,'original_max_position_embeddings':64}";
        Opened reference = open(deepseek("DeepseekV3ForCausalLM", "", n -> n), "deepseek-hf");
        Opened hfNamed = open(deepseek("MistralLarge3ForCausalLM", scaling, n -> n), "large3-hf");
        Opened nativeNamed = open(deepseek("MistralLarge3ForCausalLM", scaling,
                HfVllmAliasesSourcesTest::mistralNative), "large3-native");
        assertEquals(reference.tensors, hfNamed.tensors);
        assertEquals(reference.tensors, nativeNamed.tensors);
        assertEquals(0.1, ((Number) nativeNamed.metadata.get("deepseek2.attention.temperature_scale"))
                .doubleValue(), 1e-7);
        assertEquals(64, ((Number) nativeNamed.metadata.get("deepseek2.attention.temperature_length")).intValue());
        assertNull(reference.metadata.get("deepseek2.attention.temperature_scale"));
    }

    @Test
    public void mistralLarge3RefusesAnInvalidLlama4Scaling() {
        SyntheticHf.Builder b = deepseek("MistralLarge3ForCausalLM",
                ",'llama_4_scaling':{'beta':0.1,'original_max_position_embeddings':0}", n -> n);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(b, "large3-bad"));
        assertTrue(e.getMessage(), e.getMessage().contains("llama_4_scaling.original_max_position_embeddings"));
    }

    @Test
    public void mistralLarge3RefusesNamesThatCollide() {
        SyntheticHf.Builder b = deepseek("MistralLarge3ForCausalLM", "", n -> n);
        b.f32("norm.weight", DIM * 2);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(b, "large3-collide"));
        assertTrue(e.getMessage(), e.getMessage().contains("MistralLarge3ForCausalLM"));
    }

    // ------------------------------------------------------------------ checkpoints

    /** A Llama-shaped configuration under {@code architecture}, {@code extra} appended to its members. */
    private static String llamaConfig(String architecture, String extra) {
        return SyntheticHf.json("{'architectures':['" + architecture + "'],'hidden_size':" + DIM
                + ",'intermediate_size':" + FFN + ",'num_hidden_layers':" + LAYERS + ",'num_attention_heads':"
                + HEADS + ",'num_key_value_heads':" + KV_HEADS + ",'max_position_embeddings':128,"
                + "'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':" + VOCAB + ",'torch_dtype':'float32',"
                + "'bos_token_id':0,'eos_token_id':1" + extra + "}");
    }

    /**
     * The F32 Llama tensors under {@code rename} of their Hugging Face names, values seeded per tensor position so
     * that two calls with one shape write the same data; {@code qkvBias} adds the Qwen2 projection biases.
     */
    private static SyntheticHf.Builder llama(String config, UnaryOperator<String> rename, boolean qkvBias) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        b.f32(rename.apply("model.embed_tokens.weight"), VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.f32(rename.apply(p + "input_layernorm.weight"), DIM);
            b.f32(rename.apply(p + "self_attn.q_proj.weight"), DIM, DIM);
            b.f32(rename.apply(p + "self_attn.k_proj.weight"), KV_DIM, DIM);
            b.f32(rename.apply(p + "self_attn.v_proj.weight"), KV_DIM, DIM);
            if (qkvBias) {
                b.f32(rename.apply(p + "self_attn.q_proj.bias"), DIM);
                b.f32(rename.apply(p + "self_attn.k_proj.bias"), KV_DIM);
                b.f32(rename.apply(p + "self_attn.v_proj.bias"), KV_DIM);
            }
            b.f32(rename.apply(p + "self_attn.o_proj.weight"), DIM, DIM);
            b.f32(rename.apply(p + "post_attention_layernorm.weight"), DIM);
            b.f32(rename.apply(p + "mlp.gate_proj.weight"), FFN, DIM);
            b.f32(rename.apply(p + "mlp.up_proj.weight"), FFN, DIM);
            b.f32(rename.apply(p + "mlp.down_proj.weight"), DIM, FFN);
        }
        b.f32(rename.apply("model.norm.weight"), DIM);
        b.f32("lm_head.weight", VOCAB, DIM);
        return b;
    }

    /** The Llama checkpoint equivalent to {@link #teleChat2}: split K/V and the o_proj/down_proj biases. */
    private static SyntheticHf.Builder teleChat2Reference() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(llamaConfig("LlamaForCausalLM", ""));
        put(b, "model.embed_tokens.weight", 0, VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            int s = 100 * (l + 1);
            put(b, p + "input_layernorm.weight", s, DIM);
            put(b, p + "self_attn.q_proj.weight", s + 1, DIM, DIM);
            put(b, p + "self_attn.k_proj.weight", s + 2, KV_DIM, DIM);
            put(b, p + "self_attn.v_proj.weight", s + 3, KV_DIM, DIM);
            put(b, p + "self_attn.o_proj.weight", s + 4, DIM, DIM);
            put(b, p + "self_attn.o_proj.bias", s + 5, DIM);
            put(b, p + "post_attention_layernorm.weight", s + 6, DIM);
            put(b, p + "mlp.gate_proj.weight", s + 7, FFN, DIM);
            put(b, p + "mlp.up_proj.weight", s + 8, FFN, DIM);
            put(b, p + "mlp.down_proj.weight", s + 9, DIM, FFN);
            put(b, p + "mlp.down_proj.bias", s + 10, DIM);
        }
        put(b, "model.norm.weight", 1, DIM);
        put(b, "lm_head.weight", 2, VOCAB, DIM);
        return b;
    }

    /** A TeleChat2 checkpoint (telechat2.py names, per-head interleaved {@code key_value}). */
    private static SyntheticHf.Builder teleChat2() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['TeleChat2ForCausalLM'],'model_type':'telechat','hidden_size':" + DIM + ",'ffn_hidden_size':"
                + FFN + ",'n_layer':" + LAYERS + ",'n_head':" + HEADS + ",'num_key_value_heads':" + KV_HEADS
                + ",'layer_norm_epsilon':1e-05,'max_position_embeddings':128,'rope_theta':10000.0,'vocab_size':"
                + VOCAB + ",'torch_dtype':'float32','bos_token_id':0,'eos_token_id':1}"));
        put(b, "transformer.word_embeddings.weight", 0, VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "transformer.h." + l + ".";
            int s = 100 * (l + 1);
            put(b, p + "input_layernorm.weight", s, DIM);
            put(b, p + "self_attention.query.weight", s + 1, DIM, DIM);
            b.tensor(p + "self_attention.key_value.weight", StDType.F32, new long[]{2L * KV_DIM, DIM},
                    SyntheticHf.f32(interleave(SyntheticHf.values(s + 2, KV_DIM * DIM),
                            SyntheticHf.values(s + 3, KV_DIM * DIM))));
            put(b, p + "self_attention.dense.weight", s + 4, DIM, DIM);
            put(b, p + "self_attention.dense.bias", s + 5, DIM);
            put(b, p + "post_attention_layernorm.weight", s + 6, DIM);
            put(b, p + "mlp.gate_proj.weight", s + 7, FFN, DIM);
            put(b, p + "mlp.up_proj.weight", s + 8, FFN, DIM);
            put(b, p + "mlp.down_proj.weight", s + 9, DIM, FFN);
            put(b, p + "mlp.down_proj.bias", s + 10, DIM);
        }
        put(b, "transformer.ln_f.weight", 1, DIM);
        put(b, "lm_head.weight", 2, VOCAB, DIM);
        return b;
    }

    /** The {@code [K_h; V_h]} row blocks per key/value head of K and V ({@code [KV_DIM, DIM]} each). */
    private static float[] interleave(float[] k, float[] v) {
        float[] out = new float[k.length + v.length];
        int block = HEAD_DIM * DIM;
        for (int h = 0; h < KV_HEADS; h++) {
            System.arraycopy(k, h * block, out, 2 * h * block, block);
            System.arraycopy(v, h * block, out, (2 * h + 1) * block, block);
        }
        return out;
    }

    private static void put(SyntheticHf.Builder b, String name, int seed, long... shape) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(SyntheticHf.values(seed, n)));
    }

    /** A Gemma2 checkpoint under {@code architecture}, the SentencePiece vocabulary its converter reads. */
    private static SyntheticHf.Builder gemma2(String architecture, UnaryOperator<String> rename) {
        String config = SyntheticHf.json("{'architectures':['" + architecture + "'],'model_type':'gemma2',"
                + "'hidden_size':" + DIM + ",'intermediate_size':" + FFN + ",'num_hidden_layers':" + LAYERS
                + ",'num_attention_heads':" + HEADS + ",'num_key_value_heads':" + KV_HEADS + ",'head_dim':"
                + HEAD_DIM + ",'max_position_embeddings':128,'rms_norm_eps':1e-06,'rope_theta':10000.0,"
                + "'vocab_size':" + VOCAB + ",'attn_logit_softcapping':50.0,'final_logit_softcapping':30.0,"
                + "'sliding_window':64,'query_pre_attn_scalar':4,'torch_dtype':'float32','bos_token_id':2,"
                + "'eos_token_id':1,'pad_token_id':0}");
        SyntheticHf.Builder b = SyntheticHf.builder().config(config).tokenizerJson(null)
                .binaryFile("tokenizer.model", HfSpecialFixturesT5.unigramModel());
        b.f32(rename.apply("model.embed_tokens.weight"), VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.f32(rename.apply(p + "input_layernorm.weight"), DIM);
            b.f32(rename.apply(p + "self_attn.q_proj.weight"), DIM, DIM);
            b.f32(rename.apply(p + "self_attn.k_proj.weight"), KV_DIM, DIM);
            b.f32(rename.apply(p + "self_attn.v_proj.weight"), KV_DIM, DIM);
            b.f32(rename.apply(p + "self_attn.o_proj.weight"), DIM, DIM);
            b.f32(rename.apply(p + "post_attention_layernorm.weight"), DIM);
            b.f32(rename.apply(p + "pre_feedforward_layernorm.weight"), DIM);
            b.f32(rename.apply(p + "mlp.gate_proj.weight"), FFN, DIM);
            b.f32(rename.apply(p + "mlp.up_proj.weight"), FFN, DIM);
            b.f32(rename.apply(p + "mlp.down_proj.weight"), DIM, FFN);
            b.f32(rename.apply(p + "post_feedforward_layernorm.weight"), DIM);
        }
        b.f32(rename.apply("model.norm.weight"), DIM);
        return b;
    }

    /**
     * A two-layer DeepSeek-V3 checkpoint (one dense block, then two routed experts and a shared one) under {@code architecture};
     * {@code extra} is appended to the configuration and {@code rename} applies to every Hugging Face name.
     */
    private static SyntheticHf.Builder deepseek(String architecture, String extra, UnaryOperator<String> rename) {
        int dim = 2 * DIM;
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':['"
                + architecture + "'],'model_type':'deepseek_v3','hidden_size':" + dim + ",'intermediate_size':12,"
                + "'moe_intermediate_size':8,'num_attention_heads':2,'num_key_value_heads':2,'num_hidden_layers':2,"
                + "'max_position_embeddings':128,'vocab_size':16,'bos_token_id':0,'eos_token_id':1,"
                + "'q_lora_rank':4,'kv_lora_rank':4,'qk_nope_head_dim':2,'qk_rope_head_dim':2,'v_head_dim':1,"
                + "'n_routed_experts':2,'num_experts_per_tok':1,'n_shared_experts':1,'first_k_dense_replace':1,"
                + "'rms_norm_eps':1e-05,'rope_theta':10000.0,'torch_dtype':'float32'" + extra + "}"));
        b.f32(rename.apply("model.embed_tokens.weight"), 16, dim);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.f32(rename.apply(p + "input_layernorm.weight"), dim);
            b.f32(rename.apply(p + "self_attn.q_a_proj.weight"), 4, dim);
            b.f32(rename.apply(p + "self_attn.q_a_layernorm.weight"), 4);
            b.f32(rename.apply(p + "self_attn.q_b_proj.weight"), 2 * 4, 4);
            b.f32(rename.apply(p + "self_attn.kv_a_proj_with_mqa.weight"), 6, dim);
            b.f32(rename.apply(p + "self_attn.kv_a_layernorm.weight"), 4);
            b.f32(rename.apply(p + "self_attn.kv_b_proj.weight"), 2 * (2 + 1), 4);
            b.f32(rename.apply(p + "self_attn.o_proj.weight"), dim, 2);
            b.f32(rename.apply(p + "post_attention_layernorm.weight"), dim);
            if (l == 0) {
                b.f32(rename.apply(p + "mlp.gate_proj.weight"), 12, dim);
                b.f32(rename.apply(p + "mlp.up_proj.weight"), 12, dim);
                b.f32(rename.apply(p + "mlp.down_proj.weight"), dim, 12);
            } else {
                b.f32(rename.apply(p + "mlp.gate.weight"), 2, dim);
                b.f32(rename.apply(p + "mlp.shared_experts.gate_proj.weight"), 8, dim);
                b.f32(rename.apply(p + "mlp.shared_experts.up_proj.weight"), 8, dim);
                b.f32(rename.apply(p + "mlp.shared_experts.down_proj.weight"), dim, 8);
                for (int e = 0; e < 2; e++) {
                    String x = p + "mlp.experts." + e + ".";
                    b.f32(rename.apply(x + "gate_proj.weight"), 8, dim);
                    b.f32(rename.apply(x + "up_proj.weight"), 8, dim);
                    b.f32(rename.apply(x + "down_proj.weight"), dim, 8);
                }
            }
        }
        b.f32(rename.apply("model.norm.weight"), dim);
        b.f32(rename.apply("lm_head.weight"), 16, dim);
        return b;
    }

    /** {@code name} without the {@code model.} prefix (the headless layout of the embedding checkpoints). */
    private static String headless(String name) {
        return name.startsWith("model.") ? name.substring("model.".length()) : name;
    }

    /** The native Mistral name of a Hugging Face DeepSeek name (mistral_large_3.py's mapper, inverted). */
    private static String mistralNative(String name) {
        String n = headless(name)
                .replace(".input_layernorm.", ".attention_norm.")
                .replace(".self_attn.q_a_proj.", ".attention.wq_a.")
                .replace(".self_attn.q_a_layernorm.", ".attention.q_a_norm.")
                .replace(".self_attn.q_b_proj.", ".attention.wq_b.")
                .replace(".self_attn.kv_a_proj_with_mqa.", ".attention.wkv_a_with_mqa.")
                .replace(".self_attn.kv_a_layernorm.", ".attention.kv_a_norm.")
                .replace(".self_attn.kv_b_proj.", ".attention.wkv_b.")
                .replace(".self_attn.o_proj.", ".attention.wo.")
                .replace(".post_attention_layernorm.", ".ffn_norm.")
                .replace(".mlp.gate.", ".gate.")
                .replace(".mlp.experts.", ".experts.")
                .replace(".mlp.shared_experts.", ".shared_experts.")
                .replace(".mlp.", ".feed_forward.")
                .replace(".gate_proj.", ".w1.")
                .replace(".down_proj.", ".w2.")
                .replace(".up_proj.", ".w3.");
        switch (n) {
            case "embed_tokens.weight":
                return "tok_embeddings.weight";
            case "lm_head.weight":
                return "output.weight";
            default:
                return n;
        }
    }

    // ------------------------------------------------------------------ opening

    /** The metadata and the tensors ({@code name -> "[dims] TYPE bytes"}) of an opened checkpoint. */
    private static final class Opened {
        final Map<String, Object> metadata;
        final Map<String, String> tensors;

        Opened(Map<String, Object> metadata, Map<String, String> tensors) {
            this.metadata = metadata;
            this.tensors = tensors;
        }
    }

    /** Writes {@code b}, opens it, builds and claims its spec, and reads every tensor. */
    private Opened open(SyntheticHf.Builder b, String name) throws IOException {
        return open(b, name, true);
    }

    /** {@link #open(SyntheticHf.Builder, String)} with the MTP layers kept or dropped. */
    private Opened open(SyntheticHf.Builder b, String name, boolean includeMtp) throws IOException {
        Path dir = b.write(tmp.newFolder(name + "-hf").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(tmp.newFolder(name + "-cache").toPath())
                .includeMtp(includeMtp)
                .build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            Map<String, String> tensors = new TreeMap<>();
            try (TensorRegions regions = src.openRegions(TensorReadMode.MMAP)) {
                for (GGUFTensorInfo t : src.getTensors()) {
                    ByteBuffer data = regions.region(t, 0, t.sizeInBytes()).duplicate();
                    byte[] bytes = new byte[data.remaining()];
                    data.get(bytes);
                    String prev = tensors.put(t.name(), Arrays.toString(t.dims()) + " " + t.type() + " "
                            + Base64.getEncoder().encodeToString(bytes));
                    assertNull("tensor " + t.name() + " is listed once", prev);
                }
            }
            return new Opened(new TreeMap<>(src.getMetadata()), tensors);
        }
    }

    /** The metadata keys of {@code md} under {@code prefix}. */
    private static Map<String, Object> arch(Map<String, Object> md, String prefix) {
        Map<String, Object> out = new TreeMap<>();
        for (Map.Entry<String, Object> e : md.entrySet()) {
            if (e.getKey().startsWith(prefix)) {
                out.put(e.getKey(), e.getValue());
            }
        }
        return out;
    }
}

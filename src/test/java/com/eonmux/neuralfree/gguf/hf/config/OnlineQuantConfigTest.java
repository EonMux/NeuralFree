package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.Layer;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.LayerKind;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.Selection;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link OnlineQuantConfig}: vLLM's {@code QuantizationConfigArgs} parsing and composition
 * (V/config/quantization.py) and the per-layer choice of {@code OnlineQuantizationConfig.resolve_quant_method_cls}
 * (V/model_executor/layers/quantization/online/base.py), with vLLM's layer names.
 */
public class OnlineQuantConfigTest {

    private static final String L0 = "model.layers.0.";
    private static final Layer QKV = Layer.named(L0 + "self_attn.qkv_proj");
    private static final Layer O_PROJ = Layer.named(L0 + "self_attn.o_proj");
    private static final Layer EXPERTS = Layer.named(L0 + "mlp.experts");
    private static final Layer SHARED_DOWN = Layer.named(L0 + "mlp.shared_expert.down_proj");
    private static final Layer EMBED = Layer.named("model.embed_tokens");

    @SuppressWarnings("unchecked")
    private static OnlineQuantConfig parse(String json) {
        return OnlineQuantConfig.parse((Map<String, Object>) Json.parse(json), "test", false);
    }

    /** The layer of {@code hfTensor} in a checkpoint storing every tensor of {@code stored}. */
    private static Layer layerOf(String hfTensor, LayerKind kind, String... stored) {
        Set<String> has = new HashSet<>(Arrays.asList(stored));
        return Layer.of(hfTensor, kind, has::contains);
    }

    private static Layer fusedQkv(int layer) {
        String p = "model.layers." + layer + ".self_attn.";
        return layerOf(p + "q_proj.weight", LayerKind.LINEAR, p + "q_proj.weight", p + "k_proj.weight",
                p + "v_proj.weight");
    }

    private static OnlineQuant target(OnlineQuantConfig c, LayerKind kind, Layer layer) {
        Selection s = c.select(kind, layer);
        return s == null ? null : s.target();
    }

    private static String errorOf(Runnable r) {
        return assertThrows(IllegalArgumentException.class, r::run).getMessage();
    }

    @Test
    public void specsAreShorthandsQuantKeyNamesOrWeightObjects() {
        OnlineQuantConfig c = parse("{\"linear\": \"fp8_per_block\","
                + " \"moe\": {\"weight\": \"fp8_per_tensor_static\"}}");
        Selection s = c.select(LayerKind.LINEAR, QKV);
        assertSame(OnlineQuant.FP8_PER_BLOCK, s.target());
        assertEquals("linear: fp8_per_block_static", s.origin());
        assertSame(OnlineQuant.FP8_PER_TENSOR, target(c, LayerKind.MOE, EXPERTS));
        assertNull("embeddings are never quantized online", c.select(LayerKind.EMBEDDING, EMBED));

        OnlineQuantConfig mx = parse("{\"linear\": \"mxfp4\", \"moe\": \"mxfp4\"}");
        assertSame(OnlineQuant.MXFP4, target(mx, LayerKind.LINEAR, O_PROJ));
        assertSame(OnlineQuant.MXFP4, target(mx, LayerKind.MOE, EXPERTS));
        assertEquals(mx, OnlineQuantConfig.shorthand(OnlineQuant.MXFP4));

        assertSame(OnlineQuant.MXFP8, target(parse("{\"linear\": \"mxfp8\"}"), LayerKind.LINEAR, O_PROJ));
        assertSame(OnlineQuant.FP8_PER_CHANNEL,
                target(parse("{\"moe\": \"fp8_per_channel_static\"}"), LayerKind.MOE, EXPERTS));
        assertSame(OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY,
                target(parse("{\"moe\": \"int8_per_channel_weight_only\"}"), LayerKind.MOE, EXPERTS));
        assertSame(OnlineQuant.NVFP4_PER_TOKEN,
                target(parse("{\"moe\": \"nvfp4_per_token\"}"), LayerKind.MOE, EXPERTS));
    }

    @Test
    public void aNullWeightOrAnUnsetKindLeavesTheKindUnquantized() {
        OnlineQuantConfig c = parse("{\"linear\": {\"weight\": null}, \"moe\": \"fp8_per_block\"}");
        assertFalse(c.quantizes(LayerKind.LINEAR));
        assertTrue(c.quantizes(LayerKind.MOE));
        assertNull(c.select(LayerKind.LINEAR, O_PROJ));
        OnlineQuantConfig moeOnly = parse("{\"moe\": \"fp8_per_block\"}");
        assertNull(moeOnly.select(LayerKind.LINEAR, O_PROJ));
        assertFalse(moeOnly.quantizes(LayerKind.EMBEDDING));
    }

    @Test
    public void weightsWithoutAnOnlineMethodForTheKindAreRefusedAtTheLayer() {
        String dynamicMx = errorOf(() -> parse("{\"linear\": {\"weight\": \"mxfp4\"}}")
                .select(LayerKind.LINEAR, O_PROJ));
        assertTrue(dynamicMx, dynamicMx.contains("weight mxfp4 is not supported")
                && dynamicMx.contains(O_PROJ.name()));
        String int8Linear = errorOf(() -> parse("{\"linear\": \"int8_per_channel_static\"}")
                .select(LayerKind.LINEAR, O_PROJ));
        assertTrue(int8Linear, int8Linear.contains("linear layers with weight int8_per_channel_static"));
        assertNull("int4_per_group_32 is vLLM's MXFP4 requantization, not an online method",
                parse("{\"moe\": \"int4_per_group_32\"}").select(LayerKind.MOE, EXPERTS));
    }

    @Test
    public void malformedConfigurationsAreRefusedByMember() {
        assertTrue(errorOf(() -> parse("{\"linear\": \"int8_per_channel_weight_only\"}"))
                .contains("test: linear: online shorthand 'int8_per_channel_weight_only' has no linear spec"));
        assertTrue(errorOf(() -> parse("{\"linear\": \"fp4\"}")).contains("unknown quantization name 'fp4'"));
        assertTrue(errorOf(() -> parse("{\"linear\": {\"weight\": \"fp8_per_block_static\", \"activation\":"
                + " \"fp8_per_token\"}}")).contains("linear.activation=fp8_per_token is not supported"));
        assertTrue(errorOf(() -> parse("{\"linear\": {\"wieght\": \"mxfp8\"}}")).contains("unknown member 'wieght'"));
        assertTrue(errorOf(() -> parse("{\"quant_method\": \"online\", \"linear\": \"mxfp8\"}"))
                .contains("unknown member 'quant_method'"));
        assertTrue(errorOf(() -> parse("{\"ignore\": \"lm_head\"}")).contains("ignore must be an array"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"*\": \"fp8\"}}")).contains("is not an online shorthand"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"re:(\": \"mxfp8\"}}")).contains("pattern 're:(' is invalid"));
        assertTrue(errorOf(() -> parse("{\"linear\": \"mxfp8\", \"targets\": {\"*\": \"mxfp8\"}}"))
                .contains("targets is exclusive with"));
        assertTrue(errorOf(() -> parse("{\"linear\": 8}")).contains("linear must be a string or an object"));
    }

    @Test
    public void aCheckpointConfigCarriesQuantMethodOnline() {
        @SuppressWarnings("unchecked")
        Map<String, Object> json = (Map<String, Object>) Json.parse("{\"quant_method\": \"online\", \"moe\": \"mxfp4\","
                + " \"ignore\": [\"re:.*\\\\.mlp\\\\.gate$\"]}");
        OnlineQuantConfig c = OnlineQuantConfig.parse(json, "config.json: quantization_config", true);
        assertTrue(c.quantizes(LayerKind.MOE));
        assertEquals(Collections.singletonList("re:.*\\.mlp\\.gate$"), c.ignore());
        @SuppressWarnings("unchecked")
        Map<String, Object> fp8 = (Map<String, Object>) Json.parse("{\"quant_method\": \"fp8\"}");
        String e = errorOf(() -> OnlineQuantConfig.parse(fp8, "config.json: quantization_config", true));
        assertTrue(e, e.startsWith("config.json: quantization_config: quant_method must be 'online'"));
    }

    @Test
    public void ignoreMatchesExactNamesPrefixRegexesAndFnmatchPatterns() {
        OnlineQuantConfig c = parse("{\"linear\": \"fp8_per_block\", \"moe\": \"fp8_per_block\", \"ignore\": ["
                + "\"model.layers.0.self_attn.o_proj\", \"re:model\\\\.layers\\\\.0\\\\.mlp\\\\.shared\","
                + " \"model.layers.1.*\"]}");
        assertNull("exact", c.select(LayerKind.LINEAR, O_PROJ));
        assertNull("re: matches a prefix, as re.match", c.select(LayerKind.LINEAR, SHARED_DOWN));
        assertNull("fnmatch", c.select(LayerKind.LINEAR, fusedQkv(1)));
        assertNull("fnmatch", c.select(LayerKind.MOE, Layer.named("model.layers.1.mlp.experts")));
        assertSame(OnlineQuant.FP8_PER_BLOCK, target(c, LayerKind.LINEAR, fusedQkv(0)));
        assertSame(OnlineQuant.FP8_PER_BLOCK, target(c, LayerKind.MOE, EXPERTS));
        assertSame("an exact or fnmatch entry is no prefix", OnlineQuant.MXFP8, target(parse("{\"linear\":"
                + " \"mxfp8\", \"ignore\": [\"model.layers.0.self_attn\"]}"), LayerKind.LINEAR, O_PROJ));
    }

    @Test
    public void aFusedLayerIsIgnoredByItsNameOrByAllOfItsShards() {
        Layer qkv = fusedQkv(0);
        assertNull(parse("{\"linear\": \"mxfp8\", \"ignore\": [\"*.qkv_proj\"]}").select(LayerKind.LINEAR, qkv));
        assertNull(parse("{\"linear\": \"mxfp8\", \"ignore\": [\"*.q_proj\", \"*.k_proj\", \"*.v_proj\"]}")
                .select(LayerKind.LINEAR, qkv));
        String e = errorOf(() -> parse("{\"linear\": \"mxfp8\", \"ignore\": [\"*.q_proj\", \"*.k_proj\"]}")
                .select(LayerKind.LINEAR, qkv));
        assertTrue(e, e.contains("ignore matches only some shards of the fused layer " + qkv.name()));
    }

    @Test
    public void targetsChoosePerLayerAndLeaveTheRestUnquantized() {
        OnlineQuantConfig c = parse("{\"targets\": {\"re:.*\\\\.mlp\\\\.experts$\": \"nvfp4_per_token\","
                + " \"*.self_attn.*\": \"fp8_per_block\"}, \"ignore\": [\"model.layers.1.self_attn.qkv_proj\"]}");
        Selection experts = c.select(LayerKind.MOE, EXPERTS);
        assertSame(OnlineQuant.NVFP4_PER_TOKEN, experts.target());
        assertEquals("targets: re:.*\\.mlp\\.experts$ = nvfp4_per_token", experts.origin());
        assertSame(OnlineQuant.FP8_PER_BLOCK, target(c, LayerKind.LINEAR, fusedQkv(0)));
        assertSame(OnlineQuant.FP8_PER_BLOCK, target(c, LayerKind.LINEAR, O_PROJ));
        assertNull("no pattern", c.select(LayerKind.LINEAR, SHARED_DOWN));
        assertNull("no pattern", c.select(LayerKind.EMBEDDING, EMBED));
        assertFalse("targets set no kind spec", c.quantizes(LayerKind.LINEAR));

        assertTrue(errorOf(() -> c.select(LayerKind.LINEAR, fusedQkv(1)))
                .contains("matches both quantization_config.ignore and quantization_config.targets"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"*.mlp.*\": \"nvfp4_per_token\"}}")
                .select(LayerKind.LINEAR, SHARED_DOWN)).contains("has no spec for linear layers"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"model.*\": \"mxfp8\"}}").select(LayerKind.EMBEDDING, EMBED))
                .contains("does not apply to embeddings"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"*.o_proj\": \"mxfp8\", \"*.self_attn.*\": \"mxfp4\"}}")
                .select(LayerKind.LINEAR, O_PROJ)).contains("several quantization_config.targets patterns"));
    }

    @Test
    public void theShardsOfAFusedLayerMustMatchOneTargetMethod() {
        Layer qkv = fusedQkv(0);
        assertSame(OnlineQuant.MXFP8, target(parse("{\"targets\": {\"*.q_proj\": \"mxfp8\", \"*.k_proj\": \"mxfp8\","
                + " \"*.v_proj\": \"mxfp8\"}}"), LayerKind.LINEAR, qkv));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"*.q_proj\": \"mxfp8\", \"*.k_proj\": \"mxfp8\"}}")
                .select(LayerKind.LINEAR, qkv)).contains("every shard of a fused layer must match a target"));
        assertTrue(errorOf(() -> parse("{\"targets\": {\"*.q_proj\": \"mxfp8\", \"*.k_proj\": \"mxfp8\","
                + " \"*.v_proj\": \"mxfp4\"}}").select(LayerKind.LINEAR, qkv)).contains("of different methods"));
    }

    @Test
    public void effectiveComposesTheConfigOverTheShorthand() {
        assertNull(OnlineQuantConfig.effective(OnlineQuant.NONE, null));
        assertNull(OnlineQuantConfig.effective(OnlineQuant.Q4_0, null));
        assertEquals(OnlineQuantConfig.shorthand(OnlineQuant.FP8_PER_TENSOR),
                OnlineQuantConfig.effective(OnlineQuant.FP8_PER_TENSOR, null));
        assertTrue(errorOf(() -> OnlineQuantConfig.effective(OnlineQuant.Q4_0, parse("{\"linear\": \"mxfp8\"}")))
                .contains("q4_0 is a llama-quantize type"));
        assertTrue(errorOf(() -> OnlineQuantConfig.effective(OnlineQuant.ONLINE, null)).contains("none is given"));
        assertTrue(errorOf(() -> OnlineQuantConfig.effective(OnlineQuant.ONLINE, parse("{\"ignore\": [\"x\"]}")))
                .contains("at least one of quantization_config.linear, .moe and .targets"));
        assertTrue(errorOf(() -> OnlineQuantConfig.effective(OnlineQuant.FP8_PER_TENSOR,
                parse("{\"targets\": {\"*\": \"mxfp8\"}}"))).contains("targets is exclusive with"));

        OnlineQuantConfig ignore = OnlineQuantConfig.effective(OnlineQuant.MXFP4,
                parse("{\"ignore\": [\"*.o_proj\"]}"));
        assertSame(OnlineQuant.MXFP4, target(ignore, LayerKind.LINEAR, fusedQkv(0)));
        assertNull(ignore.select(LayerKind.LINEAR, O_PROJ));
        OnlineQuantConfig linearOff = OnlineQuantConfig.effective(OnlineQuant.FP8_PER_BLOCK,
                parse("{\"linear\": {\"weight\": null}}"));
        assertNull("an explicit null weight wins over the shorthand", linearOff.select(LayerKind.LINEAR, O_PROJ));
        assertSame(OnlineQuant.FP8_PER_BLOCK, target(linearOff, LayerKind.MOE, EXPERTS));
        assertEquals(parse("{\"linear\": \"mxfp8\"}"), OnlineQuantConfig.effective(OnlineQuant.NONE,
                parse("{\"linear\": \"mxfp8\"}")));
        assertEquals("an empty ignore counts as unset", Collections.singletonList("a"),
                parse("{\"linear\": \"mxfp8\", \"ignore\": []}")
                        .over(parse("{\"moe\": \"mxfp8\", \"ignore\": [\"a\"]}")).ignore());
    }

    @Test
    public void shorthandsAreVllmsOnlineShorthands() {
        assertNull(OnlineQuantConfig.shorthand(OnlineQuant.NONE));
        assertNull(OnlineQuantConfig.shorthand(OnlineQuant.ONLINE));
        assertNull(OnlineQuantConfig.shorthand(OnlineQuant.Q8_0));
        assertEquals(parse("{\"moe\": \"int8_per_channel_static\"}"),
                OnlineQuantConfig.shorthand(OnlineQuant.EXPERTS_INT8));
        assertEquals(parse("{\"moe\": \"int8_per_channel_weight_only\"}"),
                OnlineQuantConfig.shorthand(OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY));
        assertFalse(OnlineQuantConfig.shorthand(OnlineQuant.NVFP4_PER_TOKEN).quantizes(LayerKind.LINEAR));
        assertEquals(parse("{\"linear\": \"fp8_per_channel_static\", \"moe\": \"fp8_per_channel_static\"}"),
                OnlineQuantConfig.shorthand(OnlineQuant.FP8_PER_CHANNEL));
        assertNotEquals(OnlineQuantConfig.shorthand(OnlineQuant.MXFP8), OnlineQuantConfig.shorthand(OnlineQuant.MXFP4));
        assertEquals(OnlineQuantConfig.shorthand(OnlineQuant.MXFP8).hashCode(),
                parse("{\"linear\": \"mxfp8\", \"moe\": \"mxfp8\"}").hashCode());
    }

    @Test
    public void checkpointTensorsNameTheirVllmLayers() {
        String attn = L0 + "self_attn.";
        Layer qkv = layerOf(attn + "k_proj.weight", LayerKind.LINEAR, attn + "q_proj.weight", attn + "k_proj.weight",
                attn + "v_proj.weight");
        assertEquals(attn + "qkv_proj", qkv.name());
        assertEquals(Arrays.asList(attn + "q_proj", attn + "k_proj", attn + "v_proj"), qkv.shards());
        Layer lone = layerOf(attn + "q_proj.weight", LayerKind.LINEAR, attn + "q_proj.weight", attn + "k_proj.weight");
        assertEquals("a shard without all siblings is its own layer", attn + "q_proj", lone.name());
        assertTrue(lone.shards().isEmpty());
        Layer stored = layerOf(attn + "qkv_proj.weight", LayerKind.LINEAR);
        assertEquals(attn + "qkv_proj", stored.name());
        assertEquals(3, stored.shards().size());

        String shared = L0 + "mlp.shared_expert.";
        Layer gateUp = layerOf(shared + "up_proj.weight", LayerKind.LINEAR, shared + "gate_proj.weight",
                shared + "up_proj.weight");
        assertEquals(shared + "gate_up_proj", gateUp.name());
        assertEquals(Arrays.asList(shared + "gate_proj", shared + "up_proj"), gateUp.shards());
        assertEquals(shared + "down_proj", layerOf(shared + "down_proj.weight", LayerKind.LINEAR).name());

        assertEquals(L0 + "mlp.experts", layerOf(L0 + "mlp.experts.3.gate_proj.weight", LayerKind.MOE).name());
        assertEquals(L0 + "mlp.experts", layerOf(L0 + "mlp.experts.gate_up_proj", LayerKind.MOE).name());
        assertEquals("model.layers.0.block_sparse_moe.experts",
                layerOf("model.layers.0.block_sparse_moe.experts.1.w1.weight", LayerKind.MOE).name());
        assertEquals(qkv + "", attn + "qkv_proj " + qkv.shards());
    }

    @Test
    public void layersAreNamedAfterVllmsMapperRenames() {
        VllmWeightsMapper vl = VllmWeightsMapper.of("Qwen2_5_VLForConditionalGeneration");
        String hf = "model.language_model.layers.0.self_attn.";
        Set<String> has = new HashSet<>(Arrays.asList(hf + "q_proj.weight", hf + "k_proj.weight",
                hf + "v_proj.weight"));
        Layer qkv = Layer.of(hf + "v_proj.weight", LayerKind.LINEAR, has::contains, vl);
        String vllm = "language_model.model.layers.0.self_attn.";
        assertEquals(vllm + "qkv_proj", qkv.name());
        assertEquals(Arrays.asList(vllm + "q_proj", vllm + "k_proj", vllm + "v_proj"), qkv.shards());
        assertEquals("language_model.model.layers.0.mlp.experts", Layer.of(
                "model.language_model.layers.0.mlp.experts.7.down_proj.weight", LayerKind.MOE, n -> false, vl).name());

        VllmWeightsMapper internLm = VllmWeightsMapper.of("InternLM2ForCausalLM");
        Layer w13 = Layer.of("model.layers.1.feed_forward.w3.weight", LayerKind.LINEAR, n -> false, internLm);
        assertEquals("a stacked shard is its fused layer, whatever the checkpoint stores",
                "model.layers.1.feed_forward.gate_up_proj", w13.name());
        assertEquals(Arrays.asList("model.layers.1.feed_forward.w1", "model.layers.1.feed_forward.w3"),
                w13.shards());

        assertEquals("a weight vLLM drops keeps its checkpoint name", "mtp.layers.0.mlp.down_proj",
                Layer.of("mtp.layers.0.mlp.down_proj.weight", LayerKind.LINEAR, n -> false,
                        VllmWeightsMapper.of("Qwen3_5ForConditionalGeneration")).name());

        String unresolved = errorOf(() -> Layer.of("model.layers.0.mlp.down_proj.weight", LayerKind.LINEAR,
                n -> false, VllmWeightsMapper.of("Cosmos3EdgeForConditionalGeneration")));
        assertTrue(unresolved, unresolved.contains("Cosmos3EdgeForConditionalGeneration"));
    }

    @Test
    public void ignoreAndTargetsMatchTheMappedNames() {
        VllmWeightsMapper vl = VllmWeightsMapper.of("Qwen2_5_VLForConditionalGeneration");
        Layer down = Layer.of("model.layers.2.mlp.down_proj.weight", LayerKind.LINEAR, n -> false, vl);
        OnlineQuantConfig ignoring = parse("{\"linear\": \"fp8_per_block\","
                + " \"ignore\": [\"re:^language_model\\\\.model\\\\.layers\\\\.2\\\\..*\"]}");
        assertTrue(ignoring.matchesNames());
        assertNull(ignoring.select(LayerKind.LINEAR, down));
        assertSame("the checkpoint name does not match", OnlineQuant.FP8_PER_BLOCK,
                target(ignoring, LayerKind.LINEAR, layerOf("model.layers.2.mlp.down_proj.weight", LayerKind.LINEAR)));
        assertFalse(parse("{\"linear\": \"fp8_per_block\"}").matchesNames());
    }
}

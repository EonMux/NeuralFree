package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Hooks of {@code MistralLarge4Model} (LC/conversion/mistral4.py:15-129): the Mistral MoE format (remapped by
 * {@link HfMistralMoeHooks#remap}) with multi-head latent attention, the DeepSeek sparse-attention indexer on the
 * full-indexer layers, the {@code attention.rope_pattern} NoPE mix, the {@code attention.kv_source_map} cache sharing,
 * the LeRoPE factors and the tanh logit cap.
 *
 * <p>The class requires the layout the runtime implements (mistral4.py:19-44): a {@code dsa} block with SSMax scaling
 * that follows the main NoPE pattern, no query LoRA rank, a yarn block without magnitude scaling, and a
 * {@code kv_sharing.source_layers_str} when the block is present. The tensor hook writes the LeRoPE factors as the
 * ggml frequency factors, the per-head SSMax scales, the layer-normed indexer queries, the renamed indexer tensors
 * and the per-layer {@code wkv_b} split whose NoPE layers carry a widened non-rotary key head
 * (mistral4.py:91-129).</p>
 */
public final class HfMistralLarge4Hooks {

    /** The converter tensor key of the learned frequency factors (mistral4.py:94-95). */
    private static final String ROPE_FREQS = "ROPE_FREQS";

    private HfMistralLarge4Hooks() {
    }

    /** {@link Arch#MISTRAL_LARGE4}. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.MISTRAL_LARGE4;
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code MistralLarge4Model.__init__} (mistral4.py:19-44) over the Mistral MoE remap: the per-layer rope flags
     * ({@code nope}), indexer types ({@code dsa.sharing}) and KV sources ({@code kv_sharing}) expand over the layers,
     * and the layout the converter refuses is refused here with its key named.
     *
     * @throws GgufFormatException when a required block is missing or a value is outside the supported layout
     */
    public static HfConfig fixup(HfConfig cfg) {
        HfConfig fixed = HfMistralMoeHooks.remap(cfg);
        int nLayer = fixed.requireInt("num_hidden_layers");
        object(fixed, "moe");
        Map<String, Object> dsa = object(fixed, "dsa");
        expand(fixed, fixed.flat().get("nope"), "rope", nLayer, "nope");
        expand(fixed, dsa.get("sharing"), "full", nLayer, "dsa.sharing");
        kvSources(fixed, nLayer);

        Object ropeHeadDim = dsa.get("rope_head_dim");
        if (ropeHeadDim instanceof Number && ((Number) ropeHeadDim).intValue() != -1
                && ((Number) ropeHeadDim).intValue() != fixed.requireInt("qk_rope_head_dim")) {
            throw new GgufFormatException(fixed.source() + ": Mistral Large 4: dsa.rope_head_dim "
                    + ropeHeadDim + " differs from qk_rope_head_dim " + fixed.requireInt("qk_rope_head_dim")
                    + " (mistral4.py:40)");
        }
        if (!truthy(dsa.get("match_nope"))) {
            throw new GgufFormatException(fixed.source() + ": Mistral Large 4: the indexer must follow the NoPE"
                    + " pattern of the main attention (dsa.match_nope, mistral4.py:41)");
        }
        if (!truthy(dsa.get("ssmax_scaling"))) {
            throw new GgufFormatException(fixed.source() + ": Mistral Large 4: the indexer needs SSMax scaling"
                    + " (dsa.ssmax_scaling, mistral4.py:42)");
        }
        Object qLora = fixed.flat().get("q_lora_rank");
        if (qLora != null) {
            throw new GgufFormatException(fixed.source() + ": Mistral Large 4: q_lora_rank " + qLora
                    + " is not supported, the checkpoint must leave it unset (mistral4.py:43)");
        }
        Map<String, Object> yarn = object(fixed, "yarn");
        if (truthy(yarn.getOrDefault("apply_scale", Boolean.TRUE))) {
            throw new GgufFormatException(fixed.source() + ": Mistral Large 4: only YaRN without magnitude scaling"
                    + " is supported (yarn.apply_scale, mistral4.py:44)");
        }
        quantization(fixed);
        return fixed;
    }

    /**
     * {@code MistralLarge4Model.dequant_model} (mistral4.py:46-55): a checkpoint stored quantized must be block FP8.
     *
     * @throws GgufFormatException naming the unsupported quantization value
     */
    private static void quantization(HfConfig cfg) {
        Object raw = cfg.raw().raw("quantization");
        if (raw == null) {
            return;
        }
        if (!(raw instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": 'quantization' must be an object, got " + raw);
        }
        Map<?, ?> q = (Map<?, ?>) raw;
        if (!"fp8_e4m3".equals(q.get("qformat_weight"))) {
            throw new GgufFormatException(cfg.source() + ": Mistral Large 4: qformat_weight "
                    + q.get("qformat_weight") + " is not 'fp8_e4m3' (mistral4.py:49)");
        }
        if (!"BLOCK".equals(q.get("qscheme_weight"))) {
            throw new GgufFormatException(cfg.source() + ": Mistral Large 4: qscheme_weight "
                    + q.get("qscheme_weight") + " is not 'BLOCK' (mistral4.py:50)");
        }
    }

    /**
     * {@code expand} (mistral4.py:24-26): the comma-separated pattern repeated over the layers.
     *
     * @throws GgufFormatException when the pattern is not a string
     */
    private static List<String> expand(HfConfig cfg, Object raw, String dflt, int nLayer, String where) {
        Object value = truthy(raw) ? raw : dflt;
        if (!(value instanceof String)) {
            throw new GgufFormatException(cfg.source() + ": Mistral Large 4: '" + where + "' must be a"
                    + " comma-separated string, got " + value);
        }
        String[] values = ((String) value).split(",", -1);
        List<String> out = new ArrayList<>(nLayer);
        for (int i = 0; i < nLayer; i++) {
            out.add(values[i % values.length]);
        }
        return out;
    }

    /**
     * {@code layer_kv_source} (mistral4.py:31-37): the per-layer cache owner, its own layer when the block is absent;
     * a non-empty block without {@code source_layers_str} is not supported.
     */
    private static List<Integer> kvSources(HfConfig cfg, int nLayer) {
        Object raw = cfg.flat().get("kv_sharing");
        List<Integer> out = new ArrayList<>(nLayer);
        if (raw == null) {
            for (int il = 0; il < nLayer; il++) {
                out.add(il);
            }
            return out;
        }
        if (!(raw instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": 'kv_sharing' must be an object, got " + raw);
        }
        Map<?, ?> sharing = (Map<?, ?>) raw;
        Object sources = sharing.get("source_layers_str");
        if (sources == null) {
            if (!sharing.isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": unsupported kv_sharing config: " + sharing);
            }
            for (int il = 0; il < nLayer; il++) {
                out.add(il);
            }
            return out;
        }
        if (!(sources instanceof String)) {
            throw new GgufFormatException(cfg.source() + ": kv_sharing.source_layers_str must be a comma-separated"
                    + " string, got " + sources);
        }
        for (String part : ((String) sources).split(",", -1)) {
            try {
                out.add(Integer.parseInt(part.trim()));
            } catch (NumberFormatException e) {
                throw new GgufFormatException(cfg.source() + ": kv_sharing.source_layers_str holds '" + part
                        + "', not a layer index", e);
            }
        }
        if (out.size() != nLayer) {
            throw new GgufFormatException(cfg.source() + ": kv_sharing.source_layers_str lists " + out.size()
                    + " sources for " + nLayer + " layers (mistral4.py:37)");
        }
        return out;
    }

    private static Map<String, Object> object(HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' must be an object, got " + v);
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> object = (Map<String, Object>) v;
        return object;
    }

    private static boolean truthy(Object value) {
        if (value == null) {
            return false;
        }
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof Number) {
            return ((Number) value).doubleValue() != 0;
        }
        if (value instanceof String) {
            return !((String) value).isEmpty();
        }
        if (value instanceof List) {
            return !((List<?>) value).isEmpty();
        }
        return !(value instanceof Map) || !((Map<?, ?>) value).isEmpty();
    }

    /** The per-layer rope flags: the layers whose {@code nope} entry is {@code rope} (mistral4.py:28). */
    public static List<Boolean> ropePattern(HfConfig cfg) {
        List<String> types = expand(cfg, cfg.flat().get("nope"), "rope", cfg.requireInt("num_hidden_layers"),
                "nope");
        List<Boolean> out = new ArrayList<>(types.size());
        for (String type : types) {
            out.add("rope".equals(type));
        }
        return out;
    }

    /** The per-layer indexer flags: the layers whose {@code dsa.sharing} entry is {@code full} (mistral4.py:29). */
    public static List<Boolean> indexerTypes(HfConfig cfg) {
        Map<String, Object> dsa = object(cfg, "dsa");
        List<String> types = expand(cfg, dsa.get("sharing"), "full", cfg.requireInt("num_hidden_layers"),
                "dsa.sharing");
        List<Boolean> out = new ArrayList<>(types.size());
        for (String type : types) {
            out.add("full".equals(type));
        }
        return out;
    }

    // ------------------------------------------------------------------ filter

    /**
     * {@code MistralLarge4Model.filter_tensors} (mistral4.py:74-89): the DSpark {@code mtp_} tensors drop, the routed
     * experts take their HF names so the base class stacks them, and the inherited DeepSeek filter runs without the
     * Mistral MoE scale renames (the block-FP8 {@code qscale_*} names stay).
     */
    public static final HfNameFilter FILTER = (name, base) -> {
        if (name.startsWith("mtp_")) {
            return null;
        }
        String n = name;
        if (n.contains(".experts.") || n.endsWith(".experts")) {
            n = "model." + n.replace(".experts", ".mlp.experts")
                    .replace(".w1.", ".gate_proj.").replace(".w2.", ".down_proj.").replace(".w3.", ".up_proj.");
        }
        if (n.contains("sam_model") || n.contains("qwen2_model")) {
            return null;
        }
        return base.apply(n);
    };

    // ------------------------------------------------------------------ metadata

    /**
     * {@code MistralLarge4Model.set_gguf_parameters} (mistral4.py:57-72): the DeepSeek-V2 metadata, the Mistral yarn
     * and temperature configuration ({@link HfMistralNativeHooks#mistralConfig}), the indexer head count, key length,
     * top-k and per-layer types, the indexer LayerNorm epsilon, the rope pattern, the KV source map, the softmax
     * expert gating and the tanh final softcap.
     *
     * @throws GgufFormatException when a required key is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        HfMistralNativeHooks.mistralConfig(cfg, kv);
        Map<String, Object> dsa = object(cfg, "dsa");
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, integer(cfg, dsa, "n_heads", "dsa.n_heads"));
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, integer(cfg, dsa, "head_dim", "dsa.head_dim"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, integer(cfg, dsa, "topk", "dsa.topk"));
        kv.set(GgufKeys.ATTENTION_INDEXER_TYPES, indexerTypes(cfg));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, number(cfg, dsa, "norm_eps", "dsa.norm_eps", 1e-6));
        kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, ropePattern(cfg));
        kv.set(GgufKeys.ATTENTION_KV_SOURCE_MAP, kvSources(cfg, cfg.requireInt("num_hidden_layers")));
        kv.set(GgufKeys.EXPERT_GATING_FUNC, 1L);
        kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, cfg.findFloat("softmax_tanh_gating").orElseThrow(
                () -> new GgufFormatException(cfg.source()
                        + ": required key 'softmax_tanh_gating' is missing (mistral4.py:72)")));
    }

    private static long integer(HfConfig cfg, Map<String, Object> map, String key, String where) {
        Object v = map.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be an integer, got " + v);
        }
        return ((Number) v).longValue();
    }

    private static double number(HfConfig cfg, Map<String, Object> map, String key, String where, double dflt) {
        Object v = map.get(key);
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be a number, got " + v);
        }
        return ((Number) v).doubleValue();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code MistralLarge4Model.modify_tensors} (mistral4.py:91-129): the LeRoPE factors, the per-head SSMax scales,
     * the indexer query norm, the indexer name rewrite and the per-layer {@code wkv_b} split; every other tensor maps
     * through {@link HfDeepseekHooks#tensors} with its expert stacking.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if ("lerope.alpha".equals(hfName)) {
            out.add(ctx.names().ggufName(ROPE_FREQS, -1, ".weight"), ElementTransforms.expNeg(src));
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && (hfName.endsWith(".ssmax_scaling.s") || hfName.endsWith(".ssmax_scaling.b"))) {
            String tensor = hfName.contains(".indexer.") ? "INDEXER_SSMAX" : "ATTN_SSMAX";
            String suffix = hfName.endsWith(".s") ? ".weight" : ".bias";
            out.add(ctx.names().ggufName(tensor, bid, suffix), ElementTransforms.convert(src, GGMLType.F32));
            return;
        }
        if (bid >= 0 && hfName.contains(".indexer.q_input_norm.")) {
            String suffix = hfName.endsWith(".weight") ? ".weight" : ".bias";
            out.add(ctx.names().ggufName("INDEXER_Q_NORM", bid, suffix), src);
            return;
        }
        String name = hfName;
        if (bid >= 0 && name.contains(".indexer.")) {
            name = "model." + name.replace(".attention.", ".self_attn.");
        }
        if (bid >= 0 && name.endsWith(".wkv_b.weight")) {
            splitWkvB(ctx, name, bid, src, out);
            return;
        }
        HfDeepseekHooks.tensors(ctx, name, src, out, true, true);
    }

    /**
     * {@code MistralLarge4Model.modify_tensors}' {@code wkv_b} split (mistral4.py:111-127): the NoPE layers have a
     * larger non-rotary key head, so each layer splits into its own transposed {@code k_b} and {@code v_b}.
     */
    private static void splitWkvB(HfLoadContext ctx, String hfName, int bid, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        int nHead = cfg.requireInt("num_attention_heads");
        int vHead = cfg.requireInt("v_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        if (!ropePattern(cfg).get(bid)) {
            qkNope += cfg.requireInt("qk_rope_head_dim");
        }
        long[] shape = src.shape();
        if (shape.length != 2 || shape[0] != (long) nHead * (qkNope + vHead)) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape "
                    + java.util.Arrays.toString(shape) + "; wkv_b must have " + nHead + " heads of " + qkNope
                    + " key and " + vHead + " value rows (mistral4.py:119)");
        }
        RowSource kv = ElementTransforms.reshape(src, nHead, qkNope + vHead, -1);
        RowSource k = RowTransforms.sliceDim(kv, 1, 0, qkNope);
        RowSource v = RowTransforms.sliceDim(kv, 1, qkNope, vHead);
        out.add(ctx.names().ggufName("ATTN_K_B", bid, ".weight"), ExpertTransforms.transposeLast2(k));
        out.add(ctx.names().ggufName("ATTN_V_B", bid, ".weight"), v);
    }

    // ------------------------------------------------------------------ vocabulary

    /** {@code MistralMoeModel.set_vocab} (mistral.py:170-171): the mistral format vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.mistral(HfHookSupport.vocabInput(ctx), pre);
    }

    /** {@code DeepseekV2Model.prepare_tensors} (deepseek.py:454-462): the stacked experts must be complete. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfDeepseekHooks.extra(ctx, out);
    }
}

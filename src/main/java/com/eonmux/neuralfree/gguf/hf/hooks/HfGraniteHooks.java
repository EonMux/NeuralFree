package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.UnaryOperator;

/**
 * Hooks of llama.cpp's {@code GraniteModel} (LC/conversion/granite.py:19-75) and {@code GraniteMoeModel}
 * (granite.py:181-232), the Llama subclasses the Granite families build on: the multiplier metadata, the
 * Granite 4 Vision deepstack mapping, the multimodal tensor filter and the split of the fused JetMoe expert and shared
 * MLP projections. {@code GraniteModel} keeps {@link HfLlamaHooks#tensors}, {@link HfLlamaHooks#extra} and
 * {@link HfLlamaHooks#vocab}.
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}.</p>
 */
public final class HfGraniteHooks {

    /** {@link #filter} as an {@link HfNameFilter}, for entries and subclass filters that wrap it. */
    public static final HfNameFilter FILTER = HfGraniteHooks::filter;

    private static final String INPUT_LINEAR = "block_sparse_moe.input_linear.weight";
    private static final String SHARED_INPUT_LINEAR = "shared_mlp.input_linear.weight";
    private static final String SHARED_OUTPUT_LINEAR = "shared_mlp.output_linear.weight";
    private static final String DEEPSTACK_LAYER_MAP = "deepstack_layer_map";
    private static final String SPATIAL_TARGET_LAYERS = "spatial_target_layers";
    private static final String SPATIAL_VISION_LAYER = "spatial_vision_layer";
    /** The fused gate/up projections hold two halves along dim -2. */
    private static final int FUSED_PARTS = 2;
    private static final int FUSED_DIM = -2;

    private HfGraniteHooks() {
    }

    // ------------------------------------------------------------------ names

    /**
     * {@code GraniteModel.filter_tensors} (granite.py:62-74): the multimodal tensors (names starting with
     * {@code encoder.} or containing {@code image_}, {@code layerwise_projectors} or {@code spatial_projectors}) are
     * dropped; the rest go through {@link HfLlamaHooks#filter}.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        if (name.startsWith("encoder.") || name.contains("image_") || name.contains("layerwise_projectors")
                || name.contains("spatial_projectors")) {
            return null;
        }
        return HfLlamaHooks.filter(name, base);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code GraniteModel.set_gguf_parameters} (granite.py:23-60): {@code head_dim} is popped first, so the generic
     * metadata has no key/value lengths and the RoPE width is {@code hidden_size / num_attention_heads}; then the
     * truthy multipliers {@code attention_multiplier}, {@code embedding_multiplier}, {@code residual_multiplier} and
     * {@code logits_scaling} as the attention, embedding, residual and logit scales; then the deepstack mapping of
     * Granite 4 Vision ({@link #deepstackMapping}).
     *
     * @throws GgufFormatException when a multiplier is not a number or the deepstack layers are invalid
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.remove(GgufKeys.ATTENTION_KEY_LENGTH);
        kv.remove(GgufKeys.ATTENTION_VALUE_LENGTH);
        HfLlamaHooks.llamaKv(ctx, kv, false);
        HfConfig cfg = ctx.config();
        setTruthy(cfg, kv, "attention_multiplier", GgufKeys.ATTENTION_SCALE);
        setTruthy(cfg, kv, "embedding_multiplier", GgufKeys.EMBEDDING_SCALE);
        setTruthy(cfg, kv, "residual_multiplier", GgufKeys.RESIDUAL_SCALE);
        setTruthy(cfg, kv, "logits_scaling", GgufKeys.LOGIT_SCALE);
        if (truthy(cfg.flat().get(SPATIAL_TARGET_LAYERS)) || truthy(cfg.flat().get(DEEPSTACK_LAYER_MAP))) {
            kv.set(GgufKeys.DEEPSTACK_MAPPING, deepstackMapping(cfg, ctx.blockCount()));
        }
    }

    /**
     * {@code GraniteMoeModel.set_gguf_parameters} (granite.py:186-195): {@link #meta}, then a truthy
     * {@code shared_intermediate_size} as {@code expert_shared_feed_forward_length}.
     *
     * @throws GgufFormatException when a value has the wrong type
     */
    public static void moeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        meta(ctx, kv, plan);
        Optional<Object> shared = ctx.config().find("shared_intermediate_size");
        if (shared.isPresent() && truthy(shared.get())) {
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, ctx.config().requireInt("shared_intermediate_size"));
        }
    }

    /** Writes {@code key} from {@code name} when the value is truthy, as granite.py:39-50's walrus tests do. */
    private static void setTruthy(HfConfig cfg, HfMetadataBuilder kv, String name, String key) {
        Optional<Object> v = cfg.find(name);
        if (!v.isPresent() || !truthy(v.get())) {
            return;
        }
        if (!(v.get() instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a number, got " + v.get());
        }
        kv.set(key, ((Number) v.get()).doubleValue());
    }

    /** Python truthiness of a JSON value. */
    private static boolean truthy(Object v) {
        if (v == null) {
            return false;
        }
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof BigInteger) {
            return ((BigInteger) v).signum() != 0;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        if (v instanceof Map) {
            return !((Map<?, ?>) v).isEmpty();
        }
        return true;
    }

    /**
     * The per-layer deepstack mapping of {@code GraniteModel.set_gguf_parameters} (granite.py:52-60) over
     * {@code Granite4VisionMmprojModel.get_normalized_projector_map} (granite.py:697-732): the {@code layerwise}
     * projectors of the sorted {@code deepstack_layer_map} pairs and the {@code spatial} projectors of
     * {@code spatial_target_layers}, sorted stably by LLM layer; each projector after the first marks its LLM layer
     * with its index, every other layer is -1.
     *
     * @throws GgufFormatException when the maps are malformed, the text or vision layer count is missing, or a layer
     *                             is outside the model
     */
    public static List<Integer> deepstackMapping(HfConfig cfg, int blockCount) {
        JsonView root = cfg.raw();
        int nText = requireLayers(root, "text_config");
        int nVision = requireLayers(root, "vision_config");
        List<long[]> projectors = new ArrayList<>();
        List<long[]> pairs = new ArrayList<>();
        Object deepstack = cfg.flat().get(DEEPSTACK_LAYER_MAP);
        if (truthy(deepstack)) {
            for (Object o : requireList(cfg, DEEPSTACK_LAYER_MAP, deepstack)) {
                List<?> pair = o instanceof List ? (List<?>) o : null;
                if (pair == null || pair.size() != 2) {
                    throw new GgufFormatException(cfg.source() + ": '" + DEEPSTACK_LAYER_MAP
                            + "' must hold [vision_layer, llm_layer] pairs, got " + o);
                }
                pairs.add(new long[]{integer(cfg, DEEPSTACK_LAYER_MAP, pair.get(0)),
                        integer(cfg, DEEPSTACK_LAYER_MAP, pair.get(1))});
            }
            pairs.sort(Comparator.<long[]>comparingLong(p -> p[0]).thenComparingLong(p -> p[1]));
            for (long[] p : pairs) {
                long vision = p[0] < 0 ? nVision + p[0] : p[0];
                long llm = p[1] < 0 ? nText + p[1] : p[1];
                projectors.add(new long[]{vision, llm});
            }
        }
        Object spatial = cfg.flat().get(SPATIAL_TARGET_LAYERS);
        if (truthy(spatial)) {
            Object sv = cfg.flat().get(SPATIAL_VISION_LAYER);
            long spatialVision = sv == null ? -1 : integer(cfg, SPATIAL_VISION_LAYER, sv);
            if (spatialVision < 0) {
                spatialVision = nVision + spatialVision;
            }
            for (Object o : requireList(cfg, SPATIAL_TARGET_LAYERS, spatial)) {
                projectors.add(new long[]{spatialVision, integer(cfg, SPATIAL_TARGET_LAYERS, o)});
            }
        }
        projectors.sort(Comparator.comparingLong(p -> p[1]));
        Integer[] mapping = new Integer[blockCount];
        Arrays.fill(mapping, -1);
        for (int idx = 1; idx < projectors.size(); idx++) {
            long llm = projectors.get(idx)[1];
            long at = llm < 0 ? blockCount + llm : llm;
            if (at < 0 || at >= blockCount) {
                throw new GgufFormatException(cfg.source() + ": deepstack LLM layer " + llm + " is outside the "
                        + blockCount + " text layers");
            }
            mapping[(int) at] = idx;
        }
        return Collections.unmodifiableList(Arrays.asList(mapping));
    }

    private static int requireLayers(JsonView root, String sub) {
        Object v = root.raw(sub);
        if (!(v instanceof Map) || ((Map<?, ?>) v).get("num_hidden_layers") == null) {
            throw new GgufFormatException(root.where(sub) + ".num_hidden_layers is required for the Granite 4"
                    + " Vision deepstack mapping");
        }
        return root.obj(sub).i32("num_hidden_layers");
    }

    private static List<?> requireList(HfConfig cfg, String key, Object v) {
        if (!(v instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a list, got " + v);
        }
        return (List<?>) v;
    }

    private static long integer(HfConfig cfg, String key, Object v) {
        if (!(v instanceof Long || v instanceof Integer)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must hold integers, got " + v);
        }
        long x = ((Number) v).longValue();
        if (Math.abs(x) > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' layer " + x + " is out of range");
        }
        return x;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code GraniteMoeModel.modify_tensors} (granite.py:197-232): the JetMoe {@code block_sparse_moe.input_linear}
     * {@code [E, 2F, D]} split along dim -2 into the gate and up experts; {@code shared_mlp.input_linear} split into
     * the shared-expert gate and up (a dense FFN gate and up when {@code num_local_experts} is falsy) and, without
     * experts, {@code shared_mlp.output_linear} as the dense FFN down; every other tensor through
     * {@link HfLlamaHooks#tensors}. The split tensors bypass the Llama permutation, as the converter yields them
     * through {@code ModelBase.modify_tensors}.
     *
     * @throws GgufFormatException when a fused tensor is not twice the configured intermediate size along dim -2
     * @throws IOException         when a tensor the Llama path reads cannot be read
     */
    public static void moeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (hfName.endsWith(INPUT_LINEAR)) {
            RowSource[] gateUp = splitFused(cfg, hfName, src, "intermediate_size");
            int bid = HfHookSupport.layerOf(hfName);
            out.add(ctx.names().ggufName("FFN_GATE_EXP", bid, ".weight"), gateUp[0]);
            out.add(ctx.names().ggufName("FFN_UP_EXP", bid, ".weight"), gateUp[1]);
            return;
        }
        boolean hasExperts = truthy(cfg.flat().get("num_local_experts"));
        if (hfName.endsWith(SHARED_INPUT_LINEAR)) {
            RowSource[] gateUp = splitFused(cfg, hfName, src, "shared_intermediate_size");
            int bid = HfHookSupport.layerOf(hfName);
            out.add(ctx.names().ggufName(hasExperts ? "FFN_GATE_SHEXP" : "FFN_GATE", bid, ".weight"), gateUp[0]);
            out.add(ctx.names().ggufName(hasExperts ? "FFN_UP_SHEXP" : "FFN_UP", bid, ".weight"), gateUp[1]);
            return;
        }
        if (!hasExperts && hfName.endsWith(SHARED_OUTPUT_LINEAR)) {
            out.add(ctx.names().ggufName("FFN_DOWN", HfHookSupport.layerOf(hfName), ".weight"), src);
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /**
     * {@code data_torch.split(ffn_dim, dim=-2)} of a fused gate/up projection whose dim -2 must be
     * {@code 2 * <sizeKey>}: the gate and up halves.
     */
    private static RowSource[] splitFused(HfConfig cfg, String hfName, RowSource src, String sizeKey) {
        int ffnDim = cfg.requireInt(sizeKey);
        long[] shape = src.shape();
        if (shape.length < 2 || shape[shape.length - 2] != 2L * ffnDim) {
            throw new GgufFormatException("tensor '" + hfName + "' " + Arrays.toString(shape) + ": merged FFN tensor"
                    + " size must be 2 * " + sizeKey + " (" + 2L * ffnDim + ") along dim -2");
        }
        return ExpertTransforms.splitExperts(src, FUSED_DIM, FUSED_PARTS);
    }
}

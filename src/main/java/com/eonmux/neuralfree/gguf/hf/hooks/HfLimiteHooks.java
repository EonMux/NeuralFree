package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Hooks of {@code LimiteModel} (LC/conversion/limite.py:13-184): the pre-norm transformer with value embeddings, the
 * exclusive self attention, learned residual scales and the MUDD dense cross-layer connections.
 *
 * <p>The class pins the checkpoint layout it can convert ({@code __init__}, limite.py:18-55): an interleaved-pair RoPE
 * on the sliding-window layers, weightless RMS norms, a SwiGLU FFN, the value-embedding head slice, a global NoPE
 * window and the sigmoid logit cap. The metadata rescales the checkpoint's frequencies into ggml's convention
 * (limite.py:72-74), widens the window by the inclusive HF bound and records the sigmoid capping and the MUDD tap
 * table (limite.py:60-91); the tensor hook folds the per-layer scales into the projections, swaps the RoPE pairs,
 * pads the gate weights and merges the MUDD mixers (limite.py:115-184).</p>
 */
public final class HfLimiteHooks {

    /** {@code torch.finfo(torch.float32).eps}, the weightless RMS norm's epsilon (limite.py:67-68). */
    private static final float FLOAT32_EPS = 1.1920929E-7f;

    /** The F32 mixing weights of {@code tensor_force_quant} (limite.py:93-101). */
    private static final Pattern FULL_PRECISION = Pattern.compile(
            "blk\\.\\d+\\.(attn_gate|attn_ve_gate|mudd_down|mudd_up)(\\.(weight|bias))?$");

    /** The checkpoint layout the converter accepts (limite.py:22-38). */
    private static final Map<String, Object> EXPECTED = expected();

    private HfLimiteHooks() {
    }

    // ------------------------------------------------------------------ configuration

    /** {@link Arch#LIMITE}. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.LIMITE;
    }

    /**
     * {@code LimiteModel.__init__} (limite.py:18-55): every layout key must equal the value the runtime implements,
     * the sigmoid logit cap must be the configured kind, and a MUDD model must carry the MLP mixers.
     *
     * @throws GgufFormatException naming the unsupported key when the checkpoint deviates
     */
    public static HfConfig fixup(HfConfig cfg) {
        for (Map.Entry<String, Object> e : EXPECTED.entrySet()) {
            Object actual = cfg.flat().get(e.getKey());
            if (!same(actual, e.getValue())) {
                throw new GgufFormatException(cfg.source() + ": Limite: unsupported " + e.getKey() + "=" + actual
                        + ", expected " + e.getValue() + " (limite.py:22-38)");
            }
        }
        Object cap = cfg.flat().get("softcap_logits");
        if (!(cap instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key 'softcap_logits' is missing"
                    + " (limite.py:42-43)");
        }
        Object kind = ((Map<?, ?>) cap).get("kind");
        if (!"sigmoid".equals(kind)) {
            throw new GgufFormatException(cfg.source() + ": Limite: unsupported softcap_logits kind " + kind
                    + " (limite.py:42-43)");
        }
        int pairs = cfg.requireInt("rope_n_pairs");
        if (pairs < 2) {
            throw new GgufFormatException(cfg.source() + ": 'rope_n_pairs' must be at least 2, got " + pairs
                    + " (limite.py:48, 74)");
        }
        muddTaps(cfg);
        return cfg;
    }

    private static Map<String, Object> expected() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("pos_mode", "rope");
        m.put("qk_norm", "rms_pre_rope");
        m.put("rope_style", "interleaved_pairs_odd_lane_sign_flip");
        m.put("rope_per_layer", false);
        m.put("rms_norm_has_weight", false);
        m.put("rms_norm_eps_mode", "torch_finfo_default");
        m.put("mlp_type", "swiglu");
        m.put("attn_gate_applied", "per_head_before_o_proj");
        m.put("ve_head_slice", "first_num_key_value_heads");
        m.put("ve_applied_before_qk_norm", true);
        m.put("global_nope", true);
        m.put("global_window", -1L);
        m.put("mudd_r_site", "resid");
        m.put("tie_word_embeddings", true);
        m.put("final_softcap", 0.0);
        return Collections.unmodifiableMap(m);
    }

    /** Python equality over JSON values: numbers compare numerically, booleans by value. */
    private static boolean same(Object actual, Object expected) {
        if (actual instanceof Number && expected instanceof Number) {
            return ((Number) actual).doubleValue() == ((Number) expected).doubleValue();
        }
        return expected.equals(actual);
    }

    /**
     * {@code LimiteModel.mudd_taps} (limite.py:50-55): the layer to its tap list, empty when the checkpoint has no
     * MUDD; a MUDD model without the MLP mixers or with a tap list of another length is refused.
     */
    private static Map<Integer, List<Integer>> muddTaps(HfConfig cfg) {
        if (!truthy(cfg.flat().get("mudd"))) {
            return Collections.emptyMap();
        }
        if (!truthy(cfg.flat().get("mudd_mlp"))) {
            throw new GgufFormatException(cfg.source() + ": Limite: mudd without mudd_mlp is not supported"
                    + " (limite.py:52-53)");
        }
        Object raw = cfg.flat().get("mudd_tap_idx");
        if (!(raw instanceof Map)) {
            throw new GgufFormatException(cfg.source() + ": required key 'mudd_tap_idx' is missing"
                    + " (limite.py:54)");
        }
        int nTaps = cfg.requireInt("mudd_taps");
        Map<Integer, List<Integer>> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) raw).entrySet()) {
            int layer = layerIndex(cfg, e.getKey());
            List<Integer> taps = intList(cfg, e.getValue(), "mudd_tap_idx." + e.getKey());
            if (taps.size() != nTaps) {
                throw new GgufFormatException(cfg.source() + ": Limite: mudd_tap_idx." + e.getKey() + " has "
                        + taps.size() + " taps, mudd_taps says " + nTaps + " (limite.py:55)");
            }
            out.put(layer, taps);
        }
        return Collections.unmodifiableMap(out);
    }

    private static int layerIndex(HfConfig cfg, Object key) {
        try {
            int layer = Integer.parseInt(String.valueOf(key));
            if (layer >= 0) {
                return layer;
            }
        } catch (NumberFormatException e) {
            // fall through to the message below
        }
        throw new GgufFormatException(cfg.source() + ": mudd_tap_idx key '" + key + "' is not a layer index");
    }

    private static List<Integer> intList(HfConfig cfg, Object value, String name) {
        if (!(value instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be an array, got " + value);
        }
        List<Integer> out = new ArrayList<>();
        for (Object item : (List<?>) value) {
            if (!(item instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": '" + name + "' must hold integers, got " + item);
            }
            out.add(((Number) item).intValue());
        }
        return out;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code LimiteModel.set_gguf_parameters} (limite.py:60-91): the generic keys, then the head width, the float32
     * RMS epsilon, the attention scale, the XSA epsilon, the ggml-convention RoPE width and base, the inclusive
     * sliding window with its per-layer pattern, the sigmoid logit cap and the MUDD tap table.
     *
     * @throws GgufFormatException when a required key is missing
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int headDim = cfg.requireInt("head_dim");
        int pairs = cfg.requireInt("rope_n_pairs");
        long nLayer = cfg.requireInt("num_hidden_layers");

        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, (long) headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, (long) headDim);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, FLOAT32_EPS);
        kv.set(GgufKeys.ATTENTION_SCALE, requireDouble(cfg, "attention_softmax_scale"));
        kv.set(GgufKeys.ATTENTION_XSA_EPS, truthy(cfg.flat().get("xsa"))
                ? requireDouble(cfg, "xsa_normalize_eps") : 0.0);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (long) (2 * pairs));
        kv.set(GgufKeys.ROPE_FREQ_BASE, Math.pow(requireDouble(cfg, "rope_base_local"),
                (double) pairs / (double) (pairs - 1)));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window") + 1L);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingPattern(cfg, nLayer));

        Map<?, ?> cap = (Map<?, ?>) cfg.flat().get("softcap_logits");
        kv.set(GgufKeys.FINAL_LOGIT_SIGMOID_CAPPING, Arrays.asList(
                number(cfg, cap, "a", "softcap_logits.a"),
                number(cfg, cap, "b", "softcap_logits.b"),
                number(cfg, cap, "c", "softcap_logits.c")));

        Map<Integer, List<Integer>> taps = muddTaps(cfg);
        if (!taps.isEmpty()) {
            int nTaps = cfg.requireInt("mudd_taps");
            List<Integer> indices = new ArrayList<>(Collections.nCopies(Math.toIntExact(nLayer * nTaps), -1));
            for (Map.Entry<Integer, List<Integer>> e : taps.entrySet()) {
                for (int i = 0; i < nTaps; i++) {
                    indices.set(e.getKey() * nTaps + i, e.getValue().get(i));
                }
            }
            kv.set(GgufKeys.MUDD_FEED_FORWARD_LENGTH, (long) cfg.requireInt("mudd_inter"));
            kv.set(GgufKeys.MUDD_TAP_COUNT, (long) nTaps);
            kv.set(GgufKeys.MUDD_TAP_INDICES, indices);
        }
    }

    /** {@code [il not in global_layers for il in range(block_count)]} (limite.py:79). */
    private static List<Boolean> slidingPattern(HfConfig cfg, long nLayer) {
        List<Integer> global = cfg.findIntList(Collections.<Integer>emptyList(), "global_layers");
        List<Boolean> out = new ArrayList<>();
        for (int il = 0; il < nLayer; il++) {
            out.add(!global.contains(il));
        }
        return out;
    }

    private static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(
                cfg.source() + ": required key '" + key + "' is missing"));
    }

    private static double number(HfConfig cfg, Map<?, ?> map, String key, String where) {
        Object v = map.get(key);
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": '" + where + "' must be a number, got " + v);
        }
        return ((Number) v).doubleValue();
    }

    // ------------------------------------------------------------------ tensors

    /**
     * {@code LimiteModel.modify_tensors} (limite.py:115-184): the value-embedding table, the MUDD mixers merged from
     * their shared sources, the per-layer scale folds, the RoPE pair swap, the padded attention gates, the XSA alpha
     * and the learned residual scales; every other tensor maps through the generated pattern tables.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if ("model.value_embeds.weight".equals(hfName)) {
            out.add(ctx.names().ggufName("VALUE_EMBD", -1, ".weight"), valueEmbeddings(cfg, src));
            return;
        }
        if ("model.mudd.dense1".equals(hfName)) {
            for (int layer : muddTaps(cfg).keySet()) {
                out.add(ctx.names().ggufName("MUDD_DOWN", layer, ".weight"), src);
            }
            return;
        }
        if ("model.mudd.dense2".equals(hfName)) {
            addMuddMixers(ctx, src, out);
            return;
        }
        if (hfName.startsWith("model.mudd.")) {
            out.drop(hfName, "folded into the MUDD mixers; the converter drops it (limite.py:144-145)");
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        String prefix = "model.layers." + bid + ".";
        if (bid < 0 || !hfName.startsWith(prefix)) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        String name = hfName.substring(prefix.length());
        layerTensor(ctx, cfg, hfName, prefix, name, bid, src, out);
    }

    /** {@code model.value_embeds.weight} (limite.py:122-126): the first KV heads, padded to the head width and scaled. */
    private static RowSource valueEmbeddings(HfConfig cfg, RowSource src) {
        int kv = cfg.requireInt("num_key_value_heads");
        int headDim = cfg.requireInt("head_dim");
        int veDim = cfg.requireInt("ve_dim");
        long tokens = src.shape()[0];
        RowSource ve = ElementTransforms.reshape(src, tokens, -1, veDim);
        ve = RowTransforms.sliceDim(ve, 1, 0, kv);
        ve = ColumnTransforms.padCols(ve, headDim, 0f);
        ve = ElementTransforms.reshape(ve, tokens, -1);
        return ElementTransforms.mulScalar(ve, (float) requireDouble(cfg, "ve_gate_scale"));
    }

    /** {@code model.mudd.dense2}/{@code dense2_mlp}/{@code bias}/{@code bias_mlp} (limite.py:132-143). */
    private static void addMuddMixers(HfLoadContext ctx, RowSource dense2, TensorSink out) throws IOException {
        HfConfig cfg = ctx.config();
        RowSource dense2Mlp = ctx.index().get("model.mudd.dense2_mlp");
        RowSource bias = ctx.index().get("model.mudd.bias");
        RowSource biasMlp = ctx.index().get("model.mudd.bias_mlp");
        for (Map.Entry<Integer, List<Integer>> e : muddTaps(cfg).entrySet()) {
            int layer = e.getKey();
            int n = e.getValue().size();
            RowSource w = RowTransforms.concatRows(Arrays.asList(
                    blockSlice(dense2, layer, n), blockSlice(dense2Mlp, layer, n)));
            out.add(ctx.names().ggufName("MUDD_UP", layer, ".weight"), w);
            RowSource b = RowTransforms.concatRows(Arrays.asList(
                    blockSlice(bias, layer, n), blockSlice(biasMlp, layer, n)));
            out.add(ctx.names().ggufName("MUDD_UP", layer, ".bias"), b);
        }
    }

    /** {@code src[layer, :n]}: the first {@code n} entries of the layer's dim-1 block, its other dims kept. */
    private static RowSource blockSlice(RowSource src, int layer, int n) {
        RowSource s = RowTransforms.sliceDim(src, 1, 0, n);
        s = RowTransforms.sliceDim(s, 0, layer, 1);
        long[] shape = s.shape();
        long[] target = Arrays.copyOfRange(shape, 1, shape.length);
        return ElementTransforms.reshape(s, target);
    }

    /** One block's tensor surgery; {@code bid} is the layer and {@code name} the name inside {@code prefix}. */
    private static void layerTensor(HfLoadContext ctx, HfConfig cfg, String hfName, String prefix, String name,
                                    int bid, RowSource src, TensorSink out) throws IOException {
        int headDim = cfg.requireInt("head_dim");
        int pairs = cfg.requireInt("rope_n_pairs");
        if ("self_attn.q_proj.weight".equals(name) || "self_attn.k_proj.weight".equals(name)
                || "self_attn.v_proj.weight".equals(name)) {
            RowSource w = ElementTransforms.mulScalar(src, load(ctx, prefix + "self_attn.qkv_scale"));
            if ("self_attn.q_proj.weight".equals(name)) {
                w = swapRopePairs(w, cfg.requireInt("num_attention_heads"), headDim, pairs);
            } else if ("self_attn.k_proj.weight".equals(name)) {
                w = swapRopePairs(w, cfg.requireInt("num_key_value_heads"), headDim, pairs);
            }
            out.add(ctx.names().mapOrThrow(hfName), w);
            return;
        }
        if ("self_attn.o_proj.weight".equals(name)) {
            float scale = load(ctx, prefix + "self_attn.o_scale") * load(ctx, prefix + "post_lambda_attn")
                    * (float) requireDouble(cfg, "attn_gate_scale");
            out.add(ctx.names().mapOrThrow(hfName), ElementTransforms.mulScalar(src, scale));
            return;
        }
        if ("mlp.down_proj.weight".equals(name)) {
            out.add(ctx.names().mapOrThrow(hfName),
                    ElementTransforms.mulScalar(src, load(ctx, prefix + "post_lambda_mlp")));
            return;
        }
        if ("self_attn.attn_gate".equals(name)) {
            long hidden = cfg.requireInt("hidden_size");
            out.add(ctx.names().ggufName("ATTN_GATE", bid, ".weight"),
                    ColumnTransforms.padCols(src, hidden, 0f));
            return;
        }
        if ("self_attn.ve_gate".equals(name)) {
            long hidden = cfg.requireInt("hidden_size");
            RowSource first = RowTransforms.sliceRows(src, 0, cfg.requireInt("num_key_value_heads"));
            out.add(ctx.names().ggufName("ATTN_VE_GATE", bid, ".weight"),
                    ColumnTransforms.padCols(first, hidden, 0f));
            return;
        }
        if ("self_attn.xsa_alpha".equals(name)) {
            if (truthy(cfg.flat().get("xsa")) && xsaLayers(cfg).contains(bid)) {
                out.add(ctx.names().ggufName("ATTN_XSA_ALPHA", bid, ".weight"),
                        ElementTransforms.tanh(src));
            }
            return;
        }
        if ("resid_lambda_attn".equals(name)) {
            out.add(ctx.names().ggufName("ATTN_RESID_SCALE", bid, ".weight"),
                    ElementTransforms.reshape(src, 1));
            return;
        }
        if ("resid_lambda_mlp".equals(name)) {
            out.add(ctx.names().ggufName("FFN_RESID_SCALE", bid, ".weight"),
                    ElementTransforms.reshape(src, 1));
            return;
        }
        if ("self_attn.qkv_scale".equals(name) || "self_attn.o_scale".equals(name)
                || "post_lambda_attn".equals(name) || "post_lambda_mlp".equals(name)) {
            out.drop(hfName, "folded into the projections; the converter drops it (limite.py:180-182)");
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    private static List<Integer> xsaLayers(HfConfig cfg) {
        return cfg.findIntList("xsa_layers").orElseThrow(() -> new GgufFormatException(
                cfg.source() + ": required key 'xsa_layers' is missing (limite.py:171)"));
    }

    /** The scalar value of the checkpoint tensor {@code name}. */
    private static float load(HfLoadContext ctx, String name) throws IOException {
        float[] values = HfHookSupport.floats(ctx.index().get(name), name);
        if (values.length == 0) {
            throw new GgufFormatException("tensor '" + name + "' is empty; a scale needs one value");
        }
        return values[0];
    }

    /** {@code _swap_rope_pairs} (limite.py:103-109): the two lanes of every pair up to {@code 2 * pairs} swap. */
    private static RowSource swapRopePairs(RowSource src, int nHead, int headDim, int pairs) {
        int[] order = new int[nHead * headDim];
        for (int row = 0; row < order.length; row++) {
            int lane = row % headDim;
            int swapped = lane < 2 * pairs ? lane ^ 1 : lane;
            order[row] = (row / headDim) * headDim + swapped;
        }
        return RowTransforms.permuteRowGroups(src, 0, 1, order);
    }

    /**
     * {@code LimiteModel.tensor_force_quant} (limite.py:93-101): the gate and MUDD weights stay F32.
     */
    public static GGMLType forceType(String ggufName, int nDims, ElementFormat source,
                                     ModelSourceOptions.OutType outType) {
        return FULL_PRECISION.matcher(ggufName).find() ? GGMLType.F32 : null;
    }

    // ------------------------------------------------------------------ vocabulary

    /** {@code LimiteModel.set_vocab} (limite.py:57-58): the GPT-2 vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /** Python truthiness of a configuration value. */
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
}

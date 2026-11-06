package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksA;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Locale;
import java.util.Map;


/**
 * Hooks of the llama.cpp classic-dense converter classes, part B: {@code CommandR2Model}
 * and {@code Cohere2Model} (LC/conversion/command_r.py:17-82), {@code DeciModel} (deci.py:17-185),
 * {@code ExaoneModel} (exaone.py:19-40), {@code Exaone4Model} (exaone.py:65-102) and
 * {@code Exaone4_5_TextModel} (exaone.py:221-285).
 *
 * <p>Every method is a public static hook reference that the table composes and subclasses reuse where the converter
 * calls {@code super()}.</p>
 */
public final class HfDenseClassicHooksB {

    private HfDenseClassicHooksB() {
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                + "' is missing"));
    }

    // ------------------------------------------------------------------ Command R2 and Cohere 2

    /**
     * {@code CommandR2Model.__init__}'s context fixup (command_r.py:34-40): {@code max_position_embeddings} is
     * filled from {@code model_max_length} first, since the aya-23 models omit it.
     */
    public static HfConfig commandR2Fixup(HfConfig cfg) {
        return cfg.withFlat(Collections.singletonMap("max_position_embeddings",
                cfg.findInt(cfg.requireInt("max_position_embeddings"), "model_max_length",
                        "max_position_embeddings")));
    }

    /**
     * {@code CommandR2Model.set_gguf_parameters} (command_r.py:42-45): the inherited keys, the {@code logit_scale}
     * and an explicit no-scaling RoPE.
     */
    public static void commandR2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.LOGIT_SCALE, requireDouble(ctx.config(), "logit_scale"));
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
    }

    /**
     * {@code Cohere2Model.set_gguf_parameters} (command_r.py:47-60): the inherited keys, the {@code logit_scale},
     * the sliding window, the exact {@code vocab_size}, and the RoPE width {@code rotary_pct * hidden_size /
     * num_attention_heads} with no scaling.
     */
    public static void cohere2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        kv.set(GgufKeys.LOGIT_SCALE, requireDouble(cfg, "logit_scale"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        double pct = cfg.rope().global().partialRotaryFactor().orElse(0.5)
                * cfg.requireInt("hidden_size") / heads;
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) pct);
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
    }

    /**
     * {@code Cohere2Model.modify_tensors} (command_r.py:62-72): the runtime reads no biases; a nonzero bias tensor
     * is refused, a zero one is dropped.
     */
    public static void cohere2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".bias")) {
            if (hasNonZeroByte(src)) {
                throw new GgufFormatException("bias tensor '" + hfName + "' is not zero");
            }
            out.drop(hfName, "Cohere2 runs without biases; the zero-value tensor is skipped");
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** True when any stored byte of {@code src} is set, so a zero-valued tensor is cheap to detect. */
    private static boolean hasNonZeroByte(RowSource src) throws IOException {
        int rowBytes = Math.toIntExact(src.rowBytes());
        if (rowBytes == 0 || src.rows() == 0) {
            return false;
        }
        ByteBuffer b = ByteBuffer.allocate(rowBytes);
        for (long r = 0; r < src.rows(); r++) {
            b.clear();
            src.readRows(r, 1, b);
            for (int k = 0; k < rowBytes; k++) {
                if (b.get(k) != 0) {
                    return true;
                }
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ Deci

    /** The per-layer KV-head counts the {@code block_configs} fixup writes (deci.py:43-92). */
    static final String KV_KEY = "nf_deci_num_kv_heads";
    /** The per-layer query-head counts the {@code block_configs} fixup writes. */
    static final String HEAD_KEY = "nf_deci_num_heads";
    /** The per-layer FFN widths the {@code block_configs} fixup writes. */
    static final String FFN_KEY = "nf_deci_ffn_dims";

    /**
     * {@code DeciModel.__init__}'s {@code block_configs} derivation (deci.py:43-107): per layer, zero KV heads and
     * the full head count for a linear layer, zero of both for a layer without attention, otherwise the configured
     * quotient and head count, and the FFN width from the multiplier (the rounded 2/3 rule, deci.py:21-29).
     */
    public static HfConfig deciFixup(HfConfig cfg) {
        Object configs = cfg.raw().raw("block_configs");
        if (!(configs instanceof List) || ((List<?>) configs).isEmpty()) {
            return cfg;
        }
        List<?> blocks = (List<?>) configs;
        int heads = cfg.requireInt("num_attention_heads");
        List<Integer> kv = new ArrayList<>(blocks.size());
        List<Integer> headList = new ArrayList<>(blocks.size());
        List<Integer> ffn = new ArrayList<>(blocks.size());
        for (Object o : blocks) {
            if (!(o instanceof Map)) {
                throw new GgufFormatException(cfg.source() + ": 'block_configs' must hold objects, got " + o);
            }
            Map<?, ?> block = (Map<?, ?>) o;
            Object att = block.get("attention");
            Object f = block.get("ffn");
            if (!(att instanceof Map) || !(f instanceof Map)) {
                throw new GgufFormatException(cfg.source() + ": each 'block_configs' entry needs 'attention' and"
                        + " 'ffn' objects");
            }
            Map<?, ?> attention = (Map<?, ?>) att;
            Map<?, ?> ffnCfg = (Map<?, ?>) f;
            Object group = attention.get("n_heads_in_group");
            int kvHeads;
            int qHeads;
            if (group == null) {
                boolean linear = Boolean.TRUE.equals(attention.get("replace_with_linear"));
                kvHeads = 0;
                qHeads = linear ? heads : 0;
            } else if (group instanceof Number) {
                int perGroup = ((Number) group).intValue();
                require(cfg, perGroup > 0, "'block_configs.attention.n_heads_in_group' must be positive");
                kvHeads = heads / perGroup;
                qHeads = heads;
            } else {
                throw new GgufFormatException(cfg.source()
                        + ": 'block_configs.attention.n_heads_in_group' must be an integer or null");
            }
            Object mult = ffnCfg.get("ffn_mult");
            if (mult == null) {
                ffn.add(0);
            } else {
                require(cfg, mult instanceof Number, "'block_configs.ffn.ffn_mult' must be a number");
                ffn.add(roundTo256(2.0 * ((Number) mult).doubleValue() * cfg.requireInt("hidden_size") / 3.0));
            }
            kv.add(kvHeads);
            headList.add(qHeads);
        }
        Map<String, Object> derived = new LinkedHashMap<>();
        derived.put(KV_KEY, kv);
        derived.put(HEAD_KEY, headList);
        derived.put(FFN_KEY, ffn);
        return cfg.withFlat(derived);
    }

    /** {@code DeciModel._ffn_mult_to_intermediate_size}'s multiple-of-256 rounding (deci.py:25-29). */
    private static int roundTo256(double n) {
        int i = (int) Math.round(n);
        return i % 256 == 0 ? i : i + (256 - i % 256);
    }

    private static void require(HfConfig cfg, boolean ok, String what) {
        if (!ok) {
            throw new GgufFormatException(cfg.source() + ": " + what);
        }
    }

    /**
     * {@code DeciModel.set_gguf_parameters} (deci.py:109-146): with derived per-layer configurations the arrays and
     * per-layer key/value widths are written by hand, without the generic table; either path ends with the
     * {@code vocab_size} and the RoPE width {@code head_dim|hidden_size // num_attention_heads}.
     */
    public static void deciMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        if (cfg.flat().containsKey(KV_KEY)) {
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.flat().get(KV_KEY));
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.flat().get(HEAD_KEY));
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.flat().get(FFN_KEY));
            kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
            kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_position_embeddings"));
            kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH, cfg.requireInt("hidden_size") / heads);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, cfg.requireInt("hidden_size") / heads);
            kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, requireDouble(cfg, "rms_norm_eps"));
            OptionalDouble theta = cfg.rope().global().theta();
            if (theta.isPresent()) {
                kv.set(GgufKeys.ROPE_FREQ_BASE, theta.getAsDouble());
            }
        } else {
            kv.remove(GgufKeys.ATTENTION_HEAD_COUNT);
            kv.remove(GgufKeys.ATTENTION_HEAD_COUNT_KV);
            kv.remove(GgufKeys.FEED_FORWARD_LENGTH);
            HfGenericMetadata.apply(ctx, kv);
            Optional<List<Integer>> perLayer = cfg.findIntList("num_key_value_heads_per_layer");
            if (perLayer.isPresent()) {
                kv.remove(GgufKeys.ATTENTION_HEAD_COUNT_KV);
                kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, perLayer.get());
            }
        }
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, cfg.findInt(cfg.requireInt("hidden_size") / heads, "head_dim"));
    }

    /**
     * {@code DeciModel.modify_tensors} (deci.py:156-174): per block, q uses that layer's head count and k the layer
     * KV-head count through the Llama permutation; unblocked tensors pass on.
     */
    public static void deciTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = HfLoadContext.bidOf(hfName);
        int heads;
        int kvHeads;
        if (bid < 0) {
            heads = cfg.requireInt("num_attention_heads");
            kvHeads = cfg.findInt(heads, "num_key_value_heads");
        } else if (cfg.flat().containsKey(KV_KEY)) {
            heads = layerValue(cfg, HEAD_KEY, bid);
            kvHeads = layerValue(cfg, KV_KEY, bid);
        } else {
            heads = cfg.requireInt("num_attention_heads");
            kvHeads = cfg.findInt(heads, "num_key_value_heads_per_layer", "num_key_value_heads");
        }
        RowSource data = src;
        if (hfName.endsWith("q_proj.weight") || hfName.endsWith("q_proj.bias")) {
            data = RowTransforms.permuteQk(data, heads, heads);
        } else if (hfName.endsWith("k_proj.weight") || hfName.endsWith("k_proj.bias")) {
            data = RowTransforms.permuteQk(data, heads, kvHeads);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    private static int layerValue(HfConfig cfg, String key, int bid) {
        Object v = cfg.flat().get(key);
        if (!(v instanceof List) || ((List<?>) v).size() <= bid
                || !(((List<?>) v).get(bid) instanceof Integer)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' needs an integer per layer");
        }
        return (Integer) ((List<?>) v).get(bid);
    }

    /**
     * {@code DeciModel.generate_extra_tensors} (deci.py:176-184): the llama3 rope frequency factors of a
     * {@code rope_type: llama3} configuration, over the full-attention (or global) parameters.
     */
    public static void deciExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        RopeTensors.llama3Freqs(ctx.config()).ifPresent(freqs -> out.add(ctx.names().ggufName("ROPE_FREQS", -1,
                ".weight"), freqs));
    }

    // ------------------------------------------------------------------ Exaone

    /**
     * {@code ExaoneModel.set_gguf_parameters} (exaone.py:23-31): the inherited keys, then the RoPE width of the
     * partial rotary factor over the head width; the SiLU activation is part of the model.
     */
    public static void exaoneMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        require(cfg, "silu".equals(cfg.findString("activation_function").orElse("")),
                "the EXAONE activation function is silu");
        double partial = cfg.rope().global().partialRotaryFactor().orElse(1.0);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                (int) (partial * (cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"))));
    }

    /** {@code Exaone4Model.set_vocab} (exaone.py:80-85): the GPT-2 vocabulary of the checkpoint's own files. */
    public static Map<String, Object> exaone4Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfDenseClassicHooksA.gpt2Vocab(ctx, pre);
    }

    /**
     * {@code Exaone4Model.set_gguf_parameters} (exaone.py:87-102): the inherited keys, the exact {@code vocab_size},
     * and the boolean sliding-window pattern expanded from {@code layer_types} or the
     * {@code sliding_window_pattern} string or period.
     */
    public static void exaone4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        OptionalInt window = cfg.findInt("sliding_window");
        if (window.isEmpty()) {
            return;
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window.getAsInt());
        writeSwaPattern(ctx, kv, cfg);
    }

    /** The sliding-window pattern of exaone.py:95-101: layer_types first, then the extendable string or period. */
    private static void writeSwaPattern(HfLoadContext ctx, HfMetadataBuilder kv, HfConfig cfg) {
        Object layerTypes = cfg.flat().get("layer_types");
        int layers = cfg.requireInt("num_hidden_layers");
        List<Boolean> pattern = new ArrayList<>();
        if (layerTypes instanceof List && !((List<?>) layerTypes).isEmpty()
                && ((List<?>) layerTypes).size() == layers) {
            for (Object t : (List<?>) layerTypes) {
                pattern.add("sliding_attention".equals(String.valueOf(t)));
            }
        } else if (cfg.findString("sliding_window_pattern").isPresent()) {
            String s = cfg.findString("sliding_window_pattern").orElse("");
            require(cfg, !s.isEmpty(), "'sliding_window_pattern' must not be empty");
            for (int i = 0; i < layers; i++) {
                pattern.add(s.charAt(i % s.length()) == 'L');
            }
        } else {
            int period = cfg.findInt(0, "sliding_window_pattern");
            require(cfg, period > 0, "'sliding_window_pattern' must be a positive period");
            for (int i = 0; i < layers; i++) {
                pattern.add((i + 1) % period != 0);
            }
        }
        require(cfg, pattern.size() == layers, "'sliding_window_pattern' does not cover every layer");
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, Collections.unmodifiableList(pattern));
    }

    /**
     * {@code Exaone4Model.generate_extra_tensors} (exaone.py:104-120): the llama3 rope frequency factors, with the
     * EXAONE 4 default factor of 16.
     */
    public static void exaone4Extra(HfLoadContext ctx, TensorSink out) throws IOException {
        HfRope.Params p = ctx.config().rope().global();
        if (!"llama3".equals(p.type().orElse("").toLowerCase(Locale.ROOT))) {
            return;
        }
        HfConfig cfg = ctx.config();
        int dim = cfg.findInt(cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"), "head_dim");
        out.add(ctx.names().ggufName("ROPE_FREQS", -1, ".weight"), RopeTensors.llama3Freqs(
                p.theta().orElse(10000.0), dim, 0, p.factor().orElse(16.0), p.lowFreqFactor().orElse(1.0),
                p.highFreqFactor().orElse(4.0), p.originalMaxPositionEmbeddings().orElse(8192), false));
    }

    /** The block count of {@code Exaone4_5_TextModel.__init__} (exaone.py:225-230): nextn layers appended. */
    public static int exaone45Blocks(HfConfig cfg) {
        int nNextn = cfg.findInt(0, "num_nextn_predict_layers");
        return cfg.blockCount() + Math.max(0, nNextn);
    }

    /**
     * {@code Exaone4_5_TextModel.set_gguf_parameters} (exaone.py:232-236): the Exaone4 keys and
     * {@code nextn_predict_layers} when the checkpoint carries the MTP head.
     */
    public static void exaone45Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        exaone4Meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int nNextn = ctx.options().includeMtp() ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (nNextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nNextn);
        }
    }

    /**
     * The NextN-block renames of {@code Exaone4_5_TextModel.modify_tensors} (exaone.py:238-262): the
     * {@code mtp.layers.N.} tensors land in their appended trunk slot ({@code mtp_share_layers} replicates them into
     * every NextN block), the sidecar {@code mtp.fc}/norm tensors in the shared slots of every NextN block; tensors
     * outside {@code mtp.} pass on.
     */
    public static void exaone45Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (!hfName.startsWith("mtp.")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        HfConfig cfg = ctx.config();
        int nNextn = cfg.findInt(0, "num_nextn_predict_layers");
        if (nNextn <= 0) {
            return;
        }
        int nh = cfg.requireInt("num_hidden_layers");
        if (hfName.contains(".layers.")) {
            int bid = HfLoadContext.bidOf(hfName);
            if (bid < 0) {
                throw new GgufFormatException(cfg.source() + ": MTP tensor without a layer index: " + hfName);
            }
            if (cfg.findBool(false, "mtp_share_layers")) {
                for (int k = 0; k < nNextn; k++) {
                    applyLayer(ctx, hfName, nh + k, src, out);
                }
            } else {
                applyLayer(ctx, hfName, nh + bid, src, out);
            }
            return;
        }
        String tail = ".weight";
        String base = hfName.endsWith(tail) ? hfName.substring(0, hfName.length() - tail.length())
                : hfName;
        if (base.startsWith("mtp.")) {
            base = base.substring("mtp.".length());
        }
        String tensor;
        switch (base) {
            case "fc":
                tensor = "NEXTN_EH_PROJ";
                break;
            case "pre_fc_norm_embedding":
                tensor = "NEXTN_ENORM";
                break;
            case "pre_fc_norm_hidden":
                tensor = "NEXTN_HNORM";
                break;
            case "norm":
                tensor = "NEXTN_SHARED_HEAD_NORM";
                break;
            default:
                return;
        }
        for (int k = 0; k < nNextn; k++) {
            out.add(ctx.names().ggufName(tensor, nh + k, tail), src);
        }
    }

    /** {@link HfTensorHook#DEFAULT} over {@code hfName}, its {@code mtp.layers.N} prefix moved to layer {@code il}. */
    private static void applyLayer(HfLoadContext ctx, String hfName, int il, RowSource src, TensorSink out)
            throws IOException {
        String renamed = hfName.replaceFirst("mtp\\.layers\\.\\d+", "model.layers." + il);
        HfTensorHook.DEFAULT.apply(ctx, renamed, src, out);
    }
}

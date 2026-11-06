package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The hooks of the openPangu MoE text models (vLLM {@code vllm/model_executor/models/openpangu.py}
 * {@code PanguProMoEV2ForCausalLM} and {@code PanguUltraMoEForCausalLM}). The trunk picks its attention from the
 * configuration: when {@code qk_nope_head_dim}, {@code qk_rope_head_dim}, {@code v_head_dim} and
 * {@code kv_lora_rank} are present it is the DeepSeek-V2 MLA layout, otherwise the static sink attention when
 * {@code param_sink_number > 0}, otherwise the pangu-embedded attention with a dense lead and routed plus shared
 * experts ({@link Arch#PANGU_MOE}). A sandwich norm and an interleaved sliding window are carried into the graph: the
 * decoder's three norms map to {@code attn_norm}, the post-attention norm and the pre-FFN norm, and the per-layer
 * window list becomes the per-layer sliding window (a named window size per layer, 0 for full attention).
 *
 * <p>Routing is sigmoid in one group with top-k renormalization from the file; the class writes the defaults the
 * vLLM implementation hard-codes when the checkpoint omits them.</p>
 */
public final class HfPanguMoeHooks {

    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d+)\\..*$");

    private HfPanguMoeHooks() {
    }

    /**
     * {@code OpenPanguDecoderLayer.__init__}: MLA when the configuration carries the MLA geometry
     * ({@code OpenPanguMLAAttention} takes precedence), otherwise the static sink attention when
     * {@code param_sink_number > 0}, otherwise the pangu-embedded attention. Both non-MLA variants run over the
     * {@link Arch#PANGU_MOE} graph.
     *
     * @throws GgufFormatException naming an unsupported sink configuration
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        boolean mla = isMla(cfg);
        if (!mla) {
            sinkTokens(cfg);
            slidingWindows(cfg);
        }
        return mla ? Arch.DEEPSEEK2 : Arch.PANGU_MOE;
    }

    /**
     * {@code OpenPanguSinkAttention}'s {@code param_sink_number}: the number of learned sink tokens of the
     * non-MLA trunk, 0 when the checkpoint selects the pangu-embedded attention. The sink variant additionally
     * needs the split key/value geometry of vLLM's {@code OpenPanguSinkAttention} ({@code qk_nope_dim},
     * {@code qk_rope_dim}, {@code v_channels}), the K norm it normalizes with, and causal attention.
     *
     * @throws GgufFormatException naming a missing or inconsistent sink geometry
     */
    private static int sinkTokens(HfConfig cfg) {
        int sink = cfg.findInt(0, "param_sink_number");
        if (sink < 0) {
            throw new GgufFormatException(cfg.source() + ": 'param_sink_number' must be >= 0, got " + sink);
        }
        if (sink == 0) {
            return 0;
        }
        for (String key : new String[]{"qk_nope_dim", "qk_rope_dim", "v_channels"}) {
            if (cfg.findInt(key).isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": 'param_sink_number' = " + sink
                        + " selects openPangu's static sink attention, which needs '" + key + "'");
            }
        }
        if (cfg.findInt("qk_nope_dim").getAsInt() <= 0 || cfg.findInt("qk_rope_dim").getAsInt() <= 0
                || cfg.findInt("v_channels").getAsInt() <= 0) {
            throw new GgufFormatException(cfg.source() + ": the sink attention needs positive 'qk_nope_dim',"
                    + " 'qk_rope_dim' and 'v_channels'");
        }
        if (!cfg.findBool(true, "is_causal")) {
            throw new GgufFormatException(cfg.source() + ": 'is_causal' = false is not supported with the openPangu"
                    + " sink attention");
        }
        return sink;
    }

    private static boolean isMla(HfConfig cfg) {
        return cfg.find("qk_nope_head_dim").isPresent() && cfg.find("qk_rope_head_dim").isPresent()
                && cfg.find("v_head_dim").isPresent() && cfg.find("kv_lora_rank").isPresent();
    }

    /**
     * The per-layer sliding window of {@code interleaved_sliding_window} (an integer broadcast to every layer, or a
     * list indexed by {@code layer % length}; a non-positive or null entry is a full-attention layer), or null when
     * the key is absent. vLLM {@code OpenPanguDecoderLayer.__init__} selects
     * {@code interleaved_sliding_window[layer_idx % len]} as the layer's {@code per_layer_sliding_window}.
     *
     * @throws GgufFormatException when the value has another type or an entry is not an integer
     */
    private static int[] slidingWindows(HfConfig cfg) {
        Object raw = cfg.find("interleaved_sliding_window").orElse(null);
        if (raw == null) {
            return null;
        }
        int layers = cfg.requireInt("num_hidden_layers");
        int[] windows = new int[layers];
        if (raw instanceof Number) {
            int w = ((Number) raw).intValue();
            Arrays.fill(windows, Math.max(0, w));
            return windows;
        }
        if (!(raw instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'interleaved_sliding_window' must be an integer or a"
                    + " list of integers");
        }
        List<?> list = (List<?>) raw;
        if (list.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'interleaved_sliding_window' is an empty list");
        }
        for (int i = 0; i < layers; i++) {
            Object e = list.get(i % list.size());
            if (e == null) {
                windows[i] = 0;
            } else if (e instanceof Number) {
                windows[i] = Math.max(0, ((Number) e).intValue());
            } else {
                throw new GgufFormatException(cfg.source() + ": 'interleaved_sliding_window' entry " + i + " is '"
                        + e + "'; every entry must be an integer");
            }
        }
        return windows;
    }

    /** Distinct positive window sizes of {@code windows}; 0 or 1 when every windowed layer shares its size. */
    private static int distinctPositive(int[] windows) {
        int distinct = 0;
        int first = -1;
        for (int w : windows) {
            if (w > 0 && first < 0) {
                first = w;
                distinct = 1;
            } else if (w > 0 && w != first) {
                distinct++;
            }
        }
        return distinct;
    }

    /**
     * The routing defaults the vLLM class hard-codes: one expert group, sigmoid scoring, a top-k renormalization
     * when the file names it, and the {@code num_key_value_heads} the MLA {@code kv_b_proj} split needs.
     */
    public static HfConfig fixup(HfConfig cfg) {
        Map<String, Object> derived = new LinkedHashMap<>();
        if (!cfg.find("n_group").isPresent()) {
            derived.put("n_group", 1L);
        }
        if (!cfg.find("topk_group").isPresent()) {
            derived.put("topk_group", 1L);
        }
        if (!cfg.find("score_function", "scoring_func", "score_func", "moe_router_activation").isPresent()) {
            derived.put("scoring_func", "sigmoid");
        }
        if (!cfg.find("n_routed_experts").isPresent() && cfg.find("num_experts").isPresent()) {
            derived.put("n_routed_experts", cfg.find("num_experts").get());
        }
        if (!cfg.find("num_experts_per_tok").isPresent() && cfg.find("moe_top_k").isPresent()) {
            derived.put("num_experts_per_tok", cfg.find("moe_top_k").get());
        }
        if (isMla(cfg) && !cfg.find("num_key_value_heads").isPresent()) {
            derived.put("num_key_value_heads", (long) cfg.requireInt("num_attention_heads"));
        }
        HfConfig fixed = cfg.withFlat(derived);
        validate(fixed);
        return fixed;
    }

    private static void validate(HfConfig cfg) {
        if (!cfg.find("n_routed_experts").isPresent()) {
            throw new GgufFormatException(cfg.source() + ": 'n_routed_experts' is required: the openPangu MoE"
                    + " models have routed experts");
        }
        OptionalInt shared = cfg.findInt("n_shared_experts");
        if (!shared.isPresent() || shared.getAsInt() <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'n_shared_experts' must be positive: the openPangu"
                    + " MoE layout requires the shared-expert branch");
        }
        if (!cfg.find("first_k_dense_replace").isPresent()) {
            throw new GgufFormatException(cfg.source() + ": 'first_k_dense_replace' is required: it places the"
                    + " dense lead of the MoE stack");
        }
        if (!cfg.find("moe_intermediate_size").isPresent()) {
            throw new GgufFormatException(cfg.source() + ": 'moe_intermediate_size' is required");
        }
    }

    /**
     * The DeepSeek-V2 MLA metadata for the MLA-trunk configuration; the pangu-embedded keys plus the MoE keys for
     * the plain trunk.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        if (ctx.arch() == Arch.DEEPSEEK2) {
            HfDeepseekHooks.meta(ctx, kv, plan);
            return;
        }
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int sinks = sinkTokens(cfg);
        if (sinks > 0) {
            int nope = cfg.requireInt("qk_nope_dim");
            int rope = cfg.requireInt("qk_rope_dim");
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rope);
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH, nope + rope);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, cfg.requireInt("v_channels"));
            kv.set(GgufKeys.ROPE_TYPE, "norm");
            kv.set(GgufKeys.ATTENTION_SINK_TOKEN_COUNT, sinks);
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV,
                    cfg.findInt(cfg.requireInt("num_attention_heads"), "num_key_value_heads"));
        } else {
            int ropeDim = cfg.findInt("head_dim")
                    .orElse(cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"));
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim);
            if (!cfg.findInt("head_dim").isPresent()) {
                kv.set(GgufKeys.ATTENTION_KEY_LENGTH, ropeDim);
                kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, ropeDim);
            }
        }
        int layers = cfg.requireInt("num_hidden_layers");
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.findInt(layers, "first_k_dense_replace"));
        int expertFf = cfg.requireInt("moe_intermediate_size");
        int shared = cfg.requireInt("n_shared_experts");
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, expertFf);
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, shared);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, expertFf * shared);
        OptionalDouble scale = cfg.findFloat("routed_scaling_factor");
        if (scale.isPresent()) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, scale.getAsDouble());
        }
        if (cfg.findBool(false, "norm_topk_prob")) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, true);
        }
        int[] windows = slidingWindows(cfg);
        if (windows != null) {
            int window = 0;
            List<Boolean> pattern = new ArrayList<>(windows.length);
            List<Integer> perLayer = new ArrayList<>(windows.length);
            for (int w : windows) {
                window = Math.max(window, w);
                pattern.add(w > 0);
                perLayer.add(w);
            }
            if (window > 0) {
                kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window);
                kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
                if (distinctPositive(windows) > 1) {
                    kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PER_LAYER, perLayer);
                }
            }
        }
    }

    /**
     * {@code OpenPanguModel.load_weights}: a sandwich norm's {@code post_attention_layernorm},
     * {@code pre_mlp_layernorm} and {@code post_mlp_layernorm} become the post-attention norm, the FFN input norm and
     * the post-FFN norm of the graph; every other tensor follows the DeepSeek converter (stacked experts, tied head).
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (!out.isConsumed(hfName) && ctx.arch() != Arch.DEEPSEEK2) {
            Matcher m = LAYER.matcher(hfName);
            if (m.matches()) {
                int bid = Integer.parseInt(m.group(1));
                if (hfName.endsWith(".self_attn.k_layernorm.weight")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K_NORM)), src);
                    return;
                }
                if (hfName.endsWith(".self_attn.param_sink_key")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_SINK_K)), src);
                    return;
                }
                if (hfName.endsWith(".self_attn.param_sink_value")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_SINK_V)), src);
                    return;
                }
            }
        }
        if (!out.isConsumed(hfName) && ctx.config().findBool(false, "sandwich_norm")) {
            Matcher m = LAYER.matcher(hfName);
            if (m.matches()) {
                int bid = Integer.parseInt(m.group(1));
                if (hfName.endsWith(".post_attention_layernorm.weight")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_POST_NORM)), src);
                    return;
                }
                if (hfName.endsWith(".pre_mlp_layernorm.weight")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_NORM)), src);
                    return;
                }
                if (hfName.endsWith(".post_mlp_layernorm.weight")) {
                    out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_POST_NORM)), src);
                    return;
                }
            }
        }
        HfDeepseekHooks.tensors(ctx, hfName, src, out);
    }

    /**
     * The checkpoint's tokenizer: the SentencePiece vocabulary with the model's {@code add_prefix_space} when the
     * checkpoint ships {@code tokenizer.model}, else the GPT-2 BPE vocabulary the other vLLM text models use.
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.sentencePieceModel() != null) {
            return HfDenseClassicHooksE.panguVocab(ctx, pre);
        }
        return HfDeepseekHooks.vocab(ctx, pre);
    }
}

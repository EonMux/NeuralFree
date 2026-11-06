package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The EAGLE-3 draft layouts vLLM serves with {@code qwen3_eagle3.Eagle3Qwen3ForCausalLM}
 * (V/model_executor/models/registry.py:715-717): {@code Eagle3Qwen3ForCausalLM} and its parallel-drafting alias
 * {@code PeagleQwen3ForCausalLM}. The decoder is the {@code llama_eagle3} EAGLE-3 layer of the {@link Arch#EAGLE3}
 * row — Qwen3's {@code input_layernorm}, {@code hidden_norm} (the draft's {@code attn_norm_2}), gated FFN and final
 * norm are the names the GGUF layout already carries — with the differences the Qwen3 configuration adds:
 * <ul>
 *   <li>the per-head RMS norms {@code self_attn.q_norm} / {@code self_attn.k_norm} of Qwen3 attention
 *       ({@code blk.0.attn_q_norm} / {@code blk.0.attn_k_norm});</li>
 *   <li>the optional Q/K/V projection biases of {@code attention_bias} checkpoints (the parallel-drafting
 *       checkpoints set it);</li>
 *   <li>the optional per-tap encoder norms {@code fc_norm.<part>.weight} of checkpoints whose configuration sets
 *       {@code fc_norm} ({@code llama_eagle3.py:231-240}), applied by the {@code Eagle3DraftModel} encoder;</li>
 *   <li>the draft {@code logit_scale}, written as the GGUF key the {@code Eagle3DraftModel} applies to its logits;</li>
 *   <li>{@code draft_vocab_size}, defaulted to {@code vocab_size} ({@code qwen3_eagle3.py:291-295}) so the draft
 *       vocabulary and the {@code d2t} map agree.</li>
 * </ul>
 * The metadata and vocabulary are {@link HfEagle3Hooks}'s: the target taps are the draft configuration's
 * {@code eagle_aux_hidden_state_layer_ids} (the root's, else {@code eagle_config}'s) or, when it names none, vLLM's
 * three-layer default {@code [2, n // 2, n - 3]} of the target ({@code get_eagle3_default_aux_hidden_state_layers},
 * interfaces.py:1766-1787), which is llama.cpp's default too.
 */
public final class HfEagle3Qwen3Hooks {

    /** The Qwen3 projection norm tensors of one decoder layer. */
    private static final Pattern LAYER_QK_NORM = Pattern.compile(
            "^model\\.layers\\.(\\d{1,9})\\.self_attn\\.([qk])_norm\\.weight$");
    private static final int EAGLE3_TARGET_LAYERS = 3;

    private HfEagle3Qwen3Hooks() {
    }

    // ------------------------------------------------------------------ configuration

    /**
     * The configuration fixes of the Qwen3 EAGLE-3 layout: the {@code transformer_layer_config} flattening of
     * {@link HfEagle3Hooks#fixup}, a missing {@code num_hidden_layers} as the single decoder layer, and
     * {@code draft_vocab_size} as {@code vocab_size}. The variants
     * {@link com.eonmux.neuralfree.gguf.ops.flow.Eagle3DraftModel} does not run are refused by name:
     * {@code use_aux_hidden_state: false} (no fusion projection), {@code norm_output} (the next step reads the
     * post-norm state), a {@code sliding_attention} first layer and a target-tap or aux hidden-state count other
     * than the three the GGUF row fuses.
     *
     * @throws GgufFormatException when a value has the wrong type or names an unsupported variant
     */
    public static HfConfig fixup(HfConfig cfg) {
        HfConfig out = HfEagle3Hooks.fixup(cfg);
        Map<String, Object> extra = new LinkedHashMap<>();
        if (!out.flat().containsKey("num_hidden_layers")) {
            extra.put("num_hidden_layers", 1);
        }
        if (!out.flat().containsKey("draft_vocab_size")) {
            extra.put("draft_vocab_size", out.requireInt("vocab_size"));
        }
        if (!extra.isEmpty()) {
            out = out.withFlat(extra);
        }
        refuseVariant(out, "use_aux_hidden_state", true);
        refuseVariant(out, "norm_output", false);
        refuseSlidingLayer(out);
        validateTaps(out);
        return out;
    }

    /**
     * A draft whose single decoder layer is a sliding-window layer ({@code qwen3_eagle3.py:52-61}) runs a bounded
     * attention window the GGUF layout has no keys for; the checkpoint is refused by its {@code layer_types}.
     *
     * @throws GgufFormatException when the first {@code layer_types} entry is {@code sliding_attention}
     */
    private static void refuseSlidingLayer(HfConfig cfg) {
        Object types = cfg.flat().get("layer_types");
        if (!(types instanceof List) || ((List<?>) types).isEmpty()) {
            return;
        }
        if ("sliding_attention".equals(((List<?>) types).get(0))) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " declares a"
                    + " 'sliding_attention' first layer, which the EAGLE-3 drafter does not run");
        }
    }

    /** Refuses {@code key} when the configuration (or its nested {@code eagle_config}) contradicts it. */
    private static void refuseVariant(HfConfig cfg, String key, boolean expected) {
        JsonView raw = cfg.raw();
        Object v = raw.raw(key);
        if (v == null) {
            v = raw.optObj("eagle_config").raw(key);
        }
        if (v == null) {
            return;
        }
        if (!(v instanceof Boolean)) {
            throw new GgufFormatException(raw.where(key) + " must be a boolean");
        }
        if (((Boolean) v) != expected) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " with '" + key + "': "
                    + v + " is not supported by the EAGLE-3 drafter");
        }
    }

    /**
     * The EAGLE-3 fusion holds exactly three target taps (eagle3.cpp:9-11). The taps are the declared
     * {@code eagle_aux_hidden_state_layer_ids}; a draft that declares none gets vLLM's three-layer default
     * ({@link HfEagle3Hooks#meta}). A declared aux hidden-state count ({@code num_aux_layers}, else
     * {@code num_aux_hidden_states}) must be the number of taps, and that number must be three: another count builds
     * a fusion projection the GGUF row cannot describe.
     *
     * @throws GgufFormatException when the declared taps or aux count are not three
     */
    private static void validateTaps(HfConfig cfg) {
        List<Integer> ids = HfEagle3Hooks.declaredAuxLayers(cfg);
        int taps = ids.isEmpty() ? EAGLE3_TARGET_LAYERS : ids.size();
        if (taps != EAGLE3_TARGET_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " fuses " + taps
                    + " target layers (" + ids + "); the EAGLE-3 drafter fuses exactly " + EAGLE3_TARGET_LAYERS);
        }
        for (String key : new String[]{"num_aux_layers", "num_aux_hidden_states"}) {
            OptionalInt n = cfg.findInt(key);
            if (n.isPresent()) {
                // vLLM reads num_aux_layers first and num_aux_hidden_states only when it is absent
                if (n.getAsInt() != taps) {
                    throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " declares '" + key
                            + "' = " + n.getAsInt() + "; the EAGLE-3 drafter fuses " + taps + " target layers");
                }
                return;
            }
        }
    }

    /** {@code filter_tensors}: the checkpoint is the draft directory alone, as for every EAGLE-3 draft. */
    public static HfNameFilter filter(HfLoadContext ctx) {
        return HfNameFilter.INHERITED;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The standard Llama keys ({@link HfLlamaHooks#meta}) followed by the EAGLE-3 target keys
     * ({@link HfEagle3Hooks#meta}, which also checks the {@code fc_norm} flag against the planned per-tap norms) and
     * the optional {@code logit_scale} the draft configuration declares.
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when a value has the wrong type, the target lacks what the keys need, or the
     *                             {@code fc_norm} flag and the planned per-tap norms disagree
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        HfEagle3Hooks.meta(ctx, kv, plan);
        Object scale = cfg.raw().raw("logit_scale");
        if (scale != null) {
            if (!(scale instanceof Number) || scale instanceof Boolean) {
                throw new GgufFormatException(cfg.raw().where("logit_scale") + " must be a number");
            }
            kv.set(GgufKeys.LOGIT_SCALE, ((Number) scale).floatValue());
        }
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The tensors of the Qwen3 EAGLE-3 layout: the Qwen3 projection norms first, then the EAGLE-3 renames and
     * special tensors of {@link HfEagle3Hooks#tensor} (including the {@code fc_norm} parts), then the Llama mapping
     * of {@link HfLlamaHooks#llamaTensor} (which also un-permutes the Q/K projections and their biases).
     *
     * @throws IOException         when {@code d2t} or the target cannot be read
     * @throws GgufFormatException when a tensor has no mapping or {@code d2t} is invalid
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = HfEagle3Hooks.rename(hfName);
        if (qwenTensor(ctx, hfName, name, src, out)) {
            return;
        }
        if (HfEagle3Hooks.tensor(ctx, hfName, name, src, out)) {
            return;
        }
        HfLlamaHooks.llamaTensor(ctx, hfName, name, src, out);
    }

    /** The Qwen3-only tensors; false for any other name. */
    private static boolean qwenTensor(HfLoadContext ctx, String hfName, String name, RowSource src, TensorSink out)
            throws IOException {
        Matcher qk = LAYER_QK_NORM.matcher(name);
        if (qk.matches()) {
            int bid = Integer.parseInt(qk.group(1));
            String tensor = "q".equals(qk.group(2)) ? TensorNames.ATTN_Q_NORM : TensorNames.ATTN_K_NORM;
            out.add(TensorNames.layerName(bid, TensorNames.weight(tensor)), src);
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * The target's tokenizer ({@link HfEagle3Hooks#vocab}): the EAGLE-3 draft shares its target's vocabulary, and
     * its {@code d2t} map (when the head is reduced) is the offset table over it.
     *
     * @throws IOException         when the target or its tokenizer files cannot be read
     * @throws GgufFormatException when a GGUF target holds no tokenizer
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfEagle3Hooks.vocab(ctx, pre);
    }
}

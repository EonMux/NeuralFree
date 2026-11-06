package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The EAGLE-3 draft layouts vLLM serves with {@code deepseek_eagle3.Eagle3DeepseekV2ForCausalLM}
 * (V/model_executor/models/registry.py:722-723): {@code Eagle3DeepseekV2ForCausalLM} and its V3 alias
 * {@code Eagle3DeepseekV3ForCausalLM}. The draft keeps the {@code llama_eagle3} decoder form — the fusion
 * projection over three target taps, the separate {@code input_layernorm} / {@code hidden_norm} on the token
 * embedding and target hidden state, the concatenated attention input, the dense MLP and the output norm — but its
 * single decoder layer is DeepSeek's multi-head latent attention over the {@code DeepseekV2MLAAttention} weights:
 *
 * <ul>
 *   <li>the low-rank query path {@code q_a_proj -> q_a_layernorm -> q_b_proj}, or a plain {@code q_proj} when the
 *       configuration has no {@code q_lora_rank};</li>
 *   <li>the latent KV path {@code kv_a_proj_with_mqa -> kv_a_layernorm}, whose {@code kv_b_proj} the converter
 *       splits into the absorbed per-head {@code attn_k_b} / {@code attn_v_b} ({@code deepseek.py:434-450});</li>
 *   <li>the DeepSeek MLA metadata: {@code kv_lora_rank}, the decompressed {@code qk_nope_head_dim + qk_rope_head_dim}
 *       key and {@code v_head_dim} value widths, {@code rope.dimension_count = qk_rope_head_dim} and the YaRN log
 *       multiplier {@code 0.1 * mscale_all_dim}.</li>
 * </ul>
 *
 * <p>The draft input is the concatenation of the normed embedding and the normed target hidden state, so every
 * projection takes {@code 2 * hidden_size} columns. The decoder runs on the {@link Arch#EAGLE3} GGUF row; its
 * {@code fc_norm} layout, target taps and target-shared vocabulary are {@link HfEagle3Hooks}'s. The variants the
 * engine does not run are refused by name: {@code norm_output} (the next step reads the post-norm state),
 * {@code use_aux_hidden_state: false} (no fusion projection) and a declared target-tap or aux count other than
 * three.
 */
public final class HfEagle3MlaHooks {

    /** The DeepSeek EAGLE-3 draft names vLLM's {@code deepseek_eagle3} module serves. */
    public static final List<String> NAMES = Collections.unmodifiableList(Arrays.asList(
            "Eagle3DeepseekV2ForCausalLM", "Eagle3DeepseekV3ForCausalLM"));

    /** The MLA projection tensors of one draft decoder layer. */
    private static final Pattern MLA_PROJECTION = Pattern.compile(
            "^model\\.layers\\.(\\d{1,9})\\.self_attn\\.(q_a_proj|q_a_layernorm|q_b_proj|kv_a_proj_with_mqa"
                    + "|kv_a_layernorm|q_proj|o_proj)\\.(weight|bias)$");

    /** The GGUF base each MLA projection maps to. */
    private static final Map<String, String> MLA_TENSORS = mlaTensors();

    /** EAGLE-3 fuses exactly three target layers (eagle3.cpp:9-11). */
    private static final int EAGLE3_TARGET_LAYERS = 3;
    private static final int DEFAULT_KV_LORA_RANK = 512;

    private HfEagle3MlaHooks() {
    }

    private static Map<String, String> mlaTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("q_a_proj", TensorNames.ATTN_Q_A);
        m.put("q_a_layernorm", TensorNames.ATTN_Q_A_NORM);
        m.put("q_b_proj", TensorNames.ATTN_Q_B);
        m.put("kv_a_proj_with_mqa", TensorNames.ATTN_KV_A_MQA);
        m.put("kv_a_layernorm", TensorNames.ATTN_KV_A_NORM);
        m.put("q_proj", TensorNames.ATTN_Q);
        m.put("o_proj", TensorNames.ATTN_OUT);
        return Collections.unmodifiableMap(m);
    }

    // ------------------------------------------------------------------ configuration

    /**
     * The configuration fixes of the DeepSeek EAGLE-3 layout: the {@code transformer_layer_config} flattening of
     * {@link HfEagle3Hooks#fixup}, a missing {@code num_hidden_layers} as the single decoder layer, and
     * {@code draft_vocab_size} as {@code vocab_size} (vLLM's {@code Eagle3DeepseekV2ForCausalLM.__init__}).
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
        refuseVariant(out, "norm_output", false);
        refuseVariant(out, "use_aux_hidden_state", true);
        validateTaps(out);
        return out;
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

    /** The fusion needs exactly three target taps and no other declared aux count. */
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
                if (n.getAsInt() != taps) {
                    throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " declares '" + key
                            + "' = " + n.getAsInt() + "; the EAGLE-3 drafter fuses " + taps + " target layers");
                }
                return;
            }
        }
    }

    /** {@code filter_tensors}: the checkpoint is the draft directory alone. */
    public static HfNameFilter filter(HfLoadContext ctx) {
        return HfNameFilter.INHERITED;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The generic keys of {@link HfGenericMetadata}, the DeepSeek MLA keys, and the EAGLE-3 target keys of
     * {@link HfEagle3Hooks#meta}. The MLA keys are written after the generic pass so the head geometry describes
     * the latent attention rather than the {@code hidden_size / num_attention_heads} default.
     *
     * @throws IOException         when the target cannot be opened
     * @throws GgufFormatException when a value has the wrong type, an MLA dimension is missing, the target lacks
     *                             what the keys need, or the {@code fc_norm} flag and the planned norms disagree
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        int kvLora = cfg.findInt(DEFAULT_KV_LORA_RANK, "kv_lora_rank");
        int qkRope = cfg.requireInt("qk_rope_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, qkNope + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, vHead);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, qkRope);
        int qLora = cfg.findInt(0, "q_lora_rank");
        if (qLora > 0) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora);
        }
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        Object mscaleAllDim = cfg.rope().raw().get("mscale_all_dim");
        if (mscaleAllDim != null) {
            if (!(mscaleAllDim instanceof Number) || mscaleAllDim instanceof Boolean) {
                throw new GgufFormatException(cfg.source() + ": 'rope_parameters.mscale_all_dim' must be a number");
            }
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1 * ((Number) mscaleAllDim).doubleValue());
        }
        HfEagle3Hooks.meta(ctx, kv, plan);
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The tensors of the DeepSeek EAGLE-3 layout: the EAGLE-3 renames and special tensors of
     * {@link HfEagle3Hooks#tensor} ({@code fc}, {@code hidden_norm}, {@code fc_norm}, {@code d2t}), the MLA
     * projections with {@code kv_b_proj} split into {@code attn_k_b} / {@code attn_v_b}, and the Llama mapping of
     * the remaining layer tensors.
     *
     * @throws IOException         when {@code d2t} or the target cannot be read
     * @throws GgufFormatException when a tensor has no mapping or {@code kv_b_proj} does not have the MLA shape
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = rename(hfName);
        if (HfEagle3Hooks.tensor(ctx, hfName, name, src, out)) {
            return;
        }
        if (name.endsWith("kv_b_proj.weight")) {
            splitKvB(ctx, name, src, out);
            return;
        }
        Matcher m = MLA_PROJECTION.matcher(name);
        if (m.matches()) {
            int bid = Integer.parseInt(m.group(1));
            String base = MLA_TENSORS.get(m.group(2));
            out.add(TensorNames.layerName(bid, base + "." + m.group(3)), src);
            return;
        }
        HfLlamaHooks.llamaTensor(ctx, hfName, name, src, out);
    }

    /**
     * The EAGLE-3 {@code midlayer.} / {@code layers.0.} renames followed by the {@code model.} prefix the DeepSeek
     * draft's loader adds to every weight that is not the LM head (deepseek_eagle3.py:381-383).
     */
    static String rename(String hfName) {
        String name = HfEagle3Hooks.rename(hfName);
        if (name.startsWith("model.") || name.equals("lm_head.weight") || name.equals("lm_head.bias")
                || name.startsWith("fc.") || name.startsWith("fc_norm.") || name.equals("input_norm.weight")
                || name.equals("d2t") || name.equals("t2d") || name.equals("mask_hidden")) {
            return name;
        }
        return "model." + name;
    }

    /**
     * {@code DeepseekV2Model.modify_tensors}' {@code kv_b_proj} split (deepseek.py:434-450): the checkpoint's
     * {@code [n_head * (qk_nope + v_head), kv_lora]} tensor to the absorbed per-head {@code attn_k_b} and
     * {@code attn_v_b} the MLA mixer reads.
     *
     * @throws GgufFormatException when the tensor is not rank 2 or does not split into the configured heads
     */
    private static void splitKvB(HfLoadContext ctx, String name, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + name + "' has shape " + Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int bid = HfHookSupport.layerOf(name);
        int nHead = cfg.requireInt("num_attention_heads");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + name + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K_B)), kvB[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V_B)), kvB[1]);
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * The target's tokenizer ({@link HfEagle3Hooks#vocab}): the DeepSeek EAGLE-3 draft shares its target's
     * vocabulary and carries a {@code d2t} map only when its LM head is reduced.
     *
     * @throws IOException         when the target or its tokenizer files cannot be read
     * @throws GgufFormatException when a GGUF target holds no tokenizer
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfEagle3Hooks.vocab(ctx, pre);
    }
}

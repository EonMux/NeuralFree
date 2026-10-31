package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Hooks of the SSM hybrid converter classes of the Granite and Nemotron-H families: {@code GraniteHybridModel}
 * (LC/conversion/granite.py:436-576), {@code NemotronHModel} (nemotron.py:405-731) and
 * {@code NemotronHPuzzleModel} (nemotron.py:733-928). Both Nemotron classes build on the Granite hybrid, whose
 * attention layers route through the Granite MoE path of {@link HfGraniteHooks} and whose Mamba layers route
 * through {@link HfSsmHooks#mamba2Tensor}.
 *
 * <p>Every method is a public static hook reference; the Granite hybrid composes the read-only Granite hooks for
 * its attention layers, mirroring the class's two-parents converter.</p>
 */
public final class HfSsmHybridHooksA {

    /**
     * {@code GraniteHybridModel}: no class filter of its own; the Mamba 2 base's filter
     * ({@link HfSsmHooks#mamba2Filter}) applies before the Granite multimodal drop, the MRO order of the bases
     * Mamba2Model and GraniteMoeModel, with the plain Llama base last.
     */
    public static final HfNameFilter GRANITE_HYBRID_FILTER =
            (name, base) -> HfSsmHooks.mamba2Filter(name, s -> HfGraniteHooks.filter(s, base));

    /**
     * {@code NemotronHModel.filter_tensors} (nemotron.py:501-515): the {@code language_model.} prefix strips, the
     * {@code mtp.} head folds in (the {@code --mtp}/{@code --no-mtp} options run under their defaults), and PEFT
     * {@code model.layers.*.mixer.*} names rename to the checkpoint's {@code backbone.layers.}; then the Mamba 2
     * and Granite inherited filters.
     */
    public static final HfNameFilter NEMOTRON_H_FILTER = (name, base) -> {
        String out = name;
        if (out.startsWith("language_model.")) {
            out = out.substring("language_model.".length());
        }
        if (out.startsWith("model.layers.") && out.contains(".mixer.")) {
            out = "backbone.layers." + out.substring("model.layers.".length());
        }
        return HfSsmHooks.mamba2Filter(out, s -> HfGraniteHooks.filter(s, base));
    };

    /** {@code NemotronHPuzzleModel.filter_tensors} (nemotron.py:921-928): the MTP head drops unconditionally. */
    public static final HfNameFilter PUZZLE_FILTER = (name, base) ->
            name.startsWith("mtp.") ? null : NEMOTRON_H_FILTER.filter(name, base);

    /** A folded MTP block occupies the extra trailing block (nemotron.py:455-472); -1 when there is none. */
    static final int NO_MTP = -1;

    /** The pattern token types of the list-form Nemotron patterns (nemotron.py:424-426). */
    private static final List<String> NEMOTRON_SSM_TYPES = Arrays.asList("mamba", "linear_attention");
    private static final List<String> NEMOTRON_ATTN_TYPES = Arrays.asList("attention", "full_attention");

    /** The {@code mtp.layers.*} weights the folded block renames (nemotron.py:679-685). */
    private static final Map<String, String> MTP_SPECIAL_RENAMES;

    static {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("mtp.layers.0.enorm.weight", "model.layers.{bid}.enorm.weight");
        m.put("mtp.layers.0.hnorm.weight", "model.layers.{bid}.hnorm.weight");
        m.put("mtp.layers.0.eh_proj.weight", "model.layers.{bid}.eh_proj.weight");
        m.put("mtp.layers.1.norm.weight", "model.layers.{bid}.post_attention_layernorm.weight");
        m.put("mtp.layers.1.final_layernorm.weight", "model.layers.{bid}.shared_head.norm.weight");
        MTP_SPECIAL_RENAMES = Collections.unmodifiableMap(m);
    }

    private HfSsmHybridHooksA() {
    }

    // ------------------------------------------------------------------ shared helpers

    /** Truthiness of a flat JSON value, as the converter's walrus tests read it. */
    static boolean truthy(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        if (v instanceof Map) {
            return !((Map<?, ?>) v).isEmpty();
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        return false;
    }

    /** The layers of {@code values} up to {@code blockCount} whose member equals {@code eq}. */
    private static List<Integer> whereEquals(List<?> values, String eq, int blockCount) {
        List<Integer> out = new ArrayList<>(blockCount);
        for (int i = 0; i < Math.min(values.size(), blockCount); i++) {
            if (eq.equals(values.get(i))) {
                out.add(i);
            }
        }
        return out;
    }

    /** The layers of a string pattern whose character is {@code eq}, required to match the block count. */
    private static List<Integer> whereChar(String pattern, char eq) {
        List<Integer> out = new ArrayList<>(pattern.length());
        for (int i = 0; i < pattern.length(); i++) {
            if (pattern.charAt(i) == eq) {
                out.add(i);
            }
        }
        return out;
    }

    /** The complement of {@code selected} in {@code [0, blockCount)}. */
    private static List<Integer> complement(List<Integer> selected, int blockCount) {
        List<Integer> out = new ArrayList<>(blockCount);
        for (int i = 0; i < blockCount; i++) {
            if (!selected.contains(i)) {
                out.add(i);
            }
        }
        return out;
    }

    /** The per-block vector of {@code value} on the layers of {@code selected}, 0 elsewhere. */
    private static List<Integer> perLayer(List<Integer> selected, int value, int blockCount) {
        List<Integer> out = new ArrayList<>(blockCount);
        for (int i = 0; i < blockCount; i++) {
            out.add(selected.contains(i) ? value : 0);
        }
        return out;
    }

    /** The per-block object vector of {@code value} on {@code selected}, {@code 0} elsewhere. */
    private static List<Number> perLayerNumbers(List<Integer> selected, long value, int blockCount) {
        List<Number> out = new ArrayList<>(blockCount);
        for (int i = 0; i < blockCount; i++) {
            out.add(selected.contains(i) ? value : 0L);
        }
        return out;
    }

    /** The first present of {@code keys} and their {@code mamba_}-prefixed spellings, required. */
    private static int prefixedRequireInt(HfConfig cfg, String... keys) {
        return HfSsmHooks.prefixedRequireInt(cfg, keys);
    }

    private static OptionalDouble prefixedOptional(HfConfig cfg, String key) {
        List<String> aliases = HfSsmHooks.prefixedAliases("mamba", key);
        return cfg.findFloat(aliases.toArray(new String[0]));
    }

    /** The pattern that selects the layer types: the override string or the {@code layers_block_type} list. */
    private static List<Integer> patternLayers(HfConfig cfg, Object pattern, List<String> strings, Character token) {
        if (pattern instanceof String) {
            return whereChar((String) pattern, token == null ? '\0' : token);
        }
        List<Integer> out = new ArrayList<>();
        List<?> values = (List<?>) pattern;
        for (int i = 0; i < values.size(); i++) {
            Object v = values.get(i);
            if (!(v instanceof String)) {
                throw new GgufFormatException(cfg.source() + ": the layer pattern must hold strings, got " + v);
            }
            if (strings == null ? token != null && ((String) v).equals(token.toString()) : strings.contains(v)) {
                out.add(i);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ GraniteHybridModel (granite.py:436)

    /**
     * {@code GraniteHybridModel.get_attn_layers} (granite.py:494-512): the explicit {@code layer_types} list, else
     * the non-empty {@code attn_layer_indices}, else the layers {@code i} with {@code i % attn_layer_period ==
     * attn_layer_offset} (both required then).
     *
     * @throws GgufFormatException when no rule selects attention layers
     */
    public static List<Integer> graniteAttnLayers(HfConfig cfg, int blockCount) {
        Object layerTypes = cfg.flat().get("layer_types");
        if (layerTypes instanceof List) {
            return whereEquals((List<?>) layerTypes, "attention", blockCount);
        }
        Object indices = cfg.flat().get("attn_layer_indices");
        if (indices instanceof List && !((List<?>) indices).isEmpty()) {
            List<?> items = (List<?>) indices;
            List<Integer> out = new ArrayList<>(items.size());
            for (Object o : items) {
                if (!(o instanceof Number)) {
                    throw new GgufFormatException(cfg.source() + ": 'attn_layer_indices' must hold integers, got " + o);
                }
                out.add(((Number) o).intValue());
            }
            return out;
        }
        OptionalDouble period = cfg.findFloat("attn_layer_period");
        if (!period.isPresent() || period.getAsDouble() <= 0) {
            throw new GgufFormatException(cfg.source() + ": neither 'layer_types' nor 'attn_layer_indices' nor"
                    + " 'attn_layer_period' selects the attention layers");
        }
        OptionalDouble offset = cfg.findFloat("attn_layer_offset");
        if (!offset.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": 'attn_layer_offset' is required with"
                    + " 'attn_layer_period'");
        }
        int p = (int) period.getAsDouble();
        int o = (int) offset.getAsDouble();
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < blockCount; i++) {
            if (i % p == o) {
                out.add(i);
            }
        }
        return out;
    }

    /** The SSM layers: every block that is not an attention layer (granite.py:447-450). */
    public static List<Integer> graniteSsmLayers(HfConfig cfg, int blockCount) {
        return complement(graniteAttnLayers(cfg, blockCount), blockCount);
    }

    /**
     * The converter's {@code __init__}: GRANITE_HYBRID when SSM layers exist, else a non-hybrid granite
     * (GRANITE_MOE with experts, GRANITE without, granite.py:451-462).
     */
    public static Arch graniteHybridArch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        int blockCount = cfg.blockCount();
        if (!graniteSsmLayers(cfg, blockCount).isEmpty()) {
            return Arch.GRANITE_HYBRID;
        }
        boolean hasExperts = truthy(cfg.find("num_experts_per_tok", "num_experts_per_token").orElse(null));
        return hasExperts ? Arch.GRANITE_MOE : Arch.GRANITE;
    }

    /** {@code set_vocab}'s padding rule (granite.py:568-571): pad 8 when SSM layers exist, 1 when they do not. */
    public static HfConfig graniteHybridFixup(HfConfig cfg) {
        int pad = graniteSsmLayers(cfg, cfg.blockCount()).isEmpty() ? 1 : 8;
        return cfg.withFlat(Map.of("pad_vocab_size_multiple", pad));
    }

    /** {@link #graniteHybridMeta} over the Granite layer rules. */
    public static void graniteHybridMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        graniteCoreMeta(ctx, kv, plan, graniteAttnLayers(cfg, ctx.blockCount()),
                graniteSsmLayers(cfg, ctx.blockCount()));
    }

    /**
     * {@code GraniteHybridModel.set_gguf_parameters} (granite.py:519-558): the Granite MoE parameters, then the
     * Mamba mixer sizes (conv kernel, state size, group count, inner size and the {@code n_heads}-ranked time step
     * rank), the per-layer key/value head vector, the optional {@code attn_rotary_emb} RoPE width, and the
     * Bamba/non-hybrid RoPE flag with the {@code 2^20} context of the rope-less builds. The silu-only and
     * head-divisibility validations close.
     */
    static void graniteCoreMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, List<Integer> attnLayers,
                                List<Integer> ssmLayers) throws IOException {
        HfGraniteHooks.moeMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        int blockCount = ctx.blockCount();
        long dModel = cfg.requireInt("mamba_head_dim", "hidden_size", "d_model",
                "mamba_hidden_size", "mamba_d_model");
        long dInner = prefixedRequireInt(cfg, "expand", "num_heads") * dModel;
        long nGroup = prefixedRequireInt(cfg, "n_groups", "num_groups");
        kv.set(GgufKeys.SSM_CONV_KERNEL, (long) prefixedRequireInt(cfg, "conv_kernel", "d_conv"));
        kv.set(GgufKeys.SSM_STATE_SIZE, (long) prefixedRequireInt(cfg, "state_size", "d_state", "state_dim",
                "ssm_state_size"));
        kv.set(GgufKeys.SSM_GROUP_COUNT, nGroup);
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, (long) prefixedRequireInt(cfg, "n_heads", "num_heads"));
        int headCountKv = prefixedRequireInt(cfg, "num_key_value_heads", "n_head_kv");
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, perLayer(attnLayers, headCountKv, blockCount));
        Object ropeDim = cfg.find("attn_rotary_emb").orElse(null);
        if (ropeDim instanceof Number) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ((Number) ropeDim).longValue());
        }
        boolean useRope = "BambaForCausalLM".equals(cfg.architecture())
                || architectureList(cfg).contains("BambaForCausalLM")
                || ssmLayers.isEmpty();
        kv.set(GgufKeys.ROPE_SCALING_FINETUNED, useRope);
        if (!useRope) {
            kv.set(GgufKeys.CONTEXT_LENGTH, HfSsmHooks.DEFAULT_CONTEXT_LENGTH);
        }
        OptionalDouble dHeadOpt = prefixedOptional(cfg, "d_head");
        long dHead = dHeadOpt.isPresent() ? (long) dHeadOpt.getAsDouble() : 64L;
        Optional<String> act = cfg.findString("hidden_act");
        if (act.isPresent() && !"silu".equals(act.get())) {
            throw new GgufFormatException(cfg.source() + ": Only SILU activation supported, hidden_act '"
                    + act.get() + "' (granite.py)");
        }
        if (dInner % dHead != 0) {
            throw new GgufFormatException(cfg.source() + ": SSM inner size " + dInner + " is not a multiple of"
                    + " head dim " + dHead);
        }
    }

    /** The {@code architectures} list of the configuration (an empty list when it has none). */
    private static List<String> architectureList(HfConfig cfg) {
        Object v = cfg.flat().get("architectures");
        if (!(v instanceof List)) {
            return List.of();
        }
        List<String> out = new ArrayList<>(((List<?>) v).size());
        for (Object o : (List<?>) v) {
            if (o instanceof String) {
                out.add((String) o);
            }
        }
        return out;
    }

    /**
     * {@code GraniteHybridModel.modify_tensors} (granite.py:514-518) over the Granite layer rules.
     */
    public static void graniteHybridTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfConfig cfg = ctx.config();
        int blockCount = ctx.blockCount();
        graniteHybridTensor(ctx, hfName, src, out,
                graniteAttnLayers(cfg, blockCount), graniteSsmLayers(cfg, blockCount));
    }

    /**
     * The routing over the caller's layer lists: GraniteHybridModel.__init__ computes them from
     * {@code self.get_attn_layers()}, whose NemotronHModel override (nemotron.py:481-490) reads the
     * pattern the Nemotron readers resolve. The Granite MoE fused projections and shared MLP tensors
     * always route through {@link HfGraniteHooks#moeTensors}; the other tensors of an SSM layer go
     * through the Mamba 2 path and of an attention layer through the Granite MoE path; the rest
     * (embeddings, the final norm) map through the plain Llama path.
     */
    static void graniteHybridTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out,
                                    List<Integer> attnLayers, List<Integer> ssmLayers) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith("block_sparse_moe.input_linear.weight") || hfName.contains("shared_mlp")) {
            HfGraniteHooks.moeTensors(ctx, hfName, src, out);
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (ssmLayers.contains(bid)) {
            HfSsmHooks.mamba2Tensor(ctx, hfName, src, out);
            return;
        }
        if (attnLayers.contains(bid)) {
            HfGraniteHooks.moeTensors(ctx, hfName, src, out);
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    /**
     * {@code GraniteHybridModel.set_vocab} (granite.py:568-571, then Mamba 2's): the vocabulary padded by the
     * fixed-up {@code pad_vocab_size_multiple}, the SentencePiece, restricted-{@code .v3} and GPT-2 chain of
     * {@code Mamba2Model.set_vocab} (mamba.py:128-141), with the builtin GPT-NeoX tokenizer as the last resort.
     */
    public static Map<String, Object> graniteHybridVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfSsmHooks.ssmVocab(ctx, pre, ctx.config().findInt(8, "pad_vocab_size_multiple"), false);
    }

    // ------------------------------------------------------------------ NemotronHModel (nemotron.py:405)

    /** True when the checkpoint is a MoE build: the flat configuration holds {@code num_experts_per_tok}. */
    public static boolean nemotronIsMoe(HfConfig cfg) {
        return truthy(cfg.find("num_experts_per_tok").orElse(null));
    }

    /** NEMOTRON_H_MOE for the {@code num_experts_per_tok} checkpoints, NEMOTRON_H otherwise (nemotron.py:429-433). */
    public static Arch nemotronHArch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        return nemotronIsMoe(cfg) ? Arch.NEMOTRON_H_MOE : Arch.NEMOTRON_H;
    }

    /**
     * {@code NemotronHModel.__init__}'s fixup (nemotron.py:434-438): {@code num_hidden_layers := 
     * len(layers_block_type)}, and the MTP head folds only when {@code num_nextn_predict_layers == 1}.
     */
    public static HfConfig nemotronHFixup(HfConfig cfg) {
        HfConfig out = cfg;
        Object lbt = cfg.flat().get("layers_block_type");
        if (lbt instanceof List && !((List<?>) lbt).isEmpty()) {
            out = out.withFlat(Map.of("num_hidden_layers", (long) ((List<?>) lbt).size()));
        }
        if (nemotronIsMoe(cfg) && cfg.findLong(0, "num_nextn_predict_layers") > 1) {
            throw new GgufFormatException(cfg.source() + ": NemotronH MTP conversion supports"
                    + " num_nextn_predict_layers == 1");
        }
        return out;
    }

    /** True when the config folds an MTP block as the converter's MoE MTP head does (nemotron.py:443-459). */
    public static boolean nemotronMtpFolded(HfConfig cfg) {
        return nemotronIsMoe(cfg) && cfg.findLong(0, "num_nextn_predict_layers") > 0;
    }

    /** The nonzero block count of a Nemotron load: the trunk layers plus the folded MTP block when there is one. */
    public static int nemotronHBlockCount(HfLoadContext ctx) {
        int base = ctx.config().blockCount();
        return nemotronMtpFolded(ctx.config()) ? base + 1 : base;
    }

    /** The pattern the layer lists come from: the override string or the {@code layers_block_type} list. */
    private static Object nemotronPattern(HfConfig cfg) {
        Object p = cfg.flat().get("hybrid_override_pattern");
        return p != null && !"".equals(p) ? p : cfg.flat().get("layers_block_type");
    }

    /** {@code get_attn_layers} (nemotron.py:495-505): {@code *}/attention layers of the pattern. */
    public static List<Integer> nemotronAttnLayers(HfConfig cfg, int blockCount, boolean moe) {
        Object pattern = nemotronPattern(cfg);
        if (pattern == null) {
            return List.of();
        }
        return patternLayers(cfg, pattern, NEMOTRON_ATTN_TYPES, '*');
    }

    /** {@code __init__}'s SSM layers: {@code M}/mamba layers of the pattern. */
    public static List<Integer> nemotronSsmLayers(HfConfig cfg, int blockCount, boolean moe) {
        Object pattern = nemotronPattern(cfg);
        if (pattern == null) {
            return List.of();
        }
        return patternLayers(cfg, pattern, NEMOTRON_SSM_TYPES, 'M');
    }

    /** {@code __init__}'s MLP layers: {@code -}/moe layers of the pattern ({@code E} on MoE builds). */
    public static List<Integer> nemotronMlpLayers(HfConfig cfg, int blockCount, boolean moe) {
        Object pattern = nemotronPattern(cfg);
        if (pattern == null) {
            return List.of();
        }
        return patternLayers(cfg, pattern, List.of("moe"), moe ? 'E' : '-');
    }

    /** {@link #graniteCoreMeta} over the pattern's layer lists plus the folded MTP block. */
    public static void nemotronHMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        boolean moe = nemotronIsMoe(cfg);
        int blockCount = ctx.blockCount();
        int mtp = nemotronMtpFolded(cfg) ? cfg.blockCount() : NO_MTP;
        List<Integer> attn = new ArrayList<>(nemotronAttnLayers(cfg, blockCount, moe));
        List<Integer> ssm = new ArrayList<>(nemotronSsmLayers(cfg, blockCount, moe));
        List<Integer> mlp = new ArrayList<>(nemotronMlpLayers(cfg, blockCount, moe));
        if (mtp >= 0) {
            attn.add(mtp);
            mlp.add(mtp);
        }
        graniteCoreMeta(ctx, kv, plan, attn, ssm);
        long headDim = prefixedRequireInt(cfg, "head_dim", "attention_head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        if (!moe) {
            long nFf = prefixedRequireInt(cfg, "intermediate_size", "n_inner", "hidden_dim");
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, perLayerNumbers(mlp, nFf, blockCount));
        } else {
            long moeFfn = cfg.requireInt("moe_intermediate_size");
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, perLayerNumbers(mlp, moeFfn, blockCount));
            kv.set(GgufKeys.EXPERT_USED_COUNT,
                    (long) cfg.requireInt("num_experts_per_tok", "num_experts_per_token"));
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, perLayerNumbers(mlp, moeFfn, blockCount));
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH,
                    (long) cfg.requireInt("moe_shared_expert_intermediate_size"));
            kv.set(GgufKeys.EXPERT_COUNT, (long) cfg.requireInt("n_routed_experts"));
            kv.set(GgufKeys.EXPERT_SHARED_COUNT, (long) cfg.requireInt("n_shared_experts"));
            kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0, "routed_scaling_factor"));
            kv.set(GgufKeys.EXPERT_GROUP_COUNT, (long) cfg.requireInt("n_group"));
            OptionalDouble latent = cfg.findFloat("moe_latent_size");
            if (latent.isPresent()) {
                kv.set(GgufKeys.MOE_LATENT_SIZE, (long) latent.getAsDouble());
            }
        }
        if (mtp >= 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, cfg.requireInt("num_nextn_predict_layers"));
        }
    }

    /**
     * {@code NemotronHModel.modify_tensors} (nemotron.py:687-761): the folded MTP names rename first; then,
     * on every block of a MoE build, the MoE mixer special cases (router bias through the plain path, the dt bias
     * rename, the conv squeeze, the A/D negative-exponent single-column reshape, the grouped norm reshape and the
     * routed experts stacked down/up); everything else and the dense model through
     * {@link #graniteHybridTensor}.
     */
    public static void nemotronHTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        nemotronHTensor(ctx, hfName, src, out, nemotronIsMoe(ctx.config()));
    }

    /**
     * The Nemotron-H tensor path with the MoE flag explicit: Puzzle is always a MoE build (nemotron.py:741).
     * {@code NemotronHModel.modify_tensors} (nemotron.py:687-761): the folded MTP names rename first; then,
     * on every block of a MoE build, the MoE mixer special cases (router bias through the plain path, the dt bias
     * rename, the conv squeeze, the A/D negative-exponent single-column reshape, the grouped norm reshape and the
     * routed experts stacked down/up); everything else routes through the hybrid path over the pattern's layer
     * lists, the MTP block registered as an attention layer (nemotron.py:472-474).
     */
    static void nemotronHTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, boolean moe)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        String name = hfName;
        int mtp = nemotronMtpFolded(cfg) ? ctx.config().blockCount() : NO_MTP;
        if (mtp >= 0 && (name.startsWith("mtp.layers.0.") || name.startsWith("mtp.layers.1."))) {
            String special = MTP_SPECIAL_RENAMES.get(name);
            String suffix = name.split("\\.", 4)[3];
            name = special != null ? special.replace("{bid}", String.valueOf(mtp))
                    : "backbone.layers." + mtp + "." + suffix;
        }
        if (moe && HfLoadContext.bidOf(name) >= 0) {
            if (name.endsWith("mixer.gate.e_score_correction.bias")) {
                out.add(ctx.names().mapOrThrow(name), src);
                return;
            }
            if (name.endsWith("mixer.dt_bias")) {
                String renamed = name.replace("dt_bias", "dt.bias");
                out.add(ctx.names().mapOrThrow(renamed), src);
                return;
            }
            if (name.endsWith("mixer.conv1d.weight")) {
                out.add(ctx.names().mapOrThrow(name), ElementTransforms.squeezeConv(src));
                return;
            }
            if (name.endsWith("mixer.A_log")) {
                RowSource s = ElementTransforms.reshape(ElementTransforms.squeezeConv(
                        ElementTransforms.negExp(src)), -1L, 1L);
                out.add(ctx.names().mapOrThrow(name), s);
                return;
            }
            if (name.endsWith("mixer.D")) {
                out.add(ctx.names().mapOrThrow(name),
                        ElementTransforms.reshape(ElementTransforms.squeezeConv(src), -1L, 1L));
                return;
            }
            if (name.endsWith("mixer.norm.weight")) {
                long nGroup = prefixedRequireInt(cfg, "n_groups", "num_groups");
                out.add(ctx.names().mapOrThrow(name), ElementTransforms.reshape(src, nGroup, -1L));
                return;
            }
            if (name.contains("mixer.experts")) {
                nemotronExperts(ctx, cfg.requireInt("n_routed_experts")).offer(ctx, name, out);
                return;
            }
        }
        // the pattern's layer lists route the hybrid path; the folded MTP block registers
        // as an attention layer (nemotron.py:472-474)
        int blocks = nemotronHBlockCount(ctx);
        List<Integer> attn = new ArrayList<>(nemotronAttnLayers(cfg, blocks, moe));
        if (mtp >= 0) {
            attn.add(mtp);
        }
        graniteHybridTensor(ctx, name, src, out, attn, nemotronSsmLayers(cfg, blocks, moe));
    }

    /** The per-load Nemotron MoE expert collector (nemotron.py:668-686), two parts down/up. */
    private static HfHookSupport.ExpertCollector nemotronExperts(HfLoadContext ctx, int nExperts) {
        return ctx.state(HfSsmHybridHooksA.class.getName() + ".experts", HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(nExperts, Arrays.asList("down_proj", "up_proj"),
                        (prefix, part, tail) -> "model.layers." + HfHookSupport.layerOf(prefix)
                                + ".mlp.experts." + part + tail));
    }

    /** {@code NemotronHModel.prepare_tensors}' unprocessed-experts check (nemotron.py:763-768). */
    public static void nemotronHExtra(HfLoadContext ctx, TensorSink out) {
        if (!nemotronIsMoe(ctx.config())) {
            return;
        }
        HfHookSupport.ExpertCollector experts = ctx.state(
                HfSsmHybridHooksA.class.getName() + ".experts", HfHookSupport.ExpertCollector.class,
                () -> new HfHookSupport.ExpertCollector(ctx.config().requireInt("n_routed_experts"),
                        Arrays.asList("down_proj", "up_proj"),
                        (prefix, part, tail) -> "model.layers." + HfHookSupport.layerOf(prefix)
                                + ".mlp.experts." + part + tail));
        experts.requireComplete();
    }

    /** {@code NemotronHModel.set_vocab}'s pad-8 Mamba 2 chain (nemotron.py:600-606). */
    public static Map<String, Object> nemotronHVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfSsmHooks.ssmVocab(ctx, pre, 8, false);
    }

    // ------------------------------------------------------------------ NemotronHPuzzleModel (nemotron.py:733)

    /** {@code NemotronHPuzzleModel.__init__} (nemotron.py:743-761): every {@code block_configs} block fixes the
     * layer count and the {@code layers_block_type} the pattern reads. */
    public static HfConfig nemotronHPuzzleFixup(HfConfig cfg) {
        Object bc = cfg.flat().get("block_configs");
        if (!(bc instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": 'block_configs' is a required list of per-block"
                    + " configurations");
        }
        List<String> types = new ArrayList<>(((List<?>) bc).size());
        for (Object o : (List<?>) bc) {
            Object type = o instanceof Map ? ((Map<?, ?>) o).get("block_type") : null;
            if (!(type instanceof String)) {
                throw new GgufFormatException(cfg.source() + ": every 'block_configs' entry needs a string"
                        + " 'block_type', got " + o);
            }
            types.add((String) type);
        }
        Map<String, Object> patch = new LinkedHashMap<>();
        patch.put("num_hidden_layers", (long) types.size());
        patch.put("layers_block_type", types);
        return cfg.withFlat(patch);
    }

    /** Puzzle checkpoints are always MoE builds (nemotron.py:741). */
    public static Arch nemotronHPuzzleArch(HfConfig cfg, com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex index) {
        return Arch.NEMOTRON_H_MOE;
    }

    /**
     * {@code NemotronHPuzzleModel.set_gguf_parameters} (nemotron.py:766-800): the Granite hybrid parameters, then
     * the key/value lengths and the per-block feed-forward and expert arrays of {@code block_configs}, with the
     * shared-expert, routing and latent sizes of the configuration.
     */
    public static void nemotronHPuzzleMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        HfConfig cfg = ctx.config();
        // the Puzzle pattern rides layers_block_type (the fixup copies block_configs' block_type),
        // read through the Nemotron pattern readers like NemotronHModel's own metadata
        graniteCoreMeta(ctx, kv, plan, nemotronAttnLayers(cfg, ctx.blockCount(), true),
                nemotronSsmLayers(cfg, ctx.blockCount(), true));
        long headDim = prefixedRequireInt(cfg, "head_dim", "attention_head_dim");
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        Object bc = cfg.flat().get("block_configs");
        List<?> blocks = (List<?>) bc;
        List<Number> ffn = new ArrayList<>(blocks.size());
        List<Number> used = new ArrayList<>(blocks.size());
        for (Object o : blocks) {
            Map<?, ?> b = (Map<?, ?>) o;
            Object moeFfn = b.get("moe_intermediate_size");
            Object nUsed = b.get("num_experts_per_tok");
            ffn.add(moeFfn instanceof Number ? ((Number) moeFfn).longValue() : 0L);
            used.add(nUsed instanceof Number ? ((Number) nUsed).longValue() : 0L);
        }
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.EXPERT_USED_COUNT, used);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH,
                (long) cfg.requireInt("moe_shared_expert_intermediate_size"));
        kv.set(GgufKeys.EXPERT_COUNT, (long) cfg.requireInt("n_routed_experts"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, (long) cfg.requireInt("n_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0, "routed_scaling_factor"));
        kv.set(GgufKeys.EXPERT_GROUP_COUNT, (long) cfg.requireInt("n_group"));
        kv.set(GgufKeys.MOE_LATENT_SIZE, (long) cfg.requireInt("moe_latent_size"));
    }

    /**
     * {@code NemotronHPuzzleModel.modify_tensors} (nemotron.py:802-807): the {@code model.} trunk names of the
     * official BF16 checkpoint rename to {@code backbone.*} and the router bias to the dotted MoE form, then the
     * Nemotron-H path applies.
     */
    public static void nemotronHPuzzleTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.startsWith("model.")) {
            name = "backbone." + name.substring("model.".length());
        }
        if (name.endsWith("mixer.gate.e_score_correction_bias")) {
            name = name.substring(0, name.length() - "e_score_correction_bias".length())
                    + "e_score_correction.bias";
        }
        nemotronHTensor(ctx, name, src, out, true);
    }

    /** {@link #nemotronHVocab}: the Puzzle shares the Nemotron-H vocabulary chain. */
    public static Map<String, Object> nemotronHPuzzleVocab(HfLoadContext ctx, PreNameResolver pre)
            throws IOException {
        return nemotronHVocab(ctx, pre);
    }
}

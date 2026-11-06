package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.hf.tensor.ColumnTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.function.UnaryOperator;

/**
 * Hooks of the Hugging Face architecture names vLLM registers for model families NeuralFree converts under other
 * names (V/model_executor/models/registry.py), each composed over the family's own hooks:
 * <ul>
 *   <li>the headless embedding checkpoints ({@code MistralModel}, {@code Gemma2Model},
 *       {@code DeepseekV3BidirectionalModel}): the {@code model.} prefix the {@code *ForCausalLM} layout carries and
 *       the pooling of vLLM's embedding adapter ({@link #headless}, {@link #embeddingPooling});</li>
 *   <li>{@code TeleFLMForCausalLM} (teleflm.py, SGLang teleflm.py): Llama with the muP input and output multipliers,
 *       carried by the Granite architecture's embedding and logit scales ({@link #teleFlmMeta});</li>
 *   <li>{@code TeleChat2ForCausalLM} (telechat2.py): Llama under the {@code transformer.h} names with a per-head
 *       interleaved {@code key_value} projection ({@link #teleChat2Fixup}, {@link #teleChat2Filter},
 *       {@link #teleChat2Tensors});</li>
 *   <li>{@code TeleChat3ForCausalLM} (registry.py, transformers_utils/config.py {@code telechat3-yarn}): Llama with a
 *       YaRN attention factor of its own ({@link #teleChat3Meta});</li>
 *   <li>{@code MiMoForCausalLM} (mimo.py) and {@code MiMoMTPModel}, which vLLM serves the same checkpoint through as
 *       its MTP draft (mimo_mtp.py {@code MiMoMTP}): Qwen2 whose {@code model.mtp_layers} become the appended NextN
 *       blocks of the draft head ({@link #mimoMeta}, {@link #mimoTensors}).</li>
 * </ul>
 */
public final class HfVllmAliasHooks {

    private static final String MODEL_PREFIX = "model.";
    private static final String LM_HEAD_PREFIX = "lm_head.";
    private static final String TELECHAT2_PREFIX = "transformer.";
    private static final String KEY_VALUE = "key_value.weight";
    private static final String MIMO_MTP_PREFIX = "model.mtp_layers.";

    private HfVllmAliasHooks() {
    }

    // ------------------------------------------------------------------ headless embedding checkpoints

    /**
     * A class filter giving a headless checkpoint name the {@code model.} prefix of the {@code *ForCausalLM} layout
     * before {@code inner} runs: vLLM loads its pooling models from either layout (adapters.py
     * {@code _load_pooling_model_weights} tries the prefixes {@code ""} and {@code "model."}). Names that already
     * carry the prefix, and the {@code lm_head.} tensors, pass unchanged.
     */
    public static HfNameFilter headless(HfNameFilter inner) {
        Objects.requireNonNull(inner, "inner");
        return (name, base) -> inner.filter(withModelPrefix(name), base);
    }

    /** {@code name} with the {@code model.} prefix, unless it has it already or is an {@code lm_head.} tensor. */
    static String withModelPrefix(String name) {
        return name.startsWith(MODEL_PREFIX) || name.startsWith(LM_HEAD_PREFIX) ? name : MODEL_PREFIX + name;
    }

    /**
     * The pooling of a checkpoint vLLM serves through its embedding adapter: the sentence-transformers pooling
     * module or {@code embedding_config.json} when the checkpoint declares one ({@link HfQwenHooks#pooling}), else
     * last-token pooling, the {@code default_pooling_type} of vLLM's decoder models (interfaces_base.py).
     *
     * @throws IOException         when a pooling side file cannot be read
     * @throws GgufFormatException when a pooling side file names an unsupported pooling
     */
    public static void embeddingPooling(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        HfQwenHooks.pooling(ctx, kv);
        if (!kv.has(GgufKeys.POOLING_TYPE)) {
            kv.set(GgufKeys.POOLING_TYPE, PoolingKind.LAST.ggufCode());
        }
    }

    /** {@code inner}'s metadata, then the pooling of vLLM's embedding adapter ({@link #embeddingPooling}). */
    public static HfMetadataHook withEmbeddingPooling(HfMetadataHook inner) {
        Objects.requireNonNull(inner, "inner");
        return (ctx, kv, plan) -> {
            inner.apply(ctx, kv, plan);
            embeddingPooling(ctx, kv);
        };
    }

    // ------------------------------------------------------------------ MiMo

    /** The MTP layer count the checkpoint declares ({@code MiMoMultiTokenPredictor.num_mtp_layers}). */
    static int mimoNextn(HfConfig cfg) {
        int n = cfg.findInt(0, "num_nextn_predict_layers");
        if (n < 0) {
            throw new GgufFormatException(cfg.source() + ": num_nextn_predict_layers " + n + " is negative");
        }
        return n;
    }

    /** True when the load keeps a positive number of MiMo MTP layers. */
    private static boolean mimoKept(HfLoadContext ctx) {
        return ctx.options().includeMtp() && mimoNextn(ctx.config()) > 0;
    }

    /**
     * The block count of a MiMo load ({@code MiMoMultiTokenPredictor.__init__}): the trunk plus the
     * {@code num_nextn_predict_layers} MTP blocks, unless the MTP layers are dropped.
     */
    public static int mimoBlockCount(HfLoadContext ctx) {
        int n = ctx.config().blockCount();
        return mimoKept(ctx) ? n + mimoNextn(ctx.config()) : n;
    }

    /**
     * MiMo's {@code model.mtp_layers.<i>.*} (mimo_mtp.py {@code MiMoMultiTokenPredictorLayer}): the layer's decoder
     * block moves to the appended block {@code n + i}, its token and hidden norms and its input projection become
     * that block's {@code nextn.enorm}, {@code nextn.hnorm} and {@code nextn.eh_proj}, and its final norm becomes
     * {@code nextn.shared_head_norm}. With {@code --no-mtp} the names are dropped. Names outside the prefix pass
     * through the inherited filter.
     *
     * @throws GgufFormatException when the checkpoint holds MTP tensors without declaring
     *                             {@code num_nextn_predict_layers}, names a layer at or above the declared count, or
     *                             holds a tensor the MTP layer does not define
     */
    public static HfNameFilter mimoFilter(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int trunk = cfg.blockCount();
        int nextn = mimoNextn(cfg);
        boolean include = ctx.options().includeMtp();
        return (name, base) -> {
            if (!name.startsWith(MIMO_MTP_PREFIX)) {
                return base.apply(name);
            }
            if (!include) {
                return null;
            }
            return base.apply(mimoMtpName(name, trunk, nextn));
        };
    }

    /**
     * The metadata of a MiMo checkpoint: the Qwen2 keys, then {@code nextn_predict_layers} when the MTP layers are
     * kept.
     */
    public static void mimoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenHooks.qwen2Meta(ctx, kv, plan);
        if (!mimoKept(ctx)) {
            return;
        }
        int nextn = mimoNextn(ctx.config());
        int trunk = ctx.config().blockCount();
        for (int i = 0; i < nextn; i++) {
            for (String tensor : new String[] {"nextn.eh_proj", "nextn.enorm", "nextn.hnorm",
                    "nextn.shared_head_norm"}) {
                String name = "blk." + (trunk + i) + "." + tensor + ".weight";
                if (!plan.has(name)) {
                    throw new GgufFormatException(ctx.config().source() + ": MiMo MTP layer " + i + " has no "
                            + name + " (num_nextn_predict_layers is " + nextn + ")");
                }
            }
        }
        kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
    }

    /**
     * The tensors of a MiMo checkpoint, under the names {@link #mimoFilter(HfLoadContext)} gives them: the
     * {@code input_proj} weight ({@code nextn.eh_proj}) has its two input halves swapped because MiMo concatenates
     * the normalized hidden row before the embedded token ({@code MiMoMultiTokenPredictorLayer.forward}) while
     * {@code nextn.eh_proj} takes the embedded token first, as every llama.cpp MTP graph does; every other tensor
     * goes through the Qwen2 tensor hook.
     *
     * @throws GgufFormatException as {@link #mimoFilter(HfLoadContext)}
     */
    public static void mimoTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= ctx.config().blockCount()) {
            String nextn = nextnTensor(hfName);
            if (nextn != null) {
                out.add(HfNameTables.formatTensorName(nextn, bid) + ".weight",
                        hfName.endsWith(".eh_proj.weight") ? swappedInputProjection(ctx, hfName, src) : src);
                return;
            }
        }
        HfQwenHooks.qwen2Tensors(ctx, hfName, src, out);
    }

    /** The NextN tensor of a MiMo MTP tensor's filtered name, or null for a tensor of the block body. */
    private static String nextnTensor(String hfName) {
        if (hfName.endsWith(".eh_proj.weight")) {
            return "NEXTN_EH_PROJ";
        }
        if (hfName.endsWith(".enorm.weight")) {
            return "NEXTN_ENORM";
        }
        if (hfName.endsWith(".hnorm.weight")) {
            return "NEXTN_HNORM";
        }
        return hfName.endsWith(".shared_head.norm.weight") ? "NEXTN_SHARED_HEAD_NORM" : null;
    }

    /** The {@code input_proj} weight with MiMo's {@code [hidden | embed]} input order turned into {@code [embed | hidden]}. */
    private static RowSource swappedInputProjection(HfLoadContext ctx, String hfName, RowSource src) {
        int hidden = ctx.config().requireInt("hidden_size");
        try {
            return ColumnTransforms.permuteColGroups(src, hidden, new int[]{1, 0});
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into the two hidden halves"
                    + " of MiMo's input_proj: " + e.getMessage(), e);
        }
    }

    /** The appended-block name of a MiMo MTP tensor. */
    private static String mimoMtpName(String name, int trunk, int nextn) {
        String rest = name.substring(MIMO_MTP_PREFIX.length());
        int dot = rest.indexOf('.');
        if (dot < 0) {
            throw new GgufFormatException("tensor '" + name + "' is not a MiMo MTP layer tensor");
        }
        int index = mimoIndex(rest.substring(0, dot), name);
        if (index >= nextn) {
            throw new GgufFormatException("tensor '" + name + "' is MiMo MTP layer " + index + ", but"
                    + " num_nextn_predict_layers is " + nextn);
        }
        String suffix = rest.substring(dot + 1);
        String block = MODEL_PREFIX + "layers." + (trunk + index) + ".";
        if (suffix.equals("token_layernorm.weight")) {
            return block + "enorm.weight";
        }
        if (suffix.equals("hidden_layernorm.weight")) {
            return block + "hnorm.weight";
        }
        if (suffix.equals("input_proj.weight")) {
            return block + "eh_proj.weight";
        }
        if (suffix.equals("final_layernorm.weight")) {
            return block + "shared_head.norm.weight";
        }
        if (suffix.startsWith("self_attn.") || suffix.startsWith("mlp.")
                || suffix.equals("input_layernorm.weight") || suffix.equals("post_attention_layernorm.weight")) {
            return block + suffix;
        }
        throw new GgufFormatException("tensor '" + name + "' is not supported: a MiMo MTP layer has no '"
                + suffix + "'");
    }

    private static int mimoIndex(String index, String name) {
        if (index.isEmpty() || index.length() > 9) {
            throw new GgufFormatException("tensor '" + name + "' names no MiMo MTP layer");
        }
        for (int i = 0; i < index.length(); i++) {
            char c = index.charAt(i);
            if (c < '0' || c > '9') {
                throw new GgufFormatException("tensor '" + name + "' names no MiMo MTP layer");
            }
        }
        return Integer.parseInt(index);
    }

    // ------------------------------------------------------------------ TeleFLM

    /**
     * {@code TeleFLMForCausalLM} (teleflm.py, SGLang teleflm.py): the Llama metadata ({@link HfLlamaHooks#meta}) on
     * the Granite architecture, whose embedding and logit scales are the muP multipliers. With {@code use_mup} the
     * embeddings are multiplied by {@code input_mult} and the logits by {@code output_mult / mup_scale_factor}
     * (Granite divides the logits by its scale, so the scale written is {@code mup_scale_factor / output_mult});
     * without it both scales are 1.
     *
     * @throws GgufFormatException when a muP multiplier is missing or not a positive finite number, or the RoPE type
     *                             is {@code llama3}, whose frequency factors the Granite architecture has no tensor
     *                             for
     * @throws IOException         as the Llama metadata does
     */
    public static void teleFlmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfConfig cfg = ctx.config();
        Optional<String> ropeType = cfg.rope().global().type();
        if (ropeType.isPresent() && "llama3".equals(ropeType.get())) {
            throw new GgufFormatException(cfg.source() + ": TeleFLMForCausalLM with rope_type 'llama3' is not"
                    + " supported (the muP Granite layout has no rope_freqs tensor)");
        }
        HfLlamaHooks.meta(ctx, kv, plan);
        double logitScale = 1.0;
        if (cfg.findBool(false, "use_mup")) {
            kv.set(GgufKeys.EMBEDDING_SCALE, positive(cfg, "input_mult"));
            logitScale = positive(cfg, "mup_scale_factor") / positive(cfg, "output_mult");
        }
        kv.set(GgufKeys.LOGIT_SCALE, logitScale);
    }

    /** The positive finite number {@code key} of {@code cfg}. */
    private static double positive(HfConfig cfg, String key) {
        OptionalDouble v = cfg.findFloat(key);
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key '" + key + "' is missing");
        }
        double d = v.getAsDouble();
        if (!(d > 0) || Double.isInfinite(d)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a positive finite number, got " + d);
        }
        return d;
    }

    // ------------------------------------------------------------------ TeleChat2

    /**
     * {@code TeleChat2Model.__init__} (telechat2.py): the configuration's {@code attribute_map}
     * ({@code n_layer}, {@code n_head}, {@code ffn_hidden_size} and {@code layer_norm_epsilon} read as
     * {@code num_hidden_layers}, {@code num_attention_heads}, {@code intermediate_size} and {@code rms_norm_eps}) and
     * the SiLU activation, then the Llama fixup ({@link HfLlamaHooks#fixup}). The epsilon is an RMSNorm epsilon, so
     * {@code layer_norm_epsilon} is cleared once copied.
     */
    public static HfConfig teleChat2Fixup(HfConfig cfg) {
        Map<String, Object> fix = new LinkedHashMap<>();
        copy(fix, cfg, "n_layer", "num_hidden_layers");
        copy(fix, cfg, "n_head", "num_attention_heads");
        copy(fix, cfg, "ffn_hidden_size", "intermediate_size");
        if (copy(fix, cfg, "layer_norm_epsilon", "rms_norm_eps")) {
            fix.put("layer_norm_epsilon", null);
        }
        fix.put("hidden_act", "silu");
        return HfLlamaHooks.fixup(cfg.withFlat(fix));
    }

    /** Copies {@code from} to {@code to} when {@code cfg} has a non-null {@code from}; true when it copied. */
    private static boolean copy(Map<String, Object> fix, HfConfig cfg, String from, String to) {
        Object v = cfg.flat().get(from);
        if (v == null) {
            return false;
        }
        fix.put(to, v);
        return true;
    }

    /**
     * {@code TeleChat2ForCausalLM.hf_to_vllm_mapper} and the {@code .query} stacking of
     * {@code TeleChat2Model.hf_to_vllm_mapper} (telechat2.py): {@code transformer.} becomes {@code model.},
     * {@code .h.} {@code .layers.}, {@code .self_attention.} {@code .self_attn.}, {@code .word_embeddings.}
     * {@code .embed_tokens.}, {@code .dense.} {@code .o_proj.}, {@code .ln_f.} {@code .norm.} and
     * {@code self_attn.query} {@code self_attn.q_proj}; then the Llama filters ({@link HfLlamaHooks#filter}). The
     * fused {@code key_value} projection keeps its name for {@link #teleChat2Tensors}.
     */
    public static String teleChat2Filter(String name, UnaryOperator<String> base) {
        String n = name.startsWith(TELECHAT2_PREFIX) ? MODEL_PREFIX + name.substring(TELECHAT2_PREFIX.length())
                : name;
        n = n.replace(".h.", ".layers.")
                .replace(".self_attention.", ".self_attn.")
                .replace(".word_embeddings.", ".embed_tokens.")
                .replace(".dense.", ".o_proj.")
                .replace(".ln_f.", ".norm.")
                .replace(".self_attn.query.", ".self_attn.q_proj.");
        return HfLlamaHooks.filter(n, base);
    }

    /**
     * {@code TeleChat2Model._split_key_value} (telechat2.py) before the Llama tensors ({@link HfLlamaHooks#tensors}):
     * the {@code key_value} weight holds one {@code [K_h; V_h]} block of {@code 2 * head_dim} rows per key/value head
     * ({@code head_dim = hidden_size / num_attention_heads}); its K and V halves become the {@code k_proj} and
     * {@code v_proj} weights, and K is permuted like every Llama K projection.
     *
     * @throws GgufFormatException when a {@code key_value} tensor is not a weight of {@code 2 * kv_heads * head_dim}
     *                             rows, or the head geometry is invalid
     * @throws IOException         as the Llama tensors do
     */
    public static void teleChat2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains(".self_attn.key_value.")) {
            if (!hfName.endsWith(".self_attn." + KEY_VALUE)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not supported: TeleChat2's fused"
                        + " key_value projection has a weight only");
            }
            splitKeyValue(ctx, hfName, src, out);
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    private static void splitKeyValue(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        int nHead = cfg.requireInt("num_attention_heads");
        int hidden = cfg.requireInt("hidden_size");
        if (nHead <= 0 || hidden % nHead != 0) {
            throw new GgufFormatException(cfg.source() + ": hidden_size " + hidden + " is not a multiple of"
                    + " num_attention_heads " + nHead);
        }
        int headDim = hidden / nHead;
        int nKv = cfg.findInt(nHead, "num_key_value_heads");
        long[] shape = src.shape();
        if (shape.length != 2 || nKv <= 0 || shape[0] != 2L * nKv * headDim) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + Arrays.toString(shape)
                    + ", expected [" + (2L * nKv * headDim) + ", " + hidden + "] (" + nKv + " key/value heads of "
                    + headDim + " K and " + headDim + " V rows)");
        }
        int[] k = new int[nKv];
        int[] v = new int[nKv];
        for (int h = 0; h < nKv; h++) {
            k[h] = 2 * h;
            v[h] = 2 * h + 1;
        }
        String prefix = hfName.substring(0, hfName.length() - KEY_VALUE.length());
        HfLlamaHooks.llamaTensor(ctx, hfName, prefix + "k_proj.weight", RowTransforms.gatherRows(src, headDim, k),
                out);
        HfLlamaHooks.llamaTensor(ctx, hfName, prefix + "v_proj.weight", RowTransforms.gatherRows(src, headDim, v),
                out);
    }

    // ------------------------------------------------------------------ TeleChat3

    /**
     * {@code TeleChat3ForCausalLM} (registry.py, served as {@code LlamaForCausalLM}): the Llama metadata, then the
     * YaRN attention factor of the configuration as the RoPE attention factor. {@link HfRope} turns
     * {@code telechat3-yarn} into YaRN with {@code attention_factor = 0.07 ln(factor) + 1} as vLLM does
     * (transformers_utils/config.py {@code patch_legacy_rope_type}); vLLM's YaRN uses an explicit attention factor in
     * place of {@code 0.1 ln(factor) + 1}, while the YaRN rotation (extrapolation 1, the YaRN default of
     * llama-context.cpp) multiplies its own magnitude {@code 0.1 ln(factor) + 1} by {@code rope.scaling.attn_factor},
     * so the factor written is the ratio of the two. Nothing is written for a configuration without YaRN or without
     * an explicit attention factor.
     *
     * @throws GgufFormatException when the attention factor is not a number
     * @throws IOException         as the Llama metadata does
     */
    public static void teleChat3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        HfRope.Params rope = ctx.config().rope().global();
        Optional<String> type = rope.type();
        Optional<Object> attention = rope.get("attention_factor");
        OptionalDouble factor = rope.factor();
        if (!type.isPresent() || !"yarn".equals(type.get()) || !attention.isPresent() || !factor.isPresent()) {
            return;
        }
        double attentionFactor = number(ctx.config(), "rope_parameters.attention_factor", attention.get());
        double ggmlMagnitude = RopeMath.yarnGetMscale((float) factor.getAsDouble(), 1.0f);
        kv.set(GgufKeys.ROPE_SCALING_ATTN_FACTOR, attentionFactor / ggmlMagnitude);
    }

    private static double number(HfConfig cfg, String key, Object v) {
        if (!(v instanceof Number)) {
            throw new GgufFormatException(cfg.source() + ": '" + key + "' must be a number, got " + v);
        }
        return ((Number) v).doubleValue();
    }
}

package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The four vLLM concat-fc draft checkpoints the engine runs: {@code EagleMistralLarge3ForCausalLM},
 * {@code EagleLlama4ForCausalLM}, {@code EagleMiniCPMForCausalLM} and {@code EagleCohereForCausalLM}. Each is the
 * {@link com.eonmux.neuralfree.gguf.ops.flow.EagleFcDraftModel} layout of its family:
 *
 * <ul>
 *   <li>{@code EagleMistralLarge3ForCausalLM} (mistral_large_3_eagle.py:36-116, 118-207): DeepSeek multi-head latent
 *       attention over the DeepSeek MLA geometry, a dense FFN, the {@code llama_4_scaling} query temperature, and the
 *       {@code eagle_linear} fusion projection the class renames to {@code fc}. The draft shares the target's
 *       embedding and LM head (the class pretends to load both), so the file usually carries neither.</li>
 *   <li>{@code EagleLlama4ForCausalLM} (llama4_eagle.py:68-185, 189-274): chunked local attention with the Llama 4
 *       weightless Q/K norm on RoPE layers, the NoPE temperature on the rest, and its own {@code lm_head}; the draft
 *       checkpoint's {@code no_rope_layers} decides the single layer's pattern.</li>
 *   <li>{@code EagleMiniCPMForCausalLM} (minicpm_eagle.py:127-212, 215-330, 333-411): the {@code scale_emb}
 *       embedding, the {@code input_norm1} / {@code input_norm2} fusion-part norms, the {@code scale_depth} residual
 *       and the {@code dim_model_base / hidden_size} hidden scale.</li>
 *   <li>{@code EagleCohereForCausalLM} (cohere_eagle.py:44-160): LayerNorm blocks with the parallel residual, the
 *       biased {@code fc} and the {@code logit_scale} of the draft configuration; embeddings and LM head are the
 *       target's.</li>
 * </ul>
 *
 * <p>The draft runs over the target's post-output-norm final hidden state, so {@code target_layers} names the
 * target's last stream ({@link HfEagle3Hooks#target}) and {@code target_hidden_size} the target's hidden size. The
 * tensor names are mapped explicitly, because llama.cpp lists no tensor table for these architectures
 * ({@link HfLoadContext#names()} would refuse them). The Q/K permutation follows each family's RoPE layout: MiniCPM's
 * checkpoints are Hugging Face half-rotated and are permuted, while Llama 4's and Cohere's are already interleaved and
 * are kept as stored. A reduced draft vocabulary ({@code draft_vocab_size} below the target's) and the Llama 4
 * mixture-of-experts layers are refused by name.
 */
public final class HfEagleFcHooks {

    /** The concat-fc families, one per GGUF architecture row. */
    public enum Family {
        MISTRAL_LARGE3, LLAMA4, MINICPM, COHERE
    }

    /** Llama 4's chunked-local attention window when the draft configuration names none. */
    private static final int LLAMA4_CHUNK = 8192;

    /** One layer tensor of a checkpoint, split into its block index and its name inside the block. */
    private static final Pattern LAYER_TENSOR = Pattern.compile("^layers\\.(\\d{1,9})\\.(.+)$");

    /** The layer suffixes that map to the same GGUF base, in both the Llama and the DeepSeek layouts. */
    private static final Map<String, String> LAYER_TENSORS = layerTensors();

    private HfEagleFcHooks() {
    }

    private static Map<String, String> layerTensors() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("input_layernorm.weight", "attn_norm.weight");
        m.put("input_layernorm.bias", "attn_norm.bias");
        m.put("post_attention_layernorm.weight", "ffn_norm.weight");
        m.put("post_attention_layernorm.bias", "ffn_norm.bias");
        m.put("self_attn.q_proj.weight", "attn_q.weight");
        m.put("self_attn.q_proj.bias", "attn_q.bias");
        m.put("self_attn.k_proj.weight", "attn_k.weight");
        m.put("self_attn.k_proj.bias", "attn_k.bias");
        m.put("self_attn.v_proj.weight", "attn_v.weight");
        m.put("self_attn.v_proj.bias", "attn_v.bias");
        m.put("self_attn.o_proj.weight", "attn_output.weight");
        m.put("self_attn.o_proj.bias", "attn_output.bias");
        m.put("self_attn.q_a_proj.weight", "attn_q_a.weight");
        m.put("self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight");
        m.put("self_attn.q_b_proj.weight", "attn_q_b.weight");
        m.put("self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight");
        m.put("self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight");
        m.put("self_attn.q_norm.weight", "attn_q_norm.weight");
        m.put("self_attn.k_norm.weight", "attn_k_norm.weight");
        m.put("mlp.gate_proj.weight", "ffn_gate.weight");
        m.put("mlp.up_proj.weight", "ffn_up.weight");
        m.put("mlp.down_proj.weight", "ffn_down.weight");
        return Collections.unmodifiableMap(m);
    }

    // ------------------------------------------------------------------ entries

    public static void mistralLarge3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan)
            throws IOException {
        meta(ctx, kv, plan, Family.MISTRAL_LARGE3);
    }

    public static void mistralLarge3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        tensors(ctx, hfName, src, out, Family.MISTRAL_LARGE3);
    }

    public static void llama4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan, Family.LLAMA4);
    }

    public static void llama4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        tensors(ctx, hfName, src, out, Family.LLAMA4);
    }

    public static void miniCpmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan, Family.MINICPM);
    }

    public static void miniCpmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        tensors(ctx, hfName, src, out, Family.MINICPM);
    }

    public static void cohereMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        meta(ctx, kv, plan, Family.COHERE);
    }

    public static void cohereTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        tensors(ctx, hfName, src, out, Family.COHERE);
    }

    /** The target's tokenizer: every concat-fc draft shares its target's vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.options().tokenizerDir() != null) {
            return HfLlamaHooks.llamaVocab(ctx, pre, ctx.tokenizerFiles(), ctx.config());
        }
        HfEagle3Hooks.Target target = HfEagle3Hooks.target(ctx);
        ctx.log(ctx.config().architecture() + ": using the tokenizer of the target model " + target.path);
        if (target.config == null) {
            if (target.tokenizerKeys.isEmpty()) {
                throw new GgufFormatException("target '" + target.path + "' holds no tokenizer.* metadata");
            }
            return target.tokenizerKeys;
        }
        Map<String, Object> sizes = new LinkedHashMap<>();
        Object vocabSize = ctx.config().flat().get("vocab_size");
        sizes.put("vocab_size", vocabSize);
        sizes.put("vocab_size_per_layer_input", ctx.config().flat().containsKey("vocab_size_per_layer_input")
                ? ctx.config().flat().get("vocab_size_per_layer_input") : vocabSize);
        return HfLlamaHooks.llamaVocab(ctx, pre, HfTokenizerFiles.locate(target.path),
                target.config.withFlat(sizes));
    }

    // ------------------------------------------------------------------ metadata

    static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan, Family family) throws IOException {
        HfConfig cfg = ctx.config();
        HfEagle3Hooks.Target target = HfEagle3Hooks.target(ctx);
        HfGenericMetadata.apply(ctx, kv);
        int hidden = cfg.requireInt("hidden_size");
        if (target.hiddenSize != hidden) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " fuses a target hidden of "
                    + target.hiddenSize + " floats with its own embeddings of " + hidden);
        }
        int vocab = cfg.requireInt("vocab_size");
        if (target.vocabSize != vocab) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " predicts " + vocab
                    + " tokens; the target vocabulary has " + target.vocabSize);
        }
        kv.set(GgufKeys.VOCAB_SIZE, vocab);
        kv.set(GgufKeys.TARGET_LAYERS, Collections.singletonList(target.numLayers));
        kv.set(GgufKeys.TARGET_HIDDEN_SIZE, target.hiddenSize);
        kv.set(GgufKeys.NORM_BEFORE_RESIDUAL, false);
        kv.set(GgufKeys.NORM_BEFORE_FC, false);
        int heads = cfg.requireInt("num_attention_heads");
        int headDim = cfg.findInt(hidden / heads, "head_dim");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDim);
        double logitScale = cfg.findFloat(1.0, "logit_scale");
        if (logitScale != 1.0) {
            kv.set(GgufKeys.LOGIT_SCALE, (float) logitScale);
        }
        switch (family) {
            case MISTRAL_LARGE3:
                mistralMeta(cfg, kv);
                break;
            case LLAMA4:
                llama4Meta(cfg, kv, plan);
                break;
            case MINICPM:
                miniCpmMeta(cfg, kv);
                break;
            case COHERE:
                cohereMeta(cfg, kv);
                break;
            default:
                throw new IllegalStateException("unhandled concat-fc family " + family);
        }
    }

    /** The DeepSeek MLA geometry and the {@code llama_4_scaling} temperature of the Mistral Large 3 draft. */
    private static void mistralMeta(HfConfig cfg, HfMetadataBuilder kv) {
        int kvLora = cfg.findInt(512, "kv_lora_rank");
        int qkRope = cfg.requireInt("qk_rope_head_dim");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, kvLora + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, kvLora);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, qkNope + qkRope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, vHead);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, qkRope);
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        int qLora = cfg.findInt(0, "q_lora_rank");
        if (qLora > 0) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, qLora);
        }
        Object mscaleAllDim = cfg.rope().raw().get("mscale_all_dim");
        if (mscaleAllDim != null) {
            if (!(mscaleAllDim instanceof Number) || mscaleAllDim instanceof Boolean) {
                throw new GgufFormatException(cfg.source() + ": 'rope_parameters.mscale_all_dim' must be a number");
            }
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1 * ((Number) mscaleAllDim).doubleValue());
        }
        Object scaling = cfg.find("llama_4_scaling").orElse(null);
        if (scaling != null) {
            if (!(scaling instanceof Map)) {
                throw new GgufFormatException(cfg.source() + ": 'llama_4_scaling' must be an object");
            }
            Map<?, ?> s = (Map<?, ?>) scaling;
            Object beta = s.get("beta");
            Object orig = s.get("original_max_position_embeddings");
            if (!(beta instanceof Number)) {
                throw new GgufFormatException(cfg.source() + ": 'llama_4_scaling.beta' must be a number, got "
                        + beta);
            }
            if (!(orig instanceof Long || orig instanceof Integer) || ((Number) orig).longValue() <= 0
                    || ((Number) orig).longValue() > Integer.MAX_VALUE) {
                throw new GgufFormatException(cfg.source()
                        + ": 'llama_4_scaling.original_max_position_embeddings' must be a positive integer, got "
                        + orig);
            }
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, ((Number) beta).doubleValue());
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, ((Number) orig).intValue());
        }
    }

    /**
     * The Llama 4 draft's chunked-local attention, its per-layer RoPE pattern and the weightless Q/K norm. The draft
     * runs one layer, whose entry of {@code no_rope_layers} decides whether it ropes; a {@code yoco} cache-sharing
     * or mixture-of-experts layer has no draft graph.
     */
    private static void llama4Meta(HfConfig cfg, HfMetadataBuilder kv, HfPlanView plan) {
        for (String key : new String[]{"yoco_global_kv_layer", "yoco_local_kv_layer"}) {
            if (cfg.find(key).isPresent()) {
                throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " sets '" + key
                        + "'; the Llama 4 draft has no yoco cache sharing");
            }
        }
        boolean moe = !cfg.findIntList("moe_layers").isEmpty();
        int step = cfg.findInt(0, "interleave_moe_layer_step");
        if (moe || step == 1) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " runs mixture-of-experts"
                    + " layers; the Llama 4 draft runs dense layers only");
        }
        int chunk = cfg.findInt(LLAMA4_CHUNK, "attention_chunk_size");
        if (chunk <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'attention_chunk_size' must be positive, got " + chunk);
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, chunk);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, Collections.singletonList(true));
        kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, Collections.singletonList(ropeLayer(cfg)));
        kv.set(GgufKeys.ATTENTION_USE_QK_NORM, cfg.findBool(true, "use_qk_norm"));
    }

    /** True when the draft's single layer applies RoPE ({@code no_rope_layers[0] != 0}, default true). */
    private static boolean ropeLayer(HfConfig cfg) {
        List<Integer> flags = cfg.findIntList(Collections.singletonList(1), "no_rope_layers");
        return flags.isEmpty() || flags.get(0) != 0;
    }

    /**
     * MiniCPM's {@code scale_emb} embedding, {@code scale_depth / sqrt(mup_denominator)} residual and
     * {@code dim_model_base / hidden_size} hidden scale. The dense SILU MLP is the only FFN the draft graph has.
     */
    private static void miniCpmMeta(HfConfig cfg, HfMetadataBuilder kv) {
        if (cfg.findInt(0, "num_experts") > 0) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " runs "
                    + cfg.findInt(0, "num_experts") + " experts; the MiniCPM draft runs a dense MLP only");
        }
        String act = cfg.findStringOr("silu", "hidden_act");
        if (!"silu".equals(act)) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " uses hidden_act '" + act
                    + "'; the MiniCPM draft graph applies silu");
        }
        double scaleEmb = cfg.findFloat(1.0, "scale_emb");
        int denominator = cfg.findInt(cfg.blockCount(), "mup_denominator");
        double scaleDepth = cfg.findFloat(1.0, "scale_depth");
        double dimBase = cfg.findFloat(cfg.requireInt("hidden_size"), "dim_model_base");
        if (!(scaleEmb > 0.0) || denominator <= 0 || scaleDepth <= 0.0 || !(dimBase > 0.0)) {
            throw new GgufFormatException(cfg.source() + ": invalid MiniCPM scales (scale_emb " + scaleEmb
                    + ", mup_denominator " + denominator + ", scale_depth " + scaleDepth + ", dim_model_base "
                    + dimBase + ")");
        }
        int hidden = cfg.requireInt("hidden_size");
        kv.set(GgufKeys.EMBEDDING_SCALE, (float) scaleEmb);
        kv.set(GgufKeys.RESIDUAL_SCALE, (float) (scaleDepth / Math.sqrt(denominator)));
        kv.set(GgufKeys.OUTPUT_HIDDEN_SCALE, (float) (dimBase / hidden));
    }

    /**
     * The Cohere draft's norm geometry: a draft configuration with {@code rms_norm_eps} uses RMSNorm blocks and a
     * LayerNorm final norm over {@code layer_norm_eps}; without it both are LayerNorm.
     */
    private static void cohereMeta(HfConfig cfg, HfMetadataBuilder kv) {
        boolean rms = cfg.find("rms_norm_eps").isPresent();
        boolean layer = cfg.find("layer_norm_eps").isPresent();
        if (!rms && !layer) {
            throw new GgufFormatException(cfg.source() + ": " + cfg.architecture() + " has neither 'rms_norm_eps'"
                    + " nor 'layer_norm_eps'");
        }
        if (rms) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(1e-6, "rms_norm_eps"));
        }
        if (layer) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, cfg.findFloat(1e-5, "layer_norm_eps"));
        }
    }

    // ------------------------------------------------------------------ tensors

    static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out, Family family)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = stripModel(hfName);
        if (name.startsWith("eagle_layers.")) {
            name = "layers." + name.substring("eagle_layers.".length());
        }
        if (family == Family.MISTRAL_LARGE3) {
            name = stripModel(HfMistralLarge3Hooks.rename(name));
        }
        switch (name) {
            case "fc.weight":
            case "eagle_linear.weight":
                out.add(TensorNames.weight(TensorNames.FC), src);
                return;
            case "fc.bias":
            case "eagle_linear.bias":
                out.add(TensorNames.bias(TensorNames.FC), src);
                return;
            case "norm.weight":
                out.add(TensorNames.weight(TensorNames.OUTPUT_NORM), src);
                return;
            case "norm.bias":
                out.add(TensorNames.bias(TensorNames.OUTPUT_NORM), src);
                return;
            case "embed_tokens.weight":
                out.add(TensorNames.weight(TensorNames.TOKEN_EMBD), src);
                return;
            case "lm_head.weight":
                out.add(TensorNames.weight(TensorNames.OUTPUT), src);
                return;
            case "input_norm1.weight":
                out.add(TensorNames.weight(TensorNames.ENC_INPUT_NORM1), src);
                return;
            case "input_norm2.weight":
                out.add(TensorNames.weight(TensorNames.ENC_INPUT_NORM2), src);
                return;
            case "d2t":
            case "t2d":
                throw new GgufFormatException("tensor '" + hfName + "': a reduced draft vocabulary is not"
                        + " converted; the concat-fc drafter shares the target vocabulary");
            default:
                break;
        }
        if (name.endsWith("rotary_emb.inv_freq") || name.endsWith("rotary_emb.cos_cached")
                || name.endsWith("rotary_emb.sin_cached")) {
            out.drop(hfName, "the rope cache is recomputed at load time");
            return;
        }
        Matcher m = LAYER_TENSOR.matcher(name);
        if (!m.matches()) {
            throw new GgufFormatException("tensor '" + hfName + "' has no concat-fc draft mapping");
        }
        int bid = Integer.parseInt(m.group(1));
        String rest = m.group(2);
        if (rest.equals("self_attn.kv_b_proj.weight")) {
            splitKvB(ctx, hfName, bid, rest, src, out);
            return;
        }
        String base = LAYER_TENSORS.get(rest);
        if (base == null) {
            throw new GgufFormatException("tensor '" + hfName + "' has no concat-fc draft mapping");
        }
        out.add(TensorNames.layerName(bid, base), permuted(ctx, family, rest, src));
    }

    /**
     * The Q/K permutation of the families whose checkpoints store Hugging Face's half-rotated pairs: MiniCPM only.
     * Llama 4's and Cohere's checkpoints are already in the GGUF interleaved layout, and the MLA projections carry no
     * per-head rotary pairs.
     */
    private static RowSource permuted(HfLoadContext ctx, Family family, String rest, RowSource src) {
        if (family != Family.MINICPM) {
            return src;
        }
        boolean q = rest.equals("self_attn.q_proj.weight") || rest.equals("self_attn.q_proj.bias");
        boolean k = rest.equals("self_attn.k_proj.weight") || rest.equals("self_attn.k_proj.bias");
        if (!q && !k) {
            return src;
        }
        HfConfig cfg = ctx.config();
        int nHead = cfg.requireInt("num_attention_heads");
        int nHeadKv = cfg.findInt(nHead, "num_key_value_heads");
        return RowTransforms.permuteQk(src, q ? nHead : nHeadKv, q ? nHead : nHeadKv);
    }

    /**
     * {@code DeepseekV2Model.modify_tensors}' {@code kv_b_proj} split: the {@code [n_head * (qk_nope + v_head),
     * kv_lora]} checkpoint tensor to the absorbed per-head {@code attn_k_b} and {@code attn_v_b} the MLA mixer reads.
     */
    private static void splitKvB(HfLoadContext ctx, String hfName, int bid, String rest, RowSource src,
                                 TensorSink out) {
        HfConfig cfg = ctx.config();
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        int qkNope = cfg.requireInt("qk_nope_head_dim");
        int vHead = cfg.requireInt("v_head_dim");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, qkNope, vHead, Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into " + nHead + " heads of "
                    + qkNope + " key and " + vHead + " value rows: " + e.getMessage(), e);
        }
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K_B)), kvB[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V_B)), kvB[1]);
    }

    /** The name without the {@code model.} prefix the draft's headless checkpoint tree may carry. */
    private static String stripModel(String name) {
        return name.startsWith("model.") ? name.substring("model.".length()) : name;
    }
}

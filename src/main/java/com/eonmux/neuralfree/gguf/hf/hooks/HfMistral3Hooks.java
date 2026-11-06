package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * The hooks of {@code Mistral3Model} (LC/conversion/mistral3.py:21-91) and its three inner implementations
 * ({@code Ministral3Model} :22-32, {@code Mistral3EmbedNemotronModel} :34-44 and {@code Mistral4Model} :46-54), the
 * classes the Ministral/Mistral 3 family and the Nemotron-3 embedding checkpoints convert with, and of
 * {@code VoxtralRealtimeModel} (:93-100). The dispatch follows the {@code __init__} of :59-72: a native
 * {@code mistral4} checkpoint converts on the DeepSeek-V2 hooks ({@link HfDeepseekHooks}), the headless
 * {@code Mistral3Model} and {@code Ministral3Model} root architectures are the non-causal Nemotron embedding branch,
 * and everything else converts as a Ministral 3 on the Llama hooks of {@link HfLlamaHooks}, with its required YaRN
 * log multiplier and attention temperature.
 */
public final class HfMistral3Hooks {

    private static final Logger LOG = LogManager.getLogger(HfMistral3Hooks.class);

    /** The root architectures that resolve to the non-causal Nemotron embedding branch (mistral3.py:62). */
    public static final String MINISTRAL3_HEADLESS = "Ministral3Model";
    /** The second root architecture of the embedding branch (mistral3.py:62). */
    public static final String MISTRAL3_HEADLESS = "Mistral3Model";
    /** The {@code model_type} of Mistral 4, converted on the DeepSeek-V2 hooks (mistral3.py:67). */
    public static final String MODEL_TYPE_MISTRAL4 = "mistral4";
    /** The BPE pre-tokenizer of a {@value #MISTRAL3_HEADLESS} embedding checkpoint (mistral3.py:42-45). */
    public static final String NEMOTRON_EMBED_PRE = "nemotron-3.5-embed";
    /** The {@code model_type} of the Ministral 3 family, which requires YaRN (mistral3.py:28). */
    public static final String MODEL_TYPE_MINISTRAL3 = "ministral3";

    private static final String DOWN_PROJ = ".down_proj";
    private static final String GATE_UP_PROJ = ".gate_up_proj";
    private static final String WEIGHT = ".weight";

    private HfMistral3Hooks() {
    }

    // ------------------------------------------------------------------ architecture

    /**
     * The {@code self.impl} choice of {@code Mistral3Model.__init__} (mistral3.py:59-72):
     * {@link Arch#MISTRAL4} for a {@code mistral4} configuration (:67-68); {@link Arch#MISTRAL3} for an embedding
     * branch checkpoint, whose root architecture must state {@code is_causal: false} (:61-65, raised as
     * {@link GgufFormatException}; the converter's {@code NotImplementedError}); and {@link Arch#MISTRAL3} for the
     * Ministral 3 text checkpoints (:70-72).
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        if (MODEL_TYPE_MISTRAL4.equals(cfg.modelType())) {
            return Arch.MISTRAL4;
        }
        if (isEmbedBranch(cfg)) {
            Object causality = cfg.flat().get("is_causal");
            if (causality == null || causality instanceof Boolean && (Boolean) causality) {
                throw new GgufFormatException("Expected a Nemotron Embed model for " + cfg.architecture()
                        + ", but is_causal is not False");
            }
        }
        return Arch.MISTRAL3;
    }

    /** True for a checkpoint whose root {@code architectures[0]} is one of the headless embedding names. */
    static boolean isEmbedBranch(HfConfig cfg) {
        return MINISTRAL3_HEADLESS.equals(cfg.architecture()) || MISTRAL3_HEADLESS.equals(cfg.architecture());
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The {@code set_gguf_parameters} of the implementation chosen at resolve time. On
     * {@link Arch#MISTRAL4}, {@link HfDeepseekHooks#meta} ({@code Mistral4Model} inherits the DeepseekV2 metadata,
     * mistral3.py:46); on {@link Arch#MISTRAL3}, the Llama metadata of {@link HfLlamaHooks#meta} plus the
     * Ministral 3 keys of {@link #ministral3Kv} and, for the embedding branch, the pooling of
     * {@link HfQwenHooks#pooling} ({@code Mistral3EmbedNemotronModel.set_gguf_parameters}, mistral3.py:42-44).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        if (ctx.arch() == Arch.MISTRAL4) {
            HfDeepseekHooks.meta(ctx, kv, plan);
            return;
        }
        HfLlamaHooks.meta(ctx, kv, plan);
        ministral3Kv(ctx, kv);
        if (isEmbedBranch(ctx.config())) {
            HfQwenHooks.pooling(ctx, kv);
        }
    }

    /**
     * The heads {@code Ministral3Model.set_gguf_parameters} adds (mistral3.py:25-32): for a
     * {@value #MODEL_TYPE_MINISTRAL3} configuration, the YaRN rope parameters are required and of type
     * {@code yarn} (the converter's {@code assert}s), the YaRN log multiplier is the raw
     * {@code mscale_all_dim}, and the attention temperature scale is {@code llama_4_scaling_beta}.
     *
     * @throws GgufFormatException when a required RoPE parameter is missing or of the wrong type
     */
    static void ministral3Kv(HfLoadContext ctx, HfMetadataBuilder kv) {
        HfConfig cfg = ctx.config();
        if (!MODEL_TYPE_MINISTRAL3.equals(cfg.modelType())) {
            return;
        }
        if (cfg.rope().global().raw().isEmpty() || cfg.rope().raw().isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": ministral3 must have 'rope_parameters' config");
        }
        String type = cfg.rope().global().type().orElse("");
        if (!"yarn".equals(type.toLowerCase(Locale.ROOT))) {
            throw new GgufFormatException(cfg.source() + ": ministral3 rope_type must be 'yarn'");
        }
        Object mscale = cfg.rope().global().raw().get("mscale_all_dim");
        if (!(mscale instanceof Number) || mscale instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": required key rope_scaling.mscale_all_dim is missing");
        }
        Object beta = cfg.rope().global().raw().get("llama_4_scaling_beta");
        if (!(beta instanceof Number) || beta instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": required key rope_scaling.llama_4_scaling_beta is"
                    + " missing");
        }
        kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, ((Number) mscale).doubleValue());
        kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, ((Number) beta).doubleValue());
        LOG.info("{}: yarn log multiplier {} and attention temperature scale {}", cfg.architecture(), mscale, beta);
    }

    /**
     * {@code VoxtralRealtimeModel.set_gguf_parameters} (mistral3.py:98-100): the Llama metadata plus
     * {@code realtime_delay_tokens} from {@code default_num_delay_tokens}.
     */
    public static void voxtralMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.REALTIME_DELAY_TOKENS, ctx.config().requireInt("default_num_delay_tokens"));
    }

    // ------------------------------------------------------------------ tensors

    /**
     * The {@code modify_tensors} of the implementation: on {@link Arch#MISTRAL4}, the 3-D routed experts' names get a
     * {@code .weight} suffix (mistral3.py:51-53, {@code merge_expert = False}: the experts arrive stacked) and the
     * DeepseekV2 tensors follow ({@link HfDeepseekHooks}, {@code skip_mtp = False}, mistral3.py:47-49); on
     * {@link Arch#MISTRAL3} the Llama tensors.
     *
     * @throws IOException as the composed hooks do
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (ctx.arch() == Arch.MISTRAL4) {
            if (out.isConsumed(hfName)) {
                return;
            }
            String name = hfName.endsWith(DOWN_PROJ) || hfName.endsWith(GATE_UP_PROJ)
                    ? hfName.endsWith(WEIGHT) ? hfName : hfName + WEIGHT : hfName;
            HfDeepseekHooks.tensors(ctx, name, src, out, false, false);
            return;
        }
        HfLlamaHooks.tensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * The {@code set_vocab} of the implementation. On {@link Arch#MISTRAL4}, {@link HfDeepseekHooks#vocab}; on
     * {@link Arch#MISTRAL3}, the Llama vocabulary of {@link HfLlamaHooks#vocab}, and for the embedding branch the
     * {@code add_<t>_token(False)} of {@code Mistral3EmbedNemotronModel.set_vocab} (mistral3.py:37-40), with the
     * {@value #NEMOTRON_EMBED_PRE} pre-tokenizer of its {@code get_vocab_base_pre} for a {@value #MISTRAL3_HEADLESS}
     * root architecture (mistral3.py:42-45).
     *
     * @throws IOException when a tokenizer file cannot be read
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.arch() == Arch.MISTRAL4) {
            return HfDeepseekHooks.vocab(ctx, pre);
        }
        PreNameResolver names = MISTRAL3_HEADLESS.equals(ctx.config().architecture())
                ? (tj, encode) -> NEMOTRON_EMBED_PRE : pre;
        Map<String, Object> md = HfLlamaHooks.vocab(ctx, names);
        if (isEmbedBranch(ctx.config())) {
            Map<String, Object> out = new LinkedHashMap<>(md);
            out.put(GgufKeys.TOKENIZER_ADD_BOS, false);
            out.put(GgufKeys.TOKENIZER_ADD_EOS, false);
            return Collections.unmodifiableMap(out);
        }
        return md;
    }

    // ------------------------------------------------------------------ extra tensors

    /**
     * The {@code prepare_tensors} checks of the implementation: on {@link Arch#MISTRAL4},
     * {@link HfDeepseekHooks#extra}; on {@link Arch#MISTRAL3} the Llama expert check and the llama3
     * RoPE frequency tensor of {@link HfLlamaHooks#extra}.
     */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        if (ctx.arch() == Arch.MISTRAL4) {
            HfDeepseekHooks.extra(ctx, out);
            return;
        }
        HfLlamaHooks.extra(ctx, out);
    }
}

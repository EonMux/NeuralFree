package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.function.UnaryOperator;

/**
 * Hugging Face Param2MoE conversion (vLLM's {@code param2moe.py}): a fused-QKV, per-head Q/K normed decoder whose
 * first {@code first_k_dense_replace} layers carry a dense SwiGLU FFN and the rest sigmoid-routed experts with one
 * shared expert. The layout is llama.cpp's {@code dots1} (models/dots1.cpp), whose dense lead, biased sigmoid routing,
 * shared expert and NEOX rotary match; the conversion only maps the names, reshapes the router and zero-means its
 * per-expert bias. The graph cannot express grouped routing or a NextN head, which the configuration must not request.
 */
public final class HfParam2MoeHooks {
    /** The router aliases the generic metadata reads the gating function from. */
    private static final String[] SCORE_FUNCTION = {"score_function", "scoring_func", "score_func",
        "moe_router_activation", "moe_router_activation_func", "expert_selection_fn", "router_score_func",
        "router_score_function"};
    private static final String EXPERT_BIAS_SUFFIX = ".mlp.gate.expert_bias.bias";
    private static final String EMBEDDING = "model.embed_tokens.weight";
    private static final String WORD_EMBEDDING = "model.word_embeddings.weight";
    private static final String LM_HEAD = "lm_head.weight";
    private static final String QKV = "attention.query_key_value";
    private static final String WEIGHT = ".weight";

    private HfParam2MoeHooks() {
    }

    /**
     * Validates the variants the dots1 layout cannot carry and fills the aliases the conversion reads.
     *
     * @throws GgufFormatException naming the unsupported field or the missing one
     */
    public static HfConfig fixup(HfConfig cfg) {
        if (!cfg.findBool(false, "use_qk_norm")) {
            throw new GgufFormatException(cfg.source() + ": 'use_qk_norm': Param2MoE requires the per-head Q/K norms"
                    + " the dots1 layout carries");
        }
        if (!"silu".equals(cfg.findStringOr("silu", "hidden_act"))) {
            throw new GgufFormatException(cfg.source() + ": 'hidden_act': the dots1 layout only carries silu");
        }
        if (cfg.findInt(0, "n_group") > 1 || cfg.findInt(0, "topk_group") > 1) {
            throw new GgufFormatException(cfg.source() + ": 'n_group'/'topk_group': grouped expert routing is"
                    + " unsupported");
        }
        if (cfg.findInt(0, "num_nextn_predict_layers") > 0) {
            throw new GgufFormatException(cfg.source() + ": 'num_nextn_predict_layers': Param2MoE NextN heads are"
                    + " unsupported");
        }
        if (cfg.findBool(false, "use_qkv_bias") || cfg.findBool(false, "use_bias")) {
            throw new GgufFormatException(cfg.source() + ": 'use_qkv_bias'/'use_bias': the dots1 layout carries no"
                    + " attention projections bias");
        }
        cfg.requireInt("moe_intermediate_size");
        Map<String, Object> inject = new LinkedHashMap<>();
        inject.put("n_routed_experts", cfg.requireInt("num_experts"));
        inject.put("n_shared_experts", cfg.findInt(1, "num_shared_experts"));
        putIfAbsent(cfg, inject, "first_k_dense_replace", 1);
        putIfAbsent(cfg, inject, "num_experts_per_tok", cfg.findInt("num_experts_per_tok", "num_experts_per_token",
                "top_k_experts").orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'num_experts_per_tok' is missing")));
        if (!cfg.find(SCORE_FUNCTION).isPresent()) {
            inject.put("score_function", "sigmoid");
        }
        return HfMoeClassicHooksB.dots1Fixup(cfg.withFlat(inject));
    }

    private static void putIfAbsent(HfConfig cfg, Map<String, Object> inject, String key, int value) {
        if (!cfg.find(key).isPresent()) {
            inject.put(key, value);
        }
    }

    /**
     * The dots1 metadata (dense lead, shared expert, weight scale and norm), refusing a partial rotary width the
     * graph cannot rotate (models/dots1.cpp asserts the whole head turns).
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int headDim = HfMoeClassicHooksA.ropeWidth(cfg);
        int rotary = (int) Math.round(headDim * partialRotaryFactor(cfg));
        if (rotary != headDim) {
            throw new GgufFormatException(cfg.source() + ": 'partial_rotary_factor': the dots1 graph rotates the whole "
                    + headDim + "-wide head, got a " + rotary + "-wide rotary");
        }
        HfMoeClassicHooksB.dots1Meta(ctx, kv, plan);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDim);
    }

    /** {@code partial_rotary_factor} of the configuration or of {@code rope_parameters}, default 1. */
    private static double partialRotaryFactor(HfConfig cfg) {
        OptionalDouble top = cfg.findFloat("partial_rotary_factor");
        if (top.isPresent()) {
            return top.getAsDouble();
        }
        return cfg.rope().global().partialRotaryFactor().orElse(1.0);
    }

    /**
     * The filter: the router bias gains its {@code .bias} suffix so the name map reaches {@code exp_probs_b.bias}
     * (as the Bailing MoE conversion does). The {@code word_embeddings} and {@code attention.*} names are renamed in
     * {@link #tensors}.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        return HfMoeClassicHooksB.BAILING_V2_FILTER.filter(name, base);
    }

    /**
     * {@code Param2MoEModel.load_weights} (param2moe.py): the shared experts map by their own names, the router bias
     * is zero-meaned (vLLM's {@code _normalize_expert_bias}) and everything else goes through the Qwen2-MoE stacking.
     * A tied checkpoint (the published configuration ties the head) has no {@code lm_head}, so the token table is
     * planned as the {@code output.weight} the dots1 layout requires.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (EMBEDDING.equals(hfName) || WORD_EMBEDDING.equals(hfName)) {
            out.add(ctx.names().mapOrThrow(EMBEDDING), src);
            if (!ctx.index().has(LM_HEAD)) {
                out.add(TensorNames.weight(TensorNames.OUTPUT), src);
            }
            return;
        }
        if (hfName.endsWith(QKV + WEIGHT)) {
            splitFusedQkv(ctx, hfName, src, out);
            return;
        }
        if (hfName.endsWith(EXPERT_BIAS_SUFFIX)) {
            out.add(ctx.names().mapOrThrow(hfName), zeroMean(ctx, hfName, src));
            return;
        }
        if (hfName.contains("shared_experts")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        if (hfName.contains(".attention.")) {
            HfQwenMoeHooks.qwen2MoeTensors(ctx, llmAttentionName(hfName), src, out);
            return;
        }
        HfQwenMoeHooks.qwen2MoeTensors(ctx, hfName, src, out);
    }

    /** The Llama name of a Param2MoE attention tensor (param2moe.py:487-497, the weight-name mapper). */
    private static String llmAttentionName(String hfName) {
        if (hfName.contains(".attention.query_layernorm.")) {
            return hfName.replace(".attention.query_layernorm.", ".self_attn.q_norm.");
        }
        if (hfName.contains(".attention.key_layernorm.")) {
            return hfName.replace(".attention.key_layernorm.", ".self_attn.k_norm.");
        }
        if (hfName.contains(".attention.dense.")) {
            return hfName.replace(".attention.dense.", ".self_attn.o_proj.");
        }
        return hfName.replace(".attention.", ".self_attn.");
    }

    /**
     * Splits the checkpoint's fused {@code [Q | K | V]} weight into the three projections the dots1 layout lists
     * (param2moe.py:100-140, {@code QKVParallelLinear}).
     */
    private static void splitFusedQkv(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfConfig cfg = ctx.config();
        int nHead = cfg.requireInt("num_attention_heads");
        int nKv = cfg.requireInt("num_key_value_heads");
        long headDim = cfg.findInt("head_dim").orElse(cfg.requireInt("hidden_size") / nHead);
        if (headDim <= 0 || nHead <= 0 || nKv <= 0) {
            throw new GgufFormatException(cfg.source() + ": the fused query_key_value needs positive head counts and"
                    + " a positive head_dim");
        }
        RowSource[] qkv = RowTransforms.splitRows(src, nHead * headDim, nKv * headDim, nKv * headDim);
        out.add(ctx.names().mapOrThrow(hfName.replace(QKV, "self_attn.q_proj")), qkv[0]);
        out.add(ctx.names().mapOrThrow(hfName.replace(QKV, "self_attn.k_proj")), qkv[1]);
        out.add(ctx.names().mapOrThrow(hfName.replace(QKV, "self_attn.v_proj")), qkv[2]);
    }

    /**
     * The tensor with its mean subtracted, for {@code t - t.mean()} over the whole router bias.
     *
     * @throws GgufFormatException when the tensor is not a small 1-D row
     */
    private static RowSource zeroMean(HfLoadContext ctx, String hfName, RowSource src) throws IOException {
        RowSource values = ElementTransforms.dequant(src, GGMLType.F32);
        long elements = Math.multiplyExact(values.rows(), values.rowBytes()) / 4;
        if (elements < 1 || elements > HfPlanView.MAX_READ_ELEMENTS) {
            throw new GgufFormatException("tensor '" + hfName + "' has " + elements + " elements; the router bias"
                    + " must hold between 1 and " + HfPlanView.MAX_READ_ELEMENTS);
        }
        ByteBuffer b = ByteBuffer.allocate((int) elements * 4).order(ByteOrder.LITTLE_ENDIAN);
        values.readRows(0, Math.toIntExact(values.rows()), b);
        b.flip();
        double sum = 0;
        for (int i = 0; i < elements; i++) {
            sum += b.getFloat(i * 4);
        }
        float mean = (float) (sum / elements);
        ctx.log("router bias " + hfName + " zero-meaned by " + mean);
        return ElementTransforms.affine(values, 1f, -mean);
    }

    /** The leftovers: every layer whose expert tensors were seen must have been stacked. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfQwenMoeHooks.qwen2MoeExtra(ctx, out);
    }

    /** The GPT-2 BPE vocabulary of the Qwen-style tokenizer. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfQwenMoeHooks.qwen2MoeVocab(ctx, pre);
    }
}

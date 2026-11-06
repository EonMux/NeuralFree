package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * The hooks of llama.cpp's {@code MistralMoeModel} (LC/conversion/mistral.py:113-202), the unregistered DeepSeek-V2
 * subclass for native Mistral MoE checkpoints bound to DEEPSEEK2 by the entry: the MoE-channel hparams remap into
 * DeepseekV2 names ({@code dim}/{@code n_heads}/{@code moe.*}/...), the tensor renames that let the base
 * {@code modify_tensors} serve native checkpoints (the FP8 scales, {@code wkv_b} and the expert parts), the yarn
 * configuration block, the Deepseek metadata with the Mistral temperature length and log multiplier, and the
 * mistral format vocabulary.
 *
 * <p>Every method is a public static hook reference the synthesized entry registers with
 * {@code withUndoPermute(false)} (mistral.py:29); the FP8 scale renames of the index step are
 * {@link HfMistralNativeHooks#index}.</p>
 */
public final class HfMistralMoeHooks {

    private HfMistralMoeHooks() {
    }

    /**
     * {@code model_arch = gguf.MODEL_ARCH.DEEPSEEK2} (mistral.py:114): the native MoE format always DEEPSEEK2.
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.DEEPSEEK2;
    }

    /**
     * {@code MistralMoeModel.__init__} (mistral.py:119-168) extended with the yarn block the class's
     * {@code set_mistral_config} reads (mistral.py:98-107): the top-level and {@code moe} keys remap to their
     * DeepseekV2 names, the defaults fill the missing ones ({@code model_type} transformer, {@code hidden_act} silu,
     * untied embeddings, a 128k context), MoE routing takes {@code topk_method = None},
     * {@code norm_topk_prob = True} and {@code scoring_func = softmax}, and the {@code yarn} block becomes the
     * {@code rope_scaling} the DeepseekV2 metadata reads ({@code rope_type} yarn, the factor, the betas swapped as
     * the converter does and the original context length).
     *
     * @throws GgufFormatException when {@code yarn.factor} or {@code yarn.original_max_position_embeddings} is missing
     */
    public static HfConfig fixup(HfConfig cfg) {
        HfConfig mapped = remap(cfg);
        JsonView root = cfg.raw();
        if (root.raw("yarn") != null) {
            JsonView yarn = root.obj("yarn");
            Map<String, Object> rope = new LinkedHashMap<>();
            rope.put("rope_type", "yarn");
            rope.put("factor", yarn.f64("factor"));
            rope.put("beta_fast", yarn.f64("beta"));
            rope.put("beta_slow", yarn.f64("alpha"));
            rope.put("original_max_position_embeddings", yarn.f64("original_max_position_embeddings"));
            return mapped.withFlat(Collections.singletonMap("rope_parameters", rope));
        }
        return mapped;
    }

    /**
     * The hparam remap of {@code MistralMoeModel.__init__} (mistral.py:129-168) without the yarn bridge: the
     * top-level and {@code moe} keys remap to their DeepseekV2 names, the defaults fill the missing ones, and MoE
     * routing takes {@code topk_method = None}, {@code norm_topk_prob = True} and {@code scoring_func = softmax}.
     * Subclasses that do not read the yarn block, such as Mistral Large 4, compose this step.
     *
     * @throws GgufFormatException when the {@code moe} block is not an object
     */
    public static HfConfig remap(HfConfig cfg) {
        Map<String, Object> fix = new LinkedHashMap<>();
        putIfPresent(fix, "hidden_size", cfg, "dim");
        putIfPresent(fix, "rms_norm_eps", cfg, "norm_eps");
        putIfPresent(fix, "num_key_value_heads", cfg, "n_kv_heads");
        putIfPresent(fix, "num_hidden_layers", cfg, "n_layers");
        putIfPresent(fix, "num_attention_heads", cfg, "n_heads");
        putIfPresent(fix, "intermediate_size", cfg, "hidden_dim");
        putDefault(fix, "model_type", cfg, "model_type", "transformer");
        putDefault(fix, "hidden_act", cfg, "activation", "silu");
        putDefault(fix, "tie_word_embeddings", cfg, "tied_embeddings", false);
        Object maxPos = cfg.flat().get("max_position_embeddings");
        Object maxSeq = cfg.flat().get("max_seq_len");
        fix.put("max_seq_len", maxSeq != null ? maxSeq : maxPos != null ? maxPos : 128_000L);
        fix.put("max_position_embeddings", maxPos != null ? maxPos : 128_000L);
        Object moe = cfg.flat().get("moe");
        if (moe instanceof Map) {
            Map<String, Object> moeCfg = (Map<String, Object>) moe;
            putSub(fix, moeCfg, "moe_layer_freq", "route_every_n");
            putSub(fix, moeCfg, "first_k_dense_replace", "first_k_dense_replace");
            putSub(fix, moeCfg, "num_experts_per_tok", "num_experts_per_tok");
            putSub(fix, moeCfg, "n_routed_experts", "num_experts");
            putSub(fix, moeCfg, "moe_intermediate_size", "expert_hidden_dim");
            putSub(fix, moeCfg, "routed_scaling_factor", "routed_scale");
            putSub(fix, moeCfg, "n_shared_experts", "num_shared_experts");
            putSub(fix, moeCfg, "n_group", "num_expert_groups");
            putSub(fix, moeCfg, "topk_group", "num_expert_groups_per_tok");
        }
        fix.put("topk_method", null);
        fix.put("norm_topk_prob", true);
        fix.put("scoring_func", "softmax");
        return cfg.withFlat(fix);
    }

    private static void putIfPresent(Map<String, Object> fix, String target, HfConfig cfg, String key) {
        Object v = cfg.flat().get(key);
        if (v != null) {
            fix.put(target, v);
        }
    }

    private static void putDefault(Map<String, Object> fix, String target, HfConfig cfg, String key,
                                   Object dflt) {
        fix.put(target, cfg.flat().containsKey(key) && cfg.flat().get(key) != null ? cfg.flat().get(key) : dflt);
    }

    private static void putSub(Map<String, Object> fix, Map<String, Object> moe, String target, String key) {
        Object v = moe.get(key);
        if (v != null) {
            fix.put(target, v);
        }
    }

    /**
     * The tensor renames of {@code filter_tensors} (mistral.py:184-202), raised into the index step so the
     * quantization rewrite sees the renamed scales: {@code *.qscale_act} → {@code *.input_scale},
     * {@code *.qscale_weight} → {@code *.weight_scale}, {@code .wkv_b.} → {@code .kv_b_proj.}, and the expert
     * tensors {@code *.experts.<x>.w1|w2|w3} renamed {@code model.layers.<b>.mlp.experts.<x>.gate_proj|down_proj|
     * up_proj} with the {@code model.} prefix the base names carry.
     *
     * @throws GgufFormatException when a renamed scale collides with a tensor of the checkpoint
     */
    public static HfTensorIndex index(HfLoadContext ctx, HfTensorIndex index) {
        Map<String, RowSource> out = new LinkedHashMap<>();
        boolean renamed = false;
        for (String name : index.names()) {
            String n = name;
            if (n.endsWith(".qscale_act")) {
                n = HfHookSupport.renameTail(n, ".qscale_act", ".input_scale");
            }
            if (n.endsWith(".qscale_weight")) {
                n = HfHookSupport.renameTail(n, ".qscale_weight", ".weight_scale");
            }
            if (n.contains(".wkv_b.")) {
                n = n.replace(".wkv_b.", ".kv_b_proj.");
            }
            if (n.contains(".experts.")) {
                n = "model." + n.replace(".experts.", ".mlp.experts.")
                        .replace(".w1.", ".gate_proj.")
                        .replace(".w2.", ".down_proj.")
                        .replace(".w3.", ".up_proj.");
            }
            if (!n.equals(name)) {
                if (index.has(n)) {
                    throw new GgufFormatException("checkpoint tensors '" + name + "' and '" + n
                            + "' rename onto each other");
                }
                renamed = true;
            }
            out.put(n, index.get(name));
        }
        return renamed ? HfTensorIndex.of(out) : index;
    }

    /**
     * The entry's name filter: {@link HfDeepseekHooks#filter}, the vision-tower drop of the base class filter, which
     * the renamed names now feed directly.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        return HfDeepseekHooks.filter(name, base);
    }

    /**
     * {@code MistralMoeModel.set_gguf_parameters} (mistral.py:173-182): the DeepseekV2 metadata, then the Mistral
     * yarn configuration and the {@code llama_4_scaling.beta} attention temperature scale ({@code set_mistral_config},
     * mistral.py:96-111), the attention temperature length from the yarn block's original context, and the constant
     * yarn log multiplier {@code 0.1}.
     *
     * @throws GgufFormatException when the yarn block lacks a value or the base metadata misses one
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        JsonView root = cfg.raw();
        if (root.raw("yarn") != null) {
            JsonView yarn = root.obj("yarn");
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "yarn");
            kv.set(GgufKeys.ROPE_SCALING_FACTOR, yarn.f64("factor"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, yarn.f64("beta"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, yarn.f64("alpha"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0.1);
            kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, yarn.i64("original_max_position_embeddings"));
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, yarn.i64("original_max_position_embeddings"));
        }
        if (root.raw("llama_4_scaling") != null) {
            kv.set(GgufKeys.ATTENTION_TEMPERATURE_SCALE, root.obj("llama_4_scaling").f64("beta"));
        }
    }

    /** {@code DeepseekV2Model.modify_tensors} over the renamed index names (the base serves native format too). */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        HfDeepseekHooks.tensors(ctx, hfName, src, out, true, true);
    }

    /** {@code DeepseekV2Model.prepare_tensors}: the expert completeness of the stacked experts. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        HfDeepseekHooks.extra(ctx, out);
    }

    /** {@code MistralMoeModel.set_vocab} (mistral.py:170-171): the mistral format vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.mistral(HfHookSupport.vocabInput(ctx), pre);
    }
}

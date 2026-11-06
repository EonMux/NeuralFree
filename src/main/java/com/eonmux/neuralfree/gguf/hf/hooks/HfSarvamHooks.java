package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The hooks of Sarvam's multi-head latent attention models (vLLM
 * {@code vllm/model_executor/models/sarvam.py} {@code SarvamMLAForCausalLM}): a DeepSeek-V2-style MLA trunk with
 * routed experts and one shared expert per MoE layer. The layout, the {@code kv_b_proj} split and the expert
 * stacking are DeepSeek's, so this class composes {@link HfDeepseekHooks} around the class's own
 * configuration defaults and its zero-mean router bias.
 *
 * <p>Sarvam's defaults, which the released checkpoints omit, are written into the configuration before conversion:
 * one expert group ({@code n_group} and {@code topk_group}), top-k renormalization on, one shared expert, a dense
 * first layer, a routing scale of 2.5 and sigmoid routing. The class refuses a shared-expert count of zero (the
 * DeepSeek graph requires the shared branch) and a {@code moe_layer_freq} other than one (the DeepSeek graph
 * places every layer after the dense lead in the MoE).</p>
 */
public final class HfSarvamHooks {

    private static final Logger LOG = LogManager.getLogger(HfSarvamHooks.class);

    private static final double DEFAULT_ROUTED_SCALE = 2.5;

    private HfSarvamHooks() {
    }

    /**
     * {@code SarvamMLAForCausalLM.__init__} and {@code SarvamMLAMoE.__init__} defaults, plus the
     * {@code num_key_value_heads} the {@code kv_b_proj} split needs (the MLA stores one KV head; the trunk has as
     * many key heads as query heads).
     */
    public static HfConfig fixup(HfConfig cfg) {
        Map<String, Object> derived = new LinkedHashMap<>();
        if (!cfg.find("n_group").isPresent()) {
            derived.put("n_group", 1L);
        }
        if (!cfg.find("topk_group").isPresent()) {
            derived.put("topk_group", 1L);
        }
        if (!cfg.find("norm_topk_prob").isPresent()) {
            derived.put("norm_topk_prob", true);
        }
        if (!cfg.find("n_shared_experts").isPresent()) {
            derived.put("n_shared_experts", (long) cfg.findInt(1, "num_shared_experts"));
        }
        if (!cfg.find("first_k_dense_replace").isPresent()) {
            derived.put("first_k_dense_replace", 1L);
        }
        if (!cfg.find("moe_intermediate_size").isPresent() && cfg.find("moe_shared_expert_intermediate_size")
                .isPresent()) {
            derived.put("moe_intermediate_size", cfg.find("moe_shared_expert_intermediate_size").get());
        }
        if (!cfg.find("intermediate_size").isPresent()) {
            derived.put("dense_intermediate_size", 16384L);
        }
        if (!cfg.find("routed_scaling_factor").isPresent()) {
            derived.put("routed_scaling_factor", DEFAULT_ROUTED_SCALE);
        }
        if (!cfg.find("score_function", "scoring_func", "score_func", "moe_router_activation").isPresent()) {
            derived.put("scoring_func", "sigmoid");
        }
        if (!cfg.find("num_key_value_heads").isPresent()) {
            derived.put("num_key_value_heads", (long) cfg.requireInt("num_attention_heads"));
        }
        if (!cfg.find("n_routed_experts").isPresent() && cfg.find("num_experts").isPresent()) {
            derived.put("n_routed_experts", cfg.find("num_experts").get());
        }
        if (!cfg.find("num_experts_per_tok").isPresent() && cfg.find("moe_top_k").isPresent()) {
            derived.put("num_experts_per_tok", cfg.find("moe_top_k").get());
        }
        HfConfig fixed = cfg.withFlat(derived);
        validate(fixed);
        return fixed;
    }

    private static void validate(HfConfig cfg) {
        int moeFreq = cfg.findInt(1, "moe_layer_freq");
        if (moeFreq != 1) {
            throw new GgufFormatException(cfg.source() + ": 'moe_layer_freq' = " + moeFreq
                    + " is not supported: the DeepSeek-V2 MoE layout places every layer after the dense lead"
                    + " in the MoE");
        }
        String act = cfg.findString("hidden_act", "activation_function").orElse("silu");
        if (!"silu".equalsIgnoreCase(act)) {
            throw new GgufFormatException(cfg.source() + ": 'hidden_act' = '" + act
                    + "' is not supported: Sarvam models use the SwiGLU (silu) feed-forward");
        }
        if (cfg.findInt(1, "n_shared_experts") <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'num_shared_experts' = 0 is not supported: the"
                    + " DeepSeek-V2 MoE layout requires the shared-expert branch");
        }
    }

    /**
     * {@code SarvamMLAAttention}'s MLA metadata: the inherited DeepSeek-V2 keys.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfDeepseekHooks.meta(ctx, kv, plan);
    }

    /**
     * {@code SarvamMoEModel.load_weights}' zero-mean normalization (sarvam.py {@code _normalized_weights}) for the
     * router bias, then {@code DeepseekV2Model.modify_tensors}: the bias is centered before it is written, every
     * other tensor follows the DeepSeek layout.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = src;
        if (hfName.endsWith("e_score_correction_bias") || hfName.endsWith("e_score_correction.bias")) {
            if (src.shape().length != 1) {
                throw new GgufFormatException("tensor '" + hfName + "' has shape "
                        + java.util.Arrays.toString(src.shape()) + "; the router bias must be one-dimensional");
            }
            data = ElementTransforms.subtractRowMean(src);
            LOG.debug("{}: centered the router bias", hfName);
        }
        HfDeepseekHooks.tensors(ctx, hfName, data, out);
    }
}

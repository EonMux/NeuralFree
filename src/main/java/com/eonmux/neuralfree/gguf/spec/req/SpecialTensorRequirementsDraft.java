package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerSpan;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.ModelCondition;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.List;

/**
 * Tensor lists of the speculative-decoding drafters eagle3, gemma4-assistant and dflash. Line references are
 * to llama.cpp src/models/.
 *
 * <p>Tensors llama.cpp creates under a metadata condition (eagle3 {@code norm_before_fc}, the dflash LiLiCorr
 * and DSV4 heads, the gemma4-assistant full-attention {@code rope_freqs}) are required when that condition
 * holds; the dflash Markov and selector heads are required when their probe tensor exists. The LiLiCorr norm
 * and MLP blocks are numbered {@code blk.<n_layer_all + i>} past the drafter's own blocks.</p>
 */
public final class SpecialTensorRequirementsDraft {

    /** gemma4-assistant blocks {@code [0, n_layer_nextn)} (gemma4-assistant.cpp:47). */
    private static final LayerCondition ASSISTANT_BLOCK = (hp, il) -> il < hp.nextnLayers();
    /** dflash DeepSeek-V4 (DSpark) stages: hyper-connections enabled (dflash.cpp:248). */
    private static final LayerCondition DFLASH_DSV4 = (hp, il) -> il < hp.nLayer() && hp.dsv4HcMult() > 0;
    /** dflash attention stages without hyper-connections (dflash.cpp:298). */
    private static final LayerCondition DFLASH_PLAIN = (hp, il) -> il < hp.nLayer() && hp.dsv4HcMult() == 0;
    /** dflash stages carrying the DFlash2 convolutions ({@code conv_meta}, dflash.cpp:153 and 324). */
    private static final LayerCondition DFLASH_CONV = (hp, il) -> DFLASH_PLAIN.test(hp, il)
            && hp.dflashConvKernelSize() > 0 && hp.dflashConvGroupSize() > 0;
    /** LiLiCorr attention input projections {@code [0, lilicorr.num_layers)} (dflash.cpp:211-216). */
    private static final LayerCondition LILICORR_LAYER = (hp, il) -> il < hp.lilicorrLayers();
    /** LiLiCorr head present ({@code lilicorr.num_layers > 0}, dflash.cpp:179). */
    private static final ModelCondition LILICORR = hp -> hp.lilicorrLayers() > 0;
    /** LiLiCorr blocks {@code blk.<n_layer_all + i>}, {@code i < lilicorr.num_layers} (dflash.cpp:209-225). */
    private static final LayerCondition LILICORR_BLOCK = (hp, il) -> il >= hp.nLayerAll()
            && il < hp.nLayerAll() + hp.lilicorrLayers();
    /** LiLiCorr feature block {@code blk.<n_layer_all + lilicorr.num_layers>} (dflash.cpp:188, 196-201). */
    private static final LayerCondition LILICORR_FEATURE = (hp, il) -> hp.lilicorrLayers() > 0
            && il == hp.nLayerAll() + hp.lilicorrLayers();
    /** Layer slots up to and including the LiLiCorr feature block. */
    private static final LayerSpan LILICORR_SPAN = hp -> hp.lilicorrLayers() > 0
            ? hp.nLayerAll() + hp.lilicorrLayers() + 1 : 0;
    private static final String BLK = SpecialTensorRequirements.BLK;

    private static final String LILICORR_LAYERS = "lilicorr.layers.%d.";

    private SpecialTensorRequirementsDraft() {
    }

    /** eagle3.cpp:38-101; {@code enc.output_norm.weight} is required when {@code norm_before_fc} is set (:60-62). */
    public static List<TensorRequirement> eagle3() {
        return SpecialTensorRequirements.builder()
                .modelOptional("d2t")
                .model("fc.weight")
                .modelWhen(ModelHparams::normBeforeFc, "enc.output_norm.weight")
                .model("output_norm.weight")
                .modelOptional("output.weight", "token_embd.weight")
                .layer("attn_norm.weight", "attn_norm_2.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .ropeFreqsOptional()
                .build();
    }

    /**
     * gemma4-assistant.cpp:21-78. Only the {@code n_layer_nextn} blocks carry weights;
     * {@code nextn.pre_projection.weight} is created with block 0 and {@code rope_freqs.weight} with the first
     * full-attention block, so each is required when such a block exists.
     */
    public static List<TensorRequirement> gemma4Assistant() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("masked_embd_centroids.weight", "masked_embd_ordering")
                .model("nextn.post_projection.weight")
                .modelWhen(hp -> hp.nextnLayers() > 0, "nextn.pre_projection.weight")
                .layerWhen(ASSISTANT_BLOCK, "attn_norm.weight", "attn_q.weight", "attn_output.weight",
                        "attn_q_norm.weight", "post_attention_norm.weight", "layer_output_scale.weight")
                .modelWhen(SpecialTensorRequirementsDraft::anyFullAttentionAssistantBlock, "rope_freqs.weight")
                .layerWhen(ASSISTANT_BLOCK, "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                        "post_ffw_norm.weight")
                .build();
    }

    /** True when some assistant block uses full (non-sliding-window) attention (gemma4-assistant.cpp:66-69). */
    static boolean anyFullAttentionAssistantBlock(ModelHparams hp) {
        for (int il = 0; il < hp.nextnLayers(); il++) {
            if (!hp.isSwa(il)) {
                return true;
            }
        }
        return false;
    }

    /**
     * dflash.cpp:121-334. The DSpark Markov head is required when {@code markov_w1.weight} exists
     * (dflash.cpp:137-149) and the DFlash2 selector when {@code selector_hidden.weight} exists (:151-176).
     */
    public static List<TensorRequirement> dflash() {
        return SpecialTensorRequirements.builder()
                .modelOptional("token_embd.weight", "d2t")
                .modelOptional("markov_w1.weight")
                .modelWithTrigger("markov_w1.weight", "markov_w2.weight")
                .modelOptional("markov_w2.scale", "conf_proj.weight", "conf_proj.bias")
                .modelOptional("selector_hidden.weight")
                .modelWithTrigger("selector_hidden.weight", "selector_predecessor.weight",
                        "selector_successor.weight")
                .addAll(lilicorrHead())
                .model("fc.weight")
                .modelOptional("fc.scale")
                .model("enc.output_norm.weight", "output_norm.weight")
                .modelOptional("output.weight")
                .addAll(dflashDsv4Stages())
                .addAll(dflashPlainStages())
                .build();
    }

    /**
     * LiLiCorr head (dflash.cpp:179-237), present when {@code lilicorr.num_layers > 0}. The token projection
     * exists only when {@code lilicorr.hidden_size != n_embd}; every {@code .scale} tensor is optional.
     */
    private static List<TensorRequirement> lilicorrHead() {
        return SpecialTensorRequirements.builder()
                .modelWhen(hp -> LILICORR.test(hp) && hp.lilicorrHiddenSize() != hp.nEmbd(),
                        "lilicorr.token_proj.weight", "lilicorr.token_proj.bias")
                .modelOptional("lilicorr.token_proj.scale")
                .modelWhen(LILICORR, "lilicorr.pass_hidden_proj.weight", "lilicorr.pass_hidden_proj.bias")
                .modelOptional("lilicorr.pass_hidden_proj.scale")
                .weightBiasOver(LILICORR_SPAN, BLK, true, LILICORR_FEATURE, "ffn_norm", "ffn_up", "ffn_down")
                .modelWhen(LILICORR, "lilicorr.slot_embedding.weight", "lilicorr.rank_embedding.weight",
                        "lilicorr.attention_bias.bias", "lilicorr.context_proj.weight", "lilicorr.context_proj.bias")
                .modelOptional("lilicorr.context_proj.scale")
                .over(LILICORR_SPAN, BLK, true, LILICORR_BLOCK, "attn_norm.weight", "ffn_norm.weight")
                .weightBias(LILICORR_LAYERS, true, LILICORR_LAYER, "attn.in_proj")
                .weightBiasOver(LILICORR_SPAN, BLK, true, LILICORR_BLOCK, "attn_output", "ffn_up", "ffn_down")
                .over(LILICORR_SPAN, BLK, false, LILICORR_BLOCK, "attn_output.scale", "ffn_up.scale",
                        "ffn_down.scale")
                .modelWhen(LILICORR, "lilicorr.output_norm.weight", "lilicorr.anchor_norm.weight",
                        "lilicorr.factor_input_proj.weight", "lilicorr.factor_input_proj.bias",
                        "lilicorr.out_head.weight", "lilicorr.out_head.bias", "lilicorr.in_head.weight",
                        "lilicorr.in_head.bias", "lilicorr.anchor_out_head.weight", "lilicorr.anchor_out_head.bias")
                .build();
    }

    /** DSpark DSV4 stages (dflash.cpp:248-296); the hyper-connection head is created with them. */
    private static List<TensorRequirement> dflashDsv4Stages() {
        return SpecialTensorRequirements.builder()
                .modelWhen(hp -> hp.dsv4HcMult() > 0, "output_hc_fn.weight", "output_hc_base.weight",
                        "output_hc_scale.weight")
                .layerWhen(DFLASH_DSV4, "attn_norm.weight", "attn_sinks.weight", "attn_q_a.weight",
                        "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight",
                        "attn_output_a.weight", "attn_output_b.weight", "hc_attn_fn.weight", "hc_attn_base.weight",
                        "hc_attn_scale.weight", "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
                        "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_norm.weight", "ffn_gate_exps.weight",
                        "ffn_down_exps.weight", "ffn_up_exps.weight", "ffn_gate_shexp.weight",
                        "ffn_down_shexp.weight", "ffn_up_shexp.weight")
                .build();
    }

    /** Attention stages without hyper-connections (dflash.cpp:298-333). */
    private static List<TensorRequirement> dflashPlainStages() {
        return SpecialTensorRequirements.builder()
                .layerWhen(DFLASH_PLAIN, "attn_norm.weight", "attn_q.weight", "attn_k.weight")
                .layerOptionalWhen(DFLASH_PLAIN, "attn_v.weight")
                .layerWhen(DFLASH_PLAIN, "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layerOptionalWhen(DFLASH_PLAIN, "post_attention_norm.weight", "post_ffw_norm.weight",
                        "layer_output_scale.weight")
                .ropeFreqsOptional()
                .layerOptionalWhen(DFLASH_PLAIN, "attn_sinks.weight")
                .layerWhen(DFLASH_PLAIN, "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .layerWhen(DFLASH_CONV, "attn_conv_base", "attn_conv_proj.weight", "ffn_conv_base",
                        "ffn_conv_proj.weight")
                .build();
    }
}

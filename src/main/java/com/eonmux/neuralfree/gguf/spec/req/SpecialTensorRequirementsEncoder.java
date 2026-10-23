package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerSpan;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.List;

import static com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirements.DEC;
import static com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirements.ENC;

/**
 * Tensor lists of the encoder architectures: the BERT family, modern-bert, neo-bert, eurobert, the
 * EmbeddingGemma models and T5. Line references are to llama.cpp src/models/.
 */
public final class SpecialTensorRequirementsEncoder {

    /** nomic-bert-moe expert layers: {@code moe_every_n_layers > 0 && i % moe_every_n_layers == 1}. */
    private static final LayerCondition NOMIC_MOE = (hp, il) ->
            il < hp.nLayer() && hp.moeEveryN() > 0 && il % hp.moeEveryN() == 1;
    /** nomic-bert-moe dense layers, the complement of {@link #NOMIC_MOE}. */
    private static final LayerCondition NOMIC_DENSE = (hp, il) -> il < hp.nLayer() && !NOMIC_MOE.test(hp, il);
    /** modern-bert encoder layers {@code [1, n_layer - n_layer_decision)}; layer 0 has no attention norm. */
    private static final LayerCondition MODERN_ENC_NOT_FIRST = (hp, il) -> il > 0 && il < modernEncoderLayers(hp);
    /** modern-bert encoder layers {@code [0, n_layer - n_layer_decision)}. */
    private static final LayerCondition MODERN_ENC = (hp, il) -> il < modernEncoderLayers(hp);
    /** modern-bert decision-head layers {@code [n_layer - n_layer_decision, n_layer)}. */
    private static final LayerCondition MODERN_DECISION = (hp, il) -> il >= modernEncoderLayers(hp) && il < hp.nLayer();
    /** T5 decoder layers {@code [0, dec_n_layer)}. */
    private static final LayerCondition T5_DECODER = (hp, il) -> il < hp.decNLayer();
    /** Layer slots of the T5 decoder stack, {@code dec_n_layer}. */
    private static final LayerSpan T5_DECODER_SPAN = ModelHparams::decNLayer;

    private SpecialTensorRequirementsEncoder() {
    }

    private static int modernEncoderLayers(ModelHparams hp) {
        return hp.nLayer() - hp.nLayerDecision();
    }

    /** bert.cpp:23-62. */
    public static List<TensorRequirement> bert() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelOptional("token_types.weight")
                .model("position_embd.weight")
                .modelWeightBiasOptional("cls")
                .modelWeightBiasOptional("cls.output")
                .modelWeightBias("token_embd_norm")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layerWeightBias("attn_output_norm")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.bias")
                .layer("ffn_down.weight")
                .layerOptional("ffn_down.bias")
                .layerWeightBias("layer_output_norm")
                .build();
    }

    /** nomic-bert.cpp:13-46. */
    public static List<TensorRequirement> nomicBert() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelOptional("token_types.weight")
                .modelWeightBias("token_embd_norm")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layerWeightBias("attn_output_norm")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.bias")
                .layer("ffn_down.weight")
                .layerOptional("ffn_down.bias")
                .layer("ffn_gate.weight")
                .layerWeightBias("layer_output_norm")
                .build();
    }

    /** nomic-bert-moe.cpp:14-51; expert layers per nomic-bert-moe.cpp:35. */
    public static List<TensorRequirement> nomicBertMoe() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelOptional("token_types.weight")
                .modelWeightBias("token_embd_norm")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layerWeightBias("attn_output_norm")
                .layerWhen(NOMIC_MOE, "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_inp.weight")
                .layerWhen(NOMIC_DENSE, "ffn_up.weight")
                .layerOptionalWhen(NOMIC_DENSE, "ffn_up.bias")
                .layerWhen(NOMIC_DENSE, "ffn_down.weight")
                .layerOptionalWhen(NOMIC_DENSE, "ffn_down.bias")
                .layerWeightBias("layer_output_norm")
                .build();
    }

    /**
     * jina-bert-v2.cpp:14-61. {@code ffn_up} holds {@code n_ff} or {@code 2 * n_ff} rows (gate packed in up when
     * {@code ffn_gate} is absent, jina-bert-v2.cpp:47-50).
     */
    public static List<TensorRequirement> jinaBertV2() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "token_types.weight")
                .modelWeightBias("token_embd_norm")
                .modelWeightBiasOptional("cls")
                .qkv()
                .layerOptional("attn_q_norm.weight", "attn_q_norm.bias", "attn_k_norm.weight", "attn_k_norm.bias")
                .layerWeightBias("attn_output", "attn_output_norm")
                .layerOptional("attn_norm_2.weight", "attn_norm_2.bias", "ffn_gate.weight")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.bias")
                .layerWeightBias("ffn_down", "layer_output_norm")
                .build();
    }

    /** jina-bert-v3.cpp:13-44. */
    public static List<TensorRequirement> jinaBertV3() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelOptional("token_types.weight")
                .modelWeightBias("token_embd_norm")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layerWeightBias("attn_output_norm")
                .layer("ffn_up.weight")
                .layerOptional("ffn_up.bias")
                .layer("ffn_down.weight")
                .layerOptional("ffn_down.bias")
                .layerWeightBias("layer_output_norm")
                .build();
    }

    /**
     * modern-bert.cpp:61-132. The decision head (modern-bert.cpp:95-131) adds biased pre-norm blocks on the last
     * {@code n_layer_decision} layers and makes {@code token_types.weight}, {@code cls.bias},
     * {@code cls.norm.bias}, {@code cls.weight}, {@code cls.norm.weight} and {@code cls.output.weight/bias}
     * mandatory (modern-bert.cpp:122-131); the token-type count is checked by the hook. Without decision layers the
     * loader returns before {@code token_types}, {@code cls.bias} and {@code cls.norm.bias} are created (:95-97).
     */
    public static List<TensorRequirement> modernBert() {
        String blk = SpecialTensorRequirements.BLK;
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "token_embd_norm.weight", "output_norm.weight")
                .at(blk, false, (hp, il) -> il == 0 && modernEncoderLayers(hp) > 0, "attn_norm.weight")
                .at(blk, true, MODERN_ENC_NOT_FIRST, "attn_norm.weight")
                .at(blk, true, MODERN_ENC, "attn_qkv.weight", "attn_output.weight", "ffn_up.weight",
                        "ffn_down.weight", "ffn_norm.weight")
                .modelWeightBiasOptional("cls.output")
                .modelOptional("cls.weight", "cls.norm.weight")
                .weightBias(blk, true, MODERN_DECISION, "attn_norm", "attn_qkv", "attn_output", "ffn_norm",
                        "ffn_up", "ffn_down")
                .modelWhen(hp -> hp.nLayerDecision() > 0, "token_types.weight", "cls.bias", "cls.norm.bias",
                        "cls.weight", "cls.norm.weight", "cls.output.weight", "cls.output.bias")
                .build();
    }

    /** neo-bert.cpp:11-37. */
    public static List<TensorRequirement> neoBert() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelWeightBiasOptional("cls")
                .modelWeightBiasOptional("cls.output")
                .model("enc.output_norm.weight")
                .layer("attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "ffn_norm.weight",
                        "ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /** eurobert.cpp:11-32. */
    public static List<TensorRequirement> eurobert() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
                .build();
    }

    /** gemma-embedding.cpp:30-67. */
    public static List<TensorRequirement> gemmaEmbedding() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("output.weight", "dense_2.weight", "dense_3.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "post_attention_norm.weight", "attn_k_norm.weight", "attn_q_norm.weight",
                        "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                        "post_ffw_norm.weight")
                .build();
    }

    /** gemma-embedding2.cpp:21-73. */
    public static List<TensorRequirement> gemmaEmbedding2() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight",
                        "output_norm.weight", "output.weight")
                .layer("attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                        "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "post_ffw_norm.weight",
                        "inp_gate.weight", "proj.weight", "post_norm.weight", "layer_output_scale.weight")
                .build();
    }

    /**
     * t5.cpp:38-107: encoder blocks {@code [0, n_layer)}, decoder blocks {@code [0, dec_n_layer)}; the decoder
     * span may exceed {@code n_layer} (t5.cpp:56-60).
     */
    public static List<TensorRequirement> t5() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "enc.output_norm.weight", "dec.output_norm.weight")
                .modelOptional("output.weight")
                .addAll(t5EncoderBlocks())
                .over(T5_DECODER_SPAN, DEC, true, T5_DECODER, "attn_norm.weight")
                .over(T5_DECODER_SPAN, DEC, false, T5_DECODER, "attn_rel_b.weight")
                .over(T5_DECODER_SPAN, DEC, true, T5_DECODER, "attn_q.weight", "attn_k.weight", "attn_v.weight",
                        "attn_o.weight", "cross_attn_norm.weight")
                .over(T5_DECODER_SPAN, DEC, false, T5_DECODER, "cross_attn_rel_b.weight")
                .over(T5_DECODER_SPAN, DEC, true, T5_DECODER, "cross_attn_q.weight", "cross_attn_k.weight",
                        "cross_attn_v.weight", "cross_attn_o.weight", "ffn_norm.weight")
                .over(T5_DECODER_SPAN, DEC, false, T5_DECODER, "ffn_gate.weight")
                .over(T5_DECODER_SPAN, DEC, true, T5_DECODER, "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** t5encoder.cpp:9-40. */
    public static List<TensorRequirement> t5Encoder() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "enc.output_norm.weight")
                .modelOptional("output.weight")
                .addAll(t5EncoderBlocks())
                .build();
    }

    /** Encoder blocks shared by t5.cpp:63-78 and t5encoder.cpp:24-39. */
    private static List<TensorRequirement> t5EncoderBlocks() {
        LayerCondition trunk = TensorRequirement.TRUNK;
        return SpecialTensorRequirements.builder()
                .at(ENC, true, trunk, "attn_norm.weight")
                .at(ENC, false, trunk, "attn_rel_b.weight")
                .at(ENC, true, trunk, "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_o.weight",
                        "ffn_norm.weight")
                .at(ENC, false, trunk, "ffn_gate.weight")
                .at(ENC, true, trunk, "ffn_down.weight", "ffn_up.weight")
                .build();
    }
}

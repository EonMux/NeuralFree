package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;

import java.util.List;

/**
 * Tensor lists of the masked-diffusion (dream, llada, llada-moe, rnd1), looped (hrm_text, nanbeige) and audio
 * (pockettts, wavtokenizer-dec) architectures. Line references are to llama.cpp src/models/.
 */
public final class SpecialTensorRequirementsGen {

    private static final String POSNET = "posnet.%d.";
    private static final String CONVNEXT = "convnext.%d.";

    /** hrm_text physical blocks {@code [0, 2 * layers_per_stack)}: low stack, then high stack (hrm-text.cpp:48-53). */
    private static final LayerCondition HRM_PHYSICAL = (hp, il) -> il < 2 * hp.hrmLayersPerStack();
    /** nanbeige physical blocks; looped slots alias them (nanbeige.cpp:48). */
    private static final LayerCondition NANBEIGE_PHYSICAL = (hp, il) -> il < nanbeigePhysicalLayers(hp);
    /** wavtokenizer-dec posnet residual blocks 0, 1, 3 and 4 (wavtokenizer-dec.cpp:34-50). */
    private static final LayerCondition POSNET_RESNET = (hp, il) ->
            il < hp.posnetLayers() && (il == 0 || il == 1 || il == 3 || il == 4);
    /** wavtokenizer-dec posnet attention block 2 (wavtokenizer-dec.cpp:51-67). */
    private static final LayerCondition POSNET_ATTN = (hp, il) -> il < hp.posnetLayers() && il == 2;
    /** wavtokenizer-dec posnet final norm block 5 (wavtokenizer-dec.cpp:68-72). */
    private static final LayerCondition POSNET_NORM = (hp, il) -> il < hp.posnetLayers() && il == 5;
    /** wavtokenizer-dec ConvNeXt blocks {@code [0, convnext.block_count)} (wavtokenizer-dec.cpp:87). */
    private static final LayerCondition CONVNEXT_BLOCK = (hp, il) -> il < hp.convnextLayers();

    private SpecialTensorRequirementsGen() {
    }

    /**
     * Physical nanbeige blocks: the trunk count before loop expansion. With {@code num_loops > 1} the block
     * count was expanded to {@code n_phys * num_loops} (nanbeige.cpp:20-31).
     */
    static int nanbeigePhysicalLayers(ModelHparams hp) {
        return hp.numLoops() > 1 ? hp.nLayerAll() / hp.numLoops() : hp.nLayer();
    }

    /** dream.cpp:18-46. */
    public static List<TensorRequirement> dream() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("output.weight", "output.bias")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /**
     * llada.cpp:19-60. The optional {@code attn_output.bias} and FFN biases are accepted on load and never read: the
     * graph passes {@code NULL} for them (see {@code UnusedBiases}).
     */
    public static List<TensorRequirement> llada() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("output.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight")
                .layerOptional("attn_output.bias")
                .layer("ffn_norm.weight")
                .ropeFreqsOptional()
                .layer("ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .layerOptional("ffn_gate.bias", "ffn_down.bias", "ffn_up.bias")
                .build();
    }

    /** llada-moe.cpp:16-48; the output head is required. */
    public static List<TensorRequirement> lladaMoe() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight", "output.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ffn_norm.weight",
                        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight")
                .build();
    }

    /** rnd1.cpp:16-58. */
    public static List<TensorRequirement> rnd1() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("output.weight")
                .layer("attn_norm.weight")
                .qkv()
                .layer("attn_output.weight", "attn_k_norm.weight", "attn_q_norm.weight", "ffn_norm.weight",
                        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight")
                .build();
    }

    /**
     * hrm-text.cpp:34-86. The block norms are weightless; the later passes of each stack alias the physical
     * blocks (hrm-text.cpp:60-67).
     */
    public static List<TensorRequirement> hrmText() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelOptional("output.weight")
                .model("hrm.z_l_init")
                .qkv(HRM_PHYSICAL, true)
                .layerWhen(HRM_PHYSICAL, "attn_gate.weight", "attn_output.weight", "ffn_gate.weight", "ffn_down.weight",
                        "ffn_up.weight")
                .build();
    }

    /** nanbeige.cpp:37-74. */
    public static List<TensorRequirement> nanbeige() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight", "output_norm.weight")
                .modelOptional("output.weight")
                .layerWhen(NANBEIGE_PHYSICAL, "attn_norm.weight")
                .qkv(NANBEIGE_PHYSICAL, true)
                .layerWhen(NANBEIGE_PHYSICAL, "attn_output.weight")
                .ropeFreqsOptional()
                .layerWhen(NANBEIGE_PHYSICAL, "ffn_norm.weight", "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** pockettts.cpp:16-42; the Q/K/V projections are created with {@code TENSOR_NOT_REQUIRED}. */
    public static List<TensorRequirement> pocketTts() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelWeightBias("output_norm")
                .layerWeightBias("attn_norm")
                .qkv(TensorRequirement.TRUNK, false)
                .layer("attn_output.weight")
                .layerWeightBias("ffn_norm")
                .layer("ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /** wavtokenizer-dec.cpp:9-112. */
    public static List<TensorRequirement> wavTokenizerDec() {
        return SpecialTensorRequirements.builder()
                .model("token_embd.weight")
                .modelWeightBias("conv1d")
                .weightBias(POSNET, true, POSNET_RESNET, "norm1", "conv1", "norm2", "conv2")
                .weightBias(POSNET, true, POSNET_ATTN, "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output")
                .weightBias(POSNET, true, POSNET_NORM, "attn_norm")
                .modelWeightBias("token_embd_norm")
                .weightBias(CONVNEXT, true, CONVNEXT_BLOCK, "dw", "norm", "pw1", "pw2")
                .at(CONVNEXT, true, CONVNEXT_BLOCK, "gamma.weight")
                .modelWeightBias("output_norm")
                .modelWeightBias("output")
                .build();
    }
}

package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;

import java.util.List;

import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.NEXTN;
import static com.eonmux.neuralfree.gguf.arch.TensorRequirement.TRUNK;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.AFTER_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.DENSE_LEAD;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.and;
import static com.eonmux.neuralfree.gguf.spec.req.MoeTensorRequirementsSupport.builder;

/**
 * Tensor lists of the multi-head latent attention, DeepSeek sparse attention, DeepSeek-V4 and MiniMax sparse
 * attention architectures, referenced from {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeMla}. Each
 * method transcribes {@code load_arch_tensors} of llama.cpp src/models/&lt;file&gt;.cpp (cited
 * {@code models/<file>.cpp:lines}); conventions in {@link MoeTensorRequirementsSupport}.
 */
public final class MoeTensorRequirementsMla {

    /** Low-rank (absorbed-capable) MLA projections of the DeepSeek-V3 family. */
    private static final String[] MLA_LORA = {
        "attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_q_a.weight", "attn_q_b.weight",
        "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight", "ffn_norm.weight"};
    /** DSA lightning-indexer tensors of deepseek32-style layers. */
    private static final String[] DSA_INDEXER = {
        "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight", "indexer.attn_k.weight",
        "indexer.attn_q_b.weight"};
    /** Hyper-connection mixing tensors of each block. */
    private static final String[] HC_BLOCK = {
        "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
        "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"};
    /** Hyper-connection head tensors (model level). */
    private static final String[] HC_HEAD = {"output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"};

    private static final LayerCondition SHEXP_BY_COUNT = (hp, il) -> hp.nExpertShared() > 0;

    private MoeTensorRequirementsMla() {
    }

    /** models/minicpm3.cpp:14-56: non-absorbed MLA with a dense FFN and LongRoPE factors. */
    public static List<TensorRequirement> minicpm3() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_q_a.weight",
                        "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_b.weight", "attn_output.weight",
                        "ffn_norm.weight")
                .denseFfn(TRUNK)
                .modelOptional("rope_factors_long.weight", "rope_factors_short.weight")
                .build();
    }

    /**
     * models/deepseek2.cpp:55-153, shared by mistral4: Q LoRA only when {@code q_lora_rank > 0}; split
     * {@code attn_k_b}/{@code attn_v_b} for MLA files, the joint {@code attn_kv_b} otherwise.
     */
    public static List<TensorRequirement> deepseek2() {
        LayerCondition qLora = (hp, il) -> il < hp.nLayer() && hp.nLoraQ() > 0;
        LayerCondition noQLora = (hp, il) -> il < hp.nLayer() && hp.nLoraQ() == 0;
        LayerCondition mla = (hp, il) -> il < hp.nLayer() && hp.isMla();
        LayerCondition noMla = (hp, il) -> il < hp.nLayer() && !hp.isMla();
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight")
                .when(qLora, "attn_q_a_norm.weight").layer("attn_kv_a_norm.weight")
                .when(qLora, "attn_q_a.weight", "attn_q_b.weight").when(noQLora, "attn_q.weight")
                .layer("attn_kv_a_mqa.weight")
                .when(mla, "attn_k_b.weight", "attn_v_b.weight").when(noMla, "attn_kv_b.weight")
                .layer("attn_output.weight", "ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .probsBiasOptional(AFTER_LEAD).expertsGateUp(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .nextn()
                .build();
    }

    /** models/deepseek2ocr.cpp:23-73: the DeepSeek-V2 MoE with plain multi-head attention. */
    public static List<TensorRequirement> deepseek2ocr() {
        return builder().embdAndOutputNorm().outputOrTied()
                .qkv().layer("attn_output.weight", "ffn_norm.weight", "attn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .probsBiasOptional(AFTER_LEAD).expertsGateUp(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /** models/deepseek32.cpp:46-145: MLA plus a DSA indexer on every layer. */
    public static List<TensorRequirement> deepseek32() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer(MLA_LORA).layer(DSA_INDEXER)
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .nextn()
                .build();
    }

    /**
     * models/deepseek4.cpp:82-181: hyper-connection mixing, grouped low-rank output projection, per-layer KV
     * compressors (ratio != 0) with an indexer on ratio-4 layers, hash routing on the first
     * {@code hash_layer_count} layers and biased routing on the rest.
     */
    public static List<TensorRequirement> deepseek4() {
        LayerCondition compressed = (hp, il) -> il < hp.nLayer() && hp.compressRatio(il) != 0;
        LayerCondition ratio4 = (hp, il) -> il < hp.nLayer() && hp.compressRatio(il) == 4;
        LayerCondition hashed = (hp, il) -> il < hp.nLayer() && il < hp.dsv4HashLayerCount();
        LayerCondition biased = (hp, il) -> il < hp.nLayer() && il >= hp.dsv4HashLayerCount();
        return builder().embdAndOutputNorm().outputRequired().model(HC_HEAD)
                .layer("attn_norm.weight", "attn_sinks.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
                        "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight", "attn_output_a.weight",
                        "attn_output_b.weight")
                .layer(HC_BLOCK)
                .when(compressed, "attn_compressor_kv.weight", "attn_compressor_gate.weight",
                        "attn_compressor_ape.weight", "attn_compressor_norm.weight")
                .when(ratio4, "indexer.proj.weight", "indexer.attn_q_b.weight", "indexer_compressor_kv.weight",
                        "indexer_compressor_gate.weight", "indexer_compressor_ape.weight",
                        "indexer_compressor_norm.weight")
                .layer("ffn_gate_inp.weight")
                .when(hashed, "ffn_gate_tid2eid.weight").when(biased, "exp_probs_b.bias")
                .layerOptional("exp_probs_b_vl.bias")
                .layer("ffn_norm.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight")
                .sharedExperts(TRUNK)
                .nextn()
                .build();
    }

    /** models/glm4-moe.cpp:28-115: optional QK norms, {@code post_attention_norm} as the pre-FFN norm. */
    public static List<TensorRequirement> glm4Moe() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer("attn_norm.weight").qkv().layer("attn_output.weight")
                .layerOptional("attn_q_norm.weight", "attn_k_norm.weight")
                .layer("post_attention_norm.weight")
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias")
                .sharedExperts(and(AFTER_LEAD, SHEXP_BY_COUNT))
                .denseFfn(DENSE_LEAD)
                .nextn()
                .build();
    }

    /** models/glm-dsa.cpp:74-175: deepseek32 layout with every indexer tensor optional (absent on shared layers). */
    public static List<TensorRequirement> glmDsa() {
        return builder().embdAndOutputNorm().outputOrTied()
                .layer(MLA_LORA).layerOptional(DSA_INDEXER)
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .nextn()
                .build();
    }

    /** models/plm.cpp:13-41: MLA without Q LoRA, ungated squared-ReLU FFN, output tied to the embedding. */
    public static List<TensorRequirement> plm() {
        return builder().embdAndOutputNorm()
                .layer("attn_norm.weight", "attn_q.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                        "attn_kv_b.weight", "attn_output.weight", "ffn_norm.weight", "ffn_down.weight", "ffn_up.weight")
                .build();
    }

    /**
     * models/hy-v4.cpp:70-153: MLA with sinks and an output gate, hyper-connections, and DSA indexer tensors on
     * the full-indexer layers of a file with a positive top-k.
     */
    public static List<TensorRequirement> hyV4() {
        LayerCondition indexer = (hp, il) -> il < hp.nLayer() && hp.indexerTopK() > 0 && hp.isIndexerFull(il);
        return builder().embdAndOutputNorm().outputRequired().model(HC_HEAD)
                .layer("attn_norm.weight", "attn_sinks.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
                        "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
                        "attn_v_b.weight", "attn_output.weight", "attn_gate.weight")
                .when(indexer, "indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer.k_norm.weight",
                        "indexer.k_norm.bias", "indexer.proj.weight")
                .layer(HC_BLOCK).layer("ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).probsBiasOptional(AFTER_LEAD).sharedExperts(AFTER_LEAD)
                .build();
    }

    /**
     * models/dots3note.cpp:48-145: MLA with a rope-key norm and per-head output gate; DSA indexer tensors on the
     * full-indexer trunk layers. The NextN blocks, loaded with {@code TENSOR_SKIP | TENSOR_NOT_REQUIRED} (:78), carry
     * the MLA projections and a dense FFN (:117) but no indexer (:107), so their decoder tensors are listed here.
     */
    public static List<TensorRequirement> dots3note() {
        LayerCondition indexer = (hp, il) -> il < hp.nLayer() && hp.isIndexerFull(il);
        return builder().embdAndOutputNorm().outputOrTied()
                .layer(MLA_LORA).layer("attn_k_norm.weight", "attn_gate.weight")
                .when(indexer, DSA_INDEXER)
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias").sharedExperts(AFTER_LEAD)
                .optionalWhen(NEXTN, MLA_LORA)
                .optionalWhen(NEXTN, "attn_k_norm.weight", "attn_gate.weight")
                .denseFfnOptional(NEXTN)
                .nextnOf("eh_proj", "enorm", "hnorm", "embed_tokens", "shared_head_norm")
                .build();
    }

    /** models/minimax-m3.cpp:41-89: MoE layers carry the MiniMax sparse-attention indexer projections. */
    public static List<TensorRequirement> minimaxM3() {
        return builder().embdAndOutputNorm().outputRequired()
                .qkv().layer("attn_output.weight", "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight")
                .layer("ffn_norm.weight")
                .denseFfn(DENSE_LEAD)
                .experts(AFTER_LEAD).when(AFTER_LEAD, "exp_probs_b.bias").sharedExperts(AFTER_LEAD)
                .when(AFTER_LEAD, "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight",
                        "indexer.k_norm.weight")
                .build();
    }
}

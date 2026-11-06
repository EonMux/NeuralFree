package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooksDraft;
import com.eonmux.neuralfree.gguf.spec.req.SpecialTensorRequirementsDraft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Architecture table for the GGUF-side draft heads ({@link Arch#EAGLE} and the four concat-fc families), the rows'
 * runtime being {@link com.eonmux.neuralfree.gguf.ops.flow.Eagle1DraftModel} and
 * {@link com.eonmux.neuralfree.gguf.ops.flow.EagleFcDraftModel} ({@link ArchRegistry} merges this table with the
 * other tables; an architecture appears in exactly one table).
 *
 * <p>The EAGLE-1 drafter is the layout of vLLM's {@code speculative_method "eagle"} heads, served by SGLang's port
 * of SafeAILab EAGLE (sglang srt/models/llama_eagle.py) per target family: the checkpoint's own transformer tree
 * of one decoder layer, fused by {@code fc.weight} over {@code [token embedding ‖ target hidden]}. Following
 * llama_eagle.py:42-61 (cnets.py:427), the llama-shaped draft layer applies no norm around the fc fusion: the
 * attention runs on the fc output directly, the target hidden row is the residual, and the FFN keeps its
 * {@code ffn_norm}; the mistral-shaped checkpoints keep an input norm on the fc output and a final norm
 * (mistral_eagle.py:51-129), and the row reads them when present. There is no d2t: the draft vocabulary is the
 * target's.</p>
 *
 * <p>The same row runs vLLM's DeepSeek-MTP-shaped EAGLE draft ({@code deepseek_eagle.EagleDeepseekV3ForCausalLM},
 * the {@code EagleDeepSeekMTPModel} name): its fusion normalizes each half before {@code fc}
 * ({@code enc.enorm.weight} / {@code enc.hnorm.weight}), its single decoder layer runs DeepSeek's multi-head latent
 * attention when the file carries the MLA ranks, and its FFN is the DeepSeek routed-expert block when the file
 * carries experts ({@code ffn_gate_inp} and the {@code *_exps} tensors).</p>
 *
 * <p>The concat-fc rows are vLLM's further EAGLE drafters: the same fusion projection, applied to the target's
 * <em>post-output-norm</em> final hidden state and the token embedding, followed by the family's standard decoder
 * layers, output norm and LM head. They differ only in the attention, norm and residual geometry their files
 * describe (Mistral Large 3's MLA attention, Llama 4's chunked local attention, MiniCPM's depth-scaled residual,
 * Cohere's parallel LayerNorm block) and in the two scaling keys {@code embedding_scale} and
 * {@code output_hidden_scale}.</p>
 */
public final class ArchTableDraft {

    private ArchTableDraft() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        return Collections.unmodifiableList(Arrays.asList(
                // One decoder layer over the fc fusion, a KV cache of its own, rope like the llama-shaped draft
                // layers, and the draft rows keyed like EAGLE-3's (row offset a = 1).
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE).ropeType(RopeType.NORM).memoryKind(MemoryKind.KV)
                                .support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE, eagleTensors()),
                // vLLM's concat-fc drafters: a fusion projection over [embedding | target hidden], the family's
                // decoder layers, output norm and LM head.
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE_MISTRAL_LARGE3).ropeType(RopeType.NORM)
                                .memoryKind(MemoryKind.KV).graphFamily(Arch.DEEPSEEK2)
                                .support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE_MISTRAL_LARGE3, SpecialTensorRequirementsDraft.eagleFc(true, false)),
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE_LLAMA4).ropeType(RopeType.NORM)
                                .memoryKind(MemoryKind.KV).graphFamily(Arch.LLAMA4)
                                .support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE_LLAMA4, SpecialTensorRequirementsDraft.eagleFc(false, false)),
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE_MINICPM).ropeType(RopeType.NORM)
                                .memoryKind(MemoryKind.KV).residualKind(ResidualKind.SCALED)
                                .support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE_MINICPM, SpecialTensorRequirementsDraft.eagleFc(false, false)),
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE_COHERE).ropeType(RopeType.NORM)
                                .memoryKind(MemoryKind.KV).normKind(NormKind.LAYER)
                                .residualKind(ResidualKind.PARALLEL).support(SupportClass.DRAFT).build(),
                        SpecialHooksDraft.EAGLE_COHERE, SpecialTensorRequirementsDraft.eagleFc(false, true))));
    }

    /**
     * The EAGLE-1 tensors (the fused cnets/llama_eagle layout): {@code fc.weight} and the one decoder layer
     * without an input norm (llama_eagle.py:52-61). Embeddings and the LM head are the drafter's own when present,
     * else the target's (Eagle1DraftModel requires one of the two at bind); {@code output_norm.weight} and the
     * input norms exist only in checkpoints that keep them, and the drafter reads them when it does.
     *
     * <p>A DeepSeek-MTP-shaped file replaces the GQA projections with the MLA ones and the dense FFN with routed
     * experts: the low-rank query path ({@code attn_q_a} → {@code attn_q_a_norm} → {@code attn_q_b}, or
     * {@code attn_q} when the checkpoint has no query LoRA) and the latent KV path ({@code attn_kv_a_mqa} →
     * {@code attn_kv_a_norm}), with the {@code kv_b_proj} decompression stored absorbed ({@code attn_k_b} /
     * {@code attn_v_b}) or decompressed ({@code attn_kv_b}), and the router, routed experts and shared expert of a
     * MoE layer.</p>
     */
    private static List<TensorRequirement> eagleTensors() {
        LayerCondition mla = (hp, il) -> hp.isMla();
        LayerCondition gqa = (hp, il) -> !hp.isMla();
        LayerCondition qLora = (hp, il) -> hp.isMla() && hp.nLoraQ() > 0;
        LayerCondition qDirect = (hp, il) -> hp.isMla() && hp.nLoraQ() == 0;
        LayerCondition moe = (hp, il) -> hp.nExpert() > 0;
        LayerCondition dense = (hp, il) -> hp.nExpert() == 0;
        List<TensorRequirement> out = new ArrayList<>(Arrays.asList(
                TensorRequirement.model("fc.weight"),
                TensorRequirement.modelOptional("token_embd.weight"),
                TensorRequirement.modelOptional("output.weight"),
                TensorRequirement.modelOptional("output_norm.weight"),
                TensorRequirement.modelOptional("enc.output_norm.weight"),
                TensorRequirement.modelOptional("enc.enorm.weight"),
                TensorRequirement.modelOptional("enc.hnorm.weight"),
                TensorRequirement.layerOptional("attn_norm.weight"),
                TensorRequirement.layerOptional("attn_norm_2.weight"),
                TensorRequirement.layer("attn_q.weight", gqa),
                TensorRequirement.layer("attn_k.weight", gqa),
                TensorRequirement.layer("attn_v.weight", gqa),
                TensorRequirement.layer("attn_q_a.weight", qLora),
                TensorRequirement.layer("attn_q_a_norm.weight", qLora),
                TensorRequirement.layer("attn_q_b.weight", qLora),
                TensorRequirement.layer("attn_q.weight", qDirect),
                TensorRequirement.layer("attn_kv_a_mqa.weight", mla),
                TensorRequirement.layer("attn_kv_a_norm.weight", mla),
                TensorRequirement.layerAnyOf(mla, new String[] {"attn_k_b.weight", "attn_v_b.weight"},
                        new String[] {"attn_kv_b.weight"}),
                TensorRequirement.layer("attn_output.weight"),
                TensorRequirement.layer("ffn_norm.weight"),
                TensorRequirement.layer("ffn_gate.weight", dense),
                TensorRequirement.layer("ffn_up.weight", dense),
                TensorRequirement.layer("ffn_down.weight", dense),
                TensorRequirement.layer("ffn_gate_inp.weight", moe),
                TensorRequirement.layer("ffn_gate_exps.weight", moe),
                TensorRequirement.layer("ffn_up_exps.weight", moe),
                TensorRequirement.layer("ffn_down_exps.weight", moe),
                TensorRequirement.layerOptional("ffn_gate_shexp.weight"),
                TensorRequirement.layerOptional("ffn_up_shexp.weight"),
                TensorRequirement.layerOptional("ffn_down_shexp.weight"),
                TensorRequirement.layerOptional("exp_probs_b.bias"),
                TensorRequirement.modelOptional("rope_freqs.weight")));
        return Collections.unmodifiableList(out);
    }
}

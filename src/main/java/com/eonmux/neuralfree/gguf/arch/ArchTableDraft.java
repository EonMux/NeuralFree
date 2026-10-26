package com.eonmux.neuralfree.gguf.arch;

import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.hparams.hooks.SpecialHooks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Architecture table for the EAGLE-1 draft-head architecture ({@link Arch#EAGLE}), the row of the GGUF-side draft
 * heads ({@link ArchRegistry} merges it with the other tables; an architecture appears in exactly one table).
 * Further draft-head rows (the multi-layer EAGLE of SP-5) land here as their runtime arrives.
 *
 * <p>The EAGLE-1 drafter is the layout of vLLM's {@code speculative_method "eagle"} heads, served by SGLang's port
 * of SafeAILab EAGLE (sglang srt/models/llama_eagle.py) per target family: the checkpoint's own transformer tree
 * of one decoder layer, fused by {@code fc.weight} over {@code [token embedding ‖ target hidden]}. Following
 * llama_eagle.py:42-61 (cnets.py:427), the draft layer applies no input norm: the attention runs on the fc output
 * directly, the target hidden row is the residual, and the FFN keeps its {@code ffn_norm}. There is no final norm
 * and no d2t: the draft vocabulary is the target's.</p>
 */
public final class ArchTableDraft {

    private ArchTableDraft() {
    }

    /** The table rows. */
    public static List<ArchEntry> entries() {
        return Collections.singletonList(
                // One GQA decoder layer over the fc fusion, a KV cache of its own, rope like the llama-shaped
                // draft layers, and the draft rows keyed like EAGLE-3's (row offset a = 1, PARITY §5.5.1 S-2).
                ArchEntry.of(ArchTraits.builder(Arch.EAGLE).ropeType(RopeType.NORM).memoryKind(MemoryKind.KV)
                                .support(SupportClass.DRAFT).build(),
                        EAGLE_HPARAMS, eagleTensors()));
    }

    /**
     * EAGLE-1 hparams (llama_eagle.py:64-115, with SGLang's layer changes at llama_eagle.py:42-61): the standard
     * keys plus the EAGLE keys the HfEagle1Hooks writes — exactly one target layer (the target's last, whose row
     * the {@code DrafterFeed} reads), the hidden size the fusion shares with the drafter, and a
     * norm-before-fc/norm-before-residual pair that is always false because the drafter applies no norm around the
     * fc fusion.
     */
    private static final ArchHparamsHook EAGLE_HPARAMS = (b, kv) -> {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));
        int[] targets = kv.i32Array(GgufKeys.TARGET_LAYERS);
        if (targets == null || targets.length != 1) {
            throw SpecialHooks.invalid(b, "EAGLE model requires exactly one entry in '"
                    + kv.key(GgufKeys.TARGET_LAYERS) + "', the target's last hidden layer");
        }
        b.targetLayers(targets);
        int nEmbdTarget = kv.u32Required(GgufKeys.TARGET_HIDDEN_SIZE);
        b.targetHiddenSize(nEmbdTarget);
        b.nEmbdInpEncImpl(targets.length * nEmbdTarget);
        if (nEmbdTarget != b.nEmbd()) {
            throw SpecialHooks.invalid(b, "EAGLE fuses a target hidden of " + nEmbdTarget
                    + " floats with its own embeddings of " + b.nEmbd());
        }
        b.normBeforeResidual(false);
        b.normBeforeFc(false);
    };

    /**
     * The EAGLE-1 tensors (the fused cnets/llama_eagle layout): {@code fc.weight} and the one decoder layer
     * without an input norm (llama_eagle.py:52-61). Embeddings and the LM head are the drafter's own when present,
     * else the target's (Eagle1DraftModel requires one of the two at bind); {@code output_norm.weight} and the
     * input norms exist only in checkpoints that keep them, and the drafter reads the former when it does.
     */
    private static List<TensorRequirement> eagleTensors() {
        List<TensorRequirement> out = new ArrayList<>(Arrays.asList(
                TensorRequirement.model("fc.weight"),
                TensorRequirement.modelOptional("token_embd.weight"),
                TensorRequirement.modelOptional("output.weight"),
                TensorRequirement.modelOptional("output_norm.weight"),
                TensorRequirement.modelOptional("enc.output_norm.weight"),
                TensorRequirement.layerOptional("attn_norm.weight")));
        List<String> layerSuffixes = Arrays.asList("attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm",
                "ffn_gate", "ffn_up", "ffn_down");
        for (String s : layerSuffixes) {
            out.add(TensorRequirement.layer(s + ".weight"));
        }
        out.add(TensorRequirement.layerOptional("attn_norm_2.weight"));
        out.add(TensorRequirement.modelOptional("rope_freqs.weight"));
        return Collections.unmodifiableList(out);
    }
}

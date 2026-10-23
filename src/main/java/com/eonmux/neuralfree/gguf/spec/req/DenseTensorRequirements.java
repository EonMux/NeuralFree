package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.Scope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Shared vocabulary of the tensor lists of the plain decoder attention architectures. The per-architecture
 * lists live in {@link DenseTensorRequirementsClassic}, {@link DenseTensorRequirementsLlama} and
 * {@link DenseTensorRequirementsSandwich}; each list transcribes one llama.cpp
 * {@code llama_model_<arch>::load_arch_tensors} (src/models/&lt;arch&gt;.cpp): a {@code create_tensor} call
 * with flags 0 is a required entry, one with {@code TENSOR_NOT_REQUIRED} an optional entry.
 */
public final class DenseTensorRequirements {

    /** Every block, trunk and NextN ({@code for (int i = 0; i < n_layer_all; ++i)} loops). */
    static final LayerCondition ALL_LAYERS = (hp, il) -> il < hp.nLayerAll();
    /** Trunk layers of an architecture without experts ({@code n_expert == 0} branches). */
    static final LayerCondition DENSE_FFN = (hp, il) -> il < hp.nLayer() && hp.nExpert() == 0;
    /** Trunk layers of an architecture with experts ({@code n_expert > 0} branches). */
    static final LayerCondition MOE_FFN = (hp, il) -> il < hp.nLayer() && hp.nExpert() > 0;
    /** MoE trunk layers with a shared expert ({@code n_ff_shexp > 0}). */
    static final LayerCondition MOE_SHARED_FFN = (hp, il) -> il < hp.nLayer() && hp.nExpert() > 0 && hp.nFfShexp() > 0;

    private static final String LAYER_PREFIX = "blk.%d.";

    private DenseTensorRequirements() {
    }

    /** Starts an empty list. */
    static Builder builder() {
        return new Builder();
    }

    /** Accumulates requirements in load order; {@link #build()} returns an unmodifiable copy. */
    static final class Builder {
        private final List<TensorRequirement> out = new ArrayList<>();

        private Builder() {
        }

        /** Required model-level tensors. */
        Builder model(String... names) {
            for (String n : names) {
                out.add(TensorRequirement.model(n));
            }
            return this;
        }

        /** Optional model-level tensors. */
        Builder modelOptional(String... names) {
            for (String n : names) {
                out.add(TensorRequirement.modelOptional(n));
            }
            return this;
        }

        /** {@code token_embd.weight}, required by every architecture of this lane except codeshell. */
        Builder tokenEmbd() {
            return model("token_embd.weight");
        }

        /**
         * Untied output head: {@code output.weight} with {@code TENSOR_NOT_REQUIRED}, falling back to the
         * duplicated {@code token_embd.weight} (the common {@code if (output == NULL)} block).
         */
        Builder outputOrTied() {
            return modelOptional("output.weight");
        }

        /**
         * Per-layer RoPE frequency factors of the loaders that branch on the training-time scaling
         * ({@link LoaderHelpers#ropeFactorsByScaling()}).
         */
        Builder ropeFactors() {
            out.addAll(LoaderHelpers.ropeFactorsByScaling());
            return this;
        }

        /** The optional {@code rope_freqs} of the loaders that create no LONGROPE factors. */
        Builder ropeFreqs() {
            out.add(LoaderHelpers.ropeFreqs());
            return this;
        }

        /** Required {@code blk.%d.<suffix>} tensors on every trunk layer. */
        Builder layer(String... suffixes) {
            return layerWhen(TensorRequirement.TRUNK, suffixes);
        }

        /** Optional {@code blk.%d.<suffix>} tensors on every trunk layer. */
        Builder layerOptional(String... suffixes) {
            return layerOptionalWhen(TensorRequirement.TRUNK, suffixes);
        }

        /** Required {@code blk.%d.<suffix>} tensors on the layers selected by {@code when}. */
        Builder layerWhen(LayerCondition when, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.layer(s, when));
            }
            return this;
        }

        /** Optional {@code blk.%d.<suffix>} tensors on the layers selected by {@code when}. */
        Builder layerOptionalWhen(LayerCondition when, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.of(Scope.LAYER, false, when, new String[][] {{LAYER_PREFIX + s}}));
            }
            return this;
        }

        /**
         * {@code blk.%d.<suffix>} tensors required on trunk layers where {@code blk.%d.<trigger>} exists
         * (tensors created inside a block conditioned on another tensor).
         */
        Builder layerWithTrigger(String trigger, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.layer(s).ifPresent(LAYER_PREFIX + trigger));
            }
            return this;
        }

        /** Required layer entry satisfied by any one group of {@code blk.%d.<suffix>} names. */
        Builder layerAnyOf(LayerCondition when, String[]... groups) {
            out.add(TensorRequirement.layerAnyOf(when, groups));
            return this;
        }

        /** {@code create_tensor_qkv} with flags 0 on trunk layers ({@link LoaderHelpers#qkv}). */
        Builder qkv() {
            return qkvWhen(TensorRequirement.TRUNK);
        }

        /**
         * {@code create_tensor_qkv} with flags 0 on the layers selected by {@code when} ({@link LoaderHelpers#qkv}).
         */
        Builder qkvWhen(LayerCondition when) {
            out.addAll(LoaderHelpers.qkv(LAYER_PREFIX, when, true));
            return this;
        }

        /**
         * {@code create_tensor_qkv} with {@code TENSOR_NOT_REQUIRED} or {@code TENSOR_SKIP}
         * ({@link LoaderHelpers#qkv}): every projection optional; on NextN blocks loaded with {@code TENSOR_SKIP}
         * every name is created.
         */
        Builder qkvOptionalWhen(LayerCondition when) {
            out.addAll(LoaderHelpers.qkv(LAYER_PREFIX, when, false));
            return this;
        }

        /** Gated feed-forward weights {@code ffn_gate}, {@code ffn_down} and {@code ffn_up} on trunk layers. */
        Builder gatedFfn() {
            return layer("ffn_gate.weight", "ffn_down.weight", "ffn_up.weight");
        }

        /**
         * The llama-style FFN branch shared by refact, minicpm, granite and mistral3: dense
         * {@code ffn_gate/down/up} plus optional biases when {@code n_expert == 0}; otherwise the router
         * {@code ffn_gate_inp}, optional {@code ffn_gate_exps}, required {@code ffn_down_exps}/{@code ffn_up_exps},
         * and the shared expert when {@code n_ff_shexp > 0} (refact.cpp:50-71).
         */
        Builder llamaFfnOrExperts() {
            layerWhen(DENSE_FFN, "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight");
            layerOptionalWhen(DENSE_FFN, "ffn_gate.bias", "ffn_down.bias", "ffn_up.bias");
            layerWhen(MOE_FFN, "ffn_gate_inp.weight");
            layerOptionalWhen(MOE_FFN, "ffn_gate_exps.weight");
            layerWhen(MOE_FFN, "ffn_down_exps.weight", "ffn_up_exps.weight");
            return layerWhen(MOE_SHARED_FFN, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight");
        }

        /** The accumulated list, unmodifiable. */
        List<TensorRequirement> build() {
            return Collections.unmodifiableList(new ArrayList<>(out));
        }
    }
}

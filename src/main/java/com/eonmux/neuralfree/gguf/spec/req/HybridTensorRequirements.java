package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.ModelCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.Scope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Shared vocabulary of the tensor lists of the recurrent and hybrid architectures. The tensor-presence facts their
 * graphs branch on are decided by the spec resolvers ({@code RecurrentResolver}, {@code MoeResolver},
 * {@code MlaResolver}). The per-architecture lists live in {@link HybridTensorRequirementsSsm},
 * {@link HybridTensorRequirementsRwkv}, {@link HybridTensorRequirementsGdn} and
 * {@link HybridTensorRequirementsKda}; each transcribes one llama.cpp
 * {@code llama_model_<arch>::load_arch_tensors} (src/models/&lt;arch&gt;.cpp): a {@code create_tensor} call
 * with flags 0 is a required entry, one with {@code TENSOR_NOT_REQUIRED} an optional entry.
 *
 * <p>NextN (MTP) blocks are loaded with {@code nextn_flags().mtp}, which is {@code TENSOR_NOT_REQUIRED} for a
 * file without the first NextN block and carries {@code TENSOR_SKIP} unless MTP is loaded
 * (llama-model.cpp:3901-3920). Those flags depend on the file (a trunk-only file has no NextN blocks), so every NextN
 * entry is optional; the NextN graph checks its own inputs.
 */
public final class HybridTensorRequirements {

    /** Every layer the phase covers. */
    static final LayerCondition ALL = (hp, il) -> true;
    /** Attention (non-recurrent) layers: the {@code !hparams.is_recr(il)} branches. */
    static final LayerCondition ATTENTION = (hp, il) -> !hp.isRecurrent(il);
    /** Recurrent layers: the {@code hparams.is_recr(il)} branches. */
    static final LayerCondition RECURRENT = (hp, il) -> hp.isRecurrent(il);
    /** Leading dense layers: {@code il < n_layer_dense_lead}. */
    static final LayerCondition DENSE_LEAD = (hp, il) -> il < hp.nLayerDenseLead();
    /** Layers after the leading dense ones: {@code il >= n_layer_dense_lead}. */
    static final LayerCondition AFTER_DENSE_LEAD = (hp, il) -> il >= hp.nLayerDenseLead();

    private static final String LAYER_PREFIX = "blk.%d.";

    private HybridTensorRequirements() {
    }

    /** Starts an empty list in the trunk phase. */
    static Builder builder() {
        return new Builder();
    }

    /** Both conditions hold. */
    static LayerCondition and(LayerCondition a, LayerCondition b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return (hp, il) -> a.test(hp, il) && b.test(hp, il);
    }

    /** The condition does not hold. */
    static LayerCondition not(LayerCondition a) {
        Objects.requireNonNull(a, "a");
        return (hp, il) -> !a.test(hp, il);
    }

    /**
     * Accumulates requirements in load order; {@link #build()} returns an unmodifiable copy. Layer entries
     * are scoped to the current phase: {@link #trunk()} (layers {@code [0, n_layer)}, entries keep their
     * requiredness) or {@link #nextn()} (layers {@code [n_layer, n_layer_all)}, every entry optional).
     */
    static final class Builder {
        private final List<TensorRequirement> out = new ArrayList<>();
        private LayerCondition phase = TensorRequirement.TRUNK;
        private boolean phaseRequired = true;

        private Builder() {
        }

        /** Switches to trunk layers. */
        Builder trunk() {
            phase = TensorRequirement.TRUNK;
            phaseRequired = true;
            return this;
        }

        /** Switches to NextN layers, whose tensors are all optional (see the class comment). */
        Builder nextn() {
            phase = TensorRequirement.NEXTN;
            phaseRequired = false;
            return this;
        }

        /** Required model-level tensors (full names). */
        Builder model(String... names) {
            for (String n : names) {
                out.add(TensorRequirement.model(n));
            }
            return this;
        }

        /**
         * Model-level tensors (full names) created with the trunk flags of {@code nextn_flags}
         * (llama-model.cpp:3901-3920): required unless the file holds only the MTP blocks.
         */
        Builder modelTrunk(String... names) {
            for (String n : names) {
                out.add(TensorRequirement.model(n).withTrunkFlags());
            }
            return this;
        }

        /** Model-level tensors (full names) required when {@code cond} holds (a loader's {@code if} on hparams). */
        Builder modelWhen(ModelCondition cond, String... names) {
            for (String n : names) {
                out.add(TensorRequirement.modelWhen(cond, n));
            }
            return this;
        }

        /** Optional model-level tensors (full names). */
        Builder modelOptional(String... names) {
            for (String n : names) {
                out.add(TensorRequirement.modelOptional(n));
            }
            return this;
        }

        /**
         * Untied output head: optional {@code output.weight}, falling back to the duplicated
         * {@code token_embd.weight} (the common {@code if (output == NULL)} block).
         */
        Builder outputOrTied() {
            return modelOptional("output.weight");
        }

        /** {@code blk.%d.<suffix>} tensors on the phase's layers selected by {@code when}, required in the trunk. */
        Builder layer(LayerCondition when, String... suffixes) {
            return add(phaseRequired, when, suffixes);
        }

        /** Optional {@code blk.%d.<suffix>} tensors on the phase's layers selected by {@code when}. */
        Builder layerOptional(LayerCondition when, String... suffixes) {
            return add(false, when, suffixes);
        }

        /**
         * One entry satisfied by any one group of {@code blk.%d.<suffix>} names, required in the trunk, with every
         * group created (each name loaded with {@code TENSOR_NOT_REQUIRED}).
         */
        Builder layerAnyOf(LayerCondition when, String[]... groups) {
            out.add(TensorRequirement.of(Scope.LAYER, phaseRequired, scoped(when),
                    TensorRequirement.prefixed(LAYER_PREFIX, groups)));
            return this;
        }

        /**
         * One entry over probed groups of {@code blk.%d.<suffix>} names, required in the trunk: the leading tensor
         * of each group is created with {@code TENSOR_NOT_REQUIRED} and only the first group whose leading tensor
         * exists is loaded, else the last group ({@link TensorRequirement#probed()}).
         */
        Builder layerFirstOf(LayerCondition when, String[]... groups) {
            out.add(TensorRequirement.of(Scope.LAYER, phaseRequired, scoped(when),
                    TensorRequirement.prefixed(LAYER_PREFIX, groups)).probed());
            return this;
        }

        /**
         * {@code create_tensor_qkv} called with the phase's block flags ({@link LoaderHelpers#qkv}): required in the
         * trunk; on NextN blocks the flags carry {@code TENSOR_SKIP}, which creates every name.
         */
        Builder qkv(LayerCondition when) {
            out.addAll(LoaderHelpers.qkv(LAYER_PREFIX, scoped(when), phaseRequired));
            return this;
        }

        /**
         * {@code create_tensor_gate_up_exps} called with the phase's block flags ({@link LoaderHelpers#gateUpExps}).
         */
        Builder gateUpExps(LayerCondition when) {
            out.add(LoaderHelpers.gateUpExps(LAYER_PREFIX, scoped(when), phaseRequired));
            return this;
        }

        /** Gated dense FFN {@code ffn_gate}, {@code ffn_down}, {@code ffn_up}. */
        Builder gatedFfn(LayerCondition when) {
            return layer(when, "ffn_gate.weight", "ffn_down.weight", "ffn_up.weight");
        }

        /**
         * The NextN input fusion of the Qwen and GLM MTP blocks: {@code nextn.eh_proj}, {@code nextn.enorm},
         * {@code nextn.hnorm}, with optional {@code nextn.embed_tokens}, {@code nextn.shared_head_head} and
         * {@code nextn.shared_head_norm} (qwen3next.cpp:115-120).
         */
        Builder nextnFusion() {
            layer(ALL, "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight");
            return layerOptional(ALL, "nextn.embed_tokens.weight", "nextn.shared_head_head.weight",
                    "nextn.shared_head_norm.weight");
        }

        /** The accumulated list, unmodifiable. */
        List<TensorRequirement> build() {
            return Collections.unmodifiableList(new ArrayList<>(out));
        }

        private Builder add(boolean required, LayerCondition when, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.of(Scope.LAYER, required, scoped(when),
                        new String[][] {{LAYER_PREFIX + s}}));
            }
            return this;
        }

        private LayerCondition scoped(LayerCondition when) {
            return and(phase, when);
        }
    }
}

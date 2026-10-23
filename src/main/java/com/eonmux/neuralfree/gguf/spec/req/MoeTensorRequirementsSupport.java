package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.ModelCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.Scope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Shared vocabulary of the MoE, MLA and DSA tensor lists ({@link MoeTensorRequirements},
 * {@link MoeTensorRequirementsMla}, {@link MoeTensorRequirementsSwa}). Each list transcribes one llama.cpp
 * {@code llama_model_<arch>::load_arch_tensors} (src/models/&lt;file&gt;.cpp): a {@code create_tensor} call with
 * flags 0 is a required entry, one with {@code TENSOR_NOT_REQUIRED} an optional entry, and the loader's
 * {@code if} conditions become {@link LayerCondition}s evaluated on the loaded hparams.
 *
 * <p>NextN/MTP blocks ({@code [n_layer, n_layer_all)}) are loaded with {@code nextn_flags}
 * (llama-model.cpp:3901-3920): without {@code load_mtp} every tensor of those blocks is
 * {@code TENSOR_SKIP}, so the lists only name the NextN-specific tensors, all optional. A NextN block whose
 * loader body is the trunk body repeats the trunk tensors, which are not listed again; a list whose NextN blocks
 * differ from a trunk block lists their decoder tensors for {@code NEXTN}: exaone-moe and dots3note, whose NextN
 * blocks carry a dense FFN beside a routed trunk.
 */
final class MoeTensorRequirementsSupport {

    /** Trunk layers before the leading dense block count ({@code i < n_layer_dense_lead}). */
    static final LayerCondition DENSE_LEAD = (hp, il) -> il < hp.nLayer() && il < hp.nLayerDenseLead();
    /** Trunk layers at or after the leading dense block count ({@code i >= n_layer_dense_lead}). */
    static final LayerCondition AFTER_LEAD = (hp, il) -> il < hp.nLayer() && il >= hp.nLayerDenseLead();
    /** Trunk layers of a file without experts ({@code n_expert == 0} branches). */
    static final LayerCondition NO_EXPERTS = (hp, il) -> il < hp.nLayer() && hp.nExpert() == 0;
    /** Trunk layers of a file with experts ({@code n_expert > 0} branches). */
    static final LayerCondition WITH_EXPERTS = (hp, il) -> il < hp.nLayer() && hp.nExpert() > 0;
    /** Trunk layers whose FFN is routed ({@link com.eonmux.neuralfree.gguf.hparams.ModelHparams#isMoeLayer}). */
    static final LayerCondition MOE_LAYER = (hp, il) -> il < hp.nLayer() && hp.isMoeLayer(il);
    /** Trunk layers whose FFN is dense. */
    static final LayerCondition DENSE_LAYER = (hp, il) -> il < hp.nLayer() && !hp.isMoeLayer(il);

    private static final String LAYER_PREFIX = "blk.%d.";

    private static final String[] EXPS = {"ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight"};
    private static final String[] SHEXP = {"ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight"};
    private static final String[] DENSE = {"ffn_gate.weight", "ffn_down.weight", "ffn_up.weight"};

    private MoeTensorRequirementsSupport() {
    }

    /** Both conditions hold. */
    static LayerCondition and(LayerCondition a, LayerCondition b) {
        return (hp, il) -> a.test(hp, il) && b.test(hp, il);
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

        /** Model-level tensors required when {@code cond} holds (a loader's {@code if} on hparams). */
        Builder modelWhen(ModelCondition cond, String... names) {
            for (String n : names) {
                out.add(TensorRequirement.modelWhen(cond, n));
            }
            return this;
        }

        /** {@code token_embd.weight} and {@code output_norm.weight}, created with flags 0 by every loader here. */
        Builder embdAndOutputNorm() {
            return model("token_embd.weight", "output_norm.weight");
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

        /** Required untied {@code output.weight}. */
        Builder outputRequired() {
            return model("output.weight");
        }

        /**
         * {@code output.weight} with {@code TENSOR_NOT_REQUIRED}, falling back to the duplicated
         * {@code token_embd.weight} (the common {@code if (output == NULL)} block).
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

        /** Required {@code blk.%d.<suffix>} tensors on every trunk layer. */
        Builder layer(String... suffixes) {
            return when(TensorRequirement.TRUNK, suffixes);
        }

        /** Optional {@code blk.%d.<suffix>} tensors on every trunk layer. */
        Builder layerOptional(String... suffixes) {
            return optionalWhen(TensorRequirement.TRUNK, suffixes);
        }

        /** Required {@code blk.%d.<suffix>} tensors on the layers selected by {@code cond}. */
        Builder when(LayerCondition cond, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.layer(s, cond));
            }
            return this;
        }

        /** Optional {@code blk.%d.<suffix>} tensors on the layers selected by {@code cond}. */
        Builder optionalWhen(LayerCondition cond, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.of(Scope.LAYER, false, cond, new String[][] {{LAYER_PREFIX + s}}));
            }
            return this;
        }

        /**
         * {@code blk.%d.<suffix>} tensors required on the layers selected by {@code cond} where
         * {@code blk.%d.<trigger>} exists (tensors created inside {@code if (layer.<trigger>)}).
         */
        Builder withTrigger(String trigger, LayerCondition cond, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.layer(s, cond).ifPresent(LAYER_PREFIX + trigger));
            }
            return this;
        }

        /**
         * Optional {@code blk.%d.<suffix>} tensors on the layers selected by {@code cond} where
         * {@code blk.%d.<trigger>} does not exist (tensors created inside {@code if (!layer.<trigger>)}).
         */
        Builder optionalWithout(String trigger, LayerCondition cond, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.of(Scope.LAYER, false, cond, new String[][] {{LAYER_PREFIX + s}})
                        .ifAbsent(LAYER_PREFIX + trigger));
            }
            return this;
        }

        /** Required layer entry satisfied by any one group of {@code blk.%d.<suffix>} names, each group created. */
        Builder anyOf(LayerCondition cond, String[]... groups) {
            out.add(TensorRequirement.layerAnyOf(cond, groups));
            return this;
        }

        /**
         * Required layer entry over probed alternatives ({@link TensorRequirement#probed()}): the loader creates the
         * leading tensor of each group with {@code TENSOR_NOT_REQUIRED} and loads the first group whose leading
         * tensor exists, else the last group.
         */
        Builder firstOf(LayerCondition cond, String[]... groups) {
            out.add(TensorRequirement.layerAnyOf(cond, groups).probed());
            return this;
        }

        /**
         * {@code create_tensor_qkv} with flags 0 (llama-model.cpp:3863-3898) on trunk layers
         * ({@link LoaderHelpers#qkv}).
         */
        Builder qkv() {
            out.addAll(LoaderHelpers.qkv(LAYER_PREFIX, TensorRequirement.TRUNK, true));
            return this;
        }

        /** Gated dense FFN {@code ffn_gate/down/up.weight}, required on the layers selected by {@code cond}. */
        Builder denseFfn(LayerCondition cond) {
            return when(cond, DENSE);
        }

        /** Dense FFN created with {@code TENSOR_NOT_REQUIRED}. */
        Builder denseFfnOptional(LayerCondition cond) {
            return optionalWhen(cond, DENSE);
        }

        /** {@code ffn_gate_inp} with separate {@code ffn_gate/down/up_exps}, all required. */
        Builder experts(LayerCondition cond) {
            when(cond, "ffn_gate_inp.weight");
            return when(cond, EXPS);
        }

        /**
         * {@code ffn_gate_inp}, {@code ffn_down_exps} and {@code create_tensor_gate_up_exps} with flags 0
         * (llama-model.cpp:3846-3861, {@link LoaderHelpers#gateUpExps}).
         */
        Builder expertsGateUp(LayerCondition cond) {
            when(cond, "ffn_gate_inp.weight", "ffn_down_exps.weight");
            out.add(LoaderHelpers.gateUpExps(LAYER_PREFIX, cond, true));
            return this;
        }

        /**
         * {@code create_tensor_gate_up_exps} with flags 0 where {@code blk.%d.<trigger>} exists (the helper called
         * inside {@code if (layer.<trigger>)}).
         */
        Builder gateUpExpsWithTrigger(String trigger, LayerCondition cond) {
            out.add(LoaderHelpers.gateUpExps(LAYER_PREFIX, cond, true).ifPresent(LAYER_PREFIX + trigger));
            return this;
        }

        /**
         * {@code create_tensor_gate_up_exps} with a constant {@code TENSOR_NOT_REQUIRED}: the probed form, all
         * optional, probing on NextN blocks as well ({@link LoaderHelpers#gateUpExpsProbing}).
         */
        Builder gateUpExpsOptional(LayerCondition cond) {
            out.add(LoaderHelpers.gateUpExpsProbing(LAYER_PREFIX, cond));
            return this;
        }

        /** {@code ffn_gate_inp} plus {@code ffn_down/up_exps} with an optional {@code ffn_gate_exps}. */
        Builder expertsOptionalGate(LayerCondition cond) {
            when(cond, "ffn_gate_inp.weight");
            optionalWhen(cond, "ffn_gate_exps.weight");
            return when(cond, "ffn_down_exps.weight", "ffn_up_exps.weight");
        }

        /** Shared expert {@code ffn_gate/down/up_shexp.weight}, required on the layers selected by {@code cond}. */
        Builder sharedExperts(LayerCondition cond) {
            return when(cond, SHEXP);
        }

        /** Shared expert created with {@code TENSOR_NOT_REQUIRED}. */
        Builder sharedExpertsOptional(LayerCondition cond) {
            return optionalWhen(cond, SHEXP);
        }

        /** Optional {@code exp_probs_b.bias} (expert selection bias). */
        Builder probsBiasOptional(LayerCondition cond) {
            return optionalWhen(cond, "exp_probs_b.bias");
        }

        /**
         * The NextN block tensors {@code nextn.eh_proj/enorm/hnorm} and the optional
         * {@code nextn.embed_tokens/shared_head_head/shared_head_norm}; all are optional because the blocks
         * load with {@code TENSOR_SKIP} unless MTP is requested (llama-model.cpp:3917-3919).
         */
        Builder nextn() {
            return nextnOf("eh_proj", "enorm", "hnorm", "embed_tokens", "shared_head_head", "shared_head_norm");
        }

        /**
         * The listed {@code blk.%d.nextn.<name>.weight} tensors of the NextN blocks, all optional (see
         * {@link #nextn()}).
         */
        Builder nextnOf(String... names) {
            for (String n : names) {
                optionalWhen(TensorRequirement.NEXTN, "nextn." + n + ".weight");
            }
            return this;
        }

        /** Unmodifiable copy of the accumulated list. */
        List<TensorRequirement> build() {
            return Collections.unmodifiableList(new ArrayList<>(out));
        }
    }
}

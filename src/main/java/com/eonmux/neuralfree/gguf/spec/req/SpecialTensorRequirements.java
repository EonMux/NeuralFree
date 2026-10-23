package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerSpan;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.ModelCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.Scope;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Shared vocabulary of the tensor lists of the encoder, classifier, diffusion, drafter, audio,
 * encoder-decoder and looped architectures. The per-architecture lists live in
 * {@link SpecialTensorRequirementsEncoder}, {@link SpecialTensorRequirementsHeads},
 * {@link SpecialTensorRequirementsGen} and {@link SpecialTensorRequirementsDraft}; each list transcribes one
 * llama.cpp {@code llama_model_<arch>::load_arch_tensors} (src/models/&lt;arch&gt;.cpp): a
 * {@code create_tensor} call with flags 0 is a required entry, one with {@code TENSOR_NOT_REQUIRED} an
 * optional entry. Tensors created under an hparams condition are model entries with
 * {@link TensorRequirement#onlyWhen}, tensors created only when another tensor exists carry that trigger
 * ({@link TensorRequirement#ifPresent}), and blocks numbered past the trunk (decoder stacks, drafter heads)
 * scan their own layer span ({@link TensorRequirement#overLayers}).
 */
public final class SpecialTensorRequirements {

    /** Layer-format prefix of decoder-stack blocks. */
    static final String BLK = "blk.%d.";
    /** Layer-format prefix of T5 encoder blocks. */
    static final String ENC = "enc.blk.%d.";
    /** Layer-format prefix of T5 decoder blocks and the clef decision-head blocks. */
    static final String DEC = "dec.blk.%d.";

    private SpecialTensorRequirements() {
    }

    /**
     * Tensor list of an architecture that reuses another architecture's {@code load_arch_tensors} and then
     * creates its own tensors (C++ subclasses calling the base loader first, e.g. clef.cpp:26).
     */
    public static List<TensorRequirement> extend(List<TensorRequirement> base, List<TensorRequirement> own) {
        List<TensorRequirement> out = new ArrayList<>(base.size() + own.size());
        out.addAll(base);
        out.addAll(own);
        return Collections.unmodifiableList(out);
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

        /** Model-level tensors required when the model-level tensor {@code trigger} exists. */
        Builder modelWithTrigger(String trigger, String... names) {
            for (String n : names) {
                out.add(TensorRequirement.model(n).ifPresent(trigger));
            }
            return this;
        }

        /** Required {@code <base>.weight} and {@code <base>.bias} model-level tensors. */
        Builder modelWeightBias(String base) {
            return model(base + ".weight", base + ".bias");
        }

        /** Optional {@code <base>.weight} and {@code <base>.bias} model-level tensors. */
        Builder modelWeightBiasOptional(String base) {
            return modelOptional(base + ".weight", base + ".bias");
        }

        /** Required {@code blk.%d.<suffix>} tensors on the trunk layers. */
        Builder layer(String... suffixes) {
            return at(BLK, true, TensorRequirement.TRUNK, suffixes);
        }

        /** Optional {@code blk.%d.<suffix>} tensors on the trunk layers. */
        Builder layerOptional(String... suffixes) {
            return at(BLK, false, TensorRequirement.TRUNK, suffixes);
        }

        /** Required {@code blk.%d.<suffix>} tensors on the layers selected by {@code when}. */
        Builder layerWhen(LayerCondition when, String... suffixes) {
            return at(BLK, true, when, suffixes);
        }

        /** Optional {@code blk.%d.<suffix>} tensors on the layers selected by {@code when}. */
        Builder layerOptionalWhen(LayerCondition when, String... suffixes) {
            return at(BLK, false, when, suffixes);
        }

        /** Required {@code blk.%d.<base>.weight} and {@code .bias} tensors on the trunk layers. */
        Builder layerWeightBias(String... bases) {
            return weightBias(BLK, true, TensorRequirement.TRUNK, bases);
        }

        /**
         * Tensors named {@code <prefix><suffix>} where {@code prefix} contains the {@code %d} layer index, on
         * the layers selected by {@code when}.
         */
        Builder at(String prefix, boolean required, LayerCondition when, String... suffixes) {
            return over(TensorRequirement.ALL_LAYERS, prefix, required, when, suffixes);
        }

        /**
         * As {@link #at}, scanning layer slots {@code [0, span)}, which may extend past {@code nLayerAll};
         * {@code when} must then not read per-layer hparams for slots beyond it.
         */
        Builder over(LayerSpan span, String prefix, boolean required, LayerCondition when, String... suffixes) {
            for (String s : suffixes) {
                out.add(TensorRequirement.of(Scope.LAYER, required, when, new String[][] {{prefix + s}})
                        .overLayers(span));
            }
            return this;
        }

        /** As {@link #weightBias}, scanning layer slots {@code [0, span)} (see {@link #over}). */
        Builder weightBiasOver(LayerSpan span, String prefix, boolean required, LayerCondition when,
                               String... bases) {
            for (String base : bases) {
                over(span, prefix, required, when, base + ".weight", base + ".bias");
            }
            return this;
        }

        /** {@code <prefix><base>.weight} and {@code .bias} tensors on the layers selected by {@code when}. */
        Builder weightBias(String prefix, boolean required, LayerCondition when, String... bases) {
            for (String base : bases) {
                at(prefix, required, when, base + ".weight", base + ".bias");
            }
            return this;
        }

        /**
         * {@code create_tensor_qkv} ({@link LoaderHelpers#qkv}) on the layers selected by {@code when}: flags 0 when
         * {@code required}, {@code TENSOR_NOT_REQUIRED} otherwise.
         */
        Builder qkv(LayerCondition when, boolean required) {
            out.addAll(LoaderHelpers.qkv(BLK, when, required));
            return this;
        }

        /** {@link #qkv(LayerCondition, boolean)} on the trunk layers with flags 0. */
        Builder qkv() {
            return qkv(TensorRequirement.TRUNK, true);
        }

        /**
         * The optional {@code rope_freqs} of the loaders that create no LONGROPE factors
         * ({@link LoaderHelpers#ropeFreqs}).
         */
        Builder ropeFreqsOptional() {
            out.add(LoaderHelpers.ropeFreqs());
            return this;
        }

        /** Appends a prepared list. */
        Builder addAll(List<TensorRequirement> reqs) {
            out.addAll(reqs);
            return this;
        }

        List<TensorRequirement> build() {
            return Collections.unmodifiableList(new ArrayList<>(out));
        }
    }
}

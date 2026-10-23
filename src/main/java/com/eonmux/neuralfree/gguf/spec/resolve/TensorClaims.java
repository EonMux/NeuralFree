package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.UnusedTensorException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The llama.cpp check that every tensor of the file is loaded ({@code done_getting_tensors},
 * src/llama-model-loader.cpp:1400-1410). llama.cpp counts each tensor its loader creates, required or
 * {@code TENSOR_NOT_REQUIRED}, whether or not the graph reads it, and refuses the file when fewer tensors were created
 * than the file holds. Here a tensor is claimed when:
 * <ul>
 *   <li>an entry of the architecture's tensor list ({@code gguf.spec.req}, the transcription of its
 *       {@code load_arch_tensors}) creates it on a layer the entry applies to, as
 *       {@link TensorRequirement#created} selects: nothing when the entry's trigger does not select it, only the probed
 *       alternative of a probing entry (the fused or the split form of {@code create_tensor_qkv} and
 *       {@code create_tensor_gate_up_exps}, never both), every alternative otherwise;</li>
 *   <li>the generic pass that follows every loader creates it (src/llama-model.cpp:1717-1870): the per-tensor
 *       {@code .scale} and {@code .input_scale} of each loaded projection in {@link #SCALED_BASES} on every layer
 *       slot, and {@code output.scale} / {@code output.input_scale} when the output head is NVFP4.</li>
 * </ul>
 * NextN blocks are claimed whether or not a drafter runs them, as llama.cpp creates their tensors with
 * {@code TENSOR_SKIP} when MTP is off (src/llama-model.cpp:3901-3919); on those slots the qkv / gate-up helper entries
 * claim every name ({@link TensorRequirement#createdWhenSkipped}). A list that names trunk tensors for NextN
 * blocks claims them through those entries. A list that names only NextN-specific tensors for them
 * ({@code nextn.*}, a NextN-only norm) relies on the loader creating a NextN block with the trunk block's body
 * (models/glm4-moe.cpp:49-53, models/deepseek2.cpp:88-90, models/hy-v3.cpp:74-77): its entries are then also
 * evaluated on {@link ModelHparams#withNextnAsTrunk()} over the NextN blocks. A list without NextN entries claims
 * nothing there. Names are matched as {@link ModelSpecValidator#isPresent} finds them, so Hugging Face aliases
 * resolve to the stored tensor. Stateless.
 */
public final class TensorClaims {

    /**
     * Projections whose {@code .scale} and {@code .input_scale} the generic pass creates when the projection is
     * loaded (src/llama-model.cpp:1720-1858).
     */
    static final List<String> SCALED_BASES = Collections.unmodifiableList(Arrays.asList(
            TensorNames.ATTN_Q, TensorNames.ATTN_K, TensorNames.ATTN_V, TensorNames.ATTN_OUT, TensorNames.ATTN_QKV,
            TensorNames.ATTN_GATE, TensorNames.FFN_GATE, TensorNames.FFN_DOWN, TensorNames.FFN_UP,
            TensorNames.FFN_GATE_SHEXP, TensorNames.FFN_DOWN_SHEXP, TensorNames.FFN_UP_SHEXP,
            TensorNames.FFN_GATE_EXPS, TensorNames.FFN_DOWN_EXPS, TensorNames.FFN_UP_EXPS,
            TensorNames.SSM_IN, TensorNames.SSM_OUT, TensorNames.SSM_ALPHA, TensorNames.SSM_BETA,
            TensorNames.NEXTN_EH_PROJ, TensorNames.NEXTN_SHARED_HEAD_HEAD));

    /** Suffix of the activation scale the generic pass creates next to {@code .scale}. */
    static final String INPUT_SCALE = "input_scale";

    private TensorClaims() {
    }

    /**
     * Refuses a file with tensors its architecture never loads.
     *
     * @param requirements the architecture's tensor list
     * @throws UnusedTensorException naming every unclaimed tensor
     */
    public static void requireAllClaimed(ModelHparams hp, TensorCatalog tc, List<TensorRequirement> requirements) {
        List<String> unused = unclaimed(hp, tc, requirements);
        if (!unused.isEmpty()) {
            throw new UnusedTensorException(tc.archName(), tc.size(), unused);
        }
    }

    /** Stored names of the file's tensors that no list entry and no generic pass claims, in file order. */
    public static List<String> unclaimed(ModelHparams hp, TensorCatalog tc, List<TensorRequirement> requirements) {
        Objects.requireNonNull(hp, "hp");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(requirements, "requirements");
        Set<String> claimed = new HashSet<>();
        Predicate<String> present = name -> ModelSpecValidator.isPresent(tc, name);
        for (TensorRequirement r : requirements) {
            claimEntry(hp, tc, r, present, claimed);
        }
        if (nextnBlocksRepeatTheTrunk(hp, requirements)) {
            ModelHparams trunkView = hp.withNextnAsTrunk();
            for (TensorRequirement r : requirements) {
                if (r.scope() == TensorRequirement.Scope.LAYER) {
                    claimLayers(trunkView, tc, r, present, claimed, hp.nLayer(), r.layerSlots(trunkView), hp);
                }
            }
        }
        claimScales(hp, tc, claimed);
        List<String> out = new ArrayList<>();
        for (String name : tc.names()) {
            if (!claimed.contains(name)) {
                out.add(name);
            }
        }
        return out;
    }

    private static void claimEntry(ModelHparams hp, TensorCatalog tc, TensorRequirement r, Predicate<String> present,
                                   Set<String> claimed) {
        if (r.scope() == TensorRequirement.Scope.MODEL) {
            if (r.appliesToModel(hp)) {
                for (String name : r.created(-1, present)) {
                    claim(tc, name, claimed);
                }
            }
            return;
        }
        claimLayers(hp, tc, r, present, claimed, 0, r.layerSlots(hp), hp);
    }

    /**
     * Claims the names layer entry {@code r} creates on the slots {@code [from, to)} it applies to under {@code hp};
     * the NextN blocks of the file ({@code [nLayer, nLayerAll)} of {@code file}) are loaded with
     * {@code TENSOR_SKIP}.
     */
    private static void claimLayers(ModelHparams hp, TensorCatalog tc, TensorRequirement r, Predicate<String> present,
                                    Set<String> claimed, int from, int to, ModelHparams file) {
        for (int il = from; il < to; il++) {
            if (!r.appliesTo(hp, il)) {
                continue;
            }
            boolean skipped = il >= file.nLayer() && il < file.nLayerAll();
            List<String> names = skipped ? r.createdWhenSkipped(il, present) : r.created(il, present);
            for (String name : names) {
                claim(tc, name, claimed);
            }
        }
    }

    /**
     * True when the model has trunk and NextN blocks and the list names none of the trunk's tensors for the NextN
     * blocks (only their {@code nextn.*} tensors and NextN-only norms), so the blocks' decoder tensors are those of a
     * trunk block. A model made only of NextN blocks (gemma4-assistant) has no trunk body to repeat.
     */
    static boolean nextnBlocksRepeatTheTrunk(ModelHparams hp, List<TensorRequirement> requirements) {
        Set<String> nextn = layerFormats(hp, requirements, hp.nLayer(), hp.nLayerAll());
        if (nextn.isEmpty()) {
            return false;
        }
        Set<String> trunk = layerFormats(hp, requirements, 0, hp.nLayer());
        return !trunk.isEmpty() && Collections.disjoint(nextn, trunk);
    }

    /** Name formats of the layer entries that apply to some layer slot in {@code [from, to)}. */
    private static Set<String> layerFormats(ModelHparams hp, List<TensorRequirement> requirements, int from, int to) {
        Set<String> formats = new HashSet<>();
        for (TensorRequirement r : requirements) {
            for (int il = from; il < to; il++) {
                if (r.appliesTo(hp, il)) {
                    for (List<String> group : r.alternatives()) {
                        formats.addAll(group);
                    }
                    break;
                }
            }
        }
        return formats;
    }

    /** The generic per-tensor scale pass over every layer slot and the NVFP4 output head. */
    private static void claimScales(ModelHparams hp, TensorCatalog tc, Set<String> claimed) {
        for (int il = 0; il < hp.nLayerAll(); il++) {
            for (String base : SCALED_BASES) {
                TensorRef w = tc.layer(il, TensorNames.weight(base));
                if (w != null && claimed.contains(w.name())) {
                    claim(tc, TensorNames.layerName(il, TensorNames.scale(base)), claimed);
                    claim(tc, TensorNames.layerName(il, base + "." + INPUT_SCALE), claimed);
                }
            }
        }
        TensorRef out = claimedRef(tc, TensorNames.weight(TensorNames.OUTPUT), claimed);
        if (out == null) {
            out = claimedRef(tc, TensorNames.weight(TensorNames.TOKEN_EMBD), claimed);
        }
        if (out != null && out.type() == GGMLType.NVFP4) {
            claim(tc, TensorNames.scale(TensorNames.OUTPUT), claimed);
            claim(tc, TensorNames.OUTPUT + "." + INPUT_SCALE, claimed);
        }
    }

    private static TensorRef claimedRef(TensorCatalog tc, String name, Set<String> claimed) {
        TensorRef r = tc.get(name);
        return r != null && claimed.contains(r.name()) ? r : null;
    }

    /**
     * Claims the stored tensor that {@code name} resolves to, directly or through an alias; absent names are
     * skipped.
     */
    private static void claim(TensorCatalog tc, String name, Set<String> claimed) {
        TensorRef r = ModelSpecValidator.find(tc, name);
        if (r != null) {
            claimed.add(r.name());
        }
    }
}

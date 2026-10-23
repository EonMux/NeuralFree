package com.eonmux.neuralfree.gguf.spec.req;

import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.LayerCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.ModelCondition;
import com.eonmux.neuralfree.gguf.arch.TensorRequirement.Scope;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Tensor entries of the loader helpers every architecture list shares, transcribed from llama.cpp:
 * <ul>
 *   <li>{@code create_tensor_qkv} (llama-model.cpp:3863-3898). With flags 0 or {@code TENSOR_NOT_REQUIRED} the
 *       helper probes {@code attn_qkv.weight}: when it exists it creates the fused weight and {@code attn_qkv.bias},
 *       and the separate {@code attn_q/k/v.bias} only when the fused bias is absent; otherwise it creates
 *       {@code attn_q/k/v.weight} with the caller's flags and their optional biases, never {@code attn_qkv.bias}.
 *       With {@code TENSOR_SKIP} (a NextN block without MTP) it creates all eight names.</li>
 *   <li>{@code create_tensor_gate_up_exps} (llama-model.cpp:3846-3861): the fused {@code ffn_gate_up_exps.weight}
 *       when it exists, else {@code ffn_gate_exps} and {@code ffn_up_exps} with the caller's flags; with
 *       {@code TENSOR_SKIP} all three.</li>
 *   <li>the per-layer RoPE factors, created with {@code TENSOR_NOT_REQUIRED} and duplicated from layer 0, so the
 *       names carry no layer index: {@code rope_factors_long}/{@code _short} when {@code rope_scaling_type_train}
 *       is LONGROPE and {@code rope_freqs} otherwise (models/llama.cpp:61-67 and the loaders that copy it), or
 *       {@code rope_freqs} alone (models/arcee.cpp:37, exaone.cpp:35, cogvlm.cpp:36, llama4.cpp:73).</li>
 * </ul>
 * The helper entries are {@link TensorRequirement#probed()} and marked
 * {@link TensorRequirement#creatingAllWhenSkipped()}: they transcribe a helper called with the block's flags, so a
 * NextN block loaded with {@code TENSOR_SKIP} claims every name. A loader that passes constant flags inside a skipped
 * block probes there too and uses {@link #gateUpExpsProbing}. Stateless.
 */
final class LoaderHelpers {

    private static final String[] QKV_FUSED = {"attn_qkv.weight"};
    private static final String[] QKV_SPLIT = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
    private static final String[] QKV_FUSED_BIAS = {"attn_qkv.bias"};
    private static final String[] QKV_SPLIT_BIAS = {"attn_q.bias", "attn_k.bias", "attn_v.bias"};
    private static final String[] GATE_UP_FUSED = {"ffn_gate_up_exps.weight"};
    private static final String[] GATE_UP_SPLIT = {"ffn_gate_exps.weight", "ffn_up_exps.weight"};

    /** {@code rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE}. */
    static final ModelCondition LONGROPE = hp -> hp.ropeScalingType() == RopeScalingType.LONGROPE;
    /** Any other training-time RoPE scaling. */
    static final ModelCondition NOT_LONGROPE = hp -> hp.ropeScalingType() != RopeScalingType.LONGROPE;

    private LoaderHelpers() {
    }

    /**
     * {@code create_tensor_qkv} with flags 0 ({@code required}) or {@code TENSOR_NOT_REQUIRED} on the layers
     * selected by {@code when}.
     *
     * @param prefix layer-name prefix with one {@code %d}, e.g. {@code blk.%d.}
     */
    static List<TensorRequirement> qkv(String prefix, LayerCondition when, boolean required) {
        Objects.requireNonNull(prefix, "prefix");
        String fused = prefix + QKV_FUSED[0];
        TensorRequirement weights = TensorRequirement.of(Scope.LAYER, required, when,
                TensorRequirement.prefixed(prefix, QKV_FUSED, QKV_SPLIT)).probed().creatingAllWhenSkipped();
        TensorRequirement fusedBiases = TensorRequirement.of(Scope.LAYER, false, when,
                TensorRequirement.prefixed(prefix, QKV_FUSED_BIAS, QKV_SPLIT_BIAS)).probed().ifPresent(fused)
                .creatingAllWhenSkipped();
        TensorRequirement splitBiases = TensorRequirement.of(Scope.LAYER, false, when,
                TensorRequirement.prefixed(prefix, QKV_SPLIT_BIAS)).ifAbsent(fused).creatingAllWhenSkipped();
        return Collections.unmodifiableList(Arrays.asList(weights, fusedBiases, splitBiases));
    }

    /**
     * {@code create_tensor_gate_up_exps} with flags 0 ({@code required}) or {@code TENSOR_NOT_REQUIRED} on the
     * layers selected by {@code when}.
     */
    static TensorRequirement gateUpExps(String prefix, LayerCondition when, boolean required) {
        Objects.requireNonNull(prefix, "prefix");
        return TensorRequirement.of(Scope.LAYER, required, when,
                TensorRequirement.prefixed(prefix, GATE_UP_FUSED, GATE_UP_SPLIT)).probed().creatingAllWhenSkipped();
    }

    /**
     * {@code create_tensor_gate_up_exps} called with a constant {@code TENSOR_NOT_REQUIRED} rather than the block's
     * flags: it probes on NextN blocks loaded with {@code TENSOR_SKIP} as well.
     */
    static TensorRequirement gateUpExpsProbing(String prefix, LayerCondition when) {
        Objects.requireNonNull(prefix, "prefix");
        return TensorRequirement.of(Scope.LAYER, false, when,
                TensorRequirement.prefixed(prefix, GATE_UP_FUSED, GATE_UP_SPLIT)).probed();
    }

    /**
     * The RoPE factors of the loaders that branch on the training-time scaling: {@code rope_factors_long} and
     * {@code rope_factors_short} under LONGROPE, {@code rope_freqs} otherwise, all optional.
     */
    static List<TensorRequirement> ropeFactorsByScaling() {
        return Collections.unmodifiableList(Arrays.asList(
                TensorRequirement.modelOptional("rope_factors_long.weight").onlyWhen(LONGROPE),
                TensorRequirement.modelOptional("rope_factors_short.weight").onlyWhen(LONGROPE),
                TensorRequirement.modelOptional("rope_freqs.weight").onlyWhen(NOT_LONGROPE)));
    }

    /** The optional {@code rope_freqs} of the loaders that create no LONGROPE factors. */
    static TensorRequirement ropeFreqs() {
        return TensorRequirement.modelOptional("rope_freqs.weight");
    }
}

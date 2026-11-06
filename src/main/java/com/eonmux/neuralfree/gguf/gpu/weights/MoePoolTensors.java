package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * The stored tensors one hosted MoE layer's {@link com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn} block reads from
 * the device, so a pooled layer's block can be uploaded to the GPU that caches its experts instead of the GPU that
 * runs its attention. The router, the expert scales and biases, the post-expert norm, the shared expert and the
 * parallel dense FFN are the block's own; the norms the layer executor applies around the block
 * ({@code ffn_norm}, the post-FFN norm and the adaptive conditioning) stay with the layer's stage. The routed expert
 * stacks themselves are host-resident under the expert cache and are therefore not listed.
 *
 * <p>The set is the same one {@link com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn#bind} resolves, for the layer kinds
 * the hosted path builds ({@link com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn#checkSupported}). Immutable.</p>
 */
public final class MoePoolTensors {

    private final Set<String> names;

    private MoePoolTensors(Set<String> names) {
        this.names = names;
    }

    /**
     * The block tensors of {@code layer}, qualified with the layer's stack and weight layer.
     *
     * @throws IllegalArgumentException when the layer is not a routed MoE layer
     */
    public static MoePoolTensors of(LayerSpec layer) {
        Objects.requireNonNull(layer, "layer");
        MoeSpec moe = layer.moe();
        if (moe == null) {
            throw new IllegalArgumentException("layer " + layer.il() + " has no routed experts");
        }
        Set<String> names = new LinkedHashSet<>();
        addBase(names, layer, TensorNames.FFN_GATE_INP);
        // exp_probs_b is stored either bare or as a bias tensor (hy_v3 spells it without a suffix).
        addBase(names, layer, TensorNames.FFN_EXP_PROBS_B);
        names.add(layer.tensorName(TensorNames.FFN_EXP_PROBS_B));
        addBase(names, layer, TensorNames.FFN_EXP_PROBS_B_VL);
        addBase(names, layer, TensorNames.FFN_GATE_INP_SHEXP);
        addBase(names, layer, TensorNames.FFN_GATE_EXPS);
        addBase(names, layer, TensorNames.FFN_UP_EXPS);
        addBase(names, layer, TensorNames.FFN_DOWN_EXPS);
        addBase(names, layer, TensorNames.FFN_GATE_UP_EXPS);
        addBase(names, layer, TensorNames.FFN_POST_EXP_NORM);
        FfnSpec shared = moe.sharedExpert();
        if (shared != null) {
            addProjection(names, layer, shared.gateWeight());
            addProjection(names, layer, shared.upWeight());
            addProjection(names, layer, shared.downWeight());
        }
        if (layer.ffnKind() == com.eonmux.neuralfree.gguf.arch.FfnKind.MOE_PARALLEL_DENSE) {
            FfnSpec parallel = layer.ffn();
            addProjection(names, layer, parallel.gateWeight());
            addProjection(names, layer, parallel.upWeight());
            addProjection(names, layer, parallel.downWeight());
        }
        return new MoePoolTensors(Collections.unmodifiableSet(names));
    }

    /** The weight, bias and scale tensors of a dotted base name. */
    private static void addBase(Set<String> into, LayerSpec layer, String base) {
        into.add(layer.tensorName(TensorNames.weight(base)));
        into.add(layer.tensorName(TensorNames.bias(base)));
        into.add(layer.tensorName(TensorNames.scale(base)));
    }

    /**
     * One projection weight suffix ({@code ffn_gate_shexp.weight}) with its bias and scale siblings. A projection
     * without {@code .weight} (some fused forms) is taken as stored.
     */
    private static void addProjection(Set<String> into, LayerSpec layer, String suffix) {
        if (suffix == null) {
            return;
        }
        String base = suffix.endsWith(".weight") ? suffix.substring(0, suffix.length() - ".weight".length()) : suffix;
        into.add(layer.tensorName(suffix));
        into.add(layer.tensorName(TensorNames.bias(base)));
        into.add(layer.tensorName(TensorNames.scale(base)));
    }

    /** True when {@code storedName} is one of the block's device tensors. */
    public boolean contains(String storedName) {
        return storedName != null && names.contains(storedName);
    }

    /** The qualified tensor names of the block (unmodifiable). */
    public Set<String> names() {
        return names;
    }

    @Override
    public String toString() {
        return "MoePoolTensors" + names;
    }
}

package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

/**
 * The device-resident tensors of a model (or of one streamed layer), looked up by GGUF name.
 *
 * <p>Implementations provide {@link #find(String)} over the names actually stored in the file; every other lookup
 * derives from it and resolves aliases the way {@code TensorCatalog} does: model-level names also try
 * {@link TensorNames#modelAliases(String)} ({@code lm_head.weight} for {@code output.weight}), main-stack layer
 * names also try the Hugging Face layer aliases ({@code model.layers.N.self_attn.q_proj.weight}). Required lookups
 * throw {@link MissingTensorException} naming the architecture, the layer and the tensor.</p>
 */
public interface GpuWeights {

    /** The tensor stored under exactly this name, or null when it is not resident. */
    GpuWeight find(String name);

    /** {@code general.architecture} of the model, used in missing-tensor messages. */
    default String archName() {
        return "unknown";
    }

    /** Model-level tensor by name or model alias, or null. */
    default GpuWeight getOrNull(String name) {
        GpuWeight w = find(name);
        if (w != null) {
            return w;
        }
        for (String alias : TensorNames.modelAliases(name)) {
            w = find(alias);
            if (w != null) {
                return w;
            }
        }
        return null;
    }

    /**
     * Required model-level tensor by name or model alias.
     *
     * @throws MissingTensorException when it is not resident
     */
    default GpuWeight get(String name) {
        GpuWeight w = getOrNull(name);
        if (w == null) {
            throw new MissingTensorException(archName(), -1, name);
        }
        return w;
    }

    /** True when {@link #getOrNull(String)} finds the tensor. */
    default boolean has(String name) {
        return getOrNull(name) != null;
    }

    /** Main-stack tensor {@code blk.<il>.<suffix>} or its Hugging Face alias, or null. */
    default GpuWeight layerOrNull(int il, String suffix) {
        return stackOrNull(TensorNames.STACK_BLK, il, suffix);
    }

    /**
     * Required main-stack tensor {@code blk.<il>.<suffix>} or its Hugging Face alias.
     *
     * @throws MissingTensorException when it is not resident
     */
    default GpuWeight layer(int il, String suffix) {
        GpuWeight w = layerOrNull(il, suffix);
        if (w == null) {
            throw new MissingTensorException(archName(), il, TensorNames.layerName(il, suffix));
        }
        return w;
    }

    /**
     * Tensor {@code <stack>.<il>.<suffix>} of any stack; the main stack also tries the Hugging Face aliases.
     * Null when absent.
     */
    default GpuWeight stackOrNull(String stack, int il, String suffix) {
        GpuWeight w = find(TensorNames.layerName(stack, il, suffix));
        if (w != null || !TensorNames.STACK_BLK.equals(stack)) {
            return w;
        }
        for (String alias : TensorNames.layerAliases(suffix)) {
            w = find(TensorNames.layerName(TensorNames.HF_LAYER_PREFIX, il, alias));
            if (w != null) {
                return w;
            }
        }
        return null;
    }

    /**
     * Tensor of a layer in its own stack at its weight layer ({@code LayerSpec.weightLayer()}, so loop slots share
     * physical weights), or null.
     */
    default GpuWeight layerOrNull(LayerSpec layer, String suffix) {
        return stackOrNull(layer.stack(), layer.weightLayer(), suffix);
    }

    /**
     * Required tensor of a layer in its own stack at its weight layer.
     *
     * @throws MissingTensorException when it is not resident
     */
    default GpuWeight layer(LayerSpec layer, String suffix) {
        GpuWeight w = layerOrNull(layer, suffix);
        if (w == null) {
            throw new MissingTensorException(archName(), layer.il(), layer.tensorName(suffix));
        }
        return w;
    }
}

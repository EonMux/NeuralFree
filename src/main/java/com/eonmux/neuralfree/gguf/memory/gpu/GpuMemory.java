package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.util.List;

/**
 * Device memory of a model. Accessors take the virtual layer index; a shared-KV layer gets its owner's cache
 * (for a target-model source, the target's cache bound at construction). Accessors return null when the layer
 * has no memory of that kind.
 */
public interface GpuMemory extends ModelMemory {

    /** KV cache read and written by layer {@code il}. */
    GpuKvLayer kv(int il);

    /** T5 decoder cross-attention K/V of layer {@code il}. */
    GpuKvLayer crossKv(int il);

    GpuRecurrentState recurrent(int il);

    GpuIndexerCache indexer(int il);

    GpuMsaCache msa(int il);

    /** DSV4 compressed caches of layer {@code il} (empty when none). */
    List<GpuCompressedKv> compressed(int il);

    /** DSV4 compressed cache of one kind, or null. */
    default GpuCompressedKv compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        for (GpuCompressedKv c : compressed(il)) {
            if (c.plan().kind() == kind) {
                return c;
            }
        }
        return null;
    }

    /** Per-step position inputs. */
    GpuStepInputs stepInputs();
}

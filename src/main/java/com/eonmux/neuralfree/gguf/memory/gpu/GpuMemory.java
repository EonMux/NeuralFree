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

    /** T5 decoder cross-attention K/V of layer {@code il}: sequence 0's stream on a paged memory. */
    GpuKvLayer crossKv(int il);

    /**
     * The T5 cross-attention K/V of layer {@code il} for paged sequence {@code seq}
     * ({@link LayerMemoryPlan#crossStreams()}); a memory of one stream serves sequence 0 only.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the plan's streams
     */
    default GpuKvLayer crossKv(int il, int seq) {
        if (seq != 0) {
            throw new IndexOutOfBoundsException("memory " + this + " holds one cross-attention stream, not " + seq);
        }
        return crossKv(il);
    }

    GpuRecurrentState recurrent(int il);

    GpuIndexerCache indexer(int il);

    GpuMsaCache msa(int il);

    /** DSV4 compressed caches of layer {@code il} (empty when none): sequence 0's stream of each kind. */
    List<GpuCompressedKv> compressed(int il);

    /** DSV4 compressed cache of one kind (sequence 0's stream on a paged memory), or null. */
    default GpuCompressedKv compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        for (GpuCompressedKv c : compressed(il)) {
            if (c.plan().kind() == kind) {
                return c;
            }
        }
        return null;
    }

    /**
     * The DSV4 compressed cache of one kind for paged sequence {@code seq}
     * ({@link LayerMemoryPlan.CompressedKvPlan#streams()}), or null when the layer has none of that kind; a memory of
     * one stream serves sequence 0 only.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the plan's streams
     */
    default GpuCompressedKv compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind, int seq) {
        if (seq != 0) {
            throw new IndexOutOfBoundsException("memory " + this + " holds one DSV4 compressed stream, not " + seq);
        }
        return compressed(il, kind);
    }

    /** Per-step position inputs. */
    GpuStepInputs stepInputs();
}

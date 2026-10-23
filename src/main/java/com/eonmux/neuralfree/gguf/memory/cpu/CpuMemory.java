package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

/**
 * Heap memory of a model, mirroring {@code GpuMemory}. Accessors take the virtual layer index; a shared-KV layer gets
 * its owner's cache. Accessors return null when the layer has no memory of that kind. Side caches (DSA indexer, MSA
 * index keys, DSV4 compressed rows) are exposed as single-head K-only {@link KvAccessor}s and compressor planes as
 * {@link RecurrentStateAccessor}s with tensors {@link #COMPRESSOR_KV} and {@link #COMPRESSOR_SCORE}.
 */
public interface CpuMemory extends ModelMemory {

    /** Name of the compressor kv plane in {@link #compressorState}. */
    String COMPRESSOR_KV = "kv";
    /** Name of the compressor score plane in {@link #compressorState}. */
    String COMPRESSOR_SCORE = "score";

    KvAccessor kv(int il);

    /** T5 decoder cross-attention K/V of layer {@code il}. */
    KvAccessor crossKv(int il);

    RecurrentStateAccessor recurrent(int il);

    KvAccessor indexer(int il);

    KvAccessor msa(int il);

    KvAccessor compressed(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind);

    RecurrentStateAccessor compressorState(int il, LayerMemoryPlan.CompressedKvPlan.Kind kind);

    /** Position stored in {@code row} of KV group {@code group}, or -1 when the row is empty. */
    int positionOfRow(int group, int row);

    /**
     * Marks the rows of positions {@code [pos0, pos0 + nTokens)} as holding those positions in every KV group,
     * before the step's K/V are written.
     */
    void beginStep(int pos0, int nTokens);
}

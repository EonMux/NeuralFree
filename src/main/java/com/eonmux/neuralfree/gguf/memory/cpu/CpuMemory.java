package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

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

    /**
     * The paged sequence bookkeeping of this memory, or null for the contiguous form. Default: null, today's
     * contiguous single-sequence memory; a paged plan ({@code neuralfree.parallel > 1} or {@code neuralfree.kv.paged})
     * replaces it ({@link ModelMemory#sequences}).
     */
    default com.eonmux.neuralfree.gguf.memory.paged.PagedSequences pagedSequences() {
        return null;
    }

    /**
     * Why a paged step over {@code step} would refuse on this memory, or null when
     * {@link #beginPagedStep} can run it. Default: refused, the contiguous memory has no page tables.
     */
    default String pagedStepRefusal(PagedStep step) {
        return getClass().getSimpleName() + " holds no paged sequences";
    }

    /**
     * Marks the rows a paged step writes in every group's mirror ({@code PagedStep.kvPosRows / kvPosValues}), and
     * the mirror-only {@code PagedSequences.pendingReports()} changes (removals, shifts, partial page copies), before
     * the step's K/V are written. The host dequantizes K/V through the physical rows the step's {@code cellIdx}
     * names.
     *
     * @throws UnsupportedOperationException carrying {@link #pagedStepRefusal} when this memory is not paged
     * @throws IllegalArgumentException      when the step belongs to another memory
     */
    default void beginPagedStep(PagedStep step) {
        throw new UnsupportedOperationException(pagedStepRefusal(step));
    }

    /** Clears the paged step begun by the last {@link #beginPagedStep}. No-op without one. */
    default void endPagedStep() {
    }

    /**
     * The paged step the current forward pass runs, set by {@link #beginPagedStep} and cleared by
     * {@link #endPagedStep}, or null between steps. A mixer on a paged memory reads it to store its K/V rows at the
     * step's {@code cellIdx} rows and to attend through the step's page tables; null means the contiguous step,
     * whose store and attend rules the mixer runs unchanged. Default: always null, the contiguous form.
     */
    default com.eonmux.neuralfree.gguf.memory.paged.PagedStep currentPagedStep() {
        return null;
    }
}

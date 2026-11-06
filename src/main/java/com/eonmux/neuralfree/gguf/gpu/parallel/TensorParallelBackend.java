package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.Observers;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.TreeAccept;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuPrefillDriver;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.util.List;
import java.util.Objects;

/**
 * The {@link ForwardBackend} of a tensor split: the GPU backend of the main device with the per-rank state kept in
 * step. Every forward call goes to the main backend, whose mixers and FFN blocks run the ranks through
 * {@link TensorParallelGroup}; the step mirror forwards the main memory's uploaded positions, recorded tokens, page
 * tables and page copies to the rank memories, and {@link #reset()}, {@link #rollback(int)},
 * {@link #canRollback(int)} apply to every rank. The contiguous context shift moves every rank cache before the main
 * memory moves and the split mixers re-rotate each rank's keys; the per-sequence shift over paged rank caches moves
 * each rank's rows and re-rotates them the same way. Multi-sequence batches run over the paged main memory while
 * every rank cache follows the same page tables, cell rows and {@code kv_pos} mirror. The embedding and the output
 * head shard over the ranks' vocabulary slices, and the split covers the attention heads, the dense feed-forward and
 * the routed and shared experts.
 */
public final class TensorParallelBackend implements ForwardBackend, GpuPrefillDriver.StepMirror {

    private final ForwardBackend delegate;
    private final TensorParallelGroup group;

    /**
     * @throws IllegalArgumentException when {@code delegate} is not a GPU backend (the only kind a tensor split
     *                                  wraps)
     */
    public TensorParallelBackend(ForwardBackend delegate, TensorParallelGroup group) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.group = Objects.requireNonNull(group, "group");
        if (!(delegate instanceof GpuForwardBackend)) {
            throw new IllegalArgumentException("a tensor split wraps a GPU forward backend, got "
                    + delegate.getClass().getSimpleName());
        }
        ((GpuForwardBackend) delegate).setStepMirror(this);
        ((GpuForwardBackend) delegate).disableDecodeReplay();
        group.verifyPagedMemory(((GpuForwardBackend) delegate).memory());
    }

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        return delegate.prefill(tokens, pos0);
    }

    @Override
    public float[] prefillEmbeddings(float[][] rows, int pos0) {
        return delegate.prefillEmbeddings(rows, pos0);
    }

    @Override
    public float[] decode(int token, int pos) {
        return delegate.decode(token, pos);
    }

    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return delegate.decodeGreedyOnDevice(token, pos);
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        return delegate.forwardRows(tokens, pos0, outputRows);
    }

    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        return delegate.forwardRowsGreedy(tokens, pos0, outputRows);
    }

    @Override
    public boolean topKOnDevice() {
        return delegate.topKOnDevice();
    }

    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows) {
        return delegate.forwardRowsTopK(tokens, pos0, k, outputRows);
    }

    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k) {
        return delegate.decodeTopKOnDevice(token, pos, k);
    }

    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        return delegate.forwardAll(tokens, outputRows);
    }

    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        return delegate.hiddenRows(tokens, tokenTypes);
    }

    @Override
    public int maxSequences() {
        return delegate.maxSequences();
    }

    @Override
    public String batchRefusal() {
        return delegate.batchRefusal();
    }

    @Override
    public String embeddingRowsRefusal() {
        return delegate.embeddingRowsRefusal();
    }

    @Override
    public SequenceEncoder sequenceEncoder() {
        return delegate.sequenceEncoder();
    }

    /**
     * Runs the multi-sequence batch over the main backend's paged memory; the step mirror forwards each paged step,
     * its page copies and its page tables to every rank cache, so every rank's sharded attention reads the same
     * rows.
     */
    @Override
    public float[][] forwardBatch(ForwardBatch batch) {
        return delegate.forwardBatch(batch);
    }

    @Override
    public int[] forwardBatchGreedy(ForwardBatch batch) {
        return delegate.forwardBatchGreedy(batch);
    }

    @Override
    public float[][] hiddenBatch(ForwardBatch batch) {
        return delegate.hiddenBatch(batch);
    }

    /** Whether the main backend walks a tree verify's acceptance on its device ({@code LLM_TreeAccept}). */
    @Override
    public boolean treeAcceptOnDevice() {
        return delegate.treeAcceptOnDevice();
    }

    /**
     * The masked single-pass verify batch over the main backend, its acceptance walked on the main device; the step
     * mirror forwards the paged step and its page tables to every rank cache, so the split reads the same rows as a
     * plain {@link #forwardBatch} of the batch.
     */
    @Override
    public TreeAccept treeAcceptForward(ForwardBatch batch, DraftTree proposal) {
        return delegate.treeAcceptForward(batch, proposal);
    }

    @Override
    public Observers observers() {
        return delegate.observers();
    }

    @Override
    public void reset() {
        delegate.reset();
        group.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        return delegate.canRollback(nTokens) && group.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        if (!canRollback(nTokens)) {
            throw new IllegalStateException("the tensor-parallel model cannot drop the last " + nTokens
                    + " position(s) on every rank");
        }
        delegate.rollback(nTokens);
        group.rollback(nTokens);
    }

    /**
     * Shifts the contiguous context on every rank: the group moves each rank cache's rows, then the main backend
     * moves the main memory and every mixer re-rotates its cache (the split mixers rotate their rank's shard keys).
     * Refused by {@link #shiftSequenceRefusal()} and by the main backend's own refusal before any row moves; a failure
     * after the rows moved resets every rank cache and the main memory.
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(refusal);
        }
        group.shiftPositions(nKeep, nDiscard);
        try {
            delegate.shiftContext(nKeep, nDiscard);
        } catch (RuntimeException e) {
            try {
                group.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /** Why shifting the contiguous context would refuse, or null when {@link #shiftContext} can run. */
    private String shiftRefusal() {
        if (!(delegate instanceof GpuForwardBackend)) {
            return "a tensor-parallel split shifts the caches of its GPU backend";
        }
        String refusal = ((GpuForwardBackend) delegate).contextShiftRefusal();
        if (refusal != null) {
            return refusal;
        }
        return group.contextShiftRefusal();
    }

    /**
     * Why the per-sequence shift over paged rank caches would refuse, or null: the main backend can shift a sequence
     * and every rank cache's sequence view can move its rows.
     */
    @Override
    public String shiftSequenceRefusal() {
        if (!(delegate instanceof GpuForwardBackend)) {
            return "a tensor-parallel split shifts the caches of its GPU backend";
        }
        String refusal = delegate.shiftSequenceRefusal();
        if (refusal != null) {
            return refusal;
        }
        return group.sequenceShiftRefusal();
    }

    /**
     * Shifts one sequence on every rank: the model device plans the shift on its paged memory (removals, row moves,
     * the shift's page copies and the per-row RoPE deltas), each rank cache takes the same copies, deltas and
     * {@code kv_pos} mirror, the split mixers re-rotate every rank's keys by the same deltas, and the model device and
     * the ranks then carry their rows to the new cells.
     */
    @Override
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        String refusal = shiftSequenceRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        GpuForwardBackend gpu = (GpuForwardBackend) delegate;
        com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory main = mainDeviceMemory();
        main.shiftSequence(seq, nKeep, nDiscard);
        int[][] deltas = main.pagedShiftDeltas();
        List<com.eonmux.neuralfree.gguf.memory.paged.RowMove> moves = main.pagedSequences().pendingRowMoves();
        List<com.eonmux.neuralfree.gguf.memory.paged.PageCopy> copies = main.pagedSequences().pendingCopies();
        int[][] mirror = main.pagedMirrorSnapshot();
        try {
            group.applySequenceShift(deltas, moves, copies, mirror);
            gpu.rotateShiftedKeys(nKeep, nDiscard);
        } finally {
            try {
                main.moveShiftedRows();
            } finally {
                group.moveShiftedRows(moves);
            }
        }
    }

    /** The model device's paged memory, the one that plans every paged step and shift. */
    private com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory mainDeviceMemory() {
        ModelMemory memory = ((GpuForwardBackend) delegate).memory();
        if (!(memory instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory)) {
            throw new IllegalStateException("the per-sequence shift needs the GPU backend's device memory, got "
                    + (memory == null ? "none" : memory.getClass().getSimpleName()));
        }
        return (com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory) memory;
    }

    /**
     * Runs a sequence-level compaction (the tree verify's commit) on the split: the model device carries its queued
     * page copies and row moves, and every rank cache takes the same copies, the post-compaction {@code kv_pos}
     * mirror and row moves (no RoPE deltas — a compaction's rows keep their rotations). No-op without a GPU
     * delegate or paged memory.
     */
    @Override
    public void carryRowMoves() {
        if (!(delegate instanceof GpuForwardBackend)) {
            return;
        }
        com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory main = mainDeviceMemory();
        if (main.pagedSequences() == null) {
            return;
        }
        java.util.List<com.eonmux.neuralfree.gguf.memory.paged.RowMove> moves =
                main.pagedSequences().pendingRowMoves();
        java.util.List<com.eonmux.neuralfree.gguf.memory.paged.PageCopy> copies =
                main.pagedSequences().pendingCopies();
        int[][] mirror = main.pagedMirrorSnapshot();
        try {
            main.carryRowMoves();
        } finally {
            if (!moves.isEmpty() || !copies.isEmpty()) {
                group.applySequenceShift(null, moves, copies, mirror);
                group.moveShiftedRows(moves);
            }
        }
    }

    @Override
    public ModelMemory memory() {
        return delegate.memory();
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        delegate.setHiddenTap(tap);
    }

    @Override
    public void close() {
        if (delegate instanceof GpuForwardBackend) {
            try {
                ((GpuForwardBackend) delegate).setStepMirror(null);
            } catch (RuntimeException ignored) {
                // the driver is already gone; closing the delegate reports it
            }
        }
        delegate.close();
    }

    // ------------------------------------------------------------------ StepMirror

    @Override
    public void beforeStep(int pos0, int nTokens) {
        group.uploadStepInputs(pos0, nTokens);
    }

    @Override
    public void afterTokens(int[] tokens, int off, int n) {
        group.appendStep(tokens, off, n);
    }

    @Override
    public void afterEmbeddings(int n, int token) {
        group.appendEmbeddingStep(n, token);
    }

    @Override
    public void copyPage(com.eonmux.neuralfree.gguf.memory.paged.PageCopy copy) {
        group.copyPage(copy, mainPaged());
    }

    @Override
    public void beginPagedStep(com.eonmux.neuralfree.gguf.memory.paged.PagedStep step, int[] positions,
                               int[][] ropeSections, boolean nonCausal) {
        group.beginPagedStep(step, positions, ropeSections, nonCausal, mainPaged(), mainPagedMirror());
    }

    /** The model device's paged sequences, or null for the contiguous form. */
    private com.eonmux.neuralfree.gguf.memory.paged.PagedSequences mainPaged() {
        ModelMemory memory = ((GpuForwardBackend) delegate).memory();
        return memory instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                ? ((com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory) memory).pagedSequences()
                : null;
    }

    /**
     * The model device's paged {@code kv_pos} mirror after its own step upload applied the step's copies, removals
     * and sliding-window trims; the rank caches follow it so a bookkeeping change only the model device performed
     * reaches every rank's paged kernels.
     */
    private int[][] mainPagedMirror() {
        if (!(delegate instanceof GpuForwardBackend)) {
            return null;
        }
        ModelMemory memory = ((GpuForwardBackend) delegate).memory();
        return memory instanceof com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory
                ? ((com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory) memory).pagedMirrorSnapshot()
                : null;
    }

    @Override
    public void endPagedStep() {
        group.endPagedStep();
    }

    @Override
    public String toString() {
        return "TensorParallelBackend{" + group + '}';
    }
}

package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;

import java.util.Objects;

/**
 * Host memory seen by layers that read target caches: the caches and their row positions are the target's, and
 * the plan is the target's (attention mixers look up the rotation of the layer that owns a cache there). A step
 * stores nothing.
 */
final class SharedCpuMemory implements CpuMemory {
    private final HeapModelMemory own;
    private final CpuMemory target;

    SharedCpuMemory(HeapModelMemory own, CpuMemory target) {
        this.own = own;
        this.target = Objects.requireNonNull(target, "target memory");
    }

    @Override public KvAccessor kv(int il) { return own.kv(il); }
    @Override public KvAccessor crossKv(int il) { return own.crossKv(il); }
    @Override public RecurrentStateAccessor recurrent(int il) { return own.recurrent(il); }
    @Override public KvAccessor indexer(int il) { return own.indexer(il); }
    @Override public KvAccessor msa(int il) { return own.msa(il); }

    @Override
    public KvAccessor compressed(int il,
            LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        return own.compressed(il, kind);
    }

    @Override
    public RecurrentStateAccessor compressorState(int il,
            LayerMemoryPlan.CompressedKvPlan.Kind kind) {
        return own.compressorState(il, kind);
    }

    @Override public int positionOfRow(int group, int row) { return target.positionOfRow(group, row); }

    @Override
    public void beginStep(int pos0, int nTokens) {
        if (nTokens < 1 || nTokens > own.plan().ubatch() || pos0 < 0) {
            throw new IllegalArgumentException("step of " + nTokens + " tokens at " + pos0 + " outside [1, "
                    + own.plan().ubatch() + "]");
        }
    }

    @Override public MemoryPlan plan() { return target.plan(); }
    @Override public SequenceState sequence() { return own.sequence(); }
    @Override public void reset() { own.reset(); }
    @Override public boolean canRollback(int nTokens) { return own.canRollback(nTokens); }
    @Override public void rollback(int nTokens) { own.rollback(nTokens); }
    @Override public boolean ringsKeepWindow(int nWrite, int nDrop) { return own.ringsKeepWindow(nWrite, nDrop); }
    @Override public boolean canSnapshot() { return own.canSnapshot(); }
    @Override public void snapshot() { own.snapshot(); }
    @Override public void restore() { own.restore(); }
    @Override public boolean hasRecurrentState() { return false; }
    @Override public void close() { own.close(); }

    @Override
    public String toString() {
        return "SharedCpuMemory{" + own.plan() + '}';
    }
}

package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuCompressedKv;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuIndexerCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMsaCache;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuRecurrentState;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;

import java.util.List;
import java.util.Objects;

/**
 * Target device memory seen by layers that read target caches: the caches, {@code kv_pos} and {@code cell_idx}
 * are the target's; query and RoPE positions come from the drafter's own step buffers, so a step never marks a
 * target row.
 */
final class SharedGpuMemory implements GpuMemory {
    private final DeviceModelMemory own;
    private final GpuMemory target;
    private final GpuStepInputs inputs;

    SharedGpuMemory(DeviceModelMemory own, GpuMemory target) {
        this.own = own;
        this.target = Objects.requireNonNull(target, "target memory");
        GpuStepInputs mine = own.stepInputs();
        GpuStepInputs theirs = target.stepInputs();
        this.inputs = new GpuStepInputs() {
            @Override public int groups() { return theirs.groups(); }
            @Override public GpuBuffer kvPos(int group) { return theirs.kvPos(group); }
            @Override public GpuBuffer cellIdx(int group) { return theirs.cellIdx(group); }
            @Override public GpuBuffer qPos() { return mine.qPos(); }
            @Override public GpuBuffer ropePos() { return mine.ropePos(); }
            @Override public GpuBuffer stateRows() { return mine.stateRows(); }
            @Override public int pos0() { return mine.pos0(); }
            @Override public int nTokens() { return mine.nTokens(); }
            @Override public void upload(int pos0, int nTokens) { mine.upload(pos0, nTokens); }
        };
    }

    @Override public GpuKvLayer kv(int il) { return own.kv(il); }
    @Override public GpuKvLayer crossKv(int il) { return own.crossKv(il); }
    @Override public GpuRecurrentState recurrent(int il) { return own.recurrent(il); }
    @Override public GpuIndexerCache indexer(int il) { return own.indexer(il); }
    @Override public GpuMsaCache msa(int il) { return own.msa(il); }
    @Override public List<GpuCompressedKv> compressed(int il) { return own.compressed(il); }
    @Override public GpuStepInputs stepInputs() { return inputs; }
    @Override public MemoryPlan plan() { return own.plan(); }
    @Override public SequenceState sequence() { return own.sequence(); }
    @Override public void reset() { own.reset(); }
    @Override public boolean canRollback(int nTokens) { return own.canRollback(nTokens); }
    @Override public void rollback(int nTokens) { own.rollback(nTokens); }
    @Override public boolean ringsKeepWindow(int nWrite, int nDrop) { return own.ringsKeepWindow(nWrite, nDrop); }
    @Override public boolean canSnapshot() { return own.canSnapshot(); }
    @Override public void snapshot() { own.snapshot(); }
    @Override public void restore() { own.restore(); }
    @Override public void close() { own.close(); }

    @Override
    public String toString() {
        return "SharedGpuMemory{" + own.plan() + '}';
    }
}

package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;

import java.util.Objects;

/**
 * The {@link ModelMemory} of a hybrid CPU+GPU load over the two side memories: the device memory of the offloaded
 * layers and the heap memory of the host layers, both allocated from the same {@link MemoryPlan} and driven in
 * lockstep by the hybrid forward backend. Every sequence operation runs on both sides, so a call succeeds only when
 * each side allows it and then changes both: {@link #reset()} empties both, {@link #rollback(int)} drops the tail of
 * both, and {@link #snapshot()} / {@link #restore()} save and restore the recurrent state of both.
 *
 * <p>The sequence record the facade reports is the device side's; the backend appends every step to both, so the two
 * never disagree. Paged multi-sequence memory is a single-side mechanism and is not covered: the plan a hybrid load
 * runs with is contiguous. Not thread-safe.</p>
 */
public final class HybridMemory implements ModelMemory {

    private final DeviceModelMemory device;
    private final HeapModelMemory host;

    private HybridMemory(DeviceModelMemory device, HeapModelMemory host) {
        this.device = Objects.requireNonNull(device, "device");
        this.host = Objects.requireNonNull(host, "host");
    }

    /**
     * The memory over the two sides of a hybrid load; both were allocated from the same plan, and the backend drives
     * them in lockstep.
     *
     * @throws IllegalArgumentException when the plans differ
     */
    public static HybridMemory of(DeviceModelMemory device, HeapModelMemory host) {
        if (device.plan() != host.plan()) {
            throw new IllegalArgumentException("the two sides of a hybrid load must share one memory plan, got "
                    + device.plan() + " and " + host.plan());
        }
        return new HybridMemory(device, host);
    }

    public DeviceModelMemory device() {
        return device;
    }

    public HeapModelMemory host() {
        return host;
    }

    @Override
    public MemoryPlan plan() {
        return device.plan();
    }

    @Override
    public SequenceState sequence() {
        return device.sequence();
    }

    @Override
    public void reset() {
        host.reset();
        device.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        return host.canRollback(nTokens) && device.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        host.rollback(nTokens);
        device.rollback(nTokens);
    }

    @Override
    public boolean ringsKeepWindow(int nWrite, int nDrop) {
        return host.ringsKeepWindow(nWrite, nDrop) && device.ringsKeepWindow(nWrite, nDrop);
    }

    @Override
    public void snapshot() {
        host.snapshot();
        device.snapshot();
    }

    @Override
    public void restore() {
        host.restore();
        device.restore();
    }

    @Override
    public void close() {
        RuntimeException failure = null;
        try {
            host.close();
        } catch (RuntimeException e) {
            failure = e;
        }
        try {
            device.close();
        } catch (RuntimeException e) {
            if (failure == null) {
                failure = e;
            } else {
                failure.addSuppressed(e);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "HybridMemory{" + device.plan() + "}";
    }
}

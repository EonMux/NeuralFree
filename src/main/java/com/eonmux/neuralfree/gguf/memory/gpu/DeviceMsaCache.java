package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.Objects;

/**
 * Device MSA index-key cache of one layer (minimax-m3): {@code rows} keys of {@code width} in the plan's KV type,
 * zero-filled at allocation, row {@code r} holding the key of the position the KV group's {@code kv_pos[r]} names.
 */
public final class DeviceMsaCache implements GpuMsaCache, AutoCloseable {

    private final int layer;
    private final LayerMemoryPlan.MsaPlan plan;
    private final int group;
    private final GpuBuffer indexKeys;
    private boolean closed;

    private DeviceMsaCache(int layer, LayerMemoryPlan.MsaPlan plan, int group, GpuBuffer indexKeys) {
        this.layer = layer;
        this.plan = plan;
        this.group = group;
        this.indexKeys = indexKeys;
    }

    static DeviceMsaCache allocate(ComputeBackend backend, ComputeQueue queue, int layer, LayerMemoryPlan.MsaPlan plan,
                                   int group) {
        Objects.requireNonNull(plan, "plan");
        GpuBuffer keys = DeviceModelMemory.allocateZeroed(backend, queue, plan.bytes(),
                "MSA index-key cache of layer " + layer);
        return new DeviceMsaCache(layer, plan, group, keys);
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public LayerMemoryPlan.MsaPlan plan() {
        return plan;
    }

    @Override
    public int group() {
        return group;
    }

    @Override
    public GpuBuffer indexKeys() {
        if (closed) {
            throw new IllegalStateException("MSA cache of layer " + layer + " is closed");
        }
        return indexKeys;
    }

    /** Releases the key buffer. Idempotent. */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            indexKeys.close();
        }
    }

    @Override
    public String toString() {
        return "DeviceMsaCache{il=" + layer + ", group=" + group + ", " + plan + '}';
    }
}

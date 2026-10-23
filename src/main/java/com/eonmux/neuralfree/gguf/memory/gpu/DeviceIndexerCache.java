package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.Objects;

/**
 * Device lightning-indexer key cache of one layer: {@code rows} rows of the plan's width and element type,
 * zero-filled at allocation, row {@code r} holding the key of the position the KV group's {@code kv_pos[r]} names.
 */
public final class DeviceIndexerCache implements GpuIndexerCache, AutoCloseable {

    private final int layer;
    private final LayerMemoryPlan.IndexerPlan plan;
    private final int group;
    private final GpuBuffer keys;
    private boolean closed;

    private DeviceIndexerCache(int layer, LayerMemoryPlan.IndexerPlan plan, int group, GpuBuffer keys) {
        this.layer = layer;
        this.plan = plan;
        this.group = group;
        this.keys = keys;
    }

    static DeviceIndexerCache allocate(ComputeBackend backend, ComputeQueue queue, int layer,
                                       LayerMemoryPlan.IndexerPlan plan, int group) {
        Objects.requireNonNull(plan, "plan");
        GpuBuffer keys = DeviceModelMemory.allocateZeroed(backend, queue, plan.bytes(),
                "lightning-indexer key cache of layer " + layer);
        return new DeviceIndexerCache(layer, plan, group, keys);
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public LayerMemoryPlan.IndexerPlan plan() {
        return plan;
    }

    @Override
    public int group() {
        return group;
    }

    @Override
    public GpuBuffer keys() {
        if (closed) {
            throw new IllegalStateException("indexer cache of layer " + layer + " is closed");
        }
        return keys;
    }

    /** Releases the key buffer. Idempotent. */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            keys.close();
        }
    }

    @Override
    public String toString() {
        return "DeviceIndexerCache{il=" + layer + ", group=" + group + ", " + plan + '}';
    }
}

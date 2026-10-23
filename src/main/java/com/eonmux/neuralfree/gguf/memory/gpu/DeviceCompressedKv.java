package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.Objects;

/**
 * Device DeepSeek-V4 compressed cache of one layer and kind (src/llama-kv-cache-dsv4.cpp:1244-1340): the compressed
 * K rows and the kv / score compressor tensors, each {@code planes()} consecutive F32 planes of
 * {@code stateRows x stateWidth} (the live plane, then the rollback snapshots). All three are zeroed at allocation
 * and at every reset, because the compressor accumulates across positions.
 */
public final class DeviceCompressedKv implements GpuCompressedKv, AutoCloseable {

    private final int layer;
    private final LayerMemoryPlan.CompressedKvPlan plan;
    private final GpuBuffer cache;
    private final GpuBuffer stateKv;
    private final GpuBuffer stateScore;
    private boolean closed;

    private DeviceCompressedKv(int layer, LayerMemoryPlan.CompressedKvPlan plan, GpuBuffer cache, GpuBuffer stateKv,
                               GpuBuffer stateScore) {
        this.layer = layer;
        this.plan = plan;
        this.cache = cache;
        this.stateKv = stateKv;
        this.stateScore = stateScore;
    }

    static DeviceCompressedKv allocate(ComputeBackend backend, ComputeQueue queue, int layer,
                                       LayerMemoryPlan.CompressedKvPlan plan) {
        Objects.requireNonNull(plan, "plan");
        String what = plan.kind() + " compressed cache of layer " + layer;
        long tensorBytes = plan.stateTensorBytes();
        GpuBuffer cache = null;
        GpuBuffer kv = null;
        try {
            cache = DeviceModelMemory.allocateZeroed(backend, queue, plan.cacheBytes(), what);
            kv = DeviceModelMemory.allocateZeroed(backend, queue, tensorBytes, what + " (kv planes)");
            GpuBuffer score = DeviceModelMemory.allocateZeroed(backend, queue, tensorBytes, what + " (score planes)");
            return new DeviceCompressedKv(layer, plan, cache, kv, score);
        } catch (RuntimeException e) {
            if (kv != null) {
                kv.close();
            }
            if (cache != null) {
                cache.close();
            }
            throw e;
        }
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public LayerMemoryPlan.CompressedKvPlan plan() {
        return plan;
    }

    @Override
    public GpuBuffer cache() {
        checkOpen();
        return cache;
    }

    @Override
    public GpuBuffer stateKv() {
        checkOpen();
        return stateKv;
    }

    @Override
    public GpuBuffer stateScore() {
        checkOpen();
        return stateScore;
    }

    /**
     * Copies rollback snapshot plane {@code d} of both compressor tensors into their live plane (llama.cpp's
     * restore of plane {@code rs_idx}, llama-kv-cache-dsv4.cpp:670-681), in order on {@code queue}.
     *
     * @throws IllegalArgumentException when {@code d} is not a snapshot plane ({@code 1..snapshots})
     */
    void restorePlane(ComputeQueue queue, int d) {
        checkOpen();
        if (d < 1 || d > plan.snapshots()) {
            throw new IllegalArgumentException(plan.kind() + " compressor of layer " + layer + " has snapshot planes 1.."
                    + plan.snapshots() + ", not " + d);
        }
        long planeBytes = (long) plan.planeFloats() * Float.BYTES;
        queue.enqueueCopyBuffer(stateKv, d * planeBytes, stateKv, 0, planeBytes);
        queue.enqueueCopyBuffer(stateScore, d * planeBytes, stateScore, 0, planeBytes);
    }

    /** Zeroes the compressed rows and both compressor tensors, every plane. */
    void reset(ComputeQueue queue) {
        checkOpen();
        queue.enqueueFillBuffer(cache, 0, cache.sizeInBytes(), 0);
        queue.enqueueFillBuffer(stateKv, 0, stateKv.sizeInBytes(), 0);
        queue.enqueueFillBuffer(stateScore, 0, stateScore.sizeInBytes(), 0);
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException(plan.kind() + " compressed cache of layer " + layer + " is closed");
        }
    }

    /** Releases the three buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        cache.close();
        stateKv.close();
        stateScore.close();
    }

    @Override
    public String toString() {
        return "DeviceCompressedKv{il=" + layer + ", " + plan + '}';
    }
}

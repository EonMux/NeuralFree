package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;

import java.util.Objects;

/**
 * Device KV cache of one owning layer: a K buffer of {@code rows * rowBytesK} bytes and, unless V aliases K (MLA
 * latent cache), a V buffer of {@code rows * rowBytesV} bytes, row-major {@code [row][nHeadKv * headDim]} with V not
 * transposed. Both are zero-filled once at allocation, so rows never written hold finite values. Shared-KV layers
 * receive the owner's instance and never close it.
 */
public final class DeviceKvLayer implements GpuKvLayer, AutoCloseable {

    private final int layer;
    private final KvLayout layout;
    private final int group;
    private final RotationPlan rotation;
    private final GpuBuffer k;
    private final GpuBuffer v;
    private boolean closed;

    private DeviceKvLayer(int layer, KvLayout layout, int group, RotationPlan rotation, GpuBuffer k, GpuBuffer v) {
        this.layer = layer;
        this.layout = layout;
        this.group = group;
        this.rotation = rotation;
        this.k = k;
        this.v = v;
    }

    /**
     * Allocates and zero-fills the cache of a layer.
     *
     * @param group KV group addressing the rows, or -1 for a cache written once at rows {@code 0..n-1}
     *              (cross-attention)
     * @param label what the cache is, for error messages
     * @throws com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException when a buffer exceeds the device limit
     */
    static DeviceKvLayer allocate(ComputeBackend backend, ComputeQueue queue, int layer, KvLayout layout, int group,
                                  RotationPlan rotation, String label) {
        Objects.requireNonNull(layout, "layout");
        Objects.requireNonNull(rotation, "rotation");
        GpuBuffer k = DeviceModelMemory.allocateZeroed(backend, queue, layout.kBytes(), label + " K");
        GpuBuffer v = k;
        if (!layout.vAliasesK()) {
            try {
                v = DeviceModelMemory.allocateZeroed(backend, queue, layout.vBytes(), label + " V");
            } catch (RuntimeException e) {
                k.close();
                throw e;
            }
        }
        return new DeviceKvLayer(layer, layout, group, rotation, k, v);
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public KvLayout layout() {
        return layout;
    }

    @Override
    public int group() {
        return group;
    }

    @Override
    public RotationPlan rotation() {
        return rotation;
    }

    @Override
    public GpuBuffer k() {
        checkOpen();
        return k;
    }

    @Override
    public GpuBuffer v() {
        checkOpen();
        return v;
    }

    /** Applies a context shift's row moves to K and (when separate) V. */
    void moveRows(ComputeQueue queue, DeviceStepInputs.KvShift shift, GpuBuffer staging) {
        checkOpen();
        moveRows(queue, k, layout.rowBytesK(), shift, staging);
        if (!layout.vAliasesK()) {
            moveRows(queue, v, layout.rowBytesV(), shift, staging);
        }
    }

    /** Largest row size of this cache, for sizing the shift staging area. */
    long maxRowBytes() {
        return Math.max(layout.rowBytesK(), layout.rowBytesV());
    }

    /**
     * Executes the row moves of {@code shift} on {@code buf} (rows of {@code rowBytes}): downward moves in place in
     * ascending chunks of at most {@code src - dst} rows; otherwise through {@code staging}, which must hold
     * {@code shift.movedRows() * rowBytes} bytes. Copies are enqueued on {@code queue} in order.
     */
    static void moveRows(ComputeQueue queue, GpuBuffer buf, long rowBytes, DeviceStepInputs.KvShift shift,
                         GpuBuffer staging) {
        if (shift.runCount() == 0) {
            return;
        }
        if (!shift.needsStaging()) {
            for (int i = 0; i < shift.runCount(); i++) {
                int src = shift.runSource(i);
                int dst = shift.runTarget(i);
                int n = shift.runRows(i);
                int gap = src - dst;
                for (int off = 0; off < n; off += gap) {
                    int c = Math.min(gap, n - off);
                    queue.enqueueCopyBuffer(buf, (src + off) * rowBytes, buf, (dst + off) * rowBytes, c * rowBytes);
                }
            }
            return;
        }
        if (staging == null || staging.sizeInBytes() < shift.movedRows() * rowBytes) {
            throw new IllegalArgumentException("context shift staging of "
                    + (staging == null ? 0 : staging.sizeInBytes()) + " bytes cannot hold "
                    + shift.movedRows() + " rows of " + rowBytes + " bytes");
        }
        long at = 0;
        for (int i = 0; i < shift.runCount(); i++) {
            long bytes = shift.runRows(i) * rowBytes;
            queue.enqueueCopyBuffer(buf, shift.runSource(i) * rowBytes, staging, at, bytes);
            at += bytes;
        }
        at = 0;
        for (int i = 0; i < shift.runCount(); i++) {
            long bytes = shift.runRows(i) * rowBytes;
            queue.enqueueCopyBuffer(staging, at, buf, shift.runTarget(i) * rowBytes, bytes);
            at += bytes;
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("KV cache of layer " + layer + " is closed");
        }
    }

    /** Releases K and V. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        k.close();
        if (v != k) {
            v.close();
        }
    }

    @Override
    public String toString() {
        return "DeviceKvLayer{il=" + layer + ", group=" + group + ", " + layout + ", " + rotation + '}';
    }
}

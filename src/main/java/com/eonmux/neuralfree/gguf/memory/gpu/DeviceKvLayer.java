package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;

import java.util.Objects;

/**
 * Device KV cache of one owning layer: a K buffer of {@code rows * rowBytesK} bytes and, unless V aliases K (MLA
 * latent cache), a V buffer of {@code rows * rowBytesV} bytes, row-major {@code [row][nHeadKv * headDim]} with V not
 * transposed. A per-token-head cache also owns one F32 scale buffer holding the K and V scale rows of every cache row
 * interleaved ({@code [row0 K heads, row0 V heads, row1 K heads, ...]}), written by the store kernels and read by
 * attention through the {@code kv_scales} binding. A vendor cache owns one side buffer instead: the checkpoint KV
 * scales and the Lloyd-Max table of its K stream, then the packed side planes of its rows
 * ({@link DeviceKvFormats#sideBytes}). Every buffer is zero-filled once at allocation (the side header is uploaded),
 * so rows never written hold finite values. Shared-KV layers receive the owner's instance and never close it.
 */
public final class DeviceKvLayer implements GpuKvLayer, AutoCloseable {

    private final int layer;
    private final KvLayout layout;
    private final int group;
    private final RotationPlan rotation;
    private final GpuBuffer k;
    private final GpuBuffer v;
    private final GpuBuffer scales;
    private final GpuBuffer vendorSide;
    private boolean closed;

    private DeviceKvLayer(int layer, KvLayout layout, int group, RotationPlan rotation, GpuBuffer k, GpuBuffer v,
                          GpuBuffer scales, GpuBuffer vendorSide) {
        this.layer = layer;
        this.layout = layout;
        this.group = group;
        this.rotation = rotation;
        this.k = k;
        this.v = v;
        this.scales = scales;
        this.vendorSide = vendorSide;
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
        DeviceKvFormats.requireLayout(layout, label);
        GpuBuffer k = DeviceModelMemory.allocateZeroed(backend, queue, layout.kBytes(), label + " K");
        GpuBuffer v = k;
        GpuBuffer scales = null;
        GpuBuffer vendorSide = null;
        try {
            scales = allocateScales(backend, queue, layout, label + " scales");
            vendorSide = allocateVendorSide(backend, queue, layout, label + " side");
            if (!layout.vAliasesK()) {
                v = DeviceModelMemory.allocateZeroed(backend, queue, layout.vBytes(), label + " V");
            }
        } catch (RuntimeException e) {
            k.close();
            closeAll(scales, vendorSide);
            if (v != k) {
                v.close();
            }
            throw e;
        }
        return new DeviceKvLayer(layer, layout, group, rotation, k, v, scales, vendorSide);
    }

    private static void closeAll(GpuBuffer... buffers) {
        for (GpuBuffer b : buffers) {
            if (b != null) {
                b.close();
            }
        }
    }

    /**
     * The side buffer of a vendor cache: the checkpoint scales and Lloyd-Max table of its header, then the packed
     * plane rows of K and (separate) V. Null for every other cache type.
     */
    private static GpuBuffer allocateVendorSide(ComputeBackend backend, ComputeQueue queue, KvLayout layout,
                                                String label) {
        if (!layout.dtypeK().vendorFormat() && (layout.vAliasesK() || !layout.dtypeV().vendorFormat())) {
            return null;
        }
        GpuBuffer b = DeviceModelMemory.allocateZeroed(backend, queue, DeviceKvFormats.sideBytes(layout), label);
        float[] header = DeviceKvFormats.sideHeader(layout);
        try {
            com.eonmux.neuralfree.acceleration.backend.GpuEvent e = queue.enqueueWriteBuffer(b, 0,
                    (long) header.length * Float.BYTES, header);
            if (e != null) {
                e.close();
            }
        } catch (RuntimeException e) {
            b.close();
            throw e;
        }
        return b;
    }

    /**
     * The interleaved F32 scale buffer of a per-token-head K or V cache ({@code rows * 2 * nHeadKv} values), else
     * null.
     *
     * @throws IllegalArgumentException when a side plane is not the single per-head scale plane
     */
    private static GpuBuffer allocateScales(ComputeBackend backend, ComputeQueue queue, KvLayout layout,
                                            String label) {
        if (!layout.dtypeK().perTokenHead() && !layout.dtypeV().perTokenHead()) {
            return null;
        }
        if (!layout.dtypeK().perTokenHead() || (!layout.vAliasesK() && !layout.dtypeV().perTokenHead())) {
            throw new IllegalArgumentException(label + ": a per-token-head cache needs both sides of that type, got "
                    + layout.dtypeK() + " and " + layout.dtypeV());
        }
        long bytes = (long) layout.rows() * 2 * layout.nHeadKv() * Float.BYTES;
        return DeviceModelMemory.allocateZeroed(backend, queue, bytes, label);
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

    @Override
    public GpuBuffer rowScales() {
        checkOpen();
        return scales;
    }

    @Override
    public GpuBuffer vendorSide() {
        checkOpen();
        return vendorSide;
    }

    /** Bytes of one side-buffer row of this cache: the K plane row then the V plane row, if separate. */
    int sideRowBytes() {
        return DeviceKvFormats.sideRowBytes(layout);
    }

    /** Applies a context shift's row moves to K and (when separate) V. */
    void moveRows(ComputeQueue queue, KvShift shift, GpuBuffer staging) {
        checkOpen();
        moveRows(queue, k, layout.rowBytesK(), shift, staging);
        if (!layout.vAliasesK()) {
            moveRows(queue, v, layout.rowBytesV(), shift, staging);
        }
        if (scales != null) {
            moveRows(queue, scales, scaleRowBytes(), shift, staging);
        }
    }

    /** Largest row size of this cache (code or scale rows), for sizing the shift staging area. */
    long maxRowBytes() {
        long bytes = Math.max(layout.rowBytesK(), layout.rowBytesV());
        if (scales != null) {
            bytes = Math.max(bytes, scaleRowBytes());
        }
        bytes = Math.max(bytes, sideRowBytes());
        return bytes;
    }

    /** Floats between consecutive rows of the interleaved scale buffer: K heads then V heads. */
    private long scaleRowBytes() {
        return 2L * layout.nHeadKv() * Float.BYTES;
    }

    /**
     * Executes the row moves of {@code shift} on {@code buf} (rows of {@code rowBytes}): downward moves in place in
     * ascending chunks of at most {@code src - dst} rows; otherwise through {@code staging}, which must hold
     * {@code shift.movedRows() * rowBytes} bytes. Copies are enqueued on {@code queue} in order.
     */
    static void moveRows(ComputeQueue queue, GpuBuffer buf, long rowBytes, KvShift shift,
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
        if (scales != null) {
            scales.close();
        }
        if (vendorSide != null) {
            vendorSide.close();
        }
    }

    @Override
    public String toString() {
        return "DeviceKvLayer{il=" + layer + ", group=" + group + ", " + layout + ", " + rotation + '}';
    }
}

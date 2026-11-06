package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvFormats;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The device form of {@link DflashKvCache}: per layer K and V rows of any {@link KvDType} in device
 * buffers and a device {@code kv_pos} record, the rows the flash kernels attend ({@link #layer}, {@link #kvPos}). The
 * row geometry is the host cache's: a full-attention layer stores position {@code p} in row {@code p}, a
 * sliding-window layer in row {@code p % ringRows} ({@link DflashKvCache#ringRows}). {@link #place} records the
 * positions of the rows a call is about to write in every layer (host mirror and device record) and stages their
 * cell indices; {@link #store} writes one layer's K and V rows to those cells ({@code LLM_KV_Store_<type>}). A
 * position becomes visible to the attention at its {@link #place}; every kernel of a call runs in queue order, so a
 * layer attends only after its own store. A quantized cache carries the Hadamard rotation of {@link DflashKvCache}
 * ({@code neuralfree.kv.rotation}): every layer's {@link GpuKvLayer#rotation()} plan names the Q/K and V blocks the
 * stack rotates before a store and the attention output back ({@code GpuQuantizedKvRotation}), and {@link
 * GpuDflashBackbone#refusal} refuses a quantized type whose head size is not a multiple of 32 by name. Not
 * thread-safe.
 */
final class GpuDflashKvCache implements DflashCache, AutoCloseable {

    /** One layer's device rows, the {@link GpuKvLayer} view the flash attention binds. */
    private static final class Layer implements GpuKvLayer {
        final int il;
        final KvLayout layout;
        final RotationPlan rotation;
        final GpuBuffer k;
        final GpuBuffer v;
        final GpuBuffer kvPos;
        final int[] rowPos;
        final boolean ring;
        int rowEnd;

        Layer(int il, KvLayout layout, RotationPlan rotation, GpuBuffer k, GpuBuffer v, GpuBuffer kvPos,
              boolean ring) {
            this.il = il;
            this.layout = layout;
            this.rotation = rotation;
            this.k = k;
            this.v = v;
            this.kvPos = kvPos;
            this.rowPos = new int[layout.rows()];
            Arrays.fill(rowPos, -1);
            this.ring = ring;
        }

        @Override public int layer() { return il; }
        @Override public KvLayout layout() { return layout; }
        @Override public int group() { return 0; }
        @Override public RotationPlan rotation() { return rotation; }
        @Override public GpuBuffer k() { return k; }
        @Override public GpuBuffer v() { return v; }
    }

    private final GpuOps ops;
    private final ComputeQueue queue;
    private final KvDType dtype;
    private final int capacity;
    private final int maxRows;
    private final Layer[] layers;
    private final KvWindow[] windows;
    private final GpuBuffer fullCells;
    private final GpuBuffer ringCells;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final ByteBuffer staging;
    private int placedRows;
    private boolean closed;

    /**
     * @param capacity positions a full-attention layer holds (the target context plus one block)
     * @param rotate   apply the Hadamard rotation of quantized rows ({@code neuralfree.kv.rotation})
     * @param maxRows  most rows one {@link #place} records (the larger of an injection chunk and a block)
     * @throws IllegalArgumentException for non-positive sizes or a row width the type cannot store
     */
    GpuDflashKvCache(ComputeBackend backend, GpuOps ops, int capacity, int blockSize, int nSwa, boolean[] swa,
                     boolean causal, int[] nHeadKv, int headDim, KvDType dtype, boolean rotate, int maxRows) {
        if (capacity <= 0 || blockSize <= 0 || maxRows <= 0) {
            throw new IllegalArgumentException("dflash device KV cache needs positive capacity, block and rows, got "
                    + capacity + "/" + blockSize + "/" + maxRows);
        }
        DeviceKvFormats.requireKernel(Objects.requireNonNull(dtype, "dtype"), "the DFlash draft KV cache");
        this.ops = ops;
        this.queue = ops.queue();
        this.dtype = dtype;
        this.capacity = capacity;
        this.maxRows = maxRows;
        RotationPlan plan = RotationPlan.of(dtype, dtype, headDim, headDim, headDim, headDim, rotate, false);
        this.layers = new Layer[swa.length];
        this.windows = new KvWindow[swa.length];
        this.staging = ByteBuffer.allocate(Math.max(maxRows, 1) * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try {
            this.fullCells = allocate(backend, (long) maxRows * Integer.BYTES);
            this.ringCells = allocate(backend, (long) maxRows * Integer.BYTES);
            for (int il = 0; il < swa.length; il++) {
                boolean windowed = swa[il] && nSwa > 0;
                windows[il] = KvWindow.of(windowed ? SwaType.STANDARD : SwaType.NONE, windowed ? nSwa : 0, causal);
                int rows = windowed ? DflashKvCache.ringRows(capacity, nSwa, blockSize) : capacity;
                int width = nHeadKv[il] * headDim;
                KvLayout layout = KvLayout.of(rows, false, width, width, dtype, dtype, windows[il]);
                GpuBuffer k = allocate(backend, layout.kBytes());
                GpuBuffer v = allocate(backend, layout.vBytes());
                GpuBuffer pos = allocate(backend, (long) rows * Integer.BYTES);
                ops.fill(pos, 0, (long) rows * Integer.BYTES, -1);
                layers[il] = new Layer(il, layout, plan, k, v, pos, windowed && rows < capacity);
            }
        } catch (RuntimeException e) {
            RuntimeException f = Closeables.closeAll(owned, null);
            if (f != null) {
                e.addSuppressed(f);
            }
            throw e;
        }
    }

    private GpuBuffer allocate(ComputeBackend backend, long bytes) {
        GpuBuffer b = backend.allocateBuffer(Math.max(bytes, Integer.BYTES));
        owned.add(b);
        return b;
    }

    int nLayer() {
        return layers.length;
    }

    /** The device rows of layer {@code il}. */
    GpuKvLayer layer(int il) {
        return layers[il];
    }

    /** The device position record of layer {@code il}, -1 for an empty row. */
    GpuBuffer kvPos(int il) {
        return layers[il].kvPos;
    }

    /** Visibility rule of layer {@code il}: its window and the block's causal flag. */
    KvWindow window(int il) {
        return windows[il];
    }

    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public boolean fits(int pos, int n) {
        return pos >= 0 && (long) pos + n <= capacity;
    }

    /**
     * Records positions {@code pos0 .. pos0 + n - 1} in every layer and stages their cells for {@link #store}.
     *
     * @throws IllegalArgumentException when {@code n} exceeds the rows one call records or the positions do not fit
     */
    void place(int pos0, int n) {
        ensureOpen();
        if (n < 1 || n > maxRows || !fits(pos0, n)) {
            throw new IllegalArgumentException("dflash device KV cache cannot place " + n + " rows at " + pos0
                    + " (at most " + maxRows + " rows, " + capacity + " positions)");
        }
        writeCells(fullCells, pos0, n, capacity);
        int ringSize = -1;
        for (Layer l : layers) {
            if (l.ring) {
                ringSize = l.layout.rows();
            }
        }
        if (ringSize > 0) {
            writeCells(ringCells, pos0, n, ringSize);
        }
        for (Layer l : layers) {
            recordPositions(l, pos0, n);
        }
        placedRows = n;
    }

    /**
     * Stages the cells of the placed rows in a layer geometry of {@code rows} rows; of a placement longer than a
     * ring only the last {@code rows} positions keep a row (the earlier ones would be overwritten in order, as on
     * the host), so the others get cell -1, which the store skips.
     */
    private void writeCells(GpuBuffer cells, int pos0, int n, int rows) {
        int skip = Math.max(0, n - rows);
        staging.clear();
        for (int t = 0; t < n; t++) {
            staging.putInt(t < skip ? -1 : (pos0 + t) % rows);
        }
        upload(cells, 0, n);
    }

    /** Host mirror and device record of the placed positions: one contiguous write, or two around a ring wrap. */
    private void recordPositions(Layer l, int pos0, int n) {
        int rows = l.layout.rows();
        int start = Math.max(pos0, pos0 + n - rows);
        int count = pos0 + n - start;
        int first = start % rows;
        int head = Math.min(count, rows - first);
        for (int t = 0; t < count; t++) {
            l.rowPos[(start + t) % rows] = start + t;
        }
        l.rowEnd = l.ring ? rows : Math.max(l.rowEnd, pos0 + n);
        uploadPositions(l, first, head);
        if (head < count) {
            uploadPositions(l, 0, count - head);
        }
    }

    /**
     * Writes the {@code n} placed K rows of {@code k} (and V rows of {@code v}) from element offset {@code kOff}
     * ({@code vOff}), one row of the layer's width apart, into layer {@code il}.
     */
    void store(int il, GpuBuffer k, int kOff, GpuBuffer v, int vOff) {
        ensureOpen();
        Layer l = layers[il];
        GpuBuffer cells = l.ring ? ringCells : fullCells;
        int width = l.layout.widthK();
        storeRows(k, kOff, width, cells, l.k);
        storeRows(v, vOff, width, cells, l.v);
    }

    private void storeRows(GpuBuffer src, int srcOff, int width, GpuBuffer cells, GpuBuffer dst) {
        Object[] args = RopeKernels.kvStoreArgs(dtype, placedRows, width, width, srcOff, width, width, src, cells,
                ops.placeholders().kvScalesOnes(), dst);
        long items = RopeKernels.kvStoreWorkItems(dtype, placedRows, width);
        int local = ops.local();
        ops.enqueue(RopeKernels.kvStore(dtype), new long[]{roundUp(items, local)}, new long[]{local}, args);
    }

    @Override
    public void trim(int pos) {
        ensureOpen();
        int bound = Math.max(0, pos);
        for (Layer l : layers) {
            int start = l.ring ? 0 : bound;
            int end = l.ring ? l.layout.rows() : l.rowEnd;
            int lo = -1;
            int hi = -1;
            for (int r = start; r < end; r++) {
                if (l.rowPos[r] >= bound) {
                    l.rowPos[r] = -1;
                    lo = lo < 0 ? r : lo;
                    hi = r;
                }
            }
            if (!l.ring) {
                l.rowEnd = Math.min(l.rowEnd, bound);
            }
            if (lo >= 0) {
                uploadPositions(l, lo, hi - lo + 1);
            }
        }
    }

    @Override
    public void clear() {
        trim(0);
    }

    /** Uploads the host record of rows {@code [row0, row0 + n)} of {@code l}, in chunks of the staging size. */
    private void uploadPositions(Layer l, int row0, int n) {
        for (int off = 0; off < n; off += maxRows) {
            int count = Math.min(maxRows, n - off);
            staging.clear();
            for (int t = 0; t < count; t++) {
                staging.putInt(l.rowPos[row0 + off + t]);
            }
            upload(l.kvPos, row0 + off, count);
        }
    }

    private void upload(GpuBuffer dst, int intOffset, int count) {
        GpuEvent e = queue.enqueueWriteBuffer(dst, (long) intOffset * Integer.BYTES, (long) count * Integer.BYTES,
                staging.array());
        if (e != null) {
            e.close();
        }
    }

    /** Device bytes of a cache of these arguments: the K, V and position rows of every layer and the cell buffers. */
    static long bytes(int capacity, int blockSize, int nSwa, boolean[] swa, int[] nHeadKv, int headDim,
                      KvDType dtype, int maxRows) {
        long total = 2L * maxRows * Integer.BYTES;
        for (int il = 0; il < swa.length; il++) {
            int rows = swa[il] && nSwa > 0 ? DflashKvCache.ringRows(capacity, nSwa, blockSize) : capacity;
            KvLayout layout = KvLayout.of(rows, false, nHeadKv[il] * headDim, nHeadKv[il] * headDim, dtype, dtype,
                    KvWindow.FULL_CAUSAL);
            total = Math.addExact(total, Math.addExact(layout.totalBytes(), (long) rows * Integer.BYTES));
        }
        return total;
    }

    private static long roundUp(long n, int local) {
        return (n + local - 1) / local * local;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("dflash device KV cache is closed");
        }
    }

    /** Releases every device buffer. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException f = Closeables.closeAll(owned, null);
        owned.clear();
        if (f != null) {
            throw f;
        }
    }
}

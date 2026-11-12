package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader.TensorPlan;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.quant.W12Codec;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The row-range copier of a {@link GpuWeightUploader}, over one weight store: reusable host staging, W12 scan and
 * fill passes and the pending-write window that bounds the host memory a backend may hold for unfinished writes.
 */
final class WeightRowCopier {
    /** Staging chunks a queue may hold in pending writes before the copier finishes it. */
    static final int IN_FLIGHT_CHUNKS = 8;
    /** Bytes of one slot of the page-locked upload ring. */
    static final int RING_SLOT_BYTES = 32 << 20;
    /**
     * Least raw bytes an upload call moves for it to open the page-locked upload ring. Opening and closing the ring
     * (its page-locked slots and fill threads) costs about 20 ms on CUDA and the ring saves about 0.03 ms per MiB of
     * mapped bytes over the queue's writes, so it pays from about 640 MiB on; below this bound the ring costs at most
     * a few milliseconds more than it saves.
     */
    static final long RING_MIN_BYTES = 512L << 20;

    private final WeightStore store;
    private final int stagingBytes;
    private final W12F16Mode w12F16;
    private final long inFlightLimit;
    private final Map<ComputeQueue, long[]> pending = new IdentityHashMap<>();
    /** The backend of the page-locked upload ring, or null to write every chunk through the queue. */
    private final ComputeBackend ringBackend;
    /** The queue of {@link #ringBackend} whose uploads may take the ring; other queues write their chunks. */
    private final ComputeQueue ringQueue;
    private PinnedUploadRing ring;
    private boolean ringOpened;
    /** Whether the uploads until the next {@link #release()} move enough raw bytes to use the ring. */
    private boolean ringWanted;
    private ByteBuffer staging;
    private float[] floats;
    private float[] verify;
    private ByteBuffer w12;

    WeightRowCopier(WeightStore store, int stagingBytes, W12F16Mode w12F16) {
        this(store, stagingBytes, w12F16, null, null);
    }

    /**
     * A copier whose raw chunks of mapped tensor data enqueued on {@code ringQueue} may go through a
     * {@link PinnedUploadRing} on {@code ringBackend}, the backend that owns {@code ringQueue} (see
     * {@link #planRing}); a null {@code ringBackend} writes every chunk through its queue.
     *
     * @throws NullPointerException when {@code ringBackend} is given without its queue
     */
    WeightRowCopier(WeightStore store, int stagingBytes, W12F16Mode w12F16, ComputeBackend ringBackend,
                    ComputeQueue ringQueue) {
        if (ringBackend != null) {
            Objects.requireNonNull(ringQueue, "ringQueue");
        }
        this.ringBackend = ringBackend;
        this.ringQueue = ringQueue;
        this.store = store;
        this.stagingBytes = stagingBytes;
        this.w12F16 = Objects.requireNonNull(w12F16, "w12F16");
        this.inFlightLimit = (long) IN_FLIGHT_CHUNKS * stagingBytes;
    }

    /**
     * Declares the raw bytes the upload calls until the next {@link #release()} move: at least
     * {@link #RING_MIN_BYTES} lets their mapped chunks on the ring queue take the upload ring, which opens with the
     * first such chunk; fewer bytes write every chunk through its queue.
     */
    void planRing(long rawBytes) {
        ringWanted = ringBackend != null && rawBytes >= RING_MIN_BYTES;
    }

    /** Declares that {@code q} has been finished: none of its writes are pending any more. */
    void synced(ComputeQueue q) {
        pending.remove(q);
    }

    /** Accounts {@code bytes} of writes enqueued on {@code q}; finishes {@code q} when the window is full. */
    private void enqueued(ComputeQueue q, long bytes) {
        long[] held = pending.computeIfAbsent(q, k -> new long[1]);
        held[0] += bytes;
        if (held[0] >= inFlightLimit) {
            q.finish();
            held[0] = 0;
        }
    }

    /**
     * Enqueues the upload of rows {@code [firstRow, firstRow + nRows)} of {@code p} to {@code dst} at byte
     * {@code dstOffset}. The host staging is reusable when this returns (writes copy at enqueue). A W12 plan
     * holds its rows to {@code w12F16} (see {@link #decodeW12Chunk}) and feeds {@code moved} when rounding.
     */
    void copyRows(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset, ComputeQueue q,
                  W12Rounding moved) {
        if (p.deviceType() == GGMLType.W12) {
            copyEncoded(p, firstRow, nRows, dst, dstOffset, q, moved);
        } else if (p.needsDecode()) {
            copyDecoded(p, firstRow, nRows, dst, dstOffset, q);
        } else {
            copyRaw(p, firstRow * p.sourceRowBytes(), nRows * p.sourceRowBytes(), dst, dstOffset, q);
        }
    }

    /**
     * Uploads stored bytes as they are, in chunks of the staging size. A store that maps its data hands each chunk's
     * mapped pages over directly ({@link WeightStore#mappedSlice}); a chunk ends early at a mapping boundary. While
     * the upload ring is planned ({@link #planRing}) and {@code q} is the ring's queue, such a chunk is copied into
     * the page-locked ring by several threads and reaches the device as one device copy ({@link PinnedUploadRing});
     * otherwise the queue's write takes it. A queue of another backend never takes the ring: its device cannot read
     * the ring backend's slots. Without a mapping each chunk is read into the host staging first.
     */
    private void copyRaw(TensorPlan p, long srcOffset, long bytes, GpuBuffer dst, long dstOffset,
                         ComputeQueue q) {
        PinnedUploadRing pinned = ring(q);
        long chunk = pinned != null ? Math.min(stagingBytes, pinned.slotBytes()) : stagingBytes;
        long done = 0;
        while (done < bytes) {
            long want = Math.min(bytes - done, chunk);
            ByteBuffer src = mapped(p, srcOffset + done, want);
            int n;
            if (src != null && pinned != null) {
                n = src.capacity();
                pinned.upload(src, dst, dstOffset + done, q);
                done += n;
                continue;
            }
            if (src != null) {
                n = src.capacity();
            } else {
                src = staging(stagingBytes);
                n = (int) want;
                src.clear();
                read(p, srcOffset + done, n, src);
            }
            closeEvent(q.enqueueWriteBuffer(dst, dstOffset + done, n, src));
            enqueued(q, n);
            done += n;
        }
    }

    /**
     * Bytes {@code [byteOffset, byteOffset + n)} of the plan's stored tensor as a buffer whose index 0 is the first
     * byte: the store's mapped view when it holds the whole range, else the host staging filled by a read.
     */
    private ByteBuffer chunk(TensorPlan p, long byteOffset, int n) {
        ByteBuffer src = mapped(p, byteOffset, n);
        if (src != null && src.capacity() == n) {
            return src;
        }
        ByteBuffer buf = staging((int) Math.max(stagingBytes, n));
        buf.clear();
        read(p, byteOffset, n, buf);
        buf.flip();
        return buf;
    }

    /** The store's mapped view of the tensor's bytes from {@code byteOffset}, at most {@code maxBytes}; or null. */
    private ByteBuffer mapped(TensorPlan p, long byteOffset, long maxBytes) {
        try {
            return store.mappedSlice(p.info().offset() + byteOffset, maxBytes);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot map " + maxBytes + " bytes of tensor " + p.name() + " at byte "
                    + byteOffset, e);
        }
    }

    private void copyDecoded(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset,
                             ComputeQueue q) {
        long srcRow = p.sourceRowBytes();
        long cols = p.cols();
        long rowsPerChunk = Math.max(1, Math.min(stagingBytes / srcRow, stagingBytes / (cols * 4L)));
        long chunkFloats = rowsPerChunk * cols;
        if (rowsPerChunk * srcRow > Integer.MAX_VALUE || chunkFloats > Integer.MAX_VALUE - 8) {
            throw new IllegalStateException("tensor " + p.name() + ": one row of " + cols
                    + " elements exceeds the host staging");
        }
        float[] out = floats((int) chunkFloats);
        long done = 0;
        while (done < nRows) {
            long r = Math.min(rowsPerChunk, nRows - done);
            int n = (int) (r * srcRow);
            ByteBuffer src = chunk(p, (firstRow + done) * srcRow, n);
            CpuBlockDecoders.decodeElements(p.sourceType(), src, 0L, out, 0,
                    r * cols);
            closeEvent(q.enqueueWriteBuffer(dst, dstOffset + done * cols * 4L, r * cols * 4L, out));
            enqueued(q, r * cols * 4L);
            done += r;
        }
    }

    /**
     * Uploads rows {@code [firstRow, firstRow + nRows)} of a W12 device plan: sizes the rows' escape
     * tables by a scan pass (see {@link #scanW12}), then fills the per-row records (see {@link #fillW12}).
     * F16 source rows are held to {@code w12F16}, what the records store (see {@link #decodeW12Chunk}).
     */
    private void copyEncoded(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset,
                             ComputeQueue q, W12Rounding moved) {
        long[] escapes = new long[2];
        scanW12(p, firstRow, nRows, null, false, escapes);
        fillW12(p, firstRow, nRows, dst, dstOffset, escapes[0], escapes[1], null, false, q, moved);
    }

    /** Host staging rows of one W12 chunk: bounded by the store read and the decoded floats. */
    private long w12RowsPerChunk(TensorPlan p) {
        long srcRow = p.sourceRowBytes();
        long cols = p.cols();
        long rowsPerChunk = Math.max(1, Math.min(stagingBytes / srcRow, stagingBytes / (cols * 4L)));
        long chunkFloats = rowsPerChunk * cols;
        if (rowsPerChunk * srcRow > Integer.MAX_VALUE || chunkFloats > Integer.MAX_VALUE - 8) {
            throw new IllegalStateException("tensor " + p.name() + ": one row of " + cols
                    + " elements exceeds the host staging");
        }
        return rowsPerChunk;
    }

    /**
     * Decodes rows {@code [firstRow + done, firstRow + done + r)} of the W12 plan into {@code out}
     * row-major: from the weight store, or from {@code update} (its stored type when {@code decode},
     * else little-endian F32). BF16 sources are on the bfloat16 grid and skip the pass. F16 rows from
     * the store are held to {@code w12F16}: {@code refuse} throws {@link IllegalArgumentException} at
     * the first weight off the grid, naming the tensor, the element and
     * {@link SpeculativeOptions#PROP_W12_F16}; {@code round} brings every weight onto the grid
     * ({@link W12Codec#bf16Grid}, round-to-nearest-even — each off-grid weight loses its 3 low mantissa
     * bits) and counts those that moved in {@code moved}. Update data ({@code update != null}) always
     * rounds.
     */
    private void decodeW12Chunk(TensorPlan p, long firstRow, long done, long r, float[] out,
                                ByteBuffer update, boolean decode, W12Rounding moved) {
        int n = (int) (r * p.cols());
        boolean onGrid = p.sourceType() == GGMLType.BF16 && (update == null || decode);
        if (update == null) {
            ByteBuffer src = chunk(p, (firstRow + done) * p.sourceRowBytes(), (int) (r * p.sourceRowBytes()));
            CpuBlockDecoders.decodeElements(p.sourceType(), src, 0L, out, 0, n);
        } else if (decode) {
            CpuBlockDecoders.decodeElements(p.sourceType(), update,
                    (firstRow + done) * p.sourceRowBytes(), out, 0, n);
        } else {
            ByteBuffer chunk = update.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            long from = (firstRow + done) * p.cols() * 4L;
            chunk.position((int) from).limit((int) (from + n * 4L));
            chunk.asFloatBuffer().get(out, 0, n);
        }
        if (!onGrid) {
            if (update == null && w12F16 == W12F16Mode.REFUSE) {
                for (int i = 0; i < n; i++) {
                    float v = out[i];
                    if (!W12Codec.isOnBf16Grid(v)) {
                        throw new IllegalArgumentException(p.name() + ": row " + (firstRow + done + i / p.cols())
                                + " element " + (i % p.cols()) + " is not a bfloat16 value (f32 bits 0x"
                                + Integer.toHexString(Float.floatToRawIntBits(v)) + "); " + GGMLType.W12
                                + " stores bfloat16-representable weights losslessly only; set "
                                + SpeculativeOptions.PROP_W12_F16 + "=round to bring F16 weights onto the"
                                + " bfloat16 grid (lossy: each off-grid weight loses its 3 low mantissa bits)");
                    }
                }
                return;
            }
            for (int i = 0; i < n; i++) {
                float v = out[i];
                float grid = W12Codec.bf16Grid(v);
                out[i] = grid;
                if (moved != null) {
                    moved.add(v, grid);
                }
            }
        }
    }

    /**
     * Sums the exact and Q escape entries of rows {@code [firstRow, firstRow + nRows)} of the W12 plan into
     * {@code totals[0]} / {@code totals[1]} over a decoded scan pass. A row range whose tables leave the
     * indexable range refuses here, as does an off-grid F16 store row under {@code w12F16 = refuse}
     * (see {@link #decodeW12Chunk}). The fill pass decodes the same rows again and reports the rounding
     * there, so this pass never counts.
     */
    void scanW12(TensorPlan p, long firstRow, long nRows, ByteBuffer update, boolean decode, long[] totals) {
        long cols = p.cols();
        int colsInt = (int) cols;
        long rowsPerChunk = w12RowsPerChunk(p);
        float[] out = floats((int) (rowsPerChunk * cols));
        totals[0] = 0L;
        totals[1] = 0L;
        long done = 0;
        while (done < nRows) {
            long r = Math.min(rowsPerChunk, nRows - done);
            decodeW12Chunk(p, firstRow, done, r, out, update, decode, null);
            for (int i = 0; i < (int) r; i++) {
                W12Codec.RowPlan plan = W12Codec.scanRow(p.name(), out, (int) (i * cols), colsInt,
                        (int) (firstRow + done + i));
                totals[0] += plan.nEscapes();
                totals[1] += plan.nQEscapes();
                if (totals[0] > (Integer.MAX_VALUE - 8L) / W12Codec.ESCAPE_ENTRY_BYTES
                        || totals[1] > (Integer.MAX_VALUE - 8L) / W12Codec.Q_ESCAPE_ENTRY_BYTES) {
                    throw new IllegalStateException("tensor " + p.name() + ": row " + (firstRow + done + i)
                            + " pushes the " + GGMLType.W12 + " record's escape tables past the indexable"
                            + " range");
                }
            }
            done += r;
        }
    }

    /**
     * Fills {@code dst} at byte {@code dstOffset} with the W12 record of rows {@code [firstRow,
     * firstRow + nRows)} of the plan: the row region, the exact escape table at {@code nRows * rowBytes},
     * then the Q escape table, whose sizes {@code nEscTotal} / {@code nEscQTotal} the caller scanned
     * (see {@link #scanW12}). Each chunk's staging record is verified bit-exactly against its decoded
     * floats before the loop moves on; a scan/fill disagreement refuses. A load pass reports the weights
     * its F16 grid rounding moved through {@code moved}; live updates pass null.
     */
    void fillW12(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset, long nEscTotal,
                 long nEscQTotal, ByteBuffer update, boolean decode, ComputeQueue q, W12Rounding moved) {
        long cols = p.cols();
        int colsInt = (int) cols;
        long stride = p.deviceRowBytes();
        int planes = W12Codec.planeBytes(colsInt);
        long escBase = nRows * stride;
        long qEscBase = escBase + nEscTotal * W12Codec.ESCAPE_ENTRY_BYTES;
        long rowsPerChunk = w12RowsPerChunk(p);
        int chunkFloats = (int) (rowsPerChunk * cols);
        float[] out = floats(chunkFloats);
        float[] verify = verifyScratch(chunkFloats);
        long escPos = 0L;
        long qEscPos = 0L;
        long done = 0L;
        while (done < nRows) {
            long r = Math.min(rowsPerChunk, nRows - done);
            int chunkRows = (int) r;
            decodeW12Chunk(p, firstRow, done, r, out, update, decode, moved);
            long chunkEsc = 0L;
            long chunkQEsc = 0L;
            for (int i = 0; i < chunkRows; i++) {
                W12Codec.RowPlan plan = W12Codec.scanRow(p.name(), out, (int) (i * cols), colsInt,
                        (int) (firstRow + done + i));
                chunkEsc += plan.nEscapes();
                chunkQEsc += plan.nQEscapes();
            }
            // the chunk's staging record: its rows, then the exact entries filled so far and the
            // chunk's own, then the Q entries; the running positions keep every metadata count global
            int rowSpan = chunkRows * (int) stride;
            int escStart = rowSpan;
            int qEscStart = escStart + (int) (chunkEsc * W12Codec.ESCAPE_ENTRY_BYTES);
            long cap = qEscStart + (long) chunkQEsc * W12Codec.Q_ESCAPE_ENTRY_BYTES;
            if (cap > Integer.MAX_VALUE - 8L) {
                throw new IllegalStateException("tensor " + p.name() + ": one " + GGMLType.W12
                        + " staging chunk of " + chunkRows + " rows escapes beyond the indexable range");
            }
            ByteBuffer rec = w12Staging((int) cap);
            ByteBuffer escView = rec.duplicate();
            escView.position(escStart).limit(qEscStart);
            ByteBuffer qView = rec.duplicate();
            qView.position(qEscStart).limit((int) cap);
            ByteBuffer escSlice = escView.slice();
            ByteBuffer qSlice = qView.slice();
            long escapePos = escPos;
            long qEscapePos = qEscPos;
            for (int i = 0; i < chunkRows; i++) {
                W12Codec.RowPlan plan = W12Codec.scanRow(p.name(), out, (int) (i * cols), colsInt,
                        (int) (firstRow + done + i));
                plan.writeRow(rec, i * (int) stride, escSlice, (int) escapePos, qSlice, (int) qEscapePos);
                escapePos += plan.nEscapes() * W12Codec.ESCAPE_ENTRY_BYTES;
                qEscapePos += plan.nQEscapes() * W12Codec.Q_ESCAPE_ENTRY_BYTES;
            }
            if (escapePos != escPos + chunkEsc * W12Codec.ESCAPE_ENTRY_BYTES
                    || qEscapePos != qEscPos + chunkQEsc * W12Codec.Q_ESCAPE_ENTRY_BYTES) {
                throw new IllegalStateException("tensor " + p.name() + ": the " + GGMLType.W12
                        + " fill pass escapes differently than the scan pass that sized its buffers");
            }
            // device: the chunk's rows, then its new entries in the shard's two tables
            writeSpan(q, dst, dstOffset + done * stride, rec, 0, rowSpan);
            if (chunkEsc > 0) {
                writeSpan(q, dst, dstOffset + escBase + escPos, rec, escStart,
                        (int) (chunkEsc * W12Codec.ESCAPE_ENTRY_BYTES));
            }
            if (chunkQEsc > 0) {
                writeSpan(q, dst, dstOffset + qEscBase + qEscPos, rec, qEscStart,
                        (int) (chunkQEsc * W12Codec.Q_ESCAPE_ENTRY_BYTES));
            }
            enqueued(q, rowSpan + chunkEsc * W12Codec.ESCAPE_ENTRY_BYTES
                    + chunkQEsc * W12Codec.Q_ESCAPE_ENTRY_BYTES);
            // the enqueue copied the record; make the counts chunk-local and verify bit-exactly
            int escBefore = (int) (escPos / W12Codec.ESCAPE_ENTRY_BYTES);
            int qEscBefore = (int) (qEscPos / W12Codec.Q_ESCAPE_ENTRY_BYTES);
            for (int i = 0; i < chunkRows; i++) {
                int meta = i * (int) stride + planes;
                rec.putInt(meta + 4, rec.getInt(meta + 4) - escBefore);
                rec.putInt(meta + 8, rec.getInt(meta + 8) - qEscBefore);
            }
            W12Codec.decode(rec, verify, chunkRows, colsInt);
            for (int i = 0; i < chunkFloats; i++) {
                if (Float.floatToRawIntBits(verify[i]) != Float.floatToRawIntBits(out[i])) {
                    throw new IllegalStateException(p.name() + ": the " + GGMLType.W12 + " roundtrip of row "
                            + (firstRow + done + i / colsInt) + " element " + (i % colsInt) + " gives "
                            + verify[i] + " for " + out[i] + ", which is not bit-exact");
                }
            }
            escPos += chunkEsc * W12Codec.ESCAPE_ENTRY_BYTES;
            qEscPos += chunkQEsc * W12Codec.Q_ESCAPE_ENTRY_BYTES;
            done += r;
        }
        if (escPos != nEscTotal * W12Codec.ESCAPE_ENTRY_BYTES
                || qEscPos != nEscQTotal * W12Codec.Q_ESCAPE_ENTRY_BYTES) {
            throw new IllegalStateException("tensor " + p.name() + ": the " + GGMLType.W12 + " fill pass"
                    + " ended with " + escPos + " exact and " + qEscPos + " Q escape bytes, the scanned"
                    + " tables hold " + nEscTotal * W12Codec.ESCAPE_ENTRY_BYTES + " and "
                    + nEscQTotal * W12Codec.Q_ESCAPE_ENTRY_BYTES);
        }
    }

    /** Enqueues the write of {@code [from, from + len)} of the direct staging {@code rec} to {@code dst}. */
    private void writeSpan(ComputeQueue q, GpuBuffer dst, long dstOffset, ByteBuffer rec, int from, int len) {
        ByteBuffer view = rec.duplicate();
        view.position(from).limit(from + len);
        closeEvent(q.enqueueWriteBuffer(dst, dstOffset, len, view.slice()));
    }

    /** Decodes the single element of a one-element tensor. */
    float readScalar(TensorPlan p) {
        int n = (int) p.sourceRowBytes();
        ByteBuffer buf = ByteBuffer.allocate(n);
        byte[] raw = new byte[n];
        try {
            store.readTensor(p.info().offset(), n, raw);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read tensor " + p.name(), e);
        }
        buf.put(raw).flip();
        float[] v = new float[(int) Math.max(1, p.cols())];
        CpuBlockDecoders.decodeElements(p.sourceType(), buf, 0L, v, 0,
                p.cols());
        return v[0];
    }

    private void read(TensorPlan p, long byteOffset, int n, ByteBuffer dst) {
        try {
            store.readTensorDirect(p.info().offset() + byteOffset, n, dst);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + n + " bytes of tensor " + p.name() + " at byte "
                    + byteOffset, e);
        }
    }

    private ByteBuffer staging(int capacity) {
        if (staging == null || staging.capacity() < capacity) {
            staging = ByteBuffer.allocateDirect(capacity).order(ByteOrder.LITTLE_ENDIAN);
        }
        return staging;
    }

    private float[] floats(int n) {
        if (floats == null || floats.length < n) {
            floats = new float[n];
        }
        return floats;
    }

    /** The roundtrip-verify scratch: a second array distinct from {@link #floats}. */
    private float[] verifyScratch(int n) {
        if (verify == null || verify.length < n) {
            verify = new float[n];
        }
        return verify;
    }

    /** The reusable little-endian W12 block staging of {@code capacity} bytes. */
    private ByteBuffer w12Staging(int capacity) {
        if (w12 == null || w12.capacity() < capacity) {
            w12 = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
        }
        w12.clear().limit(capacity);
        return w12;
    }

    /**
     * The upload ring for chunks enqueued on {@code q}, opened on first use: null when {@code q} is not the ring's
     * queue, the ring is not planned, or the backend has no ring ({@link PinnedUploadRing#open}).
     */
    private PinnedUploadRing ring(ComputeQueue q) {
        if (q != ringQueue || !ringWanted) {
            return null;
        }
        if (!ringOpened) {
            ringOpened = true;
            ring = PinnedUploadRing.open(ringBackend, RING_SLOT_BYTES);
        }
        return ring;
    }

    /**
     * Drops the host staging and the pending-write accounting (writes copy their host data at enqueue), and closes
     * the upload ring once the copies reading it completed; a later upload opens it again when planned.
     */
    void release() {
        PinnedUploadRing open = ring;
        ring = null;
        ringOpened = false;
        ringWanted = false;
        if (open != null) {
            open.close();
        }
        staging = null;
        floats = null;
        verify = null;
        w12 = null;
        pending.clear();
    }

    private static void closeEvent(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }
}

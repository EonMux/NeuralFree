package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Decision;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicy.Mode;
import com.eonmux.neuralfree.gguf.hf.store.SidecarDirectory.Sidecar;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * The derived tensors of a Hugging Face model source: the produced slots of {@code [P, end)} of the
 * {@link VirtualLayout}, each a {@link PlannedTensor} whose bytes come from its {@link RowSource}. The table is
 * immutable and shared; each reader reads it through its own {@link View}, which applies that reader's
 * {@link MaterializePolicy}. Thread-safe.
 *
 * <p>Production sizes. A row-local producer is read in calls of at most {@value #SCRATCH_BYTES} bytes (and at least
 * one row). A producer that is not row-local reads more than the requested rows of its inputs, so it is produced in
 * whole matrices (the innermost two dimensions): as many as fit the call's buffer ({@value #PIECE_BYTES} bytes of
 * direct memory for a copy kept in memory, {@value #SCRATCH_BYTES} bytes of heap for one written to disk), one matrix
 * when a matrix is larger, and, for a matrix above the largest buffer, the most rows a buffer holds. Each matrix is
 * therefore produced from one pass over its inputs whenever it fits one buffer.</p>
 */
public final class ProducedSegments {

    /** Largest scratch buffer of one read of a row-local producer. */
    static final long SCRATCH_BYTES = 64L << 20;
    /** Largest direct-memory piece of a materialized tensor whose matrices fit it. */
    static final long PIECE_BYTES = 1L << 30;
    private static final long MAX_BUFFER = Integer.MAX_VALUE - 8;

    private final List<PlannedTensor> tensors;
    private final long[] offsets;
    private final long base;
    private final long end;
    private final Map<String, Integer> byName;

    private ProducedSegments(List<PlannedTensor> tensors, long[] offsets, long base, long end) {
        this.tensors = tensors;
        this.offsets = offsets;
        this.base = base;
        this.end = end;
        Map<String, Integer> names = new HashMap<>();
        for (int i = 0; i < tensors.size(); i++) {
            names.put(tensors.get(i).ggufName(), i);
        }
        this.byName = Collections.unmodifiableMap(names);
    }

    /**
     * The table of {@code tensors} at {@code offsets} (same order, increasing), in {@code [base, end)}.
     *
     * @throws IllegalArgumentException when the lists differ in length, a slot lies outside {@code [base, end)} or
     *                                  overlaps the previous one
     */
    static ProducedSegments of(List<PlannedTensor> tensors, List<Long> offsets, long base, long end) {
        if (tensors.size() != offsets.size()) {
            throw new IllegalArgumentException(tensors.size() + " produced tensors but " + offsets.size()
                    + " offsets");
        }
        if (base < 0 || end < base) {
            throw new IllegalArgumentException("produced range [" + base + ", " + end + ") is invalid");
        }
        long[] offs = new long[offsets.size()];
        long prevEnd = base;
        for (int i = 0; i < offs.length; i++) {
            PlannedTensor t = Objects.requireNonNull(tensors.get(i), "tensor " + i);
            offs[i] = offsets.get(i);
            long tEnd = Math.addExact(offs[i], t.byteSize());
            if (offs[i] < prevEnd || tEnd > end) {
                throw new IllegalArgumentException(t.ggufName() + " at [" + offs[i] + ", " + tEnd + ") overlaps the"
                        + " previous slot (ending at " + prevEnd + ") or lies past the end " + end);
            }
            prevEnd = tEnd;
        }
        return new ProducedSegments(Collections.unmodifiableList(new ArrayList<>(tensors)), offs, base, end);
    }

    /** The produced tensors in slot order. */
    public List<PlannedTensor> tensors() {
        return tensors;
    }

    /** The number of produced tensors. */
    public int size() {
        return tensors.size();
    }

    /** The virtual offset of produced tensor {@code index}. */
    public long offset(int index) {
        return offsets[index];
    }

    /** The index of the produced tensor named {@code ggufName}, or -1. */
    public int indexOf(String ggufName) {
        Integer i = byName.get(ggufName);
        return i == null ? -1 : i;
    }

    /** {@code P}: the start of the produced range. */
    public long base() {
        return base;
    }

    /** The end of the produced range. */
    public long end() {
        return end;
    }

    /**
     * A reader's view: reads follow {@code policy} (which must decide for exactly these tensors) and DISK copies go to
     * {@code sidecars}.
     */
    public View view(MaterializePolicy policy, SidecarDirectory sidecars) {
        Objects.requireNonNull(policy, "policy");
        Objects.requireNonNull(sidecars, "sidecars");
        if (policy.size() != tensors.size()) {
            throw new IllegalArgumentException("policy decides " + policy.size() + " tensors, not "
                    + tensors.size());
        }
        return new View(policy, sidecars);
    }

    /** The index of the slot at or before {@code pos}; -1 before the first slot. */
    private int slotAt(long pos) {
        int lo = 0;
        int hi = offsets.length - 1;
        int found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (offsets[mid] <= pos) {
                found = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return found;
    }

    /**
     * Rows per call when producing {@code s} whole (see the class comment): row-local producers in calls of at most
     * {@value #SCRATCH_BYTES} bytes, others in whole matrices of at most {@code matrixTarget} bytes per call.
     */
    static long chunkRows(RowSource s, long matrixTarget) throws IOException {
        long rb = Math.max(1, s.rowBytes());
        long rows = Math.max(1, s.rows());
        if (rb > MAX_BUFFER) {
            throw new IOException(s.describe() + ": a row of " + rb + " bytes exceeds one buffer");
        }
        if (s.rowLocal()) {
            return Math.min(rows, Math.max(1, SCRATCH_BYTES / rb));
        }
        long[] shape = s.shape();
        long matrix = shape.length >= 2 ? Math.max(1, shape[shape.length - 2]) : rows;
        long matrixBytes = Math.multiplyExact(matrix, rb);
        if (matrixBytes <= matrixTarget) {
            return Math.min(rows, matrix * Math.max(1, matrixTarget / matrixBytes));
        }
        return matrixBytes <= MAX_BUFFER ? Math.min(rows, matrix) : Math.max(1, MAX_BUFFER / rb);
    }

    /**
     * Writes bytes {@code [off, off + n)} of {@code t} at {@code dst}'s position, producing the covering rows. A
     * row-aligned range is produced straight into {@code dst}; otherwise through a scratch buffer.
     */
    static void produceRange(PlannedTensor t, long off, int n, ByteBuffer dst) throws IOException {
        if (n == 0) {
            return;
        }
        RowSource s = t.source();
        long rb = s.rowBytes();
        if (rb <= 0 || rb > MAX_BUFFER) {
            throw new IOException(t.ggufName() + ": rows of " + rb + " bytes cannot be read");
        }
        long first = off / rb;
        long stop = Math.addExact(off, n);
        long last = (stop + rb - 1) / rb;
        int per = (int) Math.max(1, Math.min(SCRATCH_BYTES / rb, last - first));
        if (off % rb == 0 && n % rb == 0) {
            for (long r = first; r < last; ) {
                int c = (int) Math.min(per, last - r);
                s.readRows(r, c, dst);
                r += c;
            }
            return;
        }
        ByteBuffer scratch = ByteBuffer.allocate((int) (per * rb)).order(ByteOrder.LITTLE_ENDIAN);
        long pos = off;
        for (long r = first; r < last; ) {
            int c = (int) Math.min(per, last - r);
            scratch.clear();
            s.readRows(r, c, scratch);
            long chunkStart = r * rb;
            int from = (int) (pos - chunkStart);
            int to = (int) (Math.min(stop, chunkStart + c * rb) - chunkStart);
            scratch.limit(to);
            scratch.position(from);
            dst.put(scratch);
            pos += to - from;
            r += c;
        }
    }

    @Override
    public String toString() {
        return "ProducedSegments{" + tensors.size() + " tensors in [" + base + ", " + end + ")}";
    }

    /** Bytes of one materialized tensor. */
    private interface Held {
        /** Copies {@code len} bytes at {@code off} into {@code dst}, advancing it. */
        void copy(long off, int len, ByteBuffer dst) throws IOException;

        /** The {@code len} bytes at {@code off}, little-endian, positioned at 0. */
        ByteBuffer slice(long off, int len) throws IOException;

        /** Gives the bytes up; buffers handed out stay readable. */
        void release() throws IOException;
    }

    /** A tensor in direct-memory pieces of {@code pieceBytes} bytes (the last one shorter). */
    private static final class MemoryHeld implements Held {
        private final ByteBuffer[] pieces;
        private final long pieceBytes;

        MemoryHeld(ByteBuffer[] pieces, long pieceBytes) {
            this.pieces = pieces;
            this.pieceBytes = pieceBytes;
        }

        static MemoryHeld produce(PlannedTensor t) throws IOException {
            RowSource s = t.source();
            long rb = Math.max(1, s.rowBytes());
            long rows = s.rows();
            long callRows = chunkRows(s, PIECE_BYTES);
            long pieceRows = s.rowLocal() ? Math.min(Math.max(1, rows), Math.max(1, PIECE_BYTES / rb)) : callRows;
            long pieceBytes = pieceRows * rb;
            int n = rows == 0 ? 0 : (int) ((rows + pieceRows - 1) / pieceRows);
            ByteBuffer[] pieces = new ByteBuffer[n];
            for (int p = 0; p < n; p++) {
                long r0 = p * pieceRows;
                long r1 = Math.min(rows, r0 + pieceRows);
                ByteBuffer b = allocate(t, (r1 - r0) * rb);
                for (long r = r0; r < r1; ) {
                    int c = (int) Math.min(callRows, r1 - r);
                    s.readRows(r, c, b);
                    r += c;
                }
                b.flip();
                pieces[p] = b;
            }
            return new MemoryHeld(pieces, pieceBytes);
        }

        private static ByteBuffer allocate(PlannedTensor t, long bytes) throws IOException {
            try {
                return ByteBuffer.allocateDirect((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
            } catch (OutOfMemoryError e) {
                throw new IOException("cannot allocate " + bytes + " bytes of direct memory for the derived tensor "
                        + t.ggufName() + "; raise -XX:MaxDirectMemorySize, lower "
                        + ModelSourceOptions.PROP_MEMORY_MAX + " or set "
                        + ModelSourceOptions.PROP_MATERIALIZE + "=disk", e);
            }
        }

        @Override
        public void copy(long off, int len, ByteBuffer dst) {
            long pos = off;
            int left = len;
            while (left > 0) {
                int p = (int) (pos / pieceBytes);
                int at = (int) (pos - p * pieceBytes);
                ByteBuffer src = pieces[p].duplicate();
                int n = Math.min(left, src.limit() - at);
                src.position(at);
                src.limit(at + n);
                dst.put(src);
                pos += n;
                left -= n;
            }
        }

        @Override
        public ByteBuffer slice(long off, int len) throws IOException {
            int p = len == 0 ? 0 : (int) (off / pieceBytes);
            int at = (int) (off - p * pieceBytes);
            if (pieces.length > 0 && (long) at + len <= pieces[p].limit()) {
                ByteBuffer b = pieces[p].duplicate();
                b.position(at);
                b.limit(at + len);
                return b.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
            }
            ByteBuffer out = ByteBuffer.allocateDirect(len).order(ByteOrder.LITTLE_ENDIAN);
            copy(off, len, out);
            out.flip();
            return out;
        }

        @Override
        public void release() {
        }
    }

    /** A tensor in a sealed sidecar file. */
    private static final class DiskHeld implements Held {
        private final Sidecar sidecar;

        DiskHeld(Sidecar sidecar) {
            this.sidecar = sidecar;
        }

        static DiskHeld produce(PlannedTensor t, SidecarDirectory dir) throws IOException {
            RowSource s = t.source();
            long rb = Math.max(1, s.rowBytes());
            long rows = s.rows();
            Sidecar sc = dir.create(t.ggufName(), t.byteSize());
            try {
                long callRows = chunkRows(s, SCRATCH_BYTES);
                ByteBuffer scratch = ByteBuffer.allocate((int) (Math.min(callRows, Math.max(1, rows)) * rb))
                        .order(ByteOrder.LITTLE_ENDIAN);
                for (long r = 0; r < rows; ) {
                    int c = (int) Math.min(callRows, rows - r);
                    scratch.clear();
                    s.readRows(r, c, scratch);
                    scratch.flip();
                    sc.write(r * rb, scratch);
                    r += c;
                }
                sc.seal();
            } catch (IOException | RuntimeException | Error e) {
                try {
                    sc.close();
                } catch (IOException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
                throw e;
            }
            return new DiskHeld(sc);
        }

        @Override
        public void copy(long off, int len, ByteBuffer dst) {
            sidecar.copy(off, len, dst);
        }

        @Override
        public ByteBuffer slice(long off, int len) throws IOException {
            return sidecar.slice(off, len);
        }

        @Override
        public void release() throws IOException {
            sidecar.close();
        }
    }

    /** The whole-tensor window: the last tensor materialized for the reader. */
    private static final class Window {
        final int index;
        final Held held;

        Window(int index, Held held) {
            this.index = index;
            this.held = held;
        }
    }

    /**
     * One reader's access to the derived tensors. Reads of a NONE tensor produce the covering rows; reads of a
     * MEMORY or DISK tensor materialize it whole once (under a per-tensor lock) and copy from the copy, which stays
     * for the view's life when the decision keeps it and otherwise is the view's window until another tensor is
     * materialized. Thread-safe.
     */
    public final class View implements Closeable {
        private final MaterializePolicy policy;
        private final SidecarDirectory sidecars;
        private final Object[] locks;
        private final AtomicReferenceArray<Held> kept;
        private final Object windowLock = new Object();
        private volatile Window window;
        private volatile boolean closed;

        private View(MaterializePolicy policy, SidecarDirectory sidecars) {
            this.policy = policy;
            this.sidecars = sidecars;
            this.locks = new Object[tensors.size()];
            for (int i = 0; i < locks.length; i++) {
                locks[i] = new Object();
            }
            this.kept = new AtomicReferenceArray<>(tensors.size());
        }

        /** The table the view reads. */
        public ProducedSegments segments() {
            return ProducedSegments.this;
        }

        /** The decisions the view applies. */
        public MaterializePolicy policy() {
            return policy;
        }

        /**
         * Writes the {@code len} bytes at virtual offset {@code off} at {@code dst}'s position, advancing it; the
         * alignment gaps between tensors read as zeros.
         *
         * @throws IOException when the range is not inside {@code [base(), end())}, the view is closed, or a producer
         *                     fails
         */
        public void read(long off, int len, ByteBuffer dst) throws IOException {
            Objects.requireNonNull(dst, "dst");
            if (len < 0 || len > dst.remaining()) {
                throw new IllegalArgumentException(len + " bytes requested into a buffer with " + dst.remaining()
                        + " remaining");
            }
            if (off < base || off > end - len) {
                throw new IOException("bytes [" + off + ", +" + len + ") lie outside the derived tensors [" + base
                        + ", " + end + ")");
            }
            ensureOpen();
            long pos = off;
            long stop = off + len;
            while (pos < stop) {
                int i = slotAt(pos);
                long tStart = i < 0 ? base : offsets[i];
                long tEnd = i < 0 ? base : tStart + tensors.get(i).byteSize();
                if (pos >= tEnd) {
                    long next = i + 1 < offsets.length ? offsets[i + 1] : end;
                    int z = (int) (Math.min(stop, next) - pos);
                    for (int k = 0; k < z; k++) {
                        dst.put((byte) 0);
                    }
                    pos += z;
                    continue;
                }
                int n = (int) (Math.min(stop, tEnd) - pos);
                readTensor(i, pos - tStart, n, dst);
                pos += n;
            }
        }

        private void readTensor(int i, long off, int n, ByteBuffer dst) throws IOException {
            Decision d = policy.decision(i);
            if (d.mode() == Mode.NONE) {
                produceRange(tensors.get(i), off, n, dst);
            } else {
                held(i, d).copy(off, n, dst);
            }
        }

        /**
         * The {@code length} bytes {@code byteOff} bytes into produced tensor {@code index}, little-endian and
         * positioned at 0: a new direct buffer for a row-local tensor kept in memory (and for NONE), a slice of the
         * materialized copy otherwise. The caller may keep the buffer.
         *
         * @throws IOException when the range lies outside the tensor or the bytes cannot be produced
         */
        public ByteBuffer region(int index, long byteOff, int length) throws IOException {
            PlannedTensor t = tensors.get(index);
            if (byteOff < 0 || length < 0 || byteOff > t.byteSize() - length) {
                throw new IOException("tensor " + t.ggufName() + ": region [" + byteOff + ", +" + length
                        + ") lies outside its " + t.byteSize() + " bytes");
            }
            ensureOpen();
            Decision d = policy.decision(index);
            if (d.mode() == Mode.NONE || (d.mode() == Mode.MEMORY && t.source().rowLocal())) {
                ByteBuffer b = MemoryHeld.allocate(t, length);
                produceRange(t, byteOff, length, b);
                b.flip();
                return b;
            }
            return held(index, d).slice(byteOff, length);
        }

        private Held held(int i, Decision d) throws IOException {
            if (d.keep()) {
                Held h = kept.get(i);
                if (h != null) {
                    return h;
                }
                synchronized (locks[i]) {
                    h = kept.get(i);
                    if (h == null) {
                        h = materialize(i, d.mode());
                        kept.set(i, h);
                        if (closed) {
                            kept.set(i, null);
                            h.release();
                            throw closedError();
                        }
                    }
                    return h;
                }
            }
            Window w = window;
            if (w != null && w.index == i) {
                return w.held;
            }
            synchronized (locks[i]) {
                w = window;
                if (w != null && w.index == i) {
                    return w.held;
                }
                Held h = materialize(i, d.mode());
                Window old;
                synchronized (windowLock) {
                    if (closed) {
                        h.release();
                        throw closedError();
                    }
                    old = window;
                    window = new Window(i, h);
                }
                if (old != null) {
                    old.held.release();
                }
                return h;
            }
        }

        private Held materialize(int i, Mode mode) throws IOException {
            PlannedTensor t = tensors.get(i);
            return mode == Mode.MEMORY ? MemoryHeld.produce(t) : DiskHeld.produce(t, sidecars);
        }

        private void ensureOpen() throws IOException {
            if (closed) {
                throw closedError();
            }
        }

        private IOException closedError() {
            return new IOException("the derived tensors of this " + policy.reader() + " reader are closed");
        }

        /**
         * Releases every materialized copy (deleting its sidecar file); buffers handed out stay readable. Later reads
         * fail.
         *
         * @throws IOException the first failure, with later ones suppressed
         */
        @Override
        public void close() throws IOException {
            Window w;
            synchronized (windowLock) {
                if (closed) {
                    return;
                }
                closed = true;
                w = window;
                window = null;
            }
            IOException first = null;
            List<Held> all = new ArrayList<>();
            if (w != null) {
                all.add(w.held);
            }
            for (int i = 0; i < kept.length(); i++) {
                Held h = kept.getAndSet(i, null);
                if (h != null) {
                    all.add(h);
                }
            }
            for (Held h : all) {
                try {
                    h.release();
                } catch (IOException e) {
                    if (first == null) {
                        first = e;
                    } else {
                        first.addSuppressed(e);
                    }
                }
            }
            if (first != null) {
                throw first;
            }
        }

        @Override
        public String toString() {
            return "ProducedSegments.View{" + policy + "}";
        }
    }
}

package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.MultiShardMappedWeightStore;
import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The weight store of a Hugging Face model source over its {@link VirtualLayout}: the checkpoint files
 * ({@link MultiShardMappedWeightStore}) for {@code [0, P)} and the derived tensors ({@link ProducedSegments.View}) for
 * {@code [P, end)}. A request is split where it crosses a file boundary or {@code P}; the alignment gaps between
 * files read as zeros. A request outside {@code [0, end)} throws an {@link IOException} naming the offset. Thread-safe.
 */
public final class HfWeightStore implements WeightStore {

    private final List<DataSegment> segments;
    private final MultiShardMappedWeightStore files;
    private final ProducedSegments.View produced;
    private final long producedBase;
    private final long end;
    private final AtomicBoolean closed = new AtomicBoolean();

    /**
     * A store over {@code layout} whose derived tensors are read through {@code produced}, which the store closes (also
     * when construction fails).
     *
     * @throws IOException when a checkpoint file cannot be mapped
     */
    public HfWeightStore(VirtualLayout layout, ProducedSegments.View produced) throws IOException {
        Objects.requireNonNull(layout, "layout");
        this.produced = Objects.requireNonNull(produced, "produced");
        if (produced.segments() != layout.produced()) {
            throw new IllegalArgumentException("the derived-tensor view belongs to another layout");
        }
        List<DataSegment> data = new ArrayList<>();
        for (DataSegment s : layout.segments()) {
            if (s.dataLength() > 0) {
                data.add(s);
            }
        }
        this.segments = Collections.unmodifiableList(data);
        MultiShardMappedWeightStore mapped = null;
        try {
            mapped = data.isEmpty() ? null : new MultiShardMappedWeightStore(data);
        } catch (IOException | RuntimeException e) {
            try {
                produced.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
        this.files = mapped;
        this.producedBase = layout.producedBase();
        this.end = layout.totalBytes();
    }

    @Override
    public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
        readTensor(tensorOffset, size, dst, 0);
    }

    @Override
    public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
        Objects.requireNonNull(dst, "dst");
        if (dstOffset < 0 || size < 0 || dstOffset > dst.length - size) {
            throw new IllegalArgumentException(size + " bytes at " + dstOffset + " do not fit an array of "
                    + dst.length);
        }
        readTensorDirect(tensorOffset, size, ByteBuffer.wrap(dst, dstOffset, size));
    }

    @Override
    public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        if (size < 0 || size > dst.remaining()) {
            throw new IllegalArgumentException(size + " bytes requested into a buffer with " + dst.remaining()
                    + " remaining");
        }
        if (tensorOffset < 0 || tensorOffset > end - size) {
            throw new IOException("bytes [" + tensorOffset + ", +" + size + ") lie outside the " + end
                    + " bytes of tensor data");
        }
        long pos = tensorOffset;
        long stop = tensorOffset + size;
        if (pos < producedBase) {
            long split = Math.min(stop, producedBase);
            readFiles(pos, (int) (split - pos), dst);
            pos = split;
        }
        if (pos < stop) {
            produced.read(pos, (int) (stop - pos), dst);
        }
    }

    /** Reads {@code [pos, pos + n)} of {@code [0, P)}: segment bytes from the files, gaps as zeros. */
    private void readFiles(long pos, int n, ByteBuffer dst) throws IOException {
        long stop = pos + n;
        long p = pos;
        int s = segmentAtOrAfter(p);
        while (p < stop) {
            DataSegment seg = s < segments.size() ? segments.get(s) : null;
            if (seg == null || p < seg.virtualBase()) {
                long gapEnd = seg == null ? stop : Math.min(stop, seg.virtualBase());
                for (long k = p; k < gapEnd; k++) {
                    dst.put((byte) 0);
                }
                p = gapEnd;
                continue;
            }
            int len = (int) (Math.min(stop, seg.virtualEnd()) - p);
            files.readTensorDirect(p, len, dst);
            p += len;
            s++;
        }
    }

    /** The first segment that ends after {@code pos}; {@code segments.size()} when none does. */
    private int segmentAtOrAfter(long pos) {
        int lo = 0;
        int hi = segments.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (segments.get(mid).virtualEnd() <= pos) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /**
     * Closes the derived-tensor view, then the files; the first failure is thrown with later ones suppressed. Later
     * calls do nothing.
     */
    @Override
    public void close() throws IOException {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        IOException first = null;
        try {
            produced.close();
        } catch (IOException e) {
            first = e;
        }
        if (files != null) {
            try {
                files.close();
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
        return "HfWeightStore{" + segments.size() + " files, derived from " + producedBase + " to " + end + "}";
    }
}

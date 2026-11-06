package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * {@link WeightStore} of a {@link StandardExpertView}: reads of the delegate's own tensors go to
 * the delegate store at their unchanged virtual offsets, and reads of a synthesized expert tensor
 * are served from the packed tensor its {@link PackedExpertEntry} describes, expert by expert —
 * the byte range of expert {@code e}'s role lies at {@code roleOffset + e * stride} inside the
 * packed tensor. A read must stay inside one tensor's range; one that straddles the boundary of a
 * synthesized tensor or reaches past the delegate's data is refused with the range named.
 * Thread-safe, as the delegate store is.
 */
public final class StandardExpertStore implements WeightStore {

    private final WeightStore delegate;
    private final StandardExpertView view;

    StandardExpertStore(WeightStore delegate, StandardExpertView view) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.view = Objects.requireNonNull(view, "view");
    }

    @Override
    public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
        readTensor(tensorOffset, size, dst, 0);
    }

    @Override
    public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
        Objects.requireNonNull(dst, "dst");
        StandardExpertView.Synth s = view.synthAt(tensorOffset, size);
        if (s == null) {
            delegate.readTensor(tensorOffset, size, dst, dstOffset);
            return;
        }
        checkRange(s, tensorOffset, size, dstOffset, dst.length);
        long entryBytes = s.role.entryBytes();
        long done = 0;
        while (done < size) {
            long tensorByte = tensorOffset - s.info.offset() + done;
            long e = tensorByte / entryBytes;
            long local = tensorByte - e * entryBytes;
            long run = Math.min(size - done, entryBytes - local);
            delegate.readTensor(packedByte(s, e, local), (int) run, dst, dstOffset + (int) done);
            done += run;
        }
    }

    @Override
    public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        StandardExpertView.Synth s = view.synthAt(tensorOffset, size);
        if (s == null) {
            delegate.readTensorDirect(tensorOffset, size, dst);
            return;
        }
        checkRange(s, tensorOffset, size, 0, dst.remaining());
        long entryBytes = s.role.entryBytes();
        long done = 0;
        while (done < size) {
            long tensorByte = tensorOffset - s.info.offset() + done;
            long e = tensorByte / entryBytes;
            long local = tensorByte - e * entryBytes;
            long run = Math.min(size - done, entryBytes - local);
            delegate.readTensorDirect(packedByte(s, e, local), (int) run, dst);
            done += run;
        }
    }

    private static long packedByte(StandardExpertView.Synth s, long e, long local) {
        return s.packedTensor.offset() + s.role.offset() + e * s.entry.stride() + local;
    }

    private static void checkRange(StandardExpertView.Synth s, long tensorOffset, long size, int dstOffset,
                                   int dstLength) {
        long start = tensorOffset - s.info.offset();
        if (start < 0 || size < 0 || size > s.sizeBytes() - start || dstOffset < 0
                || size > dstLength - dstOffset) {
            throw new IllegalArgumentException("read [" + tensorOffset + ", +" + size + ") of " + s.info.name()
                    + " does not lie inside the tensor's " + s.sizeBytes() + " bytes or does not fit the "
                    + (dstLength - dstOffset) + " bytes of its destination");
        }
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return view + " over " + delegate;
    }
}

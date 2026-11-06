package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.WeightStore;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * {@link WeightStore} of a {@link PackedExpertView}: reads of the delegate's own tensors go to the
 * delegate store at their unchanged virtual offsets, and reads of a packed tensor
 * {@code blk.N.ffn_exps_packed} are assembled from the delegate's expert tensors, expert by expert,
 * as {@link PackedExpertView.Packed} maps each entry byte: a role's data from the expert's own place
 * in the role's tensor, its pad from the first bytes of the next expert's same-role data (the last
 * expert repeats its own), and the alignment slack after a pad as zeros, which is what
 * {@code pack-experts.py} writes. A read must stay inside one tensor's range; one that straddles a
 * packed tensor's boundary or reaches past the delegate's data is refused with the range named.
 * Thread-safe, as the delegate store is.
 */
public final class PackedExpertStore implements WeightStore {

    private final WeightStore delegate;
    private final PackedExpertView view;

    PackedExpertStore(WeightStore delegate, PackedExpertView view) {
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
        PackedExpertView.Packed p = view.packedAt(tensorOffset, size);
        if (p == null) {
            delegate.readTensor(tensorOffset, size, dst, dstOffset);
            return;
        }
        checkRange(p, tensorOffset, size, dstOffset, dst.length);
        long done = 0;
        while (done < size) {
            long tensorByte = tensorOffset - p.info().offset() + done;
            long run = Math.min(size - done, p.runLength(tensorByte));
            long source = p.source(tensorByte);
            if (source == PackedExpertView.Packed.ZERO) {
                Arrays.fill(dst, dstOffset + (int) done, dstOffset + (int) (done + run), (byte) 0);
            } else {
                delegate.readTensor(source, (int) run, dst, dstOffset + (int) done);
            }
            done += run;
        }
    }

    @Override
    public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        PackedExpertView.Packed p = view.packedAt(tensorOffset, size);
        if (p == null) {
            delegate.readTensorDirect(tensorOffset, size, dst);
            return;
        }
        checkRange(p, tensorOffset, size, 0, dst.remaining());
        long done = 0;
        while (done < size) {
            long tensorByte = tensorOffset - p.info().offset() + done;
            long run = Math.min(size - done, p.runLength(tensorByte));
            long source = p.source(tensorByte);
            if (source == PackedExpertView.Packed.ZERO) {
                for (long b = 0; b < run; b++) {
                    dst.put((byte) 0);
                }
            } else {
                delegate.readTensorDirect(source, (int) run, dst);
            }
            done += run;
        }
    }

    private static void checkRange(PackedExpertView.Packed p, long tensorOffset, long size, int dstOffset,
                                   int dstLength) {
        long start = tensorOffset - p.info().offset();
        if (start < 0 || size < 0 || size > p.sizeBytes() - start || dstOffset < 0
                || size > dstLength - dstOffset) {
            throw new IllegalArgumentException("read [" + tensorOffset + ", +" + size + ") of " + p.info().name()
                    + " does not lie inside the tensor's " + p.sizeBytes() + " bytes or does not fit the "
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

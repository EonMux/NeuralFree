package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorRegions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * {@link TensorRegions} of a {@link StandardExpertView}: every ordinary tensor is the delegate
 * regions' own region, and every synthesized expert tensor is served from the packed tensor its
 * {@link PackedExpertEntry} describes, expert by expert — the byte range of expert {@code e}'s
 * role lies at {@code roleOffset + e * stride} inside the packed tensor. Each expert's role data is
 * one contiguous run ({@link #contiguousRunBytes}): a range inside one expert is the delegate's
 * in-place region, and a range that crosses experts is assembled into a fresh direct buffer.
 * Thread-safe.
 */
public final class StandardExpertRegions implements TensorRegions {

    private final TensorRegions delegate;
    private final StandardExpertView view;

    StandardExpertRegions(TensorRegions delegate, StandardExpertView view) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.view = Objects.requireNonNull(view, "view");
    }

    @Override
    public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
        Objects.requireNonNull(info, "info");
        StandardExpertView.Synth s = view.synth(info.name());
        if (s == null) {
            return delegate.region(info, byteOff, length);
        }
        checkRange(s, byteOff, length);
        if (length == 0) {
            return ByteBuffer.allocateDirect(0).order(ByteOrder.LITTLE_ENDIAN);
        }
        long entryBytes = s.role.entryBytes();
        long e = byteOff / entryBytes;
        long local = byteOff - e * entryBytes;
        if (length <= entryBytes - local) {
            return delegate.region(s.packedTensor, packedEntryByte(s, e, local), length);
        }
        ByteBuffer out = ByteBuffer.allocateDirect((int) length).order(ByteOrder.LITTLE_ENDIAN);
        long done = 0;
        while (done < length) {
            long tensorByte = byteOff + done;
            long expert = tensorByte / entryBytes;
            long at = tensorByte - expert * entryBytes;
            long run = Math.min(length - done, entryBytes - at);
            out.put(delegate.region(s.packedTensor, packedEntryByte(s, expert, at), run));
            done += run;
        }
        out.flip();
        return out;
    }

    /** One expert's role bytes for a synthesized expert tensor; the delegate's runs for every other tensor. */
    @Override
    public long contiguousRunBytes(GGUFTensorInfo info) {
        Objects.requireNonNull(info, "info");
        StandardExpertView.Synth s = view.synth(info.name());
        return s == null ? delegate.contiguousRunBytes(info) : s.role.entryBytes();
    }

    /** Byte offset of expert {@code e}'s role data at entry-local {@code local}, inside the packed tensor. */
    private static long packedEntryByte(StandardExpertView.Synth s, long e, long local) {
        return s.role.offset() + e * s.entry.stride() + local;
    }

    private static void checkRange(StandardExpertView.Synth s, long byteOff, long length) throws IOException {
        if (byteOff < 0 || length < 0 || length > Integer.MAX_VALUE || byteOff > s.sizeBytes() - length) {
            throw new IOException("tensor " + s.info.name() + ": region [" + byteOff + ", +" + length
                    + ") is not a valid byte range of its " + s.sizeBytes() + " bytes");
        }
    }

    @Override
    public String description() {
        return delegate.description();
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public String toString() {
        return description();
    }
}

package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorRegions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * {@link TensorRegions} of a {@link PackedExpertView}: every ordinary tensor is the delegate
 * regions' own region, and a region of a packed tensor {@code blk.N.ffn_exps_packed} is assembled
 * into a fresh direct buffer from the delegate's expert tensors, byte run by byte run, as
 * {@link PackedExpertView.Packed} maps them (role data, pad from the next expert, zero slack).
 * Thread-safe, as the delegate regions are.
 */
public final class PackedExpertRegions implements TensorRegions {

    private final TensorRegions delegate;
    private final PackedExpertView view;

    PackedExpertRegions(TensorRegions delegate, PackedExpertView view) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.view = Objects.requireNonNull(view, "view");
    }

    @Override
    public ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException {
        Objects.requireNonNull(info, "info");
        PackedExpertView.Packed p = view.packed(info.name());
        if (p == null) {
            return delegate.region(info, byteOff, length);
        }
        if (byteOff < 0 || length < 0 || length > Integer.MAX_VALUE || byteOff > p.sizeBytes() - length) {
            throw new IOException("tensor " + info.name() + ": region [" + byteOff + ", +" + length
                    + ") is not a valid byte range of its " + p.sizeBytes() + " bytes");
        }
        ByteBuffer out = ByteBuffer.allocateDirect((int) length).order(ByteOrder.LITTLE_ENDIAN);
        long done = 0;
        while (done < length) {
            long x = byteOff + done;
            int run = (int) Math.min(length - done, p.runLength(x));
            long source = p.source(x);
            if (source == PackedExpertView.Packed.ZERO) {
                out.position(out.position() + run);
            } else {
                GGUFTensorInfo t = p.sourceTensor(x);
                out.put(delegate.region(t, source - t.offset(), run));
            }
            done += run;
        }
        out.flip();
        return out;
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

package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.StTensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * A {@link RowSource} over one tensor stored in a safetensors file: rows are read with positional reads, any byte
 * offset, no mapping. Row-local; immutable and thread-safe while the file is open.
 */
public final class StRowSource implements RowSource {

    private final SafetensorsFile file;
    private final StTensor tensor;
    private final long directVirtualOffset;
    private final ElementFormat format;
    private final long rows;
    private final long innermost;

    StRowSource(SafetensorsFile file, StTensor tensor, long directVirtualOffset) {
        this.file = Objects.requireNonNull(file, "file");
        this.tensor = Objects.requireNonNull(tensor, "tensor");
        if (!tensor.equals(file.tensor(tensor.name()))) {
            throw new IllegalArgumentException("tensor '" + tensor.name() + "' is not a tensor of '" + file.path()
                    + "'");
        }
        if (directVirtualOffset < -1) {
            throw new IllegalArgumentException("virtual offset " + directVirtualOffset + " of tensor '"
                    + tensor.name() + "' is negative");
        }
        this.directVirtualOffset = directVirtualOffset;
        this.format = ElementFormat.of(tensor.dtype());
        long[] shape = tensor.shape();
        this.innermost = shape.length == 0 ? 1 : shape[shape.length - 1];
        this.rows = outerProduct(tensor.name(), shape);
    }

    /** A source over {@code tensor} of {@code file} that is not part of a checkpoint's virtual address space. */
    public static StRowSource of(SafetensorsFile file, StTensor tensor) {
        return new StRowSource(file, tensor, -1);
    }

    /** The stored tensor. */
    public StTensor tensor() {
        return tensor;
    }

    /** The file holding the tensor. */
    public SafetensorsFile file() {
        return file;
    }

    @Override
    public String describe() {
        return "tensor '" + tensor.name() + "' (" + tensor.dtype().headerName() + " "
                + Arrays.toString(tensor.shape()) + ") of '" + file.path() + "'";
    }

    @Override
    public ElementFormat format() {
        return format;
    }

    @Override
    public long[] shape() {
        return tensor.shape();
    }

    @Override
    public long rows() {
        return rows;
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException when a row of this tensor is not a whole number of bytes (a sub-byte dtype
     *                                  with an odd innermost dimension)
     */
    @Override
    public long rowBytes() {
        try {
            return format.rowBytes(innermost);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(describe() + " has no byte-addressable rows: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean rowLocal() {
        return true;
    }

    @Override
    public void readRows(long first, int count, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        if (first < 0 || count < 0 || first > rows - count) {
            throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                    + " rows of " + describe());
        }
        long rb = rowBytes();
        long bytes = Math.multiplyExact(rb, (long) count);
        if (bytes > dst.remaining()) {
            throw new IllegalArgumentException(count + " rows of " + describe() + " need " + bytes + " bytes; the"
                    + " buffer has " + dst.remaining());
        }
        ByteBuffer window = dst.duplicate();
        window.limit(window.position() + (int) bytes);
        file.read(tensor, Math.multiplyExact(first, rb), window);
        dst.position(dst.position() + (int) bytes);
    }

    @Override
    public long directVirtualOffset() {
        return directVirtualOffset;
    }

    /** The product of every dimension but the innermost; 1 for rank 0 and 1. */
    private static long outerProduct(String name, long[] shape) {
        long r = 1;
        try {
            for (int i = 0; i < shape.length - 1; i++) {
                r = Math.multiplyExact(r, shape[i]);
            }
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("tensor '" + name + "': the row count of shape "
                    + Arrays.toString(shape) + " overflows a 64-bit integer", e);
        }
        return r;
    }

    @Override
    public String toString() {
        return "StRowSource{" + describe() + "}";
    }
}

package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class GGUFTensorInfo {

    /** Maximum tensor rank ({@code GGML_MAX_DIMS}). */
    public static final int MAX_DIMS = 4;

    /** Tensor names must be shorter than this many UTF-8 bytes ({@code GGML_MAX_NAME}, which counts the NUL). */
    public static final int MAX_NAME_BYTES = 64;

    private final String name;
    private final long[] dims;
    private final GGMLType type;
    private final long offset;

    public GGUFTensorInfo(String name, long[] dims, GGMLType type, long offset) {
        this.name = name;
        this.dims = dims;
        this.type = type;
        this.offset = offset;
    }

    public String name() { return name; }
    public long[] dims() { return dims; }
    public GGMLType type() { return type; }
    public long offset() { return offset; }

    public long numElements() {
        long n = 1;
        for (long d : dims) n *= d;
        return n;
    }

    public int rank() { return dims.length; }

    public long dim(int i) {
        return i >= 0 && i < dims.length ? dims[i] : 1;
    }

    public long sizeInBytes() {
        return type.getBytesForElements(numElements());
    }

    /**
     * Rejects the tensor infos that llama.cpp rejects (llama.cpp ggml/src/gguf.cpp:657-680, 686-753): a name of
     * {@link #MAX_NAME_BYTES} UTF-8 bytes or more, more than {@link #MAX_DIMS} dimensions, a negative dimension (a
     * uint64 extent of 2^63 or more), an element count or byte size that overflows a signed 64-bit value, a type
     * without a storage layout (an id unknown to {@link GGMLType}, or the removed Q4_2 / Q4_3) and a row length
     * {@code ne[0]} that is not a whole number of blocks. Every source of tensor infos (a GGUF file, a converted
     * checkpoint, the GGUF writer) applies these rules.
     *
     * @param sourcePrefix names the source in the message, e.g. {@code Invalid GGUF file '<path>'}
     * @throws IOException {@code sourcePrefix + ": tensor '<name>' " + reason}
     */
    public static void validate(String sourcePrefix, GGUFTensorInfo info) throws IOException {
        String prefix = sourcePrefix + ": tensor '" + info.name() + "' ";
        int nameBytes = info.name().getBytes(StandardCharsets.UTF_8).length;
        if (nameBytes >= MAX_NAME_BYTES) {
            throw new IOException(prefix + "has a name of " + nameBytes + " bytes, more than the "
                    + (MAX_NAME_BYTES - 1) + " allowed");
        }
        long[] dims = info.dims();
        if (dims.length > MAX_DIMS) {
            throw new IOException(prefix + "has " + dims.length + " dimensions, more than the maximum of " + MAX_DIMS);
        }
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] < 0) {
                throw new IOException(prefix + "dimension " + i + " has " + Long.toUnsignedString(dims[i])
                        + " elements, more than a signed 64-bit count");
            }
        }
        long elements = dims.length == 0 ? 1 : dims[0];
        for (int i = 1; i < dims.length; i++) {
            if (dims[i] > 1 && Long.MAX_VALUE / dims[i] < elements) {
                throw new IOException(prefix + "has shape " + Arrays.toString(dims)
                        + ", whose element count overflows a signed 64-bit value");
            }
            elements *= dims[i];
        }
        GGMLType type = info.type();
        if (type == null || !BlockLayout.isDefined(type)) {
            throw new IOException(prefix + "has ggml type " + type + ", which is unknown or was removed from ggml");
        }
        BlockLayout layout = BlockLayout.of(type);
        long rowElements = info.dim(0);
        if (rowElements % layout.blockElements() != 0) {
            throw new IOException(prefix + "of type " + type + " has " + rowElements
                    + " elements per row, not a multiple of the block size " + layout.blockElements());
        }
        if (elements / layout.blockElements() > Long.MAX_VALUE / layout.blockBytes()) {
            throw new IOException(prefix + "of type " + type + " with shape " + Arrays.toString(dims)
                    + " has a byte size that overflows a signed 64-bit value");
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(name);
        sb.append(" [");
        for (int i = 0; i < dims.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(dims[i]);
        }
        sb.append("] type=").append(type).append(" offset=").append(offset);
        return sb.toString();
    }
}

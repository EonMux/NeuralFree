package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable description of one GGUF tensor: the name as stored in the file, its element type and its shape in
 * ggml {@code ne} order ({@code ne[0]} is the contiguous dimension, so a 2-D weight of {@code out x in} has
 * {@code ne = [in, out]}). Specs refer to optional and required tensors through this type; the data itself is
 * loaded by the weight stores.
 */
public final class TensorRef {

    private final String name;
    private final GGMLType type;
    private final long[] ne;

    private TensorRef(String name, GGMLType type, long[] ne) {
        this.name = name;
        this.type = type;
        this.ne = ne;
    }

    /**
     * Reference to a tensor described by the parser.
     *
     * @throws IllegalArgumentException when the description has no name, type or dimensions
     */
    public static TensorRef of(GGUFTensorInfo info) {
        Objects.requireNonNull(info, "tensor info");
        return of(info.name(), info.type(), info.dims());
    }

    /**
     * Reference with an explicit name, type and {@code ne} shape.
     *
     * @throws IllegalArgumentException for an empty name, a missing type, no dimensions or a negative dimension
     */
    public static TensorRef of(String name, GGMLType type, long... ne) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("tensor name is empty");
        }
        if (type == null) {
            throw new IllegalArgumentException("tensor " + name + " has no type");
        }
        if (ne == null || ne.length == 0) {
            throw new IllegalArgumentException("tensor " + name + " has no dimensions");
        }
        for (long d : ne) {
            if (d < 0) {
                throw new IllegalArgumentException("tensor " + name + " has a negative dimension: " + Arrays.toString(ne));
            }
        }
        return new TensorRef(name, type, ne.clone());
    }

    /** Name as stored in the GGUF file (may be a Hugging Face alias of the GGUF name that was asked for). */
    public String name() {
        return name;
    }

    public GGMLType type() {
        return type;
    }

    /** Shape in ggml {@code ne} order (copy). */
    public long[] ne() {
        return ne.clone();
    }

    /** Dimension {@code i} of the {@code ne} shape; 1 beyond the rank, as in ggml. */
    public long ne(int i) {
        if (i < 0) {
            throw new IllegalArgumentException("dimension index " + i + " is negative for tensor " + name);
        }
        return i < ne.length ? ne[i] : 1L;
    }

    public int rank() {
        return ne.length;
    }

    /** Length of the contiguous dimension ({@code ne[0]}): the input width of a weight matrix. */
    public long cols() {
        return ne[0];
    }

    /** Product of all dimensions above {@code ne[0]}: output rows of a weight matrix (times experts for 3-D). */
    public long rows() {
        long r = 1;
        for (int i = 1; i < ne.length; i++) {
            r *= ne[i];
        }
        return r;
    }

    public long numElements() {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        return n;
    }

    /** Storage size in bytes for the element type. */
    public long sizeInBytes() {
        return type.getBytesForElements(numElements());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof TensorRef)) {
            return false;
        }
        TensorRef t = (TensorRef) o;
        return name.equals(t.name) && type == t.type && Arrays.equals(ne, t.ne);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * name.hashCode() + type.hashCode()) + Arrays.hashCode(ne);
    }

    @Override
    public String toString() {
        return name + " " + type + " " + Arrays.toString(ne);
    }
}

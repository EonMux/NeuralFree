package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Objects;

/**
 * The element encoding of a {@link RowSource}'s bytes: a safetensors dtype, a ggml type, or both when they store the
 * same bytes (F16, BF16, F32, F64, I8, I16, I32, I64, see {@link StDType#directType()}); {@code of(StDType.F32)} and
 * {@code of(GGMLType.F32)} are therefore equal. Immutable.
 */
public final class ElementFormat {

    private final StDType dtype;
    private final GGMLType ggml;

    private ElementFormat(StDType dtype, GGMLType ggml) {
        this.dtype = dtype;
        this.ggml = ggml;
    }

    /** The format of elements stored as {@code dtype}. */
    public static ElementFormat of(StDType dtype) {
        Objects.requireNonNull(dtype, "dtype");
        return new ElementFormat(dtype, dtype.directType());
    }

    /**
     * The format of elements stored as {@code type}.
     *
     * @throws IllegalArgumentException for a type without a storage size ({@link GGMLType#UNKNOWN}, Q4_2, Q4_3)
     */
    public static ElementFormat of(GGMLType type) {
        Objects.requireNonNull(type, "type");
        if (type.getTypeSize() <= 0 || type.getBlockSize() <= 0) {
            throw new IllegalArgumentException("ggml type " + type + " has no storage size");
        }
        return new ElementFormat(StDType.forDirectType(type), type);
    }

    /** The safetensors dtype, or null for a ggml block type. */
    public StDType stDType() {
        return dtype;
    }

    /** The ggml type, or null when ggml has no type for these bytes (U8, F8 E5M2/FNUZ/E8M0, F4, F6, BOOL, unsigned, C64). */
    public GGMLType ggmlType() {
        return ggml;
    }

    /** True for a ggml block type (more than one element per block). */
    public boolean isBlock() {
        return ggml != null && ggml.isQuantized();
    }

    /**
     * True for plain floating-point elements, F16, BF16, F32 and F64: the formats a float dtype policy converts.
     * Eight-, six- and four-bit float codes and block types are not plain floats.
     */
    public boolean isFloat() {
        return ggml == GGMLType.F32 || ggml == GGMLType.F16 || ggml == GGMLType.BF16 || ggml == GGMLType.F64;
    }

    /**
     * Bytes of one row of {@code innermost} elements.
     *
     * @throws IllegalArgumentException when {@code innermost} is negative, is not a multiple of the block size of a
     *                                  block type, or makes a row that is not a whole number of bytes, or the size
     *                                  overflows
     */
    public long rowBytes(long innermost) {
        if (innermost < 0) {
            throw new IllegalArgumentException("negative row length " + innermost + " for " + this);
        }
        try {
            if (ggml != null) {
                int bs = ggml.getBlockSize();
                if (innermost % bs != 0) {
                    throw new IllegalArgumentException("a row of " + innermost + " elements is not a whole number"
                            + " of " + ggml + " blocks of " + bs);
                }
                return Math.multiplyExact(innermost / bs, (long) ggml.getTypeSize());
            }
            long bits = Math.multiplyExact(innermost, (long) dtype.bits());
            if (bits % 8 != 0) {
                throw new IllegalArgumentException("a row of " + innermost + " " + dtype.headerName()
                        + " elements is " + bits + " bits, not a whole number of bytes");
            }
            return bits / 8;
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("a row of " + innermost + " " + this + " elements overflows a"
                    + " 64-bit byte count", e);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ElementFormat)) {
            return false;
        }
        ElementFormat f = (ElementFormat) o;
        return dtype == f.dtype && ggml == f.ggml;
    }

    @Override
    public int hashCode() {
        return Objects.hash(dtype, ggml);
    }

    /** The safetensors dtype name, or the ggml type name of a block type. */
    @Override
    public String toString() {
        return dtype != null ? dtype.headerName() : ggml.name();
    }
}

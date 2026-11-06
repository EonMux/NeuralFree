package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * One tensor a live weight update carries into a running engine ({@link WeightUpdater#apply}): its stored tensor
 * name, element type, ggml shape ({@code ne[0]} first) and the little-endian bytes of the whole tensor, laid out
 * row-major over {@code ne[1..]} in rows of {@code ne[0]} elements each, exactly as the matching GGUF tensor
 * stores them. Immutable; the bytes are copied at construction, so the caller may reuse its arrays.
 *
 * <p>The type the bytes hold is the sender's choice; {@link WeightUpdater} implementations accept a tensor's
 * stored type (a checkpoint-format update, decoded by the engine the way a load decodes it) and F32 (a
 * trainer-format update) where the engine consumes the tensor as F32, and refuse anything else naming the tensor
 * and both types.
 */
public final class WeightUpdateTensor {

    private final String name;
    private final GGMLType dtype;
    private final long[] ne;
    private final byte[] data;

    private WeightUpdateTensor(String name, GGMLType dtype, long[] ne, byte[] data) {
        this.name = name;
        this.dtype = dtype;
        this.ne = ne;
        this.data = data;
    }

    /**
     * A typed tensor over raw little-endian bytes.
     *
     * @param name  stored tensor name ({@code blk.5.attn_q.weight}), non-blank
     * @param dtype element type the bytes hold
     * @param ne    ggml shape, {@code ne[0]} first (copied); every dimension positive
     * @param data  the whole tensor's bytes (copied); exactly {@link GGMLType#getBytesForElements} of the element
     *              count the shape describes
     * @throws IllegalArgumentException naming the first field that fails
     */
    public static WeightUpdateTensor of(String name, GGMLType dtype, long[] ne, byte[] data) {
        checkName(name);
        Objects.requireNonNull(dtype, "dtype");
        if (ne == null || ne.length == 0) {
            throw new IllegalArgumentException("weight update tensor " + name + " has no dimensions");
        }
        long[] shape = new long[ne.length];
        long elements = 1;
        for (int i = 0; i < ne.length; i++) {
            if (ne[i] <= 0) {
                throw new IllegalArgumentException("weight update tensor " + name + " has a non-positive dimension: "
                        + Arrays.toString(ne));
            }
            shape[i] = ne[i];
            elements = Math.multiplyExact(elements, ne[i]);
        }
        if (data == null) {
            throw new IllegalArgumentException("weight update tensor " + name + " carries no data");
        }
        long need = dtype.getBytesForElements(elements);
        if (need != data.length) {
            throw new IllegalArgumentException("weight update tensor " + name + " (" + dtype + ", ne="
                    + Arrays.toString(shape) + ") needs " + need + " bytes, got " + data.length);
        }
        return new WeightUpdateTensor(name, dtype, shape, data.clone());
    }

    /**
     * An F32 tensor over float values, little-endian in {@link #data()}.
     *
     * @throws IllegalArgumentException naming the first field that fails
     */
    public static WeightUpdateTensor f32(String name, long[] ne, float[] values) {
        checkName(name);
        Objects.requireNonNull(values, "values");
        if (ne == null || ne.length == 0) {
            throw new IllegalArgumentException("weight update tensor " + name + " has no dimensions");
        }
        long elements = 1;
        for (long d : ne) {
            if (d <= 0) {
                throw new IllegalArgumentException("weight update tensor " + name + " has a non-positive dimension: "
                        + Arrays.toString(ne));
            }
            elements = Math.multiplyExact(elements, d);
        }
        if (elements != values.length) {
            throw new IllegalArgumentException("weight update tensor " + name + " (F32, ne=" + Arrays.toString(ne)
                    + ") holds " + elements + " elements, got " + values.length + " floats");
        }
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return new WeightUpdateTensor(name, GGMLType.F32, ne.clone(), bytes.array());
    }

    private static void checkName(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("weight update tensor has no name");
        }
    }

    /** The stored tensor name the update targets. */
    public String name() {
        return name;
    }

    /** The element type the bytes hold. */
    public GGMLType dtype() {
        return dtype;
    }

    /** ggml shape, {@code ne[0]} first (copy). */
    public long[] ne() {
        return ne.clone();
    }

    /** Element count the shape describes. */
    public long numElements() {
        long elements = 1;
        for (long d : ne) {
            elements *= d;
        }
        return elements;
    }

    /** The whole tensor's little-endian bytes (copy), row-major over {@code ne[1..]}. */
    public byte[] data() {
        return data.clone();
    }

    /**
     * The tensor's bytes as a little-endian buffer over the internal array (no copy); readers must never write
     * through it. Its position is 0 and its limit the byte count.
     */
    public ByteBuffer dataBuffer() {
        return ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** The byte count of {@link #data()}. */
    public int dataBytes() {
        return data.length;
    }

    /** The parsed {@code GGMLType} of a wire type name, case-insensitive. */
    public static GGMLType parseDtype(String raw, String where) {
        if (raw != null) {
            for (GGMLType t : GGMLType.values()) {
                if (t.name().equals(raw) || t.name().toLowerCase(Locale.ROOT).equals(raw.toLowerCase(Locale.ROOT))) {
                    return t;
                }
            }
        }
        throw new IllegalArgumentException(where + ": '" + raw + "' is not a ggml type name (F32, BF16, Q8_0, ...)");
    }

    @Override
    public String toString() {
        return name + " " + dtype + " " + Arrays.toString(ne) + " (" + data.length + " B)";
    }
}

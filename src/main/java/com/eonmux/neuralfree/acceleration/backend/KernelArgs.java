package com.eonmux.neuralfree.acceleration.backend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Classification, snapshotting and little-endian packing of kernel arguments,
 * shared by the backends that marshal arguments themselves (CUDA, Vulkan).
 *
 * <p>Supported argument types:</p>
 * <ul>
 *   <li>{@link GpuBuffer} ({@link ArgKind#BUFFER});</li>
 *   <li>{@code Integer}, {@code Boolean} (0/1) and {@code int[1]} ({@link ArgKind#I32});</li>
 *   <li>{@code Float} and {@code float[1]} ({@link ArgKind#F32});</li>
 *   <li>{@code Long}, {@code long[1]}, {@code Double} and {@code double[1]} ({@link ArgKind#I64},
 *       {@link ArgKind#F64}), and other primitive arrays passed by value ({@link ArgKind#BYTES}),
 *       which only the legacy ML callers use; LLM kernels take 4-byte scalars and buffers only.</li>
 * </ul>
 */
public final class KernelArgs {

    /** How an argument is passed to the kernel. */
    public enum ArgKind {
        /** A device buffer. */
        BUFFER,
        /** A 32-bit signed integer scalar. */
        I32,
        /** A 32-bit float scalar. */
        F32,
        /** A 64-bit signed integer scalar. */
        I64,
        /** A 64-bit float scalar. */
        F64,
        /** A primitive array passed by value. */
        BYTES
    }

    private KernelArgs() {
    }

    /**
     * Classifies one argument.
     *
     * @throws IllegalArgumentException when the argument is null, an empty array,
     *                                  or of an unsupported type
     */
    public static ArgKind kindOf(Object arg) {
        if (arg == null) {
            throw new IllegalArgumentException("kernel argument is null");
        }
        if (arg instanceof GpuBuffer) {
            return ArgKind.BUFFER;
        }
        if (arg instanceof Integer || arg instanceof Boolean) {
            return ArgKind.I32;
        }
        if (arg instanceof Float) {
            return ArgKind.F32;
        }
        if (arg instanceof Long) {
            return ArgKind.I64;
        }
        if (arg instanceof Double) {
            return ArgKind.F64;
        }
        if (arg instanceof int[]) {
            return arrayKind(((int[]) arg).length, ArgKind.I32);
        }
        if (arg instanceof float[]) {
            return arrayKind(((float[]) arg).length, ArgKind.F32);
        }
        if (arg instanceof long[]) {
            return arrayKind(((long[]) arg).length, ArgKind.I64);
        }
        if (arg instanceof double[]) {
            return arrayKind(((double[]) arg).length, ArgKind.F64);
        }
        if (arg instanceof byte[]) {
            return arrayKind(((byte[]) arg).length, ArgKind.BYTES);
        }
        if (arg instanceof short[]) {
            return arrayKind(((short[]) arg).length, ArgKind.BYTES);
        }
        throw new IllegalArgumentException("unsupported kernel argument type " + arg.getClass().getName());
    }

    /**
     * Captures the value of an argument at the time it is set.
     *
     * <p>Buffers and boxed values are immutable from the queue's point of view and
     * are returned as is. A primitive array is copied: into {@code previousSnapshot}
     * when that is an array of the same type and length (no allocation), otherwise
     * into a fresh clone. Later mutation of the caller's array therefore has no
     * effect until the argument is set again.</p>
     *
     * @throws IllegalArgumentException when the argument is not supported
     */
    public static Object snapshot(Object arg, Object previousSnapshot) {
        kindOf(arg);
        if (arg instanceof int[]) {
            int[] src = (int[]) arg;
            if (previousSnapshot instanceof int[] && ((int[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        if (arg instanceof float[]) {
            float[] src = (float[]) arg;
            if (previousSnapshot instanceof float[] && ((float[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        if (arg instanceof long[]) {
            long[] src = (long[]) arg;
            if (previousSnapshot instanceof long[] && ((long[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        if (arg instanceof double[]) {
            double[] src = (double[]) arg;
            if (previousSnapshot instanceof double[] && ((double[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        if (arg instanceof byte[]) {
            byte[] src = (byte[]) arg;
            if (previousSnapshot instanceof byte[] && ((byte[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        if (arg instanceof short[]) {
            short[] src = (short[]) arg;
            if (previousSnapshot instanceof short[] && ((short[]) previousSnapshot).length == src.length
                    && previousSnapshot != src) {
                System.arraycopy(src, 0, previousSnapshot, 0, src.length);
                return previousSnapshot;
            }
            return src.clone();
        }
        return arg;
    }

    /**
     * True when setting {@code arg} would not change a stored {@code snapshot}:
     * the same buffer object, an equal boxed value, or an array of the same type
     * with equal contents.
     */
    public static boolean sameValue(Object snapshot, Object arg) {
        if (snapshot == null || arg == null) {
            return false;
        }
        if (snapshot instanceof GpuBuffer || arg instanceof GpuBuffer) {
            return snapshot == arg;
        }
        if (snapshot.getClass() != arg.getClass()) {
            return false;
        }
        if (arg instanceof int[]) {
            return Arrays.equals((int[]) snapshot, (int[]) arg);
        }
        if (arg instanceof float[]) {
            return Arrays.equals((float[]) snapshot, (float[]) arg);
        }
        if (arg instanceof long[]) {
            return Arrays.equals((long[]) snapshot, (long[]) arg);
        }
        if (arg instanceof double[]) {
            return Arrays.equals((double[]) snapshot, (double[]) arg);
        }
        if (arg instanceof byte[]) {
            return Arrays.equals((byte[]) snapshot, (byte[]) arg);
        }
        if (arg instanceof short[]) {
            return Arrays.equals((short[]) snapshot, (short[]) arg);
        }
        return snapshot.equals(arg);
    }

    /**
     * Number of bytes a by-value argument occupies in the kernel's parameter or
     * push-constant image.
     *
     * @throws IllegalArgumentException for a buffer, which is not passed by value
     */
    public static int byteSize(Object snapshot) {
        switch (kindOf(snapshot)) {
            case BUFFER:
                throw new IllegalArgumentException("a buffer argument has no by-value size");
            case I32:
            case F32:
                return 4;
            case I64:
            case F64:
                return 8;
            default:
                return arrayBytes(snapshot);
        }
    }

    /**
     * Writes a by-value argument at {@code offset} in little-endian order, using
     * absolute puts so the position of {@code dst} is unchanged.
     *
     * @return the number of bytes written
     * @throws IllegalArgumentException for a buffer argument
     * @throws IndexOutOfBoundsException when the value does not fit in {@code dst}
     */
    public static int writeLE(ByteBuffer dst, int offset, Object snapshot) {
        int size = byteSize(snapshot);
        if (offset < 0 || offset > dst.capacity() - size) {
            throw new IndexOutOfBoundsException("argument of " + size + " bytes at offset " + offset
                    + " does not fit in a buffer of " + dst.capacity() + " bytes");
        }
        ByteBuffer le = dst.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        if (snapshot instanceof Integer) {
            le.putInt(offset, (Integer) snapshot);
        } else if (snapshot instanceof Boolean) {
            le.putInt(offset, ((Boolean) snapshot) ? 1 : 0);
        } else if (snapshot instanceof Float) {
            le.putFloat(offset, (Float) snapshot);
        } else if (snapshot instanceof Long) {
            le.putLong(offset, (Long) snapshot);
        } else if (snapshot instanceof Double) {
            le.putDouble(offset, (Double) snapshot);
        } else if (snapshot instanceof int[]) {
            int[] a = (int[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.putInt(offset + 4 * i, a[i]);
            }
        } else if (snapshot instanceof float[]) {
            float[] a = (float[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.putFloat(offset + 4 * i, a[i]);
            }
        } else if (snapshot instanceof long[]) {
            long[] a = (long[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.putLong(offset + 8 * i, a[i]);
            }
        } else if (snapshot instanceof double[]) {
            double[] a = (double[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.putDouble(offset + 8 * i, a[i]);
            }
        } else if (snapshot instanceof byte[]) {
            byte[] a = (byte[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.put(offset + i, a[i]);
            }
        } else {
            short[] a = (short[]) snapshot;
            for (int i = 0; i < a.length; i++) {
                le.putShort(offset + 2 * i, a[i]);
            }
        }
        return size;
    }

    /** Short type description for error messages, for example {@code "Float"} or {@code "int[3]"}. */
    public static String describe(Object arg) {
        if (arg == null) {
            return "null";
        }
        if (arg instanceof GpuBuffer) {
            return "GpuBuffer";
        }
        if (arg.getClass().isArray()) {
            return arg.getClass().getComponentType().getSimpleName()
                    + "[" + java.lang.reflect.Array.getLength(arg) + "]";
        }
        return arg.getClass().getSimpleName();
    }

    private static ArgKind arrayKind(int length, ArgKind scalarKind) {
        if (length == 0) {
            throw new IllegalArgumentException("empty array kernel argument");
        }
        return length == 1 ? scalarKind : ArgKind.BYTES;
    }

    private static int arrayBytes(Object array) {
        if (array instanceof int[]) {
            return 4 * ((int[]) array).length;
        }
        if (array instanceof float[]) {
            return 4 * ((float[]) array).length;
        }
        if (array instanceof long[]) {
            return 8 * ((long[]) array).length;
        }
        if (array instanceof double[]) {
            return 8 * ((double[]) array).length;
        }
        if (array instanceof byte[]) {
            return ((byte[]) array).length;
        }
        return 2 * ((short[]) array).length;
    }
}

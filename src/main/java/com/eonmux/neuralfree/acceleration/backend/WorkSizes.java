package com.eonmux.neuralfree.acceleration.backend;

/**
 * Pure helpers for the dispatch contract of {@link ComputeQueue}.
 *
 * <p>{@code global[d]} is the total number of work-items in dimension {@code d}
 * and {@code local[d]} is the work-group size. Backends that launch by group
 * count (CUDA, Vulkan) use {@link #groupCounts(long[], long[])}; when the caller
 * passes no local size they use {@link #defaultLocal(long[], long)}.</p>
 */
public final class WorkSizes {

    /** Upper bound of the work-group size chosen by {@link #defaultLocal(long[], long)}. */
    public static final long DEFAULT_LOCAL_CAP = 256;

    private WorkSizes() {
    }

    /**
     * Number of work-groups per dimension: {@code ceil(global[d] / local[d])}.
     *
     * @throws IllegalArgumentException when the arrays are null, have different or
     *                                  unsupported lengths, or hold a non-positive size
     */
    public static long[] groupCounts(long[] global, long[] local) {
        validateGlobal(global);
        if (local == null) {
            throw new IllegalArgumentException("local work size must not be null when computing group counts");
        }
        if (local.length != global.length) {
            throw new IllegalArgumentException("local work size has " + local.length
                    + " dimensions but global has " + global.length);
        }
        long[] groups = new long[global.length];
        for (int d = 0; d < global.length; d++) {
            if (local[d] <= 0) {
                throw new IllegalArgumentException("local work size must be positive in dimension " + d
                        + ", got " + local[d]);
            }
            groups[d] = (global[d] - 1) / local[d] + 1;
        }
        return groups;
    }

    /**
     * The local size a backend uses when the caller passes none.
     *
     * <p>Dimension 0 gets the largest power of two that is at most
     * {@code min(256, maxWorkGroupSize)} and divides {@code global[0]} (at least 1);
     * every other dimension gets 1. Because the result divides {@code global},
     * no work-item beyond {@code global} is launched, so kernels without a tail
     * bounds check stay correct.</p>
     *
     * @throws IllegalArgumentException when {@code global} is invalid or
     *                                  {@code maxWorkGroupSize < 1}
     */
    public static long[] defaultLocal(long[] global, long maxWorkGroupSize) {
        validateGlobal(global);
        if (maxWorkGroupSize < 1) {
            throw new IllegalArgumentException("maxWorkGroupSize must be at least 1, got " + maxWorkGroupSize);
        }
        long cap = Long.highestOneBit(Math.min(DEFAULT_LOCAL_CAP, maxWorkGroupSize));
        long lowestSetBit = Long.lowestOneBit(global[0]);
        long[] local = new long[global.length];
        local[0] = Math.min(cap, lowestSetBit);
        for (int d = 1; d < local.length; d++) {
            local[d] = 1;
        }
        return local;
    }

    /**
     * Pads a 1-3 element size array to three elements.
     *
     * @param fill value for the missing trailing dimensions
     * @throws IllegalArgumentException when {@code v} is null or longer than three
     */
    public static long[] pad3(long[] v, long fill) {
        if (v == null || v.length < 1 || v.length > 3) {
            throw new IllegalArgumentException("work size must have 1 to 3 dimensions, got "
                    + (v == null ? "null" : String.valueOf(v.length)));
        }
        long[] out = new long[] {fill, fill, fill};
        System.arraycopy(v, 0, out, 0, v.length);
        return out;
    }

    /**
     * Validates a local size against the kernel's or device's work-group limit.
     *
     * @throws IllegalArgumentException when a dimension is not positive or the
     *                                  product exceeds {@code maxWorkGroupSize}
     */
    public static void validateLocal(long[] local, long maxWorkGroupSize, String kernel) {
        if (local == null || local.length < 1 || local.length > 3) {
            throw new IllegalArgumentException(kernel + ": local work size must have 1 to 3 dimensions, got "
                    + (local == null ? "null" : String.valueOf(local.length)));
        }
        for (int d = 0; d < local.length; d++) {
            if (local[d] <= 0) {
                throw new IllegalArgumentException(kernel + ": local work size must be positive in dimension "
                        + d + ", got " + local[d]);
            }
        }
        long total = product(local);
        if (total > maxWorkGroupSize) {
            throw new IllegalArgumentException(kernel + ": work-group size " + total
                    + " exceeds the limit of " + maxWorkGroupSize);
        }
    }

    /**
     * Product of all entries.
     *
     * @throws ArithmeticException when the product overflows a long
     */
    public static long product(long[] v) {
        if (v == null) {
            throw new IllegalArgumentException("work size must not be null");
        }
        long p = 1;
        for (long x : v) {
            p = Math.multiplyExact(p, x);
        }
        return p;
    }

    private static void validateGlobal(long[] global) {
        if (global == null || global.length < 1 || global.length > 3) {
            throw new IllegalArgumentException("global work size must have 1 to 3 dimensions, got "
                    + (global == null ? "null" : String.valueOf(global.length)));
        }
        for (int d = 0; d < global.length; d++) {
            if (global[d] <= 0) {
                throw new IllegalArgumentException("global work size must be positive in dimension " + d
                        + ", got " + global[d]);
            }
        }
    }
}

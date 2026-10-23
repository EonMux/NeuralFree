package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;

import java.util.Objects;

/**
 * Placement rules shared by the resident uploader ({@link GpuWeightUploader}) and the streamed layer sets
 * ({@link LayerWeightSet}), so a tensor gets the same device shards and the same output scale whichever path
 * uploads it:
 * <ul>
 *   <li>the largest single buffer a device both allocates and binds ({@link #maxBufferBytes});</li>
 *   <li>balanced row shards of whole units, one row or one expert ({@link #shardRows});</li>
 *   <li>which tensors are expert stacks ({@link #isExpert}), whose {@code .scale} is per expert rather than a scalar
 *       output scale.</li>
 * </ul>
 */
public final class WeightShards {

    private WeightShards() {
    }

    /**
     * {@code min(maxAllocSize, maxStorageBufferRange)} over the limits the device reports, or {@link Long#MAX_VALUE}
     * when it reports none.
     */
    public static long maxBufferBytes(DeviceCapabilities caps) {
        Objects.requireNonNull(caps, "caps");
        long max = Long.MAX_VALUE;
        long alloc = caps.getMaxAllocSizeBytes();
        if (alloc > 0) {
            max = Math.min(max, alloc);
        }
        long range = caps.getMaxStorageBufferRange();
        if (range > 0) {
            max = Math.min(max, range);
        }
        return max;
    }

    /**
     * Balanced row shards of whole units ({@code unitRows} rows, {@code unitBytes} bytes each) whose bytes stay within
     * {@code maxBufferBytes}: the fewest shards that fit, with unit counts differing by at most one, larger first.
     *
     * @throws IllegalArgumentException when a size is not positive, {@code rows} is not a multiple of
     *                                  {@code unitRows}, or one unit exceeds {@code maxBufferBytes}
     */
    public static long[] shardRows(long rows, long unitRows, long unitBytes, long maxBufferBytes) {
        if (rows <= 0 || unitRows <= 0 || unitBytes <= 0 || maxBufferBytes <= 0) {
            throw new IllegalArgumentException("shard sizes must be positive: rows " + rows + ", unit rows " + unitRows
                    + ", unit bytes " + unitBytes + ", buffer limit " + maxBufferBytes);
        }
        if (rows % unitRows != 0) {
            throw new IllegalArgumentException(rows + " rows are not whole units of " + unitRows + " rows");
        }
        if (unitBytes > maxBufferBytes) {
            throw new IllegalArgumentException("one unit of " + unitBytes + " bytes exceeds the buffer limit of "
                    + maxBufferBytes + " bytes");
        }
        long units = rows / unitRows;
        long maxUnits = maxBufferBytes / unitBytes;
        long n = units / maxUnits + (units % maxUnits == 0 ? 0 : 1);
        long[] out = new long[Math.toIntExact(n)];
        long base = units / n;
        long rem = units % n;
        for (int i = 0; i < n; i++) {
            out[i] = (base + (i < rem ? 1 : 0)) * unitRows;
        }
        return out;
    }

    /**
     * True for an expert stack: a tensor of rank 3 or more named {@code *_exps.weight} or {@code *_chexps.weight}
     * ({@code [cols, rows, n_expert]}).
     */
    public static boolean isExpert(String name, int rank) {
        Objects.requireNonNull(name, "name");
        return rank >= 3 && (name.endsWith("_exps.weight") || name.endsWith("_chexps.weight"));
    }
}

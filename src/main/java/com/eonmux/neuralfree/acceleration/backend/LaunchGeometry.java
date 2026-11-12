package com.eonmux.neuralfree.acceleration.backend;

import java.util.Arrays;

/**
 * The validated grid and block of a dispatch on a backend that launches by group count (CUDA, HIP), together with the
 * global and local sizes they were derived from. A kernel handle keeps the geometry of its last dispatch, so a
 * dispatch that repeats its sizes (every decode step, every replayed command) skips the derivation and its
 * allocations.
 *
 * <p>Immutable: the arrays are private copies, and {@link #grid()} and {@link #block()} return the held arrays, which
 * callers only read.</p>
 */
public final class LaunchGeometry {

    private final long[] global;
    private final long[] local;
    private final long[] grid;
    private final long[] block;

    /**
     * @param global the dispatch's global size
     * @param local  its local size, or null for the backend's default
     * @param grid   the group counts, padded to three dimensions
     * @param block  the block, padded to three dimensions
     */
    public LaunchGeometry(long[] global, long[] local, long[] grid, long[] block) {
        if (global == null || grid == null || block == null) {
            throw new IllegalArgumentException("global size, grid and block must not be null");
        }
        if (grid.length != 3 || block.length != 3) {
            throw new IllegalArgumentException("grid and block must have three dimensions, got " + grid.length
                    + " and " + block.length);
        }
        this.global = global.clone();
        this.local = local == null ? null : local.clone();
        this.grid = grid.clone();
        this.block = block.clone();
    }

    /** True when a dispatch of {@code global} and {@code local} (null: the default) has this geometry. */
    public boolean matches(long[] global, long[] local) {
        return Arrays.equals(this.global, global) && Arrays.equals(this.local, local);
    }

    /** The group counts, three dimensions; read-only. */
    public long[] grid() {
        return grid;
    }

    /** The block, three dimensions; read-only. */
    public long[] block() {
        return block;
    }

    @Override
    public String toString() {
        return "LaunchGeometry{global=" + Arrays.toString(global) + ", local=" + Arrays.toString(local) + ", grid="
                + Arrays.toString(grid) + ", block=" + Arrays.toString(block) + "}";
    }
}

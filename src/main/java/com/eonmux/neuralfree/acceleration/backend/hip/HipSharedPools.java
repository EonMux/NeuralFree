package com.eonmux.neuralfree.acceleration.backend.hip;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The buffer pools of the open {@link HipComputeBackend}s, grouped by the device their implicit
 * context belongs to. Several backends on one device allocate from one device, so memory one
 * backend keeps pooled is memory another backend's allocation can reclaim: an allocation that
 * runs out of memory trims its siblings' pools before failing, and free VRAM counts every pool
 * of the device. Devices are keyed by their ordinal, which is the same for every view of a
 * device. Safe for use from several threads.
 */
final class HipSharedPools {

    private static final Map<Integer, Set<HipBufferPool>> POOLS = new HashMap<>();

    private HipSharedPools() {
    }

    /**
     * Adds {@code pool} to the pools of {@code device}.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    static void register(int device, HipBufferPool pool) {
        requireArguments(device, pool);
        synchronized (POOLS) {
            POOLS.computeIfAbsent(device, key -> new HashSet<>()).add(pool);
        }
    }

    /**
     * Removes {@code pool} from the pools of {@code device}; nothing happens when it is not
     * registered.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    static void unregister(int device, HipBufferPool pool) {
        requireArguments(device, pool);
        synchronized (POOLS) {
            Set<HipBufferPool> pools = POOLS.get(device);
            if (pools != null && pools.remove(pool) && pools.isEmpty()) {
                POOLS.remove(device);
            }
        }
    }

    /** Bytes held by every registered pool of {@code device}. */
    static long pooledBytes(int device) {
        long total = 0;
        for (HipBufferPool pool : poolsOf(device)) {
            total += pool.pooledBytes();
        }
        return total;
    }

    /**
     * Frees the pooled blocks of every registered pool of {@code device} except {@code self}.
     * Every pool is attempted; a pool's failure is added to {@code failures}.
     *
     * @return the bytes freed
     */
    static long trimSiblings(int device, HipBufferPool self, List<RuntimeException> failures) {
        long freed = 0;
        for (HipBufferPool pool : poolsOf(device)) {
            if (pool == self || pool.pooledBytes() == 0) {
                continue;
            }
            try {
                freed += pool.trim();
            } catch (RuntimeException e) {
                failures.add(e);
            }
        }
        return freed;
    }

    private static List<HipBufferPool> poolsOf(int device) {
        synchronized (POOLS) {
            Set<HipBufferPool> pools = POOLS.get(device);
            return pools == null ? new ArrayList<>() : new ArrayList<>(pools);
        }
    }

    private static void requireArguments(int device, HipBufferPool pool) {
        if (pool == null) {
            throw new IllegalArgumentException("HIP buffer pool must not be null");
        }
    }
}

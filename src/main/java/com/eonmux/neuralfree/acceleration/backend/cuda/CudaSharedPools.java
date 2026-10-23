package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUcontext;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The buffer pools of the open {@link CudaComputeBackend}s, grouped by the primary context they
 * share. Several backends on one device allocate from one context, so memory one backend keeps
 * pooled is memory another backend's allocation can reclaim: an allocation that runs out of
 * memory trims its siblings' pools before failing, and free VRAM counts every pool of the
 * context. Contexts are keyed by their native handle, which is the same for every retain of a
 * device's primary context. Safe for use from several threads.
 */
final class CudaSharedPools {

    private static final Map<CUcontext, Set<CudaBufferPool>> POOLS = new HashMap<>();

    private CudaSharedPools() {
    }

    /**
     * Adds {@code pool} to the pools of {@code context}.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    static void register(CUcontext context, CudaBufferPool pool) {
        requireArguments(context, pool);
        synchronized (POOLS) {
            POOLS.computeIfAbsent(context, key -> new HashSet<>()).add(pool);
        }
    }

    /**
     * Removes {@code pool} from the pools of {@code context}; nothing happens when it is not registered.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    static void unregister(CUcontext context, CudaBufferPool pool) {
        requireArguments(context, pool);
        synchronized (POOLS) {
            Set<CudaBufferPool> pools = POOLS.get(context);
            if (pools != null && pools.remove(pool) && pools.isEmpty()) {
                POOLS.remove(context);
            }
        }
    }

    /** Bytes held by every registered pool of {@code context}. */
    static long pooledBytes(CUcontext context) {
        long total = 0;
        for (CudaBufferPool pool : poolsOf(context)) {
            total += pool.pooledBytes();
        }
        return total;
    }

    /**
     * Frees the pooled blocks of every registered pool of {@code context} except {@code self}.
     * Every pool is attempted; a pool's failure is added to {@code failures}.
     *
     * @return the bytes freed
     */
    static long trimSiblings(CUcontext context, CudaBufferPool self, List<RuntimeException> failures) {
        long freed = 0;
        for (CudaBufferPool pool : poolsOf(context)) {
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

    private static List<CudaBufferPool> poolsOf(CUcontext context) {
        synchronized (POOLS) {
            Set<CudaBufferPool> pools = POOLS.get(context);
            return pools == null ? new ArrayList<>() : new ArrayList<>(pools);
        }
    }

    private static void requireArguments(CUcontext context, CudaBufferPool pool) {
        if (context == null) {
            throw new IllegalArgumentException("CUDA context must not be null");
        }
        if (pool == null) {
            throw new IllegalArgumentException("CUDA buffer pool must not be null");
        }
    }
}

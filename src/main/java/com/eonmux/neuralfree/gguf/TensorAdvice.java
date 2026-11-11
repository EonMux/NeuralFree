package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.numa.NumaPlacement;

import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

/**
 * The paging advice of memory-mapped tensor data ({@value #PROP_ADVICE}), which {@link FileTensorRegions} gives
 * every mapped tensor and {@link MappedWeightStore} its mapped data region ({@link MemoryLock#advise}):
 * <ul>
 *   <li>{@code auto} (default): {@code random} while NUMA placement is active, as llama.cpp advises a mapped model
 *       on a NUMA host (where it skips its prefetch, {@code src/llama-mmap.cpp}); otherwise {@code willneed},
 *       llama.cpp's prefetch of a mapped model, unless the mapped data is larger than the host's RAM
 *       ({@link HostMemory}), where a prefetch of the whole model would only evict its own first pages, and the
 *       mapping keeps the kernel's default readahead;</li>
 *   <li>{@code willneed}, {@code sequential}, {@code random}, {@code hugepage}: that advice
 *       ({@link MemoryLock.Advice});</li>
 *   <li>{@code none}: no advice, the kernel's default readahead.</li>
 * </ul>
 * Only memory-mapped tensors are advised; read, direct and lazy loads have no mapping. The advice changes paging
 * only, never a value read.
 */
public final class TensorAdvice {

    /** The paging advice of mapped tensor data: auto, none, willneed, sequential, random or hugepage. */
    public static final String PROP_ADVICE = "neuralfree.tensor.advice";

    private TensorAdvice() {
    }

    /**
     * The advice {@code mappedBytes} of mapped tensor data take under {@code properties} on this host, or null for
     * none.
     *
     * @throws IllegalArgumentException when {@value #PROP_ADVICE} names no advice
     */
    public static MemoryLock.Advice forMapping(Properties properties, long mappedBytes) {
        return forMapping(properties, mappedBytes, HostMemory.totalBytes());
    }

    /**
     * The advice {@code mappedBytes} of mapped tensor data take under {@code properties} on a host of
     * {@code ramBytes} of RAM (-1: unknown, taken as large enough), or null for none.
     *
     * @throws IllegalArgumentException when {@value #PROP_ADVICE} names no advice
     */
    static MemoryLock.Advice forMapping(Properties properties, long mappedBytes, long ramBytes) {
        Objects.requireNonNull(properties, "properties");
        String v = properties.getProperty(PROP_ADVICE);
        String s = v == null ? "" : v.trim().toLowerCase(Locale.ROOT);
        switch (s) {
            case "":
            case "auto":
                if (NumaPlacement.activeIn(properties)) {
                    return MemoryLock.Advice.RANDOM;
                }
                return ramBytes > 0 && mappedBytes > ramBytes ? null : MemoryLock.Advice.WILLNEED;
            case "none":
                return null;
            case "willneed":
                return MemoryLock.Advice.WILLNEED;
            case "sequential":
                return MemoryLock.Advice.SEQUENTIAL;
            case "random":
                return MemoryLock.Advice.RANDOM;
            case "hugepage":
                return MemoryLock.Advice.HUGEPAGE;
            default:
                throw new IllegalArgumentException(PROP_ADVICE + "=" + v + " is invalid; use auto, none, willneed,"
                        + " sequential, random or hugepage");
        }
    }
}

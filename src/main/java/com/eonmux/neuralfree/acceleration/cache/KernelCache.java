package com.eonmux.neuralfree.acceleration.cache;

import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

@ThreadSafe("Uses ConcurrentHashMap with computeIfAbsent for atomic cache-or-compile. "
        + "compileCount is AtomicInteger, warmedUp is volatile. clear() and remove() are safe for concurrent use.")
public class KernelCache {
    private static final Logger LOGGER = LoggingUtil.getLogger(KernelCache.class);
    
    private final ConcurrentHashMap<CacheKey, KernelHandle> cache;
    private final AtomicInteger compileCount;
    private volatile boolean warmedUp;

    public KernelCache() {
        this.cache = new ConcurrentHashMap<>();
        this.compileCount = new AtomicInteger(0);
        this.warmedUp = false;
    }

    public KernelHandle getOrCompile(CacheKey key, Function<CacheKey, KernelHandle> compileFn) {
        return cache.computeIfAbsent(key, k -> {
            int count = compileCount.incrementAndGet();
            if (warmedUp) {
                LOGGER.warn("Kernel cache miss after warmup - kernel: {}, total compiles: {}", 
                        k.kernelName(), count);
            }
            LOGGER.debug("Kernel cache miss - compiling: {}, cache size: {}", 
                    k.kernelName(), cache.size() + 1);
            return compileFn.apply(k);
        });
    }
    
    public void markWarmedUp() {
        this.warmedUp = true;
        LOGGER.debug("Kernel cache marked as warmed up - {} kernels cached", cache.size());
    }

    public boolean contains(CacheKey key) {
        return cache.containsKey(key);
    }

    public int size() {
        return cache.size();
    }
    
    public int getCompileCount() {
        return compileCount.get();
    }

    public void clear() {
        cache.forEach((key, handle) -> {
            if (handle != null) {
                try {
                    handle.close();
                } catch (Exception e) {
                    LOGGER.debug("Exception closing kernel handle for {}: {}", key.kernelName(), e.getMessage());
                }
            }
        });
        cache.clear();
        compileCount.set(0);
        warmedUp = false;
    }

    public void remove(CacheKey key) {
        KernelHandle handle = cache.remove(key);
        if (handle != null) {
            try {
                handle.close();
            } catch (Exception e) {
                LOGGER.debug("Exception closing kernel handle for {}: {}", key.kernelName(), e.getMessage());
            }
        }
    }
}

package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.LogManager;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Host tier of weight streaming: reads tensor bytes from the {@link WeightStore} and keeps an execution-order-aware
 * RAM cache of the streamed per-layer tensors.
 *
 * <p>The streamed tensors are given per layer (layer {@code L} = physical weight layer {@code L}, executed in the
 * cyclic order {@code 0 .. layerCount-1}). Caching and eviction use that order instead of a generic LRU: the
 * background prefetcher loads the layers just beyond the VRAM window ({@link #setVramDepth(int)}) and evicts the
 * layer whose next use is farthest away.</p>
 *
 * <p>Tier hierarchy:</p>
 * <ul>
 *   <li><b>VRAM</b>: the device slots of
 *       {@link com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource}.</li>
 *   <li><b>RAM</b>: this class's cache, bounded by {@code maxCacheBytes} ({@code neuralfree.ram.cache.mb}).</li>
 *   <li><b>Disk</b>: the {@link WeightStore} (mmap backed by the OS page cache).</li>
 * </ul>
 *
 * <p>Every tensor of the file is addressable by name: {@link #readRange} reads any byte range of any tensor. The store
 * is not owned: the caller closes it after this provider.</p>
 *
 * <p>Thread safety: {@link #advanceExecution(int)} schedules the prefetch on one background thread. Per-layer locks
 * guard the cache slots and {@code currentCacheBytes} is atomic, so reads may run concurrently with the
 * prefetcher.</p>
 */
public final class WeightDataProvider implements AutoCloseable {

    private static final long MIB = 1024L * 1024L;
    /** Prefetch window when the per-layer size is unknown. */
    private static final int DEFAULT_RAM_WINDOW = 20;

    private final WeightStore store;
    private final Map<String, TensorRef> tensorIndex;
    private final String[][] streamedNames;
    private final int layerCount;
    private final long maxCacheBytes;
    private final long perLayerBytes;
    private volatile int vramDepth;

    private final byte[][][] ramCache;
    private final Object[] layerLocks;
    private final AtomicLong currentCacheBytes = new AtomicLong(0);

    private volatile int currentLayer;

    private final ExecutorService prefetchExecutor;
    private final AtomicBoolean prefetchScheduled = new AtomicBoolean(false);
    private volatile boolean closed;

    /** Location of one tensor in the store and, for a streamed tensor, its cache slot. */
    private static final class TensorRef {
        final String name;
        final long offset;
        final long size;
        final int layer;
        final int slot;

        TensorRef(String name, long offset, long size, int layer, int slot) {
            this.name = name;
            this.offset = offset;
            this.size = size;
            this.layer = layer;
            this.slot = slot;
        }

        boolean streamed() {
            return layer >= 0;
        }
    }

    /**
     * Provider over explicit per-layer streamed tensors, with the RAM cache size from
     * {@link EngineConfig#ramCacheMb()} and the VRAM depth from {@link EngineConfig#vramDepth()}.
     *
     * @param streamedByLayer element {@code L}: the stored names of the tensors streamed for weight layer {@code L}
     * @throws IllegalArgumentException as {@link #WeightDataProvider(WeightStore, List, List, int, long)}
     */
    public WeightDataProvider(WeightStore store, List<GGUFTensorInfo> tensors,
                              List<? extends List<String>> streamedByLayer, EngineConfig config) {
        this(store, tensors, streamedByLayer, Objects.requireNonNull(config, "config").vramDepth(),
                ramCacheBytes(config));
    }

    /**
     * Provider over explicit per-layer streamed tensors.
     *
     * @param store           tensor data access (not owned)
     * @param tensors         all tensor infos of the GGUF file
     * @param streamedByLayer element {@code L}: the stored names of the tensors streamed for weight layer {@code L}
     * @param vramDepth       layers resident in VRAM; 0 or negative = not known yet
     * @param maxCacheBytes   maximum heap bytes of the RAM cache; 0 disables it
     * @throws IllegalArgumentException when a tensor name repeats in the file, a streamed name is not in the file or
     *                                  is listed for two layers, or {@code maxCacheBytes} is negative
     */
    public WeightDataProvider(WeightStore store, List<GGUFTensorInfo> tensors,
                              List<? extends List<String>> streamedByLayer, int vramDepth, long maxCacheBytes) {
        this.store = Objects.requireNonNull(store, "store");
        Objects.requireNonNull(tensors, "tensors");
        Objects.requireNonNull(streamedByLayer, "streamedByLayer");
        if (maxCacheBytes < 0) {
            throw new IllegalArgumentException("RAM cache size must be >= 0 bytes, got " + maxCacheBytes);
        }
        this.layerCount = streamedByLayer.size();
        this.maxCacheBytes = maxCacheBytes;
        this.vramDepth = Math.max(0, vramDepth);

        Map<String, GGUFTensorInfo> infos = new HashMap<>(tensors.size() * 4 / 3 + 1);
        for (GGUFTensorInfo t : tensors) {
            if (infos.put(t.name(), t) != null) {
                throw new IllegalArgumentException("tensor '" + t.name() + "' is listed twice in the GGUF file");
            }
        }
        Map<String, int[]> streamedAt = new HashMap<>();
        this.streamedNames = new String[layerCount][];
        long maxLayerBytes = 0;
        for (int l = 0; l < layerCount; l++) {
            List<String> names = Objects.requireNonNull(streamedByLayer.get(l), "streamed tensors of layer " + l);
            streamedNames[l] = names.toArray(new String[0]);
            long layerBytes = 0;
            for (int s = 0; s < streamedNames[l].length; s++) {
                String name = streamedNames[l][s];
                GGUFTensorInfo info = infos.get(name);
                if (info == null) {
                    throw new IllegalArgumentException("streamed tensor '" + name + "' of layer " + l
                            + " is not in the GGUF file");
                }
                int[] prev = streamedAt.put(name, new int[]{l, s});
                if (prev != null) {
                    throw new IllegalArgumentException("tensor '" + name + "' is streamed for layers " + prev[0]
                            + " and " + l);
                }
                layerBytes += info.sizeInBytes();
            }
            maxLayerBytes = Math.max(maxLayerBytes, layerBytes);
        }
        this.perLayerBytes = maxLayerBytes;

        this.tensorIndex = new HashMap<>(tensors.size() * 4 / 3 + 1);
        for (GGUFTensorInfo t : tensors) {
            int[] at = streamedAt.get(t.name());
            tensorIndex.put(t.name(), new TensorRef(t.name(), t.offset(), t.sizeInBytes(),
                    at != null ? at[0] : -1, at != null ? at[1] : -1));
        }

        this.ramCache = new byte[layerCount][][];
        this.layerLocks = new Object[layerCount];
        for (int l = 0; l < layerCount; l++) {
            ramCache[l] = new byte[streamedNames[l].length][];
            layerLocks[l] = new Object();
        }

        this.prefetchExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "weight-prefetch");
            t.setDaemon(true);
            t.setPriority(Thread.MIN_PRIORITY);
            return t;
        });
    }

    /**
     * Heap bytes of the RAM cache for a configuration: {@link EngineConfig#RAM_CACHE_AUTO} is half the maximum heap,
     * 0 disables the cache, any other value is that many MiB.
     */
    public static long ramCacheBytes(EngineConfig config) {
        long mb = Objects.requireNonNull(config, "config").ramCacheMb();
        long maxHeap = Runtime.getRuntime().maxMemory();
        if (mb == EngineConfig.RAM_CACHE_AUTO) {
            return maxHeap / 2;
        }
        long bytes = mb > Long.MAX_VALUE / MIB ? Long.MAX_VALUE : mb * MIB;
        if (bytes > maxHeap) {
            LogManager.getLogger(WeightDataProvider.class).warn(
                    "{}={} MiB exceeds the maximum heap of {} MiB; the RAM cache may run the JVM out of memory",
                    EngineConfig.PROP_RAM_CACHE_MB, mb, maxHeap / MIB);
        }
        return bytes;
    }

    /** Sets the number of layers resident in VRAM; the RAM prefetch window starts that many layers ahead. */
    public void setVramDepth(int vramDepth) {
        this.vramDepth = Math.max(0, vramDepth);
    }

    /** Layers resident in VRAM (0 = not known). */
    public int vramDepth() {
        return vramDepth;
    }

    /** Number of weight layers with streamed tensors. */
    public int layerCount() {
        return layerCount;
    }

    /** Maximum heap bytes of the RAM cache. */
    public long maxCacheBytes() {
        return maxCacheBytes;
    }

    /** Checks whether a tensor name exists in the file (no disk read). */
    public boolean containsKey(String name) {
        return tensorIndex.containsKey(name);
    }

    /**
     * Puts bytes {@code [byteOffset, byteOffset + size)} of a tensor into {@code dst} at its position, from the RAM
     * cache when the tensor is cached, else from the store.
     *
     * @throws IOException               when the file has no such tensor or the read fails
     * @throws IndexOutOfBoundsException when the range is outside the tensor or {@code dst} has too little room
     */
    public void readRange(String name, long byteOffset, int size, ByteBuffer dst) throws IOException {
        readRange(require(name), byteOffset, size, dst);
    }

    private void readRange(TensorRef ref, long byteOffset, int size, ByteBuffer dst) throws IOException {
        Objects.requireNonNull(dst, "dst");
        if (byteOffset < 0 || size < 0 || byteOffset > ref.size - size) {
            throw new IndexOutOfBoundsException("range [" + byteOffset + ", " + (byteOffset + size)
                    + ") is outside tensor '" + ref.name + "' of " + ref.size + " bytes");
        }
        if (dst.remaining() < size) {
            throw new IndexOutOfBoundsException("destination has " + dst.remaining() + " bytes left, reading " + size
                    + " bytes of tensor '" + ref.name + "'");
        }
        byte[] cached = cachedLayerWeight(ref);
        if (cached != null) {
            dst.put(cached, (int) byteOffset, size);
            return;
        }
        store.readTensorDirect(ref.offset + byteOffset, size, dst);
    }

    private TensorRef require(String name) throws IOException {
        TensorRef ref = tensorIndex.get(Objects.requireNonNull(name, "name"));
        if (ref == null) {
            throw new IOException("the GGUF file has no tensor named '" + name + "'");
        }
        return ref;
    }

    /**
     * Records the execution position (weight layer about to run) and schedules the RAM prefetch of the layers beyond
     * the VRAM window. A no-op when every layer is resident in VRAM or the cache is disabled.
     *
     * @throws IllegalArgumentException when the layer is outside {@code [0, layerCount)}
     */
    public void advanceExecution(int layer) {
        if (layer < 0 || layer >= layerCount) {
            throw new IllegalArgumentException("weight layer " + layer + " outside [0, " + layerCount + ")");
        }
        this.currentLayer = layer;
        if (vramDepth >= layerCount || maxCacheBytes == 0 || closed) {
            return;
        }
        if (!prefetchScheduled.compareAndSet(false, true)) {
            return;
        }
        prefetchExecutor.submit(() -> {
            prefetchScheduled.set(false);
            int from = currentLayer;
            try {
                prefetchAhead(from);
            } catch (RuntimeException e) {
                LoggingUtil.getLogger(this).warn("RAM prefetch from layer {} failed: {}", from, e.getMessage());
            }
        });
    }

    /**
     * Loads the RAM prefetch window of a cold start (before the first token): layers
     * {@code [vramDepth, vramDepth + window)}.
     */
    public void warmup() {
        if (vramDepth >= layerCount || maxCacheBytes == 0) {
            return;
        }
        int ramWindow = computeRamWindow();
        LoggingUtil.getLogger(this).info("WeightDataProvider warmup: pre-loading {} layers into RAM cache", ramWindow);
        for (int i = 0; i < ramWindow && i + vramDepth < layerCount; i++) {
            loadLayerIntoCache((vramDepth + i) % layerCount);
        }
    }

    /** Returns current RAM cache usage in bytes. */
    public long currentCacheBytes() {
        return currentCacheBytes.get();
    }

    /** Stops the prefetcher and drops the RAM cache; the store stays open. Idempotent. */
    @Override
    public void close() {
        closed = true;
        prefetchExecutor.shutdownNow();
        try {
            if (!prefetchExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                LoggingUtil.getLogger(this).warn("Prefetch executor did not terminate in time");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        for (int l = 0; l < layerCount; l++) {
            synchronized (layerLocks[l]) {
                byte[][] slots = ramCache[l];
                for (int s = 0; s < slots.length; s++) {
                    if (slots[s] != null) {
                        currentCacheBytes.addAndGet(-slots[s].length);
                        slots[s] = null;
                    }
                }
            }
        }
    }

    // --- Private implementation ---

    private byte[] cachedLayerWeight(TensorRef ref) {
        if (!ref.streamed()) {
            return null;
        }
        synchronized (layerLocks[ref.layer]) {
            return ramCache[ref.layer][ref.slot];
        }
    }

    private byte[] loadFromStore(TensorRef ref) {
        try {
            byte[] data = new byte[(int) ref.size];
            store.readTensor(ref.offset, (int) ref.size, data);
            return data;
        } catch (IOException e) {
            LoggingUtil.getLogger(this).error("Failed to read tensor '{}' at offset {}: {}", ref.name, ref.offset,
                    e.getMessage());
            return null;
        }
    }

    private void cacheLayerWeight(TensorRef ref, byte[] data) {
        if (data.length > maxCacheBytes) {
            return;
        }
        int maxEvictions = layerCount;
        while (currentCacheBytes.get() + data.length > maxCacheBytes && maxEvictions-- > 0) {
            if (!evictFartherThan(ref.layer)) {
                return;
            }
        }
        if (currentCacheBytes.get() + data.length > maxCacheBytes) {
            return;
        }
        synchronized (layerLocks[ref.layer]) {
            if (ramCache[ref.layer][ref.slot] == null) {
                ramCache[ref.layer][ref.slot] = data;
                currentCacheBytes.addAndGet(data.length);
            }
        }
    }

    /** Distance in the cyclic execution order from the current layer to the next use of {@code layer}. */
    private int nextUseDistance(int layer, int cur) {
        return ((layer - cur) % layerCount + layerCount) % layerCount;
    }

    /**
     * Evicts the cached layer whose next use is farthest away, provided it is needed later than {@code forLayer}
     * (a layer needed sooner is never evicted to make room for one needed later).
     *
     * @return true when a layer was evicted
     */
    private boolean evictFartherThan(int forLayer) {
        int cur = currentLayer;
        int limit = nextUseDistance(forLayer, cur);
        for (int dist = layerCount - 1; dist > limit; dist--) {
            int candidate = (cur + dist) % layerCount;
            synchronized (layerLocks[candidate]) {
                boolean hadData = false;
                byte[][] slots = ramCache[candidate];
                for (int s = 0; s < slots.length; s++) {
                    if (slots[s] != null) {
                        currentCacheBytes.addAndGet(-slots[s].length);
                        slots[s] = null;
                        hadData = true;
                    }
                }
                if (hadData) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Prefetches layers {@code [fromLayer + vramDepth, fromLayer + vramDepth + window)} into the RAM cache. */
    private void prefetchAhead(int fromLayer) {
        int ramWindow = computeRamWindow();
        for (int i = 0; i < ramWindow; i++) {
            if (closed || Thread.currentThread().isInterrupted()) {
                return;
            }
            loadLayerIntoCache((fromLayer + vramDepth + i) % layerCount);
        }
    }

    private void loadLayerIntoCache(int layer) {
        String[] names = streamedNames[layer];
        for (int s = 0; s < names.length; s++) {
            synchronized (layerLocks[layer]) {
                if (ramCache[layer][s] != null) {
                    continue;
                }
            }
            TensorRef ref = tensorIndex.get(names[s]);
            if (ref.size > Integer.MAX_VALUE || ref.size > maxCacheBytes) {
                continue;
            }
            byte[] data = loadFromStore(ref);
            if (data != null) {
                cacheLayerWeight(ref, data);
            }
        }
    }

    private int computeRamWindow() {
        if (vramDepth >= layerCount || maxCacheBytes == 0) {
            return 0;
        }
        int beyond = layerCount - vramDepth;
        if (perLayerBytes <= 0) {
            return Math.min(DEFAULT_RAM_WINDOW, beyond);
        }
        return (int) Math.min(maxCacheBytes / perLayerBytes, beyond);
    }
}

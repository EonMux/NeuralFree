package com.eonmux.neuralfree.gguf.engine.lora;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The runtime adapter management of one model (llama.cpp server {@code /lora-adapters},
 * server-context.cpp:6617,6645; vLLM {@code /v1/load_lora_adapter}; SGLang {@code lora/lora_manager.py:279,385}):
 * <ul>
 *   <li>{@link #load}: reads an adapter GGUF (LLM) or a PEFT directory ({@code adapter_config.json} with
 *       {@code adapter_model.safetensors}, HF.md LA-1) through {@link LoraAdapterLoader}, which validates the base
 *       architecture and the tensor shapes against the model, refuses the loader's unsupported forms (activated
 *       LoRA, input transforms), and refuses pairs above {@link LoraConfig#maxRank()}; the handle becomes resident,
 *       evicting to the host cache when residents exceed {@link LoraConfig#maxAdapters()} and refusing when every
 *       resident is held, and its {@link com.eonmux.neuralfree.gguf.engine.lora.LoraListener} (an adapter
 *       attachment) holds its deltas;</li>
 *   <li>{@link #unload}: removes the deltas of a resident adapter through the event and caches the handle
 *       ({@link LoraConfig#cacheEntries()}, host LRU of unloaded adapters, vLLM {@code max_cpu_loras}); a later
 *       load of the same name replaces the cached entry (two loads never share in-process identity, §5.5.1
 *       N-2);</li>
 *   <li>{@link #setScale}: changes the scale of a resident adapter; a held one rescales its {@code scale * B}
 *       buffers in place on the engine thread and creates a new epoch ({@link LoraHandle#setScale}), a per-request
 *       one sets the default its later requests use (LR-2) and rewrites nothing;</li>
 *   <li>{@link #list} and {@link #handle}: the residents in load order, and the resident or cached entry by
 *       name.</li>
 * </ul>
 * Event delivery is synchronous, before the registry call returns, on the engine thread the caller owns (§3.3) —
 * no listener runs on another thread and no change is acknowledged before it is applied. PC-1's
 * {@code ParameterChanges} applies the epochs, statute fencing and invalidations of §5.5.1 O-6 as one listener.
 * Static adapters of an engine configuration ({@code neuralfree.lora}) keep loading through
 * {@code LoraAttachment.Binding.load} alone; the registry covers the runtime surface. Mutable and owned by the
 * engine thread, like {@code LoraAttachment}; one registry per loaded model.</p>
 */
public final class LoraRegistry {

    /**
     * The adapter opening step behind {@link #load}: the production loader over a base model, or a test's in-memory
     * adapter source.
     */
    public interface Loader {

        /** Opens the adapter at {@code path}, validating it against the base model.
         *
         * @throws IOException when the file cannot be read
         */
        LoraAdapter open(String path) throws IOException;
    }

    private final Loader loader;
    private final LoraConfig config;
    /** The resident adapters, load order first — the eviction order for the unheld ones. */
    private final Map<String, LoraHandle> residents = new LinkedHashMap<>();
    /** The host cache of unloaded adapters, most recent kept (vLLM max_cpu_loras). */
    private final Map<String, LoraHandle> cache = new LinkedHashMap<>();
    private final List<LoraListener> listeners = new ArrayList<>();

    private LoraRegistry(Loader loader, LoraConfig config) {
        this.loader = loader;
        this.config = config;
    }

    /**
     * The registry over the base model {@code base}: every {@link #load} opens through
     * {@link LoraAdapterLoader#load}, whose arch, tensor-shape, input-transform and rank validation run against the
     * base weights.
     */
    public static LoraRegistry open(CpuWeights base, ModelSourceOptions options, LoraConfig config) {
        Objects.requireNonNull(base, "base");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(config, "config");
        return of(path -> LoraAdapterLoader.load(path, base, options), config);
    }

    /** The registry whose {@link #load} opens adapters through {@code loader}. */
    public static LoraRegistry of(Loader loader, LoraConfig config) {
        Objects.requireNonNull(loader, "loader");
        Objects.requireNonNull(config, "config");
        return new LoraRegistry(loader, config);
    }

    /** The configuration the registry follows. */
    public LoraConfig config() {
        return config;
    }

    /** Adds {@code listener}; every following change is delivered to it before the change returns. */
    public void addListener(LoraListener listener) {
        if (listener != null) {
            listeners.add(listener);
        }
    }

    /** Removes {@code listener}. */
    public void removeListener(LoraListener listener) {
        listeners.remove(listener);
    }

    /**
     * Makes the adapter at {@code path} resident under {@code name} at {@code scale}, which a new attachment holds
     * from now on.
     *
     * @throws IOException                  when the file cannot be read or opens invalid
     * @throws IllegalArgumentException     for a blank name, a non-finite scale or a pair rank above
     *                                      {@link LoraConfig#maxRank()} (the offending tensor named)
     * @throws IllegalStateException        for a resident name, or a load past
     *                                      {@link LoraConfig#maxAdapters()} when every resident is held by an
     *                                      attachment
     */
    public LoraHandle load(String name, String path, float scale) throws IOException {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("a LoRA adapter needs a registry name, got '" + name + "'");
        }
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("LoRA adapter " + name + " has a non-finite scale " + scale);
        }
        if (residents.containsKey(name)) {
            throw new IllegalStateException("LoRA adapter " + name + " is already resident; unload it first");
        }
        LoraAdapter adapter = loader.open(Objects.requireNonNull(path, "path"));
        if (adapter == null) {
            throw new IOException("the adapter source returned no adapter for " + path);
        }
        for (String target : adapter.targets()) {
            int rank = adapter.pair(target).rank();
            if (rank > config.maxRank()) {
                throw new IllegalArgumentException("the rank " + rank + " of LoRA tensor " + target + " in " + path
                        + " is above " + LoraConfig.PROP_MAX_RANK + "=" + config.maxRank());
            }
        }
        while (residents.size() >= config.maxAdapters()) {
            if (evictOldestUnheld() == null) {
                throw new IllegalStateException(LoraConfig.PROP_MAX + "=" + config.maxAdapters() + " and every"
                        + " resident " + residents.keySet() + " is held by an attachment; unload a resident"
                        + " before loading " + name);
            }
        }
        LoraHandle handle = new LoraHandle(name, adapter, scale);
        residents.put(name, handle);
        cache.remove(name);
        try {
            for (LoraListener l : new ArrayList<>(listeners)) {
                l.loaded(handle);
            }
        } catch (RuntimeException e) {
            residents.remove(name);
            throw e;
        }
        return handle;
    }

    /** {@link #load(String, String, float)} at scale 1.0. */
    public LoraHandle load(String name, String path) throws IOException {
        return load(name, path, 1.0f);
    }

    /**
     * Removes the resident adapter {@code name}: the event detaches its deltas and the handle moves to the host
     * cache. A cached entry outside {@link LoraConfig#cacheEntries()} is dropped first.
     *
     * @throws IllegalArgumentException when {@code name} is resident nowhere
     */
    public void unload(String name) {
        LoraHandle handle = residents.remove(name);
        if (handle == null) {
            throw new IllegalArgumentException("LoRA adapter " + name + " is not resident (residents: "
                    + residents.keySet() + ", cached: " + cache.keySet() + ")");
        }
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.unloaded(handle);
        }
        while (cache.size() >= config.cacheEntries()) {
            evictOldestCached();
        }
        cache.put(name, handle);
    }

    /**
     * Changes the scale of the resident adapter {@code name}: a held one rescales its {@code scale * B} buffers in
     * place between steps, on the engine thread, and creates a new epoch ({@link LoraHandle#revision()} rises,
     * also when the value returns to an earlier one, §5.5.1 N-2); a per-request one sets the default of its later
     * requests and rewrites nothing held.
     *
     * @throws IllegalArgumentException for a non-finite {@code scale} or an adapter resident nowhere
     */
    public void setScale(String name, float scale) {
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("LoRA adapter " + name + " has a non-finite scale " + scale);
        }
        LoraHandle handle = residents.get(name);
        if (handle == null) {
            throw new IllegalArgumentException("LoRA adapter " + name + " is not resident (residents: "
                    + residents.keySet() + ", cached: " + cache.keySet() + ")");
        }
        float previous = handle.scale();
        if (handle.holds() > 0) {
            handle.setScale(scale);
        } else {
            handle.useDefaultScale(scale);
        }
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.scaleChanged(handle, previous);
        }
    }

    /** The resident handles in load order. The cached unloadeds are absent. */
    public List<LoraHandle> list() {
        return Collections.unmodifiableList(new ArrayList<>(residents.values()));
    }

    /** The resident or cached adapter by name, or null. */
    public LoraHandle handle(String name) {
        LoraHandle h = residents.get(name);
        return h != null ? h : cache.get(name);
    }

    /** The host cache entries, oldest first. */
    public List<LoraHandle> cached() {
        return Collections.unmodifiableList(new ArrayList<>(cache.values()));
    }

    /** Evicts the load-order-first unheld resident for {@code null}: null when every resident is held. */
    private LoraHandle evictOldestUnheld() {
        for (Map.Entry<String, LoraHandle> e : residents.entrySet()) {
            if (e.getValue().holds() == 0) {
                evict(e.getKey(), e.getValue());
                return e.getValue();
            }
        }
        return null;
    }

    /** Drops the oldest host cache entry. */
    private LoraHandle evictOldestCached() {
        String oldest = cache.isEmpty() ? null : cache.keySet().iterator().next();
        if (oldest == null) {
            return null;
        }
        LoraHandle evicted = cache.remove(oldest);
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.unloaded(evicted);
        }
        return evicted;
    }

    private void evict(String name, LoraHandle handle) {
        residents.remove(name);
        while (cache.size() >= config.cacheEntries()) {
            evictOldestCached();
        }
        cache.put(name, handle);
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.unloaded(handle);
        }
    }

    @Override
    public String toString() {
        return "LoraRegistry{" + residents.values() + ", cached " + cache.values() + "}";
    }
}

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
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The runtime adapter management of one model (llama.cpp server {@code /lora-adapters},
 * server-context.cpp:6617,6645; vLLM {@code /v1/load_lora_adapter}; SGLang {@code lora/lora_manager.py:279,385}):
 * <ul>
 *   <li>{@link #load}: reads an adapter GGUF (LLM) or a PEFT directory ({@code adapter_config.json} with
 *       {@code adapter_model.safetensors}) through {@link LoraAdapterLoader}, which validates the base
 *       architecture and the tensor shapes against the model, refuses the loader's unsupported forms (input
 *       transforms), and refuses pairs above {@link LoraConfig#maxRank()}; the handle becomes resident, evicting to
 *       the host cache when residents exceed {@link LoraConfig#maxAdapters()} and refusing when every resident is
 *       held, and its {@link com.eonmux.neuralfree.gguf.engine.lora.LoraListener} (an adapter attachment) holds its
 *       deltas;</li>
 *   <li>{@link #adoptStartup}: makes a startup adapter ({@code --lora}, llama.cpp lists it first in
 *       {@code /lora-adapters}) resident under a name from its path, held in slot 0 by the attachment for the
 *       model's lifetime. The startup adapters at their scales form the {@link #defaultKey}, the slot-0 state the
 *       rows of requests that select no adapter run; they count against no limit and are neither unloaded,
 *       rescaled nor evicted;</li>
 *   <li>{@link #unload}: removes the deltas of a resident adapter through the event and caches the handle
 *       ({@link LoraConfig#cacheEntries()}, host LRU of unloaded adapters, vLLM {@code max_cpu_loras}); a later
 *       load of the same name replaces the cached entry (two loads never share in-process identity);</li>
 *   <li>{@link #setScale}: changes the scale of a resident adapter; a held one rescales its {@code scale * B}
 *       buffers in place on the engine thread and creates a new epoch ({@link LoraHandle#setScale}), a per-request
 *       one sets the default its later requests use and rewrites nothing;</li>
 *   <li>{@link #list} and {@link #handle}: the residents in load order, and the resident or cached entry by
 *       name;</li>
 *   <li>the per-request slot table (vLLM's {@code max_loras} GPU slots, lora/model_manager.py
 *       {@code activate_adapter}): {@link #resolve} turns a request's {@link LoraSelection} into the
 *       {@link LoraSlotKey} of resident adapters at fixed scales — the complete set its rows run, startup adapters
 *       included only when selected — {@link #acquire} gives the request a slot of that key — an open one shared
 *       with equal keys, a free one, or the least recently used idle one displaced — and {@link #release} returns
 *       it when the request ends. An opened slot stays open while idle, so later requests of the key reuse its
 *       registered deltas; {@link LoraListener#slotOpened}, {@link LoraListener#slotRescaled} and
 *       {@link LoraListener#slotClosed} carry the changes — a key of the startup adapters at other scales opens as
 *       a rescale, whose slot runs the resident slot-0 deltas at the key's factor of their bound scales and uploads
 *       no copy, so a startup adapter above {@link LoraConfig#maxRank()} rescales within the slot reservation too.
 *       {@link LoraConfig#maxAdapters()} bounds the slots as it bounds the residents. The bare key (no adapter)
 *       runs in the permanent {@link #bareSlot}, outside the table. An adapter a running request holds in a slot is
 *       neither unloaded nor evicted;</li>
 *   <li>{@link #useSlot}: the slot of the passes whose batch sets no per-row slots, for a single-sequence run or a
 *       pooling call of one request ({@link LoraListener#sessionSlot}).</li>
 * </ul>
 * Event delivery is synchronous, before the registry call returns, on the engine thread the caller owns —
 * no listener runs on another thread and no change is acknowledged before it is applied.
 * Static adapters of an engine configuration ({@code neuralfree.lora}) load through
 * {@code LoraAttachment.Binding.load} and the serving attachment adopts them here ({@link #adoptStartup}). Mutable and owned by the
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
    /** The per-request slot table, slots {@code 1..maxAdapters}; a null entry is free (index 0 is slot 0). */
    private final OpenSlot[] slots;
    /** The use clock ordering idle slots for displacement. */
    private long clock;

    /** One open slot: its key, the requests holding it and its last use. */
    private static final class OpenSlot {
        final LoraSlotKey key;
        int refs;
        long lastUse;

        OpenSlot(LoraSlotKey key) {
            this.key = key;
        }
    }

    private LoraRegistry(Loader loader, LoraConfig config) {
        this.loader = loader;
        this.config = config;
        this.slots = new OpenSlot[config.maxAdapters() + 1];
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
        while (requestResidents() >= config.maxAdapters()) {
            if (evictOldestUnheld() == null) {
                throw new IllegalStateException(LoraConfig.PROP_MAX + "=" + config.maxAdapters() + " and every"
                        + " resident " + residents.keySet() + " is held by an attachment or a running request;"
                        + " unload a resident before loading " + name);
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

    /** The residents that are not startup adapters, the ones {@link LoraConfig#maxAdapters()} bounds. */
    private int requestResidents() {
        int n = 0;
        for (LoraHandle h : residents.values()) {
            n += h.startup() ? 0 : 1;
        }
        return n;
    }

    /**
     * Makes the startup adapter {@code adapter} resident at {@code scale} under a name taken from its path (the file
     * name without {@code .gguf}, or the PEFT directory name; {@code -2}, {@code -3}, ... appended to a name already
     * resident) and returns its handle, which the caller holds in slot 0 for the model's lifetime. No listener event
     * runs: the caller is the attachment that applies it.
     *
     * @throws IllegalArgumentException for a non-finite scale, or an activated LoRA among other startup adapters
     *                                  (an activated LoRA runs alone)
     * @throws IllegalStateException    once a per-request adapter is resident (startup adapters come first)
     */
    public LoraHandle adoptStartup(LoraAdapter adapter, float scale) {
        Objects.requireNonNull(adapter, "adapter");
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("LoRA adapter " + adapter.path() + " has a non-finite scale " + scale);
        }
        for (LoraHandle h : residents.values()) {
            if (h.startup() && (adapter.isActivated() || h.adapter().isActivated())) {
                String alora = adapter.isActivated() ? adapter.path() : h.adapter().path();
                throw new IllegalArgumentException("--lora: " + alora + " is an activated LoRA (aLoRA), which runs"
                        + " alone; load it without the other startup adapters");
            }
        }
        if (requestResidents() > 0) {
            throw new IllegalStateException("startup LoRA adapters are adopted before any per-request adapter loads");
        }
        String base = startupName(adapter.path());
        String name = base;
        for (int i = 2; residents.containsKey(name); i++) {
            name = base + "-" + i;
        }
        LoraHandle handle = new LoraHandle(name, adapter, scale, true);
        residents.put(name, handle);
        return handle;
    }

    /** The registry name of a startup adapter at {@code path}: its last path element without {@code .gguf}. */
    static String startupName(String path) {
        String trimmed = path;
        while (trimmed.length() > 1 && (trimmed.endsWith("/") || trimmed.endsWith("\\"))) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        int cut = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
        String last = trimmed.substring(cut + 1);
        if (last.toLowerCase(Locale.ROOT).endsWith(".gguf") && last.length() > 5) {
            last = last.substring(0, last.length() - 5);
        }
        return last.isEmpty() ? "lora" : last.replace(',', '_').replace('=', '_');
    }

    /**
     * The key of slot 0: the startup adapters at their scales (those at scale 0 left out), the bare key when
     * there is none.
     */
    public LoraSlotKey defaultKey() {
        List<LoraSlotKey.Part> parts = new ArrayList<>();
        for (LoraHandle h : residents.values()) {
            if (h.startup() && h.scale() != 0.0f) {
                parts.add(new LoraSlotKey.Part(h, h.scale()));
            }
        }
        return parts.isEmpty() ? LoraSlotKey.bare() : new LoraSlotKey(parts);
    }

    /**
     * Whether a slot of {@code key} would copy the startup set: true when every adapter of {@code key} is a
     * startup adapter of slot 0 at a non-zero scale, so the slot can run the resident deltas at the key's factor
     * of their bound scales instead of uploading a copy ({@link LoraListener#slotRescaled}) — which a startup
     * adapter above {@link LoraConfig#maxRank()} can only do, since its copy would exceed the slot reservation.
     * False for the key equal to the default (no slot at all) and for a key that drops or adds adapters.
     */
    public boolean isRescale(LoraSlotKey key) {
        if (key.isBare()) {
            return false;
        }
        LoraSlotKey def = defaultKey();
        if (def.isBare() || key.parts().size() != def.parts().size()) {
            return false;
        }
        for (LoraSlotKey.Part p : key.parts()) {
            LoraSlotKey.Part resident = null;
            for (LoraSlotKey.Part d : def.parts()) {
                if (d.handle() == p.handle()) {
                    resident = d;
                    break;
                }
            }
            if (resident == null || resident.scale() <= 0.0f) {
                return false;
            }
        }
        return true;
    }

    /**
     * The slot the bare key runs in: 0 when slot 0 runs no adapter, else the permanent slot past the table
     * ({@link #slotCount()} + 1), which holds no delta.
     */
    public int bareSlot() {
        return defaultKey().isBare() ? 0 : slots.length;
    }

    /** {@link #load(String, String, float)} at scale 1.0. */
    public LoraHandle load(String name, String path) throws IOException {
        return load(name, path, 1.0f);
    }

    /**
     * Removes the resident adapter {@code name}: the event detaches its deltas and the handle moves to the host
     * cache. A cached entry outside {@link LoraConfig#cacheEntries()} is dropped first; with a cache of 0 entries
     * the handle is dropped.
     *
     * @throws IllegalArgumentException when {@code name} is resident nowhere
     * @throws IllegalStateException    when a running request holds the adapter in a slot
     */
    public void unload(String name) {
        LoraHandle resident = residents.get(name);
        if (resident == null) {
            throw new IllegalArgumentException("LoRA adapter " + name + " is not resident (residents: "
                    + residents.keySet() + ", cached: " + cache.keySet() + ")");
        }
        if (resident.startup()) {
            throw new IllegalStateException("LoRA adapter " + name + " was loaded at startup (--lora) and applies"
                    + " to every request that selects no adapter; it stays loaded for the model's lifetime");
        }
        int busy = requestsOf(resident);
        if (busy > 0) {
            throw new IllegalStateException("LoRA adapter " + name + " serves " + busy + " running request(s);"
                    + " unload it once they finish");
        }
        closeSlotsOf(resident);
        LoraHandle handle = residents.remove(name);
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.unloaded(handle);
        }
        cache(name, handle);
    }

    /**
     * Changes the scale of the resident adapter {@code name}: a held one rescales its {@code scale * B} buffers in
     * place between steps, on the engine thread, and creates a new epoch ({@link LoraHandle#revision()} rises,
     * also when the value returns to an earlier one); a per-request one sets the default of its later
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
        if (handle.startup()) {
            throw new IllegalStateException("LoRA adapter " + name + " was loaded at startup (--lora) and runs at"
                    + " its startup scale " + handle.scale() + " in the requests that select no adapter; a request"
                    + " selects it at another scale");
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

    /**
     * Evicts the load-order-first resident neither held by an attachment nor serving a running request: null when
     * every resident is one of those.
     */
    private LoraHandle evictOldestUnheld() {
        for (Map.Entry<String, LoraHandle> e : residents.entrySet()) {
            if (e.getValue().holds() == 0 && requestsOf(e.getValue()) == 0) {
                LoraHandle victim = e.getValue();
                closeSlotsOf(victim);
                evict(e.getKey(), victim);
                return victim;
            }
        }
        return null;
    }

    /**
     * Moves {@code handle} into the host cache, dropping the oldest entries past {@link LoraConfig#cacheEntries()};
     * a cache of 0 entries keeps nothing.
     */
    private void cache(String name, LoraHandle handle) {
        cache.remove(name);
        if (config.cacheEntries() == 0) {
            return;
        }
        while (cache.size() >= config.cacheEntries()) {
            evictOldestCached();
        }
        cache.put(name, handle);
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
        cache(name, handle);
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.unloaded(handle);
        }
    }

    // ------------------------------------------------------------------ the per-request slot table

    /**
     * Slots of the per-request table ({@link LoraConfig#maxAdapters()}); the slots are numbered from 1, and the
     * {@link #bareSlot} follows them.
     */
    public int slotCount() {
        return slots.length - 1;
    }

    /**
     * The slot key of {@code selection}, the complete set of adapters its rows run (llama.cpp's per-request
     * {@code lora} list, where an unlisted adapter runs at 0): each named adapter must be resident; an entry without
     * a scale takes the adapter's default; entries at scale 0 are left out. Null when the key equals the
     * {@link #defaultKey} — the request runs in slot 0. Every listener checks the key
     * ({@link LoraListener#checkSlot}) unless it is a {@link #isRescale} of the startup set, whose slot uploads
     * nothing to check.
     *
     * @throws IllegalArgumentException naming an adapter that is not resident, one a following attachment applies to
     *                                  every row ({@code LoraAttachment.follow}), an activated LoRA selected with
     *                                  another adapter, or a key a listener cannot open a slot for
     */
    public LoraSlotKey resolve(LoraSelection selection) {
        Objects.requireNonNull(selection, "selection");
        List<LoraSlotKey.Part> parts = new ArrayList<>();
        for (LoraSelection.Entry e : selection.entries()) {
            LoraHandle handle = residents.get(e.name());
            if (handle == null) {
                throw new IllegalArgumentException("LoRA adapter " + e.name() + " is not loaded (loaded: "
                        + residents.keySet() + ")");
            }
            if (handle.holds() > 0 && !handle.startup()) {
                throw new IllegalArgumentException("LoRA adapter " + e.name() + " is attached to every row of the"
                        + " model; a request selects the startup and per-request adapters");
            }
            float scale = e.defaultScale() ? handle.scale() : e.scale();
            if (scale != 0.0f) {
                parts.add(new LoraSlotKey.Part(handle, scale));
            }
        }
        LoraSlotKey key = new LoraSlotKey(parts);
        LoraHandle activated = key.activated();
        if (activated != null && parts.size() > 1) {
            throw new IllegalArgumentException("lora: " + activated.name() + " is an activated LoRA (aLoRA), which"
                    + " a request runs alone (selected with " + key.label() + ")");
        }
        if (key.equals(defaultKey())) {
            return null;
        }
        if (!isRescale(key)) {
            for (LoraListener l : new ArrayList<>(listeners)) {
                l.checkSlot(key);
            }
        }
        return key;
    }

    /**
     * The key of the rows an activated key {@code key} runs before its invocation sequence (and of all its rows
     * when the prompt holds none, as llama.cpp disables the aLoRA then): the key without its activated adapter —
     * the bare key, since an activated LoRA runs alone — or null when that equals the {@link #defaultKey}.
     *
     * @throws IllegalArgumentException when {@code key} holds no activated LoRA
     */
    public LoraSlotKey preInvocation(LoraSlotKey key) {
        Objects.requireNonNull(key, "key");
        if (key.activated() == null) {
            throw new IllegalArgumentException(key + " holds no activated LoRA");
        }
        return defaultKey().isBare() ? null : LoraSlotKey.bare();
    }

    /**
     * True when {@link #acquire} of {@code key} would give a slot now: one is open for the key, free, or idle.
     */
    public boolean canAcquire(LoraSlotKey key) {
        Objects.requireNonNull(key, "key");
        return key.isBare() || find(key) > 0 || freeSlot() > 0 || idleSlot() > 0;
    }

    /**
     * A slot running {@code key} for one request, held until {@link #release}: the open slot of the key, else a
     * free slot or the least recently used idle one (displaced, {@link LoraListener#slotClosed}) opened for it
     * ({@link LoraListener#slotOpened}). -1 when every slot serves running requests of other keys; the caller
     * waits for a release.
     *
     * @throws IllegalStateException when an adapter of the key left the residents since it was resolved, or a
     *                               listener cannot serve the slot (the slot stays free)
     */
    public int acquire(LoraSlotKey key) {
        Objects.requireNonNull(key, "key");
        if (key.isBare()) {
            return bareSlot();
        }
        for (LoraSlotKey.Part p : key.parts()) {
            if (residents.get(p.handle().name()) != p.handle()) {
                throw new IllegalStateException("LoRA adapter " + p.handle().name() + " was unloaded after the"
                        + " request selected it");
            }
        }
        int slot = find(key);
        if (slot < 0) {
            slot = freeSlot();
            if (slot < 0) {
                slot = idleSlot();
                if (slot < 0) {
                    return -1;
                }
                close(slot);
            }
            open(slot, key);
        }
        slots[slot].refs++;
        slots[slot].lastUse = ++clock;
        return slot;
    }

    /**
     * Returns one request's hold of {@code slot}; the slot stays open (idle once no request holds it).
     *
     * @throws IllegalArgumentException when no request holds the slot
     */
    public void release(int slot) {
        if (slot == slots.length) {
            return;
        }
        OpenSlot open = slot > 0 && slot < slots.length ? slots[slot] : null;
        if (open == null || open.refs == 0) {
            throw new IllegalArgumentException("LoRA slot " + slot + " is held by no request");
        }
        open.refs--;
        open.lastUse = ++clock;
    }

    /** The key of open slot {@code slot} (the bare key for {@link #bareSlot}), or null when it is free. */
    public LoraSlotKey slotKey(int slot) {
        if (slot == slots.length && bareSlot() == slot) {
            return LoraSlotKey.bare();
        }
        OpenSlot open = slot > 0 && slot < slots.length ? slots[slot] : null;
        return open == null ? null : open.key;
    }

    /** The running requests holding {@code slot}, 0 for a free or idle slot. */
    public int slotRequests(int slot) {
        OpenSlot open = slot > 0 && slot < slots.length ? slots[slot] : null;
        return open == null ? 0 : open.refs;
    }

    /** The running requests holding a slot that runs {@code handle}. */
    public int requestsOf(LoraHandle handle) {
        int refs = 0;
        for (int i = 1; i < slots.length; i++) {
            if (slots[i] != null && slots[i].key.contains(handle)) {
                refs += slots[i].refs;
            }
        }
        return refs;
    }

    /**
     * Runs the rows of the following passes whose batch sets no per-row slots in {@code slot}: 0 (the startup
     * adapters), the {@link #bareSlot} or a slot a request holds ({@link #acquire}). A single-sequence run or a
     * pooling call sets its request's slot for its passes and 0 after them.
     *
     * @throws IllegalArgumentException when {@code slot} is neither 0, the bare slot nor an open slot
     */
    public void useSlot(int slot) {
        if (slot != 0 && slot != bareSlot() && slotKey(slot) == null) {
            throw new IllegalArgumentException("LoRA slot " + slot + " is not open");
        }
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.sessionSlot(slot);
        }
    }

    private int find(LoraSlotKey key) {
        for (int i = 1; i < slots.length; i++) {
            if (slots[i] != null && slots[i].key.equals(key)) {
                return i;
            }
        }
        return -1;
    }

    private int freeSlot() {
        for (int i = 1; i < slots.length; i++) {
            if (slots[i] == null) {
                return i;
            }
        }
        return -1;
    }

    /** The least recently used open slot no request holds, or -1. */
    private int idleSlot() {
        int best = -1;
        for (int i = 1; i < slots.length; i++) {
            if (slots[i] != null && slots[i].refs == 0 && (best < 0 || slots[i].lastUse < slots[best].lastUse)) {
                best = i;
            }
        }
        return best;
    }

    /** Opens {@code slot} for {@code key}; a failing listener leaves it free, the ones before it closed again. */
    private void open(int slot, LoraSlotKey key) {
        boolean rescale = isRescale(key);
        List<LoraListener> served = new ArrayList<>();
        try {
            for (LoraListener l : new ArrayList<>(listeners)) {
                if (rescale) {
                    l.slotRescaled(slot, key);
                } else {
                    l.slotOpened(slot, key);
                }
                served.add(l);
            }
        } catch (RuntimeException e) {
            for (LoraListener l : served) {
                try {
                    l.slotClosed(slot, key);
                } catch (RuntimeException suppressed) {
                    e.addSuppressed(suppressed);
                }
            }
            throw e;
        }
        slots[slot] = new OpenSlot(key);
    }

    /** Closes the idle slot {@code slot}. */
    private void close(int slot) {
        OpenSlot open = slots[slot];
        slots[slot] = null;
        for (LoraListener l : new ArrayList<>(listeners)) {
            l.slotClosed(slot, open.key);
        }
    }

    /** Closes every idle slot that runs {@code handle}. */
    private void closeSlotsOf(LoraHandle handle) {
        for (int i = 1; i < slots.length; i++) {
            if (slots[i] != null && slots[i].key.contains(handle)) {
                close(i);
            }
        }
    }

    @Override
    public String toString() {
        return "LoraRegistry{" + residents.values() + ", cached " + cache.values() + "}";
    }
}

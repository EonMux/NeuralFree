package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.cli.DeviceListing;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * The multi-model router (llama.cpp's router server, its {@code server_models} and {@code server_models-sched}):
 * the registry of servable models, each opened into its own {@link EngineService} on demand, and the load
 * scheduling over them. It is on when {@code neuralfree.server.models.dir} (a directory of GGUF files scanned at
 * startup) or {@code neuralfree.server.models.preset} (a {@link ModelPreset} INI file) is set; the properties then
 * stand beside {@code neuralfree.server.models.max}, how many models may be open at once (llama.cpp's
 * {@code --models-max}, 4, 0 for no limit), {@code neuralfree.server.models.memory.margin}, how many MiB of device
 * memory stay free, unloading the least recently used models until a load fits (llama.cpp's
 * {@code --models-memory-margin}), and {@code neuralfree.server.models.autoload}, whether a request naming an
 * unloaded model loads it (llama.cpp's {@code --models-autoload}, on by default). The single-model property
 * {@code neuralfree.server.model} refuses in router mode, as llama.cpp's router has no model of its own.
 *
 * <p>The lifecycle of one entry runs unloaded → loading → loaded → unloaded; a failed load returns it to unloaded
 * with the failure recorded, and a loaded engine that sleeps reports {@code sleeping}. A routed request touches
 * the entry's use time; {@link #makeRoom} evicts idle entries least recently used first, never one with a request
 * in flight — one still serving keeps its place (llama.cpp: "a model that is serving requests is never unloaded").
 *
 * <p>Every state change publishes an event ({@link Event}) to the subscribed queues the {@code /models/sse}
 * stream drains. Load, unload, registration and the directory rescan ({@link #reload}) serialize on the router;
 * the routing hot path ({@link #target}) reads without a lock.
 */
public final class ModelRouter implements AutoCloseable {

    /** The directory the servable models are scanned from. */
    static final String DIR_PROPERTY = "neuralfree.server.models.dir";

    /** The INI file of the model presets ({@link ModelPreset}). */
    static final String PRESET_PROPERTY = ModelPreset.PROPERTY;

    /** How many models may be open at once; 0 for no limit. */
    static final String MAX_PROPERTY = "neuralfree.server.models.max";

    /** The MiB of device memory to keep free ({@code --models-memory-margin}). */
    static final String MARGIN_PROPERTY = "neuralfree.server.models.memory.margin";

    /** Whether a request naming an unloaded model loads it. */
    static final String AUTOLOAD_PROPERTY = "neuralfree.server.models.autoload";

    /** llama.cpp's default of {@code --models-max}. */
    static final int DEFAULT_MAX = 4;

    /** How the split files of one model are named ({@code <name>-00001-of-00003.gguf}); the first shard serves. */
    private static final Pattern SPLIT = Pattern.compile("^(?<name>.+)-\\d{5}-of-\\d{5}\\.gguf$", Pattern.CASE_INSENSITIVE);

    /** Where an entry came from. */
    public enum Source {
        /** A file of {@code neuralfree.server.models.dir}. */
        MODELS_DIR("models_dir"),
        /** A section of {@code neuralfree.server.models.preset}. */
        PRESET("preset"),
        /** Registered at runtime through {@code POST /models}. */
        REGISTERED("registered");

        private final String label;

        Source(String label) {
            this.label = label;
        }

        /** The name {@code GET /models} reports. */
        public String label() {
            return label;
        }
    }

    /** The lifecycle state of an entry ({@code GET /models}' {@code status.value}). */
    public enum Status {
        UNLOADED, LOADING, LOADED
    }

    /** A model entry: its source and files, its overrides and its runtime state. */
    public static final class Entry {
        /** The model id requests route by: the file's name, the preset's section, or the registered name. */
        private final String name;
        private final Source source;
        /** The additional names that resolve to this model. */
        private final List<String> aliases;
        /** The informational tags. */
        private final List<String> tags;
        /** Whether the model loads when the server starts. */
        private final boolean loadOnStartup;
        /** The preset overrides; null is the server's value. */
        private final Integer nCtx;
        private final Integer parallel;
        private final Integer ngl;
        private final String device;
        private volatile Path path;
        private volatile Status status = Status.UNLOADED;
        /** The failure of the last load; null when it succeeded or was never tried. */
        private volatile String error;
        private volatile EngineService service;
        /** The {@link System#nanoTime} of the last routed request or load (the LRU order). */
        private volatile long lastUsedNanos;

        Entry(String name, Path path, Source source, List<String> aliases, List<String> tags, boolean loadOnStartup,
              Integer nCtx, Integer parallel, Integer ngl, String device) {
            this.name = name;
            this.path = path;
            this.source = source;
            this.aliases = List.copyOf(aliases);
            this.tags = List.copyOf(tags);
            this.loadOnStartup = loadOnStartup;
            this.nCtx = nCtx;
            this.parallel = parallel;
            this.ngl = ngl;
            this.device = device;
        }

        /** The model id. */
        public String name() {
            return name;
        }

        /** The GGUF file that serves the model. */
        public Path path() {
            return path;
        }

        /** Where the entry came from. */
        public Source source() {
            return source;
        }

        /** The additional names that resolve to this model. */
        public List<String> aliases() {
            return aliases;
        }

        /** The informational tags. */
        public List<String> tags() {
            return tags;
        }

        /** Whether the model loads when the server starts. */
        public boolean loadOnStartup() {
            return loadOnStartup;
        }

        /** The entry's open service, or null while unloaded. */
        EngineService service() {
            return service;
        }

        /** The lifecycle state. */
        public Status status() {
            return status;
        }

        /** The failure of the last load, or null. */
        public String error() {
            return error;
        }

        /** Whether the entry holds an engine an unload may close (loading, loaded or sleeping). */
        boolean running() {
            return status != Status.UNLOADED;
        }

        /** Whether the entry holds an open engine a request may serve on. */
        boolean ready() {
            return status == Status.LOADED;
        }

        /** Whether the engine holds released memory or frozen scheduling (a sleep). */
        boolean sleeping() {
            EngineService open = service;
            return open != null && open.isSleeping();
        }

        /** The engine context the entry serves with, or 0 while unloaded or sleeping. */
        int contextLength() {
            EngineService open = service;
            if (open == null || !open.loaded()) {
                return 0;
            }
            try {
                return open.contextLength();
            } catch (RuntimeException e) {
                return 0;
            }
        }

        /** The requests in flight over the entry's replicas. */
        int inFlight() {
            EngineService open = service;
            return open == null ? 0 : open.load();
        }

        /** Stamps the LRU order. */
        void touch() {
            lastUsedNanos = System.nanoTime();
        }
    }

    /** A state change of one entry, as the {@code /models/sse} stream carries it. */
    public record Event(String model, String type, Map<String, Object> data) {

        /** The event's wire form: {@code {"model", "event", "data"}}. */
        public Map<String, Object> wire() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("model", model);
            out.put("event", type);
            out.put("data", data);
            return out;
        }
    }

    /**
     * The device-memory figures the margin unloads against: the devices' total bytes, and what a loaded service
     * holds. The figures of a build without GPU devices are 0, and the margin then has no effect, as in llama.cpp.
     */
    public interface DeviceMemory {
        /** Every enumerable device's total memory in bytes. */
        long totalBytes();

        /** The device memory {@code service} holds. */
        long heldBytes(EngineService service);
    }

    /** A model name no entry carries: the 404 the request is answered with, as vLLM's unknown-model refusals are. */
    public static final class UnknownModelException extends WebApplicationException {
        private static final long serialVersionUID = 1L;

        UnknownModelException(String name) {
            super(ServerJson.error(404, "The model `" + name + "` does not exist.", "NotFoundError").getResponse());
        }
    }

    private static final Logger LOGGER = Logger.getLogger(ModelRouter.class.getName());

    private final boolean active;
    private final int modelsMax;
    private final long marginBytes;
    private final boolean autoload;
    private final EngineConfig baseConfig;
    private final ServerOptions baseOptions;
    private final MediaOptions mediaOptions;
    private final ServerLoras loras;
    private final DeviceMemory memory;
    private final Path dir;
    private final List<ModelPreset> presets;
    /** The registry, in insertion order; guarded by this router's monitor. */
    private final Map<String, Entry> entries = new LinkedHashMap<>();
    /** The subscribed event queues ({@code /models/sse}); guarded by this router's monitor. */
    private final List<LinkedBlockingQueue<Event>> listeners = new ArrayList<>();

    /**
     * The router over the directory scan at {@code dir} (null for none) and the presets of {@code presets}, under
     * the serving settings of {@code baseConfig} and {@code options} and the device figures of {@code memory}.
     *
     * @throws IllegalArgumentException when two sources name the same model id or alias
     */
    public ModelRouter(Path dir, List<ModelPreset> presets, int modelsMax, long marginBytes, boolean autoload,
                       EngineConfig baseConfig, ServerOptions baseOptions, DeviceMemory memory)
            throws IOException {
        this(dir, presets, modelsMax, marginBytes, autoload, baseConfig, baseOptions, memory, MediaOptions.none(),
                ServerLoras.NONE);
    }

    /**
     * {@link #ModelRouter(Path, List, int, long, boolean, EngineConfig, ServerOptions, DeviceMemory)} with the
     * media inputs and startup LoRA adapters every opened model serves beside.
     */
    public ModelRouter(Path dir, List<ModelPreset> presets, int modelsMax, long marginBytes, boolean autoload,
                       EngineConfig baseConfig, ServerOptions baseOptions, DeviceMemory memory,
                       MediaOptions mediaOptions, ServerLoras loras) throws IOException {
        this.active = dir != null || presets != null;
        this.dir = dir;
        this.presets = presets == null ? List.of() : List.copyOf(presets);
        this.modelsMax = modelsMax;
        this.marginBytes = marginBytes;
        this.autoload = autoload;
        this.baseConfig = Objects.requireNonNull(baseConfig, "baseConfig");
        this.baseOptions = Objects.requireNonNull(baseOptions, "baseOptions");
        this.memory = Objects.requireNonNull(memory, "memory");
        this.mediaOptions = Objects.requireNonNull(mediaOptions, "mediaOptions");
        this.loras = Objects.requireNonNull(loras, "loras");
        if (!active) {
            return;
        }
        if (dir != null) {
            scan(dir);
        }
        for (ModelPreset preset : this.presets) {
            put(new Entry(preset.name(), preset.path(), Source.PRESET, preset.aliases(), preset.tags(),
                    preset.loadOnStartup(), preset.nCtx(), preset.parallel(), preset.ngl(), preset.device()));
        }
    }

    /**
     * The router of the {@code neuralfree.server.models.*} properties; an inactive one when no property names a
     * source (the single-model mode). The single-model properties refuse in router mode.
     *
     * @throws IllegalArgumentException when a property holds a bad value, a source names a model twice, or a
     *                                 single-model property is set beside a router source
     */
    static ModelRouter fromSystemProperties(EngineConfig baseConfig, ServerOptions options) throws IOException {
        String dir = System.getProperty(DIR_PROPERTY);
        String preset = System.getProperty(PRESET_PROPERTY);
        if ((dir == null || dir.isBlank()) && (preset == null || preset.isBlank())) {
            return new ModelRouter(null, null, DEFAULT_MAX, 0, true, baseConfig, options, NO_DEVICES);
        }
        if (System.getProperty("neuralfree.server.model") != null
                || System.getProperty(ServedModels.PROPERTY) != null) {
            throw new IllegalArgumentException("neuralfree.server.model is not available in router mode: unset it"
                    + " (and " + ServedModels.PROPERTY + "), the router serves the models of " + DIR_PROPERTY
                    + " and " + PRESET_PROPERTY);
        }
        Path dirPath = dir == null || dir.isBlank() ? null : Path.of(dir.trim());
        if (dirPath != null && !Files.isDirectory(dirPath)) {
            throw new IllegalArgumentException(DIR_PROPERTY + " must name a directory, got '" + dir.trim() + "'");
        }
        Path presetPath = preset == null || preset.isBlank() ? null : Path.of(preset.trim());
        List<ModelPreset> parsed = presetPath == null ? List.of() : ModelPreset.parse(presetPath);
        long max = wholeNumber(MAX_PROPERTY, DEFAULT_MAX);
        if (max > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(MAX_PROPERTY + " is too large: " + max);
        }
        return new ModelRouter(dirPath, parsed, (int) max,
                wholeNumber(MARGIN_PROPERTY, 0) * 1024L * 1024L,
                Boolean.parseBoolean(System.getProperty(AUTOLOAD_PROPERTY, "true").trim()), baseConfig, options,
                systemDeviceMemory(), MediaOptions.fromSystemProperties(), ServerLoras.fromSystemProperties());
    }

    /** The device figures of a build whose device enumeration answers nothing: the margin stays off. */
    private static final DeviceMemory NO_DEVICES = new DeviceMemory() {
        @Override
        public long totalBytes() {
            return 0;
        }

        @Override
        public long heldBytes(EngineService service) {
            return 0;
        }
    };

    /** The device figures of the engine's device enumeration, queried on first use; the margin unloads against it. */
    static DeviceMemory systemDeviceMemory() {
        return new DeviceMemory() {
            private volatile Long total;

            @Override
            public long totalBytes() {
                Long cached = total;
                if (cached == null) {
                    long sum = 0;
                    for (DeviceListing.Entry device : DeviceListing.available()) {
                        if (device.totalBytes() > 0) {
                            sum += device.totalBytes();
                        }
                    }
                    total = cached = sum;
                }
                return cached;
            }

            @Override
            public long heldBytes(EngineService service) {
                return service.allocatedDeviceBytes();
            }
        };
    }

    private static long wholeNumber(String property, long fallback) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            long parsed = Long.parseLong(raw.trim());
            if (parsed < 0) {
                throw new IllegalArgumentException(property + " must be 0 or more, got " + parsed);
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + " must be a whole number, got '" + raw + "'");
        }
    }

    // ------------------------------------------------------------------ registry

    /** The models of a directory: the model name → its first-shard file, one entry per model, in file order. */
    private Map<String, Path> scanDir(Path dir) throws IOException {
        Map<String, Path> out = new LinkedHashMap<>();
        try (Stream<Path> files = Files.list(dir)) {
            List<Path> sorted = files.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf"))
                    .sorted()
                    .toList();
            for (Path file : sorted) {
                String fileName = file.getFileName().toString();
                Matcher split = SPLIT.matcher(fileName);
                if (split.matches()) {
                    if (!fileName.matches(".*-00001-of-\\d{5}\\.gguf")) {
                        continue; // the shards after the first serve through their first shard
                    }
                    out.put(split.group("name"), file);
                    continue;
                }
                out.put(fileName.substring(0, fileName.length() - 5), file);
            }
        }
        return out;
    }

    /** The entries of the models directory, in file order. */
    private void scan(Path dir) throws IOException {
        for (Map.Entry<String, Path> model : scanDir(dir).entrySet()) {
            put(new Entry(model.getKey(), model.getValue(), Source.MODELS_DIR, List.of(), List.of(), false,
                    null, null, null, null));
        }
    }

    /** Adds {@code entry}, refusing an id or alias another entry already answers to. */
    private void put(Entry entry) {
        requireFreeNames(entry.name(), entry.aliases());
        entries.put(entry.name(), entry);
    }

    /** Refuses, naming the clash, an id or alias another entry answers to. */
    private void requireFreeNames(String name, List<String> aliases) {
        if (entries.containsKey(name)) {
            throw new IllegalArgumentException("the model name '" + name + "' is already served by '"
                    + entries.get(name).path() + "'");
        }
        for (String alias : aliases) {
            Entry taken = resolveName(alias);
            if (taken != null) {
                throw new IllegalArgumentException("the alias '" + alias + "' is already answered by model '"
                        + taken.name() + "'");
            }
        }
    }

    /** The entry the name or alias resolves to, or null. */
    private Entry resolveName(String name) {
        Entry direct = entries.get(name);
        if (direct != null) {
            return direct;
        }
        for (Entry entry : entries.values()) {
            if (entry.aliases.contains(name)) {
                return entry;
            }
        }
        return null;
    }

    /** The entry the name or alias resolves to, refusing an unknown one ({@code GET /models}' 404). */
    private Entry require(String name) {
        Entry entry = resolveName(name);
        if (entry == null) {
            throw new UnknownModelException(name);
        }
        return entry;
    }

    /** Whether the router mode is on. */
    public boolean active() {
        return active;
    }

    /** How many models may be open at once. */
    public int modelsMax() {
        return modelsMax;
    }

    /** Whether a request naming an unloaded model loads it. */
    public boolean autoload() {
        return autoload;
    }

    /** The MiB of device memory kept free; 0 disables the margin. */
    public long memoryMarginBytes() {
        return marginBytes;
    }

    /** The entries' snapshots, in registry order ({@code GET /models}' list). */
    public synchronized List<Entry> list() {
        return new ArrayList<>(entries.values());
    }

    /** The requests in flight over every open entry (the router's {@code server_load}). */
    public synchronized int load() {
        int sum = 0;
        for (Entry entry : entries.values()) {
            sum += entry.inFlight();
        }
        return sum;
    }

    /** Aborts the live generations of every open entry, answering how many stopped. */
    public int abortAll() {
        List<EngineService> open = new ArrayList<>();
        synchronized (this) {
            for (Entry entry : entries.values()) {
                EngineService service = entry.service();
                if (entry.ready() && service != null) {
                    open.add(service);
                }
            }
        }
        int aborted = 0;
        for (EngineService service : open) {
            aborted += service.abortAll();
        }
        return aborted;
    }

    // ------------------------------------------------------------------ routing

    /**
     * The service a request serves on: the router's target for the body's {@code model} in router mode, else
     * {@code fallback} (also when {@code router} is null — the manual wiring serves the one model). A routed
     * request touches its entry's LRU order, and an unloaded entry loads when the autoload is on — the server's,
     * or the request's own {@code autoload} query value.
     *
     * @throws UnknownModelException     for a model no entry carries (404)
     * @throws WebApplicationException   400 for a missing model name, an autoload-off request naming an unloaded
     *                                   model, or the {@code autoload} query value not being true or false
     * @throws EngineUnavailableException when the load fails or no slot frees for it (503)
     */
    public static EngineService target(ModelRouter router, EngineService fallback, JsonView body,
                                       String autoloadParam) {
        if (router == null || !router.active()) {
            return fallback;
        }
        Object raw = body == null ? null : body.raw("model");
        if (!(raw instanceof String requested) || requested.isBlank()) {
            throw ServerJson.badRequest("model name is missing from the request: router mode serves the models"
                    + " of " + DIR_PROPERTY + " and " + PRESET_PROPERTY + " by name");
        }
        boolean requestAutoload = router.autoload(autoloadParam);
        Entry entry = router.require(requested.trim());
        entry.touch();
        if (entry.ready()) {
            return entry.service;
        }
        if (!requestAutoload) {
            throw ServerJson.error(400, "model '" + entry.name() + "' is not loaded: POST /models/load loads it,"
                    + " or start the server with " + AUTOLOAD_PROPERTY + "=true", "invalid_request_error");
        }
        return router.load(entry);
    }

    /** The request's own autoload choice, or the server's. */
    private boolean autoload(String autoloadParam) {
        if (autoloadParam == null || autoloadParam.isBlank()) {
            return autoload;
        }
        if (autoloadParam.equalsIgnoreCase("true") || autoloadParam.equals("1")) {
            return true;
        }
        if (autoloadParam.equalsIgnoreCase("false") || autoloadParam.equals("0")) {
            return false;
        }
        throw ServerJson.badRequest("autoload must be true or false, got '" + autoloadParam + "'");
    }

    /**
     * {@link #target} for a GET route: llama.cpp's router names the model with the {@code model} query parameter
     * (URL-encoded), the autoload override with {@code autoload}.
     */
    public static EngineService targetQuery(ModelRouter router, EngineService fallback, String model,
                                            String autoloadParam) {
        if (router == null || !router.active()) {
            return fallback;
        }
        Map<String, Object> named = new LinkedHashMap<>();
        named.put("model", model == null ? "" : model);
        return target(router, fallback, JsonView.of(named, "query"), autoloadParam);
    }

    /**
     * Loads the entry named {@code name} ({@code POST /models/load}): refuses an unknown one, and one already
     * running; makes room against the count limit and the memory margin first.
     *
     * @throws UnknownModelException     for a model no entry carries (404)
     * @throws IllegalArgumentException  when the model already runs (400)
     * @throws EngineUnavailableException when the load fails, or no slot frees within the request budget (503)
     */
    public EngineService load(String name) {
        Entry entry = require(name);
        if (entry.running()) {
            throw new IllegalArgumentException("model '" + entry.name() + "' is already "
                    + (entry.sleeping() ? "sleeping" : "running"));
        }
        return load(entry);
    }

    /** The opened service of {@code entry}, opening it after the evictions made room. */
    private synchronized EngineService load(Entry entry) {
        if (entry.ready()) {
            return entry.service; // another thread opened it while this caller waited on the monitor
        }
        if (entry.running()) {
            throw new IllegalArgumentException("model '" + entry.name() + "' is already running");
        }
        makeRoom();
        open(entry);
        return entry.service;
    }

    /**
     * Opens {@code entry}'s engine: the state runs loading → loaded, or back to unloaded with the failure recorded.
     *
     * @throws EngineUnavailableException when the model does not open, the message naming the failure
     */
    private void open(Entry entry) {
        entry.touch();
        entry.status = Status.LOADING;
        publish(entry, "loading", Map.of());
        try {
            entry.service = new EngineService(entry.path().toString(), config(entry),
                    baseOptions.withServedNames(servedNames(entry)), mediaOptions, loras);
            entry.status = Status.LOADED;
            entry.error = null;
            publish(entry, "loaded", Map.of());
            runOpenHooks(entry.service);
            LOGGER.info("router loaded model '" + entry.name() + "' from " + entry.path());
        } catch (IOException | RuntimeException e) {
            entry.service = null;
            entry.status = Status.UNLOADED;
            entry.error = String.valueOf(e.getMessage());
            publish(entry, "unloaded", Map.of("failed", true, "error", entry.error));
            throw new EngineUnavailableException("model '" + entry.name() + "' does not open: "
                    + entry.error);
        }
    }

    /** The names the entry's responses carry: the id first, the aliases after. */
    private static String servedNames(Entry entry) {
        if (entry.aliases.isEmpty()) {
            return entry.name();
        }
        return entry.name() + "," + String.join(",", entry.aliases);
    }

    /** The engine configuration the entry serves with: the server's, with the preset's overrides on top. */
    private EngineConfig config(Entry entry) {
        EngineConfig config = baseConfig;
        if (entry.device != null) {
            config = config.toBuilder().deviceFilter(entry.device).build();
        }
        if (entry.ngl != null) {
            config = config.toBuilder().gpuLayers(entry.ngl).build();
        }
        if (entry.nCtx != null || entry.parallel != null) {
            MemoryConfig.Builder memory = config.memory().toBuilder();
            if (entry.nCtx != null) {
                memory.nCtx(entry.nCtx);
            }
            if (entry.parallel != null) {
                memory.parallel(entry.parallel);
            }
            config = config.toBuilder().memory(memory.build()).build();
        }
        return config;
    }

    /**
     * Unloads the entry named {@code name}, closing its engine ({@code POST /models/unload}): refuses an unknown
     * one, and one that holds no engine.
     *
     * @throws UnknownModelException    for a model no entry carries (404)
     * @throws IllegalArgumentException when the model holds no engine (400)
     */
    public synchronized void unload(String name) {
        Entry entry = require(name);
        if (!entry.running()) {
            throw new IllegalArgumentException("model '" + entry.name() + "' is not running");
        }
        close(entry);
    }

    /** Closes {@code entry}'s engine and marks it unloaded (the state events carry the change). */
    private void close(Entry entry) {
        entry.status = Status.UNLOADED;
        entry.error = null;
        EngineService open = entry.service;
        entry.service = null;
        publish(entry, "unloaded", Map.of());
        if (open != null) {
            open.close();
            LOGGER.info("router unloaded model '" + entry.name() + "'");
        }
    }

    /**
     * Unloads the idle entries the count limit and the memory margin ask for: least recently used first, never an
     * entry with a request in flight. A count slot a serving pool of entries keeps waits for it, within the
     * request budget.
     *
     * @throws EngineUnavailableException when no slot frees within the request budget
     */
    private void makeRoom() {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(
                Math.max(1, baseOptions.requestTimeout().toMillis()));
        while (modelsMax > 0 && loadedCount() >= modelsMax) {
            Entry victim = lruIdle();
            if (victim == null) {
                if (System.nanoTime() > deadline) {
                    throw new EngineUnavailableException("every one of the " + modelsMax
                            + " model slots is serving requests: the models to unload keep serving, so the load"
                            + " waits past the " + baseOptions.budgetText() + " request budget");
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new EngineUnavailableException("the load was interrupted while waiting for a free"
                            + " model slot");
                }
                continue;
            }
            close(victim);
        }
        while (memory.totalBytes() > 0 && freeBytes() < marginBytes) {
            Entry victim = lruIdle();
            if (victim == null) {
                return; // nothing idle to unload: the load goes ahead, as llama.cpp's does when nothing fits
            }
            close(victim);
        }
    }

    /** The idle entries' held device bytes against the devices' total. */
    private long freeBytes() {
        long held = 0;
        for (Entry entry : entries.values()) {
            if (entry.ready()) {
                held += memory.heldBytes(entry.service);
            }
        }
        return memory.totalBytes() - held;
    }

    /** The loaded entries' count. */
    private int loadedCount() {
        int count = 0;
        for (Entry entry : entries.values()) {
            if (entry.ready()) {
                count++;
            }
        }
        return count;
    }

    /** The least recently used loaded entry with no request in flight, or null. */
    private Entry lruIdle() {
        Entry lru = null;
        for (Entry entry : entries.values()) {
            if (!entry.ready() || entry.inFlight() > 0) {
                continue;
            }
            if (lru == null || entry.lastUsedNanos < lru.lastUsedNanos) {
                lru = entry;
            }
        }
        return lru;
    }

    // ------------------------------------------------------------------ management (POST and DELETE /models)

    /**
     * Registers the GGUF file at {@code path} under its file name ({@code POST /models}): the model then lists
     * and routes like a scanned or preset one, and {@link #unregister} may remove it. The file must be a readable
     * {@code .gguf} file, and the name and aliases must not clash.
     *
     * @throws IllegalArgumentException when the path names no {@code .gguf} file, or a name or alias clashes
     */
    public synchronized Entry register(Path path, List<String> aliases, List<String> tags) {
        Objects.requireNonNull(path, "path");
        String fileName = path.getFileName() == null ? "" : path.getFileName().toString();
        if (!fileName.toLowerCase(Locale.ROOT).endsWith(".gguf") || !Files.isRegularFile(path)) {
            throw new IllegalArgumentException("model must name a .gguf file, got '" + path + "'");
        }
        checkDistinct(aliases, "alias");
        checkDistinct(tags, "tag");
        Matcher split = SPLIT.matcher(fileName);
        String name = split.matches() ? split.group("name")
                : fileName.substring(0, fileName.length() - 5);
        requireFreeNames(name, aliases);
        Entry entry = new Entry(name, path, Source.REGISTERED, aliases, tags, false, null, null, null, null);
        entries.put(name, entry);
        publish(entry, "unloaded", Map.of());
        return entry;
    }

    /** Refuses a list holding one entry twice, naming the kind. */
    private static void checkDistinct(List<String> values, String kind) {
        if (values == null || values.isEmpty()) {
            return;
        }
        if (new LinkedHashSet<>(values).size() != values.size()) {
            throw new IllegalArgumentException(kind + "s hold the same entry twice: " + values);
        }
    }

    /**
     * Removes the entry named {@code name} ({@code DELETE /models}), closing its engine first: only a registered
     * one may go (its files stay on disk); a scanned or preset entry is the server configuration's to keep.
     *
     * @return the removed entry
     * @throws UnknownModelException    for a model no entry carries (404)
     * @throws IllegalArgumentException when the model comes from the directory scan or a preset (400)
     */
    public synchronized Entry unregister(String name) {
        Entry entry = require(name);
        if (entry.source != Source.REGISTERED) {
            throw new IllegalArgumentException("model '" + entry.name() + "' comes from "
                    + (entry.source == Source.MODELS_DIR ? DIR_PROPERTY : PRESET_PROPERTY)
                    + " and cannot be removed: DELETE /models takes the models POST /models registered");
        }
        if (entry.running()) {
            close(entry);
        }
        entries.remove(entry.name());
        return entry;
    }

    /**
     * Re-reads the sources ({@code GET /models} with {@code reload=true}): the directory's new files and the
     * preset's new sections join, and an entry its source no longer names, or whose path changed, unloads and
     * leaves — a running model whose file went away stops serving, as llama.cpp's reload unloads it.
     */
    public synchronized void reload() throws IOException {
        Map<String, Path> dirModels = dir == null ? Map.of() : scanDir(dir);
        for (Map.Entry<String, Path> model : dirModels.entrySet()) {
            Entry entry = entries.get(model.getKey());
            if (entry == null) {
                put(new Entry(model.getKey(), model.getValue(), Source.MODELS_DIR, List.of(), List.of(), false,
                        null, null, null, null));
            } else if (!entry.path().equals(model.getValue())) {
                replace(entry, model.getValue());
            }
        }
        for (ModelPreset preset : presets) {
            Entry entry = entries.get(preset.name());
            if (entry == null) {
                put(new Entry(preset.name(), preset.path(), Source.PRESET, preset.aliases(), preset.tags(),
                        preset.loadOnStartup(), preset.nCtx(), preset.parallel(), preset.ngl(), preset.device()));
            } else if (!entry.path().equals(preset.path())) {
                replace(entry, preset.path());
            }
        }
        for (Entry entry : new ArrayList<>(entries.values())) {
            if (entry.source == Source.MODELS_DIR && !dirModels.containsKey(entry.name())) {
                if (entry.running()) {
                    close(entry);
                }
                entries.remove(entry.name());
            }
        }
    }

    /** Unloads {@code entry}, if it runs, and repoints it at {@code path} as an unloaded entry. */
    private void replace(Entry entry, Path path) {
        if (entry.running()) {
            close(entry);
        }
        entry.path = path;
        LOGGER.info("router model '" + entry.name() + "' now serves from " + path);
    }

    /** Loads every entry its sources mark {@code load-on-startup} (llama.cpp's {@code load_startup_models}). */
    public void loadStartupModels() {
        if (!active) {
            return;
        }
        for (Entry entry : list()) {
            if (!entry.loadOnStartup() || entry.running()) {
                continue;
            }
            try {
                load(entry.name());
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "the startup load of model '" + entry.name() + "' failed: "
                        + e.getMessage(), e);
            }
        }
    }

    // ------------------------------------------------------------------ events

    private final List<java.util.function.Consumer<EngineService>> openHooks = new ArrayList<>();

    /**
     * Runs {@code hook} with each service that opens from now on, and once with every open one now: an opened
     * entry binds its readers (the per-replica meters) the same way the single-model service does.
     */
    public synchronized void onOpen(java.util.function.Consumer<EngineService> hook) {
        openHooks.add(hook);
        for (Entry entry : entries.values()) {
            if (entry.service != null) {
                hook.accept(entry.service);
            }
        }
    }

    /** Runs the open hooks with the just-opened service; a failing hook is logged and leaves the entry serving. */
    private void runOpenHooks(EngineService open) {
        for (java.util.function.Consumer<EngineService> hook : openHooks) {
            try {
                hook.accept(open);
            } catch (RuntimeException e) {
                LOGGER.log(Level.WARNING, "an open hook of a router model failed", e);
            }
        }
    }

    /** Subscribes a queue the following state changes arrive on ({@code /models/sse}). */
    public synchronized LinkedBlockingQueue<Event> subscribe() {
        LinkedBlockingQueue<Event> queue = new LinkedBlockingQueue<>();
        listeners.add(queue);
        return queue;
    }

    /** Ends a subscription; the queue keeps the events it already holds. */
    public synchronized void unsubscribe(LinkedBlockingQueue<Event> queue) {
        listeners.remove(queue);
    }

    /** Publishes {@code entry}'s state change to every subscriber. */
    private void publish(Entry entry, String status, Map<String, Object> extra) {
        Event event = event(entry, status, extra);
        for (LinkedBlockingQueue<Event> listener : listeners) {
            listener.offer(event);
        }
    }

    /** The event of {@code entry}'s state, with the failure fields on an unload after a failed load. */
    private static Event event(Entry entry, String status, Map<String, Object> extra) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("status", status);
        data.putAll(extra);
        return new Event(entry.name(), "model_status", Map.copyOf(data));
    }

    // ------------------------------------------------------------------ shutdown

    /** Unloads every entry. */
    @Override
    public synchronized void close() {
        for (Entry entry : new ArrayList<>(entries.values())) {
            if (entry.running()) {
                close(entry);
            }
        }
    }
}

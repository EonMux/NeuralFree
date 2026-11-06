package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.plugin.LogitsProcessorProvider;
import com.eonmux.neuralfree.gguf.plugin.Plugins;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The custom logits processors an engine runs: the factories plugins contribute through
 * {@link LogitsProcessorProvider} plus the factory classes {@code --logits-processors}
 * ({@link EngineConfig#PROP_LOGITS_PROCESSORS}) names, merged in provider-class order and then class-name order.
 *
 * <p>Every factory is consulted for every generation ({@link #stages}); one that returns null leaves the request
 * alone, so a processor activates per request from the arguments of {@code vllm_xargs}
 * ({@link GenerationRequest#logitsProcessorArgs()}). {@link #validate} runs the factories' own checks at request
 * admission. An empty registry (no plugin factories and no configured classes) is shared, so an engine without
 * processors does no per-request work.
 *
 * <p>The table is built once per configuration and is immutable. A name or alias registered twice fails the build
 * with both sources named: providers never replace one another, as the other plugin registries hold. A configured
 * class that is missing, is not a {@link LogitsProcessorFactory}, or has no public no-argument constructor is
 * refused naming the property and the entry.
 */
public final class LogitsProcessors {

    /** The property and launch flag (vLLM's {@code --logits-processors}) naming factory classes. */
    public static final String PROP_LOGITS_PROCESSORS = EngineConfig.PROP_LOGITS_PROCESSORS;

    private static final LogitsProcessors EMPTY = new LogitsProcessors(Collections.emptyList());

    /** The registries built so far, keyed by the configured class names. */
    private static final ConcurrentHashMap<List<String>, LogitsProcessors> CACHE = new ConcurrentHashMap<>();

    private final List<LogitsProcessorFactory> factories;
    /** The name table, for rejection messages and {@link #names()}. */
    private final Map<String, LogitsProcessorFactory> table;

    private LogitsProcessors(List<LogitsProcessorFactory> factories) {
        this.factories = factories;
        Map<String, LogitsProcessorFactory> names = new LinkedHashMap<>();
        for (LogitsProcessorFactory f : factories) {
            for (String name : namesOf(f)) {
                names.put(key(name), f);
            }
        }
        this.table = Collections.unmodifiableMap(names);
    }

    /** The registry of an engine configuration over the loaded plugin providers. */
    public static LogitsProcessors forConfig(EngineConfig config) {
        Objects.requireNonNull(config, "config");
        List<String> names = config.logitsProcessors();
        List<LogitsProcessorProvider> providers = Plugins.load(LogitsProcessorProvider.class);
        if (names.isEmpty() && providers.isEmpty()) {
            return EMPTY;
        }
        List<String> key = Collections.unmodifiableList(new ArrayList<>(names));
        return CACHE.computeIfAbsent(key, ignored -> build(providers, names));
    }

    /** The registry with no processor. */
    public static LogitsProcessors empty() {
        return EMPTY;
    }

    /** A registry over {@code factories} alone (no plugin providers and no class loading). */
    public static LogitsProcessors of(List<? extends LogitsProcessorFactory> factories) {        Objects.requireNonNull(factories, "factories");
        if (factories.isEmpty()) {
            return EMPTY;
        }
        List<LogitsProcessorFactory> copy = new ArrayList<>(factories.size());
        for (LogitsProcessorFactory f : factories) {
            copy.add(Objects.requireNonNull(f, "factory"));
        }
        return new LogitsProcessors(Collections.unmodifiableList(copy));
    }

    /** Whether the engine has no processor to consult. */
    public boolean isEmpty() {
        return factories.isEmpty();
    }

    /** The factories, in their apply order (unmodifiable). */
    public List<LogitsProcessorFactory> factories() {
        if (isEmpty()) {
            return Collections.emptyList();
        }
        return Collections.unmodifiableList(factories);
    }

    /** The registered names and aliases (unmodifiable). */
    public List<String> names() {
        return Collections.unmodifiableList(new ArrayList<>(table.keySet()));
    }

    /**
     * Runs every factory's {@link LogitsProcessorFactory#validate} over the request's arguments.
     *
     * @throws IllegalArgumentException naming the processor and the argument when a factory refuses one
     */
    public void validate(GenerationRequest request) {
        Objects.requireNonNull(request, "request");
        validate(request.logitsProcessorArgs());
    }

    /**
     * Runs every factory's {@link LogitsProcessorFactory#validate} over {@code args}.
     *
     * @throws IllegalArgumentException naming the processor and the argument when a factory refuses one
     */
    public void validate(LogitsProcessorArgs args) {
        Objects.requireNonNull(args, "vllm_xargs");
        for (LogitsProcessorFactory factory : factories) {
            try {
                factory.validate(args);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("vllm_xargs: " + factory.name() + ": " + e.getMessage(), e);
            }
        }
    }

    /**
     * The stages the factories create for {@code request}, in factory order; empty when none applies. Each stage is
     * fresh state of this generation. A factory refusing an argument fails the generation the same way
     * {@link #validate(GenerationRequest)} fails admission.
     *
     * @throws IllegalArgumentException naming the processor and the argument when a factory refuses one
     */
    public List<SamplerStage> stages(GenerationRequest request) {
        Objects.requireNonNull(request, "request");
        if (factories.isEmpty()) {
            return Collections.emptyList();
        }
        LogitsProcessorArgs args = request.logitsProcessorArgs();
        validate(args);
        List<SamplerStage> stages = new ArrayList<>();
        for (LogitsProcessorFactory factory : factories) {
            SamplerStage stage;
            try {
                stage = factory.create(args);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("vllm_xargs: " + factory.name() + ": " + e.getMessage(), e);
            }
            if (stage != null) {
                stages.add(stage);
            }
        }
        return stages.isEmpty() ? Collections.emptyList() : Collections.unmodifiableList(stages);
    }

    /**
     * The table of {@code providers} plus the factories of {@code classNames}, for tests and configuration checks.
     *
     * @throws IllegalStateException    when a provider or factory violates the SPI, candid or name collision, both
     *                                  sources named
     * @throws IllegalArgumentException naming the property when a class name does not resolve to a factory
     */
    static LogitsProcessors build(List<? extends LogitsProcessorProvider> providers, List<String> classNames) {
        Objects.requireNonNull(providers, "providers");
        Objects.requireNonNull(classNames, "classNames");
        List<LogitsProcessorFactory> rows = new ArrayList<>();
        Map<String, String> source = new LinkedHashMap<>();
        for (LogitsProcessorProvider provider : providers) {
            List<LogitsProcessorFactory> contributed = provider.logitsProcessorFactories();
            if (contributed == null) {
                throw new IllegalStateException("logits processor provider " + provider.getClass().getName()
                        + " returned null factories");
            }
            for (Object row : contributed) {
                if (!(row instanceof LogitsProcessorFactory)) {
                    throw new IllegalStateException("logits processor provider " + provider.getClass().getName()
                            + " contributed " + (row == null ? "null" : row.getClass().getName())
                            + ", not a LogitsProcessorFactory");
                }
                register(rows, source, (LogitsProcessorFactory) row, provider.getClass().getName());
            }
        }
        for (LogitsProcessorFactory factory : loadClasses(classNames)) {
            register(rows, source, factory, "class " + factory.getClass().getName());
        }
        return rows.isEmpty() ? EMPTY : new LogitsProcessors(Collections.unmodifiableList(rows));
    }

    private static void register(List<LogitsProcessorFactory> rows, Map<String, String> source,
                                 LogitsProcessorFactory factory, String from) {
        List<String> names = namesOf(factory);
        for (String name : names) {
            if (name == null || name.trim().isEmpty()) {
                throw new IllegalStateException("logits processor " + factory.getClass().getName() + " of " + from
                        + " has a blank name");
            }
        }
        for (String name : names) {
            String k = key(name);
            if (source.containsKey(k)) {
                throw new IllegalStateException("logits processor name '" + name + "' of " + from
                        + " is already registered by " + source.get(k));
            }
            source.put(k, from);
        }
        rows.add(factory);
    }

    private static List<String> namesOf(LogitsProcessorFactory factory) {
        List<String> names = new ArrayList<>();
        names.add(factory.name());
        names.addAll(factory.aliases());
        return names;
    }

    /**
     * The instances of the factory classes {@code classNames}, in order.
     *
     * @throws IllegalArgumentException naming the property and the entry when one does not resolve
     */
    static List<LogitsProcessorFactory> loadClasses(List<String> classNames) {
        Objects.requireNonNull(classNames, "classNames");
        List<LogitsProcessorFactory> out = new ArrayList<>(classNames.size());
        for (String entry : classNames) {
            if (entry == null || entry.trim().isEmpty()) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + " holds a blank class name");
            }
            String name = entry.trim();
            Class<?> type;
            try {
                type = Class.forName(name, true, LogitsProcessors.class.getClassLoader());
            } catch (ClassNotFoundException e) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " is not on the class"
                        + " path", e);
            } catch (LinkageError e) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " could not be loaded: "
                        + e, e);
            }
            if (!LogitsProcessorFactory.class.isAssignableFrom(type)) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " does not implement "
                        + LogitsProcessorFactory.class.getName());
            }
            try {
                out.add((LogitsProcessorFactory) type.getConstructor().newInstance());
            } catch (NoSuchMethodException e) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " needs a public"
                        + " no-argument constructor", e);
            } catch (InvocationTargetException e) {
                Throwable cause = e.getCause() == null ? e : e.getCause();
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " failed to construct: "
                        + cause, cause);
            } catch (InstantiationException | IllegalAccessException e) {
                throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + ": " + name + " could not be"
                        + " constructed: " + e, e);
            }
        }
        return out;
    }

    private static String key(String name) {
        return name.trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }
}

package com.eonmux.neuralfree.gguf.plugin;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The dispatch of the plugin SPI: operator-installed jars on the application class
 * path contribute rows to the registries by declaring provider implementations of the SPI interfaces
 * ({@link ArchProvider}, {@link HfArchProvider}, {@link ProjectorProvider}, {@link ToolParserProvider},
 * {@link ReasoningParserProvider} and {@link LogitsProcessorProvider}) in {@code META-INF/services} entries, found
 * through {@link java.util.ServiceLoader}.
 *
 * <p>{@link #load(Class)} finds and instantiates the providers once per SPI class and returns the list in the
 * same order on every later call: the providers are sorted by their class name, independent of the order the
 * {@code META-INF/services} entries were read in. The SPI is looked up through the class loader that defines
 * this class, so a plugin must be on the application class path (a bindable loader); a provider defined by a
 * loader below that one never contributes rows. Plugins are trusted code the operator chose to install;
 * nothing is read from a model directory, and a plugin declares no configuration values of its own.
 *
 * <p>A provider's rows are merged when the receiving registry builds its table, on first use. A row whose
 * name collides with a built-in row or with a row of another provider is rejected there with both sources
 * named: providers never replace built-ins.
 */
public final class Plugins {

    private Plugins() {
    }

    /** The providers of every SPI class, loaded once on first {@link #load}. */
    private static final ConcurrentHashMap<Class<?>, List<?>> LOADED = new ConcurrentHashMap<>();

    /**
     * The plugin providers of {@code spi}, loaded once and sorted by provider class name.
     *
     * @param spi one of the SPI interfaces of {@code com.eonmux.neuralfree.gguf.plugin}
     * @return the providers, never {@code null}; never replaced by a later call
     */
    public static <S> List<S> load(Class<S> spi) {
        Objects.requireNonNull(spi, "spi");
        @SuppressWarnings("unchecked")
        List<S> providers = (List<S>) LOADED.computeIfAbsent(spi, key -> loadOnce(spi));
        return providers;
    }

    private static <S> List<S> loadOnce(Class<S> spi) {
        List<S> providers = new ArrayList<>();
        for (S provider : ServiceLoader.load(spi, Plugins.class.getClassLoader())) {
            providers.add(provider);
        }
        providers.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        return Collections.unmodifiableList(providers);
    }
}

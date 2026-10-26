package com.eonmux.neuralfree.gguf.plugin;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The dispatch of the plugin SPI (work package AX-0, row B6): operator-installed jars on the application class
 * path contribute rows to the registries by declaring provider implementations of the SPI interfaces
 * ({@link ArchProvider}, {@link HfArchProvider}, {@link ProjectorProvider}, {@link ToolParserProvider} and
 * {@link ReasoningParserProvider}) in {@code META-INF/services} entries, found through
 * {@link java.util.ServiceLoader}.
 *
 * <p>{@link #load(Class)} finds and instantiates the providers once per SPI class and returns the list in the
 * same order on every later call: the providers are sorted by their class name, independent of the order the
 * {@code META-INF/services} entries were read in. The SPI is looked up through the class loader that defines
 * this class, so a plugin must be on the application class path (a bindable loader); a provider defined by a
 * loader below that one never contributes rows. Plugins are trusted code the operator chose to install;
 * nothing is read from a model directory (&sect;3.6), and a plugin declares no configuration values of its own.
 *
 * <p>A provider's rows are merged when the receiving registry builds its table, on first use. A row whose
 * name collides with a built-in row or with a row of another provider is rejected there with both sources
 * named: providers never replace built-ins.
 *
 * <p>Plugin cache safety (&sect;5.5.1): a plugin jar is a code source that NF hashes into the runtime identity
 * (N-3 &mdash; PC-0), never a source the plugin declares, and a plugin component's configuration getters are
 * classified like the core's, so an unclassified one refuses caching (N-4 &mdash; PC-0). PC-8, the designated
 * writer over this file from Q3, adds PC-0's code scan at these loads: plugins get no
 * {@code CodeScanPolicy} allow-list rows, so a plugin method that references any scanned family or declares a
 * {@code native} method sets {@code refusal()} for the plugin's components, and a forward-path field a provider
 * did not declare (M-10, ComponentStates) sets {@code refusal()} too. A refusal turns KV reuse off for
 * models using the component; it never prevents loading.
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

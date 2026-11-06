package com.eonmux.neuralfree.acceleration.backend.spi;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * The accelerator backend providers available to the engine: one per accelerator API.
 *
 * <p>{@link #installed()} reads the application class path once through {@link ServiceLoader},
 * sorts the providers by class name so the result does not depend on the order the
 * {@code META-INF/services} entries were read in, and keeps at most one provider per
 * {@link BackendType}. Two providers for the same API are ambiguous: neither is used, and
 * {@link #failureReason(BackendType)} names both. A provider that cannot even report its API is
 * reported through {@link #loadFailures()} and skipped; the engine still starts.</p>
 *
 * <p>{@link #of(List)} is the same table over an explicit provider list, for an embedder that
 * supplies its backends itself and for tests. It refuses an invalid or duplicated provider
 * immediately, because such a list is a programming error rather than a deployment one.</p>
 */
public final class AcceleratorProviders {

    private static final Logger LOGGER = LoggingUtil.getLogger(AcceleratorProviders.class);

    private static final class Installed {
        static final AcceleratorProviders INSTANCE = load(AcceleratorProviders.class.getClassLoader());
    }

    private final Map<BackendType, AcceleratorBackendProvider> byType;
    private final Map<BackendType, String> failures;
    private final List<String> loadFailures;
    private final Set<BackendType> installedTypes;

    private AcceleratorProviders(Map<BackendType, AcceleratorBackendProvider> byType,
                                 Map<BackendType, String> failures, List<String> loadFailures) {
        EnumMap<BackendType, AcceleratorBackendProvider> providerCopy = new EnumMap<>(BackendType.class);
        providerCopy.putAll(byType);
        this.byType = Collections.unmodifiableMap(providerCopy);
        EnumMap<BackendType, String> failureCopy = new EnumMap<>(BackendType.class);
        failureCopy.putAll(failures);
        this.failures = Collections.unmodifiableMap(failureCopy);
        this.loadFailures = Collections.unmodifiableList(new ArrayList<>(loadFailures));
        EnumSet<BackendType> types = EnumSet.noneOf(BackendType.class);
        types.addAll(byType.keySet());
        this.installedTypes = Collections.unmodifiableSet(types);
    }

    /** The providers installed on the application class path, loaded once per process. */
    public static AcceleratorProviders installed() {
        return Installed.INSTANCE;
    }

    /**
     * A table over an explicit provider list.
     *
     * @throws IllegalArgumentException when a provider is null, serves a built-in or null API, or two
     *                                  providers serve the same API
     */
    public static AcceleratorProviders of(List<AcceleratorBackendProvider> providers) {
        Objects.requireNonNull(providers, "providers must not be null");
        Map<BackendType, AcceleratorBackendProvider> byType = new EnumMap<>(BackendType.class);
        for (AcceleratorBackendProvider provider : providers) {
            Objects.requireNonNull(provider, "providers must not contain null");
            BackendType type = Objects.requireNonNull(provider.backendType(),
                    "provider " + provider.getClass().getName() + " reports no API");
            if (!type.isAccelerator()) {
                throw new IllegalArgumentException("provider " + provider.getClass().getName() + " serves " + type
                        + ", which is not an accelerator API");
            }
            AcceleratorBackendProvider previous = byType.put(type, provider);
            if (previous != null) {
                throw new IllegalArgumentException("two " + type.label() + " backend providers: "
                        + nameOf(previous) + " and " + nameOf(provider) + "; keep one");
            }
        }
        return new AcceleratorProviders(byType, Collections.<BackendType, String>emptyMap(),
                Collections.<String>emptyList());
    }

    /** The provider of an accelerator API, or null when none is installed or the installed ones conflict. */
    public AcceleratorBackendProvider providerFor(BackendType api) {
        return api == null ? null : byType.get(api);
    }

    /**
     * Why an accelerator API has no usable provider: the conflicting providers when two serve it, or
     * null when one provider is installed or none is (the caller then reports the missing provider).
     */
    public String failureReason(BackendType api) {
        return api == null ? null : failures.get(api);
    }

    /** The accelerator APIs with one installed provider, in declaration order. */
    public Set<BackendType> installedTypes() {
        return installedTypes;
    }

    /** Problems met while loading the application class path, in the order they were met. */
    public List<String> loadFailures() {
        return loadFailures;
    }

    /** The reason an accelerator API has no provider installed, naming the SPI to implement. */
    public static String missingProviderReason(BackendType api) {
        Objects.requireNonNull(api, "API must not be null");
        return "no " + api.label() + " backend provider is installed; add a jar that declares a "
                + AcceleratorBackendProvider.class.getName() + " for " + api + " in META-INF/services/"
                + AcceleratorBackendProvider.class.getName();
    }

    /**
     * The providers declared in {@code META-INF/services} of a class loader, loaded once per call.
     * {@link #installed()} is this over the engine's own loader; a test or an embedder with its own
     * plugin loader calls it directly.
     *
     * @throws IllegalArgumentException when {@code loader} is null
     */
    static AcceleratorProviders load(ClassLoader loader) {
        Objects.requireNonNull(loader, "class loader must not be null");
        List<AcceleratorBackendProvider> loaded = new ArrayList<>();
        List<String> loadFailures = new ArrayList<>();
        ServiceLoader<AcceleratorBackendProvider> loaderService =
                ServiceLoader.load(AcceleratorBackendProvider.class, loader);
        Iterator<AcceleratorBackendProvider> iterator = loaderService.iterator();
        while (true) {
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                loaded.add(iterator.next());
            } catch (ServiceConfigurationError | LinkageError e) {
                loadFailures.add("an accelerator backend provider could not be loaded: " + describe(e));
            } catch (RuntimeException e) {
                loadFailures.add("an accelerator backend provider could not be created: " + describe(e));
            }
        }
        AcceleratorProviders providers = build(loaded, loadFailures);
        for (String failure : providers.loadFailures) {
            LOGGER.warn("{}", failure);
        }
        return providers;
    }

    private static AcceleratorProviders build(List<AcceleratorBackendProvider> loaded, List<String> loadFailures) {
        List<AcceleratorBackendProvider> sorted = new ArrayList<>(loaded);
        sorted.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        Map<BackendType, AcceleratorBackendProvider> byType = new EnumMap<>(BackendType.class);
        Map<BackendType, String> failures = new EnumMap<>(BackendType.class);
        Set<BackendType> conflicted = EnumSet.noneOf(BackendType.class);
        for (AcceleratorBackendProvider provider : sorted) {
            BackendType type;
            try {
                type = provider.backendType();
            } catch (RuntimeException | LinkageError e) {
                loadFailures.add("provider " + nameOf(provider) + " failed to report its API: " + describe(e));
                continue;
            }
            if (type == null) {
                loadFailures.add("provider " + nameOf(provider) + " reports no API");
                continue;
            }
            if (!type.isAccelerator()) {
                loadFailures.add("provider " + nameOf(provider) + " serves " + type
                        + ", which is not an accelerator API; it is ignored");
                continue;
            }
            if (conflicted.contains(type)) {
                continue;
            }
            AcceleratorBackendProvider previous = byType.get(type);
            if (previous != null) {
                failures.put(type, "two " + type.label() + " backend providers are installed: "
                        + nameOf(previous) + " and " + nameOf(provider) + "; keep one");
                byType.remove(type);
                conflicted.add(type);
            } else {
                byType.put(type, provider);
            }
        }
        return new AcceleratorProviders(byType, failures, loadFailures);
    }

    private static String nameOf(AcceleratorBackendProvider provider) {
        try {
            String name = provider.providerName();
            return name == null || name.isEmpty() ? provider.getClass().getName() : name;
        } catch (RuntimeException | LinkageError e) {
            return provider.getClass().getName();
        }
    }

    private static String describe(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isEmpty() ? t.getClass().getSimpleName() : message;
    }
}

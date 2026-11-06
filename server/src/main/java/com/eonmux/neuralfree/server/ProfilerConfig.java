package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import io.quarkus.runtime.Startup;
import jakarta.inject.Singleton;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * The engine profiler configuration (vLLM's {@code --profiler-config}): which profiler serves the profile window
 * and where its traces are written. The window records a Java Flight Recorder trace, the counterpart of vLLM's
 * {@code torch} profiler, so {@code torch} is the kind this server implements; vLLM's {@code cuda} and
 * {@code proton} kinds have no NeuralFree counterpart and refuse by name, as does any other kind and any field
 * beyond {@code profiler} and {@code torch_profiler_dir}.
 *
 * <p>{@code neuralfree.server.profiler.config} holds the JSON (the {@code --profiler-config} launch flag sets it).
 * The trace directory is the JSON's {@code torch_profiler_dir}; when the JSON names none, the existing
 * {@code neuralfree.server.profile.dir} setting stands in. Without a profiler the profile routes answer the 404 of
 * an unregistered vLLM route; with one, the server logs vLLM's warning once at startup. Immutable.
 */
@Startup
@Singleton
public final class ProfilerConfig {

    /** The property that configures the profiler. */
    static final String PROPERTY = "neuralfree.server.profiler.config";

    /** The launch flag that sets {@link #PROPERTY}. */
    static final String FLAG = "--profiler-config";

    /** The one profiler kind the JFR window implements: vLLM's application-level trace profiler. */
    static final String TORCH = "torch";

    /** The trace directory field of vLLM's ProfilerConfig. */
    static final String TORCH_DIR = "torch_profiler_dir";

    /** The vLLM profiler kinds with no NeuralFree counterpart. */
    static final Set<String> REFUSED_KINDS = Set.of("cuda", "proton");

    /** The fields of vLLM's ProfilerConfig this server honours. */
    private static final List<String> KEYS = List.of("profiler", TORCH_DIR);

    /** vLLM's warning when a profiler is configured. */
    static final String WARNING = "Profiler with mode '%s' is enabled in the API server."
            + " This should ONLY be used for local development!";

    private static final Logger LOGGER = Logger.getLogger(ProfilerConfig.class.getName());

    /** No profiler: the default. */
    static final ProfilerConfig OFF = new ProfilerConfig(null, null);

    private final String profiler;
    private final Path directory;

    /**
     * The configuration of the launch's properties, logging vLLM's warning once when a profiler is set.
     *
     * @throws IllegalArgumentException when the property is not a usable ProfilerConfig JSON object
     */
    public ProfilerConfig() {
        this(System.getProperties());
    }

    /** The configuration of {@code properties}, logging vLLM's warning once when a profiler is set. */
    ProfilerConfig(Map<?, ?> properties) {
        this(from(properties));
        if (enabled()) {
            LOGGER.warning(String.format(Locale.ROOT, WARNING, profiler));
        }
    }

    private ProfilerConfig(ProfilerConfig other) {
        this(other.profiler, other.directory);
    }

    private ProfilerConfig(String profiler, Path directory) {
        this.profiler = profiler;
        this.directory = directory;
    }

    /** Whether a profiler is configured: the profile routes serve only then. */
    boolean enabled() {
        return profiler != null;
    }

    /** The configured profiler kind ({@value #TORCH}), or null when profiler is off. */
    String profiler() {
        return profiler;
    }

    /** The trace directory, or null when profiler is off. */
    Path directory() {
        return directory;
    }

    /**
     * Refuses the profile window with the 404 an unregistered vLLM route answers while no profiler is configured.
     */
    void require() {
        if (!enabled()) {
            throw ServerJson.error(404, "the profile window is off: start the server with " + FLAG
                    + " '{\"profiler\": \"torch\", \"torch_profiler_dir\": \"DIR\"}' or -D" + PROPERTY
                    + " (vLLM registers /start_profile and /stop_profile only with a profiler configured)",
                    "NotFoundError");
        }
    }

    /** A profiler of {@code kind} writing its traces under {@code directory}, for tests and manual wiring. */
    static ProfilerConfig of(String kind, Path directory) {
        String resolved = kind(kind, FLAG);
        return new ProfilerConfig(resolved, directory(directory));
    }

    /**
     * The configuration of {@code properties} ({@link #PROPERTY}, and {@code neuralfree.server.profile.dir} as the
     * trace directory when the JSON names none), without logging.
     *
     * @throws IllegalArgumentException when the value is not a JSON object, names a field or kind this server does
     *                                  not honour, or names no trace directory
     */
    static ProfilerConfig from(Map<?, ?> properties) {
        String raw = text(properties.get(PROPERTY), PROPERTY);
        if (raw == null) {
            return OFF;
        }
        Object parsed;
        try {
            parsed = Json.parse(raw, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(PROPERTY + " is not valid JSON: " + e.getMessage(), e);
        }
        if (!(parsed instanceof Map<?, ?> config)) {
            throw new IllegalArgumentException(PROPERTY + " is a JSON object such as {\"profiler\": \"torch\", \""
                    + TORCH_DIR + "\": \"DIR\"}, got " + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        for (Object key : config.keySet()) {
            if (!KEYS.contains(String.valueOf(key))) {
                throw new IllegalArgumentException(PROPERTY + " key '" + key + "' is not supported; the profile"
                        + " window takes " + String.join(" and ", KEYS));
            }
        }
        String kind = text(config.get("profiler"), PROPERTY + " profiler");
        if (kind == null) {
            if (config.get(TORCH_DIR) != null) {
                throw new IllegalArgumentException(PROPERTY + " " + TORCH_DIR + " is only applicable when profiler"
                        + " is '" + TORCH + "'");
            }
            return OFF;
        }
        String resolved = kind(kind, PROPERTY);
        Path directory = path(config.get(TORCH_DIR), PROPERTY + " " + TORCH_DIR);
        if (directory == null) {
            directory = path(properties.get(EngineProfiler.PROFILE_DIR_PROPERTY),
                    EngineProfiler.PROFILE_DIR_PROPERTY);
        }
        if (directory == null) {
            throw new IllegalArgumentException(PROPERTY + " profiler '" + resolved + "' needs " + TORCH_DIR
                    + " (or " + EngineProfiler.PROFILE_DIR_PROPERTY + ") naming its trace directory");
        }
        return new ProfilerConfig(resolved, directory);
    }

    /** The profiler kind of {@code value}, refused by name unless the JFR window implements it. */
    private static String kind(String value, String source) {
        String kind = value == null ? null : value.trim();
        if (kind == null || kind.isEmpty()) {
            throw new IllegalArgumentException(source + " profiler must name a kind: " + TORCH);
        }
        String lower = kind.toLowerCase(Locale.ROOT);
        if (REFUSED_KINDS.contains(lower)) {
            throw new IllegalArgumentException(source + " profiler '" + kind + "' is not supported: the profile"
                    + " window records a Java Flight Recorder trace, the counterpart of vLLM's '" + TORCH
                    + "' profiler; vLLM's 'cuda' and 'proton' kinds have no NeuralFree counterpart");
        }
        if (!lower.equals(TORCH)) {
            throw new IllegalArgumentException(source + " profiler '" + kind + "' is not a profiler kind; use '"
                    + TORCH + "'");
        }
        return TORCH;
    }

    /** The absolute trace directory of {@code value}, or null when it is absent or blank; refused otherwise. */
    private static Path path(Object value, String field) {
        if (value == null) {
            return null;
        }
        if (!(value instanceof String text)) {
            throw new IllegalArgumentException(field + " is a string, got " + value);
        }
        if (text.isBlank()) {
            return null;
        }
        return directory(Path.of(text.trim()));
    }

    /** The trace directory, absolute and normalized. */
    private static Path directory(Path directory) {
        if (directory == null) {
            throw new IllegalArgumentException("the profiler needs a trace directory");
        }
        return directory.toAbsolutePath().normalize();
    }

    /** The trimmed text of {@code value}, or null when it is absent or blank; refused when it is not a string. */
    private static String text(Object value, String field) {
        if (value == null) {
            return null;
        }
        if (!(value instanceof String text)) {
            throw new IllegalArgumentException(field + " is a string, got " + value);
        }
        return text.isBlank() ? null : text.trim();
    }
}

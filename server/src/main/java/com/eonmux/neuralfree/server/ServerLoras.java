package com.eonmux.neuralfree.server;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * The server's per-request LoRA settings: the adapters loaded at startup (vLLM's {@code --lora-modules}), and
 * whether {@code /v1/load_lora_adapter} and {@code /v1/unload_lora_adapter} serve (vLLM's
 * {@code VLLM_ALLOW_RUNTIME_LORA_UPDATING}: loading reads a server-side path a client names, so it stays off unless
 * the operator turns it on). Immutable.
 *
 * @param modules         the adapters to load at startup, in order, names distinct
 * @param runtimeUpdating whether the runtime load and unload endpoints serve
 */
record ServerLoras(List<Module> modules, boolean runtimeUpdating) {

    /** The startup adapters, {@code name=path} entries separated by commas. */
    static final String MODULES_PROPERTY = "neuralfree.server.lora.modules";

    /** Whether the runtime load and unload endpoints serve. */
    static final String RUNTIME_PROPERTY = "neuralfree.server.lora.runtime.updating";

    /** vLLM's environment switch of the runtime endpoints, honoured when the property is unset. */
    static final String RUNTIME_ENV = "VLLM_ALLOW_RUNTIME_LORA_UPDATING";

    /** No startup adapter and the runtime endpoints off. */
    static final ServerLoras NONE = new ServerLoras(List.of(), false);

    /**
     * One startup adapter.
     *
     * @param name the name requests select it by ({@code model} or the {@code lora} list)
     * @param path the adapter GGUF or PEFT directory
     */
    record Module(String name, String path) {

        Module {
            requireName(name);
            Objects.requireNonNull(path, "path");
            if (path.isBlank()) {
                throw new IllegalArgumentException(MODULES_PROPERTY + ": adapter " + name + " names no path");
            }
        }
    }

    ServerLoras {
        modules = List.copyOf(Objects.requireNonNull(modules, "modules"));
        Set<String> names = new HashSet<>();
        for (Module m : modules) {
            if (!names.add(m.name())) {
                throw new IllegalArgumentException(MODULES_PROPERTY + " names adapter " + m.name() + " twice");
            }
        }
    }

    /**
     * The settings of {@link #MODULES_PROPERTY} and {@link #RUNTIME_PROPERTY} (else {@link #RUNTIME_ENV}).
     *
     * @throws IllegalArgumentException when a module entry is not {@code name=path}, names a path that does not
     *                                  exist, or repeats a name
     */
    static ServerLoras fromSystemProperties() {
        String runtime = System.getProperty(RUNTIME_PROPERTY);
        if (runtime == null || runtime.isBlank()) {
            runtime = System.getenv(RUNTIME_ENV);
        }
        return new ServerLoras(modules(System.getProperty(MODULES_PROPERTY)), truthy(runtime));
    }

    /** The {@code name=path} entries of {@code raw}, or none for null or blank. */
    static List<Module> modules(String raw) {
        if (raw == null || raw.isBlank()) {
            return List.of();
        }
        List<Module> out = new ArrayList<>();
        for (String entry : raw.split(",")) {
            String trimmed = entry.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            int eq = trimmed.indexOf('=');
            if (eq <= 0) {
                throw new IllegalArgumentException(MODULES_PROPERTY + " entry '" + trimmed + "' is not name=path");
            }
            String path = trimmed.substring(eq + 1).trim();
            if (!Files.exists(Path.of(path))) {
                throw new IllegalArgumentException(MODULES_PROPERTY + ": the adapter path of "
                        + trimmed.substring(0, eq).trim() + " does not exist: " + path);
            }
            out.add(new Module(trimmed.substring(0, eq).trim(), path));
        }
        return out;
    }

    /**
     * Checks an adapter name a request or a load names: not blank, and free of the comma and the equals sign the
     * module list separates entries with.
     *
     * @throws IllegalArgumentException naming the offending name
     */
    static void requireName(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("a LoRA adapter needs a name, got '" + name + "'");
        }
        if (name.indexOf(',') >= 0 || name.indexOf('=') >= 0) {
            throw new IllegalArgumentException("the LoRA adapter name '" + name + "' holds ',' or '='");
        }
    }

    private static boolean truthy(String raw) {
        if (raw == null) {
            return false;
        }
        String v = raw.trim().toLowerCase(Locale.ROOT);
        return v.equals("true") || v.equals("1");
    }
}

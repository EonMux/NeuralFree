package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The names the server answers to (vLLM's {@code --served-model-name}, llama.cpp's {@code --alias}): the
 * comma-separated {@code neuralfree.server.model.name} list, else the model path. {@code /v1/models} lists every
 * name; a response names the model the request asked for when it is served, else the first name.
 *
 * <p>With configured names a request naming another model refuses with 404, as vLLM's does; without them any name is
 * accepted, as llama.cpp's single-model server does, so clients that send a fixed model name keep working.
 * Immutable.</p>
 */
final class ServedModels {

    /** The system property listing the served names. */
    static final String PROPERTY = "neuralfree.server.model.name";

    private final List<String> names;
    private final boolean configured;

    private ServedModels(List<String> names, boolean configured) {
        this.names = Collections.unmodifiableList(new ArrayList<>(names));
        this.configured = configured;
    }

    /**
     * The served names of {@code configured} (comma-separated; null or blank falls back to {@code modelPath}).
     *
     * @throws IllegalArgumentException when the list names nothing
     */
    static ServedModels of(String configured, String modelPath) {
        if (configured == null || configured.isBlank()) {
            return new ServedModels(modelPath == null ? List.of() : List.of(modelPath), false);
        }
        Set<String> unique = new LinkedHashSet<>();
        for (String part : configured.split(",")) {
            String name = part.trim();
            if (!name.isEmpty()) {
                unique.add(name);
            }
        }
        if (unique.isEmpty()) {
            throw new IllegalArgumentException(PROPERTY + " names no model: '" + configured + "'");
        }
        return new ServedModels(new ArrayList<>(unique), true);
    }

    /** Every served name, the first the default. */
    List<String> names() {
        return names;
    }

    /** The name responses carry when the request names none: the first served name, or null when unloaded. */
    String primary() {
        return names.isEmpty() ? null : names.get(0);
    }

    /**
     * The model name a response to {@code body} carries: its {@code model} when served, else the first name.
     *
     * @throws jakarta.ws.rs.WebApplicationException 404 when names are configured and the body names another model
     */
    String resolve(JsonView body) {
        Object raw = body.raw("model");
        if (!(raw instanceof String requested) || requested.isEmpty()) {
            return primary();
        }
        if (names.contains(requested)) {
            return requested;
        }
        if (configured) {
            throw ServerJson.error(404, "The model `" + requested + "` does not exist.", "NotFoundError");
        }
        return primary();
    }
}

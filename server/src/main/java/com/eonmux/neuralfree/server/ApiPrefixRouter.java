package com.eonmux.neuralfree.server;

import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Singleton;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * The {@code --api-prefix} (llama.cpp {@code api_prefix}, {@code neuralfree.server.api.prefix}): a router installed
 * ahead of every route that answers the prefixed paths by rerouting them to the bare routes, so every route — API,
 * UI, probes and {@code /metrics} — serves under the prefix as llama.cpp serves it, and every bare path answers the
 * named 404 it answers there. The reroute runs before {@link ApiKeyFilter} sees the request, so the filter's
 * open-path checks apply after the prefix is stripped; {@link #stripped} lets it strip again when one arrives
 * prefixed. The prefix is one or more simple path segments; anything else refuses at launch by name.
 */
@Singleton
public class ApiPrefixRouter {

    /** The canonical prefix path the server serves under ({@code /api}, {@code /api/v1}). */
    static final String PROPERTY = "neuralfree.server.api.prefix";

    /** Behind the CORS preflight answer, ahead of the guards and every route the server registers. */
    private static final int ORDER = Integer.MIN_VALUE + 950;

    /** Marks the contexts this router already rerouted, so the restarted routing passes through once. */
    private static final String REROUTED = "neuralfree.api.prefix.rerouted";

    /** One prefix segment: letters, digits, dash, dot, underscore, tilde; no empty, {@code .} or {@code ..}. */
    private static final Pattern SEGMENT = Pattern.compile("[A-Za-z0-9._~-]{1,128}");

    ApiPrefixRouter() {
    }

    /** Installs the prefixed routing when {@code neuralfree.server.api.prefix} is set. */
    void install(@Observes Router router) {
        String prefix = fromProperties(System.getProperties());
        if (prefix == null) {
            return;
        }
        router.route().order(ORDER).handler(context -> route(prefix, context));
    }

    /**
     * One request under {@code prefix}: a bare path answers the named 404, the prefix itself and every path under
     * it reroutes to the path without the prefix (a query string travels with it).
     */
    static void route(String prefix, RoutingContext context) {
        if (Boolean.TRUE.equals(context.get(REROUTED))) {
            context.next();
            return;
        }
        String path = context.normalizedPath();
        String rest = stripped(path, prefix);
        if (rest.equals(path)) {
            context.response()
                    .setStatusCode(404)
                    .putHeader("Content-Type", "application/json")
                    .end(ServerJson.write(ServerJson.errorBody("unknown path '" + path
                            + "': the server serves under the prefix " + prefix, "invalid_request_error")));
            return;
        }
        context.put(REROUTED, Boolean.TRUE);
        String query = context.request().query();
        context.reroute(rest + (query == null ? "" : "?" + query));
    }

    /** The request path without {@code prefix}, or the path unchanged when it does not sit under it. */
    static String stripped(String path, String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return path;
        }
        if (path.equals(prefix)) {
            return "/";
        }
        if (path.startsWith(prefix + "/")) {
            return path.substring(prefix.length());
        }
        return path;
    }

    /**
     * The canonical prefix of {@code properties}, or null when unset or blank.
     *
     * @throws IllegalArgumentException naming the property when its value is not a prefix path
     */
    static String fromProperties(Map<?, ?> properties) {
        Object raw = properties.get(PROPERTY);
        if (raw == null || String.valueOf(raw).isBlank()) {
            return null;
        }
        return canonical(PROPERTY, String.valueOf(raw).trim());
    }

    /**
     * The canonical form of {@code value}: slash-separated simple segments with a leading slash and no trailing
     * one, so {@code api}, {@code /api} and {@code /api/} all canonicalize to {@code /api}.
     *
     * @throws IllegalArgumentException naming {@code name} for an empty, relative or unusable prefix
     */
    static String canonical(String name, String value) {
        String[] segments = value.split("/");
        StringBuilder out = new StringBuilder();
        for (String segment : segments) {
            if (segment.isEmpty()) {
                continue;
            }
            if (!SEGMENT.matcher(segment).matches() || segment.equals(".") || segment.equals("..")) {
                throw new IllegalArgumentException(name + " must be a simple path such as /api, got '" + value
                        + "': segments of letters, digits, dash, dot, underscore or tilde, no '..'");
            }
            out.append('/').append(segment);
        }
        if (out.length() == 0) {
            throw new IllegalArgumentException(name + " must be a path below the server root, got '" + value + "'");
        }
        return out.toString();
    }
}

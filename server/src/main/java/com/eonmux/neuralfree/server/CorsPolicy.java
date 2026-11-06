package com.eonmux.neuralfree.server;

import io.quarkus.vertx.http.runtime.cors.CORSConfig;
import io.quarkus.vertx.http.runtime.cors.CORSFilter;
import io.vertx.core.http.HttpServerResponse;
import jakarta.ws.rs.core.Response;
import org.eclipse.microprofile.config.Config;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * The CORS answer of the configured {@code quarkus.http.cors.*} policy: the {@code Access-Control-Allow-Origin},
 * {@code Access-Control-Allow-Credentials} and {@code Vary} headers llama.cpp's pre-routing handler stamps on every
 * response, including the refusals that never reach Quarkus's CORS filter. {@link CorsPreflight} answers OPTIONS
 * from the same policy, so the preflight and the ordinary responses agree. Immutable.
 */
final class CorsPolicy {

    static final String ENABLED = "quarkus.http.cors.enabled";
    static final String ORIGINS = "quarkus.http.cors.origins";
    static final String CREDENTIALS = "quarkus.http.cors.access-control-allow-credentials";

    static final String ALLOW_ORIGIN = "Access-Control-Allow-Origin";
    static final String ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials";
    static final String VARY = "Vary";

    private final boolean enabled;
    private final List<String> origins;
    private final List<Pattern> patterns;
    private final boolean wildcard;
    private final boolean credentials;

    private CorsPolicy(boolean enabled, Optional<List<String>> origins, boolean credentials) {
        this.enabled = enabled;
        this.origins = origins.orElse(List.of("*"));
        this.patterns = CORSFilter.parseAllowedOriginsRegex(origins);
        this.wildcard = this.origins.equals(List.of("*"));
        this.credentials = credentials;
    }

    /** The policy of Quarkus's CORS configuration. */
    static CorsPolicy from(CORSConfig cors) {
        return new CorsPolicy(cors.enabled(), cors.origins(),
                cors.accessControlAllowCredentials().orElse(true));
    }

    /** The policy of explicit dimensions, for callers that already hold the configuration. */
    static CorsPolicy of(boolean enabled, List<String> origins, boolean credentials) {
        return new CorsPolicy(enabled, Optional.of(origins), credentials);
    }

    /** The policy of the {@code quarkus.http.cors.*} runtime configuration. */
    static CorsPolicy from(Config config) {
        return new CorsPolicy(
                config.getOptionalValue(ENABLED, Boolean.class).orElse(false),
                config.getOptionalValues(ORIGINS, String.class),
                config.getOptionalValue(CREDENTIALS, Boolean.class).orElse(true));
    }

    /** A policy that answers no CORS headers and admits no origin: the defaults of the tests without a server. */
    static CorsPolicy disabled() {
        return of(false, List.of(), false);
    }

    boolean enabled() {
        return enabled;
    }

    /**
     * Whether the policy admits every origin, as llama.cpp's {@code --cors-origins *} does: enabled with the single
     * wildcard origin, or the same origin written as a regular expression.
     */
    boolean admitsEveryOrigin() {
        return enabled && (origins.equals(List.of("*")) || origins.equals(List.of("/.*/")));
    }

    /**
     * The origin headers of a response to {@code origin}, exactly as Quarkus's CORS filter and
     * {@link CorsPreflight} answer them: the wildcard echoes the origin when credentials are on and the literal
     * {@code *} otherwise, a configured origin is echoed when it matches, and {@code Access-Control-Allow-Credentials}
     * and {@code Vary: Origin} accompany every answer.
     */
    Map<String, String> headers(String origin) {
        if (!enabled) {
            return Map.of();
        }
        Map<String, String> headers = new LinkedHashMap<>();
        if (wildcard) {
            if (!credentials || origin != null) {
                headers.put(ALLOW_ORIGIN, credentials ? origin : "*");
            }
        } else if (origin != null && (origins.contains(origin) || CORSFilter.isOriginAllowedByRegex(patterns, origin))) {
            headers.put(ALLOW_ORIGIN, origin);
        }
        headers.put(ALLOW_CREDENTIALS, Boolean.toString(credentials));
        headers.put(VARY, "Origin");
        return headers;
    }

    /**
     * Stamps the origin headers on a refusal built ahead of Quarkus's CORS filter, so a browser can read the
     * response. A request without an {@code Origin} is not a cross-origin call and carries none.
     */
    void stamp(Response.ResponseBuilder response, String origin) {
        if (origin == null) {
            return;
        }
        headers(origin).forEach((name, value) -> response.header(name, value));
    }

    /** Stamps the origin headers on a Vert.x response written ahead of Quarkus's CORS filter. */
    void stamp(HttpServerResponse response, String origin) {
        if (origin == null) {
            return;
        }
        headers(origin).forEach(response::putHeader);
    }
}

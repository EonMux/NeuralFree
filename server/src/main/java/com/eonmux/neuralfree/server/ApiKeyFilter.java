package com.eonmux.neuralfree.server;

import jakarta.ws.rs.HttpMethod;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.container.PreMatching;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.Provider;

import java.util.Set;

/**
 * Requires an API key on every endpoint when keys are configured ({@link ApiKeys}): {@code Authorization: Bearer
 * <key>} (OpenAI, llama.cpp, vLLM) or {@code x-api-key: <key>} (Anthropic). The liveness endpoints stay open, as
 * llama.cpp keeps {@code /health} public: {@code /health}, {@code /v1/health} and the Quarkus {@code /q/} probes;
 * CORS preflight requests pass too. A missing or wrong key answers 401 with the OpenAI error envelope.
 * {@code /metrics} is a Vert.x route outside JAX-RS; {@link MetricsGuard} keys it the same way.
 */
@Provider
@PreMatching
public class ApiKeyFilter implements ContainerRequestFilter {

    private static final Set<String> OPEN = Set.of("health", "v1/health");

    private final ApiKeys keys;

    public ApiKeyFilter() {
        this(ApiKeys.from(System.getenv(), System.getProperties()));
    }

    ApiKeyFilter(ApiKeys keys) {
        this.keys = keys;
    }

    @Override
    public void filter(ContainerRequestContext request) {
        if (!keys.enabled() || HttpMethod.OPTIONS.equals(request.getMethod())) {
            return;
        }
        String path = request.getUriInfo().getPath();
        String relative = path.startsWith("/") ? path.substring(1) : path;
        if (OPEN.contains(relative) || relative.startsWith("q/")) {
            return;
        }
        if (keys.accepts(ApiKeys.presented(request.getHeaderString(HttpHeaders.AUTHORIZATION),
                request.getHeaderString("x-api-key")))) {
            return;
        }
        request.abortWith(Response.status(401)
                .header(HttpHeaders.WWW_AUTHENTICATE, "Bearer")
                .type(MediaType.APPLICATION_JSON)
                .entity(ApiKeys.refusalBody())
                .build());
    }
}

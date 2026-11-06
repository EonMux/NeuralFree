package com.eonmux.neuralfree.server;

import jakarta.ws.rs.HttpMethod;
import jakarta.ws.rs.container.ContainerRequestContext;
import jakarta.ws.rs.container.ContainerRequestFilter;
import jakarta.ws.rs.container.PreMatching;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.ext.Provider;
import org.eclipse.microprofile.config.ConfigProvider;

import java.util.Properties;
import java.util.Set;

/**
 * Requires an API key on every endpoint when keys are configured ({@link ApiKeys}): {@code Authorization: Bearer
 * <key>} (OpenAI, llama.cpp, vLLM) or {@code x-api-key: <key>} (Anthropic). The probe endpoints stay open, as
 * llama.cpp keeps {@code /health} public and vLLM keys only {@code /v1}: {@code /health}, {@code /v1/health},
 * {@code /ready}, {@code /readyz}, {@code /ping} and the Quarkus {@code /q/} probes; they answer status only. The
 * bundled chat UI stays open too ({@code /}, {@code /index.html} and the {@code /ui/} assets) unless the web UI is
 * disabled ({@code neuralfree.server.webui=false}, llama.cpp's {@code --no-ui}), when those paths need the key like
 * everything else. The checks run after the {@code --api-prefix} is stripped ({@link ApiPrefixRouter#stripped}), so
 * the same paths stay open under a prefix. A request whose asset path carries a {@code ..} segment never counts as
 * a UI path — the open check accepts flat asset names only. CORS preflight requests pass too. A missing or wrong
 * key answers 401 with the OpenAI error envelope and the configured CORS origin headers, so a browser on another
 * origin can read the refusal even though the filter answers ahead of Quarkus's CORS filter.
 * {@code /metrics} is a Vert.x route outside JAX-RS; {@link MetricsGuard} keys it the same way.
 * {@link GrpcAccessInterceptor} keys gRPC calls with the same key set against their {@code authorization} metadata,
 * keeping the standard {@code Health/Check} open as {@code /health} is.
 */
@Provider
@PreMatching
public class ApiKeyFilter implements ContainerRequestFilter {

    private static final Set<String> OPEN = Set.of("health", "v1/health", "ready", "readyz", "ping");

    /** The bundled UI's asset namespace: {@code /ui} and every asset under it. */
    private static final String UI_PREFIX = "ui/";

    private final ApiKeys keys;
    private final CorsPolicy cors;
    private final boolean webui;
    private final String prefix;

    public ApiKeyFilter() {
        this(ApiKeys.from(System.getenv(), System.getProperties()), CorsPolicy.from(ConfigProvider.getConfig()),
                UiResource.enabled(System.getProperties()), prefix(System.getProperties()));
    }

    ApiKeyFilter(ApiKeys keys) {
        this(keys, CorsPolicy.disabled(), true, null);
    }

    ApiKeyFilter(ApiKeys keys, boolean webui, String prefix) {
        this(keys, CorsPolicy.disabled(), webui, prefix);
    }

    ApiKeyFilter(ApiKeys keys, CorsPolicy cors, boolean webui, String prefix) {
        this.keys = keys;
        this.cors = cors;
        this.webui = webui;
        this.prefix = prefix;
    }

    @Override
    public void filter(ContainerRequestContext request) {
        if (!keys.enabled() || HttpMethod.OPTIONS.equals(request.getMethod())) {
            return;
        }
        String relative = relative(request.getUriInfo().getPath());
        if (open(relative) || openUiPath(relative)) {
            return;
        }
        if (keys.accepts(ApiKeys.presented(request.getHeaderString(HttpHeaders.AUTHORIZATION),
                request.getHeaderString("x-api-key")))) {
            return;
        }
        Response.ResponseBuilder refusal = Response.status(401)
                .header(HttpHeaders.WWW_AUTHENTICATE, "Bearer")
                .type(MediaType.APPLICATION_JSON)
                .entity(ApiKeys.refusalBody());
        cors.stamp(refusal, request.getHeaderString("Origin"));
        request.abortWith(refusal.build());
    }

    /**
     * Whether {@code relative} serves without a key: a status-only probe path or the Quarkus {@code /q/} probes.
     * The gRPC ingress asks the same question about {@code health} so its open {@code grpc.health.v1.Health/Check}
     * matches the HTTP {@code /health}.
     */
    static boolean open(String relative) {
        return OPEN.contains(relative) || relative.startsWith("q/");
    }

    /**
     * The request path the open checks apply to: the {@code --api-prefix} stripped (the {@link ApiPrefixRouter}
     * reroute strips it before JAX-RS sees the request; a prefixed path that reaches the filter strips here too),
     * then without its leading slash.
     */
    private String relative(String path) {
        String slashed = path.startsWith("/") ? path : "/" + path;
        String stripped = ApiPrefixRouter.stripped(slashed, prefix);
        return stripped.startsWith("/") ? stripped.substring(1) : stripped;
    }

    /** Whether {@code relative} is a UI path the enabled web UI serves without a key: flat asset names only. */
    private boolean openUiPath(String relative) {
        if (!webui) {
            return false;
        }
        if (relative.isEmpty() || relative.equals("index.html") || relative.equals("ui") || relative.equals("ui/")) {
            return true;
        }
        if (!relative.startsWith(UI_PREFIX)) {
            return false;
        }
        String asset = relative.substring(UI_PREFIX.length());
        return UiResource.NAME.matcher(asset).matches() && !asset.contains("..");
    }

    /** The canonical {@code --api-prefix} of {@code properties}, or null without one ({@link ApiPrefixRouter}). */
    private static String prefix(Properties properties) {
        return ApiPrefixRouter.fromProperties(properties);
    }
}

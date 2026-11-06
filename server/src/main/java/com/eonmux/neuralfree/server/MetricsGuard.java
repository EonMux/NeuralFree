package com.eonmux.neuralfree.server;

import io.quarkus.vertx.http.runtime.VertxHttpConfig;
import io.vertx.core.http.HttpHeaders;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Singleton;

/**
 * Requires the API key on {@code /metrics} when keys are configured, as llama.cpp does (only {@code /health} is
 * public there). Micrometer serves the endpoint as a Vert.x route, which the JAX-RS {@link ApiKeyFilter} never sees,
 * so the guard runs on the router ahead of it. The guard also runs ahead of Quarkus's CORS filter, so it stamps the
 * configured CORS origin headers on a refusal for a browser to read.
 */
@Singleton
public class MetricsGuard {

    /** Ahead of the Micrometer route and the Quarkus handlers that answer it. */
    private static final int ORDER = Integer.MIN_VALUE + 1000;

    void guard(@Observes Router router, VertxHttpConfig http) {
        ApiKeys keys = ApiKeys.from(System.getenv(), System.getProperties());
        if (!keys.enabled()) {
            return;
        }
        CorsPolicy cors = CorsPolicy.from(http.cors());
        router.route("/metrics*").order(ORDER).handler(context -> check(keys, cors, context));
    }

    static void check(ApiKeys keys, CorsPolicy cors, RoutingContext context) {
        String presented = ApiKeys.presented(context.request().getHeader(HttpHeaders.AUTHORIZATION),
                context.request().getHeader("x-api-key"));
        if (keys.accepts(presented)) {
            context.next();
            return;
        }
        var response = context.response()
                .setStatusCode(401)
                .putHeader("WWW-Authenticate", "Bearer")
                .putHeader(HttpHeaders.CONTENT_TYPE, "application/json");
        cors.stamp(response, context.request().getHeader("Origin"));
        response.end(ApiKeys.refusalBody());
    }
}

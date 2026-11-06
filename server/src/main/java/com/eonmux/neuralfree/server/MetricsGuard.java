package com.eonmux.neuralfree.server;

import io.vertx.core.http.HttpHeaders;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Singleton;

/**
 * Requires the API key on {@code /metrics} when keys are configured, as llama.cpp does (only {@code /health} is
 * public there). Micrometer serves the endpoint as a Vert.x route, which the JAX-RS {@link ApiKeyFilter} never sees,
 * so the guard runs on the router ahead of it.
 */
@Singleton
public class MetricsGuard {

    /** Ahead of the Micrometer route and the Quarkus handlers that answer it. */
    private static final int ORDER = Integer.MIN_VALUE + 1000;

    void guard(@Observes Router router) {
        ApiKeys keys = ApiKeys.from(System.getenv(), System.getProperties());
        if (!keys.enabled()) {
            return;
        }
        router.route("/metrics*").order(ORDER).handler(context -> check(keys, context));
    }

    static void check(ApiKeys keys, RoutingContext context) {
        String presented = ApiKeys.presented(context.request().getHeader(HttpHeaders.AUTHORIZATION),
                context.request().getHeader("x-api-key"));
        if (keys.accepts(presented)) {
            context.next();
            return;
        }
        context.response()
                .setStatusCode(401)
                .putHeader("WWW-Authenticate", "Bearer")
                .putHeader(HttpHeaders.CONTENT_TYPE, "application/json")
                .end(ApiKeys.refusalBody());
    }
}
